"""
Custom-model calibration bridge
===============================

Glue that lets a **user-defined** :class:`~backend.core.interfaces.Model`
(an SDE given by ``drift``/``diffusion`` or a characteristic function) be
calibrated to an option surface by reusing the existing surface pricers and
optimizer strategies — no per-model calibrator required.

Two pricing routes, picked from ``model.supported_engines``:

* **FFT (affine)** — when ``PricingCapability.FFT`` is advertised, the surface
  is priced by :func:`price_surface` + :class:`FFTEngine` (the model's
  characteristic function). Fast and exact.
* **Monte-Carlo (nonaffine)** — otherwise the SDE coefficients ``drift`` /
  ``diffusion`` (+ optional stochastic-variance dynamics) are Euler-discretised
  by :class:`CustomTerminalSimulator` and the surface is priced by
  :func:`price_surface_mc`.

Both routes feed a scalar :class:`ObjectiveStrategy` loss into a derivative-free
or quasi-Newton :class:`OptimizerStrategy` (Differential Evolution, Nelder-Mead,
L-BFGS-B). Levenberg-Marquardt is intentionally **unsupported**: it needs a
JAX-differentiable residual, which arbitrary user Python code does not provide.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from backend.calibration.base import BaseCalibrator, CalibrationResult
from backend.calibration.objectives import ObjectiveStrategy
from backend.calibration.optimizers.base import (
    CalibrationProblem,
    IterationLogger,
    OptimizerStrategy,
)
from backend.calibration.pricing_loop import price_surface, price_surface_mc
from backend.calibration.utils import (
    compute_rmse_iv,
    compute_rmse_price,
    model_prices_to_ivs,
)
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability
from backend.engines.fft_engine import FFTEngine
from backend.utils.logging import get_logger

logger = get_logger(__name__)

# Default Euler resolution for the MC route — a compromise between
# discretisation bias and speed for an interactive, pure-NumPy simulator.
DEFAULT_STEPS_PER_YEAR: int = 50
# Loss returned when a pricing evaluation fails or is non-finite — large
# enough to steer any scalar solver away without raising.
_FAIL_LOSS: float = 1e10


def price_custom_surface(
    model: Model,
    market_data: Any,
    *,
    n_paths: int = 20_000,
    mc_seed: int = 12_345,
    steps_per_year: int = DEFAULT_STEPS_PER_YEAR,
) -> np.ndarray:
    """Price an option surface under a custom model (FFT if affine, else MC).

    Shared by :class:`CustomModelCalibrator` (per evaluation) and the UI
    (post-fit overlay), so both use exactly the same pricing route and seed.
    """
    if PricingCapability.FFT in model.supported_engines:
        return price_surface(model, market_data, FFTEngine())
    sim = CustomTerminalSimulator(model, steps_per_year=steps_per_year)
    return price_surface_mc(sim, market_data, n_paths=n_paths, mc_seed=mc_seed)


class CustomTerminalSimulator:
    """Euler-Maruyama terminal-price simulator for an arbitrary SDE model.

    Adapts a :class:`~backend.core.interfaces.Model` exposing
    ``drift(s, v, t, r, q)`` / ``diffusion(s, v, t)`` (plus the optional
    ``variance_drift`` / ``variance_diffusion`` / ``get_correlation`` / ``jump``
    hooks) to the
    :class:`~backend.calibration.pricing_loop.MCTerminalSimulator` protocol
    consumed by :func:`price_surface_mc`.

    Unlike :class:`~backend.simulation.models.generic_euler.GenericEulerSimulator`,
    every normal is drawn from the **injected** ``rng`` (never the global NumPy
    RNG), so the surface pricer's common-random-number seed yields a smooth
    objective across optimizer evaluations, and antithetic variates are
    supported. Risk-neutral with ``q = 0`` (consistent with the existing
    nonaffine MC path, whose synthetic surfaces are dividend-free).
    """

    def __init__(
        self, model: Model, *, steps_per_year: int = DEFAULT_STEPS_PER_YEAR
    ) -> None:
        self._model = model
        self._steps_per_year = int(steps_per_year)
        self._drift = model.drift
        self._diffusion = model.diffusion
        self._has_jump = callable(getattr(model, "jump", None))
        self._has_stoch_vol = callable(
            getattr(model, "variance_drift", None)
        ) and callable(getattr(model, "variance_diffusion", None))
        params = model.get_parameters()
        self._v0 = float(params.get("v0", params.get("sigma", 0.20) ** 2))
        self._rho = (
            float(model.get_correlation())
            if callable(getattr(model, "get_correlation", None))
            else 0.0
        )

    @staticmethod
    def _normals(
        rng: np.random.Generator, n_paths: int, antithetic: bool
    ) -> np.ndarray:
        """Draw ``n_paths`` standard normals, mirrored when ``antithetic``."""
        if not antithetic:
            return rng.standard_normal(n_paths)
        half = (n_paths + 1) // 2
        z = rng.standard_normal(half)
        return np.concatenate([z, -z])[:n_paths]

    def terminals(
        self,
        s0: float,
        r: float,
        t: float,
        *,
        n_paths: int,
        rng: np.random.Generator,
        antithetic: bool = True,
    ) -> np.ndarray:
        """Simulate terminal prices ``S(T)`` — shape ``(n_paths,)``."""
        n_paths = int(n_paths)
        n_steps = max(2, round(self._steps_per_year * float(t)))
        dt = float(t) / n_steps
        sqrt_dt = np.sqrt(dt)

        # The optional jump() hook draws from the *global* NumPy RNG; reseed it
        # deterministically from the injected generator so the same mc_seed
        # reproduces the same surfaces across optimizer evaluations (CRN).
        if self._has_jump:
            np.random.seed(int(rng.integers(0, 2**31 - 1)))

        s = np.full(n_paths, float(s0), dtype=np.float64)
        v: Any = np.full(n_paths, self._v0) if self._has_stoch_vol else self._v0
        sqrt_1mrho2 = (
            np.sqrt(max(0.0, 1.0 - self._rho**2)) if self._has_stoch_vol else 0.0
        )

        for j in range(n_steps):
            t_j = j * dt
            z1 = self._normals(rng, n_paths, antithetic)
            drift_val = np.asarray(self._drift(s, v, t_j, r, 0.0), dtype=float)
            diff_val = np.asarray(self._diffusion(s, v, t_j), dtype=float)
            s = s + drift_val * dt + diff_val * sqrt_dt * z1
            if self._has_jump:
                s = s + np.asarray(self._model.jump(s, dt), dtype=float)
            np.maximum(s, 1e-10, out=s)

            if self._has_stoch_vol:
                z_indep = self._normals(rng, n_paths, antithetic)
                z2 = self._rho * z1 + sqrt_1mrho2 * z_indep
                vd = np.asarray(self._model.variance_drift(v, s, t_j), dtype=float)
                vdiff = np.asarray(
                    self._model.variance_diffusion(v, s, t_j), dtype=float
                )
                v = v + vd * dt + vdiff * sqrt_dt * z2
                np.maximum(v, 1e-10, out=v)

        return s


class CustomModelCalibrator(BaseCalibrator):
    """Calibrate a user-defined surface model with a scalar solver.

    Parameters
    ----------
    model_class
        A ``Model`` subclass whose ``__init__`` accepts ``param_names`` as
        keyword arguments.
    param_names
        Ordered calibration parameter names (the optimiser's working vector).
    bounds
        ``(low, high)`` box per parameter, same order as ``param_names``.
    objective
        Scalar loss strategy (see :func:`make_objective`).
    optimizer
        Derivative-free / quasi-Newton strategy (DE / NM / L-BFGS-B).
    n_paths, mc_seed, steps_per_year
        Monte-Carlo controls for the nonaffine route (ignored on the FFT route).
    x0
        Optional initial guess; defaults to the box midpoints.
    max_nfev, tol
        Forwarded to ``optimizer.solve``.
    """

    def __init__(
        self,
        model_class: type[Model],
        param_names: tuple[str, ...],
        bounds: list[tuple[float, float]],
        *,
        objective: ObjectiveStrategy,
        optimizer: OptimizerStrategy,
        n_paths: int = 20_000,
        mc_seed: int = 12_345,
        steps_per_year: int = DEFAULT_STEPS_PER_YEAR,
        x0: np.ndarray | list[float] | None = None,
        max_nfev: int = 300,
        tol: float = 1e-8,
        log_iterations: bool = False,
        iteration_callback: Any = None,
    ) -> None:
        self._model_class = model_class
        self._param_names = tuple(param_names)
        self._bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        self._objective = objective
        self._optimizer = optimizer
        self._n_paths = int(n_paths)
        self._mc_seed = int(mc_seed)
        self._steps_per_year = int(steps_per_year)
        self._max_nfev = int(max_nfev)
        self._tol = float(tol)
        self._log_iterations = bool(log_iterations)
        self._iteration_callback = iteration_callback
        self._fft_engine = FFTEngine()
        if x0 is None:
            x0 = [0.5 * (lo + hi) for lo, hi in self._bounds]
        self._x0 = np.asarray(x0, dtype=float)

    # -- BaseCalibrator contract ------------------------------------------- #

    def default_bounds(self) -> list[tuple[float, float]]:
        return list(self._bounds)

    def objective(self, params: np.ndarray, market_data: Any) -> float:
        try:
            prices = self._price(params, market_data)
        except (
            ValueError,
            FloatingPointError,
            RuntimeError,
            ArithmeticError,
            TypeError,
        ) as exc:
            logger.debug("custom objective pricing failed: %r", exc)
            return _FAIL_LOSS
        if not np.all(np.isfinite(prices)):
            return _FAIL_LOSS
        loss = self._objective.compute_loss(
            np.asarray(prices, dtype=float), market_data
        )
        return float(loss) if np.isfinite(loss) else _FAIL_LOSS

    def calibrate(self, market_data: Any) -> CalibrationResult:
        lo = np.array([b[0] for b in self._bounds], dtype=float)
        hi = np.array([b[1] for b in self._bounds], dtype=float)
        problem = CalibrationProblem(
            x0=self._x0.copy(),
            bounds_lo=lo,
            bounds_hi=hi,
            param_names=self._param_names,
            objective_fn=lambda x: self.objective(x, market_data),
            param_mapper=lambda x: {n: float(v) for n, v in zip(self._param_names, x)},
        )
        # Stream per-evaluation snapshots to the live UI when requested — same
        # on_snapshot contract the affine/MC backend calibrators use.
        on_snapshot = (
            self._iteration_callback
            if (self._log_iterations or self._iteration_callback)
            else None
        )
        iter_logger = IterationLogger(problem, on_snapshot=on_snapshot)
        t0 = time.perf_counter()
        opt = self._optimizer.solve(
            problem, iter_logger, max_nfev=self._max_nfev, tol=self._tol
        )
        elapsed = time.perf_counter() - t0

        x_opt = np.asarray(opt.x_optimal, dtype=float)
        model = self._build_model(x_opt)

        rmse_price: float | None = None
        rmse_iv: float | None = None
        try:
            model_prices = np.asarray(self._price(x_opt, market_data), dtype=float)
            rmse_price = compute_rmse_price(model_prices, market_data.market_prices)
            rmse_iv = self._rmse_iv(model_prices, market_data)
        except (ValueError, RuntimeError, ArithmeticError, TypeError) as exc:
            logger.debug("custom post-fit metrics failed: %r", exc)

        return CalibrationResult(
            model=model,
            objective_value=float(opt.objective_value),
            n_iterations=int(opt.n_function_evals),
            success=bool(opt.success),
            method=opt.method,
            rmse_price=rmse_price,
            rmse_iv=rmse_iv,
            elapsed_seconds=elapsed,
            diagnostics={
                "estimated_params": dict(zip(self._param_names, x_opt.tolist())),
                "route": "fft" if self._uses_fft(model) else "mc",
            },
            iteration_history=opt.iteration_history,
            optimizer_name=opt.method,
        )

    # -- internals --------------------------------------------------------- #

    def _build_model(self, x: np.ndarray) -> Model:
        return self._model_class(
            **{n: float(v) for n, v in zip(self._param_names, np.asarray(x))}
        )

    @staticmethod
    def _uses_fft(model: Model) -> bool:
        return PricingCapability.FFT in model.supported_engines

    def _price(self, x: np.ndarray, market_data: Any) -> np.ndarray:
        model = self._build_model(x)
        if self._uses_fft(model):
            return price_surface(model, market_data, self._fft_engine)
        sim = CustomTerminalSimulator(model, steps_per_year=self._steps_per_year)
        return price_surface_mc(
            sim, market_data, n_paths=self._n_paths, mc_seed=self._mc_seed
        )

    # Module-level :func:`price_custom_surface` is the public, stateless twin of
    # ``_price`` (same routing) for callers without a calibrator instance.

    def _rmse_iv(self, model_prices: np.ndarray, market_data: Any) -> float | None:
        is_calls = np.array([q.is_call for q in market_data.quotes])
        model_ivs = model_prices_to_ivs(
            model_prices=model_prices,
            spot=market_data.spot,
            strikes=market_data.strikes,
            maturities=market_data.maturities,
            rate=market_data.rate,
            is_calls=is_calls,
            dividend_yield=market_data.dividend_yield,
        )
        market_ivs = np.array(
            [
                q.implied_vol if q.implied_vol is not None else np.nan
                for q in market_data.quotes
            ]
        )
        valid = ~np.isnan(model_ivs) & ~np.isnan(market_ivs)
        if int(valid.sum()) == 0:
            return None
        return compute_rmse_iv(model_ivs[valid], market_ivs[valid])


__all__ = [
    "CustomModelCalibrator",
    "CustomTerminalSimulator",
    "DEFAULT_STEPS_PER_YEAR",
    "price_custom_surface",
]
