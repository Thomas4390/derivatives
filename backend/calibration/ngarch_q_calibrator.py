"""
Risk-neutral GARCH-Q surface calibrator — Monte-Carlo
=====================================================

Surface calibration for the risk-neutral nonaffine GARCH option family
(Duan 1995, Dorion & François §7.2). One generalised calibrator
:class:`GARCHRiskNeutralCalibrator` covers the three variants via ``garch_type``
(``garch`` / ``ngarch`` / ``gjr_garch``); :class:`NGARCHRiskNeutralCalibrator`
is the Duan-NGARCH instance kept for back-compat.

Because these models are **nonaffine** they have no closed-form characteristic
function, so model option prices are produced by Monte-Carlo
(:class:`GARCHRiskNeutralSimulator`) rather than FFT. The optimiser therefore
works on a *scalar* objective (no analytical Jacobian), with **common random
numbers** (a fixed MC seed reused across evaluations) making the objective
deterministic and smooth enough for derivative-free / finite-difference local
solvers (Nelder-Mead, L-BFGS-B) and the global Differential Evolution.

The parameter vector is a uniform 5-slot ``(omega, alpha, beta, gamma, h0)``; for
the symmetric ``garch`` variant ``gamma`` is pinned to 0 by its bounds. The
variance-stationarity condition is handled by a :class:`StationarityMode`
(OFF / SOFT penalty / HARD cap), per-variant:

    garch  : beta + alpha < 1                  HARD caps alpha
    ngarch : beta + alpha*(1 + gamma^2) < 1     HARD caps gamma
    gjr    : beta + alpha + gamma/2 < 1         HARD caps gamma

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np

from backend.calibration.base import BaseCalibrator, CalibrationResult
from backend.calibration.market_data import OptionMarketData
from backend.calibration.objectives import ObjectiveStrategy, PriceMSEObjective
from backend.calibration.optimizers import (
    CalibrationProblem,
    IterationLogger,
    NelderMeadStrategy,
    OptimizerStrategy,
)
from backend.calibration.pricing_loop import price_surface_mc
from backend.calibration.stationarity import (
    DEFAULT_STATIONARITY_WEIGHT,
    StationarityMode,
    garch_capped_alpha,
    garch_q_persistence,
    gjr_capped_gamma,
    ngarch_capped_gamma,
)
from backend.calibration.utils import (
    compute_rmse_iv,
    compute_rmse_price,
    get_atm_iv,
    model_prices_to_ivs,
)
from backend.models.ngarch_q import (
    GARCHRiskNeutralModel,
    GJRGARCHRiskNeutralModel,
    NGARCHRiskNeutralModel,
)
from backend.simulation.models.garch_q import GARCHRiskNeutralSimulator
from backend.utils.constants.calibration import (
    NGARCH_Q_PARAM_NAMES,
    RISK_NEUTRAL_GARCH_BOUNDS,
    VALID_GARCH_TYPES,
)
from backend.utils.constants.time import TRADING_DAYS_PER_YEAR
from backend.utils.logging import configure_root, get_logger

logger = get_logger(__name__)

_SPY = TRADING_DAYS_PER_YEAR
# Risk-neutral leverage seed per variant (γ slot of the uniform 5-vector).
_GAMMA_SEED: dict[str, float] = {"garch": 0.0, "ngarch": 0.8, "gjr_garch": 0.3}


class GARCHRiskNeutralCalibrator(BaseCalibrator):
    """Risk-neutral GARCH-Q surface calibrator (garch / ngarch / gjr_garch).

    Scalar Monte-Carlo objective with common random numbers; the variant is
    selected by ``garch_type`` and drives the parameter bounds, the
    stationarity persistence/cap, and the model class produced.
    """

    def __init__(
        self,
        garch_type: str = "ngarch",
        n_restarts: int = 2,
        max_nfev: int = 200,
        seed: int = 42,
        n_paths: int = 40_000,
        mc_seed: int = 20_240,
        optimizer: OptimizerStrategy | None = None,
        objective: ObjectiveStrategy | None = None,
        log_iterations: bool = False,
        iteration_callback: Any = None,
        stationarity_mode: StationarityMode | str = StationarityMode.SOFT,
        stationarity_weight: float = DEFAULT_STATIONARITY_WEIGHT,
    ) -> None:
        if garch_type not in VALID_GARCH_TYPES:
            raise ValueError(
                f"garch_type must be one of {VALID_GARCH_TYPES}, got {garch_type!r}"
            )
        self.garch_type = garch_type
        self._bounds = RISK_NEUTRAL_GARCH_BOUNDS[garch_type]
        self._lo = np.array([self._bounds[p][0] for p in NGARCH_Q_PARAM_NAMES])
        self._hi = np.array([self._bounds[p][1] for p in NGARCH_Q_PARAM_NAMES])
        self.n_restarts = max(1, int(n_restarts))
        self.max_nfev = int(max_nfev)
        self.seed = int(seed)
        self.n_paths = int(n_paths)
        self.mc_seed = int(mc_seed)
        self.stationarity_mode = StationarityMode.coerce(stationarity_mode)
        self.stationarity_weight = float(stationarity_weight)
        # Derivative-free local solver by default — robust on the MC objective.
        self.optimizer = optimizer or NelderMeadStrategy()
        if self.optimizer.requires_residuals:
            raise ValueError(
                f"GARCH-Q calibration is a scalar MC problem; optimizer "
                f"'{self.optimizer.name}' requires residuals (LM-JAX) and cannot "
                f"be used. Choose DE, NM, or L-BFGS-B."
            )
        self._obj_strategy: ObjectiveStrategy = objective or PriceMSEObjective()
        self.objective_type = self._obj_strategy.name
        self.log_iterations = bool(log_iterations)
        self.iteration_callback = iteration_callback

    # ------------------------------------------------------------------ #
    # BaseCalibrator contract
    # ------------------------------------------------------------------ #

    def default_bounds(self) -> list[tuple[float, float]]:
        return [self._bounds[p] for p in NGARCH_Q_PARAM_NAMES]

    def objective(self, params: np.ndarray, market_data: OptionMarketData) -> float:
        """Scalar surface loss + stationarity penalty (BaseCalibrator contract)."""
        model_prices = self._price_natural(self._effective_params(params), market_data)
        loss = float(self._obj_strategy.compute_loss(model_prices, market_data))
        return loss + self._stationarity_penalty(params)

    # ------------------------------------------------------------------ #
    # Parameter handling
    # ------------------------------------------------------------------ #

    def _effective_params(self, x: np.ndarray) -> np.ndarray:
        """Natural params clamped to the box; HARD mode caps the variant's knob."""
        p = np.clip(np.asarray(x, dtype=float), self._lo, self._hi)
        if self.stationarity_mode is StationarityMode.HARD:
            if self.garch_type == "ngarch":
                p[3] = min(p[3], ngarch_capped_gamma(p[1], p[2], float(self._hi[3])))
            elif self.garch_type == "gjr_garch":
                p[3] = min(p[3], gjr_capped_gamma(p[1], p[2], float(self._hi[3])))
            else:  # garch — symmetric, no gamma; cap alpha instead
                p[1] = min(p[1], garch_capped_alpha(p[2], float(self._hi[1])))
        return p

    def _stationarity_penalty(self, x: np.ndarray) -> float:
        if self.stationarity_mode is not StationarityMode.SOFT:
            return 0.0  # OFF: none; HARD: guaranteed by the cap
        _, alpha, beta, gamma, _ = (float(v) for v in self._effective_params(x))
        persistence = garch_q_persistence(self.garch_type, alpha, beta, gamma)
        excess = max(persistence - 1.0, 0.0)
        return self.stationarity_weight * excess * excess

    def _build_model(self, params: np.ndarray) -> Any:
        """Construct the variant's model from natural ``(omega, alpha, beta, gamma, h0)``."""
        omega, alpha, beta, gamma, h0 = (float(v) for v in params)
        if self.garch_type == "garch":
            return GARCHRiskNeutralModel(
                omega=omega, alpha=alpha, beta=beta, h0=h0, steps_per_year=_SPY
            )
        if self.garch_type == "gjr_garch":
            return GJRGARCHRiskNeutralModel(
                omega=omega, alpha=alpha, beta=beta, gamma=gamma, h0=h0,
                steps_per_year=_SPY,
            )
        return NGARCHRiskNeutralModel(
            omega=omega, alpha=alpha, beta=beta, gamma=gamma, h0=h0, steps_per_year=_SPY
        )

    # ------------------------------------------------------------------ #
    # Monte-Carlo surface pricing (aligned to market_data.quotes order)
    # ------------------------------------------------------------------ #

    def _price_natural(
        self, params: np.ndarray, market_data: OptionMarketData
    ) -> np.ndarray:
        """Model prices for natural ``(omega, alpha, beta, gamma, h0)``, CRN-seeded.

        Delegates the per-quote surface pricing to the shared
        :func:`backend.calibration.pricing_loop.price_surface_mc` so the calibrator
        and the app's diagnostics share one Monte-Carlo pricing path.
        """
        omega, alpha, beta, gamma, h0 = (float(v) for v in params)
        sim = GARCHRiskNeutralSimulator(
            self.garch_type, omega, alpha, beta, gamma=gamma, h0=h0, steps_per_year=_SPY
        )
        return price_surface_mc(
            sim, market_data, n_paths=self.n_paths, mc_seed=self.mc_seed
        )

    # ------------------------------------------------------------------ #
    # Main calibration
    # ------------------------------------------------------------------ #

    def calibrate(self, market_data: OptionMarketData) -> CalibrationResult:
        t_start = time.perf_counter()
        rng = np.random.default_rng(self.seed)

        # --- ATM-IV seed (per-period scale) ---
        atm_iv = get_atm_iv(market_data)
        v_per = float(np.clip(atm_iv**2 / _SPY, self._lo[4], self._hi[4]))
        beta0, alpha0 = 0.80, 0.04
        gamma0 = float(np.clip(_GAMMA_SEED[self.garch_type], self._lo[3], self._hi[3]))
        persist0 = garch_q_persistence(self.garch_type, alpha0, beta0, gamma0)
        omega0 = float(np.clip(v_per * (1.0 - persist0), self._lo[0], self._hi[0]))
        seed_params = np.array([omega0, alpha0, beta0, gamma0, v_per])

        def obj_fn(x: np.ndarray) -> float:
            model_prices = self._price_natural(self._effective_params(x), market_data)
            loss = float(self._obj_strategy.compute_loss(model_prices, market_data))
            return loss + self._stationarity_penalty(x)

        problem = CalibrationProblem(
            x0=seed_params,
            bounds_lo=self._lo.copy(),
            bounds_hi=self._hi.copy(),
            param_names=tuple(NGARCH_Q_PARAM_NAMES),
            objective_fn=obj_fn,
            param_mapper=lambda x: dict(
                zip(NGARCH_Q_PARAM_NAMES, self._effective_params(x))
            ),
        )

        starts = [seed_params]
        for _ in range(self.n_restarts - 1):
            starts.append(self._random_start(rng))

        best: dict | None = None
        run_infos = []
        for k, x0 in enumerate(starts):
            iter_logger = (
                IterationLogger(problem, on_snapshot=self.iteration_callback)
                if (self.log_iterations or self.iteration_callback)
                else None
            )
            opt_res = self.optimizer.solve(
                replace(problem, x0=np.asarray(x0, dtype=float)),
                logger=iter_logger,
                max_nfev=self.max_nfev,
            )
            run_infos.append(
                {"start": k, "loss": float(opt_res.objective_value),
                 "nfev": opt_res.n_function_evals}
            )
            if best is None or opt_res.objective_value < best["loss"]:
                best = {
                    "loss": float(opt_res.objective_value),
                    "x": np.asarray(opt_res.x_optimal, dtype=float),
                    "nfev": int(opt_res.n_function_evals),
                    "start_index": k,
                    "history": iter_logger.history if iter_logger is not None else (),
                }

        assert best is not None
        eff = self._effective_params(best["x"])
        model = self._build_model(eff)

        # --- Reporting metrics: re-price with more paths for a stable RMSE ---
        report_paths, self.n_paths = self.n_paths, max(self.n_paths, 80_000)
        try:
            model_prices = self._price_natural(eff, market_data)
        finally:
            self.n_paths = report_paths
        rmse_price = compute_rmse_price(model_prices, market_data.market_prices)
        rmse_iv = self._rmse_iv(model_prices, market_data)

        diagnostics = {
            "garch_type": self.garch_type,
            "persistence": model.persistence,
            "is_stationary": model.is_stationary,
            "loss_best": best["loss"],
            "n_restarts": self.n_restarts,
            "best_start_index": best["start_index"],
            "runs": run_infos,
            "atm_iv": atm_iv,
            "objective_name": self._obj_strategy.name,
            "stationarity_mode": self.stationarity_mode.value,
            "mc_paths": self.n_paths,
        }
        if self.log_iterations:
            diagnostics["iteration_history"] = best["history"]

        elapsed = time.perf_counter() - t_start
        logger.info(
            "%s-Q calibration done in %.2fs | RMSE_price=%.6f | RMSE_iv=%.2f bp | "
            "persistence=%.4f | nfev_best=%d",
            self.garch_type, elapsed, rmse_price, rmse_iv, model.persistence, best["nfev"],
        )

        return CalibrationResult(
            model=model,
            objective_value=best["loss"],
            n_iterations=best["nfev"],
            success=True,
            method=f"{self.optimizer.name}_mc_multistart",
            rmse_price=rmse_price,
            rmse_iv=rmse_iv,
            elapsed_seconds=elapsed,
            diagnostics=diagnostics,
            iteration_history=best["history"],
            optimizer_name=self.optimizer.name,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _random_start(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a start within the box (log-uniform for scale params).

        ``gamma`` is uniform on its box; for the symmetric ``garch`` variant the
        box is degenerate ``[0, 0]`` so gamma stays pinned at 0.
        """
        gamma = (
            rng.uniform(self._lo[3], self._hi[3]) if self._hi[3] > self._lo[3] else self._lo[3]
        )
        return np.array([
            10 ** rng.uniform(np.log10(self._lo[0]), np.log10(self._hi[0])),
            10 ** rng.uniform(np.log10(self._lo[1]), np.log10(self._hi[1])),
            rng.uniform(0.5, self._hi[2]),
            gamma,
            10 ** rng.uniform(np.log10(self._lo[4]), np.log10(self._hi[4])),
        ])

    def _rmse_iv(self, model_prices: np.ndarray, market_data: OptionMarketData) -> float:
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
            [q.implied_vol if q.implied_vol is not None else np.nan
             for q in market_data.quotes]
        )
        valid = ~np.isnan(model_ivs) & ~np.isnan(market_ivs)
        return (
            compute_rmse_iv(model_ivs[valid], market_ivs[valid])
            if valid.any()
            else float("nan")
        )


class NGARCHRiskNeutralCalibrator(GARCHRiskNeutralCalibrator):
    """Duan NGARCH-Q surface calibrator (the ``garch_type="ngarch"`` instance)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("garch_type", None)
        super().__init__("ngarch", *args, **kwargs)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    configure_root(level="INFO")
    from backend.calibration.market_data import OptionQuote
    from backend.utils.math import implied_volatility

    true = NGARCHRiskNeutralModel(
        omega=2.0e-6, alpha=0.04, beta=0.80, gamma=0.8, h0=4.0e-5
    )
    spot, rate = 100.0, 0.05
    sim = true.create_simulator()
    strikes = np.linspace(85.0, 115.0, 7)
    maturities = np.array([21, 63, 126, 252]) / 252.0

    quotes: list[OptionQuote] = []
    for T in maturities:
        prices = sim.price_strikes(spot, strikes, rate, float(T), n_paths=200_000,
                                   rng=np.random.default_rng(7))
        for k, p in zip(strikes, prices):
            try:
                iv = implied_volatility(price=float(p), spot=spot, strike=float(k),
                                        time_to_expiry=float(T), rate=rate,
                                        is_call=True, dividend_yield=0.0)
            except (ValueError, RuntimeError):
                iv = None
            quotes.append(OptionQuote(strike=float(k), maturity=float(T), is_call=True,
                                      market_price=float(max(p, 1e-6)), implied_vol=iv))

    md = OptionMarketData(spot=spot, rate=rate, dividend_yield=0.0, quotes=tuple(quotes))
    print(f"Duan NGARCH-Q synthetic surface: {md.n_quotes} quotes")

    cal = NGARCHRiskNeutralCalibrator(n_restarts=3, max_nfev=200, n_paths=40_000)
    result = cal.calibrate(md)
    print(f"\n{result}")
    print(f"  RMSE (price) : {result.rmse_price:.6f}")
    print(f"  persistence  : {result.diagnostics['persistence']:.4f}")
    for name in NGARCH_Q_PARAM_NAMES:
        t_v, c_v = true.get_parameters()[name], result.model.get_parameters()[name]
        err = abs(t_v - c_v) / (abs(t_v) + 1e-12) * 100
        print(f"  {name:6s} true={t_v:11.4e}  cal={c_v:11.4e}  err={err:6.1f}%")
    sys.exit(0 if result.success and (result.rmse_price or 1e9) < 0.5 else 1)
