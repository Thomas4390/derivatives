"""
GARCH Calibrator — exact gradients via JAX + BHHH std errors
=============================================================

MLE pipeline for GARCH(1,1), NGARCH, and GJR-GARCH with:
  - JAX-implemented NLL + analytical gradient (``jax.grad``)
  - L-BFGS-B driven by the exact gradient → fewer function evaluations,
    faster convergence
  - BHHH standard errors from per-observation score vectors
    (``jax.jacfwd`` of per-obs log-likelihood)

Supports GARCH(1,1), NGARCH, and GJR-GARCH through the standard
``BaseCalibrator`` interface — same ``calibrate`` / ``objective``
contract, same ``HistoricalReturns`` input, same ``CalibrationResult``
output with ``diagnostics`` now carrying an ``uncertainty`` table.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from backend.calibration.base import BaseCalibrator, CalibrationResult
from backend.calibration.market_data import HistoricalReturns
from backend.calibration.optimizers import (
    CalibrationProblem,
    IterationLogger,
    LBFGSStrategy,
    OptimizerStrategy,
)
from backend.calibration.uncertainty import bhhh_covariance, summary_table
from backend.engines.aad.calibration.garch_nll import (
    nll_garch_grad,
    nll_garch_jit,
    nll_gjr_grad,
    nll_gjr_jit,
    nll_ngarch_grad,
    nll_ngarch_jit,
    scores_garch_jit,
    scores_gjr_jit,
    scores_ngarch_jit,
)
from backend.models.garch import GARCHModel, GJRGARCHModel, NGARCHModel
from backend.utils.constants.calibration import (
    GARCH_BOUNDS,
    GJR_BOUNDS,
    NGARCH_BOUNDS,
    VALID_GARCH_TYPES as _VALID_TYPES,
)
from backend.utils.constants.numerical import (
    GARCH_CALIBRATION_VARIANCE_FLOOR as _VARIANCE_FLOOR,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Variance filters (numpy, used only for extracting sigma0 at the optimum)
# --------------------------------------------------------------------------- #


def _filter_variance(
    garch_type: str, params: np.ndarray, returns: np.ndarray
) -> np.ndarray:
    """Run the appropriate variance recursion and return sigma^2_t for t = 0..T."""
    omega, alpha, beta = float(params[0]), float(params[1]), float(params[2])
    T = len(returns)
    var_series = np.empty(T + 1)
    var_series[0] = float(np.var(returns))

    if garch_type == "garch":
        for t in range(T):
            vt = max(var_series[t], _VARIANCE_FLOOR)
            z = returns[t] / np.sqrt(vt)
            var_series[t + 1] = omega + alpha * vt * z * z + beta * vt
    elif garch_type == "ngarch":
        gamma = float(params[3])
        for t in range(T):
            vt = max(var_series[t], _VARIANCE_FLOOR)
            z = returns[t] / np.sqrt(vt)
            var_series[t + 1] = omega + alpha * vt * (z - gamma) ** 2 + beta * vt
    else:  # gjr_garch
        gamma = float(params[3])
        for t in range(T):
            vt = max(var_series[t], _VARIANCE_FLOOR)
            z = returns[t] / np.sqrt(vt)
            indicator = 1.0 if z < 0.0 else 0.0
            var_series[t + 1] = (
                omega + (alpha + gamma * indicator) * vt * z * z + beta * vt
            )
    return var_series


# --------------------------------------------------------------------------- #
# Calibrator
# --------------------------------------------------------------------------- #


class GARCHCalibrator(BaseCalibrator):
    """MLE calibrator for the GARCH family with JAX-driven exact gradients.

    Parameters
    ----------
    garch_type : str
        One of "garch", "ngarch", "gjr_garch".
    method : str
        Currently only "mle" is supported.
    compute_uncertainty : bool
        Whether to compute BHHH standard errors at the MLE.
    """

    def __init__(
        self,
        garch_type: str = "garch",
        method: str = "mle",
        compute_uncertainty: bool = True,
        optimizer: OptimizerStrategy | None = None,
        log_iterations: bool = False,
        iteration_callback=None,
        max_nfev: int = 500,
        param_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        # Per-run search box override ({param: (lo, hi)}). ``None`` -> the
        # canonical default box, so default-bounds runs are unchanged. Bounds
        # are in the same per-period (daily) scale the MLE searches.
        self._param_bounds = param_bounds
        if garch_type not in _VALID_TYPES:
            raise ValueError(
                f"garch_type must be one of {_VALID_TYPES}, got '{garch_type}'"
            )
        if method != "mle":
            raise ValueError(f"Only 'mle' method is supported, got '{method}'")

        self._garch_type = garch_type
        self._method = method
        self.compute_uncertainty = bool(compute_uncertainty)
        # GARCH MLE is a scalar problem — LM-JAX rejects it. Default to L-BFGS-B
        # which accepts an analytical gradient (the JAX one).
        self.optimizer: OptimizerStrategy = optimizer or LBFGSStrategy()
        if self.optimizer.requires_residuals:
            raise ValueError(
                f"GARCH calibration is a scalar MLE problem; "
                f"optimizer '{self.optimizer.name}' requires residuals "
                f"and cannot be used. Choose DE, NM, or L-BFGS-B."
            )
        self.log_iterations = bool(log_iterations)
        self.iteration_callback = iteration_callback
        self.max_nfev = int(max_nfev)

    # ------------------------------------------------------------------ #
    # BaseCalibrator contract
    # ------------------------------------------------------------------ #

    def default_bounds(self) -> list[tuple[float, float]]:
        if self._garch_type == "garch":
            base = GARCH_BOUNDS
        elif self._garch_type == "ngarch":
            base = NGARCH_BOUNDS
        else:
            base = GJR_BOUNDS
        names = self._param_names()
        box = dict(zip(names, base))
        if self._param_bounds:
            box.update(
                {
                    k: (float(v[0]), float(v[1]))
                    for k, v in self._param_bounds.items()
                    if k in box
                }
            )
        return [box[n] for n in names]

    def objective(self, params: np.ndarray, market_data: HistoricalReturns) -> float:
        return float(
            self._nll_jit()(jnp.asarray(params), jnp.asarray(market_data.log_returns))
        )

    def calibrate(self, market_data: HistoricalReturns) -> CalibrationResult:
        returns = jnp.asarray(market_data.log_returns)
        sample_var = market_data.sample_variance
        annualization_factor = market_data.annualization_factor

        x0 = self._initial_guess(sample_var)
        bounds = self.default_bounds()

        nll = self._nll_jit()
        grad = self._grad_jit()

        def f(x):
            return float(nll(jnp.asarray(x), returns))

        def g(x):
            return np.asarray(grad(jnp.asarray(x), returns))

        param_names = self._param_names()
        problem = CalibrationProblem(
            x0=np.asarray(x0, dtype=float),
            bounds_lo=np.array([lo for lo, _ in bounds], dtype=float),
            bounds_hi=np.array([hi for _, hi in bounds], dtype=float),
            param_names=tuple(param_names),
            objective_fn=f,
            gradient_fn=g,
        )

        iter_logger = (
            IterationLogger(problem, on_snapshot=self.iteration_callback)
            if (self.log_iterations or self.iteration_callback)
            else None
        )

        start = time.perf_counter()
        opt_res = self.optimizer.solve(
            problem,
            logger=iter_logger,
            max_nfev=self.max_nfev,
        )
        elapsed = time.perf_counter() - start

        params = np.asarray(opt_res.x_optimal)
        var_series = _filter_variance(
            self._garch_type, params, np.asarray(market_data.log_returns)
        )
        last_var = max(float(var_series[-1]), _VARIANCE_FLOOR)
        sigma0 = np.sqrt(annualization_factor) * np.sqrt(last_var)

        model = self._build_model(params, sigma0, annualization_factor)

        log_likelihood = -float(opt_res.objective_value)
        n = market_data.n_obs
        n_params = len(params)
        aic = 2.0 * n_params - 2.0 * log_likelihood
        bic = n_params * np.log(n) - 2.0 * log_likelihood
        persistence = self._persistence(params)

        diagnostics: dict[str, Any] = {
            "log_likelihood": log_likelihood,
            "aic": aic,
            "bic": bic,
            "persistence": persistence,
            "n_obs": n,
            "n_params": n_params,
            "grad_norm": float(np.linalg.norm(g(params))),
        }
        if self.log_iterations and iter_logger is not None:
            diagnostics["iteration_history"] = iter_logger.history

        if self.compute_uncertainty:
            try:
                scores = np.asarray(self._scores_jit()(jnp.asarray(params), returns))
                # scores shape is (T, n_params) — the Jacobian of per-obs logL
                unc = bhhh_covariance(scores)
                diagnostics["uncertainty"] = summary_table(
                    self._param_names(),
                    params,
                    unc,
                )
                diagnostics["uncertainty_condition_number"] = unc.condition_number
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.warning("BHHH covariance failed: %s", exc)
                diagnostics["uncertainty_error"] = str(exc)

        return CalibrationResult(
            model=model,
            objective_value=float(opt_res.objective_value),
            n_iterations=int(opt_res.n_iterations),
            success=bool(opt_res.success),
            method=f"mle_{self.optimizer.name}_{self._garch_type}",
            elapsed_seconds=elapsed,
            diagnostics=diagnostics,
            iteration_history=iter_logger.history if iter_logger is not None else (),
            optimizer_name=self.optimizer.name,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _nll_jit(self) -> Callable:
        if self._garch_type == "garch":
            return nll_garch_jit
        elif self._garch_type == "ngarch":
            return nll_ngarch_jit
        else:
            return nll_gjr_jit

    def _grad_jit(self) -> Callable:
        if self._garch_type == "garch":
            return nll_garch_grad
        elif self._garch_type == "ngarch":
            return nll_ngarch_grad
        else:
            return nll_gjr_grad

    def _scores_jit(self) -> Callable:
        if self._garch_type == "garch":
            return scores_garch_jit
        elif self._garch_type == "ngarch":
            return scores_ngarch_jit
        else:
            return scores_gjr_jit

    def _initial_guess(self, sample_var: float) -> np.ndarray:
        if self._garch_type == "garch":
            return np.array([sample_var * 0.05, 0.05, 0.90])
        elif self._garch_type == "ngarch":
            return np.array([sample_var * 0.05, 0.05, 0.90, 0.5])
        else:
            return np.array([sample_var * 0.05, 0.05, 0.88, 0.04])

    def _persistence(self, params: np.ndarray) -> float:
        if self._garch_type == "garch":
            return float(params[1] + params[2])
        elif self._garch_type == "ngarch":
            return float(params[1] * (1.0 + params[3] ** 2) + params[2])
        else:
            return float(params[1] + 0.5 * params[3] + params[2])

    def _param_names(self) -> list[str]:
        if self._garch_type == "garch":
            return ["omega", "alpha", "beta"]
        elif self._garch_type == "ngarch":
            return ["omega", "alpha", "beta", "gamma"]
        else:
            return ["omega", "alpha", "beta", "gamma"]

    def _build_model(
        self, params: np.ndarray, sigma0: float, annualization_factor: float
    ) -> GARCHModel | NGARCHModel | GJRGARCHModel:
        # MLE recovers per-period (e.g. daily) ω/α/β from per-period log
        # returns, but the GARCH simulator/pricer recursion is dt-free with
        # ``var0 = sigma0**2`` — so a self-consistent model needs ω and σ₀²
        # on the SAME (annualised) scale. Variance scales linearly with
        # time, hence ω_ann = ω_period × annualization_factor; α/β/θ/γ are
        # dimensionless and unchanged. σ₀ is already annualised by the
        # caller. See plan note: this keeps model.long_run_variance ≈ σ₀².
        omega = float(params[0]) * annualization_factor
        alpha, beta = float(params[1]), float(params[2])
        if self._garch_type == "garch":
            return GARCHModel(sigma0=sigma0, omega=omega, alpha=alpha, beta=beta)
        elif self._garch_type == "ngarch":
            return NGARCHModel(
                sigma0=sigma0,
                omega=omega,
                alpha=alpha,
                beta=beta,
                gamma=float(params[3]),
            )
        else:
            return GJRGARCHModel(
                sigma0=sigma0,
                omega=omega,
                alpha=alpha,
                beta=beta,
                gamma=float(params[3]),
            )

    @property
    def garch_type(self) -> str:
        return self._garch_type

    def __repr__(self) -> str:
        return f"GARCHCalibrator(type={self._garch_type}, method={self._method})"


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    from backend.simulation.models.garch import GARCHSimulator

    # Generate 10 years of daily GARCH returns
    true_sigma0 = 0.20
    true_omega = 2e-6
    true_alpha = 0.08
    true_beta = 0.90

    sim = GARCHSimulator(
        sigma0=true_sigma0, omega=true_omega, alpha=true_alpha, beta=true_beta
    )
    res = sim.simulate_paths(
        s0=100.0, mu=0.05, t=10.0, n_paths=1, n_steps=2520, seed=42
    )
    prices = res.price_paths[0, :]
    log_returns = np.diff(np.log(prices))
    md = HistoricalReturns(
        log_returns=log_returns, frequency="daily", annualization_factor=252
    )

    print(f"GARCH sample size: {md.n_obs} observations")
    print(f"Sample vol: {md.sample_volatility:.2%}")

    for variant in ["garch", "ngarch", "gjr_garch"]:
        print(f"\n--- {variant.upper()} ---")
        cal = GARCHCalibrator(garch_type=variant)
        result = cal.calibrate(md)
        est = dict(result.model.get_parameters())
        # The model stores ω on the annualised scale; show it back in the
        # per-period units of the input (true_omega) so recovery is legible.
        est["omega"] = est["omega"] / md.annualization_factor
        print(
            f"  elapsed: {result.elapsed_seconds:.3f}s | log-L: {result.diagnostics['log_likelihood']:.2f} | iter: {result.n_iterations}"
        )
        print(
            f"  params (ω per-period): {dict((k, round(v, 8)) for k, v in est.items())}"
        )
        if "uncertainty" in result.diagnostics:
            print("  BHHH std errors:")
            for name, stats in result.diagnostics["uncertainty"].items():
                print(
                    f"    {name:8s} SE={stats['std_error']:.3e}  95% CI=[{stats['ci_lo']:.4e}, {stats['ci_hi']:.4e}]"
                )

    sys.exit(0)
