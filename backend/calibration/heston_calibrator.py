"""
Heston Calibrator — Levenberg-Marquardt with JAX analytical Jacobian
=====================================================================

Industry-standard non-linear least squares calibration for Heston:

1. Reparametrize Heston's bounded parameters via sigmoid bijections so
   the optimizer runs in unconstrained R^5.
2. Build a JIT-compiled residual function returning the vector of
   (model_price - market_price) across all quotes.
3. Compute an analytical Jacobian via `jax.jacfwd` — exact, no finite
   differences.
4. Run `scipy.optimize.least_squares` with Trust-Region-Reflective
   (Levenberg-Marquardt variant that respects optional bounds) from
   multiple random starts, keep the best solution.
5. Extract parameter uncertainty from the Gauss-Newton covariance
   (``(J^T J)^{-1} * RSS / DoF``) and expose it in the result.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from backend.calibration.base import BaseCalibrator, CalibrationResult
from backend.calibration.feller import (
    DEFAULT_FELLER_WEIGHT,
    FELLER_STRICT_FACTOR,
    FellerMode,
    feller_capped_xi,
    feller_xi_to_unit,
    penalty_weight,
)
from backend.calibration.lm_helpers import make_multi_starts
from backend.calibration.market_data import OptionMarketData
from backend.calibration.objectives import (
    ObjectiveStrategy,
    PriceMSEObjective,
    VegaWeightedObjective,
)
from backend.calibration.optimizers import (
    CalibrationProblem,
    IterationLogger,
    LMJaxStrategy,
    OptimizerStrategy,
)
from backend.calibration.pricing_loop import price_surface
from backend.calibration.uncertainty import (
    least_squares_covariance,
    summary_table,
)
from backend.calibration.utils import (
    compute_rmse_iv,
    compute_rmse_price,
    get_atm_iv,
    model_prices_to_ivs,
)
from backend.engines.aad.calibration.fft import (
    JaxFFTGrids,
    price_call_strikes_jax,
)
from backend.engines.aad.calibration.heston_cf import heston_cf_jax
from backend.engines.fft_engine import FFTEngine
from backend.models.heston import HestonModel
from backend.utils.constants.calibration import HESTON_PARAM_NAMES

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Reparametrization — JAX-compatible sigmoid bijections
# --------------------------------------------------------------------------- #

_V0_LO, _V0_HI = 1e-5, 1.0
_KAPPA_LO, _KAPPA_HI = 0.01, 20.0
_THETA_LO, _THETA_HI = 1e-5, 1.0
_XI_LO, _XI_HI = 1e-3, 2.0
_RHO_LO, _RHO_HI = -0.999, 0.999


def _heston_theta_to_params(
    theta: jnp.ndarray,
    feller_mode: FellerMode = FellerMode.SOFT,
) -> tuple[float, float, float, float, float]:
    """Unconstrained R^5 -> (v0, kappa, theta, alpha, rho) in the admissible region.

    In :attr:`FellerMode.HARD` the vol-of-vol is reparametrised so that
    ``alpha <= sqrt(2*kappa*theta)`` holds by construction (Feller guaranteed).
    """
    sg = jax.nn.sigmoid(theta)
    v0 = _V0_LO + (_V0_HI - _V0_LO) * sg[0]
    kappa = _KAPPA_LO + (_KAPPA_HI - _KAPPA_LO) * sg[1]
    theta_h = _THETA_LO + (_THETA_HI - _THETA_LO) * sg[2]
    if feller_mode is FellerMode.HARD:
        alpha = feller_capped_xi(kappa, theta_h, sg[3], _XI_LO, _XI_HI, xp=jnp)
    else:
        alpha = _XI_LO + (_XI_HI - _XI_LO) * sg[3]
    rho = _RHO_LO + (_RHO_HI - _RHO_LO) * sg[4]
    return v0, kappa, theta_h, alpha, rho


def _params_to_theta(
    params: np.ndarray,
    feller_mode: FellerMode = FellerMode.SOFT,
) -> np.ndarray:
    """Inverse: constrained params -> unconstrained R^5.

    For :attr:`FellerMode.HARD` the alpha coordinate is seeded through the
    capped reparametrisation so the start point matches the forward map.
    """
    v0, kappa, theta_h, alpha, rho = [float(x) for x in params]

    def logit(x: float, lo: float, hi: float) -> float:
        frac = (x - lo) / (hi - lo)
        frac = float(np.clip(frac, 1e-8, 1.0 - 1e-8))
        return float(np.log(frac / (1.0 - frac)))

    if feller_mode is FellerMode.HARD:
        xi_unit = feller_xi_to_unit(kappa, theta_h, alpha, _XI_LO, _XI_HI)
        xi_theta = logit(xi_unit, 0.0, 1.0)
    else:
        xi_theta = logit(alpha, _XI_LO, _XI_HI)

    return np.array(
        [
            logit(v0, _V0_LO, _V0_HI),
            logit(kappa, _KAPPA_LO, _KAPPA_HI),
            logit(theta_h, _THETA_LO, _THETA_HI),
            xi_theta,
            logit(rho, _RHO_LO, _RHO_HI),
        ]
    )


# --------------------------------------------------------------------------- #
# Residual function factory
# --------------------------------------------------------------------------- #


@dataclass
class _CompiledResiduals:
    """Bundle of JIT-compiled residual and Jacobian callables."""

    residual: callable
    jacobian: callable
    n_quote_residuals: int  # excludes the final Feller penalty element


def _build_residual_fn(
    market_data: OptionMarketData,
    grids: JaxFFTGrids,
    feller_weight: float,
    objective: ObjectiveStrategy,
    feller_mode: FellerMode = FellerMode.SOFT,
) -> _CompiledResiduals:
    """JIT-compile the residual function for a fixed surface + grid config.

    The per-quote residual transformation is delegated to ``objective``
    via :meth:`ObjectiveStrategy.make_jax_residual_fn`, which closes over
    any precomputed weights (vegas, spreads, market prices, ...). The
    final residual vector is ``[obj.transform(model_prices), feller_pen]``
    where the optimizer minimises ``½‖r‖²``.
    """
    spot = float(market_data.spot)
    rate = float(market_data.rate)
    q = float(market_data.dividend_yield)
    unique_mats = [float(T) for T in market_data.unique_maturities]

    strikes_per_T: list[jnp.ndarray] = []
    is_calls_per_T: list[jnp.ndarray] = []
    quote_offsets: list[tuple[int, int]] = []
    running = 0
    for T in unique_mats:
        quotes = market_data.quotes_for_maturity(T)
        n = len(quotes)
        strikes_per_T.append(jnp.asarray([q.strike for q in quotes], dtype=jnp.float64))
        is_calls_per_T.append(jnp.asarray([q.is_call for q in quotes], dtype=jnp.bool_))
        quote_offsets.append((running, running + n))
        running += n
    n_quotes = running

    # Strategy-driven residual closure : sqrt(w_i) * (model - market) for
    # linear-weight objectives, custom logic for Huber/relative.
    objective_residual_fn = objective.make_jax_residual_fn(market_data)

    # OFF -> 0 (no penalty); SOFT -> feller_weight; HARD -> 0 (Feller is
    # guaranteed by the capped reparametrisation, the residual stays 0 and
    # is kept only to keep the vector shape constant across modes).
    eff_feller_weight = penalty_weight(feller_mode, feller_weight)

    def _residual_untraced(theta: jnp.ndarray) -> jnp.ndarray:
        v0, kappa, theta_h, alpha, rho = _heston_theta_to_params(theta, feller_mode)

        model_prices_blocks: list[jnp.ndarray] = []
        for i, tau in enumerate(unique_mats):
            strikes_T = strikes_per_T[i]
            is_calls_T = is_calls_per_T[i]

            def cf(u):
                return heston_cf_jax(u, v0, kappa, theta_h, alpha, rho, tau, rate, q)

            call_prices = price_call_strikes_jax(cf, spot, strikes_T, tau, rate, grids)

            # Put via parity: P = C - S*exp(-qT) + K*exp(-rT)
            disc_spot = spot * jnp.exp(-q * tau)
            disc_k = strikes_T * jnp.exp(-rate * tau)
            put_prices = call_prices - disc_spot + disc_k

            prices_T = jnp.where(is_calls_T, call_prices, put_prices)
            prices_T = jnp.maximum(prices_T, 0.0)
            model_prices_blocks.append(prices_T)

        model_prices = jnp.concatenate(model_prices_blocks)
        quote_residuals = objective_residual_fn(model_prices)

        # Soft Feller penalty as an extra residual: sqrt(weight) * (alpha^2 - 2*kappa*theta)_+
        feller = 2.0 * kappa * theta_h - alpha**2
        feller_res = jnp.where(
            feller < 0.0,
            jnp.sqrt(eff_feller_weight) * (-feller),
            0.0,
        )
        return jnp.concatenate([quote_residuals, jnp.array([feller_res])])

    residual_jit = jax.jit(_residual_untraced)
    jacobian_jit = jax.jit(jax.jacfwd(_residual_untraced))

    return _CompiledResiduals(
        residual=residual_jit,
        jacobian=jacobian_jit,
        n_quote_residuals=n_quotes,
    )


# --------------------------------------------------------------------------- #
# Calibrator
# --------------------------------------------------------------------------- #


class HestonCalibrator(BaseCalibrator):
    """Industry-standard Heston calibrator: LM + JAX Jacobian + multi-start.

    Parameters
    ----------
    n_restarts : int
        Number of independent LM runs from perturbed initial guesses.
        Best result (lowest RSS) is returned.
    feller_mode : FellerMode
        How the Feller condition ``2*kappa*theta > alpha**2`` is enforced:
        ``OFF`` (no penalty), ``SOFT`` (penalty with ``feller_weight`` —
        default, legacy behaviour) or ``HARD`` (guaranteed by a capped
        reparametrisation of alpha).
    feller_weight : float
        Soft penalty weight for Feller violation, added as an extra
        residual term. Only consumed in :attr:`FellerMode.SOFT`.
    max_nfev : int
        Maximum number of residual evaluations per LM run.
    ftol, xtol, gtol : float
        LM convergence tolerances.
    seed : int
        Random seed for generating multi-start perturbations.
    compute_uncertainty : bool
        If True, compute Gauss-Newton std errors at the optimum and
        store them in ``result.diagnostics["uncertainty"]``.
    """

    def __init__(
        self,
        n_restarts: int = 5,
        feller_weight: float = DEFAULT_FELLER_WEIGHT,
        max_nfev: int = 200,
        ftol: float = 1e-10,
        xtol: float = 1e-10,
        gtol: float = 1e-10,
        seed: int = 42,
        compute_uncertainty: bool = True,
        fft_grids: JaxFFTGrids | None = None,
        restart_scale_near: float = 1.0,
        restart_scale_far: float = 2.5,
        optimizer: OptimizerStrategy | None = None,
        objective: ObjectiveStrategy | None = None,
        log_iterations: bool = False,
        iteration_callback=None,
        feller_mode: FellerMode | str = FellerMode.SOFT,
    ) -> None:
        self.n_restarts = max(1, int(n_restarts))
        self.feller_weight = float(feller_weight)
        self.feller_mode = FellerMode.coerce(feller_mode)
        self.max_nfev = int(max_nfev)
        self.ftol = float(ftol)
        self.xtol = float(xtol)
        self.gtol = float(gtol)
        self.seed = int(seed)
        self.compute_uncertainty = bool(compute_uncertainty)
        self.restart_scale_near = float(restart_scale_near)
        self.restart_scale_far = float(restart_scale_far)
        self.optimizer: OptimizerStrategy = optimizer or LMJaxStrategy(
            ftol=ftol, xtol=xtol, gtol=gtol
        )

        # Resolve objective with JAX-compatibility fallback. LM-JAX cannot
        # trace through scipy-based IV inversion, so iv_mse silently falls
        # back to vega_weighted (Cont & Tankov first-order approximation).
        self.objective: ObjectiveStrategy = self._resolve_objective(
            objective or PriceMSEObjective()
        )
        # Reporting-only label retained for backwards compatibility with the
        # legacy ``objective_type`` attribute referenced by external scripts.
        self.objective_type = self.objective.name

        self.log_iterations = bool(log_iterations)
        self.iteration_callback = iteration_callback

        self._grids = fft_grids or JaxFFTGrids.build()
        self._engine = FFTEngine()

    @staticmethod
    def _resolve_objective(obj: ObjectiveStrategy) -> ObjectiveStrategy:
        """Coerce JAX-incompatible objectives into a tractable fallback.

        ``LM-JAX`` is the production solver and traces the residual
        through ``jax.jacfwd``. Objectives requiring scipy IV-inversion
        (``iv_mse``) cannot be JIT-compiled — we fall back to
        ``vega_weighted``, whose residuals are the first-order Taylor
        expansion of IV residuals around the market vega. The warning
        is logged at INFO level so it shows up in the Streamlit log
        panel without being noisy.
        """
        if obj.jax_compatible:
            return obj
        logger.info(
            "HestonCalibrator: objective '%s' is not JAX-compatible. "
            "Falling back to 'vega_weighted' (first-order IV approximation, "
            "Cont & Tankov 2004).",
            obj.name,
        )
        return VegaWeightedObjective()

    # ------------------------------------------------------------------ #
    # BaseCalibrator contract
    # ------------------------------------------------------------------ #

    def default_bounds(self) -> list[tuple[float, float]]:
        return [
            (_V0_LO, _V0_HI),
            (_KAPPA_LO, _KAPPA_HI),
            (_THETA_LO, _THETA_HI),
            (_XI_LO, _XI_HI),
            (_RHO_LO, _RHO_HI),
        ]

    def objective(self, params: np.ndarray, market_data: OptionMarketData) -> float:
        r = self.residuals(params, market_data)
        return 0.5 * float(r @ r)

    def residuals(
        self, params: np.ndarray, market_data: OptionMarketData
    ) -> np.ndarray:
        """Per-quote residual vector shaped by the chosen objective + Feller tail."""
        compiled = _build_residual_fn(
            market_data,
            self._grids,
            self.feller_weight,
            self.objective,
            self.feller_mode,
        )
        theta = _params_to_theta(np.asarray(params, dtype=float), self.feller_mode)
        return np.asarray(compiled.residual(jnp.asarray(theta)))

    # ------------------------------------------------------------------ #
    # Main calibration
    # ------------------------------------------------------------------ #

    def calibrate(self, market_data: OptionMarketData) -> CalibrationResult:
        from dataclasses import replace

        t_start = time.perf_counter()

        compiled = _build_residual_fn(
            market_data,
            self._grids,
            self.feller_weight,
            self.objective,
            self.feller_mode,
        )

        # --- Initial guess from ATM IV ---
        atm_iv = get_atm_iv(market_data)
        seed_params = np.array([atm_iv**2, 2.0, atm_iv**2, 0.3, -0.7])
        x0_base = _params_to_theta(seed_params, self.feller_mode)
        logger.info(
            "Heston calibration | ATM IV=%.2f%% | seed_params=%s",
            atm_iv * 100.0,
            np.round(seed_params, 4),
        )

        rng = np.random.default_rng(self.seed)
        starts = make_multi_starts(
            x0_base,
            n_restarts=self.n_restarts,
            near_scale=self.restart_scale_near,
            far_scale=self.restart_scale_far,
            rng=rng,
        )

        # Build the unified CalibrationProblem driving every restart.
        def _residual_np(x: np.ndarray) -> np.ndarray:
            return np.asarray(compiled.residual(jnp.asarray(x)))

        def _jacobian_np(x: np.ndarray) -> np.ndarray:
            return np.asarray(compiled.jacobian(jnp.asarray(x)))

        # Internal coords are unconstrained R^5 via sigmoid bijection.
        # We pass a wide finite box so global solvers (DE) accept the
        # problem; sigmoid saturates beyond ±10 so this loses no
        # admissible region in practice.
        problem = CalibrationProblem(
            x0=x0_base,
            bounds_lo=np.full(5, -15.0),
            bounds_hi=np.full(5, 15.0),
            param_names=tuple(HESTON_PARAM_NAMES),
            residual_fn=_residual_np,
            jacobian_fn=_jacobian_np,
            param_mapper=lambda theta_arr: dict(
                zip(
                    HESTON_PARAM_NAMES,
                    _np_theta_to_params(theta_arr, self.feller_mode),
                )
            ),
        )

        best = None
        lm_infos = []
        all_iteration_histories: list[tuple] = []
        for k, x0 in enumerate(starts):
            iter_logger = (
                IterationLogger(problem, on_snapshot=self.iteration_callback)
                if (self.log_iterations or self.iteration_callback)
                else None
            )
            opt_res = self.optimizer.solve(
                replace(problem, x0=x0),
                logger=iter_logger,
                max_nfev=self.max_nfev,
            )
            rss = 2.0 * float(opt_res.objective_value)
            lm_infos.append(
                {
                    "start": k,
                    "rss": rss,
                    "nfev": opt_res.n_function_evals,
                    "status": int(getattr(opt_res.raw_result, "status", 1)),
                    "message": opt_res.message,
                }
            )
            if iter_logger is not None:
                all_iteration_histories.append(iter_logger.history)

            if best is None or rss < best["rss"]:
                # Always compute jac/fun at best.x for uncertainty (works
                # for all optimisers, not just LM-based ones).
                jac_at_best = _jacobian_np(opt_res.x_optimal)
                fun_at_best = _residual_np(opt_res.x_optimal)
                best = {
                    "rss": rss,
                    "theta": opt_res.x_optimal,
                    "jac": jac_at_best,
                    "fun": fun_at_best,
                    "nfev": opt_res.n_function_evals,
                    "start_index": k,
                    "history": iter_logger.history if iter_logger is not None else (),
                }

        # --- Build calibrated model ---
        v0, kappa, theta_h, alpha, rho = (
            float(x) for x in _np_theta_to_params(best["theta"], self.feller_mode)
        )
        try:
            model = HestonModel(v0=v0, kappa=kappa, theta=theta_h, alpha=alpha, rho=rho)
        except ValueError as exc:
            logger.error("Invalid Heston params after calibration: %s", exc)
            err_diag = {"error": str(exc), "lm_runs": lm_infos}
            if self.log_iterations:
                err_diag["multi_start_history"] = tuple(all_iteration_histories)
            return CalibrationResult(
                model=HestonModel(v0=0.04, kappa=2.0, theta=0.04, alpha=0.3, rho=-0.7),
                objective_value=best["rss"] / 2.0,
                n_iterations=best["nfev"],
                success=False,
                method=f"{self.optimizer.name}_multistart",
                elapsed_seconds=time.perf_counter() - t_start,
                diagnostics=err_diag,
                iteration_history=best["history"],
                optimizer_name=self.optimizer.name,
            )

        # --- RMSE metrics via the shared numpy engine (consistent with the reference numpy engine) ---
        model_prices = price_surface(model, market_data, self._engine)
        rmse_price = compute_rmse_price(model_prices, market_data.market_prices)

        is_calls = np.array([qt.is_call for qt in market_data.quotes])
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
                qt.implied_vol if qt.implied_vol is not None else np.nan
                for qt in market_data.quotes
            ]
        )
        valid = ~np.isnan(model_ivs) & ~np.isnan(market_ivs)
        rmse_iv = (
            compute_rmse_iv(model_ivs[valid], market_ivs[valid])
            if valid.any()
            else float("nan")
        )

        # --- Uncertainty ---
        objective_loss = self.objective.compute_loss(model_prices, market_data)
        diagnostics = {
            "feller_ratio": model.feller_ratio,
            "rss_best": best["rss"],
            "n_restarts": self.n_restarts,
            "best_start_index": best["start_index"],
            "lm_runs": lm_infos,
            "atm_iv": atm_iv,
            "objective_name": self.objective.name,
            "objective_loss": float(objective_loss),
        }
        if self.log_iterations:
            diagnostics["multi_start_history"] = tuple(all_iteration_histories)

        if self.compute_uncertainty:
            try:
                # Strip the Feller-penalty row and map from unconstrained-space
                # Jacobian back to constrained via the chain rule:
                #   dp/dtheta_u = (hi - lo) * sigma(theta_u) * (1 - sigma(theta_u))
                J_u = np.asarray(best["jac"])[: compiled.n_quote_residuals, :]
                r_q = np.asarray(best["fun"])[: compiled.n_quote_residuals]

                dpdu = _chain_rule_jacobian(best["theta"], self.feller_mode)
                J_constrained = J_u * dpdu[np.newaxis, :]

                unc = least_squares_covariance(J_constrained, r_q)
                diagnostics["uncertainty"] = summary_table(
                    list(HESTON_PARAM_NAMES),
                    np.array([v0, kappa, theta_h, alpha, rho]),
                    unc,
                )
                diagnostics["uncertainty_condition_number"] = unc.condition_number
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.warning("Uncertainty computation failed: %s", exc)
                diagnostics["uncertainty_error"] = str(exc)

        elapsed = time.perf_counter() - t_start
        logger.info(
            "Calibration done in %.2fs | RMSE_price=%.6f | RMSE_iv=%.2f bp | nfev_best=%d",
            elapsed,
            rmse_price,
            rmse_iv,
            best["nfev"],
        )

        return CalibrationResult(
            model=model,
            objective_value=best["rss"] / 2.0,
            n_iterations=best["nfev"],
            success=True,
            method=f"{self.optimizer.name}_multistart",
            rmse_price=rmse_price,
            rmse_iv=rmse_iv,
            elapsed_seconds=elapsed,
            diagnostics=diagnostics,
            iteration_history=best["history"],
            optimizer_name=self.optimizer.name,
        )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _np_theta_to_params(
    theta: np.ndarray,
    feller_mode: FellerMode = FellerMode.SOFT,
) -> np.ndarray:
    """Numpy equivalent of _heston_theta_to_params (for post-opt reporting)."""
    sg = 1.0 / (1.0 + np.exp(-np.asarray(theta, dtype=float)))
    v0 = _V0_LO + (_V0_HI - _V0_LO) * sg[0]
    kappa = _KAPPA_LO + (_KAPPA_HI - _KAPPA_LO) * sg[1]
    theta_h = _THETA_LO + (_THETA_HI - _THETA_LO) * sg[2]
    if feller_mode is FellerMode.HARD:
        alpha = feller_capped_xi(kappa, theta_h, sg[3], _XI_LO, _XI_HI, xp=np)
    else:
        alpha = _XI_LO + (_XI_HI - _XI_LO) * sg[3]
    rho = _RHO_LO + (_RHO_HI - _RHO_LO) * sg[4]
    return np.array([v0, kappa, theta_h, alpha, rho])


def _chain_rule_jacobian(
    theta: np.ndarray,
    feller_mode: FellerMode = FellerMode.SOFT,
) -> np.ndarray:
    """Return dp/dtheta_u for each parameter (size-5 vector).

    p_i = lo_i + (hi_i - lo_i) * sigmoid(theta_i)
    dp_i/dtheta_i = (hi_i - lo_i) * sigmoid(theta_i) * (1 - sigmoid(theta_i))

    In :attr:`FellerMode.HARD` the alpha span is the capped interval width
    ``min(xi_hi, factor*sqrt(2*kappa*theta)) - min(xi_lo, ...)``. Only the
    diagonal term ``dxi/dtheta_3`` is kept; the (small) cross-coupling of the
    cap on ``kappa``/``theta`` is neglected, so the reported alpha std-error is a
    first-order approximation in HARD mode.
    """
    sg = 1.0 / (1.0 + np.exp(-np.asarray(theta, dtype=float)))
    deriv = sg * (1.0 - sg)
    if feller_mode is FellerMode.HARD:
        kappa = _KAPPA_LO + (_KAPPA_HI - _KAPPA_LO) * sg[1]
        theta_h = _THETA_LO + (_THETA_HI - _THETA_LO) * sg[2]
        xi_feller_max = FELLER_STRICT_FACTOR * float((2.0 * kappa * theta_h) ** 0.5)
        xi_upper = min(_XI_HI, xi_feller_max)
        xi_lower = min(_XI_LO, xi_upper)
        xi_span = xi_upper - xi_lower
    else:
        xi_span = _XI_HI - _XI_LO
    spans = np.array(
        [
            _V0_HI - _V0_LO,
            _KAPPA_HI - _KAPPA_LO,
            _THETA_HI - _THETA_LO,
            xi_span,
            _RHO_HI - _RHO_LO,
        ]
    )
    return spans * deriv


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    from backend.calibration.market_data import OptionQuote
    from backend.core.market import MarketEnvironment
    from backend.instruments.options import VanillaOption
    from backend.utils.math import implied_volatility

    # Build BCC1997 surface
    true_model = HestonModel(v0=0.04, kappa=1.5, theta=0.04, alpha=0.3, rho=-0.6)
    spot, rate, div = 100.0, 0.05, 0.0
    market = MarketEnvironment(spot=spot, rate=rate, dividend_yield=div)
    engine = FFTEngine()

    strikes = np.linspace(80.0, 120.0, 11)
    maturities = np.linspace(1.0 / 12.0, 2.0, 6)

    quotes: list[OptionQuote] = []
    for T in maturities:
        tpl = VanillaOption(strike=100.0, maturity=float(T), is_call=True)
        prices = engine.price_strikes(tpl, true_model, market, strikes)
        for k, p in zip(strikes, prices):
            try:
                iv = implied_volatility(
                    price=float(p),
                    spot=spot,
                    strike=float(k),
                    time_to_expiry=float(T),
                    rate=rate,
                    is_call=True,
                    dividend_yield=div,
                )
            except (ValueError, RuntimeError):
                iv = None
            quotes.append(
                OptionQuote(
                    strike=float(k),
                    maturity=float(T),
                    is_call=True,
                    market_price=float(max(p, 0.0)),
                    implied_vol=iv,
                )
            )

    md = OptionMarketData(
        spot=spot, rate=rate, dividend_yield=div, quotes=tuple(quotes)
    )
    print(f"BCC1997 surface: {md.n_quotes} quotes")

    cal = HestonCalibrator(n_restarts=3, max_nfev=100)
    result = cal.calibrate(md)

    print(f"\n{result}")
    print(f"  success       : {result.success}")
    print(f"  elapsed       : {result.elapsed_seconds:.2f}s")
    print(f"  RMSE (price)  : {result.rmse_price:.6f}")
    print(f"  RMSE (IV bp)  : {result.rmse_iv:.2f}")
    print(f"  nfev_best     : {result.n_iterations}")
    print(f"  feller_ratio  : {result.diagnostics['feller_ratio']:.3f}")

    cal_params = result.model.get_parameters()
    true_params = true_model.get_parameters()
    print("\nParameter recovery:")
    for name in HESTON_PARAM_NAMES:
        t_v, c_v = true_params[name], cal_params[name]
        err = abs(t_v - c_v) / (abs(t_v) + 1e-12) * 100
        print(f"  {name:6s} true={t_v:8.4f}  cal={c_v:8.4f}  err={err:5.2f}%")

    if "uncertainty" in result.diagnostics:
        print("\nStandard errors (Gauss-Newton):")
        for name in HESTON_PARAM_NAMES:
            s = result.diagnostics["uncertainty"][name]
            print(
                f"  {name:6s} estimate={s['estimate']:8.5f}  SE={s['std_error']:8.5f}  "
                f"95% CI=[{s['ci_lo']:8.5f}, {s['ci_hi']:8.5f}]"
            )

    sys.exit(0 if result.success and result.rmse_price < 0.1 else 1)
