"""
Heston-Nandi GARCH Calibrator — LM with JAX analytical Jacobian
================================================================

Surface calibration for the risk-neutral Heston & Nandi (2000) GARCH model,
following the exact pattern of :class:`HestonCalibrator`:

1. Reparametrize the 5 bounded parameters ``(omega, alpha, beta, gamma, h0)``
   via sigmoid bijections so the optimizer runs in unconstrained R^5.
2. Build a JIT-compiled residual vector ``(model_price - market_price)`` over
   all quotes using the closed-form characteristic function
   (``heston_nandi_cf_jax``) priced through the differentiable Carr-Madan FFT.
3. Compute the analytical Jacobian via ``jax.jacfwd``.
4. Run multi-start Levenberg-Marquardt (TRF) and keep the best fit.
5. Report Gauss-Newton parameter uncertainty at the optimum.

The variance-stationarity condition ``beta + alpha*gamma^2 < 1`` is handled by a
:class:`StationarityMode` (OFF / SOFT penalty / HARD reparametrisation),
mirroring the Feller control of the Heston/Bates calibrators.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from backend.calibration.base import BaseCalibrator, CalibrationResult
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
from backend.calibration.stationarity import (
    DEFAULT_STATIONARITY_WEIGHT,
    STATIONARITY_STRICT_FACTOR,
    StationarityMode,
    penalty_weight,
    stationarity_capped_gamma,
    stationarity_gamma_to_unit,
)
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
from backend.engines.aad.calibration.fft import JaxFFTGrids, price_call_strikes_jax
from backend.engines.aad.calibration.heston_nandi_cf import heston_nandi_cf_jax
from backend.engines.fft_engine import FFTEngine
from backend.models.heston_nandi import HestonNandiGARCHModel
from backend.utils.constants.calibration import (
    HESTON_NANDI_BOUNDS,
    HESTON_NANDI_PARAM_NAMES,
    HESTON_NANDI_STEPS_PER_YEAR,
)
from backend.utils.logging import configure_root, get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Reparametrization — JAX-compatible sigmoid bijections
# --------------------------------------------------------------------------- #

_OMEGA_LO, _OMEGA_HI = HESTON_NANDI_BOUNDS["omega"]
_ALPHA_LO, _ALPHA_HI = HESTON_NANDI_BOUNDS["alpha"]
_BETA_LO, _BETA_HI = HESTON_NANDI_BOUNDS["beta"]
_GAMMA_LO, _GAMMA_HI = HESTON_NANDI_BOUNDS["gamma"]
_H0_LO, _H0_HI = HESTON_NANDI_BOUNDS["h0"]

_SPY = HESTON_NANDI_STEPS_PER_YEAR


def _hn_theta_to_params(
    theta: jnp.ndarray,
    mode: StationarityMode = StationarityMode.SOFT,
) -> tuple[float, float, float, float, float]:
    """Unconstrained R^5 -> (omega, alpha, beta, gamma, h0) in the box.

    In :attr:`StationarityMode.HARD` gamma is reparametrised so that
    ``beta + alpha*gamma^2 < 1`` holds by construction.
    """
    sg = jax.nn.sigmoid(theta)
    omega = _OMEGA_LO + (_OMEGA_HI - _OMEGA_LO) * sg[0]
    alpha = _ALPHA_LO + (_ALPHA_HI - _ALPHA_LO) * sg[1]
    beta = _BETA_LO + (_BETA_HI - _BETA_LO) * sg[2]
    if mode is StationarityMode.HARD:
        gamma = stationarity_capped_gamma(
            alpha, beta, sg[3], _GAMMA_LO, _GAMMA_HI, xp=jnp
        )
    else:
        gamma = _GAMMA_LO + (_GAMMA_HI - _GAMMA_LO) * sg[3]
    h0 = _H0_LO + (_H0_HI - _H0_LO) * sg[4]
    return omega, alpha, beta, gamma, h0


def _np_theta_to_params(
    theta: np.ndarray,
    mode: StationarityMode = StationarityMode.SOFT,
) -> np.ndarray:
    """Numpy equivalent of :func:`_hn_theta_to_params` (post-opt reporting)."""
    sg = 1.0 / (1.0 + np.exp(-np.asarray(theta, dtype=float)))
    omega = _OMEGA_LO + (_OMEGA_HI - _OMEGA_LO) * sg[0]
    alpha = _ALPHA_LO + (_ALPHA_HI - _ALPHA_LO) * sg[1]
    beta = _BETA_LO + (_BETA_HI - _BETA_LO) * sg[2]
    if mode is StationarityMode.HARD:
        gamma = stationarity_capped_gamma(
            alpha, beta, sg[3], _GAMMA_LO, _GAMMA_HI, xp=np
        )
    else:
        gamma = _GAMMA_LO + (_GAMMA_HI - _GAMMA_LO) * sg[3]
    h0 = _H0_LO + (_H0_HI - _H0_LO) * sg[4]
    return np.array([omega, alpha, beta, gamma, h0])


def _params_to_theta(
    params: np.ndarray,
    mode: StationarityMode = StationarityMode.SOFT,
) -> np.ndarray:
    """Inverse map: constrained params -> unconstrained R^5 seed."""
    omega, alpha, beta, gamma, h0 = (float(x) for x in params)

    def logit(x: float, lo: float, hi: float) -> float:
        frac = (x - lo) / (hi - lo)
        frac = float(np.clip(frac, 1e-8, 1.0 - 1e-8))
        return float(np.log(frac / (1.0 - frac)))

    if mode is StationarityMode.HARD:
        gamma_unit = stationarity_gamma_to_unit(alpha, beta, gamma, _GAMMA_LO, _GAMMA_HI)
        gamma_theta = logit(gamma_unit, 0.0, 1.0)
    else:
        gamma_theta = logit(gamma, _GAMMA_LO, _GAMMA_HI)

    return np.array(
        [
            logit(omega, _OMEGA_LO, _OMEGA_HI),
            logit(alpha, _ALPHA_LO, _ALPHA_HI),
            logit(beta, _BETA_LO, _BETA_HI),
            gamma_theta,
            logit(h0, _H0_LO, _H0_HI),
        ]
    )


def _chain_rule_jacobian(
    theta: np.ndarray,
    mode: StationarityMode = StationarityMode.SOFT,
) -> np.ndarray:
    """Return dp/dtheta_u for each parameter (size-5 vector).

    ``p_i = lo_i + (hi_i - lo_i) * sigmoid(theta_i)`` so
    ``dp_i/dtheta_i = (hi_i - lo_i) * sigmoid(theta_i) * (1 - sigmoid(theta_i))``.
    In HARD mode the gamma span is the capped-interval width (first-order; the
    cross-coupling of the cap on alpha/beta is neglected).
    """
    sg = 1.0 / (1.0 + np.exp(-np.asarray(theta, dtype=float)))
    deriv = sg * (1.0 - sg)
    if mode is StationarityMode.HARD:
        alpha = _ALPHA_LO + (_ALPHA_HI - _ALPHA_LO) * sg[1]
        beta = _BETA_LO + (_BETA_HI - _BETA_LO) * sg[2]
        alpha_safe = max(float(alpha), 1e-12)
        gamma_stat_max = STATIONARITY_STRICT_FACTOR * float(
            ((1.0 - beta) / alpha_safe) ** 0.5
        )
        gamma_upper = min(_GAMMA_HI, gamma_stat_max)
        gamma_lower = min(_GAMMA_LO, gamma_upper)
        gamma_span = gamma_upper - gamma_lower
    else:
        gamma_span = _GAMMA_HI - _GAMMA_LO
    spans = np.array(
        [
            _OMEGA_HI - _OMEGA_LO,
            _ALPHA_HI - _ALPHA_LO,
            _BETA_HI - _BETA_LO,
            gamma_span,
            _H0_HI - _H0_LO,
        ]
    )
    return spans * deriv


# --------------------------------------------------------------------------- #
# Residual function factory
# --------------------------------------------------------------------------- #


@dataclass
class _CompiledResiduals:
    """Bundle of JIT-compiled residual and Jacobian callables."""

    residual: callable
    jacobian: callable
    n_quote_residuals: int  # excludes the final stationarity-penalty element


def _build_residual_fn(
    market_data: OptionMarketData,
    grids: JaxFFTGrids,
    stationarity_weight: float,
    objective: ObjectiveStrategy,
    mode: StationarityMode = StationarityMode.SOFT,
) -> _CompiledResiduals:
    """JIT-compile the residual function for a fixed surface + grid config."""
    spot = float(market_data.spot)
    rate = float(market_data.rate)
    q = float(market_data.dividend_yield)
    unique_mats = [float(T) for T in market_data.unique_maturities]

    strikes_per_T: list[jnp.ndarray] = []
    is_calls_per_T: list[jnp.ndarray] = []
    running = 0
    for T in unique_mats:
        quotes = market_data.quotes_for_maturity(T)
        running += len(quotes)
        strikes_per_T.append(jnp.asarray([qt.strike for qt in quotes], dtype=jnp.float64))
        is_calls_per_T.append(jnp.asarray([qt.is_call for qt in quotes], dtype=jnp.bool_))
    n_quotes = running

    objective_residual_fn = objective.make_jax_residual_fn(market_data)
    eff_weight = penalty_weight(mode, stationarity_weight)

    def _residual_untraced(theta: jnp.ndarray) -> jnp.ndarray:
        omega, alpha, beta, gamma, h0 = _hn_theta_to_params(theta, mode)

        model_prices_blocks: list[jnp.ndarray] = []
        for i, tau in enumerate(unique_mats):
            strikes_T = strikes_per_T[i]
            is_calls_T = is_calls_per_T[i]

            def cf(u, tau=tau):
                return heston_nandi_cf_jax(
                    u, omega, alpha, beta, gamma, h0, tau, rate, _SPY
                )

            call_prices = price_call_strikes_jax(cf, spot, strikes_T, tau, rate, grids)

            disc_spot = spot * jnp.exp(-q * tau)
            disc_k = strikes_T * jnp.exp(-rate * tau)
            put_prices = call_prices - disc_spot + disc_k

            prices_T = jnp.where(is_calls_T, call_prices, put_prices)
            prices_T = jnp.maximum(prices_T, 0.0)
            model_prices_blocks.append(prices_T)

        model_prices = jnp.concatenate(model_prices_blocks)
        quote_residuals = objective_residual_fn(model_prices)

        # Soft stationarity penalty: sqrt(weight) * (beta + alpha*gamma^2 - 1)_+
        persistence = beta + alpha * gamma**2
        stat_res = jnp.where(
            persistence > 1.0,
            jnp.sqrt(eff_weight) * (persistence - 1.0),
            0.0,
        )
        return jnp.concatenate([quote_residuals, jnp.array([stat_res])])

    return _CompiledResiduals(
        residual=jax.jit(_residual_untraced),
        jacobian=jax.jit(jax.jacfwd(_residual_untraced)),
        n_quote_residuals=n_quotes,
    )


# --------------------------------------------------------------------------- #
# Calibrator
# --------------------------------------------------------------------------- #


class HestonNandiGARCHCalibrator(BaseCalibrator):
    """Heston-Nandi GARCH surface calibrator: LM + JAX Jacobian + multi-start."""

    def __init__(
        self,
        n_restarts: int = 5,
        stationarity_weight: float = DEFAULT_STATIONARITY_WEIGHT,
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
        stationarity_mode: StationarityMode | str = StationarityMode.SOFT,
    ) -> None:
        self.n_restarts = max(1, int(n_restarts))
        self.stationarity_weight = float(stationarity_weight)
        self.stationarity_mode = StationarityMode.coerce(stationarity_mode)
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
        self.objective: ObjectiveStrategy = self._resolve_objective(
            objective or PriceMSEObjective()
        )
        self.objective_type = self.objective.name
        self.log_iterations = bool(log_iterations)
        self.iteration_callback = iteration_callback
        self._grids = fft_grids or JaxFFTGrids.build()
        self._engine = FFTEngine()

    @staticmethod
    def _resolve_objective(obj: ObjectiveStrategy) -> ObjectiveStrategy:
        """Coerce JAX-incompatible objectives (iv_mse) into vega_weighted."""
        if obj.jax_compatible:
            return obj
        logger.info(
            "HestonNandiGARCHCalibrator: objective '%s' is not JAX-compatible. "
            "Falling back to 'vega_weighted' (first-order IV approximation).",
            obj.name,
        )
        return VegaWeightedObjective()

    # ------------------------------------------------------------------ #
    # BaseCalibrator contract
    # ------------------------------------------------------------------ #

    def default_bounds(self) -> list[tuple[float, float]]:
        return [
            (_OMEGA_LO, _OMEGA_HI),
            (_ALPHA_LO, _ALPHA_HI),
            (_BETA_LO, _BETA_HI),
            (_GAMMA_LO, _GAMMA_HI),
            (_H0_LO, _H0_HI),
        ]

    def objective(self, params: np.ndarray, market_data: OptionMarketData) -> float:
        r = self.residuals(params, market_data)
        return 0.5 * float(r @ r)

    def residuals(
        self, params: np.ndarray, market_data: OptionMarketData
    ) -> np.ndarray:
        compiled = _build_residual_fn(
            market_data,
            self._grids,
            self.stationarity_weight,
            self.objective,
            self.stationarity_mode,
        )
        theta = _params_to_theta(np.asarray(params, dtype=float), self.stationarity_mode)
        return np.asarray(compiled.residual(jnp.asarray(theta)))

    # ------------------------------------------------------------------ #
    # Main calibration
    # ------------------------------------------------------------------ #

    def calibrate(self, market_data: OptionMarketData) -> CalibrationResult:
        t_start = time.perf_counter()

        compiled = _build_residual_fn(
            market_data,
            self._grids,
            self.stationarity_weight,
            self.objective,
            self.stationarity_mode,
        )

        # --- Initial guess from ATM IV (per-period scale) ---
        atm_iv = get_atm_iv(market_data)
        v_per = atm_iv**2 / _SPY
        beta0, gamma0 = 0.7, 200.0
        alpha0 = 0.2 / gamma0**2  # alpha*gamma^2 ~ 0.2 of persistence
        persist0 = beta0 + alpha0 * gamma0**2
        omega0 = max(v_per * (1.0 - persist0) - alpha0, _OMEGA_LO)
        seed_params = np.array([omega0, alpha0, beta0, gamma0, v_per])
        x0_base = _params_to_theta(seed_params, self.stationarity_mode)
        logger.info(
            "Heston-Nandi calibration | ATM IV=%.2f%% | seed=%s",
            atm_iv * 100.0,
            np.array2string(seed_params, precision=4),
        )

        rng = np.random.default_rng(self.seed)
        starts = make_multi_starts(
            x0_base,
            n_restarts=self.n_restarts,
            near_scale=self.restart_scale_near,
            far_scale=self.restart_scale_far,
            rng=rng,
        )

        def _residual_np(x: np.ndarray) -> np.ndarray:
            return np.asarray(compiled.residual(jnp.asarray(x)))

        def _jacobian_np(x: np.ndarray) -> np.ndarray:
            return np.asarray(compiled.jacobian(jnp.asarray(x)))

        problem = CalibrationProblem(
            x0=x0_base,
            bounds_lo=np.full(5, -15.0),
            bounds_hi=np.full(5, 15.0),
            param_names=tuple(HESTON_NANDI_PARAM_NAMES),
            residual_fn=_residual_np,
            jacobian_fn=_jacobian_np,
            param_mapper=lambda theta_arr: dict(
                zip(
                    HESTON_NANDI_PARAM_NAMES,
                    _np_theta_to_params(theta_arr, self.stationarity_mode),
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
                best = {
                    "rss": rss,
                    "theta": opt_res.x_optimal,
                    "jac": _jacobian_np(opt_res.x_optimal),
                    "fun": _residual_np(opt_res.x_optimal),
                    "nfev": opt_res.n_function_evals,
                    "start_index": k,
                    "history": iter_logger.history if iter_logger is not None else (),
                }

        # --- Build calibrated model ---
        omega, alpha, beta, gamma, h0 = (
            float(x) for x in _np_theta_to_params(best["theta"], self.stationarity_mode)
        )
        try:
            model = HestonNandiGARCHModel(
                omega=omega, alpha=alpha, beta=beta, gamma=gamma, h0=h0
            )
        except ValueError as exc:
            logger.error("Invalid Heston-Nandi params after calibration: %s", exc)
            return CalibrationResult(
                model=HestonNandiGARCHModel(
                    omega=1e-6, alpha=2e-6, beta=0.8, gamma=150.0, h0=4e-5
                ),
                objective_value=best["rss"] / 2.0,
                n_iterations=best["nfev"],
                success=False,
                method=f"{self.optimizer.name}_multistart",
                elapsed_seconds=time.perf_counter() - t_start,
                diagnostics={"error": str(exc), "lm_runs": lm_infos},
                iteration_history=best["history"],
                optimizer_name=self.optimizer.name,
            )

        # --- RMSE metrics via the shared numpy FFT engine ---
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

        objective_loss = self.objective.compute_loss(model_prices, market_data)
        diagnostics = {
            "persistence": model.persistence,
            "is_stationary": model.is_stationary,
            "rss_best": best["rss"],
            "n_restarts": self.n_restarts,
            "best_start_index": best["start_index"],
            "lm_runs": lm_infos,
            "atm_iv": atm_iv,
            "objective_name": self.objective.name,
            "objective_loss": float(objective_loss),
            "stationarity_mode": self.stationarity_mode.value,
        }
        if self.log_iterations:
            diagnostics["multi_start_history"] = tuple(all_iteration_histories)

        if self.compute_uncertainty:
            try:
                J_u = np.asarray(best["jac"])[: compiled.n_quote_residuals, :]
                r_q = np.asarray(best["fun"])[: compiled.n_quote_residuals]
                dpdu = _chain_rule_jacobian(best["theta"], self.stationarity_mode)
                J_constrained = J_u * dpdu[np.newaxis, :]
                unc = least_squares_covariance(J_constrained, r_q)
                diagnostics["uncertainty"] = summary_table(
                    list(HESTON_NANDI_PARAM_NAMES),
                    np.array([omega, alpha, beta, gamma, h0]),
                    unc,
                )
                diagnostics["uncertainty_condition_number"] = unc.condition_number
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.warning("Uncertainty computation failed: %s", exc)
                diagnostics["uncertainty_error"] = str(exc)

        elapsed = time.perf_counter() - t_start
        logger.info(
            "Heston-Nandi calibration done in %.2fs | RMSE_price=%.6f | "
            "RMSE_iv=%.2f bp | persistence=%.4f | nfev_best=%d",
            elapsed,
            rmse_price,
            rmse_iv,
            model.persistence,
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
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    configure_root(level="INFO")

    from backend.calibration.market_data import OptionQuote
    from backend.core.market import MarketEnvironment
    from backend.instruments.options import VanillaOption
    from backend.utils.math import implied_volatility

    true_model = HestonNandiGARCHModel(
        omega=1.0e-6, alpha=2.0e-6, beta=0.80, gamma=150.0, h0=4.0e-5
    )
    spot, rate, div = 100.0, 0.05, 0.0
    market = MarketEnvironment(spot=spot, rate=rate, dividend_yield=div)
    engine = FFTEngine()

    strikes = np.linspace(85.0, 115.0, 9)
    maturities = np.array([21, 63, 126, 189, 252]) / 252.0

    quotes: list[OptionQuote] = []
    for T in maturities:
        tpl = VanillaOption(strike=100.0, maturity=float(T), is_call=True)
        prices = engine.price_strikes(tpl, true_model, market, strikes)
        for k, p in zip(strikes, prices):
            try:
                iv = implied_volatility(
                    price=float(p), spot=spot, strike=float(k),
                    time_to_expiry=float(T), rate=rate, is_call=True,
                    dividend_yield=div,
                )
            except (ValueError, RuntimeError):
                iv = None
            quotes.append(
                OptionQuote(strike=float(k), maturity=float(T), is_call=True,
                            market_price=float(max(p, 0.0)), implied_vol=iv)
            )

    md = OptionMarketData(spot=spot, rate=rate, dividend_yield=div, quotes=tuple(quotes))
    print(f"Heston-Nandi synthetic surface: {md.n_quotes} quotes")

    cal = HestonNandiGARCHCalibrator(n_restarts=3, max_nfev=120)
    result = cal.calibrate(md)
    print(f"\n{result}")
    print(f"  success      : {result.success}")
    print(f"  RMSE (price) : {result.rmse_price:.6f}")
    print(f"  persistence  : {result.diagnostics['persistence']:.4f}")

    cal_params = result.model.get_parameters()
    true_params = true_model.get_parameters()
    print("\nParameter recovery:")
    for name in HESTON_NANDI_PARAM_NAMES:
        t_v, c_v = true_params[name], cal_params[name]
        err = abs(t_v - c_v) / (abs(t_v) + 1e-12) * 100
        print(f"  {name:6s} true={t_v:11.4e}  cal={c_v:11.4e}  err={err:6.1f}%")

    sys.exit(0 if result.success and result.rmse_price < 0.5 else 1)
