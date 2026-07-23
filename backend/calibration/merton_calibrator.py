"""
Merton Calibrator — Levenberg-Marquardt with JAX analytical Jacobian
=====================================================================

Follows the same architecture as ``HestonCalibrator`` (LM + JAX Jacobian +
multi-start + Gauss-Newton uncertainty) adapted for the four-parameter
Merton jump-diffusion model: ``[sigma, lam, alpha_j, sigma_j]``.

Extras specific to Merton
-------------------------
The diffusion volatility ``sigma`` and the jump parameters are known to
suffer from identifiability issues: diffusion vol can absorb part of
the jump-induced smile curvature. We carry a Tikhonov regularization
that pulls ``sigma`` toward a prior (ATM IV by default) implemented
as an additional residual term so it fits naturally into the LM
residual vector.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

from backend.calibration._reparam import (
    _CompiledResiduals,
    build_cf_residual_fn,
    logit,
)
from backend.calibration.base import BaseCalibrator, CalibrationResult
from backend.calibration.lm_helpers import make_multi_starts
from backend.calibration.market_data import OptionMarketData
from backend.calibration.search_space import MERTON_SEARCH_BOUNDS
from backend.calibration.objectives import (
    ObjectiveStrategy,
    PriceMSEObjective,
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
from backend.engines.aad.calibration.merton_cf import merton_cf_jax
from backend.engines.fft_engine import FFTEngine
from backend.models.merton import MertonModel
from backend.utils.constants.calibration import MERTON_PARAM_NAMES

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Reparametrization for Merton [sigma, lam, alpha_j, sigma_j]
# --------------------------------------------------------------------------- #

# Canonical (default) admissible box in MERTON_PARAM_NAMES order. Chosen to
# match realistic equity-index jump regimes and to prevent the jump
# compensator k = exp(alpha_j + 0.5*sigma_j^2) - 1 from exploding at extreme
# far-field multi-start points. The search universe is configurable per-run
# (see ``param_bounds``); sourced from ``backend.calibration.search_space``.
Bounds = tuple[tuple[float, float], ...]
_MERTON_BOUNDS: Bounds = tuple(MERTON_SEARCH_BOUNDS[n] for n in MERTON_PARAM_NAMES)


def _resolve_bounds(param_bounds: dict[str, tuple[float, float]] | None) -> Bounds:
    """Map a ``{param: (lo, hi)}`` override onto the ordered Merton box."""
    if not param_bounds:
        return _MERTON_BOUNDS
    return tuple(
        (float(param_bounds[n][0]), float(param_bounds[n][1]))
        if n in param_bounds
        else _MERTON_BOUNDS[i]
        for i, n in enumerate(MERTON_PARAM_NAMES)
    )


def _merton_theta_to_params(
    theta: jnp.ndarray, bounds: Bounds = _MERTON_BOUNDS
) -> tuple[float, float, float, float]:
    (s_lo, s_hi), (l_lo, l_hi), (m_lo, m_hi), (j_lo, j_hi) = bounds
    sg = jax.nn.sigmoid(theta)
    sigma = s_lo + (s_hi - s_lo) * sg[0]
    lam = l_lo + (l_hi - l_lo) * sg[1]
    alpha_j = m_lo + (m_hi - m_lo) * sg[2]
    sigma_j = j_lo + (j_hi - j_lo) * sg[3]
    return sigma, lam, alpha_j, sigma_j


def _np_theta_to_params(
    theta: np.ndarray, bounds: Bounds = _MERTON_BOUNDS
) -> np.ndarray:
    (s_lo, s_hi), (l_lo, l_hi), (m_lo, m_hi), (j_lo, j_hi) = bounds
    sg = 1.0 / (1.0 + np.exp(-np.asarray(theta, dtype=float)))
    sigma = s_lo + (s_hi - s_lo) * sg[0]
    lam = l_lo + (l_hi - l_lo) * sg[1]
    alpha_j = m_lo + (m_hi - m_lo) * sg[2]
    sigma_j = j_lo + (j_hi - j_lo) * sg[3]
    return np.array([sigma, lam, alpha_j, sigma_j])


def _params_to_theta(params: np.ndarray, bounds: Bounds = _MERTON_BOUNDS) -> np.ndarray:
    (s_lo, s_hi), (l_lo, l_hi), (m_lo, m_hi), (j_lo, j_hi) = bounds
    sigma, lam, alpha_j, sigma_j = [float(x) for x in params]

    return np.array(
        [
            logit(sigma, s_lo, s_hi),
            logit(lam, l_lo, l_hi),
            logit(alpha_j, m_lo, m_hi),
            logit(sigma_j, j_lo, j_hi),
        ]
    )


def _chain_rule_jacobian(
    theta: np.ndarray, bounds: Bounds = _MERTON_BOUNDS
) -> np.ndarray:
    sg = 1.0 / (1.0 + np.exp(-np.asarray(theta, dtype=float)))
    deriv = sg * (1.0 - sg)
    spans = np.array([hi - lo for (lo, hi) in bounds])
    return spans * deriv


# --------------------------------------------------------------------------- #
# Residual function
# --------------------------------------------------------------------------- #


def _build_residual_fn(
    market_data: OptionMarketData,
    grids: JaxFFTGrids,
    sigma_prior: float,
    reg_weight: float,
    objective: ObjectiveStrategy,
    bounds: Bounds = _MERTON_BOUNDS,
) -> _CompiledResiduals:
    """JIT-compile the Merton residual for a fixed surface + grid config.

    Surface boilerplate lives in
    :func:`backend.calibration._reparam.build_cf_residual_fn`; only the
    Merton reparametrisation, CF pricing and Tikhonov penalty on sigma
    are supplied here.
    """
    spot = float(market_data.spot)
    rate = float(market_data.rate)
    q = float(market_data.dividend_yield)

    def theta_to_params(theta: jnp.ndarray) -> tuple:
        return _merton_theta_to_params(theta, bounds)

    def price_calls(params: tuple, strikes_T: jnp.ndarray, tau: float) -> jnp.ndarray:
        sigma, lam, alpha_j, sigma_j = params

        def cf(u):
            return merton_cf_jax(u, sigma, lam, alpha_j, sigma_j, tau, rate, q)

        return price_call_strikes_jax(cf, spot, strikes_T, tau, rate, grids)

    def penalty_fn(params: tuple) -> jnp.ndarray:
        sigma, _lam, _alpha_j, _sigma_j = params
        # Tikhonov regularization on sigma: sqrt(reg) * (sigma - prior)
        # (absorbs into the LM RSS as reg * (sigma - prior)^2)
        return jnp.sqrt(reg_weight) * (sigma - sigma_prior)

    return build_cf_residual_fn(
        market_data,
        objective,
        theta_to_params=theta_to_params,
        price_calls=price_calls,
        penalty_fn=penalty_fn,
    )


# --------------------------------------------------------------------------- #
# Calibrator
# --------------------------------------------------------------------------- #


class MertonCalibrator(BaseCalibrator):
    """Industry-standard Merton calibrator: LM + JAX Jacobian + multi-start.

    Parameters
    ----------
    n_restarts : int
        Number of LM runs from perturbed initial guesses.
    reg_weight : float
        Tikhonov weight on ``(sigma - sigma_prior)^2``. Prevents the
        diffusion vol from drifting to absorb jump-induced skew.
    sigma_prior : float | None
        If None, defaults to the ATM implied vol extracted from the
        surface at calibration time.
    max_nfev, ftol, xtol, gtol : float
        LM convergence parameters.
    seed : int
        Multi-start RNG seed.
    compute_uncertainty : bool
        Whether to compute Gauss-Newton std errors at the optimum.
    """

    def __init__(
        self,
        n_restarts: int = 5,
        reg_weight: float = 10.0,
        sigma_prior: float | None = None,
        max_nfev: int = 200,
        ftol: float = 1e-10,
        xtol: float = 1e-10,
        gtol: float = 1e-10,
        seed: int = 42,
        compute_uncertainty: bool = True,
        restart_scale_near: float = 1.0,
        restart_scale_far: float = 2.5,
        fft_grids: JaxFFTGrids | None = None,
        optimizer: OptimizerStrategy | None = None,
        objective: ObjectiveStrategy | None = None,
        log_iterations: bool = False,
        iteration_callback=None,
        param_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        # Per-run search universe (the sigmoid box). ``None`` -> canonical
        # default box, so default-bounds runs are byte-identical to the legacy
        # behaviour.
        self._bounds: Bounds = _resolve_bounds(param_bounds)
        self.n_restarts = max(1, int(n_restarts))
        self.reg_weight = float(reg_weight)
        self.sigma_prior = sigma_prior
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
        # JAX-compatibility fallback (iv_mse → vega_weighted approximation)
        self.objective: ObjectiveStrategy = self._resolve_objective(
            objective or PriceMSEObjective()
        )
        self.objective_type = self.objective.name
        self.log_iterations = bool(log_iterations)
        self.iteration_callback = iteration_callback

        self._grids = fft_grids or JaxFFTGrids.build()
        self._engine = FFTEngine()

    def default_bounds(self) -> list[tuple[float, float]]:
        return [(float(lo), float(hi)) for lo, hi in self._bounds]

    def objective(self, params: np.ndarray, market_data: OptionMarketData) -> float:
        r = self.residuals(params, market_data)
        return 0.5 * float(r @ r)

    def residuals(
        self, params: np.ndarray, market_data: OptionMarketData
    ) -> np.ndarray:
        prior = (
            self.sigma_prior
            if self.sigma_prior is not None
            else get_atm_iv(market_data)
        )
        compiled = _build_residual_fn(
            market_data,
            self._grids,
            prior,
            self.reg_weight,
            self.objective,
            self._bounds,
        )
        theta = _params_to_theta(np.asarray(params, dtype=float), self._bounds)
        return np.asarray(compiled.residual(jnp.asarray(theta)))

    def calibrate(self, market_data: OptionMarketData) -> CalibrationResult:
        from dataclasses import replace

        t_start = time.perf_counter()

        atm_iv = get_atm_iv(market_data)
        prior = self.sigma_prior if self.sigma_prior is not None else atm_iv

        compiled = _build_residual_fn(
            market_data,
            self._grids,
            prior,
            self.reg_weight,
            self.objective,
            self._bounds,
        )

        # Clamp the seed into the (possibly tightened) box — a no-op for the
        # default box, so default-bounds runs are unchanged.
        _lo = np.array([b[0] for b in self._bounds])
        _hi = np.array([b[1] for b in self._bounds])
        seed_params = np.clip(np.array([atm_iv, 0.5, -0.1, 0.15]), _lo, _hi)
        x0_base = _params_to_theta(seed_params, self._bounds)
        logger.info(
            "Merton calibration | ATM IV=%.2f%% | seed_params=%s",
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

        def _residual_np(x: np.ndarray) -> np.ndarray:
            return np.asarray(compiled.residual(jnp.asarray(x)))

        def _jacobian_np(x: np.ndarray) -> np.ndarray:
            return np.asarray(compiled.jacobian(jnp.asarray(x)))

        problem = CalibrationProblem(
            x0=x0_base,
            bounds_lo=np.full(4, -15.0),
            bounds_hi=np.full(4, 15.0),
            param_names=tuple(MERTON_PARAM_NAMES),
            residual_fn=_residual_np,
            jacobian_fn=_jacobian_np,
            param_mapper=lambda theta_arr: dict(
                zip(MERTON_PARAM_NAMES, _np_theta_to_params(theta_arr, self._bounds))
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

        sigma, lam, alpha_j, sigma_j = (
            float(x) for x in _np_theta_to_params(best["theta"], self._bounds)
        )
        try:
            model = MertonModel(sigma=sigma, lam=lam, alpha_j=alpha_j, sigma_j=sigma_j)
        except ValueError as exc:
            logger.error("Invalid Merton params: %s", exc)
            err_diag = {"error": str(exc), "lm_runs": lm_infos}
            if self.log_iterations:
                err_diag["multi_start_history"] = tuple(all_iteration_histories)
            return CalibrationResult(
                model=MertonModel(sigma=0.2, lam=0.5, alpha_j=-0.1, sigma_j=0.15),
                objective_value=best["rss"] / 2.0,
                n_iterations=best["nfev"],
                success=False,
                method=f"{self.optimizer.name}_multistart",
                elapsed_seconds=time.perf_counter() - t_start,
                diagnostics=err_diag,
                iteration_history=best["history"],
                optimizer_name=self.optimizer.name,
            )

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
            "rss_best": best["rss"],
            "n_restarts": self.n_restarts,
            "best_start_index": best["start_index"],
            "lm_runs": lm_infos,
            "atm_iv": atm_iv,
            "sigma_prior": prior,
            "reg_weight": self.reg_weight,
            "objective_name": self.objective.name,
            "objective_loss": float(objective_loss),
        }
        if self.log_iterations:
            diagnostics["multi_start_history"] = tuple(all_iteration_histories)

        if self.compute_uncertainty:
            try:
                # Strip the regularization row (last) from the Jacobian
                J_u = np.asarray(best["jac"])[: compiled.n_quote_residuals, :]
                r_q = np.asarray(best["fun"])[: compiled.n_quote_residuals]
                dpdu = _chain_rule_jacobian(best["theta"], self._bounds)
                J_constrained = J_u * dpdu[np.newaxis, :]

                unc = least_squares_covariance(J_constrained, r_q)
                diagnostics["uncertainty"] = summary_table(
                    list(MERTON_PARAM_NAMES),
                    np.array([sigma, lam, alpha_j, sigma_j]),
                    unc,
                )
                diagnostics["uncertainty_condition_number"] = unc.condition_number
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.warning("Uncertainty computation failed: %s", exc)
                diagnostics["uncertainty_error"] = str(exc)

        elapsed = time.perf_counter() - t_start
        logger.info(
            "Merton done in %.2fs | RMSE_price=%.6f | RMSE_iv=%.2f bp | nfev_best=%d",
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


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    from backend.calibration.market_data import OptionQuote
    from backend.core.market import MarketEnvironment
    from backend.instruments.options import VanillaOption
    from backend.utils.math import implied_volatility

    # Merton surface (spot=100, moderate jumps)
    true_model = MertonModel(sigma=0.18, lam=0.5, alpha_j=-0.10, sigma_j=0.20)
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
    print(f"Merton surface: {md.n_quotes} quotes")

    cal = MertonCalibrator(n_restarts=5, max_nfev=200)
    result = cal.calibrate(md)

    print(f"\n{result}")
    print(f"  elapsed       : {result.elapsed_seconds:.2f}s")
    print(f"  RMSE (price)  : {result.rmse_price:.6f}")
    print(f"  RMSE (IV bp)  : {result.rmse_iv:.2f}")

    cal_params = result.model.get_parameters()
    true_params = true_model.get_parameters()
    print("\nParameter recovery:")
    for name in MERTON_PARAM_NAMES:
        t_v, c_v = true_params[name], cal_params[name]
        err = abs(t_v - c_v) / (abs(t_v) + 1e-12) * 100
        print(f"  {name:10s} true={t_v:8.4f}  cal={c_v:8.4f}  err={err:5.2f}%")

    sys.exit(0 if result.success and result.rmse_price < 0.1 else 1)
