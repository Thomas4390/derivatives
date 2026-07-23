"""
Bates Calibrator — semi-sequential + joint LM with JAX Jacobian
================================================================

Bates (1996) = Heston stochastic variance + Merton compound-Poisson
jumps — 8 parameters (v0, kappa, theta, alpha, rho, lam, alpha_j, sigma_j).

The 8-dimensional joint calibration has many local minima, so we use a
semi-sequential warmup (Heston first, then jumps, then joint refinement)
closed by a full joint Levenberg-Marquardt with an analytical JAX
Jacobian across all 8 params:

  Phase 1 — Heston sub-calibration (5 params) via HestonCalibrator
  Phase 2 — Jump-only LM (3 params) with Heston fixed
  Phase 3 — Joint LM on all 8 params with JAX Jacobian through the
            full Bates CF + Carr-Madan FFT pipeline

Uncertainty at the joint optimum uses the Gauss-Newton covariance on
the 8-param Jacobian (chained through the sigmoid reparametrization).

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

from backend.calibration.lm_helpers import make_multi_starts

from backend.calibration._reparam import (
    _CompiledResiduals,
    build_cf_residual_fn,
    logit,
)
from backend.calibration.base import BaseCalibrator, CalibrationResult
from backend.calibration.feller import (
    DEFAULT_FELLER_WEIGHT,
    FELLER_STRICT_FACTOR,
    FellerMode,
    feller_capped_alpha,
    feller_alpha_to_unit,
    penalty_weight,
)
from backend.calibration.heston_calibrator import HestonCalibrator
from backend.calibration.market_data import OptionMarketData
from backend.calibration.objectives import (
    ObjectiveStrategy,
    PriceMSEObjective,
)
from backend.calibration.search_space import BATES_SEARCH_BOUNDS
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
    model_prices_to_ivs,
)
from backend.engines.aad.calibration.fft import (
    JaxFFTGrids,
    price_call_strikes_jax,
)
from backend.engines.aad.calibration.heston_cf import bates_cf_jax
from backend.engines.fft_engine import FFTEngine
from backend.models.bates import BatesModel
from backend.models.heston import HestonModel
from backend.utils.constants.calibration import BATES_PARAM_NAMES

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 8-parameter reparametrization (Heston + jumps)
# --------------------------------------------------------------------------- #

# Canonical (default) admissible box in BATES_PARAM_NAMES order
# (v0, kappa, theta, alpha, rho, lam, alpha_j, sigma_j). The search universe is
# configurable per-run (see ``param_bounds``); sourced from the single source
# of truth in ``backend.calibration.search_space``.
Bounds = tuple[tuple[float, float], ...]
_BATES_BOUNDS: Bounds = tuple(BATES_SEARCH_BOUNDS[n] for n in BATES_PARAM_NAMES)


def _resolve_bounds(param_bounds: dict[str, tuple[float, float]] | None) -> Bounds:
    """Map a ``{param: (lo, hi)}`` override onto the ordered 8-param Bates box."""
    if not param_bounds:
        return _BATES_BOUNDS
    return tuple(
        (float(param_bounds[n][0]), float(param_bounds[n][1]))
        if n in param_bounds
        else _BATES_BOUNDS[i]
        for i, n in enumerate(BATES_PARAM_NAMES)
    )


def _bates_theta_to_params(
    theta: jnp.ndarray,
    feller_mode: FellerMode = FellerMode.SOFT,
    bounds: Bounds = _BATES_BOUNDS,
) -> tuple[float, ...]:
    (
        (v0_lo, v0_hi),
        (k_lo, k_hi),
        (t_lo, t_hi),
        (alpha_lo, alpha_hi),
        (r_lo, r_hi),
        (l_lo, l_hi),
        (m_lo, m_hi),
        (j_lo, j_hi),
    ) = bounds
    sg = jax.nn.sigmoid(theta)
    kappa = k_lo + (k_hi - k_lo) * sg[1]
    theta_h = t_lo + (t_hi - t_lo) * sg[2]
    if feller_mode is FellerMode.HARD:
        alpha = feller_capped_alpha(kappa, theta_h, sg[3], alpha_lo, alpha_hi, xp=jnp)
    else:
        alpha = alpha_lo + (alpha_hi - alpha_lo) * sg[3]
    return (
        v0_lo + (v0_hi - v0_lo) * sg[0],
        kappa,
        theta_h,
        alpha,
        r_lo + (r_hi - r_lo) * sg[4],
        l_lo + (l_hi - l_lo) * sg[5],
        m_lo + (m_hi - m_lo) * sg[6],
        j_lo + (j_hi - j_lo) * sg[7],
    )


def _np_theta_to_params(
    theta: np.ndarray,
    feller_mode: FellerMode = FellerMode.SOFT,
    bounds: Bounds = _BATES_BOUNDS,
) -> np.ndarray:
    sg = 1.0 / (1.0 + np.exp(-np.asarray(theta, dtype=float)))
    params = np.array([lo + (hi - lo) * sg[i] for i, (lo, hi) in enumerate(bounds)])
    if feller_mode is FellerMode.HARD:
        alpha_lo, alpha_hi = bounds[3]
        params[3] = feller_capped_alpha(params[1], params[2], sg[3], alpha_lo, alpha_hi, xp=np)
    return params


def _params_to_theta(
    params: np.ndarray,
    feller_mode: FellerMode = FellerMode.SOFT,
    bounds: Bounds = _BATES_BOUNDS,
) -> np.ndarray:
    params = np.asarray(params, dtype=float)

    theta = np.array(
        [logit(float(params[i]), lo, hi) for i, (lo, hi) in enumerate(bounds)]
    )
    if feller_mode is FellerMode.HARD:
        alpha_lo, alpha_hi = bounds[3]
        alpha_unit = feller_alpha_to_unit(
            float(params[1]), float(params[2]), float(params[3]), alpha_lo, alpha_hi
        )
        theta[3] = logit(alpha_unit, 0.0, 1.0)
    return theta


def _chain_rule_jacobian(
    theta: np.ndarray,
    feller_mode: FellerMode = FellerMode.SOFT,
    bounds: Bounds = _BATES_BOUNDS,
) -> np.ndarray:
    sg = 1.0 / (1.0 + np.exp(-np.asarray(theta, dtype=float)))
    spans = np.array([hi - lo for (lo, hi) in bounds])
    if feller_mode is FellerMode.HARD:
        (k_lo, k_hi), (t_lo, t_hi), (alpha_lo, alpha_hi) = bounds[1], bounds[2], bounds[3]
        kappa = k_lo + (k_hi - k_lo) * sg[1]
        theta_h = t_lo + (t_hi - t_lo) * sg[2]
        alpha_feller_max = FELLER_STRICT_FACTOR * float((2.0 * kappa * theta_h) ** 0.5)
        alpha_upper = min(alpha_hi, alpha_feller_max)
        alpha_lower = min(alpha_lo, alpha_upper)
        spans = spans.copy()
        spans[3] = alpha_upper - alpha_lower
    return spans * sg * (1.0 - sg)


# --------------------------------------------------------------------------- #
# Residual function — joint (8 params)
# --------------------------------------------------------------------------- #


def _build_residual_fn(
    market_data: OptionMarketData,
    grids: JaxFFTGrids,
    feller_weight: float,
    objective: ObjectiveStrategy,
    feller_mode: FellerMode = FellerMode.SOFT,
    bounds: Bounds = _BATES_BOUNDS,
) -> _CompiledResiduals:
    """JIT-compile the joint Bates residual for a fixed surface + grid config.

    Surface boilerplate lives in
    :func:`backend.calibration._reparam.build_cf_residual_fn`; only the
    Bates reparametrisation, CF pricing and Feller penalty are supplied
    here.
    """
    spot = float(market_data.spot)
    rate = float(market_data.rate)
    q = float(market_data.dividend_yield)

    # OFF/HARD -> 0 (HARD guarantees Feller via the capped reparametrisation);
    # SOFT -> feller_weight. See backend.calibration.feller.
    eff_feller_weight = penalty_weight(feller_mode, feller_weight)

    def theta_to_params(theta: jnp.ndarray) -> tuple:
        return _bates_theta_to_params(theta, feller_mode, bounds)

    def price_calls(params: tuple, strikes_T: jnp.ndarray, tau: float) -> jnp.ndarray:
        v0, kappa, theta_h, alpha, rho, lam, alpha_j, sigma_j = params

        def cf(u):
            return bates_cf_jax(
                u,
                v0,
                kappa,
                theta_h,
                alpha,
                rho,
                tau,
                rate,
                q,
                lam,
                alpha_j,
                sigma_j,
            )

        return price_call_strikes_jax(cf, spot, strikes_T, tau, rate, grids)

    def penalty_fn(params: tuple) -> jnp.ndarray:
        _v0, kappa, theta_h, alpha = params[0], params[1], params[2], params[3]
        feller = 2.0 * kappa * theta_h - alpha**2
        return jnp.where(feller < 0.0, jnp.sqrt(eff_feller_weight) * (-feller), 0.0)

    return build_cf_residual_fn(
        market_data,
        objective,
        theta_to_params=theta_to_params,
        price_calls=price_calls,
        penalty_fn=penalty_fn,
    )


# --------------------------------------------------------------------------- #
# Jump-only residual — used in Phase 2 (Heston fixed)
# --------------------------------------------------------------------------- #


def _build_jump_residual_fn(
    market_data: OptionMarketData,
    grids: JaxFFTGrids,
    heston_params: dict[str, float],
    jump_bounds: Bounds = _BATES_BOUNDS[5:8],
) -> _CompiledResiduals:
    spot = float(market_data.spot)
    rate = float(market_data.rate)
    q = float(market_data.dividend_yield)
    unique_mats = [float(T) for T in market_data.unique_maturities]

    strikes_per_T: list[jnp.ndarray] = []
    is_calls_per_T: list[jnp.ndarray] = []
    running = 0
    for T in unique_mats:
        quotes = market_data.quotes_for_maturity(T)
        strikes_per_T.append(jnp.asarray([q.strike for q in quotes], dtype=jnp.float64))
        is_calls_per_T.append(jnp.asarray([q.is_call for q in quotes], dtype=jnp.bool_))
        running += len(quotes)
    n_quotes = running
    market_prices = jnp.asarray(market_data.market_prices, dtype=jnp.float64)

    v0 = float(heston_params["v0"])
    kappa = float(heston_params["kappa"])
    theta_h = float(heston_params["theta"])
    alpha = float(heston_params["alpha"])
    rho = float(heston_params["rho"])

    (l_lo, l_hi), (m_lo, m_hi), (j_lo, j_hi) = jump_bounds

    def _residual_untraced(theta_jump: jnp.ndarray) -> jnp.ndarray:
        # theta_jump is 3-element unconstrained vec for [lam, alpha_j, sigma_j]
        sg = jax.nn.sigmoid(theta_jump)
        lam = l_lo + (l_hi - l_lo) * sg[0]
        alpha_j = m_lo + (m_hi - m_lo) * sg[1]
        sigma_j = j_lo + (j_hi - j_lo) * sg[2]

        model_prices_blocks: list[jnp.ndarray] = []
        for i, tau in enumerate(unique_mats):
            strikes_T = strikes_per_T[i]
            is_calls_T = is_calls_per_T[i]

            def cf(u):
                return bates_cf_jax(
                    u,
                    v0,
                    kappa,
                    theta_h,
                    alpha,
                    rho,
                    tau,
                    rate,
                    q,
                    lam,
                    alpha_j,
                    sigma_j,
                )

            call_prices = price_call_strikes_jax(cf, spot, strikes_T, tau, rate, grids)
            disc_spot = spot * jnp.exp(-q * tau)
            disc_k = strikes_T * jnp.exp(-rate * tau)
            put_prices = call_prices - disc_spot + disc_k
            prices_T = jnp.maximum(jnp.where(is_calls_T, call_prices, put_prices), 0.0)
            model_prices_blocks.append(prices_T)

        model_prices = jnp.concatenate(model_prices_blocks)
        return model_prices - market_prices

    return _CompiledResiduals(
        residual=jax.jit(_residual_untraced),
        jacobian=jax.jit(jax.jacfwd(_residual_untraced)),
        n_quote_residuals=n_quotes,
    )


# --------------------------------------------------------------------------- #
# Calibrator
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Phase1Out:
    """Result of the Heston warmup (Phase 1)."""

    heston_result: CalibrationResult
    heston_params: dict[str, float]


@dataclass(frozen=True)
class _Phase2Out:
    """Result of the jump-only LM (Phase 2)."""

    lam: float
    alpha_j: float
    sigma_j: float
    rss: float
    nfev: int
    iteration_history: tuple


@dataclass(frozen=True)
class _Phase3Out:
    """Result of the joint 8-param LM with multi-start (Phase 3)."""

    best: dict
    lm_infos: list
    iteration_histories: list
    joint_compiled: _CompiledResiduals


class BatesCalibrator(BaseCalibrator):
    """Industry-standard Bates calibrator: semi-sequential warmup + joint LM.

    Parameters
    ----------
    n_restarts_joint : int
        Multi-start count for Phase 3 (joint LM).
    feller_mode : FellerMode
        Feller-condition handling (``OFF`` / ``SOFT`` / ``HARD``), propagated
        to the Phase-1 Heston warmup and the joint Phase-3 LM. See
        :class:`backend.calibration.feller.FellerMode`.
    feller_weight : float
        Soft Feller penalty weight (only used in :attr:`FellerMode.SOFT`).
    max_nfev_heston, max_nfev_joint : int
        LM nfev budgets for the Heston warmup (Phase 1) and joint refinement
        (Phase 3). Phase 2 (jumps-only) reuses max_nfev_joint.
    ftol, xtol, gtol : float
        LM convergence tolerances.
    seed : int
        Multi-start RNG seed.
    """

    def __init__(
        self,
        n_restarts_joint: int = 3,
        feller_weight: float = DEFAULT_FELLER_WEIGHT,
        max_nfev_heston: int = 200,
        max_nfev_joint: int = 200,
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
        feller_mode: FellerMode | str = FellerMode.SOFT,
        param_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        # Per-run search universe (8-param sigmoid box). ``None`` -> the
        # canonical default box, so default-bounds runs are byte-identical to
        # the legacy behaviour. The v0..rho sub-box is forwarded to the Phase-1
        # Heston warmup; the lam/alpha_j/sigma_j sub-box drives Phase 2.
        self._bounds: Bounds = _resolve_bounds(param_bounds)
        self.n_restarts_joint = max(1, int(n_restarts_joint))
        self.feller_weight = float(feller_weight)
        self.feller_mode = FellerMode.coerce(feller_mode)
        self.max_nfev_heston = int(max_nfev_heston)
        self.max_nfev_joint = int(max_nfev_joint)
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
        # JAX-compatible fallback: iv_mse degenerates to vega_weighted under
        # the Cont-Tankov first-order approximation (see HestonCalibrator).
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
        compiled = _build_residual_fn(
            market_data,
            self._grids,
            self.feller_weight,
            self.objective,
            self.feller_mode,
            self._bounds,
        )
        theta = _params_to_theta(
            np.asarray(params, dtype=float), self.feller_mode, self._bounds
        )
        return np.asarray(compiled.residual(jnp.asarray(theta)))

    def calibrate(self, market_data: OptionMarketData) -> CalibrationResult:
        t_start = time.perf_counter()

        p1 = self._phase1_heston(market_data)
        p2 = self._phase2_jump(market_data, p1.heston_params)
        p3 = self._phase3_joint(market_data, p1.heston_params, p2)
        return self._build_calibration_result(market_data, p1, p2, p3, t_start)

    # ------------------------------------------------------------------ #
    # Phase 1 — Heston sub-calibration (5 params)
    # ------------------------------------------------------------------ #
    def _phase1_heston(self, market_data: OptionMarketData) -> _Phase1Out:
        logger.info("Phase 1: Heston sub-calibration via HestonCalibrator")
        # Forward the v0..rho sub-box (first 5 slots) to the Heston warmup so a
        # tightened Bates search universe also constrains Phase 1.
        heston_bounds = {
            name: self._bounds[i] for i, name in enumerate(BATES_PARAM_NAMES[:5])
        }
        heston_cal = HestonCalibrator(
            n_restarts=5,
            feller_weight=self.feller_weight,
            feller_mode=self.feller_mode,
            max_nfev=self.max_nfev_heston,
            seed=self.seed,
            compute_uncertainty=False,  # will compute at joint optimum
            fft_grids=self._grids,
            optimizer=self.optimizer,
            objective=self.objective,
            log_iterations=self.log_iterations,
            iteration_callback=self.iteration_callback,
            param_bounds=heston_bounds,
        )
        heston_result = heston_cal.calibrate(market_data)
        heston_model: HestonModel = heston_result.model  # type: ignore[assignment]
        heston_params = heston_model.get_parameters()
        logger.info(
            "Phase 1 done: RSS=%.6f, Heston params=%s",
            heston_result.objective_value,
            {k: round(v, 5) for k, v in heston_params.items()},
        )
        return _Phase1Out(heston_result=heston_result, heston_params=heston_params)

    # ------------------------------------------------------------------ #
    # Phase 2 — Jump-only LM (3 params, Heston fixed)
    # ------------------------------------------------------------------ #
    def _phase2_jump(
        self,
        market_data: OptionMarketData,
        heston_params: dict[str, float],
    ) -> _Phase2Out:
        logger.info("Phase 2: Jump-only LM (lam, alpha_j, sigma_j)")
        jump_bounds = self._bounds[5:8]
        (l_lo, l_hi), (m_lo, m_hi), (j_lo, j_hi) = jump_bounds
        jump_compiled = _build_jump_residual_fn(
            market_data, self._grids, heston_params, jump_bounds
        )

        def _jump_f(x):
            return np.asarray(jump_compiled.residual(jnp.asarray(x)))

        def _jump_jac(x):
            return np.asarray(jump_compiled.jacobian(jnp.asarray(x)))

        # Jump seed clamped into the (possibly tightened) jump box — no-op for
        # the default box (0.3 ∈ [1e-3, 5], -0.1 ∈ [-0.5, 0.1], 0.15 ∈ [0.01, 0.5]).
        jump_x0 = np.array(
            [
                logit(float(np.clip(0.3, l_lo, l_hi)), l_lo, l_hi),
                logit(float(np.clip(-0.1, m_lo, m_hi)), m_lo, m_hi),
                logit(float(np.clip(0.15, j_lo, j_hi)), j_lo, j_hi),
            ]
        )

        # Drive Phase 2 through self.optimizer for end-to-end consistency
        jump_problem = CalibrationProblem(
            x0=jump_x0,
            bounds_lo=np.full(3, -15.0),
            bounds_hi=np.full(3, 15.0),
            param_names=("lam", "alpha_j", "sigma_j"),
            residual_fn=_jump_f,
            jacobian_fn=_jump_jac,
        )
        jump_logger = (
            IterationLogger(jump_problem, on_snapshot=self.iteration_callback)
            if (self.log_iterations or self.iteration_callback)
            else None
        )
        jump_opt_res = self.optimizer.solve(
            jump_problem,
            logger=jump_logger,
            max_nfev=self.max_nfev_joint,
        )
        jump_rss = 2.0 * float(jump_opt_res.objective_value)
        sg = 1.0 / (1.0 + np.exp(-jump_opt_res.x_optimal))
        lam = float(l_lo + (l_hi - l_lo) * sg[0])
        alpha_j = float(m_lo + (m_hi - m_lo) * sg[1])
        sigma_j = float(j_lo + (j_hi - j_lo) * sg[2])
        nfev = int(jump_opt_res.n_function_evals)
        logger.info(
            "Phase 2 done: RSS=%.6f, lam=%.4f alpha_j=%.4f sigma_j=%.4f",
            jump_rss,
            lam,
            alpha_j,
            sigma_j,
        )
        return _Phase2Out(
            lam=lam,
            alpha_j=alpha_j,
            sigma_j=sigma_j,
            rss=jump_rss,
            nfev=nfev,
            iteration_history=(jump_logger.history if jump_logger is not None else ()),
        )

    # ------------------------------------------------------------------ #
    # Phase 3 — Joint LM (8 params) with multi-start
    # ------------------------------------------------------------------ #
    def _phase3_joint(
        self,
        market_data: OptionMarketData,
        heston_params: dict[str, float],
        p2: _Phase2Out,
    ) -> _Phase3Out:
        from dataclasses import replace as _replace

        logger.info("Phase 3: Joint LM on 8 params with multi-start")
        joint_compiled = _build_residual_fn(
            market_data,
            self._grids,
            self.feller_weight,
            self.objective,
            self.feller_mode,
            self._bounds,
        )

        x0_base = _params_to_theta(
            np.array(
                [
                    heston_params["v0"],
                    heston_params["kappa"],
                    heston_params["theta"],
                    heston_params["alpha"],
                    heston_params["rho"],
                    p2.lam,
                    p2.alpha_j,
                    p2.sigma_j,
                ]
            ),
            self.feller_mode,
            self._bounds,
        )

        rng = np.random.default_rng(self.seed + 1000)
        starts = make_multi_starts(
            x0_base,
            n_restarts=self.n_restarts_joint,
            near_scale=self.restart_scale_near,
            far_scale=self.restart_scale_far,
            rng=rng,
        )

        def _joint_residual_np(x: np.ndarray) -> np.ndarray:
            return np.asarray(joint_compiled.residual(jnp.asarray(x)))

        def _joint_jacobian_np(x: np.ndarray) -> np.ndarray:
            return np.asarray(joint_compiled.jacobian(jnp.asarray(x)))

        joint_problem = CalibrationProblem(
            x0=x0_base,
            bounds_lo=np.full(8, -15.0),
            bounds_hi=np.full(8, 15.0),
            param_names=tuple(BATES_PARAM_NAMES),
            residual_fn=_joint_residual_np,
            jacobian_fn=_joint_jacobian_np,
            param_mapper=lambda theta_arr: dict(
                zip(
                    BATES_PARAM_NAMES,
                    _np_theta_to_params(theta_arr, self.feller_mode, self._bounds),
                )
            ),
        )

        best: dict | None = None
        lm_infos: list = []
        all_iteration_histories: list[tuple] = []
        for k, x0 in enumerate(starts):
            iter_logger = (
                IterationLogger(joint_problem, on_snapshot=self.iteration_callback)
                if (self.log_iterations or self.iteration_callback)
                else None
            )
            opt_res = self.optimizer.solve(
                _replace(joint_problem, x0=x0),
                logger=iter_logger,
                max_nfev=self.max_nfev_joint,
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
                jac_at_best = _joint_jacobian_np(opt_res.x_optimal)
                fun_at_best = _joint_residual_np(opt_res.x_optimal)
                best = {
                    "rss": rss,
                    "theta": opt_res.x_optimal,
                    "jac": jac_at_best,
                    "fun": fun_at_best,
                    "nfev": opt_res.n_function_evals,
                    "start_index": k,
                    "history": iter_logger.history if iter_logger is not None else (),
                }

        assert best is not None  # n_restarts_joint >= 1 by __init__ contract
        return _Phase3Out(
            best=best,
            lm_infos=lm_infos,
            iteration_histories=all_iteration_histories,
            joint_compiled=joint_compiled,
        )

    # ------------------------------------------------------------------ #
    # Result assembly — final model + diagnostics + uncertainty
    # ------------------------------------------------------------------ #
    def _build_calibration_result(
        self,
        market_data: OptionMarketData,
        p1: _Phase1Out,
        p2: _Phase2Out,
        p3: _Phase3Out,
        t_start: float,
    ) -> CalibrationResult:
        best = p3.best
        params_final = _np_theta_to_params(
            best["theta"], self.feller_mode, self._bounds
        )
        v0, kappa, theta_h, alpha, rho, lam, alpha_j, sigma_j = (
            float(p) for p in params_final
        )

        try:
            model = BatesModel(
                v0=v0,
                kappa=kappa,
                theta=theta_h,
                alpha=alpha,
                rho=rho,
                lam=lam,
                alpha_j=alpha_j,
                sigma_j=sigma_j,
            )
        except ValueError as exc:
            return self._build_failure_result(p1, p2, p3, t_start, exc)

        # RMSE via numpy engine
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
        diagnostics: dict = {
            "feller_ratio": model.feller_ratio,
            "rss_best": best["rss"],
            "heston_rss_ph1": p1.heston_result.objective_value,
            "jump_rss_ph2": p2.rss,
            "joint_rss_ph3": best["rss"],
            "n_restarts_joint": self.n_restarts_joint,
            "best_start_index": best["start_index"],
            "lm_runs_joint": p3.lm_infos,
            "heston_warmup_params": p1.heston_params,
            "objective_name": self.objective.name,
            "objective_loss": float(objective_loss),
        }
        if self.log_iterations:
            diagnostics["multi_start_history"] = tuple(p3.iteration_histories)
            diagnostics["heston_phase_history"] = p1.heston_result.iteration_history
            if p2.iteration_history:
                diagnostics["jump_phase_history"] = p2.iteration_history

        if self.compute_uncertainty:
            self._attach_uncertainty(
                diagnostics,
                p3.joint_compiled,
                best,
                params_final,
                self.feller_mode,
                self._bounds,
            )

        elapsed = time.perf_counter() - t_start
        logger.info(
            "Bates done in %.2fs | RMSE_price=%.6f | RMSE_iv=%.2f bp",
            elapsed,
            rmse_price,
            rmse_iv,
        )

        return CalibrationResult(
            model=model,
            objective_value=best["rss"] / 2.0,
            n_iterations=(best["nfev"] + p2.nfev + p1.heston_result.n_iterations),
            success=True,
            method=f"heston+jump+joint:{self.optimizer.name}",
            rmse_price=rmse_price,
            rmse_iv=rmse_iv,
            elapsed_seconds=elapsed,
            diagnostics=diagnostics,
            iteration_history=best["history"],
            optimizer_name=self.optimizer.name,
        )

    def _build_failure_result(
        self,
        p1: _Phase1Out,
        p2: _Phase2Out,
        p3: _Phase3Out,
        t_start: float,
        exc: ValueError,
    ) -> CalibrationResult:
        """Return a CalibrationResult when BatesModel construction rejected the params."""
        logger.error("Invalid Bates params: %s", exc)
        err_diag: dict = {"error": str(exc), "lm_runs": p3.lm_infos}
        if self.log_iterations:
            err_diag["multi_start_history"] = tuple(p3.iteration_histories)
        return CalibrationResult(
            model=BatesModel(
                v0=0.04,
                kappa=2.0,
                theta=0.04,
                alpha=0.3,
                rho=-0.7,
                lam=0.1,
                alpha_j=-0.1,
                sigma_j=0.15,
            ),
            objective_value=p3.best["rss"] / 2.0,
            n_iterations=(p3.best["nfev"] + p2.nfev + p1.heston_result.n_iterations),
            success=False,
            method=f"heston+jump+joint:{self.optimizer.name}",
            elapsed_seconds=time.perf_counter() - t_start,
            diagnostics=err_diag,
            iteration_history=p3.best["history"],
            optimizer_name=self.optimizer.name,
        )

    @staticmethod
    def _attach_uncertainty(
        diagnostics: dict,
        joint_compiled: _CompiledResiduals,
        best: dict,
        params_final: np.ndarray,
        feller_mode: FellerMode = FellerMode.SOFT,
        bounds: Bounds = _BATES_BOUNDS,
    ) -> None:
        """Compute Gauss-Newton covariance on the joint Jacobian (in place)."""
        try:
            J_u = np.asarray(best["jac"])[: joint_compiled.n_quote_residuals, :]
            r_q = np.asarray(best["fun"])[: joint_compiled.n_quote_residuals]
            dpdu = _chain_rule_jacobian(best["theta"], feller_mode, bounds)
            J_constrained = J_u * dpdu[np.newaxis, :]

            unc = least_squares_covariance(J_constrained, r_q)
            diagnostics["uncertainty"] = summary_table(
                list(BATES_PARAM_NAMES),
                params_final,
                unc,
            )
            diagnostics["uncertainty_condition_number"] = unc.condition_number
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.warning("Uncertainty computation failed: %s", exc)
            diagnostics["uncertainty_error"] = str(exc)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    from backend.calibration.market_data import OptionMarketData, OptionQuote
    from backend.core.market import MarketEnvironment
    from backend.instruments.options import VanillaOption
    from backend.utils.math import implied_volatility

    true_model = BatesModel(
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        alpha=0.3,
        rho=-0.65,
        lam=0.5,
        alpha_j=-0.10,
        sigma_j=0.15,
    )
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
    print(f"Bates surface: {md.n_quotes} quotes")

    cal = BatesCalibrator(n_restarts_joint=3, max_nfev_joint=200)
    result = cal.calibrate(md)

    print(f"\n{result}")
    print(f"  elapsed       : {result.elapsed_seconds:.2f}s")
    print(f"  RMSE (price)  : {result.rmse_price:.6f}")
    print(f"  RMSE (IV bp)  : {result.rmse_iv:.2f}")

    cal_params = result.model.get_parameters()
    true_params = true_model.get_parameters()
    print("\nParameter recovery:")
    for name in BATES_PARAM_NAMES:
        t_v, c_v = true_params[name], cal_params[name]
        err = abs(t_v - c_v) / (abs(t_v) + 1e-12) * 100
        print(f"  {name:10s} true={t_v:8.4f}  cal={c_v:8.4f}  err={err:6.2f}%")

    sys.exit(0 if result.success and result.rmse_price < 0.1 else 1)
