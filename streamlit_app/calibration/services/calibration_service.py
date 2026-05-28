"""
Calibration orchestration
==========================

Wraps the backend calibrators with the chosen OptimizerStrategy and
exposes a single entry point ``calibrate_with(...)`` returning the
``CalibrationResult`` along with a set of UI-friendly metadata
(per-iteration arrays, parameter recovery error, etc.).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.calibration import (
    DEFAULT_FELLER_WEIGHT,
    DEFAULT_STATIONARITY_WEIGHT,
    BatesCalibrator,
    CalibrationResult,
    DifferentialEvolutionStrategy,
    FellerMode,
    GARCHCalibrator,
    GARCHRiskNeutralCalibrator,
    HestonCalibrator,
    HestonNandiGARCHCalibrator,
    LBFGSStrategy,
    LMJaxStrategy,
    MertonCalibrator,
    NelderMeadStrategy,
    ObjectiveStrategy,
    StationarityMode,
    make_objective,
    price_surface,
)
from backend.calibration.utils import (
    compute_rmse_iv,
    compute_rmse_price,
    model_prices_to_ivs,
)

from config.constants import RECOVERY_DENOM_CLAMP
from services.post_calibration import _engine, rebuild_model

logger = logging.getLogger(__name__)


def make_strategy(
    name: str,
    max_nfev: int = 200,
    *,
    de_seed: int = 42,
    **_extra: Any,
):
    """Instantiate one of the four solver strategies by short name.

    ``de_seed`` is only consumed by the Differential Evolution branch
    (the other strategies are deterministic given the initial point). The
    default ``42`` preserves the historical behaviour from before the
    seed was exposed.
    """
    if name == "LM-JAX":
        return LMJaxStrategy()
    if name == "DE":
        return DifferentialEvolutionStrategy(seed=de_seed, polish=False)
    if name == "NM":
        return NelderMeadStrategy()
    if name == "L-BFGS-B":
        return LBFGSStrategy()
    raise KeyError(f"Unknown solver '{name}'")


@dataclass
class CalibrationRunSummary:
    solver_name: str
    result: CalibrationResult | None
    elapsed: float
    estimated_params: dict[str, float]
    true_params: dict[str, float]
    relative_recovery_error: dict[str, float]
    objective_name: str = "price_mse"
    error: str | None = None
    # Convergence-quality metrics surfaced in the recap table. Both are
    # ``None`` for runs that did not finish (or for solvers that don't
    # expose a gradient — DE, NM, surface-LM unless we add propagation
    # in the backend). UI renders them as "—" when missing.
    final_loss: float | None = None
    grad_norm: float | None = None
    # True when this summary was reconstructed from a *stopped* run's
    # best-so-far snapshot rather than a converged fit. The run is still
    # usable (anchoring, recap table) but is not a completed calibration —
    # UI surfaces it with a distinct "partial" badge.
    partial: bool = False

    @classmethod
    def failure(
        cls,
        solver_name: str,
        true_params: dict[str, float],
        exc: BaseException,
        objective_name: str = "price_mse",
    ) -> "CalibrationRunSummary":
        return cls(
            solver_name=solver_name,
            result=None,
            elapsed=-1.0,
            estimated_params={},
            true_params=dict(true_params),
            relative_recovery_error={k: float("nan") for k in true_params},
            objective_name=objective_name,
            error=str(exc),
        )


# Surface models whose fit is summarised with RMSE (price / IV) via the FFT
# engine. GARCH (returns-MLE) and iv_gbm are excluded — their partial result
# falls back to ``objective_value`` everywhere (e.g. the landscape anchor).
_SURFACE_RMSE_MODELS: frozenset[str] = frozenset(
    {"heston", "bates", "merton", "heston_nandi"}
)


def _rebuild_for_partial(model_key: str, params: dict[str, float]) -> Any | None:
    """Rebuild a pricable model for a partial summary, or ``None``.

    Reuses :func:`services.post_calibration.rebuild_model` (heston / merton /
    bates) and adds Heston-Nandi, which is Fourier-pricable but absent from
    that surface-only helper. GARCH (returns-MLE) and iv/GBM stay ``None`` —
    their partial summary carries the parameters without a model object.
    """
    model = rebuild_model(model_key, params)
    if model is not None:
        return model
    if model_key == "heston_nandi":
        from backend.models.heston_nandi import HestonNandiGARCHModel

        return HestonNandiGARCHModel(**{k: float(v) for k, v in params.items()})
    return None


def _surface_fit_metrics(model: Any, market_data: Any) -> tuple[float | None, float | None]:
    """Best-effort ``(rmse_price, rmse_iv)`` for a fitted surface model.

    Mirrors the calibrators' own metric block. Any pricing / IV failure (or a
    non-finite result) collapses to ``None`` so a partial summary degrades
    gracefully instead of raising.
    """
    try:
        model_prices = price_surface(model, market_data, _engine())
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
            compute_rmse_iv(model_ivs[valid], market_ivs[valid]) if valid.any() else None
        )
    except (ValueError, FloatingPointError, RuntimeError, ArithmeticError, AttributeError):
        return None, None
    rp = float(rmse_price) if rmse_price is not None and math.isfinite(rmse_price) else None
    ri = float(rmse_iv) if rmse_iv is not None and math.isfinite(rmse_iv) else None
    return rp, ri


def partial_from_history(
    *,
    solver_name: str,
    objective_name: str,
    model_key: str,
    history: Sequence[Any],
    market_data: Any,
    true_params: dict[str, float],
    elapsed: float,
) -> CalibrationRunSummary:
    """Build a *partial* summary from a stopped run's best-so-far snapshot.

    Picks the lowest-objective snapshot recorded before the user pressed Stop,
    rebuilds the model at those parameters, and packages it as a non-converged
    but fully usable :class:`CalibrationRunSummary` (``partial=True``). An empty
    history — or a model that cannot be rebuilt — falls back to
    :meth:`CalibrationRunSummary.failure` (the legacy behaviour).
    """
    snaps = [s for s in history if s is not None]
    if not snaps:
        return CalibrationRunSummary.failure(
            solver_name,
            true_params,
            RuntimeError("stopped before the first evaluation"),
            objective_name=objective_name,
        )

    best = min(snaps, key=lambda s: s.objective)
    model_params = {k: float(v) for k, v in best.params_natural.items()}
    loss = float(best.objective)
    grad = getattr(best, "grad_norm", None)

    # Rebuild the model so surface consumers (landscape anchor, RMSE recap) can
    # use the stopped point. Models without a pricable rebuild (GARCH / iv_gbm)
    # — or a rebuild failure — keep the legacy failure summary, so the invariant
    # "partial ⇒ result is not None" holds for every downstream consumer.
    try:
        model = _rebuild_for_partial(model_key, model_params)
    except (TypeError, ValueError, KeyError) as exc:
        logger.warning("partial summary: cannot rebuild %s model: %s", model_key, exc)
        model = None
    if model is None:
        return CalibrationRunSummary.failure(
            solver_name,
            true_params,
            RuntimeError("stopped — no pricable model for the best-so-far point"),
            objective_name=objective_name,
        )

    # Display / recovery scale: GARCH ω is stored annualised on the model but
    # compared per-period (mirror of the success path in ``calibrate_with``).
    # A no-op for the surface models that reach here — kept for parity should a
    # GARCH rebuild ever be added to ``_rebuild_for_partial``.
    disp = dict(model_params)
    if model_key in ("garch", "ngarch", "gjr_garch") and "omega" in disp:
        af = float(getattr(market_data, "annualization_factor", 1.0))
        disp["omega"] = disp["omega"] / af
    est_filtered = {k: float(v) for k, v in disp.items() if k in true_params}
    rel_error: dict[str, float] = {}
    for k, true_v in true_params.items():
        est_v = est_filtered.get(k, float("nan"))
        denom = max(abs(true_v), RECOVERY_DENOM_CLAMP)
        rel_error[k] = float(abs(est_v - true_v) / denom)

    rmse_price: float | None = None
    rmse_iv: float | None = None
    if model_key in _SURFACE_RMSE_MODELS:
        rmse_price, rmse_iv = _surface_fit_metrics(model, market_data)
    result = CalibrationResult(
        model=model,
        objective_value=loss,
        n_iterations=len(snaps),
        success=False,
        method=f"{solver_name}/stopped",
        rmse_price=rmse_price,
        rmse_iv=rmse_iv,
        elapsed_seconds=float(elapsed),
        diagnostics={"stopped": True, "n_evaluations": len(snaps), "grad_norm": grad},
        iteration_history=tuple(snaps),
        optimizer_name=solver_name,
    )

    return CalibrationRunSummary(
        solver_name=solver_name,
        result=result,
        elapsed=float(elapsed),
        estimated_params=est_filtered,
        true_params=dict(true_params),
        relative_recovery_error=rel_error,
        objective_name=objective_name,
        error=None,
        final_loss=loss,
        grad_norm=float(grad) if grad is not None and math.isfinite(grad) else None,
        partial=True,
    )


def _calibrator_for(
    model_key: str,
    optimizer,
    objective: ObjectiveStrategy | None,
    log_iterations: bool,
    n_restarts: int,
    max_nfev: int,
    iteration_callback=None,
    feller_mode: FellerMode = FellerMode.SOFT,
    feller_weight: float = DEFAULT_FELLER_WEIGHT,
    stationarity_mode: StationarityMode = StationarityMode.SOFT,
    stationarity_weight: float = DEFAULT_STATIONARITY_WEIGHT,
):
    """Build a backend calibrator for the given model with the requested optimizer + objective.

    The physical-measure GARCH calibrators do not consume an objective (MLE on
    returns), so the ``objective`` argument is silently ignored for that family.
    The ``feller_mode`` / ``feller_weight`` controls only apply to the CIR
    stochastic-variance models (Heston, Bates); ``stationarity_mode`` /
    ``stationarity_weight`` apply to the risk-neutral GARCH surface models —
    Heston-Nandi (β + αγ² < 1) and the Duan NGARCH-Q (β + α(1 + γ²) < 1). Merton
    and the physical-measure GARCH family have neither, so nothing is forwarded
    there.
    """
    if model_key == "heston":
        return HestonCalibrator(
            n_restarts=n_restarts,
            max_nfev=max_nfev,
            optimizer=optimizer,
            objective=objective,
            log_iterations=log_iterations,
            iteration_callback=iteration_callback,
            feller_mode=feller_mode,
            feller_weight=feller_weight,
        )
    if model_key == "merton":
        return MertonCalibrator(
            n_restarts=n_restarts,
            max_nfev=max_nfev,
            optimizer=optimizer,
            objective=objective,
            log_iterations=log_iterations,
            iteration_callback=iteration_callback,
        )
    if model_key == "bates":
        return BatesCalibrator(
            n_restarts_joint=n_restarts,
            max_nfev_heston=max_nfev,
            max_nfev_joint=max_nfev,
            optimizer=optimizer,
            objective=objective,
            log_iterations=log_iterations,
            iteration_callback=iteration_callback,
            feller_mode=feller_mode,
            feller_weight=feller_weight,
        )
    if model_key == "heston_nandi":
        return HestonNandiGARCHCalibrator(
            n_restarts=n_restarts,
            max_nfev=max_nfev,
            optimizer=optimizer,
            objective=objective,
            log_iterations=log_iterations,
            iteration_callback=iteration_callback,
            stationarity_mode=stationarity_mode,
            stationarity_weight=stationarity_weight,
        )
    # Risk-neutral GARCH surface models (nonaffine, MC-priced): map the app key
    # to the simulator variant and use the generalised calibrator. NGARCH-Q keeps
    # its dedicated subclass for back-compat.
    _rn_garch_variant = {"ngarch_q": "ngarch", "garch_q": "garch", "gjr_q": "gjr_garch"}
    if model_key in _rn_garch_variant:
        # Nonaffine MC surface fit — cap restarts/paths so the interactive run
        # stays responsive (each objective eval prices the surface by MC).
        return GARCHRiskNeutralCalibrator(
            garch_type=_rn_garch_variant[model_key],
            n_restarts=min(n_restarts, 2),
            max_nfev=max_nfev,
            n_paths=20_000,
            optimizer=optimizer,
            objective=objective,
            log_iterations=log_iterations,
            iteration_callback=iteration_callback,
            stationarity_mode=stationarity_mode,
            stationarity_weight=stationarity_weight,
        )
    if model_key in ("garch", "ngarch", "gjr_garch"):
        return GARCHCalibrator(
            garch_type=model_key,
            optimizer=optimizer,
            log_iterations=log_iterations,
            iteration_callback=iteration_callback,
            max_nfev=max_nfev,
        )
    raise NotImplementedError(f"No calibrator for {model_key}")


def _build_objective(
    objective_name: str,
    objective_settings: dict[str, Any] | None,
) -> ObjectiveStrategy:
    """Instantiate an ObjectiveStrategy from the UI selection + settings dict."""
    settings = objective_settings or {}
    if objective_name == "huber":
        return make_objective("huber", delta=float(settings.get("huber_delta", 0.05)))
    if objective_name == "relative":
        return make_objective(
            "relative",
            use_log=bool(settings.get("relative_use_log", False)),
        )
    if objective_name == "vega_weighted":
        return make_objective(
            "vega_weighted",
            fallback_iv=float(settings.get("fallback_iv", 0.20)),
        )
    return make_objective(objective_name)


# Public alias: other views (e.g. the Loss Landscape tab) build the same
# ObjectiveStrategy from a UI selection without duplicating the settings logic.
build_objective = _build_objective


def _build_feller(
    constraint_settings: dict[str, Any] | None,
) -> tuple[FellerMode, float]:
    """Resolve the Feller-condition mode + soft-penalty weight from UI settings.

    Mirrors :func:`_build_objective`: the constraints panel stores a light
    ``{"feller_mode": ..., "feller_weight": ...}`` dict in session state, which
    is coerced here into the backend :class:`FellerMode`. Defaults to ``SOFT``
    at :data:`DEFAULT_FELLER_WEIGHT`, reproducing the legacy behaviour.
    """
    settings = constraint_settings or {}
    mode = FellerMode.coerce(settings.get("feller_mode"))
    weight = float(settings.get("feller_weight", DEFAULT_FELLER_WEIGHT))
    return mode, weight


def _build_stationarity(
    constraint_settings: dict[str, Any] | None,
) -> tuple[StationarityMode, float]:
    """Resolve the Heston-Nandi GARCH stationarity mode + soft-penalty weight.

    Mirrors :func:`_build_feller` for the persistence condition ``β + αγ² < 1``.
    Defaults to ``SOFT`` at :data:`DEFAULT_STATIONARITY_WEIGHT`.
    """
    settings = constraint_settings or {}
    mode = StationarityMode.coerce(settings.get("stationarity_mode"))
    weight = float(settings.get("stationarity_weight", DEFAULT_STATIONARITY_WEIGHT))
    return mode, weight


def calibrate_with(
    model_key: str,
    market_data,
    solver_name: str,
    true_params: dict[str, float],
    *,
    objective_name: str = "price_mse",
    objective_settings: dict[str, Any] | None = None,
    constraint_settings: dict[str, Any] | None = None,
    n_restarts: int = 5,
    max_nfev: int = 200,
    de_seed: int = 42,
    log_iterations: bool = True,
    iteration_callback=None,
) -> CalibrationRunSummary:
    """Run a single calibration with the chosen solver and objective."""
    optimizer = make_strategy(solver_name, max_nfev=max_nfev, de_seed=de_seed)
    # GARCH calibrators do not consume an objective — pass None.
    objective = (
        None
        if model_key in ("garch", "ngarch", "gjr_garch")
        else _build_objective(objective_name, objective_settings)
    )
    feller_mode, feller_weight = _build_feller(constraint_settings)
    stationarity_mode, stationarity_weight = _build_stationarity(constraint_settings)
    calibrator = _calibrator_for(
        model_key,
        optimizer=optimizer,
        objective=objective,
        log_iterations=log_iterations,
        n_restarts=n_restarts,
        max_nfev=max_nfev,
        iteration_callback=iteration_callback,
        feller_mode=feller_mode,
        feller_weight=feller_weight,
        stationarity_mode=stationarity_mode,
        stationarity_weight=stationarity_weight,
    )

    t0 = time.perf_counter()
    result = calibrator.calibrate(market_data)
    elapsed = time.perf_counter() - t0

    # Extract estimated params from the model
    est = (
        result.model.get_parameters() if hasattr(result.model, "get_parameters") else {}
    )
    # GARCH models store ω on the annualised scale (so the dt-free
    # simulator/pricer recursion stays self-consistent with σ₀²), but the
    # user enters / compares ω at the per-period (daily) scale. Convert the
    # estimated ω back to per-period so recovery vs. true_params is
    # apples-to-apples. α/β/θ/γ are dimensionless and unchanged.
    if model_key in ("garch", "ngarch", "gjr_garch") and "omega" in est:
        af = float(getattr(market_data, "annualization_factor", 1.0))
        est = {**est, "omega": float(est["omega"]) / af}
    # Filter out non-calibratable extras (e.g., GARCH has 'sigma0' which is not in true_params)
    est_filtered = {k: float(v) for k, v in est.items() if k in true_params}

    rel_error = {}
    for k, true_v in true_params.items():
        est_v = est_filtered.get(k, np.nan)
        denom = max(abs(true_v), RECOVERY_DENOM_CLAMP)
        rel_error[k] = float(abs(est_v - true_v) / denom)

    diag = result.diagnostics or {}
    grad = diag.get("grad_norm")
    return CalibrationRunSummary(
        solver_name=solver_name,
        result=result,
        elapsed=elapsed,
        estimated_params=est_filtered,
        true_params=dict(true_params),
        relative_recovery_error=rel_error,
        objective_name=objective_name,
        final_loss=float(result.objective_value),
        grad_norm=float(grad) if grad is not None else None,
    )


def calibrate_many(
    model_key: str,
    market_data,
    solver_names: tuple[str, ...],
    true_params: dict[str, float],
    *,
    objective_name: str = "price_mse",
    objective_settings: dict[str, Any] | None = None,
    constraint_settings: dict[str, Any] | None = None,
    n_restarts: int = 5,
    max_nfev: int = 200,
    de_seed: int = 42,
    log_iterations: bool = True,
    progress_cb=None,
) -> dict[str, CalibrationRunSummary]:
    """Run calibrations sequentially for a tuple of solver names."""
    results: dict[str, CalibrationRunSummary] = {}
    n = len(solver_names)
    for i, name in enumerate(solver_names):
        if progress_cb is not None:
            progress_cb(i, n, name)
        try:
            results[name] = calibrate_with(
                model_key=model_key,
                market_data=market_data,
                solver_name=name,
                true_params=true_params,
                objective_name=objective_name,
                objective_settings=objective_settings,
                constraint_settings=constraint_settings,
                n_restarts=n_restarts,
                max_nfev=max_nfev,
                de_seed=de_seed,
                log_iterations=log_iterations,
            )
        except Exception as exc:  # noqa: BLE001 — surfaced to the UI
            logger.exception("Solver %s failed for model %s", name, model_key)
            results[name] = CalibrationRunSummary.failure(
                name,
                true_params,
                exc,
                objective_name=objective_name,
            )
    if progress_cb is not None:
        progress_cb(n, n, None)
    return results


# Nested result shape produced by ``calibrate_multi``: indexed first by
# candidate model key, then by solver name. An ``UNSUPPORTED`` sentinel
# is stored when a solver is not available for a model so the UI can
# render a "skipped" row rather than silently dropping the pair.
MultiModelRunResult = dict[str, dict[str, CalibrationRunSummary]]


def calibrate_multi(
    candidate_models: tuple[str, ...],
    market_data,
    solver_names: tuple[str, ...],
    *,
    true_params_per_model: dict[str, dict[str, float]] | None = None,
    constraint_settings: dict[str, Any] | None = None,
    n_restarts: int = 5,
    max_nfev: int = 200,
    de_seed: int = 42,
    log_iterations: bool = True,
    progress_cb=None,
) -> MultiModelRunResult:
    """Run every ``(model, solver)`` pair sequentially.

    ``true_params_per_model`` is the synthetic ground truth used for the
    recovery-error metric. Pass an empty dict to skip recovery tracking
    (e.g. in real-data mode where there is no ground truth).
    """
    from config.model_registry import supported_solvers  # local to avoid cycle

    true_params_per_model = true_params_per_model or {}
    results: MultiModelRunResult = {}

    for model_key in candidate_models:
        model_supported = set(supported_solvers(model_key))
        per_solver: dict[str, CalibrationRunSummary] = {}
        # Use the candidate's own defaults when no synthetic truth exists
        # for it (e.g. it isn't the generator). The recovery error is
        # then a sanity proxy, not a strict identification check.
        truth = true_params_per_model.get(model_key, {})

        for solver_name in solver_names:
            if solver_name not in model_supported:
                per_solver[solver_name] = CalibrationRunSummary(
                    solver_name=solver_name,
                    result=None,
                    elapsed=-1.0,
                    estimated_params={},
                    true_params=dict(truth),
                    relative_recovery_error={},
                    error=f"solver '{solver_name}' is not supported for model '{model_key}'",
                )
                if progress_cb is not None:
                    progress_cb(model_key, solver_name, "skipped")
                continue

            if progress_cb is not None:
                progress_cb(model_key, solver_name, "running")
            try:
                per_solver[solver_name] = calibrate_with(
                    model_key=model_key,
                    market_data=market_data,
                    solver_name=solver_name,
                    true_params=truth,
                    constraint_settings=constraint_settings,
                    n_restarts=n_restarts,
                    max_nfev=max_nfev,
                    de_seed=de_seed,
                    log_iterations=log_iterations,
                )
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI
                logger.exception(
                    "Solver %s failed for model %s",
                    solver_name,
                    model_key,
                )
                per_solver[solver_name] = CalibrationRunSummary.failure(
                    solver_name,
                    truth,
                    exc,
                )

        results[model_key] = per_solver

    return results
