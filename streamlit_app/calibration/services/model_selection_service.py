"""Cross-model information criteria (AIC / BIC / LR-test).

Given the nested ``dict[model][solver]`` result produced by
``calibrate_multi``, selects the best solver per model and computes the
AIC/BIC scores so the Compare tab can rank candidates.

For surface models we lean on a Gaussian-likelihood approximation built
from the IV residual standard deviation (``rmse_iv``). The GARCH family
already stores its true MLE log-likelihood in ``result.diagnostics`` so
we reuse that directly. The Gaussian approximation is **only valid
under iid normal residuals** — the table caption warns about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import chi2

from config.constants import GARCH_FAMILY, MODEL_DISPLAY_NAMES, SURFACE_FAMILY
from config.model_registry import get_spec

# Nesting relations used for the likelihood-ratio test. ``nested`` is the
# *simpler* model — restricting parameters in the *full* model recovers
# it. Reduce the LR statistic to ``2 * (logL_full - logL_nested)``.
_NESTED_PAIRS: dict[str, str] = {
    "bates": "heston",          # Bates restrict to Heston by setting jump params to 0
    "ngarch": "garch",          # NGARCH → GARCH when leverage θ=0
    "gjr_garch": "garch",       # GJR → GARCH when γ=0
}


@dataclass(frozen=True)
class ModelSelectionRow:
    model_key: str
    model_label: str
    best_solver: str
    k: int                       # number of free parameters
    n: int                       # sample size (quotes or returns)
    rmse_iv: float | None        # surface models only
    log_likelihood: float        # log L at the calibrated optimum
    aic: float                   # 2k − 2 logL
    bic: float                   # k log n − 2 logL
    delta_aic: float = 0.0       # AIC − min(AIC) across models
    lr_p_value: float | None = None  # p-value vs nested model (None if N/A)


def _best_solver(per_solver: dict) -> tuple[str, Any] | None:
    """Pick the (solver, objective) slot with the lowest RMSE-price (surface) or
    objective_value (returns). Returns ``None`` if no slot succeeded.

    Handles both legacy ``dict[solver, Summary]`` and new
    ``dict[solver, dict[objective, Summary]]`` schemas.
    """
    best: tuple[str, Any] | None = None
    best_score = np.inf
    for solver_name, slot in per_solver.items():
        # Normalise to a list of (label, summary) pairs covering both schemas.
        if isinstance(slot, dict):
            candidates = list(slot.items())  # (objective_name, summary)
            label_fmt = lambda obj_name: f"{solver_name}/{obj_name}"  # noqa: E731
        else:
            candidates = [(solver_name, slot)]
            label_fmt = lambda _: solver_name  # noqa: E731
        for obj_label, s in candidates:
            if s.result is None:
                continue
            if s.result.rmse_price is not None and not np.isnan(s.result.rmse_price):
                score = float(s.result.rmse_price)
            else:
                score = float(s.result.objective_value)
            if score < best_score:
                best_score = score
                best = (label_fmt(obj_label), s)
    return best


def _surface_log_likelihood(rmse_iv: float, n: int) -> float:
    """Gaussian-likelihood approximation: residuals ~ N(0, σ²) with σ = RMSE.

    ``rmse_iv`` is stored as a decimal (e.g. 0.002 ≡ 20 bps) so we use it
    directly as the residual standard deviation.
    """
    resid_var = max(float(rmse_iv) ** 2, 1e-30)
    return -0.5 * n * (np.log(2.0 * np.pi) + np.log(resid_var) + 1.0)


def _extract_garch_log_likelihood(result) -> float | None:
    diag = result.diagnostics or {}
    if "log_likelihood" in diag and np.isfinite(diag["log_likelihood"]):
        return float(diag["log_likelihood"])
    # Fallback: objective_value = -log L (the GARCH calibrator stores
    # the negative log-likelihood as its scalar objective).
    if hasattr(result, "objective_value") and np.isfinite(result.objective_value):
        return -float(result.objective_value)
    return None


def _sample_size(market_data) -> int:
    """Quote count for surface, return count for returns."""
    if hasattr(market_data, "n_quotes"):
        return int(market_data.n_quotes)
    # OptionMarketData has n_quotes; HistoricalReturns exposes returns array
    if hasattr(market_data, "log_returns"):
        return int(len(market_data.log_returns))
    if hasattr(market_data, "returns"):
        return int(len(market_data.returns))
    return 1


def compute_info_criteria(
    results: dict, market_data,
) -> list[ModelSelectionRow]:
    """Build a sorted list of ``ModelSelectionRow`` — one per candidate.

    Sorted by ascending AIC (best first). Rows for candidates whose
    best solver failed (or for ``iv_gbm`` which has no objective in the
    information-criteria sense) are dropped.
    """
    rows: list[ModelSelectionRow] = []
    n = _sample_size(market_data)

    for model_key, per_solver in results.items():
        spec = get_spec(model_key)
        k = spec.n_params
        chosen = _best_solver(per_solver)
        if chosen is None:
            continue
        solver_name, summary = chosen
        result = summary.result

        rmse_iv_value = None
        if model_key in SURFACE_FAMILY:
            if result.rmse_iv is None or not np.isfinite(result.rmse_iv):
                continue
            rmse_iv_value = float(result.rmse_iv)
            log_l = _surface_log_likelihood(rmse_iv_value, n)
        elif model_key in GARCH_FAMILY:
            log_l_value = _extract_garch_log_likelihood(result)
            if log_l_value is None:
                continue
            log_l = log_l_value
        else:
            continue  # iv_gbm has no fitted likelihood

        aic = 2.0 * k - 2.0 * log_l
        bic = k * np.log(max(n, 1)) - 2.0 * log_l
        rows.append(
            ModelSelectionRow(
                model_key=model_key,
                model_label=MODEL_DISPLAY_NAMES.get(model_key, model_key),
                best_solver=solver_name,
                k=k,
                n=n,
                rmse_iv=rmse_iv_value,
                log_likelihood=log_l,
                aic=aic,
                bic=bic,
            )
        )

    if not rows:
        return rows

    # ΔAIC against the best
    min_aic = min(r.aic for r in rows)
    enriched = [
        ModelSelectionRow(
            model_key=r.model_key,
            model_label=r.model_label,
            best_solver=r.best_solver,
            k=r.k,
            n=r.n,
            rmse_iv=r.rmse_iv,
            log_likelihood=r.log_likelihood,
            aic=r.aic,
            bic=r.bic,
            delta_aic=r.aic - min_aic,
            lr_p_value=None,
        )
        for r in rows
    ]

    # Likelihood-ratio test against the nested simpler model when both
    # are present. The LR statistic is 2·(logL_full - logL_nested) with
    # df = k_full - k_nested, distributed as χ² under the null.
    by_key = {r.model_key: r for r in enriched}
    final: list[ModelSelectionRow] = []
    for r in enriched:
        nested_key = _NESTED_PAIRS.get(r.model_key)
        if nested_key is None or nested_key not in by_key:
            final.append(r)
            continue
        nested = by_key[nested_key]
        lr_stat = 2.0 * (r.log_likelihood - nested.log_likelihood)
        df = max(r.k - nested.k, 1)
        p_value = float(chi2.sf(max(lr_stat, 0.0), df))
        final.append(
            ModelSelectionRow(
                model_key=r.model_key,
                model_label=r.model_label,
                best_solver=r.best_solver,
                k=r.k,
                n=r.n,
                rmse_iv=r.rmse_iv,
                log_likelihood=r.log_likelihood,
                aic=r.aic,
                bic=r.bic,
                delta_aic=r.delta_aic,
                lr_p_value=p_value,
            )
        )

    final.sort(key=lambda r: r.aic)
    return final
