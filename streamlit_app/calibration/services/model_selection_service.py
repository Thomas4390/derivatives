"""Cross-model information criteria (AIC / BIC / LR-test).

Given the nested ``dict[model][solver][objective]`` result produced by
``run_multi_model_with_live_progress``, selects the best solver per model
and computes the AIC/BIC scores so the Compare tab can rank candidates.

For surface models we lean on a Gaussian-likelihood approximation built
from the IV residual standard deviation (``rmse_iv``). The GARCH family
already stores its true MLE log-likelihood in ``result.diagnostics`` so
we reuse that directly. The Gaussian approximation is **only valid
under iid normal residuals** — the table caption warns about it.
"""

from __future__ import annotations

from collections.abc import Iterable
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
    "bates": "heston",  # Bates restrict to Heston by setting jump params to 0
    "ngarch": "garch",  # NGARCH → GARCH when leverage θ=0
    "gjr_garch": "garch",  # GJR → GARCH when γ=0
}


@dataclass(frozen=True)
class ModelSelectionRow:
    model_key: str
    model_label: str
    best_solver: str
    k: int  # number of free parameters
    n: int  # sample size (quotes or returns)
    rmse_iv: float | None  # surface models only
    log_likelihood: float  # log L at the calibrated optimum
    aic: float  # 2k − 2 logL
    bic: float  # k log n − 2 logL
    delta_aic: float = 0.0  # AIC − min(AIC) across models
    lr_p_value: float | None = None  # p-value vs nested model (None if N/A)


def _has_price(result) -> bool:
    """True when the result carries a finite price RMSE (in dollars)."""
    rmse = getattr(result, "rmse_price", None)
    return rmse is not None and np.isfinite(rmse)


def best_slot(
    candidates: Iterable[tuple[str, Any]],
) -> tuple[str, Any] | None:
    """Pick the best ``(label, summary)`` among one model's calibration slots.

    Scoring never mixes units. ``rmse_price`` is in option-price dollars;
    ``objective_value`` is ``rss/2`` for surface calibrators or a (possibly
    large-negative) NLL for returns-GARCH — comparing the two in one ``min``
    let a failure-branch slot (which returns a default model with no
    ``rmse_price`` but a tiny ``objective_value``) beat a converged fit. So:

    - ignore failure results (``success is False``) when any success survives;
    - if **any** successful slot exposes a finite ``rmse_price``, rank **only**
      the priced slots by ``rmse_price``;
    - otherwise (the whole returns-GARCH family) rank every slot by
      ``objective_value``.

    Returns ``None`` when no slot has a result. Accepts the flattened
    ``(label, summary)`` stream from either results schema.
    """
    slots = [(lbl, s) for lbl, s in candidates if s.result is not None]
    if not slots:
        return None
    live = [(lbl, s) for lbl, s in slots if getattr(s.result, "success", True)]
    pool = live or slots  # all-failed model still anchors on its least-bad slot
    priced = [(lbl, s) for lbl, s in pool if _has_price(s.result)]
    if priced:
        return min(priced, key=lambda ls: float(ls[1].result.rmse_price))
    return min(pool, key=lambda ls: float(ls[1].result.objective_value))


def _flatten_solver_slots(per_solver: dict) -> list[tuple[str, Any]]:
    """Normalise both result schemas to a ``(label, summary)`` list.

    ``label`` is ``solver`` (legacy flat) or ``solver/objective`` (nested).
    """
    out: list[tuple[str, Any]] = []
    for solver_name, slot in per_solver.items():
        if isinstance(slot, dict):
            out.extend((f"{solver_name}/{obj}", s) for obj, s in slot.items())
        else:
            out.append((solver_name, slot))
    return out


def _best_solver(per_solver: dict) -> tuple[str, Any] | None:
    """Best ``(label, summary)`` slot for one model (both result schemas)."""
    return best_slot(_flatten_solver_slots(per_solver))


def _surface_log_likelihood(rmse_iv: float, n: int) -> float:
    """Gaussian-likelihood approximation: residuals ~ N(0, σ²) with σ = RMSE.

    ``rmse_iv`` arrives in **basis points** — ``backend.calibration.utils.
    compute_rmse_iv`` scales the IV RMSE by 1e4 — so it must be converted back
    to a decimal standard deviation before squaring. Squaring the raw bps value
    inflates the residual variance by 1e8, which flips the sign of the
    log-likelihood and makes the absolute AIC/BIC physically meaningless (the
    within-family *ranking* survives because the offset is constant across rows,
    but the displayed numbers do not).
    """
    sd = float(rmse_iv) / 1e4
    resid_var = max(sd * sd, 1e-30)
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
    results: dict,
    market_data,
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
        # The registered custom model is a surface model too (it joins the
        # surface pickers) but lives outside the static SURFACE_FAMILY tuple.
        if model_key in SURFACE_FAMILY or model_key == "custom":
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
