"""Cross-model information-criteria table rendering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from services.model_selection_service import ModelSelectionRow


def _badge(text: str, color: str) -> str:
    return (
        f"<span style='display:inline-block;padding:2px 8px;border-radius:4px;"
        f"background:{color};color:white;font-size:0.72rem;font-weight:600'>"
        f"{text}</span>"
    )


def render_info_criteria_table(rows: list[ModelSelectionRow]) -> None:
    """Render the AIC/BIC cross-model table with model-selection badges.

    Adds three pedagogical badges:
    - 🏅 Best AIC / Best BIC: lowest AIC and BIC respectively
    - 🟢 Significantly better: nested LR-test p-value < 0.05
    - 🟡 Overfit suspected: more complex model has worse BIC than its
      simpler nested counterpart (penalty wins over fit improvement)
    """
    if not rows:
        st.info(
            "Information criteria are unavailable. Run the calibration first; "
            "this section needs at least one candidate that has produced an "
            "RMSE (surface) or log-likelihood (GARCH)."
        )
        return

    best_aic = min(r.aic for r in rows)
    best_bic = min(r.bic for r in rows)
    by_key = {r.model_key: r for r in rows}

    def _badges_for(r: ModelSelectionRow) -> str:
        out: list[str] = []
        if r.aic == best_aic:
            out.append(_badge("🏅 Best AIC", "#0d9488"))
        if r.bic == best_bic:
            out.append(_badge("🏅 Best BIC", "#7c3aed"))
        if r.lr_p_value is not None and r.lr_p_value < 0.05:
            out.append(_badge("🟢 LR p < 0.05", "#059669"))
        # Overfit detection: complex model has *worse* BIC than its
        # simpler nested counterpart → the extra parameters didn't earn
        # their penalty.
        nested_keys = {"bates": "heston", "ngarch": "garch", "gjr_garch": "garch"}
        nk = nested_keys.get(r.model_key)
        if nk and nk in by_key and r.bic > by_key[nk].bic:
            out.append(_badge("🟡 Overfit?", "#d97706"))
        return " ".join(out)

    df = pd.DataFrame([
        {
            "Model": r.model_label,
            "Best solver": r.best_solver,
            "k": r.k,
            "N": r.n,
            "RMSE IV (bps)": (
                r.rmse_iv * 1e4 if r.rmse_iv is not None else np.nan
            ),
            "log L": r.log_likelihood,
            "AIC": r.aic,
            "ΔAIC": r.delta_aic,
            "BIC": r.bic,
            "LR p-value": (
                r.lr_p_value if r.lr_p_value is not None else np.nan
            ),
        }
        for r in rows
    ])

    styled = df.style.format(
        {
            "RMSE IV (bps)": "{:.2f}",
            "log L": "{:.2f}",
            "AIC": "{:.2f}",
            "ΔAIC": "{:.2f}",
            "BIC": "{:.2f}",
            "LR p-value": "{:.4f}",
        },
        na_rep="—",
    )
    st.dataframe(styled, width="stretch", hide_index=True)
    # Render the badges below as a "verdict per model" caption. We use
    # ``st.html`` rather than ``st.markdown(unsafe_allow_html=True)``
    # because recent Streamlit releases sanitise inline ``style`` attrs
    # inside markdown, which would strip the badge colours and leak the
    # raw HTML into the page.
    verdict_html = "<br>".join(
        f"<b>{r.model_label}:</b> {_badges_for(r) or '—'}"
        for r in rows
    )
    st.html(
        f"<div style='margin-top:0.3rem;font-size:0.9rem;line-height:1.6'>"
        f"{verdict_html}</div>"
    )
    st.caption(
        "AIC = 2k − 2·log L, BIC = k·log N − 2·log L. Lower is better. "
        "Surface log-likelihoods use a Gaussian iid approximation on the IV "
        "residuals (σ ≈ RMSE), so AIC/BIC are valid as a *relative* model "
        "ranking but not for absolute statistical inference. The LR-test "
        "p-value compares each model to its nested baseline (Heston ⊂ Bates, "
        "GARCH ⊂ NGARCH, GARCH ⊂ GJR-GARCH)."
    )
