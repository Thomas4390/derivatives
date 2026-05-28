"""Data-source picker — synthetic ground-truth vs embedded SPX snapshot."""

from __future__ import annotations

from typing import Any

import streamlit as st

from services import state_manager
from services.real_data_service import is_dataset_available, list_embedded_snapshots


def render(family: str) -> dict[str, Any]:
    """Returns ``{'source': 'synthetic'|'real', 'real_cfg': {...}}``.

    ``family`` is ``"surface"`` or ``"returns"``. Real SPX data is only
    meaningful for the surface family, so the toggle silently falls
    back to synthetic in returns mode.
    """
    # Real data is only meaningful for surface models — explain the
    # restriction so the implicit synthetic-only fallback doesn't feel
    # like a silent state reset when the user switches to a GARCH model.
    if family != "surface":
        st.subheader("📡 Data Source")
        st.caption(
            "⚙️ Returns-based models (GARCH family) are calibrated on a "
            "synthetic log-return series — real-data switch is hidden."
        )
        state_manager.set("calib_data_source", "synthetic")
        return {"source": "synthetic"}
    if not is_dataset_available():
        st.subheader("📡 Data Source")
        st.caption("⚙️ No SPX snapshot bundled — falling back to synthetic data.")
        state_manager.set("calib_data_source", "synthetic")
        return {"source": "synthetic"}

    st.subheader("📡 Data Source")
    # Pill-shaped segmented buttons — single-select, but matches the
    # visual language of the solver picker (also pills) so the sidebar
    # reads as one consistent set of selectors. ``segmented_control``
    # gives a more "button-row" feel than ``st.radio`` and avoids the
    # full-width radio bullet that looked out of place next to the
    # multi-pill solver row.
    options = ["Synthetic (ground truth)", "Real (SPX)"]
    current_source = state_manager.get("calib_data_source") or "synthetic"
    default = options[0] if current_source == "synthetic" else options[1]
    source = st.segmented_control(
        "source",
        options=options,
        default=default,
        selection_mode="single",
        label_visibility="collapsed",
        key="calib_data_source_picker",
    )
    # ``segmented_control`` returns ``None`` if the user clears the
    # selection — fall back to the previous source so the sidebar
    # never silently flips data sources.
    if source is None:
        source = default
    is_real = source.startswith("Real")
    state_manager.set("calib_data_source", "real" if is_real else "synthetic")

    if not is_real:
        return {"source": "synthetic"}

    entries = list_embedded_snapshots()
    if not entries:
        st.warning("No embedded SPX snapshots found.")
        return {"source": "synthetic"}
    labels = [e.label for e in entries]
    idx = st.selectbox(
        "SPX snapshot",
        range(len(entries)),
        format_func=lambda i: labels[i],
        index=0,
    )
    chosen = entries[idx]

    with st.expander("Filtering"):
        # Two-handle slider — physically impossible for the user to set
        # low > high, which the previous pair of independent sliders
        # allowed and which silently dropped every quote downstream.
        dte_min, dte_max = st.slider(
            "DTE range (days)",
            min_value=1, max_value=90,
            value=(7, 60), step=1,
            help="Keep only quotes whose days-to-expiry fall in this "
                 "window. Very short tenors (< 7 days) have unstable "
                 "bid-ask spreads, and SPX trades thinly past 60 DTE in "
                 "some snapshots.",
        )
        mny_min, mny_max = st.slider(
            "Moneyness range",
            min_value=0.5, max_value=1.5,
            value=(0.85, 1.15), step=0.01,
            help="Keep only quotes with K/S₀ in this range. Deep OTM wings "
                 "are usually noisy.",
        )
        rate = st.number_input(
            "Risk-free rate", value=0.045, step=0.005, format="%.4f",
            help="Annual continuously-compounded risk-free rate used in the "
                 "Black-Scholes IV inversion.",
        )
        div = st.number_input(
            "Dividend yield", value=0.013, step=0.005, format="%.4f",
            help="Annual continuous dividend yield. SPX hovers around 1.3–1.6 %.",
        )

    return {
        "source": "real",
        "real_cfg": {
            "snapshot_key": chosen.snap_iso,
            "dte_min": int(dte_min),
            "dte_max": int(dte_max),
            "moneyness_min": float(mny_min),
            "moneyness_max": float(mny_max),
            "rate": float(rate),
            "dividend_yield": float(div),
        },
    }
