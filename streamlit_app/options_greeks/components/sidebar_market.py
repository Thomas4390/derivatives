"""Market parameters section of the sidebar."""

import streamlit as st
from config.constants import DEFAULT_RISK_FREE_RATE, DEFAULT_SPOT_PRICE


def render_market_params() -> tuple[float, float]:
    """Render market parameters inputs."""
    st.markdown(
        """
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
        <span style="font-size: 1rem;">🌐</span>
        <span style="font-size: 0.75rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em;">Market Parameters</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        spot_price = st.number_input(
            "Spot Price ($)",
            value=DEFAULT_SPOT_PRICE,
            step=1.0,
            format="%.2f",
            help="Current underlying price",
        )

    with col2:
        risk_free_rate = st.number_input(
            "Risk-Free Rate",
            value=DEFAULT_RISK_FREE_RATE,
            step=0.01,
            format="%.3f",
            help="Annual risk-free rate",
        )

    st.markdown(
        "<div style='height: 0.75rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 1rem;'></div>",
        unsafe_allow_html=True,
    )

    return spot_price, risk_free_rate
