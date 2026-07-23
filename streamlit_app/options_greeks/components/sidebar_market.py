"""Market parameters section of the sidebar."""

import streamlit as st
from services.state_manager import MARKET_Q_KEY, MARKET_RATE_KEY, MARKET_SPOT_KEY


def render_market_params() -> tuple[float, float, float]:
    """Render market parameters inputs. Returns (spot, rate, dividend_yield)."""
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

    # The values come from session_state (seeded in init_session_state) via `key=`,
    # so a restored setup can populate them before these widgets render.
    with col1:
        spot_price = st.number_input(
            "Spot Price ($)",
            step=1.0,
            format="%.2f",
            help="Current underlying price",
            key=MARKET_SPOT_KEY,
        )

    with col2:
        risk_free_rate = st.number_input(
            "Risk-Free Rate",
            step=0.01,
            format="%.3f",
            help="Annual risk-free rate",
            key=MARKET_RATE_KEY,
        )

    dividend_yield = st.number_input(
        "Dividend Yield (q)",
        min_value=0.0,
        max_value=0.20,
        step=0.005,
        format="%.4f",
        help="Annual continuous dividend yield",
        key=MARKET_Q_KEY,
    )

    st.markdown(
        "<div style='height: 0.75rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 1rem;'></div>",
        unsafe_allow_html=True,
    )

    return spot_price, risk_free_rate, dividend_yield
