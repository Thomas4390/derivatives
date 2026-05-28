"""
Sidebar component for Options Greeks Explorer.

Thin orchestrator that delegates to sub-modules:
  - sidebar_market: Market parameters (spot, risk-free rate)
  - sidebar_strategy: Strategy builder with leg editors
  - sidebar_positions: Active positions display
"""

import streamlit as st
from components.sidebar_market import render_market_params
from components.sidebar_positions import render_positions_section
from components.sidebar_strategy import render_strategy_builder


def render_sidebar(positions: list, stock_position) -> tuple[float, float]:
    """
    Render the complete sidebar with all controls.

    Args:
        positions: List of option position dicts
        stock_position: Stock position dict or None

    Returns:
        Tuple of (spot_price, risk_free_rate)
    """
    with st.sidebar:
        # Logo/Brand section
        st.markdown(
            """
        <div style="text-align: center; padding: 0.75rem 0 1.25rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.25rem;">
            <div style="font-size: 1.75rem; margin-bottom: 0.25rem;">📊</div>
            <div style="font-size: 1rem; font-weight: 600; color: #1a365d;">Options Greeks Explorer</div>
            <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 0.15rem;">Black-Scholes Model</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Market Parameters
        spot_price, risk_free_rate = render_market_params()

        # Strategy Builder
        render_strategy_builder(spot_price, risk_free_rate)

        # Current Positions
        render_positions_section(positions, stock_position)

    return spot_price, risk_free_rate
