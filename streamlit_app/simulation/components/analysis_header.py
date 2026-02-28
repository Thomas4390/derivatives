"""
Analysis Header component for Monte Carlo Simulation Explorer.

Provides a common header for analysis tabs with Run button,
stale results warning, and summary metrics.
"""

from typing import Any

import streamlit as st
from config.constants import PRICE_MODELS, VOLATILITY_MODELS
from config.styles import stale_results_warning_html, strategy_collapsed_html
from services.state_manager import are_results_stale

# Import strategy display names
from streamlit_app.options_greeks.config.constants import STRATEGY_DISPLAY_NAMES


def render_analysis_header(
    analysis_type: str,
    params: dict[str, Any],
    result: Any | None = None
) -> bool:
    """
    Render the analysis header with Run button and status.

    Parameters
    ----------
    analysis_type : str
        Type of analysis: 'price', 'volatility', or 'pnl'
    params : dict
        Current simulation parameters
    result : Any, optional
        Current simulation result (if any)

    Returns
    -------
    bool
        True if Run button was clicked
    """
    # Check if results are stale
    is_stale = result is not None and are_results_stale(analysis_type)

    # Show stale warning if needed
    if is_stale:
        st.markdown(stale_results_warning_html(), unsafe_allow_html=True)

    # Run button
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        run_clicked = st.button(
            "🚀 Run Simulation",
            type="primary",
            width="stretch",
            key=f"run_{analysis_type}"
        )

    # Show summary metrics if we have results
    if result is not None:
        st.markdown("---")
        _render_summary_metrics(analysis_type, params, result)

    return run_clicked


def _render_summary_metrics(
    analysis_type: str,
    params: dict[str, Any],
    result: Any
) -> None:
    """Render summary metrics for the analysis result."""
    col1, col2, col3, col4 = st.columns(4)

    if analysis_type == 'price':
        with col1:
            st.metric("Model", PRICE_MODELS.get(params['price_model'], params['price_model']))
        with col2:
            st.metric("Paths Simulated", f"{result.num_paths:,}")
        with col3:
            st.metric("Time Steps", f"{result.num_steps:,}")
        with col4:
            st.metric("Computation Time", f"{result.computation_time*1000:.1f} ms")

    elif analysis_type == 'volatility':
        with col1:
            st.metric("Model", VOLATILITY_MODELS.get(params['vol_model'], params['vol_model']))
        with col2:
            st.metric("Paths Simulated", f"{result.num_paths:,}")
        with col3:
            st.metric("Time Steps", f"{result.num_steps:,}")
        with col4:
            st.metric("Computation Time", f"{result.computation_time*1000:.1f} ms")

    elif analysis_type == 'pnl':
        with col1:
            st.metric("Model", PRICE_MODELS.get(params['price_model'], params['price_model']))
        with col2:
            st.metric("Scenarios Simulated", f"{result['num_paths']:,}")
        with col3:
            st.metric("P(Profit)", f"{result['risk_metrics'].prob_profit:.1%}")
        with col4:
            st.metric("Computation Time", f"{result['computation_time']*1000:.1f} ms")


def render_strategy_summary_compact(
    params: dict[str, Any],
    collapsed: bool = True
) -> None:
    """
    Render a compact strategy summary for analysis tabs.

    Shows the current strategy configuration in a collapsed view.
    """
    positions = params.get('option_positions', [])
    stock_position = params.get('stock_position')

    if len(positions) == 0:
        st.info("No strategy configured. Go to Configuration tab to set up option positions.")
        return

    # Calculate net cost
    from streamlit_app.options_greeks.config.constants import CONTRACT_MULTIPLIER
    total_net_cost = 0.0

    # Stock cost
    if stock_position:
        stock_cost = stock_position.entry_price * stock_position.quantity
        if stock_position.position_type == 'long':
            total_net_cost -= stock_cost
        else:
            total_net_cost += stock_cost

    # Option costs
    for pos in positions:
        total_cost = pos.premium * pos.quantity * CONTRACT_MULTIPLIER
        if pos.position_type == 'long':
            total_net_cost -= total_cost
        else:
            total_net_cost += total_cost

    # Get strategy name
    strategy_name = st.session_state.get('pnl_last_strategy', 'custom')
    display_name = STRATEGY_DISPLAY_NAMES.get(strategy_name, "Custom Strategy")

    has_stock = stock_position is not None

    st.markdown(
        strategy_collapsed_html(display_name, len(positions), has_stock, total_net_cost),
        unsafe_allow_html=True
    )


def render_no_results_message(analysis_type: str) -> None:
    """Render a helpful message when no results are available."""
    if analysis_type == 'price':
        st.info("Run a price simulation to see the results.")
        st.markdown("""
        **How to use:**
        1. Configure your parameters in the Configuration tab
        2. Select a price model (GBM, Heston, Merton, Bates, or SABR)
        3. Click **Run Simulation** to generate price paths
        """)
    elif analysis_type == 'volatility':
        st.info("Run a volatility simulation to see the results.")
        st.markdown("""
        **How to use:**
        1. Configure your parameters in the Configuration tab
        2. Select a volatility model (GARCH, NGARCH, GJR-GARCH, or EGARCH)
        3. Click **Run Simulation** to generate volatility paths
        """)
    elif analysis_type == 'pnl':
        st.info("Run a P&L simulation to see the results.")
        st.markdown("""
        **How to use:**
        1. Configure your market parameters in the Configuration tab
        2. Add option legs using the Strategy Builder
        3. Set simulation parameters (model, expected return, paths)
        4. Click **Run Simulation** to analyze P&L distribution
        """)
