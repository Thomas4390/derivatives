"""
Sidebar component for Options Greeks Explorer.

Modern, clean design with improved user experience.
"""

import streamlit as st
from config.constants import (
    CONTRACT_MULTIPLIER,
    DEFAULT_SPOT_PRICE,
    DEFAULT_RISK_FREE_RATE,
    AVAILABLE_STRATEGIES,
    STRATEGY_DISPLAY_NAMES
)
from config.styles import (
    net_position_card_html,
    position_item_html,
    stock_position_html,
    section_header_html
)


def render_sidebar(
    positions: list,
    stock_position,
    portfolio_class,
    option_position_class,
    stock_position_class
) -> tuple[float, float]:
    """
    Render the complete sidebar with all controls.

    Args:
        positions: Current list of OptionPosition objects
        stock_position: Current StockPosition object or None
        portfolio_class: OptionsPortfolio class for calculations
        option_position_class: OptionPosition class for creating positions
        stock_position_class: StockPosition class for creating positions

    Returns:
        Tuple of (spot_price, risk_free_rate)
    """
    with st.sidebar:
        # Logo/Brand section
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 1.5rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.5rem;">
            <div style="font-size: 2rem; margin-bottom: 0.25rem;">📊</div>
            <div style="font-size: 0.9rem; font-weight: 600; color: #1a365d;">Greeks Explorer</div>
            <div style="font-size: 0.7rem; color: #94a3b8;">Black-Scholes Model</div>
        </div>
        """, unsafe_allow_html=True)

        # Market Parameters
        spot_price, risk_free_rate = _render_market_params()

        # Add Position Section
        _render_add_position_section(
            spot_price,
            risk_free_rate,
            portfolio_class,
            option_position_class,
            stock_position_class
        )

        # Predefined Strategies
        _render_strategies_section(
            spot_price,
            risk_free_rate,
            portfolio_class
        )

        # Current Positions
        _render_positions_section(positions, stock_position)

    return spot_price, risk_free_rate


def _render_market_params() -> tuple[float, float]:
    """Render market parameters inputs."""
    st.markdown(
        '<div class="sidebar-header">Market Parameters</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        spot_price = st.number_input(
            "Spot Price",
            value=DEFAULT_SPOT_PRICE,
            step=1.0,
            format="%.2f",
            help="Current underlying price"
        )

    with col2:
        risk_free_rate = st.number_input(
            "Risk-Free Rate",
            value=DEFAULT_RISK_FREE_RATE,
            step=0.01,
            format="%.3f",
            help="Annual risk-free rate"
        )

    return spot_price, risk_free_rate


def _render_add_position_section(
    spot_price: float,
    risk_free_rate: float,
    portfolio_class,
    option_position_class,
    stock_position_class
) -> None:
    """Render the add position section."""
    st.markdown(
        '<div class="sidebar-header">Add Position</div>',
        unsafe_allow_html=True
    )

    tab_option, tab_stock = st.tabs(["Option", "Stock"])

    with tab_option:
        _render_option_form(
            spot_price,
            risk_free_rate,
            portfolio_class,
            option_position_class
        )

    with tab_stock:
        _render_stock_form(spot_price, stock_position_class)


def _render_option_form(
    spot_price: float,
    risk_free_rate: float,
    portfolio_class,
    option_position_class
) -> None:
    """Render the option position form."""
    with st.form("add_option", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            option_type = st.selectbox(
                "Type",
                ["call", "put"],
                format_func=lambda x: x.upper()
            )

        with col2:
            position_type = st.selectbox(
                "Direction",
                ["long", "short"],
                format_func=lambda x: x.upper()
            )

        col3, col4 = st.columns(2)

        with col3:
            strike = st.number_input(
                "Strike",
                value=spot_price,
                step=1.0,
                format="%.2f"
            )

        with col4:
            quantity = st.number_input(
                "Contracts",
                value=1,
                min_value=1,
                step=1
            )

        # Calculate premium
        portfolio_temp = portfolio_class(spot_price, risk_free_rate)
        position_temp = option_position_class(option_type, position_type, strike, quantity)
        portfolio_temp.add_option_position(position_temp)
        premium = position_temp.premium_paid
        total_cost = premium * quantity * CONTRACT_MULTIPLIER

        # Display cost summary
        st.markdown(f"""
        <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; font-size: 0.85rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="color: #64748b;">Premium/contract:</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-weight: 600;">${premium:.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #64748b;">Total {'debit' if position_type == 'long' else 'credit'}:</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-weight: 600; color: {'#dc2626' if position_type == 'long' else '#059669'};">${total_cost:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.form_submit_button("Add Option", use_container_width=True, type="primary"):
            position = option_position_class(
                option_type, position_type, strike, quantity, premium
            )
            st.session_state.positions.append(position)
            st.rerun()


def _render_stock_form(spot_price: float, stock_position_class) -> None:
    """Render the stock position form."""
    with st.form("add_stock", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            stock_position_type = st.selectbox(
                "Direction",
                ["long", "short"],
                format_func=lambda x: x.upper(),
                key="stock_dir"
            )

        with col2:
            stock_quantity = st.number_input(
                "Shares",
                value=100,
                min_value=1,
                step=1
            )

        stock_entry_price = st.number_input(
            "Entry Price",
            value=spot_price,
            step=1.0,
            format="%.2f"
        )

        stock_cost = stock_entry_price * stock_quantity

        st.markdown(f"""
        <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; font-size: 0.85rem;">
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #64748b;">Total {'cost' if stock_position_type == 'long' else 'credit'}:</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-weight: 600; color: {'#dc2626' if stock_position_type == 'long' else '#059669'};">${stock_cost:,.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.form_submit_button("Add Stock", use_container_width=True, type="primary"):
            stock_pos = stock_position_class(
                stock_position_type, stock_quantity, stock_entry_price
            )
            st.session_state.stock_position = stock_pos
            st.rerun()


def _render_strategies_section(
    spot_price: float,
    risk_free_rate: float,
    portfolio_class
) -> None:
    """Render the predefined strategies section."""
    st.markdown(
        '<div class="sidebar-header">Quick Strategies</div>',
        unsafe_allow_html=True
    )

    strategy = st.selectbox(
        "Select strategy",
        AVAILABLE_STRATEGIES,
        format_func=lambda x: STRATEGY_DISPLAY_NAMES.get(x, x) if x else "Choose a strategy...",
        label_visibility="collapsed"
    )

    if st.button("Apply Strategy", use_container_width=True, disabled=not strategy):
        if strategy:
            portfolio = portfolio_class(spot_price, risk_free_rate)
            portfolio.create_strategy(strategy)
            st.session_state.positions = portfolio.positions
            st.session_state.stock_position = portfolio.stock_position
            st.rerun()


def _render_positions_section(positions: list, stock_position) -> None:
    """Render the current positions section."""
    st.markdown(
        '<div class="sidebar-header">Current Positions</div>',
        unsafe_allow_html=True
    )

    if not positions and not stock_position:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem; color: #94a3b8; font-size: 0.85rem;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📋</div>
            <div>No positions yet</div>
            <div style="font-size: 0.75rem; margin-top: 0.25rem;">Add options or stock above</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Calculate net position
    net_amount = _calculate_net_position(positions, stock_position)

    # Net position banner
    st.markdown(net_position_card_html("", net_amount), unsafe_allow_html=True)

    # Stock position
    if stock_position:
        stock_cost = stock_position.entry_price * stock_position.quantity
        st.markdown(
            stock_position_html(
                quantity=stock_position.quantity,
                position_type=stock_position.position_type,
                entry_price=stock_position.entry_price,
                stock_cost=stock_cost
            ),
            unsafe_allow_html=True
        )

    # Option positions
    for i, pos in enumerate(positions):
        total_amount = pos.premium_paid * pos.quantity * CONTRACT_MULTIPLIER
        shares_controlled = pos.quantity * CONTRACT_MULTIPLIER

        st.markdown(
            position_item_html(
                index=i + 1,
                quantity=pos.quantity,
                position_type=pos.position_type,
                option_type=pos.option_type,
                strike=pos.strike,
                premium=pos.premium_paid,
                total_amount=total_amount,
                shares_controlled=shares_controlled,
                is_long=(pos.position_type == 'long')
            ),
            unsafe_allow_html=True
        )

    # Action buttons
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear All", use_container_width=True, key="clear_all_btn"):
            st.session_state.positions = []
            st.session_state.stock_position = None
            st.rerun()

    with col2:
        if st.button("Remove Last", use_container_width=True, key="remove_last_btn"):
            if st.session_state.positions:
                st.session_state.positions.pop()
            elif st.session_state.stock_position:
                st.session_state.stock_position = None
            st.rerun()


def _calculate_net_position(positions: list, stock_position=None) -> float:
    """Calculate the net debit/credit position including stock."""
    net = sum(
        -pos.premium_paid * pos.quantity * CONTRACT_MULTIPLIER
        if pos.position_type == 'long'
        else pos.premium_paid * pos.quantity * CONTRACT_MULTIPLIER
        for pos in positions
    )

    if stock_position:
        if stock_position.position_type == 'long':
            net -= stock_position.entry_price * stock_position.quantity
        else:
            net += stock_position.entry_price * stock_position.quantity

    return net
