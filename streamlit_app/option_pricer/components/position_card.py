"""
Position card components for Options Greeks Explorer.

This module provides components for displaying option and stock positions.
"""

import streamlit as st
from config.constants import CONTRACT_MULTIPLIER
from config.styles import (
    position_item_html,
    stock_position_html,
    net_position_card_html
)


def render_net_position(net_amount: float) -> None:
    """
    Render the net position card showing total debit/credit.

    Args:
        net_amount: Net position value (positive = credit, negative = debit)
    """
    if net_amount > 0:
        position_type = "💰 Credit"
    elif net_amount < 0:
        position_type = "💸 Debit"
    else:
        position_type = "⚖️ Neutral"

    amount_text = f"${abs(net_amount):.2f}" if net_amount != 0 else "$0.00"

    st.markdown(
        net_position_card_html(position_type, amount_text),
        unsafe_allow_html=True
    )


def render_stock_position(stock_position) -> None:
    """
    Render a stock position card.

    Args:
        stock_position: StockPosition object
    """
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


def render_option_position(position, index: int) -> None:
    """
    Render an option position card.

    Args:
        position: OptionPosition object
        index: Position index (0-based, will be displayed as 1-based)
    """
    total_amount = position.premium_paid * position.quantity * CONTRACT_MULTIPLIER
    shares_controlled = position.quantity * CONTRACT_MULTIPLIER

    st.markdown(
        position_item_html(
            index=index + 1,
            quantity=position.quantity,
            position_type=position.position_type,
            option_type=position.option_type,
            strike=position.strike,
            premium=position.premium_paid,
            total_amount=total_amount,
            shares_controlled=shares_controlled,
            is_long=(position.position_type == 'long')
        ),
        unsafe_allow_html=True
    )


def render_positions_list(
    positions: list,
    stock_position,
    net_amount: float,
    on_clear_all: callable = None,
    on_remove_last: callable = None
) -> None:
    """
    Render the complete positions list with controls.

    Args:
        positions: List of OptionPosition objects
        stock_position: StockPosition object or None
        net_amount: Net position value
        on_clear_all: Callback for clear all button (optional)
        on_remove_last: Callback for remove last button (optional)
    """
    if not positions and not stock_position:
        st.info("No active positions")
        return

    # Net position card
    render_net_position(net_amount)

    # Stock position
    if stock_position:
        render_stock_position(stock_position)

    # Option positions
    for i, pos in enumerate(positions):
        render_option_position(pos, i)

    # Control buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear All", width="stretch", key="clear_all_btn"):
            if on_clear_all:
                on_clear_all()
            st.rerun()

    with col2:
        if st.button("↩️ Remove Last", width="stretch", key="remove_last_btn"):
            if on_remove_last:
                on_remove_last()
            st.rerun()


def calculate_net_position(positions: list, stock_position=None) -> float:
    """
    Calculate the net debit/credit position including stock.

    Args:
        positions: List of OptionPosition objects
        stock_position: StockPosition object or None

    Returns:
        Net position value (positive = credit, negative = debit)
    """
    net = sum(
        -pos.premium_paid * pos.quantity * CONTRACT_MULTIPLIER
        if pos.position_type == 'long'
        else pos.premium_paid * pos.quantity * CONTRACT_MULTIPLIER
        for pos in positions
    )

    # Add stock cost if present
    if stock_position:
        if stock_position.position_type == 'long':
            net -= stock_position.entry_price * stock_position.quantity
        else:  # short stock
            net += stock_position.entry_price * stock_position.quantity

    return net
