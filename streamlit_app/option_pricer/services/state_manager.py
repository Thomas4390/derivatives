"""
State management for Options Greeks Explorer.

This module provides a centralized interface for managing Streamlit session state.
"""

import streamlit as st
from typing import Any


def init_session_state() -> None:
    """Initialize all session state variables with defaults."""
    if 'positions' not in st.session_state:
        st.session_state.positions = []

    if 'stock_position' not in st.session_state:
        st.session_state.stock_position = None


def get_positions() -> list:
    """
    Get the current list of option positions.

    Returns:
        List of OptionPosition objects
    """
    return st.session_state.get('positions', [])


def get_stock_position():
    """
    Get the current stock position.

    Returns:
        StockPosition object or None
    """
    return st.session_state.get('stock_position', None)


def add_position(position) -> None:
    """
    Add an option position to the portfolio.

    Args:
        position: OptionPosition object to add
    """
    if 'positions' not in st.session_state:
        st.session_state.positions = []
    st.session_state.positions.append(position)


def remove_last_position() -> bool:
    """
    Remove the last option position from the portfolio.

    Returns:
        True if a position was removed, False otherwise
    """
    if st.session_state.positions:
        st.session_state.positions.pop()
        return True
    return False


def set_stock_position(stock_position) -> None:
    """
    Set the stock position.

    Args:
        stock_position: StockPosition object or None
    """
    st.session_state.stock_position = stock_position


def clear_stock_position() -> None:
    """Clear the stock position."""
    st.session_state.stock_position = None


def clear_positions() -> None:
    """Clear all option positions."""
    st.session_state.positions = []


def clear_all() -> None:
    """Clear all positions (options and stock)."""
    st.session_state.positions = []
    st.session_state.stock_position = None


def has_positions() -> bool:
    """
    Check if there are any positions (options or stock).

    Returns:
        True if there are any positions
    """
    return bool(st.session_state.positions) or st.session_state.stock_position is not None


def get_position_count() -> tuple[int, bool]:
    """
    Get the count of positions.

    Returns:
        Tuple of (option_count, has_stock)
    """
    option_count = len(st.session_state.get('positions', []))
    has_stock = st.session_state.get('stock_position') is not None
    return option_count, has_stock


def set_positions_from_strategy(positions: list, stock_position) -> None:
    """
    Set positions from a predefined strategy.

    Args:
        positions: List of OptionPosition objects
        stock_position: StockPosition object or None
    """
    st.session_state.positions = positions
    st.session_state.stock_position = stock_position
