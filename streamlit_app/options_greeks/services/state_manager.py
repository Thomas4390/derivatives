"""
State management for Options Greeks Explorer.

This module provides a centralized interface for managing Streamlit session state.
Uses simple dicts for positions to avoid serialization issues and improve maintainability.

Position dict formats:
    Option:
        {
            'option_type': 'call' | 'put',
            'position_type': 'long' | 'short',
            'strike': float,
            'quantity': int,
            'premium_paid': float,
            'dte_days': int (optional),
            'volatility': float (optional)
        }

    Stock:
        {
            'position_type': 'long' | 'short',
            'quantity': int,
            'entry_price': float
        }
"""

import streamlit as st


def init_session_state() -> None:
    """Initialize all session state variables with defaults."""
    if "positions" not in st.session_state:
        st.session_state.positions = []

    if "stock_position" not in st.session_state:
        st.session_state.stock_position = None

    if "sp_mode" not in st.session_state:
        st.session_state.sp_mode = False

    for key in ("sp_config", "sp_result", "sp_greeks", "sp_product_type"):
        if key not in st.session_state:
            st.session_state[key] = None


def get_positions() -> list[dict]:
    """
    Get the current list of option positions.

    Returns:
        List of option position dicts
    """
    return st.session_state.get("positions", [])


def get_stock_position() -> dict | None:
    """
    Get the current stock position.

    Returns:
        Stock position dict or None
    """
    return st.session_state.get("stock_position", None)


def add_position(position: dict) -> None:
    """
    Add an option position to the portfolio.

    Args:
        position: Option position dict with keys:
            - option_type: 'call' or 'put'
            - position_type: 'long' or 'short'
            - strike: float
            - quantity: int
            - premium_paid: float
    """
    if "positions" not in st.session_state:
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


def set_stock_position(stock_position: dict | None) -> None:
    """
    Set the stock position.

    Args:
        stock_position: Stock position dict or None with keys:
            - position_type: 'long' or 'short'
            - quantity: int
            - entry_price: float
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
    return (
        bool(st.session_state.positions) or st.session_state.stock_position is not None
    )


def get_position_count() -> tuple[int, bool]:
    """
    Get the count of positions.

    Returns:
        Tuple of (option_count, has_stock)
    """
    option_count = len(st.session_state.get("positions", []))
    has_stock = st.session_state.get("stock_position") is not None
    return option_count, has_stock


def set_positions_from_strategy(
    positions: list[dict], stock_position: dict | None
) -> None:
    """
    Set positions from a predefined strategy.

    Args:
        positions: List of option position dicts
        stock_position: Stock position dict or None
    """
    st.session_state.positions = positions
    st.session_state.stock_position = stock_position


# =============================================================================
# Position Factory Functions
# =============================================================================


def create_option_position(
    option_type: str,
    position_type: str,
    strike: float,
    quantity: int,
    premium_paid: float,
    dte_days: int | None = None,
    volatility: float | None = None,
    instrument_class: str = "vanilla",
    barrier: float | None = None,
    is_up: bool | None = None,
    is_knock_in: bool | None = None,
    rebate: float = 0.0,
    payout: float = 1.0,
    extra1: float = 0.0,
    choice_time_pct: float = 0.5,
    power_n: float = 2.0,
    gap_trigger: float | None = None,
) -> dict:
    """
    Create an option position dict.

    Args:
        option_type: 'call' or 'put'
        position_type: 'long' or 'short'
        strike: Strike price
        quantity: Number of contracts
        premium_paid: Premium per share
        dte_days: Days to expiration (optional)
        volatility: Implied volatility (optional)
        instrument_class: 'vanilla', 'barrier', 'asian', 'digital',
                          'lookback_floating', 'lookback_fixed',
                          'chooser', 'asset_or_nothing', 'power', 'gap'
        barrier: Barrier level (for barrier options)
        is_up: True for up-barrier, False for down-barrier
        is_knock_in: True for knock-in, False for knock-out
        rebate: Rebate amount (for barrier options)
        payout: Fixed payout (for digital options)
        extra1: Type-specific param (t_c for chooser, n for power, K2 for gap)
        choice_time_pct: Choice time as fraction of maturity (chooser)
        power_n: Power exponent (power options)
        gap_trigger: Trigger strike K2 (gap options)

    Returns:
        Option position dict
    """
    position = {
        "option_type": option_type,
        "position_type": position_type,
        "strike": strike,
        "quantity": quantity,
        "premium_paid": premium_paid,
    }
    if dte_days is not None:
        position["dte_days"] = dte_days
    if volatility is not None:
        position["volatility"] = volatility
    if instrument_class != "vanilla":
        position["instrument_class"] = instrument_class
        if barrier is not None:
            position["barrier"] = barrier
        if is_up is not None:
            position["is_up"] = is_up
        if is_knock_in is not None:
            position["is_knock_in"] = is_knock_in
        if rebate != 0.0:
            position["rebate"] = rebate
        if payout != 1.0:
            position["payout"] = payout
        if extra1 != 0.0:
            position["extra1"] = extra1
        if instrument_class == "chooser":
            position["choice_time_pct"] = choice_time_pct
        if instrument_class == "power":
            position["power_n"] = power_n
        if instrument_class == "gap" and gap_trigger is not None:
            position["gap_trigger"] = gap_trigger
    return position


def create_stock_position(
    position_type: str, quantity: int, entry_price: float
) -> dict:
    """
    Create a stock position dict.

    Args:
        position_type: 'long' or 'short'
        quantity: Number of shares
        entry_price: Entry price per share

    Returns:
        Stock position dict
    """
    return {
        "position_type": position_type,
        "quantity": quantity,
        "entry_price": entry_price,
    }
