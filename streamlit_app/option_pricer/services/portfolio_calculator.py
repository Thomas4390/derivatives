"""
Portfolio calculation services for Options Greeks Explorer.

This module provides functions for calculating portfolio metrics, P&L, and Greeks.
"""

import streamlit as st
import numpy as np
import json
from functools import lru_cache
from config.constants import (
    CONTRACT_MULTIPLIER,
    GREEK_NAMES,
    DTE_RANGE,
    IV_RANGE,
    STRIKE_RANGE_FACTORS,
    SPOT_RANGE_FACTOR,
    SPOT_RANGE_POINTS
)
from .risk_analyzer import check_unlimited_risk


@lru_cache(maxsize=128)
def get_portfolio_hash(portfolio_json: str) -> int:
    """
    Create a hash for portfolio caching.

    Args:
        portfolio_json: JSON string of portfolio data

    Returns:
        Hash value for caching
    """
    return hash(portfolio_json)


def prepare_portfolio_data(
    positions: list,
    stock_position,
    spot_price: float
) -> dict:
    """
    Prepare portfolio data dictionary from positions.

    Args:
        positions: List of option position dicts
        stock_position: Stock position dict or None
        spot_price: Current spot price

    Returns:
        Dictionary with portfolio data
    """
    portfolio_data = {
        'spot_price': spot_price,
        'options': [],
        'stock': None
    }

    # Add option positions (positions are already dicts)
    if positions:
        portfolio_data['options'] = [
            {
                'option_type': str(pos['option_type']),
                'position_type': str(pos['position_type']),
                'strike': pos['strike'],
                'quantity': pos['quantity'],
                'premium_paid': pos['premium_paid']
            }
            for pos in positions
        ]

    # Add stock position (stock_position is already a dict)
    if stock_position:
        portfolio_data['stock'] = {
            'position_type': str(stock_position['position_type']),
            'quantity': stock_position['quantity'],
            'entry_price': stock_position['entry_price']
        }

    return portfolio_data


def prepare_portfolio_arrays(portfolio_data: dict) -> tuple:
    """
    Prepare numpy arrays from portfolio data for calculations.

    Args:
        portfolio_data: Dictionary with portfolio data

    Returns:
        Tuple of (strikes, option_types, position_types, quantities, premiums,
                 stock_quantity, stock_entry_price)
    """
    if portfolio_data.get('options') and len(portfolio_data['options']) > 0:
        strikes = np.array([pos['strike'] for pos in portfolio_data['options']])
        option_types = np.array([
            1 if pos['option_type'] == 'call' else 0
            for pos in portfolio_data['options']
        ])
        position_types = np.array([
            1 if pos['position_type'] == 'long' else -1
            for pos in portfolio_data['options']
        ])
        quantities = np.array([
            pos['quantity'] * CONTRACT_MULTIPLIER
            for pos in portfolio_data['options']
        ])
        premiums = np.array([pos['premium_paid'] for pos in portfolio_data['options']])
    else:
        strikes = np.array([])
        option_types = np.array([], dtype=np.int32)
        position_types = np.array([], dtype=np.int32)
        quantities = np.array([], dtype=np.int32)
        premiums = np.array([])

    stock_quantity = 0
    stock_entry_price = 0
    if portfolio_data.get('stock'):
        stock = portfolio_data['stock']
        stock_quantity = stock['quantity'] * (1 if stock['position_type'] == 'long' else -1)
        stock_entry_price = stock['entry_price']

    return (strikes, option_types, position_types, quantities, premiums,
            stock_quantity, stock_entry_price)


@st.cache_data(ttl=3600)
def calculate_all_surfaces(
    portfolio_json: str,
    spot_range: tuple,
    dte_values: tuple,
    iv_values: tuple,
    risk_free_rate: float,
    _calculate_all_greeks_func,
    _calculate_pnl_at_expiry_func,
    _find_breakeven_func,
    has_positions: bool
) -> dict:
    """
    Calculate all P&L and Greeks data at once.

    Args:
        portfolio_json: JSON string of portfolio data
        spot_range: Tuple of spot prices (converted from numpy array for caching)
        dte_values: Tuple of DTE values
        iv_values: Tuple of IV values
        risk_free_rate: Risk-free interest rate
        _calculate_all_greeks_func: Function to calculate Greeks (not hashed)
        _calculate_pnl_at_expiry_func: Function to calculate P&L at expiry (not hashed)
        _find_breakeven_func: Function to find breakeven points (not hashed)
        has_positions: Whether there are active positions

    Returns:
        Dictionary with all calculated data
    """
    portfolio_data = json.loads(portfolio_json)
    spot_range_arr = np.array(spot_range)

    # Prepare arrays
    (strikes, option_types, position_types, quantities, premiums,
     stock_quantity, stock_entry_price) = prepare_portfolio_arrays(portfolio_data)

    # Calculate surfaces
    pnl_data, greeks_data = _calculate_surfaces(
        portfolio_data, spot_range_arr, dte_values, iv_values, risk_free_rate,
        strikes, option_types, position_types, quantities, premiums,
        stock_quantity, stock_entry_price, _calculate_all_greeks_func
    )

    # Calculate P&L at expiration
    expiry_pnl = _calculate_expiry_pnl(
        spot_range_arr, strikes, option_types, position_types, quantities,
        premiums, stock_quantity, stock_entry_price, _calculate_pnl_at_expiry_func
    )
    pnl_data['expiry'] = expiry_pnl

    # Find breakeven and risk analysis
    breakeven_result = _find_breakeven(
        portfolio_data, strikes, option_types, position_types, quantities,
        premiums, stock_quantity, stock_entry_price, _find_breakeven_func,
        _calculate_pnl_at_expiry_func
    )

    # Determine unlimited risk
    if not has_positions:
        # Default position is a long call, which has unlimited profit potential
        unlimited_profit, unlimited_loss = True, False
    else:
        unlimited_profit, unlimited_loss = check_unlimited_risk(portfolio_data)

    # Build result
    result = _build_result(
        pnl_data, greeks_data, breakeven_result,
        unlimited_profit, unlimited_loss, expiry_pnl
    )

    return result


def _calculate_surfaces(
    portfolio_data: dict,
    spot_range: np.ndarray,
    dte_values: tuple,
    iv_values: tuple,
    risk_free_rate: float,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    stock_quantity: int,
    stock_entry_price: float,
    calculate_all_greeks_func
) -> tuple[dict, dict]:
    """Calculate P&L and Greeks surfaces."""
    pnl_data = {}
    greeks_data = {}

    for dte in dte_values:
        time_to_expiry = dte / 365.0

        for iv in iv_values:
            key = f"{dte}_{iv}"
            iv_decimal = iv / 100.0

            pnl_values, greeks_by_name = _calculate_for_params(
                spot_range, time_to_expiry, iv_decimal, risk_free_rate,
                strikes, option_types, position_types, quantities, premiums,
                stock_quantity, stock_entry_price, calculate_all_greeks_func
            )

            pnl_data[key] = pnl_values
            greeks_data[key] = greeks_by_name

    return pnl_data, greeks_data


def _calculate_for_params(
    spot_range: np.ndarray,
    time_to_expiry: float,
    iv_decimal: float,
    risk_free_rate: float,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    stock_quantity: int,
    stock_entry_price: float,
    calculate_all_greeks_func
) -> tuple[np.ndarray, dict]:
    """Calculate P&L and Greeks for a specific DTE/IV combination."""
    pnl_values = np.zeros(len(spot_range))
    greeks_by_name = {name: np.zeros(len(spot_range)) for name in GREEK_NAMES}

    for i, spot in enumerate(spot_range):
        total_pnl = 0
        total_greeks = np.zeros(14)

        # Calculate for each option position
        for j in range(len(strikes)):
            greeks = calculate_all_greeks_func(
                spot, strikes[j], time_to_expiry,
                risk_free_rate, iv_decimal, option_types[j]
            )

            # P&L calculation
            option_value = greeks[0]
            if position_types[j] == 1:  # Long
                pnl = (option_value - premiums[j]) * quantities[j]
            else:  # Short
                pnl = (premiums[j] - option_value) * quantities[j]
            total_pnl += pnl

            # Greeks aggregation
            total_greeks += greeks * quantities[j] * position_types[j]

        # Add stock contribution
        if stock_quantity != 0:
            stock_pnl = (spot - stock_entry_price) * stock_quantity
            total_pnl += stock_pnl
            total_greeks[1] += stock_quantity  # Add to delta

        pnl_values[i] = total_pnl

        for k, name in enumerate(GREEK_NAMES):
            greeks_by_name[name][i] = total_greeks[k]

    return pnl_values, greeks_by_name


def _calculate_expiry_pnl(
    spot_range: np.ndarray,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    stock_quantity: int,
    stock_entry_price: float,
    calculate_pnl_at_expiry_func
) -> np.ndarray:
    """Calculate P&L at expiration for all spot prices."""
    expiry_pnl = np.zeros(len(spot_range))
    for i, spot in enumerate(spot_range):
        expiry_pnl[i] = calculate_pnl_at_expiry_func(
            spot, strikes, option_types, position_types,
            quantities, premiums, stock_quantity, stock_entry_price
        )
    return expiry_pnl


def _find_breakeven(
    portfolio_data: dict,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    stock_quantity: int,
    stock_entry_price: float,
    find_breakeven_func,
    calculate_pnl_at_expiry_func
):
    """Find breakeven points and max profit/loss."""
    if len(strikes) == 0 and stock_quantity == 0:
        return None

    # Use wide range for theoretical extremes
    theoretical_min = 0.01
    theoretical_max = portfolio_data.get('spot_price', 100.0) * 10.0

    breakeven_result = find_breakeven_func(
        strikes, option_types, position_types, quantities,
        premiums, stock_quantity, stock_entry_price,
        theoretical_min, theoretical_max, 20000
    )

    # Special case: For short puts, calculate P&L at exactly 0
    if portfolio_data.get('options'):
        has_short_puts = any(
            pos['option_type'] == 'put' and pos['position_type'] == 'short'
            for pos in portfolio_data['options']
        )
        if has_short_puts and breakeven_result:
            pnl_at_zero = calculate_pnl_at_expiry_func(
                0.0, strikes, option_types, position_types,
                quantities, premiums, stock_quantity, stock_entry_price
            )
            if pnl_at_zero < breakeven_result.max_loss:
                breakeven_result.max_loss = pnl_at_zero
                breakeven_result.max_loss_spot = 0.0

    return breakeven_result


def _build_result(
    pnl_data: dict,
    greeks_data: dict,
    breakeven_result,
    unlimited_profit: bool,
    unlimited_loss: bool,
    expiry_pnl: np.ndarray
) -> dict:
    """Build the final result dictionary."""
    result = {
        'pnl_data': pnl_data,
        'greeks_data': greeks_data,
        'breakeven_result': breakeven_result,
        'unlimited_profit': unlimited_profit,
        'unlimited_loss': unlimited_loss
    }

    if not breakeven_result:
        result['max_profit_display'] = 0
        result['max_loss_display'] = 0
        return result

    # Determine display values for max profit/loss
    result['max_profit_display'] = _get_profit_display(
        unlimited_profit, breakeven_result, expiry_pnl
    )
    result['max_loss_display'] = _get_loss_display(
        unlimited_loss, breakeven_result, expiry_pnl
    )

    return result


def _get_profit_display(
    unlimited_profit: bool,
    breakeven_result,
    expiry_pnl: np.ndarray
) -> float:
    """Get the display value for max profit."""
    if not unlimited_profit:
        return breakeven_result.max_profit

    if len(expiry_pnl) > 10:
        high_end_trend = expiry_pnl[-1] - expiry_pnl[-10]
        if high_end_trend > 0:
            return float('inf')

    return breakeven_result.max_profit


def _get_loss_display(
    unlimited_loss: bool,
    breakeven_result,
    expiry_pnl: np.ndarray
) -> float:
    """Get the display value for max loss."""
    if not unlimited_loss:
        return breakeven_result.max_loss

    if len(expiry_pnl) > 10:
        high_end_trend = expiry_pnl[-1] - expiry_pnl[-10]
        if high_end_trend < 0 and expiry_pnl[-1] < 0:
            return float('-inf')

    return breakeven_result.max_loss


def get_spot_range(spot_price: float) -> np.ndarray:
    """
    Get the spot price range for calculations.

    Args:
        spot_price: Current spot price

    Returns:
        Numpy array of spot prices
    """
    return np.linspace(
        spot_price * (1 - SPOT_RANGE_FACTOR),
        spot_price * (1 + SPOT_RANGE_FACTOR),
        SPOT_RANGE_POINTS
    )


@st.cache_data(ttl=3600)
def calculate_strike_surfaces(
    spot_price: float,
    spot_range: tuple,
    option_type: str,
    position_type: str,
    quantity: int,
    base_strike: float,
    risk_free_rate: float,
    _calculate_all_greeks_func,
    _calculate_pnl_at_expiry_func,
    _find_breakeven_func=None
) -> tuple[dict, dict, dict]:
    """
    Calculate P&L and Greeks data for varying strike prices (single-leg only).

    Args:
        spot_price: Current spot price
        spot_range: Tuple of spot prices
        option_type: 'call' or 'put'
        position_type: 'long' or 'short'
        quantity: Number of contracts
        base_strike: Base strike price for reference
        risk_free_rate: Risk-free interest rate
        _calculate_all_greeks_func: Function to calculate Greeks
        _calculate_pnl_at_expiry_func: Function to calculate P&L at expiry
        _find_breakeven_func: Function to find breakeven points

    Returns:
        Tuple of (pnl_data, greeks_data, breakeven_data) dictionaries keyed by strike
    """
    spot_range_arr = np.array(spot_range)
    pnl_data = {}
    greeks_data = {}
    breakeven_data = {}
    expiry_data = {}

    option_type_int = 1 if option_type == 'call' else 0
    position_sign = 1 if position_type == 'long' else -1

    # Fixed parameters for strike variation
    fixed_dte = 31
    fixed_iv = 0.25
    time_to_expiry = fixed_dte / 365.0

    for strike_factor in STRIKE_RANGE_FACTORS:
        strike = round(spot_price * strike_factor, 2)
        key = f"strike_{int(strike_factor * 100)}"

        pnl_values = np.zeros(len(spot_range_arr))
        greeks_by_name = {name: np.zeros(len(spot_range_arr)) for name in GREEK_NAMES}

        # Calculate initial premium at this strike
        initial_greeks = _calculate_all_greeks_func(
            spot_price, strike, time_to_expiry,
            risk_free_rate, fixed_iv, option_type_int
        )
        premium = initial_greeks[0]

        for i, spot in enumerate(spot_range_arr):
            greeks = _calculate_all_greeks_func(
                spot, strike, time_to_expiry,
                risk_free_rate, fixed_iv, option_type_int
            )

            option_value = greeks[0]
            if position_sign == 1:  # Long
                pnl = (option_value - premium) * quantity * CONTRACT_MULTIPLIER
            else:  # Short
                pnl = (premium - option_value) * quantity * CONTRACT_MULTIPLIER

            pnl_values[i] = pnl

            for k, name in enumerate(GREEK_NAMES):
                greeks_by_name[name][i] = greeks[k] * quantity * CONTRACT_MULTIPLIER * position_sign

        pnl_data[key] = pnl_values
        greeks_data[key] = greeks_by_name

        # Calculate expiry P&L for this strike
        expiry_pnl = np.zeros(len(spot_range_arr))
        strikes_arr = np.array([strike])
        option_types_arr = np.array([option_type_int])
        position_types_arr = np.array([position_sign])
        quantities_arr = np.array([quantity * CONTRACT_MULTIPLIER])
        premiums_arr = np.array([premium])

        for i, spot in enumerate(spot_range_arr):
            expiry_pnl[i] = _calculate_pnl_at_expiry_func(
                spot, strikes_arr, option_types_arr, position_types_arr,
                quantities_arr, premiums_arr, 0, 0
            )

        expiry_data[key] = expiry_pnl

        # Calculate breakeven for this strike
        if _find_breakeven_func:
            theoretical_min = 0.01
            theoretical_max = spot_price * 3.0
            breakeven_result = _find_breakeven_func(
                strikes_arr, option_types_arr, position_types_arr,
                quantities_arr, premiums_arr, 0, 0,
                theoretical_min, theoretical_max, 10000
            )
            breakeven_data[key] = breakeven_result

    # Store expiry data in pnl_data
    pnl_data['expiry_by_strike'] = expiry_data

    return pnl_data, greeks_data, breakeven_data


def get_strike_range(spot_price: float) -> list[float]:
    """
    Get the list of strike prices for strike variation.

    Args:
        spot_price: Current spot price

    Returns:
        List of strike prices
    """
    return [round(spot_price * factor, 2) for factor in STRIKE_RANGE_FACTORS]


@st.cache_data(ttl=3600)
def calculate_individual_leg_greeks(
    portfolio_json: str,
    spot_range: tuple,
    dte: int,
    iv: int,
    risk_free_rate: float,
    _calculate_all_greeks_func
) -> dict:
    """
    Calculate Greeks for each individual leg of a multi-leg portfolio.

    Args:
        portfolio_json: JSON string of portfolio data
        spot_range: Tuple of spot prices
        dte: Days to expiration
        iv: Implied volatility (percentage)
        risk_free_rate: Risk-free interest rate
        _calculate_all_greeks_func: Function to calculate Greeks

    Returns:
        Dictionary with Greeks data for each leg, keyed by leg index
        {
            'leg_0': {'delta': [...], 'gamma': [...], ...},
            'leg_1': {'delta': [...], 'gamma': [...], ...},
            'stock': {'delta': [...], ...} if stock position exists
        }
    """
    portfolio_data = json.loads(portfolio_json)
    spot_range_arr = np.array(spot_range)
    time_to_expiry = dte / 365.0
    iv_decimal = iv / 100.0

    leg_greeks = {}

    # Calculate Greeks for each option leg
    if portfolio_data.get('options'):
        for leg_idx, pos in enumerate(portfolio_data['options']):
            strike = pos['strike']
            option_type = 1 if pos['option_type'] == 'call' else 0
            position_sign = 1 if pos['position_type'] == 'long' else -1
            quantity = pos['quantity'] * CONTRACT_MULTIPLIER

            greeks_by_name = {name: np.zeros(len(spot_range_arr)) for name in GREEK_NAMES}

            for i, spot in enumerate(spot_range_arr):
                greeks = _calculate_all_greeks_func(
                    spot, strike, time_to_expiry,
                    risk_free_rate, iv_decimal, option_type
                )

                for k, name in enumerate(GREEK_NAMES):
                    greeks_by_name[name][i] = greeks[k] * quantity * position_sign

            # Add metadata for display
            leg_greeks[f'leg_{leg_idx}'] = {
                'greeks': greeks_by_name,
                'option_type': pos['option_type'],
                'position_type': pos['position_type'],
                'strike': strike,
                'quantity': pos['quantity']
            }

    # Calculate Greeks for stock position
    if portfolio_data.get('stock'):
        stock = portfolio_data['stock']
        stock_quantity = stock['quantity'] * (1 if stock['position_type'] == 'long' else -1)

        stock_greeks = {name: np.zeros(len(spot_range_arr)) for name in GREEK_NAMES}
        # Stock only has delta = quantity
        stock_greeks['delta'] = np.full(len(spot_range_arr), stock_quantity)

        leg_greeks['stock'] = {
            'greeks': stock_greeks,
            'position_type': stock['position_type'],
            'quantity': stock['quantity']
        }

    return leg_greeks


@st.cache_data(ttl=3600)
def calculate_all_individual_leg_greeks(
    portfolio_json: str,
    spot_range: tuple,
    dte_values: tuple,
    iv_values: tuple,
    risk_free_rate: float,
    _calculate_all_greeks_func
) -> dict:
    """
    Calculate Greeks for each individual leg for all DTE/IV combinations.

    Args:
        portfolio_json: JSON string of portfolio data
        spot_range: Tuple of spot prices
        dte_values: Tuple of DTE values
        iv_values: Tuple of IV values
        risk_free_rate: Risk-free interest rate
        _calculate_all_greeks_func: Function to calculate Greeks

    Returns:
        Dictionary keyed by "DTE_IV" with leg Greeks for each combination
        {
            '31_25': {
                'leg_0': {'greeks': {...}, 'option_type': 'call', ...},
                'leg_1': {...},
            },
            '31_30': {...},
            ...
        }
    """
    portfolio_data = json.loads(portfolio_json)
    spot_range_arr = np.array(spot_range)

    # Extract leg metadata once
    leg_metadata = {}
    if portfolio_data.get('options'):
        for leg_idx, pos in enumerate(portfolio_data['options']):
            leg_metadata[f'leg_{leg_idx}'] = {
                'option_type': pos['option_type'],
                'position_type': pos['position_type'],
                'strike': pos['strike'],
                'quantity': pos['quantity'],
                'option_type_int': 1 if pos['option_type'] == 'call' else 0,
                'position_sign': 1 if pos['position_type'] == 'long' else -1,
                'quantity_mult': pos['quantity'] * CONTRACT_MULTIPLIER
            }

    if portfolio_data.get('stock'):
        stock = portfolio_data['stock']
        leg_metadata['stock'] = {
            'position_type': stock['position_type'],
            'quantity': stock['quantity'],
            'stock_quantity': stock['quantity'] * (1 if stock['position_type'] == 'long' else -1)
        }

    all_leg_greeks = {}

    for dte in dte_values:
        time_to_expiry = dte / 365.0

        for iv in iv_values:
            key = f"{dte}_{iv}"
            iv_decimal = iv / 100.0

            leg_greeks = {}

            # Calculate Greeks for each option leg
            for leg_key, meta in leg_metadata.items():
                if leg_key == 'stock':
                    # Stock only has delta
                    stock_greeks = {name: np.zeros(len(spot_range_arr)) for name in GREEK_NAMES}
                    stock_greeks['delta'] = np.full(len(spot_range_arr), meta['stock_quantity'])
                    leg_greeks['stock'] = {
                        'greeks': stock_greeks,
                        'position_type': meta['position_type'],
                        'quantity': meta['quantity']
                    }
                else:
                    # Option leg
                    greeks_by_name = {name: np.zeros(len(spot_range_arr)) for name in GREEK_NAMES}

                    for i, spot in enumerate(spot_range_arr):
                        greeks = _calculate_all_greeks_func(
                            spot, meta['strike'], time_to_expiry,
                            risk_free_rate, iv_decimal, meta['option_type_int']
                        )

                        for k, name in enumerate(GREEK_NAMES):
                            greeks_by_name[name][i] = greeks[k] * meta['quantity_mult'] * meta['position_sign']

                    leg_greeks[leg_key] = {
                        'greeks': greeks_by_name,
                        'option_type': meta['option_type'],
                        'position_type': meta['position_type'],
                        'strike': meta['strike'],
                        'quantity': meta['quantity']
                    }

            all_leg_greeks[key] = leg_greeks

    return all_leg_greeks
