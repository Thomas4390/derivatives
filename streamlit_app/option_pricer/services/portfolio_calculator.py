"""
Portfolio calculation services for Options Greeks Explorer.

This module provides functions for calculating portfolio metrics, P&L, and Greeks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import streamlit as st
import numpy as np
import json
from functools import lru_cache

if TYPE_CHECKING:
    from .pricing_adapter import BreakevenResult
from config.constants import (
    CONTRACT_MULTIPLIER,
    GREEK_NAMES,
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
        portfolio_data['options'] = []
        for pos in positions:
            opt = {
                'option_type': str(pos['option_type']),
                'position_type': str(pos['position_type']),
                'strike': pos['strike'],
                'quantity': pos['quantity'],
                'premium_paid': pos['premium_paid'],
                'instrument_class': pos.get('instrument_class', 'vanilla'),
            }
            # Pass through exotic fields when present
            if opt['instrument_class'] != 'vanilla':
                for key in ('barrier', 'is_up', 'is_knock_in', 'rebate', 'payout'):
                    if key in pos:
                        opt[key] = pos[key]
            portfolio_data['options'].append(opt)

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
                 stock_quantity, stock_entry_price, exotic_metadata)
    """
    exotic_metadata = []
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
        for pos in portfolio_data['options']:
            exotic_metadata.append({
                'instrument_class': pos.get('instrument_class', 'vanilla'),
                'barrier': pos.get('barrier', 0.0),
                'is_up': pos.get('is_up', True),
                'is_knock_in': pos.get('is_knock_in', False),
                'rebate': pos.get('rebate', 0.0),
                'payout': pos.get('payout', 1.0),
            })
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
            stock_quantity, stock_entry_price, exotic_metadata)


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
    has_positions: bool,
    _calculate_exotic_greeks_func=None,
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
        _calculate_exotic_greeks_func: Function to calculate exotic Greeks (not hashed)

    Returns:
        Dictionary with all calculated data
    """
    portfolio_data = json.loads(portfolio_json)
    spot_range_arr = np.array(spot_range)

    # Prepare arrays
    (strikes, option_types, position_types, quantities, premiums,
     stock_quantity, stock_entry_price, exotic_metadata) = prepare_portfolio_arrays(portfolio_data)

    has_exotic_legs = any(
        m['instrument_class'] != 'vanilla' for m in exotic_metadata
    )

    # Calculate surfaces
    pnl_data, greeks_data = _calculate_surfaces(
        portfolio_data, spot_range_arr, dte_values, iv_values, risk_free_rate,
        strikes, option_types, position_types, quantities, premiums,
        stock_quantity, stock_entry_price, _calculate_all_greeks_func,
        exotic_metadata=exotic_metadata,
        calculate_exotic_greeks_func=_calculate_exotic_greeks_func,
    )

    # Calculate P&L at expiration (split vanilla / exotic)
    expiry_pnl = _calculate_expiry_pnl(
        spot_range_arr, strikes, option_types, position_types, quantities,
        premiums, stock_quantity, stock_entry_price, _calculate_pnl_at_expiry_func,
        exotic_metadata=exotic_metadata,
        portfolio_data=portfolio_data,
    )
    pnl_data['expiry'] = expiry_pnl

    # Find breakeven and risk analysis
    breakeven_result = _find_breakeven(
        portfolio_data, strikes, option_types, position_types, quantities,
        premiums, stock_quantity, stock_entry_price, _find_breakeven_func,
        _calculate_pnl_at_expiry_func,
        exotic_metadata=exotic_metadata,
        expiry_pnl=expiry_pnl,
        spot_range_arr=spot_range_arr,
    )

    # Determine unlimited risk
    if not has_positions:
        unlimited_profit, unlimited_loss = False, False
    else:
        unlimited_profit, unlimited_loss = check_unlimited_risk(portfolio_data)

    # Override for exotic portfolios — per-type logic
    if has_exotic_legs:
        for j, m in enumerate(exotic_metadata):
            if m['instrument_class'] == 'vanilla':
                continue
            is_long = position_types[j] == 1
            typ = m['instrument_class']
            # Lookback and Asian can be unbounded
            if typ in ('lookback_fixed', 'lookback_floating', 'asian'):
                if is_long:
                    unlimited_profit = True
                else:
                    unlimited_loss = True
            elif typ == 'barrier':
                is_call = portfolio_data['options'][j]['option_type'] == 'call'
                if is_long:
                    if is_call:
                        unlimited_profit = True
                else:
                    if is_call:
                        unlimited_loss = True
            # Digital: bounded payout, no change needed

    # Build result
    result = _build_result(
        pnl_data, greeks_data, breakeven_result,
        unlimited_profit, unlimited_loss, expiry_pnl,
        has_exotic_legs=has_exotic_legs,
    )
    result['has_exotic_legs'] = has_exotic_legs

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
    calculate_all_greeks_func,
    exotic_metadata: list = None,
    calculate_exotic_greeks_func=None,
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
                stock_quantity, stock_entry_price, calculate_all_greeks_func,
                exotic_metadata=exotic_metadata,
                calculate_exotic_greeks_func=calculate_exotic_greeks_func,
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
    calculate_all_greeks_func,
    exotic_metadata: list = None,
    calculate_exotic_greeks_func=None,
) -> tuple[np.ndarray, dict]:
    """Calculate P&L and Greeks for a specific DTE/IV combination."""
    pnl_values = np.zeros(len(spot_range))
    greeks_by_name = {name: np.zeros(len(spot_range)) for name in GREEK_NAMES}

    for i, spot in enumerate(spot_range):
        total_pnl = 0
        total_greeks = np.zeros(14)

        # Calculate for each option position
        for j in range(len(strikes)):
            meta = exotic_metadata[j] if exotic_metadata else None
            is_exotic = meta and meta['instrument_class'] != 'vanilla'

            if is_exotic and calculate_exotic_greeks_func:
                greeks = calculate_exotic_greeks_func(
                    spot, strikes[j], time_to_expiry,
                    risk_free_rate, iv_decimal, option_types[j],
                    exotic_type=meta['instrument_class'],
                    barrier=meta.get('barrier', 0.0),
                    is_up=meta.get('is_up', True),
                    is_knock_in=meta.get('is_knock_in', False),
                    rebate=meta.get('rebate', 0.0),
                    payout=meta.get('payout', 1.0),
                )
            else:
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

            # Greeks aggregation (guard against NaN from exotic discontinuities)
            greeks = np.where(np.isnan(greeks), 0.0, greeks)
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
    calculate_pnl_at_expiry_func,
    exotic_metadata: list = None,
    portfolio_data: dict = None,
) -> np.ndarray:
    """Calculate P&L at expiration for all spot prices.

    Vanilla legs use the fast Numba path. Exotic legs use a Python loop
    calling calculate_exotic_payoff_at_expiry.
    """
    # Separate vanilla and exotic indices
    vanilla_indices = []
    exotic_indices = []
    if exotic_metadata:
        for j, meta in enumerate(exotic_metadata):
            if meta['instrument_class'] != 'vanilla':
                exotic_indices.append(j)
            else:
                vanilla_indices.append(j)
    else:
        vanilla_indices = list(range(len(strikes)))

    # Vanilla part: use fast Numba function
    if vanilla_indices:
        v_strikes = strikes[vanilla_indices]
        v_option_types = option_types[vanilla_indices]
        v_position_types = position_types[vanilla_indices]
        v_quantities = quantities[vanilla_indices]
        v_premiums = premiums[vanilla_indices]
    else:
        v_strikes = np.array([])
        v_option_types = np.array([], dtype=np.int32)
        v_position_types = np.array([], dtype=np.int32)
        v_quantities = np.array([], dtype=np.int32)
        v_premiums = np.array([])

    expiry_pnl = np.zeros(len(spot_range))
    for i, spot in enumerate(spot_range):
        # Vanilla legs (Numba)
        if len(v_strikes) > 0 or stock_quantity != 0:
            expiry_pnl[i] = calculate_pnl_at_expiry_func(
                spot, v_strikes, v_option_types, v_position_types,
                v_quantities, v_premiums, stock_quantity, stock_entry_price
            )

        # Exotic legs (Python)
        if exotic_indices and portfolio_data:
            from .exotic_pricing_adapter import calculate_exotic_payoff_at_expiry
            options = portfolio_data.get('options', [])
            for j in exotic_indices:
                pos = options[j]
                payoff = calculate_exotic_payoff_at_expiry(spot, pos)
                premium = premiums[j]
                if position_types[j] == 1:  # Long
                    expiry_pnl[i] += (payoff - premium) * quantities[j]
                else:  # Short
                    expiry_pnl[i] += (premium - payoff) * quantities[j]

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
    calculate_pnl_at_expiry_func,
    exotic_metadata: list = None,
    expiry_pnl: np.ndarray = None,
    spot_range_arr: np.ndarray = None,
):
    """Find breakeven points and max profit/loss."""
    if len(strikes) == 0 and stock_quantity == 0:
        return None

    # When exotic legs are present, compute breakeven from the expiry_pnl array
    # (the Numba find_breakeven_func treats all legs as vanilla)
    has_exotic = exotic_metadata and any(
        m['instrument_class'] != 'vanilla' for m in exotic_metadata
    )
    if has_exotic and expiry_pnl is not None and spot_range_arr is not None:
        return _breakeven_from_pnl_curve(expiry_pnl, spot_range_arr)

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


def _breakeven_from_pnl_curve(
    expiry_pnl: np.ndarray,
    spot_range: np.ndarray,
) -> 'BreakevenResult':
    """Compute breakeven points from a PnL curve via sign-change interpolation."""
    from .pricing_adapter import BreakevenResult

    # Find sign changes (breakeven crossings)
    breakeven_points = []
    for i in range(len(expiry_pnl) - 1):
        if expiry_pnl[i] * expiry_pnl[i + 1] < 0:
            # Linear interpolation for the zero crossing
            frac = abs(expiry_pnl[i]) / (abs(expiry_pnl[i]) + abs(expiry_pnl[i + 1]))
            bp = spot_range[i] + frac * (spot_range[i + 1] - spot_range[i])
            breakeven_points.append(float(bp))

    # Max profit / loss from the PnL curve
    max_profit_idx = int(np.argmax(expiry_pnl))
    max_loss_idx = int(np.argmin(expiry_pnl))

    return BreakevenResult(
        breakeven_points=breakeven_points,
        max_profit=float(expiry_pnl[max_profit_idx]),
        max_profit_spot=float(spot_range[max_profit_idx]),
        max_loss=float(expiry_pnl[max_loss_idx]),
        max_loss_spot=float(spot_range[max_loss_idx]),
    )


def _build_result(
    pnl_data: dict,
    greeks_data: dict,
    breakeven_result,
    unlimited_profit: bool,
    unlimited_loss: bool,
    expiry_pnl: np.ndarray,
    has_exotic_legs: bool = False,
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

    # For exotic legs, use actual PnL curve values instead of breakeven-based
    # estimates, since the breakeven function treats exotic legs as vanilla
    if has_exotic_legs:
        result['max_profit_display'] = float(np.max(expiry_pnl))
        result['max_loss_display'] = float(np.min(expiry_pnl))
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
    _calculate_all_greeks_func,
    _calculate_exotic_greeks_func=None,
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
        _calculate_exotic_greeks_func: Function to calculate exotic Greeks

    Returns:
        Dictionary with Greeks data for each leg, keyed by leg index
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
            inst_class = pos.get('instrument_class', 'vanilla')
            is_exotic = inst_class != 'vanilla'

            greeks_by_name = {name: np.zeros(len(spot_range_arr)) for name in GREEK_NAMES}

            for i, spot in enumerate(spot_range_arr):
                if is_exotic and _calculate_exotic_greeks_func:
                    greeks = _calculate_exotic_greeks_func(
                        spot, strike, time_to_expiry,
                        risk_free_rate, iv_decimal, option_type,
                        exotic_type=inst_class,
                        barrier=pos.get('barrier', 0.0),
                        is_up=pos.get('is_up', True),
                        is_knock_in=pos.get('is_knock_in', False),
                        rebate=pos.get('rebate', 0.0),
                        payout=pos.get('payout', 1.0),
                    )
                else:
                    greeks = _calculate_all_greeks_func(
                        spot, strike, time_to_expiry,
                        risk_free_rate, iv_decimal, option_type
                    )

                for k, name in enumerate(GREEK_NAMES):
                    greeks_by_name[name][i] = greeks[k] * quantity * position_sign

            # Add metadata for display
            leg_data = {
                'greeks': greeks_by_name,
                'option_type': pos['option_type'],
                'position_type': pos['position_type'],
                'strike': strike,
                'quantity': pos['quantity'],
            }
            if is_exotic:
                leg_data['instrument_class'] = inst_class
            leg_greeks[f'leg_{leg_idx}'] = leg_data

    # Calculate Greeks for stock position
    if portfolio_data.get('stock'):
        stock = portfolio_data['stock']
        stock_quantity = stock['quantity'] * (1 if stock['position_type'] == 'long' else -1)

        stock_greeks = {name: np.zeros(len(spot_range_arr)) for name in GREEK_NAMES}
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
    _calculate_all_greeks_func,
    _calculate_exotic_greeks_func=None,
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
        _calculate_exotic_greeks_func: Function to calculate exotic Greeks

    Returns:
        Dictionary keyed by "DTE_IV" with leg Greeks for each combination
    """
    portfolio_data = json.loads(portfolio_json)
    spot_range_arr = np.array(spot_range)

    # Extract leg metadata once
    leg_metadata = {}
    if portfolio_data.get('options'):
        for leg_idx, pos in enumerate(portfolio_data['options']):
            meta = {
                'option_type': pos['option_type'],
                'position_type': pos['position_type'],
                'strike': pos['strike'],
                'quantity': pos['quantity'],
                'option_type_int': 1 if pos['option_type'] == 'call' else 0,
                'position_sign': 1 if pos['position_type'] == 'long' else -1,
                'quantity_mult': pos['quantity'] * CONTRACT_MULTIPLIER,
                'instrument_class': pos.get('instrument_class', 'vanilla'),
            }
            if meta['instrument_class'] != 'vanilla':
                meta['barrier'] = pos.get('barrier', 0.0)
                meta['is_up'] = pos.get('is_up', True)
                meta['is_knock_in'] = pos.get('is_knock_in', False)
                meta['rebate'] = pos.get('rebate', 0.0)
                meta['payout'] = pos.get('payout', 1.0)
            leg_metadata[f'leg_{leg_idx}'] = meta

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
                    stock_greeks = {name: np.zeros(len(spot_range_arr)) for name in GREEK_NAMES}
                    stock_greeks['delta'] = np.full(len(spot_range_arr), meta['stock_quantity'])
                    leg_greeks['stock'] = {
                        'greeks': stock_greeks,
                        'position_type': meta['position_type'],
                        'quantity': meta['quantity']
                    }
                else:
                    is_exotic = meta['instrument_class'] != 'vanilla'
                    greeks_by_name = {name: np.zeros(len(spot_range_arr)) for name in GREEK_NAMES}

                    for i, spot in enumerate(spot_range_arr):
                        if is_exotic and _calculate_exotic_greeks_func:
                            greeks = _calculate_exotic_greeks_func(
                                spot, meta['strike'], time_to_expiry,
                                risk_free_rate, iv_decimal, meta['option_type_int'],
                                exotic_type=meta['instrument_class'],
                                barrier=meta.get('barrier', 0.0),
                                is_up=meta.get('is_up', True),
                                is_knock_in=meta.get('is_knock_in', False),
                                rebate=meta.get('rebate', 0.0),
                                payout=meta.get('payout', 1.0),
                            )
                        else:
                            greeks = _calculate_all_greeks_func(
                                spot, meta['strike'], time_to_expiry,
                                risk_free_rate, iv_decimal, meta['option_type_int']
                            )

                        for k, name in enumerate(GREEK_NAMES):
                            val = greeks[k] * meta['quantity_mult'] * meta['position_sign']
                            greeks_by_name[name][i] = val if not np.isnan(val) else 0.0

                    leg_data = {
                        'greeks': greeks_by_name,
                        'option_type': meta['option_type'],
                        'position_type': meta['position_type'],
                        'strike': meta['strike'],
                        'quantity': meta['quantity'],
                    }
                    if is_exotic:
                        leg_data['instrument_class'] = meta['instrument_class']
                    leg_greeks[leg_key] = leg_data

            all_leg_greeks[key] = leg_greeks

    return all_leg_greeks
