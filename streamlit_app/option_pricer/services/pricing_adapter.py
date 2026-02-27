"""
Pricing Adapter for Streamlit App

This module provides array-based pricing functions that use the backend architecture.
These functions are specific to the Streamlit app's needs for vectorized calculations.

Performance: Uses Numba-compiled vectorized functions for fast bulk calculations.

Functions available:
- calculate_all_greeks: Calculate all 14 Greeks for a single option
- calculate_pnl_at_expiry_arrays: Calculate portfolio P&L at expiration
- find_breakeven_points: Find breakeven points and max profit/loss
- calculate_portfolio_greeks_3d_dte: 3D Greek surface (spot x DTE)
- calculate_portfolio_greeks_3d_iv: 3D Greek surface (spot x IV)
- calculate_greeks_3d_strike: 3D Greek surface (spot x strike)
- calculate_option_premium: Calculate option premium using Black-Scholes

Author: Thomas Vaudescal
"""

import numpy as np
import json
from dataclasses import dataclass
from typing import Optional

# Import Numba-optimized functions for performance
from backend.engines.vectorized_bs import (
    calculate_all_greeks as _calculate_all_greeks_numba,
)
from backend.portfolio.greeks_surfaces import (
    portfolio_greeks_surface_dte as calculate_portfolio_greeks_3d_dte_vectorized,
    portfolio_greeks_surface_iv as calculate_portfolio_greeks_3d_iv_vectorized,
    single_option_greeks_surface_strike as calculate_greeks_3d_strike_vectorized,
    calculate_portfolio_pnl_at_expiry_arrays as _calculate_pnl_numba,
    calculate_pnl_curve,
)


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class BreakevenResult:
    """Result of breakeven analysis."""
    breakeven_points: list
    max_profit: float
    max_profit_spot: float
    max_loss: float
    max_loss_spot: float


# =============================================================================
# PREMIUM CALCULATION
# =============================================================================

def calculate_option_premium(
    spot: float,
    strike: float,
    dte_days: int,
    risk_free_rate: float,
    volatility: float,
    option_type: str  # 'call' or 'put'
) -> float:
    """
    Calculate option premium using Black-Scholes.

    Args:
        spot: Current spot price
        strike: Strike price
        dte_days: Days to expiration
        risk_free_rate: Risk-free rate (decimal)
        volatility: Implied volatility (decimal)
        option_type: 'call' or 'put'

    Returns:
        Option premium per share
    """
    time_to_expiry = dte_days / 365.0
    opt_type_int = 1 if option_type == 'call' else 0
    greeks = _calculate_all_greeks_numba(
        spot, strike, time_to_expiry, risk_free_rate, volatility, opt_type_int
    )
    return greeks[0]  # Price is index 0


# =============================================================================
# CORE PRICING FUNCTIONS (Using Numba)
# =============================================================================

def calculate_all_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: int  # 1 for call, 0 for put
) -> np.ndarray:
    """
    Calculate all 14 Greeks for a single option.

    Uses Numba-compiled function for performance.

    Returns array of 14 values:
    [price, delta, gamma, vega, theta, rho,
     vanna, volga, charm, veta, speed, zomma, color, ultima]

    Args:
        spot: Current spot price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        risk_free_rate: Risk-free rate (decimal)
        volatility: Implied volatility (decimal)
        option_type: 1 for call, 0 for put

    Returns:
        np.ndarray: Array of 14 Greek values
    """
    return _calculate_all_greeks_numba(
        spot, strike, time_to_expiry, risk_free_rate, volatility, option_type
    )


def calculate_pnl_at_expiry_arrays(
    spot: float,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    stock_quantity: int,
    stock_entry_price: float
) -> float:
    """
    Calculate portfolio P&L at expiration for a single spot price.

    Uses Numba-compiled function for performance.

    Args:
        spot: Spot price at expiration
        strikes: Array of strike prices
        option_types: Array of option types (1 for call, 0 for put)
        position_types: Array of position types (1 for long, -1 for short)
        quantities: Array of quantities (already multiplied by contract multiplier)
        premiums: Array of premiums paid
        stock_quantity: Stock quantity (positive for long, negative for short)
        stock_entry_price: Stock entry price

    Returns:
        Total P&L at expiration
    """
    return _calculate_pnl_numba(
        spot, strikes, option_types, position_types,
        quantities, premiums, float(stock_quantity), stock_entry_price
    )


def find_breakeven_points(
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    stock_quantity: int,
    stock_entry_price: float,
    min_spot: float,
    max_spot: float,
    n_points: int
) -> Optional[BreakevenResult]:
    """
    Find breakeven points and max profit/loss for a portfolio.

    Uses vectorized Numba function for P&L curve calculation.

    Args:
        strikes: Array of strike prices
        option_types: Array of option types (1 for call, 0 for put)
        position_types: Array of position types (1 for long, -1 for short)
        quantities: Array of quantities
        premiums: Array of premiums
        stock_quantity: Stock quantity
        stock_entry_price: Stock entry price
        min_spot: Minimum spot for search
        max_spot: Maximum spot for search
        n_points: Number of points for search

    Returns:
        BreakevenResult or None if no positions
    """
    if len(strikes) == 0 and stock_quantity == 0:
        return None

    # Generate spot range
    spots = np.linspace(min_spot, max_spot, n_points)

    # Calculate P&L curve (vectorized)
    pnls = calculate_pnl_curve(
        spots, strikes, option_types, position_types,
        quantities, premiums, float(stock_quantity), stock_entry_price
    )

    # Find breakeven points (sign changes)
    breakeven_points = []
    for i in range(len(pnls) - 1):
        if pnls[i] * pnls[i + 1] < 0:  # Sign change
            # Linear interpolation for more precise breakeven
            ratio = abs(pnls[i]) / (abs(pnls[i]) + abs(pnls[i + 1]))
            breakeven = spots[i] + ratio * (spots[i + 1] - spots[i])
            breakeven_points.append(round(breakeven, 2))

    # Find max profit and max loss
    max_profit_idx = np.argmax(pnls)
    min_profit_idx = np.argmin(pnls)

    return BreakevenResult(
        breakeven_points=breakeven_points,
        max_profit=pnls[max_profit_idx],
        max_profit_spot=spots[max_profit_idx],
        max_loss=pnls[min_profit_idx],
        max_loss_spot=spots[min_profit_idx]
    )


# =============================================================================
# 3D SURFACE CALCULATIONS (Using Numba Vectorized Functions)
# =============================================================================

def _split_vanilla_exotic(options: list) -> tuple[list, list, list, list]:
    """Split options into vanilla and exotic lists with their indices.

    Returns:
        (vanilla_options, vanilla_indices, exotic_options, exotic_indices)
    """
    vanilla_opts, vanilla_idx = [], []
    exotic_opts, exotic_idx = [], []
    for i, opt in enumerate(options):
        if opt.get('instrument_class', 'vanilla') != 'vanilla':
            exotic_opts.append(opt)
            exotic_idx.append(i)
        else:
            vanilla_opts.append(opt)
            vanilla_idx.append(i)
    return vanilla_opts, vanilla_idx, exotic_opts, exotic_idx


def _exotic_greeks_surface_dte(
    exotic_options: list,
    spot_range: np.ndarray,
    dte_range: np.ndarray,
    risk_free_rate: float,
    base_iv: float,
    greek_index: int,
) -> np.ndarray:
    """Calculate 3D surface for exotic legs varying spot and DTE."""
    from services.exotic_pricing_adapter import calculate_exotic_all_greeks

    result = np.zeros((len(spot_range), len(dte_range)))
    for opt in exotic_options:
        opt_type_int = 1 if opt['option_type'] == 'call' else 0
        pos_sign = 1 if opt['position_type'] == 'long' else -1
        qty = opt['quantity'] * 100

        for i, spot in enumerate(spot_range):
            for j, dte in enumerate(dte_range):
                t = dte / 365.0
                greeks = calculate_exotic_all_greeks(
                    spot, opt['strike'], t,
                    risk_free_rate, base_iv, opt_type_int,
                    exotic_type=opt['instrument_class'],
                    barrier=opt.get('barrier', 0.0),
                    is_up=opt.get('is_up', True),
                    is_knock_in=opt.get('is_knock_in', False),
                    rebate=opt.get('rebate', 0.0),
                    payout=opt.get('payout', 1.0),
                )
                result[i, j] += greeks[greek_index] * qty * pos_sign
    return result


def _exotic_greeks_surface_iv(
    exotic_options: list,
    spot_range: np.ndarray,
    iv_range: np.ndarray,
    risk_free_rate: float,
    base_dte: float,
    greek_index: int,
) -> np.ndarray:
    """Calculate 3D surface for exotic legs varying spot and IV."""
    from services.exotic_pricing_adapter import calculate_exotic_all_greeks

    t = base_dte / 365.0
    result = np.zeros((len(spot_range), len(iv_range)))
    for opt in exotic_options:
        opt_type_int = 1 if opt['option_type'] == 'call' else 0
        pos_sign = 1 if opt['position_type'] == 'long' else -1
        qty = opt['quantity'] * 100

        for i, spot in enumerate(spot_range):
            for j, iv in enumerate(iv_range):
                greeks = calculate_exotic_all_greeks(
                    spot, opt['strike'], t,
                    risk_free_rate, iv, opt_type_int,
                    exotic_type=opt['instrument_class'],
                    barrier=opt.get('barrier', 0.0),
                    is_up=opt.get('is_up', True),
                    is_knock_in=opt.get('is_knock_in', False),
                    rebate=opt.get('rebate', 0.0),
                    payout=opt.get('payout', 1.0),
                )
                result[i, j] += greeks[greek_index] * qty * pos_sign
    return result


def calculate_portfolio_greeks_3d_dte(
    portfolio_json: str,
    spot_range: np.ndarray,
    dte_range: np.ndarray,
    risk_free_rate: float,
    base_iv: float,
    greek_index: int = 1  # Default to delta
) -> np.ndarray:
    """
    Calculate 3D Greek surface varying spot and DTE.

    Uses Numba for vanilla legs and Python loop for exotic legs.

    Args:
        portfolio_json: JSON string of portfolio
        spot_range: Array of spot prices
        dte_range: Array of DTEs (days)
        risk_free_rate: Risk-free rate
        base_iv: Base implied volatility (decimal)
        greek_index: Index of Greek to calculate (1=delta, 2=gamma, etc.)

    Returns:
        2D array of Greek values [len(spot_range), len(dte_range)]
    """
    portfolio_data = json.loads(portfolio_json)

    if not portfolio_data.get('options') and not portfolio_data.get('stock'):
        return np.zeros((len(spot_range), len(dte_range)))

    options = portfolio_data.get('options', [])
    vanilla_opts, _, exotic_opts, _ = _split_vanilla_exotic(options)

    result = np.zeros((len(spot_range), len(dte_range)))

    # Vanilla legs: fast Numba path
    if vanilla_opts:
        strikes = np.array([pos['strike'] for pos in vanilla_opts])
        option_types = np.array([1 if pos['option_type'] == 'call' else 0 for pos in vanilla_opts])
        position_types = np.array([1 if pos['position_type'] == 'long' else -1 for pos in vanilla_opts])
        quantities = np.array([pos['quantity'] * 100 for pos in vanilla_opts])

        result += calculate_portfolio_greeks_3d_dte_vectorized(
            strikes, option_types, position_types, quantities,
            spot_range, dte_range, risk_free_rate, base_iv, greek_index
        )

    # Exotic legs: Python loop
    if exotic_opts:
        result += _exotic_greeks_surface_dte(
            exotic_opts, spot_range, dte_range,
            risk_free_rate, base_iv, greek_index
        )

    # Add stock contribution (only affects delta)
    if portfolio_data.get('stock') and greek_index == 1:
        stock = portfolio_data['stock']
        stock_qty = stock['quantity'] * (1 if stock['position_type'] == 'long' else -1)
        result += stock_qty

    return result


def calculate_portfolio_greeks_3d_iv(
    portfolio_json: str,
    spot_range: np.ndarray,
    iv_range: np.ndarray,
    risk_free_rate: float,
    base_dte: float,
    greek_index: int = 1  # Default to delta
) -> np.ndarray:
    """
    Calculate 3D Greek surface varying spot and IV.

    Uses Numba for vanilla legs and Python loop for exotic legs.

    Args:
        portfolio_json: JSON string of portfolio
        spot_range: Array of spot prices
        iv_range: Array of IVs (decimal)
        risk_free_rate: Risk-free rate
        base_dte: Base DTE (days)
        greek_index: Index of Greek to calculate

    Returns:
        2D array of Greek values [len(spot_range), len(iv_range)]
    """
    portfolio_data = json.loads(portfolio_json)

    if not portfolio_data.get('options') and not portfolio_data.get('stock'):
        return np.zeros((len(spot_range), len(iv_range)))

    options = portfolio_data.get('options', [])
    vanilla_opts, _, exotic_opts, _ = _split_vanilla_exotic(options)

    result = np.zeros((len(spot_range), len(iv_range)))

    # Vanilla legs: fast Numba path
    if vanilla_opts:
        strikes = np.array([pos['strike'] for pos in vanilla_opts])
        option_types = np.array([1 if pos['option_type'] == 'call' else 0 for pos in vanilla_opts])
        position_types = np.array([1 if pos['position_type'] == 'long' else -1 for pos in vanilla_opts])
        quantities = np.array([pos['quantity'] * 100 for pos in vanilla_opts])

        result += calculate_portfolio_greeks_3d_iv_vectorized(
            strikes, option_types, position_types, quantities,
            spot_range, iv_range, risk_free_rate, base_dte, greek_index
        )

    # Exotic legs: Python loop
    if exotic_opts:
        result += _exotic_greeks_surface_iv(
            exotic_opts, spot_range, iv_range,
            risk_free_rate, base_dte, greek_index
        )

    # Add stock contribution (only affects delta)
    if portfolio_data.get('stock') and greek_index == 1:
        stock = portfolio_data['stock']
        stock_qty = stock['quantity'] * (1 if stock['position_type'] == 'long' else -1)
        result += stock_qty

    return result


def calculate_greeks_3d_strike(
    portfolio_json: str,
    spot_range: np.ndarray,
    strike_range: np.ndarray,
    risk_free_rate: float,
    base_iv: float,
    base_dte: float,
    greek_index: int = 1
) -> np.ndarray:
    """
    Calculate 3D Greek surface varying spot and strike for single option.

    Uses Numba-compiled parallel function for performance.

    Args:
        portfolio_json: JSON string (contains single option parameters)
        spot_range: Array of spot prices
        strike_range: Array of strikes
        risk_free_rate: Risk-free rate
        base_iv: Base IV (decimal)
        base_dte: Base DTE (days)
        greek_index: Index of Greek to calculate

    Returns:
        2D array of Greek values
    """
    portfolio_data = json.loads(portfolio_json)

    if not portfolio_data.get('options'):
        return np.zeros((len(spot_range), len(strike_range)))

    # Get option parameters from first option in portfolio
    pos = portfolio_data['options'][0]
    option_type = 1 if pos['option_type'] == 'call' else 0
    position_type = 1 if pos['position_type'] == 'long' else -1
    quantity = pos['quantity'] * 100

    # Use vectorized Numba function
    return calculate_greeks_3d_strike_vectorized(
        spot_range, strike_range, base_dte, risk_free_rate, base_iv,
        option_type, position_type, quantity, greek_index
    )
