"""
Pricing Adapter for Streamlit App

This module provides array-based pricing functions that use the new backend architecture.
These functions are specific to the Streamlit app's needs for vectorized calculations.

Performance: Uses Numba-compiled vectorized functions for fast bulk calculations.

Author: Thomas Vaudescal
"""

import numpy as np
import json
from typing import Optional
from dataclasses import dataclass

# Import Numba-optimized functions for performance
from backend.engines.vectorized_bs import (
    calculate_all_greeks as _calculate_all_greeks_numba,
    calculate_portfolio_greeks_3d_dte_vectorized,
    calculate_portfolio_greeks_3d_iv_vectorized,
    calculate_greeks_3d_strike_vectorized,
    calculate_portfolio_pnl_at_expiry as _calculate_pnl_numba,
    calculate_pnl_curve,
)

# Import from new backend architecture (for OptionPosition premium calculation)
from backend.instruments.options import VanillaOption
from backend.models.gbm import GBMModel
from backend.engines import BSAnalyticEngine
from backend.core.market import MarketEnvironment


# =============================================================================
# GLOBAL ENGINE (singleton for premium calculations)
# =============================================================================

_engine = BSAnalyticEngine()


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

    Uses Numba-compiled parallel function for performance.

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

    # Extract arrays from portfolio for Numba function
    options = portfolio_data.get('options', [])
    n_options = len(options)

    if n_options > 0:
        strikes = np.array([pos['strike'] for pos in options])
        option_types = np.array([1 if pos['option_type'] == 'call' else 0 for pos in options])
        position_types = np.array([1 if pos['position_type'] == 'long' else -1 for pos in options])
        quantities = np.array([pos['quantity'] * 100 for pos in options])  # Contract multiplier

        # Use vectorized Numba function
        result = calculate_portfolio_greeks_3d_dte_vectorized(
            strikes, option_types, position_types, quantities,
            spot_range, dte_range, risk_free_rate, base_iv, greek_index
        )
    else:
        result = np.zeros((len(spot_range), len(dte_range)))

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

    Uses Numba-compiled parallel function for performance.

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

    # Extract arrays from portfolio for Numba function
    options = portfolio_data.get('options', [])
    n_options = len(options)

    if n_options > 0:
        strikes = np.array([pos['strike'] for pos in options])
        option_types = np.array([1 if pos['option_type'] == 'call' else 0 for pos in options])
        position_types = np.array([1 if pos['position_type'] == 'long' else -1 for pos in options])
        quantities = np.array([pos['quantity'] * 100 for pos in options])

        # Use vectorized Numba function
        result = calculate_portfolio_greeks_3d_iv_vectorized(
            strikes, option_types, position_types, quantities,
            spot_range, iv_range, risk_free_rate, base_dte, greek_index
        )
    else:
        result = np.zeros((len(spot_range), len(iv_range)))

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


# =============================================================================
# LEGACY POSITION CLASSES FOR FRONTEND
# =============================================================================

class OptionPosition:
    """
    Option position for the Streamlit app.

    This is a frontend-specific class that mirrors the old API
    but uses the new backend for calculations.
    """

    def __init__(
        self,
        option_type: str,  # 'call' or 'put'
        position_type: str,  # 'long' or 'short'
        strike: float,
        quantity: int = 1,
        premium_paid: Optional[float] = None,  # If provided, use this instead of calculating
        dte_days: int = 30,
        volatility: float = 0.25,
        spot_price: float = 100.0,
        risk_free_rate: float = 0.05
    ):
        self.option_type = option_type
        self.position_type = position_type
        self.strike = strike
        self.quantity = quantity
        self.dte_days = dte_days
        self.volatility = volatility

        if premium_paid is not None:
            # Use provided premium
            self.premium_paid = premium_paid
        else:
            # Calculate premium using Numba function (faster)
            opt_type = 1 if option_type == 'call' else 0
            greeks = _calculate_all_greeks_numba(
                spot_price, strike, dte_days/365.0,
                risk_free_rate, volatility, opt_type
            )
            self.premium_paid = greeks[0]  # Price is index 0

    def __repr__(self):
        return (f"OptionPosition({self.option_type}, {self.position_type}, "
                f"K={self.strike}, qty={self.quantity})")


class StockPosition:
    """Stock position for the Streamlit app."""

    def __init__(
        self,
        position_type: str,  # 'long' or 'short'
        quantity: int,
        entry_price: float
    ):
        self.position_type = position_type
        self.quantity = quantity
        self.entry_price = entry_price

    def __repr__(self):
        return f"StockPosition({self.position_type}, qty={self.quantity}, entry={self.entry_price})"


class OptionsPortfolio:
    """
    Portfolio class for the Streamlit app.

    This mimics the old API but provides a simpler implementation
    for the frontend's needs.
    """

    def __init__(self, spot_price: float = 100.0, risk_free_rate: float = 0.05):
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.options: list[OptionPosition] = []
        self.stock: Optional[StockPosition] = None

    def add_option(self, position: OptionPosition):
        """Add an option position."""
        self.options.append(position)

    def add_stock(self, position: StockPosition):
        """Add a stock position."""
        self.stock = position

    def clear(self):
        """Clear all positions."""
        self.options.clear()
        self.stock = None
