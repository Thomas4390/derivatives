"""
Portfolio Greeks Surfaces
=========================

Numba-optimized 3D surface calculations for portfolio Greeks.

Provides high-performance computation of Greeks across:
- Spot price range
- Days to expiry (DTE) range
- Implied volatility (IV) range
- Strike range

All functions are JIT-compiled with parallel execution for optimal performance.

Author: Thomas
Created: 2025
"""

import numpy as np
from numba import njit, prange
from typing import Tuple

# Import directly from utils.math to avoid circular imports with engines
from backend.utils.math import (
    norm_cdf,
    norm_pdf,
    d1_d2,
    bs_second_order_greeks,
    bs_third_order_greeks,
    DAYS_PER_YEAR,
)

# Greek indices - defined locally to avoid circular imports
GREEK_PRICE = 0
GREEK_DELTA = 1
GREEK_GAMMA = 2
GREEK_VEGA = 3
GREEK_THETA = 4
GREEK_RHO = 5
GREEK_VANNA = 6
GREEK_VOLGA = 7
GREEK_CHARM = 8
GREEK_VETA = 9
GREEK_SPEED = 10
GREEK_ZOMMA = 11
GREEK_COLOR = 12
GREEK_ULTIMA = 13


# =============================================================================
# LOCAL GREEKS CALCULATION (to avoid circular imports with engines)
# =============================================================================

@njit(fastmath=True, cache=True)
def _calculate_first_order_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: int
) -> Tuple[float, float, float, float, float, float]:
    """Calculate first-order Black-Scholes Greeks."""
    if time_to_expiry <= 0:
        if option_type == 1:  # Call
            price = max(spot - strike, 0.0)
            delta = 1.0 if spot > strike else 0.0
        else:  # Put
            price = max(strike - spot, 0.0)
            delta = -1.0 if spot < strike else 0.0
        return price, delta, 0.0, 0.0, 0.0, 0.0

    d1, d2 = d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    sqrt_t = np.sqrt(time_to_expiry)

    n_d1 = norm_cdf(d1)
    n_d2 = norm_cdf(d2)
    n_prime_d1 = norm_pdf(d1)
    n_minus_d1 = norm_cdf(-d1)
    n_minus_d2 = norm_cdf(-d2)
    exp_rt = np.exp(-risk_free_rate * time_to_expiry)

    gamma = n_prime_d1 / (spot * volatility * sqrt_t) if volatility > 0 else 0.0
    vega = spot * n_prime_d1 * sqrt_t / 100.0

    if option_type == 1:  # Call
        price = spot * n_d1 - strike * exp_rt * n_d2
        delta = n_d1
        theta = (-spot * n_prime_d1 * volatility / (2 * sqrt_t)
                - risk_free_rate * strike * exp_rt * n_d2) / DAYS_PER_YEAR
        rho = strike * time_to_expiry * exp_rt * n_d2 / 100.0
    else:  # Put
        price = strike * exp_rt * n_minus_d2 - spot * n_minus_d1
        delta = n_d1 - 1.0
        theta = (-spot * n_prime_d1 * volatility / (2 * sqrt_t)
                + risk_free_rate * strike * exp_rt * n_minus_d2) / DAYS_PER_YEAR
        rho = -strike * time_to_expiry * exp_rt * n_minus_d2 / 100.0

    return price, delta, gamma, vega, theta, rho


@njit(fastmath=True, cache=True)
def _calculate_all_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: int
) -> np.ndarray:
    """Calculate all 14 Greeks in a single pass."""
    greeks = np.zeros(14)

    # First-order Greeks
    price, delta, gamma, vega, theta, rho = _calculate_first_order_greeks(
        spot, strike, time_to_expiry, risk_free_rate, volatility, option_type
    )
    greeks[0] = price
    greeks[1] = delta
    greeks[2] = gamma
    greeks[3] = vega
    greeks[4] = theta
    greeks[5] = rho

    # Second-order Greeks
    vanna, volga, charm, veta = bs_second_order_greeks(
        spot, strike, time_to_expiry, risk_free_rate, volatility
    )
    greeks[6] = vanna
    greeks[7] = volga
    greeks[8] = charm
    greeks[9] = veta

    # Third-order Greeks
    speed, zomma, color, ultima = bs_third_order_greeks(
        spot, strike, time_to_expiry, risk_free_rate, volatility
    )
    greeks[10] = speed
    greeks[11] = zomma
    greeks[12] = color
    greeks[13] = ultima

    return greeks


# =============================================================================
# PORTFOLIO GREEKS 3D SURFACES
# =============================================================================

@njit(fastmath=True, cache=True, parallel=True)
def portfolio_greeks_surface_dte(
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    spot_range: np.ndarray,
    dte_range: np.ndarray,
    risk_free_rate: float,
    volatility: float,
    greek_index: int = 1
) -> np.ndarray:
    """
    Calculate 2D matrix of a specific Greek varying spot and DTE.

    Uses parallel execution for performance.

    Parameters
    ----------
    strikes : np.ndarray
        Array of strike prices for each position
    option_types : np.ndarray
        Array of option types (1=call, 0=put)
    position_types : np.ndarray
        Array of position types (1=long, -1=short)
    quantities : np.ndarray
        Array of quantities (with contract multiplier)
    spot_range : np.ndarray
        Array of spot prices (x-axis)
    dte_range : np.ndarray
        Array of DTEs in days (y-axis)
    risk_free_rate : float
        Risk-free interest rate
    volatility : float
        Implied volatility (constant across surface)
    greek_index : int, default 1
        Index of Greek to return (1=delta, 2=gamma, etc.)

    Returns
    -------
    np.ndarray
        2D array of shape (n_spots, n_dte)
    """
    n_spots = len(spot_range)
    n_dte = len(dte_range)
    n_positions = len(strikes)
    result = np.zeros((n_spots, n_dte))

    for i in prange(n_spots):
        for j in range(n_dte):
            time_to_expiry = dte_range[j] / DAYS_PER_YEAR
            total_greek = 0.0

            for k in range(n_positions):
                greeks = _calculate_all_greeks(
                    spot_range[i], strikes[k], time_to_expiry,
                    risk_free_rate, volatility, option_types[k]
                )
                multiplier = quantities[k] * position_types[k]
                total_greek += greeks[greek_index] * multiplier

            result[i, j] = total_greek

    return result


@njit(fastmath=True, cache=True, parallel=True)
def portfolio_greeks_surface_iv(
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    spot_range: np.ndarray,
    iv_range: np.ndarray,
    risk_free_rate: float,
    base_dte: float,
    greek_index: int = 1
) -> np.ndarray:
    """
    Calculate 2D matrix of a specific Greek varying spot and IV.

    Parameters
    ----------
    strikes : np.ndarray
        Array of strike prices for each position
    option_types : np.ndarray
        Array of option types (1=call, 0=put)
    position_types : np.ndarray
        Array of position types (1=long, -1=short)
    quantities : np.ndarray
        Array of quantities
    spot_range : np.ndarray
        Array of spot prices (x-axis)
    iv_range : np.ndarray
        Array of IVs in decimal (y-axis)
    risk_free_rate : float
        Risk-free rate
    base_dte : float
        Base DTE in days (fixed for this surface)
    greek_index : int, default 1
        Index of Greek to return

    Returns
    -------
    np.ndarray
        2D array of shape (n_spots, n_iv)
    """
    n_spots = len(spot_range)
    n_iv = len(iv_range)
    n_positions = len(strikes)
    time_to_expiry = base_dte / DAYS_PER_YEAR
    result = np.zeros((n_spots, n_iv))

    for i in prange(n_spots):
        for j in range(n_iv):
            total_greek = 0.0

            for k in range(n_positions):
                greeks = _calculate_all_greeks(
                    spot_range[i], strikes[k], time_to_expiry,
                    risk_free_rate, iv_range[j], option_types[k]
                )
                multiplier = quantities[k] * position_types[k]
                total_greek += greeks[greek_index] * multiplier

            result[i, j] = total_greek

    return result


@njit(fastmath=True, cache=True, parallel=True)
def single_option_greeks_surface_strike(
    spot_range: np.ndarray,
    strike_range: np.ndarray,
    dte: float,
    risk_free_rate: float,
    volatility: float,
    option_type: int,
    position_type: int,
    quantity: int,
    greek_index: int = 1
) -> np.ndarray:
    """
    Calculate 2D matrix of a Greek varying spot and strike for single option.

    Parameters
    ----------
    spot_range : np.ndarray
        Array of spot prices (x-axis)
    strike_range : np.ndarray
        Array of strikes (y-axis)
    dte : float
        Days to expiration
    risk_free_rate : float
        Risk-free rate
    volatility : float
        Implied volatility
    option_type : int
        1 for call, 0 for put
    position_type : int
        1 for long, -1 for short
    quantity : int
        Quantity (with multiplier)
    greek_index : int, default 1
        Index of Greek to return

    Returns
    -------
    np.ndarray
        2D array of shape (n_spots, n_strikes)
    """
    n_spots = len(spot_range)
    n_strikes = len(strike_range)
    time_to_expiry = dte / DAYS_PER_YEAR
    multiplier = quantity * position_type
    result = np.zeros((n_spots, n_strikes))

    for i in prange(n_spots):
        for j in range(n_strikes):
            greeks = _calculate_all_greeks(
                spot_range[i], strike_range[j], time_to_expiry,
                risk_free_rate, volatility, option_type
            )
            result[i, j] = greeks[greek_index] * multiplier

    return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_greek_name(greek_index: int) -> str:
    """Get human-readable Greek name from index."""
    names = {
        0: "Price",
        1: "Delta",
        2: "Gamma",
        3: "Vega",
        4: "Theta",
        5: "Rho",
        6: "Vanna",
        7: "Volga",
        8: "Charm",
        9: "Veta",
        10: "Speed",
        11: "Zomma",
        12: "Color",
        13: "Ultima",
    }
    return names.get(greek_index, f"Greek_{greek_index}")


# =============================================================================
# P&L CALCULATIONS (for backward compatibility with Streamlit adapter)
# =============================================================================

@njit(fastmath=True, cache=True)
def calculate_portfolio_pnl_at_expiry(
    spot: float,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    stock_quantity: float,
    stock_entry_price: float
) -> float:
    """
    Calculate portfolio P&L at expiration for a given spot price.

    Parameters
    ----------
    spot : float
        Spot price at expiration
    strikes : np.ndarray
        Array of strike prices
    option_types : np.ndarray
        Array of option types (1=call, 0=put)
    position_types : np.ndarray
        Array of position types (1=long, -1=short)
    quantities : np.ndarray
        Array of quantities
    premiums : np.ndarray
        Array of premiums
    stock_quantity : float
        Stock quantity
    stock_entry_price : float
        Stock entry price

    Returns
    -------
    float
        Total P&L at expiration
    """
    pnl = 0.0

    # Initial cost of options
    for i in range(len(strikes)):
        if position_types[i] == 1:  # Long
            pnl -= premiums[i] * quantities[i]
        else:  # Short
            pnl += premiums[i] * quantities[i]

    # Initial cost of stock
    if stock_quantity != 0:
        pnl -= stock_entry_price * stock_quantity

    # Option values at expiration
    for i in range(len(strikes)):
        if option_types[i] == 1:  # Call
            payoff = max(spot - strikes[i], 0.0)
        else:  # Put
            payoff = max(strikes[i] - spot, 0.0)

        pnl += position_types[i] * quantities[i] * payoff

    # Stock value at expiration
    if stock_quantity != 0:
        pnl += stock_quantity * spot

    return pnl


@njit(fastmath=True, cache=True, parallel=True)
def calculate_pnl_curve(
    spot_range: np.ndarray,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    stock_quantity: float,
    stock_entry_price: float
) -> np.ndarray:
    """
    Calculate P&L curve across a range of spot prices.

    Parameters
    ----------
    spot_range : np.ndarray
        Array of spot prices
    strikes : np.ndarray
        Array of strike prices
    option_types : np.ndarray
        Array of option types (1=call, 0=put)
    position_types : np.ndarray
        Array of position types (1=long, -1=short)
    quantities : np.ndarray
        Array of quantities
    premiums : np.ndarray
        Array of premiums
    stock_quantity : float
        Stock quantity
    stock_entry_price : float
        Stock entry price

    Returns
    -------
    np.ndarray
        Array of P&L values
    """
    n_spots = len(spot_range)
    pnls = np.zeros(n_spots)

    for i in prange(n_spots):
        pnls[i] = calculate_portfolio_pnl_at_expiry(
            spot_range[i], strikes, option_types, position_types,
            quantities, premiums, stock_quantity, stock_entry_price
        )

    return pnls


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Surface functions
    "portfolio_greeks_surface_dte",
    "portfolio_greeks_surface_iv",
    "single_option_greeks_surface_strike",
    # P&L functions
    "calculate_portfolio_pnl_at_expiry",
    "calculate_pnl_curve",
    # Utility
    "get_greek_name",
    # Greek indices
    "GREEK_PRICE",
    "GREEK_DELTA",
    "GREEK_GAMMA",
    "GREEK_VEGA",
    "GREEK_THETA",
    "GREEK_RHO",
    "GREEK_VANNA",
    "GREEK_VOLGA",
    "GREEK_CHARM",
    "GREEK_VETA",
    "GREEK_SPEED",
    "GREEK_ZOMMA",
    "GREEK_COLOR",
    "GREEK_ULTIMA",
]
