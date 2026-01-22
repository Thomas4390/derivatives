"""
Vectorized Black-Scholes Engine
===============================

High-performance Numba-compiled functions for vectorized Greeks calculations.
This module provides fast array-based operations for the Streamlit frontend.

Reuses mathematical primitives from backend/utils/math.py for consistency.

Performance optimizations:
- @njit(fastmath=True, cache=True): JIT compilation with fast math
- @njit(parallel=True) + prange: Parallel CPU execution
- Reuses Numba-compiled functions from utils/math

Note: Portfolio-level Greeks surfaces and P&L calculations have been moved to:
- backend/portfolio/greeks_surfaces.py - Greeks 3D surface calculations
- backend/portfolio/breakeven.py - P&L at expiry calculations

Author: Thomas
Created: 2025
"""

import numpy as np
from numba import njit, prange
from typing import Tuple

# Import shared mathematical primitives
from backend.utils.math import (
    norm_cdf,
    norm_pdf,
    d1_d2,
    bs_second_order_greeks,
    bs_third_order_greeks,
    DAYS_PER_YEAR,
)


# =============================================================================
# GREEK INDEX MAPPING
# =============================================================================
# All functions return Greeks in this order (14 total):
#
# First-order Greeks (indices 0-5):
#   0: price   - Option price
#   1: delta   - ∂V/∂S (spot sensitivity)
#   2: gamma   - ∂²V/∂S² (delta sensitivity to spot)
#   3: vega    - ∂V/∂σ per 1% vol (volatility sensitivity)
#   4: theta   - ∂V/∂t per day (time decay)
#   5: rho     - ∂V/∂r per 1% rate (rate sensitivity)
#
# Second-order Greeks (indices 6-9):
#   6: vanna   - ∂²V/∂S∂σ per 1% vol (delta-vol cross)
#   7: volga   - ∂²V/∂σ² per 1%² vol (vega convexity)
#   8: charm   - ∂²V/∂S∂t per day (delta decay)
#   9: veta    - ∂²V/∂σ∂t per day per 1% vol (vega decay)
#
# Third-order Greeks (indices 10-13):
#   10: speed  - ∂³V/∂S³ (gamma sensitivity to spot)
#   11: zomma  - ∂³V/∂S²∂σ per 1% vol (gamma-vol cross)
#   12: color  - ∂³V/∂S²∂t per day (gamma decay)
#   13: ultima - ∂³V/∂σ³ per 1%³ vol (volga sensitivity to vol)

# Greek indices for use with greek_index parameter
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
# BLACK-SCHOLES FIRST-ORDER GREEKS
# =============================================================================

@njit(fastmath=True, cache=True)
def calculate_first_order_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: int
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate first-order Black-Scholes Greeks.

    Computes price and first-order sensitivities using closed-form formulas.
    Handles expiration gracefully by returning intrinsic values.

    Parameters
    ----------
    spot : float
        Current underlying price (must be > 0)
    strike : float
        Strike price (must be > 0)
    time_to_expiry : float
        Time to expiration in years (0 = at expiry)
    risk_free_rate : float
        Annualized risk-free rate (decimal, e.g., 0.05 for 5%)
    volatility : float
        Annualized volatility (decimal, e.g., 0.20 for 20%)
    option_type : int
        Option type: 1 = call, 0 = put

    Returns
    -------
    Tuple[float, float, float, float, float, float]
        (price, delta, gamma, vega, theta, rho) where:
        - price: Option price
        - delta: ∂V/∂S (raw, not percentage)
        - gamma: ∂²V/∂S² (raw)
        - vega: ∂V/∂σ per 1% vol change (scaled by 1/100)
        - theta: ∂V/∂t per calendar day (scaled by 1/365)
        - rho: ∂V/∂r per 1% rate change (scaled by 1/100)
    """
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

    # Gamma (same for calls and puts)
    gamma = n_prime_d1 / (spot * volatility * sqrt_t) if volatility > 0 else 0.0

    # Vega (per 1% change)
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


# =============================================================================
# COMBINED GREEKS CALCULATION
# =============================================================================

@njit(fastmath=True, cache=True)
def calculate_all_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: int
) -> np.ndarray:
    """
    Calculate all 14 Greeks in a single pass.

    Uses shared functions from utils/math for second and third order Greeks.

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Strike price
    time_to_expiry : float
        Time to expiry in years
    risk_free_rate : float
        Risk-free rate (decimal)
    volatility : float
        Implied volatility (decimal)
    option_type : int
        1 for call, 0 for put

    Returns
    -------
    np.ndarray
        Array of 14 Greeks: [price, delta, gamma, vega, theta, rho,
                           vanna, volga, charm, veta, speed, zomma, color, ultima]
    """
    greeks = np.zeros(14)

    # First-order Greeks
    price, delta, gamma, vega, theta, rho = calculate_first_order_greeks(
        spot, strike, time_to_expiry, risk_free_rate, volatility, option_type
    )
    greeks[0] = price
    greeks[1] = delta
    greeks[2] = gamma
    greeks[3] = vega
    greeks[4] = theta
    greeks[5] = rho

    # Second-order Greeks (from utils/math)
    vanna, volga, charm, veta = bs_second_order_greeks(
        spot, strike, time_to_expiry, risk_free_rate, volatility
    )
    greeks[6] = vanna
    greeks[7] = volga
    greeks[8] = charm
    greeks[9] = veta

    # Third-order Greeks (from utils/math)
    speed, zomma, color, ultima = bs_third_order_greeks(
        spot, strike, time_to_expiry, risk_free_rate, volatility
    )
    greeks[10] = speed
    greeks[11] = zomma
    greeks[12] = color
    greeks[13] = ultima

    return greeks


# =============================================================================
# SINGLE OPTION GREEKS (FOR INDIVIDUAL POSITION ANALYSIS)
# =============================================================================

@njit(fastmath=True, cache=True, parallel=True)
def calculate_greeks_vectorized(
    spot_range: np.ndarray,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: int
) -> np.ndarray:
    """
    Calculate all Greeks for a single option across spot range.

    Parameters
    ----------
    spot_range : np.ndarray
        Array of spot prices
    strike : float
        Strike price
    time_to_expiry : float
        Time to expiry in years
    risk_free_rate : float
        Risk-free rate
    volatility : float
        Implied volatility
    option_type : int
        1 for call, 0 for put

    Returns
    -------
    np.ndarray
        2D array of shape (n_spots, 14) with all Greeks
    """
    n_spots = len(spot_range)
    result = np.zeros((n_spots, 14))

    for i in prange(n_spots):
        result[i, :] = calculate_all_greeks(
            spot_range[i], strike, time_to_expiry,
            risk_free_rate, volatility, option_type
        )

    return result
