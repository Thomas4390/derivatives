"""
Vectorized Black-Scholes Engine
===============================

High-performance Numba-compiled functions for vectorized Greeks calculations.
This module provides fast array-based operations for the Streamlit frontend.

All calculations delegate to backend.utils.math (single source of truth).
This module provides convenience wrappers with option_type: int (1=call, 0=put)
interface expected by the frontend.

Performance optimizations:
- @njit(fastmath=True, cache=True): JIT compilation with fast math
- @njit(parallel=True) + prange: Parallel CPU execution

Note: Portfolio-level Greeks surfaces and P&L calculations have been moved to:
- backend/portfolio/greeks_surfaces.py - Greeks 3D surface calculations
- backend/portfolio/breakeven.py - P&L at expiry calculations

Author: Thomas
Created: 2025
"""

import numpy as np
from numba import njit, prange

# Import from single source of truth
from backend.utils.math import (
    bs_greeks as _bs_greeks,
)
from backend.utils.math import (
    bs_second_order_greeks,
    bs_third_order_greeks,
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
# BLACK-SCHOLES FIRST-ORDER GREEKS (Wrapper with option_type interface)
# =============================================================================

@njit(fastmath=True, cache=True)
def calculate_first_order_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: int
) -> tuple:
    """
    Calculate first-order Black-Scholes Greeks.

    Wrapper around utils.math.bs_greeks with option_type: int interface.

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
    # Convert option_type (1=call, 0=put) to is_call (bool)
    is_call = option_type == 1

    # Delegate to single source of truth
    return _bs_greeks(
        spot, strike, time_to_expiry, risk_free_rate, volatility, is_call
    )


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

    Uses shared functions from utils/math for all Greeks.

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

    # Convert option_type (1=call, 0=put) to is_call (bool)
    is_call = option_type == 1

    # First-order Greeks (from utils/math)
    price, delta, gamma, vega, theta, rho = _bs_greeks(
        spot, strike, time_to_expiry, risk_free_rate, volatility, is_call
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


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Vectorized BS Engine Smoke Test")
    print("=" * 50)

    # Test parameters
    spot, strike, t, r, vol = 100.0, 100.0, 0.5, 0.05, 0.20

    # Test first-order Greeks
    print("\n--- First-Order Greeks ---")
    price, delta, gamma, vega, theta, rho = calculate_first_order_greeks(
        spot, strike, t, r, vol, option_type=1  # Call
    )
    print("Call option (ATM):")
    print(f"  Price: ${price:.4f}")
    print(f"  Delta: {delta:.4f}")
    print(f"  Gamma: {gamma:.6f}")
    print(f"  Vega:  {vega:.4f} (per 1% vol)")
    print(f"  Theta: {theta:.4f} (per day)")
    print(f"  Rho:   {rho:.4f} (per 1% rate)")

    # Test put option
    price_put, delta_put, _, _, _, _ = calculate_first_order_greeks(
        spot, strike, t, r, vol, option_type=0  # Put
    )
    print("\nPut option (ATM):")
    print(f"  Price: ${price_put:.4f}")
    print(f"  Delta: {delta_put:.4f}")

    # Verify put-call parity for delta
    print(f"\nDelta verification: Call - Put = {delta - delta_put:.4f} (should be ~1)")

    # Test all 14 Greeks
    print("\n--- All 14 Greeks ---")
    all_greeks = calculate_all_greeks(spot, strike, t, r, vol, option_type=1)
    greek_names = [
        "price", "delta", "gamma", "vega", "theta", "rho",
        "vanna", "volga", "charm", "veta",
        "speed", "zomma", "color", "ultima"
    ]
    for i, name in enumerate(greek_names):
        print(f"  {name:>8}: {all_greeks[i]:>12.6f}")

    # Test vectorized calculation
    print("\n--- Vectorized Greeks ---")
    spot_range = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    greeks_matrix = calculate_greeks_vectorized(
        spot_range, strike, t, r, vol, option_type=1
    )
    print(f"Spot range: {spot_range}")
    print(f"Prices:     {greeks_matrix[:, GREEK_PRICE]}")
    print(f"Deltas:     {greeks_matrix[:, GREEK_DELTA]}")
    print(f"Gammas:     {greeks_matrix[:, GREEK_GAMMA]}")

    # Verify vectorized matches scalar
    print("\n--- Consistency Check ---")
    for i, s in enumerate(spot_range):
        scalar_greeks = calculate_all_greeks(s, strike, t, r, vol, option_type=1)
        assert np.allclose(greeks_matrix[i, :], scalar_greeks, rtol=1e-10), f"Mismatch at spot={s}"
    print("Vectorized matches scalar: ✓")

    print("\n" + "=" * 50)
    print("Vectorized BS Engine smoke test passed")
    print("=" * 50)
