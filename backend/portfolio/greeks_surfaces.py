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

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# Import Greeks calculation functions from vectorized_bs
# (same convention: option_type 1=call, 0=put)
from backend.engines.vectorized_bs import (
    calculate_all_greeks as _calculate_all_greeks,
)
from backend.utils.constants.greeks import (
    GREEK_CHARM,
    GREEK_COLOR,
    GREEK_DELTA,
    GREEK_GAMMA,
    GREEK_PRICE,
    GREEK_RHO,
    GREEK_SPEED,
    GREEK_THETA,
    GREEK_ULTIMA,
    GREEK_VANNA,
    GREEK_VEGA,
    GREEK_VETA,
    GREEK_VOLGA,
    GREEK_ZOMMA,
)
from backend.utils.constants.time import DAYS_PER_YEAR


# NOTE: Greeks calculations are imported from backend.engines.vectorized_bs
# to avoid code duplication. The function _calculate_all_greeks is imported above.


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
    greek_index: int = 1,
    dividend_yield: float = 0.0,
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
    dividend_yield : float, default 0.0
        Annualized continuous dividend yield (decimal)

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
                    spot_range[i],
                    strikes[k],
                    time_to_expiry,
                    risk_free_rate,
                    volatility,
                    option_types[k],
                    dividend_yield,
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
    greek_index: int = 1,
    dividend_yield: float = 0.0,
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
    dividend_yield : float, default 0.0
        Annualized continuous dividend yield (decimal)

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
                    spot_range[i],
                    strikes[k],
                    time_to_expiry,
                    risk_free_rate,
                    iv_range[j],
                    option_types[k],
                    dividend_yield,
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
    greek_index: int = 1,
    dividend_yield: float = 0.0,
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
    dividend_yield : float, default 0.0
        Annualized continuous dividend yield (decimal)

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
                spot_range[i],
                strike_range[j],
                time_to_expiry,
                risk_free_rate,
                volatility,
                option_type,
                dividend_yield,
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
# P&L CALCULATIONS (for Streamlit adapter)
# =============================================================================
# NOTE: These functions use a DIFFERENT convention than pnl.py:
#   - Here: option_type = 1 (call), 0 (put)
#   - pnl.py: option_type = 1 (call), -1 (put)
# The Streamlit frontend relies on the 0=put convention.
# Do NOT remove these in favor of pnl.py without updating the frontend.
# =============================================================================


@njit(fastmath=True, cache=True)
def calculate_portfolio_pnl_at_expiry_arrays(
    spot: float,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    stock_quantity: float,
    stock_entry_price: float,
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
    stock_entry_price: float,
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
        pnls[i] = calculate_portfolio_pnl_at_expiry_arrays(
            spot_range[i],
            strikes,
            option_types,
            position_types,
            quantities,
            premiums,
            stock_quantity,
            stock_entry_price,
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
    "calculate_portfolio_pnl_at_expiry_arrays",
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


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Greeks Surfaces Smoke Test")
    print("=" * 50)

    # Test data - Bull Call Spread: Long 100 Call, Short 110 Call
    strikes = np.array([100.0, 110.0], dtype=np.float64)
    option_types = np.array([1, 1], dtype=np.int64)  # Both calls (1=call, 0=put)
    position_types = np.array([1, -1], dtype=np.int64)  # Long, Short
    quantities = np.array(
        [100.0, 100.0], dtype=np.float64
    )  # 1 contract each (100 multiplier)
    premiums = np.array([8.0, 3.0], dtype=np.float64)  # Premium per share

    # Market parameters
    risk_free_rate = 0.05
    volatility = 0.20
    base_dte = 30.0  # 30 days to expiry

    # Ranges for surfaces
    spot_range = np.linspace(80.0, 130.0, 50)
    dte_range = np.array([7.0, 14.0, 30.0, 60.0, 90.0])
    iv_range = np.array([0.15, 0.20, 0.25, 0.30, 0.35])
    strike_range = np.linspace(90.0, 120.0, 30)

    # --- Test 1: Greek name lookup ---
    print("\n--- Test 1: Greek Name Lookup ---")
    for idx in range(14):
        name = get_greek_name(idx)
        print(f"  Index {idx:2d}: {name}")
    invalid_name = get_greek_name(99)
    print(f"  Index 99 (invalid): {invalid_name}")

    # --- Test 2: Portfolio Greeks Surface (Spot vs DTE) ---
    print("\n--- Test 2: Greeks Surface (Spot vs DTE) ---")
    delta_surface = portfolio_greeks_surface_dte(
        strikes,
        option_types.astype(np.int64),
        position_types.astype(np.int64),
        quantities,
        spot_range,
        dte_range,
        risk_free_rate,
        volatility,
        GREEK_DELTA,
    )
    print(f"  Delta surface shape: {delta_surface.shape}")
    print(f"  Delta at S=100, DTE=30: {delta_surface[24, 2]:.4f}")  # Midpoint
    print(f"  Delta range: [{delta_surface.min():.4f}, {delta_surface.max():.4f}]")

    gamma_surface = portfolio_greeks_surface_dte(
        strikes,
        option_types.astype(np.int64),
        position_types.astype(np.int64),
        quantities,
        spot_range,
        dte_range,
        risk_free_rate,
        volatility,
        GREEK_GAMMA,
    )
    print(f"  Gamma surface shape: {gamma_surface.shape}")
    print(f"  Gamma at S=100, DTE=30: {gamma_surface[24, 2]:.6f}")

    # --- Test 3: Portfolio Greeks Surface (Spot vs IV) ---
    print("\n--- Test 3: Greeks Surface (Spot vs IV) ---")
    vega_surface = portfolio_greeks_surface_iv(
        strikes,
        option_types.astype(np.int64),
        position_types.astype(np.int64),
        quantities,
        spot_range,
        iv_range,
        risk_free_rate,
        base_dte,
        GREEK_VEGA,
    )
    print(f"  Vega surface shape: {vega_surface.shape}")
    print(f"  Vega at S=100, IV=20%: {vega_surface[24, 1]:.4f}")
    print(f"  Vega range: [{vega_surface.min():.4f}, {vega_surface.max():.4f}]")

    # --- Test 4: Single Option Greeks Surface (Spot vs Strike) ---
    print("\n--- Test 4: Single Option Surface (Spot vs Strike) ---")
    single_delta = single_option_greeks_surface_strike(
        spot_range,
        strike_range,
        base_dte,
        risk_free_rate,
        volatility,
        option_type=1,  # Call
        position_type=1,  # Long
        quantity=100,
        greek_index=GREEK_DELTA,
    )
    print(f"  Single call delta surface shape: {single_delta.shape}")
    print(f"  Delta at S=100, K=100 (ATM): {single_delta[24, 10]:.4f}")

    # --- Test 5: P&L Curve ---
    print("\n--- Test 5: P&L Curve ---")
    pnl_curve = calculate_pnl_curve(
        spot_range,
        strikes,
        option_types.astype(np.float64),
        position_types.astype(np.float64),
        quantities,
        premiums,
        stock_quantity=0.0,
        stock_entry_price=0.0,
    )
    print(f"  P&L curve length: {len(pnl_curve)}")
    print(f"  P&L at S=80: ${pnl_curve[0]:.2f}")
    print(f"  P&L at S=100: ${pnl_curve[24]:.2f}")
    print(f"  P&L at S=130: ${pnl_curve[-1]:.2f}")
    print(
        f"  Max P&L: ${pnl_curve.max():.2f} at S=${spot_range[np.argmax(pnl_curve)]:.2f}"
    )
    print(
        f"  Min P&L: ${pnl_curve.min():.2f} at S=${spot_range[np.argmin(pnl_curve)]:.2f}"
    )

    # --- Test 6: P&L at Single Spot ---
    print("\n--- Test 6: P&L at Single Spot ---")
    test_spots = [80.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    for spot in test_spots:
        pnl = calculate_portfolio_pnl_at_expiry_arrays(
            spot,
            strikes,
            option_types.astype(np.float64),
            position_types.astype(np.float64),
            quantities,
            premiums,
            stock_quantity=0.0,
            stock_entry_price=0.0,
        )
        print(f"  S={spot:6.1f}: P&L = ${pnl:8.2f}")

    # --- Test 7: P&L with Stock Position ---
    print("\n--- Test 7: P&L with Stock (Covered Call) ---")
    # Covered call: Long 100 shares + Short 1 Call @ 105
    cc_strikes = np.array([105.0], dtype=np.float64)
    cc_option_types = np.array([1.0], dtype=np.float64)  # Call
    cc_position_types = np.array([-1.0], dtype=np.float64)  # Short
    cc_quantities = np.array([100.0], dtype=np.float64)
    cc_premiums = np.array([5.0], dtype=np.float64)

    for spot in [90.0, 100.0, 105.0, 115.0]:
        pnl = calculate_portfolio_pnl_at_expiry_arrays(
            spot,
            cc_strikes,
            cc_option_types,
            cc_position_types,
            cc_quantities,
            cc_premiums,
            stock_quantity=100.0,
            stock_entry_price=100.0,
        )
        print(f"  S={spot:6.1f}: P&L = ${pnl:8.2f}")

    # --- Validation: Bull Call Spread P&L ---
    print("\n--- Validation: Bull Call Spread ---")
    # Net premium paid: 8 - 3 = 5 per share = $500 total
    # Max profit at S >= 110: (110-100) * 100 - 500 = $500
    # Max loss at S <= 100: -$500 (premium paid)
    pnl_below = calculate_portfolio_pnl_at_expiry_arrays(
        90.0,
        strikes,
        option_types.astype(np.float64),
        position_types.astype(np.float64),
        quantities,
        premiums,
        0.0,
        0.0,
    )
    pnl_above = calculate_portfolio_pnl_at_expiry_arrays(
        120.0,
        strikes,
        option_types.astype(np.float64),
        position_types.astype(np.float64),
        quantities,
        premiums,
        0.0,
        0.0,
    )
    print(f"  P&L at S=90 (below both strikes): ${pnl_below:.2f} (expected: -$500)")
    print(f"  P&L at S=120 (above both strikes): ${pnl_above:.2f} (expected: $500)")

    # Verify values
    assert abs(pnl_below - (-500.0)) < 0.01, f"Expected -500, got {pnl_below}"
    assert abs(pnl_above - 500.0) < 0.01, f"Expected 500, got {pnl_above}"
    print("  ✓ Validation passed!")

    print("\n" + "=" * 50)
    print("Greeks Surfaces smoke test passed")
    print("=" * 50)
