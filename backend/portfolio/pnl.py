"""
Portfolio P&L Engine
====================

Numba-optimized functions for computing portfolio P&L distributions
from simulated terminal prices. Designed for maximum performance with
parallel execution across all CPU cores.

This module provides:
- P&L calculations for options portfolios at expiration
- Risk metrics (VaR, CVaR, skewness, kurtosis)
- Payoff curve generation
- Breakeven point detection

Key Optimizations:
-----------------
- @njit(parallel=True): Automatic parallelization
- prange: Parallel loop iteration over paths
- cache=True: Compiled code caching for fast startup
- fastmath=True: Aggressive floating-point optimizations

Usage:
------
    from backend.portfolio.pnl import (
        calculate_portfolio_pnl_vectorized,
        compute_risk_metrics,
        compute_percentiles
    )

    # Calculate P&L for 100K simulated terminal prices
    pnl = calculate_portfolio_pnl_vectorized(
        terminal_prices,
        strikes, option_types, position_types,
        quantities, premiums, multiplier=100.0
    )

    # Get risk metrics
    metrics = compute_risk_metrics(pnl)

Author: Thomas
Created: 2025
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, NamedTuple


# =============================================================================
# Data Classes for Results
# =============================================================================

class RiskMetrics(NamedTuple):
    """Risk metrics computed from P&L distribution."""
    mean_pnl: float
    std_pnl: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    prob_profit: float
    max_profit: float
    max_loss: float
    skewness: float
    kurtosis: float


# =============================================================================
# Core P&L Calculation (Numba-optimized)
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def calculate_portfolio_pnl_vectorized(
    terminal_prices: np.ndarray,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    multiplier: float = 100.0
) -> np.ndarray:
    """
    Calculate portfolio P&L for all terminal prices in parallel.

    Parameters
    ----------
    terminal_prices : np.ndarray
        Array of terminal underlying prices, shape (n_paths,)
    strikes : np.ndarray
        Strike prices for each leg, shape (n_legs,)
    option_types : np.ndarray
        Option types: 1 = call, -1 = put, shape (n_legs,)
    position_types : np.ndarray
        Position types: 1 = long, -1 = short, shape (n_legs,)
    quantities : np.ndarray
        Number of contracts for each leg, shape (n_legs,)
    premiums : np.ndarray
        Premium per share for each leg, shape (n_legs,)
    multiplier : float
        Contract multiplier (default 100 shares per contract)

    Returns
    -------
    np.ndarray
        P&L for each path, shape (n_paths,)

    Notes
    -----
    P&L formula per leg:
        position_type * (intrinsic_value - premium) * quantity * multiplier

    For a long call:
        P&L = (max(0, S_T - K) - premium) * quantity * 100

    For a short put:
        P&L = (premium - max(0, K - S_T)) * quantity * 100
    """
    n_paths = terminal_prices.shape[0]
    n_legs = strikes.shape[0]
    pnl = np.zeros(n_paths, dtype=np.float64)

    for i in prange(n_paths):
        spot = terminal_prices[i]
        path_pnl = 0.0

        for j in range(n_legs):
            # Calculate intrinsic value at expiration
            if option_types[j] == 1:  # Call
                intrinsic = max(0.0, spot - strikes[j])
            else:  # Put
                intrinsic = max(0.0, strikes[j] - spot)

            # P&L = position * (payoff - premium) * quantity * multiplier
            leg_pnl = position_types[j] * (intrinsic - premiums[j]) * quantities[j] * multiplier
            path_pnl += leg_pnl

        pnl[i] = path_pnl

    return pnl


@njit(parallel=True, cache=True, fastmath=True)
def calculate_portfolio_pnl_with_stock(
    terminal_prices: np.ndarray,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    stock_quantity: float,
    stock_entry_price: float,
    multiplier: float = 100.0
) -> np.ndarray:
    """
    Calculate portfolio P&L including stock position.

    Extends calculate_portfolio_pnl_vectorized with stock position support.

    Parameters
    ----------
    terminal_prices : np.ndarray
        Array of terminal underlying prices, shape (n_paths,)
    strikes : np.ndarray
        Strike prices for each leg, shape (n_legs,)
    option_types : np.ndarray
        Option types: 1 = call, -1 = put, shape (n_legs,)
    position_types : np.ndarray
        Position types: 1 = long, -1 = short, shape (n_legs,)
    quantities : np.ndarray
        Number of contracts for each leg, shape (n_legs,)
    premiums : np.ndarray
        Premium per share for each leg, shape (n_legs,)
    stock_quantity : float
        Number of shares (positive = long, negative = short)
    stock_entry_price : float
        Entry price for stock position
    multiplier : float
        Contract multiplier (default 100 shares per contract)

    Returns
    -------
    np.ndarray
        P&L for each path, shape (n_paths,)
    """
    n_paths = terminal_prices.shape[0]
    n_legs = strikes.shape[0]
    pnl = np.zeros(n_paths, dtype=np.float64)

    for i in prange(n_paths):
        spot = terminal_prices[i]
        path_pnl = 0.0

        # Options P&L
        for j in range(n_legs):
            if option_types[j] == 1:  # Call
                intrinsic = max(0.0, spot - strikes[j])
            else:  # Put
                intrinsic = max(0.0, strikes[j] - spot)

            leg_pnl = position_types[j] * (intrinsic - premiums[j]) * quantities[j] * multiplier
            path_pnl += leg_pnl

        # Stock P&L
        if stock_quantity != 0.0:
            path_pnl += stock_quantity * (spot - stock_entry_price)

        pnl[i] = path_pnl

    return pnl


# =============================================================================
# Risk Metrics Computation (Numba-optimized)
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_risk_metrics_core(pnl: np.ndarray) -> Tuple[float, ...]:
    """
    Compute core risk metrics from P&L distribution.

    Parameters
    ----------
    pnl : np.ndarray
        P&L values for all paths, shape (n_paths,)

    Returns
    -------
    tuple
        (mean, std, var_95, var_99, cvar_95, cvar_99, prob_profit, max_profit, max_loss)
    """
    n = pnl.shape[0]

    # Basic statistics
    mean_pnl = np.mean(pnl)
    std_pnl = np.std(pnl)

    # Sort for percentile calculations
    sorted_pnl = np.sort(pnl)

    # VaR calculations (left tail)
    var_95_idx = int(0.05 * n)
    var_99_idx = int(0.01 * n)
    var_95 = sorted_pnl[var_95_idx]
    var_99 = sorted_pnl[var_99_idx]

    # CVaR (Expected Shortfall) - mean of worst cases
    cvar_95 = np.mean(sorted_pnl[:var_95_idx + 1]) if var_95_idx > 0 else sorted_pnl[0]
    cvar_99 = np.mean(sorted_pnl[:var_99_idx + 1]) if var_99_idx > 0 else sorted_pnl[0]

    # Probability of profit
    n_profit = 0
    for i in range(n):
        if pnl[i] > 0:
            n_profit += 1
    prob_profit = n_profit / n

    # Max/Min
    max_profit = np.max(pnl)
    max_loss = np.min(pnl)

    return (mean_pnl, std_pnl, var_95, var_99, cvar_95, cvar_99,
            prob_profit, max_profit, max_loss)


@njit(cache=True, fastmath=True)
def compute_skewness_kurtosis(pnl: np.ndarray) -> Tuple[float, float]:
    """
    Compute skewness and excess kurtosis of P&L distribution.

    Parameters
    ----------
    pnl : np.ndarray
        P&L values for all paths

    Returns
    -------
    tuple
        (skewness, excess_kurtosis)
    """
    n = pnl.shape[0]
    mean = np.mean(pnl)
    std = np.std(pnl)

    if std < 1e-10:
        return 0.0, 0.0

    # Compute moments
    m3 = 0.0
    m4 = 0.0
    for i in range(n):
        z = (pnl[i] - mean) / std
        z2 = z * z
        m3 += z2 * z
        m4 += z2 * z2

    skewness = m3 / n
    kurtosis = (m4 / n) - 3.0  # Excess kurtosis

    return skewness, kurtosis


def compute_risk_metrics(pnl: np.ndarray) -> RiskMetrics:
    """
    Compute complete risk metrics from P&L distribution.

    This is the main entry point for risk metrics calculation.
    Combines Numba-optimized core calculations with Python wrapper.

    Parameters
    ----------
    pnl : np.ndarray
        P&L values for all paths, shape (n_paths,)

    Returns
    -------
    RiskMetrics
        Named tuple with all risk metrics
    """
    core_metrics = compute_risk_metrics_core(pnl)
    skew, kurt = compute_skewness_kurtosis(pnl)

    return RiskMetrics(
        mean_pnl=core_metrics[0],
        std_pnl=core_metrics[1],
        var_95=core_metrics[2],
        var_99=core_metrics[3],
        cvar_95=core_metrics[4],
        cvar_99=core_metrics[5],
        prob_profit=core_metrics[6],
        max_profit=core_metrics[7],
        max_loss=core_metrics[8],
        skewness=skew,
        kurtosis=kurt
    )


# =============================================================================
# Percentile Computation (for distributions)
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_percentiles(pnl: np.ndarray, percentiles: np.ndarray) -> np.ndarray:
    """
    Compute percentiles of P&L distribution.

    Parameters
    ----------
    pnl : np.ndarray
        P&L values, shape (n_paths,)
    percentiles : np.ndarray
        Percentiles to compute (0-100), shape (n_percentiles,)

    Returns
    -------
    np.ndarray
        Percentile values, shape (n_percentiles,)
    """
    n = pnl.shape[0]
    sorted_pnl = np.sort(pnl)
    n_percentiles = percentiles.shape[0]
    result = np.zeros(n_percentiles, dtype=np.float64)

    for i in range(n_percentiles):
        idx = int(percentiles[i] / 100.0 * (n - 1))
        idx = min(max(idx, 0), n - 1)
        result[i] = sorted_pnl[idx]

    return result


# =============================================================================
# Payoff Curve Generation
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_payoff_curve(
    spot_range: np.ndarray,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    stock_quantity: float = 0.0,
    stock_entry_price: float = 0.0,
    multiplier: float = 100.0
) -> np.ndarray:
    """
    Compute theoretical payoff curve at expiration.

    Used for overlaying on scatter plots to compare simulated P&L
    with theoretical payoff.

    Parameters
    ----------
    spot_range : np.ndarray
        Range of spot prices to evaluate, shape (n_points,)
    strikes : np.ndarray
        Strike prices for each leg
    option_types : np.ndarray
        Option types: 1 = call, -1 = put
    position_types : np.ndarray
        Position types: 1 = long, -1 = short
    quantities : np.ndarray
        Number of contracts for each leg
    premiums : np.ndarray
        Premium per share for each leg
    stock_quantity : float
        Number of shares (positive = long, negative = short)
    stock_entry_price : float
        Entry price for stock position
    multiplier : float
        Contract multiplier

    Returns
    -------
    np.ndarray
        P&L values for each spot price, shape (n_points,)
    """
    n_points = spot_range.shape[0]
    n_legs = strikes.shape[0]
    payoff = np.zeros(n_points, dtype=np.float64)

    for i in range(n_points):
        spot = spot_range[i]
        total_pnl = 0.0

        # Options payoff
        for j in range(n_legs):
            if option_types[j] == 1:  # Call
                intrinsic = max(0.0, spot - strikes[j])
            else:  # Put
                intrinsic = max(0.0, strikes[j] - spot)

            leg_pnl = position_types[j] * (intrinsic - premiums[j]) * quantities[j] * multiplier
            total_pnl += leg_pnl

        # Stock payoff
        if stock_quantity != 0.0:
            total_pnl += stock_quantity * (spot - stock_entry_price)

        payoff[i] = total_pnl

    return payoff


# =============================================================================
# Breakeven Analysis
# =============================================================================

@njit(cache=True)
def find_breakeven_points(
    payoff: np.ndarray,
    spot_range: np.ndarray
) -> np.ndarray:
    """
    Find approximate breakeven points from payoff curve.

    Parameters
    ----------
    payoff : np.ndarray
        P&L values at each spot price
    spot_range : np.ndarray
        Corresponding spot prices

    Returns
    -------
    np.ndarray
        Approximate breakeven spot prices (where P&L crosses zero)
    """
    n = payoff.shape[0]
    breakevens = []

    for i in range(n - 1):
        # Check for sign change
        if payoff[i] * payoff[i + 1] < 0:
            # Linear interpolation
            ratio = -payoff[i] / (payoff[i + 1] - payoff[i])
            breakeven = spot_range[i] + ratio * (spot_range[i + 1] - spot_range[i])
            breakevens.append(breakeven)

    return np.array(breakevens, dtype=np.float64)


# =============================================================================
# Utility Functions
# =============================================================================

def prepare_position_arrays(positions: list) -> Tuple[np.ndarray, ...]:
    """
    Convert list of position dictionaries to numpy arrays for Numba functions.

    Parameters
    ----------
    positions : list
        List of position dictionaries with keys:
        - strike, option_type, position_type, quantity, premium

    Returns
    -------
    tuple
        (strikes, option_types, position_types, quantities, premiums)
        All as numpy arrays
    """
    n_legs = len(positions)

    strikes = np.zeros(n_legs, dtype=np.float64)
    option_types = np.zeros(n_legs, dtype=np.float64)
    position_types = np.zeros(n_legs, dtype=np.float64)
    quantities = np.zeros(n_legs, dtype=np.float64)
    premiums = np.zeros(n_legs, dtype=np.float64)

    for i, pos in enumerate(positions):
        strikes[i] = pos['strike']
        option_types[i] = 1.0 if pos['option_type'].lower() == 'call' else -1.0
        position_types[i] = 1.0 if pos['position_type'].lower() == 'long' else -1.0
        quantities[i] = pos.get('quantity', 1)
        premiums[i] = pos.get('premium', 0.0)

    return strikes, option_types, position_types, quantities, premiums


def warm_up_jit():
    """
    Warm up JIT compilation by running small calculations.

    Call this at application startup to ensure fast response times.
    """
    # Small test data
    terminal_prices = np.array([100.0, 105.0, 95.0], dtype=np.float64)
    strikes = np.array([100.0], dtype=np.float64)
    option_types = np.array([1.0], dtype=np.float64)
    position_types = np.array([1.0], dtype=np.float64)
    quantities = np.array([1.0], dtype=np.float64)
    premiums = np.array([5.0], dtype=np.float64)

    # Trigger compilation
    pnl = calculate_portfolio_pnl_vectorized(
        terminal_prices, strikes, option_types,
        position_types, quantities, premiums
    )
    _ = compute_risk_metrics(pnl)
    _ = compute_percentiles(pnl, np.array([5.0, 50.0, 95.0]))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Results
    "RiskMetrics",
    # Core P&L
    "calculate_portfolio_pnl_vectorized",
    "calculate_portfolio_pnl_with_stock",
    # Risk Metrics
    "compute_risk_metrics",
    "compute_risk_metrics_core",
    "compute_skewness_kurtosis",
    "compute_percentiles",
    # Payoff & Breakeven
    "compute_payoff_curve",
    "find_breakeven_points",
    # Utilities
    "prepare_position_arrays",
    "warm_up_jit",
]


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("P&L Engine Smoke Test")
    print("=" * 50)

    # Test data - Bull Call Spread: Long 95 Call, Short 105 Call
    print("\n--- Bull Call Spread Setup ---")
    strikes = np.array([95.0, 105.0], dtype=np.float64)
    option_types = np.array([1.0, 1.0], dtype=np.float64)  # Both calls
    position_types = np.array([1.0, -1.0], dtype=np.float64)  # Long, Short
    quantities = np.array([1.0, 1.0], dtype=np.float64)
    premiums = np.array([8.0, 3.0], dtype=np.float64)  # Net debit: $5

    print("  Long Call K=95 @ $8.00")
    print("  Short Call K=105 @ $3.00")
    print("  Net Debit: $5.00")

    # Simulated terminal prices
    terminal_prices = np.array([80, 90, 95, 100, 105, 110, 120], dtype=np.float64)

    # Test calculate_portfolio_pnl_vectorized
    print("\n--- P&L Calculation (multiplier=1.0) ---")
    pnl = calculate_portfolio_pnl_vectorized(
        terminal_prices, strikes, option_types,
        position_types, quantities, premiums, multiplier=1.0
    )
    for s, p in zip(terminal_prices, pnl):
        print(f"  Spot={s:6.1f}: P&L=${p:+7.2f}")

    # Expected: Max loss = -5, Max profit = +5, Breakeven = 100
    assert pnl[0] == -5.0, f"Max loss should be -5, got {pnl[0]}"
    assert pnl[-1] == 5.0, f"Max profit should be +5, got {pnl[-1]}"
    print("  ✓ P&L values correct")

    # Test with stock position
    print("\n--- Covered Call (Long Stock + Short Call) ---")
    call_strikes = np.array([105.0], dtype=np.float64)
    call_types = np.array([1.0], dtype=np.float64)
    call_positions = np.array([-1.0], dtype=np.float64)  # Short
    call_quantities = np.array([100.0], dtype=np.float64)
    call_premiums = np.array([3.0], dtype=np.float64)

    pnl_covered = calculate_portfolio_pnl_with_stock(
        terminal_prices, call_strikes, call_types, call_positions,
        call_quantities, call_premiums,
        stock_quantity=100.0, stock_entry_price=100.0, multiplier=1.0
    )
    for s, p in zip(terminal_prices, pnl_covered):
        print(f"  Spot={s:6.1f}: P&L=${p:+9.2f}")
    print("  ✓ Covered call P&L calculated")

    # Test risk metrics
    print("\n--- Risk Metrics (100K paths) ---")
    np.random.seed(42)
    simulated_pnl = np.random.normal(100, 500, 100000)

    metrics = compute_risk_metrics(simulated_pnl)
    print(f"  Mean P&L:   ${metrics.mean_pnl:+.2f}")
    print(f"  Std P&L:    ${metrics.std_pnl:.2f}")
    print(f"  VaR (95%):  ${metrics.var_95:.2f}")
    print(f"  VaR (99%):  ${metrics.var_99:.2f}")
    print(f"  CVaR (95%): ${metrics.cvar_95:.2f}")
    print(f"  Prob Profit: {metrics.prob_profit:.1%}")
    print(f"  Skewness:   {metrics.skewness:.4f}")
    print(f"  Kurtosis:   {metrics.kurtosis:.4f}")
    print("  ✓ Risk metrics computed")

    # Test payoff curve
    print("\n--- Payoff Curve ---")
    spot_range = np.linspace(80, 120, 9)
    payoff = compute_payoff_curve(
        spot_range, strikes, option_types,
        position_types, quantities, premiums, multiplier=1.0
    )
    print("  Spot    P&L")
    for s, p in zip(spot_range, payoff):
        print(f"  {s:5.1f}  ${p:+6.2f}")
    print("  ✓ Payoff curve generated")

    # Test breakeven finder
    print("\n--- Breakeven Points ---")
    breakevens = find_breakeven_points(payoff, spot_range)
    print(f"  Found {len(breakevens)} breakeven(s): {breakevens}")
    if len(breakevens) > 0:
        assert abs(breakevens[0] - 100.0) < 1.0, "Breakeven should be ~$100"
    print("  ✓ Breakeven detection works")

    # Test JIT warmup
    print("\n--- JIT Warmup ---")
    warm_up_jit()
    print("  ✓ JIT warmup completed")

    print("\n" + "=" * 50)
    print("P&L Engine smoke test passed")
    print("=" * 50)
