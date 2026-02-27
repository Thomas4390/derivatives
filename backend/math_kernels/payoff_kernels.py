"""
Payoff Computation Kernels
==========================

Numba-optimized kernels for vectorized payoff evaluation.

These are the low-level computational primitives used by option instruments.

Author: Thomas
Created: 2025
"""

import numpy as np
from numba import njit, prange


# =============================================================================
# Vanilla Option Payoffs
# =============================================================================

@njit(fastmath=True, cache=True)
def call_payoff(spot: float, strike: float) -> float:
    """
    Scalar call payoff: max(S - K, 0).

    Parameters
    ----------
    spot : float
        Spot price at expiry
    strike : float
        Strike price

    Returns
    -------
    float
        Call payoff
    """
    return max(spot - strike, 0.0)


@njit(fastmath=True, cache=True)
def put_payoff(spot: float, strike: float) -> float:
    """
    Scalar put payoff: max(K - S, 0).

    Parameters
    ----------
    spot : float
        Spot price at expiry
    strike : float
        Strike price

    Returns
    -------
    float
        Put payoff
    """
    return max(strike - spot, 0.0)


@njit(parallel=True, fastmath=True, cache=True)
def call_payoff_vec(spots: np.ndarray, strike: float) -> np.ndarray:
    """
    Vectorized call payoff: max(S - K, 0).

    Parameters
    ----------
    spots : np.ndarray
        Array of spot prices
    strike : float
        Strike price

    Returns
    -------
    np.ndarray
        Array of payoffs
    """
    n = len(spots)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        result[i] = max(spots[i] - strike, 0.0)
    return result


@njit(parallel=True, fastmath=True, cache=True)
def put_payoff_vec(spots: np.ndarray, strike: float) -> np.ndarray:
    """
    Vectorized put payoff: max(K - S, 0).

    Parameters
    ----------
    spots : np.ndarray
        Array of spot prices
    strike : float
        Strike price

    Returns
    -------
    np.ndarray
        Array of payoffs
    """
    n = len(spots)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        result[i] = max(strike - spots[i], 0.0)
    return result


# =============================================================================
# Digital/Binary Option Payoffs
# =============================================================================

@njit(fastmath=True, cache=True)
def digital_call_payoff(spot: float, strike: float, payout: float = 1.0) -> float:
    """
    Digital call payoff: payout if S > K, else 0.

    Parameters
    ----------
    spot : float
        Spot price at expiry
    strike : float
        Strike price
    payout : float
        Fixed payout amount

    Returns
    -------
    float
        Digital call payoff
    """
    return payout if spot > strike else 0.0


@njit(fastmath=True, cache=True)
def digital_put_payoff(spot: float, strike: float, payout: float = 1.0) -> float:
    """
    Digital put payoff: payout if S < K, else 0.

    Parameters
    ----------
    spot : float
        Spot price at expiry
    strike : float
        Strike price
    payout : float
        Fixed payout amount

    Returns
    -------
    float
        Digital put payoff
    """
    return payout if spot < strike else 0.0


@njit(parallel=True, fastmath=True, cache=True)
def digital_call_payoff_vec(
    spots: np.ndarray,
    strike: float,
    payout: float = 1.0
) -> np.ndarray:
    """Vectorized digital call payoff."""
    n = len(spots)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        result[i] = payout if spots[i] > strike else 0.0
    return result


@njit(parallel=True, fastmath=True, cache=True)
def digital_put_payoff_vec(
    spots: np.ndarray,
    strike: float,
    payout: float = 1.0
) -> np.ndarray:
    """Vectorized digital put payoff."""
    n = len(spots)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        result[i] = payout if spots[i] < strike else 0.0
    return result


# =============================================================================
# Strategy Payoffs (Combinations)
# =============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def straddle_payoff(spots: np.ndarray, strike: float) -> np.ndarray:
    """
    Straddle payoff: |S - K| = max(S-K,0) + max(K-S,0).

    Parameters
    ----------
    spots : np.ndarray
        Array of spot prices
    strike : float
        Strike price

    Returns
    -------
    np.ndarray
        Array of straddle payoffs
    """
    n = len(spots)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        result[i] = abs(spots[i] - strike)
    return result


@njit(parallel=True, fastmath=True, cache=True)
def strangle_payoff(
    spots: np.ndarray,
    strike_put: float,
    strike_call: float
) -> np.ndarray:
    """
    Strangle payoff: max(K_put - S, 0) + max(S - K_call, 0).

    Parameters
    ----------
    spots : np.ndarray
        Array of spot prices
    strike_put : float
        Put strike (lower)
    strike_call : float
        Call strike (higher)

    Returns
    -------
    np.ndarray
        Array of strangle payoffs
    """
    n = len(spots)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        put = max(strike_put - spots[i], 0.0)
        call = max(spots[i] - strike_call, 0.0)
        result[i] = put + call
    return result


@njit(parallel=True, fastmath=True, cache=True)
def butterfly_payoff(
    spots: np.ndarray,
    k_low: float,
    k_mid: float,
    k_high: float
) -> np.ndarray:
    """
    Butterfly spread payoff.

    Long 1 call at K_low, short 2 calls at K_mid, long 1 call at K_high.

    Parameters
    ----------
    spots : np.ndarray
        Array of spot prices
    k_low : float
        Lower strike
    k_mid : float
        Middle strike
    k_high : float
        Upper strike

    Returns
    -------
    np.ndarray
        Array of butterfly payoffs
    """
    n = len(spots)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        s = spots[i]
        payoff = (max(s - k_low, 0.0) -
                  2.0 * max(s - k_mid, 0.0) +
                  max(s - k_high, 0.0))
        result[i] = payoff
    return result


@njit(parallel=True, fastmath=True, cache=True)
def bull_call_spread_payoff(
    spots: np.ndarray,
    k_low: float,
    k_high: float
) -> np.ndarray:
    """
    Bull call spread payoff: long call at K_low, short call at K_high.

    Parameters
    ----------
    spots : np.ndarray
        Array of spot prices
    k_low : float
        Lower strike (long call)
    k_high : float
        Upper strike (short call)

    Returns
    -------
    np.ndarray
        Array of spread payoffs
    """
    n = len(spots)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        s = spots[i]
        result[i] = max(s - k_low, 0.0) - max(s - k_high, 0.0)
    return result


# =============================================================================
# Path-Dependent Payoffs
# =============================================================================

@njit(fastmath=True, cache=True)
def asian_arithmetic_payoff(
    path: np.ndarray,
    strike: float,
    is_call: bool
) -> float:
    """
    Arithmetic Asian option payoff based on average price.

    Parameters
    ----------
    path : np.ndarray
        Price path (all time steps)
    strike : float
        Strike price
    is_call : bool
        True for call, False for put

    Returns
    -------
    float
        Asian option payoff
    """
    avg = np.mean(path)
    if is_call:
        return max(avg - strike, 0.0)
    else:
        return max(strike - avg, 0.0)


@njit(fastmath=True, cache=True)
def lookback_floating_payoff(
    path: np.ndarray,
    is_call: bool
) -> float:
    """
    Floating strike lookback option payoff.

    Call: S_T - min(S_t)
    Put: max(S_t) - S_T

    Parameters
    ----------
    path : np.ndarray
        Price path
    is_call : bool
        True for call, False for put

    Returns
    -------
    float
        Lookback option payoff
    """
    terminal = path[-1]
    if is_call:
        return terminal - np.min(path)
    else:
        return np.max(path) - terminal


@njit(fastmath=True, cache=True)
def barrier_up_out_call_payoff(
    path: np.ndarray,
    strike: float,
    barrier: float
) -> float:
    """
    Up-and-out call: call payoff if barrier never breached.

    Parameters
    ----------
    path : np.ndarray
        Price path
    strike : float
        Strike price
    barrier : float
        Upper barrier level

    Returns
    -------
    float
        Barrier option payoff (0 if knocked out)
    """
    # Check if barrier was breached
    for i in range(len(path)):
        if path[i] >= barrier:
            return 0.0

    # Not knocked out, return call payoff
    return max(path[-1] - strike, 0.0)


@njit(fastmath=True, cache=True)
def barrier_down_out_put_payoff(
    path: np.ndarray,
    strike: float,
    barrier: float
) -> float:
    """
    Down-and-out put: put payoff if barrier never breached.

    Parameters
    ----------
    path : np.ndarray
        Price path
    strike : float
        Strike price
    barrier : float
        Lower barrier level

    Returns
    -------
    float
        Barrier option payoff (0 if knocked out)
    """
    # Check if barrier was breached
    for i in range(len(path)):
        if path[i] <= barrier:
            return 0.0

    # Not knocked out, return put payoff
    return max(strike - path[-1], 0.0)


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Payoff Kernels Smoke Test")
    print("=" * 50)

    # Test vanilla payoffs
    spots = np.array([90.0, 100.0, 110.0])
    strike = 100.0

    print(f"\nVanilla payoffs (K={strike}):")
    print(f"  Spots: {spots}")
    print(f"  Call:  {call_payoff_vec(spots, strike)}")
    print(f"  Put:   {put_payoff_vec(spots, strike)}")

    # Test digital payoffs
    print(f"\nDigital payoffs (K={strike}, payout=10):")
    print(f"  Call:  {digital_call_payoff_vec(spots, strike, 10.0)}")
    print(f"  Put:   {digital_put_payoff_vec(spots, strike, 10.0)}")

    # Test straddle
    print(f"\nStraddle payoff (K={strike}):")
    print(f"  {straddle_payoff(spots, strike)}")

    # Test butterfly
    spots_wide = np.linspace(85, 115, 7)
    print("\nButterfly payoff (K_low=95, K_mid=100, K_high=105):")
    print(f"  Spots: {spots_wide}")
    print(f"  Payoff: {butterfly_payoff(spots_wide, 95, 100, 105)}")

    # Test Asian payoff
    path = np.array([100.0, 102.0, 98.0, 105.0, 110.0])
    print("\nAsian arithmetic call (K=100):")
    print(f"  Path average: {np.mean(path):.2f}")
    print(f"  Payoff: {asian_arithmetic_payoff(path, 100.0, True):.2f}")

    print("\n" + "=" * 50)
    print("Payoff Kernels smoke test passed")
    print("=" * 50)
