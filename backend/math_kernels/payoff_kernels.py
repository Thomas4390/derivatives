"""
Payoff Computation Kernels
==========================

Numba-optimized kernels for vectorized payoff evaluation.

These are the low-level computational primitives used by option instruments.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

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
    # Cash-or-nothing standard: pay at S >= K (matches the N(d2) limit and
    # instruments/payoffs.py); the two implementations previously disagreed at S==K.
    return payout if spot >= strike else 0.0


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
    spots: np.ndarray, strike: float, payout: float = 1.0
) -> np.ndarray:
    """Vectorized digital call payoff."""
    n = len(spots)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        result[i] = payout if spots[i] >= strike else 0.0  # S>=K (see scalar variant)
    return result


@njit(parallel=True, fastmath=True, cache=True)
def digital_put_payoff_vec(
    spots: np.ndarray, strike: float, payout: float = 1.0
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
    spots: np.ndarray, strike_put: float, strike_call: float
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
    spots: np.ndarray, k_low: float, k_mid: float, k_high: float
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
        payoff = max(s - k_low, 0.0) - 2.0 * max(s - k_mid, 0.0) + max(s - k_high, 0.0)
        result[i] = payoff
    return result


@njit(parallel=True, fastmath=True, cache=True)
def bull_call_spread_payoff(
    spots: np.ndarray, k_low: float, k_high: float
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
def asian_arithmetic_payoff(path: np.ndarray, strike: float, is_call: bool) -> float:
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
    return max(strike - avg, 0.0)


@njit(fastmath=True, cache=True)
def asian_geometric_payoff(path: np.ndarray, strike: float, is_call: bool) -> float:
    """
    Geometric Asian option payoff based on geometric average price.

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
        Geometric Asian option payoff
    """
    geo_avg = np.exp(np.mean(np.log(path)))
    if is_call:
        return max(geo_avg - strike, 0.0)
    return max(strike - geo_avg, 0.0)


@njit(fastmath=True, cache=True)
def lookback_floating_payoff(path: np.ndarray, is_call: bool) -> float:
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
    return np.max(path) - terminal


@njit(fastmath=True, cache=True)
def barrier_up_out_call_payoff(
    path: np.ndarray, strike: float, barrier: float
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
    path: np.ndarray, strike: float, barrier: float
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
# Batch (vectorized) Path-Dependent Payoffs
# =============================================================================
#
# These operate on the full ``(n_paths, n_steps)`` array in a single
# ``prange`` loop, computing the *identical* per-path reduction as their
# scalar counterparts above. They exist to replace the Python-level
# ``for i in range(n_paths): scalar_kernel(path[i], ...)`` loops in
# ``instruments/payoffs.py``: that idiom pays the Python→Numba dispatch cost
# once per path (≈ n_paths interpreter round-trips) and is single-threaded.
# A batch kernel pays the dispatch once and runs the rows across all cores.
# Results are bit-identical to the scalar loop (each row's reduction is
# computed sequentially within one thread; ``prange`` only splits across rows).


@njit(parallel=True, fastmath=True, cache=True)
def asian_arithmetic_payoff_batch(
    paths: np.ndarray, strike: float, is_call: bool
) -> np.ndarray:
    """Arithmetic-average Asian payoff for a batch of paths.

    Parameters
    ----------
    paths : np.ndarray
        2-D price paths, shape ``(n_paths, n_steps)``.
    strike : float
        Strike price.
    is_call : bool
        True for call, False for put.

    Returns
    -------
    np.ndarray
        1-D payoffs, shape ``(n_paths,)``.
    """
    n = paths.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        avg = np.mean(paths[i])
        out[i] = max(avg - strike, 0.0) if is_call else max(strike - avg, 0.0)
    return out


@njit(parallel=True, fastmath=True, cache=True)
def lookback_floating_payoff_batch(paths: np.ndarray, is_call: bool) -> np.ndarray:
    """Floating-strike lookback payoff for a batch of paths.

    Call: ``S_T - min(S_t)``; Put: ``max(S_t) - S_T``.

    Parameters
    ----------
    paths : np.ndarray
        2-D price paths, shape ``(n_paths, n_steps)``.
    is_call : bool
        True for call, False for put.

    Returns
    -------
    np.ndarray
        1-D payoffs, shape ``(n_paths,)``.
    """
    n = paths.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        terminal = paths[i, -1]
        if is_call:
            out[i] = terminal - np.min(paths[i])
        else:
            out[i] = np.max(paths[i]) - terminal
    return out


@njit(parallel=True, fastmath=True, cache=True)
def barrier_up_out_call_payoff_batch(
    paths: np.ndarray, strike: float, barrier: float
) -> np.ndarray:
    """Up-and-out call payoff for a batch of paths.

    Call payoff if the barrier is never breached over the path, else 0.

    Parameters
    ----------
    paths : np.ndarray
        2-D price paths, shape ``(n_paths, n_steps)``.
    strike : float
        Strike price.
    barrier : float
        Upper barrier level.

    Returns
    -------
    np.ndarray
        1-D payoffs, shape ``(n_paths,)``.
    """
    n, n_steps = paths.shape
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        knocked_out = False
        for j in range(n_steps):
            if paths[i, j] >= barrier:
                knocked_out = True
                break
        out[i] = 0.0 if knocked_out else max(paths[i, n_steps - 1] - strike, 0.0)
    return out


@njit(parallel=True, fastmath=True, cache=True)
def barrier_down_out_put_payoff_batch(
    paths: np.ndarray, strike: float, barrier: float
) -> np.ndarray:
    """Down-and-out put payoff for a batch of paths.

    Put payoff if the barrier is never breached over the path, else 0.

    Parameters
    ----------
    paths : np.ndarray
        2-D price paths, shape ``(n_paths, n_steps)``.
    strike : float
        Strike price.
    barrier : float
        Lower barrier level.

    Returns
    -------
    np.ndarray
        1-D payoffs, shape ``(n_paths,)``.
    """
    n, n_steps = paths.shape
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        knocked_out = False
        for j in range(n_steps):
            if paths[i, j] <= barrier:
                knocked_out = True
                break
        out[i] = 0.0 if knocked_out else max(strike - paths[i, n_steps - 1], 0.0)
    return out


@njit(parallel=True, fastmath=True, cache=True)
def lookback_discounted_call_payoff_batch(paths: np.ndarray) -> np.ndarray:
    """Discounted floating-strike lookback call (Globe Trotter) for a batch.

    Payoff: ``max(S_T - min(S), 0) * (S_0 / min(S))``, with the divisor floored
    at 1e-10. One ``prange`` pass with a single per-row min scan (no
    ``np.min(axis=1)`` temporary), bit-identical to the NumPy version.

    Parameters
    ----------
    paths : np.ndarray
        2-D price paths, shape ``(n_paths, n_steps)``.

    Returns
    -------
    np.ndarray
        1-D payoffs, shape ``(n_paths,)``.
    """
    n = paths.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        row = paths[i]
        s0 = row[0]
        s_t = row[-1]
        path_min = np.min(row)
        path_min_safe = path_min if path_min > 1e-10 else 1e-10
        lookback = s_t - path_min
        if lookback < 0.0:
            lookback = 0.0
        out[i] = lookback * (s0 / path_min_safe)
    return out


@njit(parallel=True, fastmath=True, cache=True)
def asian_geometric_payoff_batch(
    paths: np.ndarray, strike: float, is_call: bool
) -> np.ndarray:
    """Geometric-average Asian payoff for a batch of paths.

    Mirrors the scalar ``asian_geometric_payoff``: the geometric average is
    ``exp(mean(log(S)))`` over every column of the row (including the initial
    spot), then a vanilla payoff against ``strike``.

    Parameters
    ----------
    paths : np.ndarray
        2-D price paths, shape ``(n_paths, n_steps + 1)``.
    strike : float
        Strike price.
    is_call : bool
        True for call, False for put.

    Returns
    -------
    np.ndarray
        1-D payoffs, shape ``(n_paths,)``.
    """
    n = paths.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        geo = np.exp(np.mean(np.log(paths[i])))
        out[i] = max(geo - strike, 0.0) if is_call else max(strike - geo, 0.0)
    return out


@njit(parallel=True, fastmath=True, cache=True)
def lookback_fixed_payoff_batch(
    paths: np.ndarray, strike: float, is_call: bool
) -> np.ndarray:
    """Fixed-strike lookback payoff for a batch of paths.

    Call: ``max(max(S_t) - K, 0)``; Put: ``max(K - min(S_t), 0)``.

    Parameters
    ----------
    paths : np.ndarray
        2-D price paths, shape ``(n_paths, n_steps + 1)``.
    strike : float
        Fixed strike price.
    is_call : bool
        True for call, False for put.

    Returns
    -------
    np.ndarray
        1-D payoffs, shape ``(n_paths,)``.
    """
    n = paths.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        if is_call:
            out[i] = max(np.max(paths[i]) - strike, 0.0)
        else:
            out[i] = max(strike - np.min(paths[i]), 0.0)
    return out


@njit(parallel=True, fastmath=True, cache=True)
def barrier_payoff_batch(
    paths: np.ndarray,
    strike: float,
    barrier: float,
    is_call: bool,
    is_up: bool,
    is_knock_in: bool,
    rebate: float,
) -> np.ndarray:
    """Single-barrier knock-in/out call/put payoff for a batch of paths.

    Discretely monitored on the simulation grid: the barrier is "breached" if
    any column of the row lies on the trigger side (``>= barrier`` for an up
    barrier, ``<= barrier`` for a down barrier). A knock-out pays the vanilla
    payoff unless breached (then ``rebate``); a knock-in pays the vanilla payoff
    only if breached (else ``rebate``). The rebate is paid at expiry (the engine
    applies the discount factor); the closed-form pays an at-hit rebate, so use
    ``rebate == 0`` for an apples-to-apples cross-check.

    Note
    ----
    Discrete monitoring under-counts breaches versus a continuously-monitored
    closed form; pass a fine ``n_steps`` (or a Broadie-Glasserman-Kou barrier
    shift upstream) to control the bias.

    Parameters
    ----------
    paths : np.ndarray
        2-D price paths, shape ``(n_paths, n_steps + 1)``.
    strike : float
        Strike price.
    barrier : float
        Barrier level.
    is_call, is_up, is_knock_in : bool
        Call/put, up/down barrier, knock-in/knock-out flags.
    rebate : float
        Cash paid (at expiry) when the option expires without value.

    Returns
    -------
    np.ndarray
        1-D payoffs, shape ``(n_paths,)``.
    """
    n, n_cols = paths.shape
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        breached = False
        for j in range(n_cols):
            s = paths[i, j]
            if (is_up and s >= barrier) or ((not is_up) and s <= barrier):
                breached = True
                break
        s_t = paths[i, n_cols - 1]
        vanilla = max(s_t - strike, 0.0) if is_call else max(strike - s_t, 0.0)
        if is_knock_in:
            out[i] = vanilla if breached else rebate
        else:
            out[i] = rebate if breached else vanilla
    return out


@njit(parallel=True, fastmath=True, cache=True)
def low_point_forward_payoff_batch(paths: np.ndarray) -> np.ndarray:
    """Low-point forward (MALP) payoff for a batch of paths.

    Payoff: ``S_0 * (S_T / min(S) - 1)``, with the divisor floored at 1e-10.
    One ``prange`` pass with a single per-row min scan, bit-identical to the
    NumPy version.

    Parameters
    ----------
    paths : np.ndarray
        2-D price paths, shape ``(n_paths, n_steps)``.

    Returns
    -------
    np.ndarray
        1-D payoffs, shape ``(n_paths,)``.
    """
    n = paths.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        row = paths[i]
        s0 = row[0]
        s_t = row[-1]
        path_min = np.min(row)
        path_min_safe = path_min if path_min > 1e-10 else 1e-10
        out[i] = s0 * (s_t / path_min_safe - 1.0)
    return out


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
