"""
Private shared numeric core for the payoffs package: input validation and the
scalar Numba payoff kernels. Single source of truth imported by ``vanilla`` and
``exotic``.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _path_validity_code(arr: np.ndarray) -> int:
    """Single-pass finiteness/sign check for a 2-D path array.

    Returns
    -------
    int
        ``0`` if every value is finite and non-negative, ``1`` if any value is
        non-finite (NaN/Inf), ``2`` if all values are finite but some are
        negative. Non-finite takes precedence over negative, matching the
        original two-step ``np.any(~np.isfinite)`` then ``np.any(< 0)`` order.

    Notes
    -----
    Replaces two full-array boolean temporaries and two memory passes with one
    parallel pass and zero intermediate allocation. ``fastmath`` is **off** on
    purpose: it lets the compiler assume the absence of NaN/Inf and would elide
    the ``np.isfinite`` test entirely.
    """
    n, m = arr.shape
    n_nonfinite = 0
    n_negative = 0
    for i in prange(n):
        for j in range(m):
            v = arr[i, j]
            if not np.isfinite(v):
                n_nonfinite += 1
            elif v < 0.0:
                n_negative += 1
    if n_nonfinite > 0:
        return 1
    if n_negative > 0:
        return 2
    return 0


def _validate_spot_array(spot: np.ndarray | float, name: str = "Spot") -> np.ndarray:
    """
    Validate spot price array for payoff computation.

    Parameters
    ----------
    spot : np.ndarray
        Spot price(s) to validate
    name : str
        Variable name for error messages

    Returns
    -------
    np.ndarray
        Validated 1D float64 array

    Raises
    ------
    ValueError
        If array contains NaN, Inf, or negative values
    """
    spot_arr = np.atleast_1d(np.asarray(spot, dtype=np.float64))
    if np.any(~np.isfinite(spot_arr)):
        raise ValueError(f"{name} prices must be finite (no NaN or Inf)")
    if np.any(spot_arr < 0):
        raise ValueError(f"{name} prices must be non-negative")
    return spot_arr


def _validate_path_array(path: np.ndarray | list) -> np.ndarray:
    """
    Validate path array for path-dependent payoff computation.

    Parameters
    ----------
    path : np.ndarray
        Price path(s) to validate

    Returns
    -------
    np.ndarray
        Validated 2D float64 array with shape (n_paths, n_steps)

    Raises
    ------
    ValueError
        If array contains NaN, Inf, negative values, or has wrong dimensions
    """
    path_arr = np.atleast_2d(np.asarray(path, dtype=np.float64))
    if path_arr.ndim != 2:
        raise ValueError(f"Path must be 2D array, got {path_arr.ndim}D")
    if path_arr.shape[1] < 2:
        raise ValueError(
            f"Path must have at least 2 time steps, got {path_arr.shape[1]}"
        )
    # Single fused parallel pass (no boolean temporaries) instead of
    # ``np.any(~np.isfinite(...))`` followed by ``np.any(... < 0)``.
    validity = _path_validity_code(np.ascontiguousarray(path_arr))
    if validity == 1:
        raise ValueError("Path prices must be finite (no NaN or Inf)")
    if validity == 2:
        raise ValueError("Path prices must be non-negative")
    return path_arr


# =============================================================================
# NUMBA KERNELS (Hot Path)
# =============================================================================


@njit(cache=True, fastmath=True)
def _call_payoff(spots: np.ndarray, strike: float) -> np.ndarray:
    """Vectorized call payoff: max(S - K, 0)."""
    result = np.empty_like(spots)
    for i in range(len(spots)):
        result[i] = max(spots[i] - strike, 0.0)
    return result


@njit(cache=True, fastmath=True)
def _put_payoff(spots: np.ndarray, strike: float) -> np.ndarray:
    """Vectorized put payoff: max(K - S, 0)."""
    result = np.empty_like(spots)
    for i in range(len(spots)):
        result[i] = max(strike - spots[i], 0.0)
    return result


@njit(cache=True, fastmath=True)
def _digital_call_payoff(spots: np.ndarray, strike: float, payout: float) -> np.ndarray:
    """Vectorized digital call payoff: payout if S >= K, else 0.

    Note: Standard convention is digital call pays at spot >= strike.
    """
    result = np.empty_like(spots)
    for i in range(len(spots)):
        result[i] = payout if spots[i] >= strike else 0.0
    return result


@njit(cache=True, fastmath=True)
def _digital_put_payoff(spots: np.ndarray, strike: float, payout: float) -> np.ndarray:
    """Vectorized digital put payoff: payout if S < K, else 0."""
    result = np.empty_like(spots)
    for i in range(len(spots)):
        result[i] = payout if spots[i] < strike else 0.0
    return result
