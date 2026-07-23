"""
Numba-jitted hot-path kernels for the calibration app
=======================================================

The calibration app needs a few O(N) post-processing steps that run on
every UI rerender (residual matrices, normalised Z-scores for QQ-plots,
recovery-error vectors, log-return statistics).  We jit-compile them
with the same options as the backend simulators
(``parallel=True, cache=True, fastmath=True``) so they run at C-speed
under the Streamlit event loop.

Author: Thomas
Created: 2026
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def relative_recovery_error(
    estimated: np.ndarray, true: np.ndarray
) -> np.ndarray:
    """Per-parameter |est - true| / max(|true|, 1e-12).

    Vectorised version of the dictionary-based loop in
    :mod:`services.calibration_service` for plotting purposes.
    """
    n = estimated.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        denom = abs(true[i])
        if denom < 1e-12:
            denom = 1e-12
        out[i] = abs(estimated[i] - true[i]) / denom
    return out


@njit(cache=True, fastmath=True)
def reshape_residuals_to_grid(
    residuals: np.ndarray,
    quote_strikes: np.ndarray,
    quote_maturities: np.ndarray,
    grid_strikes: np.ndarray,
    grid_maturities: np.ndarray,
) -> np.ndarray:
    """Project a flat residual vector onto a (n_T, n_K) grid.

    Each quote is matched to its (maturity, strike) cell in the grid.
    ``grid_strikes`` is 2D ``(n_T, n_K)`` — strikes are per-maturity on the adaptive
    moneyness grid (real data tiles one shared row), so the strike match is taken
    *within* the matched maturity row. Cells without a matching quote stay NaN. Runs
    serially — parallel writes via prange would race on the (i_T, i_K) cell when two
    quotes hash to the same grid index, and the loop is microsecond-scale anyway.
    """
    n_T = grid_maturities.shape[0]
    n_K = grid_strikes.shape[1]
    out = np.full((n_T, n_K), np.nan, dtype=np.float64)
    for q in range(residuals.shape[0]):
        # closest grid index — quotes were generated on the same grid
        # so the match is exact up to floating-point noise
        T = quote_maturities[q]
        K = quote_strikes[q]
        i_T = 0
        i_K = 0
        best_dT = 1e30
        best_dK = 1e30
        for i in range(n_T):
            d = abs(grid_maturities[i] - T)
            if d < best_dT:
                best_dT = d
                i_T = i
        for j in range(n_K):
            d = abs(grid_strikes[i_T, j] - K)
            if d < best_dK:
                best_dK = d
                i_K = j
        out[i_T, i_K] = residuals[q]
    return out


@njit(cache=True, fastmath=True)
def standardised_residuals(
    residuals: np.ndarray, sigma_floor: float = 1e-12
) -> np.ndarray:
    """Z-score the residuals — used as input to the QQ-plot."""
    mu = 0.0
    n = residuals.shape[0]
    for i in range(n):
        mu += residuals[i]
    mu /= n
    var = 0.0
    for i in range(n):
        diff = residuals[i] - mu
        var += diff * diff
    var /= max(n - 1, 1)
    sigma = np.sqrt(var)
    if sigma < sigma_floor:
        sigma = sigma_floor
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = (residuals[i] - mu) / sigma
    return out


@njit(cache=True, fastmath=True)
def rolling_volatility(
    log_returns: np.ndarray, window: int, annualization: float
) -> np.ndarray:
    """Annualised rolling-window standard deviation of log-returns."""
    n = log_returns.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 1 or n < window:
        return out

    # Initial window
    s = 0.0
    s2 = 0.0
    for i in range(window):
        s += log_returns[i]
        s2 += log_returns[i] * log_returns[i]
    var = (s2 - s * s / window) / (window - 1)
    out[window - 1] = np.sqrt(max(var, 0.0)) * np.sqrt(annualization)

    # Slide
    for t in range(window, n):
        old = log_returns[t - window]
        new = log_returns[t]
        s += new - old
        s2 += new * new - old * old
        var = (s2 - s * s / window) / (window - 1)
        out[t] = np.sqrt(max(var, 0.0)) * np.sqrt(annualization)
    return out


@njit(cache=True, fastmath=True)
def correlation_matrix_from_cov(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix into the corresponding correlation matrix.

    Singular dimensions (variance ≤ 0 from rounding on a near-rank-deficient
    Gauss-Newton covariance) keep a diagonal of 1.0 and produce NaN
    off-diagonals — much safer than clamping to 1e-30 which makes neighbouring
    correlations explode toward ±∞.
    """
    n = cov.shape[0]
    out = np.empty((n, n), dtype=np.float64)
    diag = np.empty(n, dtype=np.float64)
    valid = np.empty(n, dtype=np.bool_)
    for i in range(n):
        d = cov[i, i]
        if d > 0.0:
            diag[i] = np.sqrt(d)
            valid[i] = True
        else:
            diag[i] = 0.0
            valid[i] = False
    for i in range(n):
        for j in range(n):
            if i == j:
                out[i, j] = 1.0
            elif not valid[i] or not valid[j]:
                out[i, j] = np.nan
            else:
                out[i, j] = cov[i, j] / (diag[i] * diag[j])
    return out


@njit(cache=True, fastmath=True)
def best_per_iteration(objective_history: np.ndarray) -> np.ndarray:
    """Cumulative best-so-far on an objective history (monotonic minimum)."""
    n = objective_history.shape[0]
    out = np.empty(n, dtype=np.float64)
    cur = objective_history[0]
    out[0] = cur
    for i in range(1, n):
        if objective_history[i] < cur:
            cur = objective_history[i]
        out[i] = cur
    return out
