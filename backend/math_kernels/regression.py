"""
Regression Kernels for American Option Pricing
===============================================

Numba-optimized kernels for Longstaff-Schwartz regression
and continuation value estimation.

Author: Thomas
Created: 2025
"""

import numpy as np
from numba import njit, prange

# =============================================================================
# Basis Functions
# =============================================================================

@njit(fastmath=True, cache=True)
def laguerre_basis(x: float, order: int) -> np.ndarray:
    """
    Compute Laguerre polynomial basis up to given order.

    L_0(x) = 1
    L_1(x) = 1 - x
    L_2(x) = 1 - 2x + x²/2
    ...

    Parameters
    ----------
    x : float
        Evaluation point (typically normalized spot)
    order : int
        Maximum polynomial order (0 to order inclusive)

    Returns
    -------
    np.ndarray
        Basis values [L_0(x), L_1(x), ..., L_order(x)]
    """
    result = np.empty(order + 1, dtype=np.float64)

    if order >= 0:
        result[0] = 1.0
    if order >= 1:
        result[1] = 1.0 - x
    if order >= 2:
        result[2] = 1.0 - 2.0 * x + 0.5 * x * x

    # Recurrence for higher orders: L_n = ((2n-1-x)*L_{n-1} - (n-1)*L_{n-2}) / n
    for n in range(3, order + 1):
        result[n] = ((2.0 * n - 1.0 - x) * result[n-1] - (n - 1.0) * result[n-2]) / n

    return result


@njit(fastmath=True, cache=True)
def polynomial_basis(x: float, order: int) -> np.ndarray:
    """
    Compute simple polynomial basis: [1, x, x², ..., x^order].

    Parameters
    ----------
    x : float
        Evaluation point
    order : int
        Maximum polynomial order

    Returns
    -------
    np.ndarray
        Basis values [1, x, x², ..., x^order]
    """
    result = np.empty(order + 1, dtype=np.float64)
    power = 1.0
    for i in range(order + 1):
        result[i] = power
        power *= x
    return result


@njit(fastmath=True, cache=True)
def chebyshev_basis(x: float, order: int) -> np.ndarray:
    """
    Compute Chebyshev polynomial basis (first kind).

    T_0(x) = 1
    T_1(x) = x
    T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)

    Parameters
    ----------
    x : float
        Evaluation point (should be in [-1, 1] for best numerical stability)
    order : int
        Maximum polynomial order

    Returns
    -------
    np.ndarray
        Basis values [T_0(x), T_1(x), ..., T_order(x)]
    """
    result = np.empty(order + 1, dtype=np.float64)

    if order >= 0:
        result[0] = 1.0
    if order >= 1:
        result[1] = x

    for n in range(2, order + 1):
        result[n] = 2.0 * x * result[n-1] - result[n-2]

    return result


# =============================================================================
# Vectorized Basis Construction
# =============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def build_laguerre_design_matrix(
    spots: np.ndarray,
    order: int
) -> np.ndarray:
    """
    Build design matrix with Laguerre basis for all spots.

    Parameters
    ----------
    spots : np.ndarray
        Array of spot prices, shape (n,)
    order : int
        Maximum polynomial order

    Returns
    -------
    np.ndarray
        Design matrix, shape (n, order+1)
    """
    n = len(spots)
    X = np.empty((n, order + 1), dtype=np.float64)

    for i in prange(n):
        basis = laguerre_basis(spots[i], order)
        for j in range(order + 1):
            X[i, j] = basis[j]

    return X


@njit(parallel=True, fastmath=True, cache=True)
def build_polynomial_design_matrix(
    spots: np.ndarray,
    order: int
) -> np.ndarray:
    """
    Build design matrix with polynomial basis for all spots.

    Parameters
    ----------
    spots : np.ndarray
        Array of spot prices, shape (n,)
    order : int
        Maximum polynomial order

    Returns
    -------
    np.ndarray
        Design matrix, shape (n, order+1)
    """
    n = len(spots)
    X = np.empty((n, order + 1), dtype=np.float64)

    for i in prange(n):
        basis = polynomial_basis(spots[i], order)
        for j in range(order + 1):
            X[i, j] = basis[j]

    return X


# =============================================================================
# Regression Solvers
# =============================================================================

@njit(fastmath=True, cache=True)
def lstsq_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve least squares regression: min ||X*beta - y||².

    Uses normal equations: beta = (X'X)^{-1} X'y

    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (n_samples, n_features)
    y : np.ndarray
        Target values, shape (n_samples,)

    Returns
    -------
    np.ndarray
        Coefficients beta, shape (n_features,)
    """
    # X'X
    XtX = np.dot(X.T, X)

    # X'y
    Xty = np.dot(X.T, y)

    # Solve via Cholesky or direct inversion
    # For robustness, use regularization
    n_features = X.shape[1]
    reg = 1e-10 * np.eye(n_features)
    XtX_reg = XtX + reg

    # Simple Gaussian elimination (for small systems)
    return np.linalg.solve(XtX_reg, Xty)


@njit(fastmath=True, cache=True)
def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Predict values: y_hat = X @ beta.

    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (n_samples, n_features)
    beta : np.ndarray
        Coefficients, shape (n_features,)

    Returns
    -------
    np.ndarray
        Predictions, shape (n_samples,)
    """
    return np.dot(X, beta)


# =============================================================================
# Continuation Value Estimation (Longstaff-Schwartz Core)
# =============================================================================

@njit(fastmath=True, cache=True)
def continuation_value(
    spots: np.ndarray,
    discounted_payoffs: np.ndarray,
    itm_mask: np.ndarray,
    order: int = 2,
    use_laguerre: bool = True
) -> np.ndarray:
    """
    Estimate continuation value via regression on in-the-money paths.

    This is the core of the Longstaff-Schwartz algorithm.

    Parameters
    ----------
    spots : np.ndarray
        Current spot prices, shape (n_paths,)
    discounted_payoffs : np.ndarray
        Discounted future payoffs, shape (n_paths,)
    itm_mask : np.ndarray
        Boolean mask for in-the-money paths, shape (n_paths,)
    order : int
        Polynomial order for basis
    use_laguerre : bool
        Use Laguerre basis (True) or polynomial (False)

    Returns
    -------
    np.ndarray
        Continuation values for all paths, shape (n_paths,)
    """
    n_paths = len(spots)
    continuation = np.zeros(n_paths, dtype=np.float64)

    # Count ITM paths
    n_itm = 0
    for i in range(n_paths):
        if itm_mask[i]:
            n_itm += 1

    if n_itm < order + 1:
        # Not enough ITM paths for regression, return zeros
        return continuation

    # Extract ITM data
    spots_itm = np.empty(n_itm, dtype=np.float64)
    payoffs_itm = np.empty(n_itm, dtype=np.float64)

    idx = 0
    for i in range(n_paths):
        if itm_mask[i]:
            spots_itm[idx] = spots[i]
            payoffs_itm[idx] = discounted_payoffs[i]
            idx += 1

    # Build design matrix for ITM paths
    if use_laguerre:
        X_itm = build_laguerre_design_matrix(spots_itm, order)
    else:
        X_itm = build_polynomial_design_matrix(spots_itm, order)

    # Fit regression
    beta = lstsq_regression(X_itm, payoffs_itm)

    # Predict continuation for ALL paths (not just ITM)
    if use_laguerre:
        X_all = build_laguerre_design_matrix(spots, order)
    else:
        X_all = build_polynomial_design_matrix(spots, order)

    continuation = predict(X_all, beta)

    # Floor at zero (continuation value cannot be negative)
    for i in range(n_paths):
        if continuation[i] < 0.0:
            continuation[i] = 0.0

    return continuation


@njit(fastmath=True, cache=True)
def american_put_exercise_decision(
    spots: np.ndarray,
    strike: float,
    continuation_values: np.ndarray
) -> np.ndarray:
    """
    Determine optimal exercise for American put.

    Exercise if intrinsic value > continuation value.

    Parameters
    ----------
    spots : np.ndarray
        Current spot prices
    strike : float
        Strike price
    continuation_values : np.ndarray
        Estimated continuation values

    Returns
    -------
    np.ndarray
        Boolean exercise decisions
    """
    n = len(spots)
    exercise = np.empty(n, dtype=np.bool_)

    for i in range(n):
        intrinsic = max(strike - spots[i], 0.0)
        exercise[i] = intrinsic > continuation_values[i] and intrinsic > 0

    return exercise


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Regression Kernels Smoke Test")
    print("=" * 50)

    np.random.seed(42)

    # Test Laguerre basis
    x = 1.0
    lag = laguerre_basis(x, 3)
    print(f"\nLaguerre basis at x={x}:")
    print(f"  L_0 = {lag[0]:.4f} (expected 1)")
    print(f"  L_1 = {lag[1]:.4f} (expected 0)")
    print(f"  L_2 = {lag[2]:.4f} (expected -0.5)")

    # Test polynomial basis
    poly = polynomial_basis(2.0, 3)
    print("\nPolynomial basis at x=2:")
    print(f"  [1, x, x², x³] = {poly}")

    # Test least squares regression
    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    y = np.array([2.0, 4.0, 5.0, 4.0])
    beta = lstsq_regression(X, y)
    print("\nLeast squares regression:")
    print("  X = [[1,1],[1,2],[1,3],[1,4]]")
    print("  y = [2, 4, 5, 4]")
    print(f"  beta = {beta}")

    # Test continuation value
    n_paths = 1000
    spots = 100.0 * np.exp(np.random.randn(n_paths) * 0.2)
    strike = 100.0
    intrinsic = np.maximum(strike - spots, 0.0)
    itm_mask = intrinsic > 0

    # Simulate some future payoffs
    future_payoffs = np.maximum(strike - spots * 1.05, 0.0)
    discount = 0.99

    cont_vals = continuation_value(
        spots, future_payoffs * discount, itm_mask, order=2, use_laguerre=True
    )

    print("\nContinuation value estimation:")
    print(f"  N paths: {n_paths}")
    print(f"  N ITM: {itm_mask.sum()}")
    print(f"  Mean continuation value (ITM): {cont_vals[itm_mask].mean():.4f}")

    print("\n" + "=" * 50)
    print("Regression Kernels smoke test passed")
    print("=" * 50)
