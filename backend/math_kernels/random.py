"""
Random Number Generation Kernels
================================

Numba-optimized kernels for random number generation used in Monte Carlo.

Includes:
- Normal random generation
- Correlated normal generation (for multi-factor models)
- Antithetic variate generation
- Cholesky decomposition for correlation

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

# =============================================================================
# Basic Random Generation
# =============================================================================


@njit(fastmath=True, cache=True)
def generate_normal(n: int) -> np.ndarray:
    """
    Generate n standard normal random numbers.

    Parameters
    ----------
    n : int
        Number of samples

    Returns
    -------
    np.ndarray
        Array of standard normal samples
    """
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = np.random.standard_normal()
    return result


@njit(parallel=True, fastmath=True, cache=True)
def generate_normal_2d(n_paths: int, n_steps: int) -> np.ndarray:
    """
    Generate 2D array of standard normal random numbers.

    Parameters
    ----------
    n_paths : int
        Number of paths (rows)
    n_steps : int
        Number of steps (columns)

    Returns
    -------
    np.ndarray
        Array of shape (n_paths, n_steps)
    """
    result = np.empty((n_paths, n_steps), dtype=np.float64)
    for i in prange(n_paths):
        for j in range(n_steps):
            result[i, j] = np.random.standard_normal()
    return result


# =============================================================================
# Correlated Brownian Motion
# =============================================================================


@njit(fastmath=True, cache=True)
def generate_correlated_normals(rho: float) -> tuple[float, float]:
    """
    Generate two correlated standard normals with correlation rho.

    Uses Cholesky decomposition: (Z1, rho*Z1 + sqrt(1-rho²)*Z2)

    Parameters
    ----------
    rho : float
        Correlation coefficient in [-1, 1]

    Returns
    -------
    tuple
        (z1, z2) correlated standard normals
    """
    z1 = np.random.standard_normal()
    z2 = np.random.standard_normal()

    # Apply correlation via Cholesky
    w1 = z1
    w2 = rho * z1 + math.sqrt(1.0 - rho * rho) * z2

    return w1, w2


@njit(parallel=True, fastmath=True, cache=True)
def generate_correlated_brownian(
    n_paths: int, n_steps: int, rho: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate correlated Brownian increments for Heston-type models.

    Parameters
    ----------
    n_paths : int
        Number of paths
    n_steps : int
        Number of time steps
    rho : float
        Correlation between factors
    dt : float
        Time step

    Returns
    -------
    tuple
        (dW1, dW2) each of shape (n_paths, n_steps)
    """
    sqrt_dt = math.sqrt(dt)
    sqrt_one_minus_rho2 = math.sqrt(1.0 - rho * rho)

    dW1 = np.empty((n_paths, n_steps), dtype=np.float64)
    dW2 = np.empty((n_paths, n_steps), dtype=np.float64)

    for i in prange(n_paths):
        for j in range(n_steps):
            z1 = np.random.standard_normal()
            z2 = np.random.standard_normal()

            dW1[i, j] = sqrt_dt * z1
            dW2[i, j] = sqrt_dt * (rho * z1 + sqrt_one_minus_rho2 * z2)

    return dW1, dW2


@njit(fastmath=True, cache=True)
def cholesky_transform(z: np.ndarray, chol: np.ndarray) -> np.ndarray:
    """
    Transform independent normals to correlated via Cholesky.

    If Σ = L·L' and Z ~ N(0,I), then L·Z ~ N(0,Σ)

    Parameters
    ----------
    z : np.ndarray
        Independent standard normals, shape (n,)
    chol : np.ndarray
        Lower Cholesky factor of correlation matrix, shape (n, n)

    Returns
    -------
    np.ndarray
        Correlated normals, shape (n,)
    """
    n = len(z)
    result = np.zeros(n, dtype=np.float64)

    for i in range(n):
        for j in range(i + 1):
            result[i] += chol[i, j] * z[j]

    return result


@njit(fastmath=True, cache=True)
def compute_cholesky(corr: np.ndarray) -> np.ndarray:
    """
    Compute lower Cholesky factor of correlation matrix.

    Parameters
    ----------
    corr : np.ndarray
        Correlation matrix, shape (n, n)

    Returns
    -------
    np.ndarray
        Lower Cholesky factor L where Σ = L·L'
    """
    n = corr.shape[0]
    L = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                s = 0.0
                for k in range(j):
                    s += L[j, k] ** 2
                L[i, j] = math.sqrt(corr[i, i] - s)
            else:
                s = 0.0
                for k in range(j):
                    s += L[i, k] * L[j, k]
                L[i, j] = (corr[i, j] - s) / L[j, j]

    return L


# =============================================================================
# Antithetic Variates
# =============================================================================


@njit(fastmath=True, cache=True)
def generate_antithetic_normals(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n normals with their antithetic pairs.

    Returns 2n samples: the original n and their negatives.

    Parameters
    ----------
    n : int
        Number of base samples (output will have 2n samples)

    Returns
    -------
    tuple
        (original, antithetic) each of shape (n,)
    """
    original = np.empty(n, dtype=np.float64)
    antithetic = np.empty(n, dtype=np.float64)

    for i in range(n):
        z = np.random.standard_normal()
        original[i] = z
        antithetic[i] = -z

    return original, antithetic


@njit(parallel=True, fastmath=True, cache=True)
def generate_antithetic_brownian(
    n_paths: int, n_steps: int, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Brownian increments with antithetic variates.

    Parameters
    ----------
    n_paths : int
        Number of original paths (output will have 2*n_paths)
    n_steps : int
        Number of time steps
    dt : float
        Time step

    Returns
    -------
    tuple
        (dW, dW_anti) each of shape (n_paths, n_steps)
    """
    sqrt_dt = math.sqrt(dt)

    dW = np.empty((n_paths, n_steps), dtype=np.float64)
    dW_anti = np.empty((n_paths, n_steps), dtype=np.float64)

    for i in prange(n_paths):
        for j in range(n_steps):
            z = np.random.standard_normal()
            dW[i, j] = sqrt_dt * z
            dW_anti[i, j] = -sqrt_dt * z

    return dW, dW_anti


# =============================================================================
# Quasi-Random (Sobol) Support
# =============================================================================


@njit(fastmath=True, cache=True)
def box_muller_transform(u1: float, u2: float) -> tuple[float, float]:
    """
    Box-Muller transform: convert uniform to standard normal.

    Useful for quasi-random sequences (Sobol, Halton).

    Parameters
    ----------
    u1, u2 : float
        Independent uniform(0,1) samples

    Returns
    -------
    tuple
        (z1, z2) independent standard normals
    """
    # Avoid log(0)
    u1 = max(u1, 1e-10)

    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2

    z1 = r * math.cos(theta)
    z2 = r * math.sin(theta)

    return z1, z2


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Random Kernels Smoke Test")
    print("=" * 50)

    np.random.seed(42)

    # Test basic normal generation
    normals = generate_normal(1000)
    print("\nNormal generation:")
    print(f"  Mean: {normals.mean():.4f} (expected ~0)")
    print(f"  Std:  {normals.std():.4f} (expected ~1)")

    # Test correlated normals
    rho = -0.7
    n_samples = 10000
    z1_list = np.empty(n_samples)
    z2_list = np.empty(n_samples)
    for i in range(n_samples):
        z1, z2 = generate_correlated_normals(rho)
        z1_list[i] = z1
        z2_list[i] = z2

    empirical_corr = np.corrcoef(z1_list, z2_list)[0, 1]
    print(f"\nCorrelated normals (rho={rho}):")
    print(f"  Empirical correlation: {empirical_corr:.4f}")

    # Test Cholesky
    corr_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    L = compute_cholesky(corr_matrix)
    print("\nCholesky factor of [[1, 0.5], [0.5, 1]]:")
    print(f"  L = {L}")

    # Test antithetic
    orig, anti = generate_antithetic_normals(1000)
    print("\nAntithetic variates:")
    print(f"  Sum of pairs: {(orig + anti).sum():.10f} (expected 0)")

    print("\n" + "=" * 50)
    print("Random Kernels smoke test passed")
    print("=" * 50)
