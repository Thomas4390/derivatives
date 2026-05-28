"""
Heston Characteristic Function
==============================

Numba-optimized characteristic function for Heston (1993) model.

This is the SINGLE implementation used by:
    - HestonPricer (FFT pricing)
    - HestonModel.characteristic_function() (calibration)

Uses the Gatheral (2006) formulation for numerical stability.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from backend.models.characteristic_functions._heston_core import (
    heston_cf_with_drift,
)


@njit(cache=True, fastmath=True)
def heston_characteristic_function(
    u: complex,
    s0: float,
    v0: float,
    t: float,
    r: float,
    kappa: float,
    theta: float,
    alpha: float,
    rho: float,
) -> complex:
    """
    Heston model characteristic function phi(u) = E^Q[exp(i*u*ln(S_T))].

    Uses the Gatheral (2006) formulation which is more numerically stable
    than the original Heston (1993) formulation.

    Parameters
    ----------
    u : complex
        Frequency argument
    s0 : float
        Initial spot price
    v0 : float
        Initial variance
    t : float
        Time to maturity
    r : float
        Risk-free rate
    kappa : float
        Mean reversion speed
    theta : float
        Long-run variance
    alpha : float
        Volatility of volatility
    rho : float
        Correlation

    Returns
    -------
    complex
        Characteristic function value phi(u)

    Notes
    -----
    The characteristic function is:
        phi(u) = exp(C(u,t) + D(u,t)*v0 + i*u*ln(s0))

    Where C and D satisfy Riccati ODEs derived from the Heston SDE.
    """
    return heston_cf_with_drift(u, s0, v0, t, r, kappa, theta, alpha, rho)


@njit(cache=True, fastmath=True, parallel=True)
def heston_cf_vectorized(
    u_arr: np.ndarray,
    s0: float,
    v0: float,
    t: float,
    r: float,
    kappa: float,
    theta: float,
    alpha: float,
    rho: float,
) -> np.ndarray:
    """
    Vectorized Heston characteristic function for FFT.

    Evaluates the characteristic function at multiple frequency points
    simultaneously for efficient FFT computation.

    Parameters
    ----------
    u_arr : np.ndarray
        Array of frequency arguments (complex)
    s0, v0, t, r, kappa, theta, alpha, rho : float
        Model parameters

    Returns
    -------
    np.ndarray
        Array of characteristic function values
    """
    n = len(u_arr)
    result = np.empty(n, dtype=np.complex128)

    for i in prange(n):
        result[i] = heston_characteristic_function(
            u_arr[i], s0, v0, t, r, kappa, theta, alpha, rho
        )

    return result


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Heston Characteristic Function Smoke Test")
    print("=" * 50)

    # Test parameters
    s0, v0, t, r = 100.0, 0.04, 0.5, 0.05
    kappa, theta, alpha, rho = 2.0, 0.04, 0.3, -0.7

    # Test scalar CF
    print("\n--- Scalar Characteristic Function ---")
    u = 1.0 + 0.5j
    cf = heston_characteristic_function(u, s0, v0, t, r, kappa, theta, alpha, rho)
    print(f"u = {u}")
    print(f"phi(u) = {cf}")
    print(f"|phi(u)| = {np.abs(cf):.6f}")

    # Test at u=0 (should be 1)
    cf_zero = heston_characteristic_function(
        0.0 + 0j, s0, v0, t, r, kappa, theta, alpha, rho
    )
    print(f"\nphi(0) = {cf_zero}")
    assert np.abs(cf_zero - 1.0) < 1e-10, "CF at u=0 should be 1"
    print("phi(0) = 1 ✓")

    # Test vectorized CF
    print("\n--- Vectorized Characteristic Function ---")
    u_arr = np.array([0.5, 1.0, 1.5, 2.0]) + 0.5j
    cf_vec = heston_cf_vectorized(u_arr, s0, v0, t, r, kappa, theta, alpha, rho)
    print(f"u_arr = {u_arr}")
    print(f"|phi(u_arr)| = {np.abs(cf_vec)}")

    # Verify vectorized matches scalar
    print("\n--- Consistency Check ---")
    for i, ui in enumerate(u_arr):
        cf_scalar = heston_characteristic_function(
            ui, s0, v0, t, r, kappa, theta, alpha, rho
        )
        assert np.abs(cf_vec[i] - cf_scalar) < 1e-10, f"Mismatch at index {i}"
    print("Vectorized matches scalar: ✓")

    # Test with different parameter values
    print("\n--- Parameter Sensitivity ---")
    for rho_test in [-0.9, -0.5, 0.0, 0.5]:
        cf_test = heston_characteristic_function(
            1.0 + 0j, s0, v0, t, r, kappa, theta, alpha, rho_test
        )
        print(f"rho={rho_test:+.1f}: |phi(1)| = {np.abs(cf_test):.6f}")

    print("\n" + "=" * 50)
    print("Heston CF smoke test passed")
    print("=" * 50)
