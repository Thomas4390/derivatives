"""
Bates Characteristic Function
=============================

Numba-optimized characteristic function for Bates (1996) model
(Heston stochastic volatility + Merton-style jumps).

This is the SINGLE implementation used by:
    - BatesPricer (FFT pricing)
    - BatesModel.characteristic_function() (calibration)

Author: Thomas
Created: 2025
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def bates_characteristic_function(
    u: complex,
    s0: float,
    v0: float,
    t: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float
) -> complex:
    """
    Bates model characteristic function phi(u) = E^Q[exp(i*u*ln(S_T))].

    Combines Heston stochastic volatility with Merton-style jumps.

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
    xi : float
        Volatility of volatility
    rho : float
        Correlation
    lambda_j : float
        Jump intensity (expected jumps per year)
    mu_j : float
        Mean of log-jump size
    sigma_j : float
        Std of log-jump size

    Returns
    -------
    complex
        Characteristic function value phi(u)

    Notes
    -----
    The Bates CF is the product of the Heston CF and the jump CF:
        phi_Bates(u) = phi_Heston(u) * phi_Jump(u)

    The jump compensator k = E[J-1] = exp(mu_j + 0.5*sigma_j^2) - 1
    adjusts the drift to maintain the martingale property.
    """
    i = 1j
    eps = 1e-10

    # Jump compensator
    k = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0

    # Heston component (same as heston_cf but with jump-adjusted drift)
    d = np.sqrt((rho * xi * i * u - kappa) ** 2 + xi ** 2 * (i * u + u ** 2))

    numerator = kappa - rho * xi * i * u - d
    denominator = kappa - rho * xi * i * u + d

    # Protect against division by zero in g calculation
    if np.abs(denominator) < eps:
        denominator = eps + 0j

    g = numerator / denominator

    exp_dt = np.exp(-d * t)

    # Protect against log singularity and division by zero
    one_minus_g = 1.0 - g
    one_minus_g_exp = 1.0 - g * exp_dt

    if np.abs(one_minus_g) < eps:
        one_minus_g = eps + 0j
    if np.abs(one_minus_g_exp) < eps:
        one_minus_g_exp = eps + 0j

    # Drift includes jump compensator
    drift_adj = (r - lambda_j * k) * i * u * t

    C = (
        drift_adj +
        kappa * theta / (xi ** 2) * (
            numerator * t -
            2.0 * np.log(one_minus_g_exp / one_minus_g)
        )
    )

    D = (
        numerator / (xi ** 2) *
        ((1.0 - exp_dt) / one_minus_g_exp)
    )

    # Heston component
    heston_part = np.exp(C + D * v0 + i * u * np.log(s0))

    # Jump component: phi_J(u) = exp(lambda * t * (E[exp(i*u*ln(J))] - 1))
    # For lognormal jumps: E[exp(i*u*ln(J))] = exp(i*u*mu_j - 0.5*u^2*sigma_j^2)
    jump_exponent = i * u * mu_j - 0.5 * (u ** 2) * (sigma_j ** 2)

    # Clamp real part to prevent overflow (exp(700) ≈ 1e304)
    if jump_exponent.real > 700:
        jump_exponent = 700 + jump_exponent.imag * 1j

    jump_cf = np.exp(jump_exponent)
    jump_part = np.exp(lambda_j * t * (jump_cf - 1.0))

    return heston_part * jump_part


@njit(cache=True, fastmath=True, parallel=True)
def bates_cf_vectorized(
    u_arr: np.ndarray,
    s0: float,
    v0: float,
    t: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float
) -> np.ndarray:
    """
    Vectorized Bates characteristic function for FFT.

    Parameters
    ----------
    u_arr : np.ndarray
        Array of frequency arguments (complex)
    Other parameters : float
        Model parameters

    Returns
    -------
    np.ndarray
        Array of characteristic function values
    """
    n = len(u_arr)
    result = np.empty(n, dtype=np.complex128)

    for i in range(n):
        result[i] = bates_characteristic_function(
            u_arr[i], s0, v0, t, r,
            kappa, theta, xi, rho,
            lambda_j, mu_j, sigma_j
        )

    return result


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Bates Characteristic Function Smoke Test")
    print("=" * 50)

    # Test parameters
    s0, v0, t, r = 100.0, 0.04, 0.5, 0.05
    kappa, theta, xi, rho = 2.0, 0.04, 0.3, -0.7
    lambda_j, mu_j, sigma_j = 0.5, -0.1, 0.2

    # Test scalar CF
    print("\n--- Scalar Characteristic Function ---")
    u = 1.0 + 0.5j
    cf = bates_characteristic_function(u, s0, v0, t, r, kappa, theta, xi, rho, lambda_j, mu_j, sigma_j)
    print(f"u = {u}")
    print(f"phi(u) = {cf}")
    print(f"|phi(u)| = {np.abs(cf):.6f}")

    # Test at u=0 (should be 1)
    cf_zero = bates_characteristic_function(0.0 + 0j, s0, v0, t, r, kappa, theta, xi, rho, lambda_j, mu_j, sigma_j)
    print(f"\nphi(0) = {cf_zero}")
    assert np.abs(cf_zero - 1.0) < 1e-10, "CF at u=0 should be 1"
    print("phi(0) = 1 ✓")

    # Test vectorized CF
    print("\n--- Vectorized Characteristic Function ---")
    u_arr = np.array([0.5, 1.0, 1.5, 2.0]) + 0.5j
    cf_vec = bates_cf_vectorized(u_arr, s0, v0, t, r, kappa, theta, xi, rho, lambda_j, mu_j, sigma_j)
    print(f"u_arr = {u_arr}")
    print(f"|phi(u_arr)| = {np.abs(cf_vec)}")

    # Verify vectorized matches scalar
    print("\n--- Consistency Check ---")
    for i, ui in enumerate(u_arr):
        cf_scalar = bates_characteristic_function(ui, s0, v0, t, r, kappa, theta, xi, rho, lambda_j, mu_j, sigma_j)
        assert np.abs(cf_vec[i] - cf_scalar) < 1e-10, f"Mismatch at index {i}"
    print("Vectorized matches scalar: ✓")

    # Compare with Heston (lambda=0 should reduce to Heston CF)
    print("\n--- Comparison with Heston (lambda_j=0) ---")
    from backend.models.characteristic_functions.heston_cf import (
        heston_characteristic_function,
    )

    cf_bates_no_jump = bates_characteristic_function(u, s0, v0, t, r, kappa, theta, xi, rho, 0.0, mu_j, sigma_j)
    cf_heston = heston_characteristic_function(u, s0, v0, t, r, kappa, theta, xi, rho)
    print(f"Bates (lambda=0): |phi(u)| = {np.abs(cf_bates_no_jump):.6f}")
    print(f"Heston:           |phi(u)| = {np.abs(cf_heston):.6f}")
    assert np.abs(cf_bates_no_jump - cf_heston) < 1e-10, "Bates with lambda=0 should equal Heston"
    print("Bates(lambda=0) = Heston: ✓")

    # Parameter sensitivity
    print("\n--- Jump Intensity Sensitivity ---")
    for lam in [0.0, 0.5, 1.0, 2.0]:
        cf_test = bates_characteristic_function(1.0 + 0j, s0, v0, t, r, kappa, theta, xi, rho, lam, mu_j, sigma_j)
        print(f"lambda={lam:.1f}: |phi(1)| = {np.abs(cf_test):.6f}")

    print("\n" + "=" * 50)
    print("Bates CF smoke test passed")
    print("=" * 50)
