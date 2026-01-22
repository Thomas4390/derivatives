"""
Merton Jump-Diffusion Characteristic Function
==============================================

Numba-optimized characteristic function for Merton (1976) jump-diffusion model.

This is the SINGLE implementation used by:
    - MertonPricer (FFT pricing)
    - MertonModel.characteristic_function() (calibration)

The Merton model combines GBM (constant volatility) with compound Poisson jumps:
    dS/S = (r - λk)dt + σdW + (J - 1)dN

Where:
    - σ: diffusion volatility
    - λ: jump intensity (Poisson arrival rate)
    - J: lognormal jump multiplier, ln(J) ~ N(μ_j, σ_j²)
    - k = E[J - 1] = exp(μ_j + 0.5σ_j²) - 1 (jump compensator)

References:
    Merton, R.C. (1976). "Option Pricing When Underlying Stock Returns
    Are Discontinuous." Journal of Financial Economics, 3(1-2), 125-144.

Author: Thomas
Created: 2025
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def merton_characteristic_function(
    u: complex,
    s0: float,
    t: float,
    r: float,
    sigma: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float
) -> complex:
    """
    Merton jump-diffusion characteristic function φ(u) = E^Q[exp(i·u·ln(S_T))].

    Parameters
    ----------
    u : complex
        Frequency argument
    s0 : float
        Initial spot price
    t : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Diffusion volatility
    lambda_j : float
        Jump intensity (expected jumps per year)
    mu_j : float
        Mean of log-jump size
    sigma_j : float
        Std of log-jump size

    Returns
    -------
    complex
        Characteristic function value φ(u)

    Notes
    -----
    The characteristic function decomposes as:
        φ(u) = φ_GBM(u) × φ_Jump(u)

    GBM component (log-price under Q-measure):
        φ_GBM(u) = exp(i·u·ln(S_0) + i·u·(r - 0.5σ² - λk)·t - 0.5·σ²·u²·t)

    Jump component:
        φ_Jump(u) = exp(λ·t·(E[exp(i·u·ln(J))] - 1))
                  = exp(λ·t·(exp(i·u·μ_j - 0.5·u²·σ_j²) - 1))

    The drift adjustment -λk ensures the martingale property under Q.
    """
    i = 1j

    # Jump compensator: k = E[J - 1] = exp(μ_j + 0.5σ_j²) - 1
    k = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0

    # GBM component (diffusion part)
    # ln(S_T) = ln(S_0) + (r - 0.5σ² - λk)t + σW_T + Σln(J_i)
    # φ_diffusion(u) = exp(i·u·ln(S_0) + i·u·(r - 0.5σ² - λk)·t - 0.5σ²·u²·t)
    drift = r - 0.5 * sigma ** 2 - lambda_j * k
    diffusion_exponent = (
        i * u * np.log(s0) +
        i * u * drift * t -
        0.5 * sigma ** 2 * u ** 2 * t
    )

    # Jump component
    # φ_J(u) = E[exp(i·u·ln(J))] = exp(i·u·μ_j - 0.5·u²·σ_j²)
    # For lognormal J with ln(J) ~ N(μ_j, σ_j²)
    jump_cf_exponent = i * u * mu_j - 0.5 * (u ** 2) * (sigma_j ** 2)

    # Clamp real part to prevent overflow (exp(700) ≈ 1e304)
    if jump_cf_exponent.real > 700:
        jump_cf_exponent = 700 + jump_cf_exponent.imag * 1j

    jump_cf = np.exp(jump_cf_exponent)

    # Complete jump component: exp(λ·t·(φ_J(u) - 1))
    jump_exponent = lambda_j * t * (jump_cf - 1.0)

    # Combined characteristic function
    return np.exp(diffusion_exponent + jump_exponent)


@njit(cache=True, fastmath=True)
def merton_cf_vectorized(
    u_arr: np.ndarray,
    s0: float,
    t: float,
    r: float,
    sigma: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float
) -> np.ndarray:
    """
    Vectorized Merton characteristic function for FFT.

    Evaluates the characteristic function at multiple frequency points
    simultaneously for efficient FFT computation.

    Parameters
    ----------
    u_arr : np.ndarray
        Array of frequency arguments (complex)
    s0 : float
        Initial spot price
    t : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Diffusion volatility
    lambda_j : float
        Jump intensity
    mu_j : float
        Mean of log-jump size
    sigma_j : float
        Std of log-jump size

    Returns
    -------
    np.ndarray
        Array of characteristic function values
    """
    n = len(u_arr)
    result = np.empty(n, dtype=np.complex128)

    for idx in range(n):
        result[idx] = merton_characteristic_function(
            u_arr[idx], s0, t, r, sigma, lambda_j, mu_j, sigma_j
        )

    return result


# =============================================================================
# Factory function for FFT pricer
# =============================================================================

def create_merton_cf(
    s0: float,
    t: float,
    r: float,
    sigma: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float
):
    """
    Create a characteristic function callable for FFT pricing.

    This returns a function that takes only the frequency argument u,
    with all model parameters pre-bound.

    Parameters
    ----------
    s0, t, r, sigma, lambda_j, mu_j, sigma_j : float
        Model parameters

    Returns
    -------
    callable
        Function cf(u) -> np.ndarray of characteristic function values
    """
    def cf(u: np.ndarray) -> np.ndarray:
        return merton_cf_vectorized(u, s0, t, r, sigma, lambda_j, mu_j, sigma_j)

    return cf


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Merton Characteristic Function Smoke Test")
    print("=" * 50)

    # Test parameters
    s0, t, r = 100.0, 0.5, 0.05
    sigma = 0.20
    lambda_j, mu_j, sigma_j = 0.5, -0.1, 0.2

    # Test scalar CF
    print("\n--- Scalar Characteristic Function ---")
    u = 1.0 + 0.5j
    cf = merton_characteristic_function(u, s0, t, r, sigma, lambda_j, mu_j, sigma_j)
    print(f"u = {u}")
    print(f"phi(u) = {cf}")
    print(f"|phi(u)| = {np.abs(cf):.6f}")

    # Test at u=0 (should be 1)
    cf_zero = merton_characteristic_function(0.0 + 0j, s0, t, r, sigma, lambda_j, mu_j, sigma_j)
    print(f"\nphi(0) = {cf_zero}")
    assert np.abs(cf_zero - 1.0) < 1e-10, "CF at u=0 should be 1"
    print("phi(0) = 1 ✓")

    # Test vectorized CF
    print("\n--- Vectorized Characteristic Function ---")
    u_arr = np.array([0.5, 1.0, 1.5, 2.0]) + 0.5j
    cf_vec = merton_cf_vectorized(u_arr, s0, t, r, sigma, lambda_j, mu_j, sigma_j)
    print(f"u_arr = {u_arr}")
    print(f"|phi(u_arr)| = {np.abs(cf_vec)}")

    # Verify vectorized matches scalar
    print("\n--- Consistency Check ---")
    for i, ui in enumerate(u_arr):
        cf_scalar = merton_characteristic_function(ui, s0, t, r, sigma, lambda_j, mu_j, sigma_j)
        assert np.abs(cf_vec[i] - cf_scalar) < 1e-10, f"Mismatch at index {i}"
    print("Vectorized matches scalar: ✓")

    # Test factory function
    print("\n--- Factory Function ---")
    cf_factory = create_merton_cf(s0, t, r, sigma, lambda_j, mu_j, sigma_j)
    cf_factory_result = cf_factory(u_arr)
    assert np.allclose(cf_factory_result, cf_vec), "Factory mismatch"
    print("Factory function works correctly: ✓")

    # Compare with no jumps (should approach GBM CF)
    print("\n--- Jump Intensity Sensitivity ---")
    for lam in [0.0, 0.5, 1.0, 2.0]:
        cf_test = merton_characteristic_function(1.0 + 0j, s0, t, r, sigma, lam, mu_j, sigma_j)
        print(f"lambda={lam:.1f}: |phi(1)| = {np.abs(cf_test):.6f}")

    print("\n" + "=" * 50)
    print("Merton CF smoke test passed")
    print("=" * 50)
