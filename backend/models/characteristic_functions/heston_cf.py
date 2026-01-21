"""
Heston Characteristic Function
==============================

Numba-optimized characteristic function for Heston (1993) model.

This is the SINGLE implementation used by:
    - HestonPricer (FFT pricing)
    - HestonModel.characteristic_function() (calibration)

Uses the Gatheral (2006) formulation for numerical stability.

Author: Thomas
Created: 2025
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def heston_characteristic_function(
    u: complex,
    s0: float,
    v0: float,
    t: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float
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
    xi : float
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
    i = 1j
    eps = 1e-10

    # Complex intermediate values
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

    # C coefficient (Gatheral formulation)
    C = (
        r * i * u * t +
        kappa * theta / (xi ** 2) * (
            numerator * t -
            2.0 * np.log(one_minus_g_exp / one_minus_g)
        )
    )

    # D coefficient
    D = (
        numerator / (xi ** 2) *
        ((1.0 - exp_dt) / one_minus_g_exp)
    )

    return np.exp(C + D * v0 + i * u * np.log(s0))


@njit(cache=True, fastmath=True, parallel=True)
def heston_cf_vectorized(
    u_arr: np.ndarray,
    s0: float,
    v0: float,
    t: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float
) -> np.ndarray:
    """
    Vectorized Heston characteristic function for FFT.

    Evaluates the characteristic function at multiple frequency points
    simultaneously for efficient FFT computation.

    Parameters
    ----------
    u_arr : np.ndarray
        Array of frequency arguments (complex)
    s0, v0, t, r, kappa, theta, xi, rho : float
        Model parameters

    Returns
    -------
    np.ndarray
        Array of characteristic function values
    """
    n = len(u_arr)
    result = np.empty(n, dtype=np.complex128)

    for i in range(n):
        result[i] = heston_characteristic_function(
            u_arr[i], s0, v0, t, r, kappa, theta, xi, rho
        )

    return result
