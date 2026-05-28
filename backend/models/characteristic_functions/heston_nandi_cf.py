"""
Heston-Nandi GARCH Characteristic Function
==========================================

Numba-optimized closed-form characteristic function for the Heston & Nandi
(2000) discrete-time GARCH option-pricing model. This is the SINGLE NumPy
implementation used by:
    - ``FFTEngine`` via ``HestonNandiGARCHModel.characteristic_function``
      (reference surface pricing + calibration RMSE).

The JAX twin (spot-agnostic, for the analytical Jacobian) lives in
``backend/engines/aad/calibration/heston_nandi_cf.py``.

Model (risk-neutral, per-period step, lambda* = -1/2):

    R_t = r_step - 0.5 h_t + sqrt(h_t) z_t,  z_t ~ N(0,1)
    h_{t+1} = omega + beta h_t + alpha (z_t - gamma sqrt(h_t))^2

The log-PRICE CF ``phi(u) = E^Q[exp(i u ln S_T)] = exp(i u ln S_0 + A_t + B_t h0)``
with ``phi = i u`` and the backward recursion over ``N = round(t * steps_per_year)``
steps from ``A_T = B_T = 0``:

    denom = 1 - 2 alpha B
    B_new = phi (gamma - 1/2) - 1/2 gamma^2 + beta B + 1/2 (phi - gamma)^2 / denom
    A_new = A + phi r_step + omega B - 1/2 ln(denom)

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def heston_nandi_characteristic_function(
    u: complex,
    s0: float,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    h0: float,
    t: float,
    r: float,
    steps_per_year: int,
) -> complex:
    """Heston-Nandi log-price CF ``phi(u) = E^Q[exp(i u ln S_T)]``.

    Parameters
    ----------
    u : complex
        Frequency argument.
    s0 : float
        Initial spot price (dividend-adjusted by the caller).
    omega, alpha, beta, gamma : float
        Risk-neutral GARCH(1,1) parameters (per period).
    h0 : float
        Initial conditional variance ``h_1`` (per period).
    t : float
        Time to maturity in years (``N = round(t * steps_per_year)`` steps).
    r : float
        Annual risk-free rate (per-step ``r / steps_per_year``).
    steps_per_year : int
        Trading-day discretization (e.g. 252).

    Returns
    -------
    complex
        Characteristic function value ``phi(u)``.
    """
    n_steps = int(round(t * steps_per_year))
    if n_steps < 1:
        n_steps = 1
    r_step = r / steps_per_year

    phi = 1j * u
    a = 0.0 + 0.0j
    b = 0.0 + 0.0j
    for _ in range(n_steps):
        denom = 1.0 - 2.0 * alpha * b
        b_new = (
            phi * (gamma - 0.5)
            - 0.5 * gamma * gamma
            + beta * b
            + 0.5 * (phi - gamma) ** 2 / denom
        )
        a_new = a + phi * r_step + omega * b - 0.5 * np.log(denom)
        a = a_new
        b = b_new

    return np.exp(1j * u * np.log(s0) + a + b * h0)


@njit(cache=True, fastmath=True, parallel=True)
def heston_nandi_cf_vectorized(
    u_arr: np.ndarray,
    s0: float,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    h0: float,
    t: float,
    r: float,
    steps_per_year: int,
) -> np.ndarray:
    """Vectorized Heston-Nandi characteristic function for FFT pricing."""
    n = len(u_arr)
    result = np.empty(n, dtype=np.complex128)
    for i in prange(n):
        result[i] = heston_nandi_characteristic_function(
            u_arr[i], s0, omega, alpha, beta, gamma, h0, t, r, steps_per_year
        )
    return result


if __name__ == "__main__":
    print("=" * 56)
    print("Heston-Nandi GARCH CF (Numba) Smoke Test")
    print("=" * 56)

    s0, r, spy = 100.0, 0.05, 252
    omega, alpha, beta, gamma, h0 = 1.0e-6, 2.0e-6, 0.80, 150.0, 4.0e-5
    t = 126 / spy  # ~0.5y, integer-step

    # phi(0) = exp(i*0*ln s0 + 0) = 1
    cf0 = heston_nandi_characteristic_function(
        0.0 + 0.0j, s0, omega, alpha, beta, gamma, h0, t, r, spy
    )
    print(f"\nphi(0)  = {cf0:.10f}  (expect 1)")
    assert abs(cf0 - 1.0) < 1e-9

    # Martingale: phi(-i) = E^Q[S_T] = s0 * e^{r t}
    cf_m = heston_nandi_characteristic_function(
        -1.0j, s0, omega, alpha, beta, gamma, h0, t, r, spy
    )
    target = s0 * np.exp(r * t)
    print(f"phi(-i) = {cf_m:.6f}  (expect s0·e^(r·t) = {target:.6f})")
    assert abs(cf_m - target) < 1e-6 * s0

    # Vectorized matches scalar
    u_arr = np.array([0.5, 1.0, 1.5, 2.0]) + 0.5j
    cf_vec = heston_nandi_cf_vectorized(
        u_arr, s0, omega, alpha, beta, gamma, h0, t, r, spy
    )
    for i, ui in enumerate(u_arr):
        cf_s = heston_nandi_characteristic_function(
            ui, s0, omega, alpha, beta, gamma, h0, t, r, spy
        )
        assert abs(cf_vec[i] - cf_s) < 1e-10
    print("Vectorized matches scalar: OK")

    print("\nAll Numba CF smoke tests passed")
