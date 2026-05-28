"""
Heston Characteristic-Function Core
====================================

Shared Gatheral-formulation kernel used by both the Heston CF and the
Bates CF (Bates = Heston stochastic vol + compound Poisson jumps).

Only the drift rate differs between the two callers:
    * Heston:  drift_rate = r
    * Bates :  drift_rate = r - lam * k   (with k = E[J-1])

The jump factor for Bates is multiplied in by the caller.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def heston_cf_with_drift(
    u: complex,
    s0: float,
    v0: float,
    t: float,
    drift_rate: float,
    kappa: float,
    theta: float,
    alpha: float,
    rho: float,
) -> complex:
    """Heston/Gatheral characteristic function for a given drift rate.

    Returns ``exp(C(u, t) + D(u, t) * v0 + i * u * ln(s0))`` where the
    drift contribution enters as ``drift_rate * i * u * t`` inside ``C``.
    """
    i = 1j
    eps = 1e-10

    d = np.sqrt((rho * alpha * i * u - kappa) ** 2 + alpha**2 * (i * u + u**2))

    numerator = kappa - rho * alpha * i * u - d
    denominator = kappa - rho * alpha * i * u + d

    if np.abs(denominator) < eps:
        denominator = eps + 0j

    g = numerator / denominator

    exp_dt = np.exp(-d * t)

    one_minus_g = 1.0 - g
    one_minus_g_exp = 1.0 - g * exp_dt

    if np.abs(one_minus_g) < eps:
        one_minus_g = eps + 0j
    if np.abs(one_minus_g_exp) < eps:
        one_minus_g_exp = eps + 0j

    C = drift_rate * i * u * t + kappa * theta / (alpha**2) * (
        numerator * t - 2.0 * np.log(one_minus_g_exp / one_minus_g)
    )

    D = numerator / (alpha**2) * ((1.0 - exp_dt) / one_minus_g_exp)

    return np.exp(C + D * v0 + i * u * np.log(s0))
