"""
Asset-or-nothing option pricing kernel.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.math import norm_cdf


@njit(fastmath=True, cache=True)
def asset_or_nothing_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool
) -> float:
    """
    Asset-or-nothing option price.

    Call: S * exp(-qT) * N(d1)
    Put:  S * exp(-qT) * N(-d1)

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to expiry
    r : float
        Risk-free rate
    q : float
        Continuous dividend yield
    sigma : float
        Volatility
    is_call : bool
        True for call, False for put

    Returns
    -------
    float
        Option price
    """
    if T <= 0:
        if is_call:
            return S if S > K else 0.0
        return S if S < K else 0.0

    if sigma <= 0:
        F = S * math.exp((r - q) * T)
        qd = math.exp(-q * T)
        if is_call:
            return (S * qd) if F > K else 0.0
        return (S * qd) if F < K else 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    qd = math.exp(-q * T)

    if is_call:
        return S * qd * norm_cdf(d1)
    return S * qd * norm_cdf(-d1)
