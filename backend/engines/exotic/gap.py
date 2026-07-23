"""
Gap option pricing kernel.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.math import norm_cdf


@njit(fastmath=True, cache=True)
def gap_option_price(
    S: float,
    K1: float,
    K2: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
) -> float:
    """
    Gap option price.

    K1 = payment strike, K2 = trigger strike.
    Call: (S_T - K1) if S_T > K2, else 0.
    Put:  (K1 - S_T) if S_T < K2, else 0.

    d1 = [ln(S/K2) + (r-q+0.5*sigma^2)*T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    Call = S*exp(-qT)*N(d1) - K1*exp(-rT)*N(d2)

    Note: Gap option value can be negative (when K1 > K2).

    Parameters
    ----------
    S : float
        Spot price
    K1 : float
        Payment strike
    K2 : float
        Trigger strike
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
        Option price (can be negative)
    """
    if T <= 0:
        if is_call:
            return (S - K1) if S > K2 else 0.0
        return (K1 - S) if S < K2 else 0.0

    if sigma <= 0:
        F = S * math.exp((r - q) * T)
        qd = math.exp(-q * T)
        df = math.exp(-r * T)
        if is_call:
            return (S * qd - K1 * df) if F > K2 else 0.0
        return (K1 * df - S * qd) if F < K2 else 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K2) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    qd = math.exp(-q * T)
    df = math.exp(-r * T)

    if is_call:
        return S * qd * norm_cdf(d1) - K1 * df * norm_cdf(d2)
    return K1 * df * norm_cdf(-d2) - S * qd * norm_cdf(-d1)
