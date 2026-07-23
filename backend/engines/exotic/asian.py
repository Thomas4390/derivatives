"""
Asian geometric option pricing kernel (Kemna-Vorst 1990).

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.math import norm_cdf


@njit(fastmath=True, cache=True)
def asian_geometric_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool
) -> float:
    """
    Geometric Asian option price (Kemna-Vorst 1990).

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
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)

    if sigma <= 0:
        b = r - q
        F = S * math.exp(b * T)
        df = math.exp(-r * T)
        return max(F - K, 0.0) * df if is_call else max(K - F, 0.0) * df

    # Adjusted parameters for geometric average (Kemna-Vorst 1990)
    # sigma_adj = sigma/sqrt(3), b_adj = (r-q)/2 - sigma^2/12
    sigma_adj = sigma / math.sqrt(3.0)
    b = 0.5 * (r - q - sigma * sigma / 6.0)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (b + 0.5 * sigma_adj * sigma_adj) * T) / (
        sigma_adj * sqrt_T
    )
    d2 = d1 - sigma_adj * sqrt_T

    if is_call:
        price = S * math.exp((b - r) * T) * norm_cdf(d1) - K * math.exp(
            -r * T
        ) * norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(
            (b - r) * T
        ) * norm_cdf(-d1)

    return max(price, 0.0)
