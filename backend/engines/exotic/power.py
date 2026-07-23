"""
Power option pricing kernel.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.math import norm_cdf


@njit(fastmath=True, cache=True)
def power_option_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
    n: float,
) -> float:
    """
    Power option price.

    Option on S^n: payoff = max(S_T^n - K, 0) for calls.
    mu_adj = n*(r-q) + n*(n-1)*sigma^2/2
    sigma_adj = n * sigma

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
    n : float
        Power exponent

    Returns
    -------
    float
        Option price
    """
    if n <= 0:
        # S^0 = 1 for all S, payoff = max(1-K, 0) for call
        df = math.exp(-r * T) if T > 0 else 1.0
        if is_call:
            return max(1.0 - K, 0.0) * df
        return max(K - 1.0, 0.0) * df

    if T <= 0:
        S_n = S**n
        if is_call:
            return max(S_n - K, 0.0)
        return max(K - S_n, 0.0)

    S_n = S**n
    sigma_adj = n * sigma

    if sigma <= 0 or sigma_adj <= 0:
        mu_adj = n * (r - q)
        F_n = S_n * math.exp(mu_adj * T)
        df = math.exp(-r * T)
        if is_call:
            return max(F_n - K, 0.0) * df
        return max(K - F_n, 0.0) * df

    mu_adj = n * (r - q) + 0.5 * n * (n - 1.0) * sigma * sigma

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S_n / K) + (mu_adj + 0.5 * sigma_adj * sigma_adj) * T) / (
        sigma_adj * sqrt_T
    )
    d2 = d1 - sigma_adj * sqrt_T

    df = math.exp(-r * T)

    if is_call:
        price = S_n * math.exp((mu_adj - r) * T) * norm_cdf(d1) - K * df * norm_cdf(d2)
    else:
        price = K * df * norm_cdf(-d2) - S_n * math.exp((mu_adj - r) * T) * norm_cdf(
            -d1
        )

    return max(price, 0.0)
