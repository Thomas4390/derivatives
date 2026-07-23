"""
Lookback option pricing kernels (Goldman-Sosin-Gatto 1979, Conze-Viswanathan 1991).

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.math import norm_cdf


@njit(fastmath=True, cache=True)
def lookback_floating_price(
    S: float,
    M_min: float,
    M_max: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
) -> float:
    """
    Floating strike lookback option price (Goldman-Sosin-Gatto 1979).

    Payoff:
    - Call: S_T - M_min (buy at the low)
    - Put: M_max - S_T (sell at the high)

    Parameters
    ----------
    S : float
        Spot price
    M_min : float
        Running minimum of spot
    M_max : float
        Running maximum of spot
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
            return max(S - M_min, 0.0)
        return max(M_max - S, 0.0)

    if sigma <= 0:
        # Deterministic forward: path is monotone
        b = r - q
        F = S * math.exp(b * T)
        df = math.exp(-r * T)
        if is_call:
            effective_min = min(M_min, min(S, F))
            return max(F - effective_min, 0.0) * df
        effective_max = max(M_max, max(S, F))
        return max(effective_max - F, 0.0) * df

    b = r - q
    # Guard against b ~ 0 (division by zero in sigma^2/(2b) terms)
    b_eff = b if abs(b) > 1e-10 else 1e-10

    sqrt_T = math.sqrt(T)
    M = M_min if is_call else M_max

    qd = math.exp(-q * T)
    df = math.exp(-r * T)

    b1 = (math.log(S / M) + (b_eff + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    b2 = b1 - sigma * sqrt_T

    two_b_over_sigma_sq = 2.0 * b_eff / (sigma * sigma)

    # Goldman-Sosin-Gatto / fExoticOptions formulation
    if is_call:
        term1 = S * qd * norm_cdf(b1) - M * df * norm_cdf(b2)
        term2 = (
            df
            * (sigma * sigma)
            / (2.0 * b_eff)
            * S
            * (
                (S / M) ** (-two_b_over_sigma_sq)
                * norm_cdf(-b1 + 2.0 * b_eff * sqrt_T / sigma)
                - math.exp(b_eff * T) * norm_cdf(-b1)
            )
        )
        price = term1 + term2
    else:
        term1 = M * df * norm_cdf(-b2) - S * qd * norm_cdf(-b1)
        term2 = (
            df
            * (sigma * sigma)
            / (2.0 * b_eff)
            * S
            * (
                -((S / M) ** (-two_b_over_sigma_sq))
                * norm_cdf(b1 - 2.0 * b_eff * sqrt_T / sigma)
                + math.exp(b_eff * T) * norm_cdf(b1)
            )
        )
        price = term1 + term2

    return max(price, 0.0)


@njit(fastmath=True, cache=True)
def lookback_fixed_price(
    S: float,
    K: float,
    M_min: float,
    M_max: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
) -> float:
    """
    Fixed strike lookback option price via decomposition into floating lookbacks.

    Uses the identity (Conze-Viswanathan 1991):
    - Call: fixed_call = floating_put + S - K*df  (when M_max >= K)
    - Put:  fixed_put  = K*df - S + floating_call (when K >= M_min)

    Payoff:
    - Call: max(M_max - K, 0)
    - Put: max(K - M_min, 0)

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    M_min : float
        Running minimum of spot
    M_max : float
        Running maximum of spot
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
            return max(M_max - K, 0.0)
        return max(K - M_min, 0.0)

    if sigma <= 0:
        # Deterministic forward: path is monotone, so max/min is at endpoints
        b = r - q
        F = S * math.exp(b * T)
        df = math.exp(-r * T)
        if is_call:
            path_max = max(S, F)
            effective_max = max(M_max, path_max)
            return max(effective_max - K, 0.0) * df
        path_min = min(S, F)
        effective_min = min(M_min, path_min)
        return max(K - effective_min, 0.0) * df

    df = math.exp(-r * T)

    if is_call:
        # Conze-Viswanathan: M must be max(M_max, S, K) so the
        # decomposition fixed_call = floating_put + S - K*df is exact.
        M = M_max
        if S > M:
            M = S
        if K > M:
            M = K
        float_put = lookback_floating_price(S, M_min, M, T, r, q, sigma, False)
        price = float_put + S - K * df
    else:
        # Conze-Viswanathan: M must be min(M_min, S, K) so the
        # decomposition fixed_put = K*df - S + floating_call is exact.
        M = M_min
        if S < M:
            M = S
        if K < M:
            M = K
        float_call = lookback_floating_price(S, M, M_max, T, r, q, sigma, True)
        price = K * df - S + float_call

    return max(price, 0.0)
