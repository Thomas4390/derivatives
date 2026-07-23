"""
First-order Black-Scholes Greeks and the fused ``bs_greeks``.
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.constants.time import DAYS_PER_YEAR
from backend.utils.math.black_scholes import bs_price, d1_d2
from backend.utils.math.distributions import norm_cdf, norm_pdf


@njit(fastmath=True, cache=True)
def bs_delta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    is_call: bool,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Black-Scholes delta.

    Returns
    -------
    float
        Delta (dV/dS)
    """
    if time_to_expiry <= 0 or volatility <= 0:
        if is_call:
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0

    d1, _ = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)
    forward_discount = math.exp(-dividend_yield * time_to_expiry)

    if is_call:
        return forward_discount * norm_cdf(d1)
    return forward_discount * (norm_cdf(d1) - 1.0)


@njit(fastmath=True, cache=True)
def bs_gamma(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Black-Scholes gamma.

    Returns
    -------
    float
        Gamma (d²V/dS²)
    """
    if time_to_expiry <= 0 or volatility <= 0:
        return 0.0

    d1, _ = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)
    forward_discount = math.exp(-dividend_yield * time_to_expiry)
    sqrt_t = math.sqrt(time_to_expiry)

    return forward_discount * norm_pdf(d1) / (spot * volatility * sqrt_t)


@njit(fastmath=True, cache=True)
def bs_vega(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Black-Scholes vega.

    Returns
    -------
    float
        Vega (dV/d_sigma) per 1% vol change
    """
    if time_to_expiry <= 0 or volatility <= 0:
        return 0.0

    d1, _ = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)
    forward_discount = math.exp(-dividend_yield * time_to_expiry)
    sqrt_t = math.sqrt(time_to_expiry)

    # Per 1% vol change
    return spot * forward_discount * norm_pdf(d1) * sqrt_t / 100.0


@njit(fastmath=True, cache=True)
def bs_theta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    is_call: bool,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Black-Scholes theta.

    Returns
    -------
    float
        Theta (dV/dt) per day
    """
    if time_to_expiry <= 0 or volatility <= 0:
        return 0.0

    d1, d2 = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)
    sqrt_t = math.sqrt(time_to_expiry)

    discount = math.exp(-rate * time_to_expiry)
    forward_discount = math.exp(-dividend_yield * time_to_expiry)

    # Time decay component
    time_decay = -(spot * forward_discount * norm_pdf(d1) * volatility) / (2 * sqrt_t)

    if is_call:
        theta = (
            time_decay
            + dividend_yield * spot * forward_discount * norm_cdf(d1)
            - rate * strike * discount * norm_cdf(d2)
        )
    else:
        theta = (
            time_decay
            - dividend_yield * spot * forward_discount * norm_cdf(-d1)
            + rate * strike * discount * norm_cdf(-d2)
        )

    # Convert to per day
    return theta / DAYS_PER_YEAR


@njit(fastmath=True, cache=True)
def bs_rho(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    is_call: bool,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Black-Scholes rho.

    Returns
    -------
    float
        Rho (dV/dr) per 1% rate change
    """
    if time_to_expiry <= 0 or volatility <= 0:
        return 0.0

    _, d2 = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)
    discount = math.exp(-rate * time_to_expiry)

    if is_call:
        rho = strike * time_to_expiry * discount * norm_cdf(d2)
    else:
        rho = -strike * time_to_expiry * discount * norm_cdf(-d2)

    # Per 1% rate change
    return rho / 100.0


@njit(fastmath=True, cache=True)
def bs_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    is_call: bool,
    dividend_yield: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    """
    Calculate all first-order Black-Scholes Greeks in one call.

    Returns
    -------
    Tuple[float, float, float, float, float, float]
        (price, delta, gamma, vega, theta, rho)
        - vega: per 1% vol change
        - theta: per day
        - rho: per 1% rate change
    """
    if time_to_expiry <= 0:
        # At expiry
        if is_call:
            intrinsic = max(spot - strike, 0.0)
            delta = 1.0 if spot > strike else 0.0
        else:
            intrinsic = max(strike - spot, 0.0)
            delta = -1.0 if spot < strike else 0.0
        return intrinsic, delta, 0.0, 0.0, 0.0, 0.0

    if volatility <= 0:
        price = bs_price(
            spot, strike, time_to_expiry, rate, volatility, is_call, dividend_yield
        )
        delta = bs_delta(
            spot, strike, time_to_expiry, rate, volatility, is_call, dividend_yield
        )
        return price, delta, 0.0, 0.0, 0.0, 0.0

    # Calculate d1, d2 once
    d1, d2 = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)
    sqrt_t = math.sqrt(time_to_expiry)

    discount = math.exp(-rate * time_to_expiry)
    forward_discount = math.exp(-dividend_yield * time_to_expiry)

    n_d1 = norm_cdf(d1)
    n_d2 = norm_cdf(d2)
    n_prime_d1 = norm_pdf(d1)

    # Price
    if is_call:
        price = spot * forward_discount * n_d1 - strike * discount * n_d2
    else:
        price = strike * discount * norm_cdf(-d2) - spot * forward_discount * norm_cdf(
            -d1
        )
    price = max(price, 0.0)

    # Delta
    if is_call:
        delta = forward_discount * n_d1
    else:
        delta = forward_discount * (n_d1 - 1.0)

    # Gamma (same for call and put)
    gamma = forward_discount * n_prime_d1 / (spot * volatility * sqrt_t)

    # Vega (same for call and put, per 1%)
    vega = spot * forward_discount * n_prime_d1 * sqrt_t / 100.0

    # Theta (per day)
    time_decay = -(spot * forward_discount * n_prime_d1 * volatility) / (2 * sqrt_t)
    if is_call:
        theta = (
            time_decay
            + dividend_yield * spot * forward_discount * n_d1
            - rate * strike * discount * n_d2
        )
    else:
        theta = (
            time_decay
            - dividend_yield * spot * forward_discount * norm_cdf(-d1)
            + rate * strike * discount * norm_cdf(-d2)
        )
    theta = theta / DAYS_PER_YEAR

    # Rho (per 1%)
    if is_call:
        rho = strike * time_to_expiry * discount * n_d2 / 100.0
    else:
        rho = -strike * time_to_expiry * discount * norm_cdf(-d2) / 100.0

    return price, delta, gamma, vega, theta, rho
