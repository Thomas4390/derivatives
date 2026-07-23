"""
Second- and third-order Black-Scholes Greeks (Numba-compiled).
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.constants.time import DAYS_PER_YEAR
from backend.utils.math.black_scholes import d1_d2
from backend.utils.math.distributions import norm_cdf, norm_pdf


@njit(fastmath=True, cache=True)
def bs_second_order_greeks(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
    dividend_yield: float = 0.0,
    is_call: bool = True,
) -> tuple[float, float, float, float]:
    """
    Calculate second-order Greeks for Black-Scholes.

    Parameters
    ----------
    spot : float
        Spot price
    strike : float
        Strike price
    t : float
        Time to expiry in years
    r : float
        Risk-free rate
    sigma : float
        Volatility
    dividend_yield : float
        Continuous dividend yield (default 0.0)
    is_call : bool
        True for a call, False for a put. Only affects ``charm`` and only
        when ``dividend_yield != 0`` (the dividend term differs by option
        type); vanna/volga/veta are call/put independent. Default True.

    Returns
    -------
    Tuple[float, float, float, float]
        (vanna, volga, charm, veta)
        - vanna: dDelta/dVol (per 1% vol change)
        - volga: dVega/dVol (per 1% vol change squared)
        - charm: dDelta/dt (per calendar day)
        - veta: dVega/dt (per calendar day per 1% vol)
    """
    if t <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0

    sqrt_t = math.sqrt(t)
    d1, d2 = d1_d2(spot, strike, t, r, sigma, dividend_yield)
    n_prime_d1 = norm_pdf(d1)
    forward_discount = math.exp(-dividend_yield * t)

    # Vanna - d²V/dSdσ (per 1% vol change)
    vanna = -forward_discount * n_prime_d1 * d2 / sigma / 100.0

    # Volga/Vomma - d²V/dσ² (per 1% vol change squared)
    vega_base = spot * forward_discount * n_prime_d1 * sqrt_t
    volga = vega_base * d1 * d2 / sigma / 10000.0

    # Charm - d²V/dSdt (per calendar day). Call/put-dependent via the
    # dividend term: it vanishes only when q == 0 (then call == put).
    charm_symmetric = (
        -forward_discount
        * n_prime_d1
        * (2 * (r - dividend_yield) * t - d2 * sigma * sqrt_t)
        / (2 * t * sigma * sqrt_t)
    )
    if is_call:
        charm = (
            charm_symmetric + dividend_yield * forward_discount * norm_cdf(d1)
        ) / DAYS_PER_YEAR
    else:
        charm = (
            charm_symmetric - dividend_yield * forward_discount * norm_cdf(-d1)
        ) / DAYS_PER_YEAR

    # Veta - d²V/dσdt (per calendar day per 1% vol). Bracket carries the
    # dividend-yield term so it stays correct when q != 0.
    veta = (
        spot
        * forward_discount
        * n_prime_d1
        * sqrt_t
        * (
            dividend_yield
            + (r - dividend_yield) * d1 / (sigma * sqrt_t)
            - (1 + d1 * d2) / (2 * t)
        )
        / (DAYS_PER_YEAR * 100.0)
    )

    return vanna, volga, charm, veta


@njit(fastmath=True, cache=True)
def bs_third_order_greeks(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Calculate third-order Greeks for Black-Scholes.

    Parameters
    ----------
    spot : float
        Spot price
    strike : float
        Strike price
    t : float
        Time to expiry in years
    r : float
        Risk-free rate
    sigma : float
        Volatility
    dividend_yield : float
        Continuous dividend yield (default 0.0)

    Returns
    -------
    Tuple[float, float, float, float]
        (speed, zomma, color, ultima)
        - speed: dGamma/dSpot
        - zomma: dGamma/dVol (per 1% vol change)
        - color: dGamma/dt (per day)
        - ultima: dVomma/dVol (per 1% vol change cubed)
    """
    if t <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0

    sqrt_t = math.sqrt(t)
    d1, d2 = d1_d2(spot, strike, t, r, sigma, dividend_yield)
    n_prime_d1 = norm_pdf(d1)
    forward_discount = math.exp(-dividend_yield * t)

    # Gamma for speed calculation
    gamma = forward_discount * n_prime_d1 / (spot * sigma * sqrt_t)

    # Speed - d³V/dS³
    speed = -gamma * (d1 / (sigma * sqrt_t) + 1) / spot

    # Zomma - d³V/dS²dσ (per 1% vol change)
    zomma = gamma * (d1 * d2 - 1) / sigma / 100.0

    # Color - d³V/dS²dt (per calendar day). Canonical Haug bracket
    # (2qt + 1 + ...) with the calendar-time sign so it matches the
    # finite-difference convention used by greeks/numerical.py (= -dGamma/dT).
    color = (
        forward_discount
        * n_prime_d1
        / (2 * spot * t * sigma * sqrt_t)
        * (
            2 * dividend_yield * t
            + 1
            + d1
            * (2 * (r - dividend_yield) * t - d2 * sigma * sqrt_t)
            / (sigma * sqrt_t)
        )
        / DAYS_PER_YEAR
    )

    # Ultima - d³V/dσ³ (per 1% vol change cubed)
    vega = spot * forward_discount * n_prime_d1 * sqrt_t
    ultima = (
        -vega / (sigma**3) * (d1 * d2 * (1 - d1 * d2) + d1 * d1 + d2 * d2) / 1000000.0
    )

    return speed, zomma, color, ultima
