"""
Black-Scholes d1/d2 parameters and option price (Numba-compiled).
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.math.distributions import norm_cdf


@njit(fastmath=True, cache=True)
def d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """
    Calculate d1 and d2 parameters for Black-Scholes model.

    Parameters
    ----------
    spot : float
        Current price of underlying asset
    strike : float
        Strike price of the option
    time_to_expiry : float
        Time to expiration in years
    risk_free_rate : float
        Risk-free interest rate (annualized)
    volatility : float
        Implied volatility (annualized)
    dividend_yield : float, optional
        Continuous dividend yield (default 0.0)

    Returns
    -------
    Tuple[float, float]
        (d1, d2) parameters

    Notes
    -----
    d1 = [ln(S/K) + (r - q + sigma^2/2)T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Edge cases:
    - time_to_expiry <= 0: returns (0.0, 0.0)
    - volatility <= 0: returns large values based on moneyness
    """
    if time_to_expiry <= 0:
        return 0.0, 0.0

    if volatility <= 0:
        # Handle zero volatility case based on moneyness
        forward = spot * math.exp((risk_free_rate - dividend_yield) * time_to_expiry)
        if forward > strike:
            return 1e10, 1e10  # Deep ITM
        if forward < strike:
            return -1e10, -1e10  # Deep OTM
        return 0.0, 0.0  # ATM

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility * volatility)
        * time_to_expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    return d1, d2


@njit(fastmath=True, cache=True)
def bs_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    is_call: bool,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Black-Scholes option price.

    Parameters
    ----------
    spot : float
        Current price of underlying asset
    strike : float
        Strike price of the option
    time_to_expiry : float
        Time to expiration in years
    rate : float
        Risk-free interest rate (annualized)
    volatility : float
        Implied volatility (annualized)
    is_call : bool
        True for call, False for put
    dividend_yield : float, optional
        Continuous dividend yield (default 0.0)

    Returns
    -------
    float
        Option price
    """
    if time_to_expiry <= 0:
        # At expiry - intrinsic value
        if is_call:
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    d1, d2 = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)

    discount = math.exp(-rate * time_to_expiry)
    forward_discount = math.exp(-dividend_yield * time_to_expiry)

    if is_call:
        price = spot * forward_discount * norm_cdf(d1) - strike * discount * norm_cdf(
            d2
        )
    else:
        price = strike * discount * norm_cdf(-d2) - spot * forward_discount * norm_cdf(
            -d1
        )

    return max(price, 0.0)
