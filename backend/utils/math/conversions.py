"""
Discount/forward factors and moneyness conversions (Numba-compiled).
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.math.distributions import norm_inv_cdf


@njit(fastmath=True, cache=True)
def discount_factor(rate: float, time: float) -> float:
    """
    Calculate continuous discount factor.

    Parameters
    ----------
    rate : float
        Interest rate (annualized)
    time : float
        Time period in years

    Returns
    -------
    float
        exp(-rate * time)
    """
    return math.exp(-rate * time)


@njit(fastmath=True, cache=True)
def forward_price(
    spot: float, rate: float, dividend_yield: float, time: float
) -> float:
    """
    Calculate forward price.

    Parameters
    ----------
    spot : float
        Current spot price
    rate : float
        Risk-free rate
    dividend_yield : float
        Continuous dividend yield
    time : float
        Time to maturity

    Returns
    -------
    float
        Forward price = S * exp((r - q) * T)
    """
    return spot * math.exp((rate - dividend_yield) * time)


@njit(fastmath=True, cache=True)
def log_moneyness(spot: float, strike: float) -> float:
    """
    Calculate log-moneyness.

    Returns
    -------
    float
        ln(S/K)
    """
    return math.log(spot / strike)


@njit(fastmath=True, cache=True)
def forward_log_moneyness(
    spot: float, strike: float, rate: float, dividend_yield: float, time: float
) -> float:
    """
    Calculate forward log-moneyness.

    Returns
    -------
    float
        ln(F/K) where F = S * exp((r-q)*T)
    """
    fwd = forward_price(spot, rate, dividend_yield, time)
    return math.log(fwd / strike)


@njit(fastmath=True, cache=True)
def delta_to_strike(
    spot: float,
    delta: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    is_call: bool,
    dividend_yield: float = 0.0,
) -> float:
    """
    Convert delta to strike price.

    Parameters
    ----------
    spot : float
        Current spot price
    delta : float
        Target delta (positive for calls, negative for puts)
    time_to_expiry : float
        Time to maturity
    rate : float
        Risk-free rate
    volatility : float
        Volatility
    is_call : bool
        True for call, False for put
    dividend_yield : float
        Dividend yield

    Returns
    -------
    float
        Strike price corresponding to the given delta
    """
    sqrt_t = math.sqrt(time_to_expiry)
    forward_discount = math.exp(-dividend_yield * time_to_expiry)

    if is_call:
        d1 = norm_inv_cdf(delta / forward_discount)
    else:
        d1 = norm_inv_cdf((delta / forward_discount) + 1.0)

    fwd = forward_price(spot, rate, dividend_yield, time_to_expiry)
    strike = fwd * math.exp(
        -d1 * volatility * sqrt_t + 0.5 * volatility * volatility * time_to_expiry
    )

    return strike
