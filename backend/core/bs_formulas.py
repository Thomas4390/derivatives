"""
Black-Scholes Formulas - Single Source of Truth
================================================

Unified Black-Scholes formulas used across the entire backend.
All functions are Numba-optimized for performance.

This module is the CANONICAL source for:
- Normal distribution functions (CDF, PDF)
- d1/d2 parameters
- BS pricing formulas
- First-order Greeks

IMPORTANT: Do NOT duplicate these formulas elsewhere.
All other modules should import from here.

Author: Thomas
Created: 2025
"""

import math
import numpy as np
from numba import njit
from typing import Tuple


# =============================================================================
# Normal Distribution Functions
# =============================================================================

@njit(fastmath=True, cache=True)
def norm_cdf(x: float) -> float:
    """
    Cumulative distribution function for standard normal distribution.

    Parameters
    ----------
    x : float
        Value at which to evaluate the CDF

    Returns
    -------
    float
        P(X <= x) where X ~ N(0, 1)
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@njit(fastmath=True, cache=True)
def norm_pdf(x: float) -> float:
    """
    Probability density function for standard normal distribution.

    Parameters
    ----------
    x : float
        Value at which to evaluate the PDF

    Returns
    -------
    float
        Density at x for X ~ N(0, 1)
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# =============================================================================
# Black-Scholes d1/d2 Parameters
# =============================================================================

@njit(fastmath=True, cache=True)
def d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0
) -> Tuple[float, float]:
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
    rate : float
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
        forward = spot * math.exp((rate - dividend_yield) * time_to_expiry)
        if forward > strike:
            return 1e10, 1e10  # Deep ITM
        elif forward < strike:
            return -1e10, -1e10  # Deep OTM
        else:
            return 0.0, 0.0  # ATM

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike) +
        (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    return d1, d2


# =============================================================================
# Black-Scholes Pricing
# =============================================================================

@njit(fastmath=True, cache=True)
def bs_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    is_call: bool,
    dividend_yield: float = 0.0
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
        else:
            return max(strike - spot, 0.0)

    d1, d2 = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)

    discount = math.exp(-rate * time_to_expiry)
    forward_discount = math.exp(-dividend_yield * time_to_expiry)

    if is_call:
        price = (
            spot * forward_discount * norm_cdf(d1)
            - strike * discount * norm_cdf(d2)
        )
    else:
        price = (
            strike * discount * norm_cdf(-d2)
            - spot * forward_discount * norm_cdf(-d1)
        )

    return max(price, 0.0)


# =============================================================================
# First-Order Greeks
# =============================================================================

@njit(fastmath=True, cache=True)
def bs_delta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    is_call: bool,
    dividend_yield: float = 0.0
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
        else:
            return -1.0 if spot < strike else 0.0

    d1, _ = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)
    forward_discount = math.exp(-dividend_yield * time_to_expiry)

    if is_call:
        return forward_discount * norm_cdf(d1)
    else:
        return forward_discount * (norm_cdf(d1) - 1.0)


@njit(fastmath=True, cache=True)
def bs_gamma(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0
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
    dividend_yield: float = 0.0
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
    dividend_yield: float = 0.0
) -> float:
    """
    Calculate Black-Scholes theta.

    Returns
    -------
    float
        Theta (dV/dt) per day
    """
    DAYS_PER_YEAR = 365.0

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
    dividend_yield: float = 0.0
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


# =============================================================================
# Combined Greeks Calculation
# =============================================================================

@njit(fastmath=True, cache=True)
def bs_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    is_call: bool,
    dividend_yield: float = 0.0
) -> Tuple[float, float, float, float, float, float]:
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
    DAYS_PER_YEAR = 365.0

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
        price = bs_price(spot, strike, time_to_expiry, rate, volatility, is_call, dividend_yield)
        delta = bs_delta(spot, strike, time_to_expiry, rate, volatility, is_call, dividend_yield)
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
        price = strike * discount * norm_cdf(-d2) - spot * forward_discount * norm_cdf(-d1)
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


# =============================================================================
# Vectorized Versions
# =============================================================================

@njit(fastmath=True, cache=True, parallel=True)
def norm_cdf_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized normal CDF."""
    result = np.empty_like(x)
    for i in range(len(x)):
        result[i] = 0.5 * (1.0 + math.erf(x[i] / math.sqrt(2.0)))
    return result


@njit(fastmath=True, cache=True, parallel=True)
def norm_pdf_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized normal PDF."""
    result = np.empty_like(x)
    sqrt_2pi = math.sqrt(2.0 * math.pi)
    for i in range(len(x)):
        result[i] = math.exp(-0.5 * x[i] * x[i]) / sqrt_2pi
    return result
