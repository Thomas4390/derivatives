"""
Mathematical Utilities
======================

Shared mathematical primitives used across the backend.
All functions are Numba-optimized for performance.

This module is the SINGLE SOURCE OF TRUTH for:
- Normal distribution functions (CDF, PDF)
- Black-Scholes d1/d2 parameters
- Black-Scholes pricing formulas
- First/Second/Third-order Greeks

IMPORTANT: Do NOT duplicate these formulas elsewhere.
All other modules should import from here.

Author: Thomas
Created: 2025
"""

import math
import numpy as np
from numba import njit, prange
from typing import Tuple


# =============================================================================
# Constants
# =============================================================================

DAYS_PER_YEAR = 365.0
SQRT_2PI = math.sqrt(2.0 * math.pi)
SQRT_2 = math.sqrt(2.0)


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
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


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
    return math.exp(-0.5 * x * x) / SQRT_2PI


@njit(fastmath=True, cache=True)
def norm_inv_cdf(p: float) -> float:
    """
    Inverse cumulative distribution function (quantile function) for standard normal.

    Uses rational approximation (Abramowitz and Stegun approximation).

    Parameters
    ----------
    p : float
        Probability (0 < p < 1)

    Returns
    -------
    float
        x such that P(X <= x) = p where X ~ N(0, 1)
    """
    if p <= 0.0:
        return -1e10
    if p >= 1.0:
        return 1e10

    # Rational approximation constants
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        # Lower tail
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
               ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    elif p <= p_high:
        # Central region
        q = p - 0.5
        r = q * q
        return (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q / \
               (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
    else:
        # Upper tail
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)


# =============================================================================
# Vectorized Normal Distribution Functions
# =============================================================================

@njit(fastmath=True, cache=True, parallel=True)
def norm_cdf_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized normal CDF."""
    result = np.empty_like(x)
    for i in prange(len(x)):
        result[i] = 0.5 * (1.0 + math.erf(x[i] / SQRT_2))
    return result


@njit(fastmath=True, cache=True, parallel=True)
def norm_pdf_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized normal PDF."""
    result = np.empty_like(x)
    for i in prange(len(x)):
        result[i] = math.exp(-0.5 * x[i] * x[i]) / SQRT_2PI
    return result


# =============================================================================
# Black-Scholes d1/d2 Parameters
# =============================================================================

@njit(fastmath=True, cache=True)
def d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
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
        elif forward < strike:
            return -1e10, -1e10  # Deep OTM
        else:
            return 0.0, 0.0  # ATM

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike) +
        (risk_free_rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry
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
# Combined First-Order Greeks
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
# Second-Order Greeks
# =============================================================================

@njit(fastmath=True, cache=True)
def bs_second_order_greeks(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
    dividend_yield: float = 0.0
) -> Tuple[float, float, float, float]:
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

    Returns
    -------
    Tuple[float, float, float, float]
        (vanna, volga, charm, veta)
        - vanna: dDelta/dVol (per 1% vol change)
        - volga: dVega/dVol (per 1% vol change squared)
        - charm: dDelta/dt (per day)
        - veta: dVega/dt (per day per 1% vol)
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

    # Charm - d²V/dSdt (per day)
    charm = -forward_discount * n_prime_d1 * (
        2 * (r - dividend_yield) * t - d2 * sigma * sqrt_t
    ) / (2 * t * sigma * sqrt_t) / DAYS_PER_YEAR

    # Veta - d²V/dσdt (per day per 1% vol)
    veta = spot * forward_discount * n_prime_d1 * sqrt_t * (
        (r - dividend_yield) * d1 / (sigma * sqrt_t) - (1 + d1 * d2) / (2 * t)
    ) / (DAYS_PER_YEAR * 100.0)

    return vanna, volga, charm, veta


# =============================================================================
# Third-Order Greeks
# =============================================================================

@njit(fastmath=True, cache=True)
def bs_third_order_greeks(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
    dividend_yield: float = 0.0
) -> Tuple[float, float, float, float]:
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

    # Color - d³V/dS²dt (per day)
    color = -forward_discount * n_prime_d1 / (2 * spot * t * sigma * sqrt_t) * (
        2 * (r - dividend_yield) * t - 1 +
        d1 * (2 * (r - dividend_yield) * t - d2 * sigma * sqrt_t) / (sigma * sqrt_t)
    ) / DAYS_PER_YEAR

    # Ultima - d³V/dσ³ (per 1% vol change cubed)
    vega = spot * forward_discount * n_prime_d1 * sqrt_t
    ultima = -vega / (sigma ** 3) * (
        d1 * d2 * (1 - d1 * d2) + d1 * d1 + d2 * d2
    ) / 1000000.0

    return speed, zomma, color, ultima


# =============================================================================
# Implied Volatility (Newton-Raphson)
# =============================================================================

@njit(fastmath=True, cache=True)
def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    is_call: bool,
    dividend_yield: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.

    Parameters
    ----------
    price : float
        Market price of the option
    spot : float
        Current spot price
    strike : float
        Strike price
    time_to_expiry : float
        Time to expiration in years
    rate : float
        Risk-free rate
    is_call : bool
        True for call, False for put
    dividend_yield : float
        Continuous dividend yield
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations

    Returns
    -------
    float
        Implied volatility (returns -1.0 if no convergence)
    """
    if time_to_expiry <= 0:
        return -1.0

    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = math.sqrt(2 * math.pi / time_to_expiry) * price / spot

    # Bound the initial guess
    sigma = max(0.001, min(sigma, 5.0))

    for _ in range(max_iter):
        bs_p = bs_price(spot, strike, time_to_expiry, rate, sigma, is_call, dividend_yield)
        diff = bs_p - price

        if abs(diff) < tol:
            return sigma

        # Vega for Newton step (unscaled)
        d1, _ = d1_d2(spot, strike, time_to_expiry, rate, sigma, dividend_yield)
        vega_raw = spot * math.exp(-dividend_yield * time_to_expiry) * norm_pdf(d1) * math.sqrt(time_to_expiry)

        if vega_raw < 1e-10:
            # Vega too small, bisection fallback
            if diff > 0:
                sigma *= 0.9
            else:
                sigma *= 1.1
        else:
            sigma = sigma - diff / vega_raw

        # Keep sigma in reasonable bounds
        sigma = max(0.001, min(sigma, 5.0))

    return -1.0  # No convergence


# =============================================================================
# Discount Factor Utilities
# =============================================================================

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
    spot: float,
    rate: float,
    dividend_yield: float,
    time: float
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


# =============================================================================
# Moneyness Utilities
# =============================================================================

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
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    time: float
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
    dividend_yield: float = 0.0
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
    strike = fwd * math.exp(-d1 * volatility * sqrt_t + 0.5 * volatility * volatility * time_to_expiry)

    return strike
