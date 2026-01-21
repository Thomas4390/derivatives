"""
Mathematical Utilities
======================

Shared mathematical primitives used across the backend.
All functions are Numba-optimized for performance.

These are the SINGLE source of truth for:
- Normal distribution functions (CDF, PDF)
- Black-Scholes d1/d2 parameters

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
# Black-Scholes Parameters
# =============================================================================

@njit(fastmath=True, cache=True)
def d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float
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

    Returns
    -------
    Tuple[float, float]
        (d1, d2) parameters

    Notes
    -----
    d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d2 = d1 - σ√T

    Edge cases:
    - time_to_expiry <= 0: returns (0.0, 0.0)
    - volatility <= 0: returns large values based on moneyness
    """
    if time_to_expiry <= 0:
        return 0.0, 0.0

    if volatility <= 0:
        # Handle zero volatility case based on moneyness
        forward = spot * np.exp(risk_free_rate * time_to_expiry)
        if forward > strike:
            return 1e10, 1e10  # Deep ITM
        elif forward < strike:
            return -1e10, -1e10  # Deep OTM
        else:
            return 0.0, 0.0  # ATM

    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    return d1, d2


# =============================================================================
# Vectorized Versions (for surfaces and batch calculations)
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


# =============================================================================
# Black-Scholes Higher-Order Greeks (Numba-optimized)
# =============================================================================

DAYS_PER_YEAR = 365.0


@njit(fastmath=True, cache=True)
def bs_second_order_greeks(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
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
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    n_prime_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)

    # Vanna - d²V/dSdσ (per 1% vol change)
    vanna = -n_prime_d1 * d2 / sigma / 100.0

    # Volga/Vomma - d²V/dσ² (per 1% vol change squared)
    vega_base = spot * n_prime_d1 * sqrt_t
    volga = vega_base * d1 * d2 / sigma / 10000.0

    # Charm - d²V/dSdt (per day)
    charm = -n_prime_d1 * (
        2 * r * t - d2 * sigma * sqrt_t
    ) / (2 * t * sigma * sqrt_t) / DAYS_PER_YEAR

    # Veta - d²V/dσdt (per day per 1% vol)
    veta = spot * n_prime_d1 * sqrt_t * (
        r * d1 / (sigma * sqrt_t) - (1 + d1 * d2) / (2 * t)
    ) / (DAYS_PER_YEAR * 100.0)

    return vanna, volga, charm, veta


@njit(fastmath=True, cache=True)
def bs_third_order_greeks(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
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
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    n_prime_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)

    # Gamma for speed calculation
    gamma = n_prime_d1 / (spot * sigma * sqrt_t)

    # Speed - d³V/dS³
    speed = -gamma * (d1 / (sigma * sqrt_t) + 1) / spot

    # Zomma - d³V/dS²dσ (per 1% vol change)
    zomma = gamma * (d1 * d2 - 1) / sigma / 100.0

    # Color - d³V/dS²dt (per day)
    color = -n_prime_d1 / (2 * spot * t * sigma * sqrt_t) * (
        2 * r * t - 1 +
        d1 * (2 * r * t - d2 * sigma * sqrt_t) / (sigma * sqrt_t)
    ) / DAYS_PER_YEAR

    # Ultima - d³V/dσ³ (per 1% vol change cubed)
    vega = spot * n_prime_d1 * sqrt_t
    ultima = -vega / (sigma ** 3) * (
        d1 * d2 * (1 - d1 * d2) + d1 * d1 + d2 * d2
    ) / 1000000.0

    return speed, zomma, color, ultima
