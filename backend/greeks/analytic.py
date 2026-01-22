"""
Analytic Greeks
===============

Closed-form Greeks calculations for Black-Scholes and other analytic models.

Includes all 14 Greeks:
- First order: delta, gamma, vega, theta, rho
- Second order: vanna, volga/vomma, charm, veta
- Third order: speed, zomma, color, ultima

All functions are Numba-optimized for performance.

Scaling Conventions
-------------------
Greeks are scaled to provide intuitive, market-standard values:

**First Order:**
    - Delta: raw (sensitivity to $1 move in spot)
    - Gamma: raw (change in delta per $1 move in spot)
    - Vega: per 1% volatility change (divide raw by 100)
    - Theta: per calendar day (divide raw by 365)
    - Rho: per 1% rate change (divide raw by 100)

**Second Order:**
    - Vanna: per 1% volatility change
    - Volga: per 1%² volatility change (divide raw by 10,000)
    - Charm: per calendar day
    - Veta: per day per 1% volatility

**Third Order:**
    - Speed: raw
    - Zomma: per 1% volatility change
    - Color: per calendar day
    - Ultima: per 1%³ volatility change (divide raw by 1,000,000)

To obtain raw (unscaled) values, use the `unscale_greeks()` function.

Author: Thomas
Created: 2025
"""

import numpy as np
from numba import njit
import math
from typing import NamedTuple


# =============================================================================
# Constants
# =============================================================================

DAYS_PER_YEAR = 365.0
SQRT_2PI = math.sqrt(2.0 * math.pi)

# -----------------------------------------------------------------------------
# Scaling Factors
# -----------------------------------------------------------------------------
# These factors convert raw mathematical Greeks to market-standard values.
# Raw value = scaled value * SCALE_FACTOR (or / UNSCALE_FACTOR)

VEGA_SCALE = 100.0       # Vega is per 1% vol (0.01 in decimal)
RHO_SCALE = 100.0        # Rho is per 1% rate (0.01 in decimal)
THETA_SCALE = DAYS_PER_YEAR  # Theta is per calendar day

VANNA_SCALE = 100.0      # Per 1% vol
VOLGA_SCALE = 10000.0    # Per 1%² vol (100 * 100)
CHARM_SCALE = DAYS_PER_YEAR  # Per calendar day
VETA_SCALE = DAYS_PER_YEAR * 100.0  # Per day per 1% vol

ZOMMA_SCALE = 100.0      # Per 1% vol
COLOR_SCALE = DAYS_PER_YEAR  # Per calendar day
ULTIMA_SCALE = 1000000.0  # Per 1%³ vol (100 * 100 * 100)


# =============================================================================
# Result Types
# =============================================================================

class FirstOrderGreeks(NamedTuple):
    """First-order Greeks."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class SecondOrderGreeks(NamedTuple):
    """Second-order Greeks."""
    vanna: float
    volga: float  # Also known as vomma
    charm: float
    veta: float


class ThirdOrderGreeks(NamedTuple):
    """Third-order Greeks."""
    speed: float
    zomma: float
    color: float
    ultima: float


class AllGreeks(NamedTuple):
    """All 14 Greeks."""
    # Price
    price: float
    # First order
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    # Second order
    vanna: float
    volga: float
    charm: float
    veta: float
    # Third order
    speed: float
    zomma: float
    color: float
    ultima: float


# =============================================================================
# Core Building Blocks (Numba-optimized)
# =============================================================================
#
# NOTE: These local functions duplicate logic from backend.core.bs_formulas.
# They are kept here for Numba compilation compatibility (signature matching).
# For new code, prefer using backend.core.bs_formulas as the single source of truth.
# =============================================================================

@njit(fastmath=True, cache=True)
def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@njit(fastmath=True, cache=True)
def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / SQRT_2PI


@njit(fastmath=True, cache=True)
def _d1_d2(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float
) -> tuple:
    """
    Calculate d1 and d2 parameters.

    Parameters
    ----------
    s : float
        Spot price
    k : float
        Strike price
    t : float
        Time to expiry (years)
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility

    Returns
    -------
    tuple
        (d1, d2)
    """
    if t <= 0 or sigma <= 0:
        return 0.0, 0.0

    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    return d1, d2


# =============================================================================
# First-Order Greeks
# =============================================================================

@njit(fastmath=True, cache=True)
def bs_greeks_first_order(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool
) -> tuple:
    """
    Calculate first-order Black-Scholes Greeks.

    Parameters
    ----------
    s : float
        Spot price
    k : float
        Strike price
    t : float
        Time to expiry (years)
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility
    is_call : bool
        True for call, False for put

    Returns
    -------
    tuple
        (delta, gamma, vega, theta, rho)
    """
    if t <= 0:
        # At expiry
        if is_call:
            delta = 1.0 if s > k else (0.5 if s == k else 0.0)
        else:
            delta = -1.0 if s < k else (-0.5 if s == k else 0.0)
        return delta, 0.0, 0.0, 0.0, 0.0

    if sigma <= 0:
        # Zero volatility limit
        if is_call:
            delta = 1.0 if s * math.exp((r - q) * t) > k else 0.0
        else:
            delta = -1.0 if s * math.exp((r - q) * t) < k else 0.0
        return delta, 0.0, 0.0, 0.0, 0.0

    d1, d2 = _d1_d2(s, k, t, r, q, sigma)
    sqrt_t = math.sqrt(t)
    n_d1 = _norm_cdf(d1)
    n_d2 = _norm_cdf(d2)
    n_prime_d1 = _norm_pdf(d1)

    exp_qt = math.exp(-q * t)
    exp_rt = math.exp(-r * t)

    # Delta
    if is_call:
        delta = exp_qt * n_d1
    else:
        delta = -exp_qt * (1.0 - n_d1)

    # Gamma (same for call and put)
    gamma = exp_qt * n_prime_d1 / (s * sigma * sqrt_t)

    # Vega (same for call and put, per 1% vol change)
    vega = s * exp_qt * n_prime_d1 * sqrt_t / 100.0

    # Theta (per day)
    if is_call:
        theta = (
            -s * exp_qt * n_prime_d1 * sigma / (2.0 * sqrt_t)
            + q * s * exp_qt * n_d1
            - r * k * exp_rt * n_d2
        ) / DAYS_PER_YEAR
    else:
        theta = (
            -s * exp_qt * n_prime_d1 * sigma / (2.0 * sqrt_t)
            - q * s * exp_qt * (1.0 - n_d1)
            + r * k * exp_rt * (1.0 - n_d2)
        ) / DAYS_PER_YEAR

    # Rho (per 1% rate change)
    if is_call:
        rho = k * t * exp_rt * n_d2 / 100.0
    else:
        rho = -k * t * exp_rt * (1.0 - n_d2) / 100.0

    return delta, gamma, vega, theta, rho


@njit(fastmath=True, cache=True)
def bs_delta(s: float, k: float, t: float, r: float, q: float, sigma: float, is_call: bool) -> float:
    """Calculate Black-Scholes delta."""
    if t <= 0 or sigma <= 0:
        if is_call:
            return 1.0 if s > k else 0.0
        else:
            return -1.0 if s < k else 0.0

    d1, _ = _d1_d2(s, k, t, r, q, sigma)
    exp_qt = math.exp(-q * t)

    if is_call:
        return exp_qt * _norm_cdf(d1)
    else:
        return -exp_qt * (1.0 - _norm_cdf(d1))


@njit(fastmath=True, cache=True)
def bs_gamma(s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    """Calculate Black-Scholes gamma (same for call and put)."""
    if t <= 0 or sigma <= 0:
        return 0.0

    d1, _ = _d1_d2(s, k, t, r, q, sigma)
    sqrt_t = math.sqrt(t)
    exp_qt = math.exp(-q * t)

    return exp_qt * _norm_pdf(d1) / (s * sigma * sqrt_t)


# =============================================================================
# Second-Order Greeks
# =============================================================================

@njit(fastmath=True, cache=True)
def bs_greeks_second_order(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float
) -> tuple:
    """
    Calculate second-order Black-Scholes Greeks.

    Parameters
    ----------
    s : float
        Spot price
    k : float
        Strike price
    t : float
        Time to expiry (years)
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility

    Returns
    -------
    tuple
        (vanna, volga, charm, veta)
    """
    if t <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0

    d1, d2 = _d1_d2(s, k, t, r, q, sigma)
    sqrt_t = math.sqrt(t)
    n_prime_d1 = _norm_pdf(d1)
    exp_qt = math.exp(-q * t)

    # Vanna: d²V/dSdσ (per 1% vol change)
    vanna = -exp_qt * n_prime_d1 * d2 / sigma / 100.0

    # Volga/Vomma: d²V/dσ² (per 1% vol change squared)
    vega_base = s * exp_qt * n_prime_d1 * sqrt_t
    volga = vega_base * d1 * d2 / sigma / 10000.0

    # Charm: d²V/dSdt (per day) - call option
    charm = -exp_qt * n_prime_d1 * (
        2.0 * (r - q) * t - d2 * sigma * sqrt_t
    ) / (2.0 * t * sigma * sqrt_t) / DAYS_PER_YEAR

    # Veta: d²V/dσdt (per day per 1% vol)
    veta = s * exp_qt * n_prime_d1 * sqrt_t * (
        q + (r - q) * d1 / (sigma * sqrt_t) - (1.0 + d1 * d2) / (2.0 * t)
    ) / (DAYS_PER_YEAR * 100.0)

    return vanna, volga, charm, veta


# =============================================================================
# Third-Order Greeks
# =============================================================================

@njit(fastmath=True, cache=True)
def bs_greeks_third_order(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float
) -> tuple:
    """
    Calculate third-order Black-Scholes Greeks.

    Parameters
    ----------
    s : float
        Spot price
    k : float
        Strike price
    t : float
        Time to expiry (years)
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility

    Returns
    -------
    tuple
        (speed, zomma, color, ultima)
    """
    if t <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0

    d1, d2 = _d1_d2(s, k, t, r, q, sigma)
    sqrt_t = math.sqrt(t)
    n_prime_d1 = _norm_pdf(d1)
    exp_qt = math.exp(-q * t)

    # Gamma for calculations
    gamma = exp_qt * n_prime_d1 / (s * sigma * sqrt_t)

    # Speed: d³V/dS³
    speed = -gamma * (d1 / (sigma * sqrt_t) + 1.0) / s

    # Zomma: d³V/dS²dσ (per 1% vol change)
    zomma = gamma * (d1 * d2 - 1.0) / sigma / 100.0

    # Color: d³V/dS²dt (per day)
    color = -exp_qt * n_prime_d1 / (2.0 * s * t * sigma * sqrt_t) * (
        2.0 * (r - q) * t - 1.0 +
        d1 * (2.0 * (r - q) * t - d2 * sigma * sqrt_t) / (sigma * sqrt_t)
    ) / DAYS_PER_YEAR

    # Ultima: d³V/dσ³ (per 1% vol change cubed)
    vega = s * exp_qt * n_prime_d1 * sqrt_t
    ultima = -vega / (sigma ** 3) * (
        d1 * d2 * (1.0 - d1 * d2) + d1 * d1 + d2 * d2
    ) / 1000000.0

    return speed, zomma, color, ultima


# =============================================================================
# All Greeks Combined
# =============================================================================

@njit(fastmath=True, cache=True)
def bs_all_greeks(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool
) -> tuple:
    """
    Calculate all 14 Black-Scholes Greeks plus price.

    Parameters
    ----------
    s : float
        Spot price
    k : float
        Strike price
    t : float
        Time to expiry (years)
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility
    is_call : bool
        True for call, False for put

    Returns
    -------
    tuple
        (price, delta, gamma, vega, theta, rho, vanna, volga, charm, veta,
         speed, zomma, color, ultima)
    """
    # Edge cases
    if t <= 0:
        if is_call:
            price = max(s - k, 0.0)
            delta = 1.0 if s > k else 0.0
        else:
            price = max(k - s, 0.0)
            delta = -1.0 if s < k else 0.0
        return (price, delta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0)

    if sigma <= 0:
        exp_qt = math.exp(-q * t)
        exp_rt = math.exp(-r * t)
        forward = s * exp_qt / exp_rt
        if is_call:
            price = max(forward - k, 0.0) * exp_rt
            delta = exp_qt if forward > k else 0.0
        else:
            price = max(k - forward, 0.0) * exp_rt
            delta = -exp_qt if forward < k else 0.0
        return (price, delta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0)

    # Calculate d1, d2
    d1, d2 = _d1_d2(s, k, t, r, q, sigma)
    sqrt_t = math.sqrt(t)
    n_d1 = _norm_cdf(d1)
    n_d2 = _norm_cdf(d2)
    n_prime_d1 = _norm_pdf(d1)

    exp_qt = math.exp(-q * t)
    exp_rt = math.exp(-r * t)

    # Price
    if is_call:
        price = s * exp_qt * n_d1 - k * exp_rt * n_d2
    else:
        price = k * exp_rt * (1.0 - n_d2) - s * exp_qt * (1.0 - n_d1)

    # First order
    delta, gamma, vega, theta, rho = bs_greeks_first_order(
        s, k, t, r, q, sigma, is_call
    )

    # Second order
    vanna, volga, charm, veta = bs_greeks_second_order(s, k, t, r, q, sigma)

    # Third order
    speed, zomma, color, ultima = bs_greeks_third_order(s, k, t, r, q, sigma)

    return (price, delta, gamma, vega, theta, rho, vanna, volga, charm, veta,
            speed, zomma, color, ultima)


# =============================================================================
# Vectorized Versions
# =============================================================================

def bs_greeks_surface(
    spots: np.ndarray,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool
) -> dict:
    """
    Calculate Greeks across a range of spot prices.

    Parameters
    ----------
    spots : np.ndarray
        Array of spot prices
    k : float
        Strike price
    t : float
        Time to expiry
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility
    is_call : bool
        True for call, False for put

    Returns
    -------
    dict
        Dictionary with arrays for each Greek
    """
    n = len(spots)
    result = {
        'price': np.empty(n),
        'delta': np.empty(n),
        'gamma': np.empty(n),
        'vega': np.empty(n),
        'theta': np.empty(n),
        'rho': np.empty(n),
        'vanna': np.empty(n),
        'volga': np.empty(n),
        'charm': np.empty(n),
        'veta': np.empty(n),
        'speed': np.empty(n),
        'zomma': np.empty(n),
        'color': np.empty(n),
        'ultima': np.empty(n),
    }

    for i in range(n):
        greeks = bs_all_greeks(spots[i], k, t, r, q, sigma, is_call)
        result['price'][i] = greeks[0]
        result['delta'][i] = greeks[1]
        result['gamma'][i] = greeks[2]
        result['vega'][i] = greeks[3]
        result['theta'][i] = greeks[4]
        result['rho'][i] = greeks[5]
        result['vanna'][i] = greeks[6]
        result['volga'][i] = greeks[7]
        result['charm'][i] = greeks[8]
        result['veta'][i] = greeks[9]
        result['speed'][i] = greeks[10]
        result['zomma'][i] = greeks[11]
        result['color'][i] = greeks[12]
        result['ultima'][i] = greeks[13]

    return result


# =============================================================================
# Utility Functions
# =============================================================================

def unscale_greeks(greeks: AllGreeks) -> AllGreeks:
    """
    Convert scaled Greeks back to raw mathematical values.

    This function reverses the market-standard scaling applied by default,
    returning the raw mathematical sensitivities.

    Parameters
    ----------
    greeks : AllGreeks
        Scaled Greeks from bs_all_greeks()

    Returns
    -------
    AllGreeks
        Raw (unscaled) Greeks

    Examples
    --------
    >>> scaled = bs_all_greeks(100, 100, 0.25, 0.05, 0.0, 0.20, True)
    >>> raw = unscale_greeks(scaled)
    >>> # raw.vega is now the sensitivity to a 1.0 (100%) vol change
    >>> # raw.theta is now the sensitivity per year (not per day)

    Notes
    -----
    Scaling transformations:
    - vega: scaled_value * 100 (from per 1% to per 100%)
    - theta: scaled_value * 365 (from per day to per year)
    - rho: scaled_value * 100 (from per 1% to per 100%)
    - vanna: scaled_value * 100
    - volga: scaled_value * 10000
    - charm: scaled_value * 365
    - veta: scaled_value * 36500
    - zomma: scaled_value * 100
    - color: scaled_value * 365
    - ultima: scaled_value * 1000000
    """
    return AllGreeks(
        price=greeks.price,
        # First order
        delta=greeks.delta,
        gamma=greeks.gamma,
        vega=greeks.vega * VEGA_SCALE,
        theta=greeks.theta * THETA_SCALE,
        rho=greeks.rho * RHO_SCALE,
        # Second order
        vanna=greeks.vanna * VANNA_SCALE,
        volga=greeks.volga * VOLGA_SCALE,
        charm=greeks.charm * CHARM_SCALE,
        veta=greeks.veta * VETA_SCALE,
        # Third order
        speed=greeks.speed,
        zomma=greeks.zomma * ZOMMA_SCALE,
        color=greeks.color * COLOR_SCALE,
        ultima=greeks.ultima * ULTIMA_SCALE,
    )


def scale_greeks(raw_greeks: AllGreeks) -> AllGreeks:
    """
    Apply market-standard scaling to raw Greeks.

    This is the inverse of unscale_greeks().

    Parameters
    ----------
    raw_greeks : AllGreeks
        Raw mathematical Greeks

    Returns
    -------
    AllGreeks
        Market-standard scaled Greeks
    """
    return AllGreeks(
        price=raw_greeks.price,
        # First order
        delta=raw_greeks.delta,
        gamma=raw_greeks.gamma,
        vega=raw_greeks.vega / VEGA_SCALE,
        theta=raw_greeks.theta / THETA_SCALE,
        rho=raw_greeks.rho / RHO_SCALE,
        # Second order
        vanna=raw_greeks.vanna / VANNA_SCALE,
        volga=raw_greeks.volga / VOLGA_SCALE,
        charm=raw_greeks.charm / CHARM_SCALE,
        veta=raw_greeks.veta / VETA_SCALE,
        # Third order
        speed=raw_greeks.speed,
        zomma=raw_greeks.zomma / ZOMMA_SCALE,
        color=raw_greeks.color / COLOR_SCALE,
        ultima=raw_greeks.ultima / ULTIMA_SCALE,
    )


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Analytic Greeks Smoke Test")
    print("=" * 50)

    # Test parameters
    s, k, t, r, q, sigma = 100.0, 100.0, 0.25, 0.05, 0.02, 0.20

    # First order Greeks
    delta, gamma, vega, theta, rho = bs_greeks_first_order(
        s, k, t, r, q, sigma, is_call=True
    )
    print(f"\nFirst-Order Greeks (ATM Call):")
    print(f"  Delta: {delta:.6f}")
    print(f"  Gamma: {gamma:.6f}")
    print(f"  Vega:  {vega:.6f} (per 1% vol)")
    print(f"  Theta: {theta:.6f} (per day)")
    print(f"  Rho:   {rho:.6f} (per 1% rate)")

    # Second order Greeks
    vanna, volga, charm, veta = bs_greeks_second_order(s, k, t, r, q, sigma)
    print(f"\nSecond-Order Greeks:")
    print(f"  Vanna: {vanna:.6f}")
    print(f"  Volga: {volga:.6f}")
    print(f"  Charm: {charm:.6f}")
    print(f"  Veta:  {veta:.6f}")

    # Third order Greeks
    speed, zomma, color, ultima = bs_greeks_third_order(s, k, t, r, q, sigma)
    print(f"\nThird-Order Greeks:")
    print(f"  Speed:  {speed:.8f}")
    print(f"  Zomma:  {zomma:.8f}")
    print(f"  Color:  {color:.8f}")
    print(f"  Ultima: {ultima:.10f}")

    # All Greeks
    all_greeks = bs_all_greeks(s, k, t, r, q, sigma, is_call=True)
    print(f"\nAll Greeks (price included):")
    print(f"  Price: ${all_greeks[0]:.4f}")

    print("\n" + "=" * 50)
    print("Analytic Greeks smoke test passed")
    print("=" * 50)
