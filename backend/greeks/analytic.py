"""
Analytic Greeks
===============

Closed-form Greeks calculations for Black-Scholes and other analytic models.

Includes all 14 Greeks:
- First order: delta, gamma, vega, theta, rho
- Second order: vanna, volga/vomma, charm, veta
- Third order: speed, zomma, color, ultima

All functions delegate to backend.utils.math (single source of truth).

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
from typing import NamedTuple

# Import from single source of truth
from backend.utils.math import (
    DAYS_PER_YEAR,
    bs_greeks as _bs_greeks,
    bs_second_order_greeks as _bs_second_order_greeks,
    bs_third_order_greeks as _bs_third_order_greeks,
    bs_delta as _bs_delta,
    bs_gamma as _bs_gamma,
)


# =============================================================================
# Constants
# =============================================================================

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
# First-Order Greeks
# =============================================================================

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
    # Use single source of truth from utils/math
    price, delta, gamma, vega, theta, rho = _bs_greeks(
        s, k, t, r, sigma, is_call, q
    )
    return delta, gamma, vega, theta, rho


def bs_delta(s: float, k: float, t: float, r: float, q: float, sigma: float, is_call: bool) -> float:
    """Calculate Black-Scholes delta."""
    return _bs_delta(s, k, t, r, sigma, is_call, q)


def bs_gamma(s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    """Calculate Black-Scholes gamma (same for call and put)."""
    return _bs_gamma(s, k, t, r, sigma, q)


# =============================================================================
# Second-Order Greeks
# =============================================================================

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
    return _bs_second_order_greeks(s, k, t, r, sigma, q)


# =============================================================================
# Third-Order Greeks
# =============================================================================

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
    return _bs_third_order_greeks(s, k, t, r, sigma, q)


# =============================================================================
# All Greeks Combined
# =============================================================================

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
    # First order (including price)
    price, delta, gamma, vega, theta, rho = _bs_greeks(s, k, t, r, sigma, is_call, q)

    # Second order
    vanna, volga, charm, veta = _bs_second_order_greeks(s, k, t, r, sigma, q)

    # Third order
    speed, zomma, color, ultima = _bs_third_order_greeks(s, k, t, r, sigma, q)

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
