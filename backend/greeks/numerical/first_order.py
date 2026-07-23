"""
First-order finite-difference Greeks (delta, gamma, vega, theta, rho) on a bare
``price_func``.
"""

from __future__ import annotations

from typing import Any

from backend.greeks.numerical.config import PricingFunc
from backend.utils.constants.time import MIN_TIME_TO_EXPIRY
from backend.utils.math import DAYS_PER_YEAR


def finite_difference_delta(
    price_func: PricingFunc, spot: float, bump: float = 0.01, **kwargs: Any
) -> float:
    """
    Calculate delta via central finite difference.

    ∂V/∂S ≈ (V(S+h) - V(S-h)) / (2h)

    Parameters
    ----------
    price_func : callable
        Pricing function with signature price_func(spot, **kwargs)
    spot : float
        Current spot price
    bump : float
        Relative bump size (default 1%)
    **kwargs
        Additional arguments for price_func

    Returns
    -------
    float
        Delta estimate
    """
    h = spot * bump
    v_up = price_func(spot + h, **kwargs)
    v_down = price_func(spot - h, **kwargs)
    return (v_up - v_down) / (2 * h)


def finite_difference_gamma(
    price_func: PricingFunc, spot: float, bump: float = 0.01, **kwargs: Any
) -> float:
    """
    Calculate gamma via central finite difference.

    ∂²V/∂S² ≈ (V(S+h) - 2V(S) + V(S-h)) / h²

    Parameters
    ----------
    price_func : callable
        Pricing function
    spot : float
        Current spot price
    bump : float
        Relative bump size
    **kwargs
        Additional arguments for price_func

    Returns
    -------
    float
        Gamma estimate
    """
    h = spot * bump
    v_up = price_func(spot + h, **kwargs)
    v_mid = price_func(spot, **kwargs)
    v_down = price_func(spot - h, **kwargs)
    return (v_up - 2 * v_mid + v_down) / (h**2)


def finite_difference_vega(
    price_func: PricingFunc, spot: float, vol: float, bump: float = 0.01, **kwargs: Any
) -> float:
    """
    Calculate vega via central finite difference.

    ∂V/∂σ ≈ (V(σ+h) - V(σ-h)) / (2h)

    Returns vega per 1% vol change.

    Parameters
    ----------
    price_func : callable
        Pricing function with signature price_func(spot, vol=vol, **kwargs)
    spot : float
        Current spot price
    vol : float
        Current volatility
    bump : float
        Absolute vol bump size (default 1%)
    **kwargs
        Additional arguments for price_func

    Returns
    -------
    float
        Vega estimate (per 1% vol change)
    """
    h = bump
    v_up = price_func(spot, vol=vol + h, **kwargs)
    v_down = price_func(spot, vol=vol - h, **kwargs)
    return (v_up - v_down) / (2 * h) / 100.0


def finite_difference_theta(
    price_func: PricingFunc,
    spot: float,
    time: float,
    bump_days: float = 1.0,
    **kwargs: Any,
) -> float:
    """
    Calculate theta via forward finite difference.

    ∂V/∂t ≈ (V(t-h) - V(t)) / h

    Returns theta per day (negative for long options).

    Parameters
    ----------
    price_func : callable
        Pricing function with signature price_func(spot, time=time, **kwargs)
    spot : float
        Current spot price
    time : float
        Time to expiry (years)
    bump_days : float
        Time bump in days (default 1 day)
    **kwargs
        Additional arguments for price_func

    Returns
    -------
    float
        Theta estimate (per day)
    """
    h = bump_days / DAYS_PER_YEAR
    v_now = price_func(spot, time=time, **kwargs)
    t_down = max(time - h, MIN_TIME_TO_EXPIRY)
    v_later = price_func(spot, time=t_down, **kwargs)
    # Divide by the realized step (in days), not the nominal bump: near expiry
    # the time floor clamps the step, and using bump_days would understate theta.
    realized_days = (time - t_down) * DAYS_PER_YEAR
    return (v_later - v_now) / realized_days if realized_days > 0.0 else 0.0


def finite_difference_rho(
    price_func: PricingFunc,
    spot: float,
    rate: float,
    bump: float = 0.0001,
    **kwargs: Any,
) -> float:
    """
    Calculate rho via central finite difference.

    ∂V/∂r ≈ (V(r+h) - V(r-h)) / (2h)

    Returns rho per 1% rate change.

    Parameters
    ----------
    price_func : callable
        Pricing function with signature price_func(spot, rate=rate, **kwargs)
    spot : float
        Current spot price
    rate : float
        Current risk-free rate
    bump : float
        Absolute rate bump (default 1bp)
    **kwargs
        Additional arguments for price_func

    Returns
    -------
    float
        Rho estimate (per 1% rate change)
    """
    h = bump
    v_up = price_func(spot, rate=rate + h, **kwargs)
    v_down = price_func(spot, rate=rate - h, **kwargs)
    return (v_up - v_down) / (2 * h) / 100.0
