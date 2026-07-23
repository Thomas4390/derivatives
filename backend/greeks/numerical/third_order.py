"""
Third-order finite-difference Greeks (speed, zomma, color, ultima).
"""

from __future__ import annotations

from typing import Any

from backend.greeks.numerical.config import PricingFunc
from backend.utils.constants.time import MIN_TIME_TO_EXPIRY
from backend.utils.math import DAYS_PER_YEAR


def finite_difference_speed(
    price_func: PricingFunc, spot: float, bump: float = 0.01, **kwargs: Any
) -> float:
    """
    Calculate speed (∂³V/∂S³) via finite difference.

    Parameters
    ----------
    price_func : callable
        Pricing function
    spot : float
        Current spot price
    bump : float
        Relative spot bump
    **kwargs
        Additional arguments

    Returns
    -------
    float
        Speed estimate
    """
    h = spot * bump

    # Five-point stencil for third derivative
    v_2up = price_func(spot + 2 * h, **kwargs)
    v_up = price_func(spot + h, **kwargs)
    v_down = price_func(spot - h, **kwargs)
    v_2down = price_func(spot - 2 * h, **kwargs)

    # Third derivative approximation
    return (v_2up - 2 * v_up + 2 * v_down - v_2down) / (2 * h**3)


def finite_difference_zomma(
    price_func: PricingFunc,
    spot: float,
    vol: float,
    spot_bump: float = 0.01,
    vol_bump: float = 0.01,
    **kwargs: Any,
) -> float:
    """
    Calculate zomma (∂³V/∂S²∂σ) via finite difference.

    Returns zomma per 1% vol change.

    Parameters
    ----------
    price_func : callable
        Pricing function
    spot : float
        Current spot price
    vol : float
        Current volatility
    spot_bump : float
        Relative spot bump
    vol_bump : float
        Absolute vol bump
    **kwargs
        Additional arguments

    Returns
    -------
    float
        Zomma estimate (per 1% vol)
    """
    hs = spot * spot_bump
    hv = vol_bump

    # Gamma at vol + hv
    v_up_vup = price_func(spot + hs, vol=vol + hv, **kwargs)
    v_mid_vup = price_func(spot, vol=vol + hv, **kwargs)
    v_down_vup = price_func(spot - hs, vol=vol + hv, **kwargs)
    gamma_vup = (v_up_vup - 2 * v_mid_vup + v_down_vup) / (hs**2)

    # Gamma at vol - hv
    v_up_vdown = price_func(spot + hs, vol=vol - hv, **kwargs)
    v_mid_vdown = price_func(spot, vol=vol - hv, **kwargs)
    v_down_vdown = price_func(spot - hs, vol=vol - hv, **kwargs)
    gamma_vdown = (v_up_vdown - 2 * v_mid_vdown + v_down_vdown) / (hs**2)

    return (gamma_vup - gamma_vdown) / (2 * hv) / 100.0


def finite_difference_color(
    price_func: PricingFunc,
    spot: float,
    time: float,
    spot_bump: float = 0.01,
    time_bump_days: float = 1.0,
    **kwargs: Any,
) -> float:
    """
    Calculate color (∂³V/∂S²∂t) via finite difference.

    Returns color per day.

    Parameters
    ----------
    price_func : callable
        Pricing function
    spot : float
        Current spot price
    time : float
        Time to expiry (years)
    spot_bump : float
        Relative spot bump
    time_bump_days : float
        Time bump in days
    **kwargs
        Additional arguments

    Returns
    -------
    float
        Color estimate (per day)
    """
    hs = spot * spot_bump
    ht = time_bump_days / DAYS_PER_YEAR
    time_down = max(time - ht, MIN_TIME_TO_EXPIRY)
    realized_days = (time - time_down) * DAYS_PER_YEAR

    # Gamma now
    v_up_now = price_func(spot + hs, time=time, **kwargs)
    v_mid_now = price_func(spot, time=time, **kwargs)
    v_down_now = price_func(spot - hs, time=time, **kwargs)
    gamma_now = (v_up_now - 2 * v_mid_now + v_down_now) / (hs**2)

    # Gamma later
    v_up_later = price_func(spot + hs, time=time_down, **kwargs)
    v_mid_later = price_func(spot, time=time_down, **kwargs)
    v_down_later = price_func(spot - hs, time=time_down, **kwargs)
    gamma_later = (v_up_later - 2 * v_mid_later + v_down_later) / (hs**2)

    return (gamma_later - gamma_now) / realized_days if realized_days > 0.0 else 0.0


def finite_difference_ultima(
    price_func: PricingFunc, spot: float, vol: float, bump: float = 0.01, **kwargs: Any
) -> float:
    """
    Calculate ultima (∂³V/∂σ³) via finite difference.

    Returns ultima per 1% vol change cubed.

    Parameters
    ----------
    price_func : callable
        Pricing function
    spot : float
        Current spot price
    vol : float
        Current volatility
    bump : float
        Absolute vol bump
    **kwargs
        Additional arguments

    Returns
    -------
    float
        Ultima estimate (per 1%³ vol)
    """
    h = bump

    # Five-point stencil for third derivative
    v_2up = price_func(spot, vol=vol + 2 * h, **kwargs)
    v_up = price_func(spot, vol=vol + h, **kwargs)
    v_down = price_func(spot, vol=vol - h, **kwargs)
    v_2down = price_func(spot, vol=vol - 2 * h, **kwargs)

    # Third derivative approximation
    return (v_2up - 2 * v_up + 2 * v_down - v_2down) / (2 * h**3) / 1000000.0
