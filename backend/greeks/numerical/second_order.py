"""
Second-order finite-difference Greeks (vanna, volga, charm, veta).
"""

from __future__ import annotations

from typing import Any

from backend.greeks.numerical.config import PricingFunc
from backend.utils.constants.time import MIN_TIME_TO_EXPIRY
from backend.utils.math import DAYS_PER_YEAR


def finite_difference_vanna(
    price_func: PricingFunc,
    spot: float,
    vol: float,
    spot_bump: float = 0.01,
    vol_bump: float = 0.01,
    **kwargs: Any,
) -> float:
    """
    Calculate vanna (∂²V/∂S∂σ) via cross finite difference.

    Returns vanna per 1% vol change.

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
        Vanna estimate
    """
    hs = spot * spot_bump
    hv = vol_bump

    v_up_up = price_func(spot + hs, vol=vol + hv, **kwargs)
    v_up_down = price_func(spot + hs, vol=vol - hv, **kwargs)
    v_down_up = price_func(spot - hs, vol=vol + hv, **kwargs)
    v_down_down = price_func(spot - hs, vol=vol - hv, **kwargs)

    return (v_up_up - v_up_down - v_down_up + v_down_down) / (4 * hs * hv) / 100.0


def finite_difference_volga(
    price_func: PricingFunc, spot: float, vol: float, bump: float = 0.01, **kwargs: Any
) -> float:
    """
    Calculate volga (∂²V/∂σ²) via central finite difference.

    Returns volga per 1% vol change squared.

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
        Volga estimate
    """
    h = bump
    v_up = price_func(spot, vol=vol + h, **kwargs)
    v_mid = price_func(spot, vol=vol, **kwargs)
    v_down = price_func(spot, vol=vol - h, **kwargs)

    return (v_up - 2 * v_mid + v_down) / (h**2) / 10000.0


def finite_difference_charm(
    price_func: PricingFunc,
    spot: float,
    time: float,
    spot_bump: float = 0.01,
    time_bump_days: float = 1.0,
    **kwargs: Any,
) -> float:
    """
    Calculate charm (∂²V/∂S∂t) via cross finite difference.

    Returns charm per day.

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
        Charm estimate (per day)
    """
    hs = spot * spot_bump
    ht = time_bump_days / DAYS_PER_YEAR
    time_down = max(time - ht, MIN_TIME_TO_EXPIRY)
    realized_days = (time - time_down) * DAYS_PER_YEAR

    v_up_now = price_func(spot + hs, time=time, **kwargs)
    v_up_later = price_func(spot + hs, time=time_down, **kwargs)
    v_down_now = price_func(spot - hs, time=time, **kwargs)
    v_down_later = price_func(spot - hs, time=time_down, **kwargs)

    # ∂²V/∂S∂t = ∂(∂V/∂S)/∂t
    delta_now = (v_up_now - v_down_now) / (2 * hs)
    delta_later = (v_up_later - v_down_later) / (2 * hs)

    return (delta_later - delta_now) / realized_days if realized_days > 0.0 else 0.0


def finite_difference_veta(
    price_func: PricingFunc,
    spot: float,
    vol: float,
    time: float,
    vol_bump: float = 0.01,
    time_bump_days: float = 1.0,
    **kwargs: Any,
) -> float:
    """
    Calculate veta (∂²V/∂σ∂t) via cross finite difference.

    Returns veta per day per 1% vol.

    Parameters
    ----------
    price_func : callable
        Pricing function
    spot : float
        Current spot price
    vol : float
        Current volatility
    time : float
        Time to expiry (years)
    vol_bump : float
        Absolute vol bump
    time_bump_days : float
        Time bump in days
    **kwargs
        Additional arguments

    Returns
    -------
    float
        Veta estimate (per day per 1% vol)
    """
    hv = vol_bump
    ht = time_bump_days / DAYS_PER_YEAR
    time_down = max(time - ht, MIN_TIME_TO_EXPIRY)
    realized_days = (time - time_down) * DAYS_PER_YEAR

    v_up_now = price_func(spot, vol=vol + hv, time=time, **kwargs)
    v_up_later = price_func(spot, vol=vol + hv, time=time_down, **kwargs)
    v_down_now = price_func(spot, vol=vol - hv, time=time, **kwargs)
    v_down_later = price_func(spot, vol=vol - hv, time=time_down, **kwargs)

    # ∂²V/∂σ∂t = ∂(∂V/∂σ)/∂t
    vega_now = (v_up_now - v_down_now) / (2 * hv)
    vega_later = (v_up_later - v_down_later) / (2 * hv)

    return (
        (vega_later - vega_now) / realized_days / 100.0 if realized_days > 0.0 else 0.0
    )
