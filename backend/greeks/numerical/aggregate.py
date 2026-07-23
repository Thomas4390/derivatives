"""
All-first-order finite-difference aggregator: validates spot, resolves the bump
config + overrides, runs the bump sweep inline and assembles ``NumericalGreeks``.
"""

from __future__ import annotations

from typing import Any

from backend.greeks.numerical._guards import (
    _check_finite,
    _require_positive_spot,
    _safe_div,
)
from backend.greeks.numerical.config import (
    DEFAULT_BUMP_CONFIG,
    GreeksBumpConfig,
    NumericalGreeks,
    PricingFunc,
)
from backend.utils.math import DAYS_PER_YEAR


def finite_difference_greeks(
    price_func: PricingFunc,
    spot: float,
    vol: float,
    time: float,
    rate: float,
    config: GreeksBumpConfig | None = None,
    spot_bump: float | None = None,
    vol_bump: float | None = None,
    time_bump_days: float | None = None,
    rate_bump: float | None = None,
    **kwargs: Any,
) -> NumericalGreeks:
    """
    Calculate all first-order Greeks via finite differences.

    Parameters
    ----------
    price_func : callable
        Pricing function with signature:
        price_func(spot, vol=vol, time=time, rate=rate, **kwargs)
    spot : float
        Current spot price
    vol : float
        Current volatility
    time : float
        Time to expiry (years)
    rate : float
        Risk-free rate
    config : GreeksBumpConfig, optional
        Bump configuration. If not provided, uses DEFAULT_BUMP_CONFIG.
        Individual bump parameters override config values if specified.
    spot_bump : float, optional
        Relative spot bump (overrides config)
    vol_bump : float, optional
        Absolute vol bump (overrides config)
    time_bump_days : float, optional
        Time bump in days (overrides config)
    rate_bump : float, optional
        Absolute rate bump (overrides config)
    **kwargs
        Additional arguments for price_func

    Returns
    -------
    NumericalGreeks
        Named tuple with (delta, gamma, vega, theta, rho)
    """
    _require_positive_spot(spot)

    # Use config or defaults, with individual overrides
    cfg = config or DEFAULT_BUMP_CONFIG
    _spot_bump = spot_bump if spot_bump is not None else cfg.spot_bump
    _vol_bump = vol_bump if vol_bump is not None else cfg.vol_bump
    _time_bump_days = (
        time_bump_days if time_bump_days is not None else cfg.time_bump_days
    )
    _rate_bump = rate_bump if rate_bump is not None else cfg.rate_bump

    # Cache commonly reused values
    h_s: float = spot * _spot_bump
    h_v: float = _vol_bump
    h_t: float = _time_bump_days / DAYS_PER_YEAR
    h_r: float = _rate_bump

    # Base price
    v_mid: float = _check_finite(
        price_func(spot, vol=vol, time=time, rate=rate, **kwargs), label="v_mid"
    )

    # Spot bumps (for delta and gamma)
    v_s_up: float = _check_finite(
        price_func(spot + h_s, vol=vol, time=time, rate=rate, **kwargs),
        label="v_spot_up",
    )
    v_s_down: float = _check_finite(
        price_func(spot - h_s, vol=vol, time=time, rate=rate, **kwargs),
        label="v_spot_down",
    )

    # Vol bumps
    v_v_up: float = _check_finite(
        price_func(spot, vol=vol + h_v, time=time, rate=rate, **kwargs),
        label="v_vol_up",
    )
    v_v_down: float = _check_finite(
        price_func(spot, vol=vol - h_v, time=time, rate=rate, **kwargs),
        label="v_vol_down",
    )

    # Time bump
    v_t_bump: float = _check_finite(
        price_func(spot, vol=vol, time=max(time - h_t, 0.001), rate=rate, **kwargs),
        label="v_time_bump",
    )

    # Rate bumps
    v_r_up: float = _check_finite(
        price_func(spot, vol=vol, time=time, rate=rate + h_r, **kwargs),
        label="v_rate_up",
    )
    v_r_down: float = _check_finite(
        price_func(spot, vol=vol, time=time, rate=rate - h_r, **kwargs),
        label="v_rate_down",
    )

    # Calculate Greeks (safe-div guards against pathological zero bumps)
    delta: float = _safe_div(v_s_up - v_s_down, 2 * h_s, label="delta")
    gamma: float = _safe_div(v_s_up - 2 * v_mid + v_s_down, h_s**2, label="gamma")
    vega: float = _safe_div(v_v_up - v_v_down, 2 * h_v, label="vega") / 100.0
    theta: float = _safe_div(v_t_bump - v_mid, _time_bump_days, label="theta")
    rho: float = _safe_div(v_r_up - v_r_down, 2 * h_r, label="rho") / 100.0

    return NumericalGreeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
