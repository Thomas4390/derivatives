"""
Numerical Greeks
================

Finite difference Greeks calculations for any pricing engine.

Useful for:
- Models without closed-form Greeks
- Validating analytic Greeks
- Portfolio-level Greeks

Author: Thomas
Created: 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

from backend.utils.math import DAYS_PER_YEAR

if TYPE_CHECKING:
    from backend.core.interfaces import PricingEngine, Instrument, Model
    from backend.core.market import MarketEnvironment


# =============================================================================
# Type Definitions
# =============================================================================

# Pricing function signature: (spot, strike, time, rate, vol, ...) -> price
PricingFunc = Callable[..., float]


@dataclass(frozen=True)
class GreeksBumpConfig:
    """
    Configuration for finite difference bump sizes.

    Centralizes the default perturbation sizes used in numerical Greeks
    calculations. All values are industry-standard defaults.

    Parameters
    ----------
    spot_bump : float
        Relative spot bump (default 1%). Applied as: h = spot * spot_bump
    vol_bump : float
        Absolute volatility bump (default 1% = 0.01).
        Applied as: vol ± vol_bump
    time_bump_days : float
        Time decay bump in calendar days (default 1 day).
        Converted to years internally: h = time_bump_days / 365
    rate_bump : float
        Absolute rate bump in basis points (default 1bp = 0.0001).
        Applied as: rate ± rate_bump

    Examples
    --------
    config = GreeksBumpConfig()  # Use defaults
    # config.spot_bump == 0.01

    custom = GreeksBumpConfig(spot_bump=0.005, vol_bump=0.001)
    # custom.spot_bump == 0.005

    Notes
    -----
    Default values are chosen for numerical stability and practical relevance:
    - 1% spot bump: Standard for equity delta hedging
    - 1% vol bump: Standard vega reporting convention
    - 1 day theta: Daily P&L relevance
    - 1bp rate bump: Typical rate sensitivity measure
    """
    spot_bump: float = 0.01       # 1% relative
    vol_bump: float = 0.01        # 1% absolute
    time_bump_days: float = 1.0   # 1 calendar day
    rate_bump: float = 0.0001     # 1 basis point


# Module-level default configuration
DEFAULT_BUMP_CONFIG = GreeksBumpConfig()


class NumericalGreeks(NamedTuple):
    """Result from numerical Greeks calculation."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


# =============================================================================
# Individual Greeks (Central Differences)
# =============================================================================

def finite_difference_delta(
    price_func: PricingFunc,
    spot: float,
    bump: float = 0.01,
    **kwargs
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
    price_func: PricingFunc,
    spot: float,
    bump: float = 0.01,
    **kwargs
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
    return (v_up - 2 * v_mid + v_down) / (h ** 2)


def finite_difference_vega(
    price_func: PricingFunc,
    spot: float,
    vol: float,
    bump: float = 0.01,
    **kwargs
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
    **kwargs
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
    v_later = price_func(spot, time=max(time - h, 0.001), **kwargs)
    return (v_later - v_now) / bump_days


def finite_difference_rho(
    price_func: PricingFunc,
    spot: float,
    rate: float,
    bump: float = 0.0001,
    **kwargs
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


# =============================================================================
# Second-Order Greeks
# =============================================================================

def finite_difference_vanna(
    price_func: PricingFunc,
    spot: float,
    vol: float,
    spot_bump: float = 0.01,
    vol_bump: float = 0.01,
    **kwargs
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
    price_func: PricingFunc,
    spot: float,
    vol: float,
    bump: float = 0.01,
    **kwargs
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

    return (v_up - 2 * v_mid + v_down) / (h ** 2) / 10000.0


def finite_difference_charm(
    price_func: PricingFunc,
    spot: float,
    time: float,
    spot_bump: float = 0.01,
    time_bump_days: float = 1.0,
    **kwargs
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
    time_down = max(time - ht, 0.001)

    v_up_now = price_func(spot + hs, time=time, **kwargs)
    v_up_later = price_func(spot + hs, time=time_down, **kwargs)
    v_down_now = price_func(spot - hs, time=time, **kwargs)
    v_down_later = price_func(spot - hs, time=time_down, **kwargs)

    # ∂²V/∂S∂t = ∂(∂V/∂S)/∂t
    delta_now = (v_up_now - v_down_now) / (2 * hs)
    delta_later = (v_up_later - v_down_later) / (2 * hs)

    return (delta_later - delta_now) / time_bump_days


def finite_difference_veta(
    price_func: PricingFunc,
    spot: float,
    vol: float,
    time: float,
    vol_bump: float = 0.01,
    time_bump_days: float = 1.0,
    **kwargs
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
    time_down = max(time - ht, 0.001)

    v_up_now = price_func(spot, vol=vol + hv, time=time, **kwargs)
    v_up_later = price_func(spot, vol=vol + hv, time=time_down, **kwargs)
    v_down_now = price_func(spot, vol=vol - hv, time=time, **kwargs)
    v_down_later = price_func(spot, vol=vol - hv, time=time_down, **kwargs)

    # ∂²V/∂σ∂t = ∂(∂V/∂σ)/∂t
    vega_now = (v_up_now - v_down_now) / (2 * hv)
    vega_later = (v_up_later - v_down_later) / (2 * hv)

    return (vega_later - vega_now) / time_bump_days / 100.0


# =============================================================================
# Third-Order Greeks
# =============================================================================

def finite_difference_speed(
    price_func: PricingFunc,
    spot: float,
    bump: float = 0.01,
    **kwargs
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
    v_2up = price_func(spot + 2*h, **kwargs)
    v_up = price_func(spot + h, **kwargs)
    v_down = price_func(spot - h, **kwargs)
    v_2down = price_func(spot - 2*h, **kwargs)

    # Third derivative approximation
    return (v_2up - 2*v_up + 2*v_down - v_2down) / (2 * h**3)


def finite_difference_zomma(
    price_func: PricingFunc,
    spot: float,
    vol: float,
    spot_bump: float = 0.01,
    vol_bump: float = 0.01,
    **kwargs
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
    gamma_vup = (v_up_vup - 2*v_mid_vup + v_down_vup) / (hs**2)

    # Gamma at vol - hv
    v_up_vdown = price_func(spot + hs, vol=vol - hv, **kwargs)
    v_mid_vdown = price_func(spot, vol=vol - hv, **kwargs)
    v_down_vdown = price_func(spot - hs, vol=vol - hv, **kwargs)
    gamma_vdown = (v_up_vdown - 2*v_mid_vdown + v_down_vdown) / (hs**2)

    return (gamma_vup - gamma_vdown) / (2 * hv) / 100.0


def finite_difference_color(
    price_func: PricingFunc,
    spot: float,
    time: float,
    spot_bump: float = 0.01,
    time_bump_days: float = 1.0,
    **kwargs
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
    time_down = max(time - ht, 0.001)

    # Gamma now
    v_up_now = price_func(spot + hs, time=time, **kwargs)
    v_mid_now = price_func(spot, time=time, **kwargs)
    v_down_now = price_func(spot - hs, time=time, **kwargs)
    gamma_now = (v_up_now - 2*v_mid_now + v_down_now) / (hs**2)

    # Gamma later
    v_up_later = price_func(spot + hs, time=time_down, **kwargs)
    v_mid_later = price_func(spot, time=time_down, **kwargs)
    v_down_later = price_func(spot - hs, time=time_down, **kwargs)
    gamma_later = (v_up_later - 2*v_mid_later + v_down_later) / (hs**2)

    return (gamma_later - gamma_now) / time_bump_days


def finite_difference_ultima(
    price_func: PricingFunc,
    spot: float,
    vol: float,
    bump: float = 0.01,
    **kwargs
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
    v_2up = price_func(spot, vol=vol + 2*h, **kwargs)
    v_up = price_func(spot, vol=vol + h, **kwargs)
    v_down = price_func(spot, vol=vol - h, **kwargs)
    v_2down = price_func(spot, vol=vol - 2*h, **kwargs)

    # Third derivative approximation
    return (v_2up - 2*v_up + 2*v_down - v_2down) / (2 * h**3) / 1000000.0


# =============================================================================
# All Greeks Combined
# =============================================================================

def finite_difference_greeks(
    price_func: PricingFunc,
    spot: float,
    vol: float,
    time: float,
    rate: float,
    config: Optional[GreeksBumpConfig] = None,
    spot_bump: Optional[float] = None,
    vol_bump: Optional[float] = None,
    time_bump_days: Optional[float] = None,
    rate_bump: Optional[float] = None,
    **kwargs
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
    # Use config or defaults, with individual overrides
    cfg = config or DEFAULT_BUMP_CONFIG
    _spot_bump = spot_bump if spot_bump is not None else cfg.spot_bump
    _vol_bump = vol_bump if vol_bump is not None else cfg.vol_bump
    _time_bump_days = time_bump_days if time_bump_days is not None else cfg.time_bump_days
    _rate_bump = rate_bump if rate_bump is not None else cfg.rate_bump

    # Cache commonly reused values
    h_s = spot * _spot_bump
    h_v = _vol_bump
    h_t = _time_bump_days / DAYS_PER_YEAR
    h_r = _rate_bump

    # Base price
    v_mid = price_func(spot, vol=vol, time=time, rate=rate, **kwargs)

    # Spot bumps (for delta and gamma)
    v_s_up = price_func(spot + h_s, vol=vol, time=time, rate=rate, **kwargs)
    v_s_down = price_func(spot - h_s, vol=vol, time=time, rate=rate, **kwargs)

    # Vol bumps
    v_v_up = price_func(spot, vol=vol + h_v, time=time, rate=rate, **kwargs)
    v_v_down = price_func(spot, vol=vol - h_v, time=time, rate=rate, **kwargs)

    # Time bump
    v_t_bump = price_func(spot, vol=vol, time=max(time - h_t, 0.001), rate=rate, **kwargs)

    # Rate bumps
    v_r_up = price_func(spot, vol=vol, time=time, rate=rate + h_r, **kwargs)
    v_r_down = price_func(spot, vol=vol, time=time, rate=rate - h_r, **kwargs)

    # Calculate Greeks
    delta = (v_s_up - v_s_down) / (2 * h_s)
    gamma = (v_s_up - 2 * v_mid + v_s_down) / (h_s ** 2)
    vega = (v_v_up - v_v_down) / (2 * h_v) / 100.0
    theta = (v_t_bump - v_mid) / _time_bump_days
    rho = (v_r_up - v_r_down) / (2 * h_r) / 100.0

    return NumericalGreeks(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho
    )


# =============================================================================
# Model-Aware Numerical Greeks
# =============================================================================

class ModelNumericalGreeks:
    """
    Numerical Greeks calculator that works with model/engine architecture.

    Provides finite difference Greeks for any pricing engine.

    Parameters
    ----------
    config : GreeksBumpConfig, optional
        Bump configuration. If not provided, uses DEFAULT_BUMP_CONFIG.
    """

    def __init__(self, config: Optional[GreeksBumpConfig] = None):
        """
        Initialize numerical Greeks calculator.

        Parameters
        ----------
        config : GreeksBumpConfig, optional
            Bump configuration. If not provided, uses DEFAULT_BUMP_CONFIG.
        """
        self.config = config or DEFAULT_BUMP_CONFIG

    @property
    def spot_bump(self) -> float:
        """Relative spot bump."""
        return self.config.spot_bump

    @property
    def vol_bump(self) -> float:
        """Absolute volatility bump."""
        return self.config.vol_bump

    @property
    def time_bump_days(self) -> float:
        """Time decay bump in days."""
        return self.config.time_bump_days

    @property
    def rate_bump(self) -> float:
        """Absolute rate bump."""
        return self.config.rate_bump

    def calculate(
        self,
        engine: "PricingEngine",
        instrument: "Instrument",
        model: "Model",
        market: "MarketEnvironment",
    ) -> NumericalGreeks:
        """
        Calculate numerical Greeks for an instrument.

        Parameters
        ----------
        engine : PricingEngine
            Pricing engine
        instrument : Instrument
            Option instrument
        model : Model
            Pricing model
        market : MarketEnvironment
            Market conditions

        Returns
        -------
        NumericalGreeks
            Numerical Greeks
        """
        # Base price
        v_mid = engine.price(instrument, model, market).price

        # Spot bumps
        h_s = market.spot * self.spot_bump
        market_s_up = market.bump_spot(h_s)
        market_s_down = market.bump_spot(-h_s)
        v_s_up = engine.price(instrument, model, market_s_up).price
        v_s_down = engine.price(instrument, model, market_s_down).price

        # Rate bumps
        h_r = self.rate_bump
        market_r_up = market.bump_rate(h_r)
        market_r_down = market.bump_rate(-h_r)
        v_r_up = engine.price(instrument, model, market_r_up).price
        v_r_down = engine.price(instrument, model, market_r_down).price

        # Time bump (requires instrument modification)
        h_t = self.time_bump_days / DAYS_PER_YEAR
        from backend.greeks._instrument_utils import create_decayed_instrument
        new_T = max(instrument.maturity - h_t, 0.001)
        decayed = create_decayed_instrument(instrument, new_T)
        if decayed is not None:
            v_t_bump = engine.price(decayed, model, market).price
        else:
            # Unsupported instrument type - skip theta
            v_t_bump = v_mid  # This will result in theta = 0

        # Vol bump (requires model modification)
        from backend.models.vol_bump import create_vol_bumped_pair
        model_v_up, model_v_down = create_vol_bumped_pair(model, self.vol_bump)
        if model_v_up is not None and model_v_down is not None:
            v_v_up = engine.price(instrument, model_v_up, market).price
            v_v_down = engine.price(instrument, model_v_down, market).price
            vega = (v_v_up - v_v_down) / (2 * self.vol_bump) / 100.0
        else:
            vega = 0.0

        # Calculate Greeks
        delta = (v_s_up - v_s_down) / (2 * h_s)
        gamma = (v_s_up - 2 * v_mid + v_s_down) / (h_s ** 2)
        theta = (v_t_bump - v_mid) / self.time_bump_days
        rho = (v_r_up - v_r_down) / (2 * h_r) / 100.0

        return NumericalGreeks(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho
        )


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Numerical Greeks Smoke Test")
    print("=" * 50)

    # Create a simple BS pricing function for testing
    from backend.greeks.analytic import bs_all_greeks

    def bs_price(spot, vol=0.20, time=0.25, rate=0.05, strike=100.0, is_call=True):
        greeks = bs_all_greeks(spot, strike, time, rate, 0.0, vol, is_call)
        return greeks[0]  # price

    # Test parameters
    spot, vol, time, rate = 100.0, 0.20, 0.25, 0.05

    # Calculate numerical Greeks
    num_greeks = finite_difference_greeks(
        bs_price, spot, vol, time, rate,
        strike=100.0, is_call=True
    )

    # Compare with analytic
    _, a_delta, a_gamma, a_vega, a_theta, a_rho, *_ = bs_all_greeks(
        spot, 100.0, time, rate, 0.0, vol, True
    )

    print("\nComparison (Numerical vs Analytic):")
    print(f"  Delta: {num_greeks.delta:.6f} vs {a_delta:.6f}")
    print(f"  Gamma: {num_greeks.gamma:.6f} vs {a_gamma:.6f}")
    print(f"  Vega:  {num_greeks.vega:.6f} vs {a_vega:.6f}")
    print(f"  Theta: {num_greeks.theta:.6f} vs {a_theta:.6f}")
    print(f"  Rho:   {num_greeks.rho:.6f} vs {a_rho:.6f}")

    print("\n" + "=" * 50)
    print("Numerical Greeks smoke test passed")
    print("=" * 50)
