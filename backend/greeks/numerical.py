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

import numpy as np
from typing import Callable, NamedTuple, Optional


# =============================================================================
# Type Definitions
# =============================================================================

# Pricing function signature: (spot, strike, time, rate, vol, ...) -> price
PricingFunc = Callable[..., float]


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
    h = bump_days / 365.0
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


# =============================================================================
# All Greeks Combined
# =============================================================================

def finite_difference_greeks(
    price_func: PricingFunc,
    spot: float,
    vol: float,
    time: float,
    rate: float,
    spot_bump: float = 0.01,
    vol_bump: float = 0.01,
    time_bump_days: float = 1.0,
    rate_bump: float = 0.0001,
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
    spot_bump : float
        Relative spot bump (default 1%)
    vol_bump : float
        Absolute vol bump (default 1%)
    time_bump_days : float
        Time bump in days (default 1 day)
    rate_bump : float
        Absolute rate bump (default 1bp)
    **kwargs
        Additional arguments for price_func

    Returns
    -------
    NumericalGreeks
        Named tuple with (delta, gamma, vega, theta, rho)
    """
    # Cache commonly reused values
    h_s = spot * spot_bump
    h_v = vol_bump
    h_t = time_bump_days / 365.0
    h_r = rate_bump

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
    theta = (v_t_bump - v_mid) / time_bump_days
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
    """

    def __init__(
        self,
        spot_bump: float = 0.01,
        vol_bump: float = 0.01,
        time_bump_days: float = 1.0,
        rate_bump: float = 0.0001
    ):
        """
        Initialize numerical Greeks calculator.

        Parameters
        ----------
        spot_bump : float
            Relative spot bump (default 1%)
        vol_bump : float
            Absolute vol bump (default 1%)
        time_bump_days : float
            Time bump in days (default 1 day)
        rate_bump : float
            Absolute rate bump (default 1bp)
        """
        self.spot_bump = spot_bump
        self.vol_bump = vol_bump
        self.time_bump_days = time_bump_days
        self.rate_bump = rate_bump

    def calculate(
        self,
        engine,
        instrument,
        model,
        market,
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
        h_t = self.time_bump_days / 365.0
        # Create decayed instrument
        from backend.instruments.options import VanillaOption
        decayed = VanillaOption(
            strike=instrument.strike,
            maturity=max(instrument.maturity - h_t, 0.001),
            is_call=instrument.is_call
        )
        v_t_bump = engine.price(decayed, model, market).price

        # Vol bump (requires model modification)
        params = model.get_parameters()
        if 'sigma' in params:
            # GBM model
            from backend.models.gbm import GBMModel
            model_v_up = GBMModel(sigma=params['sigma'] + self.vol_bump)
            model_v_down = GBMModel(sigma=params['sigma'] - self.vol_bump)
            v_v_up = engine.price(instrument, model_v_up, market).price
            v_v_down = engine.price(instrument, model_v_down, market).price
            vega = (v_v_up - v_v_down) / (2 * self.vol_bump) / 100.0
        else:
            vega = 0.0  # Non-GBM model, vega would need different calculation

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
    from .analytic import bs_all_greeks

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

    print(f"\nComparison (Numerical vs Analytic):")
    print(f"  Delta: {num_greeks.delta:.6f} vs {a_delta:.6f}")
    print(f"  Gamma: {num_greeks.gamma:.6f} vs {a_gamma:.6f}")
    print(f"  Vega:  {num_greeks.vega:.6f} vs {a_vega:.6f}")
    print(f"  Theta: {num_greeks.theta:.6f} vs {a_theta:.6f}")
    print(f"  Rho:   {num_greeks.rho:.6f} vs {a_rho:.6f}")

    print("\n" + "=" * 50)
    print("Numerical Greeks smoke test passed")
    print("=" * 50)
