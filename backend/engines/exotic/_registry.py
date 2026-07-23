"""
Exotic Pricer Registry
======================

Open-Closed dispatch for the closed-form exotic pricers ported from Haug.

Each new exotic registers one :class:`PricerSpec` instead of adding a branch to
``ExoticAnalyticEngine.price()`` / ``greeks()``. A spec carries an *adapter*:
given an instrument + model + market it returns a small closure
``f(S, sigma, T, r) -> price`` that has captured every static parameter
(strike, barriers, dividend yield, ...). That closure is the single seam used
both for pricing and for the generic finite-difference Greeks below, so kernels
keep pure mathematical signatures and no per-type Greek code is needed.

Design patterns: Strategy (interchangeable pricing kernel), Adapter (instrument
-> kernel-args closure), Registry / dispatch-table (Open-Closed), and a generic
finite-difference Greek (Template Method).

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from backend.core.interfaces import Instrument, Model
from backend.core.market import MarketEnvironment
from backend.utils.constants.exotic import (
    FD_RATE_BUMP,
    FD_SIGMA_FLOOR,
    FD_SPOT_REL_BUMP,
    FD_TIME_BUMP,
    FD_VOL_BUMP,
)

# A priced adapter closes over an instrument's static parameters and exposes the
# four finite-difference bump axes: spot, volatility, time-to-maturity, rate.
PriceFn = Callable[[float, float, float, float], float]  # (S, sigma, T, r) -> price
AdapterFn = Callable[[Instrument, Model, MarketEnvironment], PriceFn]


@dataclass(frozen=True)
class PricerSpec:
    """One registered closed-form exotic pricer.

    Parameters
    ----------
    instrument_type : type
        The :class:`Instrument` subclass this spec prices.
    adapter : AdapterFn
        Builds a ``(S, sigma, T, r) -> price`` closure from the instrument,
        model and market (capturing strike/barriers/dividend yield/...).
    label : str
        Human-readable name (diagnostics, error messages).
    """

    instrument_type: type
    adapter: AdapterFn
    label: str


EXOTIC_PRICER_REGISTRY: dict[type, PricerSpec] = {}


def register(spec: PricerSpec) -> None:
    """Register (or replace) a pricer spec for its instrument type."""
    EXOTIC_PRICER_REGISTRY[spec.instrument_type] = spec


def lookup(instrument: Instrument) -> PricerSpec | None:
    """Return the spec registered for ``type(instrument)``, or ``None``."""
    return EXOTIC_PRICER_REGISTRY.get(type(instrument))


def registry_price(
    spec: PricerSpec,
    instrument: Instrument,
    model: Model,
    market: MarketEnvironment,
) -> float:
    """Price ``instrument`` through its registered adapter."""
    f = spec.adapter(instrument, model, market)
    return f(market.spot, model.sigma, instrument.maturity, market.rate)  # type: ignore[attr-defined]


def fd_greeks(
    spec: PricerSpec,
    instrument: Instrument,
    model: Model,
    market: MarketEnvironment,
) -> tuple[float, float, float, float, float, float]:
    """Central finite-difference Greeks for a registered exotic.

    Returns ``(price, delta, gamma, vega, theta, rho)`` using the exact bump
    conventions of the legacy njit ``exotic_greeks_batch``: 1% spot, 1 vol
    point, 1 calendar-day one-sided theta, 1bp rate; vega and rho per 100 bps.
    """
    f = spec.adapter(instrument, model, market)
    s0: float = market.spot
    sigma0: float = model.sigma  # type: ignore[attr-defined]
    t0: float = instrument.maturity
    r0: float = market.rate

    d_s = s0 * FD_SPOT_REL_BUMP
    d_v = FD_VOL_BUMP
    d_t = FD_TIME_BUMP
    d_r = FD_RATE_BUMP
    sigma_dn = max(sigma0 - d_v, FD_SIGMA_FLOOR)
    has_theta = d_t < t0
    t_dn = t0 - d_t if has_theta else t0

    price = f(s0, sigma0, t0, r0)
    p_up = f(s0 + d_s, sigma0, t0, r0)
    p_dn = f(s0 - d_s, sigma0, t0, r0)
    p_vu = f(s0, sigma0 + d_v, t0, r0)
    p_vd = f(s0, sigma_dn, t0, r0)
    p_td = f(s0, sigma0, t_dn, r0) if has_theta else price
    p_ru = f(s0, sigma0, t0, r0 + d_r)
    p_rd = f(s0, sigma0, t0, r0 - d_r)

    delta = (p_up - p_dn) / (2.0 * d_s)
    gamma = (p_up - 2.0 * price + p_dn) / (d_s * d_s)
    vega = (p_vu - p_vd) / (2.0 * d_v) / 100.0
    theta = (p_td - price) if has_theta else 0.0
    rho = (p_ru - p_rd) / (2.0 * d_r) / 100.0
    return price, delta, gamma, vega, theta, rho
