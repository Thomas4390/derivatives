"""
Surface Pricing Loop
=====================

Factored surface-pricing routine used by all option-surface calibrators
(Heston, Merton, Bates). Replaces ~40 lines duplicated across the three
legacy calibrators with a single, well-tested helper.

Semantics
---------
`price_surface` takes a model, a market environment, an option surface
(OptionMarketData), and a Fourier/FFT pricing engine, and returns a
vector of model prices aligned with `market_data.quotes`.

Edge cases handled:
- Mixed call/put quotes at the same maturity: the engine is called once
  for the majority type via FFT, and put-call parity converts the rest.
- Non-positive prices: clamped to 0 (arbitrage violation near deep OTM).

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from backend.calibration.market_data import OptionMarketData
from backend.core.market import MarketEnvironment
from backend.instruments.options import VanillaOption

_logger = logging.getLogger(__name__)


class SurfaceEngine(Protocol):
    """Any engine exposing vectorized `price_strikes` over strikes."""

    def price_strikes(
        self,
        template: VanillaOption,
        model,  # noqa: ANN001 -- duck-typed across Heston/Merton/Bates
        market: MarketEnvironment,
        strikes: np.ndarray,
    ) -> np.ndarray: ...


class MCTerminalSimulator(Protocol):
    """Risk-neutral simulator exposing terminal prices for a single maturity."""

    def terminals(
        self,
        s0: float,
        r: float,
        t: float,
        *,
        n_paths: int,
        rng: np.random.Generator,
        antithetic: bool = ...,
    ) -> np.ndarray: ...


def price_surface(
    model,  # noqa: ANN001
    market_data: OptionMarketData,
    engine: SurfaceEngine,
) -> np.ndarray:
    """Price every quote of an option surface under a given model.

    Parameters
    ----------
    model : Any
        Fourier-pricable model exposing a characteristic function.
    market_data : OptionMarketData
        Observed option surface.
    engine : SurfaceEngine
        Pricing engine (typically FFTEngine).

    Returns
    -------
    np.ndarray
        Model prices aligned 1-to-1 with `market_data.quotes`.
        Shape: (market_data.n_quotes,).

    Raises
    ------
    ValueError, FloatingPointError, RuntimeError
        Propagated from the engine when pricing fails at a given
        maturity. Callers performing calibration typically catch these
        and return a large penalty instead.
    """
    market = MarketEnvironment(
        spot=market_data.spot,
        rate=market_data.rate,
        dividend_yield=market_data.dividend_yield,
    )

    prices_out = np.empty(market_data.n_quotes)
    idx = 0
    for T in market_data.unique_maturities:
        quotes = market_data.quotes_for_maturity(float(T))
        strikes = np.array([q.strike for q in quotes])
        is_call_template = quotes[0].is_call

        template = VanillaOption(
            strike=float(strikes[0]),
            maturity=float(T),
            is_call=is_call_template,
        )
        prices = engine.price_strikes(template, model, market, strikes)

        disc_spot = market_data.spot * np.exp(-market_data.dividend_yield * float(T))
        for i, q in enumerate(quotes):
            price = prices[i]
            if q.is_call != is_call_template:
                # Put-call parity: C - P = S*exp(-qT) - K*exp(-rT)
                fwd_diff = disc_spot - q.strike * np.exp(-market_data.rate * float(T))
                price = price + fwd_diff if q.is_call else price - fwd_diff
            prices_out[idx] = max(float(price), 0.0)
            idx += 1

    return prices_out


def price_surface_safe(
    model,  # noqa: ANN001
    market_data: OptionMarketData,
    engine: SurfaceEngine,
    fail_value: float = np.inf,
) -> np.ndarray:
    """Non-raising variant: returns a vector of `fail_value` on engine errors.

    Useful inside calibration objectives/residual functions to avoid
    exception handling at every evaluation.
    """
    try:
        return price_surface(model, market_data, engine)
    except (ValueError, FloatingPointError, RuntimeError, ArithmeticError) as exc:
        # Silent fallback is intentional for optimization loops, but record
        # the cause at DEBUG level so real bugs are traceable when needed.
        _logger.debug("price_surface_safe fallback to %s: %r", fail_value, exc)
        return np.full(market_data.n_quotes, fail_value)


def price_residuals(
    model,  # noqa: ANN001
    market_data: OptionMarketData,
    engine: SurfaceEngine,
) -> np.ndarray:
    """Return the residual vector (model_price - market_price) per quote.

    Intended for `scipy.optimize.least_squares` (Levenberg-Marquardt).
    """
    return price_surface(model, market_data, engine) - market_data.market_prices


def price_surface_mc(
    simulator: MCTerminalSimulator,
    market_data: OptionMarketData,
    *,
    n_paths: int = 30_000,
    mc_seed: int = 12_345,
) -> np.ndarray:
    """Monte-Carlo counterpart of :func:`price_surface` for nonaffine models.

    Prices every quote of an option surface under a risk-neutral GARCH simulator
    (Duan NGARCH-Q, physical-GARCH model-implied surface) by simulating terminal
    prices once per maturity and reusing them across that maturity's strikes. A
    single ``Generator(mc_seed)`` drives the whole surface so the result is
    reproducible across optimizer evaluations (common random numbers) — the key
    to a smooth objective for the derivative-free / finite-difference solvers the
    NGARCH-Q calibrator uses.

    Returns model prices aligned 1-to-1 with ``market_data.quotes``.
    """
    rng = np.random.default_rng(mc_seed)
    prices_out = np.empty(market_data.n_quotes)
    idx = 0
    for T in market_data.unique_maturities:
        quotes = market_data.quotes_for_maturity(float(T))
        strikes = np.array([q.strike for q in quotes], dtype=float)
        is_call = np.array([q.is_call for q in quotes], dtype=bool)
        terminals = simulator.terminals(
            market_data.spot, market_data.rate, float(T),
            n_paths=n_paths, rng=rng, antithetic=True,
        )
        disc = float(np.exp(-market_data.rate * float(T)))
        calls = disc * np.maximum(terminals[:, None] - strikes[None, :], 0.0).mean(0)
        puts = disc * np.maximum(strikes[None, :] - terminals[:, None], 0.0).mean(0)
        prices_out[idx : idx + len(quotes)] = np.where(is_call, calls, puts)
        idx += len(quotes)
    return prices_out
