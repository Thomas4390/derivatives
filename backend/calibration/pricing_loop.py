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
- Mixed call/put quotes at the same maturity: calls and puts are priced by
  two separate FFT passes (calls with positive damping, puts with the direct
  negative-damping Carr-Madan transform), so deep-OTM puts stay accurate
  instead of being recovered by a catastrophic call+parity subtraction.
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


class NoiseReplayTerminalSimulator(MCTerminalSimulator, Protocol):
    """Terminal simulator whose innovations can be pre-drawn and replayed.

    Lets a calibration objective draw the (parameter-independent) common
    random numbers once and reuse them across optimizer evaluations instead
    of re-drawing the identical innovations at every call.
    """

    def draw_terminal_noise(
        self,
        t: float,
        *,
        n_paths: int,
        rng: np.random.Generator,
        antithetic: bool = ...,
    ) -> np.ndarray: ...

    def terminals(  # noqa: D102 -- widens the base protocol with ``noise``
        self,
        s0: float,
        r: float,
        t: float,
        *,
        n_paths: int,
        rng: np.random.Generator,
        antithetic: bool = ...,
        noise: np.ndarray | None = ...,
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
        is_call = np.array([q.is_call for q in quotes], dtype=bool)

        # Two FFT passes per maturity (calls and puts). The put pass uses the
        # engine's direct negative-damping transform, so deep-OTM puts are exact
        # rather than reconstructed by a catastrophic call+parity subtraction.
        prices_T = np.empty(len(quotes))
        for want_call in (True, False):
            mask = is_call == want_call
            if not mask.any():
                continue
            template = VanillaOption(
                strike=float(strikes[mask][0]),
                maturity=float(T),
                is_call=bool(want_call),
            )
            prices_T[mask] = engine.price_strikes(
                template, model, market, strikes[mask]
            )

        prices_out[idx : idx + len(quotes)] = np.maximum(prices_T, 0.0)
        idx += len(quotes)

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
    noise_cache: dict[tuple[int, int, float], np.ndarray] | None = None,
) -> np.ndarray:
    """Monte-Carlo counterpart of :func:`price_surface` for nonaffine models.

    Prices every quote of an option surface under a risk-neutral GARCH simulator
    (Duan NGARCH-Q, physical-GARCH model-implied surface) by simulating terminal
    prices once per maturity (with **antithetic variates**) and reusing them across
    that maturity's strikes. A single ``Generator(mc_seed)`` drives the whole
    surface so the result is reproducible across optimizer evaluations (common
    random numbers) — the key to a smooth objective for the derivative-free /
    finite-difference solvers the NGARCH-Q calibrator uses.

    ``noise_cache`` (opt-in; requires a :class:`NoiseReplayTerminalSimulator`)
    stores the per-maturity innovations across calls: the CRN design re-draws
    bit-identical normals at every objective evaluation, and for the GARCH-Q
    kernels that draw dominates the evaluation cost. Within one optimization the
    cache keys are stable, so every call is either all-misses (first evaluation,
    draws sequential from ``Generator(mc_seed)`` exactly as without the cache)
    or all-hits — the returned prices are bit-identical either way. The caller
    owns the dict and its lifetime (clear it to release ~8 bytes × n_paths/2 ×
    total steps).

    Returns model prices aligned 1-to-1 with ``market_data.quotes``.
    """
    rng = np.random.default_rng(mc_seed)
    prices_out = np.empty(market_data.n_quotes)
    idx = 0
    for T in market_data.unique_maturities:
        quotes = market_data.quotes_for_maturity(float(T))
        strikes = np.array([q.strike for q in quotes], dtype=float)
        is_call = np.array([q.is_call for q in quotes], dtype=bool)
        if noise_cache is not None:
            key = (mc_seed, n_paths, float(T))
            noise = noise_cache.get(key)
            if noise is None:
                noise = simulator.draw_terminal_noise(  # type: ignore[attr-defined]
                    float(T), n_paths=n_paths, rng=rng, antithetic=True
                )
                noise_cache[key] = noise
            terminals = simulator.terminals(
                market_data.spot,
                market_data.rate,
                float(T),
                n_paths=n_paths,
                rng=rng,
                antithetic=True,
                noise=noise,  # type: ignore[call-arg]
            )
        else:
            terminals = simulator.terminals(
                market_data.spot,
                market_data.rate,
                float(T),
                n_paths=n_paths,
                rng=rng,
                antithetic=True,
            )
        disc = float(np.exp(-market_data.rate * float(T)))
        calls = disc * np.maximum(terminals[:, None] - strikes[None, :], 0.0).mean(0)
        puts = disc * np.maximum(strikes[None, :] - terminals[:, None], 0.0).mean(0)
        prices_out[idx : idx + len(quotes)] = np.where(is_call, calls, puts)
        idx += len(quotes)
    return prices_out
