"""
Synthetic-data generation
============================

Bridges the sidebar's "true parameters + data config" choices to the
backend's pricing / simulation engines so the user can generate ground-
truth surfaces and return series to calibrate against.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.calibration.market_data import (
    HistoricalReturns,
    OptionMarketData,
    OptionQuote,
)
from backend.calibration.custom_calibrator import price_custom_surface
from backend.calibration.pricing_loop import price_surface_mc
from backend.core.market import MarketEnvironment
from backend.engines.fft_engine import FFTEngine
from backend.instruments.options import VanillaOption
from backend.models.bates import BatesModel
from backend.models.heston import HestonModel
from backend.models.heston_nandi import HestonNandiGARCHModel
from backend.models.merton import MertonModel
from backend.simulation.models.garch import GARCHSimulator
from backend.simulation.models.gjr_garch import GJRGARCHSimulator
from backend.simulation.models.ngarch import NGARCHSimulator
from backend.utils.logging import get_logger
from backend.utils.math import implied_volatility

from config.constants import (
    DEFAULT_GBM_SIGMA,
    FALLBACK_IV,
    MC_PATHS_TRUTH,
    MC_SEED,
    MIN_PRICE_FOR_IV,
    RN_GARCH_SURFACE_MODELS,
    SURFACE_WING_CLAMP_MAX_ITERS,
    SURFACE_WING_MIN_PRICE_FFT,
    SURFACE_WING_MIN_PRICE_MC,
    SURFACE_WING_MIN_WIDTH,
)

# x-axis label for the synthetic σ√T-standardized moneyness surface. The strike
# grid is per-maturity (K = F·exp(m·σ_T·√T)); the SHARED plot axis is m, the
# number of one-σ moves from the forward, so every maturity spans the same width.
_MONEYNESS_LABEL = "Moneyness  ln(K/F) / (σ√T)"

logger = get_logger(__name__)


@dataclass(frozen=True)
class _GBMStandIn:
    """Pickle-safe stand-in for the GBM "model" returned by the IV path."""

    sigma: float
    name: str = "GBM"


@dataclass(frozen=True)
class SurfaceData:
    market_data: OptionMarketData
    true_model: Any
    iv_grid: np.ndarray  # (n_T, n_K) implied vols
    strikes: np.ndarray  # (n_T, n_K) per-maturity dollar strikes (for pricing / hover)
    maturities: np.ndarray  # (n_T,)
    # Shared 1D plot axis (n_K). For synthetic surfaces this is the σ√T-standardized
    # moneyness m = ln(K/F)/(σ_T·√T); real-data surfaces reuse it as K/S₀. ``x_label``
    # / ``atm_x`` describe how the charts should title the axis and place the ATM line.
    moneyness: np.ndarray
    x_label: str = _MONEYNESS_LABEL
    atm_x: float = 0.0
    # Half-width (±σ√T) the caller *requested*. When a wing is clamped inward to
    # avoid dead (sub-floor) cells, ``moneyness`` becomes asymmetric and narrower
    # than this — the UI compares the two to caption the reduction. ``None`` on the
    # real-data path, which does not build an adaptive moneyness grid.
    requested_width: float | None = None


@dataclass(frozen=True)
class ReturnsData:
    market_data: HistoricalReturns
    true_params: dict[str, float]
    prices: np.ndarray
    log_returns: np.ndarray
    sample_volatility_ann: float


# --------------------------------------------------------------------- #
# Surface generation (Heston / Merton / Bates / GBM)
# --------------------------------------------------------------------- #


def _build_model(model_key: str, true_params: dict) -> Any:
    if model_key == "heston":
        return HestonModel(**true_params)
    if model_key == "merton":
        return MertonModel(**true_params)
    if model_key == "bates":
        return BatesModel(**true_params)
    if model_key == "heston_nandi":
        return HestonNandiGARCHModel(**true_params)
    if model_key == "ngarch_q":
        from backend.models.ngarch_q import NGARCHRiskNeutralModel

        return NGARCHRiskNeutralModel(**true_params)
    if model_key == "garch_q":
        from backend.models.ngarch_q import GARCHRiskNeutralModel

        return GARCHRiskNeutralModel(**true_params)
    if model_key == "gjr_q":
        from backend.models.ngarch_q import GJRGARCHRiskNeutralModel

        return GJRGARCHRiskNeutralModel(**true_params)
    raise NotImplementedError(f"Surface generation for {model_key} not supported.")


def _quote_skeleton(
    strikes_grid: np.ndarray,
    maturities: np.ndarray,
    is_call_grid: np.ndarray,
    spot: float,
    rate: float,
    dividend_yield: float,
) -> OptionMarketData:
    """Per-cell ``OptionMarketData`` skeleton for the (per-maturity) strike grid."""
    n_K = strikes_grid.shape[1]
    return OptionMarketData(
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        quotes=tuple(
            OptionQuote(
                strike=float(strikes_grid[i_T, j]),
                maturity=float(T),
                is_call=bool(is_call_grid[i_T, j]),
                market_price=1.0,
                implied_vol=None,
            )
            for i_T, T in enumerate(maturities)
            for j in range(n_K)
        ),
    )


def _price_mc_surface_grid(
    model: Any,
    strikes_grid: np.ndarray,
    maturities: np.ndarray,
    is_call_grid: np.ndarray,
    spot: float,
    rate: float,
    dividend_yield: float,
) -> np.ndarray:
    """Clean ground-truth prices for a nonaffine GARCH-Q surface, ``(n_T, n_K)``.

    Prices the truth with the **same** routine and common-random-number seed the
    calibrator uses (:func:`price_surface_mc` at :data:`MC_SEED`), so the true
    parameters sit at an *attainable* optimum of the calibration objective — there
    is no structural seed/grid mismatch between the synthetic surface and the fit,
    only a (shrinking) variance gap from the higher truth path count
    (:data:`MC_PATHS_TRUTH` vs the interactive ``MC_PATHS_INTERACTIVE``).
    """
    sim = model.create_simulator()
    skeleton = _quote_skeleton(
        strikes_grid, maturities, is_call_grid, spot, rate, dividend_yield
    )
    flat = price_surface_mc(sim, skeleton, n_paths=MC_PATHS_TRUTH, mc_seed=MC_SEED)
    return np.asarray(flat, dtype=float).reshape(len(maturities), strikes_grid.shape[1])


def _price_custom_surface_grid(
    model: Any,
    strikes_grid: np.ndarray,
    maturities: np.ndarray,
    is_call_grid: np.ndarray,
    spot: float,
    rate: float,
    dividend_yield: float,
) -> np.ndarray:
    """Ground-truth prices for a user-defined custom model, ``(n_T, n_K)``.

    Routes through :func:`price_custom_surface` — FFT when the model is affine,
    Monte-Carlo (shared CRN seed) otherwise — so the truth is priced exactly the
    way the calibrator will price its candidates (an attainable optimum).
    """
    skeleton = _quote_skeleton(
        strikes_grid, maturities, is_call_grid, spot, rate, dividend_yield
    )
    flat = price_custom_surface(
        model, skeleton, n_paths=MC_PATHS_TRUTH, mc_seed=MC_SEED
    )
    return np.asarray(flat, dtype=float).reshape(len(maturities), strikes_grid.shape[1])


def _price_fft_surface_grid(
    model: Any,
    market: MarketEnvironment,
    strikes_grid: np.ndarray,
    maturities: np.ndarray,
    is_call_grid: np.ndarray,
    spot: float,
    rate: float,
    dividend_yield: float,
) -> np.ndarray:
    """Exact ground-truth prices for an affine (FFT-pricable) surface, ``(n_T, n_K)``.

    Prices calls by FFT and converts to the out-of-the-money put by put-call parity
    (exact for these models). Each maturity carries its **own** strike row
    (``strikes_grid[i_T]``) on the shared moneyness axis.
    """
    engine = FFTEngine()
    n_K = strikes_grid.shape[1]
    grid = np.empty((len(maturities), n_K), dtype=float)
    for i_T, T in enumerate(maturities):
        row = strikes_grid[i_T]
        template = VanillaOption(strike=float(row[0]), maturity=float(T), is_call=True)
        calls = engine.price_strikes(template, model, market, row)
        puts = (
            calls
            - spot * np.exp(-dividend_yield * float(T))
            + row * np.exp(-rate * float(T))
        )
        grid[i_T, :] = np.where(is_call_grid[i_T], calls, puts)
    return grid


def _price_grid(
    model_key: str,
    model: Any,
    market: MarketEnvironment,
    strikes_grid: np.ndarray,
    maturities: np.ndarray,
    is_call_grid: np.ndarray,
    spot: float,
    rate: float,
    dividend_yield: float,
) -> np.ndarray:
    """Dispatch a (per-maturity) strike grid to the model's pricing route → ``(n_T, n_K)``."""
    if model_key == "custom":
        return _price_custom_surface_grid(
            model, strikes_grid, maturities, is_call_grid, spot, rate, dividend_yield
        )
    if model_key in RN_GARCH_SURFACE_MODELS:
        return _price_mc_surface_grid(
            model, strikes_grid, maturities, is_call_grid, spot, rate, dividend_yield
        )
    return _price_fft_surface_grid(
        model,
        market,
        strikes_grid,
        maturities,
        is_call_grid,
        spot,
        rate,
        dividend_yield,
    )


def _wing_threshold(model_key: str, model: Any) -> float:
    """Dead-cell price threshold for the wing clamp.

    Monte-Carlo-priced surfaces (the risk-neutral GARCH-Q trio and nonaffine
    custom models) need the wider :data:`SURFACE_WING_MIN_PRICE_MC` margin — a
    small true price can round to zero under a finite path count. Closed-form /
    FFT surfaces use the tighter :data:`SURFACE_WING_MIN_PRICE_FFT`.
    """
    if model_key in RN_GARCH_SURFACE_MODELS:
        return SURFACE_WING_MIN_PRICE_MC
    if model_key == "custom":
        from backend.core.result_types import PricingCapability

        if PricingCapability.FFT not in model.supported_engines:
            return SURFACE_WING_MIN_PRICE_MC
    return SURFACE_WING_MIN_PRICE_FFT


def _shrunk_wing(dead_moneyness: np.ndarray, *, side: int, current: float) -> float:
    """New half-width for one wing (asymmetric clamp).

    ``side`` is ``+1`` for the call wing (``m > 0``) or ``-1`` for the put wing
    (``m < 0``). The new bound is ``0.9 ×`` the innermost dead column on that side,
    floored at :data:`SURFACE_WING_MIN_WIDTH` and never widened past ``current``. If
    that side holds no dead cell, ``current`` is returned unchanged.
    """
    wing = dead_moneyness[dead_moneyness > 0.0] if side > 0 else -dead_moneyness[dead_moneyness < 0.0]
    if wing.size == 0:
        return current
    return min(current, max(SURFACE_WING_MIN_WIDTH, 0.9 * float(wing.min())))


def _price_grid_with_wing_clamp(
    *,
    moneyness_width: float,
    n_strikes: int,
    forwards: np.ndarray,
    sigma_t: np.ndarray,
    sqrt_t: np.ndarray,
    threshold: float,
    price_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Price the adaptive strike grid, shrinking dead wings until every cell clears
    ``threshold`` (or the iteration budget is spent).

    A *dead* cell prices below ``threshold`` — its sub-floor value carries no
    invertible Black-Scholes IV (a zeroed MC tail or a negative FFT put-call parity
    value) and would render as a NaN hole. The two wings are pulled in independently
    (asymmetric): each side's new half-width is ``0.9 ×`` the innermost dead column
    on that side, never below :data:`SURFACE_WING_MIN_WIDTH`. Up to
    :data:`SURFACE_WING_CLAMP_MAX_ITERS` re-prices (an MC re-price can resurface a
    boundary dead cell). If dead cells persist, the current NaN convention is kept
    and a warning is logged.

    Returns ``(moneyness, strikes_grid, is_call_grid, price_grid)``. ``moneyness``
    may become asymmetric (``linspace(-w_put, +w_call, n_strikes)``) but always
    spans 0, so the ATM line stays at ``m = 0``. With no dead cell the grid is the
    plain symmetric ``linspace(-moneyness_width, +moneyness_width, n_strikes)`` —
    bit-identical to the pre-clamp behaviour.
    """
    n_maturities = forwards.shape[0]
    w_put = w_call = float(moneyness_width)

    def price_at(w_put_: float, w_call_: float) -> tuple[np.ndarray, ...]:
        m = np.linspace(-w_put_, w_call_, n_strikes)
        # OTM convention: K ≥ F ⇔ m ≥ 0 → call, else put (same for every maturity).
        is_call = np.broadcast_to(m >= 0.0, (n_maturities, n_strikes)).copy()
        strikes = forwards[:, None] * np.exp(m[None, :] * sigma_t[:, None] * sqrt_t[:, None])
        return m, strikes, is_call, price_fn(strikes, is_call)

    moneyness, strikes_grid, is_call_grid, price_grid = price_at(w_put, w_call)
    for _ in range(SURFACE_WING_CLAMP_MAX_ITERS):
        dead_cols = np.asarray(price_grid < threshold).any(axis=0)  # (n_K,)
        if not dead_cols.any():
            return moneyness, strikes_grid, is_call_grid, price_grid
        dead_m = moneyness[dead_cols]
        new_w_call = _shrunk_wing(dead_m, side=+1, current=w_call)
        new_w_put = _shrunk_wing(dead_m, side=-1, current=w_put)
        if new_w_call == w_call and new_w_put == w_put:
            # The dead cells sit at the ATM core, not the wings — clamping can't help.
            break
        w_call, w_put = new_w_call, new_w_put
        moneyness, strikes_grid, is_call_grid, price_grid = price_at(w_put, w_call)

    n_dead = int(np.count_nonzero(np.asarray(price_grid) < threshold))
    if n_dead:
        logger.warning(
            "synthetic surface: %d cell(s) still price below the %.1e wing "
            "threshold after %d clamp iteration(s); keeping the NaN convention",
            n_dead,
            threshold,
            SURFACE_WING_CLAMP_MAX_ITERS,
        )
    return moneyness, strikes_grid, is_call_grid, price_grid


def generate_surface(
    model_key: str,
    true_params: dict,
    spot: float,
    rate: float,
    dividend_yield: float,
    n_strikes: int,
    n_maturities: int,
    moneyness_width: float,
    maturity_min: float,
    maturity_max: float,
    noise_std: float,
    seed: int,
    custom_model_class: Any = None,
) -> SurfaceData:
    """Generate an OptionMarketData surface under the chosen model.

    The strike grid is **adaptive**: each maturity ``T`` carries its own strikes
    ``K = F·exp(m · σ_T · √T)`` placed on a *shared* standardized-moneyness axis
    ``m = linspace(−moneyness_width, +moneyness_width, n_strikes)``, where ``σ_T`` is
    that maturity's ATM-forward implied vol (priced + inverted in a first pass). Every
    maturity therefore spans the same ±``moneyness_width`` standard deviations, so the
    surface fills for any vol level (low-vol GARCH-Q as well as high-vol jump models)
    and the smile is shown in a comparable frame across maturities.

    ``custom_model_class`` is required only when ``model_key == "custom"`` — the
    user-defined ``Model`` subclass to instantiate from ``true_params``. It is
    passed explicitly (not fetched from session state) so this service stays
    Streamlit-free and the surface is priced deterministically.
    """
    if model_key == "iv_gbm":
        sigma = float(true_params.get("sigma", DEFAULT_GBM_SIGMA))
        return _generate_gbm_surface(
            sigma=sigma,
            spot=spot,
            rate=rate,
            dividend_yield=dividend_yield,
            n_strikes=n_strikes,
            n_maturities=n_maturities,
            moneyness_width=moneyness_width,
            maturity_min=maturity_min,
            maturity_max=maturity_max,
            noise_std=noise_std,
            seed=seed,
        )

    if model_key == "custom":
        if custom_model_class is None:
            raise ValueError("custom_model_class is required for the custom generator.")
        model = custom_model_class(**true_params)
    else:
        model = _build_model(model_key, true_params)
    market = MarketEnvironment(spot=spot, rate=rate, dividend_yield=dividend_yield)

    maturities = np.linspace(maturity_min, maturity_max, n_maturities)  # (n_T,)
    forwards = spot * np.exp((rate - dividend_yield) * maturities)  # (n_T,)
    sqrt_t = np.sqrt(maturities)

    def price(strikes_grid: np.ndarray, is_calls: np.ndarray) -> np.ndarray:
        return _price_grid(
            model_key,
            model,
            market,
            strikes_grid,
            maturities,
            is_calls,
            spot,
            rate,
            dividend_yield,
        )

    # Pass 1 — per-maturity ATM-forward (K=F) vol σ_T sets each maturity's strike scale.
    atm_prices = price(forwards[:, None], np.ones((n_maturities, 1), dtype=bool))
    sigma_t = np.empty(n_maturities)
    for i in range(n_maturities):
        iv = _safe_iv(
            max(float(atm_prices[i, 0]), 1e-6),
            spot,
            float(forwards[i]),
            float(maturities[i]),
            rate,
            dividend_yield,
            is_call=True,
        )
        sigma_t[i] = (
            iv if (iv is not None and np.isfinite(iv) and iv > 1e-3) else FALLBACK_IV
        )

    # Per-maturity strikes on the shared moneyness axis, then the full pricing pass —
    # with the dead (sub-floor) wings clamped inward so the surface renders NaN-free.
    moneyness, strikes_grid, is_call_grid, price_grid = _price_grid_with_wing_clamp(
        moneyness_width=moneyness_width,
        n_strikes=n_strikes,
        forwards=forwards,
        sigma_t=sigma_t,
        sqrt_t=sqrt_t,
        threshold=_wing_threshold(model_key, model),
        price_fn=price,
    )

    rng = np.random.default_rng(seed)
    iv_grid = np.full((n_maturities, n_strikes), np.nan)
    quotes: list[OptionQuote] = []
    for i_T, T in enumerate(maturities):
        for j in range(n_strikes):
            k = float(strikes_grid[i_T, j])
            is_call = bool(is_call_grid[i_T, j])
            raw_p = float(price_grid[i_T, j])
            true_p = max(raw_p, 1e-6)
            if noise_std > 0.0:
                market_p = float(
                    max(true_p * (1.0 + rng.normal(scale=noise_std)), 1e-6)
                )
            else:
                market_p = true_p
            # A raw model price at/below the floor carries no model signal — a
            # zeroed MC tail (no path crossed a deep-OTM short-maturity strike) or
            # a negative FFT call→put parity value. Inverting the floored 1e-6 would
            # fabricate a model-independent IV (the same value for every model), so
            # leave the cell missing instead. With the moneyness grid this is rare.
            if raw_p > MIN_PRICE_FOR_IV:
                iv = _safe_iv(
                    market_p, spot, k, float(T), rate, dividend_yield, is_call=is_call
                )
            else:
                iv = None
            iv_grid[i_T, j] = iv if iv is not None else np.nan
            quotes.append(
                OptionQuote(
                    strike=k,
                    maturity=float(T),
                    is_call=is_call,
                    market_price=market_p,
                    implied_vol=iv,
                )
            )

    md = OptionMarketData(
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        quotes=tuple(quotes),
    )
    return SurfaceData(
        market_data=md,
        true_model=model,
        iv_grid=iv_grid,
        strikes=strikes_grid,
        maturities=maturities,
        moneyness=moneyness,
        requested_width=float(moneyness_width),
    )


def _generate_gbm_surface(
    sigma: float,
    spot: float,
    rate: float,
    dividend_yield: float,
    n_strikes: int,
    n_maturities: int,
    moneyness_width: float,
    maturity_min: float,
    maturity_max: float,
    noise_std: float,
    seed: int,
) -> SurfaceData:
    """Closed-form Black-Scholes surface — used for the IV/GBM model.

    σ is constant, so σ_T = σ at every maturity and the moneyness grid
    ``K = F·exp(m·σ·√T)`` is exact (no ATM pre-pass needed).
    """
    from backend.utils.math import bs_price

    rng = np.random.default_rng(seed)
    maturities = np.linspace(maturity_min, maturity_max, n_maturities)
    forwards = spot * np.exp((rate - dividend_yield) * maturities)
    sqrt_t = np.sqrt(maturities)

    def price_fn(strikes_grid: np.ndarray, is_call_grid: np.ndarray) -> np.ndarray:
        out = np.empty_like(strikes_grid)
        for i_T, T in enumerate(maturities):
            for j in range(strikes_grid.shape[1]):
                out[i_T, j] = bs_price(
                    spot,
                    float(strikes_grid[i_T, j]),
                    float(T),
                    rate,
                    sigma,
                    bool(is_call_grid[i_T, j]),
                    dividend_yield,
                )
        return out

    # Same dead-wing clamp as generate_surface — σ is constant, so σ_T = σ at every
    # maturity and no ATM pre-pass is needed, but a very low σ on a wide grid still
    # pushes deep-OTM strikes below the price floor.
    moneyness, strikes_grid, is_call_grid, price_grid = _price_grid_with_wing_clamp(
        moneyness_width=moneyness_width,
        n_strikes=n_strikes,
        forwards=forwards,
        sigma_t=np.full(n_maturities, sigma),
        sqrt_t=sqrt_t,
        threshold=SURFACE_WING_MIN_PRICE_FFT,
        price_fn=price_fn,
    )
    iv_grid = np.full((n_maturities, n_strikes), sigma)

    quotes: list[OptionQuote] = []
    for i_T, T in enumerate(maturities):
        for j in range(n_strikes):
            k = float(strikes_grid[i_T, j])
            is_call = bool(is_call_grid[i_T, j])  # OTM: call at/above the forward
            raw_p = float(price_grid[i_T, j])
            price = max(raw_p, 1e-6)
            if noise_std > 0.0:
                price = max(price * (1.0 + rng.normal(scale=noise_std)), 1e-6)
            # See generate_surface: don't invert a floored, sub-resolution price
            # into a fabricated IV (matters only for a very low σ on a wide grid).
            iv = (
                _safe_iv(
                    price, spot, k, float(T), rate, dividend_yield, is_call=is_call
                )
                if raw_p > MIN_PRICE_FOR_IV
                else None
            )
            if iv is None:
                # A dead cell that survived the clamp: leave it missing rather than
                # showing the constant σ where the quote carries no invertible IV.
                iv_grid[i_T, j] = np.nan
            quotes.append(
                OptionQuote(
                    strike=k,
                    maturity=float(T),
                    is_call=is_call,
                    market_price=float(price),
                    implied_vol=iv,
                )
            )

    md = OptionMarketData(
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        quotes=tuple(quotes),
    )
    return SurfaceData(
        market_data=md,
        true_model=_GBMStandIn(sigma=float(sigma)),
        iv_grid=iv_grid,
        strikes=strikes_grid,
        maturities=maturities,
        moneyness=moneyness,
        requested_width=float(moneyness_width),
    )


def _safe_iv(price, spot, k, T, rate, q, is_call: bool = True) -> float | None:
    """Invert Black-Scholes IV from a (positive) price.

    Pass ``is_call`` matching the option that was priced — invert the
    out-of-the-money side (call above the forward, put below) so the inversion
    stays well conditioned (an in-the-money price is dominated by intrinsic
    value, which the inversion cannot resolve).

    The backend's :func:`implied_volatility` returns ``nan`` on failure
    (non-convergence or a no-arbitrage violation) — we map that to ``None``
    so downstream code can treat it as a missing value uniformly.
    """
    if price <= 0 or T <= 0:
        return None
    try:
        iv = implied_volatility(
            price=float(price),
            spot=spot,
            strike=k,
            time_to_expiry=T,
            rate=rate,
            is_call=is_call,
            dividend_yield=q,
        )
    except (ValueError, RuntimeError):
        logger.debug(
            "implied-vol inversion failed for a synthetic quote", exc_info=True
        )
        return None
    # Backend returns nan on failure / arbitrage violation; also reject
    # pathological values (nan comparisons are always False, so test it).
    if iv is None or not np.isfinite(iv) or iv <= 1e-3 or iv >= 4.99:
        return None
    return float(iv)


# --------------------------------------------------------------------- #
# Returns generation (GARCH family)
# --------------------------------------------------------------------- #


def _garch_persistence(garch_type: str, true_params: dict) -> float:
    """Variance persistence for one GARCH variant (matches the backend
    simulators' ``persistence`` property).

    - garch:      α + β
    - ngarch:     α·(1 + γ²) + β   (leverage term)
    - gjr_garch:  α + ½γ + β

    A flat α+β for all three mis-seeds σ₀ for NGARCH/GJR and lets their
    explosive regions slip past the stationarity guard.
    """
    alpha = float(true_params["alpha"])
    beta = float(true_params["beta"])
    if garch_type == "ngarch":
        gamma = float(true_params.get("gamma", 0.5))
        return alpha * (1.0 + gamma**2) + beta
    if garch_type == "gjr_garch":
        gamma = float(true_params.get("gamma", 0.04))
        return alpha + 0.5 * gamma + beta
    return alpha + beta


def _seed_sigma0_ann(
    garch_type: str, true_params: dict, annualization_factor: int
) -> float:
    """Annualised initial vol = the variant's long-run vol (√(ω/(1−persistence)))."""
    omega_per = float(true_params["omega"])
    persistence = _garch_persistence(garch_type, true_params)
    long_run_var_per = omega_per / max(1e-12, 1.0 - persistence)
    return float(np.sqrt(long_run_var_per * annualization_factor))


def generate_returns(
    garch_type: str,
    true_params: dict,
    n_periods: int,
    annualization_factor: int,
    spot: float,
    drift: float,
    seed: int,
) -> ReturnsData:
    """Simulate a single price path under the chosen GARCH variant.

    The backend ``GARCHSimulator`` works at *annualised* scale (its variance
    recursion uses annualised σ²), but the calibrator sees per-period log
    returns and recovers per-period ω/α/β.  The user enters parameters at
    the per-period scale (e.g. ω=2e-6 daily), so we convert
    ``ω_ann = ω_per × annualization_factor`` before passing to the
    simulator.  α and β are dimensionless and stay unchanged.
    """
    omega_per = float(true_params["omega"])
    alpha = float(true_params["alpha"])
    beta = float(true_params["beta"])

    # Variant-specific persistence: the guard and σ₀ seed must use the NGARCH
    # leverage / GJR asymmetry terms, not a flat α+β (which mis-seeds σ₀ and
    # lets explosive NGARCH/GJR configs slip past to a raw backend ValueError).
    persistence = _garch_persistence(garch_type, true_params)
    if persistence >= 1.0:
        raise ValueError(
            f"GARCH stationarity violated (persistence={persistence:.3f} ≥ 1). "
            f"Lower α / β / γ in the True Parameters panel."
        )

    omega_ann = omega_per * annualization_factor
    sigma0_ann = _seed_sigma0_ann(garch_type, true_params, annualization_factor)

    if garch_type == "garch":
        sim = GARCHSimulator(
            sigma0=sigma0_ann,
            omega=omega_ann,
            alpha=alpha,
            beta=beta,
        )
    elif garch_type == "ngarch":
        sim = NGARCHSimulator(
            sigma0=sigma0_ann,
            omega=omega_ann,
            alpha=alpha,
            beta=beta,
            gamma=float(true_params.get("gamma", 0.5)),
        )
    else:  # gjr_garch
        sim = GJRGARCHSimulator(
            sigma0=sigma0_ann,
            omega=omega_ann,
            alpha=alpha,
            beta=beta,
            gamma=float(true_params.get("gamma", 0.04)),
        )

    t_years = n_periods / annualization_factor
    res = sim.simulate_paths(
        s0=float(spot),
        mu=float(drift),
        t=t_years,
        n_paths=1,
        n_steps=int(n_periods),
        seed=int(seed),
    )
    prices = np.asarray(res.price_paths[0, :])
    log_returns = np.diff(np.log(prices))

    md = HistoricalReturns(
        log_returns=log_returns,
        frequency="daily",
        annualization_factor=annualization_factor,
    )
    sample_vol = float(np.std(log_returns) * np.sqrt(annualization_factor))
    return ReturnsData(
        market_data=md,
        true_params=true_params,
        prices=prices,
        log_returns=log_returns,
        sample_volatility_ann=sample_vol,
    )
