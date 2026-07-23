"""
Model-consistent pricing for the Simulation Explorer.

Single source of truth for every t=0 option premium: instead of pricing each
leg with a flat-volatility Black-Scholes proxy (the old behaviour, which made
the premium independent of the selected model), each premium is priced with the
engine that is *consistent with the model the user picked*:

    GBM                                       -> Black-Scholes analytical
    Heston / Merton / Bates / custom-with-FFT -> Carr-Madan FFT
    GARCH / NGARCH / GJR-GARCH                -> risk-neutral Monte-Carlo (cached)
    fallback (engine failure, T -> 0, custom-no-FFT) -> Black-Scholes(sigma)

Coherence rules enforced here
-----------------------------
* **Same parameters, two measures.** Pricing uses the risk-neutral drift
  ``r - q`` with the *same* diffusion parameters the simulator uses; only the
  drift differs from the real-world path simulation (``mu - q``). That asymmetry
  is intentional — it is what makes the expected P&L non-zero.
* **Engine by capability, not hardcoded lists.** Affine / custom models route to
  FFT only when ``model.supported_engines`` advertises it. The GARCH family is
  forced onto the risk-neutral MC path (never the AAD engine, which is excluded
  from the ``light`` branch).
* **Always finite, never raises.** Any engine failure falls back to
  Black-Scholes and the premium is clamped to be finite and non-negative, so the
  sidebar never shows a blank/NaN premium.

Author: Thomas Vaudescal
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import streamlit as st

from backend.utils.logging import get_logger
from backend.utils.math import bs_price as _bs_price
from config.constants import (
    CONTRACT_MULTIPLIER,
    EXOTIC_PREMIUM_MC_PATHS,
    GARCH_PREMIUM_MC_PATHS,
    GREEKS_MC_PATHS,
    MC_PRICING_SEED,
)
from services.pricing_service import (
    price_exotic_from_terminals,
    price_from_terminals,
    price_with_analytical,
    price_with_fft,
)
from services.simulation_service import _extract_model_params, get_initial_volatility

logger = get_logger(__name__)

# ── Human-readable method labels (English UI) ───────────────────────────
METHOD_BS = "Black-Scholes (analytic)"
METHOD_FFT = "FFT (Carr-Madan)"
METHOD_MC = "Risk-neutral Monte-Carlo"
METHOD_MC_PATH = "Risk-neutral Monte-Carlo (full path)"
METHOD_BS_FALLBACK = "Black-Scholes (fallback)"

_GARCH_KEYS = frozenset({"garch", "ngarch", "gjr_garch"})

# MC path counts / seed for the risk-neutral pricers live in config.constants
# (GARCH_PREMIUM_MC_PATHS, EXOTIC_PREMIUM_MC_PATHS, GREEKS_MC_PATHS,
# MC_PRICING_SEED) so the knobs are centralised. The fixed seed makes premiums
# reproducible across reruns (cached per distinct contract).

# Haug-catalog families exposed in the simulation app (priced through the
# Open/Closed registry behind ExoticAnalyticEngine, not the integer Numba
# kernels). Split by payoff shape:
#   * TERMINAL — payoff is a pure function of S(T); the GBM closed form equals
#     the terminal-payoff expectation exactly, and MC-on-terminal is exact too.
#   * BARRIER — path-dependent (the simulation monitors the realised path); under
#     GBM the registry closed form is the exact risk-neutral expectation of that
#     path payoff, so GBM uses it for the premium.
_HAUG_TERMINAL_EXOTICS = frozenset(
    {"powered", "capped_power", "log_contract", "log_option", "supershare"}
)
_HAUG_BARRIER_EXOTICS = frozenset(
    {"double_barrier", "discrete_barrier", "partial_barrier", "binary_barrier"}
)
_HAUG_EXOTICS = _HAUG_TERMINAL_EXOTICS | _HAUG_BARRIER_EXOTICS

# Exotic types whose payoff is a pure function of the terminal price: under GBM
# the closed form equals the terminal-payoff expectation exactly, so GBM may use
# it. Every other exotic (barrier/asian/lookback/chooser) is path-dependent and
# the simulation evaluates a TERMINAL approximation at maturity — so the premium
# must price that same terminal payoff by MC (for *all* models, GBM included) to
# stay consistent with the realised P&L. The Haug terminal members join the set;
# the Haug barriers stay path-dependent (GBM still uses their registry closed
# form — see :func:`price_exotic_consistent`).
_TERMINAL_EUROPEAN_EXOTICS = (
    frozenset({"digital", "asset_or_nothing", "power", "gap"}) | _HAUG_TERMINAL_EXOTICS
)

# Path-dependent exotic families the backend full-path ``ExoticMonteCarloEngine``
# can price model-consistently (they have a payoff in EXOTIC_MC_PAYOFF_REGISTRY).
# Under a non-GBM model these now price on the FULL simulated path so the premium
# matches the realised full-path P&L (the terminal approximation below ignored
# barrier breaches / path averaging). ``chooser`` and ``binary_barrier`` are not
# MC-registered yet and stay on the terminal fallback.
_BACKEND_MC_PATH_EXOTICS = frozenset(
    {
        "barrier",
        "asian",
        "lookback_floating",
        "lookback_fixed",
        "double_barrier",
        "discrete_barrier",
        "partial_barrier",
    }
)


def _finite_nonneg(x: float | None) -> float | None:
    """Return ``max(x, 0)`` if ``x`` is a finite number, else ``None``."""
    if x is None:
        return None
    xf = float(x)
    if not np.isfinite(xf):
        return None
    return max(xf, 0.0)


def _engine_for(model_key: str, params: dict[str, Any]) -> str:
    """Decide which engine will price this model — the single decision point.

    Used by both the actual pricer and :func:`pricing_method_label`, so the label
    shown in the UI can never disagree with the engine that is really used.
    """
    mk = model_key.lower()
    if mk == "gbm":
        return METHOD_BS
    if mk in _GARCH_KEYS:
        return METHOD_MC
    # Affine (Heston/Merton/Bates) and custom models: FFT iff the backend model
    # advertises the capability, otherwise the Black-Scholes fallback.
    try:
        from backend.core.result_types import PricingCapability
        from services.pricing_service import _create_model

        model = _create_model(model_key, params)
        if model is not None and PricingCapability.FFT in model.supported_engines:
            return METHOD_FFT
    except Exception:  # pragma: no cover - defensive, never break the sidebar
        logger.debug("engine capability probe failed for %s", model_key, exc_info=True)
    return METHOD_BS_FALLBACK


def _build_garch_rn_sim(model_key: str, params: dict[str, Any]):
    """Duan LRNVR (lambda = 0) risk-neutral GARCH simulator from physical params.

    The fitted asymmetry ``gamma`` is the risk-neutral ``gamma*`` (LRNVR shift is
    zero); ``from_physical_params`` owns the annualised->per-period scaling. The
    canonical keys (``sigma0``/``omega``/``alpha``/``beta``/``gamma``) are produced
    by :func:`_normalised_params` upstream.
    """
    from backend.simulation.models.garch_q import GARCHRiskNeutralSimulator

    gtype = model_key.lower()
    gamma = float(params.get("gamma", 0.0)) if gtype in ("ngarch", "gjr_garch") else 0.0
    return GARCHRiskNeutralSimulator.from_physical_params(
        gtype,
        omega_annualised=float(params.get("omega", 0.002)),
        alpha=float(params.get("alpha", 0.06)),
        beta=float(params.get("beta", 0.90)),
        gamma=gamma,
        sigma0=float(params.get("sigma0", 0.20)),
    )


def _garch_mc_price(
    model_key: str,
    params: dict[str, Any],
    spot: float,
    rate: float,
    q: float,
    strike: float,
    maturity: float,
    is_call: bool,
) -> float | None:
    """Risk-neutral MC premium for the GARCH family.

    Drifts the asset at ``r - q`` under Q, and discounts the payoff at the
    risk-free rate ``r`` (drift and discount are deliberately decoupled so a
    non-zero dividend yield is handled correctly).
    """
    try:
        sim = _build_garch_rn_sim(model_key, params)
        rng = np.random.default_rng(MC_PRICING_SEED)
        terminals = sim.terminals(
            spot,
            rate - q,
            maturity,
            n_paths=GARCH_PREMIUM_MC_PATHS,
            rng=rng,
            antithetic=True,
        )
        # price_from_terminals discounts at exp(-rate * T) — the risk-free rate,
        # NOT r - q — which is the correct discount for a (r - q)-drifted asset.
        res = price_from_terminals(terminals, strike, maturity, rate, is_call=is_call)
        return float(res["price"])
    except Exception:
        logger.warning("GARCH MC premium failed for %s", model_key, exc_info=True)
        return None


def _bs_fallback(
    spot: float,
    strike: float,
    rate: float,
    q: float,
    maturity: float,
    sigma: float,
    is_call: bool,
) -> float:
    """Flat-vol Black-Scholes premium, always finite and non-negative."""
    try:
        px = _bs_price(spot, strike, maturity, rate, sigma, is_call, dividend_yield=q)
        clean = _finite_nonneg(px)
        if clean is not None:
            return clean
    except Exception:
        logger.debug("BS fallback failed", exc_info=True)
    # Last resort: discounted intrinsic value.
    fwd = spot * np.exp((rate - q) * maturity)
    intrinsic = max(fwd - strike, 0.0) if is_call else max(strike - fwd, 0.0)
    return float(np.exp(-rate * maturity) * intrinsic)


@st.cache_data(ttl=600, show_spinner=False)
def _price_vanilla_cached(
    model_key: str,
    params_json: str,
    spot: float,
    rate: float,
    q: float,
    strike: float,
    maturity: float,
    is_call: bool,
) -> tuple[float, str]:
    """Cached dispatch: (premium, method label). Keyed on the full fingerprint."""
    params = json.loads(params_json)
    params["dividend_yield"] = q

    engine = _engine_for(model_key, params)
    price: float | None = None

    if engine == METHOD_BS:
        res = price_with_analytical(
            model_key, params, strike, maturity, spot, rate, is_call
        )
        price = _finite_nonneg(res["price"]) if res else None
    elif engine == METHOD_FFT:
        price = _finite_nonneg(
            price_with_fft(model_key, params, strike, maturity, spot, rate, is_call)
        )
    elif engine == METHOD_MC:
        price = _finite_nonneg(
            _garch_mc_price(model_key, params, spot, rate, q, strike, maturity, is_call)
        )

    if price is None:
        sigma = get_initial_volatility(model_key, params)
        return _bs_fallback(
            spot, strike, rate, q, maturity, sigma, is_call
        ), METHOD_BS_FALLBACK
    return price, engine


def price_vanilla_consistent(
    model_key: str,
    model_params: dict[str, Any],
    spot: float,
    rate: float,
    q: float,
    strike: float,
    maturity: float,
    is_call: bool,
) -> tuple[float, str]:
    """Model-consistent risk-neutral premium of a European vanilla option.

    Returns ``(premium, method_label)``. The premium is always a finite,
    non-negative float. ``method_label`` names the engine actually used (one of
    the ``METHOD_*`` constants), so the UI can surface it.
    """
    is_call = bool(is_call)
    spot, rate, q, strike = float(spot), float(rate), float(q), float(strike)
    maturity = float(maturity)

    # T -> 0: price is the (model-independent) intrinsic value; skip the engines.
    if maturity <= 1e-9:
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        return float(intrinsic), METHOD_BS

    norm = _normalised_params(model_key, model_params)
    params_json = json.dumps(norm, sort_keys=True)
    return _price_vanilla_cached(
        model_key, params_json, spot, rate, q, strike, maturity, is_call
    )


def _normalised_params(model_key: str, model_params: dict[str, Any]) -> dict[str, Any]:
    """Canonicalise UI params to backend keys and keep only hashable scalars.

    Routing through ``_extract_model_params`` resolves UI-specific aliases (e.g.
    NGARCH ``gamma_ngarch`` -> ``gamma``) so the engines below see canonical
    keys, and guarantees a deterministic JSON cache key.
    """
    try:
        norm = _extract_model_params(model_key, model_params)
    except Exception:
        norm = dict(model_params)
    return {k: v for k, v in norm.items() if isinstance(v, (int, float, bool, str))}


def pricing_method_label(
    model_key: str, model_params: dict[str, Any] | None = None
) -> str:
    """Human label for the engine that will price this model's vanilla premiums.

    Cheap (no pricing) — for a one-line UI caption. Mirrors the pricer's own
    decision via the shared :func:`_engine_for`, so label and engine never drift.
    """
    norm = _normalised_params(model_key, model_params or {})
    return _engine_for(model_key, norm)


# ════════════════════════════════════════════════════════════════════════
# Exotic premiums (terminal-only payoffs) — model-consistent risk-neutral MC
# ════════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=600, show_spinner=False)
def _rn_terminals(
    model_key: str,
    params_json: str,
    spot: float,
    rate: float,
    q: float,
    maturity: float,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Risk-neutral terminal prices S(T) under the selected model. Cached.

    The GARCH family uses the Duan LRNVR simulator; affine diffusions
    (Heston/Merton/Bates) and any other path simulator use the standard
    risk-neutral drift ``r - q`` with the same diffusion parameters — the same
    convention the vanilla FFT prices converge to.
    """
    params = json.loads(params_json)
    mk = model_key.lower()

    if mk in _GARCH_KEYS:
        sim = _build_garch_rn_sim(mk, params)
        rng = np.random.default_rng(seed)
        return np.asarray(
            sim.terminals(
                spot, rate - q, maturity, n_paths=n_paths, rng=rng, antithetic=True
            ),
            dtype=float,
        )

    from backend.simulation.factory import create_simulator

    sim_params = _extract_model_params(model_key, params)
    n_steps = max(int(round(maturity * 252)), 50)
    simulator = create_simulator(model_key, **sim_params)
    result = simulator.simulate_paths(
        s0=spot, mu=rate - q, t=maturity, n_paths=n_paths, n_steps=n_steps, seed=seed
    )
    return np.asarray(result.terminal_prices, dtype=float)


def _exotic_leg_dict(
    exotic_type: str,
    strike: float,
    is_call: bool,
    barrier: float,
    is_knock_in: bool,
    is_up: bool,
    payout: float,
    extra1: float,
    cap: float = 0.0,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Position dict consumed by calculate_exotic_payoff_at_expiry.

    ``extra1`` encodes the type-specific param (power exponent n, gap trigger K2;
    also the Haug powered/capped-power exponent); chooser's choice-time is
    irrelevant to the terminal payoff. ``cap`` is the capped-power ceiling.
    ``params`` carries any Haug family-specific named kwargs (supershare band
    ``lower_strike``/``upper_strike``, the barrier levels, ...); its keys are
    merged flat so both the terminal payoff and the registry factory see them.
    """
    leg: dict[str, Any] = {
        "instrument_class": exotic_type,
        "option_type": "call" if is_call else "put",
        "strike": float(strike),
        "barrier": float(barrier),
        "is_up": bool(is_up),
        "is_knock_in": bool(is_knock_in),
        "payout": float(payout),
        "power_n": (
            float(extra1)
            if exotic_type in ("power", "powered", "capped_power") and extra1
            else 2.0
        ),
        "gap_trigger": (
            float(extra1) if exotic_type == "gap" and extra1 else float(strike)
        ),
        "cap": float(cap),
    }
    if params:
        leg.update(params)
        leg["params"] = dict(params)
    return leg


def _gbm_exotic_price(
    exotic_type: str,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    is_call: bool,
    barrier: float,
    is_knock_in: bool,
    is_up: bool,
    rebate: float,
    payout: float,
    extra1: float,
    q: float,
    cap: float = 0.0,
    params: dict[str, Any] | None = None,
) -> float:
    """GBM closed-form exotic premium (also the fallback). Always finite, >= 0.

    Basic-8 types price through the integer Numba kernels; Haug-catalog types
    route through the registry (``cap`` / ``params`` supply their family-specific
    kwargs).
    """
    from streamlit_app.simulation.utils.exotic_loader import get_exotic_price_fn

    px = get_exotic_price_fn()(
        exotic_type=exotic_type,
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        sigma=sigma,
        is_call=is_call,
        barrier=barrier,
        is_knock_in=is_knock_in,
        is_up=is_up,
        rebate=rebate,
        payout=payout,
        extra1=extra1,
        dividend_yield=q,
        cap=cap,
        params=params,
    )
    return _finite_nonneg(px) or 0.0


def _backend_path_dependent_premium(
    model_key: str,
    model_params: dict[str, Any],
    q: float,
    *,
    exotic_type: str,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    is_call: bool,
    leg: dict[str, Any],
) -> float | None:
    """Full-path risk-neutral MC premium via the backend ``ExoticMonteCarloEngine``.

    A path-dependent exotic under a characteristic-function / affine model
    (Heston / Merton / Bates / custom) is priced on the FULL simulated path, so
    the premium is the risk-neutral expectation of the *same* payoff the realised
    P&L evaluates on each path — not the terminal approximation (which ignored
    barrier breaches and path averaging).

    Returns ``None`` (caller falls back to the terminal MC) for the GARCH family
    (its Duan-LRNVR simulator is not the plain backend model), a not-yet-registered
    family, or any failure. Backend exotic imports are lazy so the slim ``light``
    branch (which has no exotic module) never imports them.
    """
    try:
        from backend.core.market import MarketEnvironment
        from backend.engines.exotic.mc_engine import ExoticMonteCarloEngine
        from services.pricing_service import _create_model
        from streamlit_app.simulation.utils.exotic_loader import (
            get_exotic_instrument_fn,
        )

        model = _create_model(model_key, model_params)
        if model is None:  # GARCH family / unknown -> terminal fallback
            return None
        kwargs = {
            k: v
            for k, v in leg.items()
            if k not in ("instrument_class", "option_type", "strike")
        }
        instrument = get_exotic_instrument_fn()(
            exotic_type, strike, maturity, is_call, **kwargs
        )
        market = MarketEnvironment(spot=spot, rate=rate, dividend_yield=q)
        engine = ExoticMonteCarloEngine(
            n_paths=EXOTIC_PREMIUM_MC_PATHS, seed=MC_PRICING_SEED
        )
        if not engine.can_price(instrument, model):
            return None
        return _finite_nonneg(engine.price(instrument, model, market).price)
    except Exception:  # pragma: no cover - defensive, never break the sidebar
        logger.debug(
            "backend full-path exotic premium failed for %s/%s",
            model_key,
            exotic_type,
            exc_info=True,
        )
        return None


def price_exotic_consistent(
    model_key: str,
    model_params: dict[str, Any],
    q: float,
    *,
    exotic_type: str,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    is_call: bool,
    barrier: float = 0.0,
    is_knock_in: bool = False,
    is_up: bool = True,
    rebate: float = 0.0,
    payout: float = 1.0,
    extra1: float = 0.0,
    cap: float = 0.0,
    params: dict[str, Any] | None = None,
) -> tuple[float, str]:
    """Model-consistent premium of an exotic leg.

    GBM keeps the exact closed form for terminal-payoff exotics *and* the Haug
    barriers (whose registry closed form is the exact risk-neutral expectation of
    the realised-path payoff); every other model prices by risk-neutral
    Monte-Carlo on the terminal payoff (the same terminal payoff the simulation's
    P&L uses at maturity). ``cap``/``params`` carry the Haug families' extra
    kwargs. Returns ``(premium, method_label)``; the premium is always finite and
    non-negative.
    """
    spot, rate, q, strike = float(spot), float(rate), float(q), float(strike)
    maturity, sigma = float(maturity), float(sigma)
    is_call = bool(is_call)
    leg = _exotic_leg_dict(
        exotic_type,
        strike,
        is_call,
        barrier,
        is_knock_in,
        is_up,
        payout,
        extra1,
        cap=cap,
        params=params,
    )

    # T -> 0: terminal payoff at the current spot (model-independent).
    if maturity <= 1e-9:
        from streamlit_app.simulation.utils.exotic_loader import get_exotic_payoff_fn

        return float(max(get_exotic_payoff_fn()(spot, leg), 0.0)), METHOD_BS

    # GBM closed form is used for terminal-European exotics (where it equals the
    # terminal-payoff expectation) AND for the Haug barriers (where the registry
    # closed form is the exact risk-neutral expectation of the path payoff). Other
    # path-dependent exotics fall through to the terminal-MC path even under GBM.
    if model_key.lower() == "gbm" and (
        exotic_type in _TERMINAL_EUROPEAN_EXOTICS
        or exotic_type in _HAUG_BARRIER_EXOTICS
    ):
        price = _gbm_exotic_price(
            exotic_type,
            spot,
            strike,
            maturity,
            rate,
            sigma,
            is_call,
            barrier,
            is_knock_in,
            is_up,
            rebate,
            payout,
            extra1,
            q,
            cap=cap,
            params=params,
        )
        return price, METHOD_BS

    # Non-GBM path-dependent exotics: price on the FULL simulated path via the
    # backend ExoticMonteCarloEngine so the premium is the risk-neutral mean of
    # the same path payoff the realised P&L uses (barrier breaches, averaging,
    # running extremes) — fixing the terminal approximation below. GARCH and the
    # not-yet-registered families fall through to that terminal MC.
    if model_key.lower() != "gbm" and exotic_type in _BACKEND_MC_PATH_EXOTICS:
        full_path_px = _backend_path_dependent_premium(
            model_key,
            model_params,
            q,
            exotic_type=exotic_type,
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            is_call=is_call,
            leg=leg,
        )
        if full_path_px is not None:
            return full_path_px, METHOD_MC_PATH

    # All other cases: risk-neutral MC on the exact terminal payoff the
    # simulation pays at maturity — consistent across models and with the P&L.
    try:
        norm = _normalised_params(model_key, model_params)
        params_json = json.dumps(norm, sort_keys=True)
        terminals = _rn_terminals(
            model_key,
            params_json,
            spot,
            rate,
            q,
            maturity,
            EXOTIC_PREMIUM_MC_PATHS,
            MC_PRICING_SEED,
        )
        res = price_exotic_from_terminals(terminals, leg, maturity, rate)
        price = _finite_nonneg(res["price"])
        if price is not None:
            return price, METHOD_MC
    except Exception:
        logger.warning(
            "Exotic MC premium failed for %s/%s", model_key, exotic_type, exc_info=True
        )

    # Fallback: GBM closed form with the model's representative volatility.
    fb_sigma = sigma if sigma > 0 else get_initial_volatility(model_key, model_params)
    price = _gbm_exotic_price(
        exotic_type,
        spot,
        strike,
        maturity,
        rate,
        fb_sigma,
        is_call,
        barrier,
        is_knock_in,
        is_up,
        rebate,
        payout,
        extra1,
        q,
        cap=cap,
        params=params,
    )
    return price, METHOD_BS_FALLBACK


# ════════════════════════════════════════════════════════════════════════
# Model-consistent strategy value curve (for model-consistent Greeks)
# ════════════════════════════════════════════════════════════════════════
#
# Greeks are finite differences of the strategy *value* V(S) under the selected
# model. Every model here has multiplicative price dynamics, so the value is
# homogeneous of degree 1 in (S, K): V(S, K) = (S / S0) V(S0, K S0 / S). That
# lets the whole spot-grid curve be built from ONE pricing pass at a base spot —
# a single FFT call per leg (affine models) or one terminal simulation (GARCH) —
# instead of re-pricing at every grid node.


def _bumped_vol_params(
    model_key: str, params: dict[str, Any], dvol: float
) -> dict[str, Any]:
    """Shift the model's instantaneous volatility *level* by ``dvol`` (for vega).

    The bump is applied to the natural vol handle of each family: ``sigma`` for
    GBM/Merton, ``sqrt(v0)`` for Heston/Bates, ``sigma0`` for the GARCH family.
    """
    mk = model_key.lower()
    p = dict(params)
    if mk in ("gbm", "merton"):
        p["sigma"] = max(float(p.get("sigma", 0.20)) + dvol, 1e-6)
    elif mk in ("heston", "bates"):
        vol = max(np.sqrt(max(float(p.get("v0", 0.04)), 0.0)) + dvol, 1e-6)
        p["v0"] = vol**2
    elif mk in _GARCH_KEYS:
        p["sigma0"] = max(float(p.get("sigma0", 0.20)) + dvol, 1e-6)
    return p


def _fft_value_curve(
    model_key, params, strikes, is_calls, spot_grid, rate, q, maturity
):
    """Strategy-leg value curves over ``spot_grid`` via FFT + degree-1 homogeneity.

    Returns an ``(n_legs, n_spot)`` array of per-leg option values (unsigned,
    per contract), or ``None`` if the model is not FFT-priceable.
    """
    from backend.core.result_types import PricingCapability
    from backend.engines.fft_engine import FFTConfig, FFTEngine
    from backend.instruments.options import VanillaOption
    from services.pricing_service import _create_market, _create_model

    model = _create_model(model_key, params)
    if model is None or PricingCapability.FFT not in model.supported_engines:
        return None

    engine = FFTEngine(config=FFTConfig(alpha=1.5, n_fft=4096, eta=0.25))
    s0 = float(np.median(spot_grid))
    market = _create_market(s0, rate, q)
    curves = np.zeros((len(strikes), len(spot_grid)), dtype=float)
    for j, strike in enumerate(strikes):
        # Homogeneity: V(S, K) = (S / S0) * V(S0, K * S0 / S).
        eff_strikes = float(strike) * s0 / spot_grid
        template = VanillaOption(
            strike=float(eff_strikes[0]), maturity=maturity, is_call=bool(is_calls[j])
        )
        base = np.asarray(engine.price_strikes(template, model, market, eff_strikes))
        curves[j] = (spot_grid / s0) * base
    return curves


def _mc_value_curve(model_key, params, strikes, is_calls, spot_grid, rate, q, maturity):
    """Strategy-leg value curves over ``spot_grid`` via risk-neutral MC terminals.

    Used for the GARCH family (no FFT). One terminal simulation at the base spot,
    rescaled across the grid by homogeneity. Returns ``(n_legs, n_spot)`` or
    ``None`` on failure.
    """
    try:
        norm = _normalised_params(model_key, params)
        s0 = float(np.median(spot_grid))
        terminals = _rn_terminals(
            model_key,
            json.dumps(norm, sort_keys=True),
            s0,
            rate,
            q,
            maturity,
            GREEKS_MC_PATHS,
            MC_PRICING_SEED,
        )
        disc = float(np.exp(-rate * maturity))
        curves = np.zeros((len(strikes), len(spot_grid)), dtype=float)
        for i, s in enumerate(spot_grid):
            scaled = terminals * (s / s0)
            for j, strike in enumerate(strikes):
                if is_calls[j]:
                    payoff = np.maximum(scaled - float(strike), 0.0)
                else:
                    payoff = np.maximum(float(strike) - scaled, 0.0)
                curves[j, i] = disc * float(payoff.mean())
        return curves
    except Exception:
        logger.warning("MC value curve failed for %s", model_key, exc_info=True)
        return None


def _exotic_leg_value_curve(
    model_key: str,
    params: dict[str, Any],
    meta: dict[str, Any],
    strike: float,
    is_call: bool,
    spot_grid: np.ndarray,
    rate: float,
    q: float,
    maturity: float,
    step_ref_maturity: float | None = None,
) -> np.ndarray | None:
    """Model-consistent per-share value curve for ONE exotic leg over ``spot_grid``.

    Builds the *same* backend instrument the full-path premium prices (via the
    shared :func:`_exotic_leg_dict`), then dispatches **by family** (not by
    FFT-capability): a degree-1 path-dependent family (``_BACKEND_MC_PATH_EXOTICS``)
    prices on the single-matrix common-random-number ``ExoticMonteCarloEngine``
    (the homogeneity rescale is valid only for those); every other (terminal)
    exotic — including the non-degree-1 ones (powered, log, capped-power,
    supershare) — prices on the exact per-spot COS ``ExoticFourierEngine``, which
    makes no homogeneity assumption. The curve's finite differences are the
    model-consistent Greeks, the sensitivities of the same model/instrument the
    premium prices (the path-dependent level matches the displayed premium up to
    the differing MC path budget). Backend exotic imports are lazy so the slim
    ``light`` branch (no exotic module) never imports them.

    ``step_ref_maturity`` sizes the MC step grid by the *unbumped* maturity so the
    theta finite difference (which re-prices at maturity ± 1 day) stays
    common-random-number aligned; defaults to ``maturity``.

    Returns an ``(n_spot,)`` per-share value array, or ``None`` for an unsupported
    family (chooser / binary-barrier / no simulator / GARCH / a terminal exotic on
    a model with no characteristic function) so the caller falls back to the
    closed-form Greeks path.
    """
    try:
        from backend.core.market import MarketEnvironment
        from backend.engines.exotic._fourier_terminal import ExoticFourierEngine
        from backend.engines.exotic.mc_engine import ExoticMonteCarloEngine
        from services.pricing_service import _create_model
        from streamlit_app.simulation.utils.exotic_loader import (
            get_exotic_instrument_fn,
        )

        exotic_type = (meta or {}).get("instrument_class", "vanilla")
        model = _create_model(model_key, params)
        if model is None:  # GARCH family / unknown -> closed-form fallback
            return None

        # Mirror _exotic_leg_dict's extra1 convention (power exponent / gap
        # trigger) so the instrument matches the priced one exactly.
        if exotic_type in ("power", "powered", "capped_power"):
            extra1 = float(meta.get("power_n", 0.0) or 0.0)
        elif exotic_type == "gap":
            extra1 = float(meta.get("gap_trigger", 0.0) or 0.0)
        else:
            extra1 = 0.0
        leg = _exotic_leg_dict(
            exotic_type,
            float(strike),
            is_call,
            float(meta.get("barrier", 0.0)),
            bool(meta.get("is_knock_in", False)),
            bool(meta.get("is_up", True)),
            float(meta.get("payout", 1.0)),
            extra1,
            cap=float(meta.get("cap", 0.0)),
            params=dict(meta.get("params") or {}) or None,
        )
        kwargs = {
            k: v
            for k, v in leg.items()
            if k not in ("instrument_class", "option_type", "strike")
        }
        instrument = get_exotic_instrument_fn()(
            exotic_type, float(strike), float(maturity), is_call, **kwargs
        )

        grid = np.asarray(spot_grid, dtype=float)
        market = MarketEnvironment(
            spot=float(np.median(grid)), rate=rate, dividend_yield=q
        )
        # Enforce the homogeneity invariant by FAMILY, not by FFT-capability: the
        # single-matrix CRN rescale ``paths * S_i/s0`` is valid ONLY for the
        # degree-1 path-dependent families in ``_BACKEND_MC_PATH_EXOTICS``. Every
        # other (terminal) exotic — including the non-degree-1 ones (powered S^n,
        # log, capped-power, supershare) — must price on the exact per-spot COS
        # engine, which makes no homogeneity assumption; if the model has no
        # characteristic function (e.g. an MC-only custom model) we decline so the
        # caller uses the closed form rather than the invalid degree-1 rescale.
        if exotic_type in _BACKEND_MC_PATH_EXOTICS:
            engine = ExoticMonteCarloEngine(
                n_paths=GREEKS_MC_PATHS, seed=MC_PRICING_SEED
            )
            steps_ref = maturity if step_ref_maturity is None else step_ref_maturity
            n_steps = max(1, int(round(steps_ref * engine.steps_per_year)))
            curve = engine.price_curve(instrument, model, market, grid, n_steps=n_steps)
        else:
            fourier = ExoticFourierEngine()
            if not fourier.can_price(instrument, model):
                return None
            curve = fourier.price_curve(instrument, model, market, grid)
        if curve is None or not np.all(np.isfinite(curve)):
            return None
        return np.asarray(curve, dtype=float)
    except Exception:  # pragma: no cover - defensive, never break the surface
        logger.debug(
            "exotic value curve failed for %s/%s",
            model_key,
            (meta or {}).get("instrument_class"),
            exc_info=True,
        )
        return None


def strategy_value_curve(
    model_key: str,
    model_params: dict[str, Any],
    position_arrays: dict[str, Any],
    spot_grid: np.ndarray,
    rate: float,
    q: float,
    maturity: float,
    *,
    vol_bump: float = 0.0,
    rate_bump: float = 0.0,
    step_ref_maturity: float | None = None,
) -> np.ndarray | None:
    """Aggregate strategy value over a spot grid under the selected model.

    Returns the signed, quantity-weighted strategy value at each spot (premiums
    excluded — they are constants and do not affect Greeks). Vanilla legs price by
    homogeneity (FFT for affine models, MC terminals for the GARCH family); exotic
    legs price model-consistently on the backend full-path engine under a non-GBM
    characteristic-function model. Returns ``None`` — signalling the caller to fall
    back to the closed-form Greeks surface — when the model has no value-curve
    route, or when an exotic leg is present under GBM/GARCH or in an unsupported
    family (chooser / binary-barrier).
    """
    strikes = np.asarray(position_arrays.get("strikes", []), dtype=float)
    if strikes.size == 0:
        return None
    exotic_meta = position_arrays.get("exotic_metadata", [])
    classes = [
        (exotic_meta[j] or {}).get("instrument_class", "vanilla")
        if j < len(exotic_meta)
        else "vanilla"
        for j in range(strikes.size)
    ]
    has_exotic = any(c != "vanilla" for c in classes)

    mk = model_key.lower()
    # Exotic legs are model-consistent only under a non-GBM characteristic-function
    # model: GBM keeps the exact closed-form Greeks path, and the GARCH family's
    # Duan-LRNVR simulator is not the bare backend model — both decline here so the
    # caller falls back to the (correct) closed-form Greeks surface.
    if has_exotic and (mk == "gbm" or mk in _GARCH_KEYS):
        return None

    option_types = np.asarray(position_arrays.get("option_types", []), dtype=float)
    position_types = np.asarray(position_arrays.get("position_types", []), dtype=float)
    quantities = np.asarray(position_arrays.get("quantities", []), dtype=float)
    is_calls = option_types == 1.0

    params = _bumped_vol_params(model_key, model_params, vol_bump)
    r = rate + rate_bump
    spot_grid = np.asarray(spot_grid, dtype=float)

    leg_curves = np.zeros((strikes.size, spot_grid.size), dtype=float)

    # Vanilla legs: one homogeneity pass (FFT for affine models, MC terminals for
    # the GARCH family). Exotic legs: a model-consistent per-leg curve from the
    # backend engine (the same model/instrument the sidebar premium prices; the
    # path-dependent level matches it up to the differing MC path budget).
    vanilla_idx = [j for j in range(strikes.size) if classes[j] == "vanilla"]
    if vanilla_idx:
        v_strikes = strikes[vanilla_idx]
        v_calls = is_calls[vanilla_idx]
        if mk in _GARCH_KEYS:
            vcurves = _mc_value_curve(
                model_key, params, v_strikes, v_calls, spot_grid, r, q, maturity
            )
        else:
            vcurves = _fft_value_curve(
                model_key, params, v_strikes, v_calls, spot_grid, r, q, maturity
            )
        if vcurves is None:
            return None
        for k, j in enumerate(vanilla_idx):
            leg_curves[j] = vcurves[k]

    for j in range(strikes.size):
        if classes[j] == "vanilla":
            continue
        curve = _exotic_leg_value_curve(
            model_key,
            params,
            exotic_meta[j],
            float(strikes[j]),
            bool(is_calls[j]),
            spot_grid,
            r,
            q,
            maturity,
            step_ref_maturity=step_ref_maturity,
        )
        if curve is None:  # unsupported exotic family -> closed-form fallback
            return None
        leg_curves[j] = curve

    # Option leg values are per share; scale to share-equivalents (1 contract =
    # CONTRACT_MULTIPLIER shares) so the model-consistent Greeks (gradients of
    # this curve) combine with the stock leg (raw shares) on the same footing.
    signs = position_types * quantities * CONTRACT_MULTIPLIER
    value = (signs[:, None] * leg_curves).sum(axis=0)

    stock_qty = float(position_arrays.get("stock_quantity", 0.0))
    if stock_qty != 0.0:
        value = value + stock_qty * spot_grid
    return value
