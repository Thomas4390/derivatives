"""
Greeks Service for Simulation Explorer.

Computes Greeks surfaces (vs spot) for:
- Option strategies (vanilla + exotic) via analytical BS / exotic engine
- Structured products via MC bump-and-reprice

Author: Thomas Vaudescal
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from backend.engines.exotic_engine import (
    ASIAN_GEO,
    ASSET_OR_NOTHING,
    BARRIER,
    CHOOSER,
    DIGITAL,
    GAP,
    LOOKBACK_FIXED,
    LOOKBACK_FLOATING,
    POWER,
    exotic_greeks_surface,
)
from backend.engines.vectorized_bs import calculate_greeks_vectorized
from backend.utils.math import implied_volatility
from config.constants import CONTRACT_MULTIPLIER

_TYPE_MAP = {
    "barrier": BARRIER,
    "asian": ASIAN_GEO,
    "digital": DIGITAL,
    "lookback_floating": LOOKBACK_FLOATING,
    "lookback_fixed": LOOKBACK_FIXED,
    "chooser": CHOOSER,
    "asset_or_nothing": ASSET_OR_NOTHING,
    "power": POWER,
    "gap": GAP,
}

_GREEK_NAMES = ["price", "delta", "gamma", "vega", "theta", "rho"]
DISPLAY_GREEKS = ["delta", "gamma", "vega", "theta", "rho"]


def _haug_greeks_surface(
    inst_cls: str,
    spot_range: np.ndarray,
    strike: float,
    t: float,
    rate: float,
    sigma: float,
    is_call: bool,
    meta: dict,
) -> np.ndarray:
    """Per-spot registry Greeks for a Haug family that has no integer Numba kernel.

    Returns an ``(n, 6)`` array ``[price, delta, gamma, vega, theta, rho]`` built
    by differentiating the ``exotic_advanced`` instrument through the backend
    ``GreeksCalculator`` (via the options_greeks adapter). Band-only parameters
    (double-barrier corridor, supershare band) are *not* threaded into the Greeks
    instrument — the adapter builds them with default bands, the same limitation
    the options_greeks Greeks surface carries.
    """
    from streamlit_app.simulation.utils.exotic_loader import get_exotic_all_greeks_fn

    all_greeks = get_exotic_all_greeks_fn()
    p = meta.get("params", {}) or {}
    bar = float(p.get("barrier", meta.get("barrier", 0.0)))
    is_up = bool(p.get("is_up", meta.get("is_up", True)))
    is_ki = bool(p.get("is_knock_in", meta.get("is_knock_in", False)))
    extra1 = (
        float(meta.get("power_n", 0.0))
        if inst_cls in ("powered", "capped_power")
        else 0.0
    )
    cap = float(meta.get("cap", 0.0))
    opt_int = 1 if is_call else 0
    out = np.zeros((len(spot_range), 6))
    for i, s in enumerate(spot_range):
        g = all_greeks(
            float(s),
            float(strike),
            t,
            rate,
            sigma,
            opt_int,
            inst_cls,
            barrier=bar,
            is_up=is_up,
            is_knock_in=is_ki,
            extra1=extra1,
            cap=cap,
        )
        out[i] = np.where(np.isnan(g[:6]), 0.0, g[:6])
    return out


GREEK_TITLES = {
    "price": "Price (V)",
    "delta": "Delta (\u2202V/\u2202S)",
    "gamma": "Gamma (\u2202\u00b2V/\u2202S\u00b2)",
    "vega": "Vega (\u2202V/\u2202\u03c3)",
    "theta": "Theta (\u2202V/\u2202t)",
    "rho": "Rho (\u2202V/\u2202r)",
}


# ─────────────────────────────────────────────────────────────────────
# Internal: compute Greeks at a single DTE
# ─────────────────────────────────────────────────────────────────────


def _compute_greeks_at_dte(
    position_arrays: dict,
    spot_range: np.ndarray,
    rate: float,
    sigma: float,
    time_to_expiry: float,
    current_spot: float = 0.0,
    sigmas: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute aggregated Greeks across spot_range for a single time_to_expiry.

    ``sigmas`` (optional) supplies a per-leg volatility so each leg is greeked at
    its own implied vol (the practitioner-BS convention); when omitted the shared
    scalar ``sigma`` is used for every leg (flat-vol BS).
    """
    n = len(spot_range)
    strikes = position_arrays.get("strikes", np.array([]))
    option_types = position_arrays.get("option_types", np.array([]))
    position_types = position_arrays.get("position_types", np.array([]))
    quantities = position_arrays.get("quantities", np.array([]))
    exotic_metadata = position_arrays.get("exotic_metadata", [])
    stock_quantity = position_arrays.get("stock_quantity", 0.0)

    total = {name: np.zeros(n) for name in _GREEK_NAMES}
    t = max(time_to_expiry, 1e-6)

    for j in range(len(strikes)):
        meta = exotic_metadata[j] if j < len(exotic_metadata) else None
        is_exotic = meta and meta.get("instrument_class", "vanilla") != "vanilla"
        leg_sigma = float(sigmas[j]) if sigmas is not None else sigma

        if is_exotic and meta["instrument_class"] not in _TYPE_MAP:
            # Haug-catalog family (no integer Numba kernel) → registry Greeks.
            inst_cls = meta["instrument_class"]
            is_call = option_types[j] == 1.0
            leg_greeks = _haug_greeks_surface(
                inst_cls, spot_range, strikes[j], t, rate, leg_sigma, is_call, meta
            )
        elif is_exotic:
            inst_cls = meta["instrument_class"]
            opt_type = _TYPE_MAP[inst_cls]
            H = meta.get("barrier", 0.0) if inst_cls == "barrier" else 0.0
            ref_spot = (
                meta.get("ref_spot", current_spot)
                if inst_cls == "lookback_floating"
                else 0.0
            )
            extra1 = 0.0
            if inst_cls == "chooser":
                extra1 = meta.get("extra1", meta.get("choice_time_pct", 0.5) * t)
            elif inst_cls == "power":
                extra1 = meta.get("extra1", meta.get("power_n", 2.0))
            elif inst_cls == "gap":
                extra1 = meta.get("extra1", meta.get("gap_trigger", strikes[j]))

            is_call = option_types[j] == 1.0
            surface = exotic_greeks_surface(
                opt_type,
                spot_range,
                strikes[j],
                t,
                rate,
                0.0,
                leg_sigma,
                is_call,
                H,
                ref_spot,
                ref_spot,
                meta.get("is_knock_in", False),
                meta.get("is_up", True),
                meta.get("rebate", 0.0),
                meta.get("payout", 1.0),
                extra1,
            )
            leg_greeks = np.where(np.isnan(surface), 0.0, surface)
        else:
            opt_int = int(option_types[j]) if option_types[j] == 1.0 else 0
            full = calculate_greeks_vectorized(
                spot_range,
                strikes[j],
                t,
                rate,
                leg_sigma,
                opt_int,
            )
            leg_greeks = np.where(np.isnan(full[:, :6]), 0.0, full[:, :6])

        # Option Greeks are per share; scale to share-equivalents (1 contract =
        # CONTRACT_MULTIPLIER shares) so they combine with the stock leg's delta
        # (raw shares) on the same footing — see the options_greeks reference.
        scale = quantities[j] * position_types[j] * CONTRACT_MULTIPLIER
        for k, name in enumerate(_GREEK_NAMES):
            total[name] += leg_greeks[:, k] * scale

    if stock_quantity != 0:
        total["delta"] += stock_quantity

    return total


# ─────────────────────────────────────────────────────────────────────
# Public: single-DTE Greeks (kept for SP mode)
# ─────────────────────────────────────────────────────────────────────


def compute_strategy_greeks(
    position_arrays: dict,
    spot: float,
    rate: float,
    sigma: float,
    time_to_expiry: float,
    n_points: int = 200,
    spot_range_pct: float = 0.30,
) -> dict:
    """Compute Greeks across spot range at a single DTE."""
    spot_range = np.linspace(
        spot * (1 - spot_range_pct),
        spot * (1 + spot_range_pct),
        n_points,
    )
    total = _compute_greeks_at_dte(
        position_arrays,
        spot_range,
        rate,
        sigma,
        time_to_expiry,
        current_spot=spot,
    )
    current_idx = np.argmin(np.abs(spot_range - spot))
    return {
        "spot_range": spot_range,
        "greeks": {name: total[name] for name in DISPLAY_GREEKS},
        "current_values": {
            name: float(total[name][current_idx]) for name in DISPLAY_GREEKS
        },
        "price_curve": total["price"],
    }


# ─────────────────────────────────────────────────────────────────────
# Public: multi-DTE surface (for DTE slider + 3D)
# ─────────────────────────────────────────────────────────────────────


def _build_dte_range(max_dte_days: int, n_steps: int = 25) -> list[int]:
    """Build a DTE range adapted to the maturity."""
    if max_dte_days <= 1:
        return [1]
    step = max(1, max_dte_days // n_steps)
    values = list(range(1, max_dte_days + 1, step))
    if max_dte_days not in values:
        values.append(max_dte_days)
    return values


def compute_strategy_greeks_surface(
    position_arrays: dict,
    spot: float,
    rate: float,
    sigma: float,
    time_to_expiry: float,
    n_spot_points: int = 200,
    spot_range_pct: float = 0.30,
    n_dte_steps: int = 25,
    n_steps: int = 252,
) -> dict:
    """
    Compute Greeks across spot_range x DTE_range.

    The DTE axis indexes the simulation grid: 1 step = 1 "day", so the maximum
    days-to-expiration equals ``n_steps`` (the configured "Time Steps") and the
    maturity at ``d`` days is ``d * (time_to_expiry / n_steps)`` years.

    Returns dict with keys:
        spot_range, dte_values, greeks_by_dte, price_by_dte
    """
    spot_range = np.linspace(
        spot * (1 - spot_range_pct),
        spot * (1 + spot_range_pct),
        n_spot_points,
    )
    max_dte_days = max(1, n_steps)
    dt = time_to_expiry / n_steps
    dte_values = _build_dte_range(max_dte_days, n_dte_steps)

    greeks_by_dte = {}
    price_by_dte = {}

    for dte in dte_values:
        t = dte * dt
        total = _compute_greeks_at_dte(
            position_arrays,
            spot_range,
            rate,
            sigma,
            t,
            current_spot=spot,
        )
        greeks_by_dte[dte] = {name: total[name] for name in DISPLAY_GREEKS}
        price_by_dte[dte] = total["price"]

    return {
        "spot_range": spot_range,
        "dte_values": dte_values,
        "greeks_by_dte": greeks_by_dte,
        "price_by_dte": price_by_dte,
    }


def compute_strategy_greeks_surface_practitioner(
    position_arrays: dict,
    spot: float,
    rate: float,
    time_to_expiry: float,
    fallback_sigma: float,
    q: float = 0.0,
    n_spot_points: int = 121,
    spot_range_pct: float = 0.30,
    n_dte_steps: int = 12,
    n_steps: int = 252,
) -> dict:
    """Black-Scholes Greeks surface with a PER-LEG implied volatility.

    Each vanilla leg is greeked at its own implied vol, backed out from that
    leg's premium via the safeguarded-Newton BS inversion (the practitioner-BS
    convention), instead of a single flat vol shared across the whole position.
    Legs whose IV cannot be inverted (exotics, or premiums outside the
    no-arbitrage band) fall back to ``fallback_sigma``. The grid defaults match
    ``compute_strategy_greeks_surface_model`` so the two can be overlaid on a
    shared DTE slider.

    Returns the same dict shape as ``compute_strategy_greeks_surface``.
    """
    strikes = np.asarray(position_arrays.get("strikes", []), dtype=float)
    option_types = np.asarray(position_arrays.get("option_types", []), dtype=float)
    premiums = np.asarray(position_arrays.get("premiums", []), dtype=float)
    exotic_metadata = position_arrays.get("exotic_metadata", [])

    # Per-leg implied vol: vanilla legs invert BS from their own premium; exotic
    # or non-invertible legs keep the flat fallback vol.
    sigmas = np.full(len(strikes), float(fallback_sigma), dtype=float)
    for j in range(len(strikes)):
        meta = exotic_metadata[j] if j < len(exotic_metadata) else None
        is_exotic = meta and meta.get("instrument_class", "vanilla") != "vanilla"
        if is_exotic or j >= len(premiums):
            continue
        iv = implied_volatility(
            float(premiums[j]),
            spot,
            float(strikes[j]),
            time_to_expiry,
            rate,
            bool(option_types[j] == 1.0),
            q,
        )
        if np.isfinite(iv) and iv > 0.0:
            sigmas[j] = iv

    spot_range = np.linspace(
        spot * (1 - spot_range_pct), spot * (1 + spot_range_pct), n_spot_points
    )
    max_dte_days = max(1, n_steps)
    dt = time_to_expiry / n_steps
    dte_values = _build_dte_range(max_dte_days, n_dte_steps)

    greeks_by_dte: dict = {}
    price_by_dte: dict = {}
    for dte in dte_values:
        t = dte * dt
        total = _compute_greeks_at_dte(
            position_arrays,
            spot_range,
            rate,
            float(fallback_sigma),
            t,
            current_spot=spot,
            sigmas=sigmas,
        )
        greeks_by_dte[dte] = {name: total[name] for name in DISPLAY_GREEKS}
        price_by_dte[dte] = total["price"]

    return {
        "spot_range": spot_range,
        "dte_values": dte_values,
        "greeks_by_dte": greeks_by_dte,
        "price_by_dte": price_by_dte,
    }


# ─────────────────────────────────────────────────────────────────────
# Model-consistent Greeks surface (vanilla strategies)
# ─────────────────────────────────────────────────────────────────────


def compute_strategy_greeks_surface_model(
    position_arrays: dict,
    model_key: str,
    model_params: dict,
    spot: float,
    rate: float,
    q: float,
    time_to_expiry: float,
    n_spot_points: int = 121,
    spot_range_pct: float = 0.30,
    n_dte_steps: int = 12,
    n_steps: int = 252,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict | None:
    """Greeks surface under the SELECTED model (vanilla strategies only).

    Greeks are finite differences of the model-consistent strategy value
    ``strategy_value_curve`` (FFT-by-homogeneity for affine models, risk-neutral
    MC for the GARCH family): delta/gamma from the spot curve, vega/theta/rho from
    vol/maturity/rate bumps. Scales match the Black-Scholes surface (vega per 1%
    vol, theta per calendar day, rho per 1% rate) so the two are interchangeable.

    Returns ``None`` (caller falls back to the BS surface) when the strategy holds
    exotic legs or the model exposes no value-curve route.
    """
    from services.consistent_pricing import strategy_value_curve

    spot_range = np.linspace(
        spot * (1 - spot_range_pct), spot * (1 + spot_range_pct), n_spot_points
    )
    # DTE axis indexes the simulation grid: 1 step = 1 "day" (max = n_steps).
    max_dte_days = max(1, n_steps)
    dt = time_to_expiry / n_steps
    dte_values = _build_dte_range(max_dte_days, n_dte_steps)

    d_vol, d_rate = 0.005, 1e-4
    d_t = 1.0 / 365.0  # one calendar day (theta reporting scale, see THETA_SCALE)

    greeks_by_dte: dict = {}
    price_by_dte: dict = {}
    n_done, n_total = 0, len(dte_values)

    for dte in dte_values:
        t = dte * dt

        def value(vol_bump=0.0, rate_bump=0.0, mat=t, base_t=t):
            # ``step_ref_maturity=base_t`` keeps any exotic-leg MC step count fixed
            # across the theta maturity bumps (mat = t +- 1 day) so the bumped
            # curves stay common-random-number aligned (no step-count jump noise).
            return strategy_value_curve(
                model_key,
                model_params,
                position_arrays,
                spot_range,
                rate,
                q,
                mat,
                vol_bump=vol_bump,
                rate_bump=rate_bump,
                step_ref_maturity=base_t,
            )

        base = value()
        if base is None:
            return None  # exotic legs / unsupported model -> BS fallback

        v_up, v_dn = value(vol_bump=d_vol), value(vol_bump=-d_vol)
        r_up, r_dn = value(rate_bump=d_rate), value(rate_bump=-d_rate)
        t_up, t_dn = value(mat=t + d_t), value(mat=max(t - d_t, 1e-6))
        if any(c is None for c in (v_up, v_dn, r_up, r_dn, t_up, t_dn)):
            return None

        delta = np.gradient(base, spot_range)
        gamma = np.gradient(delta, spot_range)
        vega = (v_up - v_dn) / (2.0 * d_vol) / 100.0  # per 1% vol
        rho = (r_up - r_dn) / (2.0 * d_rate) / 100.0  # per 1% rate
        theta = -(t_up - t_dn) / (2.0 * d_t) / 365.0  # per calendar day

        greeks_by_dte[dte] = {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }
        price_by_dte[dte] = base
        n_done += 1
        if progress_callback:
            progress_callback(n_done / n_total, f"Model Greeks: {n_done}/{n_total} DTE")

    return {
        "spot_range": spot_range,
        "dte_values": dte_values,
        "greeks_by_dte": greeks_by_dte,
        "price_by_dte": price_by_dte,
    }


# ─────────────────────────────────────────────────────────────────────
# MC bump-and-reprice Greeks for structured products
# ─────────────────────────────────────────────────────────────────────


def compute_sp_greeks(
    sp_config: dict,
    model_key: str,
    sim_params: dict,
    n_points: int = 25,
    n_paths: int = 5000,
    n_dte_steps: int = 8,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    """
    Compute Greeks across spot_range x DTE_range for a structured product via MC.

    Returns dict with keys: spot_range, dte_values, greeks_by_dte
    (same format as compute_strategy_greeks_surface).
    """
    from backend.core.market import MarketEnvironment
    from backend.engines.structured_mc_engine import StructuredProductMCEngine
    from services.pricing_service import _create_model
    from services.simulation_service import get_initial_volatility
    from services.structured_product_service import _build_product

    spot = sim_params.get("spot_price", sim_params.get("spot", 100.0))
    rate = sim_params.get("risk_free_rate", 0.05)
    sigma = get_initial_volatility(model_key, sim_params)

    product_type = sp_config["product_type"]
    product_params = sp_config["product_params"]
    product = _build_product(product_type, product_params)

    model = _create_model(model_key, sim_params)
    if model is None:
        from backend.models.gbm import GBMModel

        model = GBMModel(sigma=sigma)

    n_steps_per_year = int(sim_params.get("n_steps", 252))
    engine = StructuredProductMCEngine(
        n_paths=n_paths,
        n_steps_per_year=n_steps_per_year,
        seed=42,
    )

    spot_range = np.linspace(spot * 0.70, spot * 1.30, n_points)

    # DTE axis indexes the simulation grid: 1 step = 1 "day", so the product's
    # full life spans ``maturity * n_steps_per_year`` days (the "Time Steps" unit).
    max_dte_days = max(1, int(round(product.maturity * n_steps_per_year)))

    # Minimum DTE must cover at least one observation period (expressed in steps)
    _freq_days = {
        "monthly": max(1, round(n_steps_per_year / 12)),
        "quarterly": max(1, round(n_steps_per_year / 4)),
        "semi_annual": max(1, round(n_steps_per_year / 2)),
        "annual": n_steps_per_year,
    }
    obs_freq = product_params.get("observation_frequency", "quarterly")
    min_dte = _freq_days.get(obs_freq, _freq_days["quarterly"])

    dte_values = [
        d for d in _build_dte_range(max_dte_days, n_dte_steps) if d >= min_dte
    ]
    if not dte_values:
        dte_values = [max_dte_days]

    greeks_by_dte = {}
    total_iters = n_points * len(dte_values)
    iter_count = 0

    for dte in dte_values:
        t_years = dte / n_steps_per_year
        adjusted_params = {**product_params, "maturity": t_years}
        adjusted_product = _build_product(product_type, adjusted_params)

        dte_greeks = {name: np.zeros(n_points) for name in DISPLAY_GREEKS}
        for i, s_i in enumerate(spot_range):
            market = MarketEnvironment(spot=s_i, rate=rate)
            gr = engine.greeks(adjusted_product, model, market)
            dte_greeks["delta"][i] = gr.delta
            dte_greeks["gamma"][i] = gr.gamma
            dte_greeks["vega"][i] = gr.vega
            dte_greeks["theta"][i] = gr.theta
            dte_greeks["rho"][i] = gr.rho
            iter_count += 1
            if progress_callback:
                progress_callback(
                    iter_count / total_iters,
                    f"MC Greeks: {iter_count}/{total_iters}",
                )
        greeks_by_dte[dte] = dte_greeks

    return {
        "spot_range": spot_range,
        "dte_values": dte_values,
        "greeks_by_dte": greeks_by_dte,
    }
