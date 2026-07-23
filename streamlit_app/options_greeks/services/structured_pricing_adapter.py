"""
Structured Products Pricing Adapter.

Bridge between Streamlit UI and the backend StructuredProductMCEngine.
Follows the same pattern as exotic_pricing_adapter.py: factory functions
+ cached wrappers for Streamlit session management.

Surface calculation:
  - Computes MC on a reduced grid (SP_DTE_RANGE × SP_IV_RANGE × SP_SPOT_RANGE_POINTS)
  - Fills the standard grid (DTE_RANGE × IV_RANGE) via nearest-neighbor
  - Returns `all_data` in the exact same format as `calculate_all_surfaces()`
    so that existing chart components work without modification.
"""

from __future__ import annotations

import json

import numpy as np
import streamlit as st

from backend.core.market import MarketEnvironment
from backend.engines.structured_mc_engine import StructuredProductMCEngine
from backend.instruments.structured.products import (
    Autocallable,
    CapitalProtectedNote,
    ReverseConvertible,
)
from backend.models.gbm import GBMModel
from backend.models.heston import HestonModel
from backend.models.vol_bump import create_vol_bumped_pair
from backend.utils.constants.monte_carlo import DEFAULT_SPOT_BUMP, DEFAULT_VOL_BUMP
from backend.utils.constants.time import CALENDAR_DAYS_PER_YEAR
from backend.utils.logging import get_logger

from config.constants import (
    DEFAULT_SCENARIO_PATHS,
    GREEK_NAMES,
    SP_DTE_RANGE,
    SP_IV_RANGE,
    SP_SURFACE_PATHS,
)

logger = get_logger(__name__)


# =============================================================================
# Factory Functions
# =============================================================================


def build_product(product_type: str, params: dict):
    """Instantiate a StructuredProduct from UI parameters."""
    if product_type == "cpn":
        return CapitalProtectedNote(
            notional_=params["notional"],
            maturity_=params["maturity"],
            participation_rate=params["participation_rate"],
            cap=params.get("cap"),
            observation_frequency=params["observation_frequency"],
            protection_level=params.get("protection_level", 1.0),
        )
    if product_type == "reverse_convertible":
        return ReverseConvertible(
            notional_=params["notional"],
            maturity_=params["maturity"],
            coupon_rate=params["coupon_rate"],
            barrier=params["barrier"],
            barrier_monitoring=params.get("barrier_monitoring", "continuous"),
            observation_frequency=params["observation_frequency"],
        )
    if product_type == "autocallable":
        return Autocallable(
            notional_=params["notional"],
            maturity_=params["maturity"],
            coupon_rate=params["coupon_rate"],
            autocall_trigger=params["autocall_trigger"],
            coupon_barrier=params["coupon_barrier"],
            ki_barrier=params["ki_barrier"],
            memory_coupon=params.get("memory_coupon", True),
            barrier_monitoring=params.get("barrier_monitoring", "continuous"),
            observation_frequency=params["observation_frequency"],
        )
    raise ValueError(f"Unknown product type: {product_type}")


def build_model(model_type: str, model_params: dict):
    """Instantiate a Model from UI parameters."""
    if model_type == "gbm":
        return GBMModel(sigma=model_params["sigma"])
    if model_type == "heston":
        return HestonModel(
            v0=model_params["v0"],
            kappa=model_params["kappa"],
            theta=model_params["theta"],
            alpha=model_params["alpha"],
            rho=model_params["rho"],
        )
    raise ValueError(f"Unknown model type: {model_type}")


def build_market(
    spot: float, rate: float, dividend_yield: float = 0.0
) -> MarketEnvironment:
    """Create a MarketEnvironment from UI inputs."""
    return MarketEnvironment(spot=spot, rate=rate, dividend_yield=dividend_yield)


# =============================================================================
# Cached Pricing Functions (sidebar — precise pricing)
# =============================================================================


@st.cache_data(ttl=300)
def price_structured_product(
    product_type: str,
    product_params_json: str,
    model_type: str,
    model_params_json: str,
    spot: float,
    rate: float,
    dividend_yield: float,
    n_paths: int,
    seed: int | None,
) -> dict:
    """
    Price a structured product via MC engine.

    Parameters are serialized as JSON for cache hashability.
    Returns a plain dict for Streamlit cache compatibility.
    """
    product_params = json.loads(product_params_json)
    model_params = json.loads(model_params_json)

    product = build_product(product_type, product_params)
    model = build_model(model_type, model_params)
    market = build_market(spot, rate, dividend_yield)

    engine = StructuredProductMCEngine(n_paths=n_paths, seed=seed)
    result = engine.price_structured(product, model, market)

    return {
        "fair_value": result.fair_value,
        "price": result.price,
        "notional": result.notional,
        "error": result.error,
        "bond_floor": result.bond_floor,
        "option_value": result.option_value,
        "expected_coupon": result.expected_coupon,
        "autocall_probability": result.autocall_probability,
        "capital_loss_probability": result.capital_loss_probability,
        "expected_return": result.expected_return,
        "worst_case_return": result.worst_case_return,
        "best_case_return": result.best_case_return,
    }


@st.cache_data(ttl=300)
def compute_greeks(
    product_type: str,
    product_params_json: str,
    model_type: str,
    model_params_json: str,
    spot: float,
    rate: float,
    dividend_yield: float,
    n_paths: int,
    seed: int | None,
) -> dict:
    """
    Compute bump-and-reprice Greeks for a structured product.

    Uses n_paths // 5 for speed (7 reprices in bump-and-reprice).
    """
    product_params = json.loads(product_params_json)
    model_params = json.loads(model_params_json)

    product = build_product(product_type, product_params)
    model = build_model(model_type, model_params)
    market = build_market(spot, rate, dividend_yield)

    greek_paths = max(n_paths // 5, 5_000)
    engine = StructuredProductMCEngine(n_paths=greek_paths, seed=seed)
    result = engine.greeks(product, model, market)

    return {
        "delta": result.delta,
        "gamma": result.gamma,
        "vega": result.vega,
        "rho": result.rho,
        "theta": result.theta,
    }


@st.cache_data(ttl=300)
def compute_scenario_analysis(
    product_type: str,
    product_params_json: str,
    model_type: str,
    model_params_json: str,
    spot: float,
    rate: float,
    dividend_yield: float,
    spot_range_tuple: tuple[float, ...],
    seed: int | None,
) -> dict:
    """
    Compute scenario analysis across a range of spot prices.

    Uses DEFAULT_SCENARIO_PATHS (10k) for speed.
    """
    product_params = json.loads(product_params_json)
    model_params = json.loads(model_params_json)

    product = build_product(product_type, product_params)
    model = build_model(model_type, model_params)
    market = build_market(spot, rate, dividend_yield)

    spot_range = np.array(spot_range_tuple)
    engine = StructuredProductMCEngine(n_paths=DEFAULT_SCENARIO_PATHS, seed=seed)
    result = engine.scenario_analysis(product, model, market, spot_range)

    return {
        "spots": result["spots"].tolist(),
        "fair_values": result["fair_values"].tolist(),
        "prices": result["prices"].tolist(),
        "deltas": result["deltas"].tolist(),
    }


# =============================================================================
# Surface Calculation — Main Entry Point
# =============================================================================


@st.cache_data(ttl=600, show_spinner=False)
def calculate_structured_surfaces(
    product_type: str,
    product_params_json: str,
    model_type: str,
    model_params_json: str,
    spot: float,
    rate: float,
    dividend_yield: float,
    spot_range_tuple: tuple[float, ...],
    entry_price: float,
    seed: int | None,
) -> dict:
    """
    Calculate all P&L and Greeks surfaces for a structured product.

    Computes MC on a reduced grid (SP_DTE_RANGE × SP_IV_RANGE × n_spots),
    then fills the standard grid via nearest-neighbor mapping.

    Returns `all_data` in the same format as `calculate_all_surfaces()`:
      - pnl_data: {"{DTE}_{IV}": np.ndarray, "expiry": np.ndarray}
      - greeks_data: {"{DTE}_{IV}": {14 Greek names → np.ndarray}}
      - breakeven_result, max_profit_display, max_loss_display, etc.
    """
    product_params = json.loads(product_params_json)
    model_params = json.loads(model_params_json)
    spot_range = np.array(spot_range_tuple)

    # Cap DTE range to product maturity
    max_dte = int(product_params["maturity"] * 365)
    dte_range = [d for d in SP_DTE_RANGE if d <= max_dte]
    if not dte_range:
        dte_range = [SP_DTE_RANGE[0]]

    # ------------------------------------------------------------------
    # 1. Compute on reduced grid
    # ------------------------------------------------------------------
    reduced_pnl = {}  # {(dte, iv): np.ndarray(n_spots)}
    reduced_greeks = {}  # {(dte, iv): dict of 14 Greek arrays}

    for dte in dte_range:
        for iv in SP_IV_RANGE:
            pnl_arr, greeks_dict = _compute_combo(
                product_type,
                product_params,
                model_type,
                model_params,
                iv / 100.0,
                spot,
                rate,
                dividend_yield,
                dte / 365.0,
                spot_range,
                entry_price,
                seed,
            )
            reduced_pnl[(dte, iv)] = pnl_arr
            reduced_greeks[(dte, iv)] = greeks_dict

    # ------------------------------------------------------------------
    # 2. Use SP_DTE_RANGE x SP_IV_RANGE directly as grid keys
    # ------------------------------------------------------------------
    pnl_data = {}
    greeks_data = {}

    for dte in dte_range:
        for iv in SP_IV_RANGE:
            key = f"{dte}_{iv}"
            pnl_data[key] = reduced_pnl[(dte, iv)]
            greeks_data[key] = reduced_greeks[(dte, iv)]

    # ------------------------------------------------------------------
    # 3. Expiry P&L — use a dense grid for the analytical payoff curve,
    #    then sample back onto the sparse MC grid for chart compatibility.
    # ------------------------------------------------------------------
    dense_spot = np.linspace(spot_range[0], spot_range[-1], 200)
    dense_expiry = _compute_expiry_pnl(
        product_type,
        product_params,
        dense_spot,
        entry_price,
        current_spot=spot,
    )
    # Interpolate onto sparse MC grid for consistent array sizes
    expiry_pnl = np.interp(spot_range, dense_spot, dense_expiry)
    pnl_data["expiry"] = expiry_pnl
    # Store dense version for high-fidelity expiry curve
    pnl_data["expiry_dense_x"] = dense_spot
    pnl_data["expiry_dense_y"] = dense_expiry

    # ------------------------------------------------------------------
    # 4. Breakeven from P&L curve
    # ------------------------------------------------------------------
    from .pricing_adapter import BreakevenResult

    breakeven_points = []
    for i in range(len(expiry_pnl) - 1):
        if expiry_pnl[i] * expiry_pnl[i + 1] < 0:
            frac = abs(expiry_pnl[i]) / (abs(expiry_pnl[i]) + abs(expiry_pnl[i + 1]))
            bp = spot_range[i] + frac * (spot_range[i + 1] - spot_range[i])
            breakeven_points.append(float(bp))

    max_profit_idx = int(np.argmax(expiry_pnl))
    max_loss_idx = int(np.argmin(expiry_pnl))

    breakeven_result = BreakevenResult(
        breakeven_points=breakeven_points,
        max_profit=float(expiry_pnl[max_profit_idx]),
        max_profit_spot=float(spot_range[max_profit_idx]),
        max_loss=float(expiry_pnl[max_loss_idx]),
        max_loss_spot=float(spot_range[max_loss_idx]),
    )

    return {
        "pnl_data": pnl_data,
        "greeks_data": greeks_data,
        "breakeven_result": breakeven_result,
        "unlimited_profit": False,
        "unlimited_loss": True,  # structured products can lose capital
        "max_profit_display": float(np.max(expiry_pnl)),
        "max_loss_display": float(np.min(expiry_pnl)),
        "has_exotic_legs": False,
        "dte_range": list(dte_range),
        "iv_range": list(SP_IV_RANGE),
    }


# =============================================================================
# 3D Surface Adapters
# =============================================================================


def calculate_structured_greeks_3d_dte(
    sp_config_json: str,
    spot_range: np.ndarray,
    dte_range: np.ndarray,
    risk_free_rate: float,
    base_iv: float,
    greek_index: int = 1,
) -> np.ndarray:
    """
    Calculate 3D Greek surface varying spot × DTE for structured products.

    Same signature as calculate_portfolio_greeks_3d_dte (called by surface_3d.py).
    Returns 2D array [len(spot_range), len(dte_range)].
    """
    config = json.loads(sp_config_json)
    product_type = config["product_type"]
    product_params = json.loads(config["product_params_json"])
    model_type = config["model_type"]
    model_params = json.loads(config["model_params_json"])
    spot = config["spot"]
    dividend_yield = config.get("dividend_yield", 0.0)
    seed = config.get("seed")
    entry_price = config.get("entry_price", 0.0)

    greek_name = GREEK_NAMES[greek_index]
    result = np.zeros((len(spot_range), len(dte_range)))

    for j, dte in enumerate(dte_range):
        iv_decimal = base_iv  # already decimal
        _, greeks_dict = _compute_combo(
            product_type,
            product_params,
            model_type,
            model_params,
            iv_decimal,
            spot,
            risk_free_rate,
            dividend_yield,
            max(dte, 1) / 365.0,
            spot_range,
            entry_price,
            seed,
        )
        result[:, j] = greeks_dict.get(greek_name, np.zeros(len(spot_range)))

    return result


def calculate_structured_greeks_3d_iv(
    sp_config_json: str,
    spot_range: np.ndarray,
    iv_range: np.ndarray,
    risk_free_rate: float,
    base_dte: float,
    greek_index: int = 1,
) -> np.ndarray:
    """
    Calculate 3D Greek surface varying spot × IV for structured products.

    Same signature as calculate_portfolio_greeks_3d_iv (called by surface_3d.py).
    Returns 2D array [len(spot_range), len(iv_range)].
    """
    config = json.loads(sp_config_json)
    product_type = config["product_type"]
    product_params = json.loads(config["product_params_json"])
    model_type = config["model_type"]
    model_params = json.loads(config["model_params_json"])
    spot = config["spot"]
    dividend_yield = config.get("dividend_yield", 0.0)
    seed = config.get("seed")
    entry_price = config.get("entry_price", 0.0)

    greek_name = GREEK_NAMES[greek_index]
    result = np.zeros((len(spot_range), len(iv_range)))

    time_to_expiry = max(base_dte, 1) / 365.0

    for j, iv in enumerate(iv_range):
        _, greeks_dict = _compute_combo(
            product_type,
            product_params,
            model_type,
            model_params,
            iv,
            spot,
            risk_free_rate,
            dividend_yield,
            time_to_expiry,
            spot_range,
            entry_price,
            seed,
        )
        result[:, j] = greeks_dict.get(greek_name, np.zeros(len(spot_range)))

    return result


# =============================================================================
# Internal Helpers
# =============================================================================


def _adjust_frequency_for_maturity(params: dict) -> dict:
    """Adjust observation_frequency if maturity is too short for the current frequency."""
    params = dict(params)
    freq_periods = {
        "monthly": 1 / 12,
        "quarterly": 0.25,
        "semi_annual": 0.5,
        "annual": 1.0,
    }
    orig_freq = params.get("observation_frequency", "annual")
    if freq_periods.get(orig_freq, 1.0) > params["maturity"]:
        for freq_name in ["monthly", "quarterly", "semi_annual", "annual"]:
            if freq_periods[freq_name] <= params["maturity"]:
                params["observation_frequency"] = freq_name
                break
        else:
            params["observation_frequency"] = "monthly"
            params["maturity"] = max(params["maturity"], 1 / 12 + 1e-6)
    return params


def _prepare_theta_product(
    product_type: str,
    params: dict,
    dt_bump: float,
) -> object | None:
    """Create a product with maturity reduced by dt_bump for theta calculation."""
    new_mat = params["maturity"] - dt_bump
    if new_mat <= 1.0 / CALENDAR_DAYS_PER_YEAR:
        return None
    try:
        theta_params = dict(params)
        theta_params["maturity"] = new_mat
        theta_params = _adjust_frequency_for_maturity(theta_params)
        return build_product(product_type, theta_params)
    except ValueError:
        logger.debug(
            "theta product build failed for %s (maturity bump)", product_type, exc_info=True
        )
        return None


def _compute_spot_greeks(
    engine: StructuredProductMCEngine,
    product,
    model,
    market: MarketEnvironment,
    spot_range: np.ndarray,
    s0_ref: float,
    model_up,
    model_down,
    h_r: float,
    theta_product,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute fair_values, prices, deltas, vegas, rhos, thetas across spot_range."""
    n_spots = len(spot_range)
    fair_values = np.zeros(n_spots)
    prices = np.zeros(n_spots)
    deltas = np.zeros(n_spots)
    vegas = np.zeros(n_spots)
    rhos = np.zeros(n_spots)
    thetas = np.zeros(n_spots)

    for i, s in enumerate(spot_range):
        bumped_market = market.with_spot(s)
        result = engine.price_structured(
            product, model, bumped_market, s0_reference=s0_ref
        )
        fair_values[i] = result.fair_value
        prices[i] = result.price

        # Numerical delta
        h = s * DEFAULT_SPOT_BUMP
        v_up = engine.price_structured(
            product, model, bumped_market.bump_spot(h), s0_reference=s0_ref
        ).price
        v_down = engine.price_structured(
            product, model, bumped_market.bump_spot(-h), s0_reference=s0_ref
        ).price
        deltas[i] = (v_up - v_down) / (2 * h)

        # Vega per spot
        if model_up is not None and model_down is not None:
            vegas[i] = (
                (
                    engine.price_structured(
                        product, model_up, bumped_market, s0_reference=s0_ref
                    ).price
                    - engine.price_structured(
                        product, model_down, bumped_market, s0_reference=s0_ref
                    ).price
                )
                / (2 * DEFAULT_VOL_BUMP)
                / 100
            )
        # Rho per spot
        rhos[i] = (
            (
                engine.price_structured(
                    product, model, bumped_market.bump_rate(h_r), s0_reference=s0_ref
                ).price
                - engine.price_structured(
                    product, model, bumped_market.bump_rate(-h_r), s0_reference=s0_ref
                ).price
            )
            / (2 * h_r)
            / 100
        )

        # Theta per spot
        if theta_product is not None:
            thetas[i] = (
                engine.price_structured(
                    theta_product, model, bumped_market, s0_reference=s0_ref
                ).price
                - prices[i]
            )

    return fair_values, prices, deltas, vegas, rhos, thetas


def _compute_combo(
    product_type: str,
    product_params: dict,
    model_type: str,
    model_params: dict,
    sigma: float,
    spot: float,
    rate: float,
    dividend_yield: float,
    maturity: float,
    spot_range: np.ndarray,
    entry_price: float,
    seed: int | None,
) -> tuple[np.ndarray, dict]:
    """Compute P&L and Greeks for one (DTE, IV) combo via MC."""
    n_spots = len(spot_range)

    # Build product with adjusted maturity and compatible observation frequency
    params = dict(product_params)
    params["maturity"] = max(maturity, 1 / 365.0)
    params = _adjust_frequency_for_maturity(params)
    product = build_product(product_type, params)

    # Build model with adjusted sigma
    mp = dict(model_params)
    if model_type == "gbm":
        mp["sigma"] = max(sigma, 0.01)
    elif model_type == "heston":
        mp["v0"] = max(sigma**2, 1e-6)
    model = build_model(model_type, mp)

    market = build_market(spot, rate, dividend_yield)
    engine = StructuredProductMCEngine(n_paths=SP_SURFACE_PATHS, seed=seed)

    # Prepare bumped models and theta product
    # Use larger bumps than vanilla options to reduce MC noise:
    #   - Rate: 10bp (vs 1bp) — reduces rho CV from ~350% to ~15%
    #   - Theta: up to 30 days, scaled down for short maturities
    model_up, model_down = create_vol_bumped_pair(model, DEFAULT_VOL_BUMP)
    h_r = 0.001  # 10bp — 10× DEFAULT_RATE_BUMP for MC noise tolerance
    # Adaptive theta bump: 30 days when maturity allows, else 1/4 of maturity
    dt_bump = min(30.0 / CALENDAR_DAYS_PER_YEAR, params["maturity"] * 0.25)
    dt_bump = max(dt_bump, 1.0 / CALENDAR_DAYS_PER_YEAR)  # floor at 1 day
    theta_product = _prepare_theta_product(product_type, params, dt_bump)

    # Compute Greeks across spot range
    fair_values, prices, deltas, vegas, rhos, thetas = _compute_spot_greeks(
        engine,
        product,
        model,
        market,
        spot_range,
        spot,
        model_up,
        model_down,
        h_r,
        theta_product,
    )

    # Gamma = numerical diff of delta
    gammas = np.gradient(deltas, spot_range)

    # P&L = price - entry_price
    pnl = prices - entry_price

    # Smooth MC Greeks to reduce noise
    if n_spots >= 5:
        from scipy.ndimage import gaussian_filter1d

        sigma_smooth = max(1.0, n_spots / 12)
        deltas = gaussian_filter1d(deltas, sigma_smooth)
        gammas = gaussian_filter1d(gammas, sigma_smooth)
        vegas = gaussian_filter1d(vegas, sigma_smooth)
        thetas = gaussian_filter1d(thetas, sigma_smooth)
        rhos = gaussian_filter1d(rhos, sigma_smooth)
        pnl = gaussian_filter1d(pnl, sigma_smooth)

    # Build greeks_dict with all 14 names
    greeks_dict = {
        "price": prices,
        "delta": deltas,
        "gamma": gammas,
        "vega": vegas,
        "theta": thetas,
        "rho": rhos,
        "vanna": np.zeros(n_spots),
        "volga": np.zeros(n_spots),
        "charm": np.zeros(n_spots),
        "veta": np.zeros(n_spots),
        "speed": np.zeros(n_spots),
        "zomma": np.zeros(n_spots),
        "color": np.zeros(n_spots),
        "ultima": np.zeros(n_spots),
    }

    return pnl, greeks_dict


def _compute_expiry_pnl(
    product_type: str,
    product_params: dict,
    spot_range: np.ndarray,
    entry_price: float,
    current_spot: float | None = None,
) -> np.ndarray:
    """
    Compute terminal payoff P&L for the structured product.

    CPN: analytical. RC/Autocallable: approximate (notional recovery at par).
    """
    notional = product_params["notional"]
    n_spots = len(spot_range)
    s0 = current_spot if current_spot is not None else spot_range[n_spots // 2]

    if product_type == "cpn":
        protection = product_params.get("protection_level", 1.0)
        participation = product_params["participation_rate"]
        cap = product_params.get("cap")
        payoffs = np.zeros(n_spots)
        for i, s in enumerate(spot_range):
            perf = s / s0
            if cap is not None:
                upside = min(cap - 1.0, max(0, perf - 1.0)) * participation
            else:
                upside = max(0, perf - 1.0) * participation
            payoffs[i] = notional * (protection + upside)
        return payoffs - entry_price

    if product_type == "reverse_convertible":
        coupon_rate = product_params["coupon_rate"]
        maturity = product_params["maturity"]
        total_coupon = notional * coupon_rate * maturity
        payoffs = np.zeros(n_spots)
        for i, s in enumerate(spot_range):
            perf = s / s0
            # Worst-case scenario (barrier breached): capital loss if S_T < S_0
            # Break at S_0 (strike), not at barrier — matches backend kernel
            if perf < 1.0:
                payoffs[i] = notional * perf + total_coupon
            else:
                payoffs[i] = notional + total_coupon
        return payoffs - entry_price

    if product_type == "autocallable":
        coupon_rate = product_params["coupon_rate"]
        maturity = product_params["maturity"]
        total_coupon = notional * coupon_rate * maturity
        payoffs = np.zeros(n_spots)
        for i, s in enumerate(spot_range):
            perf = s / s0
            # Worst-case scenario (not autocalled, KI barrier breached):
            # capital loss if S_T < S_0, break at S_0
            if perf < 1.0:
                payoffs[i] = notional * perf + total_coupon
            else:
                payoffs[i] = notional + total_coupon
        return payoffs - entry_price

    # Fallback
    return np.full(n_spots, -entry_price)
