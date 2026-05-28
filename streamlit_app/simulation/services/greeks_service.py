"""
Greeks Service for Simulation Explorer.

Computes Greeks surfaces (vs spot) for:
- Vanilla option strategies via analytical Black-Scholes
- Structured products via MC bump-and-reprice

Author: Thomas Vaudescal
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from backend.engines.vectorized_bs import calculate_greeks_vectorized

_GREEK_NAMES = ["price", "delta", "gamma", "vega", "theta", "rho"]
DISPLAY_GREEKS = ["delta", "gamma", "vega", "theta", "rho"]

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
) -> dict[str, np.ndarray]:
    """Compute aggregated Greeks across spot_range for a single time_to_expiry."""
    n = len(spot_range)
    strikes = position_arrays.get("strikes", np.array([]))
    option_types = position_arrays.get("option_types", np.array([]))
    position_types = position_arrays.get("position_types", np.array([]))
    quantities = position_arrays.get("quantities", np.array([]))
    stock_quantity = position_arrays.get("stock_quantity", 0.0)

    total = {name: np.zeros(n) for name in _GREEK_NAMES}
    t = max(time_to_expiry, 1e-6)

    for j in range(len(strikes)):
        opt_int = int(option_types[j]) if option_types[j] == 1.0 else 0
        full = calculate_greeks_vectorized(
            spot_range,
            strikes[j],
            t,
            rate,
            sigma,
            opt_int,
        )
        leg_greeks = np.where(np.isnan(full[:, :6]), 0.0, full[:, :6])

        scale = quantities[j] * position_types[j]
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
) -> dict:
    """
    Compute Greeks across spot_range x DTE_range.

    Returns dict with keys:
        spot_range, dte_values, greeks_by_dte, price_by_dte
    """
    spot_range = np.linspace(
        spot * (1 - spot_range_pct),
        spot * (1 + spot_range_pct),
        n_spot_points,
    )
    max_dte_days = max(1, int(time_to_expiry * 365))
    dte_values = _build_dte_range(max_dte_days, n_dte_steps)

    greeks_by_dte = {}
    price_by_dte = {}

    for dte in dte_values:
        t = dte / 365.0
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

    max_dte_days = max(1, int(product.maturity * 365))

    # Minimum DTE must cover at least one observation period
    _freq_days = {
        "monthly": 31,
        "quarterly": 92,
        "semi_annual": 183,
        "annual": 366,
    }
    obs_freq = product_params.get("observation_frequency", "quarterly")
    min_dte = _freq_days.get(obs_freq, 92)

    dte_values = [
        d for d in _build_dte_range(max_dte_days, n_dte_steps) if d >= min_dte
    ]
    if not dte_values:
        dte_values = [max_dte_days]

    greeks_by_dte = {}
    total_iters = n_points * len(dte_values)
    iter_count = 0

    for dte in dte_values:
        t_years = dte / 365.0
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
