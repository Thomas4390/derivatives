"""
Portfolio calculation services for Options Greeks Explorer.

This module provides functions for calculating portfolio metrics, P&L, and Greeks.
"""

import json
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import streamlit as st

if TYPE_CHECKING:
    from .pricing_adapter import BreakevenResult
from config.constants import (
    CONTRACT_MULTIPLIER,
    GREEK_NAMES,
    SPOT_RANGE_FACTOR,
    SPOT_RANGE_POINTS,
    STRIKE_RANGE_FACTORS,
)
from config.exotic_config import EXOTIC_LEG_KEYS

from .risk_analyzer import check_unlimited_risk


@lru_cache(maxsize=128)
def get_portfolio_hash(portfolio_json: str) -> int:
    """
    Create a hash for portfolio caching.

    Args:
        portfolio_json: JSON string of portfolio data

    Returns:
        Hash value for caching
    """
    return hash(portfolio_json)


def prepare_portfolio_data(positions: list, stock_position, spot_price: float) -> dict:
    """
    Prepare portfolio data dictionary from positions.

    Args:
        positions: List of option position dicts
        stock_position: Stock position dict or None
        spot_price: Current spot price

    Returns:
        Dictionary with portfolio data
    """
    portfolio_data = {"spot_price": spot_price, "options": [], "stock": None}

    # Add option positions (positions are already dicts)
    if positions:
        portfolio_data["options"] = []
        for pos in positions:
            opt = {
                "option_type": str(pos["option_type"]),
                "position_type": str(pos["position_type"]),
                "strike": pos["strike"],
                "quantity": pos["quantity"],
                "premium_paid": pos["premium_paid"],
                "instrument_class": pos.get("instrument_class", "vanilla"),
            }
            # Pass through exotic fields when present. EXOTIC_LEG_KEYS covers
            # the advanced families too (supershare corridor, double/discrete/
            # binary-barrier adv_* keys, cash, ...) so the base curve, the
            # metrics and the premium all describe the same instrument as the
            # scenario overlay.
            if opt["instrument_class"] != "vanilla":
                for key in EXOTIC_LEG_KEYS:
                    if key in pos:
                        opt[key] = pos[key]
            portfolio_data["options"].append(opt)

    # Add stock position (stock_position is already a dict)
    if stock_position:
        portfolio_data["stock"] = {
            "position_type": str(stock_position["position_type"]),
            "quantity": stock_position["quantity"],
            "entry_price": stock_position["entry_price"],
        }

    return portfolio_data


def prepare_portfolio_arrays(portfolio_data: dict) -> tuple:
    """
    Prepare numpy arrays from portfolio data for calculations.

    Args:
        portfolio_data: Dictionary with portfolio data

    Returns:
        Tuple of (strikes, option_types, position_types, quantities, premiums,
                 stock_quantity, stock_entry_price, exotic_metadata)
    """
    from .exotic_pricing_adapter import haug_factory_params

    exotic_metadata = []
    ref_spot = portfolio_data.get("spot_price", 0.0)
    if portfolio_data.get("options") and len(portfolio_data["options"]) > 0:
        strikes = np.array([pos["strike"] for pos in portfolio_data["options"]])
        option_types = np.array(
            [
                1 if pos["option_type"] == "call" else 0
                for pos in portfolio_data["options"]
            ]
        )
        position_types = np.array(
            [
                1 if pos["position_type"] == "long" else -1
                for pos in portfolio_data["options"]
            ]
        )
        quantities = np.array(
            [pos["quantity"] * CONTRACT_MULTIPLIER for pos in portfolio_data["options"]]
        )
        premiums = np.array([pos["premium_paid"] for pos in portfolio_data["options"]])
        for pos in portfolio_data["options"]:
            exotic_metadata.append(
                {
                    "instrument_class": pos.get("instrument_class", "vanilla"),
                    "barrier": pos.get("barrier", 0.0),
                    "is_up": pos.get("is_up", True),
                    "is_knock_in": pos.get("is_knock_in", False),
                    "rebate": pos.get("rebate", 0.0),
                    "payout": pos.get("payout", 1.0),
                    "extra1": pos.get("extra1", 0.0),
                    "power_n": pos.get("power_n", 2.0),
                    "gap_trigger": pos.get("gap_trigger", 0.0),
                    "cap": pos.get("cap", 0.0),
                    "params": haug_factory_params(pos),
                    "ref_spot": ref_spot,
                }
            )
    else:
        strikes = np.array([])
        option_types = np.array([], dtype=np.int32)
        position_types = np.array([], dtype=np.int32)
        quantities = np.array([], dtype=np.int32)
        premiums = np.array([])

    stock_quantity = 0
    stock_entry_price = 0
    if portfolio_data.get("stock"):
        stock = portfolio_data["stock"]
        stock_quantity = stock["quantity"] * (
            1 if stock["position_type"] == "long" else -1
        )
        stock_entry_price = stock["entry_price"]

    return (
        strikes,
        option_types,
        position_types,
        quantities,
        premiums,
        stock_quantity,
        stock_entry_price,
        exotic_metadata,
    )


@st.cache_data(ttl=3600)
def calculate_all_surfaces(
    portfolio_json: str,
    spot_range: tuple,
    dte_values: tuple,
    iv_values: tuple,
    risk_free_rate: float,
    _calculate_all_greeks_func,
    _calculate_pnl_at_expiry_func,
    _find_breakeven_func,
    has_positions: bool,
    _calculate_exotic_greeks_func=None,
    dividend_yield: float = 0.0,
) -> dict:
    """
    Calculate all P&L and Greeks data at once.

    Args:
        portfolio_json: JSON string of portfolio data
        spot_range: Tuple of spot prices (converted from numpy array for caching)
        dte_values: Tuple of DTE values
        iv_values: Tuple of IV values
        risk_free_rate: Risk-free interest rate
        _calculate_all_greeks_func: Function to calculate Greeks (not hashed)
        _calculate_pnl_at_expiry_func: Function to calculate P&L at expiry (not hashed)
        _find_breakeven_func: Function to find breakeven points (not hashed)
        has_positions: Whether there are active positions
        _calculate_exotic_greeks_func: Function to calculate exotic Greeks (not hashed)
        dividend_yield: Continuous dividend yield (decimal, default 0.0)

    Returns:
        Dictionary with all calculated data
    """
    portfolio_data = json.loads(portfolio_json)
    spot_range_arr = np.array(spot_range)

    # Prepare arrays
    (
        strikes,
        option_types,
        position_types,
        quantities,
        premiums,
        stock_quantity,
        stock_entry_price,
        exotic_metadata,
    ) = prepare_portfolio_arrays(portfolio_data)

    has_exotic_legs = any(m["instrument_class"] != "vanilla" for m in exotic_metadata)

    # Calculate surfaces
    pnl_data, greeks_data = _calculate_surfaces(
        portfolio_data,
        spot_range_arr,
        dte_values,
        iv_values,
        risk_free_rate,
        strikes,
        option_types,
        position_types,
        quantities,
        premiums,
        stock_quantity,
        stock_entry_price,
        _calculate_all_greeks_func,
        exotic_metadata=exotic_metadata,
        calculate_exotic_greeks_func=_calculate_exotic_greeks_func,
        dividend_yield=dividend_yield,
    )

    # Calculate P&L at expiration (split vanilla / exotic)
    expiry_pnl = _calculate_expiry_pnl(
        spot_range_arr,
        strikes,
        option_types,
        position_types,
        quantities,
        premiums,
        stock_quantity,
        stock_entry_price,
        _calculate_pnl_at_expiry_func,
        exotic_metadata=exotic_metadata,
        portfolio_data=portfolio_data,
    )
    pnl_data["expiry"] = expiry_pnl

    # For DTE=0, use expiry P&L directly so the curve perfectly matches
    # the "P&L at Expiration" dashed line (especially for path-dependent exotics)
    if 0 in dte_values:
        for iv in iv_values:
            pnl_data[f"0_{iv}"] = expiry_pnl.copy()
            greeks_data[f"0_{iv}"] = {
                name: np.zeros(len(spot_range_arr)) for name in GREEK_NAMES
            }

    # Find breakeven and risk analysis
    breakeven_result = _find_breakeven(
        portfolio_data,
        strikes,
        option_types,
        position_types,
        quantities,
        premiums,
        stock_quantity,
        stock_entry_price,
        _find_breakeven_func,
        _calculate_pnl_at_expiry_func,
        exotic_metadata=exotic_metadata,
        expiry_pnl=expiry_pnl,
        spot_range_arr=spot_range_arr,
    )

    # Determine unlimited risk
    if not has_positions:
        unlimited_profit, unlimited_loss = False, False
    else:
        unlimited_profit, unlimited_loss = check_unlimited_risk(portfolio_data)

    # Override for exotic portfolios — per-type logic
    if has_exotic_legs:
        for j, m in enumerate(exotic_metadata):
            if m["instrument_class"] == "vanilla":
                continue
            is_long = position_types[j] == 1
            typ = m["instrument_class"]
            # Lookback and Asian: only calls are unbounded
            if typ in ("lookback_fixed", "lookback_floating", "asian"):
                is_call = portfolio_data["options"][j]["option_type"] == "call"
                if is_call:
                    if is_long:
                        unlimited_profit = True
                    else:
                        unlimited_loss = True
            elif typ == "barrier":
                is_call = portfolio_data["options"][j]["option_type"] == "call"
                if is_long:
                    if is_call:
                        unlimited_profit = True
                else:
                    if is_call:
                        unlimited_loss = True
            elif typ == "chooser":
                # Chooser = max(call, put) → long has unlimited upside, short has unlimited downside
                if is_long:
                    unlimited_profit = True
                else:
                    unlimited_loss = True
            elif typ == "power":
                # Power options: convex payoff → unlimited for long calls
                is_call = portfolio_data["options"][j]["option_type"] == "call"
                if is_long and is_call:
                    unlimited_profit = True
                elif not is_long and is_call:
                    unlimited_loss = True
            elif typ == "gap":
                # Gap call: unlimited upside for long, unlimited downside for short
                is_call = portfolio_data["options"][j]["option_type"] == "call"
                if is_long and is_call:
                    unlimited_profit = True
                elif not is_long and is_call:
                    unlimited_loss = True
            elif typ == "asset_or_nothing":
                # Asset-or-nothing call pays spot → unbounded; put pays spot<strike → bounded
                is_call = portfolio_data["options"][j]["option_type"] == "call"
                if is_call:
                    if is_long:
                        unlimited_profit = True
                    else:
                        unlimited_loss = True
            # Digital: bounded payout, no change needed

    # Build result
    result = _build_result(
        pnl_data,
        greeks_data,
        breakeven_result,
        unlimited_profit,
        unlimited_loss,
        expiry_pnl,
        has_exotic_legs=has_exotic_legs,
    )
    result["has_exotic_legs"] = has_exotic_legs

    return result


def _calculate_surfaces(
    portfolio_data: dict,
    spot_range: np.ndarray,
    dte_values: tuple,
    iv_values: tuple,
    risk_free_rate: float,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    stock_quantity: int,
    stock_entry_price: float,
    calculate_all_greeks_func,
    exotic_metadata: list = None,
    calculate_exotic_greeks_func=None,
    dividend_yield: float = 0.0,
) -> tuple[dict, dict]:
    """Calculate P&L and Greeks surfaces."""
    pnl_data = {}
    greeks_data = {}

    for dte in dte_values:
        if dte == 0:
            continue  # Handled via expiry P&L override in calculate_all_surfaces
        time_to_expiry = dte / 365.0

        for iv in iv_values:
            key = f"{dte}_{iv}"
            iv_decimal = iv / 100.0

            pnl_values, greeks_by_name = _calculate_for_params(
                spot_range,
                time_to_expiry,
                iv_decimal,
                risk_free_rate,
                strikes,
                option_types,
                position_types,
                quantities,
                premiums,
                stock_quantity,
                stock_entry_price,
                calculate_all_greeks_func,
                exotic_metadata=exotic_metadata,
                calculate_exotic_greeks_func=calculate_exotic_greeks_func,
                dividend_yield=dividend_yield,
            )

            pnl_data[key] = pnl_values
            greeks_data[key] = greeks_by_name

    return pnl_data, greeks_data


def _calculate_for_params(
    spot_range: np.ndarray,
    time_to_expiry: float,
    iv_decimal: float,
    risk_free_rate: float,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    stock_quantity: int,
    stock_entry_price: float,
    calculate_all_greeks_func,
    exotic_metadata: list = None,
    calculate_exotic_greeks_func=None,
    dividend_yield: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Calculate P&L and Greeks for a specific DTE/IV combination."""
    from backend.engines.exotic_engine import (
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
    from backend.engines.exotic_engine import ASIAN_GEO
    from backend.engines.vectorized_bs import calculate_greeks_vectorized

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

    n = len(spot_range)
    pnl_values = np.zeros(n)
    greeks_by_name = {name: np.zeros(n) for name in GREEK_NAMES}

    for j in range(len(strikes)):
        meta = exotic_metadata[j] if exotic_metadata else None
        is_exotic = meta and meta["instrument_class"] != "vanilla"

        if is_exotic and meta["instrument_class"] in _TYPE_MAP:
            opt_type = _TYPE_MAP[meta["instrument_class"]]
            H = (
                meta.get("barrier", 0.0)
                if meta["instrument_class"] == "barrier"
                else 0.0
            )
            # For lookback_floating, pass ref spot so surface tracks extremes
            inst_cls = meta["instrument_class"]
            ref_spot = (
                meta.get("ref_spot", 0.0) if inst_cls == "lookback_floating" else 0.0
            )
            surface = exotic_greeks_surface(
                opt_type,
                spot_range,
                strikes[j],
                time_to_expiry,
                risk_free_rate,
                dividend_yield,
                iv_decimal,
                option_types[j] == 1,
                H,
                ref_spot,
                ref_spot,
                meta.get("is_knock_in", False),
                meta.get("is_up", True),
                meta.get("rebate", 0.0),
                meta.get("payout", 1.0),
                meta.get("extra1", 0.0),
            )
            # surface is (n, 6) — map to 14-element layout
            leg_greeks = np.zeros((n, 14))
            leg_greeks[:, 0] = surface[:, 0]
            leg_greeks[:, 1] = surface[:, 1]
            leg_greeks[:, 2] = surface[:, 2]
            leg_greeks[:, 3] = surface[:, 3]
            leg_greeks[:, 4] = surface[:, 4]
            leg_greeks[:, 5] = surface[:, 5]
        elif is_exotic:
            # Registry-priced (Haug advanced) families: per-cell backend path,
            # the same seam the 3D surface tab uses. Previously these fell
            # through to the vanilla branch below — Black-Scholes prices AND
            # Greeks silently displayed for exotic legs.
            from .exotic_pricing_adapter import calculate_exotic_greeks_curve

            leg_greeks = calculate_exotic_greeks_curve(
                spot_range,
                strikes[j],
                time_to_expiry,
                risk_free_rate,
                iv_decimal,
                int(option_types[j]),
                meta["instrument_class"],
                barrier=meta.get("barrier", 0.0),
                is_up=meta.get("is_up", True),
                is_knock_in=meta.get("is_knock_in", False),
                rebate=meta.get("rebate", 0.0),
                payout=meta.get("payout", 1.0),
                extra1=meta.get("extra1", 0.0),
                cap=meta.get("cap", 0.0),
                dividend_yield=dividend_yield,
                params=meta.get("params"),
            )
        else:
            leg_greeks = calculate_greeks_vectorized(
                spot_range,
                strikes[j],
                time_to_expiry,
                risk_free_rate,
                iv_decimal,
                option_types[j],
                dividend_yield,
            )

        # Guard against NaN from exotic discontinuities
        leg_greeks = np.where(np.isnan(leg_greeks), 0.0, leg_greeks)

        # P&L: per-leg vectorized
        option_values = leg_greeks[:, 0]
        if position_types[j] == 1:
            pnl_values += (option_values - premiums[j]) * quantities[j]
        else:
            pnl_values += (premiums[j] - option_values) * quantities[j]

        # Greeks aggregation — vectorized
        scale = quantities[j] * position_types[j]
        for k, name in enumerate(GREEK_NAMES):
            greeks_by_name[name] += leg_greeks[:, k] * scale

    # Stock contribution (vectorized)
    if stock_quantity != 0:
        pnl_values += (spot_range - stock_entry_price) * stock_quantity
        greeks_by_name["delta"] += stock_quantity

    return pnl_values, greeks_by_name


def _calculate_expiry_pnl(
    spot_range: np.ndarray,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    stock_quantity: int,
    stock_entry_price: float,
    calculate_pnl_at_expiry_func,
    exotic_metadata: list = None,
    portfolio_data: dict = None,
) -> np.ndarray:
    """Calculate P&L at expiration for all spot prices.

    Vanilla legs use the fast Numba path. Exotic legs use a Python loop
    calling calculate_exotic_payoff_at_expiry.

    A NaN cell of one discrete-event leg (its scenario is infeasible at that
    terminal spot) poisons the WHOLE portfolio curve at that spot by design:
    the scenario cannot occur there, so no portfolio P&L is defined — even for
    the vanilla legs. Locked by test_scenario_pnl.py.
    """
    # Separate vanilla and exotic indices
    vanilla_indices = []
    exotic_indices = []
    if exotic_metadata:
        for j, meta in enumerate(exotic_metadata):
            if meta["instrument_class"] != "vanilla":
                exotic_indices.append(j)
            else:
                vanilla_indices.append(j)
    else:
        vanilla_indices = list(range(len(strikes)))

    # Vanilla part: use fast Numba function
    if vanilla_indices:
        v_strikes = strikes[vanilla_indices]
        v_option_types = option_types[vanilla_indices]
        v_position_types = position_types[vanilla_indices]
        v_quantities = quantities[vanilla_indices]
        v_premiums = premiums[vanilla_indices]
    else:
        v_strikes = np.array([])
        v_option_types = np.array([], dtype=np.int32)
        v_position_types = np.array([], dtype=np.int32)
        v_quantities = np.array([], dtype=np.int32)
        v_premiums = np.array([])

    expiry_pnl = np.zeros(len(spot_range))

    # Vanilla legs (Numba) — per spot.
    if len(v_strikes) > 0 or stock_quantity != 0:
        for i, spot in enumerate(spot_range):
            expiry_pnl[i] = calculate_pnl_at_expiry_func(
                spot,
                v_strikes,
                v_option_types,
                v_position_types,
                v_quantities,
                v_premiums,
                stock_quantity,
                stock_entry_price,
            )

    # Exotic legs — vectorized over the whole spot range, once per leg, instead
    # of a scalar payoff call per (spot, leg) cell.
    if exotic_indices and portfolio_data:
        from config.exotic_config import PAYOFF_SCENARIOS

        from .exotic_pricing_adapter import (
            calculate_exotic_payoff_at_expiry_vec,
            conditional_exotic_payoff_vec,
        )

        options = portfolio_data.get("options", [])
        for j in exotic_indices:
            opt = options[j]
            spec = PAYOFF_SCENARIOS.get(opt.get("instrument_class"))
            if spec and spec.get("kind") == "discrete_event":
                scenario = opt.get("scenario") or spec["scenarios"][0][0]
                payoff, feasible = conditional_exotic_payoff_vec(
                    spot_range, opt, scenario
                )
                payoff = np.where(feasible, payoff, np.nan)
            else:
                payoff = calculate_exotic_payoff_at_expiry_vec(spot_range, opt)
            premium = premiums[j]
            sign = 1.0 if position_types[j] == 1 else -1.0  # long vs short
            expiry_pnl += sign * (payoff - premium) * quantities[j]

    return expiry_pnl


def _find_breakeven(
    portfolio_data: dict,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    stock_quantity: int,
    stock_entry_price: float,
    find_breakeven_func,
    calculate_pnl_at_expiry_func,
    exotic_metadata: list = None,
    expiry_pnl: np.ndarray = None,
    spot_range_arr: np.ndarray = None,
):
    """Find breakeven points and max profit/loss."""
    if len(strikes) == 0 and stock_quantity == 0:
        return None

    # When exotic legs are present, compute breakeven from the expiry_pnl array
    # (the Numba find_breakeven_func treats all legs as vanilla)
    has_exotic = exotic_metadata and any(
        m["instrument_class"] != "vanilla" for m in exotic_metadata
    )
    if has_exotic and expiry_pnl is not None and spot_range_arr is not None:
        return _breakeven_from_pnl_curve(expiry_pnl, spot_range_arr)

    # Use wide range for theoretical extremes
    theoretical_min = 0.01
    theoretical_max = portfolio_data.get("spot_price", 100.0) * 10.0

    breakeven_result = find_breakeven_func(
        strikes,
        option_types,
        position_types,
        quantities,
        premiums,
        stock_quantity,
        stock_entry_price,
        theoretical_min,
        theoretical_max,
        20000,
    )

    # Special case: For short puts, calculate P&L at exactly 0
    if portfolio_data.get("options"):
        has_short_puts = any(
            pos["option_type"] == "put" and pos["position_type"] == "short"
            for pos in portfolio_data["options"]
        )
        if has_short_puts and breakeven_result:
            pnl_at_zero = calculate_pnl_at_expiry_func(
                0.0,
                strikes,
                option_types,
                position_types,
                quantities,
                premiums,
                stock_quantity,
                stock_entry_price,
            )
            if pnl_at_zero < breakeven_result.max_loss:
                breakeven_result.max_loss = pnl_at_zero
                breakeven_result.max_loss_spot = 0.0

    return breakeven_result


def _breakeven_from_pnl_curve(
    expiry_pnl: np.ndarray,
    spot_range: np.ndarray,
) -> "BreakevenResult":
    """Compute breakeven points from a PnL curve via sign-change interpolation."""
    from .pricing_adapter import BreakevenResult

    # Find sign changes (breakeven crossings). NaN cells (infeasible regions of
    # a scenario-conditional payoff) make the product NaN, so ``< 0`` is False
    # and no spurious crossing is reported across an infeasible gap.
    breakeven_points = []
    for i in range(len(expiry_pnl) - 1):
        if expiry_pnl[i] * expiry_pnl[i + 1] < 0:
            # Linear interpolation for the zero crossing
            frac = abs(expiry_pnl[i]) / (abs(expiry_pnl[i]) + abs(expiry_pnl[i + 1]))
            bp = spot_range[i] + frac * (spot_range[i + 1] - spot_range[i])
            breakeven_points.append(float(bp))

    # Max profit / loss over the FEASIBLE (non-NaN) cells; a scenario-conditional
    # curve is NaN where the scenario cannot occur. argmax/argmin are not
    # NaN-aware, so use the nan-variants and guard the all-NaN case.
    finite = np.isfinite(expiry_pnl)
    if finite.any():
        max_profit_idx = int(np.nanargmax(expiry_pnl))
        max_loss_idx = int(np.nanargmin(expiry_pnl))
        max_profit = float(expiry_pnl[max_profit_idx])
        max_profit_spot = float(spot_range[max_profit_idx])
        max_loss = float(expiry_pnl[max_loss_idx])
        max_loss_spot = float(spot_range[max_loss_idx])
    else:
        max_profit = max_loss = 0.0
        max_profit_spot = max_loss_spot = (
            float(spot_range[0]) if len(spot_range) else 0.0
        )

    return BreakevenResult(
        breakeven_points=breakeven_points,
        max_profit=max_profit,
        max_profit_spot=max_profit_spot,
        max_loss=max_loss,
        max_loss_spot=max_loss_spot,
    )


def combine_scenario_metrics(
    curves: list[np.ndarray], spot_range: np.ndarray
) -> "BreakevenResult | None":
    """Metrics of an overlaid discrete-event chart: union of its outcome curves.

    Each curve is one conditional outcome, NaN-masked to its feasible region.
    Max profit / loss are taken across the feasible cells of ALL curves, and
    the breakeven points are the union of the per-curve crossings (deduplicated
    within half a grid step — the two outcomes share the pre-event region, so
    their crossings there coincide). Returns None if every curve is fully
    infeasible.
    """
    from .pricing_adapter import BreakevenResult

    spot = np.asarray(spot_range, dtype=np.float64)
    per_curve = [
        _breakeven_from_pnl_curve(arr, spot)
        for arr in (np.asarray(c, dtype=np.float64) for c in curves)
        if np.isfinite(arr).any()
    ]
    if not per_curve:
        return None

    step = (float(spot[-1]) - float(spot[0])) / max(len(spot) - 1, 1)
    breakevens: list[float] = []
    for res in per_curve:
        for point in res.breakeven_points:
            if not any(abs(point - seen) <= step / 2.0 for seen in breakevens):
                breakevens.append(point)
    breakevens.sort()

    best = max(per_curve, key=lambda r: r.max_profit)
    worst = min(per_curve, key=lambda r: r.max_loss)
    return BreakevenResult(
        breakeven_points=breakevens,
        max_profit=best.max_profit,
        max_profit_spot=best.max_profit_spot,
        max_loss=worst.max_loss,
        max_loss_spot=worst.max_loss_spot,
    )


def _build_result(
    pnl_data: dict,
    greeks_data: dict,
    breakeven_result,
    unlimited_profit: bool,
    unlimited_loss: bool,
    expiry_pnl: np.ndarray,
    has_exotic_legs: bool = False,
) -> dict:
    """Build the final result dictionary."""
    result = {
        "pnl_data": pnl_data,
        "greeks_data": greeks_data,
        "breakeven_result": breakeven_result,
        "unlimited_profit": unlimited_profit,
        "unlimited_loss": unlimited_loss,
    }

    if not breakeven_result:
        result["max_profit_display"] = 0
        result["max_loss_display"] = 0
        return result

    # For exotic legs, use actual PnL curve values instead of breakeven-based
    # estimates, since the breakeven function treats exotic legs as vanilla
    if has_exotic_legs:
        result["max_profit_display"] = _get_profit_display(
            unlimited_profit, breakeven_result, expiry_pnl
        )
        result["max_loss_display"] = _get_loss_display(
            unlimited_loss, breakeven_result, expiry_pnl
        )
        return result

    # Determine display values for max profit/loss
    result["max_profit_display"] = _get_profit_display(
        unlimited_profit, breakeven_result, expiry_pnl
    )
    result["max_loss_display"] = _get_loss_display(
        unlimited_loss, breakeven_result, expiry_pnl
    )

    return result


def _get_profit_display(
    unlimited_profit: bool, breakeven_result, expiry_pnl: np.ndarray
) -> float:
    """Get the display value for max profit."""
    if not unlimited_profit:
        return breakeven_result.max_profit

    if len(expiry_pnl) > 10:
        high_end_trend = expiry_pnl[-1] - expiry_pnl[-10]
        if high_end_trend > 0:
            return float("inf")

    return breakeven_result.max_profit


def _get_loss_display(
    unlimited_loss: bool, breakeven_result, expiry_pnl: np.ndarray
) -> float:
    """Get the display value for max loss."""
    if not unlimited_loss:
        return breakeven_result.max_loss

    if len(expiry_pnl) > 10:
        high_end_trend = expiry_pnl[-1] - expiry_pnl[-10]
        if high_end_trend < 0 and expiry_pnl[-1] < 0:
            return float("-inf")

    return breakeven_result.max_loss


def get_spot_range(spot_price: float) -> np.ndarray:
    """
    Get the spot price range for calculations.

    Args:
        spot_price: Current spot price

    Returns:
        Numpy array of spot prices
    """
    return np.linspace(
        spot_price * (1 - SPOT_RANGE_FACTOR),
        spot_price * (1 + SPOT_RANGE_FACTOR),
        SPOT_RANGE_POINTS,
    )


@st.cache_data(ttl=3600)
def calculate_strike_surfaces(
    spot_price: float,
    spot_range: tuple,
    option_type: str,
    position_type: str,
    quantity: int,
    base_strike: float,
    risk_free_rate: float,
    _calculate_all_greeks_func,
    _calculate_pnl_at_expiry_func,
    _find_breakeven_func=None,
    dividend_yield: float = 0.0,
) -> tuple[dict, dict, dict]:
    """
    Calculate P&L and Greeks data for varying strike prices (single-leg only).

    Args:
        spot_price: Current spot price
        spot_range: Tuple of spot prices
        option_type: 'call' or 'put'
        position_type: 'long' or 'short'
        quantity: Number of contracts
        base_strike: Base strike price for reference
        risk_free_rate: Risk-free interest rate
        _calculate_all_greeks_func: Function to calculate Greeks
        _calculate_pnl_at_expiry_func: Function to calculate P&L at expiry
        _find_breakeven_func: Function to find breakeven points

    Returns:
        Tuple of (pnl_data, greeks_data, breakeven_data) dictionaries keyed by strike
    """
    spot_range_arr = np.array(spot_range)
    pnl_data = {}
    greeks_data = {}
    breakeven_data = {}
    expiry_data = {}

    option_type_int = 1 if option_type == "call" else 0
    position_sign = 1 if position_type == "long" else -1

    # Fixed parameters for strike variation
    fixed_dte = 31
    fixed_iv = 0.25
    time_to_expiry = fixed_dte / 365.0

    for strike_factor in STRIKE_RANGE_FACTORS:
        strike = round(spot_price * strike_factor, 2)
        key = f"strike_{int(strike_factor * 100)}"

        pnl_values = np.zeros(len(spot_range_arr))
        greeks_by_name = {name: np.zeros(len(spot_range_arr)) for name in GREEK_NAMES}

        # Calculate initial premium at this strike
        initial_greeks = _calculate_all_greeks_func(
            spot_price,
            strike,
            time_to_expiry,
            risk_free_rate,
            fixed_iv,
            option_type_int,
            dividend_yield,
        )
        premium = initial_greeks[0]

        for i, spot in enumerate(spot_range_arr):
            greeks = _calculate_all_greeks_func(
                spot,
                strike,
                time_to_expiry,
                risk_free_rate,
                fixed_iv,
                option_type_int,
                dividend_yield,
            )

            option_value = greeks[0]
            if position_sign == 1:  # Long
                pnl = (option_value - premium) * quantity * CONTRACT_MULTIPLIER
            else:  # Short
                pnl = (premium - option_value) * quantity * CONTRACT_MULTIPLIER

            pnl_values[i] = pnl

            for k, name in enumerate(GREEK_NAMES):
                greeks_by_name[name][i] = (
                    greeks[k] * quantity * CONTRACT_MULTIPLIER * position_sign
                )

        pnl_data[key] = pnl_values
        greeks_data[key] = greeks_by_name

        # Calculate expiry P&L for this strike
        expiry_pnl = np.zeros(len(spot_range_arr))
        strikes_arr = np.array([strike])
        option_types_arr = np.array([option_type_int])
        position_types_arr = np.array([position_sign])
        quantities_arr = np.array([quantity * CONTRACT_MULTIPLIER])
        premiums_arr = np.array([premium])

        for i, spot in enumerate(spot_range_arr):
            expiry_pnl[i] = _calculate_pnl_at_expiry_func(
                spot,
                strikes_arr,
                option_types_arr,
                position_types_arr,
                quantities_arr,
                premiums_arr,
                0,
                0,
            )

        expiry_data[key] = expiry_pnl

        # Calculate breakeven for this strike
        if _find_breakeven_func:
            theoretical_min = 0.01
            theoretical_max = spot_price * 3.0
            breakeven_result = _find_breakeven_func(
                strikes_arr,
                option_types_arr,
                position_types_arr,
                quantities_arr,
                premiums_arr,
                0,
                0,
                theoretical_min,
                theoretical_max,
                10000,
            )
            breakeven_data[key] = breakeven_result

    # Store expiry data in pnl_data
    pnl_data["expiry_by_strike"] = expiry_data

    return pnl_data, greeks_data, breakeven_data


def get_strike_range(spot_price: float) -> list[float]:
    """
    Get the list of strike prices for strike variation.

    Args:
        spot_price: Current spot price

    Returns:
        List of strike prices
    """
    return [round(spot_price * factor, 2) for factor in STRIKE_RANGE_FACTORS]


@st.cache_data(ttl=3600)
def calculate_individual_leg_greeks(
    portfolio_json: str,
    spot_range: tuple,
    dte: int,
    iv: int,
    risk_free_rate: float,
    _calculate_all_greeks_func,
    _calculate_exotic_greeks_func=None,
    dividend_yield: float = 0.0,
) -> dict:
    """
    Calculate Greeks for each individual leg of a multi-leg portfolio.

    Args:
        portfolio_json: JSON string of portfolio data
        spot_range: Tuple of spot prices
        dte: Days to expiration
        iv: Implied volatility (percentage)
        risk_free_rate: Risk-free interest rate
        _calculate_all_greeks_func: Function to calculate Greeks
        _calculate_exotic_greeks_func: Function to calculate exotic Greeks

    Returns:
        Dictionary with Greeks data for each leg, keyed by leg index
    """
    portfolio_data = json.loads(portfolio_json)
    spot_range_arr = np.array(spot_range)
    time_to_expiry = dte / 365.0
    iv_decimal = iv / 100.0

    leg_greeks = {}

    # Calculate Greeks for each option leg
    if portfolio_data.get("options"):
        for leg_idx, pos in enumerate(portfolio_data["options"]):
            strike = pos["strike"]
            option_type = 1 if pos["option_type"] == "call" else 0
            position_sign = 1 if pos["position_type"] == "long" else -1
            quantity = pos["quantity"] * CONTRACT_MULTIPLIER
            inst_class = pos.get("instrument_class", "vanilla")
            is_exotic = inst_class != "vanilla"

            greeks_by_name = {
                name: np.zeros(len(spot_range_arr)) for name in GREEK_NAMES
            }

            for i, spot in enumerate(spot_range_arr):
                if is_exotic and _calculate_exotic_greeks_func:
                    greeks = _calculate_exotic_greeks_func(
                        spot,
                        strike,
                        time_to_expiry,
                        risk_free_rate,
                        iv_decimal,
                        option_type,
                        exotic_type=inst_class,
                        barrier=pos.get("barrier", 0.0),
                        is_up=pos.get("is_up", True),
                        is_knock_in=pos.get("is_knock_in", False),
                        rebate=pos.get("rebate", 0.0),
                        payout=pos.get("payout", 1.0),
                        extra1=pos.get("extra1", 0.0),
                        dividend_yield=dividend_yield,
                    )
                else:
                    greeks = _calculate_all_greeks_func(
                        spot,
                        strike,
                        time_to_expiry,
                        risk_free_rate,
                        iv_decimal,
                        option_type,
                        dividend_yield,
                    )

                for k, name in enumerate(GREEK_NAMES):
                    greeks_by_name[name][i] = greeks[k] * quantity * position_sign

            # Add metadata for display
            leg_data = {
                "greeks": greeks_by_name,
                "option_type": pos["option_type"],
                "position_type": pos["position_type"],
                "strike": strike,
                "quantity": pos["quantity"],
            }
            if is_exotic:
                leg_data["instrument_class"] = inst_class
            leg_greeks[f"leg_{leg_idx}"] = leg_data

    # Calculate Greeks for stock position
    if portfolio_data.get("stock"):
        stock = portfolio_data["stock"]
        stock_quantity = stock["quantity"] * (
            1 if stock["position_type"] == "long" else -1
        )

        stock_greeks = {name: np.zeros(len(spot_range_arr)) for name in GREEK_NAMES}
        stock_greeks["delta"] = np.full(len(spot_range_arr), stock_quantity)

        leg_greeks["stock"] = {
            "greeks": stock_greeks,
            "position_type": stock["position_type"],
            "quantity": stock["quantity"],
        }

    return leg_greeks


@st.cache_data(ttl=3600)
def calculate_all_individual_leg_greeks(
    portfolio_json: str,
    spot_range: tuple,
    dte_values: tuple,
    iv_values: tuple,
    risk_free_rate: float,
    _calculate_all_greeks_func,
    _calculate_exotic_greeks_func=None,
    dividend_yield: float = 0.0,
) -> dict:
    """
    Calculate Greeks for each individual leg for all DTE/IV combinations.

    Args:
        portfolio_json: JSON string of portfolio data
        spot_range: Tuple of spot prices
        dte_values: Tuple of DTE values
        iv_values: Tuple of IV values
        risk_free_rate: Risk-free interest rate
        _calculate_all_greeks_func: Function to calculate Greeks
        _calculate_exotic_greeks_func: Function to calculate exotic Greeks

    Returns:
        Dictionary keyed by "DTE_IV" with leg Greeks for each combination
    """
    portfolio_data = json.loads(portfolio_json)
    spot_range_arr = np.array(spot_range)

    # Extract leg metadata once
    leg_metadata = {}
    if portfolio_data.get("options"):
        for leg_idx, pos in enumerate(portfolio_data["options"]):
            meta = {
                "option_type": pos["option_type"],
                "position_type": pos["position_type"],
                "strike": pos["strike"],
                "quantity": pos["quantity"],
                "option_type_int": 1 if pos["option_type"] == "call" else 0,
                "position_sign": 1 if pos["position_type"] == "long" else -1,
                "quantity_mult": pos["quantity"] * CONTRACT_MULTIPLIER,
                "instrument_class": pos.get("instrument_class", "vanilla"),
            }
            if meta["instrument_class"] != "vanilla":
                meta["barrier"] = pos.get("barrier", 0.0)
                meta["is_up"] = pos.get("is_up", True)
                meta["is_knock_in"] = pos.get("is_knock_in", False)
                meta["rebate"] = pos.get("rebate", 0.0)
                meta["payout"] = pos.get("payout", 1.0)
            leg_metadata[f"leg_{leg_idx}"] = meta

    if portfolio_data.get("stock"):
        stock = portfolio_data["stock"]
        leg_metadata["stock"] = {
            "position_type": stock["position_type"],
            "quantity": stock["quantity"],
            "stock_quantity": stock["quantity"]
            * (1 if stock["position_type"] == "long" else -1),
        }

    all_leg_greeks = {}

    for dte in dte_values:
        if dte == 0:
            # At expiration, all Greeks are zero — populate zero entries
            for iv in iv_values:
                key = f"0_{iv}"
                leg_greeks = {}
                for leg_key, meta in leg_metadata.items():
                    zero_greeks = {
                        name: np.zeros(len(spot_range_arr)) for name in GREEK_NAMES
                    }
                    if leg_key == "stock":
                        zero_greeks["delta"] = np.full(
                            len(spot_range_arr), meta["stock_quantity"]
                        )
                        leg_greeks["stock"] = {
                            "greeks": zero_greeks,
                            "position_type": meta["position_type"],
                            "quantity": meta["quantity"],
                        }
                    else:
                        leg_data = {
                            "greeks": zero_greeks,
                            "option_type": meta["option_type"],
                            "position_type": meta["position_type"],
                            "strike": meta["strike"],
                            "quantity": meta["quantity"],
                        }
                        if meta["instrument_class"] != "vanilla":
                            leg_data["instrument_class"] = meta["instrument_class"]
                        leg_greeks[leg_key] = leg_data
                all_leg_greeks[key] = leg_greeks
            continue

        time_to_expiry = dte / 365.0

        for iv in iv_values:
            key = f"{dte}_{iv}"
            iv_decimal = iv / 100.0

            leg_greeks = {}

            # Calculate Greeks for each option leg
            for leg_key, meta in leg_metadata.items():
                if leg_key == "stock":
                    stock_greeks = {
                        name: np.zeros(len(spot_range_arr)) for name in GREEK_NAMES
                    }
                    stock_greeks["delta"] = np.full(
                        len(spot_range_arr), meta["stock_quantity"]
                    )
                    leg_greeks["stock"] = {
                        "greeks": stock_greeks,
                        "position_type": meta["position_type"],
                        "quantity": meta["quantity"],
                    }
                else:
                    is_exotic = meta["instrument_class"] != "vanilla"
                    greeks_by_name = {
                        name: np.zeros(len(spot_range_arr)) for name in GREEK_NAMES
                    }

                    for i, spot in enumerate(spot_range_arr):
                        if is_exotic and _calculate_exotic_greeks_func:
                            greeks = _calculate_exotic_greeks_func(
                                spot,
                                meta["strike"],
                                time_to_expiry,
                                risk_free_rate,
                                iv_decimal,
                                meta["option_type_int"],
                                exotic_type=meta["instrument_class"],
                                barrier=meta.get("barrier", 0.0),
                                is_up=meta.get("is_up", True),
                                is_knock_in=meta.get("is_knock_in", False),
                                rebate=meta.get("rebate", 0.0),
                                payout=meta.get("payout", 1.0),
                                extra1=meta.get("extra1", 0.0),
                                dividend_yield=dividend_yield,
                            )
                        else:
                            greeks = _calculate_all_greeks_func(
                                spot,
                                meta["strike"],
                                time_to_expiry,
                                risk_free_rate,
                                iv_decimal,
                                meta["option_type_int"],
                                dividend_yield,
                            )

                        for k, name in enumerate(GREEK_NAMES):
                            val = (
                                greeks[k]
                                * meta["quantity_mult"]
                                * meta["position_sign"]
                            )
                            greeks_by_name[name][i] = val if not np.isnan(val) else 0.0

                    leg_data = {
                        "greeks": greeks_by_name,
                        "option_type": meta["option_type"],
                        "position_type": meta["position_type"],
                        "strike": meta["strike"],
                        "quantity": meta["quantity"],
                    }
                    if is_exotic:
                        leg_data["instrument_class"] = meta["instrument_class"]
                    leg_greeks[leg_key] = leg_data

            all_leg_greeks[key] = leg_greeks

    return all_leg_greeks
