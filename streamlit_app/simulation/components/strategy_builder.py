"""
Strategy Builder for Monte Carlo P&L Simulation.

Provides the same strategy builder interface as options_greeks,
with editable legs and full customization support.
"""

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import streamlit as st

# Strategy constants — self-contained for the simulation app
from config.strategy_constants import (
    CONTRACT_MULTIPLIER,
    INSTRUMENT_CLASSES,
    STRATEGIES_WITH_STOCK,
    STRATEGY_DESCRIPTIONS,
    STRATEGY_DISPLAY_NAMES,
    STRATEGY_LEGS,
    STRATEGY_STOCK_POSITION,
)

# Load exotic pricing functions via importlib (avoids streamlit_app __init__ chain)
_options_greeks_path = Path(__file__).parent.parent.parent / "options_greeks"
_exotic_adapter_path = _options_greeks_path / "services" / "exotic_pricing_adapter.py"
_exotic_spec = importlib.util.spec_from_file_location(
    "exotic_pricing_adapter", _exotic_adapter_path
)
_exotic_adapter = importlib.util.module_from_spec(_exotic_spec)
_exotic_spec.loader.exec_module(_exotic_adapter)

calculate_exotic_price = _exotic_adapter.calculate_exotic_price
calculate_exotic_payoff_at_expiry = _exotic_adapter.calculate_exotic_payoff_at_expiry


# =============================================================================
# Data Classes for Positions
# =============================================================================


@dataclass
class SimulationOptionPosition:
    """Option position for P&L simulation."""

    option_type: str  # 'call' or 'put'
    position_type: str  # 'long' or 'short'
    strike: float
    quantity: int
    premium: float = 0.0
    # Exotic fields
    instrument_class: str = "vanilla"
    barrier: float = 0.0
    is_up: bool = True
    is_knock_in: bool = False
    rebate: float = 0.0
    payout: float = 1.0
    extra1: float = 0.0
    power_n: float = 2.0
    gap_trigger: float = 0.0
    choice_time_pct: float = 0.5
    # Haug-catalog families: capped-power ceiling + the family-specific registry
    # kwargs (supershare band, barrier levels, monitoring points, binary type, ...).
    cap: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class SimulationStockPosition:
    """Stock position for P&L simulation."""

    position_type: str  # 'long' or 'short'
    quantity: int
    entry_price: float


# =============================================================================
# Strategy Builder UI Component
# =============================================================================


def render_strategy_builder(
    spot_price: float,
    risk_free_rate: float,
    time_to_expiry: float,
    volatility: float,
    bs_price_function,
    exotic_price_function=None,
) -> tuple[list[SimulationOptionPosition], SimulationStockPosition | None]:
    """
    Render the strategy builder component for P&L simulation.
    Matches the options_greeks interface with editable legs.
    """
    # Strategy groups for dropdown with emojis (matching options_greeks exactly)
    strategy_groups = {
        "── Custom ──": None,
        "custom": "🎨  Custom Strategy",
        "── Directional ──": None,
        "long_call": "📈  Long Call",
        "short_call": "📉  Short Call",
        "long_put": "📈  Long Put",
        "short_put": "📉  Short Put",
        "── Vertical Spreads ──": None,
        "bull_call_spread": "🟢  Bull Call Spread",
        "bear_put_spread": "🔴  Bear Put Spread",
        "bull_put_spread": "🟢  Bull Put Spread",
        "bear_call_spread": "🔴  Bear Call Spread",
        "── Volatility ──": None,
        "long_straddle": "⚡  Long Straddle",
        "short_straddle": "⚡  Short Straddle",
        "long_strangle": "⚡  Long Strangle",
        "short_strangle": "⚡  Short Strangle",
        "── Advanced ──": None,
        "iron_condor": "🔷  Iron Condor",
        "butterfly": "🦋  Butterfly",
        "── Stock + Options ──": None,
        "covered_call": "🛡️  Covered Call",
        "protective_put": "🛡️  Protective Put",
        "collar": "🛡️  Collar",
        "── Exotic - Barrier ──": None,
        "barrier_up_out_call": "🔮  Up-and-Out Call",
        "barrier_down_out_put": "🔮  Down-and-Out Put",
        "barrier_knock_in_call": "🔮  Down-and-In Call",
        "── Exotic - Digital ──": None,
        "digital_call": "🎯  Digital Call",
        "digital_put": "🎯  Digital Put",
        "digital_range_bet": "🎯  Digital Range Bet",
        "── Exotic - Path-Dependent ──": None,
        "asian_call": "📊  Asian Call (Geometric)",
        "lookback_floating_call": "🔍  Lookback Call (Floating)",
        "lookback_fixed_call": "🔍  Lookback Call (Fixed)",
        "── Exotic - Other ──": None,
        "chooser": "🎲  Chooser Option",
        "asset_or_nothing_call": "💎  Asset-or-Nothing Call",
        "asset_or_nothing_put": "💎  Asset-or-Nothing Put",
        "power_call": "⚡  Power Call (n=2)",
        "gap_call": "📐  Gap Call",
        "── Exotic - Haug Power ──": None,
        "powered_call": "⚡  Powered Call (Esser)",
        "capped_power_call": "⚡  Capped Power Call (Esser)",
        "── Exotic - Haug Analytic ──": None,
        "log_contract": "📐  Log Contract",
        "log_option": "📐  Log Option",
        "supershare": "💠  Supershare",
        "── Exotic - Haug Barriers ──": None,
        "double_barrier_call": "🔮  Double Barrier Call",
        "discrete_barrier_call": "🔮  Discrete Barrier Call",
        "partial_barrier_call": "🔮  Partial-Time Barrier Call",
        "binary_barrier_call": "🔮  Binary Barrier",
    }

    _SP_KEYS = {"sp_cpn", "sp_reverse_convertible", "sp_autocallable"}

    # Build options list (excluding separators)
    all_options = [""] + [k for k in strategy_groups.keys() if not k.startswith("──")]

    def format_strategy(key):
        if key == "":
            return "Select a strategy..."
        if key.startswith("──"):
            return key
        return strategy_groups.get(key, STRATEGY_DISPLAY_NAMES.get(key, key))

    # Strategy selector
    selector_version = st.session_state.get("pnl_selector_version", 0)
    selected_strategy = st.selectbox(
        "Strategy",
        all_options,
        format_func=format_strategy,
        key=f"pnl_strategy_selector_v{selector_version}",
        label_visibility="collapsed",
    )

    if not selected_strategy:
        st.session_state.sp_config = None
        st.markdown(
            """
        <div style="text-align: center; padding: 2rem 1rem; color: #94a3b8; font-size: 0.85rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 10px; border: 1px dashed #cbd5e1; margin-top: 0.5rem;">
            <div style="font-size: 2rem; margin-bottom: 0.75rem; opacity: 0.7;">📈</div>
            <div style="font-weight: 500; color: #64748b;">Select a strategy to begin</div>
            <div style="font-size: 0.75rem; margin-top: 0.5rem; color: #94a3b8;">Choose from options, spreads,<br/>structured products, and more</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return [], None

    # ── Structured Products: separate rendering path ──
    if selected_strategy in _SP_KEYS:
        from components.structured_product_builder import (
            render_structured_product_builder,
        )

        st.session_state.sp_config = render_structured_product_builder(
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            time_to_expiry=time_to_expiry,
            volatility=volatility,
            product_type_key=selected_strategy[3:],  # strip "sp_" prefix
        )
        return [], None

    st.session_state.sp_config = None

    # Get strategy configuration
    is_custom = selected_strategy == "custom"
    base_strategy_legs = STRATEGY_LEGS.get(selected_strategy, [])

    # Initialize session states
    if "pnl_custom_legs" not in st.session_state:
        st.session_state.pnl_custom_legs = []
    if "pnl_additional_legs" not in st.session_state:
        st.session_state.pnl_additional_legs = {}
    if "pnl_additional_stock" not in st.session_state:
        st.session_state.pnl_additional_stock = {}
    if "pnl_custom_has_stock" not in st.session_state:
        st.session_state.pnl_custom_has_stock = False

    # Check if strategy changed
    strategy_changed = st.session_state.get("pnl_last_strategy") != selected_strategy
    if strategy_changed:
        st.session_state.pnl_additional_legs = {}
        st.session_state.pnl_additional_stock = {}
        st.session_state.pnl_custom_legs = []
        st.session_state.pnl_custom_has_stock = False

    # Calculate has_stock and strategy_legs
    has_stock = selected_strategy in STRATEGIES_WITH_STOCK

    if is_custom:
        strategy_legs = st.session_state.pnl_custom_legs
        has_stock = st.session_state.get("pnl_custom_has_stock", False)
    else:
        additional_legs_key = selected_strategy
        additional = st.session_state.pnl_additional_legs.get(additional_legs_key, [])
        strategy_legs = list(base_strategy_legs) + additional
        has_stock = has_stock or st.session_state.pnl_additional_stock.get(
            additional_legs_key, False
        )

    # Initialize state
    _initialize_strategy_state(selected_strategy, strategy_legs, has_stock, spot_price)

    # Strategy info header with hover tooltip
    strategy_name = strategy_groups.get(
        selected_strategy,
        STRATEGY_DISPLAY_NAMES.get(selected_strategy, "Custom Strategy"),
    )
    # Remove emoji prefix for display name
    clean_name = (
        strategy_name.split("  ")[-1] if "  " in strategy_name else strategy_name
    )
    num_legs = len(strategy_legs)
    desc = STRATEGY_DESCRIPTIONS.get(selected_strategy, "")
    tooltip_html = f'<div class="strat-header-tip">{desc}</div>' if desc else ""

    st.markdown(
        f"""
    <style>
    .strat-header {{position:relative;cursor:default;}}
    .strat-header-tip {{visibility:hidden;opacity:0;position:absolute;z-index:1000;top:100%;left:0;right:0;margin-top:0.25rem;padding:0.55rem 0.7rem;background:#1e293b;color:#e2e8f0;border-radius:8px;font-size:0.73rem;line-height:1.5;box-shadow:0 4px 16px rgba(0,0,0,0.4);border:1px solid #334155;transition:opacity 0.15s,visibility 0.15s;pointer-events:none;font-weight:400;}}
    .strat-header:hover .strat-header-tip {{visibility:visible;opacity:1;}}
    </style>
    <div class="strat-header" style="background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%); padding: 0.875rem 1rem; border-radius: 10px; margin: 0.75rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-weight: 600; color: #ffffff; font-size: 0.95rem;">{clean_name}</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.75rem; margin-top: 0.2rem;">
                    {num_legs} leg{"s" if num_legs != 1 else ""}{" + 100 shares" if has_stock else ""}
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.15); padding: 0.35rem 0.65rem; border-radius: 6px;">
                <span style="color: #fbbf24; font-size: 0.7rem; font-weight: 600; text-transform: uppercase;">{"Multi-Leg" if num_legs > 1 else "Single"}</span>
            </div>
        </div>
        {tooltip_html}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Track total cost
    total_net_cost = 0.0
    stock_should_remove = False

    # Determine if stock is removable
    base_has_stock = selected_strategy in STRATEGIES_WITH_STOCK
    stock_is_removable = is_custom or (has_stock and not base_has_stock)

    # Get version for widget keys
    version = st.session_state.get("pnl_strategy_version", 0)

    # Render stock position if needed
    if has_stock:
        stock_cost, stock_should_remove = _render_stock_leg_editor(
            spot_price,
            is_custom,
            selected_strategy,
            version,
            is_removable=stock_is_removable,
        )
        stock_state = st.session_state.pnl_strategy_legs_state.get("stock", {})
        if stock_state.get("position_type", "long") == "long":
            total_net_cost -= stock_cost
        else:
            total_net_cost += stock_cost
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    # Handle stock removal
    if stock_should_remove:
        if is_custom:
            st.session_state.pnl_custom_has_stock = False
        else:
            st.session_state.pnl_additional_stock[selected_strategy] = False
        if "stock" in st.session_state.pnl_strategy_legs_state:
            del st.session_state.pnl_strategy_legs_state["stock"]
        st.session_state.pnl_strategy_version = version + 1
        st.rerun()

    # Render each option leg with editable inputs
    leg_costs = []
    legs_to_remove = []
    base_leg_count = len(base_strategy_legs) if not is_custom else 0

    for i, leg in enumerate(strategy_legs):
        is_additional_leg = i >= base_leg_count
        allow_remove = is_custom or is_additional_leg

        leg_state = st.session_state.pnl_strategy_legs_state.get(i, {})
        inst_class = leg_state.get(
            "instrument_class", leg.get("instrument_class", "vanilla")
        )

        if inst_class and inst_class != "vanilla":
            leg_cost, should_remove = _render_exotic_leg_editor(
                i,
                leg,
                spot_price,
                risk_free_rate,
                time_to_expiry,
                volatility,
                num_legs,
                version,
                allow_remove=allow_remove,
                is_additional=is_additional_leg and not is_custom,
                exotic_price_function=exotic_price_function,
            )
        else:
            leg_cost, should_remove = _render_leg_editor(
                i,
                leg,
                spot_price,
                risk_free_rate,
                time_to_expiry,
                volatility,
                bs_price_function,
                num_legs,
                version,
                allow_remove=allow_remove,
                is_additional=is_additional_leg and not is_custom,
            )
        leg_costs.append(leg_cost)

        if should_remove:
            legs_to_remove.append(i)

        leg_state = st.session_state.pnl_strategy_legs_state.get(i, {})
        if leg_state.get("position_type") == "long":
            total_net_cost -= leg_cost
        else:
            total_net_cost += leg_cost

    # Handle leg removal
    if legs_to_remove:
        for idx in sorted(legs_to_remove, reverse=True):
            if is_custom:
                if idx < len(st.session_state.pnl_custom_legs):
                    st.session_state.pnl_custom_legs.pop(idx)
            else:
                additional_idx = idx - base_leg_count
                if additional_idx >= 0 and additional_idx < len(
                    st.session_state.pnl_additional_legs.get(selected_strategy, [])
                ):
                    st.session_state.pnl_additional_legs[selected_strategy].pop(
                        additional_idx
                    )
        st.session_state.pnl_strategy_legs_state = {}
        st.session_state.pnl_strategy_version = version + 1
        st.rerun()

    # Add Leg buttons (Option + Stock)
    _render_add_leg_buttons(
        spot_price, is_custom, selected_strategy, strategy_legs, has_stock
    )

    # Summary section
    _render_strategy_summary(total_net_cost, has_stock)

    # Build positions from state
    positions, stock_position = _build_positions_from_state(
        selected_strategy,
        spot_price,
        risk_free_rate,
        time_to_expiry,
        volatility,
        bs_price_function,
        is_custom,
        has_stock,
        exotic_price_function=exotic_price_function,
    )

    return positions, stock_position


def _initialize_strategy_state(
    selected_strategy: str, strategy_legs: list, has_stock: bool, spot_price: float
) -> bool:
    """Initialize or reset strategy state when strategy or spot price changes."""
    if "pnl_strategy_legs_state" not in st.session_state:
        st.session_state.pnl_strategy_legs_state = {}

    if "pnl_strategy_version" not in st.session_state:
        st.session_state.pnl_strategy_version = 0

    if "pnl_last_spot" not in st.session_state:
        st.session_state.pnl_last_spot = spot_price

    strategy_changed = st.session_state.get("pnl_last_strategy") != selected_strategy
    spot_changed = st.session_state.get("pnl_last_spot") != spot_price

    should_auto_apply = False

    if strategy_changed or spot_changed:
        st.session_state.pnl_last_strategy = selected_strategy
        st.session_state.pnl_last_spot = spot_price
        st.session_state.pnl_strategy_legs_state = {}
        st.session_state.pnl_strategy_version = (
            st.session_state.get("pnl_strategy_version", 0) + 1
        )

        # Clear old widget keys
        keys_to_remove = [
            key
            for key in list(st.session_state.keys())
            if key.startswith("pnl_leg_") or key.startswith("pnl_stock_leg_")
        ]
        for key in keys_to_remove:
            del st.session_state[key]

        for i, leg in enumerate(strategy_legs):
            state = {
                "option_type": leg["option_type"],
                "position_type": leg["position_type"],
                "strike": round(spot_price * leg["strike_factor"], 2),
                "quantity": leg["quantity"],
            }
            inst_class = leg.get("instrument_class", "vanilla")
            if inst_class != "vanilla":
                state["instrument_class"] = inst_class
                if "barrier_factor" in leg:
                    state["barrier"] = round(spot_price * leg["barrier_factor"], 2)
                for key in (
                    "is_up",
                    "is_knock_in",
                    "rebate",
                    "payout",
                    "choice_time_pct",
                    "power_n",
                ):
                    if key in leg:
                        state[key] = leg[key]
                if "gap_trigger_factor" in leg:
                    state["gap_trigger"] = round(
                        spot_price * leg["gap_trigger_factor"], 2
                    )
                # Compute extra1 for types that need it
                if inst_class == "chooser":
                    from config.constants import DEFAULT_TIME_HORIZON

                    maturity = DEFAULT_TIME_HORIZON
                    state["extra1"] = leg.get("choice_time_pct", 0.5) * maturity
                elif inst_class == "power":
                    n_val = leg.get("power_n", 2.0)
                    state["extra1"] = n_val
                    # Strike must be in S^n space for ATM
                    state["strike"] = round(spot_price**n_val, 2)
                elif inst_class == "gap":
                    state["extra1"] = round(
                        spot_price * leg.get("gap_trigger_factor", 1.05), 2
                    )
            st.session_state.pnl_strategy_legs_state[i] = state

        if has_stock:
            stock_position_type = STRATEGY_STOCK_POSITION.get(selected_strategy, "long")
            st.session_state.pnl_strategy_legs_state["stock"] = {
                "position_type": stock_position_type,
                "quantity": 100,
                "entry_price": spot_price,
            }

        should_auto_apply = True

    return should_auto_apply


def _render_leg_editor(
    leg_index: int,
    leg_config: dict,
    spot_price: float,
    risk_free_rate: float,
    time_to_expiry: float,
    volatility: float,
    bs_price_function,
    total_legs: int,
    version: int,
    allow_remove: bool = False,
    is_additional: bool = False,
) -> tuple[float, bool]:
    """Render an editable leg configuration. Returns (total_cost, should_remove)."""
    leg_state = st.session_state.pnl_strategy_legs_state.get(leg_index, {})
    should_remove = False

    option_type = leg_state.get("option_type", leg_config.get("option_type", "call"))
    position_type = leg_state.get(
        "position_type", leg_config.get("position_type", "long")
    )
    is_long = position_type == "long"

    # Visual styling
    border_color = "#10b981" if is_long else "#ef4444"
    bg_gradient = (
        "linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%)"
        if is_long
        else "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"
    )
    position_badge_bg = "#d1fae5" if is_long else "#fee2e2"
    position_badge_color = "#047857" if is_long else "#b91c1c"

    # Label for the leg
    leg_label = f"Leg {leg_index + 1}"
    if is_additional:
        leg_label = f"+ Leg {leg_index + 1}"
        bg_gradient = (
            "linear-gradient(135deg, #fefce8 0%, #fef9c3 100%)"
            if is_long
            else "linear-gradient(135deg, #fef2f2 0%, #fecaca 100%)"
        )

    added_badge = (
        '<span style="background: #fef3c7; color: #92400e; font-size: 0.55rem; font-weight: 600; padding: 0.15rem 0.35rem; border-radius: 3px; margin-left: 0.25rem;">ADDED</span>'
        if is_additional
        else ""
    )

    leg_header_html = f"""<div style="background: {bg_gradient}; border: 1px solid {border_color}40; border-left: 4px solid {border_color}; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.625rem;"><div style="display: flex; justify-content: space-between; align-items: center;"><div style="display: flex; align-items: center; gap: 0.5rem;"><span style="font-size: 0.7rem; font-weight: 700; color: #475569; text-transform: uppercase;">{leg_label}</span>{added_badge}</div><span style="background: {position_badge_bg}; color: {position_badge_color}; font-size: 0.65rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; text-transform: uppercase;">{position_type}</span></div></div>"""

    if allow_remove:
        header_col1, header_col2 = st.columns([4, 1])
        with header_col1:
            st.markdown(leg_header_html, unsafe_allow_html=True)
        with header_col2:
            st.markdown("<div style='height: 0.25rem'></div>", unsafe_allow_html=True)
            if st.button(
                "🗑️",
                key=f"pnl_remove_leg_{leg_index}_v{version}",
                help="Remove this leg",
            ):
                should_remove = True
    else:
        st.markdown(leg_header_html, unsafe_allow_html=True)

    # Editable inputs
    col1, col2 = st.columns(2)

    with col1:
        new_option_type = st.selectbox(
            "Type",
            ["call", "put"],
            index=0 if option_type == "call" else 1,
            format_func=lambda x: f"{'📈' if x == 'call' else '📉'} {x.upper()}",
            key=f"pnl_leg_{leg_index}_type_v{version}",
        )

    with col2:
        new_position_type = st.selectbox(
            "Direction",
            ["long", "short"],
            index=0 if position_type == "long" else 1,
            format_func=lambda x: f"{'🟢' if x == 'long' else '🔴'} {x.upper()}",
            key=f"pnl_leg_{leg_index}_dir_v{version}",
        )

    default_strike = leg_state.get(
        "strike", round(spot_price * leg_config.get("strike_factor", 1.0), 2)
    )

    col3, col4 = st.columns(2)

    with col3:
        new_strike = st.number_input(
            "Strike ($)",
            value=float(default_strike),
            step=1.0,
            format="%.2f",
            key=f"pnl_leg_{leg_index}_strike_v{version}",
        )

    with col4:
        default_qty = leg_state.get("quantity", leg_config.get("quantity", 1))
        new_quantity = st.number_input(
            "Contracts",
            value=int(default_qty),
            min_value=1,
            step=1,
            key=f"pnl_leg_{leg_index}_qty_v{version}",
        )

    # Update state
    st.session_state.pnl_strategy_legs_state[leg_index] = {
        "option_type": new_option_type,
        "position_type": new_position_type,
        "strike": new_strike,
        "quantity": new_quantity,
    }

    # Calculate premium
    try:
        premium = bs_price_function(
            spot_price,
            new_strike,
            risk_free_rate,
            time_to_expiry,
            volatility,
            new_option_type,
        )
    except Exception:
        premium = 0.0

    total_cost = premium * new_quantity * CONTRACT_MULTIPLIER

    # Cost display
    is_long_now = new_position_type == "long"
    cost_color = "#dc2626" if is_long_now else "#059669"
    cost_prefix = "-" if is_long_now else "+"

    st.markdown(
        f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0.75rem; background: #ffffff; border-radius: 6px; border: 1px solid #e2e8f0; margin-top: -0.5rem; margin-bottom: 0.5rem;">
        <span style="color: #64748b; font-size: 0.8rem;">
            Premium: <span style="font-family: 'JetBrains Mono', monospace; font-weight: 500;">${premium:.2f}</span>
        </span>
        <span style="color: {cost_color}; font-weight: 700; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">
            {cost_prefix}${total_cost:,.2f}
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    return total_cost, should_remove


def _render_exotic_leg_editor(
    leg_index: int,
    leg_config: dict,
    spot_price: float,
    risk_free_rate: float,
    time_to_expiry: float,
    volatility: float,
    total_legs: int,
    version: int,
    allow_remove: bool = False,
    is_additional: bool = False,
    exotic_price_function=None,
) -> tuple[float, bool]:
    """Render an editable exotic leg configuration. Returns (total_cost, should_remove)."""
    leg_state = st.session_state.pnl_strategy_legs_state.get(leg_index, {})
    should_remove = False

    option_type = leg_state.get("option_type", leg_config.get("option_type", "call"))
    position_type = leg_state.get(
        "position_type", leg_config.get("position_type", "long")
    )
    inst_class = leg_state.get(
        "instrument_class", leg_config.get("instrument_class", "barrier")
    )
    is_long = position_type == "long"

    inst_display = INSTRUMENT_CLASSES.get(inst_class, inst_class.title())

    # Visual styling — purple theme for exotic legs
    border_color = "#8b5cf6"
    bg_gradient = "linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%)"
    position_badge_bg = "#d1fae5" if is_long else "#fee2e2"
    position_badge_color = "#047857" if is_long else "#b91c1c"

    leg_label = f"Leg {leg_index + 1}"
    if is_additional:
        leg_label = f"+ Leg {leg_index + 1}"

    added_badge = (
        '<span style="background: #fef3c7; color: #92400e; font-size: 0.55rem; font-weight: 600; padding: 0.15rem 0.35rem; border-radius: 3px; margin-left: 0.25rem;">ADDED</span>'
        if is_additional
        else ""
    )

    leg_header_html = f"""<div style="background: {bg_gradient}; border: 1px solid {border_color}40; border-left: 4px solid {border_color}; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.625rem;"><div style="display: flex; justify-content: space-between; align-items: center;"><div style="display: flex; align-items: center; gap: 0.5rem;"><span style="font-size: 0.7rem; font-weight: 700; color: #475569; text-transform: uppercase;">{leg_label}</span><span style="background: #ede9fe; color: #6d28d9; font-size: 0.55rem; font-weight: 600; padding: 0.15rem 0.35rem; border-radius: 3px;">EXOTIC</span><span style="background: #ddd6fe; color: #5b21b6; font-size: 0.55rem; font-weight: 600; padding: 0.15rem 0.35rem; border-radius: 3px;">{inst_display}</span>{added_badge}</div><span style="background: {position_badge_bg}; color: {position_badge_color}; font-size: 0.65rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; text-transform: uppercase;">{position_type}</span></div></div>"""

    if allow_remove:
        header_col1, header_col2 = st.columns([4, 1])
        with header_col1:
            st.markdown(leg_header_html, unsafe_allow_html=True)
        with header_col2:
            st.markdown("<div style='height: 0.25rem'></div>", unsafe_allow_html=True)
            if st.button(
                "🗑️",
                key=f"pnl_remove_leg_{leg_index}_v{version}",
                help="Remove this leg",
            ):
                should_remove = True
    else:
        st.markdown(leg_header_html, unsafe_allow_html=True)

    # Core parameters
    col1, col2 = st.columns(2)
    with col1:
        new_option_type = st.selectbox(
            "Type",
            ["call", "put"],
            index=0 if option_type == "call" else 1,
            format_func=lambda x: f"{'📈' if x == 'call' else '📉'} {x.upper()}",
            key=f"pnl_leg_{leg_index}_type_v{version}",
        )
    with col2:
        new_position_type = st.selectbox(
            "Direction",
            ["long", "short"],
            index=0 if position_type == "long" else 1,
            format_func=lambda x: f"{'🟢' if x == 'long' else '🔴'} {x.upper()}",
            key=f"pnl_leg_{leg_index}_dir_v{version}",
        )

    default_strike = leg_state.get(
        "strike", round(spot_price * leg_config.get("strike_factor", 1.0), 2)
    )

    col3, col4 = st.columns(2)
    with col3:
        new_strike = st.number_input(
            "Strike ($)",
            value=float(default_strike),
            step=1.0,
            format="%.2f",
            key=f"pnl_leg_{leg_index}_strike_v{version}",
        )
    with col4:
        default_qty = leg_state.get("quantity", leg_config.get("quantity", 1))
        new_quantity = st.number_input(
            "Contracts",
            value=int(default_qty),
            min_value=1,
            step=1,
            key=f"pnl_leg_{leg_index}_qty_v{version}",
        )

    # Instrument-specific parameters
    new_barrier = leg_state.get(
        "barrier", round(spot_price * leg_config.get("barrier_factor", 1.10), 2)
    )
    new_is_up = leg_state.get("is_up", leg_config.get("is_up", True))
    new_is_knock_in = leg_state.get("is_knock_in", leg_config.get("is_knock_in", False))
    new_rebate = leg_state.get("rebate", leg_config.get("rebate", 0.0))
    new_payout = leg_state.get("payout", leg_config.get("payout", 1.0))
    new_choice_time_pct = leg_state.get(
        "choice_time_pct", leg_config.get("choice_time_pct", 0.5)
    )
    new_power_n = leg_state.get("power_n", leg_config.get("power_n", 2.0))
    new_gap_trigger = leg_state.get(
        "gap_trigger", round(spot_price * leg_config.get("gap_trigger_factor", 1.05), 2)
    )
    # Haug-catalog extras: the capped-power ceiling + a generic registry-kwargs
    # dict (band / barrier levels / monitoring points / binary type).
    new_cap = leg_state.get("cap", leg_config.get("cap", 0.0))
    new_params: dict = dict(leg_state.get("params", leg_config.get("params", {})))

    if inst_class == "barrier":
        col5, col6 = st.columns(2)
        with col5:
            new_barrier = st.number_input(
                "Barrier ($)",
                value=float(new_barrier),
                step=1.0,
                format="%.2f",
                key=f"pnl_leg_{leg_index}_barrier_v{version}",
            )
        with col6:
            barrier_dir = st.selectbox(
                "Barrier Dir",
                ["Up", "Down"],
                index=0 if new_is_up else 1,
                key=f"pnl_leg_{leg_index}_bdir_v{version}",
            )
            new_is_up = barrier_dir == "Up"

        col7, col8 = st.columns(2)
        with col7:
            ki_type = st.selectbox(
                "Barrier Type",
                ["Knock-Out", "Knock-In"],
                index=1 if new_is_knock_in else 0,
                key=f"pnl_leg_{leg_index}_ki_v{version}",
            )
            new_is_knock_in = ki_type == "Knock-In"
        with col8:
            new_rebate = st.number_input(
                "Rebate ($)",
                value=float(new_rebate),
                step=0.1,
                format="%.2f",
                min_value=0.0,
                disabled=new_is_knock_in,
                key=f"pnl_leg_{leg_index}_rebate_v{version}",
            )

        # Warn if barrier already breached
        if new_is_up and new_barrier <= spot_price:
            st.warning(f"Up barrier ${new_barrier:.2f} <= spot ${spot_price:.2f}")
        elif not new_is_up and new_barrier >= spot_price:
            st.warning(f"Down barrier ${new_barrier:.2f} >= spot ${spot_price:.2f}")

    elif inst_class == "digital":
        new_payout = st.number_input(
            "Payout ($)",
            value=float(new_payout),
            min_value=0.01,
            step=0.1,
            format="%.2f",
            key=f"pnl_leg_{leg_index}_payout_v{version}",
        )

    elif inst_class == "chooser":
        new_choice_time_pct = st.slider(
            "Choice time (% of maturity)",
            min_value=0.1,
            max_value=0.9,
            value=float(new_choice_time_pct),
            step=0.05,
            key=f"pnl_leg_{leg_index}_choice_v{version}",
        )

    elif inst_class == "power":
        new_power_n = st.number_input(
            "Power exponent (n)",
            value=float(new_power_n),
            min_value=1.0,
            max_value=5.0,
            step=0.5,
            help="Payoff = max(S^n - K, 0). n=1 is vanilla, n=2 is quadratic.",
            key=f"pnl_leg_{leg_index}_power_v{version}",
        )

    elif inst_class == "gap":
        new_gap_trigger = st.number_input(
            "Trigger strike K2 ($)",
            value=float(new_gap_trigger),
            step=1.0,
            format="%.2f",
            help="Call pays (S-K1) if S>K2. When K1=K2, equals vanilla.",
            key=f"pnl_leg_{leg_index}_gap_v{version}",
        )

    elif inst_class == "powered":
        new_power_n = st.number_input(
            "Power exponent (i)",
            value=int(new_power_n),
            min_value=1,
            max_value=4,
            step=1,
            help="Esser powered payoff max(η(S-K), 0)^i. Integer exponent.",
            key=f"pnl_leg_{leg_index}_powered_v{version}",
        )

    elif inst_class == "capped_power":
        cp1, cp2 = st.columns(2)
        with cp1:
            new_power_n = st.number_input(
                "Power exponent (i)",
                value=float(new_power_n),
                min_value=1.0,
                max_value=4.0,
                step=0.5,
                key=f"pnl_leg_{leg_index}_cppow_v{version}",
            )
        with cp2:
            new_cap = st.number_input(
                "Cap ($)",
                value=float(new_cap if new_cap > 0 else round(0.5 * new_strike, 2)),
                min_value=0.01,
                step=1.0,
                format="%.2f",
                help="Payoff = min(max(η(S^i - K), 0), cap).",
                key=f"pnl_leg_{leg_index}_cpcap_v{version}",
            )

    elif inst_class in ("log_contract", "log_option"):
        st.caption(
            "Pays ln(S/K)"
            if inst_class == "log_contract"
            else "Pays max(ln(S/K), 0)"
        )

    elif inst_class == "supershare":
        ss1, ss2 = st.columns(2)
        with ss1:
            ss_lo = st.number_input(
                "Lower X_L ($)",
                value=float(new_params.get("lower_strike", round(0.9 * new_strike, 2))),
                step=1.0,
                format="%.2f",
                key=f"pnl_leg_{leg_index}_sslo_v{version}",
            )
        with ss2:
            ss_hi = st.number_input(
                "Upper X_H ($)",
                value=float(new_params.get("upper_strike", round(1.1 * new_strike, 2))),
                step=1.0,
                format="%.2f",
                key=f"pnl_leg_{leg_index}_sshi_v{version}",
            )
        new_params["lower_strike"] = ss_lo
        new_params["upper_strike"] = ss_hi

    elif inst_class == "double_barrier":
        db1, db2 = st.columns(2)
        with db1:
            db_lo = st.number_input(
                "Lower L ($)",
                value=float(new_params.get("lower", round(0.8 * new_strike, 2))),
                step=1.0,
                format="%.2f",
                key=f"pnl_leg_{leg_index}_dblo_v{version}",
            )
        with db2:
            db_hi = st.number_input(
                "Upper U ($)",
                value=float(new_params.get("upper", round(1.2 * new_strike, 2))),
                step=1.0,
                format="%.2f",
                key=f"pnl_leg_{leg_index}_dbhi_v{version}",
            )
        db_ki = st.selectbox(
            "Type",
            ["Knock-Out", "Knock-In"],
            index=1 if new_params.get("is_knock_in", False) else 0,
            key=f"pnl_leg_{leg_index}_dbki_v{version}",
        )
        new_params["lower"] = db_lo
        new_params["upper"] = db_hi
        new_params["is_knock_in"] = db_ki == "Knock-In"

    elif inst_class == "discrete_barrier":
        dc1, dc2 = st.columns(2)
        with dc1:
            dc_bar = st.number_input(
                "Barrier ($)",
                value=float(new_params.get("barrier", round(1.1 * new_strike, 2))),
                step=1.0,
                format="%.2f",
                key=f"pnl_leg_{leg_index}_dcbar_v{version}",
            )
        with dc2:
            dc_dir = st.selectbox(
                "Direction",
                ["Up", "Down"],
                index=0 if new_params.get("is_up", True) else 1,
                key=f"pnl_leg_{leg_index}_dcdir_v{version}",
            )
        dc3, dc4 = st.columns(2)
        with dc3:
            dc_ki = st.selectbox(
                "Type",
                ["Knock-Out", "Knock-In"],
                index=1 if new_params.get("is_knock_in", False) else 0,
                key=f"pnl_leg_{leg_index}_dcki_v{version}",
            )
        with dc4:
            dc_m = st.number_input(
                "Monitoring dates",
                value=int(new_params.get("monitoring_points", 52)),
                min_value=1,
                step=1,
                help="Number of discrete monitoring points (BGK correction).",
                key=f"pnl_leg_{leg_index}_dcm_v{version}",
            )
        new_params["barrier"] = dc_bar
        new_params["is_up"] = dc_dir == "Up"
        new_params["is_knock_in"] = dc_ki == "Knock-In"
        new_params["monitoring_points"] = int(dc_m)

    elif inst_class == "partial_barrier":
        _pb_types = ("down_out_A", "up_out_A", "out_B1", "down_out_B2", "up_out_B2")
        pb1, pb2 = st.columns(2)
        with pb1:
            pb_bar = st.number_input(
                "Barrier ($)",
                value=float(new_params.get("barrier", round(1.1 * new_strike, 2))),
                step=1.0,
                format="%.2f",
                key=f"pnl_leg_{leg_index}_pbbar_v{version}",
            )
        with pb2:
            _pb_cur = new_params.get("barrier_type", "out_B1")
            pb_type = st.selectbox(
                "Monitoring",
                _pb_types,
                index=_pb_types.index(_pb_cur) if _pb_cur in _pb_types else 2,
                key=f"pnl_leg_{leg_index}_pbtype_v{version}",
            )
        _t1_default = (
            float(new_params.get("t1", 0.5 * time_to_expiry)) / time_to_expiry
            if time_to_expiry > 0
            else 0.5
        )
        pb_t1 = st.slider(
            "Window end t1 (% of maturity)",
            min_value=0.1,
            max_value=1.0,
            value=min(max(_t1_default, 0.1), 1.0),
            step=0.05,
            key=f"pnl_leg_{leg_index}_pbt1_v{version}",
        )
        new_params["barrier"] = pb_bar
        new_params["barrier_type"] = pb_type
        new_params["t1"] = pb_t1 * time_to_expiry

    elif inst_class == "binary_barrier":
        bb1, bb2 = st.columns(2)
        with bb1:
            bb_bar = st.number_input(
                "Barrier ($)",
                value=float(new_params.get("barrier", round(1.1 * new_strike, 2))),
                step=1.0,
                format="%.2f",
                key=f"pnl_leg_{leg_index}_bbbar_v{version}",
            )
        with bb2:
            bb_cash = st.number_input(
                "Cash ($)",
                value=float(new_params.get("cash", 10.0)),
                min_value=0.0,
                step=1.0,
                format="%.2f",
                key=f"pnl_leg_{leg_index}_bbcash_v{version}",
            )
        bb_type = st.number_input(
            "Reiner-Rubinstein type (1-28)",
            value=int(new_params.get("binary_type", 13)),
            min_value=1,
            max_value=28,
            step=1,
            help="28 binary-barrier variants (down/up x in/out x cash/asset x gate).",
            key=f"pnl_leg_{leg_index}_bbtype_v{version}",
        )
        new_params["barrier"] = bb_bar
        new_params["cash"] = bb_cash
        new_params["binary_type"] = int(bb_type)

    # Compute extra1 for backend
    new_extra1 = 0.0
    if inst_class == "chooser":
        new_extra1 = new_choice_time_pct * time_to_expiry
    elif inst_class in ("power", "powered", "capped_power"):
        new_extra1 = new_power_n
    elif inst_class == "gap":
        new_extra1 = new_gap_trigger

    # Update state
    new_state = {
        "option_type": new_option_type,
        "position_type": new_position_type,
        "strike": new_strike,
        "quantity": new_quantity,
        "instrument_class": inst_class,
        "barrier": new_barrier,
        "is_up": new_is_up,
        "is_knock_in": new_is_knock_in,
        "rebate": new_rebate,
        "payout": new_payout,
        "extra1": new_extra1,
        "choice_time_pct": new_choice_time_pct,
        "power_n": new_power_n,
        "gap_trigger": new_gap_trigger,
        "cap": new_cap,
        "params": new_params,
    }
    st.session_state.pnl_strategy_legs_state[leg_index] = new_state

    # Calculate premium via the model-consistent exotic pricer (falls back to
    # the GBM closed form when no model-aware pricer is injected).
    _exotic_pricer = exotic_price_function or calculate_exotic_price
    try:
        premium = _exotic_pricer(
            exotic_type=inst_class,
            spot=spot_price,
            strike=new_strike,
            maturity=time_to_expiry,
            rate=risk_free_rate,
            sigma=volatility,
            is_call=(new_option_type == "call"),
            barrier=new_barrier,
            is_knock_in=new_is_knock_in,
            is_up=new_is_up,
            rebate=new_rebate,
            payout=new_payout,
            extra1=new_extra1,
            cap=new_cap,
            params=new_params,
        )
    except Exception:
        premium = 0.0

    total_cost = premium * new_quantity * CONTRACT_MULTIPLIER

    # Cost display
    is_long_now = new_position_type == "long"
    cost_color = "#dc2626" if is_long_now else "#059669"
    cost_prefix = "-" if is_long_now else "+"

    st.markdown(
        f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0.75rem; background: #ffffff; border-radius: 6px; border: 1px solid #ddd6fe; margin-top: -0.5rem; margin-bottom: 0.5rem;">
        <span style="color: #64748b; font-size: 0.8rem;">
            Premium: <span style="font-family: 'JetBrains Mono', monospace; font-weight: 500;">${premium:.2f}</span>
        </span>
        <span style="color: {cost_color}; font-weight: 700; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">
            {cost_prefix}${total_cost:,.2f}
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    return total_cost, should_remove


def _render_stock_leg_editor(
    spot_price: float,
    is_custom: bool,
    selected_strategy: str,
    version: int,
    is_removable: bool = False,
) -> tuple[float, bool]:
    """Render stock position editor. Returns (stock_cost, should_remove)."""
    stock_state = st.session_state.pnl_strategy_legs_state.get(
        "stock", {"position_type": "long", "quantity": 100, "entry_price": spot_price}
    )

    should_remove = False

    header_html = """
    <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 1px solid #3b82f640; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.625rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 0.7rem; font-weight: 700; color: #475569; text-transform: uppercase;">Stock Position</span>
            </div>
            <span style="background: #dbeafe; color: #1d4ed8; font-size: 0.65rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; text-transform: uppercase;">Underlying</span>
        </div>
    </div>
    """

    if is_removable:
        header_col1, header_col2 = st.columns([4, 1])
        with header_col1:
            st.markdown(header_html, unsafe_allow_html=True)
        with header_col2:
            st.markdown("<div style='height: 0.25rem'></div>", unsafe_allow_html=True)
            if st.button(
                "🗑️", key=f"pnl_remove_stock_v{version}", help="Remove stock position"
            ):
                should_remove = True
    else:
        st.markdown(header_html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        stock_direction = st.selectbox(
            "Direction",
            ["long", "short"],
            index=0 if stock_state.get("position_type", "long") == "long" else 1,
            format_func=lambda x: f"{'🟢' if x == 'long' else '🔴'} {x.upper()}",
            key=f"pnl_stock_leg_dir_v{version}",
        )

    with col2:
        stock_qty = st.number_input(
            "Shares",
            value=int(stock_state.get("quantity", 100)),
            min_value=1,
            step=100,
            key=f"pnl_stock_leg_qty_v{version}",
        )

    stock_entry = st.number_input(
        "Entry Price ($)",
        value=float(stock_state.get("entry_price", spot_price)),
        step=1.0,
        format="%.2f",
        key=f"pnl_stock_leg_entry_v{version}",
    )

    st.session_state.pnl_strategy_legs_state["stock"] = {
        "position_type": stock_direction,
        "quantity": stock_qty,
        "entry_price": stock_entry,
    }

    stock_cost = stock_entry * stock_qty
    is_long = stock_direction == "long"
    cost_color = "#dc2626" if is_long else "#059669"
    cost_prefix = "-" if is_long else "+"

    st.markdown(
        f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0.75rem; background: #ffffff; border-radius: 6px; border: 1px solid #e2e8f0; margin-top: -0.5rem;">
        <span style="color: #64748b; font-size: 0.8rem;">
            {stock_qty:,} shares @ ${stock_entry:.2f}
        </span>
        <span style="color: {cost_color}; font-weight: 700; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">
            {cost_prefix}${stock_cost:,.2f}
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    return stock_cost, should_remove


def _render_add_leg_buttons(
    spot_price: float,
    is_custom: bool,
    selected_strategy: str,
    strategy_legs: list,
    has_stock: bool,
) -> None:
    """Render the Add Leg buttons (Option + Stock)."""
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    # Initialize additional legs tracking
    if "pnl_additional_legs" not in st.session_state:
        st.session_state.pnl_additional_legs = {}

    additional_legs_key = selected_strategy if not is_custom else "custom"
    if additional_legs_key not in st.session_state.pnl_additional_legs:
        st.session_state.pnl_additional_legs[additional_legs_key] = []

    # Three buttons: Add Option, Add Exotic, Add Stock
    col1, col2, col3 = st.columns(3)
    with col1:
        add_option_clicked = st.button(
            "➕ Option", key="pnl_add_option_btn", type="secondary"
        )
    with col2:
        add_exotic_clicked = st.button(
            "➕ Exotic", key="pnl_add_exotic_btn", type="secondary"
        )
    with col3:
        add_stock_clicked = st.button(
            "➕ Stock", key="pnl_add_stock_btn", type="secondary", disabled=has_stock
        )

    if add_option_clicked:
        new_leg = {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
        }

        if is_custom:
            st.session_state.pnl_custom_legs.append(new_leg)
            new_index = len(st.session_state.pnl_custom_legs) - 1
        else:
            st.session_state.pnl_additional_legs[additional_legs_key].append(new_leg)
            new_index = (
                len(STRATEGY_LEGS.get(selected_strategy, []))
                + len(st.session_state.pnl_additional_legs[additional_legs_key])
                - 1
            )

        st.session_state.pnl_strategy_legs_state[new_index] = {
            "option_type": "call",
            "position_type": "long",
            "strike": round(spot_price, 2),
            "quantity": 1,
        }
        st.session_state.pnl_strategy_version = (
            st.session_state.get("pnl_strategy_version", 0) + 1
        )
        st.rerun()

    if add_exotic_clicked:
        # Show exotic type selector in session state, then add on next rerun
        exotic_types = list(INSTRUMENT_CLASSES.keys())
        exotic_key = "pnl_add_exotic_type"
        if exotic_key not in st.session_state:
            st.session_state[exotic_key] = exotic_types[0]

        # Default to first exotic type (barrier)
        default_exotic = exotic_types[0]
        new_leg = {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": default_exotic,
        }
        # Add barrier defaults
        if default_exotic == "barrier":
            new_leg["barrier_factor"] = 1.10
            new_leg["is_up"] = True
            new_leg["is_knock_in"] = False

        if is_custom:
            st.session_state.pnl_custom_legs.append(new_leg)
            new_index = len(st.session_state.pnl_custom_legs) - 1
        else:
            st.session_state.pnl_additional_legs[additional_legs_key].append(new_leg)
            new_index = (
                len(STRATEGY_LEGS.get(selected_strategy, []))
                + len(st.session_state.pnl_additional_legs[additional_legs_key])
                - 1
            )

        st.session_state.pnl_strategy_legs_state[new_index] = {
            "option_type": "call",
            "position_type": "long",
            "strike": round(spot_price, 2),
            "quantity": 1,
            "instrument_class": default_exotic,
            "barrier": round(spot_price * 1.10, 2),
            "is_up": True,
            "is_knock_in": False,
            "rebate": 0.0,
            "payout": 1.0,
            "extra1": 0.0,
            "choice_time_pct": 0.5,
            "power_n": 2.0,
            "gap_trigger": round(spot_price * 1.05, 2),
        }
        st.session_state.pnl_strategy_version = (
            st.session_state.get("pnl_strategy_version", 0) + 1
        )
        st.rerun()

    if add_stock_clicked:
        if is_custom:
            st.session_state.pnl_custom_has_stock = True
        else:
            if "pnl_additional_stock" not in st.session_state:
                st.session_state.pnl_additional_stock = {}
            st.session_state.pnl_additional_stock[additional_legs_key] = True

        st.session_state.pnl_strategy_legs_state["stock"] = {
            "position_type": "long",
            "quantity": 100,
            "entry_price": spot_price,
        }
        st.session_state.pnl_strategy_version = (
            st.session_state.get("pnl_strategy_version", 0) + 1
        )
        st.rerun()

    # Show hint for custom with no legs
    if (
        is_custom
        and len(st.session_state.pnl_custom_legs) == 0
        and not st.session_state.get("pnl_custom_has_stock", False)
    ):
        st.markdown(
            """
        <div style="text-align: center; padding: 1.5rem 1rem; color: #94a3b8; font-size: 0.85rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 10px; border: 1px dashed #cbd5e1; margin-top: 0.5rem;">
            <div style="font-size: 1.5rem; margin-bottom: 0.75rem; opacity: 0.7;">🎨</div>
            <div style="font-weight: 500; color: #64748b;">Build your custom strategy</div>
            <div style="font-size: 0.75rem; margin-top: 0.5rem; color: #94a3b8;">Add options or stock positions</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def _render_strategy_summary(total_net_cost: float, has_stock: bool) -> None:
    """Render strategy cost summary."""
    is_debit = total_net_cost < 0

    if is_debit:
        summary_bg = "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"
        summary_border = "#fca5a5"
        summary_label = "Total Debit"
        summary_color = "#dc2626"
        summary_icon = "💸"
        display_amount = f"-${abs(total_net_cost):,.2f}"
    else:
        summary_bg = "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)"
        summary_border = "#86efac"
        summary_label = "Total Credit"
        summary_color = "#059669"
        summary_icon = "💰"
        display_amount = f"+${abs(total_net_cost):,.2f}"

    st.markdown(
        f"""
    <div style="background: {summary_bg}; border: 1px solid {summary_border}; border-radius: 10px; padding: 1rem; margin-top: 0.75rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.25rem;">{summary_icon}</span>
                <div>
                    <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 600;">{summary_label}</div>
                    <div style="font-size: 0.7rem; color: #94a3b8;">{"Incl. stock" if has_stock else "Options only"}</div>
                </div>
            </div>
            <div style="font-size: 1.35rem; font-weight: 700; color: {summary_color}; font-family: 'JetBrains Mono', monospace;">
                {display_amount}
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def _build_positions_from_state(
    selected_strategy: str,
    spot_price: float,
    risk_free_rate: float,
    time_to_expiry: float,
    volatility: float,
    bs_price_function,
    is_custom: bool,
    has_stock: bool,
    exotic_price_function=None,
) -> tuple[list[SimulationOptionPosition], SimulationStockPosition | None]:
    """Build position objects from session state."""
    positions = []
    stock_position = None

    # Determine number of legs
    if is_custom:
        num_legs = len(st.session_state.get("pnl_custom_legs", []))
    else:
        base_legs = len(STRATEGY_LEGS.get(selected_strategy, []))
        additional = len(
            st.session_state.pnl_additional_legs.get(selected_strategy, [])
        )
        num_legs = base_legs + additional

    # Build option positions
    for i in range(num_legs):
        leg_state = st.session_state.pnl_strategy_legs_state.get(i, {})
        if not leg_state:
            continue

        strike = leg_state["strike"]
        option_type = leg_state["option_type"]
        inst_class = leg_state.get("instrument_class", "vanilla")

        if inst_class != "vanilla":
            # Exotic premium via the model-consistent pricer (GBM closed form
            # fallback when no model-aware pricer is injected).
            extra1 = leg_state.get("extra1", 0.0)
            _exotic_pricer = exotic_price_function or calculate_exotic_price
            try:
                premium = _exotic_pricer(
                    exotic_type=inst_class,
                    spot=spot_price,
                    strike=strike,
                    maturity=time_to_expiry,
                    rate=risk_free_rate,
                    sigma=volatility,
                    is_call=(option_type == "call"),
                    barrier=leg_state.get("barrier", 0.0),
                    is_knock_in=leg_state.get("is_knock_in", False),
                    is_up=leg_state.get("is_up", True),
                    rebate=leg_state.get("rebate", 0.0),
                    payout=leg_state.get("payout", 1.0),
                    extra1=extra1,
                    cap=leg_state.get("cap", 0.0),
                    params=leg_state.get("params", {}),
                )
            except Exception:
                premium = 0.0
        else:
            try:
                premium = bs_price_function(
                    spot_price,
                    strike,
                    risk_free_rate,
                    time_to_expiry,
                    volatility,
                    option_type,
                )
            except Exception:
                premium = 0.0

        positions.append(
            SimulationOptionPosition(
                option_type=leg_state["option_type"],
                position_type=leg_state["position_type"],
                strike=leg_state["strike"],
                quantity=leg_state["quantity"],
                premium=premium,
                instrument_class=inst_class,
                barrier=leg_state.get("barrier", 0.0),
                is_up=leg_state.get("is_up", True),
                is_knock_in=leg_state.get("is_knock_in", False),
                rebate=leg_state.get("rebate", 0.0),
                payout=leg_state.get("payout", 1.0),
                extra1=leg_state.get("extra1", 0.0),
                power_n=leg_state.get("power_n", 2.0),
                gap_trigger=leg_state.get("gap_trigger", 0.0),
                choice_time_pct=leg_state.get("choice_time_pct", 0.5),
                cap=leg_state.get("cap", 0.0),
                params=dict(leg_state.get("params", {})),
            )
        )

    # Build stock position
    if has_stock:
        stock_state = st.session_state.pnl_strategy_legs_state.get("stock")
        if stock_state:
            stock_position = SimulationStockPosition(
                position_type=stock_state["position_type"],
                quantity=stock_state["quantity"],
                entry_price=stock_state["entry_price"],
            )

    return positions, stock_position


# =============================================================================
# Position Export Functions
# =============================================================================


def export_positions_for_pnl_engine(
    positions: list[SimulationOptionPosition],
    stock_position: SimulationStockPosition | None = None,
    risk_free_rate: float = 0.05,
    maturity: float = 1.0,
) -> dict[str, np.ndarray]:
    """Export positions to numpy arrays for the Numba P&L engine."""
    n_legs = len(positions)

    if n_legs == 0:
        return {
            "strikes": np.array([], dtype=np.float64),
            "option_types": np.array([], dtype=np.float64),
            "position_types": np.array([], dtype=np.float64),
            "quantities": np.array([], dtype=np.float64),
            "premiums": np.array([], dtype=np.float64),
            "stock_quantity": 0.0,
            "stock_entry_price": 0.0,
        }

    strikes = np.zeros(n_legs, dtype=np.float64)
    option_types = np.zeros(n_legs, dtype=np.float64)
    position_types = np.zeros(n_legs, dtype=np.float64)
    quantities = np.zeros(n_legs, dtype=np.float64)
    premiums = np.zeros(n_legs, dtype=np.float64)

    for i, pos in enumerate(positions):
        strikes[i] = pos.strike
        option_types[i] = 1.0 if pos.option_type == "call" else -1.0
        position_types[i] = 1.0 if pos.position_type == "long" else -1.0
        quantities[i] = pos.quantity
        premiums[i] = pos.premium

    result = {
        "strikes": strikes,
        "option_types": option_types,
        "position_types": position_types,
        "quantities": quantities,
        "premiums": premiums,
        "stock_quantity": 0.0,
        "stock_entry_price": 0.0,
    }

    # Exotic metadata for hybrid P&L calculation
    result["exotic_metadata"] = [
        {
            "instrument_class": pos.instrument_class,
            "option_type": pos.option_type,
            "position_type": pos.position_type,
            "strike": pos.strike,
            "barrier": pos.barrier,
            "is_up": pos.is_up,
            "is_knock_in": pos.is_knock_in,
            "payout": pos.payout,
            "power_n": pos.power_n,
            "gap_trigger": pos.gap_trigger,
            "choice_time_pct": pos.choice_time_pct,
            "rebate": pos.rebate,
            "cap": pos.cap,
            "params": pos.params,
            "r": risk_free_rate,
            "maturity": maturity,
        }
        for pos in positions
    ]

    if stock_position:
        sign = 1.0 if stock_position.position_type == "long" else -1.0
        result["stock_quantity"] = sign * stock_position.quantity
        result["stock_entry_price"] = stock_position.entry_price

    return result
