"""Strategy builder section of the sidebar."""

import streamlit as st
from components.sidebar_add_leg import render_add_leg_button
from components.sidebar_leg_editor import (
    render_exotic_leg_editor,
    render_leg_editor,
    render_stock_leg_editor,
)
from components.sidebar_positions import render_strategy_summary
from components.sidebar_structured import SP_KEYS, render_structured_product_section
from config.constants import (
    DEFAULT_DTE,
    DEFAULT_IV,
    STRATEGIES_WITH_STOCK,
    STRATEGY_DESCRIPTIONS,
    STRATEGY_DISPLAY_NAMES,
    STRATEGY_LEGS,
    STRATEGY_STOCK_POSITION,
)
from config.exotic_config import EXOTIC_LEG_KEYS
from services.exotic_pricing_adapter import (
    calculate_exotic_premium,
    haug_factory_params,
)
from services.pricing_adapter import calculate_option_premium
from services.state_manager import create_option_position, create_stock_position


def render_strategy_builder(
    spot_price: float, risk_free_rate: float, dividend_yield: float = 0.0
) -> None:
    """Render the strategy builder section with editable legs."""

    st.markdown(
        """
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
        <span style="font-size: 1rem;">🎯</span>
        <span style="font-size: 0.75rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em;">Strategy Builder</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Grouped strategies for better organization in dropdown
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
        "── Exotic - Digital ──": None,
        "digital_call": "🎯  Digital Call",
        "digital_put": "🎯  Digital Put",
        "digital_range_bet": "🎯  Digital Range Bet",
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

    # Build options list (excluding separators for actual selection)
    all_options = [""] + [k for k in strategy_groups.keys() if not k.startswith("──")]

    def format_strategy(key):
        if key == "":
            return "Select a strategy..."
        if key.startswith("──"):
            return key
        return strategy_groups.get(key, STRATEGY_DISPLAY_NAMES.get(key, key))

    # Strategy dropdown with custom styling
    # Use a versioned key to allow resetting the dropdown
    selector_version = st.session_state.get("selector_version", 0)
    selected_strategy = st.selectbox(
        "Strategy",
        all_options,
        format_func=format_strategy,
        key=f"strategy_selector_v{selector_version}",
        label_visibility="collapsed",
    )

    if not selected_strategy:
        st.session_state.sp_mode = False
        st.markdown(
            """
        <div style="text-align: center; padding: 2rem 1rem; color: #94a3b8; font-size: 0.85rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 10px; border: 1px dashed #cbd5e1; margin-top: 0.5rem;">
            <div style="font-size: 2rem; margin-bottom: 0.75rem; opacity: 0.7;">📈</div>
            <div style="font-weight: 500; color: #64748b;">Select a strategy to begin</div>
            <div style="font-size: 0.75rem; margin-top: 0.5rem; color: #94a3b8;">Choose from vanilla options, spreads,<br/>straddles, and more complex strategies</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

    # Structured Products: separate rendering path
    if selected_strategy in SP_KEYS:
        st.session_state.sp_mode = True
        # Clear option positions when switching to structured products
        if st.session_state.get("last_selected_strategy") != selected_strategy:
            st.session_state.positions = []
            st.session_state.stock_position = None
            st.session_state.last_selected_strategy = selected_strategy
        render_structured_product_section(selected_strategy, spot_price, risk_free_rate)
        return

    st.session_state.sp_mode = False
    # Clear stale SP state when leaving structured product mode
    for sp_key in (
        "sp_config",
        "sp_result",
        "sp_greeks",
        "sp_product_type",
        "_sp_last_priced_strategy",
        "_sp_last_priced_spot",
        "_sp_last_priced_rate",
    ):
        st.session_state.pop(sp_key, None)

    # Get strategy configuration
    is_custom = selected_strategy == "custom"
    base_strategy_legs = STRATEGY_LEGS.get(selected_strategy, [])

    # Initialize session states if needed
    if "custom_legs" not in st.session_state:
        st.session_state.custom_legs = []
    if "additional_legs" not in st.session_state:
        st.session_state.additional_legs = {}
    if "additional_stock" not in st.session_state:
        st.session_state.additional_stock = {}
    if "custom_has_stock" not in st.session_state:
        st.session_state.custom_has_stock = False

    # Check if strategy changed - if so, reset additional state BEFORE calculating has_stock/strategy_legs
    strategy_changed = (
        st.session_state.get("last_selected_strategy") != selected_strategy
    )
    if strategy_changed:
        st.session_state.additional_legs = {}
        st.session_state.additional_stock = {}
        st.session_state.custom_legs = []
        st.session_state.custom_has_stock = False
        st.session_state.removed_base_legs = {}

    # Now calculate has_stock and strategy_legs with clean state
    has_stock = selected_strategy in STRATEGIES_WITH_STOCK

    if is_custom:
        strategy_legs = st.session_state.custom_legs
        has_stock = st.session_state.get("custom_has_stock", False)
        base_leg_count = 0
        base_original_map = []
    else:
        # Filter out removed base legs
        removed_base_indices = st.session_state.get("removed_base_legs", {}).get(
            selected_strategy, set()
        )
        filtered_base = []
        base_original_map = []
        for orig_i, leg in enumerate(base_strategy_legs):
            if orig_i not in removed_base_indices:
                filtered_base.append(leg)
                base_original_map.append(orig_i)
        # Combine filtered base legs with any additional legs
        additional_legs_key = selected_strategy
        additional = st.session_state.additional_legs.get(additional_legs_key, [])
        strategy_legs = filtered_base + additional
        base_leg_count = len(filtered_base)
        # Check if stock was added manually to this strategy
        has_stock = has_stock or st.session_state.additional_stock.get(
            additional_legs_key, False
        )

    # Initialize session state and check if auto-apply is needed
    should_auto_apply = _initialize_strategy_state(
        selected_strategy, strategy_legs, has_stock, spot_price, risk_free_rate
    )

    # Auto-apply strategy when newly selected or spot price changes
    if should_auto_apply and not is_custom:
        _apply_strategy(
            spot_price, risk_free_rate, strategy_legs, has_stock, dividend_yield
        )

    # Strategy info header with hover tooltip
    strategy_name = STRATEGY_DISPLAY_NAMES.get(
        selected_strategy, "Custom Strategy" if is_custom else selected_strategy
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
                <div style="font-weight: 600; color: #ffffff; font-size: 0.95rem;">{strategy_name}</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.75rem; margin-top: 0.2rem;">
                    {num_legs} leg{"s" if num_legs > 1 else ""}{" + 100 shares" if has_stock else ""}
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

    # Track total cost for summary
    total_net_cost = 0.0
    stock_should_remove = False

    # Determine if stock is removable (added manually, not part of base strategy)
    base_has_stock = selected_strategy in STRATEGIES_WITH_STOCK
    stock_is_removable = is_custom or (has_stock and not base_has_stock)

    # Render stock position if needed
    if has_stock:
        stock_cost, stock_should_remove = render_stock_leg_editor(
            spot_price, is_custom, selected_strategy, is_removable=stock_is_removable
        )
        # Handle stock cost based on position type
        stock_state = st.session_state.strategy_legs_state.get("stock", {})
        if stock_state.get("position_type", "long") == "long":
            total_net_cost -= stock_cost  # Long stock is a debit
        else:
            total_net_cost += stock_cost  # Short stock is a credit
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    # Handle stock removal
    if stock_should_remove:
        if is_custom:
            st.session_state.custom_has_stock = False
        else:
            additional_legs_key = selected_strategy
            if "additional_stock" in st.session_state:
                st.session_state.additional_stock[additional_legs_key] = False
        if "stock" in st.session_state.strategy_legs_state:
            del st.session_state.strategy_legs_state["stock"]
        st.session_state.strategy_version = (
            st.session_state.get("strategy_version", 0) + 1
        )
        st.rerun()

    # Render each option leg with improved styling
    leg_costs = []
    legs_to_remove = []

    for i, leg in enumerate(strategy_legs):
        is_additional_leg = i >= base_leg_count
        allow_remove = True  # All legs are removable

        # Check if this is an exotic leg
        leg_state = st.session_state.strategy_legs_state.get(i, {})
        inst_class = leg_state.get(
            "instrument_class", leg.get("instrument_class", "vanilla")
        )

        if inst_class != "vanilla":
            leg_cost, should_remove = render_exotic_leg_editor(
                i,
                leg,
                spot_price,
                risk_free_rate,
                num_legs,
                allow_remove=allow_remove,
                dividend_yield=dividend_yield,
            )
        else:
            leg_cost, should_remove = render_leg_editor(
                i,
                leg,
                spot_price,
                risk_free_rate,
                num_legs,
                allow_remove=allow_remove,
                is_additional=is_additional_leg and not is_custom,
                dividend_yield=dividend_yield,
            )
        leg_costs.append(leg_cost)

        if should_remove:
            legs_to_remove.append(i)

        leg_state = st.session_state.strategy_legs_state.get(i, {})
        if leg_state.get("position_type") == "long":
            total_net_cost -= leg_cost
        else:
            total_net_cost += leg_cost

    # Handle leg removal
    if legs_to_remove:
        for idx in sorted(legs_to_remove, reverse=True):
            if is_custom:
                if idx < len(st.session_state.custom_legs):
                    st.session_state.custom_legs.pop(idx)
            else:
                if idx < base_leg_count:
                    # Base strategy leg — track as removed
                    orig_idx = base_original_map[idx]
                    if "removed_base_legs" not in st.session_state:
                        st.session_state.removed_base_legs = {}
                    if selected_strategy not in st.session_state.removed_base_legs:
                        st.session_state.removed_base_legs[selected_strategy] = set()
                    st.session_state.removed_base_legs[selected_strategy].add(orig_idx)
                else:
                    # Additional leg
                    additional_idx = idx - base_leg_count
                    additional_legs_key = selected_strategy
                    if (
                        0
                        <= additional_idx
                        < len(
                            st.session_state.additional_legs.get(
                                additional_legs_key, []
                            )
                        )
                    ):
                        st.session_state.additional_legs[additional_legs_key].pop(
                            additional_idx
                        )
        # Reset the legs state
        st.session_state.strategy_legs_state = {}
        st.session_state.strategy_version = (
            st.session_state.get("strategy_version", 0) + 1
        )
        st.rerun()

    # Add Leg button for all strategies
    render_add_leg_button(
        spot_price, is_custom, selected_strategy, strategy_legs, has_stock
    )

    # Summary section
    render_strategy_summary(total_net_cost, has_stock)

    # Apply button (for manual modifications)
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    if st.button(
        "✓  Apply Changes", width="stretch", type="primary", key="apply_strategy_btn"
    ):
        # Read values directly from widget keys for each leg to ensure we have the latest values
        version = st.session_state.get("strategy_version", 0)
        for i in range(len(strategy_legs)):
            type_key = f"leg_{i}_type_v{version}"
            dir_key = f"leg_{i}_dir_v{version}"
            strike_key = f"leg_{i}_strike_v{version}"
            qty_key = f"leg_{i}_qty_v{version}"

            # Update strategy_legs_state from widget values if they exist
            if type_key in st.session_state:
                # Preserve existing exotic fields from current state
                existing_state = st.session_state.strategy_legs_state.get(i, {})
                new_state = {
                    "option_type": st.session_state[type_key],
                    "position_type": st.session_state[dir_key],
                    "strike": st.session_state[strike_key],
                    "quantity": st.session_state[qty_key],
                }
                # Carry over exotic fields
                inst_class = existing_state.get("instrument_class", "vanilla")
                if inst_class != "vanilla":
                    new_state["instrument_class"] = inst_class
                    for ekey in ("barrier", "is_up", "is_knock_in", "rebate", "payout"):
                        # Read from widget if available, else from existing state
                        widget_key = f"leg_{i}_{ekey}_v{version}"
                        if widget_key in st.session_state:
                            new_state[ekey] = st.session_state[widget_key]
                        elif ekey in existing_state:
                            new_state[ekey] = existing_state[ekey]
                    # Handle barrier direction widget
                    barrier_dir_key = f"leg_{i}_barrier_dir_v{version}"
                    if barrier_dir_key in st.session_state:
                        new_state["is_up"] = st.session_state[barrier_dir_key] == "Up"
                    ki_key = f"leg_{i}_ki_v{version}"
                    if ki_key in st.session_state:
                        new_state["is_knock_in"] = (
                            st.session_state[ki_key] == "Knock-In"
                        )
                    payout_key = f"leg_{i}_payout_v{version}"
                    if payout_key in st.session_state:
                        new_state["payout"] = st.session_state[payout_key]
                    # Carry over any other exotic keys this widget-specific
                    # rebuild doesn't read directly (supershare corridor,
                    # double/discrete/partial/binary-barrier params, Asian
                    # avg_*, ...) so Apply Changes doesn't drop them.
                    for key in EXOTIC_LEG_KEYS:
                        if key not in new_state and key in existing_state:
                            new_state[key] = existing_state[key]
                st.session_state.strategy_legs_state[i] = new_state

        # Also update stock state from widget values
        if has_stock:
            stock_dir_key = f"stock_leg_dir_v{version}"
            stock_qty_key = f"stock_leg_qty_v{version}"
            stock_entry_key = f"stock_leg_entry_v{version}"

            if stock_dir_key in st.session_state:
                st.session_state.strategy_legs_state["stock"] = {
                    "position_type": st.session_state[stock_dir_key],
                    "quantity": st.session_state[stock_qty_key],
                    "entry_price": st.session_state[stock_entry_key],
                }

        _apply_strategy(
            spot_price, risk_free_rate, strategy_legs, has_stock, dividend_yield
        )


def _initialize_strategy_state(
    selected_strategy: str,
    strategy_legs: list,
    has_stock: bool,
    spot_price: float,
    risk_free_rate: float = 0.05,
) -> bool:
    """
    Initialize or reset strategy state when strategy or spot price changes.

    Returns:
        bool: True if the strategy should be auto-applied (new selection or spot change)
    """
    if "strategy_legs_state" not in st.session_state:
        st.session_state.strategy_legs_state = {}

    if "strategy_version" not in st.session_state:
        st.session_state.strategy_version = 0

    if "last_spot_price" not in st.session_state:
        st.session_state.last_spot_price = spot_price

    strategy_changed = (
        st.session_state.get("last_selected_strategy") != selected_strategy
    )
    spot_changed = st.session_state.get("last_spot_price") != spot_price

    should_auto_apply = False

    if strategy_changed or spot_changed:
        st.session_state.last_selected_strategy = selected_strategy
        st.session_state.last_spot_price = spot_price
        st.session_state.strategy_legs_state = {}

        # Increment version to force widget key changes and reset values
        st.session_state.strategy_version = (
            st.session_state.get("strategy_version", 0) + 1
        )

        # Clear old widget keys from session state to force reset
        keys_to_remove = [
            key
            for key in list(st.session_state.keys())
            if key.startswith("leg_") or key.startswith("stock_leg_")
        ]
        for key in keys_to_remove:
            del st.session_state[key]

        for i, leg in enumerate(strategy_legs):
            state = {
                "option_type": leg["option_type"],
                "position_type": leg["position_type"],
                "strike": round(spot_price * leg.get("strike_factor", 1.0), 2),
                "quantity": leg["quantity"],
            }
            # Preserve exotic fields if present
            inst_class = leg.get("instrument_class", "vanilla")
            if inst_class != "vanilla":
                state["instrument_class"] = inst_class
                if "barrier_factor" in leg:
                    state["barrier"] = round(spot_price * leg["barrier_factor"], 2)
                for ekey in (
                    "is_up",
                    "is_knock_in",
                    "rebate",
                    "payout",
                    "choice_time_pct",
                    "power_n",
                ):
                    if ekey in leg:
                        state[ekey] = leg[ekey]
                if "gap_trigger_factor" in leg:
                    state["gap_trigger"] = round(
                        spot_price * leg["gap_trigger_factor"], 2
                    )
                # Compute extra1 for types that need it
                if inst_class == "chooser":
                    maturity = DEFAULT_DTE / 365.0
                    state["extra1"] = leg.get("choice_time_pct", 0.5) * maturity
                elif inst_class == "power":
                    n_val = leg.get("power_n", 2.0)
                    state["extra1"] = n_val
                    # Strike must be in S^n space for ATM: K = S^n
                    state["strike"] = round(spot_price**n_val, 2)
                elif inst_class == "gap":
                    state["extra1"] = round(
                        spot_price * leg.get("gap_trigger_factor", 1.05), 2
                    )
            st.session_state.strategy_legs_state[i] = state

        if has_stock:
            # Get the stock position type from strategy definition (default to 'long')
            stock_position_type = STRATEGY_STOCK_POSITION.get(selected_strategy, "long")
            st.session_state.strategy_legs_state["stock"] = {
                "position_type": stock_position_type,
                "quantity": 100,
                "entry_price": spot_price,
            }

        # Auto-apply strategy when newly selected or spot price changes
        should_auto_apply = True

    return should_auto_apply


def _apply_strategy(
    spot_price: float,
    risk_free_rate: float,
    strategy_legs: list,
    has_stock: bool,
    dividend_yield: float = 0.0,
) -> None:
    """Apply the configured strategy to positions using dict-based storage."""
    new_positions = []

    for i, _ in enumerate(strategy_legs):
        leg_state = st.session_state.strategy_legs_state.get(i, {})
        if leg_state:
            inst_class = leg_state.get("instrument_class", "vanilla")
            is_exotic = inst_class != "vanilla"

            # Calculate premium
            if is_exotic:
                premium = calculate_exotic_premium(
                    spot=spot_price,
                    strike=leg_state["strike"],
                    dte_days=DEFAULT_DTE,
                    risk_free_rate=risk_free_rate,
                    volatility=DEFAULT_IV / 100,
                    option_type=leg_state["option_type"],
                    exotic_type=inst_class,
                    barrier=leg_state.get("barrier", 0.0),
                    is_up=leg_state.get("is_up", True),
                    is_knock_in=leg_state.get("is_knock_in", False),
                    rebate=leg_state.get("rebate", 0.0),
                    payout=leg_state.get("payout", 1.0),
                    extra1=leg_state.get("extra1", 0.0),
                    dividend_yield=dividend_yield,
                    cap=leg_state.get("cap", 0.0),
                    params=haug_factory_params(leg_state),
                )
            else:
                premium = calculate_option_premium(
                    spot=spot_price,
                    strike=leg_state["strike"],
                    dte_days=DEFAULT_DTE,
                    risk_free_rate=risk_free_rate,
                    volatility=DEFAULT_IV / 100,
                    option_type=leg_state["option_type"],
                    dividend_yield=dividend_yield,
                )

            # Create dict-based position
            position = create_option_position(
                option_type=leg_state["option_type"],
                position_type=leg_state["position_type"],
                strike=leg_state["strike"],
                quantity=leg_state["quantity"],
                premium_paid=premium,
                instrument_class=inst_class,
                barrier=leg_state.get("barrier"),
                is_up=leg_state.get("is_up"),
                is_knock_in=leg_state.get("is_knock_in"),
                rebate=leg_state.get("rebate", 0.0),
                payout=leg_state.get("payout", 1.0),
                extra1=leg_state.get("extra1", 0.0),
                choice_time_pct=leg_state.get("choice_time_pct", 0.5),
                power_n=leg_state.get("power_n", 2.0),
                gap_trigger=leg_state.get("gap_trigger"),
            )
            # create_option_position only knows the basic exotic fields; carry
            # the advanced-family keys (corridor, adv_* barrier flags, cash,
            # binary_type, ...) from the editor state so the leg keeps
            # describing the instrument the user configured.
            if is_exotic:
                for key in EXOTIC_LEG_KEYS:
                    if key not in position and key in leg_state:
                        position[key] = leg_state[key]
            new_positions.append(position)

    new_stock_position = None
    if has_stock:
        stock_state = st.session_state.strategy_legs_state.get("stock")
        if stock_state:
            new_stock_position = create_stock_position(
                position_type=stock_state["position_type"],
                quantity=stock_state["quantity"],
                entry_price=stock_state["entry_price"],
            )

    st.session_state.positions = new_positions
    st.session_state.stock_position = new_stock_position
    st.rerun()
