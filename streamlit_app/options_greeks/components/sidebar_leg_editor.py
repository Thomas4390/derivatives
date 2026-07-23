"""Leg editor components for the sidebar (vanilla, exotic, stock)."""

import streamlit as st
from config.constants import (
    BINARY_BARRIER_TYPES,
    CONTRACT_MULTIPLIER,
    DEFAULT_BARRIER_UP_FACTOR,
    DEFAULT_DIGITAL_PAYOUT,
    DEFAULT_DTE,
    DEFAULT_IV,
    INSTRUMENT_CLASSES,
    PARTIAL_BARRIER_TYPES,
)
from services.exotic_pricing_adapter import calculate_exotic_premium
from services.pricing_adapter import calculate_option_premium


def render_leg_editor(
    leg_index: int,
    leg_config: dict,
    spot_price: float,
    risk_free_rate: float,
    total_legs: int,
    allow_remove: bool = False,
    is_additional: bool = False,
    dividend_yield: float = 0.0,
) -> tuple[float, bool]:
    """Render an editable leg configuration. Returns (total_cost, should_remove)."""
    leg_state = st.session_state.strategy_legs_state.get(leg_index, {})
    should_remove = False

    option_type = leg_state.get("option_type", leg_config["option_type"])
    position_type = leg_state.get("position_type", leg_config["position_type"])
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

    # Leg container with optional remove button
    version = st.session_state.get("strategy_version", 0)

    # Label for the leg
    leg_label = f"Leg {leg_index + 1}"
    if is_additional:
        leg_label = f"+ Leg {leg_index + 1}"
        # Use a slightly different style for additional legs
        bg_gradient = (
            "linear-gradient(135deg, #fefce8 0%, #fef9c3 100%)"
            if is_long
            else "linear-gradient(135deg, #fef2f2 0%, #fecaca 100%)"
        )

    # Build the added badge HTML separately
    added_badge = (
        '<span style="background: #fef3c7; color: #92400e; font-size: 0.55rem; font-weight: 600; padding: 0.15rem 0.35rem; border-radius: 3px; margin-left: 0.25rem;">ADDED</span>'
        if is_additional
        else ""
    )

    # Build the leg header HTML
    leg_header_html = f"""<div style="background: {bg_gradient}; border: 1px solid {border_color}40; border-left: 4px solid {border_color}; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.625rem;"><div style="display: flex; justify-content: space-between; align-items: center;"><div style="display: flex; align-items: center; gap: 0.5rem;"><span style="font-size: 0.7rem; font-weight: 700; color: #475569; text-transform: uppercase;">{leg_label}</span>{added_badge}</div><span style="background: {position_badge_bg}; color: {position_badge_color}; font-size: 0.65rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; text-transform: uppercase;">{position_type}</span></div></div>"""

    if allow_remove:
        # Header with remove button integrated
        header_col1, header_col2 = st.columns([4, 1])
        with header_col1:
            st.markdown(leg_header_html, unsafe_allow_html=True)
        with header_col2:
            st.markdown("<div style='height: 0.25rem'></div>", unsafe_allow_html=True)
            if st.button(
                "🗑️",
                key=f"remove_leg_{leg_index}_v{version}",
                help="Remove this leg",
                width="stretch",
            ):
                should_remove = True
    else:
        st.markdown(leg_header_html, unsafe_allow_html=True)

    # Move inputs outside the HTML div
    col1, col2 = st.columns(2)

    with col1:
        new_option_type = st.selectbox(
            "Type",
            ["call", "put"],
            index=0 if option_type == "call" else 1,
            format_func=lambda x: f"{'📈' if x == 'call' else '📉'} {x.upper()}",
            key=f"leg_{leg_index}_type_v{version}",
        )

    with col2:
        new_position_type = st.selectbox(
            "Direction",
            ["long", "short"],
            index=0 if position_type == "long" else 1,
            format_func=lambda x: f"{'🟢' if x == 'long' else '🔴'} {x.upper()}",
            key=f"leg_{leg_index}_dir_v{version}",
        )

    # Strike and quantity inputs
    default_strike = leg_state.get(
        "strike", round(spot_price * leg_config["strike_factor"], 2)
    )

    col3, col4 = st.columns(2)

    with col3:
        new_strike = st.number_input(
            "Strike ($)",
            value=float(default_strike),
            step=1.0,
            format="%.2f",
            key=f"leg_{leg_index}_strike_v{version}",
        )

    with col4:
        default_qty = leg_state.get("quantity", leg_config["quantity"])
        new_quantity = st.number_input(
            "Contracts",
            value=int(default_qty),
            min_value=1,
            step=1,
            key=f"leg_{leg_index}_qty_v{version}",
        )

    # Update state
    st.session_state.strategy_legs_state[leg_index] = {
        "option_type": new_option_type,
        "position_type": new_position_type,
        "strike": new_strike,
        "quantity": new_quantity,
    }

    # Calculate premium using Black-Scholes
    premium = calculate_option_premium(
        spot=spot_price,
        strike=new_strike,
        dte_days=DEFAULT_DTE,
        risk_free_rate=risk_free_rate,
        volatility=DEFAULT_IV / 100,  # Convert from percentage
        option_type=new_option_type,
        dividend_yield=dividend_yield,
    )
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


def render_exotic_leg_editor(
    leg_index: int,
    leg_config: dict,
    spot_price: float,
    risk_free_rate: float,
    total_legs: int,
    allow_remove: bool = True,
    dividend_yield: float = 0.0,
) -> tuple[float, bool]:
    """Render an exotic leg editor with purple styling. Returns (total_cost, should_remove)."""
    leg_state = st.session_state.strategy_legs_state.get(leg_index, {})
    should_remove = False
    version = st.session_state.get("strategy_version", 0)

    inst_class = leg_state.get(
        "instrument_class", leg_config.get("instrument_class", "barrier")
    )
    display_name = INSTRUMENT_CLASSES.get(inst_class, inst_class)
    option_type = leg_state.get("option_type", leg_config.get("option_type", "call"))
    position_type = leg_state.get(
        "position_type", leg_config.get("position_type", "long")
    )
    is_long = position_type == "long"

    # Purple styling for exotic legs
    border_color = "#8b5cf6"
    bg_gradient = "linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%)"
    position_badge_bg = "#ddd6fe" if is_long else "#fecaca"
    position_badge_color = "#6d28d9" if is_long else "#b91c1c"

    leg_header_html = f"""<div style="background: {bg_gradient}; border: 1px solid {border_color}40; border-left: 4px solid {border_color}; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.625rem;"><div style="display: flex; justify-content: space-between; align-items: center;"><div style="display: flex; align-items: center; gap: 0.5rem;"><span style="font-size: 0.7rem; font-weight: 700; color: #475569; text-transform: uppercase;">Leg {leg_index + 1}</span><span style="background: #c4b5fd; color: #4c1d95; font-size: 0.55rem; font-weight: 600; padding: 0.15rem 0.35rem; border-radius: 3px;">EXOTIC</span><span style="background: #e9d5ff; color: #6b21a8; font-size: 0.55rem; font-weight: 500; padding: 0.15rem 0.35rem; border-radius: 3px;">{display_name}</span></div><span style="background: {position_badge_bg}; color: {position_badge_color}; font-size: 0.65rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; text-transform: uppercase;">{position_type}</span></div></div>"""

    if allow_remove:
        header_col1, header_col2 = st.columns([4, 1])
        with header_col1:
            st.markdown(leg_header_html, unsafe_allow_html=True)
        with header_col2:
            st.markdown("<div style='height: 0.25rem'></div>", unsafe_allow_html=True)
            if st.button(
                "🗑️",
                key=f"remove_leg_{leg_index}_v{version}",
                help="Remove this leg",
                width="stretch",
            ):
                should_remove = True
    else:
        st.markdown(leg_header_html, unsafe_allow_html=True)

    # Type and direction
    col1, col2 = st.columns(2)
    with col1:
        new_option_type = st.selectbox(
            "Type",
            ["call", "put"],
            index=0 if option_type == "call" else 1,
            format_func=lambda x: f"{'📈' if x == 'call' else '📉'} {x.upper()}",
            key=f"leg_{leg_index}_type_v{version}",
        )
    with col2:
        new_position_type = st.selectbox(
            "Direction",
            ["long", "short"],
            index=0 if position_type == "long" else 1,
            format_func=lambda x: f"{'🟢' if x == 'long' else '🔴'} {x.upper()}",
            key=f"leg_{leg_index}_dir_v{version}",
        )

    # Strike and quantity
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
            key=f"leg_{leg_index}_strike_v{version}",
        )
    with col4:
        default_qty = leg_state.get("quantity", leg_config.get("quantity", 1))
        new_quantity = st.number_input(
            "Contracts",
            value=int(default_qty),
            min_value=1,
            step=1,
            key=f"leg_{leg_index}_qty_v{version}",
        )

    # Exotic-specific parameters
    new_barrier = leg_state.get("barrier", 0.0)
    new_is_up = leg_state.get("is_up", True)
    new_is_knock_in = leg_state.get("is_knock_in", False)
    new_rebate = leg_state.get("rebate", 0.0)
    new_payout = leg_state.get("payout", 1.0)
    new_cap = leg_state.get("cap", round(0.5 * new_strike, 2))
    new_lower = leg_state.get("lower_strike", round(0.9 * new_strike, 2))
    new_upper = leg_state.get("upper_strike", round(1.1 * new_strike, 2))
    new_dbl_lower = leg_state.get("dbl_lower", round(0.8 * new_strike, 2))
    new_dbl_upper = leg_state.get("dbl_upper", round(1.2 * new_strike, 2))
    new_adv_barrier = leg_state.get("adv_barrier", round(1.1 * new_strike, 2))
    new_adv_is_up = leg_state.get("adv_is_up", True)
    new_adv_in = leg_state.get("adv_in", False)
    new_mon_pts = leg_state.get("monitoring_points", 252)
    new_t1_pct = leg_state.get("t1_pct", 0.5)
    new_partial_type = leg_state.get("partial_type", "out_B1")
    new_cash = leg_state.get("cash", 10.0)
    new_binary_type = leg_state.get("binary_type", 13)
    new_avg_elapsed = leg_state.get("avg_elapsed_pct", 0.0)
    new_avg_realized = leg_state.get("avg_realized", round(spot_price, 2))

    if inst_class == "barrier":
        col5, col6 = st.columns(2)
        with col5:
            new_barrier = st.number_input(
                "Barrier ($)",
                value=float(
                    leg_state.get(
                        "barrier", round(spot_price * DEFAULT_BARRIER_UP_FACTOR, 2)
                    )
                ),
                step=1.0,
                format="%.2f",
                key=f"leg_{leg_index}_barrier_v{version}",
            )
        with col6:
            barrier_dir_options = ["Up", "Down"]
            barrier_dir_idx = 0 if leg_state.get("is_up", True) else 1
            barrier_dir = st.selectbox(
                "Direction",
                barrier_dir_options,
                index=barrier_dir_idx,
                key=f"leg_{leg_index}_barrier_dir_v{version}",
            )
            new_is_up = barrier_dir == "Up"

        # Barrier options are Knock-Out only in this UI
        new_is_knock_in = False
        col7, col8 = st.columns(2)
        with col7:
            st.text("Knock-Out")
        with col8:
            new_rebate = st.number_input(
                "Rebate ($)",
                value=float(leg_state.get("rebate", 0.0)),
                step=0.1,
                format="%.2f",
                key=f"leg_{leg_index}_rebate_v{version}",
            )

        # Warn if barrier is on the wrong side of spot (immediate knock)
        if new_is_up and new_barrier <= spot_price:
            st.caption(
                f"Warning: Up-barrier ({new_barrier:.2f}) is at or below spot ({spot_price:.2f}) — option is already knocked."
            )
        elif not new_is_up and new_barrier >= spot_price:
            st.caption(
                f"Warning: Down-barrier ({new_barrier:.2f}) is at or above spot ({spot_price:.2f}) — option is already knocked."
            )

    elif inst_class == "digital":
        new_payout = st.number_input(
            "Payout ($)",
            value=float(leg_state.get("payout", DEFAULT_DIGITAL_PAYOUT)),
            step=0.1,
            format="%.2f",
            min_value=0.01,
            key=f"leg_{leg_index}_payout_v{version}",
        )

    elif inst_class == "chooser":
        new_choice_pct = st.slider(
            "Choice time (% of maturity)",
            min_value=0.1,
            max_value=0.9,
            value=float(leg_state.get("choice_time_pct", 0.5)),
            step=0.1,
            key=f"leg_{leg_index}_choice_pct_v{version}",
            help="Time at which the holder chooses call or put, as fraction of maturity.",
        )

    elif inst_class == "power":
        new_power_n = st.number_input(
            "Power exponent (n)",
            value=float(leg_state.get("power_n", 2.0)),
            min_value=1.0,
            max_value=5.0,
            step=0.5,
            format="%.1f",
            key=f"leg_{leg_index}_power_n_v{version}",
            help="Payoff = max(S^n - K, 0). n=1 is vanilla, n=2 is quadratic.",
        )

    elif inst_class == "gap":
        new_gap_trigger = st.number_input(
            "Trigger strike K2 ($)",
            value=float(leg_state.get("gap_trigger", round(spot_price * 1.05, 2))),
            step=1.0,
            format="%.2f",
            key=f"leg_{leg_index}_gap_trigger_v{version}",
            help="Call pays (S-K1) if S>K2. When K1=K2, equals vanilla.",
        )

    elif inst_class == "powered":
        new_powered_i = st.number_input(
            "Power exponent (i, integer)",
            value=int(leg_state.get("power_n", 2)),
            min_value=1,
            max_value=5,
            step=1,
            key=f"leg_{leg_index}_powered_i_v{version}",
            help="Payoff = max(S - K, 0)^i. Integer power (Esser/Haug 4.4.4).",
        )

    elif inst_class == "capped_power":
        col_cp1, col_cp2 = st.columns(2)
        with col_cp1:
            new_capped_i = st.number_input(
                "Power (i)",
                value=float(leg_state.get("power_n", 2.0)),
                min_value=1.0,
                max_value=5.0,
                step=0.5,
                format="%.1f",
                key=f"leg_{leg_index}_capped_i_v{version}",
                help="Payoff = min(max(S^i - K, 0), C).",
            )
        with col_cp2:
            new_cap = st.number_input(
                "Cap (C, $)",
                value=float(leg_state.get("cap", round(0.5 * new_strike, 2))),
                min_value=0.01,
                step=1.0,
                format="%.2f",
                key=f"leg_{leg_index}_cap_v{version}",
                help="Maximum payoff. For a put, must be below the strike.",
            )

    elif inst_class == "supershare":
        col_ss1, col_ss2 = st.columns(2)
        with col_ss1:
            new_lower = st.number_input(
                "Lower X_L ($)",
                value=float(leg_state.get("lower_strike", round(0.9 * new_strike, 2))),
                step=1.0,
                format="%.2f",
                key=f"leg_{leg_index}_lower_v{version}",
                help="Pays S/X_L if X_L < S < X_H at expiry, else 0.",
            )
        with col_ss2:
            new_upper = st.number_input(
                "Upper X_H ($)",
                value=float(leg_state.get("upper_strike", round(1.1 * new_strike, 2))),
                step=1.0,
                format="%.2f",
                key=f"leg_{leg_index}_upper_v{version}",
                help="Upper band boundary (must exceed X_L).",
            )

    elif inst_class in ("log_contract", "log_option"):
        st.caption("Log payoff on ln(S_T / K) — direction (call/put) is ignored.")

    elif inst_class == "double_barrier":
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            new_dbl_lower = st.number_input(
                "Lower barrier L ($)",
                value=float(leg_state.get("dbl_lower", round(0.8 * new_strike, 2))),
                step=1.0,
                format="%.2f",
                key=f"leg_{leg_index}_dbl_lower_v{version}",
            )
        with col_b2:
            _hi_default = 1.2
            new_dbl_upper = st.number_input(
                "Upper barrier U ($)",
                value=float(
                    leg_state.get("dbl_upper", round(_hi_default * new_strike, 2))
                ),
                step=1.0,
                format="%.2f",
                key=f"leg_{leg_index}_dbl_upper_v{version}",
            )
        new_adv_in = st.checkbox(
            "Knock-in",
            value=bool(leg_state.get("adv_in", False)),
            key=f"leg_{leg_index}_adv_in_v{version}",
            help="Knock-in (activates on touch) vs knock-out (default).",
        )

    elif inst_class == "discrete_barrier":
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            new_adv_barrier = st.number_input(
                "Barrier H ($)",
                value=float(leg_state.get("adv_barrier", round(1.1 * new_strike, 2))),
                step=1.0,
                format="%.2f",
                key=f"leg_{leg_index}_adv_barrier_v{version}",
            )
        with col_d2:
            new_mon_pts = st.number_input(
                "Monitoring points",
                value=int(leg_state.get("monitoring_points", 252)),
                min_value=1,
                step=1,
                key=f"leg_{leg_index}_mon_pts_v{version}",
                help="Equally-spaced monitoring dates (BGK continuity correction).",
            )
        col_d3, col_d4 = st.columns(2)
        with col_d3:
            _dir = st.selectbox(
                "Direction",
                ["Up", "Down"],
                index=0 if leg_state.get("adv_is_up", True) else 1,
                key=f"leg_{leg_index}_adv_dir_v{version}",
            )
            new_adv_is_up = _dir == "Up"
        with col_d4:
            new_adv_in = st.checkbox(
                "Knock-in",
                value=bool(leg_state.get("adv_in", False)),
                key=f"leg_{leg_index}_adv_in2_v{version}",
            )

    elif inst_class == "partial_barrier":
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            new_adv_barrier = st.number_input(
                "Barrier H ($)",
                value=float(leg_state.get("adv_barrier", round(1.1 * new_strike, 2))),
                step=1.0,
                format="%.2f",
                key=f"leg_{leg_index}_pb_barrier_v{version}",
            )
        with col_p2:
            new_t1_pct = st.slider(
                "Window boundary t1 (% of maturity)",
                min_value=0.1,
                max_value=1.0,
                value=float(leg_state.get("t1_pct", 0.5)),
                step=0.1,
                key=f"leg_{leg_index}_t1_pct_v{version}",
            )
        _ptypes = list(PARTIAL_BARRIER_TYPES.keys())
        _pcur = leg_state.get("partial_type", "out_B1")
        new_partial_type = st.selectbox(
            "Monitoring type",
            _ptypes,
            index=_ptypes.index(_pcur) if _pcur in _ptypes else 2,
            format_func=lambda k: PARTIAL_BARRIER_TYPES[k],
            key=f"leg_{leg_index}_ptype_v{version}",
        )

    elif inst_class == "binary_barrier":
        col_bb1, col_bb2 = st.columns(2)
        with col_bb1:
            new_adv_barrier = st.number_input(
                "Barrier H ($)",
                value=float(leg_state.get("adv_barrier", round(1.1 * new_strike, 2))),
                step=1.0,
                format="%.2f",
                key=f"leg_{leg_index}_bb_barrier_v{version}",
            )
        with col_bb2:
            new_cash = st.number_input(
                "Cash payout K ($)",
                value=float(leg_state.get("cash", 10.0)),
                min_value=0.0,
                step=1.0,
                format="%.2f",
                key=f"leg_{leg_index}_cash_v{version}",
                help="Cash-or-nothing payout (ignored for asset-or-nothing types).",
            )
        _btypes = list(BINARY_BARRIER_TYPES.keys())
        _bcur = int(leg_state.get("binary_type", 13))
        new_binary_type = st.selectbox(
            "Binary type (Reiner-Rubinstein, 1..28)",
            _btypes,
            index=_btypes.index(_bcur) if _bcur in _btypes else 12,
            format_func=lambda k: BINARY_BARRIER_TYPES[k],
            key=f"leg_{leg_index}_btype_v{version}",
        )

    elif inst_class == "arithmetic_asian":
        col_aa1, col_aa2 = st.columns(2)
        with col_aa1:
            new_avg_elapsed = st.slider(
                "Averaging elapsed (w)",
                min_value=0.0,
                max_value=0.95,
                value=float(leg_state.get("avg_elapsed_pct", 0.0)),
                step=0.05,
                key=f"leg_{leg_index}_avg_elapsed_v{version}",
                help="Fraction of the averaging window already elapsed. "
                "0 = fresh option; the ITM payoff slope becomes (1-w).",
            )
        with col_aa2:
            new_avg_realized = st.number_input(
                "Realized average SA ($)",
                value=float(leg_state.get("avg_realized", round(spot_price, 2))),
                min_value=0.01,
                step=1.0,
                format="%.2f",
                key=f"leg_{leg_index}_avg_realized_v{version}",
                help="Arithmetic average realized so far. Ignored while w = 0.",
            )

    # Compute extra1 for new exotic types
    new_extra1 = 0.0
    new_choice_time_pct = leg_state.get("choice_time_pct", 0.5)
    new_power_n_val = leg_state.get("power_n", 2.0)
    new_gap_trigger_val = leg_state.get("gap_trigger", new_strike * 1.05)

    if inst_class == "chooser":
        new_choice_time_pct = new_choice_pct
        maturity = DEFAULT_DTE / 365.0
        new_extra1 = new_choice_time_pct * maturity
    elif inst_class == "power":
        new_power_n_val = new_power_n
        new_extra1 = new_power_n_val
    elif inst_class == "gap":
        new_gap_trigger_val = new_gap_trigger
        new_extra1 = new_gap_trigger_val
    elif inst_class == "powered":
        new_power_n_val = float(new_powered_i)
        new_extra1 = new_power_n_val
    elif inst_class == "capped_power":
        new_power_n_val = new_capped_i
        new_extra1 = new_power_n_val

    # Family-specific named params for the registry-priced analytic exotics.
    new_params: dict = {}
    if inst_class == "supershare":
        new_params = {"lower_strike": new_lower, "upper_strike": new_upper}
    elif inst_class == "double_barrier":
        new_params = {
            "lower": new_dbl_lower,
            "upper": new_dbl_upper,
            "is_knock_in": new_adv_in,
        }
    elif inst_class == "discrete_barrier":
        new_params = {
            "barrier": new_adv_barrier,
            "is_up": new_adv_is_up,
            "is_knock_in": new_adv_in,
            "monitoring_points": int(new_mon_pts),
        }
    elif inst_class == "partial_barrier":
        new_params = {
            "barrier": new_adv_barrier,
            "t1": new_t1_pct * (DEFAULT_DTE / 365.0),
            "barrier_type": new_partial_type,
        }
    elif inst_class == "binary_barrier":
        new_params = {
            "barrier": new_adv_barrier,
            "cash": new_cash,
            "binary_type": int(new_binary_type),
        }
    elif inst_class == "arithmetic_asian":
        _w = min(max(float(new_avg_elapsed), 0.0), 0.95)
        new_params = {
            "average_period": (DEFAULT_DTE / 365.0) / (1.0 - _w),
            "realized_average": float(new_avg_realized),
        }

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
        "power_n": new_power_n_val,
        "gap_trigger": new_gap_trigger_val,
        "cap": new_cap,
        "lower_strike": new_lower,
        "upper_strike": new_upper,
        "dbl_lower": new_dbl_lower,
        "dbl_upper": new_dbl_upper,
        "adv_barrier": new_adv_barrier,
        "adv_is_up": new_adv_is_up,
        "adv_in": new_adv_in,
        "monitoring_points": int(new_mon_pts),
        "t1_pct": new_t1_pct,
        "partial_type": new_partial_type,
        "cash": new_cash,
        "binary_type": int(new_binary_type),
        "avg_elapsed_pct": float(new_avg_elapsed),
        "avg_realized": float(new_avg_realized),
    }
    st.session_state.strategy_legs_state[leg_index] = new_state

    # Calculate premium using exotic pricing
    premium = calculate_exotic_premium(
        spot=spot_price,
        strike=new_strike,
        dte_days=DEFAULT_DTE,
        risk_free_rate=risk_free_rate,
        volatility=DEFAULT_IV / 100,
        option_type=new_option_type,
        exotic_type=inst_class,
        barrier=new_barrier,
        is_up=new_is_up,
        is_knock_in=new_is_knock_in,
        rebate=new_rebate,
        payout=new_payout,
        extra1=new_extra1,
        dividend_yield=dividend_yield,
        cap=new_cap,
        params=new_params,
    )
    total_cost = premium * new_quantity * CONTRACT_MULTIPLIER

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


def render_stock_leg_editor(
    spot_price: float,
    is_custom: bool,
    selected_strategy: str,
    is_removable: bool = False,
) -> tuple[float, bool]:
    """Render stock position editor. Returns (stock_cost, should_remove)."""
    stock_state = st.session_state.strategy_legs_state.get(
        "stock", {"position_type": "long", "quantity": 100, "entry_price": spot_price}
    )

    # Get strategy version for unique widget keys
    version = st.session_state.get("strategy_version", 0)
    should_remove = False

    # Header with optional remove button
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
                "🗑️",
                key=f"remove_stock_v{version}",
                help="Remove stock position",
                width="stretch",
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
            key=f"stock_leg_dir_v{version}",
        )

    with col2:
        stock_qty = st.number_input(
            "Shares",
            value=int(stock_state.get("quantity", 100)),
            min_value=1,
            step=100,
            key=f"stock_leg_qty_v{version}",
        )

    stock_entry = st.number_input(
        "Entry Price ($)",
        value=float(stock_state.get("entry_price", spot_price)),
        step=1.0,
        format="%.2f",
        key=f"stock_leg_entry_v{version}",
    )

    st.session_state.strategy_legs_state["stock"] = {
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
