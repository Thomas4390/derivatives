"""Leg editor components for the sidebar (vanilla, exotic, stock)."""

import streamlit as st
from config.constants import (
    CONTRACT_MULTIPLIER,
    DEFAULT_BARRIER_UP_FACTOR,
    DEFAULT_DIGITAL_PAYOUT,
    DEFAULT_DTE,
    DEFAULT_IV,
    INSTRUMENT_CLASSES,
)

from services.pricing_adapter import calculate_option_premium


def render_leg_editor(
    leg_index: int,
    leg_config: dict,
    spot_price: float,
    risk_free_rate: float,
    total_legs: int,
    allow_remove: bool = False,
    is_additional: bool = False,
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
