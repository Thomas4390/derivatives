"""Active positions display section of the sidebar."""

import streamlit as st
from components.position_card import calculate_net_position
from config.constants import CONTRACT_MULTIPLIER, EXOTIC_TYPE_NAMES
from config.templates import (
    exotic_position_item_html,
    net_position_card_html,
    position_item_html,
    stock_position_html,
)


def render_positions_section(positions: list, stock_position) -> None:
    """Render the current positions section."""
    st.markdown(
        "<div style='height: 0.5rem; border-top: 1px solid #e2e8f0; margin-top: 1rem; padding-top: 1rem;'></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
        <span style="font-size: 1rem;">📋</span>
        <span style="font-size: 0.75rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em;">Active Positions</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if not positions and not stock_position:
        st.markdown(
            """
        <div style="text-align: center; padding: 1.5rem 1rem; color: #94a3b8; font-size: 0.85rem; background: #f8fafc; border-radius: 8px; border: 1px dashed #cbd5e1;">
            <div style="font-size: 1.25rem; margin-bottom: 0.5rem; opacity: 0.6;">∅</div>
            <div style="font-weight: 500; color: #64748b;">No active positions</div>
            <div style="font-size: 0.75rem; margin-top: 0.25rem;">Apply a strategy above to begin</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

    # Net position
    net_amount = calculate_net_position(positions, stock_position)
    st.markdown(net_position_card_html("", net_amount), unsafe_allow_html=True)

    # Stock position (now a dict)
    if stock_position:
        stock_cost = stock_position["entry_price"] * stock_position["quantity"]
        st.markdown(
            stock_position_html(
                quantity=stock_position["quantity"],
                position_type=stock_position["position_type"],
                entry_price=stock_position["entry_price"],
                stock_cost=stock_cost,
            ),
            unsafe_allow_html=True,
        )

    # Option positions (now dicts)
    for i, pos in enumerate(positions):
        total_amount = pos["premium_paid"] * pos["quantity"] * CONTRACT_MULTIPLIER
        shares_controlled = pos["quantity"] * CONTRACT_MULTIPLIER
        inst_class = pos.get("instrument_class", "vanilla")

        if inst_class != "vanilla":
            # Exotic position display (harmonized with vanilla card)
            display_name = EXOTIC_TYPE_NAMES.get(inst_class, inst_class)
            st.markdown(
                exotic_position_item_html(
                    index=i + 1,
                    quantity=pos["quantity"],
                    position_type=pos["position_type"],
                    option_type=pos["option_type"],
                    strike=pos["strike"],
                    premium=pos["premium_paid"],
                    total_amount=total_amount,
                    shares_controlled=shares_controlled,
                    is_long=(pos["position_type"] == "long"),
                    exotic_type_name=display_name,
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                position_item_html(
                    index=i + 1,
                    quantity=pos["quantity"],
                    position_type=pos["position_type"],
                    option_type=pos["option_type"],
                    strike=pos["strike"],
                    premium=pos["premium_paid"],
                    total_amount=total_amount,
                    shares_controlled=shares_controlled,
                    is_long=(pos["position_type"] == "long"),
                ),
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    if st.button("🗑️  Clear All", width="stretch", key="clear_all_btn"):
        # Clear all positions
        st.session_state.positions = []
        st.session_state.stock_position = None

        # Clear strategy state
        st.session_state.strategy_legs_state = {}
        st.session_state.additional_legs = {}
        st.session_state.additional_stock = {}
        st.session_state.custom_legs = []
        st.session_state.custom_has_stock = False

        # Reset strategy selection by incrementing selector version (creates new widget)
        st.session_state.selector_version = (
            st.session_state.get("selector_version", 0) + 1
        )
        if "last_selected_strategy" in st.session_state:
            del st.session_state.last_selected_strategy

        # Increment version to force widget reset
        st.session_state.strategy_version = (
            st.session_state.get("strategy_version", 0) + 1
        )

        # Clear widget keys
        keys_to_remove = [
            key
            for key in list(st.session_state.keys())
            if key.startswith("leg_")
            or key.startswith("stock_leg_")
            or key.startswith("strategy_selector_v")
        ]
        for key in keys_to_remove:
            del st.session_state[key]

        st.rerun()


def render_strategy_summary(total_net_cost: float, has_stock: bool) -> None:
    """Render the strategy cost summary."""
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
