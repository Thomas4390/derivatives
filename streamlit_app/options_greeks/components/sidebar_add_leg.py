"""Add leg button section of the sidebar."""

import streamlit as st
from config.constants import (
    DEFAULT_BARRIER_UP_FACTOR,
    DEFAULT_DIGITAL_PAYOUT,
    INSTRUMENT_CLASSES,
    STRATEGY_LEGS,
)


def render_add_leg_button(
    spot_price: float,
    is_custom: bool,
    selected_strategy: str,
    strategy_legs: list,
    has_stock: bool,
) -> None:
    """Render the Add Leg buttons for all strategies."""
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    # Initialize additional legs tracking
    if "additional_legs" not in st.session_state:
        st.session_state.additional_legs = {}

    # Get additional legs for current strategy
    additional_legs_key = selected_strategy if not is_custom else "custom"
    if additional_legs_key not in st.session_state.additional_legs:
        st.session_state.additional_legs[additional_legs_key] = []

    # Three buttons: Add Option, Add Exotic, Add Stock
    col1, col2, col3 = st.columns(3)
    with col1:
        add_option_clicked = st.button(
            "➕ Vanilla", width="stretch", key="add_option_leg_btn", type="secondary"
        )
    with col2:
        add_exotic_clicked = st.button(
            "➕ Exotic", width="stretch", key="add_exotic_leg_btn", type="secondary"
        )
    with col3:
        add_stock_clicked = st.button(
            "➕ Stock",
            width="stretch",
            key="add_stock_leg_btn",
            type="secondary",
            disabled=has_stock,
        )

    # Toggle exotic selector visibility on button click
    if add_exotic_clicked:
        st.session_state.show_exotic_selector = not st.session_state.get(
            "show_exotic_selector", False
        )

    # Show exotic type selector only when toggled on
    if st.session_state.get("show_exotic_selector", False):
        exotic_types = list(INSTRUMENT_CLASSES.keys())
        pending_idx = exotic_types.index(
            st.session_state.get("pending_exotic_type", "barrier")
        )
        selected_exotic = st.selectbox(
            "Exotic type",
            exotic_types,
            index=pending_idx,
            format_func=lambda x: INSTRUMENT_CLASSES[x],
            key="sidebar_exotic_type_selector",
            label_visibility="collapsed",
        )
        st.session_state.pending_exotic_type = selected_exotic

        confirm_exotic = st.button(
            "✓ Add Exotic Leg",
            width="stretch",
            key="confirm_exotic_btn",
            type="primary",
        )
        if confirm_exotic:
            exotic_type = st.session_state.get("pending_exotic_type", "barrier")
            new_leg = {
                "option_type": "call",
                "position_type": "long",
                "strike_factor": 1.0,
                "quantity": 1,
                "instrument_class": exotic_type,
            }
            if exotic_type == "barrier":
                new_leg["barrier_factor"] = DEFAULT_BARRIER_UP_FACTOR
                new_leg["is_up"] = True
                new_leg["is_knock_in"] = False

            if is_custom:
                st.session_state.custom_legs.append(new_leg)
                new_index = len(st.session_state.custom_legs) - 1
            else:
                st.session_state.additional_legs[additional_legs_key].append(new_leg)
                new_index = (
                    len(STRATEGY_LEGS.get(selected_strategy, []))
                    + len(st.session_state.additional_legs[additional_legs_key])
                    - 1
                )

            st.session_state.strategy_legs_state[new_index] = {
                "option_type": "call",
                "position_type": "long",
                "strike": round(spot_price, 2),
                "quantity": 1,
                "instrument_class": exotic_type,
                "barrier": round(spot_price * DEFAULT_BARRIER_UP_FACTOR, 2)
                if exotic_type == "barrier"
                else 0.0,
                "is_up": True,
                "is_knock_in": False,
                "rebate": 0.0,
                "payout": DEFAULT_DIGITAL_PAYOUT if exotic_type == "digital" else 1.0,
            }
            st.session_state.show_exotic_selector = False
            st.session_state.strategy_version = (
                st.session_state.get("strategy_version", 0) + 1
            )
            st.rerun()

    if add_option_clicked:
        # Add a new default option leg
        new_leg = {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
        }

        if is_custom:
            st.session_state.custom_legs.append(new_leg)
            new_index = len(st.session_state.custom_legs) - 1
        else:
            st.session_state.additional_legs[additional_legs_key].append(new_leg)
            # The new index is the base strategy legs + additional legs
            new_index = (
                len(STRATEGY_LEGS.get(selected_strategy, []))
                + len(st.session_state.additional_legs[additional_legs_key])
                - 1
            )

        # Initialize state for the new leg
        st.session_state.strategy_legs_state[new_index] = {
            "option_type": "call",
            "position_type": "long",
            "strike": round(spot_price, 2),
            "quantity": 1,
        }
        st.session_state.strategy_version = (
            st.session_state.get("strategy_version", 0) + 1
        )
        st.rerun()

    if add_stock_clicked:
        # Add stock position
        if is_custom:
            st.session_state.custom_has_stock = True
        else:
            # For predefined strategies, we track it differently
            if "additional_stock" not in st.session_state:
                st.session_state.additional_stock = {}
            st.session_state.additional_stock[additional_legs_key] = True

        # Initialize stock state
        st.session_state.strategy_legs_state["stock"] = {
            "position_type": "long",
            "quantity": 100,
            "entry_price": spot_price,
        }
        st.session_state.strategy_version = (
            st.session_state.get("strategy_version", 0) + 1
        )
        st.rerun()

    # Show hint when no legs (only for custom with no legs and no stock)
    if (
        is_custom
        and len(st.session_state.custom_legs) == 0
        and not st.session_state.get("custom_has_stock", False)
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
