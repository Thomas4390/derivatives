"""
Model Selector Component - Visual model selection with info cards.

Provides:
- Visual model selection with icons and descriptions
- Feature badges for quick overview
- Condition warnings
"""

from collections.abc import Callable

import streamlit as st
from config.model_registry import (
    MODEL_DISPLAY_ORDER,
    MODEL_REGISTRY,
    get_model,
)
from utils.model_helpers import (
    get_feature_badges,
    get_model_icon,
    get_volatility_type,
)


def render_model_selector(
    key: str = "model_selector",
    on_change: Callable | None = None
) -> str:
    """
    Render model selection interface.

    Args:
        key: Streamlit widget key
        on_change: Callback when selection changes

    Returns:
        Selected model key
    """
    st.subheader("📐 Model Selection")

    # Create options for selectbox
    model_options = {
        model_key: f"{get_model_icon(model_key)} {MODEL_REGISTRY[model_key].name}"
        for model_key in MODEL_DISPLAY_ORDER
    }

    # Append custom model if registered
    custom = st.session_state.get("custom_model")
    if custom and "spec" in custom:
        model_options["custom"] = f"🧪 {custom['spec'].name}"

    # Get current selection from session state
    current_model = st.session_state.get("selected_model", "gbm")
    options_list = list(model_options.keys())

    # Compute index
    if current_model in options_list:
        current_index = options_list.index(current_model)
    else:
        current_index = 0

    # Model selectbox
    selected = st.selectbox(
        "Select Model",
        options=options_list,
        format_func=lambda x: model_options[x],
        index=current_index,
        key=key,
        on_change=on_change,
        help="Choose simulation model"
    )

    # Store in session state
    st.session_state.selected_model = selected

    # Show model info card
    _render_model_info_card(selected)

    return selected


def _render_model_info_card(model_key: str):
    """Render information card for selected model."""
    model = get_model(model_key)

    # Model name and description
    st.markdown(f"**{model.name}**")
    st.caption(model.description)

    # Feature badges
    badges = get_feature_badges(model_key)
    badge_html = " ".join([
        f'<span style="background-color: {_get_badge_color(color)}; '
        f'color: white; padding: 2px 8px; border-radius: 12px; '
        f'font-size: 0.75rem; margin-right: 4px;">{label}</span>'
        for label, color in badges
    ])
    st.markdown(badge_html, unsafe_allow_html=True)

    # Volatility type indicator
    vol_type = get_volatility_type(model_key)
    vol_color = "🟢" if vol_type == "Constant" else "🟡" if "GARCH" in vol_type else "🔵"
    st.markdown(f"**Volatility:** {vol_color} {vol_type}")


def _get_badge_color(color_name: str) -> str:
    """Convert color name to hex."""
    colors = {
        "blue": "#1f77b4",
        "orange": "#ff7f0e",
        "green": "#2ca02c",
        "red": "#d62728",
        "violet": "#9467bd",
        "gray": "#7f7f7f",
    }
    return colors.get(color_name, "#7f7f7f")


def render_model_selector_radio(
    key: str = "model_radio",
) -> str:
    """
    Render model selection as radio buttons (alternative layout).

    Returns:
        Selected model key
    """
    st.subheader("📐 Model")

    # Group models
    continuous_models = ["gbm"]
    stoch_vol_models = ["heston", "bates"]
    jump_models = ["merton"]
    garch_models = ["garch", "ngarch", "gjr_garch"]

    current_model = st.session_state.get("selected_model", "gbm")

    # Create columns for model groups
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Continuous Time**")
        for model_key in continuous_models + stoch_vol_models + jump_models:
            model = get_model(model_key)
            icon = get_model_icon(model_key)
            if st.button(
                f"{icon} {model.short_name}",
                key=f"{key}_{model_key}",
                width="stretch",
                type="primary" if model_key == current_model else "secondary"
            ):
                st.session_state.selected_model = model_key
                st.rerun()

    with col2:
        st.markdown("**GARCH Family**")
        for model_key in garch_models:
            model = get_model(model_key)
            icon = get_model_icon(model_key)
            if st.button(
                f"{icon} {model.short_name}",
                key=f"{key}_{model_key}",
                width="stretch",
                type="primary" if model_key == current_model else "secondary"
            ):
                st.session_state.selected_model = model_key
                st.rerun()

    return st.session_state.get("selected_model", "gbm")


def render_model_comparison_table():
    """Render a comparison table of all models."""
    import pandas as pd

    data = []
    for model_key in MODEL_DISPLAY_ORDER:
        model = get_model(model_key)
        data.append({
            "Model": model.short_name,
            "Volatility": get_volatility_type(model_key),
            "Jumps": "✓" if model.has_jumps else "—",
            "BS": "✓" if "analytical" in [m.value for m in model.pricing_methods] else "—",
            "FFT": "✓" if "fft" in [m.value for m in model.pricing_methods] else "—",
            "MC": "✓",
        })

    df = pd.DataFrame(data)
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "Model": st.column_config.TextColumn("Model", width="medium"),
            "Volatility": st.column_config.TextColumn("Volatility", width="medium"),
            "Jumps": st.column_config.TextColumn("Jumps", width="small"),
            "BS": st.column_config.TextColumn("BS", width="small"),
            "FFT": st.column_config.TextColumn("FFT", width="small"),
            "MC": st.column_config.TextColumn("MC", width="small"),
        }
    )
