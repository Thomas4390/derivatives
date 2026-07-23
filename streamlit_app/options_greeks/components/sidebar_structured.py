"""Structured products builder section of the sidebar."""

import importlib.util
import sys
from pathlib import Path

import streamlit as st
from config.constants import (
    DEFAULT_MC_PATHS,
    SP_GBM_VOL_SPEC,
    SP_HESTON_PARAM_SPECS,
    STRUCTURED_MODEL_TYPES,
    STRUCTURED_PRODUCT_DEFAULTS,
    STRUCTURED_PRODUCT_TYPES,
)

# Import render_structured_product_builder from simulation without triggering
# simulation's __init__.py (which has its own dependency chain).
_sim_sp_path = str(
    Path(__file__).resolve().parent.parent.parent
    / "simulation"
    / "components"
    / "structured_product_builder.py"
)
# Ensure simulation's own imports resolve (its services/, utils/, etc.)
_sim_dir = str(Path(__file__).resolve().parent.parent.parent / "simulation")
if _sim_dir not in sys.path:
    sys.path.insert(0, _sim_dir)
_sp_spec = importlib.util.spec_from_file_location("sim_sp_builder", _sim_sp_path)
_sp_mod = importlib.util.module_from_spec(_sp_spec)
_sp_spec.loader.exec_module(_sp_mod)
render_structured_product_builder = _sp_mod.render_structured_product_builder

# Structured product strategy keys (prefix "sp_")
SP_KEYS = {f"sp_{k}" for k in STRUCTURED_PRODUCT_TYPES}


def render_structured_product_section(
    selected_strategy: str,
    spot_price: float,
    risk_free_rate: float,
) -> None:
    """Render the structured product parameter editor in the sidebar."""
    import json

    from services.structured_pricing_adapter import (
        compute_greeks,
        price_structured_product,
    )

    # Extract product type from strategy key (e.g. "sp_cpn" -> "cpn")
    product_type = selected_strategy[3:]
    defaults = STRUCTURED_PRODUCT_DEFAULTS[product_type]

    # ── Product params: use shared component from simulation ──
    sp_result = render_structured_product_builder(
        spot_price=spot_price,
        risk_free_rate=risk_free_rate,
        time_to_expiry=1.0,
        volatility=0.20,
        product_type_key=product_type,
    )
    product_params = sp_result["product_params"]

    # --- Model ---
    st.markdown('<div class="sidebar-header">Model</div>', unsafe_allow_html=True)
    model_type = st.selectbox(
        "Model",
        list(STRUCTURED_MODEL_TYPES.keys()),
        format_func=lambda k: STRUCTURED_MODEL_TYPES[k],
        key="sp_model_type",
        label_visibility="collapsed",
    )
    model_params = _render_sp_model_params(model_type)

    # --- Market Parameters ---
    st.markdown('<div class="sidebar-header">Market</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        div_yield = st.number_input(
            "Dividend Yield (y)",
            value=0.0,
            min_value=0.0,
            max_value=0.15,
            step=0.005,
            format="%.4f",
            key="sp_div",
        )
    with col2:
        n_paths = st.number_input(
            "MC Paths",
            value=DEFAULT_MC_PATHS,
            min_value=5_000,
            max_value=500_000,
            step=5_000,
            key="sp_paths",
        )
    seed = st.number_input(
        "Random Seed (0 = random)",
        value=42,
        min_value=0,
        max_value=999_999,
        step=1,
        key="sp_seed",
    )

    # --- Auto-pricing on strategy/spot/rate change ---
    sp_strategy_changed = (
        st.session_state.get("_sp_last_priced_strategy") != selected_strategy
    )
    sp_spot_changed = st.session_state.get("_sp_last_priced_spot") != spot_price
    sp_rate_changed = st.session_state.get("_sp_last_priced_rate") != risk_free_rate

    if sp_strategy_changed or sp_spot_changed or sp_rate_changed:
        default_model_type = "gbm"
        default_model_params = {"sigma": 0.20}
        product_params_json = json.dumps(defaults, sort_keys=True)
        model_params_json = json.dumps(default_model_params, sort_keys=True)
        seed_val = 42

        result = price_structured_product(
            product_type,
            product_params_json,
            default_model_type,
            model_params_json,
            spot_price,
            risk_free_rate,
            0.0,
            DEFAULT_MC_PATHS,
            seed_val,
        )
        greeks = compute_greeks(
            product_type,
            product_params_json,
            default_model_type,
            model_params_json,
            spot_price,
            risk_free_rate,
            0.0,
            DEFAULT_MC_PATHS,
            seed_val,
        )

        st.session_state.sp_result = result
        st.session_state.sp_greeks = greeks
        st.session_state.sp_product_type = product_type
        st.session_state.sp_config = {
            "product_type": product_type,
            "product_params_json": product_params_json,
            "model_type": default_model_type,
            "model_params_json": model_params_json,
            "spot": spot_price,
            "rate": risk_free_rate,
            "dividend_yield": 0.0,
            "n_paths": DEFAULT_MC_PATHS,
            "seed": seed_val,
        }
        st.session_state._sp_last_priced_strategy = selected_strategy
        st.session_state._sp_last_priced_spot = spot_price
        st.session_state._sp_last_priced_rate = risk_free_rate

    # --- Apply Changes Button ---
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    if st.button(
        "\u2713  Apply Changes", width="stretch", type="primary", key="sp_price_btn"
    ):
        seed_val = seed if seed > 0 else None
        product_params_json = json.dumps(product_params, sort_keys=True)
        model_params_json = json.dumps(model_params, sort_keys=True)

        with st.spinner("Pricing..."):
            result = price_structured_product(
                product_type=product_type,
                product_params_json=product_params_json,
                model_type=model_type,
                model_params_json=model_params_json,
                spot=spot_price,
                rate=risk_free_rate,
                dividend_yield=div_yield,
                n_paths=n_paths,
                seed=seed_val,
            )
            greeks = compute_greeks(
                product_type=product_type,
                product_params_json=product_params_json,
                model_type=model_type,
                model_params_json=model_params_json,
                spot=spot_price,
                rate=risk_free_rate,
                dividend_yield=div_yield,
                n_paths=n_paths,
                seed=seed_val,
            )

        st.session_state.sp_result = result
        st.session_state.sp_greeks = greeks
        st.session_state.sp_product_type = product_type
        st.session_state.sp_config = {
            "product_type": product_type,
            "product_params_json": product_params_json,
            "model_type": model_type,
            "model_params_json": model_params_json,
            "spot": spot_price,
            "rate": risk_free_rate,
            "dividend_yield": div_yield,
            "n_paths": n_paths,
            "seed": seed_val,
        }
        st.session_state.sp_scenario = None
        st.rerun()

    # Show quick summary if results exist
    result = st.session_state.get("sp_result")
    if result is not None:
        fv = result["fair_value"]
        price = result["price"]
        st.markdown(
            f"""
        <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border: 1px solid #86efac; border-radius: 10px; padding: 1rem; margin-top: 0.75rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 600;">Fair Value</div>
                    <div style="font-size: 0.7rem; color: #94a3b8;">of notional</div>
                </div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #059669; font-family: 'JetBrains Mono', monospace;">
                    {fv:.2f}%
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid #bbf7d0;">
                <div style="font-size: 0.75rem; color: #64748b;">Price</div>
                <div style="font-size: 0.9rem; font-weight: 600; color: #1e293b; font-family: 'JetBrains Mono', monospace;">${price:,.2f}</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def _render_sp_model_params(model_type: str) -> dict:
    """Render model-specific parameter inputs for structured products.

    Heston inputs are driven by ``SP_HESTON_PARAM_SPECS`` (bounds/defaults live in
    config) — same widgets, keys, order and 2-column layout as before.
    """
    if model_type == "gbm":
        s = SP_GBM_VOL_SPEC
        sigma = st.slider(
            s["label"], s["min"], s["max"], s["default"], s["step"],
            format=s["format"], key="sp_sigma",
        )
        return {"sigma": sigma}

    if model_type == "heston":
        cols = st.columns(2)
        params: dict = {}
        for spec in SP_HESTON_PARAM_SPECS:
            with cols[spec["col"]]:
                params[spec["name"]] = st.number_input(
                    spec["label"],
                    value=spec["default"],
                    min_value=spec["min"],
                    max_value=spec["max"],
                    step=spec["step"],
                    format=spec["format"],
                    key=spec["key"],
                )
        return params

    return {}
