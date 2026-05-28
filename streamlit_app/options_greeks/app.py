"""
Options Greeks Explorer - Main Application

A high-performance educational tool for options analysis using the Black-Scholes model.
Designed for academic use in quantitative finance courses.

Author: Thomas Vaudescal
"""

import sys
from pathlib import Path

# Add paths for imports
app_dir = Path(__file__).parent
project_root = app_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(app_dir))

import json

import numpy as np
import streamlit as st
from charts.greeks_chart import (
    render_first_order_greeks,
    render_second_order_greeks,
    render_third_order_greeks,
)
from charts.pnl_chart import render_pnl_tab
from charts.surface_3d import render_3d_tab
from components.sidebar import render_sidebar
from config.constants import (
    DEFAULT_DTE,
    DEFAULT_IV,
    DTE_RANGE,
    IV_RANGE,
    SP_SPOT_RANGE_FACTOR,
    SP_SPOT_RANGE_POINTS,
    SPOT_RANGE_FACTOR,
    SPOT_RANGE_POINTS,
)

# Local imports
from config.styles import inject_styles
from config.templates import footer_html, render_header
from guide_formula import render_guide_section
from services.portfolio_calculator import calculate_all_surfaces, prepare_portfolio_data

# Pricing functions (using backend architecture, no classes)
from services.pricing_adapter import (
    calculate_all_greeks,
    calculate_greeks_3d_strike,
    calculate_option_premium,
    calculate_portfolio_greeks_3d_dte,
    calculate_portfolio_greeks_3d_iv,
    find_breakeven_points,
)
from services.pricing_adapter import (
    calculate_pnl_at_expiry_arrays as calculate_portfolio_pnl_at_expiry,
)
from services.state_manager import init_session_state
from services.structured_pricing_adapter import (
    calculate_structured_greeks_3d_dte,
    calculate_structured_greeks_3d_iv,
    calculate_structured_surfaces,
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Options Greeks Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS styles
inject_styles()

# Initialize session state
init_session_state()


# =============================================================================
# SIDEBAR
# =============================================================================

spot_price, risk_free_rate = render_sidebar(
    positions=st.session_state.positions, stock_position=st.session_state.stock_position
)


# =============================================================================
# HEADER
# =============================================================================

render_header(
    title="Options Greeks Explorer",
    subtitle="Interactive visualization of option pricing and risk metrics using the Black-Scholes model",
    badge="Educational Tool",
)


# =============================================================================
# DATA PREPARATION
# =============================================================================


def get_default_premium() -> float:
    """Calculate default premium for a long call ATM using Black-Scholes."""
    return calculate_option_premium(
        spot=spot_price,
        strike=spot_price,
        dte_days=DEFAULT_DTE,
        risk_free_rate=risk_free_rate,
        volatility=DEFAULT_IV / 100,  # Convert from percentage
        option_type="call",
    )


def get_portfolio_json() -> str:
    """Prepare portfolio data as JSON string."""
    positions = st.session_state.positions
    stock_position = st.session_state.stock_position

    if positions or stock_position:
        portfolio_data = prepare_portfolio_data(positions, stock_position, spot_price)
    else:
        # Empty portfolio - no default position
        portfolio_data = {"spot_price": spot_price, "options": [], "stock": None}

    return json.dumps(portfolio_data)


# =============================================================================
# MAIN CONTENT — Unified flow for Options and Structured Products
# =============================================================================

sp_mode = st.session_state.get("sp_mode", False)
sp_config = st.session_state.get("sp_config")

if sp_mode and sp_config:
    # ------------------------------------------------------------------
    # STRUCTURED PRODUCTS MODE — MC surfaces + shared tabs
    # ------------------------------------------------------------------

    # Build spot range for structured products (wider to capture barriers)
    spot_range = np.linspace(
        spot_price * (1 - SP_SPOT_RANGE_FACTOR),
        spot_price * (1 + SP_SPOT_RANGE_FACTOR),
        SP_SPOT_RANGE_POINTS,
    )

    # Entry price from sidebar pricing
    sp_result = st.session_state.get("sp_result", {})
    entry_price = sp_result.get("price", 0.0)

    with st.spinner(
        "Calculating structured product surfaces (MC, may take a few minutes)..."
    ):
        all_data = calculate_structured_surfaces(
            product_type=sp_config["product_type"],
            product_params_json=sp_config["product_params_json"],
            model_type=sp_config["model_type"],
            model_params_json=sp_config["model_params_json"],
            spot=sp_config["spot"],
            rate=sp_config["rate"],
            dividend_yield=sp_config.get("dividend_yield", 0.0),
            spot_range_tuple=tuple(spot_range.tolist()),
            entry_price=entry_price,
            seed=sp_config.get("seed"),
        )

    # 3D functions: wrap sp_config into the JSON expected by surface_3d.py
    sp_config_with_entry = dict(sp_config)
    sp_config_with_entry["entry_price"] = entry_price
    sp_config_json_3d = json.dumps(sp_config_with_entry)

    greeks_3d_dte_func = calculate_structured_greeks_3d_dte
    greeks_3d_iv_func = calculate_structured_greeks_3d_iv

    # Portfolio JSON for 3D tab (uses sp_config_json_3d)
    portfolio_json = sp_config_json_3d
    has_positions = True
    has_exotic_legs = False

    # No-op functions for features not applicable to structured products
    def _noop_greeks(*args, **kwargs):
        return np.zeros(14)

    def _noop_pnl(*args, **kwargs):
        return 0.0

    def _noop_breakeven(*args, **kwargs):
        return None

    calculate_all_greeks_func = _noop_greeks
    calculate_pnl_at_expiry_func = _noop_pnl
    find_breakeven_func = _noop_breakeven
    calculate_exotic_greeks_func = None
    calculate_greeks_3d_strike_func = None

    # Positions/stock for tab rendering (empty — no legs to show)
    positions_for_tabs = []
    stock_position_for_tabs = None

elif sp_mode and not sp_config:
    # Defensive: auto-pricing should always populate sp_config, but just in case
    st.stop()

else:
    # ------------------------------------------------------------------
    # NORMAL OPTIONS MODE — unchanged
    # ------------------------------------------------------------------
    portfolio_json = get_portfolio_json()
    spot_range = np.linspace(
        spot_price * (1 - SPOT_RANGE_FACTOR),
        spot_price * (1 + SPOT_RANGE_FACTOR),
        SPOT_RANGE_POINTS,
    )
    has_positions = (
        bool(st.session_state.positions) or st.session_state.stock_position is not None
    )

    with st.spinner("Calculating options data..."):
        all_data = calculate_all_surfaces(
            portfolio_json=portfolio_json,
            spot_range=tuple(spot_range),
            dte_values=tuple(DTE_RANGE),
            iv_values=tuple(IV_RANGE),
            risk_free_rate=risk_free_rate,
            _calculate_all_greeks_func=calculate_all_greeks,
            _calculate_pnl_at_expiry_func=calculate_portfolio_pnl_at_expiry,
            _find_breakeven_func=find_breakeven_points,
            has_positions=has_positions,
        )

    has_exotic_legs = False

    greeks_3d_dte_func = calculate_portfolio_greeks_3d_dte
    greeks_3d_iv_func = calculate_portfolio_greeks_3d_iv
    calculate_greeks_3d_strike_func = calculate_greeks_3d_strike
    calculate_all_greeks_func = calculate_all_greeks
    calculate_pnl_at_expiry_func = calculate_portfolio_pnl_at_expiry
    find_breakeven_func = find_breakeven_points
    calculate_exotic_greeks_func = None
    positions_for_tabs = st.session_state.positions
    stock_position_for_tabs = st.session_state.stock_position


# =============================================================================
# SHARED TABS
# =============================================================================

tab_names = [
    "P&L Profile",
    "First-Order Greeks",
    "Second-Order Greeks",
    "Third-Order Greeks",
    "3D Surface",
    "Reference Guide",
]

tabs = st.tabs(tab_names)

tab1, tab2, tab3, tab4, tab5 = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4]
tab7 = tabs[5]

# Extract custom ranges for structured products (None for normal options mode)
sp_dte_range = all_data.get("dte_range")
sp_iv_range = all_data.get("iv_range")

# Tab 1: P&L Profile
with tab1:
    if sp_mode:
        st.info(
            "P&L Profile pour produits structurés — en cours de développement. "
            "Cette fonctionnalité sera disponible dans une prochaine mise à jour."
        )
    else:
        default_premium = get_default_premium() if not has_positions else 10.0
        render_pnl_tab(
            all_data=all_data,
            spot_range=spot_range,
            spot_price=spot_price,
            positions=positions_for_tabs,
            stock_position=stock_position_for_tabs,
            default_premium=default_premium,
            risk_free_rate=risk_free_rate,
            calculate_all_greeks_func=calculate_all_greeks_func,
            calculate_pnl_at_expiry_func=calculate_pnl_at_expiry_func,
            find_breakeven_func=find_breakeven_func,
            has_exotic_legs=all_data.get("has_exotic_legs", False),
            sp_mode=sp_mode,
            dte_range=sp_dte_range,
            iv_range=sp_iv_range,
        )

# Tab 2: First-Order Greeks
with tab2:
    if sp_mode:
        st.info(
            "Greeks pour produits structurés — en cours de développement. "
            "Cette fonctionnalité sera disponible dans une prochaine mise à jour."
        )
    else:
        render_first_order_greeks(
            greeks_data=all_data["greeks_data"],
            spot_range=spot_range,
            spot_price=spot_price,
            positions=positions_for_tabs,
            stock_position=stock_position_for_tabs,
            risk_free_rate=risk_free_rate,
            calculate_all_greeks_func=calculate_all_greeks_func,
            calculate_pnl_at_expiry_func=calculate_pnl_at_expiry_func,
            portfolio_json=portfolio_json,
            calculate_exotic_greeks_func=calculate_exotic_greeks_func,
            sp_mode=sp_mode,
            dte_range=sp_dte_range,
            iv_range=sp_iv_range,
        )

# Tab 3: Second-Order Greeks
with tab3:
    if sp_mode:
        st.info(
            "Greeks pour produits structurés — en cours de développement. "
            "Cette fonctionnalité sera disponible dans une prochaine mise à jour."
        )
    else:
        render_second_order_greeks(
            greeks_data=all_data["greeks_data"],
            spot_range=spot_range,
            spot_price=spot_price,
            positions=positions_for_tabs,
            stock_position=stock_position_for_tabs,
            risk_free_rate=risk_free_rate,
            calculate_all_greeks_func=calculate_all_greeks_func,
            calculate_pnl_at_expiry_func=calculate_pnl_at_expiry_func,
            portfolio_json=portfolio_json,
            has_exotic_legs=all_data.get("has_exotic_legs", False),
            calculate_exotic_greeks_func=calculate_exotic_greeks_func,
            sp_mode=sp_mode,
            dte_range=sp_dte_range,
            iv_range=sp_iv_range,
        )

# Tab 4: Third-Order Greeks
with tab4:
    if sp_mode:
        st.info(
            "Greeks pour produits structurés — en cours de développement. "
            "Cette fonctionnalité sera disponible dans une prochaine mise à jour."
        )
    else:
        render_third_order_greeks(
            greeks_data=all_data["greeks_data"],
            spot_range=spot_range,
            spot_price=spot_price,
            positions=positions_for_tabs,
            stock_position=stock_position_for_tabs,
            risk_free_rate=risk_free_rate,
            calculate_all_greeks_func=calculate_all_greeks_func,
            calculate_pnl_at_expiry_func=calculate_pnl_at_expiry_func,
            portfolio_json=portfolio_json,
            has_exotic_legs=all_data.get("has_exotic_legs", False),
            calculate_exotic_greeks_func=calculate_exotic_greeks_func,
            sp_mode=sp_mode,
            dte_range=sp_dte_range,
            iv_range=sp_iv_range,
        )

# Tab 5: 3D Surface
with tab5:
    if sp_mode:
        st.info(
            "Greeks pour produits structurés — en cours de développement. "
            "Cette fonctionnalité sera disponible dans une prochaine mise à jour."
        )
    else:
        render_3d_tab(
            portfolio_json=portfolio_json,
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            volatility=DEFAULT_IV / 100,
            dte=DEFAULT_DTE,
            positions=positions_for_tabs,
            stock_position=stock_position_for_tabs,
            _calculate_greeks_3d_dte_func=greeks_3d_dte_func,
            _calculate_greeks_3d_iv_func=greeks_3d_iv_func,
            _calculate_greeks_3d_strike_func=calculate_greeks_3d_strike_func,
        )

# Tab 6: Reference Guide
with tab7:
    render_guide_section()


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(footer_html(), unsafe_allow_html=True)
