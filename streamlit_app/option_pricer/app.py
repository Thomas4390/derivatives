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

import streamlit as st
import numpy as np
import json

# Pricing functions (using backend architecture, no classes)
from services.pricing_adapter import (
    calculate_all_greeks,
    calculate_option_premium,
    calculate_pnl_at_expiry_arrays as calculate_portfolio_pnl_at_expiry,
    find_breakeven_points,
    calculate_portfolio_greeks_3d_dte,
    calculate_portfolio_greeks_3d_iv,
    calculate_greeks_3d_strike,
)

# Local imports
from config.styles import inject_styles, render_header, footer_html
from config.constants import (
    DTE_RANGE,
    IV_RANGE,
    SPOT_RANGE_FACTOR,
    SPOT_RANGE_POINTS,
    DEFAULT_DTE,
    DEFAULT_IV
)
from components.sidebar import render_sidebar
from charts.pnl_chart import render_pnl_tab
from charts.greeks_chart import (
    render_first_order_greeks,
    render_second_order_greeks,
    render_third_order_greeks
)
from charts.surface_3d import render_3d_tab
from services.state_manager import init_session_state
from services.portfolio_calculator import (
    calculate_all_surfaces,
    prepare_portfolio_data
)
from guide_formula import render_guide_section
from charts.exotic_charts import render_exotic_tab
from services.exotic_pricing_adapter import calculate_exotic_all_greeks


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Options Greeks Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS styles
inject_styles()

# Initialize session state
init_session_state()


# =============================================================================
# SIDEBAR
# =============================================================================

spot_price, risk_free_rate = render_sidebar(
    positions=st.session_state.positions,
    stock_position=st.session_state.stock_position
)


# =============================================================================
# HEADER
# =============================================================================

render_header(
    title="Options Greeks Explorer",
    subtitle="Interactive visualization of option pricing and risk metrics using the Black-Scholes model",
    badge="Educational Tool"
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
        option_type='call'
    )


def get_portfolio_json() -> str:
    """Prepare portfolio data as JSON string."""
    positions = st.session_state.positions
    stock_position = st.session_state.stock_position

    if positions or stock_position:
        portfolio_data = prepare_portfolio_data(positions, stock_position, spot_price)
    else:
        # Empty portfolio - no default position
        portfolio_data = {
            'spot_price': spot_price,
            'options': [],
            'stock': None
        }

    return json.dumps(portfolio_data)


# Prepare calculation parameters
portfolio_json = get_portfolio_json()
spot_range = np.linspace(
    spot_price * (1 - SPOT_RANGE_FACTOR),
    spot_price * (1 + SPOT_RANGE_FACTOR),
    SPOT_RANGE_POINTS
)
has_positions = bool(st.session_state.positions) or st.session_state.stock_position is not None

# Calculate all data
with st.spinner('Calculating options data...'):
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
        _calculate_exotic_greeks_func=calculate_exotic_all_greeks,
    )


# =============================================================================
# MAIN TABS
# =============================================================================

has_exotic_legs = any(
    pos.get('instrument_class', 'vanilla') != 'vanilla'
    for pos in st.session_state.positions
)

tab_names = [
    "P&L Profile",
    "First-Order Greeks",
    "Second-Order Greeks",
    "Third-Order Greeks",
    "3D Surface",
]
if has_exotic_legs:
    tab_names.append("Exotic Options")
tab_names.append("Reference Guide")

tabs = st.tabs(tab_names)

tab1, tab2, tab3, tab4, tab5 = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4]
if has_exotic_legs:
    tab6 = tabs[5]
    tab7 = tabs[6]
else:
    tab6 = None
    tab7 = tabs[5]


# Tab 1: P&L Profile
with tab1:
    default_premium = get_default_premium() if not has_positions else 10.0
    render_pnl_tab(
        all_data=all_data,
        spot_range=spot_range,
        spot_price=spot_price,
        positions=st.session_state.positions,
        stock_position=st.session_state.stock_position,
        default_premium=default_premium,
        risk_free_rate=risk_free_rate,
        calculate_all_greeks_func=calculate_all_greeks,
        calculate_pnl_at_expiry_func=calculate_portfolio_pnl_at_expiry,
        find_breakeven_func=find_breakeven_points,
        has_exotic_legs=all_data.get('has_exotic_legs', False),
    )


# Tab 2: First-Order Greeks
with tab2:
    render_first_order_greeks(
        greeks_data=all_data['greeks_data'],
        spot_range=spot_range,
        spot_price=spot_price,
        positions=st.session_state.positions,
        stock_position=st.session_state.stock_position,
        risk_free_rate=risk_free_rate,
        calculate_all_greeks_func=calculate_all_greeks,
        calculate_pnl_at_expiry_func=calculate_portfolio_pnl_at_expiry,
        portfolio_json=portfolio_json,
        calculate_exotic_greeks_func=calculate_exotic_all_greeks,
    )


# Tab 3: Second-Order Greeks
with tab3:
    render_second_order_greeks(
        greeks_data=all_data['greeks_data'],
        spot_range=spot_range,
        spot_price=spot_price,
        positions=st.session_state.positions,
        stock_position=st.session_state.stock_position,
        risk_free_rate=risk_free_rate,
        calculate_all_greeks_func=calculate_all_greeks,
        calculate_pnl_at_expiry_func=calculate_portfolio_pnl_at_expiry,
        portfolio_json=portfolio_json,
        has_exotic_legs=all_data.get('has_exotic_legs', False),
        calculate_exotic_greeks_func=calculate_exotic_all_greeks,
    )


# Tab 4: Third-Order Greeks
with tab4:
    render_third_order_greeks(
        greeks_data=all_data['greeks_data'],
        spot_range=spot_range,
        spot_price=spot_price,
        positions=st.session_state.positions,
        stock_position=st.session_state.stock_position,
        risk_free_rate=risk_free_rate,
        calculate_all_greeks_func=calculate_all_greeks,
        calculate_pnl_at_expiry_func=calculate_portfolio_pnl_at_expiry,
        portfolio_json=portfolio_json,
        has_exotic_legs=all_data.get('has_exotic_legs', False),
        calculate_exotic_greeks_func=calculate_exotic_all_greeks,
    )


# Tab 5: 3D Surface
with tab5:
    render_3d_tab(
        portfolio_json=portfolio_json,
        spot_price=spot_price,
        risk_free_rate=risk_free_rate,
        volatility=DEFAULT_IV / 100,  # Convert percentage to decimal
        dte=DEFAULT_DTE,
        positions=st.session_state.positions,
        stock_position=st.session_state.stock_position,
        _calculate_greeks_3d_dte_func=calculate_portfolio_greeks_3d_dte,
        _calculate_greeks_3d_iv_func=calculate_portfolio_greeks_3d_iv,
        _calculate_greeks_3d_strike_func=calculate_greeks_3d_strike
    )


# Tab 6: Exotic Options (conditional)
if tab6 is not None:
    with tab6:
        render_exotic_tab(spot_price=spot_price, risk_free_rate=risk_free_rate)


# Tab 7: Reference Guide
with tab7:
    render_guide_section()


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(footer_html(), unsafe_allow_html=True)
