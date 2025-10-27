"""
Streamlit application for options analysis - With Stock Position Support
Performance improvements while maintaining all features
CORRECTED VERSION with proper CONTRACT_MULTIPLIER and theoretical max/min calculations
Enhanced with custom hover templates for all charts
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
from functools import lru_cache
from backend.option_pricing.options_calculator import (
    OptionsPortfolio, OptionPosition, StockPosition,
    calculate_all_greeks, calculate_portfolio_pnl_at_expiry,
    calculate_portfolio_greeks_3d_dte, calculate_portfolio_greeks_3d_iv,
    find_breakeven_points
)
from guide_formula import render_guide_section

# ============= CONFIGURATION =============
st.set_page_config(
    page_title="Options Greeks Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= CONSTANTS =============
# Options contract multiplier (1 contract = 100 shares)
CONTRACT_MULTIPLIER = 100

GREEK_NAMES = ['price', 'delta', 'gamma', 'vega', 'theta', 'rho',
               'vanna', 'volga', 'charm', 'veta', 'speed', 'zomma', 'color', 'ultima']

FIRST_ORDER = ['delta', 'gamma', 'vega', 'theta', 'rho']
SECOND_ORDER = ['vanna', 'volga', 'charm', 'veta']
THIRD_ORDER = ['speed', 'zomma', 'color', 'ultima']

GREEK_TITLES = {
    'delta': 'Delta (∂V/∂S)', 'gamma': 'Gamma (∂²V/∂S²)',
    'vega': 'Vega (∂V/∂σ)', 'theta': 'Theta (∂V/∂t)', 'rho': 'Rho (∂V/∂r)',
    'vanna': 'Vanna (∂²V/∂S∂σ)', 'volga': 'Volga/Vomma (∂²V/∂σ²)',
    'charm': 'Charm (∂²V/∂S∂t)', 'veta': 'Veta (∂²V/∂σ∂t)',
    'speed': 'Speed (∂³V/∂S³)', 'zomma': 'Zomma (∂³V/∂S²∂σ)',
    'color': 'Color (∂³V/∂S²∂t)', 'ultima': 'Ultima (∂³V/∂σ³)'
}

GREEK_COLORS = {
    'delta': '#1f77b4', 'gamma': '#ff7f0e', 'vega': '#2ca02c',
    'theta': '#d62728', 'rho': '#9467bd', 'vanna': '#e377c2',
    'volga': '#17becf', 'charm': '#bcbd22', 'veta': '#ff9896',
    'speed': '#ff9896', 'zomma': '#8c564b', 'color': '#7f7f7f', 'ultima': '#c5b0d5'
}

# ============= STYLES =============
st.markdown("""
<style>
    .metric-card {
        background: white;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 5px 0;
        border-left: 3px solid #667eea;
    }
    .metric-card h4 {
        margin: 0;
        font-size: 11px;
        color: #667eea;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card h2 {
        margin: 3px 0;
        font-size: 20px;
    }
    .position-item {
        background: #fff;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border: 1px solid #e0e0e0;
        font-size: 14px;
    }
    .position-debit { border-left: 3px solid #ef4444; }
    .position-credit { border-left: 3px solid #10b981; }
    .position-stock { border-left: 3px solid #3b82f6; }
    .net-position-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============= UTILITY FUNCTIONS =============

@lru_cache(maxsize=128)
def get_portfolio_hash(portfolio_json):
    """Create a hash for portfolio caching"""
    return hash(portfolio_json)

def calculate_net_position(positions, stock_position=None):
    """Calculate the net debit/credit position including stock"""
    net = sum(
        -pos.premium_paid * pos.quantity * CONTRACT_MULTIPLIER if pos.position_type == 'long'
        else pos.premium_paid * pos.quantity * CONTRACT_MULTIPLIER
        for pos in positions
    )

    # Add stock cost if present
    if stock_position:
        if stock_position.position_type == 'long':
            net -= stock_position.entry_price * stock_position.quantity
        else:  # short stock
            net += stock_position.entry_price * stock_position.quantity

    return net

def check_unlimited_risk(portfolio_data):
    """
    Check if portfolio has unlimited profit or loss potential
    Returns: (has_unlimited_profit, has_unlimited_loss)
    """
    # Stock positions always have unlimited potential in one direction
    if portfolio_data.get('stock'):
        stock = portfolio_data['stock']
        if stock['position_type'] == 'long':
            return True, False  # Unlimited profit upside, limited loss (stock price to 0)
        else:  # short stock
            return False, True  # Limited profit (stock to 0), unlimited loss upside

    # For options-only portfolios
    if not portfolio_data.get('options'):
        return False, False  # No positions = no unlimited risk

    long_calls = []
    short_calls = []
    long_puts = []
    short_puts = []

    for pos in portfolio_data['options']:
        strike = pos['strike']
        quantity = pos['quantity']

        if pos['option_type'] == 'call':
            if pos['position_type'] == 'long':
                long_calls.append((strike, quantity))
            else:  # short
                short_calls.append((strike, quantity))
        else:  # put
            if pos['position_type'] == 'long':
                long_puts.append((strike, quantity))
            else:  # short
                short_puts.append((strike, quantity))

    unlimited_profit = False
    unlimited_loss = False

    # Sort calls by strike
    long_calls.sort(key=lambda x: x[0])
    short_calls.sort(key=lambda x: x[0])

    # Check for unlimited profit potential
    # Net long calls at the highest strike = unlimited profit
    if long_calls or short_calls:
        # Calculate net position at each strike
        call_positions = {}
        for strike, qty in long_calls:
            call_positions[strike] = call_positions.get(strike, 0) + qty
        for strike, qty in short_calls:
            call_positions[strike] = call_positions.get(strike, 0) - qty

        # Check if we have net long calls at the highest strike
        if call_positions:
            highest_strike = max(call_positions.keys())
            if call_positions[highest_strike] > 0:
                unlimited_profit = True

    # Check for unlimited loss potential
    # Only naked short calls (without any long calls to cap the loss) create unlimited loss
    if short_calls and not long_calls:
        # We have short calls with no long calls at all
        unlimited_loss = True
    elif short_calls and long_calls:
        # Check if all short calls are covered or capped
        total_short = sum(qty for _, qty in short_calls)
        total_long = sum(qty for _, qty in long_calls)

        # For spreads like butterfly and iron condor:
        # If we have at least as many long calls as short calls (considering all strikes),
        # the loss is limited
        if total_short > total_long:
            # More short calls than long calls = potential unlimited loss
            unlimited_loss = True

    # Note: Puts can NEVER create unlimited loss as stock can't go below 0

    return unlimited_profit, unlimited_loss

@st.cache_data(ttl=3600)
def calculate_all_surfaces(portfolio_json, spot_range, dte_values, iv_values, risk_free_rate):
    """Calculate all data at once to avoid redundant calculations"""
    portfolio_data = json.loads(portfolio_json)

    # Initialize results
    pnl_data = {}
    greeks_data = {}

    # Prepare portfolio arrays - check if options list is not empty
    if portfolio_data.get('options') and len(portfolio_data['options']) > 0:
        strikes = np.array([pos['strike'] for pos in portfolio_data['options']])
        option_types = np.array([1 if pos['option_type'] == 'call' else 0
                                 for pos in portfolio_data['options']])
        position_types = np.array([1 if pos['position_type'] == 'long' else -1
                                  for pos in portfolio_data['options']])
        # Multiply by CONTRACT_MULTIPLIER to convert contracts to shares equivalent
        quantities = np.array([pos['quantity'] * CONTRACT_MULTIPLIER for pos in portfolio_data['options']])
        premiums = np.array([pos['premium_paid'] for pos in portfolio_data['options']])
    else:
        # No options positions - create empty arrays
        strikes = np.array([])
        option_types = np.array([], dtype=np.int32)
        position_types = np.array([], dtype=np.int32)
        quantities = np.array([], dtype=np.int32)
        premiums = np.array([])

    stock_quantity = 0
    stock_entry_price = 0
    if portfolio_data.get('stock'):
        stock = portfolio_data['stock']
        stock_quantity = stock['quantity'] * (1 if stock['position_type'] == 'long' else -1)
        stock_entry_price = stock['entry_price']

    # Calculate for each DTE/IV combination
    for dte in dte_values:
        time_to_expiry = dte / 365.0

        for iv in iv_values:
            key = f"{dte}_{iv}"
            iv_decimal = iv / 100.0

            # Initialize arrays for this combination
            pnl_values = np.zeros(len(spot_range))
            greeks_by_name = {name: np.zeros(len(spot_range)) for name in GREEK_NAMES}

            # Vectorized calculation for all spots
            for i, spot in enumerate(spot_range):
                total_pnl = 0
                total_greeks = np.zeros(14)

                # Calculate for each position
                for j in range(len(strikes)):
                    greeks = calculate_all_greeks(
                        spot, strikes[j], time_to_expiry,
                        risk_free_rate, iv_decimal, option_types[j]
                    )

                    # P&L calculation
                    option_value = greeks[0]
                    if position_types[j] == 1:  # Long
                        pnl = (option_value - premiums[j]) * quantities[j]
                    else:  # Short
                        pnl = (premiums[j] - option_value) * quantities[j]
                    total_pnl += pnl

                    # Greeks aggregation
                    total_greeks += greeks * quantities[j] * position_types[j]

                # Add stock P&L if present
                if stock_quantity != 0:
                    stock_pnl = (spot - stock_entry_price) * stock_quantity
                    total_pnl += stock_pnl
                    # Stock contributes delta of 1 per share
                    total_greeks[1] += stock_quantity  # Add to delta

                pnl_values[i] = total_pnl

                # Store greeks
                for k, name in enumerate(GREEK_NAMES):
                    greeks_by_name[name][i] = total_greeks[k]

            pnl_data[key] = pnl_values
            greeks_data[key] = greeks_by_name

    # Calculate P&L at expiration (for visualization range)
    expiry_pnl = np.zeros(len(spot_range))
    for i, spot in enumerate(spot_range):
        expiry_pnl[i] = calculate_portfolio_pnl_at_expiry(
            spot, strikes, option_types, position_types,
            quantities, premiums, stock_quantity, stock_entry_price
        )
    pnl_data['expiry'] = expiry_pnl

    # Find breakeven points and theoretical max/min using MUCH WIDER range
    # Use near-zero to 10x spot price to capture theoretical extremes
    # This ensures we find the true max loss (e.g., when stock goes to 0 for short puts)
    # and true max profit (for strategies with limited profit)
    theoretical_min = 0.01
    theoretical_max = portfolio_data.get('spot_price', 100.0) * 10.0

    breakeven_result = find_breakeven_points(
        strikes, option_types, position_types, quantities,
        premiums, stock_quantity, stock_entry_price,
        theoretical_min, theoretical_max, 20000  # More points for better accuracy
    ) if (len(strikes) > 0 or stock_quantity != 0) else None

    # Special case: For short puts, calculate P&L at exactly 0 for accurate max loss
    if any(portfolio_data.get('options', [])):
        has_short_puts = any(
            pos['option_type'] == 'put' and pos['position_type'] == 'short'
            for pos in portfolio_data['options']
        )
        if has_short_puts and breakeven_result:
            # Calculate P&L exactly at spot = 0
            pnl_at_zero = calculate_portfolio_pnl_at_expiry(
                0.0, strikes, option_types, position_types,
                quantities, premiums, stock_quantity, stock_entry_price
            )
            # Update max loss if P&L at 0 is worse
            if pnl_at_zero < breakeven_result.max_loss:
                breakeven_result.max_loss = pnl_at_zero
                breakeven_result.max_loss_spot = 0.0
    if not st.session_state.positions and not st.session_state.stock_position:
        # Default position is a long call, which has unlimited profit potential
        unlimited_profit = True
        unlimited_loss = False
    else:
        unlimited_profit, unlimited_loss = check_unlimited_risk(portfolio_data)

    # Create enhanced result with all information
    result = {
        'pnl_data': pnl_data,
        'greeks_data': greeks_data,
        'breakeven_result': breakeven_result,
        'unlimited_profit': unlimited_profit,
        'unlimited_loss': unlimited_loss
    }

    # Override max profit/loss for unlimited cases
    if breakeven_result:
        # Check if profit is truly unlimited
        if unlimited_profit:
            # For unlimited profit, verify P&L continues to increase at high spot prices
            if len(expiry_pnl) > 10:
                high_end_trend = expiry_pnl[-1] - expiry_pnl[-10]
                if high_end_trend > 0:  # Profit is increasing at high end
                    result['max_profit_display'] = float('inf')
                else:
                    result['max_profit_display'] = breakeven_result.max_profit
            else:
                result['max_profit_display'] = float('inf')
        else:
            # Profit is limited, use calculated value
            result['max_profit_display'] = breakeven_result.max_profit

        # Check if loss is truly unlimited
        if unlimited_loss:
            # For unlimited loss, verify P&L continues to decrease at high spot prices
            # This only applies to naked short calls or short stock
            if len(expiry_pnl) > 10:
                high_end_trend = expiry_pnl[-1] - expiry_pnl[-10]
                if high_end_trend < 0 and expiry_pnl[-1] < 0:  # Loss is worsening at high end
                    result['max_loss_display'] = float('-inf')
                else:
                    result['max_loss_display'] = breakeven_result.max_loss
            else:
                result['max_loss_display'] = float('-inf')
        else:
            # Loss is limited, use calculated value (don't override)
            result['max_loss_display'] = breakeven_result.max_loss
    else:
        result['max_profit_display'] = 0
        result['max_loss_display'] = 0

    return result

def create_greeks_subplot(greeks_list, greeks_data, slider_type, dte_values, iv_values,
                         spot_range, spot_price, subplot_rows=2, subplot_cols=2):
    """Reusable function to create Greeks subplots with custom hover templates"""
    fig = make_subplots(
        rows=subplot_rows, cols=subplot_cols,
        subplot_titles=[GREEK_TITLES[g] for g in greeks_list[:subplot_rows*subplot_cols]],
        vertical_spacing=0.20 if subplot_rows == 2 else 0.12,
        horizontal_spacing=0.12
    )

    positions = [(r+1, c+1) for r in range(subplot_rows) for c in range(subplot_cols)]

    # Add traces
    for greek_idx, greek_name in enumerate(greeks_list):
        if greek_idx >= len(positions):
            break
        row, col = positions[greek_idx]

        if slider_type == "DTE":
            fixed_iv = 25
            for dte in dte_values:
                key = f"{dte}_{fixed_iv}"
                visible = (dte == 31)

                fig.add_trace(
                    go.Scatter(
                        x=spot_range,
                        y=greeks_data[key][greek_name],
                        mode='lines',
                        name=f'{greek_name}: DTE={dte}',
                        visible=visible,
                        line=dict(width=2, color=GREEK_COLORS[greek_name]),
                        showlegend=False,
                        hovertemplate=(
                            f'<b>Underlying Price</b>: %{{x:.2f}}<br>' +
                            f'<b>DTE</b>: {dte} days<br>' +
                            f'<b>{GREEK_TITLES[greek_name]}</b>: %{{y:.4f}}<br>' +
                            '<extra></extra>'
                        )
                    ),
                    row=row, col=col
                )
        else:  # IV mode
            fixed_dte = 31
            for iv in iv_values:
                key = f"{fixed_dte}_{iv}"
                visible = (iv == 25)

                fig.add_trace(
                    go.Scatter(
                        x=spot_range,
                        y=greeks_data[key][greek_name],
                        mode='lines',
                        name=f'{greek_name}: IV={iv}%',
                        visible=visible,
                        line=dict(width=2, color=GREEK_COLORS[greek_name]),
                        showlegend=False,
                        hovertemplate=(
                            f'<b>Underlying Price</b>: %{{x:.2f}}<br>' +
                            f'<b>IV</b>: {iv}%<br>' +
                            f'<b>{GREEK_TITLES[greek_name]}</b>: %{{y:.4f}}<br>' +
                            '<extra></extra>'
                        )
                    ),
                    row=row, col=col
                )

        # Add reference lines
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3, row=row, col=col)
        fig.add_vline(x=spot_price, line_dash="dot", line_color="green", opacity=0.3, row=row, col=col)

    # Create slider
    param_values = dte_values if slider_type == "DTE" else iv_values
    steps = []

    for idx, value in enumerate(param_values):
        step = dict(
            method="update",
            args=[{"visible": [False] * (len(greeks_list) * len(param_values))}],
            label=str(value) if slider_type == "DTE" else f"{value}%"
        )
        for greek_idx in range(len(greeks_list)):
            if greek_idx < len(positions):
                trace_idx = greek_idx * len(param_values) + idx
                step["args"][0]["visible"][trace_idx] = True
        steps.append(step)

    slider_dict = dict(
        active=10,
        currentvalue={"prefix": f"{'Days to Expiration' if slider_type == 'DTE' else 'Implied Volatility'}: "},
        steps=steps,
        y=-0.10,
        len=0.9,
        x=0.05
    )

    return fig, slider_dict

# ============= SESSION STATE =============
if 'positions' not in st.session_state:
    st.session_state.positions = []
if 'stock_position' not in st.session_state:
    st.session_state.stock_position = None

# ============= MAIN TITLE =============
st.title("🎯 Options Greeks Explorer")
st.markdown("**High-performance options analysis with complete Greeks calculation**")

# ============= SIDEBAR =============
with st.sidebar:
    st.header("⚙️ Market Parameters")

    spot_price = st.number_input(
        "Spot Price (S₀)", value=100.0, step=1.0, format="%.2f",
        help="Current price of the underlying asset"
    )

    risk_free_rate = st.number_input(
        "Risk-Free Rate (r)", value=0.05, step=0.01, format="%.3f",
        help="Annual risk-free interest rate"
    )

    st.markdown("---")
    st.header("➕ Add Position")

    # Tabs for adding positions
    tab_option, tab_stock = st.tabs(["📈 Option", "💹 Stock"])

    with tab_option:
        with st.form("add_option", clear_on_submit=True):
            option_type = st.selectbox("Option Type", ["call", "put"])
            position_type = st.selectbox("Direction", ["long", "short"])
            strike = st.number_input("Strike Price (K)", value=spot_price, step=1.0, format="%.2f")
            quantity = st.number_input("Contracts (1 contract = 100 shares)", value=1, min_value=1, step=1)

            # Calculate premium
            portfolio_temp = OptionsPortfolio(spot_price, risk_free_rate)
            position_temp = OptionPosition(option_type, position_type, strike, quantity)
            portfolio_temp.add_option_position(position_temp)
            premium = position_temp.premium_paid

            # Display position type - CORRECTLY account for CONTRACT_MULTIPLIER
            total_cost = premium * quantity * CONTRACT_MULTIPLIER
            st.info(f"📊 Premium per contract: ${premium:.2f}")
            if position_type == "long":
                st.warning(f"💸 Total Debit: ${total_cost:.2f}")
            else:
                st.success(f"💰 Total Credit: ${total_cost:.2f}")

            if st.form_submit_button("Add Option", use_container_width=True, type="primary"):
                position = OptionPosition(option_type, position_type, strike, quantity, premium)
                st.session_state.positions.append(position)
                st.rerun()

    with tab_stock:
        with st.form("add_stock", clear_on_submit=True):
            stock_position_type = st.selectbox("Direction", ["long", "short"], key="stock_dir")
            stock_quantity = st.number_input("Shares", value=100, min_value=1, step=1)
            stock_entry_price = st.number_input("Entry Price", value=spot_price, step=1.0, format="%.2f")

            # Display position cost
            stock_cost = stock_entry_price * stock_quantity
            if stock_position_type == "long":
                st.info(f"💵 Buy {stock_quantity} shares @ ${stock_entry_price:.2f}")
                st.warning(f"Total Cost: ${stock_cost:.2f}")
            else:
                st.info(f"📉 Short {stock_quantity} shares @ ${stock_entry_price:.2f}")
                st.success(f"Credit Received: ${stock_cost:.2f}")

            if st.form_submit_button("Add Stock", use_container_width=True, type="primary"):
                stock_pos = StockPosition(stock_position_type, stock_quantity, stock_entry_price)
                st.session_state.stock_position = stock_pos
                st.rerun()

    # Predefined strategies
    st.markdown("---")
    st.header("🎯 Predefined Strategies")

    strategy = st.selectbox(
        "Select a Strategy",
        ["", "long_straddle", "iron_condor", "butterfly", "covered_call",
         "protective_put", "bull_call_spread", "bear_put_spread", "collar"],
        format_func=lambda x: x.replace('_', ' ').title() if x else ""
    )

    if st.button("Apply Strategy", use_container_width=True):
        if strategy:
            portfolio = OptionsPortfolio(spot_price, risk_free_rate)
            portfolio.create_strategy(strategy)
            st.session_state.positions = portfolio.positions
            st.session_state.stock_position = portfolio.stock_position
            st.rerun()

    # Current positions display
    st.markdown("---")
    st.header("📋 Current Positions")

    if st.session_state.positions or st.session_state.stock_position:
        net_amount = calculate_net_position(st.session_state.positions, st.session_state.stock_position)

        # Net position card
        position_type = "💰 Credit" if net_amount > 0 else "💸 Debit" if net_amount < 0 else "⚖️ Neutral"
        amount_text = f"${abs(net_amount):.2f}" if net_amount != 0 else "$0.00"

        st.markdown(f"""
        <div class="net-position-card">
            <strong>NET POSITION</strong><br>
            <span style="font-size: 20px;">{position_type}: {amount_text}</span>
        </div>
        """, unsafe_allow_html=True)

        # Stock position
        if st.session_state.stock_position:
            stock = st.session_state.stock_position
            stock_cost = stock.entry_price * stock.quantity
            stock_info = f"💵 Cost: ${stock_cost:.2f}" if stock.position_type == 'long' else f"📉 Credit: ${stock_cost:.2f}"

            st.markdown(f"""
            <div class="position-item position-stock">
                <strong>📊 STOCK:</strong> {stock.quantity}x {stock.position_type.upper()} @ ${stock.entry_price:.2f}
                <br><strong>{stock_info}</strong>
            </div>
            """, unsafe_allow_html=True)

        # Option positions
        for i, pos in enumerate(st.session_state.positions):
            # CORRECTED: Calculate total amount with CONTRACT_MULTIPLIER
            total_amount = pos.premium_paid * pos.quantity * CONTRACT_MULTIPLIER
            pos_class = "position-debit" if pos.position_type == 'long' else "position-credit"
            pos_info = f"💸 Total Debit: ${total_amount:.2f}" if pos.position_type == 'long' else f"💰 Total Credit: ${total_amount:.2f}"
            shares_controlled = pos.quantity * CONTRACT_MULTIPLIER

            st.markdown(f"""
            <div class="position-item {pos_class}">
                <strong>{i+1}.</strong> {pos.quantity}x {pos.position_type.upper()} {pos.option_type.upper()} @ {pos.strike:.1f}
                <br>Premium per contract: ${pos.premium_paid:.2f}
                <br>Controls: {shares_controlled} shares
                <br><strong>{pos_info}</strong>
            </div>
            """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.positions = []
                st.session_state.stock_position = None
                st.rerun()
        with col2:
            if st.button("↩️ Remove Last", use_container_width=True):
                if st.session_state.positions:
                    st.session_state.positions.pop()
                elif st.session_state.stock_position:
                    st.session_state.stock_position = None
                st.rerun()
    else:
        st.info("No active positions")

# ============= DATA PREPARATION =============
# Calculate default position premium if no positions exist
if not st.session_state.positions and not st.session_state.stock_position:
    # Create default position with Black-Scholes calculated premium
    default_portfolio = OptionsPortfolio(spot_price, risk_free_rate)
    default_position = OptionPosition('call', 'long', spot_price, 1)
    default_portfolio.add_option_position(default_position)
    default_premium = default_position.premium_paid
else:
    default_premium = 10.0  # Not used when positions exist

portfolio_data = {
    'spot_price': spot_price,
    'options': [
        {
            'option_type': pos.option_type,
            'position_type': pos.position_type,
            'strike': pos.strike,
            'quantity': pos.quantity,
            'premium_paid': pos.premium_paid
        } for pos in st.session_state.positions
    ] if st.session_state.positions else [{
        'option_type': 'call',
        'position_type': 'long',
        'strike': spot_price,
        'quantity': 1,
        'premium_paid': default_premium
    }] if not st.session_state.stock_position else [],
    'stock': None
}

if st.session_state.stock_position:
    stock = st.session_state.stock_position
    portfolio_data['stock'] = {
        'position_type': stock.position_type,
        'quantity': stock.quantity,
        'entry_price': stock.entry_price
    }

portfolio_json = json.dumps(portfolio_data)
spot_range = np.linspace(spot_price * 0.7, spot_price * 1.3, 200)
dte_values = list(range(1, 91, 3))  # Every 3 days
iv_values = list(range(5, 51, 2))   # Every 2%

# Calculate all data once
with st.spinner('Calculating options data...'):
    all_data = calculate_all_surfaces(portfolio_json, spot_range, dte_values, iv_values, risk_free_rate)

# ============= MAIN TABS =============
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 P&L Profile", "📈 First-Order Greeks", "📉 Second-Order Greeks",
    "📐 Third-Order Greeks", "🌐 3D Surface", "📚 Guide & Formulas"
])

# ============= TAB 1: P&L PROFILE =============
with tab1:
    if not st.session_state.positions and not st.session_state.stock_position:
        # For default position: 1x Long Call ATM
        # Max Profit: Unlimited (call has unlimited upside)
        # Max Loss: Premium paid
        # Breakeven: Strike + Premium
        breakeven_price = spot_price + default_premium
        st.info(f"""Default position: 1x Long Call ATM""")
    elif st.session_state.stock_position and not st.session_state.positions:
        stock = st.session_state.stock_position
        st.info(f"""Position: {stock.quantity}x {stock.position_type.upper()} Stock @ ${stock.entry_price:.2f}""")

    st.subheader("📈 Profit/Loss Profile with Breakeven Analysis")

    # Extract data
    pnl_data = all_data['pnl_data']
    breakeven_result = all_data['breakeven_result']
    unlimited_profit = all_data['unlimited_profit']
    unlimited_loss = all_data['unlimited_loss']

    # Use calculated values from backend
    if breakeven_result:
        max_profit = all_data.get('max_profit_display', breakeven_result.max_profit)
        max_loss = all_data.get('max_loss_display', breakeven_result.max_loss)
        max_profit_spot = breakeven_result.max_profit_spot
        max_loss_spot = breakeven_result.max_loss_spot
    else:
        max_profit = max_loss = 0
        max_profit_spot = max_loss_spot = spot_price

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        be_count = len(breakeven_result.breakeven_points) if breakeven_result else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>Breakeven Points</h4>
            <h2>{be_count}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if max_profit == float('inf'):
            profit_text = "∞ (Unlimited)"
        else:
            profit_text = f"${max_profit:.2f}"

        st.markdown(f"""
        <div class="metric-card">
            <h4>Maximum Profit</h4>
            <h2 style="color: #10b981;">{profit_text}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if max_loss == float('-inf'):
            loss_text = "∞ (Unlimited)"
        else:
            loss_text = f"${abs(max_loss):.2f}"

        st.markdown(f"""
        <div class="metric-card">
            <h4>Maximum Loss</h4>
            <h2 style="color: #ef4444;">{loss_text}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        if breakeven_result and breakeven_result.breakeven_points:
            be_points = sorted(breakeven_result.breakeven_points)
            if len(be_points) == 1:
                be_text = f"${be_points[0]:.2f}"
            elif len(be_points) == 2:
                be_text = f"${be_points[0]:.2f}<br>${be_points[1]:.2f}"
            else:
                be_text = "<br>".join([f"${bp:.2f}" for bp in be_points[:3]])
        else:
            be_text = "None"

        st.markdown(f"""
        <div class="metric-card">
            <h4>Breakeven Prices</h4>
            <h2 style="font-size: 16px;">{be_text}</h2>
        </div>
        """, unsafe_allow_html=True)

    # P&L Chart with slider
    col1, col2 = st.columns([1, 3])
    with col1:
        slider_type = st.radio("Parameter", ["DTE", "IV"],
                              format_func=lambda x: "Days to Expiration" if x == "DTE" else "Implied Volatility")
    with col2:
        param_info = "📊 Varying DTE | Fixed IV: 25%" if slider_type == "DTE" else "📊 Varying IV | Fixed DTE: 31 days"
        st.info(param_info)

    # Create P&L figure with custom hover templates
    fig = go.Figure()

    # Add traces based on slider type
    param_values = dte_values if slider_type == "DTE" else iv_values
    fixed_value = 25 if slider_type == "DTE" else 31

    for value in param_values:
        key = f"{value}_{fixed_value}" if slider_type == "DTE" else f"{fixed_value}_{value}"
        visible = (value == 31) if slider_type == "DTE" else (value == 25)

        # Create custom hover template based on slider type
        if slider_type == "DTE":
            hover_template = (
                '<b>Underlying Price</b>: %{x:.2f}<br>' +
                f'<b>Days to Expiration</b>: {value}<br>' +
                '<b>P&L</b>: $%{y:.2f}<br>' +
                '<extra></extra>'
            )
        else:
            hover_template = (
                '<b>Underlying Price</b>: %{x:.2f}<br>' +
                f'<b>Implied Volatility</b>: {value}%<br>' +
                '<b>P&L</b>: $%{y:.2f}<br>' +
                '<extra></extra>'
            )

        fig.add_trace(go.Scatter(
            x=spot_range,
            y=pnl_data[key],
            mode='lines',
            name=f'{slider_type}={value}',
            visible=visible,
            line=dict(width=2.5, color='blue'),
            hovertemplate=hover_template
        ))

    # Add expiration curve with custom hover template
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=pnl_data['expiry'],
        mode='lines',
        name='At Expiration',
        visible=True,
        line=dict(color='red', width=3, dash='dash'),
        hovertemplate=(
            '<b>Underlying Price</b>: %{x:.2f}<br>' +
            '<b>P&L at Expiration</b>: $%{y:.2f}<br>' +
            '<extra></extra>'
        )
    ))

    # Add breakeven points with better labels
    if breakeven_result and breakeven_result.breakeven_points:
        for i, be in enumerate(sorted(breakeven_result.breakeven_points)):
            label = f"BE{i+1}: ${be:.2f}" if len(breakeven_result.breakeven_points) > 1 else f"BE: ${be:.2f}"
            fig.add_vline(x=be, line_dash="dash", line_color="orange",
                         annotation_text=label,
                         annotation_position="right")

    # Reference lines
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=spot_price, line_dash="dot", line_color="green", opacity=0.5,
                  annotation_text="Current Spot", annotation_position="top left")

    # Create slider
    steps = []
    for idx, value in enumerate(param_values):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(param_values) + [True]}],
            label=str(value) if slider_type == "DTE" else f"{value}%"
        )
        step["args"][0]["visible"][idx] = True
        steps.append(step)

    fig.update_layout(
        sliders=[dict(
            active=10,
            currentvalue={"prefix": f"{'Days to Expiration' if slider_type == 'DTE' else 'Implied Volatility'}: "},
            steps=steps,
            y=-0.15,
            len=0.9,
            x=0.05
        )],
        title="Portfolio P&L Profile",
        xaxis_title="Underlying Price",
        yaxis_title="Profit/Loss ($)",
        height=700,
        margin=dict(b=120, t=50)
    )

    st.plotly_chart(fig, use_container_width=True)

# ============= TAB 2: FIRST-ORDER GREEKS =============
with tab2:
    st.subheader("📈 First-Order Greeks")

    if not st.session_state.positions and not st.session_state.stock_position:
        st.info(f"Default position: 1x Long Call ATM")

    col1, col2 = st.columns([1, 3])
    with col1:
        slider_type_g1 = st.radio("Parameter", ["DTE", "IV"], key="g1_slider")
    with col2:
        st.info("📊 Varying DTE | Fixed IV: 25%" if slider_type_g1 == "DTE" else "📊 Varying IV | Fixed DTE: 31 days")

    fig, slider_dict = create_greeks_subplot(
        FIRST_ORDER, all_data['greeks_data'], slider_type_g1,
        dte_values, iv_values, spot_range, spot_price, 3, 2
    )

    fig.update_layout(
        sliders=[slider_dict],
        height=750,
        title_text="First-Order Greeks",
        showlegend=False,
        margin=dict(b=100, t=80)
    )

    st.plotly_chart(fig, use_container_width=True)

# ============= TAB 3: SECOND-ORDER GREEKS =============
with tab3:
    st.subheader("📉 Second-Order Greeks")

    if not st.session_state.positions and not st.session_state.stock_position:
        st.info(f"Default position: 1x Long Call ATM")

    col1, col2 = st.columns([1, 3])
    with col1:
        slider_type_g2 = st.radio("Parameter", ["DTE", "IV"], key="g2_slider")
    with col2:
        st.info("📊 Varying DTE | Fixed IV: 25%" if slider_type_g2 == "DTE" else "📊 Varying IV | Fixed DTE: 31 days")

    fig, slider_dict = create_greeks_subplot(
        SECOND_ORDER, all_data['greeks_data'], slider_type_g2,
        dte_values, iv_values, spot_range, spot_price, 2, 2
    )

    fig.update_layout(
        sliders=[slider_dict],
        height=650,
        title_text="Second-Order Greeks",
        showlegend=False,
        margin=dict(b=120, t=80)
    )

    st.plotly_chart(fig, use_container_width=True)

# ============= TAB 4: THIRD-ORDER GREEKS =============
with tab4:
    st.subheader("📐 Third-Order Greeks")

    if not st.session_state.positions and not st.session_state.stock_position:
        st.info(f"Default position: 1x Long Call ATM")

    col1, col2 = st.columns([1, 3])
    with col1:
        slider_type_g3 = st.radio("Parameter", ["DTE", "IV"], key="g3_slider")
    with col2:
        st.info("📊 Varying DTE | Fixed IV: 25%" if slider_type_g3 == "DTE" else "📊 Varying IV | Fixed DTE: 31 days")

    fig, slider_dict = create_greeks_subplot(
        THIRD_ORDER, all_data['greeks_data'], slider_type_g3,
        dte_values, iv_values, spot_range, spot_price, 2, 2
    )

    fig.update_layout(
        sliders=[slider_dict],
        height=650,
        title_text="Third-Order Greeks",
        showlegend=False,
        margin=dict(b=120, t=80)
    )

    st.plotly_chart(fig, use_container_width=True)

# ============= TAB 5: 3D SURFACE =============
with tab5:
    st.subheader("🌐 3D Greeks Visualization")

    if not st.session_state.positions and not st.session_state.stock_position:
        st.info(f"Default position: 1x Long Call ATM")

    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        selected_greek = st.selectbox(
            "📊 Select Greek",
            options=GREEK_NAMES,
            format_func=lambda x: GREEK_TITLES.get(x, x.capitalize()),
            index=1  # Default to delta
        )

    with col2:
        surface_type = st.radio("📈 Y-Axis", ["DTE", "IV"],
                               format_func=lambda x: "Days to Expiration" if x == "DTE" else "Implied Volatility (%)")

    with col3:
        colorscale = st.selectbox("🎨 Color Palette",
                                 ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo', 'RdBu', 'Spectral'])

    # Add an anchor point for the 3D chart
    st.markdown('<div id="3d-chart-anchor"></div>', unsafe_allow_html=True)

    # Calculate 3D surface
    @st.cache_data(ttl=600)
    def calculate_3d_surface(portfolio_json, greek_name, surface_type, risk_free_rate):
        portfolio_data = json.loads(portfolio_json)

        # Check if options list is not empty
        if portfolio_data.get('options') and len(portfolio_data['options']) > 0:
            strikes = np.array([pos['strike'] for pos in portfolio_data['options']])
            option_types = np.array([1 if pos['option_type'] == 'call' else 0
                                    for pos in portfolio_data['options']])
            position_types = np.array([1 if pos['position_type'] == 'long' else -1
                                     for pos in portfolio_data['options']])
            # Multiply by CONTRACT_MULTIPLIER to convert contracts to shares equivalent
            quantities = np.array([pos['quantity'] * CONTRACT_MULTIPLIER for pos in portfolio_data['options']])
        else:
            # No options - create empty arrays
            strikes = np.array([])
            option_types = np.array([], dtype=np.int32)
            position_types = np.array([], dtype=np.int32)
            quantities = np.array([], dtype=np.int32)

        spot_base = portfolio_data.get('spot_price', 100.0)
        spot_range = np.linspace(spot_base * 0.7, spot_base * 1.3, 100)
        greek_idx = GREEK_NAMES.index(greek_name)

        if surface_type == "DTE":
            dte_range = np.linspace(1, 90, 100)
            matrix_3d = calculate_portfolio_greeks_3d_dte(
                strikes, option_types, position_types, quantities,
                spot_range, dte_range, risk_free_rate, 0.25
            )
            return spot_range, dte_range, matrix_3d[:, :, greek_idx].T, "DTE (days)"
        else:
            iv_range = np.linspace(0.05, 0.50, 100)
            matrix_3d = calculate_portfolio_greeks_3d_iv(
                strikes, option_types, position_types, quantities,
                spot_range, 30.0, risk_free_rate, iv_range
            )
            return spot_range, iv_range * 100, matrix_3d[:, :, greek_idx].T, "IV (%)"

    with st.spinner(f'Calculating 3D surface for {selected_greek}...'):
        X, Y, Z, y_label = calculate_3d_surface(portfolio_json, selected_greek, surface_type, risk_free_rate)

    # Create 3D figure with custom hover template
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title=selected_greek.capitalize(), thickness=15, len=0.7, x=1.02),
        contours={"z": {"show": True, "usecolormap": True, "project": {"z": False}}},
        hovertemplate=(
            f'<b>Underlying Price</b>: %{{x:.2f}}<br>' +
            f'<b>{y_label}</b>: %{{y:.2f}}<br>' +
            f'<b>{GREEK_TITLES.get(selected_greek, selected_greek.capitalize())}</b>: %{{z:.4f}}<br>' +
            '<extra></extra>'
        )
    )])

    fig.update_layout(
        title=f"3D Surface: {selected_greek.capitalize()} vs Price and {surface_type}",
        scene=dict(
            xaxis=dict(title='Underlying Price', gridcolor='lightgray', showbackground=True),
            yaxis=dict(title=y_label, gridcolor='lightgray', showbackground=True),
            zaxis=dict(title=selected_greek.capitalize(), gridcolor='lightgray', showbackground=True),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            aspectratio=dict(x=1, y=1, z=0.7)
        ),
        height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # JavaScript to scroll to the 3D chart anchor after page load
    st.markdown("""
    <script>
        // Wait for the page to fully load, then scroll to 3D chart
        setTimeout(function() {
            const element = document.getElementById('3d-chart-anchor');
            if (element) {
                element.scrollIntoView({behavior: 'smooth', block: 'start'});
                // Small offset to account for any headers
                window.scrollBy(0, -100);
            }
        }, 500);
    </script>
    """, unsafe_allow_html=True)

    # Info boxes
    col1, col2 = st.columns(2)
    with col1:
        option_count = len(st.session_state.positions) if st.session_state.positions else (0 if st.session_state.stock_position else 1)
        stock_count = 1 if st.session_state.stock_position else 0
        st.info(f"""
        📊 **Current Parameters:**
        - Greek: **{selected_greek.capitalize()}**
        - Y-axis: **{y_label}**
        - Option Positions: **{option_count}**
        - Stock Position: **{stock_count}**
        """)
    with col2:
        st.success("""
        💡 **3D Navigation:**
        - **Rotate**: Left click + drag
        - **Zoom**: Scroll wheel
        - **Pan**: Middle click + drag
        """)

# ============= TAB 6: GUIDE =============
with tab6:
    render_guide_section()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>📊 Options Greeks Explorer | Black-Scholes Model | Thomas Vaudescal</small>
</div>
""", unsafe_allow_html=True)