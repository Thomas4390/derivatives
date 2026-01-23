"""
Sidebar component for Options Greeks Explorer.

Modern, clean design with improved user experience.
Strategy-first approach with editable legs.
Uses dict-based positions for simplicity.
"""

import streamlit as st
from config.constants import (
    CONTRACT_MULTIPLIER,
    DEFAULT_SPOT_PRICE,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_DTE,
    DEFAULT_IV,
    AVAILABLE_STRATEGIES,
    STRATEGY_DISPLAY_NAMES,
    STRATEGY_LEGS,
    STRATEGIES_WITH_STOCK,
    STRATEGY_STOCK_POSITION
)
from config.styles import (
    net_position_card_html,
    position_item_html,
    stock_position_html
)
from services.pricing_adapter import calculate_option_premium
from services.state_manager import create_option_position, create_stock_position


def render_sidebar(
    positions: list,
    stock_position
) -> tuple[float, float]:
    """
    Render the complete sidebar with all controls.

    Args:
        positions: List of option position dicts
        stock_position: Stock position dict or None

    Returns:
        Tuple of (spot_price, risk_free_rate)
    """
    with st.sidebar:
        # Logo/Brand section
        st.markdown("""
        <div style="text-align: center; padding: 0.75rem 0 1.25rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.25rem;">
            <div style="font-size: 1.75rem; margin-bottom: 0.25rem;">📊</div>
            <div style="font-size: 1rem; font-weight: 600; color: #1a365d;">Options Greeks Explorer</div>
            <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 0.15rem;">Black-Scholes Model</div>
        </div>
        """, unsafe_allow_html=True)

        # Market Parameters
        spot_price, risk_free_rate = _render_market_params()

        # Strategy Builder
        _render_strategy_builder(spot_price, risk_free_rate)

        # Current Positions
        _render_positions_section(positions, stock_position)

    return spot_price, risk_free_rate


def _render_market_params() -> tuple[float, float]:
    """Render market parameters inputs."""
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
        <span style="font-size: 1rem;">🌐</span>
        <span style="font-size: 0.75rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em;">Market Parameters</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        spot_price = st.number_input(
            "Spot Price ($)",
            value=DEFAULT_SPOT_PRICE,
            step=1.0,
            format="%.2f",
            help="Current underlying price"
        )

    with col2:
        risk_free_rate = st.number_input(
            "Risk-Free Rate",
            value=DEFAULT_RISK_FREE_RATE,
            step=0.01,
            format="%.3f",
            help="Annual risk-free rate"
        )

    st.markdown("<div style='height: 0.75rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    return spot_price, risk_free_rate


def _render_strategy_builder(
    spot_price: float,
    risk_free_rate: float
) -> None:
    """Render the strategy builder section with editable legs."""

    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
        <span style="font-size: 1rem;">🎯</span>
        <span style="font-size: 0.75rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em;">Strategy Builder</span>
    </div>
    """, unsafe_allow_html=True)

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
    selector_version = st.session_state.get('selector_version', 0)
    selected_strategy = st.selectbox(
        "Strategy",
        all_options,
        format_func=format_strategy,
        key=f"strategy_selector_v{selector_version}",
        label_visibility="collapsed"
    )

    if not selected_strategy:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem; color: #94a3b8; font-size: 0.85rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 10px; border: 1px dashed #cbd5e1; margin-top: 0.5rem;">
            <div style="font-size: 2rem; margin-bottom: 0.75rem; opacity: 0.7;">📈</div>
            <div style="font-weight: 500; color: #64748b;">Select a strategy to begin</div>
            <div style="font-size: 0.75rem; margin-top: 0.5rem; color: #94a3b8;">Choose from vanilla options, spreads,<br/>straddles, and more complex strategies</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Get strategy configuration
    is_custom = selected_strategy == "custom"
    base_strategy_legs = STRATEGY_LEGS.get(selected_strategy, [])

    # Initialize session states if needed
    if 'custom_legs' not in st.session_state:
        st.session_state.custom_legs = []
    if 'additional_legs' not in st.session_state:
        st.session_state.additional_legs = {}
    if 'additional_stock' not in st.session_state:
        st.session_state.additional_stock = {}
    if 'custom_has_stock' not in st.session_state:
        st.session_state.custom_has_stock = False

    # Check if strategy changed - if so, reset additional state BEFORE calculating has_stock/strategy_legs
    strategy_changed = st.session_state.get('last_selected_strategy') != selected_strategy
    if strategy_changed:
        st.session_state.additional_legs = {}
        st.session_state.additional_stock = {}
        st.session_state.custom_legs = []
        st.session_state.custom_has_stock = False

    # Now calculate has_stock and strategy_legs with clean state
    has_stock = selected_strategy in STRATEGIES_WITH_STOCK

    if is_custom:
        strategy_legs = st.session_state.custom_legs
        has_stock = st.session_state.get('custom_has_stock', False)
    else:
        # Combine base strategy legs with any additional legs
        additional_legs_key = selected_strategy
        additional = st.session_state.additional_legs.get(additional_legs_key, [])
        strategy_legs = list(base_strategy_legs) + additional
        # Check if stock was added manually to this strategy
        has_stock = has_stock or st.session_state.additional_stock.get(additional_legs_key, False)

    # Initialize session state and check if auto-apply is needed
    should_auto_apply = _initialize_strategy_state(
        selected_strategy, strategy_legs, has_stock, spot_price, risk_free_rate
    )

    # Auto-apply strategy when newly selected or spot price changes
    if should_auto_apply and not is_custom:
        _apply_strategy(spot_price, risk_free_rate, strategy_legs, has_stock)

    # Strategy info header
    strategy_name = STRATEGY_DISPLAY_NAMES.get(selected_strategy, "Custom Strategy" if is_custom else selected_strategy)
    num_legs = len(strategy_legs)

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%); padding: 0.875rem 1rem; border-radius: 10px; margin: 0.75rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-weight: 600; color: #ffffff; font-size: 0.95rem;">{strategy_name}</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.75rem; margin-top: 0.2rem;">
                    {num_legs} leg{'s' if num_legs > 1 else ''}{' + 100 shares' if has_stock else ''}
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.15); padding: 0.35rem 0.65rem; border-radius: 6px;">
                <span style="color: #fbbf24; font-size: 0.7rem; font-weight: 600; text-transform: uppercase;">{'Multi-Leg' if num_legs > 1 else 'Single'}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Track total cost for summary
    total_net_cost = 0.0
    stock_should_remove = False

    # Determine if stock is removable (added manually, not part of base strategy)
    base_has_stock = selected_strategy in STRATEGIES_WITH_STOCK
    stock_is_removable = is_custom or (has_stock and not base_has_stock)

    # Render stock position if needed
    if has_stock:
        stock_cost, stock_should_remove = _render_stock_leg_editor(
            spot_price, is_custom, selected_strategy, is_removable=stock_is_removable
        )
        # Handle stock cost based on position type
        stock_state = st.session_state.strategy_legs_state.get('stock', {})
        if stock_state.get('position_type', 'long') == 'long':
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
            if 'additional_stock' in st.session_state:
                st.session_state.additional_stock[additional_legs_key] = False
        if 'stock' in st.session_state.strategy_legs_state:
            del st.session_state.strategy_legs_state['stock']
        st.session_state.strategy_version = st.session_state.get('strategy_version', 0) + 1
        st.rerun()

    # Render each option leg with improved styling
    leg_costs = []
    legs_to_remove = []
    base_leg_count = len(base_strategy_legs) if not is_custom else 0

    for i, leg in enumerate(strategy_legs):
        # Allow removal for custom legs OR additional legs on predefined strategies
        is_additional_leg = i >= base_leg_count
        allow_remove = is_custom or is_additional_leg

        leg_cost, should_remove = _render_leg_editor(
            i, leg, spot_price, risk_free_rate, num_legs,
            allow_remove=allow_remove,
            is_additional=is_additional_leg and not is_custom
        )
        leg_costs.append(leg_cost)

        if should_remove:
            legs_to_remove.append(i)

        leg_state = st.session_state.strategy_legs_state.get(i, {})
        if leg_state.get('position_type') == 'long':
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
                # Remove from additional legs (adjust index for base legs)
                additional_idx = idx - base_leg_count
                additional_legs_key = selected_strategy
                if additional_idx >= 0 and additional_idx < len(st.session_state.additional_legs.get(additional_legs_key, [])):
                    st.session_state.additional_legs[additional_legs_key].pop(additional_idx)
        # Reset the legs state
        st.session_state.strategy_legs_state = {}
        st.session_state.strategy_version = st.session_state.get('strategy_version', 0) + 1
        st.rerun()

    # Add Leg button for all strategies
    _render_add_leg_button(spot_price, is_custom, selected_strategy, strategy_legs, has_stock)

    # Summary section
    _render_strategy_summary(total_net_cost, has_stock)

    # Apply button (for manual modifications)
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    if st.button("✓  Apply Changes", width="stretch", type="primary", key="apply_strategy_btn"):
        # Read values directly from widget keys for each leg to ensure we have the latest values
        version = st.session_state.get('strategy_version', 0)
        for i in range(len(strategy_legs)):
            type_key = f"leg_{i}_type_v{version}"
            dir_key = f"leg_{i}_dir_v{version}"
            strike_key = f"leg_{i}_strike_v{version}"
            qty_key = f"leg_{i}_qty_v{version}"

            # Update strategy_legs_state from widget values if they exist
            if type_key in st.session_state:
                st.session_state.strategy_legs_state[i] = {
                    'option_type': st.session_state[type_key],
                    'position_type': st.session_state[dir_key],
                    'strike': st.session_state[strike_key],
                    'quantity': st.session_state[qty_key]
                }

        # Also update stock state from widget values
        if has_stock:
            stock_dir_key = f"stock_leg_dir_v{version}"
            stock_qty_key = f"stock_leg_qty_v{version}"
            stock_entry_key = f"stock_leg_entry_v{version}"

            if stock_dir_key in st.session_state:
                st.session_state.strategy_legs_state['stock'] = {
                    'position_type': st.session_state[stock_dir_key],
                    'quantity': st.session_state[stock_qty_key],
                    'entry_price': st.session_state[stock_entry_key]
                }

        _apply_strategy(spot_price, risk_free_rate, strategy_legs, has_stock)


def _initialize_strategy_state(
    selected_strategy: str,
    strategy_legs: list,
    has_stock: bool,
    spot_price: float,
    risk_free_rate: float = 0.05
) -> bool:
    """
    Initialize or reset strategy state when strategy or spot price changes.

    Returns:
        bool: True if the strategy should be auto-applied (new selection or spot change)
    """
    if 'strategy_legs_state' not in st.session_state:
        st.session_state.strategy_legs_state = {}

    if 'strategy_version' not in st.session_state:
        st.session_state.strategy_version = 0

    if 'last_spot_price' not in st.session_state:
        st.session_state.last_spot_price = spot_price

    strategy_changed = st.session_state.get('last_selected_strategy') != selected_strategy
    spot_changed = st.session_state.get('last_spot_price') != spot_price

    should_auto_apply = False

    if strategy_changed or spot_changed:
        st.session_state.last_selected_strategy = selected_strategy
        st.session_state.last_spot_price = spot_price
        st.session_state.strategy_legs_state = {}

        # Increment version to force widget key changes and reset values
        st.session_state.strategy_version = st.session_state.get('strategy_version', 0) + 1

        # Clear old widget keys from session state to force reset
        keys_to_remove = [key for key in list(st.session_state.keys())
                         if key.startswith('leg_') or key.startswith('stock_leg_')]
        for key in keys_to_remove:
            del st.session_state[key]

        for i, leg in enumerate(strategy_legs):
            st.session_state.strategy_legs_state[i] = {
                'option_type': leg['option_type'],
                'position_type': leg['position_type'],
                'strike': round(spot_price * leg['strike_factor'], 2),
                'quantity': leg['quantity']
            }

        if has_stock:
            # Get the stock position type from strategy definition (default to 'long')
            stock_position_type = STRATEGY_STOCK_POSITION.get(selected_strategy, 'long')
            st.session_state.strategy_legs_state['stock'] = {
                'position_type': stock_position_type,
                'quantity': 100,
                'entry_price': spot_price
            }

        # Auto-apply strategy when newly selected or spot price changes
        should_auto_apply = True

    return should_auto_apply


def _render_add_leg_button(spot_price: float, is_custom: bool, selected_strategy: str, strategy_legs: list, has_stock: bool) -> None:
    """Render the Add Leg buttons for all strategies."""
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    # Initialize additional legs tracking
    if 'additional_legs' not in st.session_state:
        st.session_state.additional_legs = {}

    # Get additional legs for current strategy
    additional_legs_key = selected_strategy if not is_custom else 'custom'
    if additional_legs_key not in st.session_state.additional_legs:
        st.session_state.additional_legs[additional_legs_key] = []

    # Two buttons: Add Option and Add Stock (always show both for consistency)
    col1, col2 = st.columns(2)
    with col1:
        add_option_clicked = st.button("➕ Option", width="stretch", key="add_option_leg_btn", type="secondary")
    with col2:
        # Disable stock button if already has stock
        add_stock_clicked = st.button(
            "➕ Stock",
            width="stretch",
            key="add_stock_leg_btn",
            type="secondary",
            disabled=has_stock
        )

    if add_option_clicked:
        # Add a new default option leg
        new_leg = {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1
        }

        if is_custom:
            st.session_state.custom_legs.append(new_leg)
            new_index = len(st.session_state.custom_legs) - 1
        else:
            st.session_state.additional_legs[additional_legs_key].append(new_leg)
            # The new index is the base strategy legs + additional legs
            new_index = len(STRATEGY_LEGS.get(selected_strategy, [])) + len(st.session_state.additional_legs[additional_legs_key]) - 1

        # Initialize state for the new leg
        st.session_state.strategy_legs_state[new_index] = {
            'option_type': 'call',
            'position_type': 'long',
            'strike': round(spot_price, 2),
            'quantity': 1
        }
        st.session_state.strategy_version = st.session_state.get('strategy_version', 0) + 1
        st.rerun()

    if add_stock_clicked:
        # Add stock position
        if is_custom:
            st.session_state.custom_has_stock = True
        else:
            # For predefined strategies, we track it differently
            if 'additional_stock' not in st.session_state:
                st.session_state.additional_stock = {}
            st.session_state.additional_stock[additional_legs_key] = True

        # Initialize stock state
        st.session_state.strategy_legs_state['stock'] = {
            'position_type': 'long',
            'quantity': 100,
            'entry_price': spot_price
        }
        st.session_state.strategy_version = st.session_state.get('strategy_version', 0) + 1
        st.rerun()

    # Show hint when no legs (only for custom with no legs and no stock)
    if is_custom and len(st.session_state.custom_legs) == 0 and not st.session_state.get('custom_has_stock', False):
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 1rem; color: #94a3b8; font-size: 0.85rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 10px; border: 1px dashed #cbd5e1; margin-top: 0.5rem;">
            <div style="font-size: 1.5rem; margin-bottom: 0.75rem; opacity: 0.7;">🎨</div>
            <div style="font-weight: 500; color: #64748b;">Build your custom strategy</div>
            <div style="font-size: 0.75rem; margin-top: 0.5rem; color: #94a3b8;">Add options or stock positions</div>
        </div>
        """, unsafe_allow_html=True)


def _render_leg_editor(
    leg_index: int,
    leg_config: dict,
    spot_price: float,
    risk_free_rate: float,
    total_legs: int,
    allow_remove: bool = False,
    is_additional: bool = False
) -> tuple[float, bool]:
    """Render an editable leg configuration. Returns (total_cost, should_remove)."""
    leg_state = st.session_state.strategy_legs_state.get(leg_index, {})
    should_remove = False

    option_type = leg_state.get('option_type', leg_config['option_type'])
    position_type = leg_state.get('position_type', leg_config['position_type'])
    is_long = position_type == 'long'

    # Visual styling
    border_color = "#10b981" if is_long else "#ef4444"
    accent_color = "#059669" if is_long else "#dc2626"
    bg_gradient = "linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%)" if is_long else "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"
    position_badge_bg = "#d1fae5" if is_long else "#fee2e2"
    position_badge_color = "#047857" if is_long else "#b91c1c"

    # Leg container with optional remove button
    version = st.session_state.get('strategy_version', 0)

    # Label for the leg
    leg_label = f"Leg {leg_index + 1}"
    if is_additional:
        leg_label = f"+ Leg {leg_index + 1}"
        # Use a slightly different style for additional legs
        bg_gradient = "linear-gradient(135deg, #fefce8 0%, #fef9c3 100%)" if is_long else "linear-gradient(135deg, #fef2f2 0%, #fecaca 100%)"

    # Build the added badge HTML separately
    added_badge = '<span style="background: #fef3c7; color: #92400e; font-size: 0.55rem; font-weight: 600; padding: 0.15rem 0.35rem; border-radius: 3px; margin-left: 0.25rem;">ADDED</span>' if is_additional else ''

    # Build the leg header HTML
    leg_header_html = f'''<div style="background: {bg_gradient}; border: 1px solid {border_color}40; border-left: 4px solid {border_color}; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.625rem;"><div style="display: flex; justify-content: space-between; align-items: center;"><div style="display: flex; align-items: center; gap: 0.5rem;"><span style="font-size: 0.7rem; font-weight: 700; color: #475569; text-transform: uppercase;">{leg_label}</span>{added_badge}</div><span style="background: {position_badge_bg}; color: {position_badge_color}; font-size: 0.65rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; text-transform: uppercase;">{position_type}</span></div></div>'''

    if allow_remove:
        # Header with remove button integrated
        header_col1, header_col2 = st.columns([4, 1])
        with header_col1:
            st.markdown(leg_header_html, unsafe_allow_html=True)
        with header_col2:
            st.markdown("<div style='height: 0.25rem'></div>", unsafe_allow_html=True)
            if st.button("🗑️", key=f"remove_leg_{leg_index}_v{version}", help="Remove this leg", width="stretch"):
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
            key=f"leg_{leg_index}_type_v{version}"
        )

    with col2:
        new_position_type = st.selectbox(
            "Direction",
            ["long", "short"],
            index=0 if position_type == "long" else 1,
            format_func=lambda x: f"{'🟢' if x == 'long' else '🔴'} {x.upper()}",
            key=f"leg_{leg_index}_dir_v{version}"
        )

    # Strike and quantity inputs
    default_strike = leg_state.get('strike', round(spot_price * leg_config['strike_factor'], 2))

    col3, col4 = st.columns(2)

    with col3:
        new_strike = st.number_input(
            "Strike ($)",
            value=float(default_strike),
            step=1.0,
            format="%.2f",
            key=f"leg_{leg_index}_strike_v{version}"
        )

    with col4:
        default_qty = leg_state.get('quantity', leg_config['quantity'])
        new_quantity = st.number_input(
            "Contracts",
            value=int(default_qty),
            min_value=1,
            step=1,
            key=f"leg_{leg_index}_qty_v{version}"
        )

    # Update state
    st.session_state.strategy_legs_state[leg_index] = {
        'option_type': new_option_type,
        'position_type': new_position_type,
        'strike': new_strike,
        'quantity': new_quantity
    }

    # Calculate premium using Black-Scholes
    premium = calculate_option_premium(
        spot=spot_price,
        strike=new_strike,
        dte_days=DEFAULT_DTE,
        risk_free_rate=risk_free_rate,
        volatility=DEFAULT_IV / 100,  # Convert from percentage
        option_type=new_option_type
    )
    total_cost = premium * new_quantity * CONTRACT_MULTIPLIER

    # Cost display
    is_long_now = new_position_type == 'long'
    cost_label = "Debit" if is_long_now else "Credit"
    cost_color = "#dc2626" if is_long_now else "#059669"
    cost_prefix = "-" if is_long_now else "+"

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0.75rem; background: #ffffff; border-radius: 6px; border: 1px solid #e2e8f0; margin-top: -0.5rem; margin-bottom: 0.5rem;">
        <span style="color: #64748b; font-size: 0.8rem;">
            Premium: <span style="font-family: 'JetBrains Mono', monospace; font-weight: 500;">${premium:.2f}</span>
        </span>
        <span style="color: {cost_color}; font-weight: 700; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">
            {cost_prefix}${total_cost:,.2f}
        </span>
    </div>
    """, unsafe_allow_html=True)

    return total_cost, should_remove


def _render_stock_leg_editor(spot_price: float, is_custom: bool, selected_strategy: str, is_removable: bool = False) -> tuple[float, bool]:
    """Render stock position editor. Returns (stock_cost, should_remove)."""
    stock_state = st.session_state.strategy_legs_state.get('stock', {
        'position_type': 'long',
        'quantity': 100,
        'entry_price': spot_price
    })

    # Get strategy version for unique widget keys
    version = st.session_state.get('strategy_version', 0)
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
            if st.button("🗑️", key=f"remove_stock_v{version}", help="Remove stock position", width="stretch"):
                should_remove = True
    else:
        st.markdown(header_html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        stock_direction = st.selectbox(
            "Direction",
            ["long", "short"],
            index=0 if stock_state.get('position_type', 'long') == 'long' else 1,
            format_func=lambda x: f"{'🟢' if x == 'long' else '🔴'} {x.upper()}",
            key=f"stock_leg_dir_v{version}"
        )

    with col2:
        stock_qty = st.number_input(
            "Shares",
            value=int(stock_state.get('quantity', 100)),
            min_value=1,
            step=100,
            key=f"stock_leg_qty_v{version}"
        )

    stock_entry = st.number_input(
        "Entry Price ($)",
        value=float(stock_state.get('entry_price', spot_price)),
        step=1.0,
        format="%.2f",
        key=f"stock_leg_entry_v{version}"
    )

    st.session_state.strategy_legs_state['stock'] = {
        'position_type': stock_direction,
        'quantity': stock_qty,
        'entry_price': stock_entry
    }

    stock_cost = stock_entry * stock_qty
    is_long = stock_direction == "long"
    cost_label = "Cost" if is_long else "Credit"
    cost_color = "#dc2626" if is_long else "#059669"
    cost_prefix = "-" if is_long else "+"

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0.75rem; background: #ffffff; border-radius: 6px; border: 1px solid #e2e8f0; margin-top: -0.5rem;">
        <span style="color: #64748b; font-size: 0.8rem;">
            {stock_qty:,} shares @ ${stock_entry:.2f}
        </span>
        <span style="color: {cost_color}; font-weight: 700; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">
            {cost_prefix}${stock_cost:,.2f}
        </span>
    </div>
    """, unsafe_allow_html=True)

    return stock_cost, should_remove


def _render_strategy_summary(total_net_cost: float, has_stock: bool) -> None:
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

    st.markdown(f"""
    <div style="background: {summary_bg}; border: 1px solid {summary_border}; border-radius: 10px; padding: 1rem; margin-top: 0.75rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.25rem;">{summary_icon}</span>
                <div>
                    <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 600;">{summary_label}</div>
                    <div style="font-size: 0.7rem; color: #94a3b8;">{'Incl. stock' if has_stock else 'Options only'}</div>
                </div>
            </div>
            <div style="font-size: 1.35rem; font-weight: 700; color: {summary_color}; font-family: 'JetBrains Mono', monospace;">
                {display_amount}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _apply_strategy(
    spot_price: float,
    risk_free_rate: float,
    strategy_legs: list,
    has_stock: bool
) -> None:
    """Apply the configured strategy to positions using dict-based storage."""
    new_positions = []

    for i, _ in enumerate(strategy_legs):
        leg_state = st.session_state.strategy_legs_state.get(i, {})
        if leg_state:
            # Calculate premium using Black-Scholes
            premium = calculate_option_premium(
                spot=spot_price,
                strike=leg_state['strike'],
                dte_days=DEFAULT_DTE,
                risk_free_rate=risk_free_rate,
                volatility=DEFAULT_IV / 100,  # Convert from percentage
                option_type=leg_state['option_type']
            )

            # Create dict-based position
            position = create_option_position(
                option_type=leg_state['option_type'],
                position_type=leg_state['position_type'],
                strike=leg_state['strike'],
                quantity=leg_state['quantity'],
                premium_paid=premium
            )
            new_positions.append(position)

    new_stock_position = None
    if has_stock:
        stock_state = st.session_state.strategy_legs_state.get('stock')
        if stock_state:
            new_stock_position = create_stock_position(
                position_type=stock_state['position_type'],
                quantity=stock_state['quantity'],
                entry_price=stock_state['entry_price']
            )

    st.session_state.positions = new_positions
    st.session_state.stock_position = new_stock_position
    st.rerun()


def _render_positions_section(positions: list, stock_position) -> None:
    """Render the current positions section."""
    st.markdown("<div style='height: 0.5rem; border-top: 1px solid #e2e8f0; margin-top: 1rem; padding-top: 1rem;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
        <span style="font-size: 1rem;">📋</span>
        <span style="font-size: 0.75rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em;">Active Positions</span>
    </div>
    """, unsafe_allow_html=True)

    if not positions and not stock_position:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 1rem; color: #94a3b8; font-size: 0.85rem; background: #f8fafc; border-radius: 8px; border: 1px dashed #cbd5e1;">
            <div style="font-size: 1.25rem; margin-bottom: 0.5rem; opacity: 0.6;">∅</div>
            <div style="font-weight: 500; color: #64748b;">No active positions</div>
            <div style="font-size: 0.75rem; margin-top: 0.25rem;">Apply a strategy above to begin</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Net position
    net_amount = _calculate_net_position(positions, stock_position)
    st.markdown(net_position_card_html("", net_amount), unsafe_allow_html=True)

    # Stock position (now a dict)
    if stock_position:
        stock_cost = stock_position['entry_price'] * stock_position['quantity']
        st.markdown(
            stock_position_html(
                quantity=stock_position['quantity'],
                position_type=stock_position['position_type'],
                entry_price=stock_position['entry_price'],
                stock_cost=stock_cost
            ),
            unsafe_allow_html=True
        )

    # Option positions (now dicts)
    for i, pos in enumerate(positions):
        total_amount = pos['premium_paid'] * pos['quantity'] * CONTRACT_MULTIPLIER
        shares_controlled = pos['quantity'] * CONTRACT_MULTIPLIER

        st.markdown(
            position_item_html(
                index=i + 1,
                quantity=pos['quantity'],
                position_type=pos['position_type'],
                option_type=pos['option_type'],
                strike=pos['strike'],
                premium=pos['premium_paid'],
                total_amount=total_amount,
                shares_controlled=shares_controlled,
                is_long=(pos['position_type'] == 'long')
            ),
            unsafe_allow_html=True
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
        st.session_state.selector_version = st.session_state.get('selector_version', 0) + 1
        if 'last_selected_strategy' in st.session_state:
            del st.session_state.last_selected_strategy

        # Increment version to force widget reset
        st.session_state.strategy_version = st.session_state.get('strategy_version', 0) + 1

        # Clear widget keys
        keys_to_remove = [key for key in list(st.session_state.keys())
                         if key.startswith('leg_') or key.startswith('stock_leg_') or key.startswith('strategy_selector_v')]
        for key in keys_to_remove:
            del st.session_state[key]

        st.rerun()


def _calculate_net_position(positions: list, stock_position=None) -> float:
    """Calculate the net debit/credit position including stock (dict-based)."""
    net = sum(
        -pos['premium_paid'] * pos['quantity'] * CONTRACT_MULTIPLIER
        if pos['position_type'] == 'long'
        else pos['premium_paid'] * pos['quantity'] * CONTRACT_MULTIPLIER
        for pos in positions
    )

    if stock_position:
        if stock_position['position_type'] == 'long':
            net -= stock_position['entry_price'] * stock_position['quantity']
        else:
            net += stock_position['entry_price'] * stock_position['quantity']

    return net
