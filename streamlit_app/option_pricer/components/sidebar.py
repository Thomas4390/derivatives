"""
Sidebar component for Options Greeks Explorer.

Modern, clean design with improved user experience.
Strategy-first approach with editable legs.
"""

import streamlit as st
from config.constants import (
    CONTRACT_MULTIPLIER,
    DEFAULT_SPOT_PRICE,
    DEFAULT_RISK_FREE_RATE,
    AVAILABLE_STRATEGIES,
    STRATEGY_DISPLAY_NAMES,
    STRATEGY_LEGS,
    STRATEGIES_WITH_STOCK
)
from config.styles import (
    net_position_card_html,
    position_item_html,
    stock_position_html
)


def render_sidebar(
    positions: list,
    stock_position,
    portfolio_class,
    option_position_class,
    stock_position_class
) -> tuple[float, float]:
    """
    Render the complete sidebar with all controls.
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
        _render_strategy_builder(
            spot_price,
            risk_free_rate,
            portfolio_class,
            option_position_class,
            stock_position_class
        )

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
    risk_free_rate: float,
    portfolio_class,
    option_position_class,
    stock_position_class
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
    selected_strategy = st.selectbox(
        "Strategy",
        all_options,
        format_func=format_strategy,
        key="strategy_selector",
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
    strategy_legs = STRATEGY_LEGS.get(selected_strategy, [])
    has_stock = selected_strategy in STRATEGIES_WITH_STOCK

    # Initialize session state
    _initialize_strategy_state(selected_strategy, strategy_legs, has_stock, spot_price)

    # Strategy info header
    strategy_name = STRATEGY_DISPLAY_NAMES.get(selected_strategy, selected_strategy)
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

    # Render stock position if needed
    if has_stock:
        stock_cost = _render_stock_leg_editor(spot_price)
        total_net_cost -= stock_cost  # Stock is typically a debit
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    # Render each option leg with improved styling
    leg_costs = []
    for i, leg in enumerate(strategy_legs):
        leg_cost = _render_leg_editor(
            i, leg, spot_price, risk_free_rate,
            portfolio_class, option_position_class,
            num_legs
        )
        leg_costs.append(leg_cost)

        leg_state = st.session_state.strategy_legs_state.get(i, {})
        if leg_state.get('position_type') == 'long':
            total_net_cost -= leg_cost
        else:
            total_net_cost += leg_cost

    # Summary section
    _render_strategy_summary(total_net_cost, has_stock)

    # Apply button
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    if st.button("✓  Apply Strategy", use_container_width=True, type="primary", key="apply_strategy_btn"):
        _apply_strategy(
            spot_price, risk_free_rate, strategy_legs, has_stock,
            portfolio_class, option_position_class, stock_position_class
        )


def _initialize_strategy_state(selected_strategy: str, strategy_legs: list, has_stock: bool, spot_price: float) -> None:
    """Initialize or reset strategy state when strategy changes."""
    if 'strategy_legs_state' not in st.session_state:
        st.session_state.strategy_legs_state = {}

    if st.session_state.get('last_selected_strategy') != selected_strategy:
        st.session_state.last_selected_strategy = selected_strategy
        st.session_state.strategy_legs_state = {}

        for i, leg in enumerate(strategy_legs):
            st.session_state.strategy_legs_state[i] = {
                'option_type': leg['option_type'],
                'position_type': leg['position_type'],
                'strike': round(spot_price * leg['strike_factor'], 2),
                'quantity': leg['quantity']
            }

        if has_stock:
            st.session_state.strategy_legs_state['stock'] = {
                'position_type': 'long',
                'quantity': 100,
                'entry_price': spot_price
            }


def _render_leg_editor(
    leg_index: int,
    leg_config: dict,
    spot_price: float,
    risk_free_rate: float,
    portfolio_class,
    option_position_class,
    total_legs: int
) -> float:
    """Render an editable leg configuration. Returns the total cost for this leg."""
    leg_state = st.session_state.strategy_legs_state.get(leg_index, {})

    option_type = leg_state.get('option_type', leg_config['option_type'])
    position_type = leg_state.get('position_type', leg_config['position_type'])
    is_long = position_type == 'long'

    # Visual styling
    border_color = "#10b981" if is_long else "#ef4444"
    accent_color = "#059669" if is_long else "#dc2626"
    bg_gradient = "linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%)" if is_long else "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"
    position_badge_bg = "#d1fae5" if is_long else "#fee2e2"
    position_badge_color = "#047857" if is_long else "#b91c1c"

    # Leg container
    st.markdown(f"""
    <div style="background: {bg_gradient}; border: 1px solid {border_color}40; border-left: 4px solid {border_color}; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.625rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.625rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 0.7rem; font-weight: 700; color: #475569; text-transform: uppercase;">Leg {leg_index + 1}</span>
            </div>
            <span style="background: {position_badge_bg}; color: {position_badge_color}; font-size: 0.65rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; text-transform: uppercase;">{position_type}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Move inputs outside the HTML div
    col1, col2 = st.columns(2)

    with col1:
        new_option_type = st.selectbox(
            "Type",
            ["call", "put"],
            index=0 if option_type == "call" else 1,
            format_func=lambda x: f"{'📈' if x == 'call' else '📉'} {x.upper()}",
            key=f"leg_{leg_index}_type"
        )

    with col2:
        new_position_type = st.selectbox(
            "Direction",
            ["long", "short"],
            index=0 if position_type == "long" else 1,
            format_func=lambda x: f"{'🟢' if x == 'long' else '🔴'} {x.upper()}",
            key=f"leg_{leg_index}_dir"
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
            key=f"leg_{leg_index}_strike"
        )

    with col4:
        default_qty = leg_state.get('quantity', leg_config['quantity'])
        new_quantity = st.number_input(
            "Contracts",
            value=int(default_qty),
            min_value=1,
            step=1,
            key=f"leg_{leg_index}_qty"
        )

    # Update state
    st.session_state.strategy_legs_state[leg_index] = {
        'option_type': new_option_type,
        'position_type': new_position_type,
        'strike': new_strike,
        'quantity': new_quantity
    }

    # Calculate premium
    portfolio_temp = portfolio_class(spot_price, risk_free_rate)
    position_temp = option_position_class(new_option_type, new_position_type, new_strike, new_quantity)
    portfolio_temp.add_option_position(position_temp)
    premium = position_temp.premium_paid
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

    return total_cost


def _render_stock_leg_editor(spot_price: float) -> float:
    """Render stock position editor. Returns the stock cost."""
    stock_state = st.session_state.strategy_legs_state.get('stock', {
        'position_type': 'long',
        'quantity': 100,
        'entry_price': spot_price
    })

    st.markdown("""
    <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 1px solid #3b82f640; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.625rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.625rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 0.7rem; font-weight: 700; color: #475569; text-transform: uppercase;">Stock Position</span>
            </div>
            <span style="background: #dbeafe; color: #1d4ed8; font-size: 0.65rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; text-transform: uppercase;">Underlying</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        stock_direction = st.selectbox(
            "Direction",
            ["long", "short"],
            index=0 if stock_state.get('position_type', 'long') == 'long' else 1,
            format_func=lambda x: f"{'🟢' if x == 'long' else '🔴'} {x.upper()}",
            key="stock_leg_dir"
        )

    with col2:
        stock_qty = st.number_input(
            "Shares",
            value=int(stock_state.get('quantity', 100)),
            min_value=1,
            step=100,
            key="stock_leg_qty"
        )

    stock_entry = st.number_input(
        "Entry Price ($)",
        value=float(stock_state.get('entry_price', spot_price)),
        step=1.0,
        format="%.2f",
        key="stock_leg_entry"
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

    return stock_cost


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
    has_stock: bool,
    portfolio_class,
    option_position_class,
    stock_position_class
) -> None:
    """Apply the configured strategy to positions."""
    new_positions = []

    for i, _ in enumerate(strategy_legs):
        leg_state = st.session_state.strategy_legs_state.get(i, {})
        if leg_state:
            portfolio_temp = portfolio_class(spot_price, risk_free_rate)
            position_temp = option_position_class(
                leg_state['option_type'],
                leg_state['position_type'],
                leg_state['strike'],
                leg_state['quantity']
            )
            portfolio_temp.add_option_position(position_temp)
            premium = position_temp.premium_paid

            position = option_position_class(
                leg_state['option_type'],
                leg_state['position_type'],
                leg_state['strike'],
                leg_state['quantity'],
                premium
            )
            new_positions.append(position)

    new_stock_position = None
    if has_stock:
        stock_state = st.session_state.strategy_legs_state.get('stock')
        if stock_state:
            new_stock_position = stock_position_class(
                stock_state['position_type'],
                stock_state['quantity'],
                stock_state['entry_price']
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

    # Stock position
    if stock_position:
        stock_cost = stock_position.entry_price * stock_position.quantity
        st.markdown(
            stock_position_html(
                quantity=stock_position.quantity,
                position_type=stock_position.position_type,
                entry_price=stock_position.entry_price,
                stock_cost=stock_cost
            ),
            unsafe_allow_html=True
        )

    # Option positions
    for i, pos in enumerate(positions):
        total_amount = pos.premium_paid * pos.quantity * CONTRACT_MULTIPLIER
        shares_controlled = pos.quantity * CONTRACT_MULTIPLIER

        st.markdown(
            position_item_html(
                index=i + 1,
                quantity=pos.quantity,
                position_type=pos.position_type,
                option_type=pos.option_type,
                strike=pos.strike,
                premium=pos.premium_paid,
                total_amount=total_amount,
                shares_controlled=shares_controlled,
                is_long=(pos.position_type == 'long')
            ),
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    if st.button("🗑️  Clear All", use_container_width=True, key="clear_all_btn"):
        st.session_state.positions = []
        st.session_state.stock_position = None
        st.session_state.strategy_legs_state = {}
        if 'last_selected_strategy' in st.session_state:
            del st.session_state.last_selected_strategy
        st.rerun()


def _calculate_net_position(positions: list, stock_position=None) -> float:
    """Calculate the net debit/credit position including stock."""
    net = sum(
        -pos.premium_paid * pos.quantity * CONTRACT_MULTIPLIER
        if pos.position_type == 'long'
        else pos.premium_paid * pos.quantity * CONTRACT_MULTIPLIER
        for pos in positions
    )

    if stock_position:
        if stock_position.position_type == 'long':
            net -= stock_position.entry_price * stock_position.quantity
        else:
            net += stock_position.entry_price * stock_position.quantity

    return net
