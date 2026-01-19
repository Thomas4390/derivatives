"""
Strategy Builder Wrapper for Monte Carlo P&L Simulation.

Provides a simplified strategy builder interface for the simulation app,
reusing constants from the option_pricer module for consistency.

Features:
- Horizontal card-based layout for option legs
- Collapsible mode for compact display
- Full edit mode in Configuration tab
"""

import streamlit as st
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# Import strategy definitions from option_pricer
from streamlit_app.option_pricer.config.constants import (
    STRATEGY_LEGS,
    STRATEGY_DISPLAY_NAMES,
    STRATEGIES_WITH_STOCK,
    STRATEGY_STOCK_POSITION,
    CONTRACT_MULTIPLIER
)

from services.state_manager import (
    get_strategy_expanded,
    set_strategy_expanded,
    toggle_strategy_expanded
)


# =============================================================================
# Data Classes for Positions
# =============================================================================

@dataclass
class SimulationOptionPosition:
    """Option position for P&L simulation."""
    option_type: str  # 'call' or 'put'
    position_type: str  # 'long' or 'short'
    strike: float
    quantity: int
    premium: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            'option_type': self.option_type,
            'position_type': self.position_type,
            'strike': self.strike,
            'quantity': self.quantity,
            'premium': self.premium
        }


@dataclass
class SimulationStockPosition:
    """Stock position for P&L simulation."""
    position_type: str  # 'long' or 'short'
    quantity: int
    entry_price: float


# =============================================================================
# Strategy Builder UI Component (Full Mode)
# =============================================================================

def render_strategy_builder(
    spot_price: float,
    risk_free_rate: float,
    time_to_expiry: float,
    volatility: float,
    bs_price_function,
    horizontal_layout: bool = True
) -> Tuple[List[SimulationOptionPosition], Optional[SimulationStockPosition]]:
    """
    Render the strategy builder component for P&L simulation.

    Parameters
    ----------
    spot_price : float
        Current underlying price
    risk_free_rate : float
        Annual risk-free rate
    time_to_expiry : float
        Time to expiration in years
    volatility : float
        Annualized volatility (decimal)
    bs_price_function : callable
        Black-Scholes pricing function(spot, strike, r, T, sigma, option_type)
    horizontal_layout : bool
        Use horizontal card layout (default True)

    Returns
    -------
    Tuple[List[SimulationOptionPosition], Optional[SimulationStockPosition]]
        List of option positions and optional stock position
    """
    # Strategy groups for dropdown
    strategy_groups = {
        "── Custom ──": None,
        "custom": "Custom Strategy",
        "── Directional ──": None,
        "long_call": "Long Call",
        "short_call": "Short Call",
        "long_put": "Long Put",
        "short_put": "Short Put",
        "── Vertical Spreads ──": None,
        "bull_call_spread": "Bull Call Spread",
        "bear_put_spread": "Bear Put Spread",
        "bull_put_spread": "Bull Put Spread",
        "bear_call_spread": "Bear Call Spread",
        "── Volatility ──": None,
        "long_straddle": "Long Straddle",
        "short_straddle": "Short Straddle",
        "long_strangle": "Long Strangle",
        "short_strangle": "Short Strangle",
        "── Advanced ──": None,
        "iron_condor": "Iron Condor",
        "butterfly": "Butterfly",
        "── Stock + Options ──": None,
        "covered_call": "Covered Call",
        "protective_put": "Protective Put",
        "collar": "Collar",
    }

    # Build options list
    all_options = [""] + [k for k in strategy_groups.keys() if not k.startswith("──")]

    def format_strategy(key):
        if key == "":
            return "Select a strategy..."
        if key.startswith("──"):
            return key
        return strategy_groups.get(key, key)

    # Strategy selector (always visible)
    selector_version = st.session_state.get('pnl_selector_version', 0)
    selected_strategy = st.selectbox(
        "Strategy",
        all_options,
        format_func=format_strategy,
        key=f"pnl_strategy_selector_v{selector_version}",
        label_visibility="collapsed"
    )

    if not selected_strategy:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 1rem; color: #94a3b8; font-size: 0.85rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 10px; border: 1px dashed #cbd5e1; margin-top: 0.5rem;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem; opacity: 0.7;">📈</div>
            <div style="font-weight: 500; color: #64748b;">Select a strategy to begin</div>
        </div>
        """, unsafe_allow_html=True)
        return [], None

    # Initialize state
    _initialize_pnl_strategy_state(selected_strategy, spot_price)

    # Get strategy configuration
    is_custom = selected_strategy == "custom"
    has_stock = selected_strategy in STRATEGIES_WITH_STOCK

    # Build positions from state
    positions, stock_position = _build_positions_from_state(
        selected_strategy, spot_price, risk_free_rate,
        time_to_expiry, volatility, bs_price_function, is_custom, has_stock
    )

    # Calculate total net cost
    total_net_cost = _calculate_total_net_cost(positions, stock_position)

    # Full strategy display
    _render_strategy_info(selected_strategy, positions, has_stock)

    if horizontal_layout:
        _render_positions_horizontal(positions, stock_position, is_custom, spot_price)
    else:
        _render_positions_vertical(positions, stock_position, is_custom, spot_price)

    # Strategy summary
    _render_strategy_summary(total_net_cost, has_stock)

    return positions, stock_position


def render_strategy_builder_compact(
    positions: List[SimulationOptionPosition],
    stock_position: Optional[SimulationStockPosition],
    strategy_name: str = "custom"
) -> None:
    """
    Render a compact read-only view of the strategy for analysis tabs.

    Parameters
    ----------
    positions : List[SimulationOptionPosition]
        Current option positions
    stock_position : Optional[SimulationStockPosition]
        Current stock position if any
    strategy_name : str
        Strategy name key
    """
    if not positions:
        st.info("No strategy configured. Configure positions in the Configuration tab.")
        return

    display_name = STRATEGY_DISPLAY_NAMES.get(strategy_name, "Custom Strategy")
    has_stock = stock_position is not None
    total_net_cost = _calculate_total_net_cost(positions, stock_position)

    # Compact header
    is_debit = total_net_cost < 0
    cost_display = f"-${abs(total_net_cost):,.0f}" if is_debit else f"+${abs(total_net_cost):,.0f}"
    cost_color = "#dc2626" if is_debit else "#059669"

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 0.75rem 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.1rem;">🎯</span>
                <div>
                    <div style="font-weight: 600; color: #f1f5f9; font-size: 0.9rem;">{display_name}</div>
                    <div style="color: #94a3b8; font-size: 0.7rem;">
                        {len(positions)} leg{'s' if len(positions) > 1 else ''}{' + stock' if has_stock else ''}
                    </div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="text-align: right;">
                    <div style="font-size: 0.65rem; color: #94a3b8; text-transform: uppercase;">Net Cost</div>
                    <div style="font-size: 1rem; font-weight: 700; color: {cost_color}; font-family: monospace;">{cost_display}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Horizontal cards for legs
    _render_leg_cards_readonly(positions, stock_position)


def _render_collapsed_summary(
    strategy: str,
    positions: list,
    stock_position: Optional[SimulationStockPosition],
    total_net_cost: float
) -> None:
    """Render collapsed summary view."""
    display_name = STRATEGY_DISPLAY_NAMES.get(strategy, "Custom Strategy")
    has_stock = stock_position is not None
    num_legs = len(positions)

    is_debit = total_net_cost < 0
    cost_display = f"-${abs(total_net_cost):,.0f}" if is_debit else f"+${abs(total_net_cost):,.0f}"
    cost_color = "#dc2626" if is_debit else "#059669"
    cost_label = "Debit" if is_debit else "Credit"

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.75rem 1rem; margin-top: 0.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
            <div>
                <div style="font-weight: 600; color: #1e293b; font-size: 0.9rem;">{display_name}</div>
                <div style="font-size: 0.7rem; color: #64748b;">
                    {num_legs} leg{'s' if num_legs > 1 else ''}{' + 100 shares' if has_stock else ''}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.6rem; color: #94a3b8; text-transform: uppercase;">{cost_label}</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {cost_color}; font-family: monospace;">{cost_display}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_positions_horizontal(
    positions: list,
    stock_position: Optional[SimulationStockPosition],
    is_custom: bool,
    spot_price: float
) -> None:
    """Render positions in horizontal card layout."""
    # Determine number of columns needed
    num_positions = len(positions) + (1 if stock_position else 0) + (1 if is_custom else 0)

    if num_positions == 0:
        if is_custom:
            _render_add_leg_button(spot_price)
        return

    # Create columns for each card
    cols = st.columns(min(num_positions, 4))
    col_idx = 0

    # Stock position card
    if stock_position:
        with cols[col_idx % len(cols)]:
            _render_stock_card(stock_position)
        col_idx += 1

    # Option position cards
    for i, pos in enumerate(positions):
        with cols[col_idx % len(cols)]:
            _render_position_card(i, pos)
        col_idx += 1

    # Add leg button for custom
    if is_custom:
        with cols[col_idx % len(cols)]:
            _render_add_leg_card(spot_price)


def _render_positions_vertical(
    positions: list,
    stock_position: Optional[SimulationStockPosition],
    is_custom: bool,
    spot_price: float
) -> None:
    """Render positions in vertical list layout (legacy)."""
    if stock_position:
        _render_stock_display(stock_position)

    for i, pos in enumerate(positions):
        _render_position_display(i, pos)

    if is_custom:
        _render_add_leg_button(spot_price)


def _render_leg_cards_readonly(
    positions: List[SimulationOptionPosition],
    stock_position: Optional[SimulationStockPosition]
) -> None:
    """Render read-only leg cards in horizontal layout."""
    num_items = len(positions) + (1 if stock_position else 0)
    if num_items == 0:
        return

    cols = st.columns(min(num_items, 4))
    col_idx = 0

    if stock_position:
        with cols[col_idx % len(cols)]:
            _render_stock_card_compact(stock_position)
        col_idx += 1

    for i, pos in enumerate(positions):
        with cols[col_idx % len(cols)]:
            _render_position_card_compact(i, pos)
        col_idx += 1


def _render_position_card(index: int, position: SimulationOptionPosition) -> None:
    """Render a single option position as a card."""
    is_long = position.position_type == 'long'
    border_color = "#10b981" if is_long else "#ef4444"
    bg_gradient = "linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%)" if is_long else "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"
    position_badge_color = "#047857" if is_long else "#b91c1c"
    position_badge_bg = "#d1fae5" if is_long else "#fee2e2"
    cost_color = "#dc2626" if is_long else "#059669"
    cost_prefix = "-" if is_long else "+"

    total_cost = position.premium * position.quantity * CONTRACT_MULTIPLIER
    option_label = f"{position.option_type.upper()}"

    st.markdown(f"""
    <div style="background: {bg_gradient}; border: 1px solid {border_color}40; border-top: 3px solid {border_color}; border-radius: 8px; padding: 0.75rem; height: 100%;">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">
            <span style="font-size: 0.65rem; font-weight: 700; color: #64748b;">LEG {index + 1}</span>
            <span style="background: {position_badge_bg}; color: {position_badge_color}; font-size: 0.55rem; font-weight: 700; padding: 0.1rem 0.3rem; border-radius: 3px; text-transform: uppercase;">{position.position_type}</span>
        </div>
        <div style="font-size: 0.95rem; font-weight: 600; color: #1e293b;">{option_label}</div>
        <div style="font-size: 0.75rem; color: #64748b; margin: 0.25rem 0;">K = ${position.strike:.0f}</div>
        <div style="font-size: 0.7rem; color: #94a3b8;">{position.quantity}x @ ${position.premium:.2f}</div>
        <div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px dashed #e2e8f0;">
            <span style="color: {cost_color}; font-weight: 700; font-size: 0.85rem; font-family: monospace;">
                {cost_prefix}${total_cost:,.0f}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_position_card_compact(index: int, position: SimulationOptionPosition) -> None:
    """Render compact read-only position card."""
    is_long = position.position_type == 'long'
    border_color = "#10b981" if is_long else "#ef4444"
    bg_color = "#f0fdf4" if is_long else "#fef2f2"
    cost_color = "#dc2626" if is_long else "#059669"
    cost_prefix = "-" if is_long else "+"

    total_cost = position.premium * position.quantity * CONTRACT_MULTIPLIER

    st.markdown(f"""
    <div style="background: {bg_color}; border-left: 3px solid {border_color}; border-radius: 6px; padding: 0.5rem 0.75rem;">
        <div style="font-size: 0.65rem; color: #64748b; text-transform: uppercase;">Leg {index + 1}</div>
        <div style="font-size: 0.85rem; font-weight: 600; color: #1e293b;">{position.position_type.title()} {position.option_type.title()}</div>
        <div style="font-size: 0.7rem; color: #64748b;">K=${position.strike:.0f} • {position.quantity}x</div>
        <div style="color: {cost_color}; font-weight: 600; font-size: 0.8rem; font-family: monospace; margin-top: 0.25rem;">
            {cost_prefix}${total_cost:,.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_stock_card(stock_position: SimulationStockPosition) -> None:
    """Render stock position as a card."""
    is_long = stock_position.position_type == 'long'
    cost_prefix = "-" if is_long else "+"
    cost_color = "#dc2626" if is_long else "#059669"
    stock_cost = stock_position.entry_price * stock_position.quantity

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 1px solid #3b82f640; border-top: 3px solid #3b82f6; border-radius: 8px; padding: 0.75rem; height: 100%;">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">
            <span style="font-size: 0.65rem; font-weight: 700; color: #1d4ed8;">STOCK</span>
            <span style="background: #bfdbfe; color: #1d4ed8; font-size: 0.55rem; font-weight: 700; padding: 0.1rem 0.3rem; border-radius: 3px; text-transform: uppercase;">{stock_position.position_type}</span>
        </div>
        <div style="font-size: 0.95rem; font-weight: 600; color: #1e293b;">100 Shares</div>
        <div style="font-size: 0.75rem; color: #64748b; margin: 0.25rem 0;">Entry: ${stock_position.entry_price:.2f}</div>
        <div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px dashed #bfdbfe;">
            <span style="color: {cost_color}; font-weight: 700; font-size: 0.85rem; font-family: monospace;">
                {cost_prefix}${stock_cost:,.0f}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_stock_card_compact(stock_position: SimulationStockPosition) -> None:
    """Render compact read-only stock card."""
    is_long = stock_position.position_type == 'long'
    cost_prefix = "-" if is_long else "+"
    cost_color = "#dc2626" if is_long else "#059669"
    stock_cost = stock_position.entry_price * stock_position.quantity

    st.markdown(f"""
    <div style="background: #eff6ff; border-left: 3px solid #3b82f6; border-radius: 6px; padding: 0.5rem 0.75rem;">
        <div style="font-size: 0.65rem; color: #1d4ed8; text-transform: uppercase;">Stock</div>
        <div style="font-size: 0.85rem; font-weight: 600; color: #1e293b;">{stock_position.position_type.title()} 100</div>
        <div style="font-size: 0.7rem; color: #64748b;">@ ${stock_position.entry_price:.2f}</div>
        <div style="color: {cost_color}; font-weight: 600; font-size: 0.8rem; font-family: monospace; margin-top: 0.25rem;">
            {cost_prefix}${stock_cost:,.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_add_leg_card(spot_price: float) -> None:
    """Render add leg card for custom strategies."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border: 2px dashed #cbd5e1; border-radius: 8px; padding: 0.75rem; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 120px;">
        <div style="font-size: 1.5rem; color: #94a3b8;">➕</div>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">Add Leg</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Add Option Leg", key="pnl_add_leg_card_btn", width="stretch"):
        _add_custom_leg(spot_price)


def _calculate_total_net_cost(
    positions: List[SimulationOptionPosition],
    stock_position: Optional[SimulationStockPosition]
) -> float:
    """Calculate total net cost/credit for all positions."""
    total_net_cost = 0.0

    # Stock cost
    if stock_position:
        stock_cost = stock_position.entry_price * stock_position.quantity
        if stock_position.position_type == 'long':
            total_net_cost -= stock_cost
        else:
            total_net_cost += stock_cost

    # Option costs
    for pos in positions:
        total_cost = pos.premium * pos.quantity * CONTRACT_MULTIPLIER
        if pos.position_type == 'long':
            total_net_cost -= total_cost
        else:
            total_net_cost += total_cost

    return total_net_cost


def _initialize_pnl_strategy_state(selected_strategy: str, spot_price: float) -> None:
    """Initialize session state for P&L strategy builder."""
    if 'pnl_strategy_legs_state' not in st.session_state:
        st.session_state.pnl_strategy_legs_state = {}

    if 'pnl_custom_legs' not in st.session_state:
        st.session_state.pnl_custom_legs = []

    strategy_changed = st.session_state.get('pnl_last_strategy') != selected_strategy
    spot_changed = st.session_state.get('pnl_last_spot') != spot_price

    if strategy_changed or spot_changed:
        st.session_state.pnl_last_strategy = selected_strategy
        st.session_state.pnl_last_spot = spot_price
        st.session_state.pnl_strategy_legs_state = {}

        if selected_strategy != "custom":
            base_legs = STRATEGY_LEGS.get(selected_strategy, [])
            for i, leg in enumerate(base_legs):
                st.session_state.pnl_strategy_legs_state[i] = {
                    'option_type': leg['option_type'],
                    'position_type': leg['position_type'],
                    'strike': round(spot_price * leg['strike_factor'], 2),
                    'quantity': leg['quantity']
                }

            if selected_strategy in STRATEGIES_WITH_STOCK:
                stock_pos_type = STRATEGY_STOCK_POSITION.get(selected_strategy, 'long')
                st.session_state.pnl_strategy_legs_state['stock'] = {
                    'position_type': stock_pos_type,
                    'quantity': 100,
                    'entry_price': spot_price
                }


def _build_positions_from_state(
    selected_strategy: str,
    spot_price: float,
    risk_free_rate: float,
    time_to_expiry: float,
    volatility: float,
    bs_price_function,
    is_custom: bool,
    has_stock: bool
) -> Tuple[List[SimulationOptionPosition], Optional[SimulationStockPosition]]:
    """Build position objects from session state."""
    positions = []
    stock_position = None

    # Determine number of legs
    if is_custom:
        num_legs = len(st.session_state.get('pnl_custom_legs', []))
    else:
        num_legs = len(STRATEGY_LEGS.get(selected_strategy, []))

    # Build option positions
    for i in range(num_legs):
        leg_state = st.session_state.pnl_strategy_legs_state.get(i, {})
        if not leg_state:
            continue

        # Calculate premium using Black-Scholes
        strike = leg_state['strike']
        option_type = leg_state['option_type']

        try:
            premium = bs_price_function(
                spot_price, strike, risk_free_rate,
                time_to_expiry, volatility, option_type
            )
        except Exception:
            premium = 0.0

        positions.append(SimulationOptionPosition(
            option_type=leg_state['option_type'],
            position_type=leg_state['position_type'],
            strike=leg_state['strike'],
            quantity=leg_state['quantity'],
            premium=premium
        ))

    # Build stock position
    if has_stock:
        stock_state = st.session_state.pnl_strategy_legs_state.get('stock')
        if stock_state:
            stock_position = SimulationStockPosition(
                position_type=stock_state['position_type'],
                quantity=stock_state['quantity'],
                entry_price=stock_state['entry_price']
            )

    return positions, stock_position


def _render_strategy_info(strategy: str, positions: list, has_stock: bool) -> None:
    """Render strategy header info."""
    strategy_name = STRATEGY_DISPLAY_NAMES.get(strategy, "Custom Strategy")
    num_legs = len(positions)

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%); padding: 0.75rem 1rem; border-radius: 10px; margin: 0.5rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-weight: 600; color: #ffffff; font-size: 0.9rem;">{strategy_name}</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.7rem; margin-top: 0.15rem;">
                    {num_legs} leg{'s' if num_legs > 1 else ''}{' + 100 shares' if has_stock else ''}
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.15); padding: 0.25rem 0.5rem; border-radius: 5px;">
                <span style="color: #fbbf24; font-size: 0.65rem; font-weight: 600; text-transform: uppercase;">{'Multi-Leg' if num_legs > 1 else 'Single'}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_stock_display(stock_position: SimulationStockPosition) -> None:
    """Render stock position display (vertical mode)."""
    is_long = stock_position.position_type == 'long'
    cost_prefix = "-" if is_long else "+"
    cost_color = "#dc2626" if is_long else "#059669"
    stock_cost = stock_position.entry_price * stock_position.quantity

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 1px solid #3b82f640; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 0.7rem; font-weight: 700; color: #1d4ed8; text-transform: uppercase;">Stock Position</span>
                <div style="font-size: 0.8rem; color: #475569; margin-top: 0.25rem;">
                    {stock_position.quantity:,} shares @ ${stock_position.entry_price:.2f}
                </div>
            </div>
            <span style="color: {cost_color}; font-weight: 700; font-size: 0.85rem; font-family: monospace;">
                {cost_prefix}${stock_cost:,.2f}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_position_display(index: int, position: SimulationOptionPosition) -> None:
    """Render option position display (vertical mode)."""
    is_long = position.position_type == 'long'
    border_color = "#10b981" if is_long else "#ef4444"
    bg_gradient = "linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%)" if is_long else "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"
    position_badge_color = "#047857" if is_long else "#b91c1c"
    cost_color = "#dc2626" if is_long else "#059669"
    cost_prefix = "-" if is_long else "+"

    total_cost = position.premium * position.quantity * CONTRACT_MULTIPLIER
    option_label = f"{position.option_type.upper()} K=${position.strike:.0f}"

    st.markdown(f"""
    <div style="background: {bg_gradient}; border: 1px solid {border_color}40; border-left: 4px solid {border_color}; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 0.7rem; font-weight: 700; color: #475569;">LEG {index + 1}</span>
                    <span style="background: {'#d1fae5' if is_long else '#fee2e2'}; color: {position_badge_color}; font-size: 0.6rem; font-weight: 700; padding: 0.15rem 0.4rem; border-radius: 4px; text-transform: uppercase;">{position.position_type}</span>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; font-weight: 500; margin-top: 0.25rem;">
                    {position.quantity}x {option_label}
                </div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.15rem;">
                    Premium: ${position.premium:.2f}/share
                </div>
            </div>
            <span style="color: {cost_color}; font-weight: 700; font-size: 0.9rem; font-family: monospace;">
                {cost_prefix}${total_cost:,.2f}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_add_leg_button(spot_price: float) -> None:
    """Render add leg button for custom strategies (vertical mode)."""
    if st.button("➕ Add Option Leg", key="pnl_add_leg_btn", width="stretch"):
        _add_custom_leg(spot_price)


def _add_custom_leg(spot_price: float) -> None:
    """Add a new custom leg to the strategy."""
    if 'pnl_custom_legs' not in st.session_state:
        st.session_state.pnl_custom_legs = []

    new_leg = {
        "option_type": "call",
        "position_type": "long",
        "strike_factor": 1.0,
        "quantity": 1
    }
    st.session_state.pnl_custom_legs.append(new_leg)

    new_idx = len(st.session_state.pnl_custom_legs) - 1
    st.session_state.pnl_strategy_legs_state[new_idx] = {
        'option_type': 'call',
        'position_type': 'long',
        'strike': round(spot_price, 2),
        'quantity': 1
    }
    st.rerun()


def _render_strategy_summary(total_net_cost: float, has_stock: bool) -> None:
    """Render strategy cost summary."""
    is_debit = total_net_cost < 0

    if is_debit:
        summary_bg = "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"
        summary_border = "#fca5a5"
        summary_label = "Total Debit"
        summary_color = "#dc2626"
        display_amount = f"-${abs(total_net_cost):,.2f}"
    else:
        summary_bg = "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)"
        summary_border = "#86efac"
        summary_label = "Total Credit"
        summary_color = "#059669"
        display_amount = f"+${abs(total_net_cost):,.2f}"

    st.markdown(f"""
    <div style="background: {summary_bg}; border: 1px solid {summary_border}; border-radius: 10px; padding: 0.75rem; margin-top: 0.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 600;">{summary_label}</div>
                <div style="font-size: 0.65rem; color: #94a3b8;">{'Incl. stock' if has_stock else 'Options only'}</div>
            </div>
            <div style="font-size: 1.2rem; font-weight: 700; color: {summary_color}; font-family: monospace;">
                {display_amount}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Position Export Functions
# =============================================================================

def export_positions_for_pnl_engine(
    positions: List[SimulationOptionPosition],
    stock_position: Optional[SimulationStockPosition] = None
) -> Dict[str, np.ndarray]:
    """
    Export positions to numpy arrays for the Numba P&L engine.

    Returns
    -------
    dict
        Dictionary with arrays ready for pnl_engine functions:
        - strikes, option_types, position_types, quantities, premiums
        - stock_quantity, stock_entry_price (if applicable)
    """
    n_legs = len(positions)

    if n_legs == 0:
        return {
            'strikes': np.array([], dtype=np.float64),
            'option_types': np.array([], dtype=np.float64),
            'position_types': np.array([], dtype=np.float64),
            'quantities': np.array([], dtype=np.float64),
            'premiums': np.array([], dtype=np.float64),
            'stock_quantity': 0.0,
            'stock_entry_price': 0.0
        }

    strikes = np.zeros(n_legs, dtype=np.float64)
    option_types = np.zeros(n_legs, dtype=np.float64)
    position_types = np.zeros(n_legs, dtype=np.float64)
    quantities = np.zeros(n_legs, dtype=np.float64)
    premiums = np.zeros(n_legs, dtype=np.float64)

    for i, pos in enumerate(positions):
        strikes[i] = pos.strike
        option_types[i] = 1.0 if pos.option_type == 'call' else -1.0
        position_types[i] = 1.0 if pos.position_type == 'long' else -1.0
        quantities[i] = pos.quantity
        premiums[i] = pos.premium

    result = {
        'strikes': strikes,
        'option_types': option_types,
        'position_types': position_types,
        'quantities': quantities,
        'premiums': premiums,
        'stock_quantity': 0.0,
        'stock_entry_price': 0.0
    }

    if stock_position:
        sign = 1.0 if stock_position.position_type == 'long' else -1.0
        result['stock_quantity'] = sign * stock_position.quantity
        result['stock_entry_price'] = stock_position.entry_price

    return result


def get_net_premium(positions: List[SimulationOptionPosition]) -> float:
    """Calculate net premium (debit is negative, credit is positive)."""
    net = 0.0
    for pos in positions:
        total = pos.premium * pos.quantity * CONTRACT_MULTIPLIER
        if pos.position_type == 'long':
            net -= total
        else:
            net += total
    return net
