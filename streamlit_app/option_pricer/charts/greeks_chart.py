"""
Greeks Chart components for Options Greeks Explorer.

Professional visualization for first, second, and third-order Greeks.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from config.constants import (
    GREEK_TITLES,
    FIRST_ORDER,
    SECOND_ORDER,
    THIRD_ORDER,
    DTE_RANGE,
    IV_RANGE,
    STRIKE_RANGE_FACTORS
)
from config.chart_theme import (
    GREEK_COLORS,
    LAYOUT_DEFAULTS,
    AXIS_DEFAULTS,
    SLIDER_DEFAULTS,
    CHART_COLORS,
    get_layout_config
)

# Colors for individual legs (distinct from aggregate)
LEG_COLORS = [
    '#ef4444',  # Red
    '#3b82f6',  # Blue
    '#22c55e',  # Green
    '#f59e0b',  # Amber
    '#8b5cf6',  # Purple
    '#ec4899',  # Pink
    '#06b6d4',  # Cyan
    '#84cc16',  # Lime
]


def create_greeks_subplot(
    greeks_list: list[str],
    greeks_data: dict,
    slider_type: str,
    dte_values: list[int],
    iv_values: list[int],
    spot_range: np.ndarray,
    spot_price: float,
    subplot_rows: int = 2,
    subplot_cols: int = 2,
    all_leg_greeks: dict = None,
    show_individual_legs: bool = False
) -> tuple[go.Figure, dict]:
    """
    Create a subplot figure with multiple Greeks.

    Args:
        greeks_list: List of Greek names to display
        greeks_data: Dictionary with Greeks data
        slider_type: "DTE" or "IV" for parameter variation
        dte_values: List of DTE values
        iv_values: List of IV values
        spot_range: Array of spot prices for x-axis
        spot_price: Current spot price for reference line
        subplot_rows: Number of subplot rows
        subplot_cols: Number of subplot columns
        all_leg_greeks: Dictionary keyed by "DTE_IV" with leg Greeks for each combination
        show_individual_legs: Whether to show individual leg traces

    Returns:
        Tuple of (Figure, slider_dict)
    """
    # Create subplot titles
    titles = [GREEK_TITLES[g] for g in greeks_list[:subplot_rows * subplot_cols]]

    fig = make_subplots(
        rows=subplot_rows,
        cols=subplot_cols,
        subplot_titles=titles,
        vertical_spacing=0.18 if subplot_rows == 2 else 0.10,
        horizontal_spacing=0.10,
        shared_xaxes='all'  # Share ALL x-axes for synchronized crosshair
    )

    positions = [(r + 1, c + 1) for r in range(subplot_rows) for c in range(subplot_cols)]

    # Add traces for each Greek
    for greek_idx, greek_name in enumerate(greeks_list):
        if greek_idx >= len(positions):
            break

        row, col = positions[greek_idx]
        _add_greek_traces(
            fig, greek_name, greeks_data, slider_type,
            dte_values, iv_values, spot_range, row, col,
            all_leg_greeks=all_leg_greeks,
            show_individual_legs=show_individual_legs,
            is_first_greek=(greek_idx == 0)
        )

        # Add reference lines with theme colors
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color=CHART_COLORS['neutral'],
            opacity=0.4,
            row=row,
            col=col
        )
        fig.add_vline(
            x=spot_price,
            line_dash="dot",
            line_color=CHART_COLORS['accent'],
            opacity=0.5,
            row=row,
            col=col
        )

    # Calculate number of leg traces per slider step
    # Get num_legs from first key in all_leg_greeks (all keys have same number of legs)
    num_legs = 0
    if show_individual_legs and all_leg_greeks:
        first_key = next(iter(all_leg_greeks), None)
        if first_key:
            num_legs = len(all_leg_greeks[first_key])

    # Create slider
    slider_dict = _create_greeks_slider(slider_type, greeks_list, positions, num_legs)

    # Configure spikelines for crosshair effect
    fig.update_xaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1.5,
        spikecolor='#6366f1',
        spikedash='solid'
    )
    fig.update_yaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='#a5b4fc',
        spikedash='dot'
    )

    return fig, slider_dict


def _add_greek_traces(
    fig: go.Figure,
    greek_name: str,
    greeks_data: dict,
    slider_type: str,
    dte_values: list[int],
    iv_values: list[int],
    spot_range: np.ndarray,
    row: int,
    col: int,
    all_leg_greeks: dict = None,
    show_individual_legs: bool = False,
    is_first_greek: bool = False
) -> None:
    """Add traces for a single Greek to the figure.

    Args:
        all_leg_greeks: Dictionary keyed by "DTE_IV" with leg Greeks for each combination
    """
    greek_color = GREEK_COLORS.get(greek_name, CHART_COLORS['primary'])
    greek_title = GREEK_TITLES.get(greek_name, greek_name.capitalize())

    if slider_type == "DTE":
        fixed_iv = 25
        for dte in dte_values:
            key = f"{dte}_{fixed_iv}"
            visible = (dte == 31)

            # Add aggregate trace
            fig.add_trace(
                go.Scatter(
                    x=spot_range,
                    y=greeks_data[key][greek_name],
                    mode='lines',
                    name=f'{greek_name}: DTE={dte}',
                    visible=visible,
                    line=dict(width=2.5, color=greek_color),
                    showlegend=False,
                    hovertemplate=(
                        f'<b>Underlying:</b> $%{{x:,.2f}}<br>' +
                        f'<b>DTE:</b> {dte} days<br>' +
                        f'<b>{greek_title}:</b> %{{y:.4f}}<br>' +
                        '<extra></extra>'
                    )
                ),
                row=row, col=col
            )

            # Add individual leg traces for this specific DTE/IV combination
            if show_individual_legs and all_leg_greeks and key in all_leg_greeks:
                show_in_legend = is_first_greek and visible
                _add_individual_leg_traces(
                    fig, greek_name, all_leg_greeks[key], spot_range, row, col,
                    param_label=f"DTE={dte}", visible=visible,
                    show_in_legend=show_in_legend
                )
    else:  # IV mode
        fixed_dte = 31
        for iv in iv_values:
            key = f"{fixed_dte}_{iv}"
            visible = (iv == 25)

            # Add aggregate trace
            fig.add_trace(
                go.Scatter(
                    x=spot_range,
                    y=greeks_data[key][greek_name],
                    mode='lines',
                    name=f'{greek_name}: IV={iv}%',
                    visible=visible,
                    line=dict(width=2.5, color=greek_color),
                    showlegend=False,
                    hovertemplate=(
                        f'<b>Underlying:</b> $%{{x:,.2f}}<br>' +
                        f'<b>IV:</b> {iv}%<br>' +
                        f'<b>{greek_title}:</b> %{{y:.4f}}<br>' +
                        '<extra></extra>'
                    )
                ),
                row=row, col=col
            )

            # Add individual leg traces for this specific DTE/IV combination
            if show_individual_legs and all_leg_greeks and key in all_leg_greeks:
                show_in_legend = is_first_greek and visible
                _add_individual_leg_traces(
                    fig, greek_name, all_leg_greeks[key], spot_range, row, col,
                    param_label=f"IV={iv}%", visible=visible,
                    show_in_legend=show_in_legend
                )


def _add_individual_leg_traces(
    fig: go.Figure,
    greek_name: str,
    leg_greeks: dict,
    spot_range: np.ndarray,
    row: int,
    col: int,
    param_label: str = "",
    visible: bool = True,
    show_in_legend: bool = False
) -> None:
    """Add individual leg traces for a Greek (with transparency).

    Args:
        show_in_legend: Only True for the first Greek's first slider step to avoid duplicates
    """
    greek_title = GREEK_TITLES.get(greek_name, greek_name.capitalize())

    leg_idx = 0
    for leg_key, leg_data in leg_greeks.items():
        if leg_key == 'stock':
            # Stock position
            leg_label = f"Stock ({leg_data['position_type'].capitalize()})"
            leg_color = '#94a3b8'  # Gray for stock
        else:
            # Option leg
            pos_type = leg_data['position_type'].capitalize()
            opt_type = leg_data['option_type'].capitalize()
            strike = leg_data['strike']
            qty = leg_data['quantity']
            leg_label = f"{pos_type} {opt_type} K=${strike:.0f}"
            if qty > 1:
                leg_label = f"{qty}x {leg_label}"
            leg_color = LEG_COLORS[leg_idx % len(LEG_COLORS)]
            leg_idx += 1

        greeks_values = leg_data['greeks'][greek_name]

        fig.add_trace(
            go.Scatter(
                x=spot_range,
                y=greeks_values,
                mode='lines',
                name=leg_label,
                visible=visible,
                line=dict(width=1.5, color=leg_color, dash='dot'),
                opacity=0.6,
                showlegend=show_in_legend,
                legendgroup=leg_key,
                hovertemplate=(
                    f'<b>{leg_label}</b><br>' +
                    f'<b>Underlying:</b> $%{{x:,.2f}}<br>' +
                    f'<b>{greek_title}:</b> %{{y:.4f}}<br>' +
                    '<extra></extra>'
                )
            ),
            row=row, col=col
        )


def create_greeks_subplot_strike(
    greeks_list: list[str],
    greeks_data: dict,
    spot_range: np.ndarray,
    spot_price: float,
    subplot_rows: int = 2,
    subplot_cols: int = 2
) -> tuple[go.Figure, dict]:
    """
    Create a subplot figure with multiple Greeks for strike variation.

    Args:
        greeks_list: List of Greek names to display
        greeks_data: Dictionary with Greeks data keyed by strike
        spot_range: Array of spot prices for x-axis
        spot_price: Current spot price for reference line
        subplot_rows: Number of subplot rows
        subplot_cols: Number of subplot columns

    Returns:
        Tuple of (Figure, slider_dict)
    """
    titles = [GREEK_TITLES[g] for g in greeks_list[:subplot_rows * subplot_cols]]

    fig = make_subplots(
        rows=subplot_rows,
        cols=subplot_cols,
        subplot_titles=titles,
        vertical_spacing=0.18 if subplot_rows == 2 else 0.10,
        horizontal_spacing=0.10,
        shared_xaxes=True  # Share x-axes for synchronized crosshair
    )

    positions = [(r + 1, c + 1) for r in range(subplot_rows) for c in range(subplot_cols)]
    num_greeks = min(len(greeks_list), len(positions))

    # Add traces for each Greek and each strike
    for greek_idx, greek_name in enumerate(greeks_list):
        if greek_idx >= len(positions):
            break

        row, col = positions[greek_idx]
        greek_color = GREEK_COLORS.get(greek_name, CHART_COLORS['primary'])
        greek_title = GREEK_TITLES.get(greek_name, greek_name.capitalize())

        for strike_idx, strike_factor in enumerate(STRIKE_RANGE_FACTORS):
            key = f"strike_{int(strike_factor * 100)}"
            strike_price = spot_price * strike_factor
            visible = (strike_factor == 1.0)  # ATM is default visible

            fig.add_trace(
                go.Scatter(
                    x=spot_range,
                    y=greeks_data[key][greek_name],
                    mode='lines',
                    name=f'{greek_name}: K=${strike_price:.0f}',
                    visible=visible,
                    line=dict(width=2.5, color=greek_color),
                    showlegend=False,
                    hovertemplate=(
                        f'<b>Underlying:</b> $%{{x:,.2f}}<br>' +
                        f'<b>Strike:</b> ${strike_price:.0f}<br>' +
                        f'<b>{greek_title}:</b> %{{y:.4f}}<br>' +
                        '<extra></extra>'
                    )
                ),
                row=row, col=col
            )

        # Add reference lines
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color=CHART_COLORS['neutral'],
            opacity=0.4,
            row=row,
            col=col
        )
        fig.add_vline(
            x=spot_price,
            line_dash="dot",
            line_color=CHART_COLORS['accent'],
            opacity=0.5,
            row=row,
            col=col
        )

    # Create slider for strike
    slider_dict = _create_greeks_strike_slider(spot_price, num_greeks)

    # Configure spikelines for crosshair effect
    fig.update_xaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1.5,
        spikecolor='#6366f1',
        spikedash='solid'
    )
    fig.update_yaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='#a5b4fc',
        spikedash='dot'
    )

    return fig, slider_dict


def _create_greeks_strike_slider(spot_price: float, num_greeks: int) -> dict:
    """Create slider configuration for Greeks strike variation."""
    steps = []
    num_strikes = len(STRIKE_RANGE_FACTORS)

    for strike_idx, strike_factor in enumerate(STRIKE_RANGE_FACTORS):
        strike = spot_price * strike_factor
        step = dict(
            method="update",
            args=[{"visible": [False] * (num_greeks * num_strikes)}],
            label=f"${strike:.0f}"
        )
        # Make this strike visible for all Greeks
        for greek_idx in range(num_greeks):
            trace_idx = greek_idx * num_strikes + strike_idx
            step["args"][0]["visible"][trace_idx] = True
        steps.append(step)

    slider = SLIDER_DEFAULTS.copy()
    slider.update({
        'active': STRIKE_RANGE_FACTORS.index(1.0) if 1.0 in STRIKE_RANGE_FACTORS else 10,
        'currentvalue': {
            **SLIDER_DEFAULTS['currentvalue'],
            'prefix': 'Strike Price: '
        },
        'steps': steps
    })

    return slider


def _create_greeks_slider(
    slider_type: str,
    greeks_list: list[str],
    positions: list[tuple[int, int]],
    num_legs: int = 0
) -> dict:
    """Create slider configuration for Greeks charts.

    Args:
        slider_type: "DTE" or "IV"
        greeks_list: List of Greek names
        positions: List of subplot positions
        num_legs: Number of individual leg traces per Greek per slider step
    """
    param_values = DTE_RANGE if slider_type == "DTE" else IV_RANGE
    num_greeks = min(len(greeks_list), len(positions))

    # Each slider step has: 1 aggregate trace + num_legs traces per Greek
    traces_per_step_per_greek = 1 + num_legs
    total_traces = num_greeks * len(param_values) * traces_per_step_per_greek

    steps = []
    for idx, value in enumerate(param_values):
        step = dict(
            method="update",
            args=[{"visible": [False] * total_traces}],
            label=str(value) if slider_type == "DTE" else f"{value}%"
        )
        # Make traces for this slider step visible for all Greeks
        for greek_idx in range(num_greeks):
            # Base index for this Greek's traces
            base_idx = greek_idx * len(param_values) * traces_per_step_per_greek
            # Index within this Greek's traces for this slider step
            step_base = base_idx + idx * traces_per_step_per_greek
            # Make aggregate trace visible
            step["args"][0]["visible"][step_base] = True
            # Make individual leg traces visible
            for leg_idx in range(num_legs):
                step["args"][0]["visible"][step_base + 1 + leg_idx] = True
        steps.append(step)

    prefix = "Days to Expiration: " if slider_type == "DTE" else "Implied Volatility: "

    slider = SLIDER_DEFAULTS.copy()
    slider.update({
        'currentvalue': {
            **SLIDER_DEFAULTS['currentvalue'],
            'prefix': prefix
        },
        'steps': steps
    })

    return slider


def render_greeks_tab(
    greeks_list: list[str],
    greeks_data: dict,
    spot_range: np.ndarray,
    spot_price: float,
    tab_title: str,
    positions: list,
    stock_position,
    subplot_rows: int = 2,
    subplot_cols: int = 2,
    slider_key: str = "g_slider",
    risk_free_rate: float = 0.05,
    calculate_all_greeks_func=None,
    calculate_pnl_at_expiry_func=None,
    portfolio_json: str = None
) -> None:
    """
    Render a complete Greeks tab.

    Args:
        greeks_list: List of Greek names to display
        greeks_data: Dictionary with Greeks data
        spot_range: Array of spot prices
        spot_price: Current spot price
        tab_title: Title for the subheader
        positions: List of option positions
        stock_position: Stock position or None
        subplot_rows: Number of subplot rows
        subplot_cols: Number of subplot columns
        slider_key: Unique key for the slider radio button
        risk_free_rate: Risk-free interest rate
        calculate_all_greeks_func: Function to calculate Greeks (for strike variation)
        calculate_pnl_at_expiry_func: Function to calculate P&L at expiry (for strike variation)
        portfolio_json: JSON string of portfolio data (for individual leg display)
    """
    from components.metrics import render_chart_controls
    from services.portfolio_calculator import calculate_strike_surfaces

    if not positions and not stock_position:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%);
            border: 1px solid #7dd3fc;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        ">
            <div style="font-size: 1.25rem;">ℹ️</div>
            <div>
                <div style="font-weight: 600; color: #0c4a6e; font-size: 0.9rem;">Default Position: Long Call ATM</div>
                <div style="color: #0369a1; font-size: 0.8rem;">Add your own positions using the sidebar to analyze custom strategies</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    from services.portfolio_calculator import calculate_all_individual_leg_greeks

    # Detect single-leg position
    is_single_leg = (
        len(positions) == 1 and stock_position is None
    ) or (
        len(positions) == 0 and stock_position is None
    )

    # Detect multi-leg position
    is_multi_leg = (len(positions) > 1) or (len(positions) >= 1 and stock_position is not None)

    # Chart controls with Strike option for single-leg
    slider_type = render_chart_controls(slider_key, is_single_leg=is_single_leg, spot_price=spot_price)

    # Toggle for showing individual legs (only for multi-leg positions)
    all_leg_greeks = None

    # Use a consistent session state key for show_legs toggle (not dependent on slider_type)
    show_legs_key = f"{slider_key}_show_legs"

    # Initialize session state if not exists
    if show_legs_key not in st.session_state:
        st.session_state[show_legs_key] = False

    # Read current value from session state
    show_individual_legs = st.session_state[show_legs_key]

    if is_multi_leg and slider_type != "Strike":
        col1, col2 = st.columns([1, 3])
        with col1:
            show_individual_legs = st.toggle(
                "Show Legs",
                key=show_legs_key,
                help="Display Greeks for each leg separately (dotted lines) alongside the aggregate (solid line)"
            )

        # Calculate individual leg Greeks for ALL DTE/IV combinations if needed
        if show_individual_legs and portfolio_json and calculate_all_greeks_func:
            all_leg_greeks = calculate_all_individual_leg_greeks(
                portfolio_json=portfolio_json,
                spot_range=tuple(spot_range),
                dte_values=tuple(DTE_RANGE),
                iv_values=tuple(IV_RANGE),
                risk_free_rate=risk_free_rate,
                _calculate_all_greeks_func=calculate_all_greeks_func
            )

    # Handle Strike variation for single-leg
    if slider_type == "Strike" and is_single_leg and calculate_all_greeks_func:
        # Get position details
        if positions:
            pos = positions[0]
            option_type = pos.option_type
            position_type = pos.position_type
            quantity = pos.quantity
            base_strike = pos.strike
        else:
            option_type = 'call'
            position_type = 'long'
            quantity = 1
            base_strike = spot_price

        # Calculate strike surfaces (returns pnl_data, greeks_data, breakeven_data)
        _, strike_greeks_data, _ = calculate_strike_surfaces(
            spot_price=spot_price,
            spot_range=tuple(spot_range),
            option_type=option_type,
            position_type=position_type,
            quantity=quantity,
            base_strike=base_strike,
            risk_free_rate=risk_free_rate,
            _calculate_all_greeks_func=calculate_all_greeks_func,
            _calculate_pnl_at_expiry_func=calculate_pnl_at_expiry_func
        )

        fig, slider_dict = create_greeks_subplot_strike(
            greeks_list=greeks_list,
            greeks_data=strike_greeks_data,
            spot_range=spot_range,
            spot_price=spot_price,
            subplot_rows=subplot_rows,
            subplot_cols=subplot_cols
        )
    else:
        # Create figure (DTE or IV)
        fig, slider_dict = create_greeks_subplot(
            greeks_list=greeks_list,
            greeks_data=greeks_data,
            slider_type=slider_type,
            dte_values=DTE_RANGE,
            iv_values=IV_RANGE,
            spot_range=spot_range,
            spot_price=spot_price,
            subplot_rows=subplot_rows,
            subplot_cols=subplot_cols,
            all_leg_greeks=all_leg_greeks,
            show_individual_legs=show_individual_legs
        )

    # Determine height based on grid size
    height = 800 if subplot_rows == 3 else 650

    # Apply professional layout
    layout = get_layout_config(height=height)
    layout.update({
        'sliders': [slider_dict],
        'showlegend': show_individual_legs,  # Show legend when individual legs are displayed
        'margin': {'l': 60, 'r': 40, 't': 60, 'b': 100 if subplot_rows == 2 else 120},
        'hovermode': 'x',  # Synchronized hover across subplots
        'hoversubplots': 'axis',  # Sync hover across subplots with shared axes
        'hoverlabel': {
            'bgcolor': 'rgba(255, 255, 255, 0.95)',
            'bordercolor': '#e2e8f0',
            'font': {'family': 'Inter, sans-serif', 'size': 12, 'color': '#1e293b'}
        },
        'spikedistance': -1,  # Enable spikes for all points
    })

    # Add legend configuration if showing individual legs
    if show_individual_legs:
        layout.update({
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'center',
                'x': 0.5,
                'font': {'size': 10},
                'bgcolor': 'rgba(255,255,255,0.9)',
                'bordercolor': '#e2e8f0',
                'borderwidth': 1
            }
        })

    fig.update_layout(**layout)

    # Style subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = {
            'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            'size': 13,
            'color': '#1e293b'
        }

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_first_order_greeks(
    greeks_data: dict,
    spot_range: np.ndarray,
    spot_price: float,
    positions: list,
    stock_position,
    risk_free_rate: float = 0.05,
    calculate_all_greeks_func=None,
    calculate_pnl_at_expiry_func=None,
    portfolio_json: str = None
) -> None:
    """Render the first-order Greeks tab."""
    render_greeks_tab(
        greeks_list=FIRST_ORDER,
        greeks_data=greeks_data,
        spot_range=spot_range,
        spot_price=spot_price,
        tab_title="First-Order Greeks",
        positions=positions,
        stock_position=stock_position,
        subplot_rows=3,
        subplot_cols=2,
        slider_key="g1_slider",
        risk_free_rate=risk_free_rate,
        calculate_all_greeks_func=calculate_all_greeks_func,
        calculate_pnl_at_expiry_func=calculate_pnl_at_expiry_func,
        portfolio_json=portfolio_json
    )


def render_second_order_greeks(
    greeks_data: dict,
    spot_range: np.ndarray,
    spot_price: float,
    positions: list,
    stock_position,
    risk_free_rate: float = 0.05,
    calculate_all_greeks_func=None,
    calculate_pnl_at_expiry_func=None,
    portfolio_json: str = None
) -> None:
    """Render the second-order Greeks tab."""
    render_greeks_tab(
        greeks_list=SECOND_ORDER,
        greeks_data=greeks_data,
        spot_range=spot_range,
        spot_price=spot_price,
        tab_title="Second-Order Greeks",
        positions=positions,
        stock_position=stock_position,
        subplot_rows=2,
        subplot_cols=2,
        slider_key="g2_slider",
        risk_free_rate=risk_free_rate,
        calculate_all_greeks_func=calculate_all_greeks_func,
        calculate_pnl_at_expiry_func=calculate_pnl_at_expiry_func,
        portfolio_json=portfolio_json
    )


def render_third_order_greeks(
    greeks_data: dict,
    spot_range: np.ndarray,
    spot_price: float,
    positions: list,
    stock_position,
    risk_free_rate: float = 0.05,
    calculate_all_greeks_func=None,
    calculate_pnl_at_expiry_func=None,
    portfolio_json: str = None
) -> None:
    """Render the third-order Greeks tab."""
    render_greeks_tab(
        greeks_list=THIRD_ORDER,
        greeks_data=greeks_data,
        spot_range=spot_range,
        spot_price=spot_price,
        tab_title="Third-Order Greeks",
        positions=positions,
        stock_position=stock_position,
        subplot_rows=2,
        subplot_cols=2,
        slider_key="g3_slider",
        risk_free_rate=risk_free_rate,
        calculate_all_greeks_func=calculate_all_greeks_func,
        calculate_pnl_at_expiry_func=calculate_pnl_at_expiry_func,
        portfolio_json=portfolio_json
    )
