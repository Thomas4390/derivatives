"""
P&L Chart components for Options Greeks Explorer.

Professional, clean chart design with interactive features.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from config.constants import DTE_RANGE, IV_RANGE, STRIKE_RANGE_FACTORS
from config.chart_theme import (
    CHART_COLORS,
    LAYOUT_DEFAULTS,
    AXIS_DEFAULTS,
    SLIDER_DEFAULTS,
    LINE_STYLES,
    get_layout_config
)


def create_pnl_figure(
    pnl_data: dict,
    spot_range: np.ndarray,
    spot_price: float,
    slider_type: str,
    breakeven_result=None
) -> go.Figure:
    """
    Create the P&L profile figure with interactive slider.

    Args:
        pnl_data: Dictionary with P&L data for different DTE/IV combinations
        spot_range: Array of spot prices for x-axis
        spot_price: Current spot price for reference line
        slider_type: "DTE" or "IV" for parameter variation
        breakeven_result: BreakevenResult object with breakeven points

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    param_values = DTE_RANGE if slider_type == "DTE" else IV_RANGE
    fixed_value = 25 if slider_type == "DTE" else 31
    default_value = 31 if slider_type == "DTE" else 25

    # Add traces for each parameter value
    for value in param_values:
        key = f"{value}_{fixed_value}" if slider_type == "DTE" else f"{fixed_value}_{value}"
        visible = (value == default_value)

        # Create hover template matching Greeks format
        if slider_type == "DTE":
            hover_template = (
                '<b>Underlying:</b> $%{x:,.2f}<br>' +
                f'<b>DTE:</b> {value} days<br>' +
                '<b>P&L:</b> $%{y:,.2f}<br>' +
                '<extra></extra>'
            )
            trace_name = f'P&L: DTE={value}'
        else:
            hover_template = (
                '<b>Underlying:</b> $%{x:,.2f}<br>' +
                f'<b>IV:</b> {value}%<br>' +
                '<b>P&L:</b> $%{y:,.2f}<br>' +
                '<extra></extra>'
            )
            trace_name = f'P&L: IV={value}%'

        fig.add_trace(go.Scatter(
            x=spot_range,
            y=pnl_data[key],
            mode='lines',
            name=trace_name,
            visible=visible,
            line=dict(
                width=LINE_STYLES['primary']['width'],
                color=CHART_COLORS['primary']
            ),
            hovertemplate=hover_template,
            fill='tozeroy',
            fillcolor='rgba(26, 54, 93, 0.08)'
        ))

    # Add expiration curve
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=pnl_data['expiry'],
        mode='lines',
        name='P&L at Expiration',
        visible=True,
        line=dict(
            color=LINE_STYLES['expiry']['color'],
            width=LINE_STYLES['expiry']['width'],
            dash=LINE_STYLES['expiry']['dash']
        ),
        hovertemplate=(
            '<b>Underlying:</b> $%{x:,.2f}<br>' +
            '<b>P&L at Expiry:</b> $%{y:,.2f}<br>' +
            '<extra></extra>'
        )
    ))

    # Add breakeven lines (pass spot_price for smart positioning)
    _add_breakeven_lines(fig, breakeven_result, spot_price)

    # Add reference lines
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color=CHART_COLORS['neutral'],
        line_width=1,
        opacity=0.6
    )
    fig.add_vline(
        x=spot_price,
        line_dash="dot",
        line_color=CHART_COLORS['accent'],
        line_width=1.5,
        opacity=0.8
    )

    # Add Current Price annotation with box style
    fig.add_annotation(
        x=spot_price,
        y=1.02,
        xref="x",
        yref="paper",
        text="Current Price",
        showarrow=False,
        font=dict(size=10, color=CHART_COLORS['accent'], weight='bold'),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=CHART_COLORS['accent'],
        borderwidth=1,
        borderpad=3
    )

    # Create slider
    slider = _create_slider(slider_type, param_values)

    # Update layout
    layout = get_layout_config(height=650)
    layout.update({
        'sliders': [slider],
        'xaxis': {
            **AXIS_DEFAULTS,
            'title': {'text': 'Underlying Price', **AXIS_DEFAULTS['title']},
            'tickprefix': '$',
            'tickformat': ',.0f'
        },
        'yaxis': {
            **AXIS_DEFAULTS,
            'title': {'text': 'Profit / Loss ($)', **AXIS_DEFAULTS['title']},
            'tickprefix': '$',
            'tickformat': ',.0f'
        },
        'margin': {'l': 70, 'r': 40, 't': 40, 'b': 100},
        'showlegend': True,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1,
            'font': {'size': 11}
        }
    })

    fig.update_layout(**layout)

    return fig


def _add_breakeven_lines(fig: go.Figure, breakeven_result, spot_price: float = None) -> None:
    """Add breakeven vertical lines to the figure with label just below Current Price."""
    if not breakeven_result or not breakeven_result.breakeven_points:
        return

    breakeven_points = sorted(breakeven_result.breakeven_points)

    for i, be in enumerate(breakeven_points):
        label = f"BE: ${be:,.0f}" if len(breakeven_points) == 1 else f"BE{i+1}: ${be:,.0f}"

        # Add the vertical line (stops before the label box at y=0.90)
        fig.add_shape(
            type="line",
            x0=be,
            x1=be,
            y0=0,
            y1=0.90,
            xref="x",
            yref="paper",
            line=dict(color=CHART_COLORS['breakeven'], width=1.5, dash="dash")
        )

        # Add annotation just below Current Price level (y=0.94)
        fig.add_annotation(
            x=be,
            y=0.94,
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            font=dict(size=10, color=CHART_COLORS['breakeven'], weight='bold'),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=CHART_COLORS['breakeven'],
            borderwidth=1,
            borderpad=3
        )


def create_pnl_figure_strike(
    pnl_data: dict,
    spot_range: np.ndarray,
    spot_price: float,
    breakeven_data: dict = None,
    option_type: str = 'call',
    position_type: str = 'long'
) -> go.Figure:
    """
    Create the P&L profile figure with Plotly slider for strike variation.
    Includes dynamic metrics (max profit, max loss, breakeven) as annotations.

    Args:
        pnl_data: Dictionary with P&L data for different strikes
        spot_range: Array of spot prices for x-axis
        spot_price: Current spot price for reference line
        breakeven_data: Dictionary with breakeven results keyed by strike
        option_type: 'call' or 'put'
        position_type: 'long' or 'short'

    Returns:
        Plotly Figure object with interactive strike slider
    """
    fig = go.Figure()

    expiry_by_strike = pnl_data.get('expiry_by_strike', {})

    # Determine unlimited profit/loss based on position type
    if option_type == 'call':
        unlimited_profit = (position_type == 'long')
        unlimited_loss = (position_type == 'short')
    else:
        unlimited_profit = False
        unlimited_loss = False

    # Default strike index (ATM = 100%)
    default_idx = STRIKE_RANGE_FACTORS.index(1.0) if 1.0 in STRIKE_RANGE_FACTORS else 10

    # Add traces for each strike (P&L at 31 DTE + Expiry)
    for idx, strike_factor in enumerate(STRIKE_RANGE_FACTORS):
        key = f"strike_{int(strike_factor * 100)}"
        strike_price_val = spot_price * strike_factor
        moneyness = "ATM" if strike_factor == 1.0 else ("ITM" if strike_factor < 1.0 else "OTM")

        visible = (idx == default_idx)

        # P&L curve at 31 DTE - full hover template
        fig.add_trace(go.Scatter(
            x=spot_range,
            y=pnl_data[key],
            mode='lines',
            name='P&L (31 DTE)',
            visible=visible,
            line=dict(
                width=LINE_STYLES['primary']['width'],
                color=CHART_COLORS['primary']
            ),
            hovertemplate=(
                '<b>Underlying:</b> $%{x:,.2f}<br>' +
                f'<b>Strike:</b> ${strike_price_val:.2f} ({moneyness})<br>' +
                '<b>P&L (31 DTE):</b> $%{y:,.2f}<br>' +
                '<extra></extra>'
            ),
            fill='tozeroy',
            fillcolor='rgba(26, 54, 93, 0.08)'
        ))

        # Expiry curve - full hover template
        expiry_pnl = expiry_by_strike.get(key, [])
        fig.add_trace(go.Scatter(
            x=spot_range,
            y=expiry_pnl,
            mode='lines',
            name='At Expiration',
            visible=visible,
            line=dict(
                color=LINE_STYLES['expiry']['color'],
                width=LINE_STYLES['expiry']['width'],
                dash=LINE_STYLES['expiry']['dash']
            ),
            hovertemplate=(
                '<b>Underlying:</b> $%{x:,.2f}<br>' +
                f'<b>Strike:</b> ${strike_price_val:.2f} ({moneyness})<br>' +
                '<b>P&L at Expiry:</b> $%{y:,.2f}<br>' +
                '<extra></extra>'
            )
        ))

    # Add reference lines (always visible)
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color=CHART_COLORS['neutral'],
        line_width=1,
        opacity=0.6
    )
    fig.add_vline(
        x=spot_price,
        line_dash="dot",
        line_color=CHART_COLORS['accent'],
        line_width=1.5,
        opacity=0.8
    )

    # Add Current Price annotation with box style
    fig.add_annotation(
        x=spot_price,
        y=1.02,
        xref="x",
        yref="paper",
        text="Current Price",
        showarrow=False,
        font=dict(size=10, color=CHART_COLORS['accent'], weight='bold'),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=CHART_COLORS['accent'],
        borderwidth=1,
        borderpad=3
    )

    # Create slider steps with dynamic annotations for metrics
    steps = []
    num_strikes = len(STRIKE_RANGE_FACTORS)
    traces_per_strike = 2  # P&L + Expiry

    for idx, strike_factor in enumerate(STRIKE_RANGE_FACTORS):
        key = f"strike_{int(strike_factor * 100)}"
        strike_price_val = spot_price * strike_factor
        moneyness = "ATM" if strike_factor == 1.0 else ("ITM" if strike_factor < 1.0 else "OTM")

        # Get metrics for this strike
        if breakeven_data and key in breakeven_data and breakeven_data[key]:
            be_result = breakeven_data[key]
            breakeven_points = be_result.breakeven_points or []
            be_count = len(breakeven_points)

            if unlimited_profit:
                max_profit_text = "Unlimited"
            else:
                max_profit_text = f"${be_result.max_profit:,.0f}"

            if unlimited_loss:
                max_loss_text = "Unlimited"
            else:
                max_loss_text = f"${abs(be_result.max_loss):,.0f}"

            if be_count == 1:
                be_text = f"${breakeven_points[0]:,.0f}"
            elif be_count == 2:
                be_text = f"${breakeven_points[0]:,.0f} / ${breakeven_points[1]:,.0f}"
            elif be_count > 2:
                be_text = f"{be_count} points"
            else:
                be_text = "N/A"
        else:
            max_profit_text = "N/A"
            max_loss_text = "N/A"
            be_text = "N/A"
            breakeven_points = []

        # Set visibility for traces
        visible = [False] * (num_strikes * traces_per_strike)
        visible[idx * traces_per_strike] = True      # P&L curve
        visible[idx * traces_per_strike + 1] = True  # Expiry curve

        # Build annotations for this step
        annotations = [
            # Strike price annotation with box style (same level as Current Price: y=1.02)
            dict(
                x=strike_price_val,
                y=1.02,
                xref="x",
                yref="paper",
                text=f"Strike: ${strike_price_val:.0f}",
                showarrow=False,
                font=dict(size=10, color="#8b5cf6", weight='bold'),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#8b5cf6",
                borderwidth=1,
                borderpad=3
            ),
            # Metrics box annotation (top left)
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=(
                    f"<b>Strike:</b> ${strike_price_val:.0f} ({moneyness})<br>"
                    f"<b>Max Profit:</b> <span style='color:#059669'>{max_profit_text}</span><br>"
                    f"<b>Max Loss:</b> <span style='color:#dc2626'>{max_loss_text}</span><br>"
                    f"<b>Breakeven:</b> {be_text}"
                ),
                showarrow=False,
                font=dict(size=12, family="Inter, sans-serif"),
                align="left",
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#e2e8f0",
                borderwidth=1,
                borderpad=10,
                xanchor="left",
                yanchor="top"
            )
        ]

        # Add breakeven vertical line annotations (just below Current Price/Strike level)
        sorted_be_points = sorted(breakeven_points)
        for i, be in enumerate(sorted_be_points):
            label = f"BE: ${be:,.0f}" if len(sorted_be_points) == 1 else f"BE{i+1}: ${be:,.0f}"

            annotations.append(dict(
                x=be,
                y=0.94,
                xref="x",
                yref="paper",
                text=label,
                showarrow=False,
                font=dict(size=10, color=CHART_COLORS['breakeven'], weight='bold'),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=CHART_COLORS['breakeven'],
                borderwidth=1,
                borderpad=3
            ))

        # Add strike vertical line shape
        shapes = [
            dict(
                type="line",
                x0=strike_price_val,
                x1=strike_price_val,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="#8b5cf6", width=1.5, dash="dashdot"),
                opacity=0.8
            )
        ]

        # Add breakeven vertical lines (stop at y=0.90, before label box)
        for be in breakeven_points:
            shapes.append(dict(
                type="line",
                x0=be,
                x1=be,
                y0=0,
                y1=0.90,
                xref="x",
                yref="paper",
                line=dict(color=CHART_COLORS['breakeven'], width=1.5, dash="dash")
            ))

        step = dict(
            method="update",
            args=[
                {"visible": visible},
                {"annotations": annotations, "shapes": shapes}
            ],
            label=f"${strike_price_val:.0f}"
        )
        steps.append(step)

    # Create slider using standard defaults for consistency
    slider = SLIDER_DEFAULTS.copy()
    slider.update({
        'active': default_idx,
        'currentvalue': {
            **SLIDER_DEFAULTS['currentvalue'],
            'prefix': 'Strike Price: '
        },
        'steps': steps
    })

    # Get initial annotations and shapes for default strike
    default_step = steps[default_idx]
    initial_annotations = default_step['args'][1]['annotations']
    initial_shapes = default_step['args'][1]['shapes']

    # Update layout
    layout = get_layout_config(height=700)
    layout.update({
        'sliders': [slider],
        'annotations': initial_annotations,
        'shapes': initial_shapes,
        'xaxis': {
            **AXIS_DEFAULTS,
            'title': {'text': 'Underlying Price', **AXIS_DEFAULTS['title']},
            'tickprefix': '$',
            'tickformat': ',.0f'
        },
        'yaxis': {
            **AXIS_DEFAULTS,
            'title': {'text': 'Profit / Loss ($)', **AXIS_DEFAULTS['title']},
            'tickprefix': '$',
            'tickformat': ',.0f'
        },
        'margin': {'l': 70, 'r': 40, 't': 60, 'b': 100},
        'showlegend': True,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1,
            'font': {'size': 11}
        }
    })

    fig.update_layout(**layout)

    return fig


def _create_strike_slider_v2(spot_price: float, num_strikes: int) -> dict:
    """Create the strike price slider configuration (2 traces per strike)."""
    steps = []
    traces_per_strike = 2  # P&L + Expiry

    for idx, strike_factor in enumerate(STRIKE_RANGE_FACTORS):
        strike = spot_price * strike_factor
        visible = [False] * (num_strikes * traces_per_strike)

        # Make this strike's traces visible
        visible[idx * traces_per_strike] = True      # P&L curve
        visible[idx * traces_per_strike + 1] = True  # Expiry curve

        step = dict(
            method="update",
            args=[{"visible": visible}],
            label=f"${strike:.0f}"
        )
        steps.append(step)

    slider = SLIDER_DEFAULTS.copy()
    slider.update({
        'active': STRIKE_RANGE_FACTORS.index(1.0) if 1.0 in STRIKE_RANGE_FACTORS else 10,
        'currentvalue': {
            **SLIDER_DEFAULTS['currentvalue'],
            'prefix': 'Strike: '
        },
        'steps': steps
    })

    return slider


def _create_slider(slider_type: str, param_values: list) -> dict:
    """Create the parameter slider configuration."""
    steps = []
    for idx, value in enumerate(param_values):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(param_values) + [True]}],
            label=str(value) if slider_type == "DTE" else f"{value}%"
        )
        step["args"][0]["visible"][idx] = True
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


def render_pnl_tab(
    all_data: dict,
    spot_range: np.ndarray,
    spot_price: float,
    positions: list,
    stock_position,
    default_premium: float,
    risk_free_rate: float = 0.05,
    calculate_all_greeks_func=None,
    calculate_pnl_at_expiry_func=None,
    find_breakeven_func=None
) -> None:
    """
    Render the complete P&L tab content.

    Args:
        all_data: Calculated portfolio data
        spot_range: Array of spot prices
        spot_price: Current spot price
        positions: List of option positions
        stock_position: Stock position or None
        default_premium: Default premium for display
        risk_free_rate: Risk-free interest rate
        calculate_all_greeks_func: Function to calculate Greeks (for strike variation)
        calculate_pnl_at_expiry_func: Function to calculate P&L at expiry (for strike variation)
        find_breakeven_func: Function to find breakeven points (for strike variation)
    """
    from components.metrics import (
        render_metrics_row,
        render_position_info_banner,
        render_chart_controls
    )
    from services.portfolio_calculator import calculate_strike_surfaces

    # Position info banner
    render_position_info_banner(positions, stock_position, default_premium)

    # Detect single-leg position
    is_single_leg = (
        len(positions) == 1 and
        stock_position is None
    ) or (
        len(positions) == 0 and
        stock_position is None
    )  # Default position is also single-leg

    # Extract base data
    pnl_data = all_data['pnl_data']
    breakeven_result = all_data['breakeven_result']

    # Get position details for strike variation
    if positions:
        pos = positions[0]
        option_type = pos.option_type
        position_type = pos.position_type
        quantity = pos.quantity
        base_strike = pos.strike
    else:
        # Default position
        option_type = 'call'
        position_type = 'long'
        quantity = 1
        base_strike = spot_price

    # Chart controls with Strike option for single-leg (render early to get slider type)
    slider_type = render_chart_controls("pnl_slider", is_single_leg=is_single_leg, spot_price=spot_price)

    # Handle Strike variation for single-leg (with Plotly slider)
    if slider_type == "Strike" and is_single_leg and calculate_all_greeks_func:
        # Calculate strike surfaces with breakeven
        strike_pnl_data, _, strike_breakeven_data = calculate_strike_surfaces(
            spot_price=spot_price,
            spot_range=tuple(spot_range),
            option_type=option_type,
            position_type=position_type,
            quantity=quantity,
            base_strike=base_strike,
            risk_free_rate=risk_free_rate,
            _calculate_all_greeks_func=calculate_all_greeks_func,
            _calculate_pnl_at_expiry_func=calculate_pnl_at_expiry_func,
            _find_breakeven_func=find_breakeven_func
        )

        # Create chart with integrated Plotly slider and metrics
        fig = create_pnl_figure_strike(
            pnl_data=strike_pnl_data,
            spot_range=spot_range,
            spot_price=spot_price,
            breakeven_data=strike_breakeven_data,
            option_type=option_type,
            position_type=position_type
        )

        st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})

    else:
        # Standard mode (DTE or IV) - show metrics row
        if breakeven_result:
            max_profit = all_data.get('max_profit_display', breakeven_result.max_profit)
            max_loss = all_data.get('max_loss_display', breakeven_result.max_loss)
            breakeven_points = breakeven_result.breakeven_points
            be_count = len(breakeven_points) if breakeven_points else 0
        else:
            max_profit = max_loss = 0
            breakeven_points = None
            be_count = 0

        # Render metrics
        render_metrics_row(be_count, max_profit, max_loss, breakeven_points)

        st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

        # Create and display chart (DTE or IV)
        fig = create_pnl_figure(
            pnl_data=pnl_data,
            spot_range=spot_range,
            spot_price=spot_price,
            slider_type=slider_type,
            breakeven_result=breakeven_result
        )

        st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})
