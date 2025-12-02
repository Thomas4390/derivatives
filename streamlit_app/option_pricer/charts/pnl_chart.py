"""
P&L Chart components for Options Greeks Explorer.

Professional, clean chart design with interactive features.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from config.constants import DTE_RANGE, IV_RANGE
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

        hover_template = _create_pnl_hover_template(slider_type, value)

        fig.add_trace(go.Scatter(
            x=spot_range,
            y=pnl_data[key],
            mode='lines',
            name=f'{slider_type}={value}',
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
        name='At Expiration',
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

    # Add breakeven lines
    _add_breakeven_lines(fig, breakeven_result)

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
        opacity=0.8,
        annotation_text="Current",
        annotation_position="top",
        annotation_font_size=10,
        annotation_font_color=CHART_COLORS['accent']
    )

    # Create slider
    slider = _create_slider(slider_type, param_values)

    # Update layout
    layout = get_layout_config(height=650)
    layout.update({
        'sliders': [slider],
        'xaxis': {
            **AXIS_DEFAULTS,
            'title': {'text': 'Underlying Price', **AXIS_DEFAULTS['title']}
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


def _create_pnl_hover_template(slider_type: str, value: int) -> str:
    """Create hover template for P&L traces."""
    if slider_type == "DTE":
        return (
            '<b>Underlying:</b> $%{x:,.2f}<br>' +
            f'<b>DTE:</b> {value} days<br>' +
            '<b>P&L:</b> $%{y:,.2f}<br>' +
            '<extra></extra>'
        )
    else:
        return (
            '<b>Underlying:</b> $%{x:,.2f}<br>' +
            f'<b>IV:</b> {value}%<br>' +
            '<b>P&L:</b> $%{y:,.2f}<br>' +
            '<extra></extra>'
        )


def _add_breakeven_lines(fig: go.Figure, breakeven_result) -> None:
    """Add breakeven vertical lines to the figure."""
    if not breakeven_result or not breakeven_result.breakeven_points:
        return

    for i, be in enumerate(sorted(breakeven_result.breakeven_points)):
        label = f"BE: ${be:,.0f}" if len(breakeven_result.breakeven_points) == 1 else f"BE{i+1}"
        fig.add_vline(
            x=be,
            line_dash="dash",
            line_color=CHART_COLORS['breakeven'],
            line_width=1.5,
            annotation_text=label,
            annotation_position="top",
            annotation_font_size=10,
            annotation_font_color=CHART_COLORS['breakeven']
        )


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
    default_premium: float
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
    """
    from components.metrics import (
        render_metrics_row,
        render_position_info_banner,
        render_chart_controls
    )

    # Position info banner
    render_position_info_banner(positions, stock_position, default_premium)

    # Extract data
    pnl_data = all_data['pnl_data']
    breakeven_result = all_data['breakeven_result']

    # Get max profit/loss values
    if breakeven_result:
        max_profit = all_data.get('max_profit_display', breakeven_result.max_profit)
        max_loss = all_data.get('max_loss_display', breakeven_result.max_loss)
        breakeven_points = breakeven_result.breakeven_points
        be_count = len(breakeven_points)
    else:
        max_profit = max_loss = 0
        breakeven_points = None
        be_count = 0

    # Render metrics
    render_metrics_row(be_count, max_profit, max_loss, breakeven_points)

    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    # Chart controls
    slider_type = render_chart_controls("pnl_slider")

    # Create and display chart
    fig = create_pnl_figure(
        pnl_data=pnl_data,
        spot_range=spot_range,
        spot_price=spot_price,
        slider_type=slider_type,
        breakeven_result=breakeven_result
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
