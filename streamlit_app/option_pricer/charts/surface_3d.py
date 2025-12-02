"""
3D Surface Chart components for Options Greeks Explorer.

Professional 3D visualization for Greeks surfaces.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import json
from config.constants import (
    GREEK_NAMES,
    GREEK_TITLES,
    CONTRACT_MULTIPLIER
)
from config.chart_theme import (
    SCENE_DEFAULTS,
    SURFACE_COLORSCALES,
    LAYOUT_DEFAULTS,
    CHART_COLORS
)


def create_3d_surface_figure(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    greek_name: str,
    y_label: str,
    colorscale: str = 'Viridis',
    surface_type: str = 'DTE'
) -> go.Figure:
    """
    Create a 3D surface figure for a Greek.

    Args:
        X: Spot price values (x-axis)
        Y: DTE or IV values (y-axis)
        Z: Greek values (z-axis)
        greek_name: Name of the Greek being displayed
        y_label: Label for y-axis
        colorscale: Plotly colorscale name
        surface_type: "DTE" or "IV"

    Returns:
        Plotly Figure object
    """
    greek_title = GREEK_TITLES.get(greek_name, greek_name.capitalize())

    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title=dict(
                text=greek_title,
                font=dict(
                    family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                    size=12,
                    color='#475569'
                )
            ),
            thickness=18,
            len=0.7,
            x=1.02,
            tickfont=dict(
                family='JetBrains Mono, monospace',
                size=10,
                color='#64748b'
            ),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e2e8f0',
            borderwidth=1
        ),
        contours={
            "z": {
                "show": True,
                "usecolormap": True,
                "project": {"z": False}
            }
        },
        hovertemplate=(
            f'<b>Underlying:</b> $%{{x:,.2f}}<br>' +
            f'<b>{y_label}:</b> %{{y:.1f}}<br>' +
            f'<b>{greek_title}:</b> %{{z:.4f}}<br>' +
            '<extra></extra>'
        )
    )])

    # Custom scene based on theme - need deep copy to avoid modifying defaults
    import copy
    scene = copy.deepcopy(SCENE_DEFAULTS)
    scene['xaxis']['title']['text'] = 'Underlying Price ($)'
    scene['yaxis']['title']['text'] = y_label
    scene['zaxis']['title']['text'] = greek_title

    fig.update_layout(
        font=LAYOUT_DEFAULTS['font'],
        paper_bgcolor=LAYOUT_DEFAULTS['paper_bgcolor'],
        scene=scene,
        height=700,
        margin=dict(l=0, r=30, t=30, b=0),
        hoverlabel=LAYOUT_DEFAULTS['hoverlabel']
    )

    return fig


@st.cache_data(ttl=600)
def calculate_3d_surface(
    portfolio_json: str,
    greek_name: str,
    surface_type: str,
    risk_free_rate: float,
    _calculate_greeks_3d_dte_func,
    _calculate_greeks_3d_iv_func
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Calculate 3D surface data for a Greek.

    Args:
        portfolio_json: JSON string of portfolio data
        greek_name: Name of the Greek to calculate
        surface_type: "DTE" or "IV"
        risk_free_rate: Risk-free interest rate
        _calculate_greeks_3d_dte_func: Function for DTE calculations (not hashed)
        _calculate_greeks_3d_iv_func: Function for IV calculations (not hashed)

    Returns:
        Tuple of (X, Y, Z, y_label)
    """
    portfolio_data = json.loads(portfolio_json)

    # Prepare portfolio arrays
    if portfolio_data.get('options') and len(portfolio_data['options']) > 0:
        strikes = np.array([pos['strike'] for pos in portfolio_data['options']])
        option_types = np.array([
            1 if pos['option_type'] == 'call' else 0
            for pos in portfolio_data['options']
        ])
        position_types = np.array([
            1 if pos['position_type'] == 'long' else -1
            for pos in portfolio_data['options']
        ])
        quantities = np.array([
            pos['quantity'] * CONTRACT_MULTIPLIER
            for pos in portfolio_data['options']
        ])
    else:
        strikes = np.array([])
        option_types = np.array([], dtype=np.int32)
        position_types = np.array([], dtype=np.int32)
        quantities = np.array([], dtype=np.int32)

    spot_base = portfolio_data.get('spot_price', 100.0)
    spot_range = np.linspace(spot_base * 0.7, spot_base * 1.3, 100)
    greek_idx = GREEK_NAMES.index(greek_name)

    if surface_type == "DTE":
        dte_range = np.linspace(1, 90, 100)
        matrix_3d = _calculate_greeks_3d_dte_func(
            strikes, option_types, position_types, quantities,
            spot_range, dte_range, risk_free_rate, 0.25
        )
        return spot_range, dte_range, matrix_3d[:, :, greek_idx].T, "DTE (days)"
    else:
        iv_range = np.linspace(0.05, 0.50, 100)
        matrix_3d = _calculate_greeks_3d_iv_func(
            strikes, option_types, position_types, quantities,
            spot_range, 30.0, risk_free_rate, iv_range
        )
        return spot_range, iv_range * 100, matrix_3d[:, :, greek_idx].T, "IV (%)"


def render_3d_tab(
    portfolio_json: str,
    risk_free_rate: float,
    positions: list,
    stock_position,
    _calculate_greeks_3d_dte_func,
    _calculate_greeks_3d_iv_func
) -> None:
    """
    Render the complete 3D surface tab.

    Args:
        portfolio_json: JSON string of portfolio data
        risk_free_rate: Risk-free interest rate
        positions: List of option positions
        stock_position: Stock position or None
        _calculate_greeks_3d_dte_func: Function for DTE calculations
        _calculate_greeks_3d_iv_func: Function for IV calculations
    """
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

    # Controls in a clean card
    st.markdown("""
    <div style="
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
    ">
        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b; margin-bottom: 0.75rem; font-weight: 600;">
            Surface Configuration
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        selected_greek = st.selectbox(
            "Greek",
            options=GREEK_NAMES,
            format_func=lambda x: GREEK_TITLES.get(x, x.capitalize()),
            index=1,  # Default to delta
            help="Select which Greek to visualize"
        )

    with col2:
        surface_type = st.radio(
            "Y-Axis Parameter",
            ["DTE", "IV"],
            format_func=lambda x: "Days to Expiration" if x == "DTE" else "Implied Volatility",
            horizontal=True
        )

    with col3:
        colorscale = st.selectbox(
            "Color Scheme",
            SURFACE_COLORSCALES['alternatives'] + [SURFACE_COLORSCALES['default']],
            index=len(SURFACE_COLORSCALES['alternatives']),  # Default to Viridis
            help="Choose the color palette for the surface"
        )

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    # Calculate surface
    with st.spinner(f'Calculating 3D surface for {GREEK_TITLES.get(selected_greek, selected_greek)}...'):
        X, Y, Z, y_label = calculate_3d_surface(
            portfolio_json,
            selected_greek,
            surface_type,
            risk_free_rate,
            _calculate_greeks_3d_dte_func,
            _calculate_greeks_3d_iv_func
        )

    # Create and display figure
    fig = create_3d_surface_figure(
        X, Y, Z,
        selected_greek,
        y_label,
        colorscale,
        surface_type
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

    # Info boxes
    _render_3d_info_boxes(selected_greek, y_label, positions, stock_position)


def _render_3d_info_boxes(
    selected_greek: str,
    y_label: str,
    positions: list,
    stock_position
) -> None:
    """Render information boxes below the 3D chart."""
    col1, col2 = st.columns(2)

    option_count = len(positions) if positions else (0 if stock_position else 1)
    stock_count = 1 if stock_position else 0
    greek_title = GREEK_TITLES.get(selected_greek, selected_greek.capitalize())

    with col1:
        st.markdown(f"""
        <div style="
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1rem 1.25rem;
        ">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b; margin-bottom: 0.5rem; font-weight: 600;">
                Current Parameters
            </div>
            <div style="font-size: 0.85rem; color: #334155; line-height: 1.8;">
                <div><span style="color: #64748b;">Greek:</span> <strong>{greek_title}</strong></div>
                <div><span style="color: #64748b;">Y-axis:</span> <strong>{y_label}</strong></div>
                <div><span style="color: #64748b;">Options:</span> <strong>{option_count}</strong></div>
                <div><span style="color: #64748b;">Stock:</span> <strong>{'Yes' if stock_count else 'No'}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
            border: 1px solid #bbf7d0;
            border-radius: 10px;
            padding: 1rem 1.25rem;
        ">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; color: #166534; margin-bottom: 0.5rem; font-weight: 600;">
                3D Navigation
            </div>
            <div style="font-size: 0.85rem; color: #166534; line-height: 1.8;">
                <div><strong>Rotate:</strong> Left click + drag</div>
                <div><strong>Zoom:</strong> Scroll wheel</div>
                <div><strong>Pan:</strong> Right click + drag</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
