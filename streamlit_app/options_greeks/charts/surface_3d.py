"""
3D Surface Chart components for Options Greeks Explorer.

Professional 3D visualization for Greeks surfaces.
"""

import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from config.chart_theme import LAYOUT_DEFAULTS, SCENE_DEFAULTS, SURFACE_COLORSCALES
from config.constants import GREEK_NAMES, GREEK_TITLES, STRIKE_RANGE_FACTORS


def create_3d_surface_figure(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    greek_name: str,
    y_label: str,
    colorscale: str = "Viridis",
    surface_type: str = "DTE",
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

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text=greek_title,
                        font=dict(
                            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                            size=12,
                            color="#475569",
                        ),
                    ),
                    thickness=18,
                    len=0.7,
                    x=1.02,
                    tickfont=dict(
                        family="JetBrains Mono, monospace", size=10, color="#64748b"
                    ),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0",
                    borderwidth=1,
                ),
                contours={
                    "z": {"show": True, "usecolormap": True, "project": {"z": False}}
                },
                hovertemplate=(
                    "<b>Underlying:</b> $%{x:,.2f}<br>"
                    + f"<b>{y_label}:</b> %{{y:.1f}}<br>"
                    + f"<b>{greek_title}:</b> %{{z:.4f}}<br>"
                    + "<extra></extra>"
                ),
            )
        ]
    )

    # Custom scene based on theme - need deep copy to avoid modifying defaults
    import copy

    scene = copy.deepcopy(SCENE_DEFAULTS)
    scene["xaxis"]["title"]["text"] = "Underlying Price ($)"
    scene["yaxis"]["title"]["text"] = y_label
    scene["zaxis"]["title"]["text"] = greek_title

    fig.update_layout(
        font=LAYOUT_DEFAULTS["font"],
        paper_bgcolor=LAYOUT_DEFAULTS["paper_bgcolor"],
        scene=scene,
        height=700,
        margin=dict(l=0, r=30, t=30, b=0),
        hoverlabel=LAYOUT_DEFAULTS["hoverlabel"],
    )

    return fig


@st.cache_data(ttl=600)
def calculate_3d_surface(
    portfolio_json: str,
    greek_name: str,
    surface_type: str,
    risk_free_rate: float,
    _calculate_greeks_3d_dte_func,
    _calculate_greeks_3d_iv_func,
    dividend_yield: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Calculate 3D surface data for a Greek (DTE or IV mode).

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
    spot_base = portfolio_data.get("spot_price", portfolio_data.get("spot", 100.0))
    spot_range = np.linspace(spot_base * 0.7, spot_base * 1.3, 100)
    greek_idx = GREEK_NAMES.index(greek_name)

    if surface_type == "DTE":
        max_dte_days = 90
        product_params_json = portfolio_data.get("product_params_json")
        if product_params_json:
            pp = (
                json.loads(product_params_json)
                if isinstance(product_params_json, str)
                else product_params_json
            )
            max_dte_days = min(int(pp.get("maturity", 0.25) * 365), 90)
        dte_range = np.linspace(1, max_dte_days, 100)
        # New signature: (portfolio_json, spot_range, dte_range, risk_free_rate, base_iv, greek_index)
        matrix_2d = _calculate_greeks_3d_dte_func(
            portfolio_json,
            spot_range,
            dte_range,
            risk_free_rate,
            0.25,  # base_iv
            greek_idx,
            dividend_yield,
        )
        return spot_range, dte_range, matrix_2d.T, "DTE (days)"
    iv_range = np.linspace(0.05, 0.50, 100)
    # New signature: (portfolio_json, spot_range, iv_range, risk_free_rate, base_dte, greek_index)
    matrix_2d = _calculate_greeks_3d_iv_func(
        portfolio_json,
        spot_range,
        iv_range,
        risk_free_rate,
        30.0,  # base_dte
        greek_idx,
        dividend_yield,
    )
    return spot_range, iv_range * 100, matrix_2d.T, "IV (%)"


@st.cache_data(ttl=600)
def calculate_3d_surface_strike(
    spot_price: float,
    greek_name: str,
    option_type: str,
    position_type: str,
    quantity: int,
    dte: float,
    risk_free_rate: float,
    volatility: float,
    _calculate_greeks_3d_strike_func,
    dividend_yield: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Calculate 3D surface data for a Greek varying by Strike (single-leg only).

    Args:
        spot_price: Current spot price
        greek_name: Name of the Greek to calculate
        option_type: 'call' or 'put'
        position_type: 'long' or 'short'
        quantity: Number of contracts
        dte: Days to expiration
        risk_free_rate: Risk-free interest rate
        volatility: Implied volatility (decimal)
        _calculate_greeks_3d_strike_func: Function for Strike calculations

    Returns:
        Tuple of (X, Y, Z, y_label)
    """
    spot_range = np.linspace(spot_price * 0.7, spot_price * 1.3, 100)
    strike_range = np.array([spot_price * f for f in STRIKE_RANGE_FACTORS])

    greek_idx = GREEK_NAMES.index(greek_name)

    # Build a portfolio JSON with the single position for the new function signature
    portfolio_data = {
        "spot_price": spot_price,
        "options": [
            {
                "option_type": option_type,
                "position_type": position_type,
                "strike": spot_price,  # Will be varied
                "quantity": quantity,
            }
        ],
        "stock": None,
    }
    portfolio_json = json.dumps(portfolio_data)

    # New signature: (portfolio_json, spot_range, strike_range, risk_free_rate, base_iv, base_dte, greek_index)
    matrix_2d = _calculate_greeks_3d_strike_func(
        portfolio_json,
        spot_range,
        strike_range,
        risk_free_rate,
        volatility,
        dte,
        greek_idx,
        dividend_yield,
    )

    # Convert strike range to percentage of spot for better visualization
    strike_pct = np.array([f * 100 for f in STRIKE_RANGE_FACTORS])

    return spot_range, strike_pct, matrix_2d.T, "Strike (% of Spot)"


def render_3d_tab(
    portfolio_json: str,
    spot_price: float,
    risk_free_rate: float,
    volatility: float,
    dte: float,
    positions: list,
    stock_position,
    _calculate_greeks_3d_dte_func,
    _calculate_greeks_3d_iv_func,
    _calculate_greeks_3d_strike_func=None,
    dividend_yield: float = 0.0,
) -> None:
    """
    Render the complete 3D surface tab.

    Args:
        portfolio_json: JSON string of portfolio data
        spot_price: Current spot price
        risk_free_rate: Risk-free interest rate
        volatility: Implied volatility (decimal)
        dte: Days to expiration
        positions: List of option positions
        stock_position: Stock position or None
        _calculate_greeks_3d_dte_func: Function for DTE calculations
        _calculate_greeks_3d_iv_func: Function for IV calculations
        _calculate_greeks_3d_strike_func: Function for Strike calculations (single-leg only)
    """
    # Detect single-leg vanilla position (Strike variation only for vanilla)
    is_single_vanilla_leg = (
        len(positions) == 1
        and stock_position is None
        and positions[0].get("instrument_class", "vanilla") == "vanilla"
    ) or (len(positions) == 0 and stock_position is None)
    is_single_exotic_leg = (
        len(positions) == 1
        and stock_position is None
        and positions[0].get("instrument_class", "vanilla") != "vanilla"
    )
    is_single_leg = is_single_vanilla_leg

    if not positions and not stock_position:
        st.markdown(
            """
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
                <div style="font-weight: 600; color: #0c4a6e; font-size: 0.9rem;">No Positions</div>
                <div style="color: #0369a1; font-size: 0.8rem;">Add positions using the sidebar to begin analysis</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Get position details for strike variation
    if positions:
        pos = positions[0]
        option_type = pos["option_type"]
        position_type = pos["position_type"]
        quantity = pos["quantity"]
    else:
        # Default position
        option_type = "call"
        position_type = "long"
        quantity = 1

    # Surface type toggle buttons (like P&L Profile)
    st.markdown(
        """
    <div style="
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
    ">
        <span style="font-size: 0.8rem; color: #64748b; font-weight: 500;">Vary by:</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize session state for 3D surface type
    if "3d_surface_type" not in st.session_state:
        st.session_state["3d_surface_type"] = "DTE"

    # Create toggle buttons
    if is_single_leg:
        btn_cols = st.columns([1, 1, 1, 3])
        options = ["DTE", "IV", "Strike"]
    else:
        btn_cols = st.columns([1, 1, 4])
        options = ["DTE", "IV"]

    for i, opt in enumerate(options):
        with btn_cols[i]:
            is_selected = st.session_state["3d_surface_type"] == opt
            if st.button(
                opt,
                key=f"3d_surface_btn_{opt}",
                type="primary" if is_selected else "secondary",
                width="stretch",
            ):
                st.session_state["3d_surface_type"] = opt
                st.rerun()

    # Inform user when Strike variation is unavailable for exotic legs
    if is_single_exotic_leg:
        st.caption(
            "Strike variation is not available for exotic options "
            "(DTE and IV variation still work)."
        )

    surface_type = st.session_state["3d_surface_type"]

    # Controls in a clean card
    st.markdown(
        """
    <div style="
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        margin-top: 1rem;
    ">
        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b; margin-bottom: 0.75rem; font-weight: 600;">
            Surface Configuration
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 3])

    with col1:
        selected_greek = st.selectbox(  # pyright: ignore[reportCallIssue]
            "Greek",
            options=GREEK_NAMES,
            format_func=lambda x: GREEK_TITLES.get(x, x.capitalize()),
            index=1,  # Default to delta
            help="Select which Greek to visualize",
        )

    with col2:
        colorscale = st.selectbox(
            "Color Scheme",
            SURFACE_COLORSCALES["alternatives"] + [SURFACE_COLORSCALES["default"]],
            index=len(SURFACE_COLORSCALES["alternatives"]),  # Default to Viridis
            help="Choose the color palette for the surface",
        )

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    # Calculate surface based on type
    with st.spinner(
        f"Calculating 3D surface for {GREEK_TITLES.get(selected_greek, selected_greek)}..."
    ):
        if (
            surface_type == "Strike"
            and is_single_leg
            and _calculate_greeks_3d_strike_func
        ):
            X, Y, Z, y_label = calculate_3d_surface_strike(
                spot_price,
                selected_greek,
                option_type,
                position_type,
                quantity,
                dte,
                risk_free_rate,
                volatility,
                _calculate_greeks_3d_strike_func,
                dividend_yield,
            )
        else:
            X, Y, Z, y_label = calculate_3d_surface(
                portfolio_json,
                selected_greek,
                surface_type if surface_type != "Strike" else "DTE",
                risk_free_rate,
                _calculate_greeks_3d_dte_func,
                _calculate_greeks_3d_iv_func,
                dividend_yield,
            )

    # Create and display figure
    fig = create_3d_surface_figure(
        X, Y, Z, selected_greek, y_label, colorscale, surface_type
    )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": True})

    # Info boxes
    _render_3d_info_boxes(
        selected_greek, y_label, positions, stock_position, surface_type
    )


def _render_3d_info_boxes(
    selected_greek: str,
    y_label: str,
    positions: list,
    stock_position,
    surface_type: str = "DTE",
) -> None:
    """Render information boxes below the 3D chart."""
    col1, col2 = st.columns(2)

    option_count = len(positions) if positions else (0 if stock_position else 1)
    stock_count = 1 if stock_position else 0
    greek_title = GREEK_TITLES.get(selected_greek, selected_greek.capitalize())

    # Variation type description
    variation_desc = {
        "DTE": "Days to Expiration",
        "IV": "Implied Volatility",
        "Strike": "Strike Price",
    }.get(surface_type, surface_type)

    with col1:
        st.markdown(
            f"""
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
                <div><span style="color: #64748b;">Varying:</span> <strong>{variation_desc}</strong></div>
                <div><span style="color: #64748b;">Options:</span> <strong>{option_count}</strong></div>
                <div><span style="color: #64748b;">Stock:</span> <strong>{"Yes" if stock_count else "No"}</strong></div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
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
        """,
            unsafe_allow_html=True,
        )
