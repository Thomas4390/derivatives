"""
P&L Chart components for Options Greeks Explorer.

Professional, clean chart design with interactive features.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from .slider_factory import create_param_slider, find_default_param_value
from config.chart_theme import (
    AXIS_DEFAULTS,
    CHART_COLORS,
    LINE_STYLES,
    SLIDER_DEFAULTS,
    get_layout_config,
)
from config.constants import DTE_RANGE, IV_RANGE, STRIKE_RANGE_FACTORS


def create_pnl_figure(
    pnl_data: dict,
    spot_range: np.ndarray,
    spot_price: float,
    slider_type: str,
    breakeven_result=None,
    dte_range=None,
    iv_range=None,
    scenario_expiries=None,
) -> go.Figure:
    """
    Create the P&L profile figure with interactive slider.

    Args:
        pnl_data: Dictionary with P&L data for different DTE/IV combinations
        spot_range: Array of spot prices for x-axis
        spot_price: Current spot price for reference line
        slider_type: "DTE" or "IV" for parameter variation
        breakeven_result: BreakevenResult object with breakeven points
        dte_range: Custom DTE range (defaults to DTE_RANGE)
        iv_range: Custom IV range (defaults to IV_RANGE)

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    effective_dte_range = dte_range or DTE_RANGE
    effective_iv_range = iv_range or IV_RANGE

    param_values = effective_dte_range if slider_type == "DTE" else effective_iv_range

    # Default slider value (DTE or IV being varied)
    default_value, _ = find_default_param_value(slider_type, param_values)

    # Fixed value for the other axis (not varied by the slider)
    if slider_type == "DTE":
        _, fixed_iv_idx = find_default_param_value("IV", effective_iv_range)
        fixed_value = effective_iv_range[fixed_iv_idx]
    else:
        # Fixed DTE: prefer 31 if available, otherwise first >= 90, otherwise middle
        if 31 in effective_dte_range:
            fixed_value = 31
        else:
            dte_ge_90 = [d for d in effective_dte_range if d >= 90]
            fixed_value = (
                dte_ge_90[0]
                if dte_ge_90
                else effective_dte_range[len(effective_dte_range) // 2]
            )

    # Add traces for each parameter value
    for value in param_values:
        key = (
            f"{value}_{fixed_value}"
            if slider_type == "DTE"
            else f"{fixed_value}_{value}"
        )
        visible = value == default_value

        # Create hover template matching Greeks format
        if slider_type == "DTE":
            hover_template = (
                "<b>Underlying:</b> $%{x:,.2f}<br>"
                + f"<b>DTE:</b> {value} days<br>"
                + "<b>P&L:</b> $%{y:,.2f}<br>"
                + "<extra></extra>"
            )
            trace_name = f"P&L: DTE={value}"
        else:
            hover_template = (
                "<b>Underlying:</b> $%{x:,.2f}<br>"
                + f"<b>IV:</b> {value}%<br>"
                + "<b>P&L:</b> $%{y:,.2f}<br>"
                + "<extra></extra>"
            )
            trace_name = f"P&L: IV={value}%"

        fig.add_trace(
            go.Scatter(
                x=spot_range,
                y=pnl_data[key],
                mode="lines",
                name=trace_name,
                visible=visible,
                line=dict(
                    width=LINE_STYLES["primary"]["width"], color=CHART_COLORS["primary"]
                ),
                hovertemplate=hover_template,
                fill="tozeroy",
                fillcolor="rgba(26, 54, 93, 0.08)",
            )
        )

    # Add expiration curve(s). For discrete-event exotics both conditional
    # outcomes are overlaid: each is NaN-masked to its feasible region, so the
    # two dashed curves tile the spot axis (no toggle needed). Otherwise a
    # single expiry curve is drawn.
    if scenario_expiries:
        _SCENARIO_COLORS = ("#7c3aed", "#ea580c")  # purple, orange — distinct
        for _idx, (_label, _curve) in enumerate(scenario_expiries):
            fig.add_trace(
                go.Scatter(
                    x=spot_range,
                    y=_curve,
                    mode="lines",
                    name=f"Expiry — {_label}",
                    visible=True,
                    connectgaps=False,
                    line=dict(
                        color=_SCENARIO_COLORS[_idx % len(_SCENARIO_COLORS)],
                        width=LINE_STYLES["expiry"]["width"],
                        dash=LINE_STYLES["expiry"]["dash"],
                    ),
                    hovertemplate=(
                        f"<b>{_label}</b><br>"
                        + "<b>Underlying:</b> $%{x:,.2f}<br>"
                        + "<b>P&L at Expiry:</b> $%{y:,.2f}<br>"
                        + "<extra></extra>"
                    ),
                )
            )
    else:
        # Single expiry curve (use dense grid for sharp payoff kinks).
        expiry_x = pnl_data.get("expiry_dense_x", spot_range)
        expiry_y = pnl_data.get("expiry_dense_y", pnl_data["expiry"])
        fig.add_trace(
            go.Scatter(
                x=expiry_x,
                y=expiry_y,
                mode="lines",
                name="P&L at Expiration",
                visible=True,
                line=dict(
                    color=LINE_STYLES["expiry"]["color"],
                    width=LINE_STYLES["expiry"]["width"],
                    dash=LINE_STYLES["expiry"]["dash"],
                ),
                hovertemplate=(
                    "<b>Underlying:</b> $%{x:,.2f}<br>"
                    + "<b>P&L at Expiry:</b> $%{y:,.2f}<br>"
                    + "<extra></extra>"
                ),
            )
        )

    # Add breakeven lines (pass spot_price for smart positioning)
    _add_breakeven_lines(fig, breakeven_result, spot_price)

    # Add reference lines
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color=CHART_COLORS["neutral"],
        line_width=1,
        opacity=0.6,
    )
    fig.add_vline(
        x=spot_price,
        line_dash="dot",
        line_color=CHART_COLORS["accent"],
        line_width=1.5,
        opacity=0.8,
    )

    # Add Current Price annotation with box style
    fig.add_annotation(
        x=spot_price,
        y=1.02,
        xref="x",
        yref="paper",
        text="Current Price",
        showarrow=False,
        font=dict(size=10, color=CHART_COLORS["accent"], weight="bold"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=CHART_COLORS["accent"],
        borderwidth=1,
        borderpad=3,
    )

    # Create slider
    slider = create_param_slider(
        slider_type, param_values, default_value, traces_per_step=1, num_subplots=1
    )

    # Update layout
    layout = get_layout_config(height=650)
    yaxis_config = {
        **AXIS_DEFAULTS,
        "title": {"text": "Profit / Loss ($)", **AXIS_DEFAULTS["title"]},
        "tickprefix": "$",
        "tickformat": ",.0f",
    }

    # Structured products: compute a FIXED Y-axis range across all DTE/IV steps
    # so the axes don't jump when the slider moves.
    if dte_range is not None:
        global_min = 0.0
        global_max = 0.0
        expiry_arr = pnl_data.get("expiry_dense_y", pnl_data.get("expiry", np.zeros(1)))
        global_min = min(global_min, float(np.min(expiry_arr)))
        global_max = max(global_max, float(np.max(expiry_arr)))
        for value in param_values:
            key = (
                f"{value}_{fixed_value}"
                if slider_type == "DTE"
                else f"{fixed_value}_{value}"
            )
            arr = pnl_data.get(key, np.zeros(1))
            global_min = min(global_min, float(np.min(arr)))
            global_max = max(global_max, float(np.max(arr)))
        pad = max((global_max - global_min) * 0.08, 20)
        yaxis_config["autorange"] = False
        yaxis_config["range"] = [global_min - pad, global_max + pad]

    layout.update(
        {
            "sliders": [slider],
            "xaxis": {
                **AXIS_DEFAULTS,
                "title": {"text": "Underlying Price", **AXIS_DEFAULTS["title"]},
                "tickprefix": "$",
                "tickformat": ",.0f",
            },
            "yaxis": yaxis_config,
            "margin": {"l": 70, "r": 40, "t": 40, "b": 100},
            "showlegend": True,
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
                "font": {"size": 11},
            },
        }
    )

    fig.update_layout(**layout)

    return fig


def _add_breakeven_lines(
    fig: go.Figure, breakeven_result, spot_price: float = None
) -> None:
    """Add breakeven vertical lines to the figure with label just below Current Price."""
    if not breakeven_result or not breakeven_result.breakeven_points:
        return

    breakeven_points = sorted(breakeven_result.breakeven_points)

    for i, be in enumerate(breakeven_points):
        label = (
            f"BE: ${be:,.0f}"
            if len(breakeven_points) == 1
            else f"BE{i + 1}: ${be:,.0f}"
        )

        # Add the vertical line (stops before the label box at y=0.90)
        fig.add_shape(
            type="line",
            x0=be,
            x1=be,
            y0=0,
            y1=0.90,
            xref="x",
            yref="paper",
            line=dict(color=CHART_COLORS["breakeven"], width=1.5, dash="dash"),
        )

        # Add annotation just below Current Price level (y=0.94)
        fig.add_annotation(
            x=be,
            y=0.94,
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            font=dict(size=10, color=CHART_COLORS["breakeven"], weight="bold"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=CHART_COLORS["breakeven"],
            borderwidth=1,
            borderpad=3,
        )


def create_pnl_figure_strike(
    pnl_data: dict,
    spot_range: np.ndarray,
    spot_price: float,
    breakeven_data: dict = None,
    option_type: str = "call",
    position_type: str = "long",
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

    expiry_by_strike = pnl_data.get("expiry_by_strike", {})

    # Determine unlimited profit/loss based on position type
    if option_type == "call":
        unlimited_profit = position_type == "long"
        unlimited_loss = position_type == "short"
    else:
        unlimited_profit = False
        unlimited_loss = False

    # Default strike index (ATM = 100%)
    default_idx = STRIKE_RANGE_FACTORS.index(1.0) if 1.0 in STRIKE_RANGE_FACTORS else 10

    # Add traces for each strike (P&L at 31 DTE + Expiry)
    for idx, strike_factor in enumerate(STRIKE_RANGE_FACTORS):
        key = f"strike_{int(strike_factor * 100)}"
        strike_price_val = spot_price * strike_factor
        moneyness = (
            "ATM" if strike_factor == 1.0 else ("ITM" if strike_factor < 1.0 else "OTM")
        )

        visible = idx == default_idx

        # P&L curve at 31 DTE - full hover template
        fig.add_trace(
            go.Scatter(
                x=spot_range,
                y=pnl_data[key],
                mode="lines",
                name="P&L (31 DTE)",
                visible=visible,
                line=dict(
                    width=LINE_STYLES["primary"]["width"], color=CHART_COLORS["primary"]
                ),
                hovertemplate=(
                    "<b>Underlying:</b> $%{x:,.2f}<br>"
                    + f"<b>Strike:</b> ${strike_price_val:.2f} ({moneyness})<br>"
                    + "<b>P&L (31 DTE):</b> $%{y:,.2f}<br>"
                    + "<extra></extra>"
                ),
                fill="tozeroy",
                fillcolor="rgba(26, 54, 93, 0.08)",
            )
        )

        # Expiry curve - full hover template
        expiry_pnl = expiry_by_strike.get(key, [])
        fig.add_trace(
            go.Scatter(
                x=spot_range,
                y=expiry_pnl,
                mode="lines",
                name="At Expiration",
                visible=visible,
                line=dict(
                    color=LINE_STYLES["expiry"]["color"],
                    width=LINE_STYLES["expiry"]["width"],
                    dash=LINE_STYLES["expiry"]["dash"],
                ),
                hovertemplate=(
                    "<b>Underlying:</b> $%{x:,.2f}<br>"
                    + f"<b>Strike:</b> ${strike_price_val:.2f} ({moneyness})<br>"
                    + "<b>P&L at Expiry:</b> $%{y:,.2f}<br>"
                    + "<extra></extra>"
                ),
            )
        )

    # Add reference lines (always visible)
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color=CHART_COLORS["neutral"],
        line_width=1,
        opacity=0.6,
    )
    fig.add_vline(
        x=spot_price,
        line_dash="dot",
        line_color=CHART_COLORS["accent"],
        line_width=1.5,
        opacity=0.8,
    )

    # Add Current Price annotation with box style
    fig.add_annotation(
        x=spot_price,
        y=1.02,
        xref="x",
        yref="paper",
        text="Current Price",
        showarrow=False,
        font=dict(size=10, color=CHART_COLORS["accent"], weight="bold"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=CHART_COLORS["accent"],
        borderwidth=1,
        borderpad=3,
    )

    # Create slider steps with dynamic annotations for metrics
    steps = []
    num_strikes = len(STRIKE_RANGE_FACTORS)
    traces_per_strike = 2  # P&L + Expiry

    for idx, strike_factor in enumerate(STRIKE_RANGE_FACTORS):
        key = f"strike_{int(strike_factor * 100)}"
        strike_price_val = spot_price * strike_factor
        moneyness = (
            "ATM" if strike_factor == 1.0 else ("ITM" if strike_factor < 1.0 else "OTM")
        )

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
        visible[idx * traces_per_strike] = True  # P&L curve
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
                font=dict(size=10, color="#8b5cf6", weight="bold"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#8b5cf6",
                borderwidth=1,
                borderpad=3,
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
                yanchor="top",
            ),
        ]

        # Add breakeven vertical line annotations (just below Current Price/Strike level)
        sorted_be_points = sorted(breakeven_points)
        for i, be in enumerate(sorted_be_points):
            label = (
                f"BE: ${be:,.0f}"
                if len(sorted_be_points) == 1
                else f"BE{i + 1}: ${be:,.0f}"
            )

            annotations.append(
                dict(
                    x=be,
                    y=0.94,
                    xref="x",
                    yref="paper",
                    text=label,
                    showarrow=False,
                    font=dict(size=10, color=CHART_COLORS["breakeven"], weight="bold"),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor=CHART_COLORS["breakeven"],
                    borderwidth=1,
                    borderpad=3,
                )
            )

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
                opacity=0.8,
            )
        ]

        # Add breakeven vertical lines (stop at y=0.90, before label box)
        for be in breakeven_points:
            shapes.append(
                dict(
                    type="line",
                    x0=be,
                    x1=be,
                    y0=0,
                    y1=0.90,
                    xref="x",
                    yref="paper",
                    line=dict(color=CHART_COLORS["breakeven"], width=1.5, dash="dash"),
                )
            )

        step = dict(
            method="update",
            args=[{"visible": visible}, {"annotations": annotations, "shapes": shapes}],
            label=f"${strike_price_val:.0f}",
        )
        steps.append(step)

    # Create slider using standard defaults for consistency
    slider = SLIDER_DEFAULTS.copy()
    slider.update(
        {
            "active": default_idx,
            "currentvalue": {
                **SLIDER_DEFAULTS["currentvalue"],
                "prefix": "Strike Price: ",
            },
            "steps": steps,
        }
    )

    # Get initial annotations and shapes for default strike
    default_step = steps[default_idx]
    initial_annotations = default_step["args"][1]["annotations"]
    initial_shapes = default_step["args"][1]["shapes"]

    # Update layout
    layout = get_layout_config(height=700)
    layout.update(
        {
            "sliders": [slider],
            "annotations": initial_annotations,
            "shapes": initial_shapes,
            "xaxis": {
                **AXIS_DEFAULTS,
                "title": {"text": "Underlying Price", **AXIS_DEFAULTS["title"]},
                "tickprefix": "$",
                "tickformat": ",.0f",
            },
            "yaxis": {
                **AXIS_DEFAULTS,
                "title": {"text": "Profit / Loss ($)", **AXIS_DEFAULTS["title"]},
                "tickprefix": "$",
                "tickformat": ",.0f",
            },
            "margin": {"l": 70, "r": 40, "t": 60, "b": 100},
            "showlegend": True,
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
                "font": {"size": 11},
            },
        }
    )

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
        visible[idx * traces_per_strike] = True  # P&L curve
        visible[idx * traces_per_strike + 1] = True  # Expiry curve

        step = dict(
            method="update", args=[{"visible": visible}], label=f"${strike:.0f}"
        )
        steps.append(step)

    slider = SLIDER_DEFAULTS.copy()
    slider.update(
        {
            "active": STRIKE_RANGE_FACTORS.index(1.0)
            if 1.0 in STRIKE_RANGE_FACTORS
            else 10,
            "currentvalue": {**SLIDER_DEFAULTS["currentvalue"], "prefix": "Strike: "},
            "steps": steps,
        }
    )

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
    find_breakeven_func=None,
    has_exotic_legs: bool = False,
    sp_mode: bool = False,
    dte_range=None,
    iv_range=None,
    dividend_yield: float = 0.0,
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
    # Empty portfolio: draw nothing (not a flat zero line). Show a short hint so
    # the tab reads as "waiting for input" rather than broken. Comes before the
    # banner/controls so nothing else renders on a blank portfolio.
    if len(positions) == 0 and stock_position is None:
        st.info(
            "No positions — add a leg in the sidebar to see the payoff diagram.",
            icon="ℹ️",
        )
        return

    from components.metrics import (
        render_chart_controls,
        render_metrics_row,
        render_position_info_banner,
    )
    from services.portfolio_calculator import calculate_strike_surfaces

    # Position info banner (skip for structured products)
    if not sp_mode:
        render_position_info_banner(positions, stock_position, default_premium)

    # Discrete-event path-dependent legs: both conditional outcomes are drawn
    # on the payoff chart together (see below), so no per-leg toggle is needed.
    has_discrete_event = False
    discrete_event_legs: list[int] = []
    if not sp_mode and has_exotic_legs:
        from config.exotic_config import PAYOFF_SCENARIOS

        for i, pos in enumerate(positions):
            spec = PAYOFF_SCENARIOS.get(pos.get("instrument_class"))
            if spec and spec.get("kind") == "discrete_event":
                has_discrete_event = True
                discrete_event_legs.append(i)
        if has_discrete_event:
            st.caption(
                "Path-dependent legs: both conditional outcomes are overlaid on "
                "the payoff chart below (each shown only over the region where it "
                "can occur), so you can compare them without toggling."
            )

    # Detect single-leg position (structured products are never single-leg)
    is_single_leg = not sp_mode and (
        (len(positions) == 1 and stock_position is None)
        or (len(positions) == 0 and stock_position is None)
    )

    # Extract base data
    pnl_data = all_data["pnl_data"]
    breakeven_result = all_data["breakeven_result"]

    # For discrete-event legs, compute BOTH conditional outcomes and overlay
    # them on the payoff chart instead of toggling one at a time. Each outcome
    # curve is NaN-masked to its feasible region, so the two dashed curves tile
    # the spot axis. The intermediate-DTE (priced) slider curves stay on the
    # scenario-agnostic base, so the time slider is unaffected; the metrics are
    # recombined from the overlay curves below.
    scenario_expiries: list[tuple[str, np.ndarray]] | None = None
    if has_discrete_event and not sp_mode and calculate_pnl_at_expiry_func is not None:
        from services.portfolio_calculator import (
            _calculate_expiry_pnl,
            prepare_portfolio_arrays,
        )
        from services.scenario_labels import scenario_options

        scenario_expiries = []
        for scen_idx in (0, 1):
            # Copy the legs so the live positions are never mutated; set every
            # discrete-event leg to its scenario ``scen_idx``. The contextual
            # scenario_options labels (same source as the per-leg payoff
            # diagram) keep the two outcome names identical across tabs.
            scen_positions = [dict(p) for p in positions]
            labels: list[str] = []
            for i in discrete_event_legs:
                key, label = scenario_options(scen_positions[i])[scen_idx]
                scen_positions[i]["scenario"] = key
                labels.append(label)
            scenario_pdata = {
                "spot_price": spot_price,
                "options": scen_positions,
                "stock": stock_position,
            }
            (
                _s_strikes,
                _s_otypes,
                _s_ptypes,
                _s_qty,
                _s_prem,
                _s_stock_qty,
                _s_stock_entry,
                _s_meta,
            ) = prepare_portfolio_arrays(scenario_pdata)
            curve = _calculate_expiry_pnl(
                spot_range,
                _s_strikes,
                _s_otypes,
                _s_ptypes,
                _s_qty,
                _s_prem,
                _s_stock_qty,
                _s_stock_entry,
                calculate_pnl_at_expiry_func,
                exotic_metadata=_s_meta,
                portfolio_data=scenario_pdata,
            )
            scenario_expiries.append((" · ".join(labels), curve))

    # For a single terminal exotic leg with a tunable primary parameter, a
    # slider sweeps it so you can see the effect on the terminal payoff (current
    # vs what-if), overlaid on the same chart. The stored leg is never changed —
    # the sweep is a non-destructive "what-if". Skipped for discrete-event legs
    # (already showing both outcomes) and multi-leg / stock positions.
    if (
        scenario_expiries is None
        and not sp_mode
        and has_exotic_legs
        and len(positions) == 1
        and stock_position is None
        and calculate_pnl_at_expiry_func is not None
    ):
        from config.exotic_config import EXOTIC_SWEEP_PARAM

        _leg = positions[0]
        _sweep = EXOTIC_SWEEP_PARAM.get(_leg.get("instrument_class"))
        if _sweep is not None:
            _cur = min(
                max(float(_leg.get(_sweep["key"], _sweep["default"])), _sweep["lo"]),
                _sweep["hi"],
            )
            _val = st.slider(
                _sweep["label"],
                _sweep["lo"],
                _sweep["hi"],
                _cur,
                _sweep["step"],
                key="exotic_param_sweep",
                help="Sweep this parameter to see how the terminal payoff "
                "changes. Your leg is left unchanged.",
            )
            if abs(_val - _cur) > 1e-9:
                from services.portfolio_calculator import (
                    _calculate_expiry_pnl,
                    prepare_portfolio_arrays,
                )

                def _sweep_expiry(_param_value: float) -> np.ndarray:
                    _mod = dict(_leg)
                    _mod[_sweep["key"]] = _param_value
                    _pd = {"spot_price": spot_price, "options": [_mod], "stock": None}
                    (_a, _b, _c, _d, _e, _f, _g, _m) = prepare_portfolio_arrays(_pd)
                    return _calculate_expiry_pnl(
                        spot_range,
                        _a,
                        _b,
                        _c,
                        _d,
                        _e,
                        _f,
                        _g,
                        calculate_pnl_at_expiry_func,
                        exotic_metadata=_m,
                        portfolio_data=_pd,
                    )

                _short = _sweep["short"]
                scenario_expiries = [
                    (f"{_short} = {_cur:g} (current)", _sweep_expiry(_cur)),
                    (f"{_short} = {_val:g}", _sweep_expiry(_val)),
                ]

    # Get position details for strike variation
    if positions:
        pos = positions[0]
        option_type = pos["option_type"]
        position_type = pos["position_type"]
        quantity = pos["quantity"]
        base_strike = pos["strike"]
    else:
        # Default position
        option_type = "call"
        position_type = "long"
        quantity = 1
        base_strike = spot_price

    # Chart controls with Strike option for single-leg (render early to get slider type)
    slider_type = render_chart_controls(
        "pnl_slider", is_single_leg=is_single_leg, spot_price=spot_price
    )

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
            _find_breakeven_func=find_breakeven_func,
            dividend_yield=dividend_yield,
        )

        # Create chart with integrated Plotly slider and metrics
        fig = create_pnl_figure_strike(
            pnl_data=strike_pnl_data,
            spot_range=spot_range,
            spot_price=spot_price,
            breakeven_data=strike_breakeven_data,
            option_type=option_type,
            position_type=position_type,
        )

        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    else:
        # Standard mode (DTE or IV) - show metrics row
        if breakeven_result:
            max_profit = all_data.get("max_profit_display", breakeven_result.max_profit)
            max_loss = all_data.get("max_loss_display", breakeven_result.max_loss)
            breakeven_points = breakeven_result.breakeven_points
            if has_discrete_event and scenario_expiries:
                # The chart overlays the two conditional outcome curves, so the
                # metrics must describe those curves too — not the
                # scenario-agnostic base. ±inf (unlimited) from the base flags
                # is kept: the grid-bounded curves cannot see an unbounded tail.
                from services.portfolio_calculator import combine_scenario_metrics

                combined = combine_scenario_metrics(
                    [curve for _, curve in scenario_expiries], spot_range
                )
                if combined is not None:
                    if max_profit != float("inf"):
                        max_profit = combined.max_profit
                    if max_loss != float("-inf"):
                        max_loss = combined.max_loss
                    breakeven_points = combined.breakeven_points
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
            breakeven_result=breakeven_result,
            dte_range=dte_range,
            iv_range=iv_range,
            scenario_expiries=scenario_expiries,
        )

        if has_discrete_event and not sp_mode:
            # Both outcome curves are drawn and NaN-masked to their feasible
            # regions, so they already tile the axis — no infeasible shading.
            from charts._exotic_annotations import add_barrier_markers

            add_barrier_markers(fig, positions)

        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
