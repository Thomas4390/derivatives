"""
Exotic Options Charts and Tab Orchestrator for Options Greeks Explorer.

Provides Price comparison and Payoff diagram charts for the Exotic Options tab.
Includes interactive DTE/IV slider controls matching the P&L Profile tab pattern.

Author: Thomas Vaudescal
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from config.chart_theme import (
    AXIS_DEFAULTS,
    CHART_COLORS,
    SLIDER_DEFAULTS,
    get_layout_config,
)
from config.constants import (
    DEFAULT_DTE,
    DEFAULT_IV,
    DTE_RANGE,
    EXOTIC_DESCRIPTIONS,
    EXOTIC_TYPE_NAMES,
    IV_RANGE,
    SPOT_RANGE_FACTOR,
    SPOT_RANGE_POINTS,
)
from charts.exotic_payoff_diagram import (  # noqa: F401  (re-export)
    create_exotic_payoff_diagram,
)
from services.exotic_pricing_adapter import (
    calculate_exotic_greeks_surface,
    calculate_vanilla_greeks_surface,
    haug_factory_params,
)

# Import spread_annotations from simulation without triggering simulation's __init__.py.
_helpers_path = str(
    Path(__file__).resolve().parent.parent.parent
    / "simulation"
    / "utils"
    / "chart_helpers.py"
)
_sim_dir = str(Path(__file__).resolve().parent.parent.parent / "simulation")
if _sim_dir not in sys.path:
    sys.path.insert(0, _sim_dir)
_spec = importlib.util.spec_from_file_location("_chart_helpers_exotic", _helpers_path)
_helpers_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers_mod)
spread_annotations = _helpers_mod.spread_annotations

# =============================================================================
# CHART HELPER
# =============================================================================


def _base_layout(title: str, height: int = 500) -> dict:
    """Get a base layout config with standard theme."""
    layout = get_layout_config(title=title, height=height)
    layout["xaxis"]["title"] = {
        "text": "Underlying Price",
        "font": AXIS_DEFAULTS["title"]["font"],
    }
    return layout


def _add_spot_vline(fig: go.Figure, spot_price: float, row=None, col=None):
    """Add a vertical dashed line at current spot."""
    fig.add_vline(
        x=spot_price,
        line_dash="dot",
        line_color=CHART_COLORS["reference"],
        line_width=1,
        row=row,
        col=col,
    )


# =============================================================================
# PRICE COMPARISON CHART (static, no slider — kept for reference/fallback)
# =============================================================================


def create_exotic_price_comparison(
    exotic_data: dict,
    vanilla_data: dict,
    spot_range: np.ndarray,
    spot_price: float,
    config: dict,
) -> go.Figure:
    """Exotic vs vanilla price with shaded fill between curves."""
    layout = _base_layout("Exotic vs Vanilla Price", height=480)
    layout["yaxis"]["title"] = {
        "text": "Option Price",
        "font": AXIS_DEFAULTS["title"]["font"],
    }
    layout["hovermode"] = "x unified"

    fig = go.Figure(layout=layout)

    # Exotic price line
    fig.add_trace(
        go.Scatter(
            x=spot_range,
            y=exotic_data["price"],
            name=config.get("exotic_label", "Exotic"),
            line=dict(color=CHART_COLORS["primary"], width=2.5),
            hovertemplate="<b>Price:</b> %{y:.2f}",
        )
    )

    # Vanilla price line (dashed)
    fig.add_trace(
        go.Scatter(
            x=spot_range,
            y=vanilla_data["price"],
            name="Vanilla",
            line=dict(color=CHART_COLORS["reference"], width=2, dash="dash"),
            hovertemplate="<b>Price:</b> %{y:.2f}",
        )
    )

    # Shaded fill between (use tonexty)
    fig.add_trace(
        go.Scatter(
            x=spot_range,
            y=exotic_data["price"],
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spot_range,
            y=vanilla_data["price"],
            fill="tonexty",
            fillcolor="rgba(26, 54, 93, 0.08)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Spot price vertical line
    _add_spot_vline(fig, spot_price)

    # Barrier level
    if config.get("barrier"):
        fig.add_vline(
            x=config["barrier"],
            line_dash="dash",
            line_color="#dc2626",
            line_width=2,
            annotation_text=f"Barrier = {config['barrier']:.1f}",
            annotation_position="top right",
            annotation_font_color="#dc2626",
        )

    # Digital payout reference line
    if config.get("payout") and config.get("discount_factor"):
        pv = config["payout"] * config["discount_factor"]
        fig.add_hline(
            y=pv,
            line_dash="dot",
            line_color="#0d9488",
            line_width=1.5,
            annotation_text=f"PV(Payout) = {pv:.2f}",
            annotation_position="top right",
            annotation_font_color="#0d9488",
        )

    # Price difference annotation at current spot
    idx = np.argmin(np.abs(spot_range - spot_price))
    exotic_at_spot = exotic_data["price"][idx]
    vanilla_at_spot = vanilla_data["price"][idx]
    diff = exotic_at_spot - vanilla_at_spot
    diff_pct = (diff / vanilla_at_spot * 100) if vanilla_at_spot != 0 else 0
    fig.add_annotation(
        x=spot_price,
        y=exotic_at_spot,
        text=f"Diff: {diff:+.2f} ({diff_pct:+.2f}%)",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        font=dict(size=11, color=CHART_COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=CHART_COLORS["primary"],
    )

    return fig


# =============================================================================
# PRICE COMPARISON WITH SLIDER (DTE or IV variation)
# =============================================================================


def create_exotic_price_comparison_with_slider(
    all_exotic_data: list[dict],
    all_vanilla_data: list[dict],
    param_values: list[int],
    spot_range: np.ndarray,
    spot_price: float,
    config: dict,
    slider_type: str,
) -> go.Figure:
    """Exotic vs vanilla price comparison with Plotly slider for DTE or IV.

    Parameters
    ----------
    all_exotic_data : list[dict]
        Exotic pricing surfaces, one per param value. Each has "price" key.
    all_vanilla_data : list[dict]
        Vanilla pricing surfaces, one per param value. Each has "price" key.
    param_values : list[int]
        DTE values (e.g. [1, 4, 7, ...]) or IV values (e.g. [5, 7, 9, ...]).
    spot_range : np.ndarray
        Array of underlying prices for x-axis.
    spot_price : float
        Current spot price for reference line.
    config : dict
        Chart config with exotic_type, barrier, payout, etc.
    slider_type : str
        "DTE" or "IV".
    """
    fig = go.Figure()

    N = len(param_values)
    default_value = DEFAULT_DTE if slider_type == "DTE" else DEFAULT_IV

    # Find closest default index
    default_idx = 0
    min_dist = abs(param_values[0] - default_value)
    for i, v in enumerate(param_values):
        if abs(v - default_value) < min_dist:
            min_dist = abs(v - default_value)
            default_idx = i

    # Add traces: 2 per param value (exotic line + vanilla line)
    for i, value in enumerate(param_values):
        visible = i == default_idx

        exotic_prices = all_exotic_data[i]["price"]
        vanilla_prices = all_vanilla_data[i]["price"]

        if slider_type == "DTE":
            hover_exotic = (
                "<b>Underlying:</b> $%{x:,.2f}<br>"
                f"<b>DTE:</b> {value} days<br>"
                "<b>Exotic Price:</b> $%{y:.2f}<br>"
                "<extra></extra>"
            )
            hover_vanilla = (
                "<b>Underlying:</b> $%{x:,.2f}<br>"
                f"<b>DTE:</b> {value} days<br>"
                "<b>Vanilla Price:</b> $%{y:.2f}<br>"
                "<extra></extra>"
            )
        else:
            hover_exotic = (
                "<b>Underlying:</b> $%{x:,.2f}<br>"
                f"<b>IV:</b> {value}%<br>"
                "<b>Exotic Price:</b> $%{y:.2f}<br>"
                "<extra></extra>"
            )
            hover_vanilla = (
                "<b>Underlying:</b> $%{x:,.2f}<br>"
                f"<b>IV:</b> {value}%<br>"
                "<b>Vanilla Price:</b> $%{y:.2f}<br>"
                "<extra></extra>"
            )

        # Trace 2*i: exotic price line
        fig.add_trace(
            go.Scatter(
                x=spot_range,
                y=exotic_prices,
                mode="lines",
                name=config.get("exotic_label", "Exotic"),
                visible=visible,
                line=dict(color=CHART_COLORS["primary"], width=2.5),
                hovertemplate=hover_exotic,
                showlegend=(i == default_idx),
            )
        )

        # Trace 2*i+1: vanilla price line (dashed)
        fig.add_trace(
            go.Scatter(
                x=spot_range,
                y=vanilla_prices,
                mode="lines",
                name="Vanilla",
                visible=visible,
                line=dict(color=CHART_COLORS["reference"], width=2, dash="dash"),
                hovertemplate=hover_vanilla,
                showlegend=(i == default_idx),
            )
        )

    # Build slider steps
    steps = []
    for i, value in enumerate(param_values):
        vis = [False] * (N * 2)
        vis[2 * i] = True
        vis[2 * i + 1] = True

        label = str(value) if slider_type == "DTE" else f"{value}%"
        step = dict(
            method="update",
            args=[{"visible": vis}],
            label=label,
        )
        steps.append(step)

    prefix = "Days to Expiration: " if slider_type == "DTE" else "Implied Volatility: "
    slider = SLIDER_DEFAULTS.copy()
    slider.update(
        {
            "active": default_idx,
            "currentvalue": {
                **SLIDER_DEFAULTS["currentvalue"],
                "prefix": prefix,
            },
            "steps": steps,
        }
    )

    # Static elements: spot vline, barrier vline
    fig.add_vline(
        x=spot_price,
        line_dash="dot",
        line_color=CHART_COLORS["accent"],
        line_width=1.5,
        opacity=0.8,
    )
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

    if config.get("barrier"):
        fig.add_vline(
            x=config["barrier"],
            line_dash="dash",
            line_color="#dc2626",
            line_width=2,
            annotation_text=f"Barrier = {config['barrier']:.1f}",
            annotation_position="top right",
            annotation_font_color="#dc2626",
        )

    if config.get("payout") and config.get("discount_factor"):
        pv = config["payout"] * config["discount_factor"]
        fig.add_hline(
            y=pv,
            line_dash="dot",
            line_color="#0d9488",
            line_width=1.5,
            annotation_text=f"PV(Payout) = {pv:.2f}",
            annotation_position="top right",
            annotation_font_color="#0d9488",
        )

    # Layout
    layout = get_layout_config(height=650)
    layout.update(
        {
            "sliders": [slider],
            "xaxis": {
                **AXIS_DEFAULTS,
                "title": {"text": "Underlying Price", **AXIS_DEFAULTS["title"]},
                "tickprefix": "$",
                "tickformat": ",.0f",
            },
            "yaxis": {
                **AXIS_DEFAULTS,
                "title": {"text": "Option Price ($)", **AXIS_DEFAULTS["title"]},
                "tickprefix": "$",
                "tickformat": ",.2f",
            },
            "margin": {"l": 70, "r": 40, "t": 40, "b": 100},
            "hovermode": "x unified",
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

    spread_annotations(fig)
    return fig


# =============================================================================
# MAIN TAB ORCHESTRATOR
# =============================================================================


def render_exotic_tab(spot_price: float, risk_free_rate: float, positions: list):
    """Main orchestrator for the Exotic Options tab.

    Reads exotic legs from the portfolio positions configured in the sidebar.
    Includes DTE/IV slider controls for interactive price comparison.
    """
    from components.metrics import render_chart_controls

    # ── Filter exotic legs from positions ──
    exotic_legs = [
        (i, pos)
        for i, pos in enumerate(positions)
        if pos.get("instrument_class", "vanilla") != "vanilla"
    ]

    if not exotic_legs:
        st.info(
            "No exotic legs in the portfolio. Add an exotic option from the sidebar to see analysis here."
        )
        return

    # ── Leg selector (only if multiple exotic legs) ──
    if len(exotic_legs) == 1:
        leg_idx, pos = exotic_legs[0]
    else:
        labels = [
            f"Leg {idx + 1} — {EXOTIC_TYPE_NAMES.get(p['instrument_class'], p['instrument_class']).split('(')[0].strip()} "
            f"{'Call' if p['option_type'] == 'call' else 'Put'} K={p['strike']}"
            for idx, p in exotic_legs
        ]
        selected = st.selectbox(
            "Select exotic leg to analyze",
            range(len(exotic_legs)),
            format_func=lambda i: labels[i],
            key="exotic_leg_selector",
        )
        leg_idx, pos = exotic_legs[selected]

    # ── Extract parameters from position ──
    exotic_type = pos["instrument_class"]
    strike = pos["strike"]
    is_call = pos["option_type"] == "call"
    barrier = pos.get("barrier", 0.0)
    is_up = pos.get("is_up", True)
    is_knock_in = pos.get("is_knock_in", False)
    rebate = pos.get("rebate", 0.0)
    payout = pos.get("payout", 1.0)
    extra1 = pos.get("extra1", 0.0)

    # Educational description
    if exotic_type in EXOTIC_DESCRIPTIONS:
        st.info(EXOTIC_DESCRIPTIONS[exotic_type])

    # ── DTE/IV toggle controls ──
    slider_type = render_chart_controls("exotic_slider", is_single_leg=False)

    # ── Compute surfaces for all parameter values ──
    spot_range = np.linspace(
        spot_price * (1 - SPOT_RANGE_FACTOR),
        spot_price * (1 + SPOT_RANGE_FACTOR),
        SPOT_RANGE_POINTS,
    )

    if slider_type == "DTE":
        param_values = DTE_RANGE
        fixed_iv = DEFAULT_IV / 100.0
    else:
        param_values = IV_RANGE
        fixed_iv = None  # will vary

    vanilla_strike = strike

    all_exotic_data = []
    all_vanilla_data = []

    for value in param_values:
        if slider_type == "DTE":
            maturity = value / 365.0
            sigma = fixed_iv
        else:
            maturity = DEFAULT_DTE / 365.0
            sigma = value / 100.0

        exotic_data = calculate_exotic_greeks_surface(
            exotic_type,
            spot_range,
            strike,
            maturity,
            risk_free_rate,
            sigma,
            is_call,
            barrier=barrier,
            is_knock_in=is_knock_in,
            is_up=is_up,
            rebate=rebate,
            payout=payout,
            extra1=extra1,
            ref_spot=spot_price,
            params=haug_factory_params(pos),
        )
        vanilla_data = calculate_vanilla_greeks_surface(
            spot_range,
            vanilla_strike,
            maturity,
            risk_free_rate,
            sigma,
            is_call,
        )

        all_exotic_data.append(exotic_data)
        all_vanilla_data.append(vanilla_data)

    # Build config dict (use default DTE for discount factor display)
    default_maturity = DEFAULT_DTE / 365.0
    chart_config = {
        "exotic_type": exotic_type,
        "exotic_label": EXOTIC_TYPE_NAMES.get(exotic_type, exotic_type),
        "is_call": is_call,
        "strike": strike,
        "spot_range": spot_range,
        "barrier": barrier if exotic_type == "barrier" else None,
        "is_up": is_up,
        "is_knock_in": is_knock_in,
        "payout": payout if exotic_type == "digital" else None,
        "discount_factor": math.exp(-risk_free_rate * default_maturity)
        if exotic_type == "digital"
        else None,
        "power_n": pos.get("power_n", 2.0),
        "gap_trigger": pos.get("gap_trigger", strike * 1.05),
    }

    # ── Price comparison chart with slider ──
    fig_price = create_exotic_price_comparison_with_slider(
        all_exotic_data,
        all_vanilla_data,
        param_values,
        spot_range,
        spot_price,
        chart_config,
        slider_type,
    )
    st.plotly_chart(fig_price, width="stretch", config={"displayModeBar": False})

    # ── Payoff diagram (static, no slider) — reads the raw leg dict ──
    fig_payoff = create_exotic_payoff_diagram(spot_range, spot_price, pos)
    st.plotly_chart(fig_payoff, width="stretch")
