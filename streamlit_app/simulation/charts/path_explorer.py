"""
Path Explorer Chart - Single-path price & volatility visualization.

Two-panel Plotly subplot:
  Top: Price path S(t) with S0 reference line
  Bottom: Volatility path sigma(t) - stochastic or flat depending on model
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any

from backend.simulation.base import SimulationResult
from services.simulation_service import (
    get_model_characteristics,
    get_initial_volatility,
    compute_long_run_volatility,
)

# ── Theme (matches simulation_paths.py) ────────────────────────────────────
_PAPER_BG = "#0e1117"
_PLOT_BG = "#161b22"
_GRID = "rgba(255,255,255,0.10)"
_AXIS_LINE = "rgba(255,255,255,0.25)"
_TICK_COLOR = "rgba(255,255,255,0.70)"
_AXIS_LABEL = "#ffffff"
_LEGEND_COLOR = "#ffffff"
_SPOT_LINE = "rgba(255,255,255,0.45)"

_PRICE_COLOR = "#58a6ff"      # bright blue
_VOL_COLOR = "#f0883e"        # bright orange
_VOL_FLAT_COLOR = "rgba(255,160,50,0.7)"
_VOL_REF_INIT = "#78c8ff"   # bright cyan — initial vol
_VOL_REF_LR = "#a078ff"     # bright purple — long-run vol


def render_path_explorer_chart(
    result: SimulationResult,
    model_key: str,
    params: Dict[str, Any],
) -> None:
    """Render single-path price + volatility chart with adaptive axes."""
    chars = get_model_characteristics(model_key)
    has_vol = chars["has_stochastic_vol"] and result.volatility_paths is not None

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
    )

    time_grid = result.time_grid

    # ── Top panel: Price path ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=time_grid,
        y=result.price_paths[0],
        mode="lines",
        line=dict(width=1.8, color=_PRICE_COLOR),
        name="S(t)",
        hovertemplate="Time: %{x:.2f} yr<br>Price: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # S0 reference
    fig.add_hline(
        y=result.initial_price, line_dash="dash",
        line_color=_SPOT_LINE, line_width=1,
        annotation_text=f"S\u2080 = {result.initial_price:.2f}",
        annotation_font_size=11,
        annotation_font_color="rgba(255,255,255,0.55)",
        row=1, col=1,
    )

    # ── Bottom panel: Volatility ───────────────────────────────────────────
    if has_vol:
        vol_pct = result.volatility_paths[0] * 100
        fig.add_trace(go.Scatter(
            x=time_grid,
            y=vol_pct,
            mode="lines",
            line=dict(width=1.8, color=_VOL_COLOR),
            name="\u03c3(t)",
            hovertemplate="Time: %{x:.2f} yr<br>Volatility: %{y:.2f}%<extra></extra>",
        ), row=2, col=1)

        # Reference lines — initial & long-run volatility
        _add_vol_references(fig, model_key, params)
    else:
        init_vol = get_initial_volatility(model_key, params) * 100
        fig.add_trace(go.Scatter(
            x=time_grid,
            y=[init_vol] * len(time_grid),
            mode="lines",
            line=dict(width=1.5, color=_VOL_FLAT_COLOR),
            name=f"\u03c3 = {init_vol:.1f}%",
            hovertemplate="Time: %{x:.2f} yr<br>Volatility: " + f"{init_vol:.2f}%" + "<extra></extra>",
        ), row=2, col=1)

    # ── Layout ─────────────────────────────────────────────────────────────
    _ax = dict(
        gridcolor=_GRID,
        zerolinecolor=_GRID,
        showline=True,
        linecolor=_AXIS_LINE,
        linewidth=1,
        tickfont=dict(size=10, color=_TICK_COLOR),
        title_font=dict(size=12, color=_AXIS_LABEL),
        autorange=True,
    )

    vol_label = "Volatility (%)" if has_vol else "Volatility (constant %)"

    fig.update_layout(
        height=520,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=11, color=_LEGEND_COLOR),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=40, b=35, l=60, r=20),
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1, **_ax)
    fig.update_xaxes(showticklabels=False, row=1, col=1, **_ax)

    fig.update_yaxes(title_text=vol_label, row=2, col=1, **_ax)
    fig.update_xaxes(title_text="Time (years)", row=2, col=1, **_ax)

    st.plotly_chart(fig, use_container_width=True)


def _add_vol_references(
    fig: go.Figure,
    model_key: str,
    params: Dict[str, Any],
) -> None:
    """Add initial-vol and long-run-vol dashed reference lines to the vol panel."""
    model_lower = model_key.lower()

    # ── Initial volatility ──────────────────────────────────────────────
    if model_lower in ("heston", "bates"):
        v0_pct = np.sqrt(params.get("v0", 0.04)) * 100
        fig.add_hline(
            y=v0_pct, line_dash="dash", line_color=_VOL_REF_INIT, line_width=1.5,
            annotation_text=f"\u221aV\u2080 = {v0_pct:.1f}%",
            annotation_font_size=11, annotation_font_color=_VOL_REF_INIT,
            annotation_position="top left",
            row=2, col=1,
        )
    elif model_lower in ("garch", "ngarch", "gjr_garch"):
        s0_pct = params.get("sigma0", 0.20) * 100
        fig.add_hline(
            y=s0_pct, line_dash="dash", line_color=_VOL_REF_INIT, line_width=1.5,
            annotation_text=f"\u03c3\u2080 = {s0_pct:.1f}%",
            annotation_font_size=11, annotation_font_color=_VOL_REF_INIT,
            annotation_position="top left",
            row=2, col=1,
        )

    # ── Long-run volatility ─────────────────────────────────────────────
    if model_lower in ("heston", "bates"):
        theta_pct = np.sqrt(params.get("theta", 0.04)) * 100
        fig.add_hline(
            y=theta_pct, line_dash="dot", line_color=_VOL_REF_LR, line_width=1.5,
            annotation_text=f"\u221a\u03b8 = {theta_pct:.1f}%",
            annotation_font_size=11, annotation_font_color=_VOL_REF_LR,
            annotation_position="bottom left",
            row=2, col=1,
        )
    elif model_lower in ("garch", "ngarch", "gjr_garch"):
        lr_vol = compute_long_run_volatility(model_key, params)
        if lr_vol is not None:
            lr_pct = lr_vol * 100
            fig.add_hline(
                y=lr_pct, line_dash="dot", line_color=_VOL_REF_LR, line_width=1,
                annotation_text=f"LR = {lr_pct:.1f}%",
                annotation_font_size=11, annotation_font_color=_VOL_REF_LR,
                annotation_position="bottom left",
                row=2, col=1,
            )
