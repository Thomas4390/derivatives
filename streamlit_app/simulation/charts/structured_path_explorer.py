"""
Structured Product Path Explorer for Simulation Explorer.

Uses the same dark theme and two-panel layout (price + volatility)
as the options path_explorer.py for visual consistency.

Author: Thomas Vaudescal
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))

from functools import partial

from config.chart_theme import (
    ANN_BG as _ANN_BG,
    AXIS_LINE as _AXIS_LINE,
    AXIS_LABEL as _AXIS_LABEL,
    GRID as _GRID,
    LEGEND_COLOR as _LEGEND_COLOR,
    PAPER_BG as _PAPER_BG,
    PLOT_BG as _PLOT_BG,
    TICK_COLOR as _TICK_COLOR,
    badge,
)
from config.constants import SP_PRODUCT_COLORS
from services.simulation_service import (
    compute_long_run_volatility,
    get_initial_volatility,
    get_model_characteristics,
)

_badge = partial(badge, font_size=11)

_SPOT_LINE = "rgba(255,255,255,0.45)"
_PROFIT_COLOR = "#22c55e"
_LOSS_COLOR = "#ef4444"
_PRICE_COLOR = "#58a6ff"
_VOL_COLOR = "#f0883e"
_VOL_FLAT_COLOR = "rgba(255,160,50,0.7)"
_VOL_REF_INIT = "#78c8ff"
_VOL_REF_LR = "#a078ff"


def render_structured_path_explorer(
    result,
    sp_result,
    model_key: str,
    params: dict[str, Any],
    product_config: dict,
) -> float | None:
    """
    Render single-path structured product explorer.

    Same two-panel layout as options path explorer:
    - Top: Price path with SP overlays (barriers, triggers, obs dates, events)
    - Bottom: Volatility path (stochastic or flat)

    Returns the single-path P&L value.
    """
    chars = get_model_characteristics(model_key)
    has_vol = chars["has_stochastic_vol"] and result.volatility_paths is not None

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
    )

    time_grid = result.time_grid
    price_path = result.price_paths[0]
    product_type = product_config["product_type"]
    product_params = product_config["product_params"]
    s0 = params.get("spot", float(price_path[0]))
    notional = product_params["notional"]

    # Per-path values
    path_return = float(sp_result.per_path_returns[0])
    path_pnl = path_return * notional
    barrier_hit = bool(sp_result.barrier_breached[0])

    is_autocalled = False
    autocall_obs_idx = -1
    if product_type == "autocallable":
        is_autocalled = bool(sp_result.autocall_called[0])
        autocall_obs_idx = int(sp_result.autocall_date_index[0])

    obs_times = sp_result.observation_times
    obs_indices = sp_result.observation_indices

    # ── Path color from P&L ──
    path_color = _PROFIT_COLOR if path_pnl >= 0 else _LOSS_COLOR

    # ── Top panel: Price path ──
    if is_autocalled and autocall_obs_idx >= 0:
        end_step = obs_indices[autocall_obs_idx]
        # Full path faded
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=price_path,
                mode="lines",
                line=dict(color="rgba(148, 163, 184, 0.25)", width=1),
                name="Post-autocall",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        # Active path solid
        fig.add_trace(
            go.Scatter(
                x=time_grid[: end_step + 1],
                y=price_path[: end_step + 1],
                mode="lines",
                line=dict(width=1.8, color=path_color),
                name="S(t)",
                hovertemplate="<b>Time:</b> %{x:.2f} yr<br><b>Price:</b> $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=price_path,
                mode="lines",
                line=dict(width=1.8, color=path_color),
                name="S(t)",
                hovertemplate="<b>Time:</b> %{x:.2f} yr<br><b>Price:</b> $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # S0 reference
    fig.add_hline(
        y=s0,
        line_dash="dash",
        line_color=_SPOT_LINE,
        line_width=1,
        annotation_text=f"S\u2080 = {s0:.2f}",
        annotation_font_size=11,
        annotation_font_color="rgba(255,255,255,0.55)",
        row=1,
        col=1,
    )

    # ── Product-level overlays (barriers, triggers, cap) ──
    _add_product_levels(fig, product_type, product_params, s0)

    # ── Observation date markers with events ──
    for j, (t_obs, oi) in enumerate(zip(obs_times, obs_indices)):
        spot_at_obs = price_path[oi]

        # Vertical obs line
        fig.add_vline(
            x=t_obs,
            line_dash="dot",
            line_color="rgba(255,255,255,0.12)",
            line_width=1,
            row=1,
            col=1,
        )

        # Event marker
        marker_color = "rgba(148, 163, 184, 0.6)"
        marker_symbol = "circle"
        hover_text = f"Obs {j + 1}: S={spot_at_obs:.2f}"

        if product_type == "autocallable":
            perf = spot_at_obs / s0
            if is_autocalled and autocall_obs_idx == j:
                marker_color = SP_PRODUCT_COLORS["autocall"]
                marker_symbol = "star"
                hover_text += " | AUTOCALLED"
            elif perf >= product_params.get("coupon_barrier", 0):
                marker_color = SP_PRODUCT_COLORS["coupon_pv"]
                marker_symbol = "diamond"
                hover_text += " | Coupon paid"
            else:
                marker_color = _LOSS_COLOR
                marker_symbol = "circle"
                hover_text += " | Coupon missed"

        fig.add_trace(
            go.Scatter(
                x=[t_obs],
                y=[spot_at_obs],
                mode="markers",
                marker=dict(
                    symbol=marker_symbol,
                    size=8,
                    color=marker_color,
                    line=dict(width=1, color="rgba(255,255,255,0.4)"),
                ),
                showlegend=False,
                hoverinfo="text",
                hovertext=hover_text,
            ),
            row=1,
            col=1,
        )

    # ── Barrier breach marker ──
    if barrier_hit and product_type != "cpn":
        if product_type == "reverse_convertible":
            barrier_level = product_params["barrier"] * s0
        else:
            barrier_level = product_params["ki_barrier"] * s0

        breach_steps = np.where(price_path <= barrier_level)[0]
        if len(breach_steps) > 0:
            first = breach_steps[0]
            fig.add_trace(
                go.Scatter(
                    x=[time_grid[first]],
                    y=[price_path[first]],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        size=12,
                        color=SP_PRODUCT_COLORS["barrier"],
                        line=dict(width=2, color="rgba(255,255,255,0.5)"),
                    ),
                    name="Barrier Breach",
                    showlegend=True,
                    hoverinfo="text",
                    hovertext=f"Barrier breached at t={time_grid[first]:.3f}",
                ),
                row=1,
                col=1,
            )

    # ── P&L annotation at terminal point ──
    pnl_label = f"P&L: ${path_pnl:+,.2f}"
    term_x = time_grid[-1]
    if is_autocalled and autocall_obs_idx >= 0:
        term_x = time_grid[obs_indices[autocall_obs_idx]]
    term_y = float(
        price_path[min(int(np.searchsorted(time_grid, term_x)), len(price_path) - 1)]
    )
    fig.add_annotation(
        x=term_x,
        y=term_y,
        text=f" {pnl_label} ",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor=path_color,
        font=dict(size=12, color=path_color),
        bgcolor=_ANN_BG,
        bordercolor=path_color,
        borderwidth=1,
        borderpad=3,
        xref="x",
        yref="y",
        row=1,
        col=1,
    )

    # ── Bottom panel: Volatility ──
    if has_vol:
        vol_pct = result.volatility_paths[0] * 100
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=vol_pct,
                mode="lines",
                line=dict(width=1.8, color=_VOL_COLOR),
                name="\u03c3(t)",
                hovertemplate="<b>Time:</b> %{x:.2f} yr<br><b>Volatility:</b> %{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )
        _add_vol_references(fig, model_key, params)
    else:
        init_vol = get_initial_volatility(model_key, params) * 100
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=[init_vol] * len(time_grid),
                mode="lines",
                line=dict(width=1.5, color=_VOL_FLAT_COLOR),
                name=f"\u03c3 = {init_vol:.1f}%",
                hovertemplate="<b>Time:</b> %{x:.2f} yr<br><b>Volatility:</b> "
                + f"{init_vol:.2f}%"
                + "<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # ── Layout (identical to path_explorer.py) ──
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
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color=_LEGEND_COLOR),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=40, b=35, l=60, r=20),
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1, **_ax)
    fig.update_xaxes(showticklabels=False, row=1, col=1, **_ax)

    fig.update_yaxes(title_text=vol_label, row=2, col=1, **_ax)
    fig.update_xaxes(title_text="Time (years)", row=2, col=1, **_ax)

    st.plotly_chart(fig, width="stretch")
    return path_pnl


# ── Product-level overlays ──────────────────────────────────────────────


def _add_product_levels(fig, product_type, params, s0):
    """Horizontal reference lines for structured product levels."""
    if product_type == "cpn":
        protection = params.get("protection_level", 1.0)
        fig.add_hline(
            y=s0 * protection,
            line_dash="dash",
            line_color=SP_PRODUCT_COLORS["bond_floor"],
            line_width=1.5,
            annotation_text=f" Protection ({protection:.0%}) ",
            **_badge(SP_PRODUCT_COLORS["bond_floor"]),
            row=1,
            col=1,
        )
        cap = params.get("cap")
        if cap:
            fig.add_hline(
                y=s0 * cap,
                line_dash="dash",
                line_color=SP_PRODUCT_COLORS["option_value"],
                line_width=1.5,
                annotation_text=f" Cap ({cap:.0%}) ",
                **_badge(SP_PRODUCT_COLORS["option_value"]),
                row=1,
                col=1,
            )

    elif product_type == "reverse_convertible":
        barrier = params["barrier"]
        fig.add_hline(
            y=s0 * barrier,
            line_dash="dash",
            line_color=SP_PRODUCT_COLORS["barrier"],
            line_width=1.5,
            annotation_text=f" KI Barrier ({barrier:.0%}) ",
            **_badge(SP_PRODUCT_COLORS["barrier"]),
            row=1,
            col=1,
        )

    elif product_type == "autocallable":
        fig.add_hline(
            y=s0 * params["autocall_trigger"],
            line_dash="dash",
            line_color=SP_PRODUCT_COLORS["autocall"],
            line_width=1.5,
            annotation_text=f" Autocall ({params['autocall_trigger']:.0%}) ",
            annotation_position="top right",
            **_badge(SP_PRODUCT_COLORS["autocall"]),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=s0 * params["coupon_barrier"],
            line_dash="dot",
            line_color=SP_PRODUCT_COLORS["coupon_pv"],
            line_width=1,
            annotation_text=f" Coupon ({params['coupon_barrier']:.0%}) ",
            annotation_position="bottom right",
            **_badge(SP_PRODUCT_COLORS["coupon_pv"]),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=s0 * params["ki_barrier"],
            line_dash="dash",
            line_color=SP_PRODUCT_COLORS["barrier"],
            line_width=1.5,
            annotation_text=f" KI Barrier ({params['ki_barrier']:.0%}) ",
            annotation_position="bottom right",
            **_badge(SP_PRODUCT_COLORS["barrier"]),
            row=1,
            col=1,
        )


# ── Volatility references (reused from path_explorer.py) ───────────────


def _add_vol_references(fig, model_key: str, params: dict) -> None:
    model_lower = model_key.lower()
    if model_lower in ("heston", "bates"):
        v0_pct = np.sqrt(params.get("v0", 0.04)) * 100
        fig.add_hline(
            y=v0_pct,
            line_dash="dash",
            line_color=_VOL_REF_INIT,
            line_width=1.5,
            annotation_text=f"\u221aV\u2080 = {v0_pct:.1f}%",
            annotation_font_size=11,
            annotation_font_color=_VOL_REF_INIT,
            annotation_position="top left",
            row=2,
            col=1,
        )
    elif model_lower in ("garch", "ngarch", "gjr_garch"):
        s0_pct = params.get("sigma0", 0.20) * 100
        fig.add_hline(
            y=s0_pct,
            line_dash="dash",
            line_color=_VOL_REF_INIT,
            line_width=1.5,
            annotation_text=f"\u03c3\u2080 = {s0_pct:.1f}%",
            annotation_font_size=11,
            annotation_font_color=_VOL_REF_INIT,
            annotation_position="top left",
            row=2,
            col=1,
        )

    if model_lower in ("heston", "bates"):
        theta_pct = np.sqrt(params.get("theta", 0.04)) * 100
        fig.add_hline(
            y=theta_pct,
            line_dash="dot",
            line_color=_VOL_REF_LR,
            line_width=1.5,
            annotation_text=f"\u221a\u03b8 = {theta_pct:.1f}%",
            annotation_font_size=11,
            annotation_font_color=_VOL_REF_LR,
            annotation_position="bottom left",
            row=2,
            col=1,
        )
    elif model_lower in ("garch", "ngarch", "gjr_garch"):
        lr_vol = compute_long_run_volatility(model_key, params)
        if lr_vol is not None:
            lr_pct = lr_vol * 100
            fig.add_hline(
                y=lr_pct,
                line_dash="dot",
                line_color=_VOL_REF_LR,
                line_width=1.5,
                annotation_text=f"LR = {lr_pct:.1f}%",
                annotation_font_size=11,
                annotation_font_color=_VOL_REF_LR,
                annotation_position="bottom left",
                row=2,
                col=1,
            )
