"""
Simulation Paths — Profit-colored price & volatility paths.

Green paths: strategy is profitable at T (P&L >= 0)
Red paths: strategy is unprofitable at T (P&L < 0)
When no strategy: neutral blue paths.

Volatility subplot uses the same green/red coloring for stochastic vol models.
Constant vol models (GBM, Merton) show a flat line.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, Optional

from backend.simulation.base import SimulationResult
from services.simulation_service import get_model_characteristics, get_initial_volatility

# ── Color palette ─────────────────────────────────────────────────────────
_GREEN = "rgba(34, 197, 94, 0.40)"
_RED = "rgba(239, 68, 68, 0.32)"
_BLUE = "rgba(99, 160, 255, 0.35)"
_BAND_FILL = "rgba(250, 204, 21, 0.12)"
_BAND_EDGE = "#e5b820"
_VOL_ORANGE = "rgba(255, 160, 50, 0.35)"
_VOL_BAND_FILL = "rgba(255, 160, 50, 0.13)"
_SPOT_LINE = "rgba(255, 255, 255, 0.45)"
_BE_LINE = "#a78bfa"

# Chart background — dark enough to contrast with Streamlit's grey UI
_PAPER_BG = "#0e1117"
_PLOT_BG = "#161b22"
_GRID = "rgba(255,255,255,0.10)"
_AXIS_LINE = "rgba(255,255,255,0.25)"

# Text — white for readability on dark background
_TITLE_COLOR = "#ffffff"
_AXIS_LABEL = "#ffffff"
_TICK_COLOR = "rgba(255,255,255,0.70)"
_LEGEND_COLOR = "#ffffff"

MODEL_DISPLAY_NAMES = {
    "gbm": "GBM",
    "heston": "Heston",
    "merton": "Merton",
    "bates": "Bates",
    "garch": "GARCH",
    "ngarch": "NGARCH",
    "gjr_garch": "GJR-GARCH",
}


def render_simulation_chart(
    result: SimulationResult,
    model_key: str,
    params: Dict[str, Any],
    n_display: int = 150,
    show_bands: bool = True,
    pnl_values: Optional[np.ndarray] = None,
    breakeven_prices: Optional[np.ndarray] = None,
) -> None:
    """Render price paths + volatility paths chart."""
    chars = get_model_characteristics(model_key)
    has_vol = chars["has_stochastic_vol"] and result.volatility_paths is not None
    has_pnl = pnl_values is not None and len(pnl_values) == result.price_paths.shape[0]

    # Subplot layout
    if has_vol:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
        )
        chart_height = 740
    else:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.78, 0.22],
        )
        chart_height = 600

    time_grid = result.time_grid
    n_paths = result.price_paths.shape[0]

    # Sample indices
    if n_paths > n_display:
        idx_sample = np.random.choice(n_paths, n_display, replace=False)
    else:
        idx_sample = np.arange(n_paths)

    # ── Price paths ──────────────────────────────────────────────────────
    if has_pnl:
        profit_mask = pnl_values[idx_sample] >= 0
        _add_paths(fig, time_grid, result.price_paths, idx_sample[profit_mask],
                   _GREEN, "Profit", "profit", row=1)
        _add_paths(fig, time_grid, result.price_paths, idx_sample[~profit_mask],
                   _RED, "Loss", "loss", row=1)
    else:
        _add_paths(fig, time_grid, result.price_paths, idx_sample,
                   _BLUE, "Path", "paths", row=1)

    # 5-95 percentile band
    if show_bands:
        pct = result.percentile_paths([5, 95])
        fig.add_trace(go.Scatter(
            x=time_grid, y=pct[1], mode="lines",
            line=dict(width=1.2, color=_BAND_EDGE, dash="dot"),
            name="95th pct", legendgroup="band",
            hovertemplate="95th: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=time_grid, y=pct[0], mode="lines",
            line=dict(width=1.2, color=_BAND_EDGE, dash="dot"),
            fill="tonexty", fillcolor=_BAND_FILL,
            name="5th pct", legendgroup="band",
            hovertemplate="5th: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # Spot reference
    fig.add_hline(
        y=result.initial_price, line_dash="dash",
        line_color=_SPOT_LINE, line_width=1,
        annotation_text=f"S\u2080 = {result.initial_price:.2f}",
        annotation_font_size=11, annotation_font_color="rgba(255,255,255,0.55)",
        row=1, col=1,
    )

    # Breakeven lines
    if breakeven_prices is not None and len(breakeven_prices) > 0:
        for be in breakeven_prices:
            fig.add_hline(
                y=be, line_dash="dot", line_color=_BE_LINE, line_width=1,
                annotation_text=f"BE ${be:.1f}",
                annotation_font_size=10, annotation_font_color=_BE_LINE,
                annotation_position="top right",
                row=1, col=1,
            )

    # ── Volatility paths ─────────────────────────────────────────────────
    if has_vol:
        vol_paths = result.volatility_paths

        if has_pnl:
            _add_paths(fig, time_grid, vol_paths * 100,
                       idx_sample[profit_mask], _GREEN, None, "profit",
                       row=2, show_legend=False)
            _add_paths(fig, time_grid, vol_paths * 100,
                       idx_sample[~profit_mask], _RED, None, "loss",
                       row=2, show_legend=False)
        else:
            _add_paths(fig, time_grid, vol_paths * 100,
                       idx_sample, _VOL_ORANGE, "Vol Path", "vol",
                       row=2)

        # Vol bands
        if show_bands:
            vol_pct = np.percentile(vol_paths * 100, [5, 95], axis=0)
            fig.add_trace(go.Scatter(
                x=time_grid, y=vol_pct[1], mode="lines",
                line=dict(width=0.8, color="rgba(255,160,50,0.30)", dash="dot"),
                showlegend=False, hoverinfo="skip",
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=time_grid, y=vol_pct[0], mode="lines",
                line=dict(width=0.8, color="rgba(255,160,50,0.30)", dash="dot"),
                fill="tonexty", fillcolor=_VOL_BAND_FILL,
                name="Vol 5-95 %", legendgroup="vol_band",
            ), row=2, col=1)
    else:
        # Constant vol (GBM, Merton)
        initial_vol = get_initial_volatility(model_key, params) * 100
        fig.add_trace(go.Scatter(
            x=time_grid, y=[initial_vol] * len(time_grid),
            mode="lines", line=dict(width=1.5, color="rgba(255,160,50,0.7)"),
            name=f"\u03c3 = {initial_vol:.1f} %",
        ), row=2, col=1)

    # ── Layout ────────────────────────────────────────────────────────────
    _axis_style = dict(
        gridcolor=_GRID,
        zerolinecolor=_GRID,
        showline=True,
        linecolor=_AXIS_LINE,
        linewidth=1,
        tickfont=dict(size=10, color=_TICK_COLOR),
        title_font=dict(size=12, color=_AXIS_LABEL),
    )

    vol_label = "Volatility Paths" if has_vol else "Volatility (Constant)"

    fig.update_layout(
        height=chart_height,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=11, color=_LEGEND_COLOR),
            bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
            tracegroupgap=0,
        ),
        margin=dict(t=50, b=35, l=60, r=20),
    )

    # Time axis ticks: "Time: X.XX yr"
    t_max = float(time_grid[-1])
    _tvals = np.linspace(0, t_max, 6).tolist()
    _ttexts = [f"Time: {v:.2f} yr" for v in _tvals]
    _time_kw = dict(tickmode="array", tickvals=_tvals, ticktext=_ttexts)

    # Price subplot axes
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, **_axis_style)
    fig.update_xaxes(
        showticklabels=not has_vol,
        row=1, col=1, **_axis_style, **_time_kw,
    )

    # Vol subplot axes
    fig.update_yaxes(title_text=vol_label, row=2, col=1, **_axis_style)
    fig.update_xaxes(
        row=2, col=1, **_axis_style, **_time_kw,
    )

    st.plotly_chart(fig, width="stretch")


# ── helpers ───────────────────────────────────────────────────────────────

def _add_paths(
    fig, time_grid, data, indices, color, legend_name, legend_group,
    row=1, show_legend=True,
):
    """Add a batch of paths to the figure."""
    first = True
    for idx in indices:
        fig.add_trace(go.Scatter(
            x=time_grid, y=data[idx],
            mode="lines", line=dict(width=0.7, color=color),
            name=legend_name if (first and show_legend) else None,
            showlegend=(first and show_legend and legend_name is not None),
            legendgroup=legend_group,
            hoverinfo="skip",
        ), row=row, col=1)
        first = False


def render_path_controls() -> Dict[str, Any]:
    """Controls for path visualization."""
    col1, col2 = st.columns(2)
    with col1:
        n_paths = st.slider("Display Paths", 20, 500, 150, 10)
    with col2:
        show_bands = st.checkbox("Show 5-95% Bands", value=True)
    return {"n_display": n_paths, "show_bands": show_bands}
