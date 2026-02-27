"""
P&L Analysis Charts — Simulated P&L scatter with marginal histogram + 3D scatter.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Optional

from backend.simulation.base import SimulationResult

# ── Color palette (consistent with simulation_paths.py) ──────────────────
_PAPER_BG = "#0e1117"
_PLOT_BG = "#161b22"
_GRID = "rgba(255,255,255,0.10)"
_AXIS_LINE = "rgba(255,255,255,0.25)"
_AXIS_LABEL = "#ffffff"
_TICK_COLOR = "rgba(255,255,255,0.70)"
_LEGEND_COLOR = "#ffffff"

_GREEN_MARKER = "rgba(34, 197, 94, 0.55)"
_RED_MARKER = "rgba(239, 68, 68, 0.45)"
_GREEN_HIST = "rgba(34, 197, 94, 0.60)"
_RED_HIST = "rgba(239, 68, 68, 0.50)"
_ZERO_LINE = "rgba(255, 255, 255, 0.30)"

_AXIS_STYLE = dict(
    gridcolor=_GRID,
    zerolinecolor=_GRID,
    showline=True,
    linecolor=_AXIS_LINE,
    linewidth=1,
    tickfont=dict(size=10, color=_TICK_COLOR),
    title_font=dict(size=12, color=_AXIS_LABEL),
)


def render_payoff_with_distribution(
    result: SimulationResult,
    pnl_values: np.ndarray,
    breakeven_prices: Optional[np.ndarray] = None,
    spot: float = 100.0,
    max_scatter: int = 5000,
) -> None:
    """MC simulated P&L scatter (green/red) + marginal P&L histogram."""

    pnl_values = np.round(pnl_values, 2)
    terminal = result.terminal_prices
    n = len(terminal)

    # Subsample for performance
    if n > max_scatter:
        idx = np.random.choice(n, max_scatter, replace=False)
    else:
        idx = np.arange(n)

    tp = terminal[idx]
    pnl = pnl_values[idx]
    profit_mask = pnl >= 0

    # ── Figure: scatter (left) + marginal histogram (right) ──────────────
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.82, 0.18],
        shared_yaxes=True,
        horizontal_spacing=0.02,
    )

    # Scatter — profit paths (green)
    fig.add_trace(go.Scatter(
        x=tp[profit_mask], y=pnl[profit_mask],
        mode="markers",
        marker=dict(size=3.5, color=_GREEN_MARKER),
        name="Profit", legendgroup="profit",
        hovertemplate="<b>S(T):</b> $%{x:.2f}<br><b>P&L:</b> $%{y:+.2f}<extra></extra>",
    ), row=1, col=1)

    # Scatter — loss paths (red)
    fig.add_trace(go.Scatter(
        x=tp[~profit_mask], y=pnl[~profit_mask],
        mode="markers",
        marker=dict(size=3.5, color=_RED_MARKER),
        name="Loss", legendgroup="loss",
        hovertemplate="<b>S(T):</b> $%{x:.2f}<br><b>P&L:</b> $%{y:+.2f}<extra></extra>",
    ), row=1, col=1)

    # P&L = 0 line
    fig.add_hline(
        y=0, line_dash="dash", line_color=_ZERO_LINE, line_width=1,
        row=1, col=1,
    )

    # Spot reference
    fig.add_vline(
        x=spot, line_dash="dot", line_color="rgba(255,255,255,0.25)",
        line_width=1,
        annotation_text=f"S\u2080={spot:.0f}",
        annotation_font_size=10,
        annotation_font_color="rgba(255,255,255,0.5)",
        row=1, col=1,
    )

    # Breakeven vertical lines
    if breakeven_prices is not None:
        for be in breakeven_prices:
            fig.add_vline(
                x=be, line_dash="dot", line_color="#a78bfa", line_width=1,
                annotation_text=f"BE ${be:.1f}",
                annotation_font_size=9,
                annotation_font_color="#a78bfa",
                annotation_position="top",
                row=1, col=1,
            )

    # ── Marginal histogram of P&L (right panel) ─────────────────────────
    profit_pnl = pnl_values[pnl_values >= 0]
    loss_pnl = pnl_values[pnl_values < 0]

    n_bins = 60
    all_min, all_max = float(np.min(pnl_values)), float(np.max(pnl_values))
    bin_size = (all_max - all_min) / n_bins if all_max > all_min else 1.0

    if len(profit_pnl) > 0:
        fig.add_trace(go.Histogram(
            y=profit_pnl,
            ybins=dict(start=0, end=all_max, size=bin_size),
            marker_color=_GREEN_HIST, showlegend=False,
            hovertemplate="<b>P&L:</b> $%{y:+.2f}<br><b>Count:</b> %{x}<extra></extra>",
        ), row=1, col=2)

    if len(loss_pnl) > 0:
        fig.add_trace(go.Histogram(
            y=loss_pnl,
            ybins=dict(start=all_min, end=0, size=bin_size),
            marker_color=_RED_HIST, showlegend=False,
            hovertemplate="<b>P&L:</b> $%{y:+.2f}<br><b>Count:</b> %{x}<extra></extra>",
        ), row=1, col=2)

    # ── Layout ───────────────────────────────────────────────────────────
    fig.update_layout(
        height=550,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.4,
            font=dict(size=11, color=_LEGEND_COLOR),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=50, b=45, l=60, r=20),
        bargap=0.02,
    )

    fig.update_xaxes(title_text="Terminal Price S(T)", row=1, col=1, tickprefix="$", tickformat=",.2f", hoverformat=",.2f", **_AXIS_STYLE)
    fig.update_yaxes(title_text="P&L ($)", row=1, col=1, tickprefix="$", tickformat=",.2f", hoverformat=",.2f", **_AXIS_STYLE)
    fig.update_xaxes(title_text="Count", row=1, col=2, showticklabels=False, **_AXIS_STYLE)
    fig.update_yaxes(row=1, col=2, showticklabels=False, **_AXIS_STYLE)

    st.plotly_chart(fig, width="stretch")


def render_3d_pnl_chart(
    result: SimulationResult,
    pnl_values: np.ndarray,
    time_horizon: float = 1.0,
    max_points: int = 4000,
) -> None:
    """3D scatter: realized volatility x terminal price x P&L, with formula."""

    pnl_values = np.round(pnl_values, 2)
    terminal = result.terminal_prices
    n_steps = result.n_steps

    # Annualization: sqrt(N / T) so that RV is in annual terms
    ann_factor = np.sqrt(n_steps / time_horizon)
    realized_vol = result.realized_volatility(annualization_factor=ann_factor) * 100
    n = len(terminal)

    # Subsample for performance
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
    else:
        idx = np.arange(n)

    tp = terminal[idx]
    rv = realized_vol[idx]
    pnl = pnl_values[idx]

    # Symmetric color range centered on 0
    pnl_abs_max = max(abs(float(np.min(pnl))), abs(float(np.max(pnl))))
    if pnl_abs_max == 0:
        pnl_abs_max = 1.0

    # Diverging red → dark → green
    colorscale = [
        [0.00, "#fca5a5"],   # red-300
        [0.15, "#f87171"],   # red-400
        [0.30, "#ef4444"],   # red-500
        [0.42, "#b91c1c"],   # red-700
        [0.50, "#1c1917"],   # stone-900 (dark midpoint)
        [0.58, "#166534"],   # green-800
        [0.70, "#22c55e"],   # green-500
        [0.85, "#4ade80"],   # green-400
        [1.00, "#86efac"],   # green-300
    ]

    fig = go.Figure(data=[go.Scatter3d(
        x=rv,
        y=tp,
        z=pnl,
        mode="markers",
        marker=dict(
            size=2.5,
            color=pnl,
            colorscale=colorscale,
            cmin=-pnl_abs_max,
            cmax=pnl_abs_max,
            opacity=0.90,
            colorbar=dict(
                title=dict(text="P&L ($)", font=dict(color=_AXIS_LABEL, size=11)),
                tickfont=dict(color=_TICK_COLOR, size=9),
                tickprefix="$",
                tickformat=",.2f",
                thickness=14,
                len=0.6,
                outlinewidth=0,
                bgcolor="rgba(0,0,0,0)",
            ),
        ),
        hovertemplate=(
            "<b>Realized Vol:</b> %{x:.2f}%<br>"
            "<b>S(T):</b> $%{y:.2f}<br>"
            "<b>P&L:</b> $%{z:+.2f}<extra></extra>"
        ),
    )])

    _scene_axis = dict(
        backgroundcolor=_PLOT_BG,
        gridcolor="rgba(255,255,255,0.08)",
        showline=True,
        linecolor=_AXIS_LINE,
        tickfont=dict(size=9, color=_TICK_COLOR),
        title_font=dict(size=11, color=_AXIS_LABEL),
    )

    fig.update_layout(
        height=650,
        paper_bgcolor=_PAPER_BG,
        margin=dict(t=20, b=10, l=10, r=10),
        scene=dict(
            xaxis=dict(title="Realized Volatility (%)", ticksuffix="%", tickformat=".1f", **_scene_axis),
            yaxis=dict(title="Terminal Price S(T)", tickprefix="$", tickformat=",.2f", **_scene_axis),
            zaxis=dict(title="P&L ($)", tickprefix="$", tickformat=",.2f", **_scene_axis),
            bgcolor=_PLOT_BG,
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
        ),
    )

    st.plotly_chart(fig, width="stretch")

    # Realized vol formula — adaptive to selected N and T
    ann_ratio = f"{n_steps}" if time_horizon == 1.0 else rf"\frac{{{n_steps}}}{{{time_horizon:.2g}}}"
    st.latex(
        r"\sigma_{\mathrm{realized}}^{(i)} "
        r"= \sqrt{" + ann_ratio + r"}"
        r" \;\cdot\; "
        r"\mathrm{std}\!\left(\,r_1^{(i)},\; r_2^{(i)},\; \dots,\; r_{"
        + str(n_steps) +
        r"}^{(i)}\,\right)"
        r"\qquad\text{where}\quad "
        r"r_t^{(i)} = \ln\!\frac{S_{t}^{(i)}}{S_{t-1}^{(i)}}"
    )
