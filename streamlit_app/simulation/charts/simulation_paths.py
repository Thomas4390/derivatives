"""
Simulation Paths — Profit-colored price & volatility paths.

Green paths: strategy is profitable at T (P&L >= 0)
Red paths: strategy is unprofitable at T (P&L < 0)
When no strategy: neutral blue paths.

Volatility subplot uses the same green/red coloring for stochastic vol models.
Constant vol models (GBM, Merton) show a flat line.

Exotic option overlays:
- Barrier: horizontal barrier line + first-crossing markers + greyed-out post-hit segments
- Asian: running geometric mean on highlighted paths
- Lookback: running min/max on highlighted paths
- Digital / Asset-or-Nothing: payoff zone shading above/below strike
- Chooser: vertical choice-time line
- Gap: dual hlines for strike K1 and trigger K2
- Power: effective breakeven line at K^(1/n)
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from services.simulation_service import (
    compute_long_run_volatility,
    get_initial_volatility,
    get_model_characteristics,
)
from utils.chart_helpers import spread_annotations

from backend.simulation.base import SimulationResult

# ── Color palette ─────────────────────────────────────────────────────────
_GREEN = "rgba(34, 197, 94, 0.40)"
_RED = "rgba(239, 68, 68, 0.32)"
_BLUE = "rgba(99, 160, 255, 0.35)"
_BAND_FILL = "rgba(250, 204, 21, 0.12)"
_BAND_EDGE = "#e5b820"
_VOL_ORANGE = "rgba(255, 160, 50, 0.35)"
_VOL_BAND_FILL = "rgba(255, 160, 50, 0.18)"
_VOL_BAND_EDGE = "#ffa032"
_BE_LINE = "#a78bfa"

# Vol reference lines
_VOL_REF_INIT = "#78c8ff"  # bright cyan — initial vol (V0, sigma0)
_VOL_REF_LR = "#a078ff"  # bright purple — long-run vol (theta, LR)

# Exotic overlay colors
_BARRIER_LINE = "#f59e0b"  # amber
_BARRIER_HIT_KO = "#ef4444"  # red marker for knock-out hit
_BARRIER_HIT_KI = "#22c55e"  # green marker for knock-in activation
_BARRIER_DEAD = "rgba(120,120,120,0.18)"  # grey for knocked-out path segments
_ASIAN_AVG = "#06b6d4"  # cyan for running average
_LOOKBACK_EXTREME = "#c084fc"  # light purple for running min/max
_DIGITAL_ZONE = "rgba(139, 92, 246, 0.06)"  # very faint violet for payoff zone
_CHOOSER_LINE = "#d946ef"  # fuchsia for choice time
_GAP_TRIGGER = "#fb923c"  # orange for trigger level
_POWER_BE = "#14b8a6"  # teal for effective breakeven

from config.chart_theme import (
    AXIS_STYLE,
    CHART_HEIGHT_MD,
    LEGEND_COLOR as _LEGEND_COLOR,
    PAPER_BG as _PAPER_BG,
    PLOT_BG as _PLOT_BG,
    badge as _badge,
)

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
    params: dict[str, Any],
    n_display: int = 150,
    show_bands: bool = True,
    pnl_values: np.ndarray | None = None,
    breakeven_prices: np.ndarray | None = None,
    exotic_metadata: list[dict] | None = None,
    exotic_viz: dict[str, Any] | None = None,
    overlay_fn: Any | None = None,
) -> None:
    """Render price paths + volatility paths chart with exotic overlays.

    Parameters
    ----------
    overlay_fn : callable, optional
        Called as ``overlay_fn(fig, price_paths, time_grid, idx_sample)``
        just before the chart is displayed. Use to add custom overlays.
    """
    chars = get_model_characteristics(model_key)
    has_vol = chars["has_stochastic_vol"] and result.volatility_paths is not None
    has_pnl = pnl_values is not None and len(pnl_values) == result.price_paths.shape[0]

    # Subplot layout
    if has_vol:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
        )
        chart_height = 740
    else:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.78, 0.22],
        )
        chart_height = CHART_HEIGHT_MD

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
        _add_paths(
            fig,
            time_grid,
            result.price_paths,
            idx_sample[profit_mask],
            _GREEN,
            "Profit",
            "profit",
            row=1,
        )
        _add_paths(
            fig,
            time_grid,
            result.price_paths,
            idx_sample[~profit_mask],
            _RED,
            "Loss",
            "loss",
            row=1,
        )
    else:
        _add_paths(
            fig,
            time_grid,
            result.price_paths,
            idx_sample,
            _BLUE,
            "Path",
            "paths",
            row=1,
        )

    # 5-95 percentile band
    if show_bands:
        pct = result.percentile_paths([5, 95])
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=pct[1],
                mode="lines",
                line=dict(width=1.2, color=_BAND_EDGE, dash="dot"),
                name="95th pct",
                legendgroup="band",
                hovertemplate="<b>Time:</b> %{x:.2f} yr<br><b>95th Percentile:</b> $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=pct[0],
                mode="lines",
                line=dict(width=1.2, color=_BAND_EDGE, dash="dot"),
                fill="tonexty",
                fillcolor=_BAND_FILL,
                name="5th pct",
                legendgroup="band",
                hovertemplate="<b>Time:</b> %{x:.2f} yr<br><b>5th Percentile:</b> $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Breakeven lines
    if breakeven_prices is not None and len(breakeven_prices) > 0:
        for be in breakeven_prices:
            fig.add_hline(
                y=be,
                line_dash="dot",
                line_color=_BE_LINE,
                line_width=1,
                annotation_text=f" BE ${be:.1f} ",
                annotation_position="top right",
                **_badge(_BE_LINE),
                row=1,
                col=1,
            )

    # ── Exotic overlays ──────────────────────────────────────────────────
    _eviz = exotic_viz or {}
    if exotic_metadata and _eviz.get("show_overlays", True):
        _add_exotic_overlays(
            fig,
            result.price_paths,
            time_grid,
            idx_sample,
            exotic_metadata,
            params,
            _eviz,
        )

    # ── Volatility paths ─────────────────────────────────────────────────
    if has_vol:
        vol_paths = result.volatility_paths

        if has_pnl:
            _add_paths(
                fig,
                time_grid,
                vol_paths * 100,
                idx_sample[profit_mask],
                _GREEN,
                None,
                "profit",
                row=2,
                show_legend=False,
                hover_fmt="<b>Volatility:</b> %{y:.2f}%",
            )
            _add_paths(
                fig,
                time_grid,
                vol_paths * 100,
                idx_sample[~profit_mask],
                _RED,
                None,
                "loss",
                row=2,
                show_legend=False,
                hover_fmt="<b>Volatility:</b> %{y:.2f}%",
            )
        else:
            _add_paths(
                fig,
                time_grid,
                vol_paths * 100,
                idx_sample,
                _VOL_ORANGE,
                "Vol Path",
                "vol",
                row=2,
                hover_fmt="<b>Volatility:</b> %{y:.2f}%",
            )

        # Vol 5-95 percentile bands
        if show_bands:
            vol_pct = np.percentile(vol_paths * 100, [5, 95], axis=0)
            fig.add_trace(
                go.Scatter(
                    x=time_grid,
                    y=vol_pct[1],
                    mode="lines",
                    line=dict(width=1.2, color=_VOL_BAND_EDGE, dash="dot"),
                    name="Vol 95th",
                    legendgroup="vol_band",
                    hovertemplate="<b>Time:</b> %{x:.2f} yr<br><b>Volatility 95th:</b> %{y:.2f}%<extra></extra>",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=time_grid,
                    y=vol_pct[0],
                    mode="lines",
                    line=dict(width=1.2, color=_VOL_BAND_EDGE, dash="dot"),
                    fill="tonexty",
                    fillcolor=_VOL_BAND_FILL,
                    name="Vol 5th",
                    legendgroup="vol_band",
                    hovertemplate="<b>Time:</b> %{x:.2f} yr<br><b>Volatility 5th:</b> %{y:.2f}%<extra></extra>",
                ),
                row=2,
                col=1,
            )

        # Vol reference lines (V0, theta, sigma0, long-run)
        _add_vol_references(fig, model_key, params)

    else:
        # Constant vol (GBM, Merton)
        initial_vol = get_initial_volatility(model_key, params) * 100
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=[initial_vol] * len(time_grid),
                mode="lines",
                line=dict(width=1.5, color="rgba(255,160,50,0.7)"),
                name=f"\u03c3 = {initial_vol:.1f} %",
                hovertemplate="<b>Time:</b> %{x:.2f} yr<br><b>Volatility:</b> "
                + f"{initial_vol:.2f}%"
                + "<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # ── Layout ────────────────────────────────────────────────────────────

    vol_label = "Volatility Paths" if has_vol else "Volatility (Constant)"

    fig.update_layout(
        height=chart_height,
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
            itemsizing="constant",
            tracegroupgap=0,
        ),
        margin=dict(t=50, b=35, l=60, r=20),
    )

    # Y-axes
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, **AXIS_STYLE)
    fig.update_yaxes(title_text=vol_label, row=2, col=1, **AXIS_STYLE)

    # X-axes
    fig.update_xaxes(row=1, col=1, **AXIS_STYLE)
    fig.update_xaxes(title_text="Time (years)", row=2, col=1, **AXIS_STYLE)

    # Hide x-axis labels on row 1 when shared
    if has_vol:
        fig.update_xaxes(showticklabels=False, row=1, col=1)

    # Custom overlays (e.g. structured product barriers/triggers)
    if overlay_fn is not None:
        overlay_fn(fig, result.price_paths, time_grid, idx_sample)

    spread_annotations(fig)
    st.plotly_chart(fig, width="stretch")


# ── helpers ───────────────────────────────────────────────────────────────


def _add_paths(
    fig,
    time_grid,
    data,
    indices,
    color,
    legend_name,
    legend_group,
    row=1,
    show_legend=True,
    hover_fmt="<b>Price:</b> $%{y:.2f}",
):
    """Add a batch of paths to the figure."""
    ht = f"<b>Time:</b> %{{x:.2f}} yr<br>{hover_fmt}<extra></extra>"
    first = True
    for idx in indices:
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=data[idx],
                mode="lines",
                line=dict(width=0.7, color=color),
                name=legend_name if (first and show_legend) else None,
                showlegend=(first and show_legend and legend_name is not None),
                legendgroup=legend_group,
                hovertemplate=ht,
            ),
            row=row,
            col=1,
        )
        first = False


# ── Exotic overlay logic ──────────────────────────────────────────────────


def _add_exotic_overlays(
    fig: go.Figure,
    price_paths: np.ndarray,
    time_grid: np.ndarray,
    idx_sample: np.ndarray,
    exotic_metadata: list[dict],
    params: dict[str, Any],
    exotic_viz: dict[str, Any] | None = None,
) -> None:
    """Add exotic-specific visual overlays to the price path chart."""
    time_horizon = params.get("time_horizon", 1.0)
    eviz = exotic_viz or {}

    for meta in exotic_metadata:
        inst = meta.get("instrument_class", "vanilla")
        if inst == "vanilla":
            continue

        if inst == "barrier":
            _overlay_barrier(
                fig,
                price_paths,
                time_grid,
                idx_sample,
                meta,
                show_hits=eviz.get("barrier_show_hits", True),
                show_dead=eviz.get("barrier_show_dead", True),
                max_paths=eviz.get("barrier_n_paths", 200),
            )
        elif inst in ("digital", "asset_or_nothing"):
            if eviz.get("digital_show_zone", True):
                _overlay_digital_zone(fig, price_paths, time_grid, meta)
        elif inst == "asian":
            n_avg = eviz.get("asian_n_avg", 5)
            if n_avg > 0:
                _overlay_running_average(
                    fig, price_paths, time_grid, idx_sample, n_highlight=n_avg
                )
        elif inst == "lookback_floating":
            n_ext = eviz.get("lookback_n_extreme", 5)
            if n_ext > 0:
                _overlay_running_extreme(
                    fig,
                    price_paths,
                    time_grid,
                    idx_sample,
                    is_call=(meta.get("option_type") == "call"),
                    n_highlight=n_ext,
                )
        elif inst == "lookback_fixed":
            n_ext = eviz.get("lookback_n_extreme", 5)
            if n_ext > 0:
                _overlay_running_extreme(
                    fig,
                    price_paths,
                    time_grid,
                    idx_sample,
                    is_call=(meta.get("option_type") == "call"),
                    fixed=True,
                    n_highlight=n_ext,
                )
        elif inst == "chooser":
            choice_pct = meta.get("choice_time_pct", 0.5)
            _overlay_chooser(fig, time_grid, choice_pct, time_horizon)
        elif inst == "gap":
            _overlay_gap(fig, meta)
        elif inst == "power":
            _overlay_power_breakeven(fig, meta)


def _overlay_barrier(
    fig: go.Figure,
    price_paths: np.ndarray,
    time_grid: np.ndarray,
    idx_sample: np.ndarray,
    meta: dict,
    show_hits: bool = True,
    show_dead: bool = True,
    max_paths: int = 200,
) -> None:
    """Barrier overlays: barrier hline + first-hit markers + greyed post-hit segments."""
    barrier = meta.get("barrier", 0.0)
    is_up = meta.get("is_up", True)
    is_knock_in = meta.get("is_knock_in", False)

    if barrier <= 0:
        return

    # 1) Barrier horizontal line — always shown
    ko_ki = "KI" if is_knock_in else "KO"
    direction = "\u2191" if is_up else "\u2193"
    fig.add_hline(
        y=barrier,
        line_dash="dash",
        line_color=_BARRIER_LINE,
        line_width=1.8,
        annotation_text=f" {direction}{ko_ki} ${barrier:.0f} ",
        annotation_position="bottom left" if is_up else "top left",
        **_badge(_BARRIER_LINE),
        row=1,
        col=1,
    )

    if not show_hits and not show_dead:
        return

    # 2) First-hit markers + greyed-out post-hit segments
    hit_color = _BARRIER_HIT_KI if is_knock_in else _BARRIER_HIT_KO
    hit_symbol = "circle" if is_knock_in else "x"
    hit_label = "Activated" if is_knock_in else "Knocked Out"

    hit_times = []
    hit_prices = []

    max_overlay_paths = min(len(idx_sample), max_paths)

    for idx in idx_sample[:max_overlay_paths]:
        path = price_paths[idx]
        if is_up:
            crossings = np.where(path >= barrier)[0]
        else:
            crossings = np.where(path <= barrier)[0]

        if len(crossings) > 0:
            first_cross = crossings[0]
            if show_hits:
                hit_times.append(time_grid[first_cross])
                hit_prices.append(path[first_cross])

            if show_dead and not is_knock_in and first_cross < len(path) - 1:
                fig.add_trace(
                    go.Scatter(
                        x=time_grid[first_cross:],
                        y=path[first_cross:],
                        mode="lines",
                        line=dict(width=1.2, color=_BARRIER_DEAD),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )

    if show_hits and hit_times:
        fig.add_trace(
            go.Scatter(
                x=hit_times,
                y=hit_prices,
                mode="markers",
                marker=dict(
                    size=5 if len(hit_times) > 50 else 7,
                    color=hit_color,
                    symbol=hit_symbol,
                    line=dict(width=0.5, color="white"),
                ),
                name=hit_label,
                legendgroup="barrier_hit",
                hovertemplate=(
                    f"<b>{hit_label}</b><br>"
                    "<b>Time:</b> %{x:.3f} yr<br>"
                    "<b>Price:</b> $%{y:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )


def _overlay_digital_zone(
    fig: go.Figure,
    price_paths: np.ndarray,
    time_grid: np.ndarray,
    meta: dict,
) -> None:
    """Digital / Asset-or-Nothing: shade payoff zone above or below strike."""
    strike = meta.get("strike", 0.0)
    is_call = meta.get("option_type") == "call"
    inst = meta.get("instrument_class", "digital")

    if strike <= 0:
        return

    label = "Digital" if inst == "digital" else "AoN"

    # Use hrect for the payoff zone
    if is_call:
        y_max = float(np.max(price_paths)) * 1.1
        fig.add_hrect(
            y0=strike,
            y1=y_max,
            fillcolor=_DIGITAL_ZONE,
            line_width=0,
            row=1,
            col=1,
        )
    else:
        y_min = float(np.min(price_paths)) * 0.9
        fig.add_hrect(
            y0=y_min,
            y1=strike,
            fillcolor=_DIGITAL_ZONE,
            line_width=0,
            row=1,
            col=1,
        )

    # Strike line — RIGHT side with badge (barriers use LEFT)
    fig.add_hline(
        y=strike,
        line_dash="dashdot",
        line_color="#8b5cf6",
        line_width=1.2,
        annotation_text=f" {label} K=${strike:.0f} ",
        annotation_position="top right",
        **_badge("#8b5cf6"),
        row=1,
        col=1,
    )


def _overlay_running_average(
    fig: go.Figure,
    price_paths: np.ndarray,
    time_grid: np.ndarray,
    idx_sample: np.ndarray,
    n_highlight: int = 5,
) -> None:
    """Asian option: show running geometric average on a few highlighted paths."""
    highlight_idx = idx_sample[
        np.linspace(0, len(idx_sample) - 1, n_highlight, dtype=int)
    ]

    first = True
    for idx in highlight_idx:
        path = price_paths[idx]
        log_prices = np.log(np.maximum(path, 1e-10))
        cum_avg = np.cumsum(log_prices) / np.arange(1, len(log_prices) + 1)
        geo_avg = np.exp(cum_avg)

        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=geo_avg,
                mode="lines",
                line=dict(width=1.3, color=_ASIAN_AVG, dash="dot"),
                name="Geo. Average" if first else None,
                showlegend=first,
                legendgroup="asian_avg",
                hovertemplate="<b>Time:</b> %{x:.2f} yr<br><b>Geo. Avg:</b> $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        first = False


def _overlay_running_extreme(
    fig: go.Figure,
    price_paths: np.ndarray,
    time_grid: np.ndarray,
    idx_sample: np.ndarray,
    is_call: bool = True,
    fixed: bool = False,
    n_highlight: int = 5,
) -> None:
    """Lookback option: show running min (floating call) or running max on highlighted paths."""
    highlight_idx = idx_sample[
        np.linspace(0, len(idx_sample) - 1, n_highlight, dtype=int)
    ]

    if fixed:
        use_max = is_call
        label = "Running Max" if use_max else "Running Min"
    else:
        use_max = not is_call
        label = "Running Min" if is_call else "Running Max"

    first = True
    for idx in highlight_idx:
        path = price_paths[idx]
        if use_max:
            running = np.maximum.accumulate(path)
        else:
            running = np.minimum.accumulate(path)

        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=running,
                mode="lines",
                line=dict(width=1.3, color=_LOOKBACK_EXTREME, dash="dot"),
                name=label if first else None,
                showlegend=first,
                legendgroup="lookback_ext",
                hovertemplate=f"<b>Time:</b> %{{x:.2f}} yr<br><b>{label}:</b> $%{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        first = False


def _overlay_chooser(
    fig: go.Figure,
    time_grid: np.ndarray,
    choice_pct: float,
    time_horizon: float,
) -> None:
    """Chooser option: vertical line at choice time."""
    t_choice = choice_pct * time_horizon

    fig.add_vline(
        x=t_choice,
        line_dash="dash",
        line_color=_CHOOSER_LINE,
        line_width=1.5,
        annotation_text=f" Choice t={t_choice:.2f}y ",
        annotation_position="top right",
        **_badge(_CHOOSER_LINE),
        row=1,
        col=1,
    )


def _overlay_gap(fig: go.Figure, meta: dict) -> None:
    """Gap option: two horizontal lines for strike K1 and trigger K2."""
    strike = meta.get("strike", 0.0)
    trigger = meta.get("gap_trigger", strike)

    # Opposite sides to avoid overlap
    if trigger > 0 and trigger != strike:
        fig.add_hline(
            y=trigger,
            line_dash="dashdot",
            line_color=_GAP_TRIGGER,
            line_width=1.5,
            annotation_text=f" K\u2082=${trigger:.0f} ",
            annotation_position="top left",
            **_badge(_GAP_TRIGGER),
            row=1,
            col=1,
        )
    if strike > 0:
        fig.add_hline(
            y=strike,
            line_dash="dot",
            line_color="#60a5fa",
            line_width=1.2,
            annotation_text=f" K\u2081=${strike:.0f} ",
            annotation_position="top right",
            **_badge("#60a5fa"),
            row=1,
            col=1,
        )


def _overlay_power_breakeven(fig: go.Figure, meta: dict) -> None:
    """Power option: show effective breakeven at K^(1/n)."""
    strike = meta.get("strike", 0.0)
    n = meta.get("power_n", 2.0)

    if strike <= 0 or n <= 0:
        return

    effective_be = strike ** (1.0 / n)
    fig.add_hline(
        y=effective_be,
        line_dash="dashdot",
        line_color=_POWER_BE,
        line_width=1.5,
        annotation_text=f" S*=${effective_be:.1f} ",
        annotation_position="top left",
        **_badge(_POWER_BE),
        row=1,
        col=1,
    )


def _add_vol_references(
    fig: go.Figure,
    model_key: str,
    params: dict[str, Any],
) -> None:
    """Add initial-vol and long-run-vol reference lines to the vol panel."""
    model_lower = model_key.lower()

    # Collect both values to detect overlap
    init_pct: float | None = None
    lr_pct: float | None = None

    if model_lower in ("heston", "bates"):
        init_pct = np.sqrt(params.get("v0", 0.04)) * 100
        lr_pct = np.sqrt(params.get("theta", 0.04)) * 100
    elif model_lower in ("garch", "ngarch", "gjr_garch"):
        init_pct = params.get("sigma0", 0.20) * 100
        lr_vol = compute_long_run_volatility(model_key, params)
        if lr_vol is not None:
            lr_pct = lr_vol * 100

    # Build labels
    if model_lower in ("heston", "bates"):
        init_label = (
            f" \u221aV\u2080 = {init_pct:.1f}% " if init_pct is not None else None
        )
        lr_label = f" \u221a\u03b8 = {lr_pct:.1f}% " if lr_pct is not None else None
    else:
        init_label = (
            f" \u03c3\u2080 = {init_pct:.1f}% " if init_pct is not None else None
        )
        lr_label = f" LR = {lr_pct:.1f}% " if lr_pct is not None else None

    # ── Initial volatility — label on the LEFT ─────────────────────────
    if init_pct is not None and init_label is not None:
        fig.add_hline(
            y=init_pct,
            line_dash="dash",
            line_color=_VOL_REF_INIT,
            line_width=1.5,
            annotation_text=init_label,
            annotation_position="top left",
            **_badge(_VOL_REF_INIT),
            row=2,
            col=1,
        )

    # ── Long-run volatility — label on the RIGHT ────────────────────────
    if lr_pct is not None and lr_label is not None:
        fig.add_hline(
            y=lr_pct,
            line_dash="dot",
            line_color=_VOL_REF_LR,
            line_width=1.5,
            annotation_text=lr_label,
            annotation_position="top right",
            **_badge(_VOL_REF_LR),
            row=2,
            col=1,
        )


def render_path_controls(
    exotic_metadata: list[dict] | None = None,
    sp_config: dict | None = None,
) -> dict[str, Any]:
    """Controls for path visualization — exotic and structured product overlays."""

    # ── styled helpers ───────────────────────────────────────────────
    def _section(title: str, color: str):
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:0.45rem;"
            f"margin:0.75rem 0 0.3rem;padding-bottom:0.2rem;"
            f"border-bottom:1px solid {color}30;'>"
            f"<span style='width:7px;height:7px;border-radius:50%;"
            f"background:{color};display:inline-block;flex-shrink:0;'></span>"
            f"<span style='font-size:0.7rem;font-weight:700;color:{color};"
            f"text-transform:uppercase;letter-spacing:0.06em;'>"
            f"{title}</span></div>",
            unsafe_allow_html=True,
        )

    def _label(text: str, color: str):
        st.markdown(
            f"<div style='font-size:0.73rem;font-weight:600;color:{color};"
            f"margin:0.35rem 0 0.1rem;opacity:0.9;'>"
            f"\u25b8 {text}</div>",
            unsafe_allow_html=True,
        )

    # ═══ General display ═════════════════════════════════════════════
    _section("Display", "#94a3b8")
    show_bands = st.checkbox("Show 5-95% Bands", value=True)
    n_paths = st.slider("Display Paths", 20, 500, 150, 10)

    result: dict[str, Any] = {"n_display": n_paths, "show_bands": show_bands}

    # ═══ Exotic overlays ═════════════════════════════════════════════
    exotic_types: set[str] = set()
    if exotic_metadata:
        exotic_types = {
            m.get("instrument_class", "vanilla") for m in exotic_metadata
        } - {"vanilla"}

    if exotic_types:
        _section("Exotic Overlays", "#7c3aed")

        exotic_viz: dict[str, Any] = {}
        exotic_viz["show_overlays"] = st.checkbox(
            "Show exotic overlays",
            value=True,
            key="viz_exotic_toggle",
        )

        if exotic_viz["show_overlays"]:
            # Barrier
            if "barrier" in exotic_types:
                _label("Barrier", _BARRIER_LINE)
                c1, c2 = st.columns(2)
                with c1:
                    exotic_viz["barrier_show_hits"] = st.checkbox(
                        "Hit markers",
                        value=True,
                        key="viz_barrier_hits",
                    )
                with c2:
                    exotic_viz["barrier_show_dead"] = st.checkbox(
                        "Grey knocked-out",
                        value=True,
                        key="viz_barrier_dead",
                    )
                if exotic_viz.get("barrier_show_hits") or exotic_viz.get(
                    "barrier_show_dead"
                ):
                    exotic_viz["barrier_n_paths"] = st.slider(
                        "Overlay paths",
                        0,
                        min(n_paths, 200),
                        50,
                        5,
                        key="viz_barrier_n",
                        help="Paths showing barrier hit markers and KO segments",
                    )

            # Asian
            if "asian" in exotic_types:
                _label("Asian", _ASIAN_AVG)
                exotic_viz["asian_n_avg"] = st.slider(
                    "Running average paths",
                    0,
                    15,
                    5,
                    1,
                    key="viz_asian_n",
                    help="Paths showing the running geometric mean",
                )

            # Lookback
            if "lookback_floating" in exotic_types or "lookback_fixed" in exotic_types:
                _label("Lookback", _LOOKBACK_EXTREME)
                exotic_viz["lookback_n_extreme"] = st.slider(
                    "Running min/max paths",
                    0,
                    15,
                    5,
                    1,
                    key="viz_lookback_n",
                    help="Paths showing the running extreme",
                )

            # Digital / Asset-or-Nothing
            if "digital" in exotic_types or "asset_or_nothing" in exotic_types:
                _label("Digital / Asset-or-Nothing", "#8b5cf6")
                exotic_viz["digital_show_zone"] = st.checkbox(
                    "Payoff zone shading",
                    value=True,
                    key="viz_digital_zone",
                )

        result["exotic_viz"] = exotic_viz

    # ═══ Structured product overlays ═════════════════════════════════
    if sp_config:
        product_type = sp_config.get("product_type", "")
        sp_viz: dict[str, Any] = {}

        if product_type == "reverse_convertible":
            _section("Product Overlays", "#0891b2")
            _label("Reverse Convertible \u2014 Barrier", "#ef4444")
            sp_viz["rc_n_markers"] = st.slider(
                "Breach marker paths",
                0,
                min(n_paths, 100),
                30,
                5,
                key="viz_rc_markers",
                help="Paths showing KI barrier breach markers",
            )

        elif product_type == "autocallable":
            _section("Product Overlays", "#0891b2")
            _label("Autocallable \u2014 Events", "#22c55e")
            sp_viz["autocall_n_markers"] = st.slider(
                "Event marker paths",
                0,
                min(n_paths, 100),
                30,
                5,
                key="viz_autocall_markers",
                help="Paths showing autocall event markers",
            )

        if sp_viz:
            result["sp_viz"] = sp_viz

    return result
