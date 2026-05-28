"""
P&L Analysis Charts — Simulated P&L scatter with marginal histogram + 3D scatter.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config.chart_theme import (
    ANN_BG as _ANN_BG,
    ANN_FONT as _ANN_FONT,
    AXIS_LINE as _AXIS_LINE,
    AXIS_LABEL as _AXIS_LABEL,
    AXIS_STYLE as _AXIS_STYLE,
    CHART_HEIGHT_LG,
    LEGEND_COLOR as _LEGEND_COLOR,
    PAPER_BG as _PAPER_BG,
    PLOT_BG as _PLOT_BG,
    TICK_COLOR as _TICK_COLOR,
    badge as _badge,
)

from backend.simulation.base import SimulationResult

_GREEN_MARKER = "rgba(34, 197, 94, 0.55)"
_RED_MARKER = "rgba(239, 68, 68, 0.45)"
_GREEN_HIST = "rgba(34, 197, 94, 0.60)"
_RED_HIST = "rgba(239, 68, 68, 0.50)"
_ZERO_LINE = "rgba(255, 255, 255, 0.30)"

_BARRIER_LINE_PNL = "#f59e0b"
_GAP_TRIGGER_PNL = "#fb923c"
_DIGITAL_ZONE_PNL = "rgba(139, 92, 246, 0.06)"
_POWER_BE_PNL = "#14b8a6"
_CHOOSER_LINE_PNL = "#d946ef"


def render_payoff_with_distribution(
    result: SimulationResult,
    pnl_values: np.ndarray,
    breakeven_prices: np.ndarray | None = None,
    spot: float = 100.0,
    max_scatter: int = 5000,
    exotic_metadata: list[dict] | None = None,
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
        rows=1,
        cols=2,
        column_widths=[0.82, 0.18],
        shared_yaxes=True,
        horizontal_spacing=0.02,
    )

    # Scatter — profit paths (green)
    fig.add_trace(
        go.Scatter(
            x=tp[profit_mask],
            y=pnl[profit_mask],
            mode="markers",
            marker=dict(size=3.5, color=_GREEN_MARKER),
            name="Profit",
            legendgroup="profit",
            hovertemplate="<b>S(T):</b> $%{x:.2f}<br><b>P&L:</b> $%{y:+.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Scatter — loss paths (red)
    fig.add_trace(
        go.Scatter(
            x=tp[~profit_mask],
            y=pnl[~profit_mask],
            mode="markers",
            marker=dict(size=3.5, color=_RED_MARKER),
            name="Loss",
            legendgroup="loss",
            hovertemplate="<b>S(T):</b> $%{x:.2f}<br><b>P&L:</b> $%{y:+.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # P&L = 0 line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=_ZERO_LINE,
        line_width=1,
        row=1,
        col=1,
    )

    # Spot reference — always top-left (isolated from other annotations)
    fig.add_vline(
        x=spot,
        line_dash="dot",
        line_color="rgba(255,255,255,0.25)",
        line_width=1,
        annotation_text=f" S\u2080={spot:.0f} ",
        annotation_position="top left",
        **_badge("rgba(255,255,255,0.5)"),
        row=1,
        col=1,
    )

    # ── Collect vline annotations, then render with smart positions ────────
    _vline_annotations: list[dict] = []

    # Breakeven vertical lines
    if breakeven_prices is not None:
        for be in breakeven_prices:
            _vline_annotations.append(
                dict(
                    x=float(be),
                    line_dash="dot",
                    color="#a78bfa",
                    line_width=1,
                    text=f" BE ${be:.1f} ",
                )
            )

    # Exotic-specific annotations
    if exotic_metadata:
        for meta in exotic_metadata:
            inst = meta.get("instrument_class", "vanilla")
            if inst == "vanilla":
                continue

            if inst == "barrier":
                barrier = meta.get("barrier", 0.0)
                is_up = meta.get("is_up", True)
                is_ki = meta.get("is_knock_in", False)
                if barrier > 0:
                    ko_ki = "KI" if is_ki else "KO"
                    direction = "\u2191" if is_up else "\u2193"
                    _vline_annotations.append(
                        dict(
                            x=float(barrier),
                            line_dash="dash",
                            color=_BARRIER_LINE_PNL,
                            line_width=1.5,
                            text=f" {direction}{ko_ki} ${barrier:.0f} ",
                        )
                    )

            elif inst in ("digital", "asset_or_nothing"):
                strike = meta.get("strike", 0.0)
                is_call = meta.get("option_type") == "call"
                if strike > 0:
                    if is_call:
                        fig.add_vrect(
                            x0=strike,
                            x1=float(np.max(tp)) * 1.05,
                            fillcolor=_DIGITAL_ZONE_PNL,
                            line_width=0,
                            row=1,
                            col=1,
                        )
                    else:
                        fig.add_vrect(
                            x0=float(np.min(tp)) * 0.95,
                            x1=strike,
                            fillcolor=_DIGITAL_ZONE_PNL,
                            line_width=0,
                            row=1,
                            col=1,
                        )

            elif inst == "gap":
                trigger = meta.get("gap_trigger", 0.0)
                if trigger > 0:
                    _vline_annotations.append(
                        dict(
                            x=float(trigger),
                            line_dash="dashdot",
                            color=_GAP_TRIGGER_PNL,
                            line_width=1.5,
                            text=f" K\u2082=${trigger:.0f} ",
                        )
                    )

            elif inst == "power":
                strike = meta.get("strike", 0.0)
                n = meta.get("power_n", 2.0)
                if strike > 0 and n > 0:
                    eff_be = strike ** (1.0 / n)
                    _vline_annotations.append(
                        dict(
                            x=float(eff_be),
                            line_dash="dashdot",
                            color=_POWER_BE_PNL,
                            line_width=1.5,
                            text=f" S*=${eff_be:.1f} ",
                        )
                    )

    # Render vlines — alternate top-right / bottom-right (all on right side,
    # S₀ is already isolated on the left)
    _right_positions = ["top right", "bottom right"]
    _vline_annotations.sort(key=lambda a: a["x"])
    for i, ann in enumerate(_vline_annotations):
        fig.add_vline(
            x=ann["x"],
            line_dash=ann["line_dash"],
            line_color=ann["color"],
            line_width=ann["line_width"],
            annotation_text=ann["text"],
            annotation_position=_right_positions[i % 2],
            **_badge(ann["color"]),
            row=1,
            col=1,
        )

    # ── Marginal histogram of P&L (right panel) ─────────────────────────
    profit_pnl = pnl_values[pnl_values >= 0]
    loss_pnl = pnl_values[pnl_values < 0]

    n_bins = 60
    all_min, all_max = float(np.min(pnl_values)), float(np.max(pnl_values))
    bin_size = (all_max - all_min) / n_bins if all_max > all_min else 1.0

    if len(profit_pnl) > 0:
        fig.add_trace(
            go.Histogram(
                y=profit_pnl,
                ybins=dict(start=0, end=all_max, size=bin_size),
                marker_color=_GREEN_HIST,
                showlegend=False,
                hovertemplate="<b>P&L:</b> $%{y:+.2f}<br><b>Count:</b> %{x}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    if len(loss_pnl) > 0:
        fig.add_trace(
            go.Histogram(
                y=loss_pnl,
                ybins=dict(start=all_min, end=0, size=bin_size),
                marker_color=_RED_HIST,
                showlegend=False,
                hovertemplate="<b>P&L:</b> $%{y:+.2f}<br><b>Count:</b> %{x}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    # ── Layout ───────────────────────────────────────────────────────────
    fig.update_layout(
        height=550,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.4,
            font=dict(size=11, color=_LEGEND_COLOR),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=50, b=45, l=60, r=20),
        bargap=0.02,
    )

    fig.update_xaxes(
        title_text="Terminal Price S(T)",
        row=1,
        col=1,
        tickprefix="$",
        tickformat=",.2f",
        hoverformat=",.2f",
        **_AXIS_STYLE,
    )
    fig.update_yaxes(
        title_text="P&L ($)",
        row=1,
        col=1,
        tickprefix="$",
        tickformat=",.2f",
        hoverformat=",.2f",
        **_AXIS_STYLE,
    )
    fig.update_xaxes(
        title_text="Count", row=1, col=2, showticklabels=False, **_AXIS_STYLE
    )
    fig.update_yaxes(row=1, col=2, showticklabels=False, **_AXIS_STYLE)

    st.plotly_chart(fig, width="stretch")


# ═════════════════════════════════════════════════════════════════════════
# P&L by DTE — scatter + histogram + dynamic breakevens with DTE slider
# ═════════════════════════════════════════════════════════════════════════

_BASE_TRACES = 4  # profit scatter, loss scatter, profit hist, loss hist


def render_pnl_by_dte(
    result: SimulationResult,
    position_arrays: dict,
    rate: float,
    sigma: float,
    time_to_expiry: float,
    spot: float = 100.0,
    n_checkpoints: int = 8,
    max_scatter: int = 5000,
    exotic_metadata: list[dict] | None = None,
    terminal_pnl: np.ndarray | None = None,
    terminal_breakevens: np.ndarray | None = None,
) -> None:
    """
    Unified P&L chart: scatter (green/red) + histogram + dynamic breakevens,
    navigable via a Plotly DTE slider.

    Merges the former *Mark-to-Market P&L by DTE* and *Payoff Diagram* into a
    single visualisation.  DTE=0 corresponds to the classical expiry payoff.

    Parameters
    ----------
    terminal_pnl : np.ndarray, optional
        Pre-computed MC P&L at expiry (used at DTE=0 instead of BS repricing,
        important for exotic legs whose payoff is path-dependent).
    terminal_breakevens : np.ndarray, optional
        Pre-computed breakeven prices at expiry (from hybrid payoff curve).
    """
    from scipy.stats import norm as sp_norm
    from backend.portfolio.pnl import find_breakeven_points

    strikes = position_arrays["strikes"]
    option_types = position_arrays["option_types"]
    position_types = position_arrays["position_types"]
    quantities = position_arrays["quantities"]
    premiums = position_arrays["premiums"]
    multiplier = 100.0
    n_legs = len(strikes)

    if n_legs == 0:
        return

    n_paths = result.n_paths
    n_steps = result.n_steps
    dt = time_to_expiry / n_steps

    # Subsample for performance
    if n_paths > max_scatter:
        idx = np.random.choice(n_paths, max_scatter, replace=False)
    else:
        idx = np.arange(n_paths)
    paths = result.price_paths[idx]

    stock_qty = position_arrays.get("stock_quantity", 0.0)
    stock_entry = position_arrays.get("stock_entry_price", 0.0)

    # ── DTE checkpoints (ascending: 0 → T-1, exclude T_days) ─────────
    T_days = int(round(time_to_expiry * 365))
    if T_days <= 30:
        step_d = max(T_days // n_checkpoints, 1)
    else:
        step_d = max(T_days // n_checkpoints, 5)
    dte_days = list(range(0, T_days, step_d))  # exclude T_days
    dte_days = sorted(set(dte_days))  # ascending, deduplicated
    if not dte_days or dte_days[0] != 0:
        dte_days.insert(0, 0)
    n_dte = len(dte_days)

    # ── Precompute P&L (MC scatter) + breakevens (BS curve) per DTE ──
    all_pnl: dict[int, np.ndarray] = {}
    all_spots: dict[int, np.ndarray] = {}
    all_breakevens: dict[int, np.ndarray] = {}
    global_pnl_min = float("inf")
    global_pnl_max = float("-inf")

    be_spot_grid = np.linspace(spot * 0.5, spot * 1.5, 1000)

    for dte in dte_days:
        tau = dte / 365.0
        elapsed = time_to_expiry - tau
        step_idx = min(int(round(elapsed / dt)), n_steps)
        S_t = paths[:, step_idx]

        # ── At DTE=0, use real MC P&L if provided (exact for exotics) ─
        if dte == 0 and terminal_pnl is not None:
            pnl = np.round(terminal_pnl[idx], 2)
        else:
            from services.simulation_runner import compute_mtm_pnl_at_step

            pnl = np.round(
                compute_mtm_pnl_at_step(
                    paths=paths,
                    step_idx=step_idx,
                    position_arrays=position_arrays,
                    rate=rate,
                    sigma=sigma,
                    time_to_expiry=time_to_expiry,
                ),
                2,
            )

        all_pnl[dte] = pnl
        all_spots[dte] = S_t
        global_pnl_min = min(global_pnl_min, float(np.min(pnl)))
        global_pnl_max = max(global_pnl_max, float(np.max(pnl)))

        # ── Breakevens ────────────────────────────────────────────────
        if dte == 0 and terminal_breakevens is not None:
            # Use pre-computed breakevens (exact for exotics)
            all_breakevens[dte] = terminal_breakevens
        else:
            # Theoretical BS payoff curve → breakevens
            be_pnl = np.zeros_like(be_spot_grid)
            for j in range(n_legs):
                K = strikes[j]
                is_call = option_types[j] == 1
                d = position_types[j]
                qty = quantities[j]
                prem = premiums[j]

                if tau < 1e-10:
                    v = (
                        np.maximum(be_spot_grid - K, 0)
                        if is_call
                        else np.maximum(K - be_spot_grid, 0)
                    )
                else:
                    sqrt_tau = np.sqrt(tau)
                    d1 = (np.log(be_spot_grid / K) + (rate + 0.5 * sigma**2) * tau) / (
                        sigma * sqrt_tau
                    )
                    d2 = d1 - sigma * sqrt_tau
                    if is_call:
                        v = be_spot_grid * sp_norm.cdf(d1) - K * np.exp(
                            -rate * tau
                        ) * sp_norm.cdf(d2)
                    else:
                        v = K * np.exp(-rate * tau) * sp_norm.cdf(
                            -d2
                        ) - be_spot_grid * sp_norm.cdf(-d1)

                be_pnl += d * (v - prem) * qty * multiplier

            if stock_qty != 0.0:
                be_pnl += stock_qty * (be_spot_grid - stock_entry)

            all_breakevens[dte] = find_breakeven_points(be_pnl, be_spot_grid)

    # ── Determine max breakeven count (for fixed trace count per DTE) ─
    max_be = max((len(b) for b in all_breakevens.values()), default=0)
    traces_per_dte = _BASE_TRACES + max_be

    # ── Histogram bins (consistent across all DTEs) ───────────────────
    n_bins = 60
    bin_size = (
        (global_pnl_max - global_pnl_min) / n_bins
        if global_pnl_max > global_pnl_min
        else 1.0
    )

    # ── Build figure: subplot scatter (82%) + histogram (18%) ─────────
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.82, 0.18],
        shared_yaxes=True,
        horizontal_spacing=0.02,
    )

    for i, dte in enumerate(dte_days):
        pnl = all_pnl[dte]
        S_t = all_spots[dte]
        profit_mask = pnl >= 0
        is_default = i == n_dte - 1  # show max DTE by default

        # Profit scatter
        fig.add_trace(
            go.Scatter(
                x=S_t[profit_mask],
                y=pnl[profit_mask],
                mode="markers",
                marker=dict(size=3.5, color=_GREEN_MARKER),
                name="Profit",
                legendgroup="profit",
                showlegend=is_default,
                visible=is_default,
                hovertemplate=(
                    f"<b>S(t):</b> $%{{x:.2f}}<br>"
                    f"<b>P&L:</b> $%{{y:+.2f}}<br>"
                    f"<b>DTE:</b> {dte}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # Loss scatter
        fig.add_trace(
            go.Scatter(
                x=S_t[~profit_mask],
                y=pnl[~profit_mask],
                mode="markers",
                marker=dict(size=3.5, color=_RED_MARKER),
                name="Loss",
                legendgroup="loss",
                showlegend=is_default,
                visible=is_default,
                hovertemplate=(
                    f"<b>S(t):</b> $%{{x:.2f}}<br>"
                    f"<b>P&L:</b> $%{{y:+.2f}}<br>"
                    f"<b>DTE:</b> {dte}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # Profit histogram
        profit_pnl = pnl[pnl >= 0]
        fig.add_trace(
            go.Histogram(
                y=profit_pnl if len(profit_pnl) > 0 else [],
                ybins=dict(start=0, end=global_pnl_max, size=bin_size),
                marker_color=_GREEN_HIST,
                showlegend=False,
                visible=is_default,
                hovertemplate=(
                    "<b>P&L:</b> $%{y:+.2f}<br><b>Count:</b> %{x}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

        # Loss histogram
        loss_pnl = pnl[pnl < 0]
        fig.add_trace(
            go.Histogram(
                y=loss_pnl if len(loss_pnl) > 0 else [],
                ybins=dict(start=global_pnl_min, end=0, size=bin_size),
                marker_color=_RED_HIST,
                showlegend=False,
                visible=is_default,
                hovertemplate=(
                    "<b>P&L:</b> $%{y:+.2f}<br><b>Count:</b> %{x}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

        # Breakeven vertical lines (toggled with slider — lines only, no text)
        bes = all_breakevens[dte]
        for b_idx in range(max_be):
            if b_idx < len(bes):
                be_price = float(bes[b_idx])
                fig.add_trace(
                    go.Scatter(
                        x=[be_price, be_price],
                        y=[global_pnl_min, global_pnl_max],
                        mode="lines",
                        line=dict(color="#a78bfa", width=1, dash="dot"),
                        showlegend=False,
                        visible=is_default,
                        hovertemplate=f"<b>Breakeven:</b> ${be_price:.2f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
            else:
                # Pad with empty trace to keep fixed count
                fig.add_trace(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode="lines",
                        showlegend=False,
                        visible=False,
                    ),
                    row=1,
                    col=1,
                )

    # ── Helper: build a badge-style annotation dict ───────────────────
    def _ann(x, text, color, yref="paper", y=0.97, xanchor="left"):
        return dict(
            x=x,
            y=y,
            xref="x",
            yref=yref,
            text=text,
            showarrow=False,
            xanchor=xanchor,
            yanchor="top" if y > 0.5 else "bottom",
            font=dict(size=_ANN_FONT, color=color),
            bgcolor=_ANN_BG,
            bordercolor=color,
            borderwidth=1,
            borderpad=3,
        )

    # ── Persistent annotations (S₀ + exotic) ─────────────────────────
    persistent_anns: list[dict] = [
        _ann(
            spot,
            f" S\u2080={spot:.0f} ",
            "rgba(255,255,255,0.5)",
            y=0.97,
            xanchor="right",
        ),
    ]

    # Static exotic shapes + annotations
    if exotic_metadata:
        _exotic_ann_list: list[dict] = []
        for meta in exotic_metadata:
            inst = meta.get("instrument_class", "vanilla")
            if inst == "vanilla":
                continue

            if inst == "barrier":
                barrier = meta.get("barrier", 0.0)
                is_up = meta.get("is_up", True)
                is_ki = meta.get("is_knock_in", False)
                if barrier > 0:
                    ko_ki = "KI" if is_ki else "KO"
                    direction = "\u2191" if is_up else "\u2193"
                    fig.add_vline(
                        x=float(barrier),
                        line_dash="dash",
                        line_color=_BARRIER_LINE_PNL,
                        line_width=1.5,
                        row=1,
                        col=1,
                    )
                    _exotic_ann_list.append(
                        dict(
                            x=float(barrier),
                            text=f" {direction}{ko_ki} ${barrier:.0f} ",
                            color=_BARRIER_LINE_PNL,
                        )
                    )

            elif inst in ("digital", "asset_or_nothing"):
                strike = meta.get("strike", 0.0)
                is_call = meta.get("option_type") == "call"
                if strike > 0:
                    _all_s = all_spots[dte_days[-1]]
                    if is_call:
                        fig.add_vrect(
                            x0=strike,
                            x1=float(np.max(_all_s)) * 1.05,
                            fillcolor=_DIGITAL_ZONE_PNL,
                            line_width=0,
                            row=1,
                            col=1,
                        )
                    else:
                        fig.add_vrect(
                            x0=float(np.min(_all_s)) * 0.95,
                            x1=strike,
                            fillcolor=_DIGITAL_ZONE_PNL,
                            line_width=0,
                            row=1,
                            col=1,
                        )

            elif inst == "gap":
                trigger = meta.get("gap_trigger", 0.0)
                if trigger > 0:
                    fig.add_vline(
                        x=float(trigger),
                        line_dash="dashdot",
                        line_color=_GAP_TRIGGER_PNL,
                        line_width=1.5,
                        row=1,
                        col=1,
                    )
                    _exotic_ann_list.append(
                        dict(
                            x=float(trigger),
                            text=f" K\u2082=${trigger:.0f} ",
                            color=_GAP_TRIGGER_PNL,
                        )
                    )

            elif inst == "power":
                p_strike = meta.get("strike", 0.0)
                p_n = meta.get("power_n", 2.0)
                if p_strike > 0 and p_n > 0:
                    eff_be = p_strike ** (1.0 / p_n)
                    fig.add_vline(
                        x=float(eff_be),
                        line_dash="dashdot",
                        line_color=_POWER_BE_PNL,
                        line_width=1.5,
                        row=1,
                        col=1,
                    )
                    _exotic_ann_list.append(
                        dict(
                            x=float(eff_be),
                            text=f" S*=${eff_be:.1f} ",
                            color=_POWER_BE_PNL,
                        )
                    )

        # Exotic annotations: alternate top-right / bottom-right
        _exotic_ann_list.sort(key=lambda a: a["x"])
        for vi, ea in enumerate(_exotic_ann_list):
            y_pos = 0.97 if vi % 2 == 0 else 0.03
            persistent_anns.append(
                _ann(ea["x"], ea["text"], ea["color"], y=y_pos, xanchor="left")
            )

    # ── Per-DTE breakeven annotations (stacked at the bottom) ─────────
    dte_be_anns: dict[int, list[dict]] = {}
    for dte in dte_days:
        bes = all_breakevens[dte]
        anns_for_dte: list[dict] = []
        sorted_bes = sorted(bes)
        for b_idx, be_price in enumerate(sorted_bes):
            # Stack from the bottom up — never collides with S₀ at the top
            y_pos = 0.03 + b_idx * 0.06
            anns_for_dte.append(
                _ann(
                    float(be_price),
                    f" BE ${be_price:.1f} ",
                    "#a78bfa",
                    y=y_pos,
                    xanchor="left",
                )
            )
        dte_be_anns[dte] = anns_for_dte

    # ── Reference lines (shapes only, no text — text is in annotations)
    fig.add_hline(
        y=0, line_dash="dash", line_color=_ZERO_LINE, line_width=1, row=1, col=1
    )
    fig.add_vline(
        x=spot,
        line_dash="dot",
        line_color="rgba(255,255,255,0.25)",
        line_width=1,
        row=1,
        col=1,
    )

    # ── Set initial annotations (persistent + default DTE breakevens) ─
    default_dte = dte_days[-1]
    fig.layout.annotations = list(persistent_anns) + dte_be_anns[default_dte]

    # ── DTE slider (visibility toggling + annotation update) ──────────
    total_traces = traces_per_dte * n_dte
    steps = []
    for i, dte in enumerate(dte_days):
        visible = [False] * total_traces
        for k in range(traces_per_dte):
            visible[traces_per_dte * i + k] = True
        step_anns = list(persistent_anns) + dte_be_anns[dte]
        steps.append(
            dict(
                method="update",
                args=[
                    {"visible": visible},
                    {"annotations": step_anns},
                ],
                label=str(dte),
            )
        )

    slider = dict(
        active=n_dte - 1,
        currentvalue=dict(
            prefix="Days to Expiration: ",
            font=dict(size=13, color=_AXIS_LABEL),
        ),
        pad=dict(t=30),
        steps=steps,
        bgcolor="#1e293b",
        activebgcolor="#0d9488",
        bordercolor="rgba(255,255,255,0.15)",
        font=dict(color=_TICK_COLOR, size=10),
    )

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(
        height=550,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.4,
            font=dict(size=11, color=_LEGEND_COLOR),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=50, b=45, l=60, r=20),
        bargap=0.02,
        sliders=[slider],
    )

    fig.update_xaxes(
        title_text="Spot Price S(t)",
        row=1,
        col=1,
        tickprefix="$",
        tickformat=",.2f",
        hoverformat=",.2f",
        **_AXIS_STYLE,
    )
    fig.update_yaxes(
        title_text="Mark-to-Market P&L ($)",
        row=1,
        col=1,
        tickprefix="$",
        tickformat=",.2f",
        hoverformat=",.2f",
        **_AXIS_STYLE,
    )
    fig.update_xaxes(
        title_text="Count", row=1, col=2, showticklabels=False, **_AXIS_STYLE
    )
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

    # Asymmetric color range mapped to actual P&L min/max
    pnl_min = float(np.min(pnl))
    pnl_max = float(np.max(pnl))
    if pnl_min == pnl_max:
        pnl_min, pnl_max = pnl_min - 1.0, pnl_max + 1.0
    pnl_range = pnl_max - pnl_min

    # Position of zero on the [0, 1] normalized scale
    zero_pos = max(0.01, min(0.99, -pnl_min / pnl_range))

    # Diverging red → neutral → green, with midpoint at actual zero
    # Build colorscale stops proportionally around the zero position
    colorscale = []
    # Red side (loss): from pnl_min to 0
    if zero_pos > 0.02:
        colorscale.append([0.00, "#fca5a5"])  # large loss
        colorscale.append([zero_pos * 0.35, "#f87171"])
        colorscale.append([zero_pos * 0.65, "#ef4444"])
        colorscale.append([zero_pos * 0.85, "#f97066"])
    # Neutral zone around zero
    colorscale.append([max(0, zero_pos - 0.03), "#d4d4d8"])
    colorscale.append([zero_pos, "#e4e4e7"])  # midpoint = 0
    colorscale.append([min(1, zero_pos + 0.03), "#d4d4d8"])
    # Green side (profit): from 0 to pnl_max
    if zero_pos < 0.98:
        green_start = zero_pos
        green_range = 1.0 - green_start
        colorscale.append([green_start + green_range * 0.15, "#4ade80"])
        colorscale.append([green_start + green_range * 0.35, "#22c55e"])
        colorscale.append([green_start + green_range * 0.65, "#4ade80"])
        colorscale.append([1.00, "#86efac"])  # large profit

    # Deduplicate and sort (ensure strictly increasing positions)
    _seen = set()
    _clean = []
    for pos, col in colorscale:
        rounded = round(pos, 4)
        if rounded not in _seen:
            _seen.add(rounded)
            _clean.append([rounded, col])
    colorscale = sorted(_clean, key=lambda x: x[0])

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=rv,
                y=tp,
                z=pnl,
                mode="markers",
                marker=dict(
                    size=2.5,
                    color=pnl,
                    colorscale=colorscale,
                    cmin=pnl_min,
                    cmax=pnl_max,
                    opacity=0.90,
                    colorbar=dict(
                        title=dict(
                            text="P&L ($)", font=dict(color=_AXIS_LABEL, size=11)
                        ),
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
            )
        ]
    )

    _scene_axis = dict(
        backgroundcolor=_PLOT_BG,
        gridcolor="rgba(255,255,255,0.08)",
        showline=True,
        linecolor=_AXIS_LINE,
        tickfont=dict(size=9, color=_TICK_COLOR),
        title_font=dict(size=11, color=_AXIS_LABEL),
    )

    fig.update_layout(
        height=CHART_HEIGHT_LG,
        paper_bgcolor=_PAPER_BG,
        margin=dict(t=20, b=10, l=10, r=10),
        scene=dict(
            xaxis=dict(
                title="Realized Volatility (%)",
                ticksuffix="%",
                tickformat=".1f",
                **_scene_axis,
            ),
            yaxis=dict(
                title="Terminal Price S(T)",
                tickprefix="$",
                tickformat=",.2f",
                **_scene_axis,
            ),
            zaxis=dict(
                title="P&L ($)", tickprefix="$", tickformat=",.2f", **_scene_axis
            ),
            bgcolor=_PLOT_BG,
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
        ),
    )

    st.plotly_chart(fig, width="stretch")

    # Realized vol formula — adaptive to selected N and T
    ann_ratio = (
        f"{n_steps}"
        if time_horizon == 1.0
        else rf"\frac{{{n_steps}}}{{{time_horizon:.2g}}}"
    )
    st.latex(
        r"\sigma_{\mathrm{realized}}^{(i)} "
        r"= \sqrt{" + ann_ratio + r"}"
        r" \;\cdot\; "
        r"\mathrm{std}\!\left(\,r_1^{(i)},\; r_2^{(i)},\; \dots,\; r_{"
        + str(n_steps)
        + r"}^{(i)}\,\right)"
        r"\qquad\text{where}\quad "
        r"r_t^{(i)} = \ln\!\frac{S_{t}^{(i)}}{S_{t-1}^{(i)}}"
    )
