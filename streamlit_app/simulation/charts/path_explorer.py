"""
Path Explorer Chart - Single-path price & volatility visualization.

Two-panel Plotly subplot:
  Top: Price path S(t) with S0 reference line, P&L coloring, breakeven lines,
       and exotic overlays (barrier, asian, lookback, digital, chooser, gap, power)
  Bottom: Volatility path sigma(t) - stochastic or flat depending on model
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

from config.chart_theme import (
    AXIS_LINE as _AXIS_LINE,
    AXIS_LABEL as _AXIS_LABEL,
    GRID as _GRID,
    LEGEND_COLOR as _LEGEND_COLOR,
    PAPER_BG as _PAPER_BG,
    PLOT_BG as _PLOT_BG,
    TICK_COLOR as _TICK_COLOR,
    badge as _badge,
)

from backend.simulation.base import SimulationResult
from charts.simulation_paths import (
    _add_exotic_overlays,
    _BE_LINE,
)


# ── Path-aware P&L for single path ────────────────────────────────────────


def _compute_path_aware_pnl(
    price_path: np.ndarray,
    position_arrays: dict,
    multiplier: float = 100.0,
) -> float:
    """Compute P&L for a single simulated path, using the full path for
    path-dependent exotics (barrier, asian, lookback).

    Non-path-dependent legs (vanilla, digital, power, gap,
    asset_or_nothing) use terminal price only.
    Chooser is path-dependent (optimal choice at t_c).
    """
    from backend.portfolio.pnl import calculate_portfolio_pnl_vectorized

    exotic_metadata = position_arrays.get("exotic_metadata", [])
    terminal_price = float(price_path[-1])
    pnl = 0.0

    # ── Vanilla legs via Numba ──────────────────────────────────────────
    vanilla_idx = [
        i for i, m in enumerate(exotic_metadata) if m["instrument_class"] == "vanilla"
    ]
    if vanilla_idx:
        v_idx = np.array(vanilla_idx)
        v_pnl = calculate_portfolio_pnl_vectorized(
            np.array([terminal_price]),
            position_arrays["strikes"][v_idx],
            position_arrays["option_types"][v_idx],
            position_arrays["position_types"][v_idx],
            position_arrays["quantities"][v_idx],
            position_arrays["premiums"][v_idx],
            multiplier,
        )
        pnl += float(v_pnl[0])

    # ── Exotic legs — path-aware ────────────────────────────────────────
    exotic_idx = [
        i for i, m in enumerate(exotic_metadata) if m["instrument_class"] != "vanilla"
    ]
    for j in exotic_idx:
        meta = exotic_metadata[j]
        premium = float(position_arrays["premiums"][j])
        direction = float(position_arrays["position_types"][j])
        qty = float(position_arrays["quantities"][j])
        payoff = _exotic_payoff_from_path(price_path, meta)
        pnl += direction * (payoff - premium) * qty * multiplier

    # ── Stock component ─────────────────────────────────────────────────
    stock_qty = position_arrays.get("stock_quantity", 0.0)
    if stock_qty != 0.0:
        stock_entry = position_arrays.get("stock_entry_price", 0.0)
        pnl += stock_qty * (terminal_price - stock_entry)

    return pnl


def _exotic_payoff_from_path(price_path: np.ndarray, meta: dict) -> float:
    """Compute per-share exotic payoff using the full simulated path.

    For path-dependent options (barrier, asian, lookback), this gives the
    correct payoff. For terminal-price-only options, it falls back to the
    standard calculation.
    """
    inst = meta.get("instrument_class", "vanilla")
    is_call = meta.get("option_type") == "call"
    strike = float(meta.get("strike", 0.0))
    spot_t = float(price_path[-1])

    # ── Barrier ─────────────────────────────────────────────────────────
    if inst == "barrier":
        barrier = float(meta.get("barrier", 0.0))
        is_up = meta.get("is_up", True)
        is_knock_in = meta.get("is_knock_in", False)
        vanilla_payoff = (
            max(spot_t - strike, 0.0) if is_call else max(strike - spot_t, 0.0)
        )

        # Check if barrier was hit at any point during the path
        if is_up:
            barrier_hit = bool(np.any(price_path >= barrier))
        else:
            barrier_hit = bool(np.any(price_path <= barrier))

        if is_knock_in:
            # KI: payoff only if barrier was activated
            return vanilla_payoff if barrier_hit else 0.0
        else:
            # KO: payoff zeroed if barrier was hit
            return 0.0 if barrier_hit else vanilla_payoff

    # ── Asian (geometric average) ───────────────────────────────────────
    if inst == "asian":
        # Geometric average of the full path
        log_avg = float(np.mean(np.log(price_path)))
        geo_avg = float(np.exp(log_avg))
        if is_call:
            return max(geo_avg - strike, 0.0)
        return max(strike - geo_avg, 0.0)

    # ── Lookback floating ───────────────────────────────────────────────
    if inst == "lookback_floating":
        if is_call:
            # Call: S_T - min(path)
            return max(spot_t - float(np.min(price_path)), 0.0)
        # Put: max(path) - S_T
        return max(float(np.max(price_path)) - spot_t, 0.0)

    # ── Lookback fixed ──────────────────────────────────────────────────
    if inst == "lookback_fixed":
        if is_call:
            # Call: max(max(path) - K, 0)
            return max(float(np.max(price_path)) - strike, 0.0)
        # Put: max(K - min(path), 0)
        return max(strike - float(np.min(price_path)), 0.0)

    # ── Chooser — optimal choice at t_c, payoff at T ─────────────────
    if inst == "chooser":
        choice_pct = float(meta.get("choice_time_pct", 0.5))
        n_steps = len(price_path) - 1
        step_tc = max(1, min(int(choice_pct * n_steps), n_steps - 1))
        s_tc = float(price_path[step_tc])
        r = float(meta.get("r", 0.0))
        maturity = float(meta.get("maturity", 1.0))
        t_c = choice_pct * maturity
        threshold = strike * np.exp(-(r) * (maturity - t_c))
        if s_tc > threshold:
            return max(spot_t - strike, 0.0)
        return max(strike - spot_t, 0.0)

    # ── Non-path-dependent exotics — delegate to standard function ──────
    from services.simulation_runner import _get_exotic_payoff_fn

    return _get_exotic_payoff_fn()(spot_t, meta)


_SPOT_LINE = "rgba(255,255,255,0.45)"

_PRICE_COLOR = "#58a6ff"  # bright blue (no strategy)
_PROFIT_COLOR = "#22c55e"  # green (profit)
_LOSS_COLOR = "#ef4444"  # red (loss)
_VOL_COLOR = "#f0883e"  # bright orange
_VOL_FLAT_COLOR = "rgba(255,160,50,0.7)"
_VOL_REF_INIT = "#78c8ff"  # bright cyan — initial vol
_VOL_REF_LR = "#a078ff"  # bright purple — long-run vol


def render_path_explorer_chart(
    result: SimulationResult,
    model_key: str,
    params: dict[str, Any],
    position_arrays: dict | None = None,
    exotic_metadata: list[dict] | None = None,
) -> float | None:
    """Render single-path price + volatility chart with P&L coloring and exotic overlays.

    Returns the single-path P&L value if a strategy is active, else None.
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

    # ── Compute P&L if strategy is active ──────────────────────────────────
    single_pnl = None
    path_color = _PRICE_COLOR

    if position_arrays is not None:
        single_pnl = _compute_path_aware_pnl(result.price_paths[0], position_arrays)
        is_profit = single_pnl >= 0
        path_color = _PROFIT_COLOR if is_profit else _LOSS_COLOR

    # ── Top panel: Price path ──────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=time_grid,
            y=result.price_paths[0],
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
        y=result.initial_price,
        line_dash="dash",
        line_color=_SPOT_LINE,
        line_width=1,
        annotation_text=f"S\u2080 = {result.initial_price:.2f}",
        annotation_font_size=11,
        annotation_font_color="rgba(255,255,255,0.55)",
        row=1,
        col=1,
    )

    # ── P&L annotation at terminal point ───────────────────────────────────
    if single_pnl is not None:
        pnl_label = f"P&L: ${single_pnl:+,.2f}"
        fig.add_annotation(
            x=time_grid[-1],
            y=float(result.price_paths[0, -1]),
            text=f" {pnl_label} ",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=path_color,
            font=dict(size=12, color=path_color),
            bgcolor="rgba(14, 17, 23, 0.85)",
            bordercolor=path_color,
            borderwidth=1,
            borderpad=3,
            xref="x",
            yref="y",
            row=1,
            col=1,
        )

    # ── Breakeven lines ────────────────────────────────────────────────────
    if position_arrays is not None:
        from services.simulation_runner import compute_hybrid_payoff_curve as _hpc
        from backend.portfolio.pnl import find_breakeven_points as _fbp

        s0 = float(params.get("spot_price", params.get("spot", 100.0)))
        spot_range = np.linspace(s0 * 0.5, s0 * 1.5, 500)
        payoff_curve = _hpc(spot_range, position_arrays)
        breakevens = _fbp(payoff_curve, spot_range)

        if breakevens is not None and len(breakevens) > 0:
            _be_positions = ["bottom right", "top right"]
            for i, be in enumerate(breakevens):
                fig.add_hline(
                    y=float(be),
                    line_dash="dot",
                    line_color=_BE_LINE,
                    line_width=1.5,
                    annotation_text=f" BE ${float(be):.2f} ",
                    annotation_position=_be_positions[i % 2],
                    **_badge(_BE_LINE),
                    row=1,
                    col=1,
                )

    # ── Exotic overlays ────────────────────────────────────────────────────
    if exotic_metadata:
        exotic_viz = {
            "show_overlays": True,
            "barrier_show_hits": True,
            "barrier_show_dead": True,
            "asian_n_avg": 1,
            "lookback_n_extreme": 1,
            "digital_show_zone": True,
        }
        _add_exotic_overlays(
            fig,
            result.price_paths,
            time_grid,
            np.array([0]),
            exotic_metadata,
            params,
            exotic_viz,
        )

    # ── Bottom panel: Volatility ───────────────────────────────────────────
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

        # Reference lines — initial & long-run volatility
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
    return single_pnl


def _add_vol_references(
    fig: go.Figure,
    model_key: str,
    params: dict[str, Any],
) -> None:
    """Add initial-vol and long-run-vol dashed reference lines to the vol panel."""
    model_lower = model_key.lower()

    # ── Initial volatility ──────────────────────────────────────────────
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

    # ── Long-run volatility ─────────────────────────────────────────────
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
