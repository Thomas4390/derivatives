"""
Pricing Comparison — Plotly-native animated MC convergence.

Two-panel chart: MC price convergence (top) + error convergence (bottom),
both animated with Play/Pause and scenario slider.
All data is pre-computed, then rendered via Plotly frames.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import List

from services.pricing_service import (
    price_with_analytical,
    price_with_fft,
    price_from_terminals,
    get_available_pricing_methods,
)
from services.simulation_service import _extract_model_params
from backend.simulation.factory import create_simulator

# ── Color palette ──────────────────────────────────────────────────────────
_PAPER_BG = "#0e1117"
_PLOT_BG = "#161b22"
_GRID = "rgba(255,255,255,0.10)"
_AXIS_LINE = "rgba(255,255,255,0.25)"
_AXIS_LABEL = "#ffffff"
_TICK_COLOR = "rgba(255,255,255,0.70)"
_LEGEND_COLOR = "#ffffff"
_THEORY_LINE = "rgba(255,255,255,0.20)"
_REF_LINE = "rgba(255, 215, 0, 0.50)"

LEG_COLORS = ["#60a5fa", "#34d399", "#fb923c", "#f472b6", "#a78bfa", "#fbbf24"]
LEG_FILLS = [
    "rgba(96,165,250,0.12)", "rgba(52,211,153,0.12)", "rgba(251,146,60,0.12)",
    "rgba(244,114,182,0.12)", "rgba(167,139,250,0.12)", "rgba(251,191,36,0.12)",
]

_AXIS_STYLE = dict(
    gridcolor=_GRID,
    zerolinecolor=_GRID,
    showline=True,
    linecolor=_AXIS_LINE,
    linewidth=1,
    tickfont=dict(size=10, color=_TICK_COLOR),
    title_font=dict(size=12, color=_AXIS_LABEL),
)


def _build_n_paths_grid(max_n: int) -> list:
    """Build a logarithmically-spaced grid ending at max_n."""
    base = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000]
    grid = [n for n in base if n < max_n]
    grid.append(max_n)
    # Ensure at least 4 points
    if len(grid) < 4:
        grid = sorted(set([max(50, max_n // 8), max(100, max_n // 4), max(200, max_n // 2), max_n]))
    return grid


# ═══════════════════════════════════════════════════════════════════════════
# Leg helpers
# ═══════════════════════════════════════════════════════════════════════════

def extract_legs(position_arrays: dict, default_spot: float) -> List[dict]:
    """Extract option legs from position_arrays. Falls back to ATM call."""
    strikes = position_arrays.get("strikes", [])
    if len(strikes) == 0:
        return [{
            "strike": round(default_spot),
            "is_call": True,
            "direction": 1,
            "quantity": 1.0,
            "premium": 0.0,
        }]
    return [
        {
            "strike": float(strikes[i]),
            "is_call": int(position_arrays["option_types"][i]) == 1,
            "direction": int(position_arrays["position_types"][i]),
            "quantity": float(position_arrays["quantities"][i]),
            "premium": float(position_arrays["premiums"][i]),
        }
        for i in range(len(strikes))
    ]


def compute_reference_prices(
    model_key: str, params: dict, legs: list,
    T: float, spot: float, r: float,
) -> None:
    """Compute and attach reference price to each leg (in-place)."""
    for leg in legs:
        leg["ref_price"] = None
        leg["ref_method"] = None

        ana = price_with_analytical(
            model_key, params, leg["strike"], T, spot, r, leg["is_call"]
        )
        if ana is not None:
            leg["ref_price"] = ana["price"]
            leg["ref_method"] = "Black-Scholes"
            continue

        fft = price_with_fft(
            model_key, params, leg["strike"], T, spot, r, leg["is_call"]
        )
        if fft is not None:
            leg["ref_price"] = fft
            leg["ref_method"] = "FFT (Carr-Madan)"


def _leg_label(leg: dict) -> str:
    pos = "Long" if leg["direction"] == 1 else "Short"
    typ = "C" if leg["is_call"] else "P"
    return f"{pos} {typ} K={leg['strike']:.0f}"


def render_legs_summary(legs: list) -> None:
    """Read-only table showing the legs being priced."""
    rows = []
    for i, leg in enumerate(legs):
        rows.append({
            "#": i + 1,
            "Type": "Call" if leg["is_call"] else "Put",
            "Strike": f"${leg['strike']:.1f}",
            "Position": "Long" if leg["direction"] == 1 else "Short",
            "Qty": int(leg["quantity"]),
            "Reference": f"${leg['ref_price']:.2f}" if leg.get("ref_price") else "MC only",
            "Method": leg.get("ref_method") or "\u2014",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# Pre-computation
# ═══════════════════════════════════════════════════════════════════════════

def precompute_convergence(
    model_key: str,
    params: dict,
    legs: list,
    T: float,
    spot: float,
    r: float,
    n_steps: int = 252,
    max_n: int = 50_000,
    seed: int = 42,
) -> dict:
    """Run all MC simulations at once and return the full convergence data."""
    n_paths_grid = _build_n_paths_grid(max_n)

    model_params = _extract_model_params(model_key, params)
    simulator = create_simulator(model_key, **model_params)
    has_ref = any(leg.get("ref_price") is not None for leg in legs)

    acc = {
        i: {"prices": [], "errors": [], "se": [], "ci_lo": [], "ci_hi": []}
        for i in range(len(legs))
    }

    for n_paths in n_paths_grid:
        terminal = simulator.simulate_terminal(
            s0=spot, mu=r, t=T,
            n_paths=n_paths, n_steps=n_steps, seed=seed,
        )
        for i, leg in enumerate(legs):
            mc = price_from_terminals(terminal, leg["strike"], T, r, leg["is_call"])
            acc[i]["prices"].append(mc["price"])
            acc[i]["se"].append(mc["std_error"])
            acc[i]["ci_lo"].append(mc["confidence_interval"][0])
            acc[i]["ci_hi"].append(mc["confidence_interval"][1])
            ref = leg.get("ref_price")
            acc[i]["errors"].append(abs(mc["price"] - ref) if ref else mc["std_error"])

    return {
        "n_done": n_paths_grid,
        "acc": acc,
        "legs": legs,
        "has_ref": has_ref,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotly-native animated two-panel chart
# ═══════════════════════════════════════════════════════════════════════════

def render_animated_convergence_chart(conv: dict) -> None:
    """Two-panel Plotly animation: MC price (top) + error (bottom).

    Trace order (must match between initial traces and frame data):
      Row 1 — price panel:
        [0 .. n_legs-1]              price lines
        [n_legs + 2*i]               CI upper for leg i
        [n_legs + 2*i + 1]           CI lower for leg i  (fill="tonexty")
      Row 2 — error panel:
        [3*n_legs .. 4*n_legs-1]     error lines
        [4*n_legs]                   O(1/√N) theory line
      Total: 4*n_legs + 1
    """
    legs = conv["legs"]
    n_all = conv["n_done"]
    acc = conv["acc"]
    has_ref = conv["has_ref"]
    n_frames = len(n_all)
    n_legs = len(legs)

    # ── Compute fixed y-axis ranges from all data ─────────────────────
    all_ci_lo, all_ci_hi, all_prices = [], [], []
    for i in range(n_legs):
        all_prices.extend(acc[i]["prices"])
        all_ci_lo.extend(acc[i]["ci_lo"])
        all_ci_hi.extend(acc[i]["ci_hi"])
        ref = legs[i].get("ref_price")
        if ref is not None:
            all_prices.append(ref)

    price_min = min(all_ci_lo) if all_ci_lo else min(all_prices)
    price_max = max(all_ci_hi) if all_ci_hi else max(all_prices)
    price_pad = (price_max - price_min) * 0.12 if price_max > price_min else 0.5
    price_range = [price_min - price_pad, price_max + price_pad]

    all_errors = []
    for i in range(n_legs):
        all_errors.extend(acc[i]["errors"])
    err_max = max(all_errors) if all_errors else 1.0
    err_range = [0, err_max * 1.15]

    c_fit = acc[0]["errors"][0] * np.sqrt(n_all[0])
    x_range = [np.log10(n_all[0]) - 0.15, np.log10(n_all[-1]) + 0.15]

    # ── Create subplots ───────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.10,
        subplot_titles=["MC Price Convergence", "Pricing Error Convergence"],
    )

    # ── Row 1: Price lines [0 .. n_legs-1] ────────────────────────────
    for i, leg in enumerate(legs):
        c = LEG_COLORS[i % len(LEG_COLORS)]
        fig.add_trace(go.Scatter(
            x=[n_all[0]], y=[acc[i]["prices"][0]],
            mode="lines+markers",
            line=dict(width=2.5, color=c),
            marker=dict(size=6, color=c, symbol="circle"),
            name=_leg_label(leg),
            legendgroup=f"leg{i}",
            hovertemplate=(
                f"{_leg_label(leg)}<br>"
                "N=%{x:,.0f}<br>"
                "MC=$%{y:.2f}<extra></extra>"
            ),
        ), row=1, col=1)

    # Row 1: CI bands — interleaved upper/lower per leg [n_legs .. 3*n_legs-1]
    for i in range(n_legs):
        fill_c = LEG_FILLS[i % len(LEG_FILLS)]
        # CI upper (invisible boundary)
        fig.add_trace(go.Scatter(
            x=[n_all[0]], y=[acc[i]["ci_hi"][0]],
            mode="lines", line=dict(width=0),
            showlegend=False, legendgroup=f"leg{i}",
            hoverinfo="skip",
        ), row=1, col=1)
        # CI lower (fill to the CI upper immediately above)
        fig.add_trace(go.Scatter(
            x=[n_all[0]], y=[acc[i]["ci_lo"][0]],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=fill_c,
            showlegend=False, legendgroup=f"leg{i}",
            hoverinfo="skip",
        ), row=1, col=1)

    # ── Row 2: Error lines [3*n_legs .. 4*n_legs-1] ──────────────────
    for i, leg in enumerate(legs):
        c = LEG_COLORS[i % len(LEG_COLORS)]
        fig.add_trace(go.Scatter(
            x=[n_all[0]], y=[acc[i]["errors"][0]],
            mode="lines+markers",
            line=dict(width=2.5, color=c),
            marker=dict(size=6, color=c, symbol="circle"),
            legendgroup=f"leg{i}",
            showlegend=False,
            hovertemplate=(
                f"{_leg_label(leg)}<br>"
                "N=%{x:,.0f}<br>"
                f"{'|Error|' if has_ref else 'SE'}=$%{{y:.4f}}"
                "<extra></extra>"
            ),
        ), row=2, col=1)

    # Row 2: Theory line [4*n_legs]
    fig.add_trace(go.Scatter(
        x=[n_all[0]],
        y=[c_fit / np.sqrt(n_all[0])],
        mode="lines",
        line=dict(width=1.5, color=_THEORY_LINE, dash="dot"),
        name="O(1/\u221AN)",
        showlegend=True,
    ), row=2, col=1)

    # ── Reference price hlines (layout shapes — persist across frames)
    for i, leg in enumerate(legs):
        ref = leg.get("ref_price")
        if ref is not None:
            c = LEG_COLORS[i % len(LEG_COLORS)]
            fig.add_hline(
                y=ref, row=1, col=1,
                line_dash="dash", line_color=c, line_width=1.2,
                opacity=0.6,
                annotation_text=f"Ref {_leg_label(leg)}: ${ref:.4f}",
                annotation_font_size=9,
                annotation_font_color=c,
                annotation_position="top left" if i % 2 == 0 else "bottom left",
            )

    # ── Build frames ──────────────────────────────────────────────────
    frames = []
    for k in range(n_frames):
        n_sub = n_all[: k + 1]
        fdata = []

        # Row 1: price lines
        for i in range(n_legs):
            c = LEG_COLORS[i % len(LEG_COLORS)]
            fdata.append(go.Scatter(
                x=n_sub, y=acc[i]["prices"][: k + 1],
                mode="lines+markers",
                line=dict(width=2.5, color=c),
                marker=dict(size=6, color=c, symbol="circle"),
                xaxis="x", yaxis="y",
            ))

        # Row 1: CI bands (interleaved upper/lower per leg)
        for i in range(n_legs):
            fill_c = LEG_FILLS[i % len(LEG_FILLS)]
            # CI upper
            fdata.append(go.Scatter(
                x=n_sub, y=acc[i]["ci_hi"][: k + 1],
                mode="lines", line=dict(width=0),
                xaxis="x", yaxis="y",
            ))
            # CI lower (fill to upper)
            fdata.append(go.Scatter(
                x=n_sub, y=acc[i]["ci_lo"][: k + 1],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=fill_c,
                xaxis="x", yaxis="y",
            ))

        # Row 2: error lines
        for i in range(n_legs):
            c = LEG_COLORS[i % len(LEG_COLORS)]
            fdata.append(go.Scatter(
                x=n_sub, y=acc[i]["errors"][: k + 1],
                mode="lines+markers",
                line=dict(width=2.5, color=c),
                marker=dict(size=6, color=c, symbol="circle"),
                xaxis="x2", yaxis="y2",
            ))

        # Row 2: theory line
        if k >= 1:
            n_sm = np.linspace(n_all[0], n_all[k], 120)
            th = c_fit / np.sqrt(n_sm)
        else:
            n_sm = np.array([n_all[0]])
            th = np.array([c_fit / np.sqrt(n_all[0])])

        fdata.append(go.Scatter(
            x=n_sm.tolist(), y=th.tolist(),
            mode="lines",
            line=dict(width=1.5, color=_THEORY_LINE, dash="dot"),
            xaxis="x2", yaxis="y2",
        ))

        frames.append(go.Frame(data=fdata, name=f"{n_all[k]:,}"))

    fig.frames = frames

    # ── Animation controls ────────────────────────────────────────────
    slider_steps = [
        dict(
            args=[
                [f"{n:,}"],
                {
                    "frame": {"duration": 400, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 250},
                },
            ],
            label=f"{n:,}",
            method="animate",
        )
        for n in n_all
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.08,
                x=0.0,
                xanchor="left",
                buttons=[
                    dict(
                        label="\u25B6  Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                                "transition": {
                                    "duration": 300,
                                    "easing": "cubic-in-out",
                                },
                            },
                        ],
                    ),
                    dict(
                        label="\u23F8  Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
                font=dict(color=_AXIS_LABEL, size=12),
                bgcolor="rgba(255,255,255,0.08)",
                bordercolor="rgba(255,255,255,0.25)",
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=slider_steps,
                x=0.0,
                len=1.0,
                y=-0.05,
                currentvalue=dict(
                    prefix="N = ",
                    visible=True,
                    xanchor="center",
                    font=dict(size=13, color=_AXIS_LABEL),
                ),
                font=dict(size=9, color=_TICK_COLOR),
                tickcolor="rgba(255,255,255,0.3)",
                bordercolor="rgba(255,255,255,0.15)",
                bgcolor="rgba(255,255,255,0.05)",
            )
        ],
    )

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(
        height=750,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color=_LEGEND_COLOR),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=75, b=80, l=65, r=20),
    )

    # Subplot title styling
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color=_AXIS_LABEL)

    # Row 1 axes (price)
    fig.update_xaxes(
        type="log", title="",
        range=x_range,
        **_AXIS_STYLE,
        row=1, col=1,
    )
    fig.update_yaxes(
        title="MC Price ($)",
        range=price_range,
        **_AXIS_STYLE,
        row=1, col=1,
    )

    # Row 2 axes (error)
    fig.update_xaxes(
        type="log",
        title="Number of Scenarios (log scale)",
        range=x_range,
        **_AXIS_STYLE,
        row=2, col=1,
    )
    fig.update_yaxes(
        title="|Error| ($)" if has_ref else "Std Error ($)",
        range=err_range,
        **_AXIS_STYLE,
        row=2, col=1,
    )

    st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════
# Final summary table
# ═══════════════════════════════════════════════════════════════════════════

def render_final_table(conv: dict) -> None:
    """Summary table with per-leg MC price vs reference at max N."""
    legs = conv["legs"]
    acc = conv["acc"]
    max_n = conv["n_done"][-1]
    rows = []
    for i, leg in enumerate(legs):
        mc = acc[i]["prices"][-1]
        se = acc[i]["se"][-1]
        ref = leg.get("ref_price")
        err = abs(mc - ref) if ref else None
        rows.append({
            "Leg": _leg_label(leg),
            f"MC Price ({max_n:,})": f"${mc:.2f}",
            "Std Error": f"${se:.4f}",
            "Reference": f"${ref:.2f}" if ref else "\u2014",
            "|Error|": f"${err:.4f}" if err is not None else "\u2014",
            "Error (%)": f"{err / ref * 100:.2f}%" if err is not None and ref else "\u2014",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
