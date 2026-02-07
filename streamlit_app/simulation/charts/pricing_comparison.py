"""
Pricing Comparison — Plotly-native animated MC convergence.

All convergence data is pre-computed, then rendered as a Plotly figure
with native Play/Pause buttons and a scenario slider.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Any, List

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

LEG_COLORS = ["#60a5fa", "#34d399", "#fb923c", "#f472b6", "#a78bfa", "#fbbf24"]

_AXIS_STYLE = dict(
    gridcolor=_GRID,
    zerolinecolor=_GRID,
    showline=True,
    linecolor=_AXIS_LINE,
    linewidth=1,
    tickfont=dict(size=10, color=_TICK_COLOR),
    title_font=dict(size=12, color=_AXIS_LABEL),
)

N_PATHS_GRID = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000]


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
            "Reference": f"${leg['ref_price']:.4f}" if leg.get("ref_price") else "MC only",
            "Method": leg.get("ref_method") or "\u2014",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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
    seed: int = 42,
) -> dict:
    """Run all MC simulations at once and return the full convergence data."""
    model_params = _extract_model_params(model_key, params)
    simulator = create_simulator(model_key, **model_params)
    has_ref = any(leg.get("ref_price") is not None for leg in legs)

    acc = {
        i: {"prices": [], "errors": [], "se": [], "ci_lo": [], "ci_hi": []}
        for i in range(len(legs))
    }

    for n_paths in N_PATHS_GRID:
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
        "n_done": list(N_PATHS_GRID),
        "acc": acc,
        "legs": legs,
        "has_ref": has_ref,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotly-native animated chart
# ═══════════════════════════════════════════════════════════════════════════

def render_animated_error_chart(conv: dict) -> None:
    """Plotly figure with frames, Play/Pause buttons, and a slider."""
    legs = conv["legs"]
    n_all = conv["n_done"]
    acc = conv["acc"]
    has_ref = conv["has_ref"]
    n_frames = len(n_all)
    n_legs = len(legs)

    # Theoretical O(1/√N) fitted from first leg
    c_fit = acc[0]["errors"][0] * np.sqrt(n_all[0])

    # ── Initial traces (first frame data) ──────────────────────────────
    fig = go.Figure()

    for i, leg in enumerate(legs):
        c = LEG_COLORS[i % len(LEG_COLORS)]
        fig.add_trace(go.Scatter(
            x=[n_all[0]], y=[acc[i]["errors"][0]],
            mode="lines+markers",
            line=dict(width=2.5, color=c),
            marker=dict(size=7, color=c, symbol="circle"),
            name=_leg_label(leg),
            hovertemplate=(
                f"{_leg_label(leg)}<br>"
                "N=%{x:,.0f}<br>"
                f"{'|Error|' if has_ref else 'SE'}=$%{{y:.5f}}"
                "<extra></extra>"
            ),
        ))

    # Theory line (initial: single point)
    fig.add_trace(go.Scatter(
        x=[n_all[0]],
        y=[c_fit / np.sqrt(n_all[0])],
        mode="lines",
        line=dict(width=1.5, color=_THEORY_LINE, dash="dot"),
        name="O(1/\u221AN)",
    ))

    # ── Frames ─────────────────────────────────────────────────────────
    frames = []
    for k in range(n_frames):
        n_sub = n_all[: k + 1]
        fdata = []

        for i in range(n_legs):
            c = LEG_COLORS[i % len(LEG_COLORS)]
            fdata.append(go.Scatter(
                x=n_sub,
                y=acc[i]["errors"][: k + 1],
                mode="lines+markers",
                line=dict(width=2.5, color=c),
                marker=dict(size=7, color=c, symbol="circle"),
            ))

        # Theory extends to current x-range
        if k >= 1:
            n_sm = np.linspace(n_all[0], n_all[k], 120)
            th = c_fit / np.sqrt(n_sm)
        else:
            n_sm = np.array([n_all[0]])
            th = np.array([c_fit / np.sqrt(n_all[0])])

        fdata.append(go.Scatter(
            x=n_sm.tolist(),
            y=th.tolist(),
            mode="lines",
            line=dict(width=1.5, color=_THEORY_LINE, dash="dot"),
        ))

        frames.append(go.Frame(data=fdata, name=f"{n_all[k]:,}"))

    fig.frames = frames

    # ── Animation controls ─────────────────────────────────────────────
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
                y=1.18,
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
                y=-0.08,
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

    # ── Layout ─────────────────────────────────────────────────────────
    fig.update_layout(
        height=500,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        hovermode="x unified",
        xaxis=dict(
            type="log",
            title="Number of Scenarios (log scale)",
            range=[np.log10(n_all[0]) - 0.15, np.log10(n_all[-1]) + 0.15],
            **_AXIS_STYLE,
        ),
        yaxis=dict(
            title="|Error| ($)" if has_ref else "Std Error ($)",
            rangemode="tozero",
            **_AXIS_STYLE,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color=_LEGEND_COLOR),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=80, b=95, l=65, r=20),
    )

    st.plotly_chart(fig, use_container_width=True)


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
            f"MC Price ({max_n:,})": f"${mc:.4f}",
            "Std Error": f"${se:.4f}",
            "Reference": f"${ref:.4f}" if ref else "\u2014",
            "|Error|": f"${err:.4f}" if err is not None else "\u2014",
            "Error (%)": f"{err / ref * 100:.2f}%" if err is not None and ref else "\u2014",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
