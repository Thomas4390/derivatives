"""
Pricing Comparison — Animated MC convergence to theoretical price (BS / FFT).

Automatically prices the strategy legs defined in the sidebar.
Play button triggers a step-by-step animation showing MC error
shrinking as the number of scenarios increases.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import time
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
LEG_FILLS = [
    "rgba(96,165,250,0.12)",
    "rgba(52,211,153,0.12)",
    "rgba(251,146,60,0.12)",
    "rgba(244,114,182,0.12)",
    "rgba(167,139,250,0.12)",
    "rgba(251,191,36,0.12)",
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
            "Method": leg.get("ref_method") or "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# Animated convergence
# ═══════════════════════════════════════════════════════════════════════════

def _leg_label(leg: dict) -> str:
    pos = "Long" if leg["direction"] == 1 else "Short"
    typ = "C" if leg["is_call"] else "P"
    return f"{pos} {typ} K={leg['strike']:.0f}"


def run_animated_convergence(
    model_key: str,
    params: dict,
    legs: list,
    T: float,
    spot: float,
    r: float,
    n_steps: int = 252,
    seed: int = 42,
) -> dict:
    """Animate MC convergence step by step, returning final results."""

    model_params = _extract_model_params(model_key, params)
    simulator = create_simulator(model_key, **model_params)
    has_ref = any(leg.get("ref_price") is not None for leg in legs)

    chart_ph = st.empty()
    progress = st.progress(0)

    acc = {
        i: {"prices": [], "errors": [], "se": [], "ci_lo": [], "ci_hi": []}
        for i in range(len(legs))
    }
    n_done = []

    for step, n_paths in enumerate(N_PATHS_GRID):
        terminal = simulator.simulate_terminal(
            s0=spot, mu=r, t=T,
            n_paths=n_paths, n_steps=n_steps, seed=seed,
        )
        n_done.append(n_paths)

        for i, leg in enumerate(legs):
            mc = price_from_terminals(terminal, leg["strike"], T, r, leg["is_call"])
            acc[i]["prices"].append(mc["price"])
            acc[i]["se"].append(mc["std_error"])
            acc[i]["ci_lo"].append(mc["confidence_interval"][0])
            acc[i]["ci_hi"].append(mc["confidence_interval"][1])
            ref = leg.get("ref_price")
            acc[i]["errors"].append(abs(mc["price"] - ref) if ref else mc["std_error"])

        fig = _build_figure(legs, n_done, acc, has_ref)
        chart_ph.plotly_chart(fig, use_container_width=True)

        progress.progress((step + 1) / len(N_PATHS_GRID))
        if step < len(N_PATHS_GRID) - 1:
            time.sleep(0.3)

    progress.empty()

    return {"n_done": n_done, "acc": acc, "legs": legs, "has_ref": has_ref}


def render_static_convergence(conv: dict) -> None:
    """Render the final convergence chart (no animation)."""
    fig = _build_figure(
        conv["legs"], conv["n_done"], conv["acc"], conv["has_ref"],
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Figure builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_figure(legs, n_list, acc, has_ref):
    """Two-row chart: MC price convergence (top) + error (bottom)."""

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.55, 0.45],
        subplot_titles=(
            "MC Price vs Reference",
            "Absolute Error" if has_ref else "Standard Error",
        ),
    )

    for i, leg in enumerate(legs):
        c = LEG_COLORS[i % len(LEG_COLORS)]
        fill = LEG_FILLS[i % len(LEG_FILLS)]
        label = _leg_label(leg)

        # ── Top: MC price with CI band ─────────────────────────────────
        fig.add_trace(go.Scatter(
            x=n_list, y=acc[i]["ci_hi"],
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=n_list, y=acc[i]["ci_lo"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=fill,
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=n_list, y=acc[i]["prices"],
            mode="lines+markers",
            line=dict(width=2, color=c),
            marker=dict(size=5, color=c),
            name=label,
            hovertemplate=(
                f"{label}<br>N=%{{x:,.0f}}<br>"
                f"MC=$%{{y:.4f}}<extra></extra>"
            ),
        ), row=1, col=1)

        # Reference line
        ref = leg.get("ref_price")
        if ref is not None:
            fig.add_hline(
                y=ref, line_dash="dash", line_color=c, line_width=1,
                row=1, col=1,
            )

        # ── Bottom: error ──────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=n_list, y=acc[i]["errors"],
            mode="lines+markers",
            line=dict(width=2, color=c),
            marker=dict(size=5, color=c),
            showlegend=False,
            hovertemplate=(
                f"{label}<br>N=%{{x:,.0f}}<br>"
                f"{'|Err|' if has_ref else 'SE'}=$%{{y:.5f}}<extra></extra>"
            ),
        ), row=2, col=1)

    # Theoretical O(1/√N) fitted from first leg
    if len(n_list) >= 2:
        c_fit = acc[0]["errors"][0] * np.sqrt(n_list[0])
        n_smooth = np.linspace(n_list[0], n_list[-1], 100)
        theory = c_fit / np.sqrt(n_smooth)
        fig.add_trace(go.Scatter(
            x=n_smooth.tolist(), y=theory.tolist(),
            mode="lines",
            line=dict(width=1.5, color=_THEORY_LINE, dash="dot"),
            name="O(1/\u221aN)",
        ), row=2, col=1)

    # ── Layout ─────────────────────────────────────────────────────────
    fig.update_layout(
        height=620,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.06,
            xanchor="center", x=0.5,
            font=dict(size=11, color=_LEGEND_COLOR),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=70, b=45, l=65, r=20),
    )

    for ann in fig.layout.annotations:
        ann.font.color = _AXIS_LABEL
        ann.font.size = 13

    fig.update_xaxes(type="log", row=1, col=1, **_AXIS_STYLE)
    fig.update_xaxes(
        type="log", title_text="Number of Scenarios (log scale)",
        row=2, col=1, **_AXIS_STYLE,
    )
    fig.update_yaxes(title_text="Option Price ($)", row=1, col=1, **_AXIS_STYLE)
    fig.update_yaxes(
        title_text="|Error| ($)" if has_ref else "Std Error ($)",
        row=2, col=1, **_AXIS_STYLE,
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Final summary table
# ═══════════════════════════════════════════════════════════════════════════

def render_final_table(conv: dict) -> None:
    """Summary table after convergence (last step = 50k paths)."""
    legs = conv["legs"]
    acc = conv["acc"]
    rows = []
    for i, leg in enumerate(legs):
        mc = acc[i]["prices"][-1]
        se = acc[i]["se"][-1]
        ref = leg.get("ref_price")
        err = abs(mc - ref) if ref else None
        rows.append({
            "Leg": _leg_label(leg),
            "MC Price (50k)": f"${mc:.4f}",
            "Std Error": f"${se:.4f}",
            "Reference": f"${ref:.4f}" if ref else "—",
            "|Error|": f"${err:.4f}" if err is not None else "—",
            "Error (%)": f"{err / ref * 100:.2f}%" if err is not None and ref else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
