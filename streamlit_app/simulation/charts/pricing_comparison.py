"""
Pricing Comparison — MC convergence to theoretical price (BS / FFT).

Shows how Monte Carlo pricing error shrinks as n_paths increases,
with reference price from closed-form (BS) or semi-analytical (FFT).
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, Optional, List, Tuple

from services.pricing_service import (
    price_with_analytical,
    price_with_fft,
    price_from_terminals,
    get_available_pricing_methods,
)
from services.simulation_service import _extract_model_params
from backend.simulation.factory import create_simulator

# ── Color palette (consistent with other charts) ──────────────────────────
_PAPER_BG = "#0e1117"
_PLOT_BG = "#161b22"
_GRID = "rgba(255,255,255,0.10)"
_AXIS_LINE = "rgba(255,255,255,0.25)"
_AXIS_LABEL = "#ffffff"
_TICK_COLOR = "rgba(255,255,255,0.70)"
_LEGEND_COLOR = "#ffffff"

_MC_COLOR = "#60a5fa"          # blue-400
_MC_FILL = "rgba(96,165,250,0.15)"
_REF_BS = "#f59e0b"            # amber-500
_REF_FFT = "#a78bfa"           # violet-400
_THEORETICAL = "rgba(255,255,255,0.25)"
_ERROR_POS = "rgba(239,68,68,0.70)"   # red
_ERROR_NEG = "rgba(34,197,94,0.70)"   # green

_AXIS_STYLE = dict(
    gridcolor=_GRID,
    zerolinecolor=_GRID,
    showline=True,
    linecolor=_AXIS_LINE,
    linewidth=1,
    tickfont=dict(size=10, color=_TICK_COLOR),
    title_font=dict(size=12, color=_AXIS_LABEL),
)


# ═══════════════════════════════════════════════════════════════════════════
# Convergence study
# ═══════════════════════════════════════════════════════════════════════════

N_PATHS_GRID = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000]


def run_convergence_study(
    model_key: str,
    params: Dict[str, Any],
    strike: float,
    time_to_maturity: float,
    spot: float,
    risk_free_rate: float,
    is_call: bool = True,
    n_paths_list: Optional[List[int]] = None,
    n_steps: int = 252,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run MC pricing at increasing n_paths and compare to reference price.

    Returns dict with:
        - n_paths_list, mc_prices, mc_std_errors, mc_ci_lower, mc_ci_upper
        - ref_price, ref_method (or None if GARCH)
    """
    if n_paths_list is None:
        n_paths_list = N_PATHS_GRID

    model_params = _extract_model_params(model_key, params)

    # Reference price
    ref_price = None
    ref_method = None

    analytical = price_with_analytical(
        model_key, params, strike, time_to_maturity, spot, risk_free_rate, is_call
    )
    if analytical is not None:
        ref_price = analytical["price"]
        ref_method = "Black-Scholes"

    if ref_price is None:
        fft = price_with_fft(
            model_key, params, strike, time_to_maturity, spot, risk_free_rate, is_call
        )
        if fft is not None:
            ref_price = fft
            ref_method = "FFT (Carr-Madan)"

    # MC convergence — risk-neutral (mu = r)
    mc_prices = []
    mc_std_errors = []
    mc_ci_lower = []
    mc_ci_upper = []

    simulator = create_simulator(model_key, **model_params)

    for n_paths in n_paths_list:
        terminal = simulator.simulate_terminal(
            s0=spot,
            mu=risk_free_rate,
            t=time_to_maturity,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
        )
        mc = price_from_terminals(
            terminal, strike, time_to_maturity, risk_free_rate, is_call
        )
        mc_prices.append(mc["price"])
        mc_std_errors.append(mc["std_error"])
        mc_ci_lower.append(mc["confidence_interval"][0])
        mc_ci_upper.append(mc["confidence_interval"][1])

    return {
        "n_paths_list": n_paths_list,
        "mc_prices": np.array(mc_prices),
        "mc_std_errors": np.array(mc_std_errors),
        "mc_ci_lower": np.array(mc_ci_lower),
        "mc_ci_upper": np.array(mc_ci_upper),
        "ref_price": ref_price,
        "ref_method": ref_method,
        "strike": strike,
        "is_call": is_call,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Charts
# ═══════════════════════════════════════════════════════════════════════════

def render_convergence_chart(study: Dict[str, Any]) -> None:
    """Two-row chart: MC price convergence (top) + absolute error (bottom)."""

    n_paths = study["n_paths_list"]
    mc_prices = study["mc_prices"]
    mc_ci_lower = study["mc_ci_lower"]
    mc_ci_upper = study["mc_ci_upper"]
    mc_errors = study["mc_std_errors"]
    ref_price = study["ref_price"]
    ref_method = study["ref_method"]
    has_ref = ref_price is not None

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.60, 0.40],
        subplot_titles=(
            "MC Price Convergence",
            "Absolute Error" if has_ref else "Standard Error",
        ),
    )

    # ── Top panel: MC price with 95% CI ────────────────────────────────────
    # CI band
    fig.add_trace(go.Scatter(
        x=n_paths, y=mc_ci_upper,
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=n_paths, y=mc_ci_lower,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=_MC_FILL,
        name="95% CI", legendgroup="ci",
    ), row=1, col=1)

    # MC price line
    fig.add_trace(go.Scatter(
        x=n_paths, y=mc_prices,
        mode="lines+markers",
        line=dict(width=2, color=_MC_COLOR),
        marker=dict(size=6, color=_MC_COLOR),
        name="MC Price",
        hovertemplate="N=%{x:,.0f}<br>Price=$%{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # Reference price line
    if has_ref:
        fig.add_hline(
            y=ref_price, line_dash="dash", line_color=_REF_BS if "Black" in ref_method else _REF_FFT,
            line_width=1.5,
            annotation_text=f"{ref_method}: ${ref_price:.4f}",
            annotation_font_size=11,
            annotation_font_color=_REF_BS if "Black" in ref_method else _REF_FFT,
            annotation_position="top left",
            row=1, col=1,
        )

    # ── Bottom panel: error or std error ───────────────────────────────────
    if has_ref:
        abs_errors = np.abs(mc_prices - ref_price)

        # Theoretical convergence: C / sqrt(N)
        # Fit C from the first MC run
        c_fit = abs_errors[0] * np.sqrt(n_paths[0])
        n_smooth = np.linspace(n_paths[0], n_paths[-1], 200)
        theoretical = c_fit / np.sqrt(n_smooth)

        fig.add_trace(go.Scatter(
            x=n_smooth.tolist(), y=theoretical.tolist(),
            mode="lines",
            line=dict(width=1.5, color=_THEORETICAL, dash="dot"),
            name="O(1/√N) theory",
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=n_paths, y=abs_errors.tolist(),
            mode="lines+markers",
            line=dict(width=2, color="#fb923c"),
            marker=dict(size=6, color="#fb923c"),
            name="|MC − Ref|",
            hovertemplate="N=%{x:,.0f}<br>Error=$%{y:.5f}<extra></extra>",
        ), row=2, col=1)
    else:
        # No reference — show std error
        fig.add_trace(go.Scatter(
            x=n_paths, y=mc_errors.tolist(),
            mode="lines+markers",
            line=dict(width=2, color="#fb923c"),
            marker=dict(size=6, color="#fb923c"),
            name="Std Error",
            hovertemplate="N=%{x:,.0f}<br>SE=$%{y:.5f}<extra></extra>",
        ), row=2, col=1)

        # Theoretical: SE ∝ 1/√N
        c_fit = mc_errors[0] * np.sqrt(n_paths[0])
        n_smooth = np.linspace(n_paths[0], n_paths[-1], 200)
        theoretical = c_fit / np.sqrt(n_smooth)

        fig.add_trace(go.Scatter(
            x=n_smooth.tolist(), y=theoretical.tolist(),
            mode="lines",
            line=dict(width=1.5, color=_THEORETICAL, dash="dot"),
            name="O(1/√N) theory",
        ), row=2, col=1)

    # ── Layout ─────────────────────────────────────────────────────────────
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

    # Subtitle font color
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
        title_text="Error ($)" if has_ref else "Std Error ($)",
        row=2, col=1, **_AXIS_STYLE,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_method_comparison(
    model_key: str,
    params: Dict[str, Any],
    terminal_prices: np.ndarray,
    strike: float,
    time_to_maturity: float,
    spot: float,
    risk_free_rate: float,
    is_call: bool = True,
) -> None:
    """Bar chart comparing prices across available methods."""

    methods = get_available_pricing_methods(model_key)
    names = []
    prices = []
    colors = []

    # MC
    mc = price_from_terminals(terminal_prices, strike, time_to_maturity, risk_free_rate, is_call)
    names.append(f"Monte Carlo ({mc['n_paths']:,} paths)")
    prices.append(mc["price"])
    colors.append(_MC_COLOR)

    # Analytical
    if "analytical" in methods:
        ana = price_with_analytical(model_key, params, strike, time_to_maturity, spot, risk_free_rate, is_call)
        if ana is not None:
            names.append("Black-Scholes (Analytic)")
            prices.append(ana["price"])
            colors.append(_REF_BS)

    # FFT
    if "fft" in methods:
        fft = price_with_fft(model_key, params, strike, time_to_maturity, spot, risk_free_rate, is_call)
        if fft is not None:
            names.append("FFT (Carr-Madan)")
            prices.append(fft)
            colors.append(_REF_FFT)

    if len(prices) < 2:
        st.caption("Only Monte Carlo pricing available for this model — no reference price for comparison.")
        return

    fig = go.Figure(go.Bar(
        x=names,
        y=prices,
        marker_color=colors,
        text=[f"${p:.4f}" for p in prices],
        textposition="outside",
        textfont=dict(size=12, color=_AXIS_LABEL),
        hovertemplate="%{x}<br>$%{y:.6f}<extra></extra>",
    ))

    fig.update_layout(
        height=350,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        margin=dict(t=30, b=20, l=60, r=20),
        yaxis=dict(title_text="Option Price ($)", **_AXIS_STYLE),
        xaxis=dict(tickfont=dict(size=11, color=_TICK_COLOR)),
        bargap=0.35,
    )

    st.plotly_chart(fig, use_container_width=True)
