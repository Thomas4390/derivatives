"""
Exotic Options Charts and Tab Orchestrator for Options Greeks Explorer.

Provides Price comparison and Payoff diagram charts for the Exotic Options tab.

Author: Thomas Vaudescal
"""

import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from config.constants import (
    EXOTIC_TYPE_NAMES,
    BARRIER_SUBTYPES,
    EXOTIC_DESCRIPTIONS,
    DEFAULT_EXOTIC_DTE,
    DEFAULT_DIGITAL_PAYOUT,
    DEFAULT_BARRIER_UP_FACTOR,
    DEFAULT_BARRIER_DOWN_FACTOR,
    SPOT_RANGE_FACTOR,
    SPOT_RANGE_POINTS,
)
from config.chart_theme import (
    CHART_COLORS,
    AXIS_DEFAULTS,
    get_layout_config,
)
from services.exotic_pricing_adapter import (
    calculate_exotic_greeks_surface,
    calculate_vanilla_greeks_surface,
)


# =============================================================================
# CHART HELPER
# =============================================================================

def _base_layout(title: str, height: int = 500) -> dict:
    """Get a base layout config with standard theme."""
    layout = get_layout_config(title=title, height=height)
    layout["xaxis"]["title"] = {"text": "Underlying Price", "font": AXIS_DEFAULTS["title"]["font"]}
    return layout


def _add_spot_vline(fig: go.Figure, spot_price: float, row=None, col=None):
    """Add a vertical dashed line at current spot."""
    fig.add_vline(
        x=spot_price, line_dash="dot", line_color=CHART_COLORS["reference"],
        line_width=1, row=row, col=col,
    )


# =============================================================================
# PRICE COMPARISON CHART
# =============================================================================

def create_exotic_price_comparison(
    exotic_data: dict,
    vanilla_data: dict,
    spot_range: np.ndarray,
    spot_price: float,
    config: dict,
) -> go.Figure:
    """Exotic vs vanilla price with shaded fill between curves."""
    layout = _base_layout("Exotic vs Vanilla Price", height=480)
    layout["yaxis"]["title"] = {"text": "Option Price", "font": AXIS_DEFAULTS["title"]["font"]}
    layout["hovermode"] = "x unified"

    fig = go.Figure(layout=layout)

    # Exotic price line
    fig.add_trace(go.Scatter(
        x=spot_range, y=exotic_data["price"],
        name=config.get("exotic_label", "Exotic"),
        line=dict(color=CHART_COLORS["primary"], width=2.5),
        hovertemplate="<b>Price:</b> %{y:.2f}",
    ))

    # Vanilla price line (dashed)
    fig.add_trace(go.Scatter(
        x=spot_range, y=vanilla_data["price"],
        name="Vanilla",
        line=dict(color=CHART_COLORS["reference"], width=2, dash="dash"),
        hovertemplate="<b>Price:</b> %{y:.2f}",
    ))

    # Shaded fill between (use tonexty)
    fig.add_trace(go.Scatter(
        x=spot_range, y=exotic_data["price"],
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=spot_range, y=vanilla_data["price"],
        fill="tonexty",
        fillcolor="rgba(26, 54, 93, 0.08)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # Spot price vertical line
    _add_spot_vline(fig, spot_price)

    # Barrier level
    if config.get("barrier"):
        fig.add_vline(
            x=config["barrier"],
            line_dash="dash", line_color="#dc2626", line_width=2,
            annotation_text=f"Barrier = {config['barrier']:.1f}",
            annotation_position="top right",
            annotation_font_color="#dc2626",
        )

    # Digital payout reference line
    if config.get("payout") and config.get("discount_factor"):
        pv = config["payout"] * config["discount_factor"]
        fig.add_hline(
            y=pv, line_dash="dot", line_color="#0d9488", line_width=1.5,
            annotation_text=f"PV(Payout) = {pv:.2f}",
            annotation_position="top right",
            annotation_font_color="#0d9488",
        )

    # Price difference annotation at current spot
    idx = np.argmin(np.abs(spot_range - spot_price))
    exotic_at_spot = exotic_data["price"][idx]
    vanilla_at_spot = vanilla_data["price"][idx]
    diff = exotic_at_spot - vanilla_at_spot
    diff_pct = (diff / vanilla_at_spot * 100) if vanilla_at_spot != 0 else 0
    fig.add_annotation(
        x=spot_price, y=exotic_at_spot,
        text=f"Diff: {diff:+.2f} ({diff_pct:+.2f}%)",
        showarrow=True, arrowhead=2, arrowsize=1,
        font=dict(size=11, color=CHART_COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.9)", bordercolor=CHART_COLORS["primary"],
    )

    return fig


# =============================================================================
# PAYOFF DIAGRAM
# =============================================================================

def create_exotic_payoff_diagram(
    spot_range: np.ndarray,
    spot_price: float,
    config: dict,
) -> go.Figure:
    """Terminal payoff diagram for exotic option types."""
    exotic_type = config["exotic_type"]
    is_call = config["is_call"]
    strike = config["strike"]
    layout = _base_layout("Terminal Payoff at Expiration", height=450)
    layout["yaxis"]["title"] = {"text": "Payoff", "font": AXIS_DEFAULTS["title"]["font"]}

    fig = go.Figure(layout=layout)

    if exotic_type == "barrier":
        barrier = config["barrier"]
        is_up = config["is_up"]
        is_knock_in = config["is_knock_in"]

        # Vanilla payoff for reference
        vanilla_payoff = np.maximum(spot_range - strike, 0) if is_call else np.maximum(strike - spot_range, 0)
        fig.add_trace(go.Scatter(
            x=spot_range, y=vanilla_payoff,
            name="Vanilla Payoff", line=dict(color=CHART_COLORS["reference"], width=1.5, dash="dash"),
        ))

        # Barrier payoff: same as vanilla but zeroed out in knocked-out zone
        barrier_payoff = vanilla_payoff.copy()
        if is_knock_in:
            # Knock-in: only active if barrier was hit (at expiry, simplified)
            if is_up:
                barrier_payoff[spot_range < barrier] = 0
            else:
                barrier_payoff[spot_range > barrier] = 0
        else:
            # Knock-out: deactivated if barrier was hit
            if is_up:
                barrier_payoff[spot_range >= barrier] = 0
            else:
                barrier_payoff[spot_range <= barrier] = 0

        fig.add_trace(go.Scatter(
            x=spot_range, y=barrier_payoff,
            name="Barrier Payoff (at expiry)", line=dict(color=CHART_COLORS["primary"], width=2.5),
        ))

        fig.add_vline(x=barrier, line_dash="dash", line_color="#dc2626", line_width=2,
                       annotation_text=f"Barrier = {barrier:.1f}")

    elif exotic_type == "digital":
        payout = config.get("payout", 1.0)
        if is_call:
            payoff = np.where(spot_range > strike, payout, 0.0)
        else:
            payoff = np.where(spot_range < strike, payout, 0.0)

        fig.add_trace(go.Scatter(
            x=spot_range, y=payoff,
            name=f"Digital {'Call' if is_call else 'Put'} Payoff",
            line=dict(color=CHART_COLORS["primary"], width=2.5, shape="hv"),
        ))

        fig.add_hline(y=payout, line_dash="dot", line_color="#0d9488", line_width=1,
                       annotation_text=f"Payout = {payout:.2f}")

    elif exotic_type == "asian":
        # Show vanilla payoff for reference; Asian payoff depends on path
        vanilla_payoff = np.maximum(spot_range - strike, 0) if is_call else np.maximum(strike - spot_range, 0)
        fig.add_trace(go.Scatter(
            x=spot_range, y=vanilla_payoff,
            name="Vanilla Payoff (terminal)", line=dict(color=CHART_COLORS["reference"], width=1.5, dash="dash"),
        ))
        # Approximate: geometric average payoff shifted toward ATM
        # Show narrower payoff to illustrate averaging effect
        avg_shift = 0.5  # Geometric average is between spot and strike
        if is_call:
            avg_payoff = np.maximum(spot_range * avg_shift + spot_price * (1 - avg_shift) - strike, 0)
        else:
            avg_payoff = np.maximum(strike - (spot_range * avg_shift + spot_price * (1 - avg_shift)), 0)

        fig.add_trace(go.Scatter(
            x=spot_range, y=avg_payoff,
            name="Illustrative Average Payoff", line=dict(color=CHART_COLORS["primary"], width=2.5),
        ))

        fig.add_annotation(
            x=0.5, y=0.02, xref="paper", yref="paper",
            text="Note: Actual payoff depends on the path of prices (geometric average)",
            showarrow=False, font=dict(size=10, color="#64748b"),
        )

    elif exotic_type in ("lookback_floating", "lookback_fixed"):
        vanilla_payoff = np.maximum(spot_range - strike, 0) if is_call else np.maximum(strike - spot_range, 0)
        fig.add_trace(go.Scatter(
            x=spot_range, y=vanilla_payoff,
            name="Vanilla Payoff", line=dict(color=CHART_COLORS["reference"], width=1.5, dash="dash"),
        ))

        if exotic_type == "lookback_floating":
            # Floating: call payoff = S_T - S_min (always >= vanilla call payoff since S_min <= K possible)
            fig.add_annotation(
                x=0.5, y=0.02, xref="paper", yref="paper",
                text="Floating lookback payoff depends on running min/max (path-dependent)",
                showarrow=False, font=dict(size=10, color="#64748b"),
            )
            # Illustrative: payoff as if S_min = 0.9*S_T (call) or S_max = 1.1*S_T (put)
            if is_call:
                lb_payoff = np.maximum(spot_range - spot_range * 0.90, 0)
            else:
                lb_payoff = np.maximum(spot_range * 1.10 - spot_range, 0)
            fig.add_trace(go.Scatter(
                x=spot_range, y=lb_payoff,
                name="Illustrative Lookback Payoff", line=dict(color=CHART_COLORS["primary"], width=2.5),
            ))
        else:
            # Fixed: call payoff = max(M_max - K, 0)
            fig.add_annotation(
                x=0.5, y=0.02, xref="paper", yref="paper",
                text="Fixed lookback payoff: max(M_max - K, 0) for call, max(K - M_min, 0) for put",
                showarrow=False, font=dict(size=10, color="#64748b"),
            )
            # Illustrative: as if M_max = max(S_T, 1.1*S_0)
            if is_call:
                m_max = np.maximum(spot_range, spot_price * 1.1)
                lb_payoff = np.maximum(m_max - strike, 0)
            else:
                m_min = np.minimum(spot_range, spot_price * 0.9)
                lb_payoff = np.maximum(strike - m_min, 0)
            fig.add_trace(go.Scatter(
                x=spot_range, y=lb_payoff,
                name="Illustrative Lookback Payoff", line=dict(color=CHART_COLORS["primary"], width=2.5),
            ))

    fig.add_vline(x=strike, line_dash="dot", line_color="#64748b", line_width=1,
                   annotation_text=f"K = {strike:.1f}")
    _add_spot_vline(fig, spot_price)

    return fig


# =============================================================================
# MAIN TAB ORCHESTRATOR
# =============================================================================

def render_exotic_tab(spot_price: float, risk_free_rate: float):
    """Main orchestrator for the Exotic Options tab."""

    # ── Exotic type selector ──
    exotic_type_keys = list(EXOTIC_TYPE_NAMES.keys())
    exotic_type = st.selectbox(
        "Exotic Option Type",
        exotic_type_keys,
        format_func=lambda k: EXOTIC_TYPE_NAMES[k],
        key="exotic_type_selector",
    )

    # Educational description
    st.info(EXOTIC_DESCRIPTIONS[exotic_type])

    # ── Parameter controls ──
    col1, col2 = st.columns(2)

    with col1:
        strike = st.number_input("Strike Price", value=spot_price, min_value=1.0, step=1.0, key="exotic_strike")
        dte = st.number_input("Days to Expiry", value=DEFAULT_EXOTIC_DTE, min_value=1, max_value=730, step=1, key="exotic_dte")
        sigma_pct = st.slider("Volatility (%)", min_value=5, max_value=100, value=25, step=1, key="exotic_vol")
        sigma = sigma_pct / 100.0

    # Type-specific parameters
    is_call = True
    barrier = 0.0
    is_knock_in = False
    is_up = True
    rebate = 0.0
    payout = DEFAULT_DIGITAL_PAYOUT

    with col2:
        if exotic_type == "barrier":
            barrier_subtype_key = st.selectbox(
                "Barrier Type",
                list(BARRIER_SUBTYPES.keys()),
                format_func=lambda k: BARRIER_SUBTYPES[k]["label"],
                key="barrier_subtype",
            )
            subtype = BARRIER_SUBTYPES[barrier_subtype_key]
            is_call = subtype["is_call"]
            is_up = subtype["is_up"]
            is_knock_in = subtype["is_knock_in"]

            default_barrier = spot_price * (DEFAULT_BARRIER_UP_FACTOR if is_up else DEFAULT_BARRIER_DOWN_FACTOR)
            barrier = st.number_input("Barrier Level", value=default_barrier, min_value=1.0, step=1.0, key="barrier_level")
            rebate = st.number_input("Rebate (knock-out only)", value=0.0, min_value=0.0, step=0.5, key="barrier_rebate")

        elif exotic_type == "digital":
            option_side = st.radio("Option Type", ["Call", "Put"], horizontal=True, key="digital_side")
            is_call = option_side == "Call"
            payout = st.number_input("Payout Amount", value=DEFAULT_DIGITAL_PAYOUT, min_value=0.01, step=0.1, key="digital_payout")

        elif exotic_type == "asian":
            option_side = st.radio("Option Type", ["Call", "Put"], horizontal=True, key="asian_side")
            is_call = option_side == "Call"
            st.caption("Geometric average — closed-form pricing (Kemna-Vorst 1990)")

        elif exotic_type in ("lookback_floating", "lookback_fixed"):
            option_side = st.radio("Option Type", ["Call", "Put"], horizontal=True, key="lookback_side")
            is_call = option_side == "Call"
            if exotic_type == "lookback_floating":
                st.caption("Fresh option: M_min = M_max = Spot (Goldman-Sosin-Gatto 1979)")
            else:
                st.caption("Fresh option: M_min = M_max = Spot (Conze-Viswanathan 1991)")

    # ── Compute surfaces ──
    maturity = dte / 365.0
    spot_range = np.linspace(
        spot_price * (1 - SPOT_RANGE_FACTOR),
        spot_price * (1 + SPOT_RANGE_FACTOR),
        SPOT_RANGE_POINTS,
    )

    exotic_data = calculate_exotic_greeks_surface(
        exotic_type, spot_range, strike, maturity, risk_free_rate, sigma, is_call,
        barrier=barrier, is_knock_in=is_knock_in, is_up=is_up,
        rebate=rebate, payout=payout,
    )

    # For lookback floating, strike is not used in vanilla comparison — use spot as strike
    vanilla_strike = strike if exotic_type != "lookback_floating" else spot_price
    vanilla_data = calculate_vanilla_greeks_surface(
        spot_range, vanilla_strike, maturity, risk_free_rate, sigma, is_call,
    )

    # Build config dict
    chart_config = {
        "exotic_type": exotic_type,
        "exotic_label": EXOTIC_TYPE_NAMES[exotic_type],
        "is_call": is_call,
        "strike": strike,
        "spot_range": spot_range,
        "barrier": barrier if exotic_type == "barrier" else None,
        "is_up": is_up,
        "is_knock_in": is_knock_in,
        "payout": payout if exotic_type == "digital" else None,
        "discount_factor": math.exp(-risk_free_rate * maturity) if exotic_type == "digital" else None,
    }

    # ── Charts ──
    # Price comparison
    fig_price = create_exotic_price_comparison(exotic_data, vanilla_data, spot_range, spot_price, chart_config)
    st.plotly_chart(fig_price, width="stretch")

    # Payoff diagram
    fig_payoff = create_exotic_payoff_diagram(spot_range, spot_price, chart_config)
    st.plotly_chart(fig_payoff, width="stretch")
