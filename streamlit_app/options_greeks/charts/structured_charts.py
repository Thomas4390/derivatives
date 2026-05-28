"""
Structured Products — Summary Header Display.

Renders pricing results, decomposition, and probability metrics
as a compact header above the shared tabs.
"""

from __future__ import annotations

import streamlit as st

import plotly.graph_objects as go

from config.chart_theme import CHART_COLORS, get_layout_config
from config.constants import (
    STRUCTURED_PRODUCT_DESCRIPTIONS,
)
from config.templates import metric_card_html


# =============================================================================
# Entry Point
# =============================================================================


def render_structured_summary_header() -> None:
    """Render structured product summary metrics above the shared tabs."""

    product_type = st.session_state.get("sp_product_type")
    result = st.session_state.get("sp_result")

    if result is None:
        return

    _render_pricing_metrics(result)
    _render_decomposition_chart(result)
    _render_probability_metrics(result, product_type)

    # Educational description
    with st.expander("About this product"):
        st.markdown(STRUCTURED_PRODUCT_DESCRIPTIONS.get(product_type, ""))  # pyright: ignore[reportCallIssue]

    st.markdown("---")

    # DTE variation disclaimer
    st.caption(
        "DTE/IV variation is approximated by re-pricing with adjusted maturity/volatility. "
        "2nd and 3rd order Greeks are set to zero (MC limitation)."
    )


# =============================================================================
# Results Display
# =============================================================================


def _render_pricing_metrics(result: dict) -> None:
    """Render the 4 main pricing metric cards."""
    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            metric_card_html(
                "Fair Value", f"{result['fair_value']:.2f}%", subtext="of notional"
            ),
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            metric_card_html(
                "Price",
                f"${result['price']:,.2f}",
                subtext=f"Notional: ${result['notional']:,.0f}",
            ),
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            metric_card_html(
                "MC Error",
                f"\u00b1${result['error']:,.2f}",
                subtext=f"{int(st.session_state.get('sp_config', {}).get('n_paths', 0)):,} paths",
            ),
            unsafe_allow_html=True,
        )
    with cols[3]:
        ret = result["expected_return"]
        cls = "profit" if ret >= 0 else "loss"
        st.markdown(
            metric_card_html(
                "Expected Return", f"{ret:.2%}", value_class=cls, subtext="total return"
            ),
            unsafe_allow_html=True,
        )


def _render_decomposition_chart(result: dict) -> None:
    """Render horizontal stacked bar chart of product decomposition."""
    bond = result["bond_floor"]
    option = result["option_value"]
    coupon = result["expected_coupon"]

    if bond == 0 and option == 0 and coupon == 0:
        return

    fig = go.Figure()

    components = []
    values = []
    colors = []

    if bond != 0:
        components.append("Bond Floor")
        values.append(bond)
        colors.append(CHART_COLORS["primary"])
    if option != 0:
        components.append("Option Value")
        values.append(option)
        colors.append(CHART_COLORS["accent"])
    if coupon != 0:
        components.append("Expected Coupon")
        values.append(coupon)
        colors.append("#c9a227")  # Gold

    for comp, val, color in zip(components, values, colors):
        fig.add_trace(
            go.Bar(
                y=["Product"],
                x=[val],
                name=f"{comp} ({val:.1f}%)",
                orientation="h",
                marker_color=color,
                text=f"{val:.1f}%",
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=f"{comp}: %{{x:.2f}}% of notional<extra></extra>",
            )
        )

    layout = get_layout_config(title="Value Decomposition (% of Notional)", height=180)
    layout["barmode"] = "stack"
    layout["showlegend"] = True
    layout["legend"] = dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    )
    layout["margin"] = {"l": 20, "r": 20, "t": 50, "b": 20}
    layout["yaxis"]["visible"] = False
    fig.update_layout(**layout)

    st.plotly_chart(fig, width="stretch")


def _render_probability_metrics(result: dict, product_type: str) -> None:
    """Render probability and scenario metric cards."""
    cols = st.columns(4)

    with cols[0]:
        if product_type == "autocallable":
            st.markdown(
                metric_card_html(
                    "Autocall Prob.", f"{result['autocall_probability']:.1%}"
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                metric_card_html(
                    "Autocall Prob.", "N/A", value_class="", subtext="Not applicable"
                ),
                unsafe_allow_html=True,
            )

    with cols[1]:
        loss_prob = result["capital_loss_probability"]
        cls = "loss" if loss_prob > 0 else ""
        st.markdown(
            metric_card_html("Capital Loss Prob.", f"{loss_prob:.1%}", value_class=cls),
            unsafe_allow_html=True,
        )

    with cols[2]:
        worst = result["worst_case_return"]
        st.markdown(
            metric_card_html("Worst Case (5th)", f"{worst:.2%}", value_class="loss"),
            unsafe_allow_html=True,
        )

    with cols[3]:
        best = result["best_case_return"]
        st.markdown(
            metric_card_html("Best Case (95th)", f"{best:.2%}", value_class="profit"),
            unsafe_allow_html=True,
        )
