"""Terminal payoff diagram for the Exotic tab.

One dispatch instead of a per-family if/elif chain: terminal families draw the
exact vectorized adapter payoff (true breaks at discontinuities), discrete-event
families overlay BOTH conditional outcomes — the same semantics and colors as
the Portfolio P&L overlay — and the illustrative path-dependent families
(asian / lookback, reachable through imported setups only) keep their sketch.
No family renders an empty chart.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from charts._exotic_annotations import add_barrier_markers
from config.chart_theme import (
    AXIS_DEFAULTS,
    CHART_COLORS,
    SCENARIO_COLORS,
    get_layout_config,
)
from config.exotic_config import EXOTIC_TYPE_NAMES, PAYOFF_SCENARIOS
from services.exotic_pricing_adapter import (
    calculate_exotic_payoff_at_expiry_vec,
    conditional_exotic_payoff_vec,
    payoff_curve_with_gaps,
)
from services.scenario_labels import scenario_options

_ANNOTATION_COLOR = "#0d9488"
_NOTE_FONT = {"size": 10, "color": "#64748b"}

# Vanilla-derived payoffs where the dashed vanilla reference gives the reader a
# baseline to compare against.
_VANILLA_REFERENCE_FAMILIES = frozenset(
    {
        "barrier",
        "double_barrier",
        "discrete_barrier",
        "partial_barrier",
        "power",
        "gap",
        "powered",
        "capped_power",
        "log_option",
        "arithmetic_asian",
    }
)


def _family_label(inst_class: str) -> str:
    name = EXOTIC_TYPE_NAMES.get(inst_class, inst_class)
    return name.split("(")[0].strip()


def _add_vanilla_reference(
    fig: go.Figure, spot_range: np.ndarray, strike: float, is_call: bool
) -> None:
    vanilla = (
        np.maximum(spot_range - strike, 0.0)
        if is_call
        else np.maximum(strike - spot_range, 0.0)
    )
    fig.add_trace(
        go.Scatter(
            x=spot_range,
            y=vanilla,
            name="Vanilla Payoff",
            line=dict(color=CHART_COLORS["reference"], width=1.5, dash="dash"),
        )
    )


def _add_terminal_payoff(
    fig: go.Figure, spot_range: np.ndarray, position: dict
) -> None:
    """Single exact terminal curve, broken (double point + NaN) at each jump."""
    x, y = payoff_curve_with_gaps(
        spot_range,
        position,
        lambda s: calculate_exotic_payoff_at_expiry_vec(s, position),
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name=f"{_family_label(position['instrument_class'])} Payoff",
            line=dict(color=CHART_COLORS["primary"], width=2.5),
            connectgaps=False,
        )
    )


def _add_outcome_overlay(
    fig: go.Figure, spot_range: np.ndarray, position: dict
) -> None:
    """Overlay the two conditional outcomes of a discrete-event leg.

    Each curve is NaN-masked to its feasible region (an up-and-out that was
    never touched cannot end above the barrier), so with ``connectgaps=False``
    the two curves tile the spot axis — same reading as the P&L overlay.
    """

    def _conditional(scen_key: str):
        def payoff_fn(s: np.ndarray) -> np.ndarray:
            payoff, feasible = conditional_exotic_payoff_vec(s, position, scen_key)
            return np.where(feasible, payoff, np.nan)

        return payoff_fn

    for idx, (scen_key, label) in enumerate(scenario_options(position)):
        x, y = payoff_curve_with_gaps(spot_range, position, _conditional(scen_key))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=label,
                line=dict(color=SCENARIO_COLORS[idx], width=2.5),
                connectgaps=False,
            )
        )


def _add_illustrative_payoff(
    fig: go.Figure, spot_range: np.ndarray, spot_price: float, position: dict
) -> None:
    """Sketch for the non-picker path-dependent families (asian, lookbacks)."""
    inst = position["instrument_class"]
    strike = float(position["strike"])
    is_call = position["option_type"] == "call"

    if inst == "asian":
        _add_vanilla_reference(fig, spot_range, strike, is_call)
        fig.data[-1].name = "Vanilla Payoff (upper bound)"
        # Geometric averaging reduces effective volatility by 1/sqrt(3),
        # which compresses the payoff toward ATM by the same factor.
        compression = 1.0 / math.sqrt(3.0)
        diff = (spot_range - strike) if is_call else (strike - spot_range)
        fig.add_trace(
            go.Scatter(
                x=spot_range,
                y=np.maximum(diff * compression, 0.0),
                name="Illustrative Asian Payoff",
                line=dict(color=CHART_COLORS["primary"], width=2.5),
            )
        )
        note = (
            "Note: Geometric averaging compresses payoff by 1/√3. "
            "Actual payoff depends on price path."
        )
    elif inst == "lookback_floating":
        # Floating: call pays S_T - min(S). Always ITM — assume 10% excursion.
        if is_call:
            m_min = np.minimum(spot_range, spot_price * 0.90)
            lb_payoff = np.maximum(spot_range - m_min, 0.0)
        else:
            m_max = np.maximum(spot_range, spot_price * 1.10)
            lb_payoff = np.maximum(m_max - spot_range, 0.0)
        fig.add_trace(
            go.Scatter(
                x=spot_range,
                y=lb_payoff,
                name="Illustrative Lookback Payoff",
                line=dict(color=CHART_COLORS["primary"], width=2.5),
            )
        )
        note = (
            "Floating lookback: call pays S_T - min(S). Always ITM. "
            "Curve assumes 10% path excursion from spot."
        )
    else:  # lookback_fixed
        _add_vanilla_reference(fig, spot_range, strike, is_call)
        if is_call:
            m_max = np.maximum(spot_range, spot_price * 1.10)
            lb_payoff = np.maximum(m_max - strike, 0.0)
        else:
            m_min = np.minimum(spot_range, spot_price * 0.90)
            lb_payoff = np.maximum(strike - m_min, 0.0)
        fig.add_trace(
            go.Scatter(
                x=spot_range,
                y=lb_payoff,
                name="Illustrative Lookback Payoff",
                line=dict(color=CHART_COLORS["primary"], width=2.5),
            )
        )
        note = (
            "Fixed lookback payoff: max(M_max - K, 0) for call. Vanilla payoff "
            "shown as lower bound; upper curve assumes 10% path excursion."
        )

    fig.add_annotation(
        x=0.5,
        y=0.02,
        xref="paper",
        yref="paper",
        text=note,
        showarrow=False,
        font=_NOTE_FONT,
    )


def _add_family_annotations(fig: go.Figure, position: dict) -> None:
    """Level markers specific to the family (payout, trigger, corridor, cap)."""
    inst = position["instrument_class"]
    strike = float(position["strike"])

    if inst == "digital":
        payout = float(position.get("payout", 1.0))
        fig.add_hline(
            y=payout,
            line_dash="dot",
            line_color=_ANNOTATION_COLOR,
            line_width=1,
            annotation_text=f"Payout = {payout:.2f}",
            annotation_position="top left",
        )
    elif inst == "gap":
        trigger = float(position.get("gap_trigger", strike))
        fig.add_vline(
            x=trigger,
            line_dash="dash",
            line_color="#dc2626",
            line_width=2,
            annotation_text=f"Trigger K2 = {trigger:.1f}",
            annotation_position="top left",
        )
    elif inst == "supershare":
        lower = float(position.get("lower_strike", 0.9 * strike))
        upper = float(position.get("upper_strike", 1.1 * strike))
        for level, name in ((lower, "K_L"), (upper, "K_U")):
            fig.add_vline(
                x=level,
                line_dash="dash",
                line_color="#dc2626",
                line_width=1.5,
                annotation_text=f"{name} = {level:.1f}",
                annotation_position="top left",
            )
    elif inst == "capped_power":
        cap = float(position.get("cap", 0.0))
        if cap > 0.0:
            fig.add_hline(
                y=cap,
                line_dash="dot",
                line_color=_ANNOTATION_COLOR,
                line_width=1,
                annotation_text=f"Cap = {cap:.2f}",
                annotation_position="top left",
            )
    elif inst == "arithmetic_asian":
        w = min(max(float(position.get("avg_elapsed_pct", 0.0)), 0.0), 0.95)
        sa = float(position.get("avg_realized", 0.0))
        fig.add_annotation(
            x=0.5,
            y=0.02,
            xref="paper",
            yref="paper",
            text=(
                f"Averaging: A_T = {w:.0%}·SA + {1 - w:.0%}·S_T (SA = {sa:.1f}) — "
                "assumes the remaining averaging path sits at S_T."
            ),
            showarrow=False,
            font=_NOTE_FONT,
        )

    # Barrier levels of any discrete-event leg (dotted red, shared helper).
    add_barrier_markers(fig, [position])


def create_exotic_payoff_diagram(
    spot_range: np.ndarray,
    spot_price: float,
    position: dict,
) -> go.Figure:
    """Terminal payoff diagram for one exotic leg (raw position dict)."""
    inst = position.get("instrument_class", "vanilla")
    strike = float(position["strike"])
    is_call = position["option_type"] == "call"

    layout = get_layout_config(title="Terminal Payoff at Expiration", height=450)
    layout["xaxis"]["title"] = {
        "text": "Underlying Price",
        "font": AXIS_DEFAULTS["title"]["font"],
    }
    layout["yaxis"]["title"] = {
        "text": "Payoff",
        "font": AXIS_DEFAULTS["title"]["font"],
    }
    fig = go.Figure(layout=layout)

    if inst in _VANILLA_REFERENCE_FAMILIES:
        _add_vanilla_reference(fig, spot_range, strike, is_call)

    spec = PAYOFF_SCENARIOS.get(inst)
    if inst in ("asian", "lookback_floating", "lookback_fixed"):
        _add_illustrative_payoff(fig, spot_range, spot_price, position)
    elif spec is not None and spec["kind"] == "discrete_event":
        _add_outcome_overlay(fig, spot_range, position)
    else:
        # Terminal families — and the safety net for any future class: the
        # vectorized adapter falls back to the terminal intrinsic, so no
        # family can render an empty chart again.
        _add_terminal_payoff(fig, spot_range, position)

    _add_family_annotations(fig, position)

    # K and S vlines — place on opposite sides when close to avoid overlap.
    close = abs(spot_price - strike) < (spot_price * 0.06)
    fig.add_vline(
        x=strike,
        line_dash="dot",
        line_color="#64748b",
        line_width=1,
        annotation_text=f"K = {strike:.1f}",
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=spot_price,
        line_dash="dot",
        line_color=CHART_COLORS["reference"],
        line_width=1,
        annotation_text=f"S = {spot_price:.1f}",
        annotation_position="top left" if close else "top right",
    )

    return fig
