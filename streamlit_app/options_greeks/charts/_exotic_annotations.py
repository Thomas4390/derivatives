"""Plotly barrier-vline annotations for discrete-event exotic legs. Kept separate
from the pure ``services.scenario_labels`` helper so the label logic stays
Plotly-free and testable, and so the P&L, payoff-diagram and Greeks charts share
one annotation implementation. (Infeasible-region shading was retired with the
OG-1 overlay: the NaN-masked outcome curves already tile the spot axis.)
"""

from __future__ import annotations

import plotly.graph_objects as go

from config.exotic_config import PAYOFF_SCENARIOS
from services.scenario_labels import barrier_levels

_BARRIER_COLOR = "#dc2626"


def add_barrier_markers(fig: go.Figure, positions: list[dict]) -> None:
    """Add a dotted vline at each barrier level of every discrete-event leg."""
    seen: set[float] = set()
    for pos in positions or []:
        spec = PAYOFF_SCENARIOS.get(pos.get("instrument_class"))
        if not spec or spec.get("kind") != "discrete_event":
            continue
        for lvl in barrier_levels(pos):
            if lvl in seen:
                continue
            seen.add(lvl)
            fig.add_vline(
                x=lvl,
                line_dash="dot",
                line_color=_BARRIER_COLOR,
                line_width=1,
                annotation_text=f"barrier {lvl:.0f}",
                annotation_position="top",
                annotation_font_size=10,
                annotation_font_color=_BARRIER_COLOR,
                exclude_empty_subplots=False,
            )
