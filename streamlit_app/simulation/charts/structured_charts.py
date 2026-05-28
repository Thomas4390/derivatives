"""
Structured Product Overlays for Simulation Explorer.

Adds barrier lines, triggers, observation dates, and event markers
to the main simulation paths chart.

Author: Thomas Vaudescal
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from functools import partial

from config.chart_theme import badge
from config.constants import SP_PRODUCT_COLORS

_badge = partial(badge, font_size=11)


def add_structured_overlays(
    fig: go.Figure,
    price_paths: np.ndarray,
    time_grid: np.ndarray,
    idx_sample: np.ndarray,
    product_config: dict,
    sp_result,
    viz_options: dict | None = None,
) -> None:
    """Add structured product overlays to the simulation paths chart."""
    product_type = product_config["product_type"]
    params = product_config["product_params"]
    s0 = price_paths[0, 0]
    obs_times = sp_result.observation_times

    # ── Observation date vertical lines ──
    if len(obs_times) > 0:
        for t_obs in obs_times:
            fig.add_vline(
                x=t_obs,
                line_dash="dot",
                line_color="rgba(255, 255, 255, 0.12)",
                line_width=1,
                row=1,
                col=1,
            )

    vopts = viz_options or {}

    if product_type == "cpn":
        _overlay_cpn(fig, s0, params)
    elif product_type == "reverse_convertible":
        _overlay_rc(
            fig, s0, params, price_paths, idx_sample, time_grid, sp_result, vopts
        )
    elif product_type == "autocallable":
        _overlay_autocallable(
            fig, s0, params, price_paths, idx_sample, time_grid, sp_result, vopts
        )


def _overlay_cpn(fig, s0, params):
    """CPN overlays: protection floor + cap line."""
    protection = params.get("protection_level", 1.0)
    cap = params.get("cap")

    fig.add_hline(
        y=s0 * protection,
        line_dash="dash",
        line_color=SP_PRODUCT_COLORS["bond_floor"],
        line_width=1.5,
        annotation_text=f" Protection ({protection:.0%}) ",
        annotation_position="bottom right",
        **_badge(SP_PRODUCT_COLORS["bond_floor"]),
        row=1,
        col=1,
    )

    if cap is not None:
        fig.add_hline(
            y=s0 * cap,
            line_dash="dash",
            line_color=SP_PRODUCT_COLORS["option_value"],
            line_width=1.5,
            annotation_text=f" Cap ({cap:.0%}) ",
            annotation_position="top right",
            **_badge(SP_PRODUCT_COLORS["option_value"]),
            row=1,
            col=1,
        )


def _overlay_rc(
    fig, s0, params, paths, idx_sample, time_grid, sp_result, viz_options=None
):
    """RC overlays: barrier line + breach markers."""
    barrier = params["barrier"]

    fig.add_hline(
        y=s0 * barrier,
        line_dash="dash",
        line_color=SP_PRODUCT_COLORS["barrier"],
        line_width=1.5,
        annotation_text=f" KI Barrier ({barrier:.0%}) ",
        annotation_position="bottom right",
        **_badge(SP_PRODUCT_COLORS["barrier"]),
        row=1,
        col=1,
    )

    # Breach markers on displayed paths
    vopts = viz_options or {}
    n_markers = vopts.get("rc_n_markers", 30)
    barrier_level = s0 * barrier
    for i in idx_sample[:n_markers]:
        if sp_result.barrier_breached[i]:
            path = paths[i]
            breach_steps = np.where(path <= barrier_level)[0]
            if len(breach_steps) > 0:
                first = breach_steps[0]
                fig.add_trace(
                    go.Scatter(
                        x=[time_grid[first]],
                        y=[path[first]],
                        mode="markers",
                        marker=dict(
                            symbol="x",
                            size=8,
                            color=SP_PRODUCT_COLORS["barrier"],
                            line=dict(width=1, color="rgba(255,255,255,0.4)"),
                        ),
                        showlegend=False,
                        hoverinfo="text",
                        hovertext=f"Barrier breach t={time_grid[first]:.2f}",
                    )
                )


def _overlay_autocallable(
    fig, s0, params, paths, idx_sample, time_grid, sp_result, viz_options=None
):
    """Autocallable overlays: trigger + coupon barrier + KI barrier + autocall markers."""
    trigger = params["autocall_trigger"]
    coupon_barrier = params["coupon_barrier"]
    ki_barrier = params["ki_barrier"]

    fig.add_hline(
        y=s0 * trigger,
        line_dash="dash",
        line_color=SP_PRODUCT_COLORS["autocall"],
        line_width=1.5,
        annotation_text=f" Autocall ({trigger:.0%}) ",
        annotation_position="top right",
        **_badge(SP_PRODUCT_COLORS["autocall"]),
        row=1,
        col=1,
    )

    fig.add_hline(
        y=s0 * coupon_barrier,
        line_dash="dot",
        line_color=SP_PRODUCT_COLORS["coupon_pv"],
        line_width=1,
        annotation_text=f" Coupon ({coupon_barrier:.0%}) ",
        annotation_position="bottom right",
        **_badge(SP_PRODUCT_COLORS["coupon_pv"]),
        row=1,
        col=1,
    )

    fig.add_hline(
        y=s0 * ki_barrier,
        line_dash="dash",
        line_color=SP_PRODUCT_COLORS["barrier"],
        line_width=1.5,
        annotation_text=f" KI Barrier ({ki_barrier:.0%}) ",
        annotation_position="bottom right",
        **_badge(SP_PRODUCT_COLORS["barrier"]),
        row=1,
        col=1,
    )

    # Autocall event markers on displayed paths
    vopts = viz_options or {}
    n_markers = vopts.get("autocall_n_markers", 30)
    obs_indices = sp_result.observation_indices
    for i in idx_sample[:n_markers]:
        obs_idx = sp_result.autocall_date_index[i]
        if obs_idx >= 0:
            step = obs_indices[obs_idx]
            fig.add_trace(
                go.Scatter(
                    x=[time_grid[step]],
                    y=[paths[i, step]],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=9,
                        color=SP_PRODUCT_COLORS["autocall"],
                        line=dict(width=1, color="rgba(255,255,255,0.4)"),
                    ),
                    showlegend=False,
                    hoverinfo="text",
                    hovertext=f"Autocalled t={time_grid[step]:.2f}",
                )
            )
