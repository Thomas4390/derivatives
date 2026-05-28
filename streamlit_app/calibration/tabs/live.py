"""Tab 2 — Live Calibration: convergence replay + parameter trajectories.

Every selected ``(model, solver, objective)`` run is superimposed on the
**same animated** charts (via ``series_view_filter``): the convergence
chart overlays each run's objective curve, and the parameter trajectories
overlay each run's path (faceted by model). Both open on the converged
state so the result is visible immediately after a fit — ▶ Replay animates
the descent from the start.
"""

from __future__ import annotations

import streamlit as st

from charts.live_convergence import (
    render_loss_overlay_animated,
    render_parameter_trajectories_animated,
)
from config.constants import MODEL_DISPLAY_NAMES
from services import state_manager
from streamlit_app.simulation.config.styles import (  # type: ignore
    render_metric_row,
    section_header_html,
)
from tabs._helpers import series_view_filter, termination_reason
from utils.plotly_theme import series_style


def render(ctx: dict) -> None:
    true_params = ctx["true_params"]
    data_source = ctx["data_source"]
    generator_model = ctx.get("generator_model")

    results = state_manager.get("calib_results") or {}
    if not results:
        st.info(
            "Click the **▶ Run calibration** button in the sidebar to start. "
            "Once a solver is running, the convergence trace and the "
            "parameter trajectories will stream into this tab."
        )
        return

    selected = series_view_filter(results, key="live")
    if not selected:
        st.info("Select at least one run above to display its convergence.")
        return

    # Distinct colour PER RUN (multi_model=False) so several variations of the
    # SAME model are still told apart on the legend — the model stays in the
    # label and the dash encodes the solver. The same style is reused on the
    # convergence + parameter charts so a run keeps one colour across both.
    styles = {
        s.key: series_style(s.model, s.solver, multi_model=False, index=i)
        for i, s in enumerate(selected)
    }

    # ── Convergence — every run's objective overlaid, animated ───────────
    st.markdown(
        section_header_html("⏱", "Convergence · objectives superimposed · ▶ replay"),
        unsafe_allow_html=True,
    )
    if len(selected) == 1:
        # A single run keeps the at-a-glance metric strip for context.
        s = selected[0]
        res = s.summary.result
        render_metric_row(
            [
                ("Solver", s.solver, "active fit"),
                (
                    "RMSE price",
                    f"{res.rmse_price:.3e}" if res.rmse_price is not None else "—",
                    "price units",
                ),
                ("Elapsed", f"{s.summary.elapsed:.2f} s", "wall-clock"),
                ("Iterations", f"{res.n_iterations}", "logged callbacks"),
            ]
        )
    conv_traces = [
        (s.label, s.summary.result.iteration_history, styles[s.key]) for s in selected
    ]
    st.plotly_chart(
        render_loss_overlay_animated(conv_traces, title=""),
        width="stretch",
        key="live_convergence",
    )
    # Why each fit stopped — the optimiser's termination message (surface
    # solvers) or the success flag (GARCH MLE), with the evaluation count.
    st.caption(
        "  \n".join(
            f"{'✓' if s.summary.result.success else '⏸' if getattr(s.summary, 'partial', False) else '⚠'} "
            f"**{s.label}** — "
            f"{termination_reason(s.summary)}  ·  "
            f"{s.summary.result.n_iterations} evals"
            for s in selected
        )
    )

    # ── Parameter trajectories — faceted by model ────────────────────────
    # Different models have different parameter sets, so one overlaid
    # trajectory chart only makes sense within a model.
    st.markdown(
        section_header_html("🎯", "Parameter trajectories · runs superimposed"),
        unsafe_allow_html=True,
    )
    by_model: dict[str, list] = {}
    for s in selected:
        by_model.setdefault(s.model, []).append(s)

    for model_key, model_series in by_model.items():
        if len(by_model) > 1:
            st.caption(MODEL_DISPLAY_NAMES.get(model_key, model_key))
        # True params overlay only when the model IS the synthetic generator.
        show_truth = data_source == "synthetic" and model_key == generator_model
        tp = true_params if show_truth else None
        traj_traces = [
            (s.label, s.summary.result.iteration_history, styles[s.key])
            for s in model_series
        ]
        st.plotly_chart(
            render_parameter_trajectories_animated(traj_traces, true_params=tp),
            width="stretch",
            key=f"live_traj_{model_key}",
        )
