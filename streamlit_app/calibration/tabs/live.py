"""Tab 2 — Live Calibration: the *fit anatomy* (loss + parameter roles).

Every selected ``(model, solver, objective)`` run is superimposed (via
``series_view_filter``) on one **animated** figure per model: the objective's
descent on top, co-registered with each parameter's path in a 2-column grid
below, on the shared evaluation axis — so the role of each parameter in the
fit is legible. A cross-model loss overlay is shown above only when more than
one model is selected. Charts open on the converged state — ▶ Replay animates
the descent from the start.
"""

from __future__ import annotations

import streamlit as st

from charts.live_convergence import (
    render_fit_anatomy_animated,
    render_loss_overlay_animated,
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
            "Once a solver is running, the fit anatomy — the loss descent and "
            "each parameter's path — will stream into this tab."
        )
        return

    selected = series_view_filter(results, key="live")
    if not selected:
        st.info("Select at least one run above to display its convergence.")
        return

    # Distinct colour PER RUN (multi_model=False) so several variations of the
    # SAME model are still told apart on the legend — the model stays in the
    # label and the dash encodes the solver. The same style is reused across the
    # loss row + parameter panels so a run keeps one colour throughout.
    styles = {
        s.key: series_style(s.model, s.solver, multi_model=False, index=i)
        for i, s in enumerate(selected)
    }

    by_model: dict[str, list] = {}
    for s in selected:
        by_model.setdefault(s.model, []).append(s)

    # ── Cross-model convergence — only when >1 model is overlaid ──────────
    # Parameter sets differ per model, so the per-model anatomy figures below
    # can't carry the cross-model comparison; with a single model this overlay
    # is redundant with that figure's loss row, so it's skipped here (the
    # Compare tab still keeps a static cross-model loss overlay).
    if len(by_model) > 1:
        st.markdown(
            section_header_html("⏱", "Convergence · all models · ▶ replay"),
            unsafe_allow_html=True,
        )
        conv_traces = [
            (s.label, s.summary.result.iteration_history, styles[s.key]) for s in selected
        ]
        st.plotly_chart(
            render_loss_overlay_animated(conv_traces, title=""),
            width="stretch",
            key="live_convergence",
        )

    # A single run keeps the at-a-glance metric strip for context.
    if len(selected) == 1:
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

    # ── Fit anatomy — loss + parameter roles, one figure per model ───────
    # The objective's descent (top) is co-registered with each parameter's path
    # (2-column grid below) on the shared evaluation axis; one slider / ▶ Replay
    # drives them together so the role of each parameter in the fit is legible.
    # Per-model because different models have different parameter sets.
    st.markdown(
        section_header_html("🔬", "Fit anatomy · loss + parameter roles · ▶ replay"),
        unsafe_allow_html=True,
    )
    st.caption(
        "Top: the objective's descent. Below: each parameter's path to its fitted "
        "value (red dashed = true value on synthetic data). Scrub or ▶ Replay to "
        "watch the loss fall as the parameters settle into place."
    )
    for model_key, model_series in by_model.items():
        if len(by_model) > 1:
            st.caption(MODEL_DISPLAY_NAMES.get(model_key, model_key))
        # True params overlay only when the model IS the synthetic generator.
        show_truth = data_source == "synthetic" and model_key == generator_model
        tp = true_params if show_truth else None
        anat_traces = [
            (s.label, s.summary.result.iteration_history, styles[s.key])
            for s in model_series
        ]
        st.plotly_chart(
            render_fit_anatomy_animated(anat_traces, true_params=tp, model_key=model_key),
            width="stretch",
            key=f"live_anatomy_{model_key}",
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
