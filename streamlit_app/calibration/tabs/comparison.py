"""Tab — Compare & Restarts: cross-(model × solver) recap, Pareto, overlays."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from charts.comparison import (
    multi_comparison_table,
    render_multi_overlaid_loss,
    render_multi_pareto,
    render_parameter_recovery_bars,
)
from charts.info_criteria_table import render_info_criteria_table
from config.constants import MODEL_DISPLAY_NAMES
from services import state_manager
from services.model_selection_service import compute_info_criteria
from streamlit_app.simulation.config.styles import section_header_html  # type: ignore
from tabs._helpers import series_view_filter, subdict_from_series

GARCH_FAMILY_KEYS = frozenset({"garch", "ngarch", "gjr_garch"})


def render(ctx: dict) -> None:
    data_source = ctx["data_source"]
    true_params = ctx["true_params"]
    generator_model = ctx.get("generator_model")
    ensure_data = ctx["ensure_data"]

    results_all = state_manager.get("calib_results") or {}
    if not results_all:
        st.info(
            "Pick ≥ 1 candidate model and ≥ 1 solver in the sidebar, then click "
            "**▶ Run calibration** to populate this comparison."
        )
        return

    # ── Cross-model information criteria (top of tab) ─────────────────
    # Central pedagogical artefact: which model wins on AIC/BIC, and is
    # the win statistically significant against the nested baseline?
    market_data, _meta = ensure_data()
    if market_data is not None:
        st.markdown(
            section_header_html(
                "🏆", "Cross-model selection · AIC / BIC / LR-test",
            ),
            unsafe_allow_html=True,
        )
        info_rows = compute_info_criteria(results_all, market_data)
        render_info_criteria_table(info_rows)

    n_models = len(results_all)
    n_pairs = sum(len(v) for v in results_all.values())
    st.markdown(
        section_header_html(
            "📊",
            f"Recap · {n_models} model(s) × ≤{n_pairs // max(n_models, 1)} solvers = {n_pairs} runs",
        ),
        unsafe_allow_html=True,
    )
    _render_run_metadata()
    recap_df = pd.DataFrame(multi_comparison_table(results_all))
    styled = recap_df.style.format(
        {
            "RMSE price": "{:.3e}",
            "RMSE IV (bps)": "{:.2f}",
            "Final loss": "{:.3e}",
            "Grad norm": "{:.3e}",
            "Iterations": "{:d}",
            "Elapsed (s)": "{:.2f}",
        },
        na_rep="—",
    )
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        column_config={
            "Error": st.column_config.TextColumn(
                "Error",
                help=(
                    "Solver failure message — empty when the run completed. "
                    "Click a cell to see the full text; the complete traceback "
                    "is logged to the console where the app was started."
                ),
                width="medium",
            ),
        },
    )
    if any(k in GARCH_FAMILY_KEYS for k in results_all):
        st.caption(
            "ℹ️ For GARCH-family models, **ω is stored on the annualised "
            "scale** (multiplied by `annualization_factor`, 252 by default); "
            "per-period diagnostics divide ω by that factor before running "
            "the variance filter. α, β, γ and θ are dimensionless and "
            "unchanged."
        )

    # Per-chart view filter — pick exactly which (model, solver, objective)
    # runs to plot without re-running. Defaults to every run selected; the
    # pruned selection feeds the multi-* builders via a rebuilt sub-dict
    # (same nested shape they already consume).
    selected = series_view_filter(results_all, key="compare_charts")
    if not selected:
        st.caption("Nothing to plot — select at least one run above.")
        return
    visible_sub = subdict_from_series(selected)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            section_header_html("⚡", "Pareto frontier · speed × accuracy"),
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            render_multi_pareto(visible_sub),
            width="stretch",
            key="compare_pareto",
        )
        # Encoding hint moved out of the (former) in-figure title so the
        # legend has the top of the chart to itself.
        st.caption("Colour = model · shape = solver.")
    with col2:
        st.markdown(
            section_header_html("📉", "Loss overlay · every (model, solver)"),
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            render_multi_overlaid_loss(visible_sub),
            width="stretch",
            key="compare_loss_overlay",
        )

    # Parameter recovery only makes sense for the generator's model
    # (we know its true params). Show that block scoped to it.
    if data_source == "synthetic" and generator_model is not None \
            and generator_model in results_all:
        st.markdown(
            section_header_html(
                "🎯",
                f"Parameter recovery · {MODEL_DISPLAY_NAMES.get(generator_model, generator_model)} (generator)",
            ),
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            render_parameter_recovery_bars(
                results_all[generator_model], true_params,
            ),
            width="stretch",
        )
        st.caption(
            "Recovery bars are scoped to the generator model — only it has "
            "a ground truth. Other candidates compare their fit quality via "
            "RMSE / Pareto above instead."
        )
    else:
        st.caption(
            "Parameter recovery requires synthetic data with the generator "
            "among the calibrated candidates."
        )

    # LM-JAX multi-start details, fused from the old dedicated tab.
    with st.expander("🔁 LM-JAX multi-start landscape (per model)", expanded=False):
        st.caption(
            "Expand to inspect basins of attraction across LM-JAX restarts. "
            "Available only for surface models (Heston / Merton / Bates) where "
            "the calibrator loops over `n_restarts` independent starts."
        )
        _render_multistart_expander(results_all)


def _render_run_metadata() -> None:
    """Caption with timestamp + truncated config hashes.

    Shows when the on-screen numbers were produced and which config
    produced them. If the user moves a sidebar slider after running,
    the data_hash on the right will drift away from the displayed
    config_hash and the caption acts as a visual cue that a re-run is
    needed. Pedagogical only — no persistence.
    """
    last_run_ts = state_manager.get("calib_last_run_ts")
    if last_run_ts is None:
        return
    local_ts = (
        last_run_ts.astimezone()
        if isinstance(last_run_ts, datetime)
        else last_run_ts
    )
    data_hash = state_manager.get("calib_data_hash") or ""
    config_hash = state_manager.get("calib_results_hash") or ""
    st.caption(
        f"⏱ Calibrated at {local_ts:%H:%M:%S} · "
        f"data `{data_hash[:8] or '—'}` · "
        f"config `{config_hash[:8] or '—'}`"
    )


def _render_multistart_expander(results_all: dict) -> None:
    """Render the per-model LM-JAX multi-start panel (fused from the
    standalone Multi-Start tab). One sub-section per model that has a
    successful LM-JAX result with restart history."""
    from charts.live_convergence import render_multi_start_loss

    from tabs._helpers import _is_nested_objectives  # noqa: PLC0415 — local

    rendered = False
    for model_key, per_solver in results_all.items():
        lm = per_solver.get("LM-JAX")
        # New schema: lm is dict[objective, Summary]. Resolve the active
        # objective (or fall back to first) before dereferencing .result.
        if _is_nested_objectives(lm):
            from services import state_manager as _sm  # noqa: PLC0415
            obj_key = _sm.get("calib_active_objective") or next(iter(lm), None)
            lm = lm.get(obj_key) if obj_key else None
        if lm is None or lm.result is None:
            continue
        diag = lm.result.diagnostics or {}
        msh = diag.get("multi_start_history")
        best_idx = diag.get("best_start_index")
        if not msh:
            continue
        rendered = True
        st.markdown(
            f"**{MODEL_DISPLAY_NAMES.get(model_key, model_key)}** · "
            f"{len(msh)} restart(s)"
        )
        st.plotly_chart(
            render_multi_start_loss(msh, best_index=best_idx),
            width="stretch",
            key=f"multistart_{model_key}",
        )
        lm_runs = diag.get("lm_runs") or diag.get("lm_runs_joint") or []
        if lm_runs:
            lm_df = pd.DataFrame(lm_runs)
            format_map: dict[str, str] = {}
            if "rss" in lm_df.columns:
                format_map["rss"] = "{:.3e}"
            for col in ("nfev", "status", "start"):
                if col in lm_df.columns:
                    format_map[col] = "{:d}"
            st.dataframe(
                lm_df.style.format(format_map, na_rep="—"),
                width="stretch",
                hide_index=True,
                key=f"multistart_table_{model_key}",
            )
    if not rendered:
        st.info(
            "No multi-start history captured for any candidate. Enable LM-JAX "
            "as a solver and set **n_restarts > 1** in the sidebar advanced "
            "settings before re-running."
        )
