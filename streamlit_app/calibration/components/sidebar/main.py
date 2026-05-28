"""Top-level sidebar orchestrator — composes the smaller panels.

Post multi-model refactor:
  family → data source → generator (synth only) + true_params (synth only)
       → synth config (synth only) → candidates → solvers → run.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from components.sidebar import (
    constraints_panel,
    data_source,
    family_picker,
    model_picker,
    objective_panel,
    solver_panel,
    synthetic_config,
    true_params,
)
from services import state_manager


def _render_run_button(blocking_error: str | None) -> bool:
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    return st.button(
        "▶  Run calibration",
        width="stretch",
        type="primary",
        help=blocking_error or "Run every selected solver on every candidate model and stream live progress.",
        disabled=blocking_error is not None,
    )


def render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        family = family_picker.render()
        st.markdown("---")
        data_src = data_source.render(family)
        st.markdown("---")

        if data_src["source"] == "synthetic":
            generator_model = model_picker.render_generator(family)
            st.markdown("---")
            params = true_params.render(generator_model, hide_for_real=False)
            st.markdown("---")
            data_config = synthetic_config.render(generator_model)
        else:
            generator_model = None
            params = {}
            data_config = {"_mode": "real", **data_src["real_cfg"]}
        st.markdown("---")

        candidate_models = model_picker.render_candidates(family)
        st.markdown("---")

        solvers, solver_settings = solver_panel.render(candidate_models)
        st.markdown("---")

        objectives, objective_settings = objective_panel.render(candidate_models)
        st.markdown("---")

        constraint_settings = constraints_panel.render(candidate_models)
        st.markdown("---")

        # true_params.render has already populated calib_blocking_error for
        # us — reading it here keeps the ordering robust even if more panels
        # start contributing validation errors later.
        blocking_error = state_manager.get("calib_blocking_error")
        run_clicked = _render_run_button(blocking_error)

    return {
        "data_family": family,
        "generator_model": generator_model,
        "candidate_models": candidate_models,
        "data_source": data_src["source"],
        "true_params": params,
        "data_config": data_config,
        "solvers": solvers,
        "solver_settings": solver_settings,
        "objectives": objectives,
        "objective_settings": objective_settings,
        "constraint_settings": constraint_settings,
        "run_clicked": run_clicked,
        "blocking_error": blocking_error,
    }
