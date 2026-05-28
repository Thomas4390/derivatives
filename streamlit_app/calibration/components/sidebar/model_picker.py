"""Generator + candidate model pickers (multi-model refactor).

* ``render_generator(family)`` — selectbox over family models. In synthetic
  mode, the generator's ``true_params`` produce the ground-truth market
  data. Hidden in real-data mode.
* ``render_candidates(family)`` — multiselect over family models. These
  are the models that will be calibrated against the market data; ≥ 1
  selection enforced.
"""

from __future__ import annotations

import streamlit as st

from config.constants import (
    FAMILY_DEFAULT_MODEL,
    FAMILY_MODELS,
    MODEL_DISPLAY_NAMES,
    MODEL_GROUP_HELP_CANDIDATES,
    MODEL_GROUP_HELP_GENERATOR,
    MODEL_HOVER,
    MODEL_ICONS,
)
from services import state_manager

from ._option_help import compose_help


def _format(key: str) -> str:
    return f"{MODEL_ICONS.get(key, '·')}  {MODEL_DISPLAY_NAMES[key]}"


def render_generator(family: str) -> str:
    """Pill-style picker for the synthetic ground-truth model."""
    st.subheader("🧬 Generator (ground truth)")
    options = list(FAMILY_MODELS[family])
    current = state_manager.get("calib_generator_model") or FAMILY_DEFAULT_MODEL[family]
    if current not in options:
        current = FAMILY_DEFAULT_MODEL[family]
    key = st.segmented_control(
        "generator",
        options=options,
        format_func=_format,
        default=current,
        selection_mode="single",
        label_visibility="collapsed",
        help=compose_help(
            MODEL_GROUP_HELP_GENERATOR, tuple(options), MODEL_HOVER, _format
        ),
        key=f"generator_picker_{family}",
    )
    if key is None:
        key = current
    state_manager.set("calib_generator_model", key)
    return key


def render_candidates(family: str) -> tuple[str, ...]:
    """Multi-pill picker for the calibration candidates. Always ≥ 1."""
    st.subheader("🎯 Candidate Models")
    options = list(FAMILY_MODELS[family])
    stored = state_manager.get("calib_candidate_models") or (FAMILY_DEFAULT_MODEL[family],)
    valid_stored = [m for m in stored if m in options] or [FAMILY_DEFAULT_MODEL[family]]
    picked = st.pills(
        "candidates",
        options=options,
        default=valid_stored,
        selection_mode="multi",
        format_func=_format,
        label_visibility="collapsed",
        help=compose_help(
            MODEL_GROUP_HELP_CANDIDATES, tuple(options), MODEL_HOVER, _format
        ),
        key=f"candidates_picker_{family}",
    )
    if not picked:
        picked = [FAMILY_DEFAULT_MODEL[family]]
        st.warning(
            f"No candidate selected → defaulted to {MODEL_DISPLAY_NAMES[picked[0]]}."
        )
    candidates = tuple(picked)
    state_manager.set("calib_candidate_models", candidates)
    return candidates
