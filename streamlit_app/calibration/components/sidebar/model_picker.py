"""Generator + candidate model pickers (multi-model refactor).

* ``render_generator(family)`` — segmented-button picker over family models.
  In synthetic mode, the generator's ``true_params`` produce the ground-truth
  market data. Hidden in real-data mode.
* ``render_candidates(family)`` — multi-pill picker over family models. These
  are the models that will be calibrated against the market data; ≥ 1
  selection enforced.

Both pickers' option lists are **dynamic**: a registered custom model joins the
surface family. ``_purge_stale_selection`` drops a persisted widget value that
is no longer a valid option (e.g. ``"custom"`` after it was unregistered) before
the widget instantiates, so the button group re-seeds from its default instead
of carrying a dangling selection.
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
from services.custom_model_service import is_registered

from ._option_help import compose_help


def _format(key: str) -> str:
    return f"{MODEL_ICONS.get(key, '·')}  {MODEL_DISPLAY_NAMES.get(key, key)}"


def _family_options(family: str) -> list[str]:
    """Family models, plus the registered custom model on the surface family.

    The custom model is a surface model, so it joins the surface picker once it
    has been registered in the Custom Model tab. Gated on the per-session
    ``is_registered`` flag so an unregistered slot never shows a stale option.
    """
    options = list(FAMILY_MODELS[family])
    if family == "surface" and is_registered() and "custom" not in options:
        options.append("custom")
    return options


def _purge_stale_selection(wkey: str, options: list[str]) -> None:
    """Drop a persisted widget value that is no longer a valid option.

    The option list is dynamic (the custom model appears/disappears), so a
    keyed widget could otherwise keep a dangling value (e.g. ``"custom"``
    after it was unregistered) and raise. The stale key is deleted *before*
    the widget instantiates so Streamlit re-seeds from the default, which
    already carries the surviving selection via ``state_manager``. Deletion
    only — a Session-State *write* on a key whose widget also passes
    ``default=`` makes Streamlit warn on every rerun.
    """
    val = st.session_state.get(wkey)
    if val is None:
        return
    stale = (
        any(m not in options for m in val)
        if isinstance(val, list)
        else val not in options
    )
    if stale:
        del st.session_state[wkey]


def render_generator(family: str) -> str:
    """Segmented-button picker for the synthetic ground-truth model."""
    st.subheader("🧬 Generator (ground truth)")
    options = _family_options(family)
    current = state_manager.get("calib_generator_model") or FAMILY_DEFAULT_MODEL[family]
    if current not in options:
        current = FAMILY_DEFAULT_MODEL[family]
    _purge_stale_selection(f"generator_picker_{family}", options)
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
    options = _family_options(family)
    stored = state_manager.get("calib_candidate_models") or (FAMILY_DEFAULT_MODEL[family],)
    valid_stored = [m for m in stored if m in options] or [FAMILY_DEFAULT_MODEL[family]]
    _purge_stale_selection(f"candidates_picker_{family}", options)
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
