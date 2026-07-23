"""Custom Model tab — define and register a user model.

Creation only: write a Python ``Model`` subclass, validate it, and register it.
Once registered, the model joins the sidebar's **Generator** dropdown and
**Candidate Models** picker (surface family), and is calibrated through the
normal Run pipeline like any built-in model — there is deliberately no
calibrate button here.
"""

from __future__ import annotations

import streamlit as st

from components.custom_model_editor import render_custom_model_editor
from services.custom_model_service import get_custom_meta, is_registered


def render(ctx: dict) -> None:  # noqa: ARG001 — uniform tab signature
    """Render the define → validate → register workflow."""
    render_custom_model_editor()

    if is_registered():
        meta = get_custom_meta() or {}
        st.markdown("---")
        st.success(
            f"**{meta.get('name', 'Your model')}** is registered. It now appears "
            "in the sidebar under **🧬 Generator** and **🎯 Candidate Models** "
            "(surface family). Select it there and press **▶ Run calibration** to "
            "fit it — results show up in the Live, Diagnostics and "
            "Compare & Restarts tabs like any built-in model."
        )
        st.caption(
            "Tip: pick a built-in model (e.g. Heston) as the Generator and your "
            "custom model as a Candidate to fit it to a known surface — or set "
            "your model as the Generator to calibrate other models against it."
        )
