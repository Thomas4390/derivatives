"""Tab — Theory.

Pedagogical, solver-focused tab. Two audiences in one page:

* **Undergraduate / Bachelor reader** — first-pass: intuition,
  mechanics in plain English, strengths / weaknesses, mini key-property
  cards.
* **M2 quant reader** — second-pass (in expanders): KaTeX update rules,
  convergence conditions, references to the canonical papers.

Six sections, in reading order:

1. Concept primer — what a solver *is* and the vocabulary used.
2. Loss being minimised (model-aware LaTeX). Unchanged from prior.
3. Parameter cheat-sheet (model-aware). Unchanged from prior.
4. Three-axes framework — the orthogonal design axes that distinguish
   our four solvers.
5. Per-solver deep dives — LM-JAX, DE, NM, L-BFGS-B. The centrepiece.
6. Picker — head-to-head matrix + decision tree.
7. Further reading — consolidated bibliography (textbooks + tutorials).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from charts.solver_gallery import SOLVER_FIGURES
from components.model_equations import render_model_card
from config.constants import (
    SOLVER_BADGES,
    SOLVER_ICONS,
)
from config.solver_theory import (
    AXES_CAPTION_MD,
    AXES_TABLE_ROWS,
    COMPARISON_MATRIX_ROWS,
    CONCEPT_INTRO_MD,
    DECISION_AXES_MD,
    DECISION_TREE_CAPTION_MD,
    DECISION_TREE_MD,
    FURTHER_READING_MD,
    SOLVER_DEEP_DIVES,
    VOCAB_EXPANDER_MD,
)
from streamlit_app.simulation.config.styles import section_header_html  # type: ignore


def render(ctx: dict) -> None:
    generator = ctx.get("generator_model")
    candidates = ctx.get("candidate_models") or ()
    # Focus the model-aware blocks (loss + cheat-sheet) on the generator
    # when synthetic; otherwise the first candidate. iv_gbm is skipped
    # if anything else is available since it has no SDE of interest.
    focus = generator or (candidates[0] if candidates else "heston")

    st.markdown(section_header_html("📚", "Theory"), unsafe_allow_html=True)
    st.markdown(
        "Understand **what each solver actually does** and **when to "
        "pick it**. The page is layered: every block opens at "
        "undergraduate level; each *Want more detail?* expander pushes "
        "to M2-quant depth (update rules, convergence, references)."
    )

    _section_intro()
    render_model_card(focus)
    _section_axes_framework()
    _section_solver_deep_dives()
    _section_picker()
    _section_further_reading()


# ──────────────────────────────────────────────────────────────────────


def _section_intro() -> None:
    st.markdown(
        section_header_html("🧭", "What is a solver, really?"),
        unsafe_allow_html=True,
    )
    st.markdown(CONCEPT_INTRO_MD)
    with st.expander(
        "📖 Optimisation vocabulary in plain English", expanded=False,
    ):
        st.markdown(VOCAB_EXPANDER_MD)


# ── Solver pedagogy (centrepiece) ─────────────────────────────────────


def _section_axes_framework() -> None:
    st.markdown(
        section_header_html(
            "🧭", "Three axes that distinguish the four solvers",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(DECISION_AXES_MD)
    axes_df = pd.DataFrame(
        AXES_TABLE_ROWS,
        columns=["Solver", "Local / Global", "Loss shape", "Derivative info"],
    )
    st.dataframe(axes_df, width="stretch", hide_index=True)
    st.caption(AXES_CAPTION_MD)


def _badge_pill_html(label: str) -> str:
    """Inline pill rendered via ``st.html`` (avoids markdown sanitisation)."""
    return (
        "<span style='display:inline-block;padding:3px 10px;border-radius:9999px;"
        "background:#ecfeff;color:#0e7490;border:1px solid #67e8f9;"
        "font-size:0.78rem;font-weight:600;letter-spacing:0.01em'>"
        f"{label}</span>"
    )


def _properties_html(properties: dict[str, str]) -> str:
    rows = "".join(
        "<tr>"
        f"<td style='color:#475569;padding:3px 8px 3px 0'>{k}</td>"
        f"<td style='color:#0f172a;font-weight:600;padding:3px 0'>{v}</td>"
        "</tr>"
        for k, v in properties.items()
    )
    return (
        "<div style='font-size:0.85rem;line-height:1.55'>"
        "<div style='font-weight:600;color:#0f172a;margin-bottom:6px'>"
        "Key properties</div>"
        f"<table style='border-collapse:collapse'>{rows}</table>"
        "</div>"
    )


def _render_solver_deep_dive(dive) -> None:
    icon = SOLVER_ICONS.get(dive.key, "🛠️")
    st.markdown(
        section_header_html(icon, f"{dive.key} · {dive.long_name}"),
        unsafe_allow_html=True,
    )
    st.html(_badge_pill_html(SOLVER_BADGES.get(dive.key, "")))

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**🎯 Intuition** — {dive.intuition_md}")
        st.markdown(f"**⚙️ How it works** — {dive.mechanics_md}")
        with st.expander("📖 Want more detail? (M2 deep dive)", expanded=False):
            st.markdown(dive.deep_dive_md)
            if dive.update_rule_latex:
                st.markdown("**Update rule**")
                st.latex(dive.update_rule_latex)
            st.markdown("**References**")
            ref_lines = [f"- [{label}]({url})" for label, url in dive.references]
            st.markdown("\n".join(ref_lines))
    with right:
        fig_fn = SOLVER_FIGURES.get(dive.key)
        if fig_fn is not None:
            st.plotly_chart(
                fig_fn(), width="stretch", key=f"theory_dive_fig_{dive.key}",
            )
        with st.container(border=True):
            st.html(_properties_html(dive.properties))

    shines_col, struggles_col = st.columns(2)
    with shines_col:
        st.markdown("**✅ When it shines**")
        st.markdown("\n".join(f"- {b}" for b in dive.shines))
    with struggles_col:
        st.markdown("**⚠️ When it struggles**")
        st.markdown("\n".join(f"- {b}" for b in dive.struggles))
    st.divider()


def _section_solver_deep_dives() -> None:
    for dive in SOLVER_DEEP_DIVES:
        _render_solver_deep_dive(dive)


def _section_picker() -> None:
    st.markdown(
        section_header_html("🧮", "Which solver should I pick?"),
        unsafe_allow_html=True,
    )
    matrix_df = pd.DataFrame(
        COMPARISON_MATRIX_ROWS,
        columns=["Criterion", "LM-JAX", "DE", "NM", "L-BFGS-B"],
    )
    st.dataframe(matrix_df, width="stretch", hide_index=True)
    st.markdown(DECISION_TREE_MD)
    st.caption(DECISION_TREE_CAPTION_MD)


def _section_further_reading() -> None:
    st.markdown(
        section_header_html("📖", "Further reading"),
        unsafe_allow_html=True,
    )
    with st.expander(
        "📚 Citations, textbooks and online tutorials", expanded=False,
    ):
        st.markdown(FURTHER_READING_MD)
