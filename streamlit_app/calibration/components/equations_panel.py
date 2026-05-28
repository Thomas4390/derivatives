"""Pedagogical *Equations* panel at the top of the Setup & Data tab.

Mirrors the user's current sidebar selection — the generator, candidate
models, optimisers, and objective functions — and renders the
mathematical content (dynamics, loss, update rules, residual / scalar
loss) associated with each axis. Lives inside a single collapsible
expander with four inner ``st.tabs`` so the page stays readable when
several candidates × solvers × objectives are active at once.

All LaTeX and prose come from existing modules:
* ``config.formulas``      → ``MODEL_SDE_LATEX``, ``OBJECTIVE_*_LATEX``,
                              ``DUAN_LRNVR_LATEX``, ``STATIONARITY_*``
* ``config.solver_theory`` → ``SOLVER_DEEP_DIVES`` (per-solver dive)
* ``config.constants``     → ``MODEL_DISPLAY_NAMES``, ``SOLVER_BADGES``,
                              ``OBJECTIVE_DISPLAY_NAMES`` etc.
* ``components.model_equations`` → ``render_model_card`` (shared with
                                    the Theory tab).
"""

from __future__ import annotations

import streamlit as st

from components.model_equations import render_model_card
from config.constants import (
    GARCH_FAMILY,
    MODEL_DISPLAY_NAMES,
    MODEL_ICONS,
    OBJECTIVE_BADGES,
    OBJECTIVE_DESCRIPTIONS,
    OBJECTIVE_DISPLAY_NAMES,
    OBJECTIVE_ICONS,
    SOLVER_BADGES,
    SOLVER_ICONS,
    SURFACE_FAMILY,
)
from config.formulas import (
    GARCH_LOSS_LATEX,
    OBJECTIVE_AGGREGATE_LATEX,
    OBJECTIVE_RESIDUAL_LATEX,
)
from config.solver_theory import SOLVER_DEEP_DIVES
from services import state_manager
from streamlit_app.simulation.config.styles import (  # type: ignore
    p_measure_badge_html,
    q_measure_badge_html,
    section_header_html,
)

# Solver lookup by canonical key (SOLVER_DEEP_DIVES is an ordered tuple).
_SOLVER_DIVES = {dive.key: dive for dive in SOLVER_DEEP_DIVES}


def render(ctx: dict) -> None:
    """Render the Equations panel for the current sidebar selection."""

    generator = ctx.get("generator_model")
    candidates = tuple(ctx.get("candidate_models") or ())
    solvers = tuple(state_manager.get("calib_solver_selection") or ())
    objectives = tuple(state_manager.get("calib_objective_selection") or ())
    mode = ctx.get("mode", "surface")
    constraints = state_manager.get("calib_constraint_settings") or {}
    feller_mode = constraints.get("feller_mode")
    stationarity_mode = constraints.get("stationarity_mode")

    has_results = bool(state_manager.get("calib_results") or {})

    surface_candidates = tuple(c for c in candidates if c in SURFACE_FAMILY)
    show_objectives_tab = mode == "surface" and any(
        c not in ("iv_gbm",) for c in surface_candidates
    )

    tab_labels: list[str] = ["🎲 Generator"]
    tab_labels.append(f"🧪 Candidates ({len(candidates)})")
    tab_labels.append(f"🛠 Optimizers ({len(solvers)})")
    if show_objectives_tab:
        tab_labels.append(f"🎯 Objectives ({len(objectives)})")

    with st.expander(
        "📐 Equations for the current selection",
        expanded=not has_results,
    ):
        st.caption(
            "Live reference for whatever you've picked in the sidebar. "
            "Equations update as you change the generator, candidates, "
            "solvers, or objectives. Full pedagogy lives in the 📚 Theory "
            "tab."
        )

        _render_garch_scale_note(generator, candidates)

        tabs = st.tabs(tab_labels)
        with tabs[0]:
            _tab_generator(generator, feller_mode, stationarity_mode)
        with tabs[1]:
            _tab_candidates(
                candidates, generator, feller_mode, stationarity_mode
            )
        with tabs[2]:
            _tab_optimizers(solvers, mode)
        if show_objectives_tab:
            with tabs[3]:
                _tab_objectives(objectives, surface_candidates)

        _render_active_triangles_caption(
            candidates, solvers, objectives, show_objectives_tab
        )


# ─────────────────────────────────────────────────────────────────────
# Tab renderers
# ─────────────────────────────────────────────────────────────────────


def _tab_generator(
    generator: str | None,
    feller_mode: str | None,
    stationarity_mode: str | None,
) -> None:
    if not generator:
        st.info(
            "Pick a generator in the sidebar to see the true "
            "data-generating dynamics."
        )
        return

    _render_model_header(generator, suffix=" · ground-truth dynamics")
    if generator == "iv_gbm":
        st.caption(
            r"GBM baseline. A constant volatility $\sigma$ gives a flat "
            r"smile. Calibrating against an iv_gbm-generated surface is "
            r"just a closed-form Black-Scholes IV inversion per quote, "
            r"with no iterative loss."
        )
    render_model_card(
        generator,
        wrap_in_expander=False,
        show_cheatsheet=True,
        feller_mode=feller_mode,
        stationarity_mode=stationarity_mode,
    )


def _tab_candidates(
    candidates: tuple[str, ...],
    generator: str | None,
    feller_mode: str | None,
    stationarity_mode: str | None,
) -> None:
    if not candidates:
        st.info(
            "Pick at least one candidate model in the sidebar to populate "
            "this tab."
        )
        return

    for idx, candidate in enumerate(candidates):
        with st.container(border=True):
            recovery = candidate == generator
            suffix = " · also the generator (recovery test)" if recovery else ""
            _render_model_header(candidate, suffix=suffix)
            render_model_card(
                candidate,
                wrap_in_expander=False,
                show_cheatsheet=True,
                feller_mode=feller_mode,
                stationarity_mode=stationarity_mode,
            )
        if idx < len(candidates) - 1:
            st.write("")  # vertical breathing room between cards


def _tab_optimizers(solvers: tuple[str, ...], mode: str) -> None:
    if not solvers:
        st.info(
            "Pick at least one solver in the sidebar to see its update "
            "rule and intuition."
        )
        return

    if mode != "surface":
        st.caption(
            "Returns mode (GARCH MLE) uses a scalar negative-log-likelihood. "
            "L-BFGS-B is the default. NM is a derivative-free sanity check."
        )

    for idx, solver_key in enumerate(solvers):
        dive = _SOLVER_DIVES.get(solver_key)
        with st.container(border=True):
            icon = SOLVER_ICONS.get(solver_key, "🛠️")
            long_name = dive.long_name if dive else solver_key
            st.markdown(
                section_header_html(icon, f"{solver_key} · {long_name}"),
                unsafe_allow_html=True,
            )
            badge = SOLVER_BADGES.get(solver_key)
            if badge:
                st.markdown(
                    f"<span style='display:inline-block;padding:2px 8px;"
                    f"border-radius:9999px;background:#ecfeff;color:#0e7490;"
                    f"border:1px solid #67e8f9;font-size:0.78rem;"
                    f"font-weight:600'>{badge}</span>",
                    unsafe_allow_html=True,
                )
            if dive is not None:
                st.markdown(f"**Intuition.** {dive.intuition_md}")
                if dive.update_rule_latex:
                    st.markdown("**Update rule**")
                    st.latex(dive.update_rule_latex)
                else:
                    st.caption(
                        "No closed-form update rule. The simplex moves "
                        "geometrically (see 📚 Theory for the four-move "
                        "sequence)."
                    )
                st.caption(
                    "Full mechanics and references live in the 📚 Theory tab."
                )
            else:
                st.caption(
                    f"No pedagogical material registered for solver "
                    f"`{solver_key}`."
                )
        if idx < len(solvers) - 1:
            st.write("")


def _tab_objectives(
    objectives: tuple[str, ...], surface_candidates: tuple[str, ...]
) -> None:
    # Surface candidates that aren't iv_gbm consume an objective.
    real_surface = tuple(c for c in surface_candidates if c != "iv_gbm")
    if not real_surface:
        st.caption(
            "Selected candidates use a closed-form IV inversion "
            "(`iv_gbm`). There is no iterative loss to choose."
        )
        return

    if not objectives:
        st.info(
            "Pick at least one objective in the sidebar to see its residual "
            "and scalar-loss formulas."
        )
        st.markdown("**Generic surface objective (template)**")
        st.latex(GARCH_LOSS_LATEX)  # reused as a generic ½ ‖r‖² template
        st.caption(
            r"Each candidate exposes the residual $r_i$ defined by the "
            r"chosen objective. The scalar loss is the RMSE of $r$. "
            r"Huber uses its own aggregate."
        )
        return

    for idx, obj_key in enumerate(objectives):
        with st.container(border=True):
            icon = OBJECTIVE_ICONS.get(obj_key, "🎯")
            name = OBJECTIVE_DISPLAY_NAMES.get(obj_key, obj_key)
            st.markdown(
                section_header_html(icon, name), unsafe_allow_html=True
            )
            badge = OBJECTIVE_BADGES.get(obj_key)
            if badge:
                st.markdown(
                    f"<span style='display:inline-block;padding:2px 8px;"
                    f"border-radius:9999px;background:#fef3c7;color:#92400e;"
                    f"border:1px solid #fbbf24;font-size:0.78rem;"
                    f"font-weight:600'>{badge}</span>",
                    unsafe_allow_html=True,
                )
            description = OBJECTIVE_DESCRIPTIONS.get(obj_key)
            if description:
                st.markdown(description)
            residual = OBJECTIVE_RESIDUAL_LATEX.get(obj_key)
            if residual:
                st.markdown("**Per-quote residual**")
                st.latex(residual)
            aggregate = OBJECTIVE_AGGREGATE_LATEX.get(obj_key)
            if aggregate:
                st.markdown("**Scalar loss**")
                st.latex(aggregate)
        if idx < len(objectives) - 1:
            st.write("")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


# Surface-side GARCH variants whose sliders share the P-family scale under
# LRNVR. ``heston_nandi`` is deliberately excluded — its γ is O(100) and α
# is O(1e-6), so it keeps its own bespoke slider/bounds box.
_LRNVR_INVARIANT_Q_MODELS: tuple[str, ...] = ("garch_q", "ngarch_q", "gjr_q")


def _render_garch_scale_note(
    generator: str | None, candidates: tuple[str, ...]
) -> None:
    """Show the GARCH per-period scale convention when GARCH-family models
    (P or Q) are in scope, so the user understands why the sliders for
    ω/α/β look the same across `garch ↔ garch_q`, etc."""
    in_scope = {generator, *candidates}
    has_p_garch = bool(in_scope & set(GARCH_FAMILY))
    has_q_garch = bool(in_scope & set(_LRNVR_INVARIANT_Q_MODELS))
    if not (has_p_garch or has_q_garch):
        return
    st.info(
        "**GARCH scale convention** — sliders show **ω at the per-period "
        "(daily) scale**; α, β, γ are dimensionless. Under Duan's LRNVR "
        "the one-step-ahead conditional variance is invariant under "
        "ℙ → ℚ, so the ω/α/β sliders for `garch ↔ garch_q`, "
        "`ngarch ↔ ngarch_q`, `gjr_garch ↔ gjr_q` share the same range. "
        "Only γ is measure-specific (γ\\* = γ + λ). Heston-Nandi is a "
        "separate case — its γ is 𝒪(100) and its α is 𝒪(10⁻⁶).",
        icon="ℹ️",
    )


def _render_model_header(model_key: str, *, suffix: str = "") -> None:
    icon = MODEL_ICONS.get(model_key, "📐")
    name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
    badge_html = (
        p_measure_badge_html()
        if model_key in GARCH_FAMILY
        else q_measure_badge_html()
    )
    header_html = section_header_html(icon, f"{name}{suffix}")
    st.markdown(
        f"{header_html}&nbsp;{badge_html}",
        unsafe_allow_html=True,
    )


def _render_active_triangles_caption(
    candidates: tuple[str, ...],
    solvers: tuple[str, ...],
    objectives: tuple[str, ...],
    show_objectives_tab: bool,
) -> None:
    n_c = max(len(candidates), 1)
    n_s = max(len(solvers), 1)
    if show_objectives_tab:
        n_o = max(len(objectives), 1)
    else:
        n_o = 1  # GARCH MLE / iv_gbm: a single implicit objective
    n_runs = n_c * n_s * n_o

    def _pluralise(n: int, word: str) -> str:
        return f"{n} {word}{'s' if n != 1 else ''}"

    parts = [
        _pluralise(len(candidates), "candidate"),
        _pluralise(len(solvers), "solver"),
    ]
    if show_objectives_tab:
        parts.append(_pluralise(len(objectives), "objective"))
    triangle = " × ".join(parts)
    suffix = "s" if n_runs != 1 else ""
    st.caption(
        rf"**Active triangles**. {triangle} = **{n_runs} run{suffix}**. "
        r"Each one minimises the chosen objective $L(\theta)$ over the "
        r"candidate's parameters using the chosen solver."
    )


__all__ = ["render"]
