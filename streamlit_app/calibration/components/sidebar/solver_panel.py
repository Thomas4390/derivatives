"""Solver picker + advanced settings panel (multi-model aware)."""

from __future__ import annotations

from typing import Any

import streamlit as st

from config.constants import (
    GARCH_FAMILY,
    MODEL_DISPLAY_NAMES,
    SOLVER_GROUP_HELP,
    SOLVER_HOVER,
    SOLVER_ICONS,
)
from config.model_registry import supported_solvers
from services import state_manager

from ._option_help import compose_help


def _format_solver(name: str) -> str:
    return f"{SOLVER_ICONS.get(name, '·')} {name}"


def _intersection(candidates: tuple[str, ...]) -> tuple[str, ...]:
    """Solvers supported by **all** candidate models — only these can run
    on every model in one click."""
    if not candidates:
        return ()
    sets = [set(supported_solvers(m)) for m in candidates]
    inter = sets[0]
    for s in sets[1:]:
        inter &= s
    # Preserve a stable ordering: follow the order of supported_solvers
    # for the first candidate, then anything else (shouldn't happen).
    first_order = supported_solvers(candidates[0])
    return tuple(s for s in first_order if s in inter)


def _union_unsupported_pairs(
    candidates: tuple[str, ...], inter: tuple[str, ...]
) -> list[tuple[str, str]]:
    """List ``(model, solver)`` pairs that would be silently skipped."""
    inter_set = set(inter)
    pairs: list[tuple[str, str]] = []
    for m in candidates:
        for s in supported_solvers(m):
            if s not in inter_set:
                pairs.append((m, s))
    return pairs


def render(candidates: tuple[str, ...]) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Render the solver multi-select.

    ``candidates`` is the tuple of model keys chosen in the sidebar. The
    multiselect surfaces the **intersection** of supported solvers (so a
    single run can apply every selected solver to every model), and
    warns when a solver supported by some but not all candidates is
    being skipped.
    """
    inter = _intersection(candidates)
    st.subheader("🚀 Solvers")
    if not inter:
        if not candidates:
            st.info("Pick at least one candidate model to enable solver selection.")
        else:
            st.info(
                "iv_gbm uses closed-form inversion — no iterative solver. "
                "Pick another candidate to enable a solver."
            )
        return (), {}

    current = state_manager.get("calib_solver_selection") or ("LM-JAX",)
    valid_current = [s for s in current if s in inter] or [inter[0]]

    picked = st.pills(
        "Active solvers",
        options=list(inter),
        default=valid_current,
        selection_mode="multi",
        format_func=_format_solver,
        key="solver_pick_multi",
        help=compose_help(SOLVER_GROUP_HELP, inter, SOLVER_HOVER, _format_solver),
    )
    selection = list(picked) if picked else []
    if not selection:
        selection = [inter[0]]
        st.warning(f"No solver selected → defaulted to {inter[0]}.")

    # Inform the user about pairs that won't run because the intersection
    # excludes them. They aren't an error — just a transparency note.
    skipped = _union_unsupported_pairs(candidates, inter)
    if skipped:
        msgs = ", ".join(
            f"{MODEL_DISPLAY_NAMES[m]} × {s}" for m, s in skipped
        )
        st.caption(
            f"ℹ️  Solvers not shared across all candidates are hidden: {msgs}."
        )

    settings = dict(state_manager.get("calib_solver_settings") or {})
    is_garch_only = all(m in GARCH_FAMILY for m in candidates)

    settings["max_nfev"] = st.slider(
        "Max function evaluations", 50, 5000,
        int(settings.get("max_nfev", 200)), step=50,
        help=(
            "Per-restart evaluation budget, honoured by every solver. LM "
            "converges in ~50–200; DE / NM are derivative-free and need "
            "1000+ to actually explore. DE rounds the budget to whole "
            "generations and stops at the cap (overshoot ≤ one generation)."
        ),
        key="solver_max_nfev",
    )
    if "DE" in selection and settings["max_nfev"] < 1000:
        st.warning(
            "**DE budget is low** — the global solver typically needs ≥ 1000 "
            "evaluations to actually explore the parameter space. Increase "
            "`Max function evaluations` above before running."
        )

    with st.expander("⚙️ Advanced settings", expanded=False):
        if is_garch_only:
            n_restarts_help = (
                "Multi-start is not applicable to GARCH MLE — each "
                "calibrator runs a single L-BFGS-B fit. Switch family "
                "to use restarts."
            )
        else:
            n_restarts_help = (
                "Number of independent restarts run by each surface "
                "calibrator. More restarts → better odds of finding the "
                "global minimum at the cost of `n_restarts × max_nfev` "
                "evaluations per (model, solver) pair."
            )
        settings["n_restarts"] = st.slider(
            "Multi-start restarts",
            1, 10,
            int(settings.get("n_restarts", 3)), step=1,
            disabled=is_garch_only,
            help=n_restarts_help,
            key="solver_n_restarts",
        )
        if "DE" in selection:
            settings["de_seed"] = int(st.number_input(
                "Seed (DE)",
                min_value=0,
                max_value=2**32 - 1,
                value=int(settings.get("de_seed", 42)),
                step=1,
                help=(
                    "Random seed forwarded to Differential Evolution. "
                    "DE is the only stochastic solver wired here — changing "
                    "the seed produces a different exploration trajectory "
                    "(and usually a slightly different final fit). Hold it "
                    "fixed to make a run exactly reproducible; vary it to "
                    "study how much of the result depends on luck."
                ),
                key="solver_de_seed",
            ))
    state_manager.update(
        calib_solver_settings=settings,
        calib_solver_selection=tuple(selection),
    )
    return tuple(selection), settings
