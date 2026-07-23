"""Solver picker + advanced settings panel (multi-model aware)."""

from __future__ import annotations

from typing import Any

import streamlit as st

from config.constants import (
    GARCH_FAMILY,
    MODEL_DISPLAY_NAMES,
    RN_GARCH_SURFACE_MODELS,
    SOLVER_GROUP_HELP,
    SOLVER_HOVER,
    SOLVER_ICONS,
)
from config.model_registry import default_solver, supported_solvers
from services import state_manager

from ._option_help import compose_help


def _format_solver(name: str) -> str:
    return f"{SOLVER_ICONS.get(name, '·')} {name}"


def _default_selection(
    candidates: tuple[str, ...], inter: tuple[str, ...]
) -> list[str]:
    """Recommended default solver for the candidates, restricted to the shared
    intersection. Prefers the first candidate's ``default_solver`` (so the trio
    defaults to its measured NM / L-BFGS-B rather than the starved DE that would
    win as ``inter[0]``); falls back to the first shared solver."""
    for m in candidates:
        d = default_solver(m)
        if d in inter:
            return [d]
    return [inter[0]]


def _zero_solver_candidates(candidates: tuple[str, ...]) -> tuple[str, ...]:
    """Candidates with no iterative solver (closed-form models like iv_gbm)."""
    return tuple(m for m in candidates if not supported_solvers(m))


def _is_single_seed_only(candidates: tuple[str, ...]) -> bool:
    """True when the whole selection runs a single seeded start (no multi-start).

    GARCH MLE, the nonaffine GARCH-Q surface trio, and the custom model all lack
    a restart loop, so the multi-start slider must be disabled — leaving it
    enabled promised ``n_restarts × max_nfev`` work that never ran."""
    return bool(candidates) and all(
        m in GARCH_FAMILY or m in RN_GARCH_SURFACE_MODELS or m == "custom"
        for m in candidates
    )


def _intersection(candidates: tuple[str, ...]) -> tuple[str, ...]:
    """Solvers supported by every **solver-bearing** candidate model.

    Closed-form models (iv_gbm has no iterative solver) are dropped before
    intersecting — otherwise adding iv_gbm to the pills emptied the whole set
    and dead-ended Run for a valid heston candidate. When every candidate is
    closed-form the result is empty (no solver to run)."""
    solver_cands = tuple(m for m in candidates if supported_solvers(m))
    if not solver_cands:
        return ()
    sets = [set(supported_solvers(m)) for m in solver_cands]
    inter = sets[0]
    for s in sets[1:]:
        inter &= s
    # Preserve a stable ordering: follow supported_solvers of the first
    # solver-bearing candidate.
    first_order = supported_solvers(solver_cands[0])
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
    closed_form = _zero_solver_candidates(candidates)
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
    if closed_form:
        # Solver-bearing candidates still get a solver; note the closed-form
        # ones that won't be iteratively calibrated instead of silently
        # letting them empty the intersection.
        st.caption(
            f"Closed-form (no solver): {', '.join(closed_form)} — calibrated by "
            "direct inversion, not by the solvers below."
        )

    current = state_manager.get("calib_solver_selection") or ("LM-JAX",)
    panel_default = _default_selection(candidates, inter)
    valid_current = [s for s in current if s in inter] or panel_default

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
        selection = panel_default
        st.warning(f"No solver selected → defaulted to {panel_default[0]}.")

    # Inform the user about pairs that won't run because the intersection
    # excludes them. They aren't an error — just a transparency note.
    skipped = _union_unsupported_pairs(candidates, inter)
    if skipped:
        msgs = ", ".join(f"{MODEL_DISPLAY_NAMES[m]} × {s}" for m, s in skipped)
        st.caption(f"ℹ️  Solvers not shared across all candidates are hidden: {msgs}.")

    settings = dict(state_manager.get("calib_solver_settings") or {})
    # Single-seed family: GARCH MLE runs one L-BFGS-B fit, and the nonaffine
    # GARCH-Q surface trio runs a single seeded start (measured: random restarts
    # only waste the interactive budget in worse basins). Both disable the
    # multi-start slider.
    is_single_seed_only = _is_single_seed_only(candidates)

    settings["max_nfev"] = st.slider(
        "Max function evaluations",
        50,
        5000,
        int(settings.get("max_nfev", 200)),
        step=50,
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
        if is_single_seed_only:
            n_restarts_help = (
                "Multi-start is disabled for this selection — GARCH MLE runs a "
                "single L-BFGS-B fit, the nonaffine GARCH-Q surface models run a "
                "single seeded start, and the custom model has no restart loop. "
                "Switch to a built-in surface model (Heston / Bates / Merton) to "
                "use restarts."
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
            1,
            10,
            int(settings.get("n_restarts", 3)),
            step=1,
            disabled=is_single_seed_only,
            help=n_restarts_help,
            key="solver_n_restarts",
        )
        if "DE" in selection:
            settings["de_seed"] = int(
                st.number_input(
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
                )
            )
    state_manager.update(
        calib_solver_settings=settings,
        calib_solver_selection=tuple(selection),
    )
    return tuple(selection), settings
