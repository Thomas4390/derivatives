"""Shared helpers used by multiple tab renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import streamlit as st

from charts.comparison import _iter_flat, _label
from config.constants import (
    MODEL_DISPLAY_NAMES,
    MODEL_ICONS,
    OBJECTIVE_DISPLAY_NAMES,
    OBJECTIVE_ICONS,
)
from services import state_manager


def _format_model(key: str) -> str:
    icon = MODEL_ICONS.get(key, "•")
    name = MODEL_DISPLAY_NAMES.get(key, key)
    return f"{icon} {name}"


def _format_objective(key: str) -> str:
    icon = OBJECTIVE_ICONS.get(key, "•")
    name = OBJECTIVE_DISPLAY_NAMES.get(key, key)
    return f"{icon} {name}"


def _is_nested_objectives(value) -> bool:
    """Return ``True`` when value is the new ``dict[objective] -> Summary``
    layer rather than a single :class:`CalibrationRunSummary`.

    Used to migrate tabs that previously consumed a flat
    ``dict[solver, Summary]`` to the new
    ``dict[solver, dict[objective, Summary]]`` schema while gracefully
    handling the legacy shape.
    """
    if not isinstance(value, dict):
        return False
    # CalibrationRunSummary has a ``solver_name`` attribute; nested
    # objective dicts contain those instead as their leaf values.
    sample = next(iter(value.values()), None)
    return sample is None or hasattr(sample, "solver_name")


# ──────────────────────────────────────────────────────────────────────
# Multi-series overlay model
# ──────────────────────────────────────────────────────────────────────
# A "series" is one ``(model, solver, objective)`` combination — the unit
# the user overlays on a chart. ``enumerate_series`` flattens the nested
# ``calib_results`` dict into these; ``series_view_filter`` is the multi-pill
# picker that lets every tab choose which combinations to superimpose
# (default = all), with a soft cap so dense overlays stay legible. Pure
# helpers are kept separate from the Streamlit wrappers so the selection
# logic is unit-testable without a UI runtime.

_SERIES_CAP_DEFAULT = 8


@dataclass(frozen=True)
class Series:
    """One overlaid ``(model, solver, objective)`` combination."""

    model: str
    solver: str
    objective: str
    summary: Any

    @property
    def key(self) -> str:
        """Stable identity used as the ``st.pills`` option value (so the
        selection persists across reruns) — ``"model|solver|objective"``."""
        return f"{self.model}|{self.solver}|{self.objective}"

    @property
    def label(self) -> str:
        """Human-readable composite legend label, identical to the one the
        Compare-tab charts emit (``charts.comparison._label``)."""
        return _label(self.model, self.solver, self.objective)


def enumerate_series(results: dict) -> list[Series]:
    """Flatten ``dict[model][solver][objective] -> Summary`` (or the legacy
    ``dict[model][solver] -> Summary``) into a list of :class:`Series`.

    Reuses ``charts.comparison._iter_flat`` so the traversal stays the
    single source of truth across both schemas.
    """
    if not results:
        return []
    return [
        Series(model, solver, objective, summary)
        for model, solver, objective, summary in _iter_flat(results)
    ]


def subdict_from_series(series: Iterable[Series]) -> dict:
    """Rebuild a ``dict[model][solver][objective] -> Summary`` from a list
    of :class:`Series`.

    Lets a tab feed a *pruned* selection straight into the existing
    ``charts.comparison.render_multi_*`` builders, which consume that
    nested shape — so overlaying "only these combinations" reuses the
    battle-tested Compare-tab rendering instead of duplicating it.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for s in series:
        out.setdefault(s.model, {}).setdefault(s.solver, {})[s.objective] = s.summary
    return out


def successful_series(series: Iterable[Series]) -> list[Series]:
    """Keep only series whose run produced a result — failed/skipped runs
    can't be overlaid and would otherwise clutter the picker."""
    return [s for s in series if s.summary is not None and s.summary.result is not None]


def filter_series(series: Iterable[Series], chosen_keys: Iterable[str]) -> list[Series]:
    """Return the series whose :attr:`Series.key` is in ``chosen_keys``,
    preserving enumeration order. Unknown keys are ignored."""
    chosen = set(chosen_keys)
    return [s for s in series if s.key in chosen]


def termination_reason(summary) -> str:
    """Human-readable reason the fit stopped, for one run.

    Surface calibrators store the optimiser's termination message per restart
    in ``diagnostics['lm_runs'][best]['message']`` (e.g. "``ftol`` termination
    condition is satisfied." or "The maximum number of function evaluations is
    exceeded."); the best restart is ``diagnostics['best_start_index']``. GARCH
    MLE doesn't record a message, so fall back to the ``success`` flag.
    """
    if summary is None or summary.result is None:
        return f"failed: {getattr(summary, 'error', '') or 'unknown'}"
    res = summary.result
    diag = res.diagnostics or {}
    if diag.get("stopped"):
        n = diag.get("n_evaluations", res.n_iterations)
        return f"stopped by user — best of {n} evaluations"
    runs = diag.get("lm_runs") or diag.get("lm_runs_joint") or []
    if runs:
        best_idx = diag.get("best_start_index")
        chosen = None
        if best_idx is not None:
            chosen = next((r for r in runs if r.get("start") == best_idx), None)
        chosen = chosen or runs[0]
        msg = chosen.get("message")
        if msg:
            return str(msg)
    return "converged" if res.success else "did not converge (max evaluations reached?)"


def series_view_filter(
    results: dict,
    *,
    key: str,
    cap: int = _SERIES_CAP_DEFAULT,
    label: str = "Show on chart",
    only_successful: bool = True,
) -> list[Series]:
    """Multi-pill picker over ``(model, solver, objective)`` combinations.

    Defaults to **all runnable series selected** (the user prunes via the
    pills) and emits a soft-cap ``st.warning`` past ``cap`` series so dense
    overlays stay legible — nothing is blocked, the user decides. Selection
    persists across reruns via ``key=series_view_<key>``.
    """
    series = enumerate_series(results)
    if only_successful:
        series = successful_series(series)
    if not series:
        return []
    label_by_key = {s.key: s.label for s in series}
    options = list(label_by_key)
    chosen = st.pills(  # pyright: ignore[reportCallIssue]
        label,
        options=options,
        default=options,
        selection_mode="multi",
        format_func=lambda k: label_by_key.get(k, k),
        key=f"series_view_{key}",
    )
    selected = filter_series(series, list(chosen) if chosen else [])
    if len(selected) > cap:
        st.warning(
            f"Overlaying {len(selected)} series — charts get hard to read "
            f"past {cap}. Deselect some above to declutter.",
            icon="⚠️",
        )
    return selected


def facet_grid(
    series: list[Series],
    render_one: Callable[[Series], Any],
    *,
    max_cols: int = 2,
    key_prefix: str = "facet",
) -> None:
    """Lay selected series out as **small multiples**.

    For charts that cannot be physically superimposed (heatmaps,
    squared-residual ACF bars), render one mini-figure per series in a
    responsive ``st.columns`` grid, each titled with the series' composite
    label. ``render_one(series)`` returns a Plotly figure (or ``None`` to
    skip that cell).

    Each chart gets an explicit ``key=f"{key_prefix}_{series.key}"`` so two
    structurally-identical mini-figures (or a figure that also appears on
    another tab) can't collide on Streamlit's auto-generated element id.
    """
    if not series:
        return
    n_cols = max(1, min(max_cols, len(series)))
    for row_start in range(0, len(series), n_cols):
        cols = st.columns(n_cols)
        for col, s in zip(cols, series[row_start : row_start + n_cols]):
            with col:
                st.caption(s.label)
                fig = render_one(s)
                if fig is not None:
                    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_{s.key}")


def active_model_picker(key_suffix: str = "") -> str | None:
    """Segmented control selecting the model the inspection tab focuses on.

    Returns ``None`` when no calibration has run yet. When only one
    candidate has results the picker is suppressed but the state is
    still synced so downstream helpers don't see a stale value.
    """
    nested = state_manager.get("calib_results") or {}
    models = list(nested.keys())
    if not models:
        return None
    active = state_manager.get("calib_active_model") or models[0]
    if active not in models:
        active = models[0]
    if len(models) == 1:
        if state_manager.get("calib_active_model") != active:
            state_manager.update(calib_active_model=active)
        return active
    pick = st.segmented_control(
        "Inspect model",
        options=models,
        default=active,
        selection_mode="single",
        format_func=_format_model,
        key=f"active_model_picker_{key_suffix}",
    )
    if pick is None:
        pick = active
    if pick != active:
        # When the user switches model, reset the active solver/objective
        # so downstream pickers don't keep stale state.
        new_solvers = list(nested[pick].keys())
        first_solver = new_solvers[0] if new_solvers else None
        first_objective: str | None = None
        if first_solver is not None:
            slot = nested[pick][first_solver]
            if _is_nested_objectives(slot) and slot:
                first_objective = next(iter(slot))
        state_manager.update(
            calib_active_model=pick,
            calib_active_solver=first_solver,
            calib_active_objective=first_objective,
        )
    return pick


def active_objective_picker(key_suffix: str = "") -> str | None:
    """Segmented control selecting the objective within (model, solver).

    Returns ``None`` when results don't expose multiple objectives
    (e.g. GARCH MLE or single-objective runs). Suppresses the widget
    when only one objective is available but still syncs the state.
    """
    nested = state_manager.get("calib_results") or {}
    if not nested:
        return None
    active_model = state_manager.get("calib_active_model")
    active_solver = state_manager.get("calib_active_solver")
    if not active_model or not active_solver:
        return None
    slot = nested.get(active_model, {}).get(active_solver)
    if not _is_nested_objectives(slot) or not slot:
        return None
    objectives = list(slot.keys())
    active = state_manager.get("calib_active_objective") or objectives[0]
    if active not in objectives:
        active = objectives[0]
    if len(objectives) == 1:
        if state_manager.get("calib_active_objective") != active:
            state_manager.update(calib_active_objective=active)
        return active
    pick = st.segmented_control(
        "Inspect objective",
        options=objectives,
        default=active,
        selection_mode="single",
        format_func=_format_objective,
        key=f"active_objective_picker_{key_suffix}",
    )
    if pick is None:
        pick = active
    if pick != active:
        state_manager.update(calib_active_objective=pick)
    return pick


def active_objectives_picker(key_suffix: str = "") -> list[str]:
    """Multi-select objectives to overlay (Loss Landscape tab).

    Returns the chosen objectives with the **primary** one
    (``calib_active_objective``) first, so callers can use ``[0]`` for the
    anchor / base-params / labels. Returns ``[]`` when results don't expose
    objectives (GARCH MLE / single-objective legacy runs) and a single-item
    list (no extra widget) when only one objective is available — i.e. the
    behaviour is identical to :func:`active_objective_picker` in that case.
    """
    nested = state_manager.get("calib_results") or {}
    if not nested:
        return []
    active_model = state_manager.get("calib_active_model")
    active_solver = state_manager.get("calib_active_solver")
    if not active_model or not active_solver:
        return []
    slot = nested.get(active_model, {}).get(active_solver)
    if not _is_nested_objectives(slot) or not slot:
        return []
    objectives = list(slot.keys())
    active = state_manager.get("calib_active_objective") or objectives[0]
    if active not in objectives:
        active = objectives[0]
    if len(objectives) == 1:
        if state_manager.get("calib_active_objective") != active:
            state_manager.update(calib_active_objective=active)
        return [active]
    chosen = st.pills(
        "Overlay loss landscapes",
        options=objectives,
        default=[active],
        selection_mode="multi",
        format_func=_format_objective,
        key=f"landscape_objectives_{key_suffix}",
    )
    chosen = list(chosen) if chosen else [active]
    # Promote the first chosen objective to primary when the active one was
    # deselected, so the anchor / labels stay consistent with what's shown.
    if active not in chosen:
        active = chosen[0]
        state_manager.update(calib_active_objective=active)
    return [active] + [o for o in chosen if o != active]
