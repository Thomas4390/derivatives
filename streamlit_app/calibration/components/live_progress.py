"""
Streamlit-side rendering of the live calibration runner
========================================================

Consumes :mod:`services.live_runner` (which is Streamlit-free) and
binds it to ``st.empty`` placeholders + Plotly charts. This is the
ONLY module that wires live progress to the Streamlit canvas.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from charts.live_convergence import (
    _filter_history_for_display,
    _latest_restart_segment,
)
from config.constants import (
    LIVE_POLL_INTERVAL_SEC,
    MODEL_DISPLAY_NAMES,
    MODEL_ICONS,
)
from config.model_registry import supported_solvers
from services import state_manager
from services.calibration_service import CalibrationRunSummary, partial_from_history
from services.live_runner import (
    LiveRunHandle,
    drain_handle,
    iter_snapshots,
    start_run,
)
from tabs._helpers import fmt_param_value
from utils.plotly_theme import COLORS, FONT_FAMILY, MONO_FAMILY

# Streamlit raises one of these (subclasses of ``BaseException``) to tear the
# script down when a widget callback requests a rerun — e.g. the ⏹ Stop button.
# Matched by class name so the catch survives Streamlit's periodic moves of the
# exception module path across versions.
_STREAMLIT_CONTROL_EXC: frozenset[str] = frozenset({"RerunException", "StopException"})

# session_state key for the list of in-flight LiveRunHandles. The Stop
# button's on_click callback iterates this list and sets each handle's
# ``cancel_event`` so the worker threads exit cooperatively. Reset at
# the start of every run_multi_model_with_live_progress call.
_LIVE_HANDLES_KEY = "_calib_live_handles"

# Column label for the live loss curve. The live convergence is shown with a
# native st.line_chart re-rendered into an st.empty slot (Vega-Lite reconciles in
# place at the same position) rather than a re-rendered Plotly figure, so it
# updates without the per-frame DOM remount that made the chart flicker — and
# without the deprecated add_rows. ``log10`` keeps the multi-decade descent
# legible on the otherwise-linear native chart.
_LIVE_LOSS_COL = "loss  (log₁₀, best-so-far)"


def _handle_alive(handle) -> bool:
    """True when a run handle's worker thread is still running."""
    thread = getattr(handle, "thread", None)
    return thread is not None and thread.is_alive()


def _cancel_handles(handles) -> None:
    """Signal every handle's worker to exit cooperatively."""
    for handle in handles:
        handle.cancel_event.set()


def _request_cancel() -> None:
    """Streamlit on_click callback for the ⏹ Stop button.

    Flips every active handle's ``cancel_event`` so the workers raise
    :class:`CalibrationCancelled` on their next iteration snapshot, and
    records the cancellation on session_state so the next script run can
    surface a banner. Only records the cancellation when a worker is actually
    still alive — clicking the still-present ⏹ after a run finished used to
    raise a false 'calibration stopped' banner (and a phantom ⏸ badge) on the
    next rerun. Thread termination is asynchronous — at most one poll later.
    """
    handles = st.session_state.get(_LIVE_HANDLES_KEY, [])
    _cancel_handles(handles)
    if any(_handle_alive(h) for h in handles):
        st.session_state["calib_was_cancelled"] = True


# Tokens shared by every card in this module.
#
# The card background is intentionally NOT ``COLORS["plot"]`` (white)
# because Streamlit's own canvas is also white — pure-white cards on a
# white canvas would disappear, leaving only the 4-px accent stripe to
# delimit them. A very light slate (#f8fafc) gives just enough contrast
# for the card border to read while staying compatible with the
# plotly_white theme used by the inline charts.
_CARD_BG = "#f8fafc"  # light slate, distinguishable from canvas white
_CARD_BORDER = "rgba(15, 23, 42, 0.12)"  # darker than COLORS["grid"] for a visible edge
_LABEL_COLOR = COLORS["text_muted"]  # rgba(0,0,0,0.50)
_VALUE_COLOR = COLORS["text"]  # #1f2937
_SUB_COLOR = COLORS["text_dim"]  # rgba(0,0,0,0.70)


def _fmt_param_value(v: float) -> str:
    """Format a natural-scale parameter without collapsing tiny knobs to 0.

    Thin alias for the shared :func:`tabs._helpers.fmt_param_value` so the live
    card and the landscape hidden-parameter footer obey one rule.
    """
    return fmt_param_value(v)


def _format_params(d: dict, n: int | None = None) -> str:
    """Render a key-value list of natural-scale parameters.

    ``n=None`` shows every parameter — the previous default of 4 silently
    hid jump params on Bates (which has 8) and made the live card lie
    about which knobs were moving.
    """
    items = list(d.items()) if n is None else list(d.items())[:n]
    rows = [
        f'<span style="color:{_LABEL_COLOR}">{k}</span> = '
        f'<span style="color:{_VALUE_COLOR};font-weight:500">{_fmt_param_value(v)}</span>'
        for k, v in items
    ]
    return "<br>".join(rows)


def _metric_card(label: str, value: str, sub: str, accent: str) -> str:
    return f"""
    <div style="background:{_CARD_BG};border:1px solid {_CARD_BORDER};border-radius:12px;
                border-left:4px solid {accent};padding:0.85rem 1.05rem">
      <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;
                   color:{_LABEL_COLOR};font-weight:600;font-family:{FONT_FAMILY}">{label}</div>
      <div style="font-size:1.15rem;font-weight:600;color:{_VALUE_COLOR};margin-top:0.25rem;
                   font-family:{MONO_FAMILY}">{value}</div>
      <div style="font-size:0.72rem;color:{_SUB_COLOR};margin-top:0.25rem;
                   font-family:{FONT_FAMILY}">{sub}</div>
    </div>
    """


def _live_metrics_html(handle: LiveRunHandle) -> str:
    """Tiny HTML block summarising the current solver state."""
    if not handle.history:
        return f"""
        <div style="background:{_CARD_BG};border:1px solid {_CARD_BORDER};border-radius:12px;
                    border-left:4px solid {COLORS['primary']};padding:1rem 1.25rem;">
          <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;
                       color:{_LABEL_COLOR};font-weight:600;font-family:{FONT_FAMILY}">Solver</div>
          <div style="font-size:1.4rem;font-weight:600;color:{_VALUE_COLOR};margin-top:0.25rem;
                       font-family:{FONT_FAMILY}">
            {handle.solver_name}
          </div>
          <div style="font-size:0.78rem;color:{_SUB_COLOR};margin-top:0.3rem;
                       font-family:{FONT_FAMILY}">
            waiting for first iteration…
          </div>
        </div>
        """
    last = handle.history[-1]
    best_obj = min(s.objective for s in handle.history)
    # Per-restart view: count evaluations in the restart currently running
    # (so the figure exceeds neither the chart nor the per-restart max_nfev
    # budget). ``best loss`` stays the global minimum across all restarts.
    current = _filter_history_for_display(_latest_restart_segment(handle.history))
    iter_count = len(current)
    restart_k = sum(1 for s in handle.history if getattr(s, "iteration", 0) == 0)
    restart_tag = (
        f"restart {restart_k}/{handle.n_restarts} · " if handle.n_restarts > 1 else ""
    )
    elapsed = last.elapsed_seconds

    # ``auto-fit minmax(180px, 1fr)`` lets the cards reflow into 1, 2, 3
    # or 4 columns depending on the viewport — the previous fixed
    # 4-column grid crushed labels on tablets.
    return f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));
                gap:0.6rem;margin:0.4rem 0 0.8rem">
      {_metric_card("solver", handle.solver_name,
                    f"{restart_tag}{iter_count} evals · {elapsed:.2f}s", COLORS["primary"])}
      {_metric_card("current loss", f"{last.objective:.3e}",
                    f"iter #{last.iteration}", COLORS["accent"])}
      {_metric_card("best loss", f"{best_obj:.3e}",
                    "monotone minimum", COLORS["primary"])}
      {_metric_card("latest params", _format_params(last.params_natural),
                    "natural scale", COLORS["info"])}
    </div>
    """


def _solver_header_html(s_idx: int, solver_name: str) -> str:
    return f"""
    <div style='display:flex;align-items:center;justify-content:space-between;
                margin:1.0rem 0 0.6rem;padding-bottom:0.4rem;
                border-bottom:1px solid {_CARD_BORDER}'>
      <div style='font-family:{MONO_FAMILY};font-size:0.95rem;
                  font-weight:600;color:{_VALUE_COLOR}'>
        <span style='color:{COLORS["primary"]}'>{s_idx + 1:02d}</span>  ·  {solver_name}
      </div>
      <div style='display:inline-flex;align-items:center;gap:0.4rem;
                  padding:0.18rem 0.6rem;border-radius:999px;
                  background:rgba(217,119,6,0.18);color:{COLORS["accent"]};
                  font-size:0.7rem;font-weight:600;letter-spacing:0.08em;
                  text-transform:uppercase;font-family:{FONT_FAMILY}'>
        <span style='width:6px;height:6px;border-radius:50%;background:{COLORS["accent"]};
                      animation:pulse 1.4s ease-in-out infinite'></span>
        LIVE
      </div>
    </div>
    <style>@keyframes pulse {{ 0%,100% {{opacity:1}} 50% {{opacity:0.4}} }}</style>
    """


def _completion_html(
    solver_name: str, summary: CalibrationRunSummary, dropped: int
) -> str:
    rmse_str = (
        f"RMSE price = {summary.result.rmse_price:.3e}"
        if summary.result is not None and summary.result.rmse_price is not None
        else "MLE done"
    )
    dropped_str = (
        f" · {dropped} snapshots dropped (UI throttled)" if dropped > 0 else ""
    )
    return (
        f"<div style='margin:0.4rem 0 1.2rem;padding-left:0.4rem;"
        f"font-family:{MONO_FAMILY};font-size:0.8rem;color:#10b981'>"
        f"✓ {solver_name} done in {summary.elapsed:.2f}s · {rmse_str}{dropped_str}"
        f"</div>"
    )


def _model_header_html(model_key: str) -> str:
    icon = MODEL_ICONS.get(model_key, "·")
    name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
    return (
        f"<div style='display:flex;align-items:center;gap:0.6rem;"
        f"margin:0.8rem 0 0.2rem;padding:0.4rem 0.8rem;border-radius:8px;"
        f"background:rgba(13,148,136,0.12);border-left:3px solid {COLORS['primary']}'>"
        f"<span style='font-size:1.2rem'>{icon}</span>"
        f"<span style='font-family:{FONT_FAMILY};font-size:0.95rem;font-weight:600;"
        f"color:{COLORS['text']}'>{name}</span>"
        f"</div>"
    )


def _render_live_losses(
    chart_slot, losses: list[float], history, streamed: int, best: float
):
    """Re-render the live loss curve with the full best-so-far series.

    Accumulates ``log10(best-so-far)`` into ``losses`` and re-renders the **native**
    ``st.line_chart`` into ``chart_slot`` (an ``st.empty`` placeholder). Native
    Vega-Lite charts reconcile in place at the same slot position, so the curve
    updates smoothly without the per-frame remount a re-rendered Plotly figure
    caused — and without the (now-deprecated) ``add_rows``. Returns the updated
    streamed count + running best. ``log10`` keeps the multi-decade descent
    legible; the rare non-positive loss (GARCH NLL) falls back to its raw value.
    """
    for snap in history[streamed:]:
        if snap.objective < best:
            best = snap.objective
        losses.append(math.log10(best) if best > 0 else float(best))
    if losses:
        chart_slot.line_chart(pd.DataFrame({_LIVE_LOSS_COL: losses}), height=280)
    return len(history), best


def run_multi_model_with_live_progress(
    *,
    candidate_models: tuple[str, ...],
    market_data: Any,
    true_params_per_model: dict[str, dict[str, float]],
    solver_names: tuple[str, ...],
    objective_names: tuple[str, ...] = ("price_mse",),
    objective_settings: dict[str, Any] | None = None,
    constraint_settings: dict[str, Any] | None = None,
    search_bounds_per_model: dict[str, dict[str, tuple[float, float]]] | None = None,
    n_restarts: int = 5,
    max_nfev: int = 200,
    de_seed: int = 42,
    poll_interval: float = LIVE_POLL_INTERVAL_SEC,
) -> dict[str, dict[str, dict[str, CalibrationRunSummary]]]:
    """Run every ``(model, solver, objective)`` triple sequentially with live progress.

    Solvers not supported by a given model are recorded as an
    ``unsupported`` :class:`CalibrationRunSummary` for each objective
    rather than crashing, so the UI can render a "skipped" row downstream.

    Returns a triple-nested mapping ``results[model][solver][objective]``.
    """
    if not solver_names or not candidate_models or not objective_names:
        return {}

    n_runs = sum(
        1
        for m in candidate_models
        for s in solver_names
        if s in set(supported_solvers(m))
        for _ in objective_names
    )
    overall_status = st.status(
        f"Calibrating {len(candidate_models)} model(s) × "
        f"{len(solver_names)} solver(s) × {len(objective_names)} objective(s) "
        f"= {n_runs} runs…",
        expanded=True,
        state="running",
    )
    outer = overall_status.container()

    # Fresh handle registry per run — the Stop button's on_click callback
    # iterates this list to cancel in-flight workers. Cancel any workers still
    # alive from a previous run BEFORE dropping the registry: a non-Stop rerun
    # (sidebar tweak / tab click) leaves the old worker running, and resetting
    # the list here would make it permanently uncancellable — spamming Run then
    # stacks background calibrations with unbounded snapshot queues.
    _cancel_handles(st.session_state.get(_LIVE_HANDLES_KEY, []))
    st.session_state[_LIVE_HANDLES_KEY] = []
    outer.button(
        "⏹  Stop calibration",
        on_click=_request_cancel,
        key="cancel_calib_btn",
        type="secondary",
        help=(
            "Signals every running solver to exit at its next iteration. "
            "Cancellation is cooperative — DE / NM may take up to one "
            "poll interval (~50 ms) to actually stop."
        ),
    )

    results: dict[str, dict[str, dict[str, CalibrationRunSummary]]] = {}

    for model_key in candidate_models:
        outer.markdown(_model_header_html(model_key), unsafe_allow_html=True)
        truth = true_params_per_model.get(model_key, {})
        model_bounds = (search_bounds_per_model or {}).get(model_key)
        model_supported = set(supported_solvers(model_key))
        per_solver: dict[str, dict[str, CalibrationRunSummary]] = {}

        # Each model gets its own metric + chart slots so a long-running
        # second model doesn't blank out the first one's final state.
        metrics_slot = outer.empty()
        chart_slot = outer.empty()

        for s_idx, solver_name in enumerate(solver_names):
            if solver_name not in model_supported:
                outer.caption(
                    f"⊘  {solver_name} not supported for "
                    f"{MODEL_DISPLAY_NAMES.get(model_key, model_key)} — skipped."
                )
                per_solver[solver_name] = {
                    objective_name: CalibrationRunSummary(
                        solver_name=solver_name,
                        result=None,
                        elapsed=-1.0,
                        estimated_params={},
                        true_params=dict(truth),
                        relative_recovery_error={},
                        objective_name=objective_name,
                        error=(
                            f"solver '{solver_name}' is not supported for "
                            f"model '{model_key}'"
                        ),
                    )
                    for objective_name in objective_names
                }
                continue

            per_objective: dict[str, CalibrationRunSummary] = {}
            for objective_name in objective_names:
                outer.markdown(
                    _solver_header_html(s_idx, solver_name),
                    unsafe_allow_html=True,
                )
                handle = start_run(
                    model_key=model_key,
                    market_data=market_data,
                    solver_name=solver_name,
                    true_params=truth,
                    objective_name=objective_name,
                    objective_settings=objective_settings,
                    constraint_settings=constraint_settings,
                    search_bounds=model_bounds,
                    n_restarts=n_restarts,
                    max_nfev=max_nfev,
                    de_seed=de_seed,
                )
                st.session_state[_LIVE_HANDLES_KEY].append(handle)
                # ``in_flight`` tracks the running handle until the worker
                # terminates on its own. If the user presses ⏹ Stop, Streamlit
                # unwinds this synchronous loop via a Rerun/Stop exception at the
                # next ``st.*`` call; the except below catches it, keeps the
                # interrupted run's best-so-far point, and persists every
                # already-completed run before re-raising so the rerun doesn't
                # discard them. (Cleared to ``None`` once the worker finishes so a
                # later interruption can't overwrite a completed result.)
                in_flight: LiveRunHandle | None = handle
                # The live loss curve is a native st.line_chart re-rendered with
                # the full best-so-far series into the same st.empty slot each
                # tick. Native Vega-Lite charts reconcile in place, so the curve
                # updates smoothly — unlike the old per-frame Plotly redraw under a
                # unique key, which remounted the chart each tick (the flicker).
                losses: list[float] = []
                streamed = 0
                best = float("inf")
                try:
                    for _n_evals in iter_snapshots(handle, poll_interval=poll_interval):
                        metrics_slot.markdown(
                            _live_metrics_html(handle),
                            unsafe_allow_html=True,
                        )
                        streamed, best = _render_live_losses(
                            chart_slot,
                            losses,
                            handle.history,
                            streamed,
                            best,
                        )
                    # Worker terminated on its own (success or backend error) — no
                    # longer in flight.
                    in_flight = None

                    if handle.error is not None:
                        outer.error(
                            f"Solver {solver_name} ({objective_name}) failed: "
                            f"{handle.error}"
                        )
                        per_objective[objective_name] = CalibrationRunSummary.failure(
                            solver_name,
                            truth,
                            handle.error,
                            objective_name=objective_name,
                        )
                    else:
                        summary = handle.result
                        assert summary is not None
                        per_objective[objective_name] = summary
                        metrics_slot.markdown(
                            _live_metrics_html(handle),
                            unsafe_allow_html=True,
                        )
                        # Final render with any points that arrived after the last
                        # tick (the full series is re-rendered, so this is complete).
                        _render_live_losses(
                            chart_slot,
                            losses,
                            handle.history,
                            streamed,
                            best,
                        )
                        outer.markdown(
                            _completion_html(solver_name, summary, handle.n_dropped),
                            unsafe_allow_html=True,
                        )
                except BaseException as exc:  # noqa: BLE001 — re-raised below
                    if type(exc).__name__ not in _STREAMLIT_CONTROL_EXC:
                        raise
                    # ⏹ Stop / rerun → Streamlit is tearing the run down. Cancel
                    # the interrupted worker (ANY interruption, not just Stop —
                    # otherwise a plain rerun orphans it), then salvage its
                    # best-so-far point and persist every completed run before
                    # the local ``results`` is lost.
                    if in_flight is not None:
                        in_flight.cancel_event.set()
                        # The worker may have finished (result set, _DONE queued)
                        # between the last tick and the interruption — drain and
                        # record the REAL summary instead of demoting a completed
                        # run to a best-so-far partial.
                        drain_handle(in_flight)
                        if in_flight.result is not None:
                            per_objective[objective_name] = in_flight.result
                        else:
                            per_objective[objective_name] = partial_from_history(
                                solver_name=solver_name,
                                objective_name=objective_name,
                                model_key=model_key,
                                history=in_flight.history,
                                market_data=market_data,
                                true_params=truth,
                                elapsed=(
                                    in_flight.history[-1].elapsed_seconds
                                    if in_flight.history
                                    else 0.0
                                ),
                            )
                    per_solver[solver_name] = per_objective
                    results[model_key] = per_solver
                    state_manager.commit_results(results)
                    raise

            per_solver[solver_name] = per_objective

        results[model_key] = per_solver

    # Every run finished normally: drop the handle registry so the still-present
    # ⏹ Stop button (in the now-collapsed status card) can't fire a false
    # cancellation on the next rerun (see _request_cancel).
    st.session_state[_LIVE_HANDLES_KEY] = []

    overall_status.update(
        label=f"Calibration complete · {n_runs} run(s) finished",
        state="complete",
        expanded=False,
    )
    return results
