"""
Live-convergence and parameter-trajectory charts (animated)
============================================================

* :func:`render_loss_overlay_animated` — best-so-far loss curves for
  1..N runs superimposed, animated, opening on the converged state with a
  ▶ Replay / Pause / slider.
* :func:`render_parameter_trajectories_animated` — small-multiples
  (one sub-plot per parameter) overlaying 1..N runs, animated, same
  open-on-final + replay behaviour.
* :func:`render_multi_start_loss` — overlaid restart curves (static).
* :func:`render_live_progress_chart` — quick non-animated chart used
  during live calibration to show ongoing progress.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.numba_kernels import best_per_iteration
from utils.plotly_theme import COLORS, FONT_FAMILY, PALETTE, apply_lab_theme

# Unified-hover header (``hovermode="x unified"``). Plotly otherwise shows the
# bare x value (the iteration index) as an unlabelled header; we label it once
# here so the per-trace lines never need to repeat ``iter %{x}``.
_ITER_HOVER_TITLE = dict(text="<b>iter %{x}</b>")


def _latest_restart_segment(history):
    """Return the tail of ``history`` belonging to the current restart.

    Surface calibrators run ``n_restarts`` independent fits, each with a
    *fresh* ``IterationLogger`` whose first snapshot therefore has
    ``iteration == 0``. The live stream concatenates every restart, so the
    last ``iteration == 0`` marks where the restart in progress began. The
    live view slices to that segment: concatenating restarts made the
    iteration index non-monotonic (one curve drawn back over another) and
    over-counted evaluations against the per-restart ``max_nfev`` budget.
    """
    if not history:
        return history
    start = 0
    for i, snap in enumerate(history):
        if getattr(snap, "iteration", 0) == 0:
            start = i
    return history[start:]


def _adaptive_loss_y_axis(objs: np.ndarray, title_hint: str = "Loss") -> dict:
    """Choose a Plotly ``update_yaxes`` config that fits the objective.

    Log scale is great for surface calibrations (RSS / 2 spans 8+
    decades from start to optimum), but Plotly silently drops every
    non-positive sample from a log axis. GARCH-family calibrators
    minimise the negative log-likelihood, which is **negative** for
    typical daily returns — so the live convergence chart used to
    look empty even though ``handle.history`` was full.

    We detect any non-positive sample and fall back to a linear axis
    with a 3-significant-digit tick format. Both branches now set an
    explicit ``range`` over the full ``objs`` span so that animated
    charts (``render_loss_chart_animated``), which initialise the
    figure with only the first sample, still display every frame —
    the layout's auto-range used to lock onto ``objs[0]`` and clip
    every later frame's "best so far" line off-screen.
    """
    finite = objs[np.isfinite(objs)]
    if finite.size == 0:
        return dict(
            type="linear",
            title=f"{title_hint}  (linear)",
            tickformat=".3g",
        )
    if (finite <= 0.0).any():
        ymin = float(finite.min())
        ymax = float(finite.max())
        span = ymax - ymin
        # Padding scales with the span, with floors based on |ymax|
        # and |ymin| so identical-value or near-zero runs still get a
        # visible window.
        pad = max(span * 0.05, abs(ymax) * 0.01, abs(ymin) * 0.01, 0.1)
        return dict(
            type="linear",
            title=f"{title_hint}  (linear)",
            tickformat=".3g",
            range=[ymin - pad, ymax + pad],
        )
    # Bracket the ACTUAL data min/max. The old ``max(…, 1e-15)`` floor raised
    # the bottom to 1e-15, so a loss that converged below that (perfect
    # synthetic fits reach ~1e-16 or smaller) fell *under* the range and the
    # curve clipped off the bottom of the chart. ``finite`` is strictly
    # positive here (non-positive samples take the linear branch), so log10 is
    # well-defined; a 1e-30 guard only caps a pathological denormal without
    # clipping any real converged loss.
    lo = float(np.log10(max(finite.min(), 1e-30)))
    hi = float(np.log10(max(finite.max(), 1e-30)))
    # Pad BOTH ends (in decades): 5 % of the span, with a 0.15-decade floor so
    # flat/near-constant series still get a visible margin and the converged
    # tail + its marker never sit on the edge.
    pad = max(0.05 * (hi - lo), 0.15)
    return dict(
        type="log",
        title=f"{title_hint}  (log scale)",
        tickformat=".0e",
        range=[lo - pad, hi + pad],
    )


def _filter_history_for_display(history):
    # Show every *objective evaluation* so the convergence trace reflects the
    # true work done (≈ max_nfev points), matching the x-axis "Objective
    # Evaluation Count". Solvers that record both per-evaluation and
    # per-generation snapshots (DE, NM, L-BFGS-B) would otherwise collapse to a
    # handful of callback marks. The best-so-far line and the running-best
    # parameter projection already smooth DE's noisy per-trial cloud. Falls back
    # to the full history if a solver emitted no per-evaluation snapshots.
    ev = [s for s in history if getattr(s, "source", None) == "evaluation"]
    return ev if ev else list(history)


def _project_to_running_best(history) -> tuple[np.ndarray, list[dict]]:
    """Return (objectives, params_list) where each entry corresponds to the
    running best objective seen so far.

    Global solvers (DE) report population-best per generation, but the
    population best can jump anywhere in the parameter space when DE finds
    a better region — making the raw trajectory chaotic even when the
    objective is improving monotonically. Projecting on the running best
    keeps params constant between true improvements, producing a
    step-function trajectory that mirrors the convergence story told by
    best_per_iteration on the loss chart.
    """
    if not history:
        return np.empty(0, dtype=np.float64), []
    best_obj = float("inf")
    best_params: dict | None = None
    out_obj: list[float] = []
    out_params: list[dict] = []
    for s in history:
        if s.objective < best_obj:
            best_obj = float(s.objective)
            best_params = dict(s.params_natural)
        out_obj.append(best_obj)
        # Keep a fresh dict each step so downstream consumers can mutate without
        # cross-contamination.
        out_params.append(dict(best_params) if best_params is not None else {})
    return np.array(out_obj, dtype=np.float64), out_params


def _frame_step(n: int, max_frames: int = 80) -> int:
    if n <= max_frames:
        return 1
    return max(1, n // max_frames)


def _animation_buttons(frame_duration: int = 80, *, from_current: bool = True) -> list[dict]:
    # ``from_current=False`` makes ▶ always restart from the first frame —
    # used by the charts that open on their *final* state so the button
    # replays the path from the start rather than resuming from the end.
    play_label = "▶  Play" if from_current else "▶  Replay"
    return [
        dict(
            type="buttons",
            direction="left",
            x=0.0, y=-0.18,
            xanchor="left", yanchor="top",
            pad=dict(t=4, r=10),
            showactive=False,
            # Teal-on-white call-to-action — matches the iv_surface
            # animation buttons so users see the same primary control
            # cluster across every animated chart in the app.
            bgcolor=COLORS["primary"],
            bordercolor=COLORS["primary_dim"],
            borderwidth=1,
            font=dict(family=FONT_FAMILY, color="#ffffff", size=12),
            buttons=[
                dict(
                    label=play_label,
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=frame_duration, redraw=False),
                        fromcurrent=from_current,
                        transition=dict(duration=0),
                    )],
                ),
                dict(
                    label="❚❚  Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
        )
    ]


def _slider_template(steps: list[dict]) -> dict:
    return dict(
        active=0,
        x=0.12, y=-0.18,
        len=0.85,
        xanchor="left", yanchor="top",
        currentvalue=dict(
            prefix="iter: ",
            font=dict(family=FONT_FAMILY, color=COLORS["text"], size=12),
        ),
        steps=steps,
        font=dict(family=FONT_FAMILY, color=COLORS["axis"], size=10),
        # Light grey rail readable on the plotly_white canvas; active
        # step keeps the teal primary so the current snapshot pops.
        bgcolor="rgba(0,0,0,0.06)",
        activebgcolor=COLORS["primary"],
        bordercolor=COLORS["grid_strong"],
        tickcolor=COLORS["axis"],
    )


# ─────────────────────────────────────────────────────────── Loss ──── #


def render_loss_overlay_animated(traces, *, title: str = "Convergence") -> go.Figure:
    """Animated overlay of best-so-far loss curves for 1..N runs.

    ``traces`` is a list of ``(label, iteration_history, style)`` where
    ``style`` is the ``utils.plotly_theme.series_style`` dict. Each run gets
    one *best-so-far* line (colour/dash from ``style``); a single selected
    run also keeps the translucent evaluation markers for context.

    The figure opens on the **final** state (full curves) so the result is
    visible immediately after a fit; the slider sits on the last frame and
    ▶ Replay animates the descent from the start. Per-iteration callback
    snapshots are preferred (drops DE's per-generation exploration noise);
    series of unequal length freeze at their final value on later frames.
    """
    from utils.plotly_theme import empty_state_figure

    series = []
    for label, history, style in traces:
        if not history:
            continue
        objs = np.array(
            [s.objective for s in _filter_history_for_display(history)],
            dtype=np.float64,
        )
        if objs.size == 0:
            continue
        iters = np.arange(objs.size, dtype=np.int64)
        series.append((label, iters, objs, best_per_iteration(objs), style))
    if not series:
        return empty_state_figure("Run a calibration to see the convergence trace.")

    single = len(series) == 1
    fig = go.Figure()
    animated: list[tuple[int, np.ndarray, np.ndarray]] = []
    all_y: list[float] = []

    if single:
        _label, iters, objs, best, style = series[0]
        all_y.extend(objs.tolist())
        # Evaluation markers (raw, translucent) so the bold best-so-far line
        # always reads on top — same contrast rationale as the live chart.
        fig.add_trace(
            go.Scatter(
                x=iters, y=objs, mode="markers",
                marker=dict(color=COLORS["accent"], size=5, opacity=0.45,
                            line=dict(color=COLORS["plot"], width=0.5)),
                name="evaluation",
                hovertemplate="obj %{y:.4e}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=iters, y=best, mode="lines+markers",
                line=dict(color=style["color"], width=3),
                marker=dict(color=style["color"], size=8,
                            line=dict(color=COLORS["plot"], width=1.2)),
                name="best so far",
                hovertemplate="best %{y:.4e}<extra></extra>",
            )
        )
        animated.append((0, iters, objs))
        animated.append((1, iters, best))
    else:
        for i, (label, iters, _objs, best, style) in enumerate(series):
            all_y.extend(best.tolist())
            fig.add_trace(
                go.Scatter(
                    x=iters, y=best, mode="lines",
                    line=dict(color=style["color"], dash=style["dash"], width=2.4),
                    name=label,
                    hovertemplate=(
                        f"<b>{label}</b><br>best %{{y:.4e}}<extra></extra>"
                    ),
                )
            )
            animated.append((i, iters, best))

    max_len = max(len(iters) for _l, iters, *_r in series)
    step = _frame_step(max_len)
    frame_indices = sorted(set(list(range(0, max_len, step)) + [max_len - 1]))
    fig.frames = [
        go.Frame(
            name=str(idx),
            data=[go.Scatter(x=x[: idx + 1], y=y[: idx + 1]) for _t, x, y in animated],
            traces=[t for t, _x, _y in animated],
        )
        for idx in frame_indices
    ]
    slider_steps = [
        dict(
            method="animate",
            args=[[str(i)], dict(mode="immediate",
                                 frame=dict(duration=0, redraw=False),
                                 transition=dict(duration=0))],
            label=str(i),
        )
        for i in frame_indices
    ]
    slider = _slider_template(slider_steps)
    slider["active"] = len(frame_indices) - 1  # open on the converged frame

    suffix = f"{max_len} evaluations" if single else f"{len(series)} runs"
    head = f"{title}  ·  {suffix}".strip(" ·")
    apply_lab_theme(fig, height=460, title=head,
                    margin=(60, 25, 74, 130), legend_horizontal=True)
    fig.update_layout(
        updatemenus=_animation_buttons(from_current=False),
        sliders=[slider],
    )
    x_max = float(max_len - 1)
    x_pad = max(x_max * 0.02, 0.5)
    fig.update_xaxes(
        title="Objective Evaluation Count",
        range=[-x_pad, x_max + x_pad],
        unifiedhovertitle=_ITER_HOVER_TITLE,
    )
    fig.update_yaxes(
        **_adaptive_loss_y_axis(np.array(all_y, dtype=np.float64),
                                title_hint="Objective  ½‖r‖²")
    )
    return fig


def render_live_progress_chart(history, *, height: int = 240) -> go.Figure:
    """Non-animated quick chart used while a calibration is RUNNING."""
    if not history:
        from utils.plotly_theme import empty_state_figure
        return empty_state_figure("Waiting for first iteration…")
    # Only the restart in progress (see _latest_restart_segment), and only its
    # per-evaluation snapshots, so the curve is one clean descent over a
    # monotone evaluation index instead of every restart overlaid.
    segment = _filter_history_for_display(_latest_restart_segment(history))
    objs = np.array([s.objective for s in segment], dtype=np.float64)
    iters = np.arange(objs.size, dtype=np.int64)
    best = best_per_iteration(objs)
    fig = go.Figure()
    # See render_loss_chart_animated for the visibility rationale: the
    # eval markers stay dim so the bold best-so-far overlay reads on top
    # even when the solver is monotone (L-BFGS-B / GARCH).
    fig.add_trace(
        go.Scatter(
            x=iters, y=objs,
            mode="markers",
            marker=dict(color=COLORS["accent"], size=3, opacity=0.4),
            name="evaluation",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iters, y=best,
            mode="lines+markers",
            line=dict(color=COLORS["primary"], width=2.6),
            marker=dict(
                color=COLORS["primary"],
                size=6,
                line=dict(color=COLORS["plot"], width=1.0),
            ),
            name="best",
            showlegend=False,
        )
    )
    apply_lab_theme(fig, height=height, margin=(45, 15, 10, 35), legend_horizontal=False)
    fig.update_yaxes(**_adaptive_loss_y_axis(objs, title_hint="Loss  ½‖r‖²"))
    fig.update_xaxes(title="Objective Evaluation Count")
    return fig


# ────────────────────────────────────── Parameter trajectories ──── #


def render_parameter_trajectories_animated(
    traces,
    *,
    true_params: dict[str, float] | None = None,
) -> go.Figure:
    """Animated overlay of parameter trajectories for 1..N runs of ONE model.

    ``traces`` is a list of ``(label, iteration_history, style)``. Each run
    contributes one animated line per parameter sub-plot (colour/dash from
    ``style``, legend-grouped so a click toggles the whole run). All runs
    must share the same model (identical parameter set) — the Live tab
    facets the selection by model before calling this.

    Opens on the **final** trajectories (full paths visible immediately);
    ▶ Replay animates from the start. Each snapshot is projected onto the
    running-best objective so the path tells the convergence story instead
    of per-generation jitter.
    """
    from utils.plotly_theme import empty_state_figure

    runs = []
    for label, history, style in traces:
        if not history:
            continue
        _, params_list = _project_to_running_best(_filter_history_for_display(history))
        if params_list:
            runs.append((label, params_list, style))
    if not runs:
        return empty_state_figure("No iteration history available.")

    names = [k for k in runs[0][1][0] if not k.startswith("_")]
    cols = 2 if len(names) > 1 else 1
    rows = (len(names) + cols - 1) // cols
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=names,
        vertical_spacing=0.18, horizontal_spacing=0.10,
    )

    # One (run × parameter) Scatter each; base data = full path (final), and
    # ``animated`` mirrors them so the frames reveal each progressively.
    animated: list[tuple[int, np.ndarray, np.ndarray]] = []
    extents: dict[str, list[float]] = {name: [] for name in names}
    max_len = 1
    trace_idx = 0
    for label, params_list, style in runs:
        iters = np.arange(len(params_list), dtype=np.int64)
        max_len = max(max_len, len(iters))
        for p_i, name in enumerate(names):
            y = np.array([p.get(name, np.nan) for p in params_list], dtype=np.float64)
            extents[name].extend(y[np.isfinite(y)].tolist())
            fig.add_trace(
                go.Scatter(
                    x=iters, y=y, mode="lines",
                    line=dict(color=style["color"], dash=style["dash"], width=2.2),
                    name=label, legendgroup=label, showlegend=(p_i == 0),
                    hovertemplate=f"<b>{label}</b><br>{name} %{{y:.4g}}<extra></extra>",
                ),
                row=p_i // cols + 1, col=p_i % cols + 1,
            )
            animated.append((trace_idx, iters, y))
            trace_idx += 1

    step = _frame_step(max_len)
    frame_indices = sorted(set(list(range(0, max_len, step)) + [max_len - 1]))
    fig.frames = [
        go.Frame(
            name=str(idx),
            data=[go.Scatter(x=x[: idx + 1], y=y[: idx + 1]) for _t, x, y in animated],
            traces=[t for t, _x, _y in animated],
        )
        for idx in frame_indices
    ]
    slider_steps = [
        dict(
            method="animate",
            args=[[str(i)], dict(mode="immediate",
                                 frame=dict(duration=0, redraw=False),
                                 transition=dict(duration=0))],
            label=str(i),
        )
        for i in frame_indices
    ]
    slider = _slider_template(slider_steps)
    slider["active"] = len(frame_indices) - 1

    last_row = (len(names) - 1) // cols + 1
    for p_i, name in enumerate(names):
        vals = extents[name]
        if true_params and name in true_params:
            vals = vals + [float(true_params[name])]
        v_min = min(vals) if vals else 0.0
        v_max = max(vals) if vals else 1.0
        pad = max(abs(v_max - v_min) * 0.1, 1e-6)
        row = p_i // cols + 1
        col = p_i % cols + 1
        if true_params and name in true_params:
            fig.add_hline(
                y=float(true_params[name]),
                line=dict(color=COLORS["danger"], width=1.4, dash="dash"),
                row=row, col=col,
            )
        x_title = "Objective Evaluation Count" if row == last_row else None
        fig.update_xaxes(range=[0.0, float(max_len - 1)], title=x_title, row=row, col=col)
        fig.update_yaxes(
            range=[v_min - pad, v_max + pad],
            title=f"{name} (natural scale)", row=row, col=col,
        )

    chart_height = min(210 * rows + 160, 900)
    title = "Parameter trajectories  ·  every selected run superimposed"
    if true_params:
        title += "  ·  red dashed line = true value"
    apply_lab_theme(
        fig, height=chart_height, title=title,
        margin=(50, 25, 92, 130), legend_horizontal=True,
    )
    fig.update_layout(
        updatemenus=_animation_buttons(from_current=False),
        sliders=[slider],
    )
    # Label the unified-hover header on every sub-plot's x-axis (no row/col
    # selector → applies to all) so the bare iteration index is never shown raw.
    fig.update_xaxes(unifiedhovertitle=_ITER_HOVER_TITLE)
    for ann in fig.layout.annotations:
        ann.font = dict(family=FONT_FAMILY, color=COLORS["text"], size=12)
    return fig


# ─────────────────────────────────────────────── Multi-start ──── #


def render_multi_start_loss(
    multi_start_history: Iterable,
    *,
    best_index: int | None = None,
) -> go.Figure:
    fig = go.Figure()
    any_data = False
    for k, history in enumerate(multi_start_history):
        if not history:
            continue
        any_data = True
        # Same callback-preference logic as the single-run convergence chart,
        # then enforce a monotone best-so-far so each restart yields a clean
        # convergence curve regardless of solver stochasticity.
        display_history = _filter_history_for_display(history)
        objs = np.array([s.objective for s in display_history], dtype=np.float64)
        objs = best_per_iteration(objs)
        iters = np.arange(len(display_history), dtype=np.int64)
        is_best = (best_index is not None and k == best_index)
        fig.add_trace(
            go.Scatter(
                x=iters, y=objs,
                mode="lines+markers",
                line=dict(
                    color=PALETTE[k % len(PALETTE)],
                    width=3.0 if is_best else 1.6,
                    dash="solid" if is_best else "dot",
                ),
                marker=dict(size=4 if is_best else 3),
                name=f"start {k}" + (" ★ best" if is_best else ""),
            )
        )
    if not any_data:
        from utils.plotly_theme import empty_state_figure
        return empty_state_figure("No multi-start history captured for this run.")
    apply_lab_theme(fig, height=460, title="Multi-start convergence  ·  best-so-far per restart")
    all_objs = np.concatenate([
        np.array([s.objective for s in _filter_history_for_display(h)], dtype=np.float64)
        for h in multi_start_history if h
    ]) if multi_start_history else np.array([1.0])
    fig.update_yaxes(**_adaptive_loss_y_axis(all_objs, title_hint="Objective  ½‖r‖²"))
    fig.update_xaxes(title="Objective Evaluation Count")
    return fig
