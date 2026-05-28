"""Solver-comparison charts (themed)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from config.constants import (
    MODEL_DISPLAY_NAMES,
    MODEL_ICONS,
    OBJECTIVE_DISPLAY_NAMES,
)
from utils.plotly_theme import (
    _SOLVER_DASH,
    _SOLVER_SYMBOL,
    COLORS,
    FONT_FAMILY,
    SOLVER_COLOR_MAP,
    apply_lab_theme,
    model_color,
)

# ``_SOLVER_SYMBOL`` / ``_SOLVER_DASH`` now live in ``utils.plotly_theme`` so
# the Compare-tab charts and the per-tab overlays share one encoding source.


def _label(model_key: str, solver_name: str, objective_name: str | None = None) -> str:
    icon = MODEL_ICONS.get(model_key, "")
    base = f"{icon} {MODEL_DISPLAY_NAMES.get(model_key, model_key)} · {solver_name}"
    # Always show the objective (incl. the default ``price_mse``) so that when
    # several objectives are overlaid the price_mse run is named in the legend
    # instead of looking like an unlabelled baseline. Use the same display name
    # as the sidebar / pickers ("Price MSE", not the raw "price_mse" key).
    if objective_name:
        base += f" · {OBJECTIVE_DISPLAY_NAMES.get(objective_name, objective_name)}"
    return base


def _iter_flat(results: dict):
    """Yield ``(model, solver, objective, summary)`` over both legacy and
    nested ``dict[solver][objective] -> Summary`` schemas."""
    for model_key, per_solver in results.items():
        for solver_name, slot in per_solver.items():
            if isinstance(slot, dict):
                for obj_name, summary in slot.items():
                    yield model_key, solver_name, obj_name, summary
            else:
                yield model_key, solver_name, getattr(slot, "objective_name", "price_mse"), slot


def _pareto_y_axis(results_iter) -> tuple[bool, str, str]:
    """Decide the Pareto y-axis based on the available loss metric.

    Surface-family results expose ``rmse_price`` directly; returns-family
    (GARCH) calibrators don't price quotes and leave it as ``None``, so
    the chart used to silently drop them. We fall back to the per-run
    ``diagnostics["objective_loss"]`` (negative log-likelihood for
    GARCH MLE) and label the axis accordingly. Returns ``(use_price,
    y_title, hover_label)``.
    """
    has_price = any(
        s.result is not None
        and getattr(s.result, "rmse_price", None) is not None
        for s in results_iter
    )
    if has_price:
        return True, "Pricing RMSE (log scale, $)", "RMSE price"
    return False, "Objective loss (log scale, NLL)", "Objective loss"


def _pareto_y_value(summary, use_price: bool) -> float:
    if use_price:
        v = getattr(summary.result, "rmse_price", None)
        return float(v) if v is not None else float("nan")
    diag = summary.result.diagnostics or {}
    v = diag.get("objective_loss")
    if v is None:
        # Some backends expose the final loss on the result object directly.
        v = getattr(summary.result, "objective_value", None)
    return float(v) if v is not None else float("nan")


def _adaptive_loss_axis_kwargs(
    values,
    *,
    log_title: str,
    linear_title: str,
) -> dict:
    """Pick log vs linear Plotly y-axis kwargs based on the data sign.

    Mirrors ``charts.live_convergence._adaptive_loss_y_axis`` but takes
    caller-supplied axis titles since the Compare-tab charts label
    their axis "Pricing RMSE" / "Objective loss" rather than the
    generic "Loss". Any non-positive sample (NaN-tolerant via the
    finite filter) forces a linear axis — Plotly silently drops
    negative samples from a log axis, which is exactly what made
    GARCH NLL points disappear from the Pareto frontier and overlay.

    Both branches set an explicit ``range`` over the actual data so
    that animated or partially-populated figures don't lock the
    layout onto a single sample and clip the rest.
    """
    arr = np.asarray(
        [v for v in values if v is not None and np.isfinite(v)],
        dtype=np.float64,
    )
    if arr.size == 0:
        return dict(type="linear", title=linear_title, tickformat=".3g")
    if (arr <= 0.0).any():
        ymin = float(arr.min())
        ymax = float(arr.max())
        span = ymax - ymin
        pad = max(span * 0.05, abs(ymax) * 0.01, abs(ymin) * 0.01, 0.1)
        return dict(
            type="linear",
            title=linear_title,
            tickformat=".3g",
            range=[ymin - pad, ymax + pad],
        )
    return dict(
        type="log",
        title=log_title,
        tickformat=".0e",
        range=[
            # 0.1 decade of head- and foot-room: without the lower pad the
            # marker at ``arr.min()`` is centred on the bottom plot edge
            # and its bottom half is clipped (most visible on the Pareto
            # frontier where the lowest-loss point reads as half a dot).
            float(np.log10(max(arr.min(), 1e-15)) - 0.1),
            float(np.log10(max(arr.max(), 1e-15)) + 0.1),
        ],
    )


def render_pareto_chart(summaries) -> go.Figure:
    fig = go.Figure()
    # Iterate via _iter_per_solver to handle both legacy dict[solver, Summary]
    # and the new dict[solver, dict[objective, Summary]] schemas.
    summaries_list = list(_iter_per_solver(summaries))
    use_price, y_title, hover_label = _pareto_y_axis(s for _, _, s in summaries_list)
    collected_y: list[float] = []
    for label, solver, s in summaries_list:
        if s.result is None:
            continue
        y = _pareto_y_value(s, use_price)
        elapsed = float(s.elapsed)
        if np.isnan(y) or elapsed < 0:
            continue
        collected_y.append(y)
        fig.add_trace(
            go.Scatter(
                x=[elapsed], y=[y],
                mode="markers+text",
                text=[label], textposition="top center",
                textfont=dict(family=FONT_FAMILY, color=COLORS["text"], size=11),
                marker=dict(
                    size=22,
                    color=SOLVER_COLOR_MAP.get(solver, COLORS["primary"]),
                    line=dict(color=COLORS["plot"], width=2),
                ),
                name=label,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "elapsed %{x:.2f}s<br>"
                    f"{hover_label} %{{y:.4e}}<extra></extra>"
                ),
            )
        )
    apply_lab_theme(fig, height=420,
                     title="Pareto frontier  ·  ↙ ideal (fast & accurate)",
                     legend_horizontal=False)
    # Surface the legend so dense runs (many solvers / objectives) stay
    # readable when the inline text annotations would otherwise overlap.
    # The markers+text mode keeps the at-a-glance labels for the
    # canonical, sparse case.
    fig.update_xaxes(title="Wall-clock Time (seconds)")
    # Drop the "(log scale)" suffix from y_title — the adaptive helper
    # owns the scale-suffix labelling and we don't want a stale label
    # to claim log when we actually emitted a linear axis.
    base_title = y_title.replace(" (log scale, $)", " ($)").replace(" (log scale, NLL)", " (NLL)")
    fig.update_yaxes(
        **_adaptive_loss_axis_kwargs(
            collected_y,
            log_title=f"{base_title} (log scale)",
            linear_title=base_title,
        )
    )
    return fig


def _iter_per_solver(summaries):
    """Yield ``(label, summary)`` pairs handling legacy + nested objective schemas.

    Legacy : ``summaries`` is ``dict[solver, Summary]``.
    New    : ``summaries`` is ``dict[solver, dict[objective, Summary]]``.

    ``label`` is the bare solver name in the legacy case, ``solver/objective``
    when several objectives are stored under one solver so plot legends stay
    unambiguous.
    """
    for solver, slot in summaries.items():
        if isinstance(slot, dict):
            multi = len(slot) > 1
            for obj_name, s in slot.items():
                label = (
                    f"{solver}/{obj_name}"
                    if multi and obj_name != "price_mse"
                    else solver
                )
                yield label, solver, s
        else:
            yield solver, solver, slot


def render_parameter_recovery_bars(
    summaries,
    true_params: dict[str, float],
) -> go.Figure:
    names = [k for k in true_params if k != "_model"]
    fig = go.Figure()
    for label, solver, s in _iter_per_solver(summaries):
        if s.result is None:
            continue
        errs = [s.relative_recovery_error.get(n, np.nan) * 100.0 for n in names]
        fig.add_trace(
            go.Bar(
                name=label,
                x=names,
                y=errs,
                marker=dict(color=SOLVER_COLOR_MAP.get(solver, COLORS["primary"])),
                hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.2f}}%<extra></extra>",
            )
        )
    apply_lab_theme(fig, height=420,
                     title="Parameter recovery  ·  |estimated − true| / |true|")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Model Parameter")
    fig.update_yaxes(title="Relative Recovery Error (%)", ticksuffix="%")
    return fig


def render_overlaid_loss(summaries) -> go.Figure:
    fig = go.Figure()
    all_y: list[float] = []
    for label, solver, s in _iter_per_solver(summaries):
        if s.result is None or not s.result.iteration_history:
            continue
        hist = s.result.iteration_history
        x = np.array([snap.iteration for snap in hist])
        y = np.array([snap.objective for snap in hist])
        all_y.extend(y.tolist())
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode="lines",
                line=dict(color=SOLVER_COLOR_MAP.get(solver, COLORS["primary"]), width=2.4),
                name=label,
            )
        )
    # Adapt the y-axis to the data: surface RSS/2 spans many decades and
    # benefits from log scale, GARCH NLL crosses zero and needs linear
    # (Plotly drops non-positive samples from a log axis).
    y_kwargs = _adaptive_loss_axis_kwargs(
        all_y,
        log_title="Objective  ½‖r‖²  (log scale)",
        linear_title="Objective",
    )
    title_scale = "log scale" if y_kwargs["type"] == "log" else "linear"
    apply_lab_theme(
        fig, height=420,
        title=f"Convergence comparison  ·  every solver superimposed ({title_scale})",
    )
    fig.update_xaxes(title="Objective Evaluation Count")
    fig.update_yaxes(**y_kwargs)
    return fig


def comparison_table(summaries) -> list[dict]:
    rows = []
    for label, _solver, s in _iter_per_solver(summaries):
        if s.result is None:
            rows.append({
                "Solver": label, "Status": "✗ failed",
                "RMSE price": np.nan, "RMSE IV (bps)": np.nan,
                "Iterations": 0,
                "Elapsed (s)": s.elapsed if s.elapsed >= 0 else np.nan,
                # Surface the exception text so the user can diagnose
                # without digging through the console. Empty for runs
                # that finished cleanly. Full traceback is logged via
                # logger.exception in calibrate_many / live_runner.
                "Error": s.error or "",
            })
            continue
        rows.append({
            "Solver": label,
            "Status": "✓" if s.result.success else "⚠",
            "RMSE price": float(s.result.rmse_price) if s.result.rmse_price is not None else np.nan,
            # rmse_iv is stored as a decimal (e.g. 0.002) — multiply by 1e4 to
            # match the column header that promises basis points.
            "RMSE IV (bps)": float(s.result.rmse_iv) * 1e4 if s.result.rmse_iv is not None else np.nan,
            "Iterations": int(s.result.n_iterations),
            "Elapsed (s)": float(s.elapsed),
            "Error": "",
        })
    return rows


# ──────────────────────────────────────────────────────────────────────
# Multi-model variants
# ──────────────────────────────────────────────────────────────────────
# Each accepts ``results: dict[model_key, dict[solver_name, summary]]``
# and overlays every ``(model, solver)`` pair on the same chart. Colour
# is keyed by model (so the eye associates "Heston is teal" across the
# whole app), shape/dash by solver, so the user can read both axes at a
# glance.


def multi_comparison_table(results: dict) -> list[dict]:
    """Flatten nested results into ``(model, solver, objective)`` rows."""
    rows = []
    for model_key, solver_name, obj_name, s in _iter_flat(results):
        model_label = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        if s.result is None:
            status = (
                "⊘ skipped"
                if (s.error or "").startswith("solver '") and "not supported" in (s.error or "")
                else "✗ failed"
            )
            rows.append({
                "Model": model_label,
                "Solver": solver_name,
                "Objective": obj_name,
                "Status": status,
                "RMSE price": np.nan,
                "RMSE IV (bps)": np.nan,
                "Final loss": np.nan,
                "Grad norm": np.nan,
                "Iterations": 0,
                "Elapsed (s)": s.elapsed if s.elapsed >= 0 else np.nan,
                # Surface the exception text so the user can diagnose
                # without digging through the console. Empty for runs
                # that finished cleanly. Full traceback is logged via
                # logger.exception in calibrate_many / live_runner.
                "Error": s.error or "",
            })
            continue
        rows.append({
            "Model": model_label,
            "Solver": solver_name,
            "Objective": obj_name,
            "Status": (
                "⏸ stopped"
                if getattr(s, "partial", False)
                else "✓"
                if s.result.success
                else "⚠"
            ),
            "RMSE price": (
                float(s.result.rmse_price)
                if s.result.rmse_price is not None else np.nan
            ),
            "RMSE IV (bps)": (
                float(s.result.rmse_iv) * 1e4
                if s.result.rmse_iv is not None else np.nan
            ),
            # ``final_loss`` reads ``result.objective_value``, which is set
            # for every calibrator (surface uses RSS/2, GARCH uses NLL).
            # The previous "Objective loss" column was sourced from
            # ``diag["objective_loss"]`` — populated for Heston/Bates/
            # Merton only and NaN for GARCH-family, making the column
            # half-empty.
            "Final loss": (
                float(s.final_loss) if s.final_loss is not None else np.nan
            ),
            # Only solvers that touch the gradient surface (L-BFGS-B,
            # LM-JAX, GARCH MLE) populate this; DE / NM leave it None
            # and the dataframe renders it as "—".
            "Grad norm": (
                float(s.grad_norm) if s.grad_norm is not None else np.nan
            ),
            "Iterations": int(s.result.n_iterations),
            "Elapsed (s)": float(s.elapsed),
            "Error": "",
        })
    return rows


def render_multi_pareto(results: dict, *, visible_models: set[str] | None = None) -> go.Figure:
    """Pareto frontier overlaying all ``(model, solver, objective)`` triples.

    Colour = model · marker shape = solver. ``visible_models`` lets a
    chart-level view-filter hide a subset without re-running.

    The y-axis is the **pricing RMSE** when at least one displayed
    result exposes it (surface models) and falls back to the
    **objective loss / NLL** when all visible results come from the
    returns family (GARCH calibrators don't price quotes). Without
    the fallback, GARCH runs used to disappear silently from the
    chart.
    """
    fig = go.Figure()
    visible_summaries = [
        s for model_key, _, _, s in _iter_flat(results)
        if (visible_models is None or model_key in visible_models)
        and s.result is not None
    ]
    use_price, y_title, hover_label = _pareto_y_axis(visible_summaries)
    collected_y: list[float] = []
    for model_key, solver_name, obj_name, s in _iter_flat(results):
        if visible_models is not None and model_key not in visible_models:
            continue
        if s.result is None:
            continue
        y = _pareto_y_value(s, use_price)
        elapsed = float(s.elapsed)
        if np.isnan(y) or elapsed < 0:
            continue
        collected_y.append(y)
        col = model_color(model_key)
        fig.add_trace(
            go.Scatter(
                x=[elapsed], y=[y],
                mode="markers",
                marker=dict(
                    size=18,
                    color=col,
                    symbol=_SOLVER_SYMBOL.get(solver_name, "circle"),
                    line=dict(color=COLORS["plot"], width=2),
                ),
                name=_label(model_key, solver_name, obj_name),
                hovertemplate=(
                    f"<b>{_label(model_key, solver_name, obj_name)}</b><br>"
                    "elapsed %{x:.2f}s<br>"
                    f"{hover_label} %{{y:.4e}}<extra></extra>"
                ),
            )
        )
    # No in-figure title — the ``⚡ Pareto frontier · speed × accuracy``
    # section-header markdown above the chart already labels it. Keeping
    # the duplicate title caused the same collision the loss-overlay
    # chart had: with ~16 series the horizontal legend wraps to 3-4 rows
    # that climb back up into the title's paper-y band. The "colour =
    # model, shape = solver" encoding hint that used to live in the title
    # is preserved as an ``st.caption`` rendered by the tab.
    apply_lab_theme(fig, height=440)
    fig.update_xaxes(title="Wall-clock Time (seconds)")
    # Drop the "(log scale, $)"/"(log scale, NLL)" suffix already baked
    # into ``y_title`` — the adaptive helper appends the actual scale.
    base_title = y_title.replace(" (log scale, $)", " ($)").replace(" (log scale, NLL)", " (NLL)")
    fig.update_yaxes(
        **_adaptive_loss_axis_kwargs(
            collected_y,
            log_title=f"{base_title} (log scale)",
            linear_title=base_title,
        )
    )
    return fig


def render_multi_overlaid_loss(results: dict, *, visible_models: set[str] | None = None) -> go.Figure:
    """Loss curve per ``(model, solver, objective)``. Colour = model · dash = solver.

    The y-axis switches to linear when any displayed objective dips
    non-positive (GARCH MLE minimises ``-log_likelihood`` which is
    negative for typical daily returns) — a log axis would silently
    drop those points and leave the overlay looking empty.
    """
    fig = go.Figure()
    all_y: list[float] = []
    for model_key, solver_name, obj_name, s in _iter_flat(results):
        if visible_models is not None and model_key not in visible_models:
            continue
        if s.result is None or not s.result.iteration_history:
            continue
        col = model_color(model_key)
        hist = s.result.iteration_history
        x = np.array([snap.iteration for snap in hist])
        y = np.array([snap.objective for snap in hist])
        all_y.extend(y.tolist())
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode="lines",
                line=dict(
                    color=col,
                    dash=_SOLVER_DASH.get(solver_name, "solid"),
                    width=2.2,
                ),
                name=_label(model_key, solver_name, obj_name),
            )
        )
    y_arr = np.asarray(all_y, dtype=np.float64) if all_y else np.array([1.0])
    use_linear = (y_arr <= 0.0).any() or not np.isfinite(y_arr).any()
    if use_linear:
        y_kwargs = dict(title="Objective", type="linear", tickformat=".3g")
    else:
        y_kwargs = dict(
            title="Objective  ½‖r‖²  (log scale)",
            type="log",
            tickformat=".0e",
        )
    # No in-figure title — the section header rendered above the chart
    # ("📉 Loss overlay · every (model, solver)") already labels it, and the
    # y-axis title carries the (log scale) / (linear) distinction. Removing
    # the duplicate title also kills the layout collision that was happening
    # with the horizontal legend: with ~16 series the legend wraps to 3-4
    # rows that climb back up into the same paper-y band as the title.
    apply_lab_theme(fig, height=440)
    fig.update_xaxes(title="Objective Evaluation Count")
    fig.update_yaxes(**y_kwargs)
    return fig
