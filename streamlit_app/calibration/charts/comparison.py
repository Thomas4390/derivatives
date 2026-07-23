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
    PALETTE,
    adaptive_loss_axis,
    apply_lab_theme,
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
                yield (
                    model_key,
                    solver_name,
                    getattr(slot, "objective_name", "price_mse"),
                    slot,
                )


def _series_colors(results: dict) -> dict[tuple[str, str, str], str]:
    """A distinct palette colour per ``(model, solver, objective)`` run.

    Keyed by run identity — colouring by *model* alone (or *solver* alone) made
    several objectives of one solver, or several solvers of one model, collapse
    to a single hue. Built from the shared ``_iter_flat`` order so a run reads the
    same colour across the Pareto and loss overlays. ``PALETTE`` deliberately
    excludes the amber eval / red truth hues, so the runs never clash with them.
    """
    return {
        (m, s, o): PALETTE[i % len(PALETTE)]
        for i, (m, s, o, _summary) in enumerate(_iter_flat(results))
    }


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
        s.result is not None and getattr(s.result, "rmse_price", None) is not None
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
    """Compare-tab wrapper over the shared :func:`adaptive_loss_axis`.

    The Compare charts label their axis "Pricing RMSE" / "Objective loss"
    rather than the Live chart's generic "Loss", so they pass explicit titles;
    the log/linear decision and range numerics come from the one shared helper
    (previously a hand-synced copy that had already drifted a log floor).
    """
    return adaptive_loss_axis(values, log_title=log_title, linear_title=linear_title)


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
    # Distinct colour per run (solver/objective) — colouring by solver alone made
    # several objectives of one solver collapse to one hue. Same enumeration as the
    # overlays, so a run reads the same colour as in the single-model Pareto / loss.
    runs = list(_iter_per_solver(summaries))
    colors = {
        label: PALETTE[i % len(PALETTE)] for i, (label, _sv, _s) in enumerate(runs)
    }
    fig = go.Figure()
    for label, _solver, s in runs:
        if s.result is None:
            continue
        errs = [s.relative_recovery_error.get(n, np.nan) * 100.0 for n in names]
        fig.add_trace(
            go.Bar(
                name=label,
                x=names,
                y=errs,
                marker=dict(color=colors[label]),
                hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.2f}}%<extra></extra>",
            )
        )
    # No in-figure title — the section header above the chart already labels it,
    # and the duplicate title collided with the horizontal run legend (same fix
    # the Pareto / loss overlays use). A log y-axis keeps the small-error bars
    # readable when one parameter (e.g. weakly-identified kappa) recovers an order
    # of magnitude worse than the others.
    apply_lab_theme(fig, height=440)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Model Parameter")
    fig.update_yaxes(
        title="Relative Recovery Error (%, log)",
        type="log",
        ticksuffix="%",
    )
    return fig


def garch_q_aggregate_recovery(
    summaries, model_key: str, true_params: dict[str, float]
) -> list[dict]:
    """Identified-aggregate recovery rows for the nonaffine GARCH-Q trio.

    An option surface only weakly identifies ω, γ and h₀ individually
    (Christoffersen-Jacobs 2004): many (ω, γ, h₀) triples price almost the same
    smile. What it *does* pin down are two aggregates — the variance
    **persistence** and the annualised **long-run volatility** σ_LT. This builds
    one ``Run`` row per fit comparing those aggregates against the generator's
    truth, reading both sides off the model's own diagnostics (same formula on
    each side). The first row is the generator truth. Returns ``[]`` when the
    generator is not one of the MC-priced GARCH-Q models.
    """
    from backend.models.ngarch_q import (
        GARCHRiskNeutralModel,
        GJRGARCHRiskNeutralModel,
        NGARCHRiskNeutralModel,
    )

    classes = {
        "ngarch_q": NGARCHRiskNeutralModel,
        "garch_q": GARCHRiskNeutralModel,
        "gjr_q": GJRGARCHRiskNeutralModel,
    }
    cls = classes.get(model_key)
    if cls is None:
        return []
    true_model = cls(**{k: float(v) for k, v in true_params.items() if k != "_model"})
    tp = float(true_model.persistence)
    tsig = float(true_model.long_run_volatility) * 100.0
    rows: list[dict] = [
        {
            "Run": "🎯 True (generator)",
            "Persistence": tp,
            "|Δ persist|": 0.0,
            "σ_LT ann (%)": tsig,
            "|Δ σ_LT| (%)": 0.0,
        }
    ]
    for label, _solver, s in _iter_per_solver(summaries):
        if s.result is None or getattr(s.result, "model", None) is None:
            continue
        m = s.result.model
        if not hasattr(m, "persistence"):
            continue
        p = float(m.persistence)
        sig = float(m.long_run_volatility) * 100.0
        rows.append(
            {
                "Run": label,
                "Persistence": p,
                "|Δ persist|": abs(p - tp),
                "σ_LT ann (%)": sig,
                "|Δ σ_LT| (%)": abs(sig - tsig),
            }
        )
    return rows


# ──────────────────────────────────────────────────────────────────────
# Multi-model variants
# ──────────────────────────────────────────────────────────────────────
# Each accepts ``results: dict[model_key, dict[solver_name, summary]]``
# and overlays every ``(model, solver, objective)`` run on the same chart.
# Colour is keyed per run (so several objectives of one solver, or several
# solvers of one model, never collapse to a single hue); shape/dash = solver.


def multi_comparison_table(results: dict) -> list[dict]:
    """Flatten nested results into ``(model, solver, objective)`` rows."""
    rows = []
    for model_key, solver_name, obj_name, s in _iter_flat(results):
        model_label = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        if s.result is None:
            status = (
                "⊘ skipped"
                if (s.error or "").startswith("solver '")
                and "not supported" in (s.error or "")
                else "✗ failed"
            )
            rows.append(
                {
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
                    # logger.exception in live_runner.
                    "Error": s.error or "",
                }
            )
            continue
        rows.append(
            {
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
                    if s.result.rmse_price is not None
                    else np.nan
                ),
                "RMSE IV (bps)": (
                    float(s.result.rmse_iv) if s.result.rmse_iv is not None else np.nan
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
            }
        )
    return rows


def render_multi_pareto(
    results: dict, *, visible_models: set[str] | None = None
) -> go.Figure:
    """Pareto frontier overlaying all ``(model, solver, objective)`` triples.

    A distinct colour per run (model, solver, objective) · marker shape = solver.
    ``visible_models`` lets a chart-level view-filter hide a subset without re-running.

    The y-axis is the **pricing RMSE** when at least one displayed
    result exposes it (surface models) and falls back to the
    **objective loss / NLL** when all visible results come from the
    returns family (GARCH calibrators don't price quotes). Without
    the fallback, GARCH runs used to disappear silently from the
    chart.
    """
    fig = go.Figure()
    visible_summaries = [
        s
        for model_key, _, _, s in _iter_flat(results)
        if (visible_models is None or model_key in visible_models)
        and s.result is not None
    ]
    use_price, y_title, hover_label = _pareto_y_axis(visible_summaries)
    colors = _series_colors(results)
    collected_y: list[float] = []
    dropped_labels: list[str] = []
    for model_key, solver_name, obj_name, s in _iter_flat(results):
        if visible_models is not None and model_key not in visible_models:
            continue
        if s.result is None:
            continue
        y = _pareto_y_value(s, use_price)
        elapsed = float(s.elapsed)
        if np.isnan(y) or elapsed < 0:
            # On a price axis, returns-family runs (no rmse_price) map to NaN
            # and would vanish with no trace. Record them so a caption can name
            # the hidden runs instead of dropping them silently.
            if use_price and _pareto_y_value(s, False) == _pareto_y_value(s, False):
                dropped_labels.append(_label(model_key, solver_name, obj_name))
            continue
        collected_y.append(y)
        col = colors[(model_key, solver_name, obj_name)]
        fig.add_trace(
            go.Scatter(
                x=[elapsed],
                y=[y],
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
    base_title = y_title.replace(" (log scale, $)", " ($)").replace(
        " (log scale, NLL)", " (NLL)"
    )
    fig.update_yaxes(
        **_adaptive_loss_axis_kwargs(
            collected_y,
            log_title=f"{base_title} (log scale)",
            linear_title=base_title,
        )
    )
    if dropped_labels:
        # The price axis can't place returns-family runs (they don't price
        # quotes). Name them so they aren't silently missing from the frontier.
        shown = ", ".join(dropped_labels[:4])
        if len(dropped_labels) > 4:
            shown += f" +{len(dropped_labels) - 4} more"
        fig.add_annotation(
            text=(
                f"⚠ {len(dropped_labels)} returns-family run(s) not shown on the "
                f"price axis: {shown}"
            ),
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.02,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=11, color=COLORS["text_muted"]),
        )
    return fig


def render_multi_overlaid_loss(
    results: dict, *, visible_models: set[str] | None = None
) -> go.Figure:
    """Loss curve per ``(model, solver, objective)``. Distinct colour per run · dash = solver.

    The y-axis switches to linear when any displayed objective dips
    non-positive (GARCH MLE minimises ``-log_likelihood`` which is
    negative for typical daily returns) — a log axis would silently
    drop those points and leave the overlay looking empty.
    """
    fig = go.Figure()
    colors = _series_colors(results)
    all_y: list[float] = []
    for model_key, solver_name, obj_name, s in _iter_flat(results):
        if visible_models is not None and model_key not in visible_models:
            continue
        if s.result is None or not s.result.iteration_history:
            continue
        col = colors[(model_key, solver_name, obj_name)]
        hist = s.result.iteration_history
        x = np.array([snap.iteration for snap in hist])
        y = np.array([snap.objective for snap in hist])
        all_y.extend(y.tolist())
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
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
