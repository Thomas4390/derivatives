"""Tab — Loss Landscape (P2).

Slices the calibration objective along any pair of model parameters and
overlays every solver's trajectory + multi-start endpoints. Only
surface models are supported — GARCH MLE has a different objective.

Post-audit fixes (B1-B6) + pedagogical extras (A1-A6) + UX (C3):
  * Initial-guess marker uses the *first evaluation snapshot* from the
    real iteration history (not the registry default).
  * Slice extents default to a zoom of ±3×|estimated| around the
    optimiser's best point, with a toggle for the full spec range.
  * Synthetic-mode pre-run uses ``true_params`` as the slice base.
  * Caveat caption explains the 2-D slice approximation.
  * Slice-minimum coordinates / loss are reported numerically.
  * Basin curvature (Hessian eigenvalues + condition number) printed.
  * 3-D surface toggle complements the contour view.
  * Progress bar for resolution ≥ 40.
  * Auto-recompute toggle + "center on solver optimum" toggle.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import streamlit as st

from charts.landscape import LandscapeLayer, render_landscape
from config.constants import MODEL_DISPLAY_NAMES
from config.model_registry import get_spec
from services import state_manager
from services.calibration_service import build_objective
from services.landscape_service import (
    basin_curvature,
    compute_loss_grid,
    feller_boundary_segments,
    feller_feasible_window,
    initial_point_from_history,
    is_supported,
    slice_minimum,
    trajectory_points,
)
from streamlit_app.simulation.config.styles import (  # type: ignore
    render_stats_row,
    section_header_html,
)
from tabs._helpers import active_model_picker, active_objectives_picker


# Default param pair per model — chosen for visual interest.
_DEFAULT_PAIRS: dict[str, tuple[str, str]] = {
    "heston": ("kappa", "theta"),
    "bates": ("kappa", "rho"),
    "merton": ("lam", "sigma_j"),
}

# Zoom factor (±) applied around the estimated point for B3.
_DEFAULT_ZOOM_FACTOR = 3.0


def _zoom_range(
    estimate: float,
    spec_lo: float,
    spec_hi: float,
    factor: float = _DEFAULT_ZOOM_FACTOR,
) -> tuple[float, float]:
    """Build a window around ``estimate`` of half-width ``factor·|estimate|``.

    Clamped to ``[spec_lo, spec_hi]``. Falls back to a fraction of the
    spec range when ``estimate`` is ~0 so we don't end up with a
    zero-width window.
    """
    half = factor * abs(float(estimate))
    if half < 1e-12:
        half = 0.25 * (spec_hi - spec_lo)
    lo = max(spec_lo, estimate - half)
    hi = min(spec_hi, estimate + half)
    if hi - lo < 1e-9:
        # Estimate sat on a boundary — widen toward the interior.
        if estimate <= spec_lo + 1e-9:
            hi = min(spec_hi, spec_lo + half * 2.0)
            lo = spec_lo
        else:
            lo = max(spec_lo, spec_hi - half * 2.0)
            hi = spec_hi
    return float(lo), float(hi)


def _fit_window(
    zoom_lo: float,
    zoom_hi: float,
    content: np.ndarray,
    spec_lo: float,
    spec_hi: float,
    pad: float = 0.05,
) -> tuple[float, float]:
    """Widen ``[zoom_lo, zoom_hi]`` to also cover every coordinate in
    ``content`` (solver trajectories, endpoints, markers), so overlays sit on
    the surface instead of floating off it. Adds a small padding and clamps to
    the spec bounds. The zoom window is the *minimum* extent — a tight basin
    stays tight when the runs don't escape it.
    """
    lo, hi = float(zoom_lo), float(zoom_hi)
    if content is not None and len(content):
        finite = content[np.isfinite(content)]
        if finite.size:
            lo = min(lo, float(finite.min()))
            hi = max(hi, float(finite.max()))
    span = hi - lo
    if span > 0:
        lo -= pad * span
        hi += pad * span
    lo = max(float(spec_lo), lo)
    hi = min(float(spec_hi), hi)
    return float(lo), float(hi)


def _render_pedagogy_panel(model_key_hint: str | None) -> None:
    """Pedagogical explainer rendered above the chart.

    Collapsed by default so the panel does not push the chart below the
    fold for repeat users, but always available for first-timers. The
    content focuses on *reading* the chart — what every colour, marker,
    and *empty* region means — and on the subtle but important fact that
    a 2-D slice is not the full objective.
    """
    with st.expander(
        "📖 How to read this chart (read once before exploring)",
        expanded=False,
    ):
        st.markdown(
            """
**What is a loss landscape?**

Calibration tunes the model parameters $\\theta$ to reproduce the market.
For every candidate $\\theta$, we measure how badly the model misprices the
surface — that number is the **loss** $L(\\theta)$. The set of all
$(\\theta, L(\\theta))$ pairs is a high-dimensional landscape; calibration
is the descent from a starting guess down into a valley.

For a model with $p$ parameters, the landscape lives in $\\mathbb{R}^p$ and
you cannot see it. **What this tab shows is a 2-D *slice*** of that
landscape: two parameters vary on the $(x, y)$ axes, every other parameter
is **frozen** at the values listed at the bottom of the chart.

**How to read the colour map**

- **Dark blue / purple** → low loss → model fits the market well.
- **Yellow / red** → high loss → model misprices the market.
- The **valley** (lowest dark region) is the slice's local optimum.

**The overlays**

| Marker | Meaning |
|---|---|
| 🟢 *Star (green)* | Synthetic-mode ground truth — the parameters that **generated** the data. The solver should land here. |
| 🟠 *Diamond (orange)* | The solver's **initial guess** $x_0$ — typically a heuristic (ATM IV for Heston, etc.). |
| 🔴 *Polyline (red)* | The **trajectory** — every step the optimiser made, from $x_0$ to its final answer. |
| 🟣 *Dots (purple)* | **Multi-start endpoints** (LM-JAX only) — final point of each restart. They tell you whether all restarts agreed. |
| ⚪ *White curve* | The **Feller boundary** $2\\kappa\\sigma^2 = \\alpha^2$ (Heston/Bates only). Crossing it makes variance negative. |

**Why doesn't the colour cover the whole rectangle?**

The blank / white regions are **not bugs** — they are **infeasible
regions** of parameter space:

1. **Feller violation (Heston/Bates).** If $2\\kappa\\sigma^2 < \\alpha^2$, the
   variance process can hit zero in finite time and the model is rejected.
   The pricer returns `NaN`, the contour stays blank.
2. **Pricing failures.** Some configurations make the characteristic
   function or the FFT inversion diverge (e.g. $\\alpha \\to 0$, extreme
   correlations $|\\rho| \\to 1$). The cell is left as `NaN` and stays
   blank.
3. **Numerical exceptions** (overflow, division by zero in the closed-form
   formulas) are caught and reported as `NaN` so a single bad cell does
   not crash the whole grid.

This is also why **the contour map is sometimes very narrow**: the user
toggle "Full spec range" is OFF by default and we zoom **±3 ×|estimate|**
around the solver's best point — a tight window where the curvature is
visible. Turn the toggle ON to see the full bounded region, including
how much of it is actually infeasible.

**Caveat: 2-D slice ≠ full optimisation problem**

The solver does not walk along this slice. It walks in $\\mathbb{R}^p$,
varying *all* parameters at once. The trajectory you see is **projected**
onto the $(x, y)$ plane. A point that looks like a yellow plateau here
may actually be a deep valley in the full $p$-dimensional space because
the hidden parameters were different at that step. Use this chart to
build intuition, not to second-guess the optimiser.
"""
        )


def _flatten_per_solver(per_solver: dict):
    """Yield ``(label, summary)`` pairs handling both legacy and nested schemas.

    ``label`` is the bare solver name for the legacy ``Summary`` shape, and
    ``solver/objective`` for the nested ``dict[objective, Summary]`` shape —
    including the default ``price_mse`` (so an overlaid price_mse trajectory
    reads as ``LM-JAX/price_mse`` instead of an unlabelled ``LM-JAX``).
    """
    for solver_name, slot in per_solver.items():
        if isinstance(slot, dict):
            for obj_name, s in slot.items():
                yield f"{solver_name}/{obj_name}", s
        else:
            yield solver_name, slot


def _pick_anchor_solver(per_solver: dict) -> tuple[str | None, Any]:
    """Pick the slot whose calibrated point anchors the slice (best RMSE).

    Returns ``(label, summary)`` or ``(None, None)`` when no run
    succeeded yet. Honours the user's current ``active_objective`` when
    one exists so the anchor follows the inspected slice.
    """
    active_obj = state_manager.get("calib_active_objective")
    best_name: str | None = None
    best_summary = None
    best_score = math.inf
    for label, summary in _flatten_per_solver(per_solver):
        if summary.result is None:
            continue
        # Prefer entries matching the active objective when one is set.
        if active_obj is not None and "/" in label and not label.endswith(f"/{active_obj}"):
            continue
        rmse = summary.result.rmse_price
        score = float(rmse) if rmse is not None else float(summary.result.objective_value)
        if score < best_score:
            best_score = score
            best_name = label
            best_summary = summary
    return best_name, best_summary


def _gather_overlays_by_objective(per_solver: dict, px: str, py: str, primary_obj: str):
    """Group solver trajectories + multi-start endpoints **by objective**.

    Returns ``(traj_by_obj, ms_by_obj, content_x, content_y)`` where
    ``traj_by_obj[obj][label] = (xs, ys)``, ``ms_by_obj[obj] = (xs, ys)`` and
    ``content_x/_y`` collect every overlay coordinate (used to fit the slice
    window so the runs sit on the surface). Labels are ``solver/objective``
    (nested schema) or the bare solver name (legacy → ``primary_obj``).
    """
    traj_by_obj: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    ms_by_obj: dict[str, tuple[list[float], list[float]]] = {}
    cx: list[float] = []
    cy: list[float] = []
    for label, summary in _flatten_per_solver(per_solver):
        if summary.result is None:
            continue
        obj = label.split("/", 1)[1] if "/" in label else primary_obj
        xs, ys = trajectory_points(summary.result.iteration_history, px, py)
        if len(xs) > 1:
            traj_by_obj.setdefault(obj, {})[label] = (xs, ys)
            cx.extend(xs.tolist())
            cy.extend(ys.tolist())
        if label.split("/", 1)[0].startswith("LM-JAX"):
            diag = summary.result.diagnostics or {}
            ms_xs: list[float] = []
            ms_ys: list[float] = []
            for history in diag.get("multi_start_history") or ():
                if not history:
                    continue
                final = getattr(history[-1], "params_natural", None) or {}
                if px in final and py in final:
                    ms_xs.append(float(final[px]))
                    ms_ys.append(float(final[py]))
            if ms_xs:
                ms_by_obj[obj] = (ms_xs, ms_ys)
                cx.extend(ms_xs)
                cy.extend(ms_ys)
    return (
        traj_by_obj,
        ms_by_obj,
        np.asarray(cx, dtype=float),
        np.asarray(cy, dtype=float),
    )


def render(ctx: dict) -> None:
    ensure_data = ctx["ensure_data"]
    data_source = ctx["data_source"]
    true_params = ctx.get("true_params") or {}
    generator = ctx.get("generator_model")
    candidates = ctx.get("candidate_models") or ()
    compute_grid_fn = ctx.get("compute_loss_grid") or compute_loss_grid

    st.markdown(
        section_header_html("🗺️", "Loss landscape · 2-D slice"),
        unsafe_allow_html=True,
    )
    st.caption(
        "Pick two parameters to vary while every other parameter is held at "
        "its current value. The contour (or 3-D surface) shows the calibration "
        "loss under the **selected objective** (switch it with the picker "
        "below). Overlay the optimiser's trajectory to see how each solver "
        "climbs out of (or settles into) the basin."
    )

    _render_pedagogy_panel(model_key_hint=None)

    inspected = active_model_picker(key_suffix="landscape")
    # The objective picker ties the inspected slice to the user's active
    # objective(s): the contour values, the anchor, and the axis label all
    # follow them. Multi-select overlays one loss surface per chosen
    # objective; the first (primary) drives the anchor + base params. Fall
    # back to the first selected objective (then price_mse) when no run has
    # produced multiple objectives yet.
    selected_objectives = active_objectives_picker(key_suffix="landscape")
    if not selected_objectives:
        selection = state_manager.get("calib_objective_selection") or ()
        selected_objectives = [selection[0] if selection else "price_mse"]
    active_obj_name = selected_objectives[0]
    model_key = inspected or generator or (candidates[0] if candidates else None)
    if model_key is None:
        st.info("Pick at least one candidate model in the sidebar.")
        return

    ok, reason = is_supported(model_key)
    if not ok:
        st.info(
            f"Landscape unavailable for "
            f"{MODEL_DISPLAY_NAMES.get(model_key, model_key)}: {reason}"
        )
        return

    market_data, meta = ensure_data()
    if market_data is None:
        st.info("Data unavailable — fix the sidebar config first.")
        return

    spec = get_spec(model_key)
    param_names = [p.name for p in spec.params]
    if len(param_names) < 2:
        st.info("Model has < 2 parameters — nothing to slice.")
        return

    default_x, default_y = _DEFAULT_PAIRS.get(
        model_key, (param_names[0], param_names[1]),
    )

    col_x, col_y, col_res = st.columns([2, 2, 1])
    with col_x:
        px = st.selectbox(
            "Parameter X", options=param_names,
            index=param_names.index(default_x) if default_x in param_names else 0,
            key=f"landscape_x_{model_key}",
        )
    with col_y:
        py_options = [p for p in param_names if p != px]
        py_default = default_y if default_y in py_options else py_options[0]
        py = st.selectbox(
            "Parameter Y", options=py_options,
            index=py_options.index(py_default),
            key=f"landscape_y_{model_key}",
        )
    with col_res:
        resolution = st.select_slider(
            "Grid",
            options=[15, 20, 30, 40, 60],
            value=20,
            help=(
                "Higher resolution = smoother contour but quadratic compute. "
                "20 ≈ 1-2 s per Heston slice; 60 ≈ 8-15 s."
            ),
            key=f"landscape_res_{model_key}",
        )

    # ── Pick base_params + anchor solver ─────────────────────────────
    nested = state_manager.get("calib_results") or {}
    per_solver = nested.get(model_key, {})
    anchor_name, anchor_summary = _pick_anchor_solver(per_solver)

    # Default base_params:
    #   1. If a solver ran on this model → use its estimate (B6 partial).
    #   2. Else if synthetic mode AND inspected == generator AND we have
    #      true_params → use those (B6 main fix).
    #   3. Else → spec defaults.
    base_params: dict[str, float] = {p.name: float(p.default) for p in spec.params}
    if anchor_summary is not None and anchor_summary.estimated_params:
        for k, v in anchor_summary.estimated_params.items():
            if k in base_params:
                base_params[k] = float(v)
    elif (
        data_source == "synthetic"
        and inspected == generator
        and true_params
    ):
        for k, v in true_params.items():
            if k in base_params and isinstance(v, (int, float)):
                base_params[k] = float(v)

    px_spec = next(p for p in spec.params if p.name == px)
    py_spec = next(p for p in spec.params if p.name == py)

    # ── Toggles row ──────────────────────────────────────────────────
    opt_a, opt_b, opt_c = st.columns(3)
    with opt_a:
        log_scale = st.toggle(
            "Log scale",
            value=True,
            help="Show log₁₀(loss). Reveals the basin near the optimum.",
            key=f"landscape_log_{model_key}",
        )
    with opt_b:
        full_range = st.toggle(
            "Full spec range",
            value=False,
            help=(
                "Off: zoom around the estimated point (±3×|estimate|). "
                "On: use the full parameter bounds from the registry."
            ),
            key=f"landscape_full_range_{model_key}",
        )
    with opt_c:
        view_3d = st.toggle(
            "3-D view",
            value=False,
            help="Swap the contour for a Plotly 3-D surface. The basin "
                 "becomes a literal valley.",
            key=f"landscape_3d_{model_key}",
        )

    # ── Overlays (gathered BEFORE the window so it can be fit to them) ──
    true_point = None
    if data_source == "synthetic" and inspected == generator and true_params:
        if px in true_params and py in true_params:
            true_point = (float(true_params[px]), float(true_params[py]))

    # B1: extract real initial guess from history when available; fall
    # back to the spec default (B1 → B6 chain) when no solver ran.
    initial_point: tuple[float, float] | None = None
    if anchor_summary is not None and anchor_summary.result is not None:
        initial_point = initial_point_from_history(
            anchor_summary.result.iteration_history, px, py,
        )
    if initial_point is None:
        defaults = spec.true_param_dict()
        initial_point = (float(defaults[px]), float(defaults[py]))

    # Trajectories + multi-start endpoints grouped by objective, plus the
    # bounding box of every overlay coordinate used to fit the slice window.
    traj_by_obj, ms_by_obj, content_x, content_y = _gather_overlays_by_objective(
        per_solver, px, py, active_obj_name,
    )
    extra_x = [p[0] for p in (true_point, initial_point) if p is not None]
    extra_y = [p[1] for p in (true_point, initial_point) if p is not None]
    if extra_x:
        content_x = np.concatenate([content_x, np.asarray(extra_x, dtype=float)])
        content_y = np.concatenate([content_y, np.asarray(extra_y, dtype=float)])

    # ── Slice extents — fit the window to the runs so they lie on the surface ──
    if full_range:
        x_range = (float(px_spec.lo), float(px_spec.hi))
        y_range = (float(py_spec.lo), float(py_spec.hi))
    elif anchor_summary is None:
        x_range = _zoom_range(base_params[px], px_spec.lo, px_spec.hi)
        y_range = _zoom_range(base_params[py], py_spec.lo, py_spec.hi)
    else:
        zx_lo, zx_hi = _zoom_range(base_params[px], px_spec.lo, px_spec.hi)
        zy_lo, zy_hi = _zoom_range(base_params[py], py_spec.lo, py_spec.hi)
        x_range = _fit_window(zx_lo, zx_hi, content_x, px_spec.lo, px_spec.hi)
        y_range = _fit_window(zy_lo, zy_hi, content_y, py_spec.lo, py_spec.hi)
        # CIR models: crop the window onto the Feller-feasible region so the
        # infeasible band (NaN → blank) doesn't waste the plot — most visible in
        # Feller-HARD where the optimum sits on the boundary. Never crop away an
        # overlaid point (HARD trajectories are feasible; SOFT/OFF may stray, so
        # we keep them visible by unioning the feasible box with the content).
        if model_key in ("heston", "bates"):
            fb = feller_feasible_window(px, py, base_params, x_range, y_range)
            if fb is not None:
                fxlo, fxhi, fylo, fyhi = fb
                cxlo, cxhi = (
                    (float(content_x.min()), float(content_x.max()))
                    if len(content_x) else (fxlo, fxhi)
                )
                cylo, cyhi = (
                    (float(content_y.min()), float(content_y.max()))
                    if len(content_y) else (fylo, fyhi)
                )
                nxlo, nxhi = min(fxlo, cxlo), max(fxhi, cxhi)
                nylo, nyhi = min(fylo, cylo), max(fyhi, cyhi)
                if nxhi - nxlo > 1e-9 and nyhi - nylo > 1e-9:
                    x_range, y_range = (nxlo, nxhi), (nylo, nyhi)

    # Diagnostic: overlay points still outside the (fitted) window — should be
    # ~0 now; the warning below only fires if a spec-bound clamp prevents full
    # coverage.
    out_of_range = 0
    if len(content_x):
        ox = (content_x < x_range[0]) | (content_x > x_range[1])
        oy = (content_y < y_range[0]) | (content_y > y_range[1])
        out_of_range = int(np.count_nonzero(ox | oy))

    # ── Trajectory overlay picker (across all selected objectives) ──────
    all_traj_labels = sorted({lbl for d in traj_by_obj.values() for lbl in d})
    chosen_set = set(all_traj_labels)
    if all_traj_labels:
        chosen = st.pills(
            "Overlay solver trajectories",
            options=all_traj_labels,
            default=all_traj_labels,
            selection_mode="multi",
            key=f"landscape_traj_{model_key}",
        )
        chosen_set = set(chosen) if chosen else set()

    # ── One loss surface per selected objective; assemble overlay layers ──
    # The grid is cached (by model / params / ranges / objective / resolution),
    # so unchanged renders are instant; the spinner covers a genuine recompute.
    objective_settings = state_manager.get("calib_objective_settings") or {}
    layers: list[LandscapeLayer] = []
    selected_trajectory_labels: list[str] = []
    with st.spinner(
        f"Pricing {len(selected_objectives)} surface(s) × "
        f"{resolution * resolution} configurations…"
    ):
        for obj_name in selected_objectives:
            objective = build_objective(obj_name, objective_settings)
            objective_key = f"{obj_name}|" + "|".join(
                f"{k}={objective_settings[k]}" for k in sorted(objective_settings)
            )
            res = compute_grid_fn(
                model_key=model_key,
                market_data=market_data,
                meta=meta,
                base_params=base_params,
                param_x=px,
                param_y=py,
                x_range=x_range,
                y_range=y_range,
                objective=objective,
                objective_key=objective_key,
                resolution=int(resolution),
            )
            obj_traj = {
                lbl: xy
                for lbl, xy in traj_by_obj.get(obj_name, {}).items()
                if lbl in chosen_set
            }
            selected_trajectory_labels.extend(obj_traj)
            ms = ms_by_obj.get(obj_name)
            # Multi-start endpoints belong to LM-JAX — drop them when that
            # trajectory is deselected so the overlay stays self-consistent.
            if ms is not None and not any(lbl.startswith("LM-JAX") for lbl in obj_traj):
                ms = None
            layers.append(LandscapeLayer(obj_name, res, obj_traj or None, ms))

    # Primary surface drives the downstream slice-minimum / curvature panels.
    result = layers[0].result
    # Kept for the legend caption below (list of overlaid trajectory labels).
    solver_trajectories = selected_trajectory_labels

    feller_curve = None
    if model_key in ("heston", "bates"):
        # Clip the boundary to the visible (fitted) window so it stays on the
        # surface instead of running off to the full parameter bounds.
        feller_curve = feller_boundary_segments(
            {p.name: (p.lo, p.hi) for p in spec.params},
            px, py, base_params,
            x_range=x_range, y_range=y_range,
        )

    st.plotly_chart(
        render_landscape(
            layers,
            true_point=true_point,
            initial_point=initial_point,
            feller_curve=feller_curve,
            log_scale=log_scale,
            view_3d=view_3d,
        ),
        width="stretch",
        # Stable key so Streamlit keeps the same chart element across the
        # always-on auto-recompute re-renders, letting Plotly's uirevision
        # preserve the camera / zoom. The ``_chart_`` prefix avoids colliding
        # with the "3-D view" toggle's key (landscape_3d_<model>); distinct
        # per view so the 2-D and 3-D figures don't share frontend state.
        key=f"landscape_chart_{'3d' if view_3d else 'contour'}_{model_key}",
    )

    _render_convergence_at_seed_caption(layers, x_range, y_range)

    _render_slice_diagnostics(
        result=result,
        anchor_summary=anchor_summary,
        anchor_name=anchor_name,
        px=px,
        py=py,
        out_of_range=out_of_range,
        solver_trajectories=solver_trajectories,
        base_params=base_params,
        param_names=param_names,
        data_source=data_source,
        inspected=inspected,
        generator=generator,
    )


# Below this fraction of the slice window, a solver trajectory is
# indistinguishable from the x₀ marker and the user sees no path at
# all. Two-axis tolerance (max of relative spans) so we only fire when
# *both* parameters barely moved.
_DEGENERATE_TRAJECTORY_THRESHOLD: float = 0.01


def _render_convergence_at_seed_caption(
    layers: list[LandscapeLayer],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    """Tell the student why a solver path can shrink into the seed marker.

    LM-JAX seeded from the ATM-IV inverse converges almost instantly when
    the synthetic surface uses the default true parameters — the path
    spans a fraction of a percent of the window and disappears under x₀.
    Flag the case explicitly so they don't read it as a bug.
    """
    window_x = max(abs(x_range[1] - x_range[0]), 1e-12)
    window_y = max(abs(y_range[1] - y_range[0]), 1e-12)
    degenerate: list[tuple[str, int]] = []
    for layer in layers:
        for solver_name, (xs, ys) in (layer.solver_trajectories or {}).items():
            if len(xs) < 2:
                continue
            rel_x = float(np.ptp(xs)) / window_x
            rel_y = float(np.ptp(ys)) / window_y
            if max(rel_x, rel_y) < _DEGENERATE_TRAJECTORY_THRESHOLD:
                degenerate.append((solver_name, len(xs)))
    if not degenerate:
        return
    names = ", ".join(f"`{name}` ({nfev} evals)" for name, nfev in degenerate)
    st.caption(
        f"ℹ️ {names} converged right at the seed. On synthetic surfaces "
        f"the ATM-IV-derived x₀ already sits inside the basin, so the "
        f"trajectory spans less than 1% of the window and hides under "
        f"the x₀ marker. Change the true parameters in the sidebar to "
        f"displace the optimum and watch the path stretch out."
    )


# Basin condition-number verdict → stats-card colour variant.
_KAPPA_VARIANT: dict[str, str] = {
    "well-conditioned": "teal",
    "moderately elongated": "amber",
    "highly elongated": "red",
}


def _render_slice_diagnostics(
    *,
    result: Any,
    anchor_summary: Any,
    anchor_name: str | None,
    px: str,
    py: str,
    out_of_range: int,
    solver_trajectories: list[str],
    base_params: dict[str, float],
    param_names: list[str],
    data_source: str,
    inspected: str,
    generator: str | None,
) -> None:
    """Render the slice-diagnostics panel below the landscape chart.

    A coloured stat-card row (slice minimum, solver optimum, the gap between
    them, and the basin condition number κ(H)), a one-line legend, and a
    collapsible explainer holding the 2-D-slice caveat and the anchoring rule.
    Pure presentation — every value is read from already-computed objects.
    """
    minimum = slice_minimum(result)

    anchor_loss: float | None = None
    anchor_coords: tuple[float, float] | None = None
    if anchor_summary is not None and anchor_summary.estimated_params:
        est = anchor_summary.estimated_params
        if px in est and py in est:
            anchor_coords = (float(est[px]), float(est[py]))
        if anchor_summary.result is not None and anchor_summary.result.rmse_iv is not None:
            anchor_loss = float(anchor_summary.result.rmse_iv) ** 2  # variance proxy

    curvature = basin_curvature(result)

    # ── Stat cards — built dynamically; render_stats_row fits the columns ──
    stats: list[tuple[str, str, str]] = []
    variants: list[str] = []
    if minimum is not None:
        xm, ym, lm = minimum
        stats.append(
            ("Slice minimum", f"{lm:.3e}", f"{px} = {xm:.3g} · {py} = {ym:.3g}")
        )
        variants.append("blue")
    if anchor_name is not None and anchor_coords is not None:
        loss_val = f"{anchor_loss:.2e}" if anchor_loss is not None else "—"
        stats.append(
            (
                f"{anchor_name} optimum",
                loss_val,
                f"{px} = {anchor_coords[0]:.3g} · {py} = {anchor_coords[1]:.3g}",
            )
        )
        variants.append("green")
    if minimum is not None and anchor_coords is not None:
        dx = abs(anchor_coords[0] - minimum[0])
        dy = abs(anchor_coords[1] - minimum[1])
        stats.append(("Min ↔ optimum gap", f"Δ{px} = {dx:.3g}", f"Δ{py} = {dy:.3g}"))
        variants.append("amber")
    if curvature is not None:
        kappa = curvature["kappa"]
        lam_min = curvature["lambda_min"]
        lam_max = curvature["lambda_max"]
        verdict = (
            "well-conditioned" if kappa < 5
            else "moderately elongated" if kappa < 30
            else "highly elongated"
        )
        stats.append(
            (
                "Condition κ(H)",
                f"{kappa:,.0f}",
                f"{verdict} · λ ∈ [{lam_min:.1e}, {lam_max:.1e}]",
            )
        )
        variants.append(_KAPPA_VARIANT[verdict])

    if stats:
        st.markdown(
            section_header_html("📐", "Slice diagnostics"), unsafe_allow_html=True
        )
        render_stats_row(stats, variants)

    # ── Out-of-window trajectory warning ──────────────────────────────
    if out_of_range > 0:
        st.caption(
            f"⚠ {out_of_range} trajectory point(s) fall outside the slice "
            "window — toggle **Full spec range** above to widen the view."
        )

    # ── Legend (the map key — stays visible) ──────────────────────────
    overlaid = ", ".join(solver_trajectories) if solver_trajectories else "none selected"
    st.caption(
        "Legend: 🟢 ground truth (synthetic) · 🟠 initial guess (x₀) · "
        "🔴 solver trajectory · 🟣 multi-start endpoints (LM-JAX) · "
        f"⚪ Feller boundary.  Trajectories overlaid: {overlaid}."
    )

    # ── Collapsible explainer: slice caveat + anchoring rule ──────────
    with st.expander("ℹ️ How to read this slice", expanded=False):
        if minimum is not None and anchor_coords is not None:
            st.markdown(
                "Large gaps between the **slice minimum** and the **solver "
                "optimum** mean the basin curves through the *hidden* parameters."
            )
        if curvature is not None and curvature["kappa"] > 30:
            st.markdown(
                "Gradient-free solvers (NM, DE) struggle when the condition "
                "number κ(H) ≫ 1 — the basin is a long, narrow valley."
            )
        hidden_str = ", ".join(
            f"`{n}` = {base_params[n]:.4f}"
            for n in param_names if n not in (px, py)
        )
        st.markdown(
            "**⚠ This is a 2-D slice of a high-dimensional objective.** The "
            f"contour shows the loss with hidden parameters frozen at "
            f"{hidden_str or '(none)'}. Solver trajectories pass through "
            "*different* hidden-param values along the way, so the true loss at "
            "a trajectory point is generally lower than what the contour reads "
            "at the same (x, y) location."
        )
        if anchor_name is not None:
            stopped = " *(best-so-far — solver was stopped)*" if getattr(
                anchor_summary, "partial", False
            ) else ""
            st.markdown(
                f"Slice anchored at the **{anchor_name}** estimated point.{stopped} "
                "Toggle **Center on solver optimum** is implicit here — the "
                f"hidden parameters and zoom window are derived from {anchor_name}'s fit."
            )
        elif data_source == "synthetic" and inspected == generator:
            st.markdown(
                "Slice anchored at the **synthetic ground truth** parameters "
                "(no calibration has run yet). Run a calibration to anchor the "
                "slice at the optimiser's optimum instead."
            )
        else:
            st.markdown(
                "Slice anchored at the model's **registry defaults** (no run "
                "yet). Run a calibration to anchor the slice at the optimum."
            )
