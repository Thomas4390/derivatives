"""Tab — Loss Landscape (P2).

Slices the calibration objective along any pair of model parameters and
overlays every solver's trajectory + multi-start endpoints. Every model
family has a landscape backend (see ``landscape_service.loss_backend``):
FFT surface models and FFT-capable custom models price in closed form, the
MC trio / MC-only custom models price with a reduced common-random-number
budget, and the returns-GARCH family plots its MLE negative log-likelihood.

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

from typing import Any

import numpy as np
import streamlit as st

from charts.landscape import LandscapeLayer, render_landscape
from config.constants import (
    GARCH_FAMILY,
    LANDSCAPE_MC_PATHS,
    LANDSCAPE_RESOLUTION_DEFAULT,
    LANDSCAPE_RESOLUTION_DEFAULT_MC,
    MC_PATHS_INTERACTIVE,
    MODEL_DISPLAY_NAMES,
    RN_GARCH_SURFACE_MODELS,
)
from config.model_registry import get_spec
from services import state_manager
from services.calibration_service import build_objective
from services.model_selection_service import best_slot
from services.landscape_service import (
    compute_loss_grid,
    evaluate_slice_points,
    feller_boundary_segments,
    feller_feasible_window,
    initial_point_from_history,
    is_supported,
    loss_backend,
    mask_nonstationary,
    stationarity_boundary_segments,
    stationarity_label,
    trajectory_points,
)
from streamlit_app.simulation.config.styles import (  # type: ignore
    section_header_html,
)
from tabs._helpers import active_model_picker
from tabs._landscape_panels import (
    render_convergence_at_seed_caption,
    render_pedagogy_panel,
    render_slice_diagnostics,
)


# Default param pair per model — chosen for visual interest. The GARCH
# families default to (α, β): the persistence ridge α+β ≈ 1 (and its
# leverage-adjusted variants) is the pedagogical story of those slices.
_DEFAULT_PAIRS: dict[str, tuple[str, str]] = {
    "heston": ("kappa", "theta"),
    "bates": ("kappa", "rho"),
    "merton": ("lam", "sigma_j"),
    "heston_nandi": ("beta", "gamma"),
    "garch": ("alpha", "beta"),
    "ngarch": ("alpha", "beta"),
    "gjr_garch": ("alpha", "beta"),
    "garch_q": ("alpha", "beta"),
    "ngarch_q": ("alpha", "beta"),
    "gjr_q": ("alpha", "beta"),
}

# Zoom factor (±) applied around the estimated point for B3.
_DEFAULT_ZOOM_FACTOR = 3.0

# The multiplicative ±3×|estimate| zoom is right for scale parameters
# (Heston κ, θ, …) but degenerate for the GARCH persistence parameters:
# β̂ ≈ 0.9 gives a 2.7 half-width that spans the whole [0, 1) domain, so the
# grid cells end up far wider than the NLL valley. For those parameters the
# half-width is capped at a fraction of the spec span. ω (log-scale, a scale
# parameter) keeps the multiplicative rule; heston_nandi's α/γ live on very
# different scales and are excluded on purpose.
_ADDITIVE_ZOOM_MODELS = frozenset(GARCH_FAMILY) | frozenset(RN_GARCH_SURFACE_MODELS)
_PERSISTENCE_ZOOM_PARAMS = frozenset({"alpha", "beta", "gamma"})
_PERSISTENCE_ZOOM_FRAC = 0.15  # half-width as a fraction of the spec span


def _zoom_range(
    estimate: float,
    spec_lo: float,
    spec_hi: float,
    factor: float = _DEFAULT_ZOOM_FACTOR,
    *,
    additive_half: float | None = None,
) -> tuple[float, float]:
    """Build a window around ``estimate`` of half-width ``factor·|estimate|``.

    Clamped to ``[spec_lo, spec_hi]``. Falls back to a fraction of the
    spec range when ``estimate`` is ~0 so we don't end up with a
    zero-width window. ``additive_half`` caps the half-width (persistence
    parameters: a β̂ near 1 must not blow the window up to the full domain).
    """
    half = factor * abs(float(estimate))
    if additive_half is not None:
        half = min(half, float(additive_half))
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


def _objectives_from_labels(labels: set[str], primary: str | None) -> list[str]:
    """Unique objectives among ``solver/objective`` labels, the primary first.

    A legacy label without ``/`` maps to ``primary`` (or ``price_mse``). When the
    ``primary`` objective is no longer represented in the selection, the first
    (sorted) objective becomes the anchor — the caller promotes
    ``calib_active_objective`` to match.
    """
    objs: list[str] = []
    seen: set[str] = set()
    for lbl in sorted(labels):
        obj = lbl.split("/", 1)[1] if "/" in lbl else (primary or "price_mse")
        if obj not in seen:
            seen.add(obj)
            objs.append(obj)
    if primary and primary in objs:
        return [primary] + [o for o in objs if o != primary]
    return objs


def _lmjax_selected_for_objective(
    chosen_set: set[str], obj_name: str, primary_obj: str
) -> bool:
    """True when the user has an LM-JAX run for ``obj_name`` selected.

    Multi-start endpoints are an LM-JAX artefact, so they are shown only while
    an LM-JAX run for their objective is selected. This gates on the **picker
    selection** (``chosen_set``) rather than on the *drawn* trajectory dict: a
    seed that reaches the optimum in one step produces a 1-point path that the
    ``len(xs) > 1`` gate in :func:`_gather_overlays_by_objective` rightly skips,
    but its endpoints stay meaningful and must not vanish with the polyline
    (the synthetic Heston→Heston default case, ATM-IV seed on truth). Labels are
    ``solver/objective`` (nested schema) or a bare solver name (legacy →
    ``primary_obj``).
    """
    return any(
        lbl.split("/", 1)[0].startswith("LM-JAX")
        and (lbl.split("/", 1)[1] if "/" in lbl else primary_obj) == obj_name
        for lbl in chosen_set
    )


def _runs_picker(per_solver: dict, *, key_suffix: str) -> tuple[set[str], list[str]]:
    """ONE multi-pill over the model's successful ``solver/objective`` runs.

    Returns ``(chosen_labels, selected_objectives)``: the chosen runs drive BOTH the
    loss surfaces (their unique objectives, primary first) AND the overlaid solver
    paths. Replaces the old separate "objectives" + "trajectories" pickers — one
    run = its objective's surface + its solver path.
    """
    labels = sorted(
        {lbl for lbl, s in _flatten_per_solver(per_solver) if s.result is not None}
    )
    if not labels:
        return set(), []
    chosen = st.pills(
        "Overlay runs · loss surface + solver path",
        options=labels,
        default=labels,
        selection_mode="multi",
        key=f"landscape_runs_{key_suffix}",
    )
    chosen_set = set(chosen) if chosen else set(labels)
    primary = state_manager.get("calib_active_objective")
    objs = _objectives_from_labels(chosen_set, primary)
    if objs and objs[0] != primary:
        # Promote the anchor when the active objective is no longer selected
        # (mirrors the old active_objectives_picker behaviour).
        state_manager.update(calib_active_objective=objs[0])
    return chosen_set, objs


def _pick_anchor_solver(per_solver: dict) -> tuple[str | None, Any]:
    """Pick the slot whose calibrated point anchors the slice.

    Returns ``(label, summary)`` or ``(None, None)`` when no run succeeded
    yet. Honours the user's current ``active_objective`` when one exists so
    the anchor follows the inspected slice. Scoring is delegated to the shared
    :func:`best_slot`, which never compares ``rmse_price`` (dollars) with
    ``objective_value`` (rss/2 or NLL) — that unit mix used to let a
    failure-branch slot anchor the slice at unfitted parameters.
    """
    active_obj = state_manager.get("calib_active_objective")
    candidates = [
        (label, summary)
        for label, summary in _flatten_per_solver(per_solver)
        # Prefer entries matching the active objective when one is set.
        if not (
            active_obj is not None
            and "/" in label
            and not label.endswith(f"/{active_obj}")
        )
    ]
    chosen = best_slot(candidates)
    if chosen is None:
        return None, None
    return chosen


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


def _exact_overlay_losses(
    eval_fn,
    *,
    model_key: str,
    market_data,
    meta: dict,
    base_params: dict[str, float],
    px: str,
    py: str,
    obj_traj: dict[str, tuple[np.ndarray, np.ndarray]],
    ms: tuple[list[float], list[float]] | None,
    objective,
    objective_key: str,
    initial_point: tuple[float, float] | None,
    true_point: tuple[float, float] | None,
) -> tuple[dict[str, float] | None, np.ndarray | None, float | None, float | None]:
    """Exact slice losses for one layer's marker points (one batched call).

    Covers the final vertex of every trajectory, the multi-start endpoints,
    and — when given — x₀ / ground truth. Exact heights keep the 3-D markers
    on the surface: a nearest-cell lookup could snap a converged endpoint up
    the valley wall (or beyond the stationarity boundary), drawing the final
    fit above the seed.
    """
    labels = list(obj_traj)
    points: list[tuple[float, float]] = [
        (float(obj_traj[lbl][0][-1]), float(obj_traj[lbl][1][-1])) for lbl in labels
    ]
    n_ms = len(ms[0]) if ms is not None else 0
    if n_ms:
        points.extend((float(x), float(y)) for x, y in zip(ms[0], ms[1]))
    if initial_point is not None:
        points.append((float(initial_point[0]), float(initial_point[1])))
    if true_point is not None:
        points.append((float(true_point[0]), float(true_point[1])))
    if not points:
        return None, None, None, None
    losses = eval_fn(
        model_key=model_key,
        market_data=market_data,
        meta=meta,
        base_params=base_params,
        param_x=px,
        param_y=py,
        points=tuple(points),
        objective=objective,
        objective_key=objective_key,
    )
    end_losses = {lbl: float(losses[i]) for i, lbl in enumerate(labels)} or None
    ms_losses = (
        np.asarray(losses[len(labels) : len(labels) + n_ms], dtype=float)
        if n_ms
        else None
    )
    k = len(labels) + n_ms
    init_loss: float | None = None
    true_loss: float | None = None
    if initial_point is not None:
        init_loss = float(losses[k])
        k += 1
    if true_point is not None:
        true_loss = float(losses[k])
    return end_losses, ms_losses, init_loss, true_loss


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
        "Pick two parameters to vary while every other parameter is held at its "
        "current value. The contour (or 3-D surface) shows the calibration loss "
        "under each selected run's objective. Use the **Overlay runs** picker below "
        "to choose which calibration runs to show — each adds its objective's loss "
        "surface and overlays its solver's trajectory across the basin."
    )

    render_pedagogy_panel(model_key_hint=None)

    inspected = active_model_picker(key_suffix="landscape")
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

    backend = loss_backend(model_key)
    nll_mode = backend == "nll"

    spec = get_spec(model_key)
    param_names = [p.name for p in spec.params]
    if len(param_names) < 2:
        st.info("Model has < 2 parameters — nothing to slice.")
        return

    default_x, default_y = _DEFAULT_PAIRS.get(
        model_key,
        (param_names[0], param_names[1]),
    )

    col_x, col_y, col_res = st.columns([2, 2, 1])
    with col_x:
        px = st.selectbox(
            "Parameter X",
            options=param_names,
            index=param_names.index(default_x) if default_x in param_names else 0,
            key=f"landscape_x_{model_key}",
        )
    with col_y:
        py_options = [p for p in param_names if p != px]
        py_default = default_y if default_y in py_options else py_options[0]
        py = st.selectbox(
            "Parameter Y",
            options=py_options,
            index=py_options.index(py_default),
            key=f"landscape_y_{model_key}",
        )
    with col_res:
        resolution = st.select_slider(
            "Grid",
            options=[15, 20, 30, 40, 60],
            value=(
                LANDSCAPE_RESOLUTION_DEFAULT_MC
                if backend == "mc"
                else LANDSCAPE_RESOLUTION_DEFAULT
            ),
            help=(
                "Higher resolution = smoother contour but quadratic compute. "
                "Monte-Carlo models re-price the surface at every cell, so "
                "they default coarser (15 ≈ 5-15 s); FFT / NLL backends run "
                "20 ≈ 1-2 s and 60 ≈ 8-20 s."
            ),
            key=f"landscape_res_{model_key}",
        )
    if backend == "mc":
        st.caption(
            f"🎲 Monte-Carlo pricing at every cell ({LANDSCAPE_MC_PATHS:,} "
            f"paths, fixed seed → common random numbers keep the surface "
            f"smooth and deterministic; the calibrator used "
            f"{MC_PATHS_INTERACTIVE:,} paths). The loss *level* therefore "
            "differs from the solver's objective value — the diagnostics "
            "card below shows both, labelled; the basin shape does not move."
        )

    # ── Pick base_params + anchor solver ─────────────────────────────
    nested = state_manager.get("calib_results") or {}
    per_solver = nested.get(model_key, {})

    # ── ONE overlay picker (replaces the old objectives + trajectories menus) ──
    # Each chosen "solver/objective" run draws its solver path AND ensures its
    # objective's loss surface is shown (surfaces dedupe by objective). Runs BEFORE
    # the anchor pick so a promoted primary objective anchors the slice this rerun.
    chosen_set, selected_objectives = _runs_picker(per_solver, key_suffix=model_key)
    if not selected_objectives:
        selection = state_manager.get("calib_objective_selection") or ()
        selected_objectives = [selection[0] if selection else "price_mse"]
    if nll_mode:
        # Returns-GARCH runs are stored under a pseudo "price_mse" objective
        # key, but their loss IS the MLE negative log-likelihood — one single
        # "nll" surface regardless of how the runs were labelled.
        selected_objectives = ["nll"]
    active_obj_name = selected_objectives[0]

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
    elif data_source == "synthetic" and inspected == generator and true_params:
        for k, v in true_params.items():
            if k in base_params and isinstance(v, (int, float)):
                base_params[k] = float(v)

    px_spec = next(p for p in spec.params if p.name == px)
    py_spec = next(p for p in spec.params if p.name == py)

    # ── Toggles row ──────────────────────────────────────────────────
    opt_a, opt_b, opt_c = st.columns(3)
    with opt_a:
        if nll_mode:
            # The NLL is typically negative, so log₁₀(loss) is meaningless —
            # _safe_log10 would floor the whole grid to a constant.
            log_scale = False
            st.caption("Linear loss scale — the NLL can be negative.")
        else:
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
            anchor_summary.result.iteration_history,
            px,
            py,
        )
    if initial_point is None:
        defaults = spec.true_param_dict()
        initial_point = (float(defaults[px]), float(defaults[py]))

    # Trajectories + multi-start endpoints grouped by objective, plus the
    # bounding box of every overlay coordinate used to fit the slice window.
    traj_by_obj, ms_by_obj, content_x, content_y = _gather_overlays_by_objective(
        per_solver,
        px,
        py,
        active_obj_name,
    )
    if nll_mode:
        # Collapse every stored objective onto the single "nll" surface so the
        # runs' trajectories land on the one layer drawn below.
        traj_by_obj = {
            "nll": {lbl: xy for d in traj_by_obj.values() for lbl, xy in d.items()}
        }
        ms_all_x = [x for xs, _ys in ms_by_obj.values() for x in xs]
        ms_all_y = [y for _xs, ys in ms_by_obj.values() for y in ys]
        ms_by_obj = {"nll": (ms_all_x, ms_all_y)} if ms_all_x else {}
    extra_x = [p[0] for p in (true_point, initial_point) if p is not None]
    extra_y = [p[1] for p in (true_point, initial_point) if p is not None]
    if extra_x:
        content_x = np.concatenate([content_x, np.asarray(extra_x, dtype=float)])
        content_y = np.concatenate([content_y, np.asarray(extra_y, dtype=float)])

    # ── Slice extents — fit the window to the runs so they lie on the surface ──
    def _additive_half(spec, name: str) -> float | None:
        if model_key in _ADDITIVE_ZOOM_MODELS and name in _PERSISTENCE_ZOOM_PARAMS:
            return _PERSISTENCE_ZOOM_FRAC * (float(spec.hi) - float(spec.lo))
        return None

    if full_range:
        x_range = (float(px_spec.lo), float(px_spec.hi))
        y_range = (float(py_spec.lo), float(py_spec.hi))
    elif anchor_summary is None:
        x_range = _zoom_range(
            base_params[px],
            px_spec.lo,
            px_spec.hi,
            additive_half=_additive_half(px_spec, px),
        )
        y_range = _zoom_range(
            base_params[py],
            py_spec.lo,
            py_spec.hi,
            additive_half=_additive_half(py_spec, py),
        )
    else:
        zx_lo, zx_hi = _zoom_range(
            base_params[px],
            px_spec.lo,
            px_spec.hi,
            additive_half=_additive_half(px_spec, px),
        )
        zy_lo, zy_hi = _zoom_range(
            base_params[py],
            py_spec.lo,
            py_spec.hi,
            additive_half=_additive_half(py_spec, py),
        )
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
                    if len(content_x)
                    else (fxlo, fxhi)
                )
                cylo, cyhi = (
                    (float(content_y.min()), float(content_y.max()))
                    if len(content_y)
                    else (fylo, fyhi)
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

    # ── One loss surface per selected objective; assemble overlay layers ──
    # ``chosen_set`` (the runs picked above) selects which trajectories appear.
    # The grid is cached (by model / params / ranges / objective / resolution),
    # so unchanged renders are instant; the spinner covers a genuine recompute.
    objective_settings = state_manager.get("calib_objective_settings") or {}
    eval_points_fn = ctx.get("evaluate_slice_points") or evaluate_slice_points
    initial_point_loss: float | None = None
    true_point_loss: float | None = None
    primary_objective = None
    primary_objective_key = ""
    layers: list[LandscapeLayer] = []
    selected_trajectory_labels: list[str] = []
    with st.spinner(
        f"Evaluating {len(selected_objectives)} loss surface(s) × "
        f"{resolution * resolution} configurations…"
    ):
        for obj_idx, obj_name in enumerate(selected_objectives):
            if nll_mode:
                # The NLL backend evaluates the likelihood directly — there is
                # no ObjectiveStrategy to build.
                objective = None
                objective_key = "nll"
            else:
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
            if model_key in _ADDITIVE_ZOOM_MODELS:
                # GARCH families: blank the non-stationary region (persistence
                # ≥ 1) so its exploding soft-barrier loss doesn't crush the
                # colour/z scale — the dashed boundary curve marks the edge.
                res = mask_nonstationary(res, model_key)
            obj_traj = {
                lbl: xy
                for lbl, xy in traj_by_obj.get(obj_name, {}).items()
                if lbl in chosen_set
            }
            selected_trajectory_labels.extend(obj_traj)
            ms = ms_by_obj.get(obj_name)
            # Multi-start endpoints belong to LM-JAX — keep them while an LM-JAX
            # run for this objective is selected (the picker), NOT while its drawn
            # trajectory survives the ``len(xs) > 1`` gate. See
            # :func:`_lmjax_selected_for_objective` for the one-step-seed case.
            if ms is not None and not _lmjax_selected_for_objective(
                chosen_set, obj_name, active_obj_name
            ):
                ms = None
            # Exact slice losses at the marker points (3-D height override —
            # x₀ / ground truth ride the primary surface, so only that layer
            # evaluates them).
            end_losses, ms_losses, init_loss, true_loss = _exact_overlay_losses(
                eval_points_fn,
                model_key=model_key,
                market_data=market_data,
                meta=meta,
                base_params=base_params,
                px=px,
                py=py,
                obj_traj=obj_traj,
                ms=ms,
                objective=objective,
                objective_key=objective_key,
                initial_point=initial_point if obj_idx == 0 else None,
                true_point=true_point if obj_idx == 0 else None,
            )
            if obj_idx == 0:
                initial_point_loss = init_loss
                true_point_loss = true_loss
                primary_objective = objective
                primary_objective_key = objective_key
            layers.append(
                LandscapeLayer(
                    obj_name,
                    res,
                    obj_traj or None,
                    ms,
                    end_losses,
                    ms_losses,
                )
            )

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
            px,
            py,
            base_params,
            x_range=x_range,
            y_range=y_range,
        )

    # GARCH families (P and Q) + Heston-Nandi: persistence = 1 boundary,
    # clipped to the visible window exactly like the Feller curve.
    stat_curve = stationarity_boundary_segments(
        model_key,
        {p.name: (p.lo, p.hi) for p in spec.params},
        px,
        py,
        base_params,
        x_range=x_range,
        y_range=y_range,
    )

    st.plotly_chart(
        render_landscape(
            layers,
            true_point=true_point,
            initial_point=initial_point,
            feller_curve=feller_curve,
            stationarity_curve=stat_curve,
            stationarity_label=stationarity_label(model_key) or "stationarity boundary",
            log_scale=log_scale,
            view_3d=view_3d,
            initial_point_loss=initial_point_loss,
            true_point_loss=true_point_loss,
        ),
        width="stretch",
        # Stable key so Streamlit keeps the same chart element across the
        # always-on auto-recompute re-renders, letting Plotly's uirevision
        # preserve the camera / zoom. The ``_chart_`` prefix avoids colliding
        # with the "3-D view" toggle's key (landscape_3d_<model>); distinct
        # per view so the 2-D and 3-D figures don't share frontend state.
        key=f"landscape_chart_{'3d' if view_3d else 'contour'}_{model_key}",
    )

    if view_3d and true_point is not None and anchor_summary is not None:
        # The marker's z is a *slice* loss (hidden params frozen at the fit),
        # so after a run the truth generally rides the valley wall — spell
        # that out under the chart so it doesn't read as a plotting bug.
        st.caption(
            f"Ground-truth height = loss at (true {px}, true {py}) with the "
            "remaining parameters frozen at the **fit** — a point on this "
            "slice, not the loss of the full true vector (≈ 0). The valley "
            "floor passes through the fit, so the truth sits up the wall "
            "unless the fit recovered every parameter; the dotted dropline "
            "measures that gap (markers also carry a small constant "
            "anti-clipping lift)."
        )

    render_convergence_at_seed_caption(layers, x_range, y_range)

    # Grid-comparable loss at the anchor's estimate — the same cell loss the
    # primary surface sweeps (cached alongside the marker evaluations). The
    # solver's own objective_value is NOT comparable (rss/2 vs the objective's
    # compute_loss; reduced MC budget), so the card shows both, labelled.
    anchor_slice_loss: float | None = None
    if anchor_summary is not None and anchor_summary.estimated_params:
        est = anchor_summary.estimated_params
        if px in est and py in est:
            anchor_slice_loss = float(
                eval_points_fn(
                    model_key=model_key,
                    market_data=market_data,
                    meta=meta,
                    base_params=base_params,
                    param_x=px,
                    param_y=py,
                    points=((float(est[px]), float(est[py])),),
                    objective=primary_objective,
                    objective_key=primary_objective_key,
                )[0]
            )

    render_slice_diagnostics(
        result=result,
        anchor_summary=anchor_summary,
        anchor_name=anchor_name,
        anchor_slice_loss=anchor_slice_loss,
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
