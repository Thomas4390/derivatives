"""Loss-landscape computation for the Loss Landscape tab.

Given a surface model and two of its parameters, sweep a 2-D meshgrid
fixing every other parameter at the user-provided base point. At each
``(p_x, p_y)`` cell, re-price the model on the market grid and record
the loss under the **chosen calibration objective** (the same
``ObjectiveStrategy`` minimised by the solver). The result is rendered
as a contour plot with the optimiser's trajectory overlaid.

Only surface models are supported — GARCH MLE has a different objective
(needs the variance filter), so the tab degrades gracefully there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.calibration.objectives import ObjectiveStrategy
from backend.calibration.pricing_loop import price_surface
from config.constants import SURFACE_FAMILY
from config.model_registry import get_spec
from services.post_calibration import _engine, rebuild_model


@dataclass(frozen=True)
class LandscapeResult:
    param_x: str
    param_y: str
    x_values: np.ndarray
    y_values: np.ndarray
    loss_grid: np.ndarray            # shape (n_y, n_x), NaN for failures
    base_params: dict[str, float]    # frozen at compute time


def is_supported(model_key: str) -> tuple[bool, str | None]:
    """Return ``(ok, reason_if_not)``. The tab uses this to short-circuit."""
    if model_key not in SURFACE_FAMILY:
        return False, (
            "Loss-landscape view is only available for surface models. "
            "GARCH-family losses depend on the variance filter and don't "
            "decompose cleanly into a (p_x, p_y) slice."
        )
    if model_key in ("ngarch_q", "garch_q", "gjr_q"):
        return False, (
            "Nonaffine risk-neutral GARCH models price by Monte-Carlo — re-pricing "
            "the whole surface by MC at every one of the ~900 grid cells is too "
            "costly for an interactive landscape (the affine Heston-Nandi has one "
            "because it prices in closed form)."
        )
    if model_key == "iv_gbm":
        return False, "iv_gbm has a single parameter — nothing to slice."
    spec = get_spec(model_key)
    if spec.n_params < 2:
        return False, "Need ≥ 2 parameters to render a 2-D slice."
    return True, None


def _objective_loss(
    model_key: str,
    params: dict[str, float],
    market_data,
    objective: ObjectiveStrategy,
) -> float:
    """Loss of the model at ``params`` under the chosen objective.

    Prices the surface once, then delegates the residual shaping /
    weighting to ``objective.compute_loss`` so the contour reflects the
    very quantity the solver minimised (price MSE, IV MSE, vega-weighted,
    Huber, …).
    """
    model = rebuild_model(model_key, params)
    if model is None:
        return float("nan")
    prices = price_surface(model, market_data, _engine())
    return float(objective.compute_loss(prices, market_data))


def compute_loss_grid(
    model_key: str,
    market_data,
    meta: dict,
    base_params: dict[str, float],
    param_x: str,
    param_y: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    objective: ObjectiveStrategy,
    resolution: int = 30,
    objective_key: str | None = None,
) -> LandscapeResult:
    """Sweep a ``resolution × resolution`` grid of ``(p_x, p_y)`` values.

    ``base_params`` provides the held-fixed coordinates; ``x_range`` and
    ``y_range`` set the slice extents. ``objective`` is the
    ``ObjectiveStrategy`` whose loss is evaluated at every cell — pass the
    same strategy the user selected so the landscape matches the solver's
    target. NaN cells correspond to model invocations that failed (e.g.
    Feller-violating Heston configurations). ``meta`` is retained for call
    compatibility with the cached wrapper; ``objective_key`` is a cache-key
    hint consumed by that wrapper and ignored by the direct computation.
    """
    if param_x == param_y:
        raise ValueError("param_x and param_y must differ to form a 2-D slice")
    x_values = np.linspace(float(x_range[0]), float(x_range[1]), int(resolution))
    y_values = np.linspace(float(y_range[0]), float(y_range[1]), int(resolution))

    base = dict(base_params)

    loss = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
    for j, yv in enumerate(y_values):
        for i, xv in enumerate(x_values):
            params = dict(base)
            params[param_x] = float(xv)
            params[param_y] = float(yv)
            try:
                loss[j, i] = _objective_loss(model_key, params, market_data, objective)
            except (ValueError, RuntimeError, FloatingPointError):
                loss[j, i] = float("nan")

    return LandscapeResult(
        param_x=param_x,
        param_y=param_y,
        x_values=x_values,
        y_values=y_values,
        loss_grid=loss,
        base_params=dict(base_params),
    )


def trajectory_points(
    history, param_x: str, param_y: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the ``(p_x, p_y)`` optimiser path from an iteration history.

    Prefers the per-iteration *best* (``source == "callback"``) snapshots when
    present (≥2), falling back to the full history otherwise. This matters for
    population solvers (DE): their per-evaluation snapshots are a cloud that
    spans the whole bound box, which — since this polyline also feeds the
    window-fit — would blow the landscape window up to the full bounds and
    flatten the surface. The callback snapshots track the population best and
    give the actual descent path. (Opposite of the convergence chart's filter,
    which deliberately shows every evaluation.) LM has no callback → falls back
    to its tightly-converging evaluation path.

    When we fall through to the callback path, the *first* callback fires
    **after** iteration 1 — already off the seed x₀ — so a polyline drawn
    from ``cb[0]`` would leave a visual gap to the ``initial guess (x₀)``
    marker rendered separately, and the enlarged first callback dot would
    read as a duplicate starting point. To close that gap we prepend the
    very first ``source == "evaluation"`` snapshot (the seed the wrapped
    objective sees on call #1) so the trajectory starts exactly at x₀.
    """
    cb = [s for s in history if getattr(s, "source", None) == "callback"]
    if len(cb) >= 2:
        first_eval = next(
            (s for s in history if getattr(s, "source", None) == "evaluation"),
            None,
        )
        snaps = ([first_eval, *cb] if first_eval is not None else list(cb))
    else:
        snaps = list(history)
    xs: list[float] = []
    ys: list[float] = []
    for snap in snaps:
        nat = snap.params_natural
        if param_x in nat and param_y in nat:
            xs.append(float(nat[param_x]))
            ys.append(float(nat[param_y]))
    return np.asarray(xs), np.asarray(ys)


def initial_point_from_history(
    history, param_x: str, param_y: str,
) -> tuple[float, float] | None:
    """Return the *first evaluation* snapshot's natural (p_x, p_y).

    The previous implementation used ``ParamSpec.default`` for the
    'initial' marker, which is the registry default — not the seed the
    calibrator actually started from (ATM-IV-derived for Heston, etc.).
    Pulling the first ``source='evaluation'`` snapshot gives the true
    starting point, which is what the user expects when they read the
    contour plot.
    """
    for snap in history:
        if getattr(snap, "source", None) == "evaluation":
            nat = snap.params_natural
            if param_x in nat and param_y in nat:
                return float(nat[param_x]), float(nat[param_y])
            return None
    # No 'evaluation' snapshot — fall back to the very first snapshot.
    if history:
        nat = history[0].params_natural
        if param_x in nat and param_y in nat:
            return float(nat[param_x]), float(nat[param_y])
    return None


def slice_minimum(
    result: "LandscapeResult",
) -> tuple[float, float, float] | None:
    """Return ``(x_min, y_min, loss_min)`` — the lowest cell on the slice.

    Returns ``None`` when every cell is NaN (e.g. every model evaluation
    raised). The user-facing caption uses this to anchor the plot
    visually against a numerical value.
    """
    grid = result.loss_grid
    if not np.isfinite(grid).any():
        return None
    flat_idx = int(np.nanargmin(grid))
    j, i = np.unravel_index(flat_idx, grid.shape)
    return (
        float(result.x_values[i]),
        float(result.y_values[j]),
        float(grid[j, i]),
    )


def basin_curvature(
    result: "LandscapeResult",
) -> dict[str, float] | None:
    """Estimate the 2×2 Hessian by finite differences around the slice
    minimum.

    Uses the 9-point stencil ``(i±1, j±1)`` centred on the minimum cell.
    Returns ``{"lambda_min", "lambda_max", "kappa", "angle_deg"}``
    where ``kappa = lambda_max / lambda_min`` is the condition number
    and ``angle_deg`` is the principal axis orientation (degrees from
    the x-axis). Returns ``None`` when the minimum sits on a boundary
    or NaN cells prevent the stencil.
    """
    grid = result.loss_grid
    if not np.isfinite(grid).any():
        return None
    flat_idx = int(np.nanargmin(grid))
    j, i = np.unravel_index(flat_idx, grid.shape)
    nx, ny = grid.shape[1], grid.shape[0]
    if not (1 <= i < nx - 1 and 1 <= j < ny - 1):
        return None
    # Step sizes in natural units
    dx = float(result.x_values[1] - result.x_values[0])
    dy = float(result.y_values[1] - result.y_values[0])
    stencil = grid[j - 1: j + 2, i - 1: i + 2]
    if not np.isfinite(stencil).all():
        return None
    # Central finite differences for the Hessian of f(x, y)
    fxx = (stencil[1, 2] - 2 * stencil[1, 1] + stencil[1, 0]) / (dx * dx)
    fyy = (stencil[2, 1] - 2 * stencil[1, 1] + stencil[0, 1]) / (dy * dy)
    fxy = (stencil[2, 2] - stencil[2, 0] - stencil[0, 2] + stencil[0, 0]) / (4.0 * dx * dy)
    H = np.array([[fxx, fxy], [fxy, fyy]], dtype=np.float64)
    try:
        eigs = np.linalg.eigvalsh(H)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(eigs).all() or (eigs <= 0).any():
        # Not a positive-definite Hessian → not a clean basin minimum.
        return None
    lam_min = float(eigs[0])
    lam_max = float(eigs[-1])
    # Principal axis from the eigenvector of the largest eigenvalue
    try:
        _, vecs = np.linalg.eigh(H)
        v = vecs[:, -1]
        angle_deg = float(np.degrees(np.arctan2(v[1], v[0])))
    except np.linalg.LinAlgError:
        angle_deg = float("nan")
    return {
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "kappa": lam_max / max(lam_min, 1e-30),
        "angle_deg": angle_deg,
    }


def feller_boundary_segments(
    spec_lookup: dict[str, tuple[float, float]],
    param_x: str,
    param_y: str,
    base_params: dict[str, float],
    n: int = 80,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> dict[str, np.ndarray] | None:
    """Return ``(x_curve, y_curve)`` for the 2κθ = ξ² boundary in
    (κ, θ) / (κ, ξ) / (θ, ξ) slices. Returns ``None`` when the slice does
    not contain the relevant pair.

    When ``x_range`` / ``y_range`` are given (the visible slice window),
    the curve is **clipped** to that rectangle so it stays on the plotted
    surface instead of running off to the full parameter bounds. Density is
    bumped when clipping so the visible arc stays smooth, and ``None`` is
    returned if no point of the curve falls inside the window.
    """
    pair = {param_x, param_y}
    if not pair.issubset({"kappa", "theta", "alpha"}):
        return None

    clip = x_range is not None and y_range is not None
    n_gen = max(n, 400) if clip else n

    if pair == {"kappa", "theta"}:
        alpha = float(base_params.get("alpha", 0.0))
        kappa_lo, kappa_hi = spec_lookup["kappa"]
        kappa = np.linspace(max(kappa_lo, 1e-3), kappa_hi, n_gen)
        theta = alpha * alpha / (2.0 * kappa)
        x_arr, y_arr = (kappa, theta) if param_x == "kappa" else (theta, kappa)
    elif pair == {"kappa", "alpha"}:
        theta = float(base_params.get("theta", 0.0))
        kappa_lo, kappa_hi = spec_lookup["kappa"]
        kappa = np.linspace(max(kappa_lo, 1e-3), kappa_hi, n_gen)
        alpha = np.sqrt(np.clip(2.0 * kappa * theta, 0.0, None))
        x_arr, y_arr = (kappa, alpha) if param_x == "kappa" else (alpha, kappa)
    else:  # pair == {"theta", "alpha"}
        kappa = float(base_params.get("kappa", 0.0))
        xi_lo, xi_hi = spec_lookup["alpha"]
        alpha = np.linspace(max(xi_lo, 1e-3), xi_hi, n_gen)
        theta = alpha * alpha / (2.0 * max(kappa, 1e-3))
        x_arr, y_arr = (theta, alpha) if param_x == "theta" else (alpha, theta)

    if clip:
        mask = (
            (x_arr >= x_range[0])
            & (x_arr <= x_range[1])
            & (y_arr >= y_range[0])
            & (y_arr <= y_range[1])
        )
        if not np.any(mask):
            return None
        x_arr, y_arr = x_arr[mask], y_arr[mask]

    return {"x": x_arr, "y": y_arr}


def feller_feasible_window(
    param_x: str,
    param_y: str,
    base_params: dict[str, float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    n: int = 200,
) -> tuple[float, float, float, float] | None:
    """Bounding box of the Feller-feasible region (``2κθ ≥ α²``) inside
    ``[x_range] × [y_range]`` for a κ/θ/α slice.

    Used to crop the landscape window onto the feasible side so the infeasible
    band — which prices to NaN and renders blank — doesn't waste the plot
    (notably in Feller-HARD mode, where the optimum sits on the boundary). The
    test is purely algebraic (no pricing), so this is instant.

    Returns ``(x_lo, x_hi, y_lo, y_hi)``, or ``None`` when the pair isn't
    Feller-constrained or no cell in the window is feasible.
    """
    if not {param_x, param_y}.issubset({"kappa", "theta", "alpha"}):
        return None
    xs = np.linspace(x_range[0], x_range[1], n)
    ys = np.linspace(y_range[0], y_range[1], n)
    grid_x, grid_y = np.meshgrid(xs, ys)
    axes = {param_x: grid_x, param_y: grid_y}
    kappa = axes.get("kappa", float(base_params.get("kappa", 0.0)))
    theta = axes.get("theta", float(base_params.get("theta", 0.0)))
    alpha = axes.get("alpha", float(base_params.get("alpha", 0.0)))
    feasible = 2.0 * kappa * theta >= alpha**2
    if not np.any(feasible):
        return None
    fx, fy = grid_x[feasible], grid_y[feasible]
    return float(fx.min()), float(fx.max()), float(fy.min()), float(fy.max())
