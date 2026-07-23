"""Loss-landscape Plotly chart — contour or 3-D surface, + overlays.

Supports overlaying **multiple loss surfaces** (one per calibration
objective) via :class:`LandscapeLayer`. With a single layer the chart keeps
its original look (filled Viridis contour / opaque 3-D surface). With more
than one layer it switches to a comparison style — line-contours in 2-D and
semi-transparent single-tint surfaces in 3-D — so the basins of different
objectives can be read side by side without occluding each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from config.constants import OBJECTIVE_DISPLAY_NAMES
from utils.plotly_theme import (
    COLORS,
    FONT_FAMILY,
    PALETTE,
    SOLVER_COLOR_MAP,
    apply_lab_theme,
)

# Reserved marker hues for the shared (objective-independent) overlays. Kept as
# module constants so a regression test can assert they never collide with any
# per-solver ``SOLVER_COLOR_MAP`` colour (that map is an inter-tab identity — we
# do NOT remap the solvers, we keep the annotations off their hues).
#   * x₀ — dark slate with a white ring: a neutral marker that reads clearly on
#     the Viridis surface and is distinct from the amber DE trajectory.
#   * multi-start — pink open diamonds: distinct from the purple NM trajectory.
#   * ground truth — green (star in 2-D, diamond in 3-D).
X0_MARKER_COLOR = "#1e293b"  # dark slate
MULTISTART_MARKER_COLOR = "#db2777"  # pink
TRUTH_MARKER_COLOR = "#10b981"  # green


@dataclass(frozen=True)
class LandscapeLayer:
    """One overlaid loss surface (a single objective) + its own overlays.

    ``solver_trajectories`` / ``multi_start_points`` belong to this layer's
    objective and are rendered on top of *this* surface (their z is sampled
    from this layer's grid in 3-D).
    """

    objective_name: str | None
    result: object  # LandscapeResult (duck-typed: x_values/y_values/loss_grid)
    solver_trajectories: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
    multi_start_points: tuple[np.ndarray, np.ndarray] | None = None
    # Exact (raw, non-log) slice losses for the point markers — same cell loss
    # as the grid, evaluated at the marker's own coordinates. When present they
    # replace the surface-sampled height in 3-D so a converged endpoint can
    # never be snapped up a valley wall (keys match ``solver_trajectories``).
    trajectory_end_losses: dict[str, float] | None = None
    multi_start_losses: np.ndarray | None = None


def _safe_log10(z: np.ndarray) -> np.ndarray:
    """log10 with a small epsilon floor for non-positive cells; NaN stays NaN.

    NaN cells are masked/failed evaluations (the display mask and the
    ``_z_interp`` NaN-break contract both rely on them). ``np.where(z > 0, …)``
    treats NaN as the else-branch, which would floor them to ``log10(1e-12) =
    -12`` — rendering as the deepest 'valley' and compressing the real basin
    into a colour sliver. Re-mask them back to NaN after the floor.
    """
    eps = 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.log10(np.where(z > 0, z, eps))
    return np.where(np.isnan(z), np.nan, out)


def _loss_label(objective_name: str | None, log_scale: bool) -> str:
    """Axis / colorbar caption naming the objective whose loss is shown."""
    base = "loss"
    if objective_name:
        base = f"{OBJECTIVE_DISPLAY_NAMES.get(objective_name, objective_name)} loss"
    return f"log₁₀ {base}" if log_scale else base


def _obj_label(objective_name: str | None) -> str:
    """Short human-readable objective label for legends."""
    if not objective_name:
        return "loss"
    return OBJECTIVE_DISPLAY_NAMES.get(objective_name, objective_name)


def _mono_colorscale(color: str) -> list[list]:
    """A flat single-tint colorscale so a 3-D surface reads as one colour
    (shading comes from the scene lighting, not the loss gradient). Used
    when overlaying multiple surfaces so they stay visually distinct."""
    return [[0.0, color], [1.0, color]]


def _z_interp(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """Bilinear surface-height interpolation for the 3-D overlays.

    The overlays are plotted on top of the surface; their z must match the
    surface's local value so lines sit on the basin instead of clipping
    through it. Bilinear interpolation (the axes are uniform ``linspace``
    grids) replaces the old nearest-cell lookup, whose snapping pushed a
    converged point up the valley wall whenever the basin was narrower than
    a cell. Points outside the window clamp to the edge cells; NaN corner
    cells propagate to NaN (Plotly breaks the line rather than inventing a
    height over masked cells).
    """
    out = np.full(len(xs), np.nan, dtype=np.float64)
    nx, ny = len(x_values), len(y_values)
    for k, (xv, yv) in enumerate(zip(xs, ys)):
        if not (np.isfinite(xv) and np.isfinite(yv)):
            continue
        fi = (
            0.0
            if nx == 1
            else float(
                np.clip(
                    (xv - x_values[0]) / (x_values[-1] - x_values[0]) * (nx - 1),
                    0.0,
                    nx - 1,
                )
            )
        )
        fj = (
            0.0
            if ny == 1
            else float(
                np.clip(
                    (yv - y_values[0]) / (y_values[-1] - y_values[0]) * (ny - 1),
                    0.0,
                    ny - 1,
                )
            )
        )
        i0, j0 = int(fi), int(fj)
        i1, j1 = min(i0 + 1, nx - 1), min(j0 + 1, ny - 1)
        tx, ty = fi - i0, fj - j0
        corners = np.array(
            [
                z_grid[j0, i0],
                z_grid[j0, i1],
                z_grid[j1, i0],
                z_grid[j1, i1],
            ]
        )
        weights = np.array(
            [
                (1 - tx) * (1 - ty),
                tx * (1 - ty),
                (1 - tx) * ty,
                tx * ty,
            ]
        )
        live = weights > 0
        if np.isnan(corners[live]).any():
            continue
        out[k] = float(corners[live] @ weights[live])
    return out


def _exact_z(loss: float, log_scale: bool) -> float:
    """Plotted z of an exact raw loss — same transform as the surface."""
    if log_scale:
        return float(_safe_log10(np.asarray([loss], dtype=np.float64))[0])
    return float(loss)


def _robust_zmax(z_plot: np.ndarray) -> float | None:
    """Colour-scale cap (99th percentile) for spike-dominated grids.

    The GARCH NLL rises by orders of magnitude in the strip approaching the
    stationarity boundary that survives the display mask — one such cell
    would stretch the colour scale until the basin reads as a flat sheet
    (the NLL view cannot log-scale: the likelihood is negative). Returns the
    cap only when the spike truly dominates; benign grids (Heston, log-scale
    FFT) fail the trigger and keep the full auto range.
    """
    finite = z_plot[np.isfinite(z_plot)]
    if finite.size < 8:
        return None
    q99 = float(np.quantile(finite, 0.99))
    lo = float(finite.min())
    hi = float(finite.max())
    if q99 <= lo:
        return None
    if hi - q99 > 3.0 * (q99 - lo):
        return q99
    return None


def _normalize_layers(
    layers_or_result,
    *,
    solver_trajectories,
    multi_start_points,
    objective_name,
) -> list[LandscapeLayer]:
    """Accept either a list of :class:`LandscapeLayer` (new multi-surface API)
    or a single ``LandscapeResult`` plus the legacy per-call overlay kwargs."""
    if isinstance(layers_or_result, (list, tuple)):
        return list(layers_or_result)
    return [
        LandscapeLayer(
            objective_name=objective_name,
            result=layers_or_result,
            solver_trajectories=solver_trajectories,
            multi_start_points=multi_start_points,
        )
    ]


def render_landscape(
    layers_or_result,
    *,
    true_point: tuple[float, float] | None = None,
    initial_point: tuple[float, float] | None = None,
    feller_curve: dict[str, np.ndarray] | None = None,
    stationarity_curve: dict[str, np.ndarray] | None = None,
    stationarity_label: str = "stationarity boundary",
    log_scale: bool = True,
    view_3d: bool = False,
    # Exact raw losses at the shared markers (3-D height override; optional).
    initial_point_loss: float | None = None,
    true_point_loss: float | None = None,
    # Legacy single-surface kwargs (used when the first arg is one result).
    solver_trajectories: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    multi_start_points: tuple[np.ndarray, np.ndarray] | None = None,
    objective_name: str | None = None,
) -> go.Figure:
    """Render a contour (default) or 3-D surface of ``loss(p_x, p_y)``.

    ``layers_or_result`` is either a list of :class:`LandscapeLayer` (one per
    overlaid objective) or a single ``LandscapeResult`` for the legacy
    single-surface path. Shared overlays — ground truth, initial guess, the
    Feller boundary, and the GARCH stationarity boundary — are
    objective-independent and drawn once.

    Marker Z-order (top to bottom):
      1. initial guess (dark-slate circle) — drawn last so the ground-truth
         star (which overflows it) can never bury it entirely
      2. ground truth (green star)        — synthetic mode only
      3. solver trajectory lines + markers
      4. multi-start endpoints (pink diamonds)
      5. Feller / stationarity boundary
      6. background contour(s) / surface(s)
    """
    layers = _normalize_layers(
        layers_or_result,
        solver_trajectories=solver_trajectories,
        multi_start_points=multi_start_points,
        objective_name=objective_name,
    )
    if view_3d:
        return _render_3d(
            layers,
            true_point=true_point,
            initial_point=initial_point,
            feller_curve=feller_curve,
            stationarity_curve=stationarity_curve,
            stationarity_label=stationarity_label,
            log_scale=log_scale,
            initial_point_loss=initial_point_loss,
            true_point_loss=true_point_loss,
        )
    return _render_contour(
        layers,
        true_point=true_point,
        initial_point=initial_point,
        feller_curve=feller_curve,
        stationarity_curve=stationarity_curve,
        stationarity_label=stationarity_label,
        log_scale=log_scale,
    )


def _render_contour(
    layers: list[LandscapeLayer],
    *,
    true_point,
    initial_point,
    feller_curve,
    stationarity_curve=None,
    stationarity_label: str = "stationarity boundary",
    log_scale: bool,
) -> go.Figure:
    primary = layers[0]
    result = primary.result
    multi = len(layers) > 1
    fig = go.Figure()

    # ── Surfaces (contours) ─────────────────────────────────────────
    for idx, layer in enumerate(layers):
        res = layer.result
        z_plot = _safe_log10(res.loss_grid) if log_scale else res.loss_grid
        if multi:
            obj_color = PALETTE[idx % len(PALETTE)]
            fig.add_trace(
                go.Contour(
                    x=res.x_values,
                    y=res.y_values,
                    z=z_plot,
                    customdata=res.loss_grid,
                    contours=dict(coloring="lines", showlines=True),
                    colorscale=[[0.0, obj_color], [1.0, obj_color]],
                    line=dict(width=1.6),
                    showscale=False,
                    name=_obj_label(layer.objective_name),
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{_obj_label(layer.objective_name)}</b><br>"
                        f"{res.param_x}: %{{x:.4f}}<br>"
                        f"{res.param_y}: %{{y:.4f}}<br>loss: %{{customdata:.4e}}<extra></extra>"
                    ),
                )
            )
        else:
            zmax = _robust_zmax(z_plot)
            finite = z_plot[np.isfinite(z_plot)]
            fig.add_trace(
                go.Contour(
                    x=res.x_values,
                    y=res.y_values,
                    z=z_plot,
                    customdata=res.loss_grid,
                    zmin=float(finite.min()) if zmax is not None else None,
                    zmax=zmax,
                    colorscale="Viridis",
                    contours=dict(showlines=True, coloring="heatmap"),
                    line=dict(width=0.6),
                    colorbar=dict(
                        title=_loss_label(layer.objective_name, log_scale),
                        tickfont=dict(
                            family=FONT_FAMILY, color=COLORS["text"], size=10
                        ),
                        len=0.78,
                        x=1.01,
                    ),
                    hovertemplate=(
                        f"{res.param_x}: %{{x:.4f}}<br>"
                        f"{res.param_y}: %{{y:.4f}}<br>loss: %{{customdata:.4e}}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

    # Feller boundary — drawn early so trajectories / markers overlay.
    if feller_curve is not None:
        fig.add_trace(
            go.Scatter(
                x=feller_curve["x"],
                y=feller_curve["y"],
                mode="lines",
                line=dict(color="rgba(15,23,42,0.85)", width=2.0, dash="dot"),
                name="Feller boundary  2κθ = α²",
                hovertemplate="Feller boundary<extra></extra>",
            )
        )

    # GARCH stationarity boundary (persistence = 1) — same early z-order.
    if stationarity_curve is not None:
        fig.add_trace(
            go.Scatter(
                x=stationarity_curve["x"],
                y=stationarity_curve["y"],
                mode="lines",
                line=dict(color="rgba(15,23,42,0.85)", width=2.0, dash="dash"),
                name=stationarity_label,
                hovertemplate="stationarity boundary<extra></extra>",
            )
        )

    # ── Multi-start endpoints (drawn first, below x₀ + trajectories) ──
    for layer in layers:
        suffix = f" · {_obj_label(layer.objective_name)}" if multi else ""
        if layer.multi_start_points is not None:
            ms_x, ms_y = layer.multi_start_points
            if len(ms_x) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=ms_x,
                        y=ms_y,
                        mode="markers",
                        marker=dict(
                            size=14,
                            color=MULTISTART_MARKER_COLOR,
                            symbol="diamond-open",
                            line=dict(color=MULTISTART_MARKER_COLOR, width=1.8),
                        ),
                        name=f"multi-start endpoints{suffix}",
                        hovertemplate=(
                            f"<b>multi-start endpoint</b><br>{result.param_x}=%{{x:.4f}}<br>"
                            f"{result.param_y}=%{{y:.4f}}<extra></extra>"
                        ),
                    )
                )

    # ── Solver trajectories (x₀ is drawn last, after the truth star) ──
    for layer in layers:
        suffix = f" · {_obj_label(layer.objective_name)}" if multi else ""
        for k, (solver_name, (xs, ys)) in enumerate(
            (layer.solver_trajectories or {}).items()
        ):
            if len(xs) == 0:
                continue
            # Trajectories are keyed by the composite "solver/objective" label;
            # the bare solver name is what SOLVER_COLOR_MAP and the legend need
            # (the objective is already carried by ``suffix`` and the contour
            # colour). The old code looked the composite key up in the map, which
            # always missed and fell back to the reserved amber accent for every
            # run — that is the legend bug being fixed. The solver hues (amber
            # DE, purple NM) are now free to be themselves: x₀ moved to a
            # dark-slate circle and the multi-start endpoints to pink, so a
            # trajectory can no longer be mistaken for either annotation.
            bare_solver = solver_name.split("/", 1)[0]
            color = SOLVER_COLOR_MAP.get(bare_solver, PALETTE[k % len(PALETTE)])
            sizes = np.full(len(xs), 6.0)
            # First marker stays the standard size so it doesn't read as a
            # second starting point next to the x₀ marker (size 19, drawn
            # below): the trajectory always starts at x₀, and the dedicated
            # x₀ marker is already there to call out the seed.
            # Final marker has to overflow the x₀ marker so it remains
            # visible when the solver converged right at the seed.
            sizes[-1] = 22.0
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    line=dict(color=color, width=2.4),
                    marker=dict(
                        size=sizes, color=color, line=dict(color="white", width=1)
                    ),
                    name=f"{bare_solver} trajectory{suffix}",
                    hovertemplate=(
                        f"<b>{bare_solver} trajectory</b><br>{result.param_x}=%{{x:.4f}}<br>"
                        f"{result.param_y}=%{{y:.4f}}<extra></extra>"
                    ),
                )
            )
    # Ground-truth star drawn before x₀ so its size-20 point overflows the
    # size-19 x₀ circle even when the seed sits exactly on the truth.
    if true_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[true_point[0]],
                y=[true_point[1]],
                mode="markers",
                marker=dict(
                    size=20,
                    color=TRUTH_MARKER_COLOR,
                    symbol="star",
                    line=dict(color="white", width=2),
                ),
                name="ground truth",
                hovertemplate=(
                    f"<b>ground truth</b><br>{result.param_x}=%{{x:.4f}}<br>"
                    f"{result.param_y}=%{{y:.4f}}<extra></extra>"
                ),
            )
        )

    # The initial guess is the very last trace. On synthetic surfaces where the
    # ATM-IV seed already sits on the truth, x₀, the truth star and the solver
    # endpoint stack on one spot; drawing x₀ last (dark slate, size 19) keeps it
    # visible while the size-20 star and size-22 endpoint still poke past it.
    if initial_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[initial_point[0]],
                y=[initial_point[1]],
                mode="markers",
                marker=dict(
                    size=19,
                    color=X0_MARKER_COLOR,
                    symbol="circle",
                    line=dict(color="white", width=2),
                ),
                name="initial guess (x₀)",
                hovertemplate=(
                    f"<b>initial guess (x₀)</b><br>{result.param_x}=%{{x:.4f}}<br>"
                    f"{result.param_y}=%{{y:.4f}}<extra></extra>"
                ),
            )
        )

    title = f"Loss landscape · ({result.param_x}, {result.param_y}) slice"
    if multi:
        title += f"  ·  {len(layers)} objectives overlaid"
    apply_lab_theme(fig, height=580, margin=(60, 20, 70, 50), legend_horizontal=False)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family=FONT_FAMILY, color=COLORS["text"], size=15),
            x=0.0,
            xanchor="left",
            y=0.985,
            yanchor="top",
            pad=dict(b=10, t=6),
        ),
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.985,
            font=dict(family=FONT_FAMILY, color=COLORS["text"], size=10),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=COLORS["primary"],
            borderwidth=1,
        ),
    )
    fig.update_layout(
        hovermode="closest",
        uirevision=f"landscape-{result.param_x}-{result.param_y}",
    )
    fig.update_xaxes(title=result.param_x)
    fig.update_yaxes(title=result.param_y)
    return fig


def _render_3d(
    layers: list[LandscapeLayer],
    *,
    true_point,
    initial_point,
    feller_curve,
    stationarity_curve=None,
    stationarity_label: str = "stationarity boundary",
    log_scale: bool,
    initial_point_loss: float | None = None,
    true_point_loss: float | None = None,
) -> go.Figure:
    primary = layers[0]
    result = primary.result
    multi = len(layers) > 1
    z_plot_primary = (
        _safe_log10(primary.result.loss_grid) if log_scale else primary.result.loss_grid
    )

    fig = go.Figure()

    # ── Surfaces ────────────────────────────────────────────────────
    all_finite: list[np.ndarray] = []
    for idx, layer in enumerate(layers):
        res = layer.result
        z_plot = _safe_log10(res.loss_grid) if log_scale else res.loss_grid
        finite = z_plot[np.isfinite(z_plot)]
        if finite.size:
            all_finite.append(finite)
        X, Y = np.meshgrid(res.x_values, res.y_values)
        if multi:
            obj_color = PALETTE[idx % len(PALETTE)]
            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=z_plot,
                    customdata=res.loss_grid,
                    colorscale=_mono_colorscale(obj_color),
                    showscale=False,
                    opacity=0.66,
                    name=_obj_label(layer.objective_name),
                    hovertemplate=(
                        f"<b>{_obj_label(layer.objective_name)}</b><br>"
                        f"{res.param_x}: %{{x:.4f}}<br>{res.param_y}: %{{y:.4f}}<br>"
                        "loss: %{customdata:.4e}<extra></extra>"
                    ),
                )
            )
            # Surfaces don't appear in the legend — add a colour proxy.
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(size=8, color=obj_color, symbol="square"),
                    name=_obj_label(layer.objective_name),
                    showlegend=True,
                )
            )
        else:
            cmax = _robust_zmax(z_plot)
            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=z_plot,
                    customdata=res.loss_grid,
                    cmin=float(finite.min())
                    if (cmax is not None and finite.size)
                    else None,
                    cmax=cmax,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title=_loss_label(layer.objective_name, log_scale),
                        tickfont=dict(
                            family=FONT_FAMILY, color=COLORS["text"], size=10
                        ),
                        len=0.65,
                        x=1.01,
                    ),
                    hovertemplate=(
                        f"{res.param_x}: %{{x:.4f}}<br>{res.param_y}: %{{y:.4f}}<br>"
                        "loss: %{customdata:.4e}<extra></extra>"
                    ),
                    contours=dict(
                        z=dict(
                            show=True, usecolormap=True, project=dict(z=True), width=2
                        )
                    ),
                    opacity=0.92,
                    name="loss surface",
                )
            )

    # Markers/trajectories sit just above the surfaces; scale the offset to
    # the combined z range so it tracks the colour scale. ``z_floor`` is the
    # lowest finite surface height — the last-resort height for a marker whose
    # exact loss is non-finite and whose (x, y) falls on masked cells, so it
    # can never silently drop out of the 3-D scene.
    z_offset = 0.0
    z_floor = 0.0
    if all_finite:
        merged = np.concatenate(all_finite)
        z_offset = 0.05 * (merged.max() - merged.min())
        z_floor = float(merged.min())

    if feller_curve is not None:
        fx = np.asarray(feller_curve["x"], dtype=np.float64)
        fy = np.asarray(feller_curve["y"], dtype=np.float64)
        fz = (
            _z_interp(result.x_values, result.y_values, z_plot_primary, fx, fy)
            + z_offset
        )
        fig.add_trace(
            go.Scatter3d(
                x=fx,
                y=fy,
                z=fz,
                mode="lines",
                line=dict(color="rgba(15,23,42,0.9)", width=4, dash="dot"),
                name="Feller boundary  2κθ = α²",
                hovertemplate="<b>Feller boundary</b>  2κθ = α²<extra></extra>",
            )
        )

    if stationarity_curve is not None:
        sx = np.asarray(stationarity_curve["x"], dtype=np.float64)
        sy = np.asarray(stationarity_curve["y"], dtype=np.float64)
        sz = (
            _z_interp(result.x_values, result.y_values, z_plot_primary, sx, sy)
            + z_offset
        )
        fig.add_trace(
            go.Scatter3d(
                x=sx,
                y=sy,
                z=sz,
                mode="lines",
                line=dict(color="rgba(15,23,42,0.9)", width=4, dash="dash"),
                name=stationarity_label,
                hovertemplate="<b>stationarity boundary</b><extra></extra>",
            )
        )

    # ── Per-layer trajectories + endpoints (z sampled on own surface) ──
    for layer in layers:
        res = layer.result
        z_layer = _safe_log10(res.loss_grid) if log_scale else res.loss_grid
        suffix = f" · {_obj_label(layer.objective_name)}" if multi else ""
        for k, (solver_name, (xs, ys)) in enumerate(
            (layer.solver_trajectories or {}).items()
        ):
            if len(xs) == 0:
                continue
            bare_solver = solver_name.split("/", 1)[0]
            color = SOLVER_COLOR_MAP.get(bare_solver, PALETTE[k % len(PALETTE)])
            xs_arr = np.asarray(xs, dtype=np.float64)
            ys_arr = np.asarray(ys, dtype=np.float64)
            zs = (
                _z_interp(res.x_values, res.y_values, z_layer, xs_arr, ys_arr)
                + z_offset
            )
            end_loss = (layer.trajectory_end_losses or {}).get(solver_name)
            if end_loss is not None and np.isfinite(end_loss):
                # The converged endpoint gets its exact slice loss — never a
                # neighbouring cell's height up the valley wall.
                zs[-1] = _exact_z(float(end_loss), log_scale) + z_offset
            sizes = np.full(len(xs_arr), 4.0)
            sizes[0] = 8.0
            sizes[-1] = 11.0
            fig.add_trace(
                go.Scatter3d(
                    x=xs_arr,
                    y=ys_arr,
                    z=zs,
                    mode="lines+markers",
                    line=dict(color=color, width=5),
                    marker=dict(size=sizes, color=color),
                    name=f"{bare_solver} trajectory{suffix}",
                    hovertemplate=(
                        f"<b>{bare_solver} trajectory</b><br>{res.param_x}=%{{x:.4f}}<br>"
                        f"{res.param_y}=%{{y:.4f}}<extra></extra>"
                    ),
                )
            )
        if layer.multi_start_points is not None:
            ms_x = np.asarray(layer.multi_start_points[0], dtype=np.float64)
            ms_y = np.asarray(layer.multi_start_points[1], dtype=np.float64)
            if len(ms_x) > 0:
                ms_z = (
                    _z_interp(res.x_values, res.y_values, z_layer, ms_x, ms_y)
                    + z_offset
                )
                if layer.multi_start_losses is not None:
                    exact = (
                        np.array(
                            [
                                _exact_z(float(v), log_scale)
                                for v in np.asarray(
                                    layer.multi_start_losses, dtype=np.float64
                                )
                            ]
                        )
                        + z_offset
                    )
                    if len(exact) == len(ms_z):
                        ms_z = np.where(np.isfinite(exact), exact, ms_z)
                fig.add_trace(
                    go.Scatter3d(
                        x=ms_x,
                        y=ms_y,
                        z=ms_z,
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=MULTISTART_MARKER_COLOR,
                            symbol="diamond-open",
                            line=dict(color=MULTISTART_MARKER_COLOR, width=1.5),
                        ),
                        name=f"multi-start endpoints{suffix}",
                        hovertemplate=(
                            f"<b>multi-start endpoint</b><br>{res.param_x}=%{{x:.4f}}<br>"
                            f"{res.param_y}=%{{y:.4f}}<extra></extra>"
                        ),
                    )
                )

    if initial_point is not None:
        ix, iy = initial_point
        iz = np.nan
        if initial_point_loss is not None and np.isfinite(initial_point_loss):
            iz = _exact_z(float(initial_point_loss), log_scale) + z_offset
        if not np.isfinite(iz):
            iz = (
                float(
                    _z_interp(
                        result.x_values,
                        result.y_values,
                        z_plot_primary,
                        np.array([ix]),
                        np.array([iy]),
                    )[0]
                )
                + z_offset
            )
        if not np.isfinite(iz):
            # Non-finite exact loss AND a masked surface at (ix, iy): pin the
            # marker to the surface z-floor so it is never dropped from the scene.
            iz = z_floor + z_offset
        fig.add_trace(
            go.Scatter3d(
                x=[ix],
                y=[iy],
                z=[iz],
                mode="markers",
                marker=dict(
                    size=11,
                    color=X0_MARKER_COLOR,
                    symbol="circle",
                    line=dict(color="white", width=2),
                ),
                name="initial guess (x₀)",
                hovertemplate=(
                    f"<b>initial guess (x₀)</b><br>{result.param_x}=%{{x:.4f}}<br>"
                    f"{result.param_y}=%{{y:.4f}}<extra></extra>"
                ),
            )
        )
    if true_point is not None:
        tx, ty = true_point
        # Surface height on THIS slice at the truth (x, y). The truth marker's
        # own z is its exact slice loss (the hidden params are frozen at the
        # fit, so it floats above the visible valley); a thin dropline down to
        # the surface makes that offset legible — "loss at the truth on this cut".
        surface_tz = (
            float(
                _z_interp(
                    result.x_values,
                    result.y_values,
                    z_plot_primary,
                    np.array([tx]),
                    np.array([ty]),
                )[0]
            )
            + z_offset
        )
        if true_point_loss is not None and np.isfinite(true_point_loss):
            tz = _exact_z(float(true_point_loss), log_scale) + z_offset
        else:
            tz = surface_tz
        if np.isfinite(tz) and np.isfinite(surface_tz):
            fig.add_trace(
                go.Scatter3d(
                    x=[tx, tx],
                    y=[ty, ty],
                    z=[surface_tz, tz],
                    mode="lines",
                    line=dict(color="rgba(100,116,139,0.7)", width=2, dash="dot"),
                    showlegend=False,
                    name="truth → surface",
                    hovertemplate=(
                        "<b>truth dropline</b><br>gap between the truth's exact "
                        "slice loss and the drawn surface<extra></extra>"
                    ),
                )
            )
        fig.add_trace(
            go.Scatter3d(
                x=[tx],
                y=[ty],
                z=[tz],
                mode="markers",
                marker=dict(
                    size=13,
                    color=TRUTH_MARKER_COLOR,
                    symbol="diamond",
                    line=dict(color="white", width=1.5),
                ),
                name="ground truth",
                hovertemplate=(
                    f"<b>ground truth</b><br>{result.param_x}=%{{x:.4f}}<br>"
                    f"{result.param_y}=%{{y:.4f}}<extra></extra>"
                ),
            )
        )

    title = f"Loss landscape · ({result.param_x}, {result.param_y}) — 3-D view"
    if multi:
        title += f"  ·  {len(layers)} objectives overlaid"
    loss_label = _loss_label(primary.objective_name, log_scale)
    apply_lab_theme(fig, height=640, margin=(40, 20, 80, 30), legend_horizontal=False)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family=FONT_FAMILY, color=COLORS["text"], size=15),
            x=0.0,
            xanchor="left",
            y=0.985,
            yanchor="top",
            pad=dict(b=10, t=6),
        ),
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.02,
            font=dict(family=FONT_FAMILY, color=COLORS["text"], size=10),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=COLORS["primary"],
            borderwidth=1,
        ),
        uirevision="landscape-3d",
        scene=dict(
            xaxis_title=result.param_x,
            yaxis_title=result.param_y,
            zaxis_title=loss_label,
            camera=dict(eye=dict(x=1.7, y=1.4, z=1.0)),
            uirevision="landscape-3d-camera",
        ),
    )
    return fig
