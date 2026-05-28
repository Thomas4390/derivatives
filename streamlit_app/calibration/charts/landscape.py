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


def _safe_log10(z: np.ndarray) -> np.ndarray:
    """log10 with a small epsilon floor to keep NaN cells out of the
    finite range. NaN inputs stay NaN."""
    eps = 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log10(np.where(z > 0, z, eps))


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


def _z_lookup(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """Nearest-cell loss lookup for a trajectory in 3-D rendering.

    The trajectory is plotted on top of the surface; we need its z
    coordinate to match the surface's local value rather than 0 so the
    line sits on top of the basin instead of clipping through it.
    """
    out = np.full(len(xs), np.nan, dtype=np.float64)
    for k, (xv, yv) in enumerate(zip(xs, ys)):
        if not (np.isfinite(xv) and np.isfinite(yv)):
            continue
        i = int(np.clip(np.argmin(np.abs(x_values - xv)), 0, len(x_values) - 1))
        j = int(np.clip(np.argmin(np.abs(y_values - yv)), 0, len(y_values) - 1))
        out[k] = z_grid[j, i]
    return out


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
    log_scale: bool = True,
    view_3d: bool = False,
    # Legacy single-surface kwargs (used when the first arg is one result).
    solver_trajectories: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    multi_start_points: tuple[np.ndarray, np.ndarray] | None = None,
    objective_name: str | None = None,
) -> go.Figure:
    """Render a contour (default) or 3-D surface of ``loss(p_x, p_y)``.

    ``layers_or_result`` is either a list of :class:`LandscapeLayer` (one per
    overlaid objective) or a single ``LandscapeResult`` for the legacy
    single-surface path. Shared overlays — ground truth, initial guess, and
    the Feller boundary — are objective-independent and drawn once.

    Marker Z-order (top to bottom):
      1. ground truth (green star)        — synthetic mode only
      2. initial guess (orange circle)
      3. solver trajectory lines + markers
      4. multi-start endpoints (purple diamonds)
      5. Feller-condition boundary
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
            log_scale=log_scale,
        )
    return _render_contour(
        layers,
        true_point=true_point,
        initial_point=initial_point,
        feller_curve=feller_curve,
        log_scale=log_scale,
    )


def _render_contour(
    layers: list[LandscapeLayer],
    *,
    true_point,
    initial_point,
    feller_curve,
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
            fig.add_trace(go.Contour(
                x=res.x_values, y=res.y_values, z=z_plot,
                contours=dict(coloring="lines", showlines=True),
                colorscale=[[0.0, obj_color], [1.0, obj_color]],
                line=dict(width=1.6),
                showscale=False,
                name=_obj_label(layer.objective_name),
                showlegend=True,
                hovertemplate=(
                    f"<b>{_obj_label(layer.objective_name)}</b><br>"
                    f"{res.param_x}: %{{x:.4f}}<br>"
                    f"{res.param_y}: %{{y:.4f}}<br>loss: %{{z:.4e}}<extra></extra>"
                ),
            ))
        else:
            fig.add_trace(go.Contour(
                x=res.x_values, y=res.y_values, z=z_plot,
                colorscale="Viridis",
                contours=dict(showlines=True, coloring="heatmap"),
                line=dict(width=0.6),
                colorbar=dict(
                    title=_loss_label(layer.objective_name, log_scale),
                    tickfont=dict(family=FONT_FAMILY, color=COLORS["text"], size=10),
                    len=0.78, x=1.01,
                ),
                hovertemplate=(
                    f"{res.param_x}: %{{x:.4f}}<br>"
                    f"{res.param_y}: %{{y:.4f}}<br>loss: %{{z:.4e}}<extra></extra>"
                ),
                showlegend=False,
            ))

    # Feller boundary — drawn early so trajectories / markers overlay.
    if feller_curve is not None:
        fig.add_trace(go.Scatter(
            x=feller_curve["x"], y=feller_curve["y"],
            mode="lines",
            line=dict(color="rgba(15,23,42,0.85)", width=2.0, dash="dot"),
            name="Feller boundary  2κθ = ξ²",
            hovertemplate="Feller boundary<extra></extra>",
        ))

    # ── Multi-start endpoints (drawn first, below x₀ + trajectories) ──
    for layer in layers:
        suffix = f" · {_obj_label(layer.objective_name)}" if multi else ""
        if layer.multi_start_points is not None:
            ms_x, ms_y = layer.multi_start_points
            if len(ms_x) > 0:
                fig.add_trace(go.Scatter(
                    x=ms_x, y=ms_y, mode="markers",
                    marker=dict(size=14, color="#7c3aed", symbol="diamond-open",
                                line=dict(color="#7c3aed", width=1.8)),
                    name=f"multi-start endpoints{suffix}",
                    hovertemplate=(
                        f"<b>multi-start endpoint</b><br>{result.param_x}=%{{x:.4f}}<br>"
                        f"{result.param_y}=%{{y:.4f}}<extra></extra>"
                    ),
                ))

    # The initial guess marker is drawn *before* the trajectories so the
    # solver endpoint always stays visible on top. Otherwise, when LM
    # converges right at the seed (typical of synthetic surfaces where
    # the ATM-IV-derived x₀ is already optimal), the gros marker x₀
    # entirely masked the trajectory and the user saw no path at all.
    if initial_point is not None:
        fig.add_trace(go.Scatter(
            x=[initial_point[0]], y=[initial_point[1]], mode="markers",
            marker=dict(size=19, color="#d97706", symbol="circle",
                        line=dict(color="white", width=2)),
            name="initial guess (x₀)",
            hovertemplate=(
                f"<b>initial guess (x₀)</b><br>{result.param_x}=%{{x:.4f}}<br>"
                f"{result.param_y}=%{{y:.4f}}<extra></extra>"
            ),
        ))

    # ── Solver trajectories (drawn after x₀ so the endpoint is on top) ──
    for layer in layers:
        suffix = f" · {_obj_label(layer.objective_name)}" if multi else ""
        for solver_name, (xs, ys) in (layer.solver_trajectories or {}).items():
            if len(xs) == 0:
                continue
            color = SOLVER_COLOR_MAP.get(solver_name, COLORS["accent"])
            sizes = np.full(len(xs), 6.0)
            # First marker stays the standard size so it doesn't read as a
            # second starting point next to the x₀ marker (size 19, drawn
            # below): the trajectory always starts at x₀, and the dedicated
            # x₀ marker is already there to call out the seed.
            # Final marker has to overflow the x₀ marker so it remains
            # visible when the solver converged right at the seed.
            sizes[-1] = 22.0
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                line=dict(color=color, width=2.4),
                marker=dict(size=sizes, color=color, line=dict(color="white", width=1)),
                name=f"{solver_name} trajectory{suffix}",
                hovertemplate=(
                    f"<b>{solver_name} trajectory</b><br>{result.param_x}=%{{x:.4f}}<br>"
                    f"{result.param_y}=%{{y:.4f}}<extra></extra>"
                ),
            ))
    # Ground-truth star is the last trace so it sits on top.
    if true_point is not None:
        fig.add_trace(go.Scatter(
            x=[true_point[0]], y=[true_point[1]], mode="markers",
            marker=dict(size=20, color="#10b981", symbol="star",
                        line=dict(color="white", width=2)),
            name="ground truth",
            hovertemplate=(
                f"<b>ground truth</b><br>{result.param_x}=%{{x:.4f}}<br>"
                f"{result.param_y}=%{{y:.4f}}<extra></extra>"
            ),
        ))

    title = f"Loss landscape · ({result.param_x}, {result.param_y}) slice"
    if multi:
        title += f"  ·  {len(layers)} objectives overlaid"
    apply_lab_theme(fig, height=580, margin=(60, 20, 70, 50), legend_horizontal=False)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family=FONT_FAMILY, color=COLORS["text"], size=15),
            x=0.0, xanchor="left", y=0.985, yanchor="top", pad=dict(b=10, t=6),
        ),
        legend=dict(
            orientation="v", yanchor="bottom", y=0.01, xanchor="right", x=0.985,
            font=dict(family=FONT_FAMILY, color=COLORS["text"], size=10),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=COLORS["primary"], borderwidth=1,
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
    log_scale: bool,
) -> go.Figure:
    primary = layers[0]
    result = primary.result
    multi = len(layers) > 1
    z_plot_primary = _safe_log10(primary.result.loss_grid) if log_scale else primary.result.loss_grid

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
            fig.add_trace(go.Surface(
                x=X, y=Y, z=z_plot,
                colorscale=_mono_colorscale(obj_color), showscale=False,
                opacity=0.66, name=_obj_label(layer.objective_name),
                hovertemplate=(
                    f"<b>{_obj_label(layer.objective_name)}</b><br>"
                    f"{res.param_x}: %{{x:.4f}}<br>{res.param_y}: %{{y:.4f}}<br>"
                    "loss: %{z:.4e}<extra></extra>"
                ),
            ))
            # Surfaces don't appear in the legend — add a colour proxy.
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers",
                marker=dict(size=8, color=obj_color, symbol="square"),
                name=_obj_label(layer.objective_name), showlegend=True,
            ))
        else:
            fig.add_trace(go.Surface(
                x=X, y=Y, z=z_plot,
                colorscale="Viridis", showscale=True,
                colorbar=dict(
                    title=_loss_label(layer.objective_name, log_scale),
                    tickfont=dict(family=FONT_FAMILY, color=COLORS["text"], size=10),
                    len=0.65, x=1.01,
                ),
                hovertemplate=(
                    f"{res.param_x}: %{{x:.4f}}<br>{res.param_y}: %{{y:.4f}}<br>"
                    "loss: %{z:.4e}<extra></extra>"
                ),
                contours=dict(z=dict(show=True, usecolormap=True,
                                     project=dict(z=True), width=2)),
                opacity=0.92, name="loss surface",
            ))

    # Markers/trajectories sit just above the surfaces; scale the offset to
    # the combined z range so it tracks the colour scale.
    z_offset = 0.0
    if all_finite:
        merged = np.concatenate(all_finite)
        z_offset = 0.05 * (merged.max() - merged.min())

    if feller_curve is not None:
        fx = np.asarray(feller_curve["x"], dtype=np.float64)
        fy = np.asarray(feller_curve["y"], dtype=np.float64)
        fz = _z_lookup(result.x_values, result.y_values, z_plot_primary, fx, fy) + z_offset
        fig.add_trace(go.Scatter3d(
            x=fx, y=fy, z=fz, mode="lines",
            line=dict(color="rgba(15,23,42,0.9)", width=4, dash="dot"),
            name="Feller boundary  2κθ = ξ²",
            hovertemplate="<b>Feller boundary</b>  2κθ = ξ²<extra></extra>",
        ))

    # ── Per-layer trajectories + endpoints (z sampled on own surface) ──
    for layer in layers:
        res = layer.result
        z_layer = _safe_log10(res.loss_grid) if log_scale else res.loss_grid
        suffix = f" · {_obj_label(layer.objective_name)}" if multi else ""
        for solver_name, (xs, ys) in (layer.solver_trajectories or {}).items():
            if len(xs) == 0:
                continue
            color = SOLVER_COLOR_MAP.get(solver_name, COLORS["accent"])
            xs_arr = np.asarray(xs, dtype=np.float64)
            ys_arr = np.asarray(ys, dtype=np.float64)
            zs = _z_lookup(res.x_values, res.y_values, z_layer, xs_arr, ys_arr) + z_offset
            sizes = np.full(len(xs_arr), 4.0)
            sizes[0] = 8.0
            sizes[-1] = 11.0
            fig.add_trace(go.Scatter3d(
                x=xs_arr, y=ys_arr, z=zs, mode="lines+markers",
                line=dict(color=color, width=5),
                marker=dict(size=sizes, color=color),
                name=f"{solver_name} trajectory{suffix}",
                hovertemplate=(
                    f"<b>{solver_name} trajectory</b><br>{res.param_x}=%{{x:.4f}}<br>"
                    f"{res.param_y}=%{{y:.4f}}<extra></extra>"
                ),
            ))
        if layer.multi_start_points is not None:
            ms_x = np.asarray(layer.multi_start_points[0], dtype=np.float64)
            ms_y = np.asarray(layer.multi_start_points[1], dtype=np.float64)
            if len(ms_x) > 0:
                ms_z = _z_lookup(res.x_values, res.y_values, z_layer, ms_x, ms_y) + z_offset
                fig.add_trace(go.Scatter3d(
                    x=ms_x, y=ms_y, z=ms_z, mode="markers",
                    marker=dict(size=10, color="#7c3aed", symbol="diamond-open",
                                line=dict(color="#7c3aed", width=1.5)),
                    name=f"multi-start endpoints{suffix}",
                    hovertemplate=(
                        f"<b>multi-start endpoint</b><br>{res.param_x}=%{{x:.4f}}<br>"
                        f"{res.param_y}=%{{y:.4f}}<extra></extra>"
                    ),
                ))

    if initial_point is not None:
        ix, iy = initial_point
        iz = float(_z_lookup(result.x_values, result.y_values, z_plot_primary,
                             np.array([ix]), np.array([iy]))[0]) + z_offset
        fig.add_trace(go.Scatter3d(
            x=[ix], y=[iy], z=[iz], mode="markers",
            marker=dict(size=11, color="#d97706", symbol="circle",
                        line=dict(color="white", width=1.5)),
            name="initial guess (x₀)",
            hovertemplate=(
                f"<b>initial guess (x₀)</b><br>{result.param_x}=%{{x:.4f}}<br>"
                f"{result.param_y}=%{{y:.4f}}<extra></extra>"
            ),
        ))
    if true_point is not None:
        tx, ty = true_point
        tz = float(_z_lookup(result.x_values, result.y_values, z_plot_primary,
                             np.array([tx]), np.array([ty]))[0]) + z_offset
        fig.add_trace(go.Scatter3d(
            x=[tx], y=[ty], z=[tz], mode="markers",
            marker=dict(size=13, color="#10b981", symbol="diamond",
                        line=dict(color="white", width=1.5)),
            name="ground truth",
            hovertemplate=(
                f"<b>ground truth</b><br>{result.param_x}=%{{x:.4f}}<br>"
                f"{result.param_y}=%{{y:.4f}}<extra></extra>"
            ),
        ))

    title = f"Loss landscape · ({result.param_x}, {result.param_y}) — 3-D view"
    if multi:
        title += f"  ·  {len(layers)} objectives overlaid"
    loss_label = _loss_label(primary.objective_name, log_scale)
    apply_lab_theme(fig, height=640, margin=(40, 20, 80, 30), legend_horizontal=False)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family=FONT_FAMILY, color=COLORS["text"], size=15),
            x=0.0, xanchor="left", y=0.985, yanchor="top", pad=dict(b=10, t=6),
        ),
        legend=dict(
            orientation="v", yanchor="bottom", y=0.02, xanchor="left", x=0.02,
            font=dict(family=FONT_FAMILY, color=COLORS["text"], size=10),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=COLORS["primary"], borderwidth=1,
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
