"""IV surface 3D + 2D smile + returns summary — themed for Quant Lab."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.plotly_theme import COLORS, FONT_FAMILY, PALETTE, apply_lab_theme


# Two complementary colorscales — cool blue for the market layer, warm
# amber for the model layer — span similar lightness ranges so neither
# dominates and stay readable for users with red-green color vision
# deficiency. Used identically in the static overlay and the animation
# so the converged frame matches the standalone fit chart pixel-for-pixel.
_MARKET_COLORSCALE = [
    [0.0, "#0c4a6e"],  # deep blue
    [0.5, "#0284c7"],  # medium blue
    [1.0, "#7dd3fc"],  # light cyan
]
_MARKET_LINE = "#7dd3fc"  # wireframe contour color for the market layer

# Per-solver colorscales used when overlaying *multiple* model surfaces.
# Each scale spans a similar lightness range as the market scale so no
# surface visually dominates, and hues are chosen to be mutually distinct
# (amber / violet / green / pink). The bright contour color used for the
# matching wireframe is the top stop of each scale.
# Per-solver surface colorscales. The mid stop is each solver's canonical
# identity colour from ``SOLVER_COLOR_MAP`` so a solver reads the SAME hue on
# the IV surfaces as on the Landscape / Live / Compare line charts (the two used
# to disagree — LM-JAX was teal on one and amber on the other). L-BFGS-B is the
# ONE exception: its identity blue (#0284c7) is exactly the market surface's
# medium blue on this chart type, so it keeps a distinct pink band here to stay
# separable from the (blue/cyan) market layer.
_SOLVER_COLORSCALES: dict[str, list[list]] = {
    "LM-JAX": [[0.0, "#134e4a"], [0.5, "#0d9488"], [1.0, "#5eead4"]],  # teal
    "DE": [[0.0, "#7c2d12"], [0.5, "#d97706"], [1.0, "#fcd34d"]],  # amber
    "NM": [[0.0, "#4c1d95"], [0.5, "#7c3aed"], [1.0, "#c4b5fd"]],  # purple
    "L-BFGS-B": [
        [0.0, "#831843"],
        [0.5, "#db2777"],
        [1.0, "#f9a8d4"],
    ],  # pink (see note)
}
_SOLVER_LINES: dict[str, str] = {
    "LM-JAX": "#5eead4",
    "DE": "#fcd34d",
    "NM": "#c4b5fd",
    "L-BFGS-B": "#f9a8d4",
}
# Fallback for solvers not in the registry — cycle through a small palette
# so unknown / future solvers still render with distinct colors.
_SOLVER_FALLBACK_SCALES: list[list[list]] = [
    [[0.0, "#7c2d12"], [0.5, "#d97706"], [1.0, "#fcd34d"]],  # amber
    [[0.0, "#4c1d95"], [0.5, "#7c3aed"], [1.0, "#c4b5fd"]],  # violet
    [[0.0, "#064e3b"], [0.5, "#059669"], [1.0, "#86efac"]],  # green
    [[0.0, "#831843"], [0.5, "#db2777"], [1.0, "#f9a8d4"]],  # pink
]
_SOLVER_FALLBACK_LINES: list[str] = ["#fcd34d", "#c4b5fd", "#86efac", "#f9a8d4"]


def _solver_colorscale(solver_name: str, slot: int) -> list[list]:
    return _SOLVER_COLORSCALES.get(
        solver_name,
        _SOLVER_FALLBACK_SCALES[slot % len(_SOLVER_FALLBACK_SCALES)],
    )


def _solver_line(solver_name: str, slot: int) -> str:
    return _SOLVER_LINES.get(
        solver_name,
        _SOLVER_FALLBACK_LINES[slot % len(_SOLVER_FALLBACK_LINES)],
    )


def _z_range(*arrays: np.ndarray, pad_pct: float = 0.10) -> tuple[float, float]:
    """Compute a padded ``[z_min, z_max]`` covering every input array.

    Used to lock the IV (%) axis identically across the static overlay
    and the animation so the converged frame of the animation reads the
    same axis ticks as the standalone fit chart.
    """
    stacked = np.concatenate([np.asarray(a).ravel() for a in arrays])
    z_min = float(np.nanmin(stacked))
    z_max = float(np.nanmax(stacked))
    pad = max(1.0, pad_pct * (z_max - z_min))
    return z_min - pad, z_max + pad


_MARKET_POINT_COLOR = "#dc2626"  # red — matches MARKET_COLOR on the smile chart


def _vis(key: str, default):
    """Read a presentation-only visual setting from the Streamlit session.

    Falls back to ``default`` outside a Streamlit runtime (e.g. in tests), so the
    chart builders stay pure and testable while the Setup "Visual settings"
    expander can retune markers/opacity without regenerating the surface data.
    """
    try:
        import streamlit as st

        val = st.session_state.get(key, default)
        return default if val is None else val
    except Exception:
        return default


def _market_points_scatter(
    M: np.ndarray,
    T_days: np.ndarray,
    z_market: np.ndarray,
    *,
    x_hover_label: str,
    x_hover_fmt: str,
    legendgroup: str | None = None,
    reference_label: str = "Market",
) -> go.Scatter3d:
    """Scatter overlay of the actual target IV quotes.

    ``reference_label`` names the surface these quotes belong to — "Market" for
    real data, or the generating model's name (e.g. "Heston") for a synthetic
    surface.

    NaN cells (missing quotes) are filtered out so only real observations
    appear. Drawn on top of the surface so the viewer can see where the
    surface interpolates versus where it sits on an actual quote.

    Colour: red (``_MARKET_POINT_COLOR``) — matches the smile chart's
    market trace, distinct from any model/solver hue.

    ``legendgroup`` lets the caller bundle this scatter with the matching
    market ``go.Surface`` so a single legend click toggles both at once
    (the surface owns the visible legend entry; this scatter rides along).
    """
    M_flat = np.asarray(M).ravel()
    T_flat = np.asarray(T_days).ravel()
    Z_flat = np.asarray(z_market).ravel()
    valid = ~np.isnan(Z_flat)
    return go.Scatter3d(
        x=M_flat[valid],
        y=T_flat[valid],
        z=Z_flat[valid],
        mode="markers",
        name=f"{reference_label} IV quotes",
        # Filled red dot by default (``size=5``) so the market quotes read as
        # solid points at typical chart sizes. Flip ``calib_vis_marker_fill``
        # off to get ``circle-open`` — the only scatter3d symbol with an
        # unfilled outline — which turns each point into a ring so a perfect
        # fit shows the solver's filled diamond *inside* it rather than hidden
        # beneath. ``line.width=2`` keeps the outline readable either way (the
        # same-colour contour is harmless on a filled point).
        marker=dict(
            size=_vis("calib_vis_marker_size", 5),
            color=_vis("calib_vis_marker_color", _MARKET_POINT_COLOR),
            symbol="circle" if _vis("calib_vis_marker_fill", True) else "circle-open",
            line=dict(
                color=_vis("calib_vis_marker_color", _MARKET_POINT_COLOR), width=2
            ),
            opacity=_vis("calib_vis_marker_opacity", 0.95),
        ),
        hovertemplate=(
            f"<b>{reference_label} quote</b><br>{x_hover_label}=%{{x:{x_hover_fmt}}}<br>"
            "T=%{y:.0f}d<br>σ_IV=%{z:.2f}%<extra></extra>"
        ),
        showlegend=False,
        legendgroup=legendgroup,
    )


def _model_points_scatter(
    M: np.ndarray,
    T_days: np.ndarray,
    z_model: np.ndarray,
    *,
    color: str,
    solver_name: str,
    x_hover_label: str,
    x_hover_fmt: str,
    legendgroup: str | None = None,
    mask: np.ndarray | None = None,
) -> go.Scatter3d:
    """Scatter overlay of the *calibrated* model IV at the same (K, T) grid.

    Uses ``diamond`` markers in the solver-specific colour so the eye can
    immediately distinguish ``market`` (red circles) from ``model``
    (solver-coloured diamonds) — even when the two surfaces overlap.

    ``legendgroup`` ties this scatter to its parent ``go.Surface`` so a
    single legend click toggles surface + dots together. ``mask`` (flat bool)
    selects which cells to plot; pass the MARKET grid's valid mask for the
    animated overlay so the initial frame shows the same diamond set every
    animation frame restyles to (they use the market mask) — otherwise the
    dot count silently changed on the first Replay/slider.
    """
    M_flat = np.asarray(M).ravel()
    T_flat = np.asarray(T_days).ravel()
    Z_flat = np.asarray(z_model).ravel()
    valid = ~np.isnan(Z_flat) if mask is None else np.asarray(mask).ravel()
    return go.Scatter3d(
        x=M_flat[valid],
        y=T_flat[valid],
        z=Z_flat[valid],
        mode="markers",
        name=f"{solver_name} IV fit",
        # Filled solver-coloured diamond sized just under the inner
        # diameter of the market ring (size=8, open). At a perfect fit
        # the diamond nests inside the ring with both shapes still
        # distinguishable; at a poor fit the two markers separate
        # vertically and the gap reads as residual error.
        marker=dict(
            size=4,
            color=color,
            symbol="diamond",
            line=dict(color="#ffffff", width=0.8),
            opacity=1.0,
        ),
        hovertemplate=(
            f"<b>{solver_name} fit</b><br>{x_hover_label}=%{{x:{x_hover_fmt}}}<br>"
            "T=%{y:.0f}d<br>σ_IV=%{z:.2f}%<extra></extra>"
        ),
        showlegend=False,
        legendgroup=legendgroup,
    )


def _scene(
    *,
    x_axis_title: str,
    x_tickformat: str,
    z_range: tuple[float, float],
    uirevision: str,
) -> dict:
    """Shared 3D scene layout used by every IV-surface chart.

    Locking the camera, the aspect mode (cube — proportions stay fixed
    regardless of data range) and the Z-axis range here is the only
    way to guarantee that the static overlay and the animation produce
    visually identical surfaces when fed the same fit. Without these
    constraints Plotly auto-rescales each scene independently and the
    same converged surface ends up looking different in the two charts.

    ``uirevision`` is set *inside* the scene (not only on layout). For
    3D scenes during animation Play, plotly.js does **not** honour the
    layout-level token (known bug — plotly.js #5050, #6359, streamlit
    #4653): the ``redraw=True`` triggered by Play re-applies whatever
    ``camera`` value is hardcoded in ``layout.scene``, snapping the
    user's manual rotation back to that hardcoded ``eye``. The only
    Python-only workaround that actually works is to **never set an
    explicit camera value** here — Plotly uses its built-in default
    ``eye=(1.25, 1.25, 1.25)`` on first render, the user rotates from
    there, and subsequent redraws have nothing hardcoded to fight
    against the user's state.
    """
    return dict(
        xaxis=dict(title=x_axis_title, tickformat=x_tickformat),
        yaxis=dict(title="Time to Maturity T (days)", tickformat=".0f"),
        zaxis=dict(
            title="Implied Volatility σ (%)",
            tickformat=".1f",
            range=list(z_range),
        ),
        bgcolor=COLORS["plot"],
        # "Droit" front-view default — looking down the +Y (maturity)
        # axis with Z (IV %) up, slight elevation for depth. Plotly's
        # built-in default eye=(1.25, 1.25, 1.25) gave a strongly
        # diagonal view that the user found awkward to read. The
        # camera-reset-on-Play risk is mitigated by the per-frame
        # ``scene.uirevision`` token set on every go.Frame.layout
        # (plotly.js honours uirevision per-frame even when it would
        # otherwise re-apply the layout-level camera value).
        camera=dict(
            eye=dict(x=0.0, y=2.2, z=0.6),
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
        ),
        aspectmode="cube",
        # ``turntable`` is Plotly's default and the more intuitive
        # rotation gesture for an IV surface (rotates around a fixed
        # vertical axis, keeping "up" stable). Camera persistence
        # across redraws relies on the per-frame ``scene.uirevision``
        # token set on every go.Frame.layout instead.
        dragmode="turntable",
        uirevision=uirevision,
    )


def _resolve_axis(
    moneyness: np.ndarray,
    x_label: str,
    atm_x: float,
    *,
    hover_label: str | None = None,
    hover_fmt: str | None = None,
    tickformat: str = ".2f",
) -> tuple[np.ndarray, str, str, str, str]:
    """Shared plot-axis spec for every surface/smile chart.

    ``moneyness`` is the x-vector — 1D ``(n_K,)`` when shared across maturities
    (σ√T-standardized moneyness, ATM at 0; or K/S₀ for real data, ATM at 1), or
    2D ``(n_T, n_K)`` for a per-maturity standard axis (strike $, ln(K/F), K/F).
    When ``hover_label`` is ``None`` it is derived from ``atm_x`` (legacy ``m`` /
    ``K/S₀`` behaviour). Returns ``(x_vals, axis_title, tickformat, hover_label,
    hover_fmt)``.
    """
    if hover_label is None:
        is_ratio = float(atm_x) != 0.0
        hover_label = "K/S₀" if is_ratio else "m"
        hover_fmt = ".3f" if is_ratio else ".2f"
    return (
        np.asarray(moneyness, dtype=float),
        x_label,
        tickformat,
        hover_label,
        hover_fmt or ".2f",
    )


def _resolve_grid(
    x_vals: np.ndarray, maturities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return matching 2D ``(M, T_days)`` grids for ``go.Surface``/scatter.

    ``x_vals`` may be 1D (shared axis → meshgrid, the legacy behaviour) or 2D
    (per-maturity axis → passthrough with ``T_days`` broadcast to the same
    shape). The two cases coincide for a shared axis because every meshgrid row
    equals the 1D vector, so callers can always index ``M[i, :]`` for maturity
    ``i``.
    """
    x = np.asarray(x_vals, dtype=float)
    mat = np.asarray(maturities, dtype=float)
    if x.ndim == 1:
        M, T = np.meshgrid(x, mat)
    else:
        M = x
        T = np.broadcast_to(mat[:, None], x.shape)
    return M, np.asarray(T, dtype=float) * 365.0


def render_iv_surface_3d(
    iv_grid: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    *,
    moneyness: np.ndarray,
    x_label: str = "Moneyness  ln(K/F) / (σ√T)",
    atm_x: float = 0.0,
    hover_label: str | None = None,
    hover_fmt: str | None = None,
    tickformat: str = ".2f",
    title: str = "Implied Volatility Surface",
    spot: float | None = None,
    reference_label: str = "Market",
) -> go.Figure:
    # Plot on the chosen display axis (σ√T-standardized moneyness, ln(K/F), K/F
    # or the dollar strike — 1D when shared, 2D when per-maturity); the
    # per-maturity dollar strikes ride along as customdata so the hover still
    # names a concrete K=$.
    x_axis_vals, x_axis_title, x_tickformat, x_hover_label, x_hover_fmt = _resolve_axis(
        moneyness,
        x_label,
        atm_x,
        hover_label=hover_label,
        hover_fmt=hover_fmt,
        tickformat=tickformat,
    )

    M, T_days = _resolve_grid(x_axis_vals, maturities)
    z = iv_grid * 100.0  # %

    fig = go.Figure(
        data=[
            go.Surface(
                x=M,
                y=T_days,
                z=z,
                customdata=np.asarray(strikes, dtype=float),
                colorscale=_MARKET_COLORSCALE,
                colorbar=dict(
                    title=dict(
                        text="Implied Volatility σ (%)",
                        font=dict(family=FONT_FAMILY, color=COLORS["axis"]),
                    ),
                    tickfont=dict(family=FONT_FAMILY, color=COLORS["axis"]),
                    thickness=14,
                    xpad=8,
                    outlinewidth=0,
                ),
                showscale=True,
                contours=dict(
                    z=dict(
                        show=True,
                        usecolormap=True,
                        highlightcolor=COLORS["primary"],
                        project_z=True,
                    )
                ),
                hovertemplate=(
                    f"{x_hover_label}=%{{x:{x_hover_fmt}}}<br>"
                    "K=$%{customdata:.0f}<br>"
                    "T=%{y:.0f}d<br>"
                    "σ_IV=%{z:.2f}%<extra></extra>"
                ),
            ),
            _market_points_scatter(
                M,
                T_days,
                z,
                x_hover_label=x_hover_label,
                x_hover_fmt=x_hover_fmt,
                reference_label=reference_label,
            ),
        ]
    )
    apply_lab_theme(fig, height=540, title=title, margin=(0, 0, 50, 0))
    z_range = _z_range(z)
    fig.update_layout(
        scene=_scene(
            x_axis_title=x_axis_title,
            x_tickformat=x_tickformat,
            z_range=z_range,
            uirevision="iv-surface-market",
        ),
        showlegend=False,
        # Preserve user-driven camera rotation across Streamlit reruns.
        uirevision="iv-surface-market",
    )
    return fig


def _smile_maturity_menu(
    maturities: np.ndarray,
    order: np.ndarray,
    *,
    traces_per_maturity: int,
    n_legend_proxies: int = 0,
) -> dict:
    """Plotly dropdown that toggles smile traces by maturity.

    Each maturity owns ``traces_per_maturity`` consecutive traces (1 for
    the market-only chart, 2 for the market+fit overlay), laid out first.
    ``n_legend_proxies`` trailing legend-only proxy traces (the compact
    2-axis key on the overlay) stay visible for every button so the legend
    always reads as a static key. The button list starts with an "All
    maturities" option and then walks through ``order`` short→long.
    """
    n = len(order)
    total = n * traces_per_maturity
    proxy_tail = [True] * n_legend_proxies
    buttons = [
        dict(
            label="All maturities",
            method="update",
            args=[{"visible": [True] * total + proxy_tail}],
        )
    ]
    for slot, i in enumerate(order):
        visible = [False] * total
        for k in range(traces_per_maturity):
            visible[slot * traces_per_maturity + k] = True
        buttons.append(
            dict(
                label=f"T = {maturities[i] * 365:.0f}d",
                method="update",
                args=[{"visible": visible + proxy_tail}],
            )
        )
    return dict(
        type="dropdown",
        direction="down",
        x=1.0,
        y=1.18,
        xanchor="right",
        yanchor="top",
        pad=dict(t=2, r=4, b=2, l=4),
        showactive=True,
        # White-on-light dropdown that reads cleanly on plotly_white;
        # teal border ties it to the Play/Pause cluster for visual
        # consistency across the chart's interactive chrome.
        bgcolor="#ffffff",
        bordercolor=COLORS["primary"],
        font=dict(family=FONT_FAMILY, color=COLORS["text"], size=11),
        buttons=buttons,
    )


def render_smile_slices(
    iv_grid: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    *,
    moneyness: np.ndarray,
    x_label: str = "Moneyness  ln(K/F) / (σ√T)",
    atm_x: float = 0.0,
    hover_label: str | None = None,
    hover_fmt: str | None = None,
    tickformat: str = ".2f",
    spot: float = 100.0,
) -> go.Figure:
    fig = go.Figure()
    strikes = np.asarray(strikes, dtype=float)  # (n_T, n_K) per-maturity dollar strikes
    x_axis_vals, _, _, x_hover_label, x_hover_fmt = _resolve_axis(
        moneyness,
        x_label,
        atm_x,
        hover_label=hover_label,
        hover_fmt=hover_fmt,
        tickformat=tickformat,
    )
    # Per-maturity x rows: M[i, :] equals the shared 1D axis when it is shared,
    # and the maturity-specific strikes/moneyness when the axis is 2D.
    M, _ = _resolve_grid(x_axis_vals, maturities)
    # Sort maturities ascending so the legend reads short→long, matching
    # the visual ordering of the smiles bottom→top.
    order = np.argsort(maturities)
    for plot_idx, i in enumerate(order):
        T = float(maturities[i])
        label = f"T = {T * 365:.0f}d"
        fig.add_trace(
            go.Scatter(
                x=M[i, :],
                y=iv_grid[i, :] * 100.0,
                customdata=strikes[i, :],
                mode="lines+markers",
                name=label,
                legendgroup=label,
                legendgrouptitle_text=("Maturities" if plot_idx == 0 else None),
                line=dict(color=PALETTE[i % len(PALETTE)], width=2.2),
                marker=dict(size=7, line=dict(width=1, color=COLORS["plot"])),
                hovertemplate=(
                    f"{x_hover_label}=%{{x:{x_hover_fmt}}}<br>K=$%{{customdata:.0f}}"
                    "<br>IV=%{y:.2f}%<extra></extra>"
                ),
            )
        )
    fig.add_vline(
        x=atm_x,
        line=dict(color=COLORS["axis"], dash="dot", width=1),
        annotation_text="ATM",
        annotation_position="top",
        annotation_font=dict(family=FONT_FAMILY, color=COLORS["axis"], size=10),
    )
    # No Plotly title — the section header rendered by the tab already
    # names the panel; a second title inside the chart would duplicate it.
    apply_lab_theme(fig, height=420)
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title="Implied Volatility σ (%)")
    fig.update_layout(
        updatemenus=[_smile_maturity_menu(maturities, order, traces_per_maturity=1)]
    )
    return fig


_FIT_DASH_STYLES = ["dash", "dot", "dashdot", "longdash", "longdashdot"]


def render_smile_slices_overlay(
    market_iv_grid: np.ndarray,
    model_iv_grids: np.ndarray | dict[str, np.ndarray],
    strikes: np.ndarray,
    maturities: np.ndarray,
    *,
    moneyness: np.ndarray,
    x_label: str = "Moneyness  ln(K/F) / (σ√T)",
    atm_x: float = 0.0,
    hover_label: str | None = None,
    hover_fmt: str | None = None,
    tickformat: str = ".2f",
    spot: float = 100.0,
    solver_name: str = "model",
    reference_label: str = "Market",
) -> go.Figure:
    """Per-maturity smile with the target quotes + one or more model fits.

    ``reference_label`` names the target series — "Market" for real data, or the
    generating model's name (e.g. "Heston") for a synthetic surface.

    Color encodes *maturity* (consistent across solvers and the
    market-only chart). Line style encodes *solver* — first solver gets
    a dashed line, then dotted, dash-dot, long-dash, etc. Market always
    appears as filled markers. This factoring keeps a 4-solver overlay
    readable: the eye finds a maturity by color, then disambiguates
    solvers by line stroke.

    Accepts either a single ``np.ndarray`` (legacy single-solver call —
    paired with ``solver_name``) or a dict ``{solver_name: iv_grid}``.
    """
    if isinstance(model_iv_grids, dict):
        models = model_iv_grids
    else:
        models = {solver_name: np.asarray(model_iv_grids)}

    fig = go.Figure()
    strikes = np.asarray(strikes, dtype=float)  # (n_T, n_K) per-maturity dollar strikes
    x_axis_vals, _, _, x_hover_label, x_hover_fmt = _resolve_axis(
        moneyness,
        x_label,
        atm_x,
        hover_label=hover_label,
        hover_fmt=hover_fmt,
        tickformat=tickformat,
    )
    # Per-maturity x rows (M[i, :]); identical to the shared 1D axis when shared.
    M, _ = _resolve_grid(x_axis_vals, maturities)
    # Sort maturities short→long so the dropdown reads in chronological
    # order — the user thinks "1m, 3m, 6m…", not "row 0, row 1, row 2".
    order = np.argsort(maturities)
    traces_per_maturity = 1 + len(models)  # 1 market + N model fits

    # Data traces first, in maturity-major order (market, then each fit) so the
    # maturity dropdown below can address them by block. Identity is carried by
    # colour (maturity) and dash (fit), so these hold NO legend entry — a
    # compact 2-axis key is added afterwards. The previous one-entry-per
    # (maturity × series) legend overflowed the chart and the verbose series
    # label was repeated for every maturity.
    for i in order:
        col = PALETTE[i % len(PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=M[i, :],
                y=market_iv_grid[i, :] * 100.0,
                customdata=strikes[i, :],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=8, color=col, line=dict(width=1, color=COLORS["plot"])
                ),
                hovertemplate=(
                    f"{reference_label}<br>{x_hover_label}=%{{x:{x_hover_fmt}}}"
                    "<br>K=$%{customdata:.0f}<br>IV=%{y:.2f}%<extra></extra>"
                ),
            )
        )
        for slot, (name, grid) in enumerate(models.items()):
            dash_style = _FIT_DASH_STYLES[slot % len(_FIT_DASH_STYLES)]
            fig.add_trace(
                go.Scatter(
                    x=M[i, :],
                    y=grid[i, :] * 100.0,
                    mode="lines",
                    showlegend=False,
                    line=dict(color=col, width=2.2, dash=dash_style),
                    hovertemplate=(
                        f"{name}<br>{x_hover_label}=%{{x:{x_hover_fmt}}}"
                        "<br>IV=%{y:.2f}%<extra></extra>"
                    ),
                )
            )

    # Compact 2-axis legend via legend-only proxy traces (no plotted data):
    # one swatch per maturity (colour) + one per fit (dash) + a market marker.
    # The long series label now appears exactly once instead of per maturity.
    _MUTED = "#94a3b8"
    for slot_i, i in enumerate(order):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines+markers",
                name=f"T = {maturities[i] * 365:.0f}d",
                legendgroup="mat",
                legendgrouptitle_text="Maturities" if slot_i == 0 else None,
                line=dict(color=PALETTE[i % len(PALETTE)], width=2.2),
                marker=dict(size=8, color=PALETTE[i % len(PALETTE)]),
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=reference_label,
            legendgroup="fit",
            legendgrouptitle_text="Fits",
            marker=dict(size=8, color=_MUTED, line=dict(width=1, color=COLORS["plot"])),
            hoverinfo="skip",
        )
    )
    for slot, name in enumerate(models):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=name,
                legendgroup="fit",
                line=dict(
                    color=_MUTED,
                    width=2.2,
                    dash=_FIT_DASH_STYLES[slot % len(_FIT_DASH_STYLES)],
                ),
                hoverinfo="skip",
            )
        )
    n_legend_proxies = len(order) + 1 + len(models)

    fig.add_vline(
        x=atm_x,
        line=dict(color=COLORS["axis"], dash="dot", width=1),
        annotation_text="ATM",
        annotation_position="top",
        annotation_font=dict(family=FONT_FAMILY, color=COLORS["axis"], size=10),
    )
    # No Plotly title — the tab renders its own section header above
    # the chart, so a duplicate title inside the figure would clutter.
    apply_lab_theme(fig, height=460)
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title="Implied Volatility σ (%)")
    fig.update_layout(
        updatemenus=[
            _smile_maturity_menu(
                maturities,
                order,
                traces_per_maturity=traces_per_maturity,
                n_legend_proxies=n_legend_proxies,
            )
        ]
    )
    return fig


def render_iv_surface_overlay_3d(
    market_iv_grid: np.ndarray,
    model_iv_grids: np.ndarray | dict[str, np.ndarray],
    strikes: np.ndarray,
    maturities: np.ndarray,
    *,
    moneyness: np.ndarray,
    x_label: str = "Moneyness  ln(K/F) / (σ√T)",
    atm_x: float = 0.0,
    hover_label: str | None = None,
    hover_fmt: str | None = None,
    tickformat: str = ".2f",
    title: str = "Target vs calibrated surface",
    solver_name: str = "model",
    spot: float | None = None,
    reference_label: str = "Market",
) -> go.Figure:
    """One reference surface (cool blue) overlaid with N model surfaces.

    ``reference_label`` names the reference surface — "Market" for real data, or
    the generating model's name (e.g. "Heston") for a synthetic surface.

    Accepts either a single ``np.ndarray`` (legacy single-solver call —
    paired with ``solver_name``) or a dict ``{solver_name: iv_grid}`` for
    multi-solver comparison. Each model surface gets a distinct hue from
    ``_SOLVER_COLORSCALES``; the opacity drops slightly when more than
    one model is overlaid so the user can see through the stack.
    """
    if isinstance(model_iv_grids, dict):
        models = model_iv_grids
    else:
        models = {solver_name: np.asarray(model_iv_grids)}

    x_axis_vals, x_axis_title, x_tickformat, x_hover_label, x_hover_fmt = _resolve_axis(
        moneyness,
        x_label,
        atm_x,
        hover_label=hover_label,
        hover_fmt=hover_fmt,
        tickformat=tickformat,
    )

    M, T_days = _resolve_grid(x_axis_vals, maturities)
    z_market = market_iv_grid * 100.0

    fig = go.Figure()
    # Bundle every market trace into the "market" legendgroup and every
    # model's (surface + fit scatter) pair into a group keyed on the model
    # name. A single click on a legend entry then toggles both visuals at
    # once — the user gets full control over what's drawn on the scene
    # when several candidate models are stacked together.
    _add_market_surface(
        fig,
        M,
        T_days,
        z_market,
        x_hover_label=x_hover_label,
        x_hover_fmt=x_hover_fmt,
        legendgroup="market",
        reference_label=reference_label,
    )
    fig.add_trace(
        _market_points_scatter(
            M,
            T_days,
            z_market,
            x_hover_label=x_hover_label,
            x_hover_fmt=x_hover_fmt,
            legendgroup="market",
            reference_label=reference_label,
        )
    )
    overlay_opacity = 0.85 if len(models) <= 1 else 0.65
    for slot, (name, grid) in enumerate(models.items()):
        _add_model_surface(
            fig,
            M,
            T_days,
            grid * 100.0,
            solver_name=name,
            slot=slot,
            opacity=overlay_opacity,
            x_hover_label=x_hover_label,
            x_hover_fmt=x_hover_fmt,
            legendgroup=name,
        )
        fig.add_trace(
            _model_points_scatter(
                M,
                T_days,
                grid * 100.0,
                color=_solver_line(name, slot),
                solver_name=name,
                x_hover_label=x_hover_label,
                x_hover_fmt=x_hover_fmt,
                legendgroup=name,
            )
        )
    apply_lab_theme(fig, height=540, title=title, margin=(0, 0, 50, 0))
    z_range = _z_range(z_market, *(g * 100.0 for g in models.values()))
    fig.update_layout(
        scene=_scene(
            x_axis_title=x_axis_title,
            x_tickformat=x_tickformat,
            z_range=z_range,
            uirevision="iv-surface-overlay",
        ),
        # Interactive Plotly legend (replaces the previous static
        # annotation block): single-click an entry to hide/show its
        # surface, double-click to isolate a single model. Anchored
        # top-right inside the scene so it matches where the user is
        # already used to looking for the model labels.
        showlegend=True,
        legend=dict(
            orientation="v",
            x=0.99,
            y=0.97,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=COLORS["axis"],
            borderwidth=1,
            font=dict(family=FONT_FAMILY, size=11),
            itemsizing="constant",
        ),
        # Preserve user-driven camera rotation across Streamlit reruns and
        # animation frames; without this Plotly resets the view on every redraw.
        uirevision="iv-surface-overlay",
    )
    return fig


def _add_market_surface(
    fig: go.Figure,
    M: np.ndarray,
    T_days: np.ndarray,
    z_market: np.ndarray,
    *,
    x_hover_label: str,
    x_hover_fmt: str,
    legendgroup: str | None = None,
    row: int | None = None,
    col: int | None = None,
    reference_label: str = "Market",
) -> None:
    """Append the cool-blue reference (target) surface to ``fig``.

    ``reference_label`` names the surface — "Market" for real data, or the
    generating model's name (e.g. "Heston") for a synthetic surface.

    ``legendgroup`` enables Plotly's bundled click-to-toggle: when the
    caller (the multi-model overlay) gives both this surface and the
    paired scatter the same group key, a single legend click hides or
    re-shows the *whole* reference layer (surface + ring scatter) at once.
    Default ``None`` keeps single-solver call sites (the animation)
    legend-neutral.
    """
    fig.add_trace(
        go.Surface(
            x=M,
            y=T_days,
            z=z_market,
            name=reference_label,
            colorscale=_MARKET_COLORSCALE,
            showscale=False,
            opacity=_vis("calib_vis_market_surf_opacity", 0.45),
            # Wireframe contours reinforce the market layer even when the
            # model surface lies right on top of it.
            contours=dict(
                x=dict(show=True, color=_MARKET_LINE, width=1, highlight=False),
                y=dict(show=True, color=_MARKET_LINE, width=1, highlight=False),
            ),
            hovertemplate=(
                f"{reference_label}<br>{x_hover_label}=%{{x:{x_hover_fmt}}}<br>"
                "T=%{y:.0f}d<br>IV=%{z:.2f}%<extra></extra>"
            ),
            legendgroup=legendgroup,
            # ``go.Surface`` defaults to ``showlegend=False`` (unlike
            # ``go.Scatter``) — a layout-level ``showlegend=True`` does NOT
            # propagate, so without this explicit opt-in the multi-model
            # overlay renders an empty legend and the click-to-toggle
            # affordance is gone. Setting it True here makes the surface
            # itself the visible legend entry that toggles its whole
            # ``legendgroup`` (surface + paired scatter) on click.
            showlegend=True,
        ),
        row=row,
        col=col,
    )


def _add_model_surface(
    fig: go.Figure,
    M: np.ndarray,
    T_days: np.ndarray,
    z_model: np.ndarray,
    *,
    solver_name: str,
    x_hover_label: str,
    x_hover_fmt: str,
    slot: int = 0,
    opacity: float | None = None,
    legendgroup: str | None = None,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Append a model surface to ``fig`` with solver-specific coloring.

    ``slot`` is the position of this solver in the multi-overlay order;
    it falls back to a rotating palette when the solver is not in
    ``_SOLVER_COLORSCALES``. Opacity defaults to 0.85 for a single
    overlay and drops to 0.65 when several model surfaces co-exist so
    the user can see through the upper layers to the ones beneath.

    ``legendgroup`` lets the caller bundle this surface with its
    matching fit-points scatter so a single legend click hides or shows
    both at once. Default ``None`` keeps the animation call site
    legend-neutral (it disables the legend at the layout level anyway).
    """
    line_color = _solver_line(solver_name, slot)
    fig.add_trace(
        go.Surface(
            x=M,
            y=T_days,
            z=z_model,
            name=solver_name,
            colorscale=_solver_colorscale(solver_name, slot),
            showscale=False,
            opacity=0.85 if opacity is None else opacity,
            contours=dict(
                x=dict(show=True, color=line_color, width=1, highlight=False),
                y=dict(show=True, color=line_color, width=1, highlight=False),
            ),
            hovertemplate=(
                f"{solver_name}<br>{x_hover_label}=%{{x:{x_hover_fmt}}}<br>"
                "T=%{y:.0f}d<br>IV=%{z:.2f}%<extra></extra>"
            ),
            legendgroup=legendgroup,
            # Per-trace opt-in: ``go.Surface`` defaults to hidden in the
            # legend even when ``layout.showlegend=True``. This is the
            # entry the user clicks to hide/show this model's surface +
            # its paired fit scatter (linked via legendgroup).
            showlegend=True,
        ),
        row=row,
        col=col,
    )


def _surface_legend_annotations(
    solver_names: list[str], reference_label: str = "Market"
) -> list[dict]:
    """Inline color-coded labels stacked in the top-right corner.

    Replaces the absent legend for ``go.Surface`` traces — the user needs
    to know at a glance which surface is the reference (``reference_label``:
    "Market" or the generating model's name) and which is each (calibrated)
    model. Stacks vertically so multiple solver rows stay readable without
    overlapping.
    """
    annotations = [
        dict(
            xref="paper",
            yref="paper",
            x=0.99,
            y=0.97,
            xanchor="right",
            yanchor="top",
            text=f"<span style='color:{_MARKET_LINE}'>● {reference_label}</span>",
            showarrow=False,
            font=dict(family=FONT_FAMILY, size=12, color=_MARKET_LINE),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=_MARKET_LINE,
            borderwidth=1,
            borderpad=3,
        )
    ]
    for slot, name in enumerate(solver_names):
        line_color = _solver_line(name, slot)
        annotations.append(
            dict(
                xref="paper",
                yref="paper",
                x=0.99,
                y=0.91 - slot * 0.06,
                xanchor="right",
                yanchor="top",
                text=f"<span style='color:{line_color}'>● {name}</span>",
                showarrow=False,
                font=dict(family=FONT_FAMILY, size=12, color=line_color),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor=line_color,
                borderwidth=1,
                borderpad=3,
            )
        )
    return annotations


def render_iv_surface_animation(
    frames,
    market_iv_grid: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    *,
    moneyness: np.ndarray,
    x_label: str = "Moneyness  ln(K/F) / (σ√T)",
    atm_x: float = 0.0,
    hover_label: str | None = None,
    hover_fmt: str | None = None,
    tickformat: str = ".2f",
    solver_name: str = "model",
    spot: float | None = None,
    reference_label: str = "Market",
) -> go.Figure:
    """Animate the model IV surface across iteration snapshots.

    ``frames`` is a list of ``IVAnimationFrame`` (iter_index, objective,
    iv_grid). The market surface is rendered as a static reference at
    low opacity; the model surface morphs frame-by-frame via Plotly
    animations with a Play/Pause + slider.
    """
    if not frames:
        from utils.plotly_theme import empty_state_figure

        return empty_state_figure(
            "No animation frames available — calibration may have failed."
        )

    x_axis_vals, x_axis_title, x_tickformat, x_hover_label, x_hover_fmt = _resolve_axis(
        moneyness,
        x_label,
        atm_x,
        hover_label=hover_label,
        hover_fmt=hover_fmt,
        tickformat=tickformat,
    )

    M, T_days = _resolve_grid(x_axis_vals, maturities)
    z_market = market_iv_grid * 100.0

    # Stack every frame's IV grid up-front so the z-range below covers the
    # *whole* convergence trajectory — locking the range to the first frame
    # alone clipped the model surface as soon as the optimizer pushed IVs
    # outside the initial bounds.
    z_frames = np.stack([fr.iv_grid * 100.0 for fr in frames])
    # Open on the CONVERGED surface so the fit is visible immediately after a
    # run; ▶ Replay (fromcurrent=False below) animates the descent from the
    # first iteration on demand.
    z_init = z_frames[-1]
    fig = go.Figure()
    # Trace 0: market reference. Trace 1: model surface — animated below.
    # Helpers guarantee the color/opacity/contour treatment stays in lock-step
    # with render_iv_surface_overlay_3d (single-solver path, slot=0), so the
    # converged frame matches the static fit chart exactly.
    _add_market_surface(
        fig,
        M,
        T_days,
        z_market,
        x_hover_label=x_hover_label,
        x_hover_fmt=x_hover_fmt,
        reference_label=reference_label,
    )
    _add_model_surface(
        fig,
        M,
        T_days,
        z_init,
        solver_name=solver_name,
        slot=0,
        x_hover_label=x_hover_label,
        x_hover_fmt=x_hover_fmt,
    )
    # Trace 2: target IV quotes (red circles) — static reference.
    fig.add_trace(
        _market_points_scatter(
            M,
            T_days,
            z_market,
            x_hover_label=x_hover_label,
            x_hover_fmt=x_hover_fmt,
            reference_label=reference_label,
        )
    )
    # Pre-flatten the MARKET valid mask for the animated model scatter —
    # markers only exist where the market grid has a real quote, and every
    # frame restyles to this mask, so the initial trace must use it too (the
    # model grid's own NaN mask differs, which changed the dot count on the
    # first Replay/slider).
    M_flat = np.asarray(M).ravel()
    T_flat_days = np.asarray(T_days).ravel()
    valid_mask = ~np.isnan(np.asarray(z_market).ravel())
    M_valid = M_flat[valid_mask]
    T_valid = T_flat_days[valid_mask]

    # Trace 3: model IV at the same grid — animated alongside the surface
    # so the user can see the calibrated dots converge onto the market
    # dots iteration by iteration.
    model_point_color = _solver_line(solver_name, 0)
    fig.add_trace(
        _model_points_scatter(
            M,
            T_days,
            z_init,
            color=model_point_color,
            solver_name=solver_name,
            x_hover_label=x_hover_label,
            x_hover_fmt=x_hover_fmt,
            mask=valid_mask,
        )
    )

    plotly_frames = []
    slider_steps = []
    for i, fr in enumerate(frames):
        z = fr.iv_grid * 100.0
        z_valid = z.ravel()[valid_mask]
        plotly_frames.append(
            go.Frame(
                name=str(i),
                # Restyle trace 1 (model surface) and trace 3 (model
                # scatter) in lock-step so dots and mesh always show the
                # same iteration.
                data=[
                    go.Surface(x=M, y=T_days, z=z),
                    go.Scatter3d(x=M_valid, y=T_valid, z=z_valid),
                ],
                traces=[1, 3],
                # Repeating the scene-level uirevision inside each frame
                # gives plotly.js a second hook to preserve the user's
                # camera through ``Plotly.animate`` redraws. Without
                # this, even with no hardcoded camera in the base
                # layout, the redraw=True path resets the eye to the
                # built-in default on every frame transition.
                layout=dict(scene=dict(uirevision="iv-surface-overlay")),
            )
        )
        slider_steps.append(
            dict(
                method="animate",
                args=[
                    [str(i)],
                    dict(
                        mode="immediate",
                        # 3D surfaces require redraw=True for the mesh to
                        # actually re-render with the new z values; with
                        # redraw=False the model surface vanishes from view.
                        # Camera preservation relies instead on the per-frame
                        # ``scene.uirevision`` token + dragmode='orbit'.
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0),
                    ),
                ],
                label=f"#{fr.iter_index}",
            )
        )
    fig.frames = plotly_frames

    apply_lab_theme(
        fig,
        # Tall enough to make rotation feel meaningful on a wide screen
        # — the static fit chart sits at 540 and the animation has the
        # extra Play/Pause + slider chrome below it, so it benefits
        # from a noticeably larger viewport. Bottom margin also widened
        # so the snapshot slider has breathing room under the 3D scene.
        height=840,
        title=f"Surface fit progression  ·  {solver_name}  ·  {len(frames)} frames",
        margin=(0, 0, 120, 110),
    )
    # Lock the Z-axis to the *converged* frame so the animation's final
    # state reads exactly the same axis ticks as render_iv_surface_overlay_3d
    # when given the same fit. Earlier-iteration frames may overshoot
    # this range and clip at the box edges — that is intentional and
    # informative (it visually flags "this iteration was wildly off the
    # market"). Locking instead to the whole-trajectory min/max would
    # match the previous behaviour but at the cost of a different aspect
    # ratio than the static chart, which is precisely the bug we're fixing.
    z_converged = z_frames[-1]
    z_range = _z_range(z_market, z_converged)
    fig.update_layout(
        scene=_scene(
            x_axis_title=x_axis_title,
            x_tickformat=x_tickformat,
            z_range=z_range,
            # Same scene-level uirevision as the static overlay so the
            # camera rotation the user set on either chart carries over
            # to the other — and survives every Play / frame redraw.
            uirevision="iv-surface-overlay",
        ),
        # Preserve the user's camera rotation across animation frames and
        # Streamlit reruns. Without this, Plotly snaps back to the default
        # eye on every frame redraw and the user cannot inspect the fit
        # from any angle they choose.
        uirevision="iv-animation",
        showlegend=False,
        annotations=_surface_legend_annotations([solver_name], reference_label),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                # Buttons sit just under the plot; the slider rides further
                # below so neither overlaps when the chart shrinks on a
                # narrower viewport.
                x=0.0,
                y=-0.08,
                xanchor="left",
                yanchor="top",
                pad=dict(t=4, r=10),
                showactive=False,
                # Solid teal background with white font: high-contrast on
                # the plotly_white canvas, instantly recognisable as a
                # call-to-action. Border is the darker teal so the button
                # cluster keeps a defined edge without a fragile 1-px line.
                bgcolor=COLORS["primary"],
                bordercolor=COLORS["primary_dim"],
                borderwidth=1,
                font=dict(family=FONT_FAMILY, color="#ffffff", size=12),
                buttons=[
                    dict(
                        label="▶  Replay",
                        method="animate",
                        args=[
                            None,
                            dict(
                                # redraw=True is required for 3D surfaces —
                                # without it the model mesh stops rendering.
                                # The per-frame ``scene.uirevision`` (set
                                # on every go.Frame) is what tells plotly.js
                                # to keep the user's camera through the
                                # redraws.
                                frame=dict(duration=400, redraw=True),
                                # The chart opens on the converged surface, so
                                # replay from the first iteration on demand.
                                fromcurrent=False,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="❚❚  Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                # Open on the last snapshot (converged), matching the base
                # surface shown above.
                active=len(slider_steps) - 1,
                # Center the snapshot slider under the 3D scene so the
                # Play/Pause cluster on the left no longer pulls it
                # visually off-axis.
                x=0.5,
                y=-0.16,
                len=0.78,
                xanchor="center",
                yanchor="top",
                currentvalue=dict(
                    prefix="snapshot: ",
                    font=dict(family=FONT_FAMILY, color=COLORS["text"], size=12),
                ),
                steps=slider_steps,
                font=dict(family=FONT_FAMILY, color=COLORS["axis"], size=10),
                # Light grey rail on the white canvas (was near-black,
                # invisible against the plotly_white background). Active
                # step keeps the teal primary so it pops against the rail.
                bgcolor="rgba(0,0,0,0.06)",
                activebgcolor=COLORS["primary"],
                bordercolor=COLORS["grid_strong"],
                tickcolor=COLORS["axis"],
            )
        ],
    )
    return fig


def render_surface_fit_anatomy_animated(
    model_groups,
    market_iv_grid: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    *,
    moneyness: np.ndarray,
    x_label: str = "Moneyness  ln(K/F) / (σ√T)",
    atm_x: float = 0.0,
    hover_label: str | None = None,
    hover_fmt: str | None = None,
    tickformat: str = ".2f",
    spot: float | None = None,
    reference_label: str = "Market",
    model_key: str | None = None,
    true_params: dict[str, float] | None = None,
) -> go.Figure:
    """Surface fit progression (3D) synchronized with its parameter trajectories.

    Overlays 1..N models, each with 1..N variations (solver / objective), inside a
    SINGLE figure. The morphing model-vs-market IV surfaces all share one 3D scene
    (LEFT, full height) over the shared market surface; every model gets its OWN
    column of per-parameter panels on the right (a model's variations overlaid in
    its column) because distinct models' parameter sets are not comparable and must
    not be merged onto one axis. One ▶ Replay / slider advances every run in
    lock-step; runs of unequal length freeze at their last frame.

    ``model_groups`` is a list of ``(model_key, display_name, runs, true_params)``
    where ``runs`` is ``[(label, solver, frames), …]`` and ``frames`` a list of
    ``IVAnimationFrame`` (``iv_grid`` + the ``params_natural`` that priced it).
    ``true_params`` (per group, may be ``None``) draws a red dashed reference line
    per parameter in that model's column. A native, clickable Plotly legend toggles
    each run's whole layer (surface + fit scatter + parameter lines share its
    ``legendgroup``); in multi-model mode the legend label is prefixed by the model.
    A single-model call reproduces the historical single-column layout exactly.
    """
    from utils.plotly_theme import empty_state_figure

    from charts.live_convergence import _display_labels

    # Backward-compatible legacy form: a bare ``runs`` list of
    # ``(label, solver, frames)`` triples plus ``model_key`` / ``true_params``
    # kwargs (single model). Wrap it into a one-element ``model_groups`` so the
    # single-model layout is byte-for-byte the historical one.
    if (
        model_key is not None
        or true_params is not None
        or (model_groups and len(model_groups[0]) == 3)
    ):
        model_groups = [(model_key, model_key or "", model_groups, true_params)]

    # Keep only groups that still have priceable frames.
    groups = []
    for model_key, display_name, runs, tp in model_groups:
        kept = [(label, solver, frames) for (label, solver, frames) in runs if frames]
        if kept:
            groups.append((model_key, display_name, kept, tp))
    if not groups:
        return empty_state_figure(
            "No animation frames available — calibration may have failed."
        )
    multi_model = len(groups) > 1

    # Surface axes on the chosen display frame (σ√T-standardized / K/S₀ / ln(K/F)…).
    x_axis_vals, x_axis_title, x_tickformat, x_hover_label, x_hover_fmt = _resolve_axis(
        moneyness,
        x_label,
        atm_x,
        hover_label=hover_label,
        hover_fmt=hover_fmt,
        tickformat=tickformat,
    )
    M, T_days = _resolve_grid(x_axis_vals, maturities)
    z_market = market_iv_grid * 100.0

    # Per group: its parameter names (all its variations share one model) and the
    # Greek display labels. The grid spans the widest model; shorter columns leave
    # their lower cells empty.
    group_names = [
        [k for k in runs[0][2][0].params_natural if not k.startswith("_")]
        for _mk, _dn, runs, _tp in groups
    ]
    group_disp = [
        _display_labels(mk, group_names[m])
        for m, (mk, _dn, _runs, _tp) in enumerate(groups)
    ]
    n_models = len(groups)
    max_rows = max((len(nm) for nm in group_names), default=1)
    max_rows = max(max_rows, 1)
    max_len = max(
        len(frames) for _mk, _dn, runs, _tp in groups for _l, _s, frames in runs
    )
    fx = np.arange(max_len, dtype=np.int64)

    # Geometry: the 3D scene spans the full height of column 1; each model owns one
    # column of stacked parameter panels (col = model index + 2). ``None`` cells
    # below a shorter model's last parameter stay empty.
    cols = 1 + n_models
    specs: list[list] = []
    for r in range(max_rows):
        row_spec: list = []
        for c in range(cols):
            if c == 0:
                row_spec.append(
                    {"type": "scene", "rowspan": max_rows} if r == 0 else None
                )
            else:
                row_spec.append({"type": "xy"} if r < len(group_names[c - 1]) else None)
        specs.append(row_spec)

    # Subplot titles in make_subplots' cell-iteration order (row-major, skipping
    # ``None`` cells). In multi-model mode the top panel of each column also carries
    # the model's display name as a column header.
    titles: list[str] = []
    for r in range(max_rows):
        for c in range(cols):
            if c == 0:
                if r == 0:
                    titles.append("")
                continue
            m = c - 1
            if r < len(group_names[m]):
                nm = group_names[m][r]
                label = group_disp[m][nm]
                if multi_model and r == 0:
                    label = f"<b>{groups[m][1]}</b><br>{label}"
                titles.append(label)

    if multi_model:
        scene_w = 0.48
        column_widths = [scene_w] + [(1.0 - scene_w) / n_models] * n_models
    else:
        column_widths = [0.62, 0.38]
    fig = make_subplots(
        rows=max_rows,
        cols=cols,
        specs=specs,
        column_widths=column_widths,
        vertical_spacing=min(0.045, 0.5 / max(max_rows, 1)),
        horizontal_spacing=0.06 if multi_model else 0.09,
        subplot_titles=titles,
    )

    # Animated model scatter shows markers only where the market has a quote.
    valid_mask = ~np.isnan(np.asarray(z_market).ravel())
    M_valid = np.asarray(M).ravel()[valid_mask]
    T_valid = np.asarray(T_days).ravel()[valid_mask]

    # ── Reference surface (legendgroup "market") — static, shared by all models ──
    _add_market_surface(
        fig,
        M,
        T_days,
        z_market,
        x_hover_label=x_hover_label,
        x_hover_fmt=x_hover_fmt,
        legendgroup="market",
        row=1,
        col=1,
        reference_label=reference_label,
    )
    fig.add_trace(
        _market_points_scatter(
            M,
            T_days,
            z_market,
            x_hover_label=x_hover_label,
            x_hover_fmt=x_hover_fmt,
            legendgroup="market",
            reference_label=reference_label,
        ),
        row=1,
        col=1,
    )

    # ── Per run (flattened across models): model surface + fit scatter (scene) +
    #    one line per parameter (this model's column). ``slot`` is a GLOBAL colour
    #    index so no two runs collide, even across models. Trace indices are tracked
    #    per run for the frame updates.
    overlay_opacity = _vis("calib_vis_model_surf_opacity", 0.8)
    ghost_opacity = _vis("calib_vis_ghost_opacity", 0.55)
    run_records: list[dict] = []  # per run: model idx, col, names, py, trace indices
    trace_idx = 2  # 0, 1 are the static market surface + scatter
    slot = 0
    for m, (model_key, display_name, runs, _tp) in enumerate(groups):
        col = m + 2
        names = group_names[m]
        disp = group_disp[m]
        for label, _solver, frames in runs:
            leg_label = f"{display_name} · {label}" if multi_model else label
            legendgroup = f"{model_key}::{label}" if multi_model else label
            color = _solver_line(leg_label, slot)
            py = {
                nm: np.array(
                    [fr.params_natural.get(nm, np.nan) for fr in frames],
                    dtype=np.float64,
                )
                for nm in names
            }
            z_init = frames[-1].iv_grid * 100.0  # open on this run's converged surface
            _add_model_surface(
                fig,
                M,
                T_days,
                z_init,
                solver_name=leg_label,
                slot=slot,
                opacity=overlay_opacity,
                legendgroup=legendgroup,
                x_hover_label=x_hover_label,
                x_hover_fmt=x_hover_fmt,
                row=1,
                col=1,
            )
            surf_i = trace_idx
            trace_idx += 1
            fig.add_trace(
                _model_points_scatter(
                    M,
                    T_days,
                    z_init,
                    color=color,
                    solver_name=leg_label,
                    x_hover_label=x_hover_label,
                    x_hover_fmt=x_hover_fmt,
                    legendgroup=legendgroup,
                    mask=valid_mask,
                ),
                row=1,
                col=1,
            )
            scat_i = trace_idx
            trace_idx += 1
            param_i: list[int] = []
            for p_i, nm in enumerate(names):
                # Static full-trajectory ghost (thin): always shows the whole
                # parameter path behind the animated snapshot line. NOT registered
                # in param_i, so the frames never slice it — a more pedagogical
                # replay (you see where the parameter is heading, not just where it
                # has been).
                fig.add_trace(
                    go.Scatter(
                        x=fx[: len(frames)],
                        y=py[nm],
                        mode="lines",
                        line=dict(color=color, width=1.0),
                        opacity=ghost_opacity,
                        name=leg_label,
                        legendgroup=legendgroup,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=p_i + 1,
                    col=col,
                )
                trace_idx += 1
                fig.add_trace(
                    go.Scatter(
                        x=fx[: len(frames)],
                        y=py[nm],
                        mode="lines+markers",
                        line=dict(color=color, width=2.2),
                        marker=dict(size=3, color=color),
                        name=leg_label,
                        legendgroup=legendgroup,
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{leg_label}</b><br>{disp[nm]} %{{y:.4g}}<br>"
                            "snapshot %{x}<extra></extra>"
                        ),
                    ),
                    row=p_i + 1,
                    col=col,
                )
                param_i.append(trace_idx)
                trace_idx += 1
            run_records.append(
                {
                    "m": m,
                    "col": col,
                    "names": names,
                    "frames": frames,
                    "py": py,
                    "surf_i": surf_i,
                    "scat_i": scat_i,
                    "param_i": param_i,
                }
            )
            slot += 1

    # ── ONE frame set advances EVERY run's surface + scatter + params in step.
    #    Shorter runs freeze at their last frame. ──
    lead = run_records[0]["frames"]
    plotly_frames = []
    slider_steps = []
    for k in range(max_len):
        data = []
        traces = []
        for rec in run_records:
            frames = rec["frames"]
            kk = min(k, len(frames) - 1)
            z = frames[kk].iv_grid * 100.0
            data.append(go.Surface(x=M, y=T_days, z=z))
            traces.append(rec["surf_i"])
            data.append(go.Scatter3d(x=M_valid, y=T_valid, z=z.ravel()[valid_mask]))
            traces.append(rec["scat_i"])
            for p_i, nm in enumerate(rec["names"]):
                data.append(go.Scatter(x=fx[: kk + 1], y=rec["py"][nm][: kk + 1]))
                traces.append(rec["param_i"][p_i])
        plotly_frames.append(
            go.Frame(
                name=str(k),
                data=data,
                traces=traces,
                # Per-frame scene uirevision preserves the user's camera through
                # the redraw=True the 3D mesh requires (see _scene docstring).
                layout=dict(scene=dict(uirevision="iv-surface-overlay")),
            )
        )
        lead_k = min(k, len(lead) - 1)
        slider_steps.append(
            dict(
                method="animate",
                args=[
                    [str(k)],
                    dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0),
                    ),
                ],
                label=f"#{lead[lead_k].iter_index}",
            )
        )
    fig.frames = plotly_frames

    # Parameter axes: each model's parameters live in its own column. Shared
    # snapshot index on x (labels only on each column's bottom panel); the panel
    # title carries the parameter name; a red dashed line marks the true value when
    # that model is the synthetic generator. The y-range spans that model's runs.
    for m, (_mk, _dn, _runs, tp) in enumerate(groups):
        col = m + 2
        names = group_names[m]
        n_p = len(names)
        for p_i, nm in enumerate(names):
            row = p_i + 1
            vals: list[float] = []
            for rec in run_records:
                if rec["m"] == m:
                    py = rec["py"][nm]
                    vals.extend(py[np.isfinite(py)].tolist())
            if tp and nm in tp:
                vals.append(float(tp[nm]))
            v_min = min(vals) if vals else 0.0
            v_max = max(vals) if vals else 1.0
            pad = max(abs(v_max - v_min) * 0.1, 1e-6)
            if tp and nm in tp:
                # A horizontal Scatter, NOT add_hline: add_hline's axis-spanning
                # logic chokes on this mixed scene+xy figure. Added after the frames
                # so it stays a static reference line (never animated).
                fig.add_trace(
                    go.Scatter(
                        x=[-0.5, float(max_len - 1) + 0.5],
                        y=[float(tp[nm])] * 2,
                        mode="lines",
                        line=dict(color=COLORS["danger"], width=1.4, dash="dash"),
                        name="_truth",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )
            is_bottom = p_i == n_p - 1
            fig.update_xaxes(
                range=[-0.5, float(max_len - 1) + 0.5],
                title="snapshot" if is_bottom else None,
                showticklabels=is_bottom,
                row=row,
                col=col,
            )
            fig.update_yaxes(range=[v_min - pad, v_max + pad], row=row, col=col)

    height = min(170 * max_rows + 200, 1500)
    total_runs = len(run_records)
    if multi_model:
        head_id = f"{n_models} models"
    else:
        head_id = groups[0][2][0][1] if total_runs == 1 else f"{total_runs} variations"
    apply_lab_theme(
        fig,
        height=height,
        title=(
            f"Surface fit progression + parameter roles  ·  {head_id}  ·  "
            f"{max_len} frames"
        ),
        margin=(0, 0, 96, 70),
    )
    z_range = _z_range(
        z_market, *[rec["frames"][-1].iv_grid * 100.0 for rec in run_records]
    )
    scene_cfg = _scene(
        x_axis_title=x_axis_title,
        x_tickformat=x_tickformat,
        z_range=z_range,
        uirevision="iv-surface-overlay",
    )
    # A 3/4 diagonal default view shows the smile (K) and term-structure (T)
    # relief far better than the previous flat head-on view; the per-frame
    # ``scene.uirevision`` still preserves whatever rotation the user sets.
    scene_cfg["camera"] = dict(
        eye=dict(x=1.6, y=1.5, z=0.9),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.05),
    )
    scene_cfg["aspectmode"] = "manual"
    scene_cfg["aspectratio"] = dict(x=1.7, y=1.2, z=0.9)
    fig.update_layout(
        scene=scene_cfg,
        uirevision="iv-animation",
        # Native, clickable legend (top-left of the scene): one click toggles a
        # variation's whole layer — its 3D surface, fit scatter and parameter lines
        # share the legendgroup, so they show/hide together; "market" toggles the
        # reference layer. Double-click isolates a single variation.
        showlegend=True,
        legend=dict(
            orientation="v",
            x=0.0,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor=COLORS["axis"],
            borderwidth=1,
            font=dict(family=FONT_FAMILY, size=10),
            itemsizing="constant",
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=-0.03,
                xanchor="left",
                yanchor="top",
                pad=dict(t=4, r=10),
                showactive=False,
                bgcolor=COLORS["primary"],
                bordercolor=COLORS["primary_dim"],
                borderwidth=1,
                font=dict(family=FONT_FAMILY, color="#ffffff", size=12),
                buttons=[
                    dict(
                        label="▶  Replay",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(
                                    duration=_vis("calib_vis_frame_duration_ms", 400),
                                    redraw=True,
                                ),
                                fromcurrent=False,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="❚❚  Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=len(slider_steps) - 1,
                x=0.5,
                y=-0.06,
                len=0.78,
                xanchor="center",
                yanchor="top",
                currentvalue=dict(
                    prefix="snapshot: ",
                    font=dict(family=FONT_FAMILY, color=COLORS["text"], size=12),
                ),
                steps=slider_steps,
                font=dict(family=FONT_FAMILY, color=COLORS["axis"], size=10),
                bgcolor="rgba(0,0,0,0.06)",
                activebgcolor=COLORS["primary"],
                bordercolor=COLORS["grid_strong"],
                tickcolor=COLORS["axis"],
            )
        ],
    )
    # Keep the sub-plot-title annotations on-theme (the native legend above
    # replaces the old custom market/model corner annotations).
    for ann in fig.layout.annotations:
        ann.font = dict(family=FONT_FAMILY, color=COLORS["text"], size=12)
    return fig


def render_returns_summary(
    prices: np.ndarray, log_returns: np.ndarray, ann: int
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Synthetic price path", "Return distribution"),
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.10,
    )
    t = np.arange(prices.size)
    fig.add_trace(
        go.Scatter(
            x=t,
            y=prices,
            mode="lines",
            line=dict(color=COLORS["primary"], width=1.4),
            hovertemplate="t=%{x}<br>S=%{y:.2f}<extra></extra>",
            name="price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=log_returns,
            nbinsx=60,
            marker=dict(
                color=COLORS["info"],
                line=dict(color="#0a1018", width=0.5),
            ),
            opacity=0.85,
            name="returns",
        ),
        row=1,
        col=2,
    )
    sigma_ann = float(np.std(log_returns) * np.sqrt(ann)) * 100
    mu_ann = float(np.mean(log_returns) * ann) * 100
    apply_lab_theme(
        fig,
        height=400,
        title=f"σ_ann = {sigma_ann:.2f}%   |   μ_ann = {mu_ann:.2f}%   |   N = {len(log_returns)}",
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Time step t (index)", row=1, col=1)
    fig.update_yaxes(title="Underlying price Sₜ (level)", row=1, col=1)
    fig.update_xaxes(title="Daily log-return rₜ = ln(Sₜ / Sₜ₋₁)", row=1, col=2)
    fig.update_yaxes(title="Frequency (observation count)", row=1, col=2)
    # Match the subplot-title styling used in render_parameter_trajectories so
    # the multi-subplot annotations are not rendered with Plotly's default grey.
    for ann in fig.layout.annotations:
        ann.font = dict(family=FONT_FAMILY, color=COLORS["text"], size=12)
    return fig
