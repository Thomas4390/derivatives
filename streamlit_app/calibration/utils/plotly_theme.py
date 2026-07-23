"""
Plotly theme — white background ("plotly_white" classic)
=========================================================

Light theme for the calibration explorer. Uses Plotly's stock
``plotly_white`` template as the canvas, then overrides typography
and the palette so it matches the rest of the app's identity.

Public surface (kept stable for downstream charts):
    - ``apply_lab_theme(fig, ...)``
    - ``empty_state_figure(message, height=...)``
    - ``COLORS`` dict, ``PALETTE``, ``MODEL_COLOR_MAP``,
      ``SOLVER_COLOR_MAP``, ``MARKET_COLOR``, ``model_color()``
    - ``series_style()``, ``_SOLVER_SYMBOL``, ``_SOLVER_DASH`` (overlay
      legend encoding shared by Compare + per-tab overlays)
    - ``FONT_FAMILY``, ``MONO_FAMILY``
"""

from __future__ import annotations

import zlib

import numpy as np
import plotly.graph_objects as go


def adaptive_loss_axis(
    values,
    *,
    log_title: str,
    linear_title: str,
    empty_title: str | None = None,
) -> dict:
    """Plotly ``update_yaxes`` kwargs picking log vs linear from the data sign.

    Log scale suits surface calibrations (½‖r‖² spans 8+ decades), but Plotly
    silently drops non-positive samples from a log axis — and GARCH calibrators
    minimise a *negative* log-likelihood, so a naive log axis renders empty.
    Any non-positive (or all-NaN) sample forces a linear axis; otherwise a log
    axis bracketed to the actual data.

    Single source of truth for both the Live convergence chart and the Compare
    Pareto/loss overlays, which previously kept two hand-synced copies that had
    already drifted (a 1e-30 vs 1e-15 log floor made a converged ~1e-16 loss
    show on Live but clip below the axis on Compare). The 1e-30 floor and the
    adaptive decade padding below are the correct behaviour.
    """
    arr = np.asarray(
        [v for v in np.asarray(values, dtype=np.float64).ravel() if np.isfinite(v)],
        dtype=np.float64,
    )
    if arr.size == 0:
        return dict(type="linear", title=empty_title or linear_title, tickformat=".3g")
    if (arr <= 0.0).any():
        ymin = float(arr.min())
        ymax = float(arr.max())
        span = ymax - ymin
        # Span-scaled padding with |ymax|/|ymin| floors so identical-value or
        # near-zero runs still get a visible window.
        pad = max(span * 0.05, abs(ymax) * 0.01, abs(ymin) * 0.01, 0.1)
        return dict(
            type="linear",
            title=linear_title,
            tickformat=".3g",
            range=[ymin - pad, ymax + pad],
        )
    # Strictly positive here: bracket the ACTUAL data min/max in log10. The
    # 1e-30 guard only caps a pathological denormal without clipping a real
    # converged loss (perfect synthetic fits reach ~1e-16); pad both ends by
    # 5 % of the span with a 0.15-decade floor so the tail marker isn't clipped.
    lo = float(np.log10(max(float(arr.min()), 1e-30)))
    hi = float(np.log10(max(float(arr.max()), 1e-30)))
    pad = max(0.05 * (hi - lo), 0.15)
    return dict(
        type="log",
        title=log_title,
        tickformat=".0e",
        range=[lo - pad, hi + pad],
    )


# --------------------------------------------------------------------------- #
# Canvas constants (classic Plotly white)
# --------------------------------------------------------------------------- #

PAPER_BG = "white"
PLOT_BG = "white"

GRID = "rgba(0,0,0,0.08)"
AXIS_LINE = "rgba(0,0,0,0.30)"
AXIS_LABEL = "#1f2937"  # dark slate for titles
TICK_COLOR = "#374151"  # slightly lighter for ticks
LEGEND_COLOR = "#1f2937"

AXIS_STYLE = dict(
    gridcolor=GRID,
    zerolinecolor=GRID,
    showline=True,
    linecolor=AXIS_LINE,
    linewidth=1,
    tickfont=dict(size=10, color=TICK_COLOR),
    title_font=dict(size=12, color=AXIS_LABEL),
)


# --------------------------------------------------------------------------- #
# Palette (unchanged — colours are visible on both light and dark backgrounds)
# --------------------------------------------------------------------------- #

COLORS = {
    "bg": PAPER_BG,
    "plot": PLOT_BG,
    "grid": GRID,
    "grid_strong": "rgba(0,0,0,0.18)",
    "axis": AXIS_LABEL,
    "text": AXIS_LABEL,
    "text_dim": "rgba(0,0,0,0.70)",
    "text_muted": "rgba(0,0,0,0.50)",
    "primary": "#0d9488",  # teal — primary lines / best
    "primary_dim": "#0f766e",
    "accent": "#d97706",  # amber — evaluations / iterations
    "info": "#0284c7",  # cyan-ish — informational
    "secondary": "#1a365d",  # deep navy
    "danger": "#dc2626",
    "purple": "#7c3aed",
    "pink": "#db2777",
}

SOLVER_COLOR_MAP = {
    "LM-JAX": "#0d9488",  # teal
    "DE": "#d97706",  # amber
    "NM": "#7c3aed",  # purple
    "L-BFGS-B": "#0284c7",  # blue
}

MODEL_COLOR_MAP = {
    "heston": "#0d9488",  # teal — canonical SV
    "merton": "#d97706",  # amber — jump-diffusion
    "bates": "#7c3aed",  # purple — SV+jumps combo
    "garch": "#0284c7",  # blue — variance-recursion baseline
    "ngarch": "#059669",  # green — asymmetric GARCH variant
    "gjr_garch": "#db2777",  # pink — indicator-based variant
    "iv_gbm": "#475569",  # slate — flat-vol baseline
}

MARKET_COLOR = "#dc2626"  # red

# Per-run overlay palette. Two hues are DELIBERATELY excluded so a run can
# never be mistaken for a reserved annotation:
#   * amber ``#d97706`` == ``COLORS["accent"]`` — the translucent "evaluation"
#     markers on the loss row.
#   * red   ``#dc2626`` == ``COLORS["danger"]`` == ``MARKET_COLOR`` — the dashed
#     "true value" reference line (and market data on Compare).
# A 2nd fit used to land on amber (≈ the red truth line) and a 5th on pure red
# (== it), so overlaid fits blurred into the reference. Order is tuned for
# maximum separation across the common 2-4 run case (teal → purple → blue →
# pink). Do NOT re-add amber/red here — keep them reserved for their roles.
PALETTE = (
    "#0d9488",  # teal   — primary / "best" identity
    "#7c3aed",  # purple — max contrast vs teal and vs the red truth line
    "#0284c7",  # blue
    "#db2777",  # pink / magenta
    "#059669",  # green
    "#475569",  # slate
    "#9333ea",  # violet
    "#0891b2",  # cyan
)


def model_color(model_key: str) -> str:
    """Return the canonical colour for ``model_key`` (falls back to a
    deterministic PALETTE rotation for unknown keys so adding a model can't
    crash a chart).

    The fallback uses a stable ``zlib.crc32`` digest, not the builtin
    ``hash()`` — string ``hash`` is ``PYTHONHASHSEED``-salted per process, so
    the custom model (and any key absent from ``MODEL_COLOR_MAP``) used to pick
    a different chart colour on every ``streamlit run``."""
    if model_key in MODEL_COLOR_MAP:
        return MODEL_COLOR_MAP[model_key]
    digest = zlib.crc32(model_key.encode("utf-8"))
    return PALETTE[digest % len(PALETTE)]


# --------------------------------------------------------------------------- #
# Per-solver shape / dash encoding (shared by every overlay chart)
# --------------------------------------------------------------------------- #
# One marker shape and one line dash per solver so a reader decodes both
# axes of an overlaid chart at a glance: colour = model (or per-series),
# shape / dash = solver. Lives here — not in a chart module — so the
# Compare-tab charts and the per-tab overlays share a single source.

_SOLVER_SYMBOL = {
    "LM-JAX": "circle",
    "DE": "diamond",
    "NM": "triangle-up",
    "L-BFGS-B": "square",
}
_SOLVER_DASH = {
    "LM-JAX": "solid",
    "DE": "dash",
    "NM": "dot",
    "L-BFGS-B": "dashdot",
}


def series_style(
    model_key: str,
    solver_name: str,
    *,
    multi_model: bool,
    index: int = 0,
) -> dict[str, str]:
    """Colour / dash / symbol for one overlaid ``(model, solver)`` series.

    Encoding rule, kept consistent across the whole app:

    * **colour** — keyed by *model* when the overlaid selection spans more
      than one model (so "Heston is teal" reads the same everywhere), and
      by a per-series ``PALETTE`` rotation (``index``) when a single model
      is compared across solvers/objectives (distinct hues are clearer).
    * **dash / symbol** — always keyed by *solver*, falling back to a solid
      line / circle marker for an unknown solver so a new backend cannot
      crash a chart.
    """
    color = model_color(model_key) if multi_model else PALETTE[index % len(PALETTE)]
    return {
        "color": color,
        "dash": _SOLVER_DASH.get(solver_name, "solid"),
        "symbol": _SOLVER_SYMBOL.get(solver_name, "circle"),
    }


FONT_FAMILY = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
MONO_FAMILY = "'JetBrains Mono', 'Menlo', monospace"


# --------------------------------------------------------------------------- #
# Theme application
# --------------------------------------------------------------------------- #


def apply_lab_theme(
    fig: go.Figure,
    *,
    height: int | None = None,
    title: str | None = None,
    margin: tuple[int, int, int, int] | None = None,
    legend_horizontal: bool = True,
) -> go.Figure:
    """Apply the classic Plotly-white theme to ``fig`` in place."""
    layout: dict = dict(
        template="plotly_white",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(
            family=FONT_FAMILY,
            color=AXIS_LABEL,
            size=11,
        ),
        hoverlabel=dict(
            font=dict(family=FONT_FAMILY, size=12, color=AXIS_LABEL),
            bgcolor="white",
            bordercolor=AXIS_LINE,
        ),
        hovermode="x unified",
    )
    if height is not None:
        layout["height"] = int(height)
    if title is not None:
        layout["title"] = dict(
            text=title,
            font=dict(family=FONT_FAMILY, size=14, color=AXIS_LABEL),
            x=0.0,
            xanchor="left",
            y=0.97,
            yanchor="top",
            pad=dict(b=18, t=6),
        )
    if margin is not None:
        layout["margin"] = dict(l=margin[0], r=margin[1], t=margin[2], b=margin[3])
    else:
        layout["margin"] = dict(l=50, r=20, t=64, b=40)
    if legend_horizontal:
        layout["legend"] = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(family=FONT_FAMILY, color=LEGEND_COLOR, size=10),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=AXIS_LINE,
            borderwidth=1,
        )

    fig.update_layout(**layout)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)

    if any(getattr(d, "type", None) in ("surface", "scatter3d") for d in fig.data):
        for axis in ("xaxis", "yaxis", "zaxis"):
            fig.update_layout(
                **{
                    f"scene.{axis}": dict(
                        backgroundcolor=PLOT_BG,
                        gridcolor=GRID,
                        showbackground=True,
                        color=AXIS_LABEL,
                        title=dict(
                            font=dict(family=FONT_FAMILY, color=AXIS_LABEL, size=11)
                        ),
                    )
                }
            )

    return fig


def empty_state_figure(message: str, *, height: int = 280) -> go.Figure:
    """Plain figure displaying ``message`` centred — used as a graceful
    fallback when a chart has no data. ``height`` is configurable so a
    tab with multiple charts in flow can avoid pinning every empty slot
    to 280 px and creating jarring whitespace.
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(family=FONT_FAMILY, color=TICK_COLOR, size=14),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    apply_lab_theme(fig, height=height)
    return fig
