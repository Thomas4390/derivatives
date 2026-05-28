"""Centralized dark-theme constants for Plotly charts.

Single source of truth for background colors, grid/axis styling,
annotation badges, and standard chart heights. Shared between the
simulation and calibration Streamlit apps.
"""

# ── Background ────────────────────────────────────────────────────────
PAPER_BG = "#0e1117"
PLOT_BG = "#161b22"

# ── Grid & axes ───────────────────────────────────────────────────────
GRID = "rgba(255,255,255,0.10)"
AXIS_LINE = "rgba(255,255,255,0.25)"
AXIS_LABEL = "#ffffff"
TICK_COLOR = "rgba(255,255,255,0.70)"
LEGEND_COLOR = "#ffffff"

AXIS_STYLE = dict(
    gridcolor=GRID,
    zerolinecolor=GRID,
    showline=True,
    linecolor=AXIS_LINE,
    linewidth=1,
    tickfont=dict(size=10, color=TICK_COLOR),
    title_font=dict(size=12, color=AXIS_LABEL),
)

# ── Annotation badge ─────────────────────────────────────────────────
ANN_BG = "rgba(14, 17, 23, 0.85)"
ANN_FONT = 12


def badge(color: str, font_size: int = ANN_FONT) -> dict:
    """Return annotation_* kwargs for a readable badge with dark background."""
    return dict(
        annotation_font_size=font_size,
        annotation_font_color=color,
        annotation_bgcolor=ANN_BG,
        annotation_bordercolor=color,
        annotation_borderwidth=1,
        annotation_borderpad=3,
    )


# ── Standard chart heights ────────────────────────────────────────────
CHART_HEIGHT_SM = 450
CHART_HEIGHT_MD = 600
CHART_HEIGHT_LG = 650
