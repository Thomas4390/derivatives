"""
Chart Helper Functions for Monte Carlo Simulation Explorer.

Provides consistent styling and common patterns for Plotly charts
used across the application.
"""

# Import constants from config
import re
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.constants import (
    CHART_HEIGHT_STANDARD,
)

# =============================================================================
# LAYOUT HELPERS
# =============================================================================


def apply_default_layout(
    fig: go.Figure,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    height: int = CHART_HEIGHT_STANDARD,
    show_legend: bool = True,
    hovermode: str = "x unified",
) -> go.Figure:
    """
    Apply consistent default layout to a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to update
    title : str
        Chart title
    xaxis_title : str
        X-axis label
    yaxis_title : str
        Y-axis label
    height : int
        Chart height in pixels
    show_legend : bool
        Whether to show legend
    hovermode : str
        Hover interaction mode

    Returns
    -------
    go.Figure
        Updated figure
    """
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        hovermode=hovermode,
        showlegend=show_legend,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def create_base_figure(
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = CHART_HEIGHT_STANDARD,
) -> go.Figure:
    """
    Create a new figure with default styling applied.

    Parameters
    ----------
    title : str
        Chart title
    xaxis_title : str
        X-axis label
    yaxis_title : str
        Y-axis label
    height : int
        Chart height in pixels

    Returns
    -------
    go.Figure
        New figure with default layout
    """
    fig = go.Figure()
    return apply_default_layout(fig, title, xaxis_title, yaxis_title, height)


# =============================================================================
# TRACE HELPERS
# =============================================================================


def add_percentile_band(
    fig: go.Figure,
    time_grid: np.ndarray,
    p_lower: np.ndarray,
    p_upper: np.ndarray,
    name: str = "5-95% Range",
    color: str = "rgba(13, 148, 136, 0.2)",
    show_legend: bool = True,
) -> go.Figure:
    """
    Add a filled percentile band to a figure.

    Parameters
    ----------
    fig : go.Figure
        Figure to add band to
    time_grid : np.ndarray
        X-axis values (time points)
    p_lower : np.ndarray
        Lower percentile values
    p_upper : np.ndarray
        Upper percentile values
    name : str
        Legend name for the band
    color : str
        Fill color (rgba format recommended)
    show_legend : bool
        Whether to show in legend

    Returns
    -------
    go.Figure
        Updated figure
    """
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([time_grid, time_grid[::-1]]),
            y=np.concatenate([p_upper, p_lower[::-1]]),
            fill="toself",
            fillcolor=color,
            line=dict(color="rgba(0,0,0,0)"),
            name=name,
            hoverinfo="skip",
            showlegend=show_legend,
        )
    )
    return fig


def add_confidence_band(
    fig: go.Figure,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    n_std: float = 1.96,
    name: str = "95% CI",
    color: str = "rgba(59, 130, 246, 0.2)",
) -> go.Figure:
    """
    Add a confidence band around a mean line.

    Parameters
    ----------
    fig : go.Figure
        Figure to add band to
    x : np.ndarray
        X-axis values
    y_mean : np.ndarray
        Mean values
    y_std : np.ndarray
        Standard deviation values
    n_std : float
        Number of standard deviations for band (1.96 for 95% CI)
    name : str
        Legend name
    color : str
        Fill color

    Returns
    -------
    go.Figure
        Updated figure
    """
    y_lower = y_mean - n_std * y_std
    y_upper = y_mean + n_std * y_std
    return add_percentile_band(fig, x, y_lower, y_upper, name, color)


def add_horizontal_line(
    fig: go.Figure,
    y: float,
    line_dash: str = "dot",
    line_color: str = "#64748b",
    annotation_text: str | None = None,
    annotation_position: str = "top right",
) -> go.Figure:
    """
    Add a horizontal reference line to a figure.

    Parameters
    ----------
    fig : go.Figure
        Figure to add line to
    y : float
        Y-value for horizontal line
    line_dash : str
        Line style ('solid', 'dot', 'dash', 'dashdot')
    line_color : str
        Line color
    annotation_text : Optional[str]
        Text annotation for the line
    annotation_position : str
        Position of annotation

    Returns
    -------
    go.Figure
        Updated figure
    """
    fig.add_hline(
        y=y,
        line_dash=line_dash,
        line_color=line_color,
        annotation_text=annotation_text,
        annotation_position=annotation_position,
    )
    return fig


def add_vertical_line(
    fig: go.Figure,
    x: float,
    line_dash: str = "dash",
    line_color: str = "#64748b",
    annotation_text: str | None = None,
    annotation_position: str = "top left",
) -> go.Figure:
    """
    Add a vertical reference line to a figure.

    Parameters
    ----------
    fig : go.Figure
        Figure to add line to
    x : float
        X-value for vertical line
    line_dash : str
        Line style ('solid', 'dot', 'dash', 'dashdot')
    line_color : str
        Line color
    annotation_text : Optional[str]
        Text annotation for the line
    annotation_position : str
        Position of annotation

    Returns
    -------
    go.Figure
        Updated figure
    """
    fig.add_vline(
        x=x,
        line_dash=line_dash,
        line_color=line_color,
        annotation_text=annotation_text,
        annotation_position=annotation_position,
    )
    return fig


# =============================================================================
# SAMPLE PATHS HELPER
# =============================================================================


def add_sample_paths(
    fig: go.Figure,
    time_grid: np.ndarray,
    paths: np.ndarray,
    max_paths: int = 50,
    color: str = "rgba(59, 130, 246, 0.3)",
    line_width: float = 0.8,
) -> go.Figure:
    """
    Add multiple sample paths to a figure.

    Parameters
    ----------
    fig : go.Figure
        Figure to add paths to
    time_grid : np.ndarray
        X-axis values (time points)
    paths : np.ndarray
        2D array of shape (n_paths, n_steps)
    max_paths : int
        Maximum number of paths to display
    color : str
        Line color for paths
    line_width : float
        Width of path lines

    Returns
    -------
    go.Figure
        Updated figure
    """
    n_display = min(max_paths, paths.shape[0])

    for i in range(n_display):
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=paths[i],
                mode="lines",
                line=dict(color=color, width=line_width),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    return fig


def add_mean_path(
    fig: go.Figure,
    time_grid: np.ndarray,
    mean_path: np.ndarray,
    name: str = "Mean",
    color: str = "#1e3a5f",
    line_width: float = 2.5,
) -> go.Figure:
    """
    Add a mean path line to a figure.

    Parameters
    ----------
    fig : go.Figure
        Figure to add path to
    time_grid : np.ndarray
        X-axis values
    mean_path : np.ndarray
        Mean values
    name : str
        Legend name
    color : str
        Line color
    line_width : float
        Line width

    Returns
    -------
    go.Figure
        Updated figure
    """
    fig.add_trace(
        go.Scatter(
            x=time_grid,
            y=mean_path,
            mode="lines",
            name=name,
            line=dict(color=color, width=line_width),
        )
    )
    return fig


# =============================================================================
# FORMATTING HELPERS
# =============================================================================


def format_currency(value: float, decimals: int = 2, symbol: str = "$") -> str:
    """
    Format a number as currency.

    Parameters
    ----------
    value : float
        Value to format
    decimals : int
        Number of decimal places
    symbol : str
        Currency symbol

    Returns
    -------
    str
        Formatted currency string
    """
    return f"{symbol}{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1, multiply: bool = True) -> str:
    """
    Format a number as percentage.

    Parameters
    ----------
    value : float
        Value to format (0.5 or 50 depending on multiply flag)
    decimals : int
        Number of decimal places
    multiply : bool
        If True, multiply by 100 before formatting

    Returns
    -------
    str
        Formatted percentage string
    """
    if multiply:
        value = value * 100
    return f"{value:.{decimals}f}%"


def format_large_number(value: float, decimals: int = 1) -> str:
    """
    Format large numbers with K/M/B suffixes.

    Parameters
    ----------
    value : float
        Value to format
    decimals : int
        Number of decimal places

    Returns
    -------
    str
        Formatted string with suffix
    """
    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    if abs_value >= 1e9:
        return f"{sign}{abs_value / 1e9:.{decimals}f}B"
    if abs_value >= 1e6:
        return f"{sign}{abs_value / 1e6:.{decimals}f}M"
    if abs_value >= 1e3:
        return f"{sign}{abs_value / 1e3:.{decimals}f}K"
    return f"{sign}{abs_value:.{decimals}f}"


# =============================================================================
# COLOR UTILITIES
# =============================================================================


def get_profit_loss_color(
    value: float, profit_color: str = "#22c55e", loss_color: str = "#ef4444"
) -> str:
    """
    Return color based on profit/loss.

    Parameters
    ----------
    value : float
        P&L value
    profit_color : str
        Color for positive values
    loss_color : str
        Color for negative values

    Returns
    -------
    str
        Color string
    """
    return profit_color if value > 0 else loss_color


def create_color_scale(values: np.ndarray, colorscale: str = "RdYlGn") -> list[str]:
    """
    Create a list of colors based on values using a colorscale.

    Parameters
    ----------
    values : np.ndarray
        Values to map to colors
    colorscale : str
        Plotly colorscale name

    Returns
    -------
    List[str]
        List of color strings
    """
    # Normalize values to [0, 1]
    min_val = values.min()
    max_val = values.max()
    if max_val == min_val:
        normalized = np.zeros_like(values) + 0.5
    else:
        normalized = (values - min_val) / (max_val - min_val)

    # This is a simplified version - for full colorscale support,
    # use plotly.colors module
    colors = []
    for v in normalized:
        if v < 0.5:
            # Red to yellow
            r = 239
            g = int(68 + (250 - 68) * (v * 2))
            b = 68
        else:
            # Yellow to green
            r = int(250 - (250 - 34) * ((v - 0.5) * 2))
            g = int(250 - (250 - 197) * ((v - 0.5) * 2))
            b = int(34 + (94 - 34) * ((v - 0.5) * 2))
        colors.append(f"rgb({r},{g},{b})")

    return colors


# =============================================================================
# ANNOTATION ANTI-OVERLAP
# =============================================================================

_AVG_CHAR_WIDTH_FACTOR = 0.6
_DEFAULT_FONT_SIZE = 12


def _estimate_ann_size(ann):
    """Estimate (width_px, height_px) of an annotation label."""
    text = getattr(ann, "text", "") or ""
    stripped = re.sub(r"<[^>]+>", "", text)

    font = getattr(ann, "font", None)
    font_size = (getattr(font, "size", None) if font else None) or _DEFAULT_FONT_SIZE

    borderpad = getattr(ann, "borderpad", None) or 0
    borderwidth = getattr(ann, "borderwidth", None) or 0
    pad = borderpad + borderwidth

    width = len(stripped) * _AVG_CHAR_WIDTH_FACTOR * font_size + 2 * pad
    height = font_size * 1.3 + 2 * pad
    return width, height


def _get_axis_range(fig, axis_ref):
    """Get (min, max) data range for an axis like 'y', 'y2', 'x', 'x2'."""
    base = axis_ref[0] + "axis"
    suffix = axis_ref[1:] if len(axis_ref) > 1 else ""
    prop = base + suffix

    axis_obj = getattr(fig.layout, prop, None)
    if axis_obj is not None:
        r = axis_obj.range
        if r is not None and len(r) == 2:
            try:
                return float(r[0]), float(r[1])
            except (TypeError, ValueError):
                pass

    # Fallback: scan trace data
    is_y = axis_ref.startswith("y")
    vals = []
    for trace in fig.data:
        trace_ax = getattr(trace, "yaxis" if is_y else "xaxis", None)
        if trace_ax is None:
            trace_ax = axis_ref[0]
        if trace_ax != axis_ref:
            continue
        data = getattr(trace, "y" if is_y else "x", None)
        if data is not None:
            arr = np.asarray(data, dtype=float)
            finite = arr[np.isfinite(arr)]
            if len(finite) > 0:
                vals.extend([float(np.min(finite)), float(np.max(finite))])

    if vals:
        return min(vals), max(vals)
    return None


def _get_axis_domain(fig, axis_ref):
    """Get domain (d0, d1) fraction for an axis."""
    base = axis_ref[0] + "axis"
    suffix = axis_ref[1:] if len(axis_ref) > 1 else ""
    prop = base + suffix

    axis_obj = getattr(fig.layout, prop, None)
    if axis_obj is not None:
        d = axis_obj.domain
        if d is not None and len(d) == 2:
            return float(d[0]), float(d[1])
    return 0.0, 1.0


def _label_bbox_offset(anchor, size):
    """Return (low_offset, high_offset) from the data-mapped pixel position.

    'bottom'/'left' → label extends positively (upward/rightward).
    'top'/'right'   → label extends negatively (downward/leftward).
    """
    if anchor in ("top", "right"):
        return -size, 0.0
    elif anchor in ("middle", "center"):
        return -size / 2.0, size / 2.0
    else:  # 'bottom', 'left', 'auto', None
        return 0.0, size


def _spread_group(
    anns,
    indices,
    data_range,
    panel_px,
    pos_attr,
    shift_attr,
    anchor_attr,
    size_idx,
    min_gap_px,
):
    """Resolve overlaps within one group of annotations (greedy + cascade)."""
    data_span = data_range[1] - data_range[0]
    if data_span <= 0 or panel_px <= 0:
        return

    px_per_unit = panel_px / data_span

    sorted_idx = sorted(
        indices,
        key=lambda i: float(getattr(anns[i], pos_attr, 0) or 0),
    )

    # Two passes: second handles cascading shifts
    for _ in range(2):
        for j in range(1, len(sorted_idx)):
            prev = anns[sorted_idx[j - 1]]
            curr = anns[sorted_idx[j]]

            prev_val = float(getattr(prev, pos_attr, 0) or 0)
            curr_val = float(getattr(curr, pos_attr, 0) or 0)

            prev_shift = float(getattr(prev, shift_attr, 0) or 0)
            curr_shift = float(getattr(curr, shift_attr, 0) or 0)

            eff_prev = (prev_val - data_range[0]) * px_per_unit + prev_shift
            eff_curr = (curr_val - data_range[0]) * px_per_unit + curr_shift

            prev_size = _estimate_ann_size(prev)[size_idx]
            curr_size = _estimate_ann_size(curr)[size_idx]

            prev_anchor = getattr(prev, anchor_attr, "auto") or "auto"
            curr_anchor = getattr(curr, anchor_attr, "auto") or "auto"

            _, prev_top = _label_bbox_offset(prev_anchor, prev_size)
            curr_bot, _ = _label_bbox_offset(curr_anchor, curr_size)

            overlap = (eff_prev + prev_top) + min_gap_px - (eff_curr + curr_bot)
            if overlap > 0:
                setattr(curr, shift_attr, curr_shift + overlap)


def spread_annotations(fig, min_gap_px=4):
    """Post-process fig.layout.annotations to resolve overlapping labels.

    Identifies annotations created by add_hline / add_vline (those with
    'domain' in one axis ref), groups them by edge and subplot panel,
    then applies pixel shifts to eliminate overlaps.
    """
    anns = fig.layout.annotations
    if len(anns) < 2:
        return fig

    chart_height = fig.layout.height or 600
    chart_width = fig.layout.width or 900
    margins = fig.layout.margin
    margin_t = margins.t if margins and margins.t is not None else 60
    margin_b = margins.b if margins and margins.b is not None else 60
    margin_l = margins.l if margins and margins.l is not None else 60
    margin_r = margins.r if margins and margins.r is not None else 40

    plot_height = chart_height - margin_t - margin_b
    plot_width = chart_width - margin_l - margin_r

    # Classify annotations into groups: (line_type, axis_ref, side) → [indices]
    groups = {}

    for i, ann in enumerate(anns):
        xref = str(getattr(ann, "xref", "") or "")
        yref = str(getattr(ann, "yref", "") or "")

        if "domain" in xref and "domain" not in yref and yref:
            # hline annotation: x is domain-based, y is data-based
            side = "right" if (getattr(ann, "x", 0) or 0) > 0.5 else "left"
            key = ("hline", yref, side)
            groups.setdefault(key, []).append(i)
        elif "domain" in yref and "domain" not in xref and xref:
            # vline annotation: y is domain-based, x is data-based
            side = "top" if (getattr(ann, "y", 0) or 0) > 0.5 else "bottom"
            key = ("vline", xref, side)
            groups.setdefault(key, []).append(i)

    for key, indices in groups.items():
        if len(indices) < 2:
            continue

        line_type, axis_ref, _ = key
        data_range = _get_axis_range(fig, axis_ref)
        domain = _get_axis_domain(fig, axis_ref)
        if data_range is None:
            continue

        if line_type == "hline":
            panel_px = plot_height * (domain[1] - domain[0])
            _spread_group(
                anns,
                indices,
                data_range,
                panel_px,
                "y",
                "yshift",
                "yanchor",
                1,
                min_gap_px,
            )
        else:
            panel_px = plot_width * (domain[1] - domain[0])
            _spread_group(
                anns,
                indices,
                data_range,
                panel_px,
                "x",
                "xshift",
                "xanchor",
                0,
                min_gap_px,
            )

    return fig
