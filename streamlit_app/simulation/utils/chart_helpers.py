"""
Chart Helper Functions for Monte Carlo Simulation Explorer.

Provides consistent styling and common patterns for Plotly charts
used across the application.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, List

# Import constants from config
import sys
from pathlib import Path
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
    hovermode: str = 'x unified'
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
        title=dict(
            text=title,
            font=dict(size=16)
        ),
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
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=60, r=40, t=60, b=60)
    )
    return fig


def create_base_figure(
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = CHART_HEIGHT_STANDARD
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
    name: str = '5-95% Range',
    color: str = 'rgba(13, 148, 136, 0.2)',
    show_legend: bool = True
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
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_grid, time_grid[::-1]]),
        y=np.concatenate([p_upper, p_lower[::-1]]),
        fill='toself',
        fillcolor=color,
        line=dict(color='rgba(0,0,0,0)'),
        name=name,
        hoverinfo='skip',
        showlegend=show_legend
    ))
    return fig


def add_confidence_band(
    fig: go.Figure,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    n_std: float = 1.96,
    name: str = '95% CI',
    color: str = 'rgba(59, 130, 246, 0.2)'
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
    annotation_text: Optional[str] = None,
    annotation_position: str = "top right"
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
        annotation_position=annotation_position
    )
    return fig


def add_vertical_line(
    fig: go.Figure,
    x: float,
    line_dash: str = "dash",
    line_color: str = "#64748b",
    annotation_text: Optional[str] = None,
    annotation_position: str = "top left"
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
        annotation_position=annotation_position
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
    color: str = 'rgba(59, 130, 246, 0.3)',
    line_width: float = 0.8
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
        fig.add_trace(go.Scatter(
            x=time_grid,
            y=paths[i],
            mode='lines',
            line=dict(color=color, width=line_width),
            hoverinfo='skip',
            showlegend=False
        ))

    return fig


def add_mean_path(
    fig: go.Figure,
    time_grid: np.ndarray,
    mean_path: np.ndarray,
    name: str = 'Mean',
    color: str = '#1e3a5f',
    line_width: float = 2.5
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
    fig.add_trace(go.Scatter(
        x=time_grid,
        y=mean_path,
        mode='lines',
        name=name,
        line=dict(color=color, width=line_width)
    ))
    return fig


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def format_currency(
    value: float,
    decimals: int = 2,
    symbol: str = "$"
) -> str:
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


def format_percentage(
    value: float,
    decimals: int = 1,
    multiply: bool = True
) -> str:
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
        return f"{sign}{abs_value/1e9:.{decimals}f}B"
    elif abs_value >= 1e6:
        return f"{sign}{abs_value/1e6:.{decimals}f}M"
    elif abs_value >= 1e3:
        return f"{sign}{abs_value/1e3:.{decimals}f}K"
    else:
        return f"{sign}{abs_value:.{decimals}f}"


# =============================================================================
# COLOR UTILITIES
# =============================================================================

def get_profit_loss_color(value: float, profit_color: str = '#22c55e', loss_color: str = '#ef4444') -> str:
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


def create_color_scale(
    values: np.ndarray,
    colorscale: str = 'RdYlGn'
) -> List[str]:
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
        colors.append(f'rgb({r},{g},{b})')

    return colors
