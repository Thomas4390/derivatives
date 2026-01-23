"""
Unified Paths Chart - Combined price and volatility visualization.

Provides:
- Price paths with percentiles and mean
- Volatility paths (or constant line) in synchronized subplot
- Interactive controls
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List

from backend.simulation.base import SimulationResult
from services.simulation_service import (
    get_model_characteristics,
    get_initial_volatility,
)


def render_unified_paths(
    result: SimulationResult,
    model_key: str,
    params: Dict[str, Any],
    n_sample_paths: int = 100,
    show_percentiles: bool = True,
    show_mean: bool = True,
    height: int = 700,
):
    """
    Render unified price and volatility paths chart.

    Args:
        result: SimulationResult from simulation
        model_key: Model identifier
        params: All parameters
        n_sample_paths: Number of sample paths to display
        show_percentiles: Whether to show percentile bands
        show_mean: Whether to show mean path
        height: Chart height in pixels
    """
    characteristics = get_model_characteristics(model_key)
    has_vol_paths = (
        characteristics["has_stochastic_vol"] and
        result.volatility_paths is not None
    )

    # Create subplots
    if has_vol_paths:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
            subplot_titles=("Price Paths", "Volatility Paths")
        )
    else:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
            subplot_titles=("Price Paths", "Volatility (Constant)")
        )

    # Time grid
    time_grid = result.time_grid
    n_paths = result.price_paths.shape[0]

    # Sample paths for display (to avoid overplotting)
    if n_paths > n_sample_paths:
        sample_idx = np.random.choice(n_paths, n_sample_paths, replace=False)
    else:
        sample_idx = np.arange(n_paths)

    # Colors
    price_color = "rgba(31, 119, 180, 0.3)"  # Blue with transparency
    vol_color = "rgba(255, 127, 14, 0.3)"    # Orange with transparency
    mean_color = "rgb(44, 160, 44)"          # Green
    percentile_color = "rgba(214, 39, 40, 0.2)"  # Red with transparency

    # === PRICE PATHS ===
    # Sample paths
    for i, idx in enumerate(sample_idx):
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=result.price_paths[idx],
                mode="lines",
                line=dict(width=0.5, color=price_color),
                name="Sample Path" if i == 0 else None,
                showlegend=(i == 0),
                legendgroup="price_samples",
                hoverinfo="skip"
            ),
            row=1, col=1
        )

    # Mean path
    if show_mean:
        mean_path = result.mean_path
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=mean_path,
                mode="lines",
                line=dict(width=2, color=mean_color),
                name="Mean",
                legendgroup="price_mean"
            ),
            row=1, col=1
        )

    # Percentile bands
    if show_percentiles:
        percentiles = [5, 25, 75, 95]
        pct_paths = result.percentile_paths(percentiles)

        # 5-95 band
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=pct_paths[3],  # 95th
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=pct_paths[0],  # 5th
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=percentile_color,
                name="5-95%",
                legendgroup="price_pct"
            ),
            row=1, col=1
        )

    # Initial price reference
    fig.add_hline(
        y=result.initial_price,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"S₀ = {result.initial_price:.2f}",
        row=1, col=1
    )

    # === VOLATILITY PATHS ===
    if has_vol_paths:
        vol_paths = result.volatility_paths

        # Sample paths
        for i, idx in enumerate(sample_idx):
            fig.add_trace(
                go.Scatter(
                    x=time_grid,
                    y=vol_paths[idx] * 100,  # Convert to percentage
                    mode="lines",
                    line=dict(width=0.5, color=vol_color),
                    name="Vol Path" if i == 0 else None,
                    showlegend=(i == 0),
                    legendgroup="vol_samples",
                    hoverinfo="skip"
                ),
                row=2, col=1
            )

        # Mean volatility
        if show_mean:
            mean_vol = result.mean_volatility_path * 100
            fig.add_trace(
                go.Scatter(
                    x=time_grid,
                    y=mean_vol,
                    mode="lines",
                    line=dict(width=2, color=mean_color),
                    name="Mean Vol",
                    legendgroup="vol_mean"
                ),
                row=2, col=1
            )

        # Percentile bands for volatility
        if show_percentiles:
            vol_pct = np.percentile(vol_paths * 100, [5, 95], axis=0)
            fig.add_trace(
                go.Scatter(
                    x=time_grid,
                    y=vol_pct[1],  # 95th
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=time_grid,
                    y=vol_pct[0],  # 5th
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(255, 127, 14, 0.2)",
                    name="Vol 5-95%",
                    legendgroup="vol_pct"
                ),
                row=2, col=1
            )

    else:
        # Constant volatility line
        initial_vol = get_initial_volatility(model_key, params) * 100
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=[initial_vol] * len(time_grid),
                mode="lines",
                line=dict(width=2, color="orange", dash="solid"),
                name=f"σ = {initial_vol:.1f}%"
            ),
            row=2, col=1
        )

    # Layout
    fig.update_layout(
        height=height,
        title=dict(
            text=f"<b>{_get_model_name(model_key)} Simulation</b>",
            x=0.5
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=80, b=40)
    )

    # Axis labels
    fig.update_xaxes(title_text="Time (years)", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_price_paths_only(
    result: SimulationResult,
    model_key: str,
    n_sample_paths: int = 100,
    show_percentiles: bool = True,
    show_mean: bool = True,
    height: int = 450,
):
    """Render price paths chart only."""
    fig = go.Figure()
    time_grid = result.time_grid
    n_paths = result.price_paths.shape[0]

    # Sample paths
    if n_paths > n_sample_paths:
        sample_idx = np.random.choice(n_paths, n_sample_paths, replace=False)
    else:
        sample_idx = np.arange(n_paths)

    price_color = "rgba(31, 119, 180, 0.3)"
    mean_color = "rgb(44, 160, 44)"

    # Sample paths
    for i, idx in enumerate(sample_idx):
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=result.price_paths[idx],
                mode="lines",
                line=dict(width=0.5, color=price_color),
                name="Sample Path" if i == 0 else None,
                showlegend=(i == 0),
                hoverinfo="skip"
            )
        )

    # Mean
    if show_mean:
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=result.mean_path,
                mode="lines",
                line=dict(width=2, color=mean_color),
                name="Mean"
            )
        )

    # Percentiles
    if show_percentiles:
        pct_paths = result.percentile_paths([5, 95])
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=pct_paths[1],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=pct_paths[0],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(214, 39, 40, 0.2)",
                name="5-95%"
            )
        )

    # Initial price
    fig.add_hline(
        y=result.initial_price,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"S₀ = {result.initial_price:.2f}"
    )

    fig.update_layout(
        height=height,
        title=f"<b>Price Paths - {_get_model_name(model_key)}</b>",
        xaxis_title="Time (years)",
        yaxis_title="Price ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_volatility_paths_only(
    result: SimulationResult,
    model_key: str,
    params: Dict[str, Any],
    n_sample_paths: int = 100,
    height: int = 350,
):
    """Render volatility paths chart only."""
    fig = go.Figure()
    characteristics = get_model_characteristics(model_key)
    time_grid = result.time_grid

    if characteristics["has_stochastic_vol"] and result.volatility_paths is not None:
        vol_paths = result.volatility_paths
        n_paths = vol_paths.shape[0]

        if n_paths > n_sample_paths:
            sample_idx = np.random.choice(n_paths, n_sample_paths, replace=False)
        else:
            sample_idx = np.arange(n_paths)

        vol_color = "rgba(255, 127, 14, 0.3)"
        mean_color = "rgb(44, 160, 44)"

        # Sample paths
        for i, idx in enumerate(sample_idx):
            fig.add_trace(
                go.Scatter(
                    x=time_grid,
                    y=vol_paths[idx] * 100,
                    mode="lines",
                    line=dict(width=0.5, color=vol_color),
                    name="Vol Path" if i == 0 else None,
                    showlegend=(i == 0),
                    hoverinfo="skip"
                )
            )

        # Mean
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=result.mean_volatility_path * 100,
                mode="lines",
                line=dict(width=2, color=mean_color),
                name="Mean Vol"
            )
        )

        title = f"<b>Stochastic Volatility - {_get_model_name(model_key)}</b>"

    else:
        # Constant volatility
        initial_vol = get_initial_volatility(model_key, params) * 100
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=[initial_vol] * len(time_grid),
                mode="lines",
                line=dict(width=2, color="orange"),
                name=f"σ = {initial_vol:.1f}%"
            )
        )
        title = f"<b>Constant Volatility - {_get_model_name(model_key)}</b>"

    fig.update_layout(
        height=height,
        title=title,
        xaxis_title="Time (years)",
        yaxis_title="Volatility (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)


def _get_model_name(model_key: str) -> str:
    """Get display name for model."""
    names = {
        "gbm": "GBM",
        "heston": "Heston",
        "merton": "Merton",
        "bates": "Bates",
        "garch": "GARCH",
        "ngarch": "NGARCH",
        "gjr_garch": "GJR-GARCH",
    }
    return names.get(model_key.lower(), model_key)


def render_path_controls() -> Dict[str, Any]:
    """Render controls for path visualization."""
    col1, col2, col3 = st.columns(3)

    with col1:
        n_paths = st.slider(
            "Sample Paths",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Number of paths to display"
        )

    with col2:
        show_percentiles = st.checkbox("Show Percentiles", value=True)

    with col3:
        show_mean = st.checkbox("Show Mean", value=True)

    return {
        "n_sample_paths": n_paths,
        "show_percentiles": show_percentiles,
        "show_mean": show_mean,
    }


# Alias for consistency
render_unified_paths_chart = render_unified_paths
