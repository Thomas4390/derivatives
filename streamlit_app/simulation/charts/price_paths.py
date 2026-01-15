"""
Price path visualization charts for Monte Carlo Simulation Explorer.

Provides interactive visualizations for simulated price paths using Plotly.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, Optional

from config.constants import (
    CHART_HEIGHT_STANDARD,
    CHART_HEIGHT_LARGE,
    PATH_COLORS,
    MODEL_COLORS,
    PRICE_MODELS
)


def render_price_paths_tab(
    simulation_result,
    params: Dict[str, Any],
    show_variance_paths: bool = False
) -> None:
    """
    Render the price paths visualization tab.

    Args:
        simulation_result: SimulationResult object from backend
        params: Dictionary of simulation parameters
        show_variance_paths: Whether to show variance paths (for Heston/SABR)
    """
    if simulation_result is None:
        st.info("Run a simulation to see price paths visualization.")
        return

    st.markdown("### Simulated Price Paths")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Sample Paths", "Statistics View", "Log Returns"])

    with tab1:
        _render_sample_paths(simulation_result, params)

    with tab2:
        _render_statistics_view(simulation_result, params)

    with tab3:
        _render_log_returns(simulation_result, params)

    # If Heston or SABR, show variance/volatility paths
    if show_variance_paths and hasattr(simulation_result, 'variance_paths'):
        st.markdown("### Variance/Volatility Paths")
        _render_variance_paths(simulation_result, params)


def _render_sample_paths(result, params: Dict[str, Any]) -> None:
    """Render sample price paths with percentile bands."""
    paths = result.paths
    time_grid = result.time_grid
    n_paths = paths.shape[0]

    # Limit display paths
    max_display = min(params.get('max_display_paths', 50), n_paths)

    fig = go.Figure()

    # Sample paths (translucent)
    for i in range(max_display):
        fig.add_trace(go.Scatter(
            x=time_grid,
            y=paths[i],
            mode='lines',
            line=dict(
                color=PATH_COLORS['sample_paths'],
                width=0.8
            ),
            hoverinfo='skip',
            showlegend=False
        ))

    # Mean path
    if params.get('show_mean', True):
        mean_path = result.mean_path
        fig.add_trace(go.Scatter(
            x=time_grid,
            y=mean_path,
            mode='lines',
            name='Mean Path',
            line=dict(
                color=PATH_COLORS['mean_path'],
                width=2.5
            )
        ))

    # Percentile bands
    if params.get('show_percentiles', True):
        percentiles = result.percentile_paths([5, 50, 95])
        p5, p50, p95 = percentiles[0], percentiles[1], percentiles[2]

        # 5-95% band
        fig.add_trace(go.Scatter(
            x=np.concatenate([time_grid, time_grid[::-1]]),
            y=np.concatenate([p95, p5[::-1]]),
            fill='toself',
            fillcolor=PATH_COLORS['percentile_band'],
            line=dict(color='rgba(0,0,0,0)'),
            name='5-95% Range',
            hoverinfo='skip'
        ))

        # Median
        fig.add_trace(go.Scatter(
            x=time_grid,
            y=p50,
            mode='lines',
            name='Median',
            line=dict(
                color='#0d9488',
                width=2,
                dash='dash'
            )
        ))

    # Initial price line
    fig.add_hline(
        y=params['spot_price'],
        line_dash="dot",
        line_color=PATH_COLORS['initial_price'],
        annotation_text=f"S0 = ${params['spot_price']:.2f}"
    )

    fig.update_layout(
        title=dict(
            text=f"{PRICE_MODELS.get(params['price_model'], 'Model')} - {result.num_paths:,} Paths",
            font=dict(size=16)
        ),
        xaxis_title="Time (Years)",
        yaxis_title="Price ($)",
        height=CHART_HEIGHT_LARGE,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=60, r=40, t=60, b=60)
    )

    st.plotly_chart(fig, width="stretch")

    # Path statistics summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Mean Terminal Price",
            f"${result.terminal_values.mean():.2f}"
        )
    with col2:
        st.metric(
            "Std Dev",
            f"${result.terminal_values.std():.2f}"
        )
    with col3:
        st.metric(
            "Min Terminal",
            f"${result.terminal_values.min():.2f}"
        )
    with col4:
        st.metric(
            "Max Terminal",
            f"${result.terminal_values.max():.2f}"
        )


def _render_statistics_view(result, params: Dict[str, Any]) -> None:
    """Render statistical view with confidence intervals over time."""
    paths = result.paths
    time_grid = result.time_grid

    # Compute statistics at each time point
    mean_path = np.mean(paths, axis=0)
    std_path = np.std(paths, axis=0)
    percentiles = np.percentile(paths, [1, 5, 25, 50, 75, 95, 99], axis=0)

    fig = go.Figure()

    # 1-99% range (very light)
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_grid, time_grid[::-1]]),
        y=np.concatenate([percentiles[6], percentiles[0][::-1]]),
        fill='toself',
        fillcolor='rgba(13, 148, 136, 0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='1-99%',
        hoverinfo='skip'
    ))

    # 5-95% range
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_grid, time_grid[::-1]]),
        y=np.concatenate([percentiles[5], percentiles[1][::-1]]),
        fill='toself',
        fillcolor='rgba(13, 148, 136, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='5-95%',
        hoverinfo='skip'
    ))

    # 25-75% range (IQR)
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_grid, time_grid[::-1]]),
        y=np.concatenate([percentiles[4], percentiles[2][::-1]]),
        fill='toself',
        fillcolor='rgba(13, 148, 136, 0.3)',
        line=dict(color='rgba(0,0,0,0)'),
        name='25-75% (IQR)',
        hoverinfo='skip'
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=time_grid,
        y=percentiles[3],
        mode='lines',
        name='Median',
        line=dict(color='#0d9488', width=2)
    ))

    # Mean with std band
    fig.add_trace(go.Scatter(
        x=time_grid,
        y=mean_path,
        mode='lines',
        name='Mean',
        line=dict(color='#1a365d', width=2, dash='dash')
    ))

    # Initial price
    fig.add_hline(
        y=params['spot_price'],
        line_dash="dot",
        line_color='#dc2626',
        annotation_text="S0"
    )

    fig.update_layout(
        title="Price Distribution Over Time",
        xaxis_title="Time (Years)",
        yaxis_title="Price ($)",
        height=CHART_HEIGHT_STANDARD,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01
        )
    )

    st.plotly_chart(fig, width="stretch")


def _render_log_returns(result, params: Dict[str, Any]) -> None:
    """Render log returns analysis."""
    paths = result.paths

    # Compute log returns
    log_returns = np.diff(np.log(paths), axis=1)
    time_grid = result.time_grid[1:]

    # Aggregate statistics
    mean_returns = np.mean(log_returns, axis=0)
    std_returns = np.std(log_returns, axis=0)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Mean Log Return Over Time",
            "Volatility (Std Dev) Over Time",
            "Sample Log Returns Distribution",
            "Q-Q Plot vs Normal"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Mean returns over time
    fig.add_trace(
        go.Scatter(
            x=time_grid,
            y=mean_returns,
            mode='lines',
            name='Mean Return',
            line=dict(color='#0d9488')
        ),
        row=1, col=1
    )

    # Volatility over time
    fig.add_trace(
        go.Scatter(
            x=time_grid,
            y=std_returns,
            mode='lines',
            name='Std Dev',
            line=dict(color='#d97706')
        ),
        row=1, col=2
    )

    # Terminal returns histogram
    terminal_returns = log_returns[:, -1]
    fig.add_trace(
        go.Histogram(
            x=terminal_returns,
            nbinsx=50,
            name='Terminal Returns',
            marker_color='#0d9488',
            opacity=0.7
        ),
        row=2, col=1
    )

    # Q-Q plot
    sorted_returns = np.sort(terminal_returns)
    n = len(sorted_returns)
    theoretical_quantiles = np.array([
        _norm_ppf((i - 0.5) / n) for i in range(1, n + 1)
    ])

    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_returns,
            mode='markers',
            name='Q-Q',
            marker=dict(color='#0d9488', size=3)
        ),
        row=2, col=2
    )

    # Add reference line for Q-Q
    fig.add_trace(
        go.Scatter(
            x=[-3, 3],
            y=[-3 * terminal_returns.std() + terminal_returns.mean(),
               3 * terminal_returns.std() + terminal_returns.mean()],
            mode='lines',
            name='Normal Reference',
            line=dict(color='red', dash='dash')
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=CHART_HEIGHT_LARGE,
        showlegend=False
    )

    st.plotly_chart(fig, width="stretch")

    # Return statistics
    st.markdown("#### Return Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        annualized_return = np.mean(log_returns) * 252
        st.metric("Annualized Return", f"{annualized_return*100:.2f}%")

    with col2:
        annualized_vol = np.std(log_returns) * np.sqrt(252)
        st.metric("Annualized Volatility", f"{annualized_vol*100:.2f}%")

    with col3:
        from scipy.stats import skew
        skewness = skew(terminal_returns)
        st.metric("Skewness", f"{skewness:.3f}")

    with col4:
        from scipy.stats import kurtosis
        kurt = kurtosis(terminal_returns)
        st.metric("Excess Kurtosis", f"{kurt:.3f}")


def _render_variance_paths(result, params: Dict[str, Any]) -> None:
    """Render variance/volatility paths for stochastic vol models."""
    if not hasattr(result, 'variance_paths') or result.variance_paths is None:
        return

    variance_paths = result.variance_paths
    time_grid = result.time_grid
    n_paths = variance_paths.shape[0]
    max_display = min(params.get('max_display_paths', 50), n_paths)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Variance Paths", "Volatility Paths")
    )

    # Variance paths
    for i in range(max_display):
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=variance_paths[i],
                mode='lines',
                line=dict(color='rgba(217, 119, 6, 0.15)', width=0.8),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    # Mean variance
    mean_var = np.mean(variance_paths, axis=0)
    fig.add_trace(
        go.Scatter(
            x=time_grid,
            y=mean_var,
            mode='lines',
            name='Mean Variance',
            line=dict(color='#d97706', width=2)
        ),
        row=1, col=1
    )

    # Volatility paths
    vol_paths = np.sqrt(variance_paths)
    for i in range(max_display):
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=vol_paths[i] * 100,
                mode='lines',
                line=dict(color='rgba(13, 148, 136, 0.15)', width=0.8),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=2
        )

    # Mean volatility
    mean_vol = np.mean(vol_paths, axis=0) * 100
    fig.add_trace(
        go.Scatter(
            x=time_grid,
            y=mean_vol,
            mode='lines',
            name='Mean Volatility',
            line=dict(color='#0d9488', width=2)
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=CHART_HEIGHT_STANDARD,
        showlegend=True
    )

    fig.update_xaxes(title_text="Time (Years)")
    fig.update_yaxes(title_text="Variance", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=1, col=2)

    st.plotly_chart(fig, width="stretch")


def _norm_ppf(p: float) -> float:
    """Approximate normal inverse CDF (percent point function)."""
    # Simple approximation using Abramowitz and Stegun
    import math

    if p <= 0:
        return -10
    if p >= 1:
        return 10

    if p < 0.5:
        return -_norm_ppf(1 - p)

    t = math.sqrt(-2 * math.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
