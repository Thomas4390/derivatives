"""
Statistics and summary charts for Monte Carlo Simulation Explorer.

Provides comprehensive statistical analysis and model comparison tools.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, List, Optional
from scipy import stats

from config.constants import (
    CHART_HEIGHT_STANDARD,
    CHART_HEIGHT_LARGE,
    PRICE_MODELS,
    VOLATILITY_MODELS,
    MODEL_COLORS
)
from config.styles import render_stats_row, stats_table_html


def _get_param(params: Dict, *keys, default=None):
    """Get parameter value with fallback for different naming conventions."""
    for key in keys:
        if key in params:
            return params[key]
    return default


def _get_result_attr(result, *attrs, default=None):
    """Get attribute from result with fallback for different interfaces."""
    for attr in attrs:
        if hasattr(result, attr):
            val = getattr(result, attr)
            if val is not None:
                return val
    return default


def render_statistics_tab(
    simulation_result,
    params: Dict[str, Any],
    result_type: str = "price"
) -> None:
    """
    Render comprehensive statistics tab.

    Args:
        simulation_result: Simulation result object
        params: Dictionary of simulation parameters
        result_type: "price" or "volatility"
    """
    if simulation_result is None:
        st.info("Run a simulation to see statistics.")
        return

    st.markdown("### Simulation Statistics")

    tab1, tab2, tab3 = st.tabs([
        "Summary Statistics",
        "Convergence Analysis",
        "Computation Performance"
    ])

    with tab1:
        _render_summary_statistics(simulation_result, params, result_type)

    with tab2:
        _render_convergence_analysis(simulation_result, params, result_type)

    with tab3:
        _render_performance_metrics(simulation_result, params)


def _render_summary_statistics(result, params: Dict[str, Any], result_type: str) -> None:
    """Render comprehensive summary statistics."""
    st.markdown("#### Simulation Summary")

    # Get values with compatibility
    n_paths = _get_result_attr(result, 'num_paths', default=result.price_paths.shape[0] if hasattr(result, 'price_paths') else 0)
    n_steps = _get_result_attr(result, 'num_steps', default=result.price_paths.shape[1]-1 if hasattr(result, 'price_paths') else 0)
    spot_price = _get_param(params, 'spot_price', 'spot', default=100.0)
    risk_free_rate = _get_param(params, 'risk_free_rate', default=0.05)
    volatility = _get_param(params, 'volatility', 'sigma', 'sigma0', default=0.20)
    time_horizon = _get_param(params, 'time_horizon', default=1.0)
    num_steps_param = _get_param(params, 'num_steps', 'n_steps', default=n_steps)

    # Simulation parameters with styled cards
    st.markdown("##### ⚡ Simulation Parameters")
    sim_stats = [
        ("Paths", f"{n_paths:,}", "Number of simulations"),
        ("Steps", f"{n_steps:,}", "Time discretization"),
        ("Total Samples", f"{n_paths * n_steps:,}", "paths × steps"),
    ]
    render_stats_row(sim_stats, ["teal", "blue", "purple"])

    st.markdown("")  # Spacer

    # Market parameters with styled cards
    st.markdown("##### 📊 Market Parameters")
    market_stats = [
        ("S₀ (Spot)", f"${spot_price:.2f}", "Initial price"),
        ("r (Risk-free)", f"{risk_free_rate*100:.2f}%", "Annual rate"),
        ("σ (Volatility)", f"{volatility*100:.1f}%", "Annual vol"),
    ]
    render_stats_row(market_stats, ["green", "blue", "amber"])

    st.markdown("")  # Spacer

    # Time parameters with styled cards
    st.markdown("##### ⏱️ Time Parameters")
    time_stats = [
        ("T (Horizon)", f"{time_horizon:.2f} yr", "Simulation period"),
        ("dt (Step size)", f"{time_horizon/num_steps_param:.6f}", "Time increment"),
        ("Seed", f"{params.get('seed', 'Random')}", "Random state"),
    ]
    render_stats_row(time_stats, ["slate", "purple", "teal"])

    st.markdown("---")

    # Terminal value statistics
    if result_type == "price":
        terminal = _get_result_attr(result, 'terminal_prices', 'terminal_values')
        name = "Terminal Price"
        unit = "$"
    else:
        vol_paths = _get_result_attr(result, 'volatility_paths')
        if vol_paths is not None:
            terminal = vol_paths[:, -1] * 100
        else:
            terminal = _get_result_attr(result, 'terminal_volatility', default=np.array([0])) * 100
        name = "Terminal Volatility"
        unit = "%"

    st.markdown(f"#### {name} Statistics")

    # Create statistics using pandas DataFrame
    import pandas as pd

    stats_data = [
        {'Statistic': 'Count', 'Value': f"{len(terminal):,}"},
        {'Statistic': 'Mean', 'Value': f"{terminal.mean():.4f}{unit}"},
        {'Statistic': 'Std Dev', 'Value': f"{terminal.std():.4f}{unit}"},
        {'Statistic': 'Min', 'Value': f"{terminal.min():.4f}{unit}"},
        {'Statistic': '5th Percentile', 'Value': f"{np.percentile(terminal, 5):.4f}{unit}"},
        {'Statistic': '25th Percentile', 'Value': f"{np.percentile(terminal, 25):.4f}{unit}"},
        {'Statistic': 'Median', 'Value': f"{np.median(terminal):.4f}{unit}"},
        {'Statistic': '75th Percentile', 'Value': f"{np.percentile(terminal, 75):.4f}{unit}"},
        {'Statistic': '95th Percentile', 'Value': f"{np.percentile(terminal, 95):.4f}{unit}"},
        {'Statistic': 'Max', 'Value': f"{terminal.max():.4f}{unit}"},
        {'Statistic': 'Skewness', 'Value': f"{stats.skew(terminal):.4f}"},
        {'Statistic': 'Kurtosis', 'Value': f"{stats.kurtosis(terminal):.4f}"},
    ]

    df_stats = pd.DataFrame(stats_data)
    st.dataframe(
        df_stats,
        width="stretch",
        hide_index=True,
        column_config={
            "Statistic": st.column_config.TextColumn("Statistic", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="medium")
        }
    )

    # Path statistics over time
    st.markdown("#### Path Evolution Statistics")

    if result_type == "price":
        paths = _get_result_attr(result, 'price_paths', 'paths')
    else:
        vol_paths = _get_result_attr(result, 'volatility_paths')
        paths = vol_paths * 100 if vol_paths is not None else None

    if paths is None:
        st.warning("Path data not available for evolution statistics.")
        return

    time_grid = result.time_grid

    # Sample points for statistics
    sample_points = [0, len(time_grid)//4, len(time_grid)//2, 3*len(time_grid)//4, -1]
    sample_times = [time_grid[i] for i in sample_points]

    stats_evolution = []
    for i in sample_points:
        vals = paths[:, i]
        stats_evolution.append({
            't': time_grid[i],
            'mean': vals.mean(),
            'std': vals.std(),
            'min': vals.min(),
            'max': vals.max()
        })

    # Create evolution chart
    fig = go.Figure()

    means = [s['mean'] for s in stats_evolution]
    stds = [s['std'] for s in stats_evolution]
    times = [s['t'] for s in stats_evolution]

    # Mean with std bands
    fig.add_trace(go.Scatter(
        x=times + times[::-1],
        y=[m + 2*s for m, s in zip(means, stds)] + [m - 2*s for m, s in zip(means[::-1], stds[::-1])],
        fill='toself',
        fillcolor='rgba(13, 148, 136, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='+/- 2 Std'
    ))

    fig.add_trace(go.Scatter(
        x=times,
        y=means,
        mode='lines+markers',
        name='Mean',
        line=dict(color='#0d9488', width=2)
    ))

    fig.update_layout(
        title=f"{name} Evolution Over Time",
        xaxis_title="Time",
        yaxis_title=f"{name} ({unit})",
        height=CHART_HEIGHT_STANDARD
    )

    st.plotly_chart(fig, width="stretch")


def _render_convergence_analysis(result, params: Dict[str, Any], result_type: str) -> None:
    """Analyze Monte Carlo convergence."""
    st.markdown("#### Monte Carlo Convergence Analysis")

    spot_price = _get_param(params, 'spot_price', 'spot', default=100.0)
    risk_free_rate = _get_param(params, 'risk_free_rate', default=0.05)
    time_horizon = _get_param(params, 'time_horizon', default=1.0)

    if result_type == "price":
        terminal = _get_result_attr(result, 'terminal_prices', 'terminal_values')
        true_mean = spot_price * np.exp(risk_free_rate * time_horizon)
        name = "Terminal Price"
    else:
        vol_paths = _get_result_attr(result, 'volatility_paths')
        if vol_paths is not None:
            terminal = vol_paths[:, -1]
        else:
            terminal = _get_result_attr(result, 'terminal_volatility', default=np.array([0]))
        true_mean = None  # No closed-form for GARCH
        name = "Terminal Volatility"

    n_paths = len(terminal)

    # Compute running mean and std error
    sample_sizes = np.unique(np.logspace(1, np.log10(n_paths), 100).astype(int))
    sample_sizes = sample_sizes[sample_sizes <= n_paths]

    running_means = []
    running_stds = []
    std_errors = []

    for n in sample_sizes:
        sample = terminal[:n]
        running_means.append(sample.mean())
        running_stds.append(sample.std())
        std_errors.append(sample.std() / np.sqrt(n))

    running_means = np.array(running_means)
    std_errors = np.array(std_errors)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Mean Convergence", "Standard Error Decay")
    )

    # Mean convergence
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=running_means,
            mode='lines',
            name='Running Mean',
            line=dict(color='#0d9488', width=2)
        ),
        row=1, col=1
    )

    # Confidence band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([sample_sizes, sample_sizes[::-1]]),
            y=np.concatenate([running_means + 1.96*std_errors,
                             (running_means - 1.96*std_errors)[::-1]]),
            fill='toself',
            fillcolor='rgba(13, 148, 136, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name='95% CI'
        ),
        row=1, col=1
    )

    if true_mean is not None:
        fig.add_hline(y=true_mean, line_dash="dash", line_color="red",
                      annotation_text="Theoretical", row=1, col=1)

    # Standard error decay
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=std_errors,
            mode='lines',
            name='Std Error',
            line=dict(color='#d97706', width=2)
        ),
        row=1, col=2
    )

    # Theoretical 1/sqrt(n) decay
    theoretical_se = terminal.std() / np.sqrt(sample_sizes)
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=theoretical_se,
            mode='lines',
            name='1/sqrt(n)',
            line=dict(color='gray', width=1, dash='dash')
        ),
        row=1, col=2
    )

    fig.update_xaxes(type='log', title_text='Number of Paths')
    fig.update_yaxes(title_text=f'Mean {name}', row=1, col=1)
    fig.update_yaxes(type='log', title_text='Standard Error', row=1, col=2)

    fig.update_layout(
        height=CHART_HEIGHT_STANDARD,
        showlegend=True
    )

    st.plotly_chart(fig, width="stretch")

    # Convergence statistics with styled cards
    final_se = terminal.std() / np.sqrt(n_paths)
    ci_width = 2 * 1.96 * final_se

    conv_stats = [
        ("Final Std Error", f"{final_se:.6f}", "σ / √n"),
        ("95% CI Width", f"±{ci_width:.6f}", "1.96 × SE"),
    ]
    conv_variants = ["teal", "blue"]

    if true_mean is not None:
        error = abs(terminal.mean() - true_mean)
        error_pct = error / true_mean * 100 if true_mean > 0 else 0
        error_variant = "green" if error_pct < 1 else "amber" if error_pct < 5 else "red"
        conv_stats.append(("Absolute Error", f"{error:.6f}", f"{error_pct:.2f}% of theoretical"))
        conv_variants.append(error_variant)
    else:
        conv_stats.append(("Absolute Error", "N/A", "No closed-form solution"))
        conv_variants.append("slate")

    render_stats_row(conv_stats, conv_variants)


def _render_performance_metrics(result, params: Dict[str, Any]) -> None:
    """Render computation performance metrics."""
    st.markdown("#### Computation Performance")

    comp_time = _get_result_attr(result, 'computation_time', default=0.0)
    if comp_time == 0:
        # Fallback: try to get from session state
        comp_time = st.session_state.get('execution_time', 0.001)

    n_paths = _get_result_attr(result, 'num_paths', default=result.price_paths.shape[0] if hasattr(result, 'price_paths') else 1000)
    n_steps = _get_result_attr(result, 'num_steps', default=result.price_paths.shape[1]-1 if hasattr(result, 'price_paths') else 252)
    total_samples = n_paths * n_steps

    paths_per_sec = n_paths / comp_time
    samples_per_sec = total_samples / comp_time

    # Performance metrics with styled cards
    perf_stats = [
        ("Computation Time", f"{comp_time*1000:.1f} ms", "Total generation time"),
        ("Paths/Second", f"{paths_per_sec:,.0f}", "Throughput"),
        ("Samples/Second", f"{samples_per_sec/1e6:.1f}M", "Million samples/sec"),
    ]
    render_stats_row(perf_stats, ["teal", "green", "blue"])

    # Performance comparison chart
    st.markdown("##### Scaling Analysis")

    # Simulated scaling data
    path_counts = [1000, 5000, 10000, 50000, 100000]
    estimated_times = [comp_time * (n / n_paths) for n in path_counts]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=path_counts,
        y=[t * 1000 for t in estimated_times],
        mode='lines+markers',
        name='Estimated Time',
        line=dict(color='#0d9488', width=2)
    ))

    # Theoretical linear scaling
    fig.add_trace(go.Scatter(
        x=path_counts,
        y=[path_counts[0] * comp_time / n_paths * n * 1000 for n in path_counts],
        mode='lines',
        name='Linear Scaling',
        line=dict(color='gray', dash='dash')
    ))

    fig.update_layout(
        title="Computation Time Scaling",
        xaxis_title="Number of Paths",
        yaxis_title="Time (ms)",
        height=CHART_HEIGHT_STANDARD
    )

    st.plotly_chart(fig, width="stretch")

    # Memory usage estimate with styled cards
    st.markdown("##### Memory Usage Estimate")

    bytes_per_float = 8
    path_memory = n_paths * (n_steps + 1) * bytes_per_float
    path_memory_mb = path_memory / (1024 * 1024)

    memory_stats = [
        ("Memory", f"{path_memory_mb:.2f} MB", "Array storage"),
        ("Dimensions", f"{n_paths:,} × {n_steps + 1:,}", "paths × steps"),
        ("Total Elements", f"{n_paths * (n_steps + 1):,}", "Float values"),
        ("Data Type", "float64", "8 bytes/element"),
    ]
    render_stats_row(memory_stats, ["purple", "blue", "teal", "slate"])
