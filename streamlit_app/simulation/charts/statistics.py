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

    # Simulation parameters summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Simulation Parameters**")
        st.write(f"- Paths: {result.num_paths:,}")
        st.write(f"- Steps: {result.num_steps:,}")
        st.write(f"- Total samples: {result.num_paths * result.num_steps:,}")

    with col2:
        st.markdown("**Model Parameters**")
        st.write(f"- S0: ${params['spot_price']:.2f}")
        st.write(f"- r: {params['risk_free_rate']*100:.2f}%")
        st.write(f"- sigma: {params['volatility']*100:.1f}%")

    with col3:
        st.markdown("**Time Parameters**")
        st.write(f"- T: {params['time_horizon']:.2f} years")
        st.write(f"- dt: {params['time_horizon']/params['num_steps']:.6f}")
        st.write(f"- Seed: {params.get('seed', 'Random')}")

    st.markdown("---")

    # Terminal value statistics
    if result_type == "price":
        terminal = result.terminal_values
        name = "Terminal Price"
        unit = "$"
    else:
        terminal = result.terminal_volatility * 100
        name = "Terminal Volatility"
        unit = "%"

    st.markdown(f"#### {name} Statistics")

    # Create statistics table
    stats_data = {
        'Statistic': [
            'Count', 'Mean', 'Std Dev', 'Min', '5th Percentile',
            '25th Percentile', 'Median', '75th Percentile',
            '95th Percentile', 'Max', 'Skewness', 'Kurtosis'
        ],
        'Value': [
            f"{len(terminal):,}",
            f"{terminal.mean():.4f}",
            f"{terminal.std():.4f}",
            f"{terminal.min():.4f}",
            f"{np.percentile(terminal, 5):.4f}",
            f"{np.percentile(terminal, 25):.4f}",
            f"{np.median(terminal):.4f}",
            f"{np.percentile(terminal, 75):.4f}",
            f"{np.percentile(terminal, 95):.4f}",
            f"{terminal.max():.4f}",
            f"{stats.skew(terminal):.4f}",
            f"{stats.kurtosis(terminal):.4f}"
        ]
    }

    # Display as columns
    col1, col2 = st.columns(2)

    with col1:
        for i in range(len(stats_data['Statistic']) // 2):
            st.write(f"**{stats_data['Statistic'][i]}**: {stats_data['Value'][i]}{unit if i > 0 and i < 10 else ''}")

    with col2:
        for i in range(len(stats_data['Statistic']) // 2, len(stats_data['Statistic'])):
            st.write(f"**{stats_data['Statistic'][i]}**: {stats_data['Value'][i]}{unit if i < 10 else ''}")

    # Path statistics over time
    st.markdown("#### Path Evolution Statistics")

    paths = result.paths if result_type == "price" else result.volatility_paths * 100
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

    st.plotly_chart(fig, use_container_width=True)


def _render_convergence_analysis(result, params: Dict[str, Any], result_type: str) -> None:
    """Analyze Monte Carlo convergence."""
    st.markdown("#### Monte Carlo Convergence Analysis")

    if result_type == "price":
        terminal = result.terminal_values
        true_mean = params['spot_price'] * np.exp(params['risk_free_rate'] * params['time_horizon'])
        name = "Terminal Price"
    else:
        terminal = result.terminal_volatility
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

    st.plotly_chart(fig, use_container_width=True)

    # Convergence statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        final_se = terminal.std() / np.sqrt(n_paths)
        st.metric("Final Standard Error", f"{final_se:.6f}")

    with col2:
        ci_width = 2 * 1.96 * final_se
        st.metric("95% CI Width", f"{ci_width:.6f}")

    with col3:
        if true_mean is not None:
            error = abs(terminal.mean() - true_mean)
            st.metric("Absolute Error", f"{error:.6f}")


def _render_performance_metrics(result, params: Dict[str, Any]) -> None:
    """Render computation performance metrics."""
    st.markdown("#### Computation Performance")

    comp_time = result.computation_time
    n_paths = result.num_paths
    n_steps = result.num_steps
    total_samples = n_paths * n_steps

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Computation Time", f"{comp_time*1000:.2f} ms")

    with col2:
        paths_per_sec = n_paths / comp_time
        st.metric("Paths/Second", f"{paths_per_sec:,.0f}")

    with col3:
        samples_per_sec = total_samples / comp_time
        st.metric("Samples/Second", f"{samples_per_sec/1e6:.2f}M")

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

    st.plotly_chart(fig, use_container_width=True)

    # Memory usage estimate
    st.markdown("##### Memory Usage Estimate")

    bytes_per_float = 8
    path_memory = n_paths * (n_steps + 1) * bytes_per_float
    path_memory_mb = path_memory / (1024 * 1024)

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Path Array Size**: {path_memory_mb:.2f} MB")
        st.write(f"**Dimensions**: ({n_paths:,} x {n_steps + 1:,})")

    with col2:
        st.write(f"**Data Type**: float64 (8 bytes)")
        st.write(f"**Total Elements**: {n_paths * (n_steps + 1):,}")
