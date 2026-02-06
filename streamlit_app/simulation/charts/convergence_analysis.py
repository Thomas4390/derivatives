"""
Monte Carlo Convergence Analysis Module.

Pedagogical module teaching:
1. √N convergence: Standard error decreases as SE = σ/√N
2. Running statistics: How estimates stabilize with more paths
3. Confidence intervals: Quantifying estimation uncertainty

Key insight: "With 10K paths, your estimate has ±$X uncertainty"
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, Optional
from scipy import stats

from config.constants import CHART_HEIGHT_STANDARD, CHART_HEIGHT_LARGE
from config.styles import render_stats_row


def render_convergence_analysis_tab(
    simulation_result,
    pnl_result: Optional[Dict[str, Any]],
    params: Dict[str, Any]
) -> None:
    """
    Render Monte Carlo convergence analysis educational tab.

    Shows how estimates converge as number of paths increases.
    """
    if simulation_result is None:
        st.info("Run a simulation to see convergence analysis.")
        return

    st.markdown("### Monte Carlo Convergence Analysis")
    st.caption(
        "Understand how estimation uncertainty decreases with more simulation paths. "
        "The key insight: standard error decreases as 1/√N."
    )

    # Educational formula box
    st.markdown("""
    <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #0284c7;">
        <h5 style="color: #0c4a6e; margin-top: 0;">Monte Carlo Standard Error</h5>
        <p style="font-family: monospace; font-size: 1.1em; margin-bottom: 0.5rem;">
            SE(mean) = σ / √N
        </p>
        <p style="margin-bottom: 0; font-size: 0.9em; color: #475569;">
            To halve uncertainty, you need 4x more paths. To reduce by 10x, you need 100x more paths.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Get data
    has_pnl = pnl_result is not None and 'pnl_values' in pnl_result

    if has_pnl:
        data = pnl_result['pnl_values']
        data_label = "P&L"
        data_unit = "$"
    else:
        data = simulation_result.terminal_prices
        data_label = "Terminal Price"
        data_unit = "$"

    n_paths = len(data)

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "Running Statistics",
        "Confidence Bands",
        "Convergence Calculator"
    ])

    with tab1:
        _render_running_statistics(data, data_label, data_unit, n_paths)

    with tab2:
        _render_confidence_bands(data, data_label, data_unit, n_paths)

    with tab3:
        _render_convergence_calculator(data, data_label, data_unit, n_paths)


def _render_running_statistics(
    data: np.ndarray,
    data_label: str,
    data_unit: str,
    n_paths: int
) -> None:
    """Show how statistics stabilize as N increases."""
    st.markdown("#### Running Statistics")
    st.caption(
        "Watch how mean, VaR, and standard deviation estimates stabilize "
        "as we include more simulation paths."
    )

    # Calculate running statistics at different sample sizes
    sample_points = np.unique(np.concatenate([
        np.arange(100, min(1000, n_paths), 100),
        np.arange(1000, min(10000, n_paths), 1000),
        np.arange(10000, n_paths + 1, 5000),
        [n_paths]
    ]))
    sample_points = sample_points[sample_points <= n_paths]

    running_mean = []
    running_std = []
    running_var95 = []
    running_se = []

    for n in sample_points:
        sample = data[:n]
        running_mean.append(np.mean(sample))
        running_std.append(np.std(sample))
        running_var95.append(np.percentile(sample, 5))  # 5th percentile = VaR 95%
        running_se.append(np.std(sample) / np.sqrt(n))

    running_mean = np.array(running_mean)
    running_std = np.array(running_std)
    running_var95 = np.array(running_var95)
    running_se = np.array(running_se)

    # Final estimates
    final_mean = running_mean[-1]
    final_std = running_std[-1]
    final_se = running_se[-1]

    # Create figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Running Mean {data_label}",
            "Standard Error of Mean",
            f"Running VaR 95% ({data_label})",
            "Running Standard Deviation"
        ),
        vertical_spacing=0.12
    )

    # 1. Running Mean
    fig.add_trace(
        go.Scatter(
            x=sample_points,
            y=running_mean,
            mode='lines',
            line=dict(color='#2563eb', width=2),
            name='Running Mean'
        ),
        row=1, col=1
    )
    fig.add_hline(y=final_mean, line_dash="dash", line_color="red", row=1, col=1)

    # Add confidence band around mean
    upper = running_mean + 1.96 * running_se
    lower = running_mean - 1.96 * running_se
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([sample_points, sample_points[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill='toself',
            fillcolor='rgba(37, 99, 235, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name='95% CI',
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    # 2. Standard Error
    fig.add_trace(
        go.Scatter(
            x=sample_points,
            y=running_se,
            mode='lines',
            line=dict(color='#dc2626', width=2),
            name='Standard Error'
        ),
        row=1, col=2
    )

    # Theoretical 1/sqrt(N) curve
    theoretical_se = final_std / np.sqrt(sample_points)
    fig.add_trace(
        go.Scatter(
            x=sample_points,
            y=theoretical_se,
            mode='lines',
            line=dict(color='gray', dash='dash', width=1),
            name='σ/√N (theory)'
        ),
        row=1, col=2
    )

    # 3. Running VaR 95%
    fig.add_trace(
        go.Scatter(
            x=sample_points,
            y=running_var95,
            mode='lines',
            line=dict(color='#f59e0b', width=2),
            name='Running VaR 95%'
        ),
        row=2, col=1
    )
    fig.add_hline(y=running_var95[-1], line_dash="dash", line_color="red", row=2, col=1)

    # 4. Running Std Dev
    fig.add_trace(
        go.Scatter(
            x=sample_points,
            y=running_std,
            mode='lines',
            line=dict(color='#059669', width=2),
            name='Running Std'
        ),
        row=2, col=2
    )
    fig.add_hline(y=final_std, line_dash="dash", line_color="red", row=2, col=2)

    # Update axes
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text="Number of Paths", type="log", row=row, col=col)

    fig.update_yaxes(title_text=f"Mean {data_label} ({data_unit})", row=1, col=1)
    fig.update_yaxes(title_text=f"Std Error ({data_unit})", row=1, col=2)
    fig.update_yaxes(title_text=f"VaR 95% ({data_unit})", row=2, col=1)
    fig.update_yaxes(title_text=f"Std Dev ({data_unit})", row=2, col=2)

    fig.update_layout(height=CHART_HEIGHT_LARGE, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    stats_data = [
        (f"Final Mean {data_label}", f"{data_unit}{final_mean:,.2f}", f"N = {n_paths:,}"),
        ("Standard Error", f"{data_unit}{final_se:,.2f}", "Estimation uncertainty"),
        ("95% Confidence Interval",
         f"{data_unit}{final_mean - 1.96*final_se:,.2f} to {data_unit}{final_mean + 1.96*final_se:,.2f}",
         "True mean likely in this range"),
        ("Relative Error", f"{final_se/abs(final_mean)*100:.2f}%" if final_mean != 0 else "N/A",
         "SE/Mean"),
    ]
    render_stats_row(stats_data, ["blue", "red", "teal", "amber"])


def _render_confidence_bands(
    data: np.ndarray,
    data_label: str,
    data_unit: str,
    n_paths: int
) -> None:
    """Show confidence bands narrowing with more paths."""
    st.markdown("#### Confidence Bands vs Sample Size")
    st.caption(
        "See how the 95% confidence interval narrows as you add more paths. "
        "The width decreases proportionally to 1/√N."
    )

    # Calculate mean and SE at various sample sizes
    sample_sizes = np.logspace(np.log10(100), np.log10(n_paths), 50).astype(int)
    sample_sizes = np.unique(sample_sizes)

    means = []
    lower_95 = []
    upper_95 = []
    lower_99 = []
    upper_99 = []

    for n in sample_sizes:
        sample = data[:n]
        mean = np.mean(sample)
        se = np.std(sample) / np.sqrt(n)
        means.append(mean)
        lower_95.append(mean - 1.96 * se)
        upper_95.append(mean + 1.96 * se)
        lower_99.append(mean - 2.576 * se)
        upper_99.append(mean + 2.576 * se)

    fig = go.Figure()

    # 99% CI band
    fig.add_trace(go.Scatter(
        x=np.concatenate([sample_sizes, sample_sizes[::-1]]),
        y=np.concatenate([upper_99, lower_99[::-1]]),
        fill='toself',
        fillcolor='rgba(239, 68, 68, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='99% CI',
        hoverinfo='skip'
    ))

    # 95% CI band
    fig.add_trace(go.Scatter(
        x=np.concatenate([sample_sizes, sample_sizes[::-1]]),
        y=np.concatenate([upper_95, lower_95[::-1]]),
        fill='toself',
        fillcolor='rgba(37, 99, 235, 0.3)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% CI',
        hoverinfo='skip'
    ))

    # Mean line
    fig.add_trace(go.Scatter(
        x=sample_sizes,
        y=means,
        mode='lines',
        line=dict(color='#1e3a5f', width=2),
        name='Running Mean'
    ))

    # Final value reference
    fig.add_hline(
        y=np.mean(data),
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Final: {data_unit}{np.mean(data):,.2f}"
    )

    fig.update_layout(
        title=f"{data_label} Estimate with Confidence Bands",
        xaxis_title="Number of Paths (log scale)",
        yaxis_title=f"{data_label} ({data_unit})",
        xaxis_type="log",
        height=CHART_HEIGHT_STANDARD,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Educational insight
    st.markdown("""
    <div style="background: #e8f4f8; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <h5 style="color: #1e3a5f; margin-top: 0;">Key Insight: The √N Rule</h5>
        <p style="margin-bottom: 0;">
            Notice how the confidence bands narrow slowly. To cut uncertainty in half,
            you need <strong>4x more paths</strong> (since SE ∝ 1/√N). This is why:
        </p>
        <ul style="margin-bottom: 0;">
            <li>Going from 1,000 to 10,000 paths only reduces error by ~68%</li>
            <li>Diminishing returns kick in quickly</li>
            <li>For production, 10K-50K paths is often sufficient</li>
            <li>Variance reduction techniques (antithetic variates, control variates) can help more than brute force</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def _render_convergence_calculator(
    data: np.ndarray,
    data_label: str,
    data_unit: str,
    n_paths: int
) -> None:
    """Interactive calculator for required sample size."""
    st.markdown("#### Convergence Calculator")
    st.caption("Calculate how many paths you need for a target precision.")

    # Current statistics
    current_mean = np.mean(data)
    current_std = np.std(data)
    current_se = current_std / np.sqrt(n_paths)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Current Simulation:**")
        st.write(f"- Paths: {n_paths:,}")
        st.write(f"- Mean: {data_unit}{current_mean:,.2f}")
        st.write(f"- Std Dev: {data_unit}{current_std:,.2f}")
        st.write(f"- Std Error: {data_unit}{current_se:,.2f}")
        st.write(f"- 95% CI Width: ±{data_unit}{1.96*current_se:,.2f}")

    with col2:
        st.markdown("**Target Precision:**")

        precision_type = st.radio(
            "Specify target as:",
            ["Absolute Error", "Relative Error (%)"],
            key="convergence_precision_type"
        )

        if precision_type == "Absolute Error":
            target_se = st.number_input(
                f"Target Standard Error ({data_unit})",
                min_value=0.01,
                max_value=float(current_se * 10),
                value=float(current_se / 2),
                step=0.1,
                key="convergence_target_se"
            )
        else:
            target_rel_error = st.number_input(
                "Target Relative Error (%)",
                min_value=0.1,
                max_value=20.0,
                value=1.0,
                step=0.1,
                key="convergence_target_rel"
            )
            target_se = abs(current_mean) * target_rel_error / 100

    # Calculate required paths
    required_paths = int(np.ceil((current_std / target_se) ** 2))
    improvement_factor = n_paths / required_paths if required_paths > 0 else 0

    st.markdown("---")
    st.markdown("**Required Paths:**")

    if required_paths <= n_paths:
        st.success(f"Current simulation ({n_paths:,} paths) already meets target!")
        st.write(f"You could reduce to {required_paths:,} paths and still achieve this precision.")
    else:
        st.warning(f"Need **{required_paths:,}** paths ({required_paths/n_paths:.1f}x current)")

        # Cost implication
        time_multiplier = required_paths / n_paths
        st.write(f"- This would take approximately {time_multiplier:.1f}x longer to simulate")
        st.write(f"- Consider variance reduction techniques instead")

    # Comparison table
    st.markdown("#### Precision vs Sample Size")

    comparison_n = [100, 500, 1000, 5000, 10000, 50000, 100000]
    comparison_n = [n for n in comparison_n if n <= max(100000, n_paths * 10)]

    table_data = []
    for n in comparison_n:
        se = current_std / np.sqrt(n)
        ci_width = 1.96 * se
        rel_error = se / abs(current_mean) * 100 if current_mean != 0 else np.inf
        table_data.append({
            "Paths": f"{n:,}",
            f"Std Error ({data_unit})": f"{se:,.2f}",
            f"95% CI Width ({data_unit})": f"±{ci_width:,.2f}",
            "Relative Error": f"{rel_error:.2f}%"
        })

    import pandas as pd
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Educational note
    st.markdown("""
    <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <h5 style="color: #92400e; margin-top: 0;">Practical Guidance</h5>
        <ul style="margin-bottom: 0;">
            <li><strong>Pricing</strong>: 1-2% relative error usually sufficient → 5K-10K paths</li>
            <li><strong>Risk metrics (VaR, CVaR)</strong>: Need more paths for tail estimation → 50K-100K</li>
            <li><strong>Greeks via bump-and-revalue</strong>: Need very low noise → 100K+ paths</li>
            <li><strong>Research/illustration</strong>: 1K-5K paths often fine for qualitative insights</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
