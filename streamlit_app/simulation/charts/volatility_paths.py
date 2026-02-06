"""
Volatility path visualization charts for Monte Carlo Simulation Explorer.

Provides interactive visualizations for GARCH-family volatility simulations.
Consolidated into 3 focused pedagogical tabs:
1. Volatility Dynamics - paths + variance analysis combined
2. GARCH Effects - leverage effect + news impact combined
3. Theory vs Simulation - unchanged
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any

from config.constants import (
    CHART_HEIGHT_STANDARD,
    CHART_HEIGHT_LARGE,
    PATH_COLORS,
    MODEL_COLORS,
    VOLATILITY_MODELS
)
from config.styles import render_stats_row


def render_volatility_paths_tab(
    simulation_result,
    params: Dict[str, Any]
) -> None:
    """
    Render the volatility paths visualization tab.

    Args:
        simulation_result: VolatilitySimulationResult object from backend
        params: Dictionary of simulation parameters
    """
    if simulation_result is None:
        st.info("Run a simulation to see volatility paths visualization.")
        return

    # Section header with native help
    st.markdown("### Simulated Volatility Paths")
    st.caption(
        "GARCH-family models capture volatility clustering and leverage effects. "
        "High volatility tends to follow high volatility, and negative returns increase volatility."
    )

    # Create 3 consolidated tabs
    tab1, tab2, tab3 = st.tabs([
        "Volatility Dynamics",
        "GARCH Effects",
        "Theory vs Simulation"
    ])

    with tab1:
        _render_volatility_dynamics(simulation_result, params)

    with tab2:
        _render_garch_effects(simulation_result, params)

    with tab3:
        _render_theory_comparison(simulation_result, params)


def _render_volatility_dynamics(result, params: Dict[str, Any]) -> None:
    """
    Consolidated volatility dynamics tab combining:
    - Sample volatility paths with statistics
    - Variance analysis including variance ratio tests
    """
    vol_paths = result.volatility_paths * 100  # Convert to percentage
    variance_paths = result.variance_paths
    time_grid = result.time_grid
    n_paths = vol_paths.shape[0]
    max_display = min(params.get('max_display_paths', 50), n_paths)

    # === Part 1: Volatility Paths ===
    st.markdown("#### Sample Volatility Paths")

    fig_paths = go.Figure()

    # Sample paths
    for i in range(max_display):
        fig_paths.add_trace(go.Scatter(
            x=time_grid,
            y=vol_paths[i],
            mode='lines',
            line=dict(color='rgba(13, 148, 136, 0.15)', width=0.8),
            hoverinfo='skip',
            showlegend=False
        ))

    # Mean volatility
    if params.get('show_mean', True):
        mean_vol = result.mean_volatility_path * 100
        fig_paths.add_trace(go.Scatter(
            x=time_grid,
            y=mean_vol,
            mode='lines',
            name='Mean Volatility',
            line=dict(color='#0d9488', width=2.5)
        ))

    # Percentile bands
    if params.get('show_percentiles', True):
        percentiles = result.percentile_volatility_paths([5, 50, 95])
        p5, p50, p95 = percentiles[0] * 100, percentiles[1] * 100, percentiles[2] * 100

        fig_paths.add_trace(go.Scatter(
            x=np.concatenate([time_grid, time_grid[::-1]]),
            y=np.concatenate([p95, p5[::-1]]),
            fill='toself',
            fillcolor='rgba(13, 148, 136, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name='5-95% Range',
            hoverinfo='skip'
        ))

        fig_paths.add_trace(go.Scatter(
            x=time_grid,
            y=p50,
            mode='lines',
            name='Median',
            line=dict(color='#0f766e', width=2, dash='dash')
        ))

    # Initial volatility
    initial_vol = params['volatility'] * 100
    fig_paths.add_hline(
        y=initial_vol,
        line_dash="dot",
        line_color='#dc2626',
        annotation_text=f"sigma_0 = {initial_vol:.1f}%"
    )

    # Long-run volatility (if applicable)
    if hasattr(result, 'long_run_variance') and not np.isnan(result.long_run_variance):
        long_run_vol = np.sqrt(result.long_run_variance) * 100
        fig_paths.add_hline(
            y=long_run_vol,
            line_dash="dashdot",
            line_color='#059669',
            annotation_text=f"Long-run = {long_run_vol:.1f}%"
        )

    model_name = VOLATILITY_MODELS.get(params.get('vol_model', 'garch'), 'GARCH')

    fig_paths.update_layout(
        title=dict(
            text=f"{model_name} Volatility Paths - {result.num_paths:,} Simulations",
            font=dict(size=16)
        ),
        xaxis_title="Time (Normalized)",
        yaxis_title="Volatility (%)",
        height=CHART_HEIGHT_STANDARD,
        hovermode='x unified',
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )

    st.plotly_chart(fig_paths, use_container_width=True)

    # Statistics with styled cards
    terminal_vol = result.terminal_volatility * 100

    vol_stats = [
        ("Mean Terminal Vol", f"{terminal_vol.mean():.2f}%", "Average ending volatility"),
        ("Std Dev", f"{terminal_vol.std():.2f}%", "Dispersion measure"),
        ("Min Terminal", f"{terminal_vol.min():.2f}%", "Lowest ending vol"),
        ("Max Terminal", f"{terminal_vol.max():.2f}%", "Highest ending vol"),
    ]
    render_stats_row(vol_stats, ["teal", "amber", "slate", "slate"])

    # === Part 2: Variance Analysis ===
    st.markdown("---")
    st.markdown("#### Variance Analysis")
    st.caption("Understanding variance dynamics, clustering, and autocorrelation.")

    fig_var = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Mean Variance Over Time",
            "Variance Distribution at T",
            "Variance Autocorrelation",
            "Variance Clustering Sample"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    # Mean variance path
    mean_var = np.mean(variance_paths, axis=0)
    std_var = np.std(variance_paths, axis=0)

    fig_var.add_trace(
        go.Scatter(
            x=time_grid,
            y=mean_var,
            mode='lines',
            name='Mean',
            line=dict(color='#d97706', width=2)
        ),
        row=1, col=1
    )

    # +/- 1 std band
    fig_var.add_trace(
        go.Scatter(
            x=np.concatenate([time_grid, time_grid[::-1]]),
            y=np.concatenate([mean_var + std_var, (mean_var - std_var)[::-1]]),
            fill='toself',
            fillcolor='rgba(217, 119, 6, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name='+/- 1 Std',
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    # Terminal variance histogram
    terminal_var = result.terminal_variance
    fig_var.add_trace(
        go.Histogram(
            x=terminal_var,
            nbinsx=50,
            name='Terminal Var',
            marker_color='#d97706',
            opacity=0.7
        ),
        row=1, col=2
    )

    # Autocorrelation of variance changes
    var_changes = np.diff(variance_paths, axis=1)
    mean_var_change = np.mean(var_changes, axis=0)

    # Sample ACF
    acf_lags = min(50, len(mean_var_change) - 1)
    acf = np.correlate(mean_var_change - mean_var_change.mean(),
                       mean_var_change - mean_var_change.mean(), mode='full')
    acf = acf[len(acf)//2:len(acf)//2 + acf_lags]
    if acf[0] > 0:
        acf = acf / acf[0]

    fig_var.add_trace(
        go.Bar(
            x=list(range(acf_lags)),
            y=acf,
            name='ACF',
            marker_color='#0d9488'
        ),
        row=2, col=1
    )

    # Variance clustering - sample path
    sample_path = variance_paths[0]
    fig_var.add_trace(
        go.Scatter(
            x=time_grid,
            y=sample_path,
            mode='lines',
            name='Sample Path',
            line=dict(color='#1a365d', width=1)
        ),
        row=2, col=2
    )

    fig_var.update_layout(
        height=CHART_HEIGHT_LARGE,
        showlegend=False
    )

    st.plotly_chart(fig_var, use_container_width=True)

    # Educational insight
    st.markdown("""
    <div style="background: #e8f4f8; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <h5 style="color: #1e3a5f; margin-top: 0;">Key Insight: Volatility Clustering</h5>
        <p style="margin-bottom: 0;">
            The autocorrelation plot shows <strong>persistent volatility</strong>: high variance today
            predicts high variance tomorrow. This is captured by the β parameter in GARCH models.
            A higher persistence (α+β close to 1) means shocks to volatility die out slowly.
        </p>
    </div>
    """, unsafe_allow_html=True)


def _render_garch_effects(result, params: Dict[str, Any]) -> None:
    """
    Consolidated GARCH effects tab combining:
    - Leverage effect analysis
    - News impact curve
    """
    # === Part 1: Leverage Effect ===
    st.markdown("#### Leverage Effect Analysis")
    st.caption("How returns affect subsequent volatility changes.")

    returns = result.return_paths[:, :-1].flatten()
    vol_paths = np.sqrt(result.variance_paths)
    vol_changes = (vol_paths[:, 2:] - vol_paths[:, 1:-1]).flatten()

    # Bin returns and compute mean volatility change
    n_bins = 20
    bins = np.percentile(returns, np.linspace(0, 100, n_bins + 1))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []

    for i in range(len(bins) - 1):
        mask = (returns >= bins[i]) & (returns < bins[i + 1])
        if mask.sum() > 0:
            bin_means.append(vol_changes[mask].mean())
        else:
            bin_means.append(np.nan)

    fig_leverage = go.Figure()

    fig_leverage.add_trace(go.Scatter(
        x=bin_centers * 100,
        y=np.array(bin_means) * 100,
        mode='markers+lines',
        marker=dict(size=8, color='#0d9488'),
        line=dict(color='#0d9488', width=2),
        name='Mean Vol Change'
    ))

    fig_leverage.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_leverage.add_vline(x=0, line_dash="dash", line_color="gray")

    model_name = VOLATILITY_MODELS.get(params.get('vol_model', 'garch'), 'GARCH')

    fig_leverage.update_layout(
        title=f"Leverage Effect - {model_name}",
        xaxis_title="Return (%)",
        yaxis_title="Next Period Volatility Change (%)",
        height=CHART_HEIGHT_STANDARD - 50
    )

    st.plotly_chart(fig_leverage, use_container_width=True)

    # Correlation statistic
    valid_mask = ~np.isnan(vol_changes) & ~np.isnan(returns)
    if valid_mask.sum() > 100:
        correlation = np.corrcoef(returns[valid_mask], vol_changes[valid_mask])[0, 1]
        corr_variant = "green" if correlation < -0.1 else "amber" if correlation < 0 else "red"
        leverage_stats = [
            ("Return-Vol Correlation", f"{correlation:.4f}", "Negative = leverage effect"),
        ]
        render_stats_row(leverage_stats, [corr_variant])

    # === Part 2: News Impact Curve ===
    st.markdown("---")
    st.markdown("#### News Impact Curve")
    st.caption("How shocks of different magnitudes affect next-period volatility.")

    model = params.get('vol_model', 'garch')
    sigma = params.get('volatility', 0.20)
    alpha = params.get('garch_alpha', 0.05)
    beta = params.get('garch_beta', 0.90)

    # Range of shocks
    epsilon_range = np.linspace(-0.15, 0.15, 200)
    sigma_sq = sigma ** 2

    # Compute omega (to maintain long-run variance)
    if model == 'ngarch':
        theta = params.get('ngarch_theta', 0.5)
        persistence = alpha * (1 + theta ** 2) + beta
        omega = sigma_sq * (1 - persistence) if persistence < 1 else 0.00001
    elif model == 'gjr_garch':
        gamma = params.get('gjr_gamma', 0.05)
        persistence = alpha + beta + 0.5 * gamma
        omega = sigma_sq * (1 - persistence) if persistence < 1 else 0.00001
    else:  # Standard GARCH
        persistence = alpha + beta
        omega = sigma_sq * (1 - persistence) if persistence < 1 else 0.00001

    fig_nic = go.Figure()

    # GARCH (symmetric)
    garch_response = omega + alpha * epsilon_range**2 + beta * sigma_sq
    fig_nic.add_trace(go.Scatter(
        x=epsilon_range * 100,
        y=np.sqrt(garch_response) * 100,
        mode='lines',
        name='GARCH(1,1) - Symmetric',
        line=dict(color='#1a365d', width=2)
    ))

    if model == 'ngarch':
        theta = params.get('ngarch_theta', 0.5)
        ngarch_response = omega + alpha * (epsilon_range - theta * sigma)**2 + beta * sigma_sq
        fig_nic.add_trace(go.Scatter(
            x=epsilon_range * 100,
            y=np.sqrt(ngarch_response) * 100,
            mode='lines',
            name=f'NGARCH (theta={theta})',
            line=dict(color='#dc2626', width=2)
        ))

    elif model == 'gjr_garch':
        gamma = params.get('gjr_gamma', 0.05)
        indicator = (epsilon_range < 0).astype(float)
        gjr_response = omega + (alpha + gamma * indicator) * epsilon_range**2 + beta * sigma_sq
        fig_nic.add_trace(go.Scatter(
            x=epsilon_range * 100,
            y=np.sqrt(gjr_response) * 100,
            mode='lines',
            name=f'GJR-GARCH (gamma={gamma})',
            line=dict(color='#059669', width=2)
        ))

    fig_nic.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig_nic.update_layout(
        title="News Impact Curve",
        xaxis_title="Return Shock epsilon (%)",
        yaxis_title="Next Period Volatility (%)",
        height=CHART_HEIGHT_STANDARD - 50,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01
        )
    )

    st.plotly_chart(fig_nic, use_container_width=True)

    # Combined educational insight
    st.markdown("""
    <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <h5 style="color: #92400e; margin-top: 0;">Key Insight: Asymmetric Volatility Response</h5>
        <ul style="margin-bottom: 0;">
            <li><strong>GARCH(1,1)</strong>: Symmetric - positive and negative shocks have equal impact</li>
            <li><strong>NGARCH/GJR-GARCH</strong>: Asymmetric - negative shocks increase volatility more</li>
            <li>This asymmetry, called the <strong>leverage effect</strong>, is empirically observed in equity markets</li>
            <li>Implication: Put options tend to be more expensive than calls at same distance from ATM</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def _render_theory_comparison(result, params: Dict[str, Any]) -> None:
    """Compare simulated volatility with theoretical GARCH predictions."""
    model = params.get('vol_model', 'garch')
    model_name = VOLATILITY_MODELS.get(model, 'GARCH')

    st.markdown(f"#### {model_name} - Simulation vs Theory")
    st.caption(
        "Compare Monte Carlo simulation results against theoretical GARCH predictions. "
        "Deviations indicate finite-sample effects or numerical issues."
    )

    # Extract GARCH parameters
    alpha = params.get('garch_alpha', 0.05)
    beta = params.get('garch_beta', 0.90)
    sigma_0 = params.get('volatility', 0.20)

    # Model-specific persistence calculation
    if model == 'ngarch':
        theta = params.get('ngarch_theta', 0.5)
        persistence = alpha * (1 + theta ** 2) + beta
    elif model == 'gjr_garch':
        gamma = params.get('gjr_gamma', 0.05)
        persistence = alpha + beta + 0.5 * gamma
    else:  # Standard GARCH
        persistence = alpha + beta

    # Theoretical long-run variance
    if persistence < 1:
        if model == 'ngarch':
            omega = sigma_0 ** 2 * (1 - persistence)
            long_run_var_theory = omega / (1 - persistence)
        elif model == 'gjr_garch':
            omega = sigma_0 ** 2 * (1 - persistence)
            long_run_var_theory = omega / (1 - alpha - beta - 0.5 * gamma)
        else:
            omega = sigma_0 ** 2 * (1 - alpha - beta)
            long_run_var_theory = omega / (1 - alpha - beta)
    else:
        long_run_var_theory = np.nan

    # Get simulated values
    variance_paths = result.variance_paths
    terminal_var_sim = result.terminal_variance
    mean_var_sim = np.mean(variance_paths[:, -1])

    # Long-run variance from simulation
    long_run_var_sim = (
        result.long_run_variance
        if hasattr(result, 'long_run_variance') and result.long_run_variance
        else np.mean(variance_paths)
    )

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Long-run Variance Convergence",
            "Terminal Variance Distribution",
            "Variance Autocorrelation Decay",
            "Mean Reversion Check"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    time_grid = result.time_grid

    # 1. Long-run variance convergence
    mean_var_path = np.mean(variance_paths, axis=0)

    fig.add_trace(
        go.Scatter(
            x=time_grid,
            y=mean_var_path,
            mode='lines',
            name='Simulated Mean',
            line=dict(color='#0d9488', width=2)
        ),
        row=1, col=1
    )

    if not np.isnan(long_run_var_theory):
        fig.add_hline(
            y=long_run_var_theory,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=f"Theory: {long_run_var_theory:.6f}",
            annotation_position="right",
            row=1, col=1
        )

    # 2. Terminal variance distribution
    fig.add_trace(
        go.Histogram(
            x=terminal_var_sim,
            nbinsx=50,
            name='Simulated',
            marker_color='#0d9488',
            opacity=0.7
        ),
        row=1, col=2
    )

    if not np.isnan(long_run_var_theory):
        fig.add_vline(
            x=long_run_var_theory,
            line_dash="dash",
            line_color="#dc2626",
            row=1, col=2
        )

    # 3. Theoretical ACF of variance for GARCH(1,1): rho(k) = (alpha + beta)^k
    max_lag = min(50, len(time_grid) - 1)
    lags = np.arange(max_lag)
    theoretical_acf = persistence ** lags

    # Empirical ACF of variance
    var_series = variance_paths[0]  # Sample path
    var_centered = var_series - var_series.mean()
    autocorr = np.correlate(var_centered, var_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:len(autocorr)//2 + max_lag]
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]

    fig.add_trace(
        go.Scatter(
            x=lags,
            y=theoretical_acf,
            mode='lines',
            name='Theoretical ACF',
            line=dict(color='#dc2626', width=2, dash='dash')
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=lags,
            y=autocorr,
            name='Empirical ACF',
            marker_color='#0d9488',
            opacity=0.5
        ),
        row=2, col=1
    )

    # 4. Mean reversion: variance should revert to long-run
    if not np.isnan(long_run_var_theory):
        above_mask = variance_paths[:, 0] > long_run_var_theory
        below_mask = ~above_mask

        if above_mask.sum() > 0:
            above_mean = np.mean(variance_paths[above_mask], axis=0)
            fig.add_trace(
                go.Scatter(
                    x=time_grid,
                    y=above_mean,
                    mode='lines',
                    name='Started Above',
                    line=dict(color='#f59e0b', width=2)
                ),
                row=2, col=2
            )

        if below_mask.sum() > 0:
            below_mean = np.mean(variance_paths[below_mask], axis=0)
            fig.add_trace(
                go.Scatter(
                    x=time_grid,
                    y=below_mean,
                    mode='lines',
                    name='Started Below',
                    line=dict(color='#3b82f6', width=2)
                ),
                row=2, col=2
            )

        fig.add_hline(
            y=long_run_var_theory,
            line_dash="dash",
            line_color="#dc2626",
            row=2, col=2
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=mean_var_path,
                mode='lines',
                name='Mean Variance',
                line=dict(color='#0d9488', width=2)
            ),
            row=2, col=2
        )

    fig.update_layout(
        height=CHART_HEIGHT_LARGE + 50,
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="right", x=0.99,
            font=dict(size=9)
        )
    )

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Variance", row=1, col=2)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_yaxes(title_text="Variance", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Autocorrelation", row=2, col=1)
    fig.update_yaxes(title_text="Variance", row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics comparison
    st.markdown("#### Theory vs Simulation Summary")

    # Calculate error metrics
    if not np.isnan(long_run_var_theory):
        lr_var_error = (mean_var_sim - long_run_var_theory) / long_run_var_theory * 100
        lr_variant = "green" if abs(lr_var_error) < 5 else "amber" if abs(lr_var_error) < 15 else "red"
    else:
        lr_var_error = np.nan
        lr_variant = "slate"

    pers_variant = "green" if persistence < 0.99 else "amber" if persistence < 1 else "red"

    theory_stats = [
        ("Persistence (alpha+beta)", f"{persistence:.4f}", "< 1 for stationarity"),
        ("Long-run Vol (Theory)",
         f"{np.sqrt(long_run_var_theory)*100:.2f}%" if not np.isnan(long_run_var_theory) else "Non-stationary",
         "Theoretical sigma_inf"),
        ("Long-run Vol (Sim)", f"{np.sqrt(mean_var_sim)*100:.2f}%", "Simulated mean"),
        ("Deviation",
         f"{lr_var_error:+.2f}%" if not np.isnan(lr_var_error) else "N/A",
         "Simulation error"),
    ]
    render_stats_row(theory_stats, [pers_variant, "blue", "teal", lr_variant])

    # Interpretation
    st.markdown("---")

    if persistence >= 1:
        st.warning("Model is non-stationary (persistence >= 1). Variance has no long-run mean.")
    elif persistence > 0.99:
        st.info("High persistence - variance shocks are very persistent. Long convergence time expected.")
    else:
        if not np.isnan(lr_var_error) and abs(lr_var_error) < 5:
            st.success("Simulation closely matches theoretical predictions.")
        elif not np.isnan(lr_var_error) and abs(lr_var_error) < 15:
            st.info("Moderate deviation from theory - increase sample size or time horizon for better convergence.")
        elif not np.isnan(lr_var_error):
            st.warning("Significant deviation from theory - check parameters or increase simulation length.")

    # Educational insight box
    st.markdown("""
    <div style="background: #e8f4f8; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <h5 style="color: #1e3a5f; margin-top: 0;">Key Insight: Stationarity and Convergence</h5>
        <p style="margin-bottom: 0;">
            For GARCH models to have a well-defined long-run variance, the <strong>persistence must be less than 1</strong>.
            High persistence (close to 1) means the model is nearly "integrated" (IGARCH),
            and shocks to variance persist for a very long time. This explains why equity volatility
            tends to remain elevated after market crashes.
        </p>
    </div>
    """, unsafe_allow_html=True)
