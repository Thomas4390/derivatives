"""
Distribution visualization charts for Monte Carlo Simulation Explorer.

Provides interactive visualizations for terminal distributions and density analysis.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, Optional
from scipy import stats

from config.constants import (
    CHART_HEIGHT_STANDARD,
    CHART_HEIGHT_LARGE,
    PATH_COLORS,
    MODEL_COLORS,
    PRICE_MODELS,
    VOLATILITY_MODELS
)


def render_distributions_tab(
    simulation_result,
    params: Dict[str, Any],
    result_type: str = "price"
) -> None:
    """
    Render the distributions visualization tab.

    Args:
        simulation_result: Simulation result object from backend
        params: Dictionary of simulation parameters
        result_type: "price" for price simulations, "volatility" for vol simulations
    """
    if simulation_result is None:
        st.info("Run a simulation to see distribution analysis.")
        return

    if result_type == "price":
        st.markdown("### Terminal Price Distribution")
        terminal_values = simulation_result.terminal_values
        value_name = "Price"
        unit = "$"
    else:
        st.markdown("### Terminal Volatility Distribution")
        terminal_values = simulation_result.terminal_volatility * 100
        value_name = "Volatility"
        unit = "%"

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "Distribution Analysis",
        "Comparison with Theory",
        "Risk Metrics"
    ])

    with tab1:
        _render_distribution_analysis(terminal_values, value_name, unit, params)

    with tab2:
        if result_type == "price":
            _render_theoretical_comparison(terminal_values, params)
        else:
            _render_vol_theoretical_comparison(terminal_values, params)

    with tab3:
        _render_risk_metrics(terminal_values, value_name, unit, params, result_type)


def _render_distribution_analysis(
    values: np.ndarray,
    name: str,
    unit: str,
    params: Dict[str, Any]
) -> None:
    """Render comprehensive distribution analysis."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Histogram with KDE",
            "Empirical CDF",
            "Box Plot",
            "Q-Q Plot vs Normal"
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ],
        vertical_spacing=0.12
    )

    # 1. Histogram with KDE
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=75,
            name='Histogram',
            marker_color='rgba(13, 148, 136, 0.6)',
            histnorm='probability density'
        ),
        row=1, col=1
    )

    # KDE overlay
    kde_x = np.linspace(values.min(), values.max(), 200)
    kde = stats.gaussian_kde(values)
    fig.add_trace(
        go.Scatter(
            x=kde_x,
            y=kde(kde_x),
            mode='lines',
            name='KDE',
            line=dict(color='#1a365d', width=2)
        ),
        row=1, col=1
    )

    # 2. Empirical CDF
    sorted_vals = np.sort(values)
    ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    fig.add_trace(
        go.Scatter(
            x=sorted_vals,
            y=ecdf,
            mode='lines',
            name='ECDF',
            line=dict(color='#0d9488', width=2)
        ),
        row=1, col=2
    )

    # Add percentile markers
    percentiles = [5, 25, 50, 75, 95]
    for p in percentiles:
        val = np.percentile(values, p)
        fig.add_trace(
            go.Scatter(
                x=[val, val],
                y=[0, p/100],
                mode='lines',
                line=dict(color='gray', dash='dot', width=1),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=2
        )

    # 3. Box plot
    fig.add_trace(
        go.Box(
            y=values,
            name=name,
            boxmean='sd',
            marker_color='#0d9488',
            fillcolor='rgba(13, 148, 136, 0.3)'
        ),
        row=2, col=1
    )

    # 4. Q-Q plot
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.001, 0.999, n))

    # Standardize values
    std_vals = (sorted_vals - values.mean()) / values.std()

    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=std_vals,
            mode='markers',
            name='Q-Q',
            marker=dict(color='#0d9488', size=3)
        ),
        row=2, col=2
    )

    # Reference line
    fig.add_trace(
        go.Scatter(
            x=[-4, 4],
            y=[-4, 4],
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

    fig.update_xaxes(title_text=f"{name} ({unit})", row=1, col=1)
    fig.update_xaxes(title_text=f"{name} ({unit})", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Probability", row=1, col=2)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
    fig.update_yaxes(title_text="Sample Quantiles (Standardized)", row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # Distribution statistics table
    st.markdown("#### Distribution Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean", f"{values.mean():.4f}{unit}")
        st.metric("Std Dev", f"{values.std():.4f}{unit}")

    with col2:
        st.metric("Median", f"{np.median(values):.4f}{unit}")
        st.metric("IQR", f"{np.percentile(values, 75) - np.percentile(values, 25):.4f}{unit}")

    with col3:
        st.metric("Skewness", f"{stats.skew(values):.4f}")
        st.metric("Kurtosis", f"{stats.kurtosis(values):.4f}")

    with col4:
        st.metric("Min", f"{values.min():.4f}{unit}")
        st.metric("Max", f"{values.max():.4f}{unit}")


def _render_theoretical_comparison(values: np.ndarray, params: Dict[str, Any]) -> None:
    """Compare simulated distribution with theoretical (for price models)."""
    model = params.get('price_model', 'gbm')
    S0 = params['spot_price']
    r = params['risk_free_rate']
    sigma = params['volatility']
    T = params['time_horizon']

    fig = go.Figure()

    # Histogram of simulated values
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=75,
        name='Simulated',
        marker_color='rgba(13, 148, 136, 0.6)',
        histnorm='probability density'
    ))

    # Theoretical distribution (for GBM)
    if model == 'gbm':
        # Log-normal distribution
        mu = np.log(S0) + (r - 0.5 * sigma**2) * T
        sigma_total = sigma * np.sqrt(T)

        x_range = np.linspace(values.min(), values.max(), 200)
        lognorm_pdf = stats.lognorm.pdf(x_range, s=sigma_total, scale=np.exp(mu))

        fig.add_trace(go.Scatter(
            x=x_range,
            y=lognorm_pdf,
            mode='lines',
            name='Theoretical (Log-Normal)',
            line=dict(color='#dc2626', width=2)
        ))

        # Theoretical mean and std
        theoretical_mean = S0 * np.exp(r * T)
        theoretical_std = theoretical_mean * np.sqrt(np.exp(sigma**2 * T) - 1)

        st.markdown("#### GBM Theoretical vs Simulated")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Mean**")
            st.write(f"Theoretical: ${theoretical_mean:.2f}")
            st.write(f"Simulated: ${values.mean():.2f}")
            error = abs(values.mean() - theoretical_mean) / theoretical_mean * 100
            st.write(f"Error: {error:.2f}%")

        with col2:
            st.markdown("**Std Dev**")
            st.write(f"Theoretical: ${theoretical_std:.2f}")
            st.write(f"Simulated: ${values.std():.2f}")
            error = abs(values.std() - theoretical_std) / theoretical_std * 100
            st.write(f"Error: {error:.2f}%")

        with col3:
            st.markdown("**Monte Carlo Error**")
            mc_error = values.std() / np.sqrt(len(values))
            st.write(f"Standard Error: ${mc_error:.4f}")
            st.write(f"95% CI: +/- ${1.96 * mc_error:.4f}")

    else:
        st.info(f"Theoretical comparison not available for {PRICE_MODELS.get(model, model)}. "
                "Closed-form solutions don't exist for stochastic volatility models.")

    fig.update_layout(
        title="Simulated vs Theoretical Distribution",
        xaxis_title="Terminal Price ($)",
        yaxis_title="Probability Density",
        height=CHART_HEIGHT_STANDARD
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_vol_theoretical_comparison(values: np.ndarray, params: Dict[str, Any]) -> None:
    """Compare simulated volatility distribution with steady-state."""
    model = params.get('vol_model', 'garch')

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=75,
        name='Simulated Terminal Vol',
        marker_color='rgba(13, 148, 136, 0.6)',
        histnorm='probability density'
    ))

    st.info(f"Terminal volatility distribution for {VOLATILITY_MODELS.get(model, model)}. "
            "The distribution approaches the ergodic distribution as simulation length increases.")

    fig.update_layout(
        title="Terminal Volatility Distribution",
        xaxis_title="Volatility (%)",
        yaxis_title="Probability Density",
        height=CHART_HEIGHT_STANDARD
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_risk_metrics(
    values: np.ndarray,
    name: str,
    unit: str,
    params: Dict[str, Any],
    result_type: str
) -> None:
    """Render risk metrics (VaR, CVaR, etc.)."""
    st.markdown("#### Risk Metrics")

    if result_type == "price":
        S0 = params['spot_price']
        # Returns for risk calculations
        returns = (values - S0) / S0
        log_returns = np.log(values / S0)
    else:
        initial_vol = params['volatility'] * 100
        returns = (values - initial_vol) / initial_vol
        log_returns = returns

    # VaR at different confidence levels
    confidence_levels = [0.90, 0.95, 0.99]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Value at Risk (VaR)")
        for cl in confidence_levels:
            var = np.percentile(returns, (1 - cl) * 100)
            if result_type == "price":
                var_dollar = var * S0
                st.write(f"VaR {int(cl*100)}%: {var*100:.2f}% (${var_dollar:.2f})")
            else:
                st.write(f"VaR {int(cl*100)}%: {var*100:.2f}%")

    with col2:
        st.markdown("##### Conditional VaR (CVaR/ES)")
        for cl in confidence_levels:
            var = np.percentile(returns, (1 - cl) * 100)
            cvar = returns[returns <= var].mean()
            if result_type == "price":
                cvar_dollar = cvar * S0
                st.write(f"CVaR {int(cl*100)}%: {cvar*100:.2f}% (${cvar_dollar:.2f})")
            else:
                st.write(f"CVaR {int(cl*100)}%: {cvar*100:.2f}%")

    # Visualization of VaR/CVaR
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=75,
        name='Returns',
        marker_color='rgba(13, 148, 136, 0.6)',
        histnorm='probability density'
    ))

    # Add VaR lines
    colors = ['#fbbf24', '#f97316', '#dc2626']
    for cl, color in zip(confidence_levels, colors):
        var = np.percentile(returns, (1 - cl) * 100) * 100
        fig.add_vline(
            x=var,
            line_dash="dash",
            line_color=color,
            annotation_text=f"VaR {int(cl*100)}%: {var:.1f}%"
        )

    fig.update_layout(
        title="Return Distribution with VaR Levels",
        xaxis_title="Return (%)",
        yaxis_title="Probability Density",
        height=CHART_HEIGHT_STANDARD
    )

    st.plotly_chart(fig, use_container_width=True)

    # Additional risk statistics
    st.markdown("#### Additional Risk Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Probability of loss
        prob_loss = (returns < 0).mean() * 100
        st.metric("Probability of Loss", f"{prob_loss:.1f}%")

    with col2:
        # Maximum drawdown (simplified)
        if result_type == "price":
            max_dd = (values.max() - values.min()) / values.max() * 100
            st.metric("Range/Max", f"{max_dd:.1f}%")

    with col3:
        # Sharpe-like ratio
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 / params['num_steps'])
            st.metric("Return/Risk Ratio", f"{sharpe:.3f}")
