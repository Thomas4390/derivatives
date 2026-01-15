"""
Interactive single path analysis for Monte Carlo Simulation Explorer.

Provides real-time parameter adjustment using Plotly sliders for better responsiveness.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, Optional
from numba import njit

from config.constants import (
    CHART_HEIGHT_LARGE,
    PRICE_MODELS,
    VOLATILITY_MODELS
)


# =============================================================================
# FAST SINGLE PATH SIMULATION FUNCTIONS (No Numba overhead for single path)
# =============================================================================

def simulate_single_gbm_path(
    s0: float, r: float, sigma: float, t: float, n_steps: int, seed: int
) -> tuple:
    """Simulate a single GBM path."""
    np.random.seed(seed)
    dt = t / n_steps
    time_grid = np.linspace(0, t, n_steps + 1)

    # Generate path
    dW = np.random.standard_normal(n_steps) * np.sqrt(dt)
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * dW

    path = np.zeros(n_steps + 1)
    path[0] = s0
    path[1:] = s0 * np.exp(np.cumsum(log_returns))

    return time_grid, path, np.concatenate([[0], log_returns])


def simulate_single_heston_path(
    s0: float, v0: float, r: float, kappa: float, theta: float,
    xi: float, rho: float, t: float, n_steps: int, seed: int
) -> tuple:
    """Simulate a single Heston path with variance."""
    np.random.seed(seed)
    dt = t / n_steps
    time_grid = np.linspace(0, t, n_steps + 1)

    # Correlated Brownians
    z1 = np.random.standard_normal(n_steps)
    z2 = np.random.standard_normal(n_steps)
    dW_s = z1 * np.sqrt(dt)
    dW_v = (rho * z1 + np.sqrt(1 - rho**2) * z2) * np.sqrt(dt)

    path = np.zeros(n_steps + 1)
    variance = np.zeros(n_steps + 1)
    returns = np.zeros(n_steps + 1)

    path[0] = s0
    variance[0] = v0

    for i in range(n_steps):
        v_curr = max(variance[i], 0)
        sqrt_v = np.sqrt(v_curr)

        # Milstein scheme for variance
        variance[i + 1] = (v_curr + kappa * (theta - v_curr) * dt +
                          xi * sqrt_v * dW_v[i] +
                          0.25 * xi**2 * (dW_v[i]**2 - dt))
        variance[i + 1] = max(variance[i + 1], 0)

        # Log-Euler for price
        log_return = (r - 0.5 * v_curr) * dt + sqrt_v * dW_s[i]
        returns[i + 1] = log_return
        path[i + 1] = path[i] * np.exp(log_return)

    return time_grid, path, variance, returns


def simulate_single_garch_path(
    sigma0: float, omega: float, alpha: float, beta: float,
    n_steps: int, seed: int
) -> tuple:
    """Simulate a single GARCH(1,1) variance path."""
    np.random.seed(seed)

    z = np.random.standard_normal(n_steps)

    variance = np.zeros(n_steps + 1)
    volatility = np.zeros(n_steps + 1)
    returns = np.zeros(n_steps + 1)

    variance[0] = sigma0 ** 2
    volatility[0] = sigma0

    for i in range(n_steps):
        epsilon = np.sqrt(variance[i]) * z[i]
        returns[i + 1] = epsilon
        variance[i + 1] = omega + alpha * epsilon**2 + beta * variance[i]
        volatility[i + 1] = np.sqrt(variance[i + 1])

    time_grid = np.arange(n_steps + 1)
    return time_grid, variance, volatility, returns


def simulate_single_ngarch_path(
    sigma0: float, omega: float, alpha: float, beta: float, theta: float,
    n_steps: int, seed: int
) -> tuple:
    """Simulate a single NGARCH variance path."""
    np.random.seed(seed)

    z = np.random.standard_normal(n_steps)

    variance = np.zeros(n_steps + 1)
    volatility = np.zeros(n_steps + 1)
    returns = np.zeros(n_steps + 1)

    variance[0] = sigma0 ** 2
    volatility[0] = sigma0

    for i in range(n_steps):
        sigma_i = np.sqrt(variance[i])
        epsilon = sigma_i * z[i]
        returns[i + 1] = epsilon
        variance[i + 1] = omega + alpha * (epsilon - theta * sigma_i)**2 + beta * variance[i]
        volatility[i + 1] = np.sqrt(variance[i + 1])

    time_grid = np.arange(n_steps + 1)
    return time_grid, variance, volatility, returns


def simulate_single_gjr_garch_path(
    sigma0: float, omega: float, alpha: float, beta: float, gamma: float,
    n_steps: int, seed: int
) -> tuple:
    """Simulate a single GJR-GARCH variance path."""
    np.random.seed(seed)

    z = np.random.standard_normal(n_steps)

    variance = np.zeros(n_steps + 1)
    volatility = np.zeros(n_steps + 1)
    returns = np.zeros(n_steps + 1)

    variance[0] = sigma0 ** 2
    volatility[0] = sigma0

    for i in range(n_steps):
        epsilon = np.sqrt(variance[i]) * z[i]
        returns[i + 1] = epsilon
        indicator = 1.0 if epsilon < 0 else 0.0
        variance[i + 1] = omega + (alpha + gamma * indicator) * epsilon**2 + beta * variance[i]
        volatility[i + 1] = np.sqrt(variance[i + 1])

    time_grid = np.arange(n_steps + 1)
    return time_grid, variance, volatility, returns


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_rolling_volatility(returns: np.ndarray, window: int = 30) -> np.ndarray:
    """Compute rolling historical volatility."""
    n = len(returns)
    rolling_vol = np.full(n, np.nan)

    for i in range(window, n):
        rolling_vol[i] = np.std(returns[i-window+1:i+1]) * np.sqrt(252)

    return rolling_vol


def compute_unconditional_variance_garch(omega: float, alpha: float, beta: float) -> float:
    """Compute GARCH unconditional variance."""
    persistence = alpha + beta
    if persistence < 1:
        return omega / (1 - persistence)
    return np.nan


def compute_unconditional_variance_ngarch(
    omega: float, alpha: float, beta: float, theta: float
) -> float:
    """Compute NGARCH unconditional variance."""
    persistence = alpha * (1 + theta**2) + beta
    if persistence < 1:
        return omega / (1 - persistence)
    return np.nan


def compute_unconditional_variance_gjr(
    omega: float, alpha: float, beta: float, gamma: float
) -> float:
    """Compute GJR-GARCH unconditional variance."""
    persistence = alpha + beta + 0.5 * gamma
    if persistence < 1:
        return omega / (1 - persistence)
    return np.nan


# =============================================================================
# MAIN RENDERING FUNCTIONS
# =============================================================================

def render_interactive_path_tab(params: Dict[str, Any]) -> None:
    """
    Render the interactive single path analysis tab.

    Args:
        params: Dictionary of simulation parameters
    """
    simulation_mode = params.get('simulation_mode', 'price')

    st.markdown("### Interactive Single Path Analysis")
    st.markdown("*Adjust parameters using the sliders below for real-time visualization*")

    if simulation_mode == 'price':
        _render_interactive_price_path(params)
    else:
        _render_interactive_volatility_path(params)


def _render_interactive_price_path(params: Dict[str, Any]) -> None:
    """Render interactive price path visualization."""
    model = params.get('price_model', 'gbm')

    # Parameter controls in columns
    st.markdown("#### Model Parameters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        s0 = st.slider("Initial Price (S₀)", 50.0, 200.0,
                       float(params.get('spot_price', 100.0)), 1.0,
                       key="interactive_s0")
    with col2:
        sigma = st.slider("Volatility (σ)", 0.05, 0.80,
                         float(params.get('volatility', 0.20)), 0.01,
                         key="interactive_sigma")
    with col3:
        r = st.slider("Risk-free Rate (r)", 0.0, 0.15,
                     float(params.get('risk_free_rate', 0.05)), 0.005,
                     key="interactive_r")
    with col4:
        seed = st.slider("Random Seed", 0, 1000,
                        int(params.get('seed', 42)), 1,
                        key="interactive_seed")

    n_steps = params.get('num_steps', 252)
    t = params.get('time_horizon', 1.0)

    # Model-specific parameters
    if model == 'heston':
        st.markdown("#### Heston Parameters")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            v0 = st.slider("V₀", 0.01, 0.25,
                          float(params.get('heston_v0', 0.04)), 0.01,
                          key="interactive_v0")
        with col2:
            kappa = st.slider("κ (mean rev.)", 0.5, 10.0,
                             float(params.get('heston_kappa', 2.0)), 0.1,
                             key="interactive_kappa")
        with col3:
            theta = st.slider("θ (long-run)", 0.01, 0.25,
                             float(params.get('heston_theta', 0.04)), 0.01,
                             key="interactive_theta")
        with col4:
            xi = st.slider("ξ (vol of vol)", 0.1, 1.0,
                          float(params.get('heston_xi', 0.3)), 0.05,
                          key="interactive_xi")
        with col5:
            rho = st.slider("ρ (correlation)", -0.95, 0.0,
                           float(params.get('heston_rho', -0.7)), 0.05,
                           key="interactive_rho")

        # Simulate Heston
        time_grid, path, variance, returns = simulate_single_heston_path(
            s0, v0, r, kappa, theta, xi, rho, t, n_steps, seed
        )

        # Create figure with price and variance
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
            subplot_titles=("Price Path", "Instantaneous Variance")
        )

        # Price path
        fig.add_trace(
            go.Scatter(x=time_grid, y=path, mode='lines',
                      name='Price', line=dict(color='#0d9488', width=1.5)),
            row=1, col=1
        )

        # Initial price line
        fig.add_hline(y=s0, line_dash="dot", line_color="#dc2626",
                     annotation_text=f"S₀={s0:.0f}",
                     annotation_position="top left",
                     row=1, col=1)

        # Variance path
        fig.add_trace(
            go.Scatter(x=time_grid, y=variance, mode='lines',
                      name='Variance', line=dict(color='#d97706', width=1.5)),
            row=2, col=1
        )

        # Long-run variance
        fig.add_hline(y=theta, line_dash="dash", line_color="#059669",
                     annotation_text=f"θ={theta:.4f} ({np.sqrt(theta)*100:.1f}%)",
                     annotation_position="top left",
                     row=2, col=1)

        # Rolling volatility (from variance)
        vol_path = np.sqrt(variance) * 100
        rolling_vol = compute_rolling_volatility(returns, window=30)

        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Variance", row=2, col=1)
        fig.update_xaxes(title_text="Time (Years)", row=2, col=1)

    else:  # GBM
        # Simulate GBM
        time_grid, path, returns = simulate_single_gbm_path(
            s0, r, sigma, t, n_steps, seed
        )

        # Rolling volatility
        rolling_vol = compute_rolling_volatility(returns, window=30)

        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
            subplot_titles=("Price Path", "Rolling Volatility (30-day window)")
        )

        # Price path
        fig.add_trace(
            go.Scatter(x=time_grid, y=path, mode='lines',
                      name='Price', line=dict(color='#0d9488', width=1.5)),
            row=1, col=1
        )

        # Initial price
        fig.add_hline(y=s0, line_dash="dot", line_color="#dc2626",
                     annotation_text=f"S₀={s0:.0f}",
                     annotation_position="top left",
                     row=1, col=1)

        # Rolling volatility
        fig.add_trace(
            go.Scatter(x=time_grid, y=rolling_vol * 100, mode='lines',
                      name='Rolling Vol', line=dict(color='#d97706', width=1.5)),
            row=2, col=1
        )

        # True volatility
        fig.add_hline(y=sigma * 100, line_dash="dash", line_color="#059669",
                     annotation_text=f"σ={sigma*100:.1f}%",
                     annotation_position="top left",
                     row=2, col=1)

        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_xaxes(title_text="Time (Years)", row=2, col=1)

    fig.update_layout(
        height=CHART_HEIGHT_LARGE,
        showlegend=False,
        margin=dict(l=60, r=40, t=40, b=40),
        hovermode='x unified'
    )

    st.plotly_chart(fig, width="stretch")

    # Statistics
    _display_path_statistics(path, returns, s0, "Price")


def _render_interactive_volatility_path(params: Dict[str, Any]) -> None:
    """Render interactive volatility path visualization."""
    model = params.get('vol_model', 'garch')

    # Common parameters
    st.markdown("#### Model Parameters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sigma0 = st.slider("Initial Vol (σ₀)", 0.05, 0.50,
                          float(params.get('volatility', 0.20)), 0.01,
                          key="interactive_sigma0")
    with col2:
        alpha = st.slider("α (ARCH)", 0.01, 0.30,
                         float(params.get('garch_alpha', 0.05)), 0.01,
                         key="interactive_alpha")
    with col3:
        beta = st.slider("β (GARCH)", 0.50, 0.98,
                        float(params.get('garch_beta', 0.90)), 0.01,
                        key="interactive_beta")
    with col4:
        seed = st.slider("Random Seed", 0, 1000,
                        int(params.get('seed', 42)), 1,
                        key="interactive_seed_vol")

    n_steps = params.get('num_steps', 252)

    # Model-specific parameters and simulation
    if model == 'ngarch':
        col1, col2 = st.columns([1, 3])
        with col1:
            theta_ngarch = st.slider("θ (leverage)", 0.0, 2.0,
                                    float(params.get('ngarch_theta', 0.5)), 0.1,
                                    key="interactive_theta_ngarch")

        # Compute omega for target long-run variance
        persistence = alpha * (1 + theta_ngarch**2) + beta
        if persistence < 1:
            omega = sigma0**2 * (1 - persistence)
        else:
            omega = 0.00001
            st.warning(f"Non-stationary: α(1+θ²)+β = {persistence:.3f} ≥ 1")

        time_grid, variance, volatility, returns = simulate_single_ngarch_path(
            sigma0, omega, alpha, beta, theta_ngarch, n_steps, seed
        )
        unconditional_var = compute_unconditional_variance_ngarch(omega, alpha, beta, theta_ngarch)
        model_name = "NGARCH"

    elif model == 'gjr_garch':
        col1, col2 = st.columns([1, 3])
        with col1:
            gamma = st.slider("γ (asymmetry)", 0.0, 0.20,
                             float(params.get('gjr_gamma', 0.05)), 0.01,
                             key="interactive_gamma_gjr")

        persistence = alpha + beta + 0.5 * gamma
        if persistence < 1:
            omega = sigma0**2 * (1 - persistence)
        else:
            omega = 0.00001
            st.warning(f"Non-stationary: α+β+γ/2 = {persistence:.3f} ≥ 1")

        time_grid, variance, volatility, returns = simulate_single_gjr_garch_path(
            sigma0, omega, alpha, beta, gamma, n_steps, seed
        )
        unconditional_var = compute_unconditional_variance_gjr(omega, alpha, beta, gamma)
        model_name = "GJR-GARCH"

    else:  # Standard GARCH
        persistence = alpha + beta
        if persistence < 1:
            omega = sigma0**2 * (1 - persistence)
        else:
            omega = 0.00001
            st.warning(f"Non-stationary: α+β = {persistence:.3f} ≥ 1")

        time_grid, variance, volatility, returns = simulate_single_garch_path(
            sigma0, omega, alpha, beta, n_steps, seed
        )
        unconditional_var = compute_unconditional_variance_garch(omega, alpha, beta)
        model_name = "GARCH(1,1)"

    # Compute rolling historical volatility
    rolling_vol = compute_rolling_volatility(returns, window=30)

    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.5],
        subplot_titles=(
            f"{model_name} Variance Path",
            "Volatility: Instantaneous vs Rolling (30-day)"
        )
    )

    # Variance path
    fig.add_trace(
        go.Scatter(x=time_grid, y=variance, mode='lines',
                  name='Variance', line=dict(color='#d97706', width=1.5)),
        row=1, col=1
    )

    # Unconditional variance
    if not np.isnan(unconditional_var):
        fig.add_hline(
            y=unconditional_var,
            line_dash="dash",
            line_color="#059669",
            row=1, col=1
        )
        # Add annotation separately for better positioning
        fig.add_annotation(
            x=0.02, y=unconditional_var,
            xref="paper", yref="y",
            text=f"E[σ²]={unconditional_var:.6f} ({np.sqrt(unconditional_var)*100:.1f}%)",
            showarrow=False,
            font=dict(size=11, color="#059669"),
            bgcolor="rgba(255,255,255,0.8)",
            borderpad=3,
            row=1, col=1
        )

    # Initial variance
    fig.add_hline(
        y=sigma0**2,
        line_dash="dot",
        line_color="#dc2626",
        row=1, col=1
    )
    fig.add_annotation(
        x=0.98, y=sigma0**2,
        xref="paper", yref="y",
        text=f"σ₀²={sigma0**2:.6f}",
        showarrow=False,
        font=dict(size=11, color="#dc2626"),
        bgcolor="rgba(255,255,255,0.8)",
        borderpad=3,
        xanchor="right",
        row=1, col=1
    )

    # Instantaneous volatility
    fig.add_trace(
        go.Scatter(x=time_grid, y=volatility * 100, mode='lines',
                  name='Instantaneous Vol',
                  line=dict(color='#0d9488', width=1.5)),
        row=2, col=1
    )

    # Rolling volatility
    fig.add_trace(
        go.Scatter(x=time_grid, y=rolling_vol * 100, mode='lines',
                  name='Rolling Vol (30d)',
                  line=dict(color='#6366f1', width=1.5, dash='dot')),
        row=2, col=1
    )

    # Unconditional volatility
    if not np.isnan(unconditional_var):
        unconditional_vol = np.sqrt(unconditional_var) * 100
        fig.add_hline(
            y=unconditional_vol,
            line_dash="dash",
            line_color="#059669",
            row=2, col=1
        )
        fig.add_annotation(
            x=0.02, y=unconditional_vol,
            xref="paper", yref="y2",
            text=f"E[σ]={unconditional_vol:.1f}%",
            showarrow=False,
            font=dict(size=11, color="#059669"),
            bgcolor="rgba(255,255,255,0.8)",
            borderpad=3,
            row=2, col=1
        )

    fig.update_yaxes(title_text="Variance (σ²)", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
    fig.update_xaxes(title_text="Time Step (days)", row=2, col=1)

    fig.update_layout(
        height=CHART_HEIGHT_LARGE,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(l=60, r=40, t=60, b=40),
        hovermode='x unified'
    )

    st.plotly_chart(fig, width="stretch")

    # Statistics
    _display_volatility_statistics(variance, volatility, unconditional_var, model_name)


def _display_path_statistics(
    path: np.ndarray,
    returns: np.ndarray,
    initial: float,
    name: str
) -> None:
    """Display statistics for price path."""
    st.markdown("#### Path Statistics")

    col1, col2, col3, col4, col5 = st.columns(5)

    total_return = (path[-1] - initial) / initial * 100

    with col1:
        st.metric("Terminal Value", f"${path[-1]:.2f}")
    with col2:
        st.metric("Total Return", f"{total_return:+.2f}%")
    with col3:
        st.metric("Max Value", f"${path.max():.2f}")
    with col4:
        st.metric("Min Value", f"${path.min():.2f}")
    with col5:
        # Realized volatility
        if len(returns) > 1:
            realized_vol = np.std(returns[1:]) * np.sqrt(252) * 100
            st.metric("Realized Vol", f"{realized_vol:.1f}%")


def _display_volatility_statistics(
    variance: np.ndarray,
    volatility: np.ndarray,
    unconditional_var: float,
    model_name: str
) -> None:
    """Display statistics for volatility path."""
    st.markdown("#### Volatility Statistics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Mean Vol", f"{volatility.mean()*100:.2f}%")
    with col2:
        st.metric("Terminal Vol", f"{volatility[-1]*100:.2f}%")
    with col3:
        st.metric("Max Vol", f"{volatility.max()*100:.2f}%")
    with col4:
        st.metric("Min Vol", f"{volatility.min()*100:.2f}%")
    with col5:
        if not np.isnan(unconditional_var):
            st.metric("Unconditional", f"{np.sqrt(unconditional_var)*100:.2f}%")
        else:
            st.metric("Unconditional", "N/A (non-stat)")
