"""
Sidebar component for Monte Carlo Simulation Explorer.

Provides model selection and parameter configuration interface.
"""

import streamlit as st
from typing import Tuple, Dict, Any

from config.constants import (
    PRICE_MODELS,
    VOLATILITY_MODELS,
    MODEL_DESCRIPTIONS,
    STATIONARITY_CONDITIONS,
    DEFAULT_SPOT_PRICE,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_VOLATILITY,
    DEFAULT_TIME_HORIZON,
    DEFAULT_NUM_PATHS,
    DEFAULT_NUM_STEPS
)
from config.styles import param_group_html


def render_sidebar() -> Dict[str, Any]:
    """
    Render the sidebar with model selection and parameters.

    Returns:
        Dictionary containing all simulation parameters
    """
    with st.sidebar:
        st.markdown("## Simulation Settings")

        # =================================================================
        # SIMULATION MODE
        # =================================================================
        st.markdown('<div class="sidebar-header">Simulation Mode</div>', unsafe_allow_html=True)

        simulation_mode_display = st.radio(
            "Choose simulation type",
            options=["Price Paths", "Volatility Paths"],
            horizontal=True,
            label_visibility="collapsed"
        )

        # Map display names to internal keys
        simulation_mode = "price" if simulation_mode_display == "Price Paths" else "volatility"

        # =================================================================
        # MODEL SELECTION
        # =================================================================
        st.markdown('<div class="sidebar-header">Model Selection</div>', unsafe_allow_html=True)

        if simulation_mode == "price":
            price_model = st.selectbox(
                "Price Model",
                options=list(PRICE_MODELS.keys()),
                format_func=lambda x: PRICE_MODELS[x],
                key="price_model_select"
            )

            # Show model description in expander
            with st.expander("About this model", expanded=False):
                st.markdown(MODEL_DESCRIPTIONS.get(price_model, ""))
            vol_model = "garch"  # Default, not used
        else:
            price_model = "gbm"  # Default, not used
            vol_model = st.selectbox(
                "Volatility Model",
                options=list(VOLATILITY_MODELS.keys()),
                format_func=lambda x: VOLATILITY_MODELS[x],
                key="vol_model_select"
            )

            # Show model description
            with st.expander("About this model", expanded=False):
                st.markdown(MODEL_DESCRIPTIONS.get(vol_model, ""))

            # Stationarity condition
            if vol_model in STATIONARITY_CONDITIONS:
                st.info(f"Stationarity: {STATIONARITY_CONDITIONS[vol_model]}")

        # =================================================================
        # COMMON PARAMETERS
        # =================================================================
        st.markdown('<div class="sidebar-header">Common Parameters</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            spot_price = st.number_input(
                "Initial Price (S0)",
                min_value=1.0,
                max_value=10000.0,
                value=DEFAULT_SPOT_PRICE,
                step=1.0,
                format="%.2f"
            )

        with col2:
            risk_free_rate = st.number_input(
                "Risk-Free Rate (r)",
                min_value=0.0,
                max_value=0.20,
                value=DEFAULT_RISK_FREE_RATE,
                step=0.005,
                format="%.3f"
            )

        col1, col2 = st.columns(2)

        with col1:
            volatility = st.number_input(
                "Volatility (sigma)",
                min_value=0.01,
                max_value=1.0,
                value=DEFAULT_VOLATILITY,
                step=0.01,
                format="%.2f"
            )

        with col2:
            time_horizon = st.number_input(
                "Time Horizon (T)",
                min_value=0.1,
                max_value=10.0,
                value=DEFAULT_TIME_HORIZON,
                step=0.1,
                format="%.1f"
            )

        # =================================================================
        # SIMULATION SETTINGS
        # =================================================================
        st.markdown('<div class="sidebar-header">Simulation Settings</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            num_paths = st.number_input(
                "Number of Paths",
                min_value=10,
                max_value=100000,
                value=DEFAULT_NUM_PATHS,
                step=100
            )

        with col2:
            num_steps = st.number_input(
                "Time Steps",
                min_value=10,
                max_value=1000,
                value=DEFAULT_NUM_STEPS,
                step=10
            )

        seed = st.number_input(
            "Random Seed (for reproducibility)",
            min_value=0,
            max_value=99999,
            value=42,
            step=1
        )

        # =================================================================
        # MODEL-SPECIFIC PARAMETERS
        # =================================================================
        params = {
            'simulation_mode': simulation_mode,
            'price_model': price_model,
            'vol_model': vol_model,
            'spot_price': spot_price,
            'risk_free_rate': risk_free_rate,
            'volatility': volatility,
            'time_horizon': time_horizon,
            'num_paths': int(num_paths),
            'num_steps': int(num_steps),
            'seed': int(seed)
        }

        # Model-specific parameters based on simulation mode
        if simulation_mode == "price":
            params.update(_render_price_model_params(price_model, volatility))
        else:
            params.update(_render_volatility_model_params(vol_model, volatility))

        # =================================================================
        # VISUALIZATION OPTIONS
        # =================================================================
        st.markdown('<div class="sidebar-header">Visualization Options</div>', unsafe_allow_html=True)

        show_percentiles = st.checkbox("Show percentile bands (5%-95%)", value=True)
        show_mean = st.checkbox("Show mean path", value=True)

        max_display = st.slider(
            "Max paths to display",
            min_value=10,
            max_value=200,
            value=50,
            step=10
        )

        params['show_percentiles'] = show_percentiles
        params['show_mean'] = show_mean
        params['max_display_paths'] = max_display

        # =================================================================
        # ACTION BUTTONS
        # =================================================================
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            run_simulation = st.button(
                "Run Simulation",
                type="primary",
                width="stretch"
            )

        with col2:
            reset = st.button(
                "Reset Defaults",
                type="secondary",
                width="stretch"
            )

        params['run_simulation'] = run_simulation
        params['reset'] = reset

        return params


def _render_price_model_params(model: str, base_volatility: float) -> Dict[str, Any]:
    """Render parameters specific to price models."""
    params = {}

    if model == "heston":
        st.markdown('<div class="sidebar-header">Heston Parameters</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            v0 = st.number_input(
                "Initial Variance (V0)",
                min_value=0.001,
                max_value=1.0,
                value=base_volatility ** 2,
                step=0.01,
                format="%.4f"
            )
            kappa = st.number_input(
                "Mean Reversion (kappa)",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                format="%.2f"
            )

        with col2:
            theta = st.number_input(
                "Long-term Var (theta)",
                min_value=0.001,
                max_value=1.0,
                value=base_volatility ** 2,
                step=0.01,
                format="%.4f"
            )
            xi = st.number_input(
                "Vol of Vol (xi)",
                min_value=0.01,
                max_value=1.0,
                value=0.3,
                step=0.05,
                format="%.2f"
            )

        rho = st.slider(
            "Correlation (rho)",
            min_value=-0.99,
            max_value=0.99,
            value=-0.7,
            step=0.01
        )

        # Feller condition check
        feller = 2 * kappa * theta / (xi ** 2)
        if feller < 1:
            st.warning(f"Feller condition not satisfied ({feller:.2f} < 1). Variance may reach zero.")

        params['heston_v0'] = v0
        params['heston_kappa'] = kappa
        params['heston_theta'] = theta
        params['heston_xi'] = xi
        params['heston_rho'] = rho

    elif model == "merton":
        st.markdown('<div class="sidebar-header">Merton Jump Parameters</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            lambda_j = st.number_input(
                "Jump Intensity (lambda)",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                step=0.1,
                format="%.2f",
                help="Expected number of jumps per year"
            )
            mu_j = st.number_input(
                "Mean Log-Jump (mu_j)",
                min_value=-0.5,
                max_value=0.5,
                value=-0.1,
                step=0.01,
                format="%.3f"
            )

        with col2:
            sigma_j = st.number_input(
                "Jump Volatility (sigma_j)",
                min_value=0.01,
                max_value=0.5,
                value=0.2,
                step=0.01,
                format="%.2f"
            )

        params['merton_lambda'] = lambda_j
        params['merton_mu_j'] = mu_j
        params['merton_sigma_j'] = sigma_j

    elif model == "sabr":
        st.markdown('<div class="sidebar-header">SABR Parameters</div>', unsafe_allow_html=True)

        beta = st.slider(
            "CEV Exponent (beta)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0=Normal, 1=Lognormal"
        )

        col1, col2 = st.columns(2)

        with col1:
            nu = st.number_input(
                "Vol of Vol (nu)",
                min_value=0.01,
                max_value=1.0,
                value=0.4,
                step=0.05,
                format="%.2f"
            )

        with col2:
            rho = st.slider(
                "Correlation (rho)",
                min_value=-0.99,
                max_value=0.99,
                value=-0.3,
                step=0.01,
                key="sabr_rho"
            )

        params['sabr_beta'] = beta
        params['sabr_nu'] = nu
        params['sabr_rho'] = rho

    return params


def _render_volatility_model_params(model: str, base_volatility: float) -> Dict[str, Any]:
    """Render parameters specific to volatility models."""
    params = {}

    st.markdown(f'<div class="sidebar-header">{VOLATILITY_MODELS[model]} Parameters</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        alpha = st.number_input(
            "ARCH coef (alpha)",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            step=0.01,
            format="%.3f",
            help="Reaction to past shocks"
        )

    with col2:
        beta = st.number_input(
            "GARCH coef (beta)",
            min_value=0.0,
            max_value=0.99,
            value=0.90,
            step=0.01,
            format="%.3f",
            help="Persistence of variance"
        )

    params['garch_alpha'] = alpha
    params['garch_beta'] = beta

    # Model-specific additional parameters
    if model == "ngarch":
        theta = st.number_input(
            "Leverage (theta)",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.1,
            format="%.2f",
            help="Asymmetry parameter"
        )
        params['ngarch_theta'] = theta

        # Check stationarity
        persistence = alpha * (1 + theta ** 2) + beta
        if persistence >= 1:
            st.error(f"Non-stationary! alpha(1+theta^2) + beta = {persistence:.3f} >= 1")
        else:
            st.success(f"Stationary: persistence = {persistence:.3f}")

    elif model == "gjr_garch":
        gamma = st.number_input(
            "Asymmetry (gamma)",
            min_value=0.0,
            max_value=0.3,
            value=0.05,
            step=0.01,
            format="%.3f",
            help="Additional weight for negative shocks"
        )
        params['gjr_gamma'] = gamma

        # Check stationarity
        persistence = alpha + beta + 0.5 * gamma
        if persistence >= 1:
            st.error(f"Non-stationary! alpha + beta + gamma/2 = {persistence:.3f} >= 1")
        else:
            st.success(f"Stationary: persistence = {persistence:.3f}")

    elif model == "egarch":
        gamma = st.number_input(
            "Asymmetry (gamma)",
            min_value=-0.5,
            max_value=0.5,
            value=-0.1,
            step=0.01,
            format="%.3f",
            help="Negative for leverage effect"
        )
        params['egarch_gamma'] = gamma

        if abs(beta) >= 1:
            st.error(f"Non-stationary! |beta| = {abs(beta):.3f} >= 1")
        else:
            st.success(f"Stationary: |beta| = {abs(beta):.3f}")

    else:  # Standard GARCH
        persistence = alpha + beta
        if persistence >= 1:
            st.error(f"Non-stationary! alpha + beta = {persistence:.3f} >= 1")
        else:
            st.success(f"Stationary: persistence = {persistence:.3f}")

    # Compute long-run variance/volatility
    if model == "garch":
        if alpha + beta < 1:
            omega = base_volatility ** 2 * (1 - alpha - beta)
            long_run_var = omega / (1 - alpha - beta)
            st.info(f"Long-run volatility: {100*long_run_var**0.5:.1f}%")
            params['garch_omega'] = omega

    return params
