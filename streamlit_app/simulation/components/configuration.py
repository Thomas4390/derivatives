"""
Configuration component for Monte Carlo Simulation Explorer.

Provides a centralized configuration interface in the main content area
(replacing the sidebar-based configuration).
"""

import streamlit as st
from typing import Dict, Any

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
    DEFAULT_NUM_STEPS,
    DEFAULT_EXPECTED_RETURN
)
from config.styles import config_card_html, p_measure_badge_html

# Import Black-Scholes for premium calculation
from backend.option_pricing.options_calculator import (
    black_scholes_call_price,
    black_scholes_put_price
)


def render_configuration_tab() -> Dict[str, Any]:
    """
    Render the centralized configuration tab.

    Returns:
        Dictionary containing all simulation parameters
    """
    params = {}

    # =================================================================
    # MARKET PARAMETERS (single row)
    # =================================================================
    st.markdown(config_card_html("📊", "Market Parameters"), unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        spot_price = st.number_input(
            "S₀ (Spot)",
            min_value=1.0,
            max_value=10000.0,
            value=DEFAULT_SPOT_PRICE,
            step=1.0,
            format="%.2f",
            help="Initial underlying price"
        )

    with col2:
        volatility = st.number_input(
            "σ (Volatility)",
            min_value=0.01,
            max_value=1.0,
            value=DEFAULT_VOLATILITY,
            step=0.01,
            format="%.2f",
            help="Annualized volatility (decimal)"
        )

    with col3:
        time_horizon = st.number_input(
            "T (Years)",
            min_value=0.1,
            max_value=10.0,
            value=DEFAULT_TIME_HORIZON,
            step=0.1,
            format="%.1f",
            help="Time horizon in years"
        )

    with col4:
        risk_free_rate = st.number_input(
            "r (Risk-free)",
            min_value=0.0,
            max_value=0.20,
            value=DEFAULT_RISK_FREE_RATE,
            step=0.005,
            format="%.3f",
            help="Annual risk-free rate"
        )

    with col5:
        # Expected return with P-measure badge
        st.markdown(
            f'<label style="font-size: 0.875rem; font-weight: 500; margin-bottom: 0.25rem; display: block;">'
            f'μ (Expected) {p_measure_badge_html()}</label>',
            unsafe_allow_html=True
        )
        expected_return = st.number_input(
            "Expected Return",
            min_value=-0.20,
            max_value=0.50,
            value=DEFAULT_EXPECTED_RETURN,
            step=0.01,
            format="%.2f",
            help="Annual expected return under P-measure (for P&L simulation)",
            label_visibility="collapsed"
        )

    params['spot_price'] = spot_price
    params['volatility'] = volatility
    params['time_horizon'] = time_horizon
    params['risk_free_rate'] = risk_free_rate
    params['expected_return'] = expected_return

    st.markdown("</div>", unsafe_allow_html=True)

    # =================================================================
    # SIMULATION SETTINGS
    # =================================================================
    st.markdown(config_card_html("⚡", "Simulation Settings"), unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        num_paths = st.number_input(
            "Number of Paths",
            min_value=10,
            max_value=100000,
            value=DEFAULT_NUM_PATHS,
            step=100,
            help="Number of Monte Carlo paths to simulate"
        )

    with col2:
        num_steps = st.number_input(
            "Time Steps",
            min_value=10,
            max_value=1000,
            value=DEFAULT_NUM_STEPS,
            step=10,
            help="Number of time steps (252 = daily for 1 year)"
        )

    with col3:
        seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=99999,
            value=42,
            step=1,
            help="Seed for reproducibility"
        )

    params['num_paths'] = int(num_paths)
    params['num_steps'] = int(num_steps)
    params['seed'] = int(seed)

    st.markdown("</div>", unsafe_allow_html=True)

    # =================================================================
    # MODEL CONFIGURATION (side-by-side)
    # =================================================================
    col_price, col_vol = st.columns(2)

    with col_price:
        st.markdown(config_card_html("📈", "Price Model"), unsafe_allow_html=True)

        price_model = st.selectbox(
            "Select Model",
            options=list(PRICE_MODELS.keys()),
            format_func=lambda x: PRICE_MODELS[x],
            key="config_price_model"
        )

        # Model description
        with st.expander("Model Info", expanded=False):
            st.markdown(MODEL_DESCRIPTIONS.get(price_model, ""), unsafe_allow_html=True)

        # Model-specific parameters
        price_params = _render_price_model_params(price_model, volatility)
        params.update(price_params)
        params['price_model'] = price_model

        st.markdown("</div>", unsafe_allow_html=True)

    with col_vol:
        st.markdown(config_card_html("📊", "Volatility Model"), unsafe_allow_html=True)

        vol_model = st.selectbox(
            "Select Model",
            options=list(VOLATILITY_MODELS.keys()),
            format_func=lambda x: VOLATILITY_MODELS[x],
            key="config_vol_model"
        )

        # Model description
        with st.expander("Model Info", expanded=False):
            st.markdown(MODEL_DESCRIPTIONS.get(vol_model, ""), unsafe_allow_html=True)
            if vol_model in STATIONARITY_CONDITIONS:
                st.info(f"Stationarity: {STATIONARITY_CONDITIONS[vol_model]}")

        # Model-specific parameters
        vol_params = _render_volatility_model_params(vol_model, volatility)
        params.update(vol_params)
        params['vol_model'] = vol_model

        st.markdown("</div>", unsafe_allow_html=True)

    # =================================================================
    # STRATEGY BUILDER
    # =================================================================
    st.markdown(config_card_html("🎯", "Option Strategy"), unsafe_allow_html=True)

    def bs_price(s, k, r, t, sigma, opt_type):
        """Black-Scholes pricing wrapper."""
        if opt_type == 'call':
            return black_scholes_call_price(s, k, t, r, sigma)
        else:
            return black_scholes_put_price(s, k, t, r, sigma)

    from components.strategy_builder import (
        render_strategy_builder,
        export_positions_for_pnl_engine
    )

    positions, stock_position = render_strategy_builder(
        spot_price=spot_price,
        risk_free_rate=risk_free_rate,
        time_to_expiry=time_horizon,
        volatility=volatility,
        bs_price_function=bs_price
    )

    # Export positions for Numba engine
    position_arrays = export_positions_for_pnl_engine(positions, stock_position)

    params['option_positions'] = positions
    params['stock_position'] = stock_position
    params['position_arrays'] = position_arrays

    st.markdown("</div>", unsafe_allow_html=True)

    # =================================================================
    # VISUALIZATION OPTIONS (compact)
    # =================================================================
    st.markdown(config_card_html("🎨", "Visualization Options"), unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        show_percentiles = st.checkbox("Show percentile bands (5%-95%)", value=True)

    with col2:
        show_mean = st.checkbox("Show mean path", value=True)

    with col3:
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

    st.markdown("</div>", unsafe_allow_html=True)

    return params


def _render_price_model_params(model: str, base_volatility: float) -> Dict[str, Any]:
    """Render parameters specific to price models."""
    params = {}

    if model == "heston":
        col1, col2 = st.columns(2)

        with col1:
            v0 = st.number_input(
                "V₀ (Initial Var)",
                min_value=0.001,
                max_value=1.0,
                value=base_volatility ** 2,
                step=0.01,
                format="%.4f",
                key="config_heston_v0"
            )
            kappa = st.number_input(
                "κ (Mean Rev)",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                format="%.2f",
                key="config_heston_kappa"
            )

        with col2:
            theta = st.number_input(
                "θ (Long-term Var)",
                min_value=0.001,
                max_value=1.0,
                value=base_volatility ** 2,
                step=0.01,
                format="%.4f",
                key="config_heston_theta"
            )
            xi = st.number_input(
                "ξ (Vol of Vol)",
                min_value=0.01,
                max_value=1.0,
                value=0.3,
                step=0.05,
                format="%.2f",
                key="config_heston_xi"
            )

        rho = st.slider(
            "ρ (Correlation)",
            min_value=-0.99,
            max_value=0.99,
            value=-0.7,
            step=0.01,
            key="config_heston_rho"
        )

        # Feller condition check
        feller = 2 * kappa * theta / (xi ** 2)
        if feller < 1:
            st.warning(f"Feller condition not satisfied ({feller:.2f} < 1)")

        params['heston_v0'] = v0
        params['heston_kappa'] = kappa
        params['heston_theta'] = theta
        params['heston_xi'] = xi
        params['heston_rho'] = rho

    elif model == "merton":
        col1, col2 = st.columns(2)

        with col1:
            lambda_j = st.number_input(
                "λ (Jump Intensity)",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                step=0.1,
                format="%.2f",
                key="config_merton_lambda"
            )
            mu_j = st.number_input(
                "μⱼ (Mean Log-Jump)",
                min_value=-0.5,
                max_value=0.5,
                value=-0.1,
                step=0.01,
                format="%.3f",
                key="config_merton_mu_j"
            )

        with col2:
            sigma_j = st.number_input(
                "σⱼ (Jump Vol)",
                min_value=0.01,
                max_value=0.5,
                value=0.2,
                step=0.01,
                format="%.2f",
                key="config_merton_sigma_j"
            )

        params['merton_lambda'] = lambda_j
        params['merton_mu_j'] = mu_j
        params['merton_sigma_j'] = sigma_j

    elif model == "bates":
        st.markdown("**Stochastic Volatility**", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            v0 = st.number_input(
                "V₀ (Initial Var)",
                min_value=0.001,
                max_value=1.0,
                value=base_volatility ** 2,
                step=0.01,
                format="%.4f",
                key="config_bates_v0"
            )
            kappa = st.number_input(
                "κ (Mean Rev)",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                format="%.2f",
                key="config_bates_kappa"
            )

        with col2:
            theta = st.number_input(
                "θ (Long-term Var)",
                min_value=0.001,
                max_value=1.0,
                value=base_volatility ** 2,
                step=0.01,
                format="%.4f",
                key="config_bates_theta"
            )
            xi = st.number_input(
                "ξ (Vol of Vol)",
                min_value=0.01,
                max_value=1.0,
                value=0.3,
                step=0.05,
                format="%.2f",
                key="config_bates_xi"
            )

        rho = st.slider(
            "ρ (Correlation)",
            min_value=-0.99,
            max_value=0.99,
            value=-0.7,
            step=0.01,
            key="config_bates_rho"
        )

        # Feller condition
        feller = 2 * kappa * theta / (xi ** 2)
        if feller < 1:
            st.warning(f"Feller condition not satisfied ({feller:.2f} < 1)")

        st.markdown("**Jump Component**", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            lambda_j = st.number_input(
                "λ (Jump Intensity)",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                step=0.1,
                format="%.2f",
                key="config_bates_lambda"
            )
            mu_j = st.number_input(
                "μⱼ (Mean Log-Jump)",
                min_value=-0.5,
                max_value=0.5,
                value=-0.1,
                step=0.01,
                format="%.3f",
                key="config_bates_mu_j"
            )

        with col2:
            sigma_j = st.number_input(
                "σⱼ (Jump Vol)",
                min_value=0.01,
                max_value=0.5,
                value=0.2,
                step=0.01,
                format="%.2f",
                key="config_bates_sigma_j"
            )

        params['bates_v0'] = v0
        params['bates_kappa'] = kappa
        params['bates_theta'] = theta
        params['bates_xi'] = xi
        params['bates_rho'] = rho
        params['bates_lambda'] = lambda_j
        params['bates_mu_j'] = mu_j
        params['bates_sigma_j'] = sigma_j

    elif model == "sabr":
        beta = st.slider(
            "β (CEV Exponent)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0=Normal, 1=Lognormal",
            key="config_sabr_beta"
        )

        col1, col2 = st.columns(2)

        with col1:
            nu = st.number_input(
                "ν (Vol of Vol)",
                min_value=0.01,
                max_value=1.0,
                value=0.4,
                step=0.05,
                format="%.2f",
                key="config_sabr_nu"
            )

        with col2:
            rho = st.slider(
                "ρ (Correlation)",
                min_value=-0.99,
                max_value=0.99,
                value=-0.3,
                step=0.01,
                key="config_sabr_rho"
            )

        params['sabr_beta'] = beta
        params['sabr_nu'] = nu
        params['sabr_rho'] = rho

    return params


def _render_volatility_model_params(model: str, base_volatility: float) -> Dict[str, Any]:
    """Render parameters specific to volatility models."""
    params = {}

    col1, col2 = st.columns(2)

    with col1:
        alpha = st.number_input(
            "α (ARCH coef)",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            step=0.01,
            format="%.3f",
            help="Reaction to past shocks",
            key="config_garch_alpha"
        )

    with col2:
        beta = st.number_input(
            "β (GARCH coef)",
            min_value=0.0,
            max_value=0.99,
            value=0.90,
            step=0.01,
            format="%.3f",
            help="Persistence of variance",
            key="config_garch_beta"
        )

    params['garch_alpha'] = alpha
    params['garch_beta'] = beta

    # Model-specific additional parameters
    if model == "ngarch":
        theta = st.number_input(
            "θ (Leverage)",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.1,
            format="%.2f",
            key="config_ngarch_theta"
        )
        params['ngarch_theta'] = theta

        persistence = alpha * (1 + theta ** 2) + beta
        if persistence >= 1:
            st.error(f"Non-stationary! α(1+θ²) + β = {persistence:.3f} >= 1")
        else:
            st.success(f"Stationary: {persistence:.3f}")

    elif model == "gjr_garch":
        gamma = st.number_input(
            "γ (Asymmetry)",
            min_value=0.0,
            max_value=0.3,
            value=0.05,
            step=0.01,
            format="%.3f",
            key="config_gjr_gamma"
        )
        params['gjr_gamma'] = gamma

        persistence = alpha + beta + 0.5 * gamma
        if persistence >= 1:
            st.error(f"Non-stationary! α + β + γ/2 = {persistence:.3f} >= 1")
        else:
            st.success(f"Stationary: {persistence:.3f}")

    elif model == "egarch":
        gamma = st.number_input(
            "γ (Asymmetry)",
            min_value=-0.5,
            max_value=0.5,
            value=-0.1,
            step=0.01,
            format="%.3f",
            key="config_egarch_gamma"
        )
        params['egarch_gamma'] = gamma

        if abs(beta) >= 1:
            st.error(f"Non-stationary! |β| = {abs(beta):.3f} >= 1")
        else:
            st.success(f"Stationary: |β| = {abs(beta):.3f}")

    else:  # Standard GARCH
        persistence = alpha + beta
        if persistence >= 1:
            st.error(f"Non-stationary! α + β = {persistence:.3f} >= 1")
        else:
            st.success(f"Stationary: {persistence:.3f}")

    # Compute long-run variance
    if model == "garch" and alpha + beta < 1:
        omega = base_volatility ** 2 * (1 - alpha - beta)
        params['garch_omega'] = omega

    return params
