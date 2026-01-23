"""
Parameter Panel Component - Dynamic parameter inputs based on model.

Provides:
- Market parameters (common to all models)
- Model-specific parameters (dynamic)
- Simulation settings
- Condition validation warnings
"""

import streamlit as st
from typing import Dict, Any, Optional

from config.model_registry import (
    MARKET_PARAMETERS,
    SIMULATION_PARAMETERS,
    get_model,
    get_parameter_defaults,
)
from utils.model_helpers import (
    get_condition_display,
    check_garch_stationarity,
    check_feller_condition,
)


def render_market_parameters(key_prefix: str = "market") -> Dict[str, float]:
    """
    Render market parameters section.

    Returns:
        Dictionary of market parameter values
    """
    st.subheader("📊 Market Parameters")

    params = {}

    # Spot price
    params["spot"] = st.number_input(
        "Spot Price (S₀)",
        min_value=1.0,
        max_value=10000.0,
        value=st.session_state.get(f"{key_prefix}_spot", 100.0),
        step=1.0,
        key=f"{key_prefix}_spot_input",
        help="Current asset price"
    )
    st.session_state[f"{key_prefix}_spot"] = params["spot"]

    # Two columns for rate and drift
    col1, col2 = st.columns(2)

    with col1:
        params["risk_free_rate"] = st.number_input(
            "Risk-Free Rate (r)",
            min_value=0.0,
            max_value=0.20,
            value=st.session_state.get(f"{key_prefix}_rate", 0.05),
            step=0.005,
            format="%.3f",
            key=f"{key_prefix}_rate_input",
            help="Annualized risk-free rate"
        )
        st.session_state[f"{key_prefix}_rate"] = params["risk_free_rate"]

    with col2:
        params["drift"] = st.number_input(
            "Expected Return (μ)",
            min_value=-0.20,
            max_value=0.50,
            value=st.session_state.get(f"{key_prefix}_drift", 0.08),
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_drift_input",
            help="Annualized expected return"
        )
        st.session_state[f"{key_prefix}_drift"] = params["drift"]

    # Time horizon
    params["time_horizon"] = st.number_input(
        "Time Horizon (T)",
        min_value=0.1,
        max_value=10.0,
        value=st.session_state.get(f"{key_prefix}_time", 1.0),
        step=0.1,
        format="%.1f",
        key=f"{key_prefix}_time_input",
        help="Simulation horizon in years"
    )
    st.session_state[f"{key_prefix}_time"] = params["time_horizon"]

    return params


def render_model_parameters(
    model_key: str,
    key_prefix: str = "model"
) -> Dict[str, float]:
    """
    Render model-specific parameters.

    Args:
        model_key: Selected model key
        key_prefix: Prefix for widget keys

    Returns:
        Dictionary of model parameter values
    """
    model = get_model(model_key)
    defaults = get_parameter_defaults(model_key)
    params = {}

    st.subheader("⚙️ Model Parameters")

    # Show model name
    st.caption(f"Parameters for {model.name}")

    # Render parameters based on model
    model_lower = model_key.lower()

    if model_lower == "gbm":
        params.update(_render_gbm_params(defaults, key_prefix))

    elif model_lower == "heston":
        params.update(_render_heston_params(defaults, key_prefix))
        _render_feller_warning(params)

    elif model_lower == "merton":
        params.update(_render_merton_params(defaults, key_prefix))

    elif model_lower == "bates":
        params.update(_render_bates_params(defaults, key_prefix))
        _render_feller_warning(params)

    elif model_lower == "garch":
        params.update(_render_garch_params(defaults, key_prefix))
        _render_stationarity_warning(model_key, params)

    elif model_lower == "ngarch":
        params.update(_render_ngarch_params(defaults, key_prefix))
        _render_stationarity_warning(model_key, params)

    elif model_lower == "gjr_garch":
        params.update(_render_gjr_garch_params(defaults, key_prefix))
        _render_stationarity_warning(model_key, params)

    return params


def _render_gbm_params(defaults: Dict, key_prefix: str) -> Dict[str, float]:
    """Render GBM parameters."""
    params = {}

    params["sigma"] = st.slider(
        "Volatility (σ)",
        min_value=0.01,
        max_value=1.0,
        value=st.session_state.get(f"{key_prefix}_sigma", defaults.get("sigma", 0.20)),
        step=0.01,
        format="%.2f",
        key=f"{key_prefix}_sigma_input",
        help="Annualized volatility (0.20 = 20%)"
    )
    st.session_state[f"{key_prefix}_sigma"] = params["sigma"]

    # Display as percentage
    st.caption(f"σ = {params['sigma']*100:.1f}%")

    return params


def _render_heston_params(defaults: Dict, key_prefix: str) -> Dict[str, float]:
    """Render Heston parameters."""
    params = {}

    # Initial variance
    params["v0"] = st.slider(
        "Initial Variance (V₀)",
        min_value=0.001,
        max_value=0.50,
        value=st.session_state.get(f"{key_prefix}_v0", defaults.get("v0", 0.04)),
        step=0.005,
        format="%.3f",
        key=f"{key_prefix}_v0_input",
        help="Initial variance level"
    )
    st.session_state[f"{key_prefix}_v0"] = params["v0"]
    st.caption(f"Initial vol = {params['v0']**0.5*100:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        # Mean reversion speed
        params["kappa"] = st.slider(
            "Mean Reversion (κ)",
            min_value=0.1,
            max_value=10.0,
            value=st.session_state.get(f"{key_prefix}_kappa", defaults.get("kappa", 2.0)),
            step=0.1,
            format="%.1f",
            key=f"{key_prefix}_kappa_input",
            help="Speed of variance mean reversion"
        )
        st.session_state[f"{key_prefix}_kappa"] = params["kappa"]

    with col2:
        # Long-run variance
        params["theta"] = st.slider(
            "Long-Run Var (θ)",
            min_value=0.001,
            max_value=0.50,
            value=st.session_state.get(f"{key_prefix}_theta", defaults.get("theta", 0.04)),
            step=0.005,
            format="%.3f",
            key=f"{key_prefix}_theta_input",
            help="Long-run variance level"
        )
        st.session_state[f"{key_prefix}_theta"] = params["theta"]
        st.caption(f"Long-run vol = {params['theta']**0.5*100:.1f}%")

    col3, col4 = st.columns(2)

    with col3:
        # Vol of vol
        params["xi"] = st.slider(
            "Vol of Vol (ξ)",
            min_value=0.01,
            max_value=1.0,
            value=st.session_state.get(f"{key_prefix}_xi", defaults.get("xi", 0.3)),
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_xi_input",
            help="Volatility of variance"
        )
        st.session_state[f"{key_prefix}_xi"] = params["xi"]

    with col4:
        # Correlation
        params["rho"] = st.slider(
            "Correlation (ρ)",
            min_value=-0.99,
            max_value=0.99,
            value=st.session_state.get(f"{key_prefix}_rho", defaults.get("rho", -0.7)),
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_rho_input",
            help="Price-variance correlation"
        )
        st.session_state[f"{key_prefix}_rho"] = params["rho"]

    return params


def _render_merton_params(defaults: Dict, key_prefix: str) -> Dict[str, float]:
    """Render Merton parameters."""
    params = {}

    # Diffusion volatility
    params["sigma"] = st.slider(
        "Diffusion Vol (σ)",
        min_value=0.01,
        max_value=1.0,
        value=st.session_state.get(f"{key_prefix}_sigma", defaults.get("sigma", 0.20)),
        step=0.01,
        format="%.2f",
        key=f"{key_prefix}_sigma_input",
        help="Diffusion volatility"
    )
    st.session_state[f"{key_prefix}_sigma"] = params["sigma"]

    st.markdown("**Jump Parameters**")

    col1, col2 = st.columns(2)

    with col1:
        params["lambda_j"] = st.slider(
            "Jump Intensity (λ)",
            min_value=0.0,
            max_value=5.0,
            value=st.session_state.get(f"{key_prefix}_lambda", defaults.get("lambda_j", 0.5)),
            step=0.1,
            format="%.1f",
            key=f"{key_prefix}_lambda_input",
            help="Expected jumps per year"
        )
        st.session_state[f"{key_prefix}_lambda"] = params["lambda_j"]

    with col2:
        params["mu_j"] = st.slider(
            "Mean Jump (μⱼ)",
            min_value=-0.5,
            max_value=0.5,
            value=st.session_state.get(f"{key_prefix}_muj", defaults.get("mu_j", -0.1)),
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_muj_input",
            help="Mean of log-jump size"
        )
        st.session_state[f"{key_prefix}_muj"] = params["mu_j"]

    params["sigma_j"] = st.slider(
        "Jump Vol (σⱼ)",
        min_value=0.01,
        max_value=0.5,
        value=st.session_state.get(f"{key_prefix}_sigmaj", defaults.get("sigma_j", 0.2)),
        step=0.01,
        format="%.2f",
        key=f"{key_prefix}_sigmaj_input",
        help="Volatility of log-jump size"
    )
    st.session_state[f"{key_prefix}_sigmaj"] = params["sigma_j"]

    return params


def _render_bates_params(defaults: Dict, key_prefix: str) -> Dict[str, float]:
    """Render Bates parameters (Heston + Merton jumps)."""
    params = {}

    # Heston part
    st.markdown("**Stochastic Volatility**")
    params.update(_render_heston_params(defaults, key_prefix))

    # Jump part
    st.markdown("**Jump Parameters**")

    col1, col2 = st.columns(2)

    with col1:
        params["lambda_j"] = st.slider(
            "Jump Intensity (λ)",
            min_value=0.0,
            max_value=5.0,
            value=st.session_state.get(f"{key_prefix}_lambda", defaults.get("lambda_j", 0.5)),
            step=0.1,
            format="%.1f",
            key=f"{key_prefix}_lambda_input",
            help="Expected jumps per year"
        )
        st.session_state[f"{key_prefix}_lambda"] = params["lambda_j"]

    with col2:
        params["mu_j"] = st.slider(
            "Mean Jump (μⱼ)",
            min_value=-0.5,
            max_value=0.5,
            value=st.session_state.get(f"{key_prefix}_muj", defaults.get("mu_j", -0.1)),
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_muj_input",
            help="Mean of log-jump size"
        )
        st.session_state[f"{key_prefix}_muj"] = params["mu_j"]

    params["sigma_j"] = st.slider(
        "Jump Vol (σⱼ)",
        min_value=0.01,
        max_value=0.5,
        value=st.session_state.get(f"{key_prefix}_sigmaj", defaults.get("sigma_j", 0.2)),
        step=0.01,
        format="%.2f",
        key=f"{key_prefix}_sigmaj_input",
        help="Volatility of log-jump size"
    )
    st.session_state[f"{key_prefix}_sigmaj"] = params["sigma_j"]

    return params


def _render_garch_params(defaults: Dict, key_prefix: str) -> Dict[str, float]:
    """Render GARCH parameters."""
    params = {}

    # Initial volatility
    params["sigma0"] = st.slider(
        "Initial Vol (σ₀)",
        min_value=0.01,
        max_value=1.0,
        value=st.session_state.get(f"{key_prefix}_sigma0", defaults.get("sigma0", 0.20)),
        step=0.01,
        format="%.2f",
        key=f"{key_prefix}_sigma0_input",
        help="Initial volatility level"
    )
    st.session_state[f"{key_prefix}_sigma0"] = params["sigma0"]
    st.caption(f"σ₀ = {params['sigma0']*100:.1f}%")

    # Omega (constant)
    params["omega"] = st.number_input(
        "Constant (ω)",
        min_value=0.0000001,
        max_value=0.001,
        value=st.session_state.get(f"{key_prefix}_omega", defaults.get("omega", 0.000001)),
        step=0.0000001,
        format="%.7f",
        key=f"{key_prefix}_omega_input",
        help="Variance constant term"
    )
    st.session_state[f"{key_prefix}_omega"] = params["omega"]

    col1, col2 = st.columns(2)

    with col1:
        params["alpha"] = st.slider(
            "ARCH Coef (α)",
            min_value=0.001,
            max_value=0.5,
            value=st.session_state.get(f"{key_prefix}_alpha", defaults.get("alpha", 0.05)),
            step=0.01,
            format="%.3f",
            key=f"{key_prefix}_alpha_input",
            help="Response to past shocks"
        )
        st.session_state[f"{key_prefix}_alpha"] = params["alpha"]

    with col2:
        params["beta"] = st.slider(
            "GARCH Coef (β)",
            min_value=0.0,
            max_value=0.99,
            value=st.session_state.get(f"{key_prefix}_beta", defaults.get("beta", 0.90)),
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_beta_input",
            help="Persistence of volatility"
        )
        st.session_state[f"{key_prefix}_beta"] = params["beta"]

    return params


def _render_ngarch_params(defaults: Dict, key_prefix: str) -> Dict[str, float]:
    """Render NGARCH parameters."""
    params = _render_garch_params(defaults, key_prefix)

    # Leverage parameter
    params["theta_ngarch"] = st.slider(
        "Leverage (θ)",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.get(f"{key_prefix}_theta_ng", defaults.get("theta", 0.5)),
        step=0.1,
        format="%.1f",
        key=f"{key_prefix}_theta_ng_input",
        help="Leverage effect parameter"
    )
    st.session_state[f"{key_prefix}_theta_ng"] = params["theta_ngarch"]

    return params


def _render_gjr_garch_params(defaults: Dict, key_prefix: str) -> Dict[str, float]:
    """Render GJR-GARCH parameters."""
    params = _render_garch_params(defaults, key_prefix)

    # Asymmetry parameter
    params["gamma"] = st.slider(
        "Asymmetry (γ)",
        min_value=0.0,
        max_value=0.3,
        value=st.session_state.get(f"{key_prefix}_gamma", defaults.get("gamma", 0.05)),
        step=0.01,
        format="%.3f",
        key=f"{key_prefix}_gamma_input",
        help="Asymmetry for negative shocks"
    )
    st.session_state[f"{key_prefix}_gamma"] = params["gamma"]

    return params


def _render_feller_warning(params: Dict[str, float]):
    """Render Feller condition warning."""
    satisfied, lhs, rhs = check_feller_condition(params)

    if satisfied:
        st.success(f"✓ Feller: 2κθ = {lhs:.4f} > {rhs:.4f} = ξ²")
    else:
        st.warning(
            f"⚠️ Feller violated: 2κθ = {lhs:.4f} ≤ {rhs:.4f} = ξ²\n\n"
            "Variance may become negative. Full Truncation scheme will be used."
        )


def _render_stationarity_warning(model_key: str, params: Dict[str, float]):
    """Render stationarity condition warning."""
    stationary, persistence = check_garch_stationarity(model_key, params)

    if stationary:
        st.success(f"✓ Stationary: persistence = {persistence:.4f} < 1")
    else:
        st.warning(
            f"⚠️ Non-stationary: persistence = {persistence:.4f} ≥ 1\n\n"
            "Volatility may explode. Consider reducing α or β."
        )


def render_simulation_settings(key_prefix: str = "sim") -> Dict[str, Any]:
    """
    Render simulation settings section.

    Returns:
        Dictionary of simulation settings
    """
    st.subheader("🎲 Simulation Settings")

    settings = {}

    col1, col2 = st.columns(2)

    with col1:
        settings["n_paths"] = st.number_input(
            "Number of Paths",
            min_value=100,
            max_value=100000,
            value=st.session_state.get(f"{key_prefix}_paths", 10000),
            step=100,
            key=f"{key_prefix}_paths_input",
            help="Number of Monte Carlo paths"
        )
        st.session_state[f"{key_prefix}_paths"] = settings["n_paths"]

    with col2:
        settings["n_steps"] = st.number_input(
            "Time Steps",
            min_value=10,
            max_value=1000,
            value=st.session_state.get(f"{key_prefix}_steps", 252),
            step=1,
            key=f"{key_prefix}_steps_input",
            help="Steps per path (252 = daily)"
        )
        st.session_state[f"{key_prefix}_steps"] = settings["n_steps"]

    settings["seed"] = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=99999,
        value=st.session_state.get(f"{key_prefix}_seed", 42),
        step=1,
        key=f"{key_prefix}_seed_input",
        help="0 = random seed"
    )
    st.session_state[f"{key_prefix}_seed"] = settings["seed"]

    return settings


def render_option_parameters(key_prefix: str = "option") -> Dict[str, Any]:
    """
    Render option parameters for pricing comparison.

    Returns:
        Dictionary of option parameters
    """
    st.subheader("📜 Option Parameters")

    params = {}

    col1, col2 = st.columns(2)

    with col1:
        params["strike"] = st.number_input(
            "Strike Price (K)",
            min_value=1.0,
            max_value=10000.0,
            value=st.session_state.get(f"{key_prefix}_strike", 100.0),
            step=1.0,
            key=f"{key_prefix}_strike_input",
            help="Option strike price"
        )
        st.session_state[f"{key_prefix}_strike"] = params["strike"]

    with col2:
        params["is_call"] = st.selectbox(
            "Option Type",
            options=[True, False],
            format_func=lambda x: "Call" if x else "Put",
            index=0 if st.session_state.get(f"{key_prefix}_is_call", True) else 1,
            key=f"{key_prefix}_type_input",
        )
        st.session_state[f"{key_prefix}_is_call"] = params["is_call"]

    return params


def render_parameter_panel(
    model_key: str,
    key_prefix: str = "param"
) -> Dict[str, Any]:
    """
    Render complete parameter panel for a model.

    Args:
        model_key: Selected model key
        key_prefix: Prefix for widget keys

    Returns:
        Dictionary with all parameters (market, model, simulation, option)
    """
    all_params = {}

    # Market parameters
    market_params = render_market_parameters(f"{key_prefix}_market")
    all_params.update(market_params)

    st.markdown("---")

    # Model-specific parameters
    model_params = render_model_parameters(model_key, f"{key_prefix}_model")
    all_params.update(model_params)

    st.markdown("---")

    # Simulation settings
    sim_settings = render_simulation_settings(f"{key_prefix}_sim")
    all_params.update(sim_settings)

    return all_params
