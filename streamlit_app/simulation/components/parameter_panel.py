"""
Parameter Panel Component - Dynamic parameter inputs based on model.

Provides:
- Market parameters (common to all models)
- Model-specific parameters (dynamic)
- Simulation settings
- Condition validation warnings
"""

from typing import Any

import streamlit as st
from config.model_registry import (
    get_model,
    get_parameter_defaults,
)
from utils.model_helpers import (
    check_feller_condition,
    check_garch_stationarity,
)


def render_market_parameters(key_prefix: str = "market") -> dict[str, float]:
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
        help="Current asset price",
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
            help="Annualized risk-free rate",
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
            help="Annualized expected return",
        )
        st.session_state[f"{key_prefix}_drift"] = params["drift"]

    # Dividend yield and time horizon
    col3, col4 = st.columns(2)

    with col3:
        params["dividend_yield"] = st.number_input(
            "Dividend Yield (q)",
            min_value=0.0,
            max_value=0.20,
            value=st.session_state.get(f"{key_prefix}_div", 0.0),
            step=0.005,
            format="%.3f",
            key=f"{key_prefix}_div_input",
            help="Annual continuous dividend yield (reduces price drift to μ − q)",
        )
        st.session_state[f"{key_prefix}_div"] = params["dividend_yield"]

    with col4:
        params["time_horizon"] = st.number_input(
            "Time Horizon (T)",
            min_value=0.1,
            max_value=10.0,
            value=st.session_state.get(f"{key_prefix}_time", 1.0),
            step=0.1,
            format="%.1f",
            key=f"{key_prefix}_time_input",
            help="Simulation horizon in years",
        )
        st.session_state[f"{key_prefix}_time"] = params["time_horizon"]

    return params


def render_model_parameters(
    model_key: str, key_prefix: str = "model"
) -> dict[str, float]:
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

    else:
        params.update(_render_dynamic_params(model_key, defaults, key_prefix))

    return params


def _render_gbm_params(defaults: dict, key_prefix: str) -> dict[str, float]:
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
        help="Annualized volatility (0.20 = 20%)",
    )
    st.session_state[f"{key_prefix}_sigma"] = params["sigma"]

    # Display as percentage
    st.caption(f"σ = {params['sigma'] * 100:.1f}%")

    return params


def _render_heston_params(defaults: dict, key_prefix: str) -> dict[str, float]:
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
        help="Initial variance level",
    )
    st.session_state[f"{key_prefix}_v0"] = params["v0"]
    st.caption(f"Initial vol = {params['v0'] ** 0.5 * 100:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        # Mean reversion speed
        params["kappa"] = st.slider(
            "Mean Reversion (κ)",
            min_value=0.1,
            max_value=10.0,
            value=st.session_state.get(
                f"{key_prefix}_kappa", defaults.get("kappa", 2.0)
            ),
            step=0.1,
            format="%.1f",
            key=f"{key_prefix}_kappa_input",
            help="Speed of variance mean reversion",
        )
        st.session_state[f"{key_prefix}_kappa"] = params["kappa"]

    with col2:
        # Long-run variance
        params["theta"] = st.slider(
            "Long-Run Var (σ²)",
            min_value=0.001,
            max_value=0.50,
            value=st.session_state.get(
                f"{key_prefix}_theta", defaults.get("theta", 0.04)
            ),
            step=0.005,
            format="%.3f",
            key=f"{key_prefix}_theta_input",
            help="Long-run variance level",
        )
        st.session_state[f"{key_prefix}_theta"] = params["theta"]
        st.caption(f"Long-run vol = {params['theta'] ** 0.5 * 100:.1f}%")

    col3, col4 = st.columns(2)

    with col3:
        # Vol of vol
        params["alpha"] = st.slider(
            "Vol of Vol (α)",
            min_value=0.01,
            max_value=1.0,
            value=st.session_state.get(f"{key_prefix}_xi", defaults.get("alpha", 0.3)),
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_xi_input",
            help="Volatility of variance",
        )
        st.session_state[f"{key_prefix}_xi"] = params["alpha"]

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
            help="Price-variance correlation",
        )
        st.session_state[f"{key_prefix}_rho"] = params["rho"]

    return params


def _render_merton_params(defaults: dict, key_prefix: str) -> dict[str, float]:
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
        help="Diffusion volatility",
    )
    st.session_state[f"{key_prefix}_sigma"] = params["sigma"]

    st.markdown("**Jump Parameters**")

    col1, col2 = st.columns(2)

    with col1:
        params["lam"] = st.slider(
            "Jump Intensity (λ)",
            min_value=0.0,
            max_value=5.0,
            value=st.session_state.get(
                f"{key_prefix}_lambda", defaults.get("lam", 0.5)
            ),
            step=0.1,
            format="%.1f",
            key=f"{key_prefix}_lambda_input",
            help="Expected jumps per year",
        )
        st.session_state[f"{key_prefix}_lambda"] = params["lam"]

    with col2:
        params["alpha_j"] = st.slider(
            "Mean Jump (α_J)",
            min_value=-0.5,
            max_value=0.5,
            value=st.session_state.get(f"{key_prefix}_muj", defaults.get("alpha_j", -0.1)),
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_muj_input",
            help="Mean of log-jump size",
        )
        st.session_state[f"{key_prefix}_muj"] = params["alpha_j"]

    params["sigma_j"] = st.slider(
        "Jump Vol (σ_J)",
        min_value=0.01,
        max_value=0.5,
        value=st.session_state.get(
            f"{key_prefix}_sigmaj", defaults.get("sigma_j", 0.2)
        ),
        step=0.01,
        format="%.2f",
        key=f"{key_prefix}_sigmaj_input",
        help="Volatility of log-jump size",
    )
    st.session_state[f"{key_prefix}_sigmaj"] = params["sigma_j"]

    return params


def _render_bates_params(defaults: dict, key_prefix: str) -> dict[str, float]:
    """Render Bates parameters (Heston + Merton jumps)."""
    params = {}

    # Heston part
    st.markdown("**Stochastic Volatility**")
    params.update(_render_heston_params(defaults, key_prefix))

    # Jump part
    st.markdown("**Jump Parameters**")

    col1, col2 = st.columns(2)

    with col1:
        params["lam"] = st.slider(
            "Jump Intensity (λ)",
            min_value=0.0,
            max_value=5.0,
            value=st.session_state.get(
                f"{key_prefix}_lambda", defaults.get("lam", 0.5)
            ),
            step=0.1,
            format="%.1f",
            key=f"{key_prefix}_lambda_input",
            help="Expected jumps per year",
        )
        st.session_state[f"{key_prefix}_lambda"] = params["lam"]

    with col2:
        params["alpha_j"] = st.slider(
            "Mean Jump (α_J)",
            min_value=-0.5,
            max_value=0.5,
            value=st.session_state.get(f"{key_prefix}_muj", defaults.get("alpha_j", -0.1)),
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_muj_input",
            help="Mean of log-jump size",
        )
        st.session_state[f"{key_prefix}_muj"] = params["alpha_j"]

    params["sigma_j"] = st.slider(
        "Jump Vol (σ_J)",
        min_value=0.01,
        max_value=0.5,
        value=st.session_state.get(
            f"{key_prefix}_sigmaj", defaults.get("sigma_j", 0.2)
        ),
        step=0.01,
        format="%.2f",
        key=f"{key_prefix}_sigmaj_input",
        help="Volatility of log-jump size",
    )
    st.session_state[f"{key_prefix}_sigmaj"] = params["sigma_j"]

    return params


def _render_garch_params(defaults: dict, key_prefix: str) -> dict[str, float]:
    """Render GARCH parameters.

    Theoretical constraints:
    - omega > 0  (intercept, drives long-run variance)
    - alpha >= 0, beta >= 0  (non-negativity)
    - alpha + beta < 1  (stationarity)
    - Long-run vol = sqrt(omega / (1 - alpha - beta))
    """
    params = {}

    params["sigma0"] = st.slider(
        "Initial Vol (σ₀)",
        min_value=0.01,
        max_value=2.0,
        value=st.session_state.get(
            f"{key_prefix}_sigma0", defaults.get("sigma0", 0.20)
        ),
        step=0.01,
        format="%.2f",
        key=f"{key_prefix}_sigma0_input",
        help="Initial volatility level",
    )
    st.session_state[f"{key_prefix}_sigma0"] = params["sigma0"]
    st.caption(f"σ₀ = {params['sigma0'] * 100:.1f}%")

    params["omega"] = st.number_input(
        "Constant (ω)",
        min_value=1e-7,
        max_value=0.1,
        value=st.session_state.get(f"{key_prefix}_omega", defaults.get("omega", 0.002)),
        step=0.0001,
        format="%.4f",
        key=f"{key_prefix}_omega_input",
        help="Variance intercept — long-run vol ≈ √(ω / (1 − α − β))",
    )
    st.session_state[f"{key_prefix}_omega"] = params["omega"]

    col1, col2 = st.columns(2)

    with col1:
        params["alpha"] = st.slider(
            "ARCH Coef (α)",
            min_value=0.0,
            max_value=0.50,
            value=st.session_state.get(
                f"{key_prefix}_alpha", defaults.get("alpha", 0.06)
            ),
            step=0.01,
            format="%.3f",
            key=f"{key_prefix}_alpha_input",
            help="Reaction to past shocks — must satisfy α + β < 1",
        )
        st.session_state[f"{key_prefix}_alpha"] = params["alpha"]

    with col2:
        params["beta"] = st.slider(
            "GARCH Coef (β)",
            min_value=0.0,
            max_value=0.99,
            value=st.session_state.get(
                f"{key_prefix}_beta", defaults.get("beta", 0.90)
            ),
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_beta_input",
            help="Volatility persistence — must satisfy α + β < 1",
        )
        st.session_state[f"{key_prefix}_beta"] = params["beta"]

    return params


def _render_ngarch_params(defaults: dict, key_prefix: str) -> dict[str, float]:
    """Render NGARCH parameters."""
    params = _render_garch_params(defaults, key_prefix)

    # Leverage parameter
    params["gamma_ngarch"] = st.slider(
        "Leverage (γ)",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.get(
            f"{key_prefix}_theta_ng", defaults.get("theta", 0.5)
        ),
        step=0.1,
        format="%.1f",
        key=f"{key_prefix}_theta_ng_input",
        help="Leverage effect parameter",
    )
    st.session_state[f"{key_prefix}_theta_ng"] = params["gamma_ngarch"]

    return params


def _render_gjr_garch_params(defaults: dict, key_prefix: str) -> dict[str, float]:
    """Render GJR-GARCH parameters."""
    params = _render_garch_params(defaults, key_prefix)

    # Asymmetry parameter
    params["gamma"] = st.slider(
        "Asymmetry (γ)",
        min_value=0.0,
        max_value=0.5,
        value=st.session_state.get(f"{key_prefix}_gamma", defaults.get("gamma", 0.03)),
        step=0.01,
        format="%.3f",
        key=f"{key_prefix}_gamma_input",
        help="Extra vol reaction to negative shocks — persistence = α + γ/2 + β",
    )
    st.session_state[f"{key_prefix}_gamma"] = params["gamma"]

    return params


def _render_dynamic_params(
    model_key: str, defaults: dict, key_prefix: str
) -> dict[str, float]:
    """Render parameters dynamically from ModelSpec (used for custom models)."""
    params = {}
    model = get_model(model_key)

    for p in model.parameters:
        params[p.name] = st.slider(
            p.display_name,
            min_value=float(p.min_value),
            max_value=float(p.max_value),
            value=st.session_state.get(
                f"{key_prefix}_{p.name}", float(defaults.get(p.name, p.default))
            ),
            step=float(p.step),
            format=p.format,
            key=f"{key_prefix}_{p.name}_input",
            help=p.description,
        )
        st.session_state[f"{key_prefix}_{p.name}"] = params[p.name]

    return params


def _render_feller_warning(params: dict[str, float]):
    """Render Feller condition warning."""
    satisfied, lhs, rhs = check_feller_condition(params)

    if satisfied:
        st.success(f"✓ Feller: 2κσ² = {lhs:.4f} > {rhs:.4f} = α²")
    else:
        st.warning(
            f"⚠️ Feller violated: 2κσ² = {lhs:.4f} ≤ {rhs:.4f} = α²\n\n"
            "Variance may become negative. Full Truncation scheme will be used."
        )


def _render_stationarity_warning(model_key: str, params: dict[str, float]):
    """Render stationarity condition warning with long-run volatility."""
    stationary, persistence = check_garch_stationarity(model_key, params)

    # Compute long-run vol (same formula as simulation_service.compute_long_run_volatility)
    omega = params.get("omega", 0.002)
    lr_vol_str = ""
    if stationary and (1 - persistence) > 0:
        lr_vol = (omega / (1 - persistence)) ** 0.5 * 100
        lr_vol_str = f" · LR vol ≈ {lr_vol:.1f}%"

    # Build persistence formula label
    model_lower = model_key.lower()
    if model_lower == "ngarch":
        formula = "α(1+γ²)+β"
    elif model_lower == "gjr_garch":
        formula = "α+β+γ/2"
    else:
        formula = "α+β"

    if stationary:
        st.success(f"✓ Stationary: {formula} = {persistence:.4f} < 1{lr_vol_str}")
    else:
        st.warning(
            f"⚠️ Non-stationary: {formula} = {persistence:.4f} ≥ 1\n\n"
            "Volatility may explode. Consider reducing α or β."
        )


def render_simulation_settings(key_prefix: str = "sim") -> dict[str, Any]:
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
            help="Number of Monte Carlo paths",
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
            help="Steps per path (252 = daily)",
        )
        st.session_state[f"{key_prefix}_steps"] = settings["n_steps"]

    settings["seed"] = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=99999,
        value=st.session_state.get(f"{key_prefix}_seed", 42),
        step=1,
        key=f"{key_prefix}_seed_input",
        help="0 = random seed",
    )
    st.session_state[f"{key_prefix}_seed"] = settings["seed"]

    return settings


def render_option_parameters(key_prefix: str = "option") -> dict[str, Any]:
    """
    Render option parameters for pricing comparison with options_greeks style.

    Returns:
        Dictionary of option parameters
    """
    # Get current values for styling
    current_type = st.session_state.get(f"{key_prefix}_type", "call")
    is_call_display = current_type == "call"

    # Dynamic styling based on option type
    border_color = "#10b981" if is_call_display else "#ef4444"
    bg_gradient = (
        "linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%)"
        if is_call_display
        else "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"
    )
    type_badge_bg = "#d1fae5" if is_call_display else "#fee2e2"
    type_badge_color = "#047857" if is_call_display else "#b91c1c"
    type_label = "CALL" if is_call_display else "PUT"

    # Styled header like options_greeks
    st.markdown(
        f"""
    <div style="background: {bg_gradient}; border: 1px solid {border_color}40; border-left: 4px solid {border_color}; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.625rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1rem;">📜</span>
                <span style="font-size: 0.7rem; font-weight: 700; color: #475569; text-transform: uppercase;">Option Parameters</span>
            </div>
            <span style="background: {type_badge_bg}; color: {type_badge_color}; font-size: 0.65rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; text-transform: uppercase;">{type_label}</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    params = {}

    # Option type selector with emoji format
    option_type = st.selectbox(
        "Type",
        options=["call", "put"],
        format_func=lambda x: f"{'📈' if x == 'call' else '📉'} {x.upper()}",
        index=0 if st.session_state.get(f"{key_prefix}_type", "call") == "call" else 1,
        key=f"{key_prefix}_type",
    )
    params["is_call"] = option_type == "call"

    # Strike price
    params["strike"] = st.number_input(
        "Strike ($)",
        min_value=1.0,
        max_value=10000.0,
        value=st.session_state.get(f"{key_prefix}_strike", 100.0),
        step=1.0,
        format="%.2f",
        key=f"{key_prefix}_strike_input",
        help="Option strike price",
    )
    st.session_state[f"{key_prefix}_strike"] = params["strike"]

    return params


def render_parameter_panel(model_key: str, key_prefix: str = "param") -> dict[str, Any]:
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
