"""
Monte Carlo Simulation Explorer - Main Application

A high-performance educational tool for Monte Carlo simulation of price paths
and volatility dynamics. Features multiple stochastic models including GBM,
Heston, SABR, and GARCH-family volatility models.

Author: Thomas Vaudescal
"""

import sys
from pathlib import Path

# Add paths for imports
app_dir = Path(__file__).parent
project_root = app_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(app_dir))

import streamlit as st
import numpy as np

# Backend imports
from backend.simulation.simulate_paths import SimulationResult
from backend.simulation.simulate_volatility import VolatilitySimulationResult
from backend.simulation.pnl_engine import RiskMetrics

# Local imports
from config.styles import (
    inject_styles,
    render_compact_header,
    footer_html,
    stale_results_warning_html,
    metric_card_html
)
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
from charts.price_paths import render_price_paths_tab
from charts.volatility_paths import render_volatility_paths_tab
from charts.distributions import render_distributions_tab
from charts.statistics import render_statistics_tab
from charts.interactive_path import render_interactive_path_tab
from charts.pnl_distribution import render_pnl_distribution_tab, render_risk_metrics_tab
from charts.scenario_analysis import render_scenario_analysis_tab
from services.state_manager import (
    init_session_state,
    mark_results_current,
    are_results_stale,
    check_params_changed
)
from services.simulation_runner import (
    run_price_simulation,
    run_volatility_simulation,
    calculate_pnl_from_paths
)

# Black-Scholes for premium calculation
from backend.option_pricing.options_calculator import (
    black_scholes_call_price,
    black_scholes_put_price
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Monte Carlo Simulation Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS styles
inject_styles()

# Initialize session state
init_session_state()


# =============================================================================
# HEADER (Compact)
# =============================================================================

render_compact_header(
    title="Monte Carlo Simulation Explorer",
    subtitle="Interactive visualization of stochastic price and volatility simulations",
    badge="Educational Tool"
)


# =============================================================================
# MAIN TABS NAVIGATION
# =============================================================================

tab_price, tab_volatility = st.tabs([
    "📈 Price & Option P&L",
    "📊 Volatility Paths"
])


# =============================================================================
# SIDEBAR CONFIGURATION FUNCTIONS
# =============================================================================

def render_market_params(key_prefix: str, include_expected_return: bool = False) -> dict:
    """Render market parameters with unique keys."""
    params = {}

    st.markdown("### 📊 Market Parameters")

    col1, col2 = st.columns(2)
    with col1:
        spot_price = st.number_input(
            "S₀ (Spot)",
            min_value=1.0,
            max_value=10000.0,
            value=DEFAULT_SPOT_PRICE,
            step=1.0,
            format="%.2f",
            key=f"{key_prefix}_spot"
        )
    with col2:
        volatility = st.number_input(
            "σ (Volatility)",
            min_value=0.01,
            max_value=1.0,
            value=DEFAULT_VOLATILITY,
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_vol"
        )

    col1, col2 = st.columns(2)
    with col1:
        time_horizon = st.number_input(
            "T (Years)",
            min_value=0.1,
            max_value=10.0,
            value=DEFAULT_TIME_HORIZON,
            step=0.1,
            format="%.1f",
            key=f"{key_prefix}_time"
        )
    with col2:
        risk_free_rate = st.number_input(
            "r (Risk-free)",
            min_value=0.0,
            max_value=0.20,
            value=DEFAULT_RISK_FREE_RATE,
            step=0.005,
            format="%.3f",
            key=f"{key_prefix}_rate"
        )

    # Expected return (only for price simulation)
    if include_expected_return:
        expected_return = st.number_input(
            "μ (Expected Return)",
            min_value=-0.20,
            max_value=0.50,
            value=DEFAULT_EXPECTED_RETURN,
            step=0.01,
            format="%.2f",
            help="Annual expected return under P-measure",
            key=f"{key_prefix}_mu"
        )
        params['expected_return'] = expected_return

    params.update({
        'spot_price': spot_price,
        'volatility': volatility,
        'time_horizon': time_horizon,
        'risk_free_rate': risk_free_rate
    })

    return params


def render_simulation_settings(key_prefix: str) -> dict:
    """Render simulation settings with unique keys."""
    st.markdown("### ⚡ Simulation Settings")

    col1, col2 = st.columns(2)
    with col1:
        num_paths = st.number_input(
            "Paths",
            min_value=10,
            max_value=100000,
            value=DEFAULT_NUM_PATHS,
            step=100,
            key=f"{key_prefix}_paths"
        )
    with col2:
        num_steps = st.number_input(
            "Steps",
            min_value=10,
            max_value=1000,
            value=DEFAULT_NUM_STEPS,
            step=10,
            key=f"{key_prefix}_steps"
        )

    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=99999,
        value=42,
        step=1,
        key=f"{key_prefix}_seed"
    )

    return {
        'num_paths': int(num_paths),
        'num_steps': int(num_steps),
        'seed': int(seed)
    }


def render_price_model_config(base_volatility: float, key_prefix: str) -> dict:
    """Render price model configuration with unique keys."""
    params = {}

    st.markdown("### 🔬 Price Model")

    price_model = st.selectbox(
        "Select Model",
        options=list(PRICE_MODELS.keys()),
        format_func=lambda x: PRICE_MODELS[x],
        key=f"{key_prefix}_model_select"
    )
    params['price_model'] = price_model

    # Model-specific parameters
    if price_model == "heston":
        with st.expander("Heston Parameters", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                params['heston_v0'] = st.number_input(
                    "V₀", min_value=0.001, max_value=1.0,
                    value=base_volatility ** 2, step=0.01, format="%.4f",
                    key=f"{key_prefix}_h_v0"
                )
                params['heston_kappa'] = st.number_input(
                    "κ (Mean Rev.)", min_value=0.1, max_value=10.0,
                    value=2.0, step=0.1, format="%.2f", key=f"{key_prefix}_h_kappa"
                )
            with col2:
                params['heston_theta'] = st.number_input(
                    "θ (Long-term)", min_value=0.001, max_value=1.0,
                    value=base_volatility ** 2, step=0.01, format="%.4f",
                    key=f"{key_prefix}_h_theta"
                )
                params['heston_xi'] = st.number_input(
                    "ξ (Vol of Vol)", min_value=0.01, max_value=1.0,
                    value=0.3, step=0.05, format="%.2f", key=f"{key_prefix}_h_xi"
                )
            params['heston_rho'] = st.slider(
                "ρ (Correlation)", -0.99, 0.99, -0.7, 0.01, key=f"{key_prefix}_h_rho"
            )
            if abs(params['heston_rho']) > 0.95:
                st.info("ℹ️ Extreme correlation may cause numerical instability")
            feller = 2 * params['heston_kappa'] * params['heston_theta'] / (params['heston_xi'] ** 2)
            if feller < 1:
                st.warning(f"⚠️ Feller: {feller:.2f} < 1")

    elif price_model == "merton":
        with st.expander("Merton Jump Parameters", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                params['merton_lambda'] = st.number_input(
                    "λ (Intensity)", min_value=0.0, max_value=5.0,
                    value=0.5, step=0.1, format="%.2f", key=f"{key_prefix}_m_lambda"
                )
                params['merton_mu_j'] = st.number_input(
                    "μⱼ (Mean)", min_value=-0.5, max_value=0.5,
                    value=-0.1, step=0.01, format="%.3f", key=f"{key_prefix}_m_mu_j"
                )
            with col2:
                params['merton_sigma_j'] = st.number_input(
                    "σⱼ (Vol)", min_value=0.01, max_value=0.5,
                    value=0.2, step=0.01, format="%.2f", key=f"{key_prefix}_m_sigma_j"
                )

    elif price_model == "bates":
        with st.expander("Bates Parameters", expanded=True):
            st.markdown("**Stochastic Vol**")
            col1, col2 = st.columns(2)
            with col1:
                params['bates_v0'] = st.number_input(
                    "V₀", min_value=0.001, max_value=1.0,
                    value=base_volatility ** 2, step=0.01, format="%.4f",
                    key=f"{key_prefix}_b_v0"
                )
                params['bates_kappa'] = st.number_input(
                    "κ", min_value=0.1, max_value=10.0,
                    value=2.0, step=0.1, format="%.2f", key=f"{key_prefix}_b_kappa"
                )
            with col2:
                params['bates_theta'] = st.number_input(
                    "θ", min_value=0.001, max_value=1.0,
                    value=base_volatility ** 2, step=0.01, format="%.4f",
                    key=f"{key_prefix}_b_theta"
                )
                params['bates_xi'] = st.number_input(
                    "ξ", min_value=0.01, max_value=1.0,
                    value=0.3, step=0.05, format="%.2f", key=f"{key_prefix}_b_xi"
                )
            params['bates_rho'] = st.slider(
                "ρ", -0.99, 0.99, -0.7, 0.01, key=f"{key_prefix}_b_rho"
            )
            if abs(params['bates_rho']) > 0.95:
                st.info("ℹ️ Extreme correlation may cause numerical instability")
            feller = 2 * params['bates_kappa'] * params['bates_theta'] / (params['bates_xi'] ** 2)
            if feller < 1:
                st.warning(f"⚠️ Feller: {feller:.2f} < 1")

            st.markdown("**Jumps**")
            col1, col2 = st.columns(2)
            with col1:
                params['bates_lambda'] = st.number_input(
                    "λ", min_value=0.0, max_value=5.0,
                    value=0.5, step=0.1, format="%.2f", key=f"{key_prefix}_b_lambda"
                )
                params['bates_mu_j'] = st.number_input(
                    "μⱼ", min_value=-0.5, max_value=0.5,
                    value=-0.1, step=0.01, format="%.3f", key=f"{key_prefix}_b_mu_j"
                )
            with col2:
                params['bates_sigma_j'] = st.number_input(
                    "σⱼ", min_value=0.01, max_value=0.5,
                    value=0.2, step=0.01, format="%.2f", key=f"{key_prefix}_b_sigma_j"
                )

    elif price_model == "sabr":
        with st.expander("SABR Parameters", expanded=True):
            params['sabr_beta'] = st.slider(
                "β (CEV)", 0.0, 1.0, 0.5, 0.1, key=f"{key_prefix}_s_beta",
                help="0=Normal, 1=Lognormal"
            )
            col1, col2 = st.columns(2)
            with col1:
                params['sabr_nu'] = st.number_input(
                    "ν (Vol of Vol)", min_value=0.01, max_value=1.0,
                    value=0.4, step=0.05, format="%.2f", key=f"{key_prefix}_s_nu"
                )
            with col2:
                params['sabr_rho'] = st.slider(
                    "ρ", -0.99, 0.99, -0.3, 0.01, key=f"{key_prefix}_s_rho"
                )
            if abs(params['sabr_rho']) > 0.95:
                st.info("ℹ️ Extreme correlation may cause numerical instability")

    return params


def render_volatility_model_config(base_volatility: float, key_prefix: str) -> dict:
    """Render volatility model configuration with unique keys."""
    params = {}

    st.markdown("### 🔬 Volatility Model")

    vol_model = st.selectbox(
        "Select Model",
        options=list(VOLATILITY_MODELS.keys()),
        format_func=lambda x: VOLATILITY_MODELS[x],
        key=f"{key_prefix}_vol_model_select"
    )
    params['vol_model'] = vol_model

    with st.expander(f"{VOLATILITY_MODELS[vol_model]} Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            params['garch_alpha'] = st.number_input(
                "α (ARCH)", min_value=0.001, max_value=0.5,
                value=0.05, step=0.01, format="%.3f", key=f"{key_prefix}_g_alpha"
            )
        with col2:
            params['garch_beta'] = st.number_input(
                "β (GARCH)", min_value=0.0, max_value=0.99,
                value=0.90, step=0.01, format="%.3f", key=f"{key_prefix}_g_beta"
            )

        if vol_model == "ngarch":
            params['ngarch_theta'] = st.number_input(
                "θ (Leverage)", min_value=0.0, max_value=2.0,
                value=0.5, step=0.1, format="%.2f", key=f"{key_prefix}_ng_theta"
            )
            persistence = params['garch_alpha'] * (1 + params['ngarch_theta'] ** 2) + params['garch_beta']
            if persistence >= 1:
                st.error(f"Non-stationary: {persistence:.3f} ≥ 1")
            else:
                st.success(f"Stationary: {persistence:.3f}")

        elif vol_model == "gjr_garch":
            params['gjr_gamma'] = st.number_input(
                "γ (Asymmetry)", min_value=0.0, max_value=0.3,
                value=0.05, step=0.01, format="%.3f", key=f"{key_prefix}_gjr_gamma"
            )
            persistence = params['garch_alpha'] + params['garch_beta'] + 0.5 * params['gjr_gamma']
            if persistence >= 1:
                st.error(f"Non-stationary: {persistence:.3f} ≥ 1")
            else:
                st.success(f"Stationary: {persistence:.3f}")

        elif vol_model == "egarch":
            params['egarch_gamma'] = st.number_input(
                "γ (Asymmetry)", min_value=-0.5, max_value=0.5,
                value=-0.1, step=0.01, format="%.3f", key=f"{key_prefix}_eg_gamma"
            )
            if abs(params['garch_beta']) >= 1:
                st.error(f"Non-stationary: |β| ≥ 1")
            else:
                st.success(f"Stationary: |β| = {abs(params['garch_beta']):.3f}")

        else:  # Standard GARCH
            persistence = params['garch_alpha'] + params['garch_beta']
            if persistence >= 1:
                st.error(f"Non-stationary: {persistence:.3f} ≥ 1")
            else:
                st.success(f"Stationary: {persistence:.3f}")

        # Compute omega
        alpha_beta = params['garch_alpha'] + params['garch_beta']
        if alpha_beta < 1:
            params['garch_omega'] = base_volatility ** 2 * (1 - alpha_beta)

    return params


def render_strategy_builder_sidebar(
    spot_price: float,
    risk_free_rate: float,
    time_horizon: float,
    volatility: float
) -> dict:
    """Render strategy builder in sidebar."""
    from components.strategy_builder import (
        render_strategy_builder,
        export_positions_for_pnl_engine
    )

    st.markdown("### 🎯 Option Strategy")

    def bs_price(s, k, r, t, sigma, opt_type):
        if opt_type == 'call':
            return black_scholes_call_price(s, k, t, r, sigma)
        else:
            return black_scholes_put_price(s, k, t, r, sigma)

    positions, stock_position = render_strategy_builder(
        spot_price=spot_price,
        risk_free_rate=risk_free_rate,
        time_to_expiry=time_horizon,
        volatility=volatility,
        bs_price_function=bs_price
    )

    position_arrays = export_positions_for_pnl_engine(positions, stock_position)

    return {
        'option_positions': positions,
        'stock_position': stock_position,
        'position_arrays': position_arrays
    }


def render_visualization_options(key_prefix: str) -> dict:
    """Render visualization options with unique keys."""
    st.markdown("### 🎨 Visualization")

    show_percentiles = st.checkbox("Percentile bands", value=True, key=f"{key_prefix}_viz_percentiles")
    show_mean = st.checkbox("Mean path", value=True, key=f"{key_prefix}_viz_mean")
    max_display = st.slider("Max paths", 10, 200, 50, 10, key=f"{key_prefix}_viz_max_paths")

    return {
        'show_percentiles': show_percentiles,
        'show_mean': show_mean,
        'max_display_paths': max_display
    }


# =============================================================================
# SIDEBAR - COMMON FOR BOTH TABS (rendered once)
# =============================================================================

with st.sidebar:
    # Market parameters with expected return
    market_params = render_market_params("price", include_expected_return=True)

    # Simulation settings
    sim_params = render_simulation_settings("sim")

    # Price model configuration
    price_model_params = render_price_model_config(market_params['volatility'], "price")

    # Strategy builder
    strategy_params = render_strategy_builder_sidebar(
        market_params['spot_price'],
        market_params['risk_free_rate'],
        market_params['time_horizon'],
        market_params['volatility']
    )

    st.markdown("---")

    # Volatility model configuration
    vol_model_params = render_volatility_model_config(market_params['volatility'], "vol")

    st.markdown("---")

    # Visualization options
    viz_params = render_visualization_options("viz")

    # Combine params for price tab
    price_params = {
        **market_params,
        **sim_params,
        **price_model_params,
        **strategy_params,
        **viz_params
    }
    st.session_state.price_params = price_params

    # Combine params for volatility tab
    vol_params = {
        **market_params,
        **sim_params,
        **vol_model_params,
        **viz_params
    }
    st.session_state.vol_params = vol_params

    check_params_changed(price_params)


# =============================================================================
# PRICE & OPTION P&L TAB
# =============================================================================

with tab_price:
    params = st.session_state.get('price_params', {})

    # Check for stale results
    price_result = st.session_state.get('price_result')
    is_stale = price_result is not None and are_results_stale('price')

    if is_stale:
        st.markdown(stale_results_warning_html(), unsafe_allow_html=True)

    # Run button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_sim = st.button(
            "🚀 Run Simulation",
            type="primary",
            width="stretch",
            key="run_price_pnl"
        )

    if run_sim:
        with st.spinner('Running simulation...'):
            try:
                result = run_price_simulation(params)
                st.session_state.price_result = result
                mark_results_current('price')
                price_result = result

                # Also calculate P&L if strategy is defined
                pnl_result = calculate_pnl_from_paths(result, params)
                if pnl_result is not None:
                    st.session_state.pnl_result = pnl_result
                    mark_results_current('pnl')

                st.success(f"Simulation completed in {result.computation_time*1000:.1f} ms")
            except Exception as e:
                st.error(f"Simulation error: {str(e)}")
                st.exception(e)

    if price_result is not None:
        # Summary metrics with styled cards
        st.markdown("---")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(metric_card_html(
                "Model",
                PRICE_MODELS.get(params.get('price_model', 'gbm'), 'GBM')
            ), unsafe_allow_html=True)
        with mc2:
            st.markdown(metric_card_html(
                "Paths",
                f"{price_result.num_paths:,}"
            ), unsafe_allow_html=True)
        with mc3:
            st.markdown(metric_card_html(
                "Steps",
                f"{price_result.num_steps:,}"
            ), unsafe_allow_html=True)
        with mc4:
            st.markdown(metric_card_html(
                "Computation Time",
                f"{price_result.computation_time*1000:.1f} ms"
            ), unsafe_allow_html=True)

        # Check if we have option positions for P&L tab
        has_positions = len(params.get('option_positions', [])) > 0
        pnl_result = st.session_state.get('pnl_result')

        show_variance = params.get('price_model') in ['heston', 'sabr', 'bates']

        # Sub-tabs based on whether we have P&L data
        if has_positions and pnl_result is not None:
            sub_tabs = st.tabs([
                "🎛️ Interactive",
                "📈 Price Paths",
                "📊 Distribution",
                "📋 Statistics",
                "💰 P&L Distribution",
                "📋 Risk Metrics",
                "🎯 Scenarios"
            ])

            with sub_tabs[0]:
                render_interactive_path_tab(params=params, key_prefix="price_pnl")

            with sub_tabs[1]:
                render_price_paths_tab(
                    simulation_result=price_result,
                    params=params,
                    show_variance_paths=show_variance
                )

            with sub_tabs[2]:
                render_distributions_tab(
                    simulation_result=price_result,
                    params=params,
                    result_type="price"
                )

            with sub_tabs[3]:
                render_statistics_tab(
                    simulation_result=price_result,
                    params=params,
                    result_type="price"
                )

            with sub_tabs[4]:
                render_pnl_distribution_tab(
                    pnl_values=pnl_result['pnl_values'],
                    risk_metrics=pnl_result['risk_metrics'],
                    params=params
                )

            with sub_tabs[5]:
                render_risk_metrics_tab(
                    metrics=pnl_result['risk_metrics'],
                    params=params
                )

            with sub_tabs[6]:
                render_scenario_analysis_tab(
                    terminal_prices=pnl_result['terminal_prices'],
                    pnl_values=pnl_result['pnl_values'],
                    position_arrays=params.get('position_arrays', {}),
                    risk_metrics=pnl_result['risk_metrics'],
                    params=params
                )
        else:
            # No P&L - just price tabs
            sub_tabs = st.tabs([
                "🎛️ Interactive",
                "📈 Price Paths",
                "📊 Distribution",
                "📋 Statistics"
            ])

            with sub_tabs[0]:
                render_interactive_path_tab(params=params, key_prefix="price_only")

            with sub_tabs[1]:
                render_price_paths_tab(
                    simulation_result=price_result,
                    params=params,
                    show_variance_paths=show_variance
                )

            with sub_tabs[2]:
                render_distributions_tab(
                    simulation_result=price_result,
                    params=params,
                    result_type="price"
                )

            with sub_tabs[3]:
                render_statistics_tab(
                    simulation_result=price_result,
                    params=params,
                    result_type="price"
                )

            if has_positions:
                st.info("💡 Option strategy defined. Run simulation to see P&L analysis.")
    else:
        st.info("Configure parameters in the sidebar and click **Run Simulation**.")

        # Show strategy summary if defined
        positions = params.get('option_positions', [])
        if len(positions) > 0:
            from streamlit_app.option_pricer.config.constants import CONTRACT_MULTIPLIER

            total_net = 0.0
            for pos in positions:
                cost = pos.premium * pos.quantity * CONTRACT_MULTIPLIER
                if pos.position_type == 'long':
                    total_net -= cost
                else:
                    total_net += cost

            st.markdown(
                f"**Strategy configured:** {len(positions)} leg(s) | "
                f"Net: {'+'if total_net >= 0 else ''}${total_net:,.0f}"
            )


# =============================================================================
# VOLATILITY PATHS TAB
# =============================================================================

with tab_volatility:
    vol_params = st.session_state.get('vol_params', {})

    # Check for stale results
    vol_result = st.session_state.get('vol_result')
    is_stale = vol_result is not None and are_results_stale('volatility')

    if is_stale:
        st.markdown(stale_results_warning_html(), unsafe_allow_html=True)

    # Run button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_vol = st.button(
            "🚀 Run Volatility Simulation",
            type="primary",
            width="stretch",
            key="run_vol"
        )

    if run_vol:
        with st.spinner('Running volatility simulation...'):
            try:
                result = run_volatility_simulation(vol_params)
                st.session_state.vol_result = result
                mark_results_current('volatility')
                vol_result = result
                st.success(f"Simulation completed in {result.computation_time*1000:.1f} ms")
            except Exception as e:
                st.error(f"Simulation error: {str(e)}")
                st.exception(e)

    if vol_result is not None:
        # Summary metrics with styled cards
        st.markdown("---")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(metric_card_html(
                "Model",
                VOLATILITY_MODELS.get(vol_params.get('vol_model', 'garch'), 'GARCH')
            ), unsafe_allow_html=True)
        with mc2:
            st.markdown(metric_card_html(
                "Paths",
                f"{vol_result.num_paths:,}"
            ), unsafe_allow_html=True)
        with mc3:
            st.markdown(metric_card_html(
                "Steps",
                f"{vol_result.num_steps:,}"
            ), unsafe_allow_html=True)
        with mc4:
            st.markdown(metric_card_html(
                "Computation Time",
                f"{vol_result.computation_time*1000:.1f} ms"
            ), unsafe_allow_html=True)

        sub_tabs = st.tabs([
            "🎛️ Interactive",
            "📈 Volatility Paths",
            "📊 Distribution",
            "📋 Statistics"
        ])

        with sub_tabs[0]:
            render_interactive_path_tab(params=vol_params, key_prefix="vol")

        with sub_tabs[1]:
            render_volatility_paths_tab(
                simulation_result=vol_result,
                params=vol_params
            )

        with sub_tabs[2]:
            render_distributions_tab(
                simulation_result=vol_result,
                params=vol_params,
                result_type="volatility"
            )

        with sub_tabs[3]:
            render_statistics_tab(
                simulation_result=vol_result,
                params=vol_params,
                result_type="volatility"
            )
    else:
        st.info("Configure parameters in the sidebar and click **Run Volatility Simulation**.")


# =============================================================================
# MODEL EQUATIONS (Expander)
# =============================================================================

with st.expander("ℹ️ Model Equations", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Price Models")

        st.markdown("**GBM**")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")

        st.markdown("**Heston**")
        st.latex(r"dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^{(1)}")
        st.latex(r"dv_t = \kappa(\theta - v_t)dt + \xi\sqrt{v_t}dW_t^{(2)}")

        st.markdown("**Merton Jump-Diffusion**")
        st.latex(r"dS_t = (\mu - \lambda\bar{k})S_t dt + \sigma S_t dW_t + S_t dJ_t")

        st.markdown("**Bates**")
        st.latex(r"dS_t = (\mu - \lambda\bar{k})S_t dt + \sqrt{v_t} S_t dW_t^{(1)} + S_t dJ_t")
        st.latex(r"dv_t = \kappa(\theta - v_t)dt + \xi\sqrt{v_t}dW_t^{(2)}")

        st.markdown("**SABR**")
        st.latex(r"dF_t = \sigma_t F_t^\beta dW_t^{(1)}")
        st.latex(r"d\sigma_t = \nu \sigma_t dW_t^{(2)}")

    with col2:
        st.markdown("### Volatility Models")

        st.markdown("**GARCH(1,1)**")
        st.latex(r"\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2")

        st.markdown("**NGARCH**")
        st.latex(r"\sigma_t^2 = \omega + \alpha(\epsilon_{t-1} - \theta\sigma_{t-1})^2 + \beta \sigma_{t-1}^2")

        st.markdown("**GJR-GARCH**")
        st.latex(r"\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \gamma \epsilon_{t-1}^2 \mathbf{1}_{(\epsilon<0)} + \beta \sigma_{t-1}^2")

        st.markdown("**EGARCH**")
        st.latex(r"\ln(\sigma_t^2) = \omega + \alpha|z_{t-1}| + \gamma z_{t-1} + \beta \ln(\sigma_{t-1}^2)")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(footer_html(), unsafe_allow_html=True)
