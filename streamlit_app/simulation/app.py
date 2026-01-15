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
import time

# Backend imports
from backend.simulation.simulate_paths import (
    simulate_paths,
    SimulationResult
)
from backend.simulation.simulate_volatility import (
    simulate_volatility_paths,
    VolatilitySimulationResult
)

# Local imports
from config.styles import inject_styles, render_header, footer_html
from config.constants import (
    PRICE_MODELS,
    VOLATILITY_MODELS,
    MODEL_DESCRIPTIONS,
    SIMULATION_MODES
)
from components.sidebar import render_sidebar
from charts.price_paths import render_price_paths_tab
from charts.volatility_paths import render_volatility_paths_tab
from charts.distributions import render_distributions_tab
from charts.statistics import render_statistics_tab
from charts.interactive_path import render_interactive_path_tab
from services.state_manager import init_session_state


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
# SIDEBAR
# =============================================================================

params = render_sidebar()


# =============================================================================
# HEADER
# =============================================================================

render_header(
    title="Monte Carlo Simulation Explorer",
    subtitle="Interactive visualization of stochastic price and volatility simulations",
    badge="Educational Tool"
)


# =============================================================================
# SIMULATION EXECUTION
# =============================================================================

def run_price_simulation(params: dict) -> SimulationResult:
    """Execute price path simulation with given parameters."""
    model = params['price_model']

    # Common parameters (using backend parameter names)
    common_params = {
        's0': params['spot_price'],
        'r': params['risk_free_rate'],
        'sigma': params['volatility'],
        't': params['time_horizon'],
        'n_paths': params['num_paths'],
        'n_steps': params['num_steps'],
        'seed': params.get('seed')
    }

    # Model-specific parameters
    if model == 'heston':
        common_params.update({
            'v0': params['heston_v0'],
            'kappa': params['heston_kappa'],
            'theta': params['heston_theta'],
            'xi': params['heston_xi'],
            'rho': params['heston_rho']
        })
    elif model == 'merton':
        common_params.update({
            'lambda_j': params['merton_lambda'],
            'mu_j': params['merton_mu_j'],
            'sigma_j': params['merton_sigma_j']
        })
    elif model == 'sabr':
        # SABR uses alpha0 (initial vol), beta (CEV exponent), rho (correlation), nu (vol of vol)
        # alpha0 defaults to sigma in the backend if not provided
        common_params.update({
            'alpha0': params['volatility'],  # Use base volatility as initial alpha
            'beta': params['sabr_beta'],
            'rho': params['sabr_rho'],
            'nu': params['sabr_nu']
        })

    return simulate_paths(model=model, **common_params)


def run_volatility_simulation(params: dict) -> VolatilitySimulationResult:
    """Execute volatility simulation with given parameters."""
    model = params['vol_model']

    # Common parameters (using backend parameter names)
    common_params = {
        'sigma0': params['volatility'],
        'n_paths': params['num_paths'],
        'n_steps': params['num_steps'],
        'seed': params.get('seed')
    }

    # GARCH parameters
    garch_params = {
        'omega': params.get('garch_omega', 0.00001),
        'alpha': params['garch_alpha'],
        'beta': params['garch_beta']
    }

    if model == 'garch':
        common_params.update(garch_params)
    elif model == 'ngarch':
        common_params.update(garch_params)
        common_params['theta'] = params['ngarch_theta']
    elif model == 'gjr_garch':
        common_params.update(garch_params)
        common_params['gamma'] = params['gjr_gamma']
    elif model == 'egarch':
        # EGARCH uses same alpha/beta as other GARCH models, plus gamma for asymmetry
        common_params.update(garch_params)
        common_params['gamma'] = params['egarch_gamma']

    return simulate_volatility_paths(model=model, **common_params)


# =============================================================================
# MAIN CONTENT
# =============================================================================

# Simulation mode selection
simulation_mode = params.get('simulation_mode', 'price')

# Detect model/mode changes and reset results
current_price_model = params.get('price_model', 'gbm')
current_vol_model = params.get('vol_model', 'garch')

# Initialize tracking variables if not present
if 'prev_simulation_mode' not in st.session_state:
    st.session_state.prev_simulation_mode = simulation_mode
if 'prev_price_model' not in st.session_state:
    st.session_state.prev_price_model = current_price_model
if 'prev_vol_model' not in st.session_state:
    st.session_state.prev_vol_model = current_vol_model

# Check if mode or model changed
mode_changed = st.session_state.prev_simulation_mode != simulation_mode
price_model_changed = st.session_state.prev_price_model != current_price_model
vol_model_changed = st.session_state.prev_vol_model != current_vol_model

if mode_changed or price_model_changed or vol_model_changed:
    # Reset results when model changes
    st.session_state.price_result = None
    st.session_state.vol_result = None
    # Update tracking variables
    st.session_state.prev_simulation_mode = simulation_mode
    st.session_state.prev_price_model = current_price_model
    st.session_state.prev_vol_model = current_vol_model

# Run simulation button
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    run_button = st.button(
        "🚀 Run Simulation",
        width="stretch",
        type="primary"
    )

# Execute simulation
if run_button:
    with st.spinner('Running Monte Carlo simulation...'):
        start_time = time.time()

        try:
            if simulation_mode == 'price':
                result = run_price_simulation(params)
                st.session_state.price_result = result
                st.session_state.vol_result = None
            else:
                result = run_volatility_simulation(params)
                st.session_state.vol_result = result
                st.session_state.price_result = None

            elapsed = time.time() - start_time
            st.success(f"Simulation completed in {elapsed*1000:.1f} ms")

        except Exception as e:
            st.error(f"Simulation error: {str(e)}")
            st.exception(e)

# Get current results
price_result = st.session_state.get('price_result')
vol_result = st.session_state.get('vol_result')

# Display simulation info
if price_result is not None or vol_result is not None:
    st.markdown("---")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    if simulation_mode == 'price' and price_result is not None:
        with col1:
            st.metric("Model", PRICE_MODELS.get(params['price_model'], params['price_model']))
        with col2:
            st.metric("Paths Simulated", f"{price_result.num_paths:,}")
        with col3:
            st.metric("Time Steps", f"{price_result.num_steps:,}")
        with col4:
            st.metric("Computation Time", f"{price_result.computation_time*1000:.1f} ms")

    elif simulation_mode == 'volatility' and vol_result is not None:
        with col1:
            st.metric("Model", VOLATILITY_MODELS.get(params['vol_model'], params['vol_model']))
        with col2:
            st.metric("Paths Simulated", f"{vol_result.num_paths:,}")
        with col3:
            st.metric("Time Steps", f"{vol_result.num_steps:,}")
        with col4:
            st.metric("Computation Time", f"{vol_result.computation_time*1000:.1f} ms")


# =============================================================================
# VISUALIZATION TABS
# =============================================================================

if simulation_mode == 'price':
    # Price simulation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎛️ Interactive Path",
        "📈 Price Paths",
        "📊 Terminal Distribution",
        "📋 Statistics"
    ])

    # Check if we have stochastic volatility
    show_variance = params['price_model'] in ['heston', 'sabr']

    with tab1:
        render_interactive_path_tab(params=params)

    with tab2:
        render_price_paths_tab(
            simulation_result=price_result,
            params=params,
            show_variance_paths=show_variance
        )

    with tab3:
        render_distributions_tab(
            simulation_result=price_result,
            params=params,
            result_type="price"
        )

    with tab4:
        render_statistics_tab(
            simulation_result=price_result,
            params=params,
            result_type="price"
        )

else:
    # Volatility simulation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎛️ Interactive Path",
        "📈 Volatility Paths",
        "📊 Terminal Distribution",
        "📋 Statistics"
    ])

    with tab1:
        render_interactive_path_tab(params=params)

    with tab2:
        render_volatility_paths_tab(
            simulation_result=vol_result,
            params=params
        )

    with tab3:
        render_distributions_tab(
            simulation_result=vol_result,
            params=params,
            result_type="volatility"
        )

    with tab4:
        render_statistics_tab(
            simulation_result=vol_result,
            params=params,
            result_type="volatility"
        )


# =============================================================================
# MODEL INFORMATION
# =============================================================================

with st.expander("ℹ️ Model Information", expanded=False):
    if simulation_mode == 'price':
        model = params['price_model']
        st.markdown(f"### {PRICE_MODELS.get(model, model)}")
        st.markdown(MODEL_DESCRIPTIONS.get(model, "No description available."))

        # Model-specific formulas
        if model == 'gbm':
            st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
            st.markdown(r"""
            **Parameters:**
            - $\mu$ (drift): Expected return rate
            - $\sigma$ (volatility): Constant volatility
            - $S_0$: Initial stock price
            """)
        elif model == 'heston':
            st.latex(r"dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^{(1)}")
            st.latex(r"dv_t = \kappa(\theta - v_t)dt + \xi\sqrt{v_t}dW_t^{(2)}")
            st.latex(r"dW_t^{(1)} dW_t^{(2)} = \rho dt")
            st.markdown("""
            **Parameters:**
            - $\\kappa$: Mean reversion speed
            - $\\theta$: Long-run variance
            - $\\xi$: Volatility of volatility
            - $\\rho$: Correlation between price and variance
            - $v_0$: Initial variance
            """)
        elif model == 'merton':
            st.latex(r"dS_t = (\mu - \lambda\bar{k})S_t dt + \sigma S_t dW_t + S_t dJ_t")
            st.markdown(r"""
            **Parameters:**
            - $\lambda$: Jump intensity (expected jumps per year)
            - $\mu_J$: Mean jump size (log scale)
            - $\sigma_J$: Jump size volatility
            """)
        elif model == 'sabr':
            st.latex(r"dF_t = \sigma_t F_t^\beta dW_t^{(1)}")
            st.latex(r"d\sigma_t = \nu \sigma_t dW_t^{(2)}")
            st.latex(r"dW_t^{(1)} dW_t^{(2)} = \rho dt")
            st.markdown("""
            **Parameters:**
            - $\\alpha$: Initial volatility
            - $\\beta$: CEV exponent (0=normal, 1=lognormal)
            - $\\rho$: Correlation
            - $\\nu$: Volatility of volatility
            """)

    else:
        model = params['vol_model']
        st.markdown(f"### {VOLATILITY_MODELS.get(model, model)}")
        st.markdown(MODEL_DESCRIPTIONS.get(model, "No description available."))

        if model == 'garch':
            st.latex(r"\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2")
            st.markdown("""
            **Parameters:**
            - $\\omega$: Constant term
            - $\\alpha$: ARCH coefficient (shock impact)
            - $\\beta$: GARCH coefficient (persistence)

            **Stationarity condition:** $\\alpha + \\beta < 1$
            """)
        elif model == 'ngarch':
            st.latex(r"\sigma_t^2 = \omega + \alpha(\epsilon_{t-1} - \theta\sigma_{t-1})^2 + \beta \sigma_{t-1}^2")
            st.markdown("""
            **Parameters:**
            - $\\theta$: Leverage parameter (asymmetry)

            **Stationarity condition:** $\\alpha(1 + \\theta^2) + \\beta < 1$
            """)
        elif model == 'gjr_garch':
            st.latex(r"\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \gamma \epsilon_{t-1}^2 I(\epsilon_{t-1}<0) + \beta \sigma_{t-1}^2")
            st.markdown("""
            **Parameters:**
            - $\\gamma$: Leverage coefficient (negative shock amplification)

            **Stationarity condition:** $\\alpha + 0.5\\gamma + \\beta < 1$
            """)
        elif model == 'egarch':
            st.latex(r"\ln(\sigma_t^2) = \omega + \alpha|z_{t-1}| + \gamma z_{t-1} + \beta \ln(\sigma_{t-1}^2)")
            st.markdown("""
            **Parameters:**
            - $\\gamma$: Leverage parameter (asymmetric response)

            **Note:** EGARCH ensures variance is always positive.
            """)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(footer_html(), unsafe_allow_html=True)
