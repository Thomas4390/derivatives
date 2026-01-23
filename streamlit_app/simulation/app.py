"""
Unified Monte Carlo Simulation Explorer

A comprehensive educational tool for Monte Carlo simulation featuring:
- 7 stochastic models (GBM, Heston, Merton, Bates, GARCH, NGARCH, GJR-GARCH)
- Unified price and volatility path visualization
- Option Strategy P&L analysis with full strategy builder
- Risk metrics (VaR, CVaR, skewness, kurtosis)

Author: Thomas Vaudescal
"""

import sys
from pathlib import Path
import time

# Add paths for imports
app_dir = Path(__file__).parent
project_root = app_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(app_dir))

import streamlit as st
import numpy as np

# Backend imports
from backend.simulation.base import SimulationResult
from backend.portfolio.pnl import RiskMetrics

# Local config and components
from config.styles import (
    inject_styles,
    render_compact_header,
    footer_html,
    metric_card_html
)
from config.model_registry import (
    MODEL_REGISTRY,
    MODEL_DISPLAY_ORDER,
    get_model,
    PricingMethod,
)

# Components
from components.model_selector import render_model_selector, render_model_comparison_table
from components.parameter_panel import (
    render_market_parameters,
    render_model_parameters,
    render_simulation_settings,
)
from components.strategy_builder import (
    render_strategy_builder,
    export_positions_for_pnl_engine
)
from components.results_summary import (
    render_results_summary,
    render_simulation_info,
    render_model_equations,
    render_percentile_table,
)

# Charts
from charts.unified_paths import (
    render_unified_paths,
    render_price_paths_only,
    render_volatility_paths_only,
    render_path_controls,
)
from charts.distributions import render_distributions_tab
from charts.statistics import render_statistics_tab
from charts.pnl_distribution import render_pnl_distribution_tab, render_risk_metrics_tab

# Services
from services.simulation_service import (
    run_simulation,
    get_model_characteristics,
    check_model_conditions,
    MODEL_NAMES,
)
from services.simulation_runner import (
    run_price_simulation,
    calculate_pnl_from_paths,
)

# Black-Scholes functions for premium calculation
from scipy.stats import norm


def black_scholes_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes put price."""
    if T <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


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
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gbm"
if "simulation_result" not in st.session_state:
    st.session_state.simulation_result = None
if "pnl_result" not in st.session_state:
    st.session_state.pnl_result = None
if "all_params" not in st.session_state:
    st.session_state.all_params = {}


# =============================================================================
# HEADER
# =============================================================================

render_compact_header(
    title="Monte Carlo Simulation Explorer",
    subtitle="Option Strategy P&L Analysis with 7 Stochastic Models",
    badge="Educational Tool"
)


# =============================================================================
# SIDEBAR - UNIFIED CONFIGURATION
# =============================================================================

with st.sidebar:
    # Model Selection
    model_key = render_model_selector()

    st.markdown("---")

    # Market Parameters
    market_params = render_market_parameters()

    st.markdown("---")

    # Model-Specific Parameters
    model_params = render_model_parameters(model_key)

    st.markdown("---")

    # Simulation Settings
    sim_settings = render_simulation_settings()

    st.markdown("---")

    # Strategy Builder
    st.markdown("### 🎯 Option Strategy")

    def bs_price(s, k, r, t, sigma, opt_type):
        if opt_type == 'call':
            return black_scholes_call_price(s, k, t, r, sigma)
        else:
            return black_scholes_put_price(s, k, t, r, sigma)

    positions, stock_position = render_strategy_builder(
        spot_price=market_params.get('spot', 100.0),
        risk_free_rate=market_params.get('risk_free_rate', 0.05),
        time_to_expiry=market_params.get('time_horizon', 1.0),
        volatility=market_params.get('sigma', 0.20),
        bs_price_function=bs_price
    )

    position_arrays = export_positions_for_pnl_engine(positions, stock_position)

    st.markdown("---")

    # Combine all parameters
    all_params = {
        **market_params,
        **model_params,
        **sim_settings,
        "model": model_key,
        "price_model": model_key,
        "option_positions": positions,
        "stock_position": stock_position,
        "position_arrays": position_arrays,
    }
    st.session_state.all_params = all_params

    # Model condition check
    conditions = check_model_conditions(model_key, all_params)
    if not conditions["is_valid"]:
        for cond in conditions["conditions"]:
            if not cond["satisfied"]:
                st.warning(f"⚠️ {cond['name']}: {cond['message']}")

    # Run Simulation Button
    run_clicked = st.button(
        "▶️ Run Simulation",
        type="primary",
        use_container_width=True,
        key="run_simulation_btn"
    )

    if run_clicked:
        with st.spinner("Running simulation..."):
            start_time = time.time()
            try:
                result = run_simulation(model_key, all_params)
                execution_time = time.time() - start_time
                st.session_state.simulation_result = result
                st.session_state.execution_time = execution_time
                st.session_state.simulation_model = model_key
                st.session_state.simulation_params = all_params.copy()

                # Calculate P&L if strategy is defined
                if len(position_arrays.get('strikes', [])) > 0:
                    pnl_result = calculate_pnl_from_paths(result, all_params)
                    st.session_state.pnl_result = pnl_result
                else:
                    st.session_state.pnl_result = None

                st.success(f"✓ Completed in {execution_time*1000:.0f}ms")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)


# =============================================================================
# MAIN CONTENT - TABS
# =============================================================================

# Check if P&L analysis is available
pnl_result = st.session_state.get("pnl_result")
has_pnl = pnl_result is not None

if has_pnl:
    tab_sim, tab_pnl, tab_risk, tab_edu = st.tabs([
        "📈 Simulation",
        "💰 P&L Distribution",
        "📊 Risk Metrics",
        "📚 Education"
    ])
else:
    tab_sim, tab_edu = st.tabs([
        "📈 Simulation",
        "📚 Education"
    ])
    tab_pnl = None
    tab_risk = None


# =============================================================================
# TAB 1: SIMULATION
# =============================================================================

with tab_sim:
    result = st.session_state.get("simulation_result")

    if result is None:
        st.info("👈 Configure parameters and strategy in the sidebar, then click **Run Simulation**")

        # Show strategy summary if defined
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

        # Show model comparison table
        with st.expander("📋 Model Comparison", expanded=True):
            render_model_comparison_table()
    else:
        # Get stored model and params
        sim_model = st.session_state.get("simulation_model", model_key)
        sim_params = st.session_state.get("simulation_params", all_params)
        exec_time = st.session_state.get("execution_time", 0)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(metric_card_html(
                "Model",
                MODEL_NAMES.get(sim_model, sim_model)
            ), unsafe_allow_html=True)
        with col2:
            st.markdown(metric_card_html(
                "Paths",
                f"{int(sim_params.get('n_paths', 10000)):,}"
            ), unsafe_allow_html=True)
        with col3:
            st.markdown(metric_card_html(
                "Steps",
                f"{int(sim_params.get('n_steps', 252)):,}"
            ), unsafe_allow_html=True)
        with col4:
            st.markdown(metric_card_html(
                "Time",
                f"{exec_time*1000:.0f} ms"
            ), unsafe_allow_html=True)

        st.markdown("---")

        # Visualization controls
        with st.expander("🎨 Visualization Options", expanded=False):
            viz_opts = render_path_controls()

        # Unified paths chart
        render_unified_paths(
            result=result,
            model_key=sim_model,
            params=sim_params,
            n_sample_paths=viz_opts.get("n_sample_paths", 100),
            show_percentiles=viz_opts.get("show_percentiles", True),
            show_mean=viz_opts.get("show_mean", True),
        )

        # Results summary
        render_results_summary(result, sim_model, sim_params)

        # Detailed tabs
        detail_tabs = st.tabs(["📊 Distribution", "📋 Statistics", "📐 Percentiles"])

        with detail_tabs[0]:
            render_distributions_tab(
                simulation_result=result,
                params=sim_params,
                result_type="price"
            )

        with detail_tabs[1]:
            render_statistics_tab(
                simulation_result=result,
                params=sim_params,
                result_type="price"
            )

        with detail_tabs[2]:
            render_percentile_table(result)


# =============================================================================
# TAB 2: P&L DISTRIBUTION (Only shown when strategy is defined)
# =============================================================================

if tab_pnl is not None:
    with tab_pnl:
        pnl_result = st.session_state.get("pnl_result")
        sim_params = st.session_state.get("simulation_params", all_params)

        if pnl_result is not None:
            render_pnl_distribution_tab(
                pnl_values=pnl_result['pnl_values'],
                risk_metrics=pnl_result['risk_metrics'],
                params=sim_params
            )
        else:
            st.info("👈 Define an option strategy and run simulation to see P&L distribution.")


# =============================================================================
# TAB 3: RISK METRICS (Only shown when strategy is defined)
# =============================================================================

if tab_risk is not None:
    with tab_risk:
        pnl_result = st.session_state.get("pnl_result")
        sim_params = st.session_state.get("simulation_params", all_params)

        if pnl_result is not None:
            render_risk_metrics_tab(
                metrics=pnl_result['risk_metrics'],
                params=sim_params
            )
        else:
            st.info("👈 Define an option strategy and run simulation to see risk metrics.")


# =============================================================================
# TAB: EDUCATION (Always shown)
# =============================================================================

with tab_edu:
    st.subheader("📚 Model Education")

    # Model selector for education
    edu_model = st.selectbox(
        "Select Model to Learn About",
        options=MODEL_DISPLAY_ORDER,
        format_func=lambda x: f"{MODEL_REGISTRY[x].short_name} - {MODEL_REGISTRY[x].name}",
        key="edu_model_select"
    )

    model_spec = get_model(edu_model)

    # Model overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"### {model_spec.name}")
        st.markdown(model_spec.description)

        # Equations
        st.markdown("#### Model Equations")
        st.latex(model_spec.equation_main)

        if model_spec.equation_vol:
            st.markdown("*Volatility dynamics:*")
            st.latex(model_spec.equation_vol)

        if model_spec.equation_jump:
            st.markdown("*Jump distribution:*")
            st.latex(model_spec.equation_jump)

    with col2:
        st.markdown("#### Characteristics")

        # Feature badges
        if model_spec.has_stochastic_vol:
            st.success("✓ Stochastic Volatility")
        else:
            st.info("○ Constant Volatility")

        if model_spec.has_jumps:
            st.success("✓ Jump Component")
        else:
            st.info("○ No Jumps")

        st.markdown("**Pricing Methods:**")
        for method in model_spec.pricing_methods:
            if method == PricingMethod.ANALYTICAL:
                st.markdown("- 🎯 Black-Scholes (Analytical)")
            elif method == PricingMethod.FFT:
                st.markdown("- 📊 FFT (Carr-Madan)")
            elif method == PricingMethod.MONTE_CARLO:
                st.markdown("- 🎲 Monte Carlo")

    # Conditions
    if model_spec.feller_condition or model_spec.stationarity_condition:
        st.markdown("---")
        st.markdown("#### Model Conditions")

        if model_spec.feller_condition:
            st.info(f"**Feller Condition:** ${model_spec.feller_condition}$\n\nEnsures variance stays positive in Heston-type models.")

        if model_spec.stationarity_condition:
            st.info(f"**Stationarity Condition:** ${model_spec.stationarity_condition}$\n\nEnsures volatility mean-reverts to long-run level.")

    # Parameters
    st.markdown("---")
    st.markdown("#### Parameters")

    import pandas as pd
    param_data = []
    for p in model_spec.parameters:
        param_data.append({
            "Parameter": p.display_name,
            "Symbol": p.name,
            "Default": p.default,
            "Range": f"[{p.min_value}, {p.max_value}]",
            "Description": p.description
        })

    if param_data:
        st.dataframe(pd.DataFrame(param_data), use_container_width=True, hide_index=True)

    # Model comparison
    st.markdown("---")
    st.markdown("#### All Models Comparison")
    render_model_comparison_table()


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(footer_html(), unsafe_allow_html=True)
