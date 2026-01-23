"""
Unified Monte Carlo Simulation Explorer

A comprehensive educational tool for Monte Carlo simulation featuring:
- 7 stochastic models (GBM, Heston, Merton, Bates, GARCH, NGARCH, GJR-GARCH)
- Unified price and volatility path visualization
- Pricing comparison (MC vs Analytical/FFT)
- Option P&L analysis

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
    render_option_parameters,
)
from components.pricing_comparison import (
    render_pricing_comparison,
    render_convergence_guide,
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
from charts.pricing_comparison_chart import (
    render_pricing_comparison_chart,
    render_single_price_comparison,
)
from charts.distributions import render_distributions_tab
from charts.statistics import render_statistics_tab

# Services
from services.simulation_service import (
    run_simulation,
    get_model_characteristics,
    check_model_conditions,
    MODEL_NAMES,
)
from services.pricing_service import (
    compare_pricing,
    price_multiple_strikes,
    get_available_pricing_methods,
    compute_option_pnl,
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
if "all_params" not in st.session_state:
    st.session_state.all_params = {}


# =============================================================================
# HEADER
# =============================================================================

render_compact_header(
    title="Monte Carlo Simulation Explorer",
    subtitle="Unified visualization of 7 stochastic models",
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

    # Option Parameters (for pricing comparison)
    option_params = render_option_parameters()

    st.markdown("---")

    # Combine all parameters
    all_params = {
        **market_params,
        **model_params,
        **sim_settings,
        **option_params,
        "model": model_key,
    }
    st.session_state.all_params = all_params

    # Model condition check
    conditions = check_model_conditions(model_key, all_params)
    if not conditions["is_valid"]:
        for cond in conditions["conditions"]:
            if not cond["satisfied"]:
                st.warning(f"⚠️ {cond['name']}: {cond['message']}")

    # Run Simulation Button
    st.markdown("---")
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
                st.success(f"✓ Completed in {execution_time*1000:.0f}ms")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)


# =============================================================================
# MAIN CONTENT - TABS
# =============================================================================

tab_sim, tab_pricing, tab_pnl, tab_edu = st.tabs([
    "📈 Simulation",
    "💰 Pricing Comparison",
    "📊 Option P&L",
    "📚 Education"
])


# =============================================================================
# TAB 1: SIMULATION
# =============================================================================

with tab_sim:
    result = st.session_state.get("simulation_result")

    if result is None:
        st.info("👈 Configure parameters in the sidebar and click **Run Simulation**")

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
# TAB 2: PRICING COMPARISON
# =============================================================================

with tab_pricing:
    result = st.session_state.get("simulation_result")

    if result is None:
        st.info("👈 Run a simulation first to compare pricing methods")
        render_convergence_guide()
    else:
        sim_model = st.session_state.get("simulation_model", model_key)
        sim_params = st.session_state.get("simulation_params", all_params)

        # Available methods
        available_methods = get_available_pricing_methods(sim_model)
        st.caption(f"Available methods for {MODEL_NAMES.get(sim_model, sim_model)}: {', '.join(m.upper() for m in available_methods)}")

        # Single strike comparison
        st.subheader("Single Strike Comparison")

        # Get current values for styling
        pricing_type_val = st.session_state.get("pricing_type", "call")
        is_call_display = pricing_type_val == "call"

        # Dynamic styling based on option type
        border_color = "#10b981" if is_call_display else "#ef4444"
        bg_gradient = "linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%)" if is_call_display else "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"
        type_badge_bg = "#d1fae5" if is_call_display else "#fee2e2"
        type_badge_color = "#047857" if is_call_display else "#b91c1c"
        type_label = "CALL" if is_call_display else "PUT"

        # Option configuration with option_pricer style
        st.markdown(f"""
        <div style="background: {bg_gradient}; border: 1px solid {border_color}40; border-left: 4px solid {border_color}; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.625rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1rem;">📜</span>
                    <span style="font-size: 0.7rem; font-weight: 700; color: #475569; text-transform: uppercase;">Option Configuration</span>
                </div>
                <span style="background: {type_badge_bg}; color: {type_badge_color}; font-size: 0.65rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; text-transform: uppercase;">{type_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            option_type = st.selectbox(
                "Type",
                options=["call", "put"],
                format_func=lambda x: f"{'📈' if x == 'call' else '📉'} {x.upper()}",
                key="pricing_type"
            )
            is_call = option_type == "call"

        with col2:
            strike = st.number_input(
                "Strike ($)",
                value=sim_params.get("strike", 100.0),
                min_value=1.0,
                step=1.0,
                format="%.2f",
                key="pricing_strike"
            )

        time_to_mat = sim_params.get("time_horizon", 1.0)

        # Premium display styled like option_pricer
        spot = sim_params.get("spot", 100.0)
        rate = sim_params.get("risk_free_rate", 0.05)
        char = get_model_characteristics(sim_model)
        if char["volatility_type"] == "constant":
            vol = sim_params.get("sigma", 0.20)
        else:
            vol = np.sqrt(sim_params.get("v0", 0.04)) if "v0" in sim_params else sim_params.get("sigma0", 0.20)

        if is_call:
            premium = black_scholes_call_price(spot, strike, time_to_mat, rate, vol)
        else:
            premium = black_scholes_put_price(spot, strike, time_to_mat, rate, vol)

        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0.75rem; background: #ffffff; border-radius: 6px; border: 1px solid #e2e8f0; margin-bottom: 0.5rem;">
            <span style="color: #64748b; font-size: 0.8rem;">
                Premium (BS): <span style="font-family: 'JetBrains Mono', monospace; font-weight: 500;">${premium:.2f}</span>
            </span>
            <span style="color: #64748b; font-size: 0.8rem;">
                Maturity: <span style="font-family: 'JetBrains Mono', monospace; font-weight: 500;">{time_to_mat:.2f}y</span>
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Run pricing comparison
        if st.button("Compare Pricing", key="compare_pricing_btn"):
            with st.spinner("Computing prices..."):
                comparison = compare_pricing(
                    model_key=sim_model,
                    params=sim_params,
                    terminal_prices=result.terminal_prices,
                    strike=strike,
                    time_to_maturity=time_to_mat,
                    spot=sim_params.get("spot", 100.0),
                    risk_free_rate=sim_params.get("risk_free_rate", 0.05),
                    is_call=is_call
                )
                st.session_state.pricing_comparison = comparison

        comparison = st.session_state.get("pricing_comparison")
        if comparison:
            render_pricing_comparison(comparison)
            render_single_price_comparison(comparison)

        st.markdown("---")

        # Multi-strike comparison
        st.subheader("Multi-Strike Analysis")

        col1, col2 = st.columns(2)
        with col1:
            strike_min = st.number_input("Min Strike", value=80.0, key="strike_min")
        with col2:
            strike_max = st.number_input("Max Strike", value=120.0, key="strike_max")

        n_strikes = st.slider("Number of Strikes", 5, 20, 11, key="n_strikes")

        if st.button("Run Multi-Strike Analysis", key="multi_strike_btn"):
            strikes = np.linspace(strike_min, strike_max, n_strikes)

            with st.spinner("Computing prices across strikes..."):
                multi_result = price_multiple_strikes(
                    model_key=sim_model,
                    params=sim_params,
                    simulation_result=result,
                    strikes=strikes,
                    time_to_maturity=time_to_mat,
                    spot=sim_params.get("spot", 100.0),
                    risk_free_rate=sim_params.get("risk_free_rate", 0.05),
                    is_call=is_call
                )
                st.session_state.multi_strike_result = multi_result

        multi_result = st.session_state.get("multi_strike_result")
        if multi_result:
            render_pricing_comparison_chart(
                strikes=multi_result["strikes"],
                mc_prices=multi_result["mc_prices"],
                mc_errors=multi_result["mc_errors"],
                analytical_prices=multi_result.get("analytical_prices"),
                fft_prices=multi_result.get("fft_prices"),
                spot=sim_params.get("spot", 100.0),
                is_call=is_call
            )

        # Convergence guide
        with st.expander("📚 Understanding MC Convergence"):
            render_convergence_guide()


# =============================================================================
# TAB 3: OPTION P&L
# =============================================================================

with tab_pnl:
    result = st.session_state.get("simulation_result")

    if result is None:
        st.info("👈 Run a simulation first to analyze option P&L")
    else:
        sim_model = st.session_state.get("simulation_model", model_key)
        sim_params = st.session_state.get("simulation_params", all_params)

        st.subheader("Option P&L Analysis")

        # Get current values for styling
        pnl_type_val = st.session_state.get("pnl_type", "call")
        pnl_position_val = st.session_state.get("pnl_position", "long")
        is_call_display = pnl_type_val == "call"
        is_long_display = pnl_position_val == "long"

        # Dynamic styling based on position (long = green, short = red)
        border_color = "#10b981" if is_long_display else "#ef4444"
        bg_gradient = "linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%)" if is_long_display else "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"
        position_badge_bg = "#d1fae5" if is_long_display else "#fee2e2"
        position_badge_color = "#047857" if is_long_display else "#b91c1c"

        # Option configuration styled container
        st.markdown(f"""
        <div style="background: {bg_gradient}; border: 1px solid {border_color}40; border-left: 4px solid {border_color}; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.625rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1rem;">💰</span>
                    <span style="font-size: 0.7rem; font-weight: 700; color: #475569; text-transform: uppercase;">P&L Configuration</span>
                </div>
                <span style="background: {position_badge_bg}; color: {position_badge_color}; font-size: 0.65rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; text-transform: uppercase;">{pnl_position_val.upper()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Row 1: Type and Direction
        col1, col2 = st.columns(2)

        with col1:
            option_type = st.selectbox(
                "Type",
                options=["call", "put"],
                format_func=lambda x: f"{'📈' if x == 'call' else '📉'} {x.upper()}",
                key="pnl_type"
            )
            is_call = option_type == "call"

        with col2:
            position_type = st.selectbox(
                "Direction",
                options=["long", "short"],
                format_func=lambda x: f"{'🟢' if x == 'long' else '🔴'} {x.upper()}",
                key="pnl_position"
            )
            is_long = position_type == "long"

        # Row 2: Strike and Contracts
        col3, col4 = st.columns(2)

        with col3:
            strike = st.number_input(
                "Strike ($)",
                value=sim_params.get("strike", 100.0),
                min_value=1.0,
                step=1.0,
                format="%.2f",
                key="pnl_strike"
            )

        with col4:
            quantity = st.number_input(
                "Contracts",
                value=1,
                min_value=1,
                max_value=1000,
                step=1,
                key="pnl_quantity"
            )

        # Premium calculation
        spot = sim_params.get("spot", 100.0)
        rate = sim_params.get("risk_free_rate", 0.05)
        time_to_mat = sim_params.get("time_horizon", 1.0)

        # Get volatility for BS pricing
        char = get_model_characteristics(sim_model)
        if char["volatility_type"] == "constant":
            vol = sim_params.get("sigma", 0.20)
        else:
            vol = np.sqrt(sim_params.get("v0", 0.04)) if "v0" in sim_params else sim_params.get("sigma0", 0.20)

        # Calculate premium
        if is_call:
            premium = black_scholes_call_price(spot, strike, time_to_mat, rate, vol)
        else:
            premium = black_scholes_put_price(spot, strike, time_to_mat, rate, vol)

        total_cost = premium * quantity * 100  # 100 shares per contract
        cost_color = "#dc2626" if is_long else "#059669"
        cost_prefix = "-" if is_long else "+"
        cost_label = "Debit" if is_long else "Credit"

        # Premium display styled like option_pricer
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0.75rem; background: #ffffff; border-radius: 6px; border: 1px solid #e2e8f0; margin-bottom: 0.5rem;">
            <span style="color: #64748b; font-size: 0.8rem;">
                Premium: <span style="font-family: 'JetBrains Mono', monospace; font-weight: 500;">${premium:.2f}</span>
            </span>
            <span style="color: {cost_color}; font-weight: 700; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">
                {cost_prefix}${total_cost:,.2f}
            </span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Calculate P&L", key="calc_pnl_btn"):
            pnl = compute_option_pnl(
                price_paths=result.price_paths,
                strike=strike,
                premium=premium,
                is_call=is_call,
                is_long=is_long,
                quantity=quantity
            )
            st.session_state.pnl_result = pnl

        pnl = st.session_state.get("pnl_result")
        if pnl is not None:
            st.markdown("---")

            # P&L Statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Mean P&L", f"${np.mean(pnl):,.2f}")
            with col2:
                st.metric("Median P&L", f"${np.median(pnl):,.2f}")
            with col3:
                st.metric("Max P&L", f"${np.max(pnl):,.2f}")
            with col4:
                st.metric("Min P&L", f"${np.min(pnl):,.2f}")

            # P&L Distribution
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=pnl,
                nbinsx=50,
                marker_color="rgba(31, 119, 180, 0.7)",
                name="P&L"
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.add_vline(x=np.mean(pnl), line_dash="solid", line_color="green",
                         annotation_text=f"Mean: ${np.mean(pnl):.2f}")

            fig.update_layout(
                title="<b>P&L Distribution</b>",
                xaxis_title="P&L ($)",
                yaxis_title="Count",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Risk metrics
            with st.expander("📊 Risk Metrics", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Value at Risk**")
                    var_95 = np.percentile(pnl, 5)
                    var_99 = np.percentile(pnl, 1)
                    st.metric("VaR 95%", f"${var_95:,.2f}")
                    st.metric("VaR 99%", f"${var_99:,.2f}")

                with col2:
                    st.markdown("**Additional Stats**")
                    win_rate = np.mean(pnl > 0) * 100
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                    st.metric("Std Dev", f"${np.std(pnl):,.2f}")


# =============================================================================
# TAB 4: EDUCATION
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
