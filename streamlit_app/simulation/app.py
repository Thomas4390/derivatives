"""
Monte Carlo Simulation Explorer

Author: Thomas Vaudescal
"""

import sys
from pathlib import Path
import time

app_dir = Path(__file__).parent
project_root = app_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(app_dir))

import streamlit as st
import numpy as np
from scipy.stats import norm
from backend.portfolio.pnl import compute_payoff_curve, find_breakeven_points

from config.styles import inject_styles, render_compact_header, footer_html, metric_card_html
from config.model_registry import get_model

from components.model_selector import render_model_selector
from components.parameter_panel import (
    render_market_parameters,
    render_model_parameters,
    render_simulation_settings,
)
from components.strategy_builder import render_strategy_builder, export_positions_for_pnl_engine

from charts.simulation_paths import render_simulation_chart, render_path_controls
from charts.pnl_analysis import render_payoff_with_distribution, render_3d_pnl_chart
from charts.path_explorer import render_path_explorer_chart
from charts.pricing_comparison import (
    extract_legs,
    compute_reference_prices,
    render_legs_summary,
    precompute_convergence,
    render_animated_convergence_chart,
    render_final_table,
)

from services.simulation_service import (
    run_simulation,
    check_model_conditions,
    get_initial_volatility,
    MODEL_NAMES,
)
from components.path_explorer_params import render_explorer_params
from services.pricing_service import get_available_pricing_methods
from services.simulation_runner import calculate_pnl_from_paths
from components.custom_model_editor import render_custom_model_editor
from services.simulation_service import get_model_display_name


# ── Black-Scholes helpers (for strategy builder premium calc) ────────────

def _bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bs_put(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ═════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Monte Carlo Simulation Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

for key, default in [
    ("simulation_result", None),
    ("pnl_result", None),
    ("all_params", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ═════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════

render_compact_header(
    title="Monte Carlo Simulation Explorer",
    subtitle="Option Strategy P&L Analysis with 7+ Stochastic Models",
    badge="Educational Tool",
)

# ═════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════

with st.sidebar:
    model_key = render_model_selector()
    st.markdown("---")

    market_params = render_market_parameters()
    st.markdown("---")

    model_params = render_model_parameters(model_key)
    st.markdown("---")

    sim_settings = render_simulation_settings()
    st.markdown("---")

    # Strategy builder
    st.markdown("### Option Strategy")

    def _bs_price(s, k, r, t, sigma, opt_type):
        return _bs_call(s, k, t, r, sigma) if opt_type == "call" else _bs_put(s, k, t, r, sigma)

    positions, stock_position = render_strategy_builder(
        spot_price=market_params.get("spot", 100.0),
        risk_free_rate=market_params.get("risk_free_rate", 0.05),
        time_to_expiry=market_params.get("time_horizon", 1.0),
        volatility=get_initial_volatility(model_key, model_params),
        bs_price_function=_bs_price,
    )
    position_arrays = export_positions_for_pnl_engine(positions, stock_position)
    st.markdown("---")

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

    # Condition warnings
    conditions = check_model_conditions(model_key, all_params)
    if not conditions["is_valid"]:
        for c in conditions["conditions"]:
            if not c["satisfied"]:
                st.warning(f"⚠️ {c['name']}: {c['description']}")

    # Run button
    if st.button("Run Simulation", type="primary", width="stretch", key="run_btn"):
        with st.spinner("Running simulation..."):
            t0 = time.time()
            try:
                result = run_simulation(model_key, all_params)
                dt = time.time() - t0
                st.session_state.simulation_result = result
                st.session_state.execution_time = dt
                st.session_state.simulation_model = model_key
                st.session_state.simulation_params = all_params.copy()

                if len(position_arrays.get("strikes", [])) > 0:
                    st.session_state.pnl_result = calculate_pnl_from_paths(result, all_params)
                else:
                    st.session_state.pnl_result = None

                st.session_state.convergence_result = None
                st.success(f"Done — {dt*1000:.0f} ms")
            except Exception as e:
                st.error(str(e))
                st.exception(e)


# ═════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═════════════════════════════════════════════════════════════════════════

result = st.session_state.get("simulation_result")

sim_model = st.session_state.get("simulation_model", model_key)
sim_params = st.session_state.get("simulation_params", all_params)
exec_time = st.session_state.get("execution_time", 0)
pnl_result = st.session_state.get("pnl_result")
has_strategy = result is not None and pnl_result is not None

# Precompute shared data
pnl_vals = pnl_result["pnl_values"] if has_strategy else None
breakevens = None
payoff_curve = None
spot_range = None

if has_strategy:
    pa = sim_params.get("position_arrays", {})
    if len(pa.get("strikes", [])) > 0:
        spot = sim_params.get("spot_price", sim_params.get("spot", 100.0))
        spot_range = np.linspace(spot * 0.5, spot * 1.5, 1000)
        payoff_curve = compute_payoff_curve(
            spot_range, pa["strikes"], pa["option_types"],
            pa["position_types"], pa["quantities"], pa["premiums"],
            pa.get("stock_quantity", 0.0), pa.get("stock_entry_price", 0.0),
        )
        breakevens = find_breakeven_points(payoff_curve, spot_range)

# ── Tabs (always show all) ────────────────────────────────────────
tab_names = ["Simulation"]
if has_strategy:
    tab_names.append("P&L Analysis")
tab_names.append("Pricing Comparison")
tab_names.append("Path Explorer")
tab_names.append("Custom Model")

tabs = st.tabs(tab_names)

# ── TAB 1: Simulation ─────────────────────────────────────────────
with tabs[0]:
    if result is None:
        st.info("Configure parameters in the sidebar and click **Run Simulation**.")
    else:
        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card_html("Model", get_model_display_name(sim_model)), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card_html("Paths", f"{int(sim_params.get('n_paths', 10000)):,}"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card_html("Steps", f"{int(sim_params.get('n_steps', 252)):,}"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card_html("Time", f"{exec_time*1000:.0f} ms"), unsafe_allow_html=True)

        st.markdown("---")

        with st.expander("Visualization Options", expanded=False):
            viz = render_path_controls()

        render_simulation_chart(
            result=result,
            model_key=sim_model,
            params=sim_params,
            n_display=viz["n_display"],
            show_bands=viz["show_bands"],
            pnl_values=pnl_vals,
            breakeven_prices=breakevens,
        )

        # Empirical statistics
        if has_strategy:
            risk = pnl_result["risk_metrics"]
            be_str = " / ".join(f"${b:.2f}" for b in breakevens) if breakevens is not None and len(breakevens) > 0 else "N/A"

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.markdown(metric_card_html("P(Profit)", f"{risk.prob_profit:.1%}"), unsafe_allow_html=True)
            with s2:
                st.markdown(metric_card_html("Breakeven", be_str), unsafe_allow_html=True)
            with s3:
                st.markdown(metric_card_html("Expected P&L", f"${risk.mean_pnl:+,.2f}"), unsafe_allow_html=True)
            with s4:
                st.markdown(metric_card_html("VaR 95%", f"${risk.var_95:+,.2f}", subtext="Max loss at 95% confidence"), unsafe_allow_html=True)
        else:
            tp = result.terminal_prices
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.markdown(metric_card_html("E[S(T)]", f"${np.mean(tp):,.2f}"), unsafe_allow_html=True)
            with s2:
                st.markdown(metric_card_html("Std[S(T)]", f"${np.std(tp):,.2f}"), unsafe_allow_html=True)
            with s3:
                ret = np.mean(tp / result.initial_price - 1) * 100
                st.markdown(metric_card_html("E[Return]", f"{ret:+.2f}%"), unsafe_allow_html=True)
            with s4:
                p5 = np.percentile(tp, 5)
                st.markdown(metric_card_html("5th Percentile", f"${p5:,.2f}", subtext="Worst 5% of outcomes"), unsafe_allow_html=True)

# ── TAB 2: P&L Analysis (only when strategy exists) ───────────────
if has_strategy:
    with tabs[1]:
        # Payoff diagram + MC distribution
        st.subheader("Payoff Diagram & Simulated P&L")
        render_payoff_with_distribution(
            result=result,
            pnl_values=pnl_vals,
            breakeven_prices=breakevens,
            spot=sim_params.get("spot_price", sim_params.get("spot", 100.0)),
        )

        st.markdown("---")

        # 3D scatter
        st.subheader("Realized Volatility / Terminal Price / P&L")
        render_3d_pnl_chart(
            result=result,
            pnl_values=pnl_vals,
            time_horizon=sim_params.get("time_horizon", 1.0),
        )

# ── TAB: Pricing Comparison ────────────────────────────────────────
pricing_tab_idx = 2 if has_strategy else 1
with tabs[pricing_tab_idx]:
    if result is None:
        st.info("Run a simulation first to see pricing comparison results.")
    else:
        spot_val = sim_params.get("spot_price", sim_params.get("spot", 100.0))
        r_val = sim_params.get("risk_free_rate", 0.05)
        T_val = sim_params.get("time_horizon", 1.0)
        n_steps_val = int(sim_params.get("n_steps", 252))

        methods = get_available_pricing_methods(sim_model)
        method_str = " \u00b7 ".join(m.replace("_", " ").title() for m in methods)
        st.caption(f"Available methods for {get_model_display_name(sim_model)}: **{method_str}**")

        # Extract legs from strategy (read-only)
        pa = sim_params.get("position_arrays", {})
        legs = extract_legs(pa, spot_val)
        compute_reference_prices(sim_model, sim_params, legs, T_val, spot_val, r_val)

        # Surface diagnostic when expected FFT/analytical pricing failed
        has_ref_method = "fft" in methods or "analytical" in methods
        all_mc_only = all(leg.get("ref_price") is None for leg in legs)
        if has_ref_method and all_mc_only:
            st.warning(
                "Reference pricing (FFT/Analytical) returned no result. "
                "If you just updated the code, try restarting the Streamlit server. "
                "Check terminal logs for details."
            )

        if not has_strategy:
            st.info("No strategy defined \u2014 pricing a single ATM Call option.")

        render_legs_summary(legs)
        st.markdown("---")

        # Auto-compute convergence (invalidate if n_paths changed)
        n_paths_val = int(sim_params.get("n_paths", 10000))
        conv = st.session_state.get("convergence_result")
        if conv is not None and conv["n_done"][-1] != n_paths_val:
            conv = None
        if conv is None:
            n_sims = len([n for n in [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000] if n < n_paths_val]) + 1
            with st.spinner(f"Computing convergence ({n_sims} simulations up to N={n_paths_val:,})\u2026"):
                conv = precompute_convergence(
                    model_key=sim_model, params=sim_params,
                    legs=legs, T=T_val, spot=spot_val, r=r_val,
                    n_steps=n_steps_val, max_n=n_paths_val,
                )
                st.session_state.convergence_result = conv

        render_animated_convergence_chart(conv)

        st.latex(
            r"\text{SE}(\hat{C}_{\mathrm{MC}}) = "
            r"\frac{\sigma_{\text{payoff}}}{\sqrt{N}}"
            r"\quad\Longrightarrow\quad"
            r"\text{Error} = O\!\left(\frac{1}{\sqrt{N}}\right)"
        )

        st.markdown("---")
        max_n_display = conv["n_done"][-1]
        st.subheader(f"Final Comparison (N = {max_n_display:,})")
        render_final_table(conv)

# ── TAB: Path Explorer ────────────────────────────────────────────
explorer_tab_idx = pricing_tab_idx + 1
with tabs[explorer_tab_idx]:
    if result is None:
        st.info("Run a simulation first to explore individual paths.")
    else:
        explorer_params = render_explorer_params(sim_model)

        try:
            explorer_result = run_simulation(sim_model, explorer_params)

            render_path_explorer_chart(
                result=explorer_result,
                model_key=sim_model,
                params=explorer_params,
            )

            # Terminal metrics
            tp = explorer_result.price_paths[0, -1]
            s0 = explorer_params["spot"]
            init_vol = get_initial_volatility(sim_model, explorer_params) * 100
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(metric_card_html("Terminal Price", f"${tp:,.2f}"), unsafe_allow_html=True)
            with m2:
                ret = (tp / s0 - 1) * 100
                st.markdown(metric_card_html("Return", f"{ret:+.2f}%"), unsafe_allow_html=True)
            with m3:
                st.markdown(metric_card_html("Initial Vol", f"{init_vol:.1f}%"), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Simulation error: {e}")

# ── TAB: Custom Model ─────────────────────────────────────────────
custom_tab_idx = explorer_tab_idx + 1
with tabs[custom_tab_idx]:
    render_custom_model_editor()


# ═════════════════════════════════════════════════════════════════════════
# MODEL EQUATIONS (collapsible, bottom of page)
# ═════════════════════════════════════════════════════════════════════════

active_model = st.session_state.get("simulation_model", model_key)
try:
    spec = get_model(active_model)
    with st.expander(f"Model Equations — {spec.name}", expanded=False):
        # ── SDE dynamics ──
        st.markdown("**Stochastic Dynamics**")
        st.latex(spec.equation_main)
        if spec.equation_vol:
            st.caption("Volatility dynamics:")
            st.latex(spec.equation_vol)
        if spec.equation_jump:
            st.caption("Jump distribution:")
            st.latex(spec.equation_jump)

        # ── Pricing method equations ──
        eq_analytical = getattr(spec, "equation_analytical", None)
        eq_cf = getattr(spec, "equation_cf", None)
        eq_mc = getattr(spec, "equation_mc", None)
        if eq_analytical or eq_cf or eq_mc:
            st.markdown("---")
            st.markdown("**Pricing Methods**")
            if eq_analytical:
                st.caption("Analytical (Black-Scholes):")
                st.latex(eq_analytical)
            if eq_cf:
                st.caption("FFT — Characteristic Function:")
                st.latex(eq_cf)
            if eq_mc:
                st.caption("Monte Carlo — Discretization:")
                st.latex(eq_mc)
except ValueError:
    pass

# ═════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════

st.markdown(footer_html(), unsafe_allow_html=True)
