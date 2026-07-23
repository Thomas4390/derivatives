"""
Monte Carlo Simulation Explorer

Author: Thomas Vaudescal
"""

import hashlib
import json
import sys
import time
from pathlib import Path

app_dir = Path(__file__).parent
project_root = app_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(app_dir))

import numpy as np
import streamlit as st
from charts.path_explorer import render_path_explorer_chart
from charts.greeks_charts import (
    render_colorscale_selector,
    render_greek_selector,
    render_greeks_3d_surface,
    render_greeks_with_dte_slider,
)
from charts.pnl_analysis import (
    render_3d_pnl_chart,
    render_payoff_with_distribution,
    render_pnl_by_dte,
    render_vol_pnl_scatter,
    render_vol_pnl_scatter_by_dte,
)
from charts.pricing_comparison import (
    _build_n_paths_grid,
    compute_cpn_analytical_reference,
    compute_reference_prices,
    extract_legs,
    precompute_convergence,
    precompute_sp_convergence,
    render_animated_convergence_chart,
    render_final_table,
    render_legs_summary,
    render_sp_summary,
)
from charts.simulation_paths import render_path_controls, render_simulation_chart
from charts.structured_charts import add_structured_overlays
from charts.structured_path_explorer import render_structured_path_explorer
from components.custom_model_editor import render_custom_model_editor
from components.model_selector import render_model_selector
from components.parameter_panel import (
    render_market_parameters,
    render_model_parameters,
    render_simulation_settings,
)
from components.path_explorer_params import render_explorer_params
from components.strategy_builder import (
    export_positions_for_pnl_engine,
    render_strategy_builder,
)
from config.constants import SP_PRODUCT_DESCRIPTIONS
from config.model_registry import get_model
from config.strategy_equations import get_option_strategy_equations
from config.styles import (
    footer_html,
    inject_styles,
    render_compact_header,
    render_metric_row,
)
from services.consistent_pricing import (
    price_exotic_consistent,
    price_vanilla_consistent,
    pricing_method_label,
)
from services.pricing_service import get_available_pricing_methods
from services.greeks_service import (
    compute_strategy_greeks_surface_model,
    compute_strategy_greeks_surface_practitioner,
)
from services.simulation_runner import (
    calculate_pnl_from_paths,
    compute_hybrid_payoff_curve,
    has_exotic_legs,
)
from services.simulation_service import (
    check_model_conditions,
    get_initial_volatility,
    get_model_display_name,
    get_spot,
    run_simulation,
)
from services.structured_product_service import evaluate_structured_product_on_paths

from backend.portfolio.pnl import compute_payoff_curve, find_breakeven_points


def _model_greeks_cached(model_key, params, position_arrays, spot, rate, q, time_to_expiry):
    """Session-cached model-consistent Greeks surface.

    Recomputes only when the model / parameters / positions / market fingerprint
    changes, so toggling the Greek selector does not re-run the FFT/MC surface.
    Returns ``None`` for exotic strategies (caller falls back to the BS surface).
    """
    fingerprint = hashlib.md5(
        json.dumps(
            {
                "model": model_key,
                "params": {
                    k: v
                    for k, v in params.items()
                    if isinstance(v, (int, float, str, bool))
                },
                "strikes": np.asarray(position_arrays.get("strikes", [])).tolist(),
                "opt_types": np.asarray(position_arrays.get("option_types", [])).tolist(),
                "pos_types": np.asarray(position_arrays.get("position_types", [])).tolist(),
                "qtys": np.asarray(position_arrays.get("quantities", [])).tolist(),
                "stock_qty": float(position_arrays.get("stock_quantity", 0.0)),
                "exotic": [
                    (m or {}).get("instrument_class", "vanilla")
                    for m in position_arrays.get("exotic_metadata", [])
                ],
                "spot": round(float(spot), 6),
                "rate": round(float(rate), 8),
                "q": round(float(q), 8),
                "T": round(float(time_to_expiry), 8),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    cached = st.session_state.get("_model_greeks_cache")
    if cached and cached[0] == fingerprint:
        return cached[1]

    with st.spinner(
        f"Computing model-consistent Greeks under {get_model_display_name(model_key)}…"
    ):
        surface = compute_strategy_greeks_surface_model(
            position_arrays=position_arrays,
            model_key=model_key,
            model_params=params,
            spot=spot,
            rate=rate,
            q=q,
            time_to_expiry=time_to_expiry,
            n_steps=int(params.get("n_steps", 252)),
        )
    st.session_state["_model_greeks_cache"] = (fingerprint, surface)
    return surface


# ═════════════════════════════════════════════════════════════════════════
# STRUCTURED PRODUCT EQUATIONS
# ═════════════════════════════════════════════════════════════════════════


def _get_structured_product_equations(product_type: str, params: dict) -> dict:
    """Return LaTeX equations for a structured product type."""
    if product_type == "cpn":
        return {
            "name": "Capital Protected Note (CPN)",
            "description": "Bond floor (zero-coupon) + capped upside participation on the underlying.",
            "equations": [
                {
                    "label": "Payoff at Maturity",
                    "latex": r"\text{Payoff}(T) = N \cdot \max\!\Big(\alpha,\; \alpha + p \cdot \min\!\big(\text{perf} - 1,\; C - 1\big)^+\Big)",
                },
                {
                    "label": "Performance",
                    "latex": r"\text{perf} = \frac{S_T}{S_0}",
                },
                {
                    "label": "Fair Value (MC)",
                    "latex": r"V_0 = \frac{1}{M} \sum_{i=1}^{M} e^{-rT} \cdot \text{Payoff}^{(i)}(T)",
                },
                {
                    "label": "Decomposition",
                    "latex": r"V_0 = \underbrace{\alpha \cdot N \cdot e^{-rT}}_{\text{Bond Floor}} + \underbrace{N \cdot p \cdot \text{Call Spread}(S_0, C \cdot S_0)}_{\text{Option Component}}",
                },
            ],
        }
    if product_type == "reverse_convertible":
        return {
            "name": "Reverse Convertible",
            "description": "Fixed coupon stream + knock-in put exposure at maturity.",
            "equations": [
                {
                    "label": "Coupon Payments",
                    "latex": r"\text{Coupon PV} = \sum_{j=1}^{n} c \cdot \Delta t_j \cdot N \cdot D(0, t_j)",
                },
                {
                    "label": "Terminal Payoff",
                    "latex": r"\text{Terminal}(T) = \begin{cases} N \cdot D(0,T) & \text{if no KI breach} \\ N \cdot \min\!\left(\frac{S_T}{S_0},\, 1\right) \cdot D(0,T) & \text{if KI breached} \end{cases}",
                },
                {
                    "label": "Knock-In Condition",
                    "latex": r"\text{KI Breached} \iff \exists\, t \in [0,T] : S_t \leq B \cdot S_0",
                },
                {
                    "label": "Total PV",
                    "latex": r"V_0 = \text{Coupon PV} + \mathbb{E}^{\mathbb{Q}}\!\left[\text{Terminal}(T)\right]",
                },
            ],
        }
    if product_type == "autocallable":
        return {
            "name": "Autocallable",
            "description": "Conditional coupons with memory, autocall trigger for early redemption, and knock-in put protection.",
            "equations": [
                {
                    "label": "Autocall Condition (at observation $t_j$)",
                    "latex": r"\text{Autocalled at } t_j \iff \frac{S_{t_j}}{S_0} \geq H_{\text{call}} \quad\text{(first such } j\text{)}",
                },
                {
                    "label": "Conditional Coupon",
                    "latex": r"\text{Coupon}_j = \begin{cases} c \cdot \Delta t_j \cdot N & \text{if } S_{t_j}/S_0 \geq H_{\text{coupon}} \\ 0 & \text{otherwise}\end{cases}",
                },
                {
                    "label": "Memory Coupon",
                    "latex": r"\text{If memory: unpaid coupons accumulate and are paid at first observation where } S_{t_j}/S_0 \geq H_{\text{coupon}}",
                },
                {
                    "label": "Terminal Payoff (if not autocalled)",
                    "latex": r"\text{Terminal} = \begin{cases} N \cdot D(0,T) & \text{if no KI breach} \\ N \cdot \min\!\left(\frac{S_T}{S_0},\, 1\right) \cdot D(0,T) & \text{if KI breached} \end{cases}",
                },
                {
                    "label": "Total PV",
                    "latex": r"V_0 = \frac{1}{M}\sum_{i=1}^{M}\left[\sum_{j} \text{Coupon}_j^{(i)} \cdot D(0,t_j) + \text{Redemption}^{(i)}\right]",
                },
            ],
        }
    return {"name": product_type, "description": "", "equations": []}


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
    ("sp_result", None),
    ("sp_config", None),
    ("all_params", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ═════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════

render_compact_header(
    title="Monte Carlo Simulation Explorer",
    subtitle="Option Strategy & Structured Product Analysis with 7+ Stochastic Models",
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

    # ── Unified Strategy Builder (options + structured products) ──
    st.markdown(
        """
<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
    <span style="font-size: 1rem;">🎯</span>
    <span style="font-size: 0.75rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em;">Strategy Builder</span>
</div>
""",
        unsafe_allow_html=True,
    )

    # Premiums are priced with the engine consistent with the SELECTED model
    # (BS for GBM, FFT for Heston/Merton/Bates, risk-neutral MC for the GARCH
    # family) — not a flat-vol BS proxy. The (s, k, r, t, sigma, opt_type)
    # signature is preserved for the strategy builder; `sigma` is only the
    # last-resort Black-Scholes fallback inside price_vanilla_consistent.
    _div_yield = market_params.get("dividend_yield", 0.0)

    def _model_premium(s, k, r, t, sigma, opt_type):
        premium, _method = price_vanilla_consistent(
            model_key,
            model_params,
            spot=s,
            rate=r,
            q=_div_yield,
            strike=k,
            maturity=t,
            is_call=(opt_type == "call"),
        )
        return premium

    def _model_exotic_premium(
        exotic_type,
        spot,
        strike,
        maturity,
        rate,
        sigma,
        is_call,
        barrier=0.0,
        is_knock_in=False,
        is_up=True,
        rebate=0.0,
        payout=1.0,
        extra1=0.0,
        cap=0.0,
        params=None,
    ):
        premium, _method = price_exotic_consistent(
            model_key,
            model_params,
            _div_yield,
            exotic_type=exotic_type,
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            sigma=sigma,
            is_call=is_call,
            barrier=barrier,
            is_knock_in=is_knock_in,
            is_up=is_up,
            rebate=rebate,
            payout=payout,
            extra1=extra1,
            cap=cap,
            params=params,
        )
        return premium

    positions, stock_position = render_strategy_builder(
        spot_price=market_params.get("spot", 100.0),
        risk_free_rate=market_params.get("risk_free_rate", 0.05),
        time_to_expiry=market_params.get("time_horizon", 1.0),
        volatility=get_initial_volatility(model_key, model_params),
        bs_price_function=_model_premium,
        exotic_price_function=_model_exotic_premium,
    )
    st.caption(
        f"Premiums priced under {get_model_display_name(model_key)} "
        f"· {pricing_method_label(model_key, model_params)}"
    )
    position_arrays = export_positions_for_pnl_engine(
        positions,
        stock_position,
        risk_free_rate=market_params.get("risk_free_rate", 0.05),
        maturity=market_params.get("time_horizon", 1.0),
    )

    # sp_config is set by the strategy builder when a structured product is selected
    sp_config = st.session_state.get("sp_config")
    if sp_config:
        st.caption(SP_PRODUCT_DESCRIPTIONS.get(sp_config["product_type"], ""))

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
    # Sync maturity from structured product to time_horizon
    if sp_config:
        all_params["time_horizon"] = sp_config["maturity"]
        all_params["sp_config"] = sp_config
    st.session_state.all_params = all_params

    # Condition warnings
    conditions = check_model_conditions(model_key, all_params)
    if not conditions["is_valid"]:
        for c in conditions["conditions"]:
            if not c["satisfied"]:
                st.warning(f"⚠️ {c['name']}: {c['description']}")

    # ── Auto-run: detect config changes via hash ──────────────────────
    def _params_hash(params, pa, sp_cfg):
        """Fingerprint of all simulation-relevant parameters."""
        d = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
        d["_strikes"] = pa.get("strikes", np.array([])).tolist()
        d["_opt_types"] = pa.get("option_types", np.array([])).tolist()
        d["_pos_types"] = pa.get("position_types", np.array([])).tolist()
        d["_qtys"] = pa.get("quantities", np.array([])).tolist()
        d["_premiums"] = [
            round(p, 6) for p in pa.get("premiums", np.array([])).tolist()
        ]
        d["_stock_qty"] = pa.get("stock_quantity", 0.0)
        # Include exotic metadata (instrument classes, barriers, etc.)
        for i, m in enumerate(pa.get("exotic_metadata", [])):
            for mk, mv in m.items():
                d[f"_ex{i}_{mk}"] = mv
        # Include structured product config
        if sp_cfg:
            d["_sp_type"] = sp_cfg.get("product_type", "")
            d["_sp_params"] = json.dumps(
                sp_cfg.get("product_params", {}), sort_keys=True, default=str
            )
        return hashlib.md5(
            json.dumps(d, sort_keys=True, default=str).encode()
        ).hexdigest()

    _current_hash = _params_hash(all_params, position_arrays, sp_config)
    _last_hash = st.session_state.get("_sim_config_hash")
    _config_changed = _current_hash != _last_hash
    _has_positions = len(position_arrays.get("strikes", [])) > 0
    _is_sp_mode = sp_config is not None

    # Auto-run when config changed (with positions or in SP mode)
    manual_run = st.button(
        "Re-run Simulation"
        if st.session_state.get("simulation_result") is not None
        else "Run Simulation",
        type="secondary" if (_has_positions or _is_sp_mode) else "primary",
        width="stretch",
        key="run_btn",
    )

    should_run = manual_run or (_config_changed and (_has_positions or _is_sp_mode))

    if should_run:
        with st.spinner("Running simulation..."):
            t0 = time.time()
            try:
                result = run_simulation(model_key, all_params)
                dt = time.time() - t0
                st.session_state.simulation_result = result
                st.session_state.execution_time = dt
                st.session_state.simulation_model = model_key
                st.session_state.simulation_params = all_params.copy()
                st.session_state._sim_config_hash = _current_hash

                if _is_sp_mode:
                    st.session_state.sp_result = evaluate_structured_product_on_paths(
                        result,
                        sp_config,
                        all_params,
                    )
                    st.session_state.pnl_result = None
                else:
                    st.session_state.sp_result = None
                    if _has_positions:
                        st.session_state.pnl_result = calculate_pnl_from_paths(
                            result, all_params
                        )
                    else:
                        st.session_state.pnl_result = None

                st.session_state.convergence_result = None
                st.session_state.sp_convergence_result = None
            except (ValueError, RuntimeError) as e:
                st.error(str(e))


# ═════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═════════════════════════════════════════════════════════════════════════

result = st.session_state.get("simulation_result")

sim_model = st.session_state.get("simulation_model", model_key)
sim_params = st.session_state.get("simulation_params", all_params)
exec_time = st.session_state.get("execution_time", 0)
pnl_result = st.session_state.get("pnl_result")
sp_result = st.session_state.get("sp_result")

is_sp_mode = sp_result is not None
has_strategy = result is not None and (pnl_result is not None or is_sp_mode)

# Precompute shared data — unified for both modes
pnl_vals = None
breakevens = None
payoff_curve = None
spot_range = None

if is_sp_mode and result is not None:
    # Convert SP returns to P&L values (same semantics as option P&L)
    notional = sp_result.product_config["product_params"]["notional"]
    pnl_vals = np.round(sp_result.per_path_returns * notional, 2)
elif pnl_result is not None:
    pnl_vals = np.round(pnl_result["pnl_values"], 2)

if not is_sp_mode and pnl_result is not None and result is not None:
    pa = sim_params.get("position_arrays", {})
    if len(pa.get("strikes", [])) > 0:
        spot = get_spot(sim_params)
        spot_range = np.linspace(spot * 0.5, spot * 1.5, 1000)
        has_exotic = has_exotic_legs(pa)
        if has_exotic:
            payoff_curve = compute_hybrid_payoff_curve(spot_range, pa)
        else:
            payoff_curve = compute_payoff_curve(
                spot_range,
                pa["strikes"],
                pa["option_types"],
                pa["position_types"],
                pa["quantities"],
                pa["premiums"],
                pa.get("stock_quantity", 0.0),
                pa.get("stock_entry_price", 0.0),
            )
        breakevens = find_breakeven_points(payoff_curve, spot_range)

# ── Tabs ────────────────────────────────────────────────────────────
tab_names = ["Simulation"]
if has_strategy:
    tab_names.append("P&L Analysis")
    tab_names.append("Greeks")
tab_names.append("Pricing Comparison")
tab_names.append("Path Explorer")
tab_names.append("Custom Model")

tabs = st.tabs(tab_names)

# ═════════════════════════════════════════════════════════════════════════
# TAB 1: Simulation
# ═════════════════════════════════════════════════════════════════════════
with tabs[0]:
    if result is None:
        st.info("Configure parameters in the sidebar and click **Run Simulation**.")
    else:
        # Metric cards
        if is_sp_mode:
            summary = sp_result.pricing_summary
            _sp_metrics = [
                ("Fair Value", f"{summary['fair_value']:.2f}%", "% of notional"),
                ("Expected Return", f"{summary['expected_return']:+.2%}"),
                ("P(Capital Loss)", f"{summary['prob_capital_loss']:.1%}"),
            ]
            if summary.get("autocall_probability") is not None:
                _sp_metrics.append(
                    ("P(Autocall)", f"{summary['autocall_probability']:.1%}")
                )
            else:
                _sp_metrics.append(("Model", get_model_display_name(sim_model)))
            render_metric_row(_sp_metrics)
        else:
            render_metric_row(
                [
                    ("Model", get_model_display_name(sim_model)),
                    ("Paths", f"{int(sim_params.get('n_paths', 10000)):,}"),
                    ("Steps", f"{int(sim_params.get('n_steps', 252)):,}"),
                    ("Time", f"{exec_time * 1000:.0f} ms"),
                ]
            )

        st.markdown("---")

        # Visualization options & path chart
        _pa_viz = sim_params.get("position_arrays", {})
        _exotic_meta_viz = _pa_viz.get("exotic_metadata", [])
        _has_exotic_viz = has_exotic_legs(_pa_viz)

        with st.expander("Visualization Options", expanded=False):
            viz = render_path_controls(
                exotic_metadata=_exotic_meta_viz if _has_exotic_viz else None,
                sp_config=sp_result.product_config if is_sp_mode else None,
            )

        # Build SP overlay callback if in structured product mode
        _sp_overlay_fn = None
        if is_sp_mode:
            _sp_cfg = sp_result.product_config
            _sp_res = sp_result
            _sp_viz = viz.get("sp_viz")

            def _sp_overlay_fn(fig, paths, tg, idx):
                add_structured_overlays(fig, paths, tg, idx, _sp_cfg, _sp_res, _sp_viz)

        render_simulation_chart(
            result=result,
            model_key=sim_model,
            params=sim_params,
            n_display=viz.get("n_display", 150),
            show_bands=viz["show_bands"],
            pnl_values=pnl_vals,
            breakeven_prices=breakevens if not is_sp_mode else None,
            exotic_metadata=_exotic_meta_viz if _has_exotic_viz else None,
            exotic_viz=viz.get("exotic_viz"),
            overlay_fn=_sp_overlay_fn,
            path_view=viz.get("path_view", "Lines"),
            path_alpha=viz.get("path_alpha"),
            balanced_sampling=viz.get("balanced_sampling", False),
            sort_vol_pnl=viz.get("sort_vol_pnl", True),
        )

        # Empirical statistics
        if is_sp_mode:
            summary = sp_result.pricing_summary
            render_metric_row(
                [
                    ("P(Profit)", f"{summary['prob_profit']:.1%}"),
                    ("Worst 5%", f"{summary['worst_case_return']:+.2%}"),
                    ("Best 95%", f"{summary['best_case_return']:+.2%}"),
                    ("Std Error", f"${summary['error']:.2f}"),
                ]
            )
        elif pnl_result is not None:
            risk = pnl_result["risk_metrics"]
            be_str = (
                " / ".join(f"${b:.2f}" for b in breakevens)
                if breakevens is not None and len(breakevens) > 0
                else "N/A"
            )
            render_metric_row(
                [
                    ("P(Profit)", f"{risk.prob_profit:.1%}"),
                    ("Breakeven", be_str),
                    ("Expected P&L", f"${risk.mean_pnl:+,.2f}"),
                    ("VaR 95%", f"${risk.var_95:+,.2f}", "Max loss at 95% confidence"),
                ]
            )
        else:
            tp = result.terminal_prices
            ret = np.mean(tp / result.initial_price - 1) * 100
            p5 = np.percentile(tp, 5)
            render_metric_row(
                [
                    ("E[S(T)]", f"${np.mean(tp):,.2f}"),
                    ("Std[S(T)]", f"${np.std(tp):,.2f}"),
                    ("E[Return]", f"{ret:+.2f}%"),
                    ("5th Percentile", f"${p5:,.2f}", "Worst 5% of outcomes"),
                ]
            )

# ── TAB 2: P&L Analysis (when strategy or SP exists) ─────────────────
if has_strategy:
    with tabs[1]:
        if is_sp_mode:
            # Reuse same charts with SP P&L data
            st.subheader("P&L Distribution & Simulated Returns")
            render_payoff_with_distribution(
                result=result,
                pnl_values=pnl_vals,
                breakeven_prices=None,
                spot=get_spot(sim_params),
            )

            st.markdown("---")

            st.subheader("Realized Volatility vs P&L")
            render_vol_pnl_scatter(
                result=result,
                pnl_values=pnl_vals,
                time_horizon=sim_params.get("time_horizon", 1.0),
                sigma0=get_initial_volatility(sim_model, sim_params),
            )

            st.markdown("---")

            st.subheader("Realized Volatility / Terminal Price / P&L")
            render_3d_pnl_chart(
                result=result,
                pnl_values=pnl_vals,
                time_horizon=sim_params.get("time_horizon", 1.0),
            )
        else:
            _pa_pnl = sim_params.get("position_arrays", {})
            _has_exotic_pnl = has_exotic_legs(_pa_pnl)
            _exotic_meta_pnl = _pa_pnl.get("exotic_metadata", [])

            if result is not None:
                st.subheader("Payoff Diagram & Mark-to-Market P&L")
                _pnl_rate = sim_params.get("risk_free_rate", 0.05)
                _pnl_sigma = get_initial_volatility(sim_model, sim_params)
                _pnl_T = sim_params.get("time_horizon", 1.0)
                render_pnl_by_dte(
                    result=result,
                    position_arrays=_pa_pnl,
                    rate=_pnl_rate,
                    sigma=_pnl_sigma,
                    time_to_expiry=_pnl_T,
                    spot=get_spot(sim_params),
                    exotic_metadata=_exotic_meta_pnl if _exotic_meta_pnl else None,
                    terminal_pnl=pnl_vals,
                    terminal_breakevens=breakevens,
                )

                st.markdown("---")

                st.subheader("Realized Volatility vs P&L")
                render_vol_pnl_scatter_by_dte(
                    result=result,
                    position_arrays=_pa_pnl,
                    rate=_pnl_rate,
                    sigma=_pnl_sigma,
                    time_to_expiry=_pnl_T,
                    sigma0=_pnl_sigma,
                    terminal_pnl=pnl_vals,
                )

            st.markdown("---")

            st.subheader("Realized Volatility / Terminal Price / P&L")
            render_3d_pnl_chart(
                result=result,
                pnl_values=pnl_vals,
                time_horizon=sim_params.get("time_horizon", 1.0),
            )

# ── TAB: Greeks (when strategy or SP exists) ─────────────────────────
if has_strategy:
    with tabs[2]:
        if is_sp_mode:
            st.info(
                "Greeks for structured products — under development. "
                "This feature will be available in a future update."
            )
        else:
            _spot = get_spot(sim_params)
            _rate = sim_params.get("risk_free_rate", 0.05)
            _sigma = get_initial_volatility(sim_model, sim_params)
            _T = sim_params.get("time_horizon", 1.0)

            _selected_greek = render_greek_selector()

            _pa_greeks = sim_params.get("position_arrays", {})
            _q_greeks = sim_params.get("dividend_yield", 0.0)
            _n_steps = int(sim_params.get("n_steps", 252))

            # Model-consistent Greeks under the selected model (None for exotic
            # legs / unsupported models -> practitioner-BS only).
            model_surface = _model_greeks_cached(
                sim_model, sim_params, _pa_greeks, _spot, _rate, _q_greeks, _T
            )
            # Practitioner Black-Scholes: each vanilla leg greeked at its OWN
            # implied vol (backed out from its premium), on the model grid so the
            # two series overlay on a shared DTE slider.
            bs_surface = compute_strategy_greeks_surface_practitioner(
                position_arrays=_pa_greeks,
                spot=_spot,
                rate=_rate,
                time_to_expiry=_T,
                fallback_sigma=_sigma,
                q=_q_greeks,
                n_steps=_n_steps,
            )
            if model_surface is None:
                st.caption(
                    "Model-consistent Greeks are available for vanilla strategies "
                    "under GBM/Heston/Merton/Bates/GARCH — showing practitioner "
                    "Black-Scholes Greeks only."
                )
            elif sim_model.lower() in ("garch", "ngarch", "gjr_garch"):
                st.caption(
                    "Model-consistent GARCH Greeks are Monte-Carlo estimates: "
                    "delta/gamma are robust; vega/theta/rho carry some MC noise."
                )

            render_greeks_with_dte_slider(
                bs_surface,
                spot=_spot,
                selected_greek=_selected_greek,
                model_surface=model_surface,
                primary_label="Black-Scholes (per-leg IV)",
                model_label="Model-consistent",
            )

            # 3D Surface — model-consistent when available, else practitioner-BS.
            st.markdown("---")
            st.subheader("3D Greeks Surface")
            _surface_3d = model_surface if model_surface is not None else bs_surface
            _3d_col1, _3d_col2 = st.columns([2, 2])
            with _3d_col1:
                _greek_3d = render_greek_selector(key="greeks_3d")
            with _3d_col2:
                _colorscale = render_colorscale_selector()
            render_greeks_3d_surface(
                _surface_3d,
                spot=_spot,
                selected_greek=_greek_3d,
                colorscale=_colorscale,
            )

# ── TAB: Pricing Comparison ───────────────────────────────────────────
_next_tab_idx = 3 if has_strategy else 1

with tabs[_next_tab_idx]:
    if result is None:
        st.info("Run a simulation first to see pricing comparison results.")
    elif is_sp_mode:
        # ── Structured Product MC convergence ──
        spot_val = get_spot(sim_params)
        r_val = sim_params.get("risk_free_rate", 0.05)
        T_val = sim_params.get("time_horizon", 1.0)
        n_steps_val = int(sim_params.get("n_steps", 252))
        n_paths_val = int(sim_params.get("n_paths", 10000))
        sp_max_n = min(n_paths_val, 25_000)

        product_type = sp_config["product_type"]
        st.caption(
            f"MC Convergence — {SP_PRODUCT_DESCRIPTIONS.get(product_type, product_type)}"
        )

        # CPN analytical reference (GBM only)
        ref_price, ref_method = None, None
        if product_type == "cpn" and sim_model.lower() == "gbm":
            sigma = sim_params.get("sigma", 0.20)
            ref_price = compute_cpn_analytical_reference(
                sp_config, spot_val, r_val, sigma
            )
            ref_method = "Bond + Call Spread (BS)" if ref_price else None

        render_sp_summary(sp_config, ref_price, ref_method)
        st.markdown("---")

        # Cache + compute
        conv = st.session_state.get("sp_convergence_result")
        if conv is not None and conv["n_done"][-1] != sp_max_n:
            conv = None
        if conv is None:
            n_sims = len(_build_n_paths_grid(sp_max_n))
            with st.spinner(
                f"Computing SP convergence ({n_sims} simulations up to N={sp_max_n:,})..."
            ):
                conv = precompute_sp_convergence(
                    model_key=sim_model,
                    params=sim_params,
                    sp_config=sp_config,
                    spot=spot_val,
                    r=r_val,
                    n_steps=n_steps_val,
                    max_n=sp_max_n,
                )
                st.session_state.sp_convergence_result = conv

        render_animated_convergence_chart(conv)
        st.latex(
            r"\text{SE}(\hat{V}_{\mathrm{MC}}) = \frac{\sigma_{\text{PV}}}{\sqrt{N}}"
        )
        st.markdown("---")
        st.subheader(f"Final Convergence (N = {conv['n_done'][-1]:,})")
        render_final_table(conv)
    else:
        # ── Options MC convergence (existing) ──
        spot_val = get_spot(sim_params)
        r_val = sim_params.get("risk_free_rate", 0.05)
        T_val = sim_params.get("time_horizon", 1.0)
        n_steps_val = int(sim_params.get("n_steps", 252))

        methods = get_available_pricing_methods(sim_model)
        method_str = " \u00b7 ".join(m.replace("_", " ").title() for m in methods)
        st.caption(
            f"Available methods for {get_model_display_name(sim_model)}: **{method_str}**"
        )

        pa = sim_params.get("position_arrays", {})
        legs = extract_legs(pa, spot_val)
        compute_reference_prices(sim_model, sim_params, legs, T_val, spot_val, r_val)

        has_ref_method = "fft" in methods or "analytical" in methods
        all_mc_only = all(leg.get("ref_price") is None for leg in legs)
        _has_exotic_pricing = any(
            leg.get("instrument_class", "vanilla") != "vanilla" for leg in legs
        )
        has_vanilla_legs = any(
            leg.get("instrument_class", "vanilla") == "vanilla" for leg in legs
        )

        if all_mc_only:
            if _has_exotic_pricing and not has_vanilla_legs:
                if sim_model.lower() != "gbm":
                    st.info(
                        "Exotic analytic reference pricing is only available with the **GBM** model. "
                        "Switch to GBM to see reference prices, or compare using MC convergence below."
                    )
                elif has_ref_method:
                    st.warning(
                        "Exotic analytic pricing returned no result. "
                        "Check terminal logs for details."
                    )
            elif has_ref_method:
                st.warning(
                    "Reference pricing (FFT/Analytical) returned no result. "
                    "If you just updated the code, try restarting the Streamlit server. "
                    "Check terminal logs for details."
                )

        if pnl_result is None:
            st.info("No strategy defined \u2014 pricing a single ATM Call option.")

        render_legs_summary(legs)
        st.markdown("---")

        n_paths_val = int(sim_params.get("n_paths", 10000))
        conv = st.session_state.get("convergence_result")
        if conv is not None and conv["n_done"][-1] != n_paths_val:
            conv = None
        if conv is None:
            n_sims = (
                len(
                    [
                        n
                        for n in [
                            100,
                            250,
                            500,
                            1000,
                            2500,
                            5000,
                            10000,
                            25000,
                            50000,
                            100000,
                        ]
                        if n < n_paths_val
                    ]
                )
                + 1
            )
            with st.spinner(
                f"Computing convergence ({n_sims} simulations up to N={n_paths_val:,})\u2026"
            ):
                conv = precompute_convergence(
                    model_key=sim_model,
                    params=sim_params,
                    legs=legs,
                    T=T_val,
                    spot=spot_val,
                    r=r_val,
                    n_steps=n_steps_val,
                    max_n=n_paths_val,
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

_next_tab_idx += 1

# ── TAB: Path Explorer ────────────────────────────────────────────────
explorer_tab_idx = _next_tab_idx
with tabs[explorer_tab_idx]:
    if result is None:
        st.info("Run a simulation first to explore individual paths.")
    elif is_sp_mode:
        explorer_params = render_explorer_params(sim_model)

        try:
            explorer_result = run_simulation(sim_model, explorer_params)

            _sp_explorer = evaluate_structured_product_on_paths(
                explorer_result,
                sp_result.product_config,
                explorer_params,
            )

            render_structured_path_explorer(
                result=explorer_result,
                sp_result=_sp_explorer,
                model_key=sim_model,
                params=explorer_params,
                product_config=sp_result.product_config,
            )

            tp = explorer_result.price_paths[0, -1]
            s0 = explorer_params["spot"]
            init_vol = get_initial_volatility(sim_model, explorer_params) * 100
            path_pnl = float(
                _sp_explorer.per_path_returns[0]
                * _sp_explorer.product_config["product_params"]["notional"]
            )
            status = "Profit" if path_pnl >= 0 else "Loss"
            render_metric_row(
                [
                    ("Terminal Price", f"${tp:,.2f}"),
                    ("Return", f"{(tp / s0 - 1) * 100:+.2f}%"),
                    ("Initial Vol", f"{init_vol:.1f}%"),
                    ("Path P&L", f"${path_pnl:+,.2f}"),
                    ("Status", status),
                ]
            )
        except (ValueError, RuntimeError) as e:
            st.error(f"Simulation error: {e}")
    else:
        explorer_params = render_explorer_params(sim_model)

        _pa_explorer = sim_params.get("position_arrays", {})
        _has_exotic_explorer = has_exotic_legs(_pa_explorer)
        _exotic_meta_explorer = _pa_explorer.get("exotic_metadata", [])
        _has_strategy_explorer = len(_pa_explorer.get("strikes", [])) > 0

        try:
            explorer_result = run_simulation(sim_model, explorer_params)

            # Seed-sensitivity overlay: run extra seeds (base_seed + i) with the
            # SAME parameters so the user can see how much the seed alone moves
            # the path.
            _n_seeds = int(explorer_params.get("n_seeds", 1))
            _seed_paths = None
            if _n_seeds > 1:
                _base_seed = int(explorer_params.get("seed", 42))
                _extra = []
                for _i in range(1, _n_seeds):
                    _p = dict(explorer_params)
                    _p["seed"] = _base_seed + _i
                    _extra.append(run_simulation(sim_model, _p).price_paths[0])
                _seed_paths = np.asarray(_extra)

            explorer_pnl = render_path_explorer_chart(
                result=explorer_result,
                model_key=sim_model,
                params=explorer_params,
                position_arrays=_pa_explorer if _has_strategy_explorer else None,
                exotic_metadata=_exotic_meta_explorer if _has_exotic_explorer else None,
                seed_paths=_seed_paths,
            )

            tp = explorer_result.price_paths[0, -1]
            s0 = explorer_params["spot"]
            init_vol = get_initial_volatility(sim_model, explorer_params) * 100

            _explorer_metrics = [
                ("Terminal Price", f"${tp:,.2f}"),
                ("Return", f"{(tp / s0 - 1) * 100:+.2f}%"),
                ("Initial Vol", f"{init_vol:.1f}%"),
            ]
            if _has_strategy_explorer and explorer_pnl is not None:
                status = "Profit" if explorer_pnl >= 0 else "Loss"
                _explorer_metrics.extend(
                    [
                        ("Path P&L", f"${explorer_pnl:+,.2f}"),
                        ("Status", status),
                    ]
                )
            render_metric_row(_explorer_metrics)
        except (ValueError, RuntimeError) as e:
            st.error(f"Simulation error: {e}")

# ── TAB: Custom Model ────────────────────────────────────────────────
custom_tab_idx = explorer_tab_idx + 1
with tabs[custom_tab_idx]:
    render_custom_model_editor()


st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════
# STRATEGY EQUATIONS (options — vanilla, exotic, spreads)
# ═════════════════════════════════════════════════════════════════════════

_active_strategy = st.session_state.get("pnl_last_strategy")
_active_sp_config = st.session_state.get("sp_config")

if _active_strategy and not _active_sp_config:
    _strat_equations = get_option_strategy_equations(_active_strategy)
    if _strat_equations:
        with st.expander(
            f"Strategy Equations — {_strat_equations['name']}", expanded=False
        ):
            st.markdown(f"**{_strat_equations['name']}**")
            st.caption(_strat_equations["description"])
            for eq_block in _strat_equations["equations"]:
                st.markdown(f"**{eq_block['label']}**")
                st.latex(eq_block["latex"])

# ═════════════════════════════════════════════════════════════════════════
# PRODUCT EQUATIONS (structured products, above model equations)
# ═════════════════════════════════════════════════════════════════════════

if _active_sp_config:
    _sp_type = _active_sp_config["product_type"]
    _sp_equations = _get_structured_product_equations(
        _sp_type, _active_sp_config.get("product_params", {})
    )
    with st.expander(f"Product Equations — {_sp_equations['name']}", expanded=False):
        st.markdown(f"**{_sp_equations['name']}**")
        st.caption(_sp_equations["description"])
        for eq_block in _sp_equations["equations"]:
            st.markdown(f"**{eq_block['label']}**")
            st.latex(eq_block["latex"])

# ═════════════════════════════════════════════════════════════════════════
# MODEL EQUATIONS (collapsible, bottom of page)
# ═════════════════════════════════════════════════════════════════════════

active_model = st.session_state.get("simulation_model", model_key)
try:
    spec = get_model(active_model)
    with st.expander(f"Model Equations — {spec.name}", expanded=False):
        st.markdown("**Stochastic Dynamics**")
        st.latex(spec.equation_main)
        if spec.equation_vol:
            st.caption("Volatility dynamics:")
            st.latex(spec.equation_vol)
        if spec.equation_jump:
            st.caption("Jump distribution:")
            st.latex(spec.equation_jump)

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
