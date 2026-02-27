"""
Results Summary Component - Display simulation results and statistics.

Provides:
- Key metrics display
- Distribution statistics
- Model information summary
"""

import streamlit as st
import numpy as np
from typing import Dict, Any, Optional

from backend.simulation.base import SimulationResult
from utils.model_helpers import (
    get_model_icon,
    get_volatility_type,
    format_volatility_display,
    compute_summary_statistics,
)
from services.simulation_service import (
    get_model_characteristics,
    compute_long_run_volatility,
    get_initial_volatility,
)


def render_results_summary(
    result: SimulationResult,
    model_key: str,
    params: Dict[str, Any]
):
    """
    Render comprehensive results summary.

    Args:
        result: SimulationResult from simulation
        model_key: Model identifier
        params: All parameters used
    """
    # Key metrics row
    _render_key_metrics(result, model_key, params)

    # Distribution statistics
    with st.expander("📊 Distribution Statistics", expanded=True):
        _render_distribution_stats(result)

    # Volatility info
    _render_volatility_info(result, model_key, params)


def _render_key_metrics(
    result: SimulationResult,
    model_key: str,
    params: Dict[str, Any]
):
    """Render key simulation metrics."""
    col1, col2, col3, col4 = st.columns(4)

    terminal_prices = result.terminal_prices
    initial_price = result.initial_price

    with col1:
        mean_terminal = np.mean(terminal_prices)
        expected_return = (mean_terminal / initial_price - 1) * 100
        st.metric(
            "Mean Terminal",
            f"${mean_terminal:.2f}",
            f"{expected_return:+.1f}%"
        )

    with col2:
        median_terminal = np.median(terminal_prices)
        st.metric(
            "Median Terminal",
            f"${median_terminal:.2f}"
        )

    with col3:
        std_terminal = np.std(terminal_prices)
        cv = std_terminal / mean_terminal
        st.metric(
            "Std Dev",
            f"${std_terminal:.2f}",
            f"CV: {cv:.2%}"
        )

    with col4:
        # Realized volatility
        initial_vol = get_initial_volatility(model_key, params)
        realized_vol = result.terminal_volatility

        if realized_vol is not None:
            st.metric(
                "Realized Vol (ann.)",
                f"{realized_vol * 100:.1f}%",
                f"vs {initial_vol * 100:.1f}% initial"
            )
        else:
            # For constant vol models, display the constant volatility
            st.metric(
                "Volatility (ann.)",
                f"{initial_vol * 100:.1f}%",
                "constant"
            )


def _render_distribution_stats(result: SimulationResult):
    """Render distribution statistics."""
    terminal_prices = result.terminal_prices
    initial_price = result.initial_price
    returns = np.log(terminal_prices / initial_price)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Price Statistics**")
        stats_price = {
            "Mean": f"${np.mean(terminal_prices):.2f}",
            "Median": f"${np.median(terminal_prices):.2f}",
            "Std Dev": f"${np.std(terminal_prices):.2f}",
            "Min": f"${np.min(terminal_prices):.2f}",
            "Max": f"${np.max(terminal_prices):.2f}",
        }
        for label, value in stats_price.items():
            st.markdown(f"- {label}: {value}")

    with col2:
        st.markdown("**Return Statistics**")
        stats_return = {
            "Mean Return": f"{np.mean(returns) * 100:.2f}%",
            "Median Return": f"{np.median(returns) * 100:.2f}%",
            "Volatility": f"{np.std(returns) * 100:.2f}%",
            "Min Return": f"{np.min(returns) * 100:.2f}%",
            "Max Return": f"{np.max(returns) * 100:.2f}%",
        }
        for label, value in stats_return.items():
            st.markdown(f"- {label}: {value}")

    with col3:
        st.markdown("**Risk Metrics**")
        var_95 = np.percentile(terminal_prices, 5)
        var_99 = np.percentile(terminal_prices, 1)
        cvar_95 = np.mean(terminal_prices[terminal_prices <= var_95])

        # Skewness and kurtosis
        n = len(returns)
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        skewness = np.mean(((returns - mean_r) / std_r) ** 3)
        kurtosis = np.mean(((returns - mean_r) / std_r) ** 4) - 3

        stats_risk = {
            "VaR 95%": f"${var_95:.2f}",
            "VaR 99%": f"${var_99:.2f}",
            "CVaR 95%": f"${cvar_95:.2f}",
            "Skewness": f"{skewness:.3f}",
            "Excess Kurt.": f"{kurtosis:.3f}",
        }
        for label, value in stats_risk.items():
            st.markdown(f"- {label}: {value}")


def _render_volatility_info(
    result: SimulationResult,
    model_key: str,
    params: Dict[str, Any]
):
    """Render volatility-specific information."""
    characteristics = get_model_characteristics(model_key)

    if characteristics["has_stochastic_vol"] and result.volatility_paths is not None:
        with st.expander("📈 Volatility Analysis", expanded=False):
            vol_paths = result.volatility_paths

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                mean_vol = np.mean(vol_paths[:, -1])
                st.metric("Mean Terminal Vol", f"{mean_vol * 100:.1f}%")

            with col2:
                median_vol = np.median(vol_paths[:, -1])
                st.metric("Median Terminal Vol", f"{median_vol * 100:.1f}%")

            with col3:
                long_run = compute_long_run_volatility(model_key, params)
                if long_run:
                    st.metric("Long-Run Vol", f"{long_run * 100:.1f}%")
                else:
                    st.metric("Long-Run Vol", "N/A")

            with col4:
                vol_of_vol = np.std(vol_paths[:, -1])
                st.metric("Vol of Vol", f"{vol_of_vol * 100:.2f}%")

            # Volatility percentiles
            st.markdown("**Terminal Volatility Percentiles**")
            percentiles = [5, 25, 50, 75, 95]
            terminal_vols = vol_paths[:, -1] * 100
            pct_values = np.percentile(terminal_vols, percentiles)

            cols = st.columns(len(percentiles))
            for i, (pct, val) in enumerate(zip(percentiles, pct_values)):
                with cols[i]:
                    st.metric(f"P{pct}", f"{val:.1f}%")


def render_simulation_info(
    model_key: str,
    params: Dict[str, Any],
    execution_time: Optional[float] = None
):
    """
    Render simulation configuration info.

    Args:
        model_key: Model identifier
        params: All parameters
        execution_time: Simulation execution time in seconds
    """
    from config.model_registry import get_model

    model = get_model(model_key)

    st.markdown("### Simulation Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Model:** {get_model_icon(model_key)} {model.name}")
        st.markdown(f"**Volatility Type:** {get_volatility_type(model_key)}")
        st.markdown(f"**Spot Price:** ${params.get('spot', 100):.2f}")
        st.markdown(f"**Time Horizon:** {params.get('time_horizon', 1.0):.1f} years")

    with col2:
        st.markdown(f"**Paths:** {int(params.get('n_paths', 10000)):,}")
        st.markdown(f"**Steps:** {int(params.get('n_steps', 252)):,}")
        seed = params.get('seed', 42)
        st.markdown(f"**Seed:** {seed if seed else 'Random'}")
        if execution_time is not None:
            st.markdown(f"**Execution Time:** {execution_time:.2f}s")


def render_percentile_table(result: SimulationResult):
    """Render percentile table for terminal prices."""
    import pandas as pd

    terminal_prices = result.terminal_prices
    initial_price = result.initial_price

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    values = np.percentile(terminal_prices, percentiles)
    returns = (values / initial_price - 1) * 100

    df = pd.DataFrame({
        "Percentile": [f"{p}%" for p in percentiles],
        "Price": [f"${v:.2f}" for v in values],
        "Return": [f"{r:+.1f}%" for r in returns],
    })

    st.dataframe(df, width="stretch", hide_index=True)


def render_paths_summary(result: SimulationResult, n_display: int = 5):
    """Render summary of individual paths."""
    import pandas as pd

    terminal_prices = result.terminal_prices
    initial_price = result.initial_price

    # Sort by terminal value
    sorted_idx = np.argsort(terminal_prices)

    # Get best and worst paths
    worst_idx = sorted_idx[:n_display]
    best_idx = sorted_idx[-n_display:][::-1]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Best Performing Paths**")
        best_data = {
            "Path": [f"#{i+1}" for i in best_idx],
            "Terminal": [f"${terminal_prices[i]:.2f}" for i in best_idx],
            "Return": [f"{(terminal_prices[i]/initial_price - 1)*100:+.1f}%" for i in best_idx],
        }
        st.dataframe(pd.DataFrame(best_data), width="stretch", hide_index=True)

    with col2:
        st.markdown("**Worst Performing Paths**")
        worst_data = {
            "Path": [f"#{i+1}" for i in worst_idx],
            "Terminal": [f"${terminal_prices[i]:.2f}" for i in worst_idx],
            "Return": [f"{(terminal_prices[i]/initial_price - 1)*100:+.1f}%" for i in worst_idx],
        }
        st.dataframe(pd.DataFrame(worst_data), width="stretch", hide_index=True)


def render_model_equations(model_key: str, params: Dict[str, Any]):
    """Render model equations with LaTeX."""
    from config.model_registry import get_model
    from utils.model_helpers import get_model_equations

    model = get_model(model_key)
    equations = get_model_equations(model_key)

    with st.expander("📐 Model Equations", expanded=False):
        st.markdown(f"**{model.name}**")

        st.latex(equations["main"])

        if "volatility" in equations:
            st.markdown("*Volatility dynamics:*")
            st.latex(equations["volatility"])

        if "jump" in equations:
            st.markdown("*Jump distribution:*")
            st.latex(equations["jump"])

        # Conditions
        if model.feller_condition:
            st.markdown(f"*Feller condition:* ${model.feller_condition}$")

        if model.stationarity_condition:
            st.markdown(f"*Stationarity:* ${model.stationarity_condition}$")

        # Pricing method equations
        has_pricing_eq = "analytical" in equations or "cf" in equations or "mc" in equations
        if has_pricing_eq:
            st.markdown("---")
            st.markdown("**Pricing Methods**")
            if "analytical" in equations:
                st.markdown("*Analytical (Black-Scholes):*")
                st.latex(equations["analytical"])
            if "cf" in equations:
                st.markdown("*FFT — Characteristic Function:*")
                st.latex(equations["cf"])
            if "mc" in equations:
                st.markdown("*Monte Carlo — Discretization:*")
                st.latex(equations["mc"])
