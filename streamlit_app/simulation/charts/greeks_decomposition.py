"""
Greeks P&L Decomposition Chart Module.

Pedagogical module teaching that:
P&L = Delta P&L + Gamma P&L + Theta P&L + Vega P&L + Residual

Formula:
Total P&L ≈ Δ·ΔS + ½Γ·(ΔS)² + Θ·Δt + V·Δσ + residual

Key insight: "Long gamma = profits on large moves regardless of direction"
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, List, Optional, NamedTuple
from dataclasses import dataclass

from backend.greeks.analytic import (
    bs_greeks_first_order,
    bs_all_greeks,
    VEGA_SCALE,
    THETA_SCALE
)
from config.constants import CHART_HEIGHT_STANDARD, CHART_HEIGHT_LARGE
from config.styles import render_stats_row


@dataclass
class GreeksDecomposition:
    """Results of Greeks-based P&L decomposition."""
    delta_pnl: np.ndarray       # Δ·ΔS per path
    gamma_pnl: np.ndarray       # ½Γ·(ΔS)² per path
    theta_pnl: np.ndarray       # Θ·Δt (scalar broadcast)
    vega_pnl: np.ndarray        # V·Δσ per path (for stoch vol models)
    residual: np.ndarray        # Unexplained P&L
    total_pnl: np.ndarray       # Actual P&L

    # Aggregates
    portfolio_delta: float
    portfolio_gamma: float
    portfolio_theta: float
    portfolio_vega: float

    # Model info
    has_stochastic_vol: bool
    time_horizon: float


def calculate_portfolio_greeks(
    spot: float,
    strikes: np.ndarray,
    option_types: np.ndarray,  # 1 for call, -1 for put
    position_types: np.ndarray,  # 1 for long, -1 for short
    quantities: np.ndarray,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0
) -> Dict[str, float]:
    """
    Calculate aggregate portfolio Greeks.

    Returns
    -------
    dict with keys: delta, gamma, vega, theta, rho
    """
    portfolio_greeks = {
        'delta': 0.0,
        'gamma': 0.0,
        'vega': 0.0,
        'theta': 0.0,
        'rho': 0.0
    }

    for i in range(len(strikes)):
        is_call = option_types[i] == 1
        sign = position_types[i]
        qty = quantities[i]

        delta, gamma, vega, theta, rho = bs_greeks_first_order(
            s=spot,
            k=strikes[i],
            t=time_to_expiry,
            r=risk_free_rate,
            q=dividend_yield,
            sigma=volatility,
            is_call=is_call
        )

        # Accumulate with position sign and quantity
        # Note: Each contract is for 100 shares
        multiplier = sign * qty * 100
        portfolio_greeks['delta'] += delta * multiplier
        portfolio_greeks['gamma'] += gamma * multiplier
        portfolio_greeks['vega'] += vega * multiplier
        portfolio_greeks['theta'] += theta * multiplier
        portfolio_greeks['rho'] += rho * multiplier

    return portfolio_greeks


def decompose_pnl_by_greeks(
    simulation_result,
    pnl_values: np.ndarray,
    params: Dict[str, Any]
) -> Optional[GreeksDecomposition]:
    """
    Decompose simulated P&L into Greeks components.

    Parameters
    ----------
    simulation_result : SimulationResult
        Contains price_paths, volatility_paths (if stochastic vol), time_grid
    pnl_values : np.ndarray
        Actual P&L per simulation path
    params : dict
        Simulation and strategy parameters

    Returns
    -------
    GreeksDecomposition or None
    """
    position_arrays = params.get('position_arrays', {})
    if len(position_arrays.get('strikes', [])) == 0:
        return None

    spot = params.get('spot_price', 100.0)
    time_horizon = params.get('time_horizon', 1.0)
    risk_free_rate = params.get('risk_free_rate', 0.05)
    volatility = params.get('volatility', 0.20)

    # Calculate portfolio Greeks at t=0
    portfolio_greeks = calculate_portfolio_greeks(
        spot=spot,
        strikes=position_arrays['strikes'],
        option_types=position_arrays['option_types'],
        position_types=position_arrays['position_types'],
        quantities=position_arrays['quantities'],
        time_to_expiry=time_horizon,
        risk_free_rate=risk_free_rate,
        volatility=volatility
    )

    # Get terminal prices
    terminal_prices = simulation_result.terminal_prices
    n_paths = len(terminal_prices)

    # Price change per path
    delta_s = terminal_prices - spot

    # Check for stochastic volatility
    has_stoch_vol = (
        hasattr(simulation_result, 'volatility_paths') and
        simulation_result.volatility_paths is not None
    )

    # Delta P&L: Δ · ΔS
    delta_pnl = portfolio_greeks['delta'] * delta_s

    # Gamma P&L: ½Γ · (ΔS)²
    gamma_pnl = 0.5 * portfolio_greeks['gamma'] * (delta_s ** 2)

    # Theta P&L: Θ · Δt (in days, so unscale)
    # portfolio_greeks['theta'] is per day, time_horizon is in years
    theta_pnl = np.full(n_paths, portfolio_greeks['theta'] * time_horizon * 365)

    # Vega P&L: V · Δσ
    if has_stoch_vol:
        # For stochastic vol models, compute realized vol change
        terminal_vol = simulation_result.volatility_paths[:, -1]
        initial_vol = simulation_result.volatility_paths[:, 0]
        delta_sigma = terminal_vol - initial_vol
        # portfolio_greeks['vega'] is per 1% vol, delta_sigma is decimal
        vega_pnl = portfolio_greeks['vega'] * (delta_sigma * 100)
    else:
        # For constant vol models, no vega P&L from vol changes
        vega_pnl = np.zeros(n_paths)

    # Sum of Greeks-explained P&L
    greeks_explained = delta_pnl + gamma_pnl + theta_pnl + vega_pnl

    # Residual: what the Taylor expansion doesn't capture
    residual = pnl_values - greeks_explained

    return GreeksDecomposition(
        delta_pnl=delta_pnl,
        gamma_pnl=gamma_pnl,
        theta_pnl=theta_pnl,
        vega_pnl=vega_pnl,
        residual=residual,
        total_pnl=pnl_values,
        portfolio_delta=portfolio_greeks['delta'],
        portfolio_gamma=portfolio_greeks['gamma'],
        portfolio_theta=portfolio_greeks['theta'],
        portfolio_vega=portfolio_greeks['vega'],
        has_stochastic_vol=has_stoch_vol,
        time_horizon=time_horizon
    )


def render_greeks_decomposition_tab(
    simulation_result,
    pnl_result: Dict[str, Any],
    params: Dict[str, Any]
) -> None:
    """
    Render the Greeks P&L Decomposition educational tab.

    Shows how Total P&L = Delta P&L + Gamma P&L + Theta P&L + Vega P&L + Residual
    """
    if simulation_result is None or pnl_result is None:
        st.info("Run a simulation with a strategy defined to see Greeks P&L decomposition.")
        return

    st.markdown("### Greeks P&L Decomposition")
    st.caption(
        "Understand your strategy's P&L drivers using the Taylor expansion: "
        "how much did Delta, Gamma, Theta, and Vega contribute?"
    )

    # Calculate decomposition
    decomp = decompose_pnl_by_greeks(
        simulation_result,
        pnl_result['pnl_values'],
        params
    )

    if decomp is None:
        st.warning("Unable to calculate Greeks decomposition. Check strategy definition.")
        return

    # Educational formula box
    st.markdown("""
    <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #0284c7;">
        <h5 style="color: #0c4a6e; margin-top: 0;">P&L Decomposition Formula</h5>
        <p style="font-family: monospace; font-size: 1.1em; margin-bottom: 0.5rem;">
            Total P&L ≈ <span style="color: #2563eb;">Δ·ΔS</span> +
            <span style="color: #16a34a;">½Γ·(ΔS)²</span> +
            <span style="color: #dc2626;">Θ·Δt</span> +
            <span style="color: #9333ea;">V·Δσ</span> +
            <span style="color: #64748b;">Residual</span>
        </p>
        <p style="margin-bottom: 0; font-size: 0.9em; color: #475569;">
            This Taylor expansion shows how first-order Greeks explain most of your P&L.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Portfolio Greeks summary
    st.markdown("#### Portfolio Greeks at Inception")

    delta_variant = "green" if abs(decomp.portfolio_delta) < 10 else "amber" if abs(decomp.portfolio_delta) < 50 else "blue"
    gamma_sign = "+" if decomp.portfolio_gamma > 0 else ""
    gamma_variant = "green" if decomp.portfolio_gamma > 0 else "red"

    greeks_stats = [
        ("Delta (Δ)", f"{decomp.portfolio_delta:+.2f}", "Shares equivalent"),
        ("Gamma (Γ)", f"{gamma_sign}{decomp.portfolio_gamma:.4f}", "Long = benefits from moves"),
        ("Theta (Θ)", f"${decomp.portfolio_theta:.2f}/day", "Time decay"),
        ("Vega (V)", f"${decomp.portfolio_vega:.2f}/1%", "Vol sensitivity"),
    ]
    render_stats_row(greeks_stats, [delta_variant, gamma_variant, "red", "purple"])

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs([
        "Attribution Chart",
        "Component Analysis",
        "Residual Analysis"
    ])

    with tab1:
        _render_attribution_chart(decomp)

    with tab2:
        _render_component_analysis(decomp)

    with tab3:
        _render_residual_analysis(decomp)


def _render_attribution_chart(decomp: GreeksDecomposition) -> None:
    """Render stacked bar chart of P&L components."""
    st.markdown("#### P&L Attribution by Greek")

    # Calculate means
    components = {
        'Delta P&L': np.mean(decomp.delta_pnl),
        'Gamma P&L': np.mean(decomp.gamma_pnl),
        'Theta P&L': np.mean(decomp.theta_pnl),
        'Vega P&L': np.mean(decomp.vega_pnl) if decomp.has_stochastic_vol else 0.0,
        'Residual': np.mean(decomp.residual)
    }

    # Create waterfall-style bar chart
    labels = list(components.keys()) + ['Total P&L']
    values = list(components.values()) + [np.mean(decomp.total_pnl)]

    colors = ['#2563eb', '#16a34a', '#dc2626', '#9333ea', '#64748b', '#1e3a5f']

    fig = go.Figure()

    # Main bars
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"${v:+,.0f}" for v in values],
        textposition='outside',
        hovertemplate='%{x}: $%{y:,.2f}<extra></extra>'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="Mean P&L Attribution by Greek Component",
        yaxis_title="P&L ($)",
        height=CHART_HEIGHT_STANDARD,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Percentage breakdown
    total_abs = sum(abs(v) for v in list(components.values())[:-1])  # Exclude residual
    if total_abs > 0:
        st.markdown("**Contribution Breakdown:**")
        cols = st.columns(5)
        for i, (name, val) in enumerate(components.items()):
            pct = abs(val) / total_abs * 100 if total_abs > 0 else 0
            with cols[i]:
                st.metric(name, f"{pct:.1f}%", f"${val:+,.0f}")

    # Key insight based on strategy type
    if decomp.portfolio_gamma > 0:
        st.markdown("""
        <div style="background: #dcfce7; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <h5 style="color: #166534; margin-top: 0;">Key Insight: Long Gamma Position</h5>
            <p style="margin-bottom: 0;">
                Your positive Gamma P&L means you <strong>profit from large price moves</strong> regardless
                of direction. This is the "convexity" advantage of owning options - your P&L curves upward
                on big moves. The cost is the negative Theta (time decay).
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif decomp.portfolio_gamma < 0:
        st.markdown("""
        <div style="background: #fef2f2; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <h5 style="color: #991b1b; margin-top: 0;">Key Insight: Short Gamma Position</h5>
            <p style="margin-bottom: 0;">
                Your negative Gamma means large price moves <strong>hurt your position</strong>.
                You're essentially "selling insurance" against big moves. The benefit is positive Theta
                (you collect time decay), but the risk is unlimited on the downside.
            </p>
        </div>
        """, unsafe_allow_html=True)


def _render_component_analysis(decomp: GreeksDecomposition) -> None:
    """Detailed analysis of each P&L component."""
    st.markdown("#### Component-by-Component Analysis")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Delta P&L vs Price Move",
            "Gamma P&L vs Price Move",
            "P&L Component Distributions",
            "Cumulative Component P&L"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # Sort by price change for cleaner visualization
    delta_s = decomp.total_pnl - decomp.residual - decomp.gamma_pnl - decomp.theta_pnl - decomp.vega_pnl
    # Actually compute ΔS from delta_pnl
    if decomp.portfolio_delta != 0:
        delta_s_approx = decomp.delta_pnl / decomp.portfolio_delta
    else:
        delta_s_approx = np.zeros_like(decomp.delta_pnl)

    # Sample for scatter plots (avoid overplotting)
    sample_size = min(1000, len(decomp.delta_pnl))
    idx = np.random.choice(len(decomp.delta_pnl), sample_size, replace=False)

    # 1. Delta P&L vs ΔS - should be linear
    fig.add_trace(
        go.Scatter(
            x=delta_s_approx[idx],
            y=decomp.delta_pnl[idx],
            mode='markers',
            marker=dict(color='#2563eb', size=3, opacity=0.5),
            name='Delta P&L',
            hovertemplate='ΔS: $%{x:.2f}<br>Delta P&L: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add theoretical line
    x_range = np.linspace(delta_s_approx.min(), delta_s_approx.max(), 100)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=decomp.portfolio_delta * x_range,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Theoretical',
            showlegend=False
        ),
        row=1, col=1
    )

    # 2. Gamma P&L vs ΔS - should be parabolic
    fig.add_trace(
        go.Scatter(
            x=delta_s_approx[idx],
            y=decomp.gamma_pnl[idx],
            mode='markers',
            marker=dict(color='#16a34a', size=3, opacity=0.5),
            name='Gamma P&L'
        ),
        row=1, col=2
    )

    # Theoretical parabola
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=0.5 * decomp.portfolio_gamma * (x_range ** 2),
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Theoretical',
            showlegend=False
        ),
        row=1, col=2
    )

    # 3. Distribution of components
    for name, data, color in [
        ('Delta', decomp.delta_pnl, '#2563eb'),
        ('Gamma', decomp.gamma_pnl, '#16a34a'),
        ('Theta', decomp.theta_pnl, '#dc2626'),
        ('Residual', decomp.residual, '#64748b'),
    ]:
        fig.add_trace(
            go.Histogram(
                x=data,
                name=name,
                marker_color=color,
                opacity=0.6,
                nbinsx=50
            ),
            row=2, col=1
        )

    # 4. Sorted cumulative P&L by component
    sorted_idx = np.argsort(decomp.total_pnl)
    n = len(sorted_idx)
    x_pct = np.arange(n) / n * 100

    for name, data, color in [
        ('Total P&L', decomp.total_pnl, '#1e3a5f'),
        ('Delta', decomp.delta_pnl, '#2563eb'),
        ('Gamma', decomp.gamma_pnl, '#16a34a'),
    ]:
        fig.add_trace(
            go.Scatter(
                x=x_pct,
                y=data[sorted_idx],
                mode='lines',
                name=name,
                line=dict(color=color, width=2 if name == 'Total P&L' else 1)
            ),
            row=2, col=2
        )

    fig.update_xaxes(title_text="Price Change (ΔS)", row=1, col=1)
    fig.update_xaxes(title_text="Price Change (ΔS)", row=1, col=2)
    fig.update_xaxes(title_text="P&L ($)", row=2, col=1)
    fig.update_xaxes(title_text="Percentile", row=2, col=2)
    fig.update_yaxes(title_text="Delta P&L ($)", row=1, col=1)
    fig.update_yaxes(title_text="Gamma P&L ($)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=2, col=2)

    fig.update_layout(
        height=CHART_HEIGHT_LARGE,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_residual_analysis(decomp: GreeksDecomposition) -> None:
    """Analyze what the Taylor expansion doesn't capture."""
    st.markdown("#### Residual Analysis")
    st.caption(
        "The residual shows what our first-order Taylor expansion doesn't explain. "
        "Large residuals indicate higher-order effects, jumps, or model limitations."
    )

    # Statistics
    residual_mean = np.mean(decomp.residual)
    residual_std = np.std(decomp.residual)
    total_pnl_std = np.std(decomp.total_pnl)
    r_squared = 1 - (residual_std ** 2) / (total_pnl_std ** 2) if total_pnl_std > 0 else 0

    stats = [
        ("Mean Residual", f"${residual_mean:+,.2f}", "Should be near zero"),
        ("Residual Std Dev", f"${residual_std:,.2f}", "Unexplained variation"),
        ("R-squared", f"{r_squared:.1%}", "Variance explained by Greeks"),
        ("Residual/Total Std", f"{residual_std/total_pnl_std:.1%}" if total_pnl_std > 0 else "N/A",
         "Lower is better"),
    ]

    r2_variant = "green" if r_squared > 0.8 else "amber" if r_squared > 0.5 else "red"
    render_stats_row(stats, ["slate", "amber", r2_variant, "slate"])

    # Residual histogram
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=decomp.residual,
        nbinsx=50,
        marker_color='#64748b',
        opacity=0.7,
        name='Residual'
    ))

    # Add mean line
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=residual_mean, line_dash="solid", line_color="red",
                  annotation_text=f"Mean: ${residual_mean:+,.0f}")

    fig.update_layout(
        title="Residual Distribution (Unexplained P&L)",
        xaxis_title="Residual P&L ($)",
        yaxis_title="Frequency",
        height=CHART_HEIGHT_STANDARD - 100
    )

    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    if r_squared > 0.9:
        st.success(
            f"The Greeks decomposition explains {r_squared:.1%} of P&L variance. "
            "First-order sensitivities are sufficient for this strategy."
        )
    elif r_squared > 0.7:
        st.info(
            f"Greeks explain {r_squared:.1%} of variance. Some P&L comes from "
            "higher-order effects (second-order Greeks, path dependency, or jumps)."
        )
    else:
        st.warning(
            f"Only {r_squared:.1%} of variance explained by first-order Greeks. "
            "Consider higher-order Greeks or the impact of path-dependent features."
        )

    # What contributes to residual
    st.markdown("""
    <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <h5 style="color: #334155; margin-top: 0;">What Causes Residuals?</h5>
        <ul style="margin-bottom: 0;">
            <li><strong>Higher-order Greeks</strong>: Vanna (ΔΓ/Δσ), Volga (ΔV/Δσ), Charm (ΔΔ/Δt)</li>
            <li><strong>Large price moves</strong>: Taylor expansion is local - breaks down for big ΔS</li>
            <li><strong>Jumps</strong>: Merton/Bates models have discontinuous price paths</li>
            <li><strong>Stochastic volatility</strong>: Vol changes affect all Greeks dynamically</li>
            <li><strong>Discrete hedging</strong>: Greeks assume continuous rebalancing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
