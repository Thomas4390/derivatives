"""
Volatility Impact on P&L Module.

Pedagogical module teaching how volatility affects option P&L:

1. IV vs Realized Vol: Compare implied volatility to realized volatility
   - If RV > IV: option buyers profit (long gamma wins)
   - If RV < IV: option sellers profit (short gamma wins)

2. Vega P&L Attribution: Track cumulative P&L from vol changes (stochastic vol models)

3. Fat Tails Explained: Why Heston/Bates have different tails than GBM

4. Price-Volatility Surface: How option value changes with spot and vol
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, Optional
from scipy import stats

from backend.greeks.analytic import bs_all_greeks
from config.constants import CHART_HEIGHT_STANDARD, CHART_HEIGHT_LARGE
from config.styles import render_stats_row


def calculate_realized_volatility(price_paths: np.ndarray, time_horizon: float) -> np.ndarray:
    """
    Calculate annualized realized volatility for each simulation path.

    Parameters
    ----------
    price_paths : np.ndarray
        Shape (n_paths, n_steps+1) price paths
    time_horizon : float
        Time horizon in years

    Returns
    -------
    np.ndarray
        Annualized realized volatility per path
    """
    n_steps = price_paths.shape[1] - 1
    dt = time_horizon / n_steps

    # Log returns
    log_returns = np.diff(np.log(price_paths), axis=1)

    # Standard deviation of returns
    return_std = np.std(log_returns, axis=1)

    # Annualize: std * sqrt(252 trading days / steps_per_year)
    annualization_factor = np.sqrt(1.0 / dt)
    realized_vol = return_std * annualization_factor

    return realized_vol


def render_volatility_impact_tab(
    simulation_result,
    pnl_result: Optional[Dict[str, Any]],
    params: Dict[str, Any]
) -> None:
    """
    Render the Volatility Impact educational tab.

    Shows how volatility affects option P&L through multiple visualizations.
    """
    if simulation_result is None:
        st.info("Run a simulation to see volatility impact analysis.")
        return

    st.markdown("### Volatility Impact on P&L")
    st.caption(
        "Understand how volatility drives option P&L. For long gamma positions, "
        "higher realized volatility means higher profits. For short gamma, the opposite."
    )

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "IV vs Realized Vol",
        "Vega P&L Attribution",
        "Fat Tails Explained",
        "Price-Vol Surface"
    ])

    with tab1:
        _render_iv_vs_rv(simulation_result, pnl_result, params)

    with tab2:
        _render_vega_attribution(simulation_result, pnl_result, params)

    with tab3:
        _render_fat_tails(simulation_result, params)

    with tab4:
        _render_price_vol_surface(params)


def _render_iv_vs_rv(simulation_result, pnl_result, params: Dict[str, Any]) -> None:
    """Render IV vs Realized Volatility scatter with P&L coloring."""
    st.markdown("#### Implied vs Realized Volatility")
    st.caption(
        "Each point is one simulation path. Realized volatility (RV) is computed from "
        "the actual price path. When RV > IV, option buyers tend to profit."
    )

    # Get implied volatility (what we priced the option at)
    iv = params.get('volatility', 0.20)

    # Calculate realized volatility for each path
    price_paths = simulation_result.price_paths
    time_horizon = params.get('time_horizon', 1.0)
    realized_vol = calculate_realized_volatility(price_paths, time_horizon)

    # Check if we have P&L data
    has_pnl = pnl_result is not None and 'pnl_values' in pnl_result
    pnl_values = pnl_result['pnl_values'] if has_pnl else np.zeros(len(realized_vol))

    # Summary statistics
    rv_mean = np.mean(realized_vol)
    rv_std = np.std(realized_vol)
    rv_vs_iv = rv_mean - iv

    variant = "green" if abs(rv_vs_iv) < 0.02 else "amber"
    direction = "higher" if rv_vs_iv > 0 else "lower"

    stats_data = [
        ("Implied Vol (IV)", f"{iv*100:.1f}%", "Used for pricing"),
        ("Mean Realized Vol", f"{rv_mean*100:.1f}%", f"{direction} than IV"),
        ("RV Std Dev", f"{rv_std*100:.1f}%", "Volatility of volatility"),
        ("RV - IV Spread", f"{rv_vs_iv*100:+.2f}%", "Positive = buyer wins"),
    ]
    render_stats_row(stats_data, ["blue", "teal", "amber", variant])

    # Create scatter plot
    fig = go.Figure()

    # Sample for performance if needed
    n_display = min(5000, len(realized_vol))
    idx = np.random.choice(len(realized_vol), n_display, replace=False)

    if has_pnl:
        # Color by P&L
        fig.add_trace(go.Scatter(
            x=realized_vol[idx] * 100,
            y=pnl_values[idx],
            mode='markers',
            marker=dict(
                size=4,
                color=pnl_values[idx],
                colorscale='RdYlGn',
                cmin=-np.percentile(np.abs(pnl_values), 95),
                cmax=np.percentile(np.abs(pnl_values), 95),
                colorbar=dict(title="P&L ($)"),
                opacity=0.6
            ),
            hovertemplate=(
                'Realized Vol: %{x:.1f}%<br>'
                'P&L: $%{y:,.0f}<extra></extra>'
            )
        ))

        # Add IV reference line
        fig.add_vline(
            x=iv * 100,
            line_dash="dash",
            line_color="red",
            annotation_text=f"IV = {iv*100:.1f}%",
            annotation_position="top"
        )

        # Add trend line
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            realized_vol[idx], pnl_values[idx]
        )
        x_line = np.array([realized_vol.min(), realized_vol.max()]) * 100
        y_line = slope * x_line / 100 + intercept

        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            line=dict(color='black', dash='dot', width=2),
            name=f'Trend (R²={r_value**2:.2f})'
        ))

        fig.update_layout(
            title="Realized Volatility vs P&L",
            xaxis_title="Realized Volatility (%)",
            yaxis_title="P&L ($)",
            height=CHART_HEIGHT_STANDARD,
            showlegend=True
        )

    else:
        # No P&L - just show RV distribution
        fig.add_trace(go.Histogram(
            x=realized_vol * 100,
            nbinsx=50,
            marker_color='#0d9488',
            opacity=0.7,
            name='Realized Vol'
        ))

        fig.add_vline(
            x=iv * 100,
            line_dash="dash",
            line_color="red",
            annotation_text=f"IV = {iv*100:.1f}%"
        )

        fig.update_layout(
            title="Realized Volatility Distribution",
            xaxis_title="Realized Volatility (%)",
            yaxis_title="Frequency",
            height=CHART_HEIGHT_STANDARD
        )

    st.plotly_chart(fig, use_container_width=True)

    # Educational insight
    st.markdown("""
    <div style="background: #e8f4f8; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <h5 style="color: #1e3a5f; margin-top: 0;">Key Insight: The Volatility Premium</h5>
        <p style="margin-bottom: 0;">
            If realized volatility consistently exceeds implied volatility, <strong>option buyers profit</strong>
            (and market makers / sellers lose). Historically, IV tends to exceed RV on average - this is the
            <strong>volatility risk premium</strong> that option sellers collect. However, when markets crash,
            RV spikes dramatically above IV, causing large losses for short gamma positions.
        </p>
    </div>
    """, unsafe_allow_html=True)


def _render_vega_attribution(simulation_result, pnl_result, params: Dict[str, Any]) -> None:
    """Render Vega P&L attribution for stochastic vol models."""
    st.markdown("#### Vega P&L Attribution")

    # Check for stochastic volatility
    has_stoch_vol = (
        hasattr(simulation_result, 'volatility_paths') and
        simulation_result.volatility_paths is not None
    )

    if not has_stoch_vol:
        st.info(
            "Vega P&L attribution requires a stochastic volatility model (Heston, Bates, or GARCH). "
            "With constant volatility models like GBM, there's no volatility path to track."
        )

        # Show conceptual explanation instead
        st.markdown("""
        <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <h5 style="color: #92400e; margin-top: 0;">Why Vega P&L Matters</h5>
            <p>
                Even though GBM assumes constant volatility, in real markets, implied volatility changes.
                Your position's Vega tells you how much you gain/lose per 1% move in IV:
            </p>
            <ul style="margin-bottom: 0;">
                <li><strong>Long Vega</strong> (long options): Profit when IV increases</li>
                <li><strong>Short Vega</strong> (short options): Profit when IV decreases</li>
                <li>Around earnings/events, IV typically rises then falls ("vol crush")</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return

    st.caption(
        "For stochastic volatility models, we can track how volatility changes along each path "
        "and attribute P&L to Vega exposure."
    )

    # Get volatility paths
    vol_paths = simulation_result.volatility_paths
    time_grid = simulation_result.time_grid
    n_paths = vol_paths.shape[0]

    # Initial and terminal volatility
    initial_vol = vol_paths[:, 0]
    terminal_vol = vol_paths[:, -1]
    vol_change = terminal_vol - initial_vol

    # Get portfolio Vega if we have positions
    position_arrays = params.get('position_arrays', {})
    has_positions = len(position_arrays.get('strikes', [])) > 0

    if has_positions and pnl_result is not None:
        from .greeks_decomposition import calculate_portfolio_greeks

        portfolio_greeks = calculate_portfolio_greeks(
            spot=params.get('spot_price', 100.0),
            strikes=position_arrays['strikes'],
            option_types=position_arrays['option_types'],
            position_types=position_arrays['position_types'],
            quantities=position_arrays['quantities'],
            time_to_expiry=params.get('time_horizon', 1.0),
            risk_free_rate=params.get('risk_free_rate', 0.05),
            volatility=params.get('volatility', 0.20)
        )

        portfolio_vega = portfolio_greeks['vega']

        # Vega P&L = Vega * Δσ (in percentage points)
        vega_pnl = portfolio_vega * vol_change * 100

        # Create 2x2 subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Volatility Change Distribution",
                "Vega P&L vs Vol Change",
                "Volatility Paths Over Time",
                "Cumulative Vega P&L Impact"
            ),
            vertical_spacing=0.12
        )

        # 1. Vol change distribution
        fig.add_trace(
            go.Histogram(
                x=vol_change * 100,
                nbinsx=50,
                marker_color='#9333ea',
                opacity=0.7,
                name='Vol Change'
            ),
            row=1, col=1
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=1)

        # 2. Vega P&L vs Vol Change scatter
        sample_size = min(2000, n_paths)
        idx = np.random.choice(n_paths, sample_size, replace=False)

        fig.add_trace(
            go.Scatter(
                x=vol_change[idx] * 100,
                y=vega_pnl[idx],
                mode='markers',
                marker=dict(
                    size=4,
                    color=pnl_result['pnl_values'][idx],
                    colorscale='RdYlGn',
                    opacity=0.5
                ),
                hovertemplate=(
                    'Vol Change: %{x:.2f}%<br>'
                    'Vega P&L: $%{y:,.0f}<extra></extra>'
                )
            ),
            row=1, col=2
        )

        # Add theoretical line (slope = portfolio vega)
        x_range = np.array([vol_change.min(), vol_change.max()]) * 100
        y_range = portfolio_vega * x_range
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_range,
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Theoretical',
                showlegend=False
            ),
            row=1, col=2
        )

        # 3. Sample volatility paths
        max_display = min(50, n_paths)
        for i in range(max_display):
            fig.add_trace(
                go.Scatter(
                    x=time_grid,
                    y=vol_paths[i] * 100,
                    mode='lines',
                    line=dict(color='rgba(147, 51, 234, 0.2)', width=0.8),
                    hoverinfo='skip',
                    showlegend=False
                ),
                row=2, col=1
            )

        # Mean path
        mean_vol = np.mean(vol_paths, axis=0) * 100
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=mean_vol,
                mode='lines',
                line=dict(color='#9333ea', width=2),
                name='Mean Vol'
            ),
            row=2, col=1
        )

        # Initial vol reference
        fig.add_hline(
            y=vol_paths[0, 0] * 100,
            line_dash="dot",
            line_color="red",
            row=2, col=1
        )

        # 4. Cumulative Vega contribution vs Total P&L
        sorted_idx = np.argsort(pnl_result['pnl_values'])
        percentiles = np.arange(n_paths) / n_paths * 100

        fig.add_trace(
            go.Scatter(
                x=percentiles,
                y=pnl_result['pnl_values'][sorted_idx],
                mode='lines',
                line=dict(color='#1e3a5f', width=2),
                name='Total P&L'
            ),
            row=2, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=percentiles,
                y=vega_pnl[sorted_idx],
                mode='lines',
                line=dict(color='#9333ea', width=2),
                name='Vega P&L'
            ),
            row=2, col=2
        )

        fig.update_xaxes(title_text="Vol Change (%)", row=1, col=1)
        fig.update_xaxes(title_text="Vol Change (%)", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="Percentile", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Vega P&L ($)", row=1, col=2)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=2, col=2)

        fig.update_layout(height=CHART_HEIGHT_LARGE, showlegend=True)

        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        vega_contribution = np.mean(vega_pnl)
        total_pnl = np.mean(pnl_result['pnl_values'])
        pct_from_vega = vega_contribution / total_pnl * 100 if total_pnl != 0 else 0

        stats_data = [
            ("Portfolio Vega", f"${portfolio_vega:+,.2f}/1%", "Per 1% vol move"),
            ("Mean Vol Change", f"{np.mean(vol_change)*100:+.2f}%", "Terminal - Initial"),
            ("Vega P&L Contribution", f"${vega_contribution:+,.0f}", "From vol changes"),
            ("% of Total P&L", f"{pct_from_vega:+.1f}%", "Attributed to Vega"),
        ]
        vega_sign = "green" if portfolio_vega > 0 else "red"
        render_stats_row(stats_data, [vega_sign, "purple", "purple", "slate"])

    else:
        # Just show volatility dynamics
        st.info("Define a strategy to see Vega P&L attribution.")

        fig = go.Figure()
        max_display = min(100, n_paths)
        for i in range(max_display):
            fig.add_trace(go.Scatter(
                x=time_grid,
                y=vol_paths[i] * 100,
                mode='lines',
                line=dict(color='rgba(147, 51, 234, 0.2)', width=0.8),
                showlegend=False
            ))

        mean_vol = np.mean(vol_paths, axis=0) * 100
        fig.add_trace(go.Scatter(
            x=time_grid,
            y=mean_vol,
            mode='lines',
            line=dict(color='#9333ea', width=2.5),
            name='Mean Volatility'
        ))

        fig.update_layout(
            title="Stochastic Volatility Paths",
            xaxis_title="Time",
            yaxis_title="Volatility (%)",
            height=CHART_HEIGHT_STANDARD
        )

        st.plotly_chart(fig, use_container_width=True)


def _render_fat_tails(simulation_result, params: Dict[str, Any]) -> None:
    """Explain fat tails in stochastic vol models vs GBM."""
    st.markdown("#### Fat Tails: Why Stochastic Vol Matters")
    st.caption(
        "Stochastic volatility creates fatter tails than GBM. When volatility is high, "
        "prices move more, and high volatility tends to persist."
    )

    terminal_prices = simulation_result.terminal_prices
    spot = params.get('spot_price', 100.0)
    volatility = params.get('volatility', 0.20)
    time_horizon = params.get('time_horizon', 1.0)
    model = params.get('price_model', 'gbm')

    # Calculate returns
    returns = (terminal_prices - spot) / spot * 100

    # Generate theoretical GBM distribution for comparison
    mu = params.get('expected_return', 0.05)
    gbm_mean = spot * np.exp(mu * time_horizon)
    gbm_std = spot * np.sqrt(np.exp(2 * mu * time_horizon) * (np.exp(volatility**2 * time_horizon) - 1))

    # Create comparison figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Terminal Price Distribution",
            "Return Distribution (Log Scale)"
        )
    )

    # 1. Price distribution
    fig.add_trace(
        go.Histogram(
            x=terminal_prices,
            nbinsx=80,
            name=f'{model.upper()} Simulation',
            marker_color='#0d9488',
            opacity=0.7,
            histnorm='probability density'
        ),
        row=1, col=1
    )

    # Add lognormal overlay
    x_prices = np.linspace(terminal_prices.min(), terminal_prices.max(), 200)
    # Lognormal PDF parameters
    log_mean = np.log(spot) + (mu - 0.5 * volatility**2) * time_horizon
    log_std = volatility * np.sqrt(time_horizon)
    lognormal_pdf = stats.lognorm.pdf(x_prices, log_std, scale=np.exp(log_mean))

    fig.add_trace(
        go.Scatter(
            x=x_prices,
            y=lognormal_pdf,
            mode='lines',
            name='GBM Theory',
            line=dict(color='red', dash='dash', width=2)
        ),
        row=1, col=1
    )

    # 2. Return distribution with focus on tails
    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=100,
            name='Simulated Returns',
            marker_color='#0d9488',
            opacity=0.7,
            histnorm='probability density'
        ),
        row=1, col=2
    )

    # Normal distribution overlay
    x_returns = np.linspace(returns.min(), returns.max(), 200)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    normal_pdf = stats.norm.pdf(x_returns, mean_ret, std_ret)

    fig.add_trace(
        go.Scatter(
            x=x_returns,
            y=normal_pdf,
            mode='lines',
            name='Normal Fit',
            line=dict(color='red', dash='dash', width=2)
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Terminal Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="Probability Density", row=1, col=1)
    fig.update_yaxes(title_text="Probability Density", type="log", row=1, col=2)

    fig.update_layout(height=CHART_HEIGHT_STANDARD, showlegend=True)

    st.plotly_chart(fig, use_container_width=True)

    # Calculate tail statistics
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)  # Excess kurtosis (normal = 0)

    # VaR comparison
    var_95_sim = np.percentile(returns, 5)
    var_95_normal = stats.norm.ppf(0.05, mean_ret, std_ret)

    tail_stats = [
        ("Skewness", f"{skewness:.3f}", "0 for symmetric"),
        ("Excess Kurtosis", f"{kurtosis:.3f}", "0 for normal, >0 = fat tails"),
        ("5% VaR (Simulated)", f"{var_95_sim:.1f}%", "Worst 5% return"),
        ("5% VaR (Normal)", f"{var_95_normal:.1f}%", "Would be if normal"),
    ]

    kurt_variant = "red" if kurtosis > 1 else "amber" if kurtosis > 0 else "green"
    render_stats_row(tail_stats, ["slate", kurt_variant, "red", "blue"])

    # Educational insight
    if model in ['heston', 'bates', 'garch', 'ngarch', 'gjr_garch']:
        st.markdown("""
        <div style="background: #fef2f2; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <h5 style="color: #991b1b; margin-top: 0;">Key Insight: Volatility Clustering Creates Fat Tails</h5>
            <p style="margin-bottom: 0;">
                In stochastic volatility models, extreme returns are more likely than a normal distribution predicts.
                <strong>Why?</strong> When volatility spikes, it tends to stay high (clustering), creating extended
                periods of large moves. This means:
            </p>
            <ul style="margin-bottom: 0;">
                <li>Deep OTM options are worth more than Black-Scholes suggests (volatility smile)</li>
                <li>VaR based on normal distributions underestimates true tail risk</li>
                <li>Crash protection (puts) should be valued higher</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #e8f4f8; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <h5 style="color: #1e3a5f; margin-top: 0;">Note: GBM Assumes Lognormal Returns</h5>
            <p style="margin-bottom: 0;">
                The GBM model produces approximately lognormal terminal prices. In reality, financial
                returns exhibit <strong>fat tails</strong> and <strong>volatility clustering</strong>.
                Try Heston, Bates, or GARCH models to see more realistic tail behavior.
            </p>
        </div>
        """, unsafe_allow_html=True)


def _render_price_vol_surface(params: Dict[str, Any]) -> None:
    """Render 3D surface showing how option price varies with spot and vol."""
    st.markdown("#### Option Price Surface: Spot vs Volatility")
    st.caption(
        "See how option value changes across different spot prices and volatility levels. "
        "This visualizes Delta (slope in spot direction) and Vega (slope in vol direction)."
    )

    # Get base parameters
    spot = params.get('spot_price', 100.0)
    strike = spot  # ATM for illustration
    time_to_expiry = params.get('time_horizon', 1.0)
    risk_free_rate = params.get('risk_free_rate', 0.05)
    base_vol = params.get('volatility', 0.20)

    # Allow user to select option type
    col1, col2 = st.columns(2)
    with col1:
        option_type = st.selectbox(
            "Option Type",
            ["Call", "Put"],
            key="vol_surface_option_type"
        )
    with col2:
        strike_pct = st.slider(
            "Strike (% of Spot)",
            min_value=80,
            max_value=120,
            value=100,
            step=5,
            key="vol_surface_strike"
        )

    strike = spot * strike_pct / 100
    is_call = option_type == "Call"

    # Create grid
    spot_range = np.linspace(spot * 0.7, spot * 1.3, 30)
    vol_range = np.linspace(0.05, 0.50, 25)

    prices = np.zeros((len(vol_range), len(spot_range)))
    deltas = np.zeros_like(prices)
    vegas = np.zeros_like(prices)

    for i, vol in enumerate(vol_range):
        for j, s in enumerate(spot_range):
            greeks = bs_all_greeks(s, strike, time_to_expiry, risk_free_rate, 0.0, vol, is_call)
            prices[i, j] = greeks[0]  # price
            deltas[i, j] = greeks[1]  # delta
            vegas[i, j] = greeks[3]   # vega

    # Create 3D surface
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=spot_range,
        y=vol_range * 100,
        z=prices,
        colorscale='Viridis',
        name='Option Price',
        hovertemplate=(
            'Spot: $%{x:.1f}<br>'
            'Vol: %{y:.1f}%<br>'
            'Price: $%{z:.2f}<extra></extra>'
        )
    ))

    # Add current point
    current_price = bs_all_greeks(spot, strike, time_to_expiry, risk_free_rate, 0.0, base_vol, is_call)[0]
    fig.add_trace(go.Scatter3d(
        x=[spot],
        y=[base_vol * 100],
        z=[current_price],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='Current'
    ))

    fig.update_layout(
        title=f"{option_type} Option Price Surface (Strike=${strike:.0f})",
        scene=dict(
            xaxis_title="Spot Price ($)",
            yaxis_title="Volatility (%)",
            zaxis_title="Option Price ($)"
        ),
        height=CHART_HEIGHT_LARGE
    )

    st.plotly_chart(fig, use_container_width=True)

    # Greeks at current point
    current_greeks = bs_all_greeks(spot, strike, time_to_expiry, risk_free_rate, 0.0, base_vol, is_call)
    greeks_stats = [
        ("Current Price", f"${current_greeks[0]:.2f}", ""),
        ("Delta", f"{current_greeks[1]:.4f}", "Slope in spot direction"),
        ("Vega", f"${current_greeks[3]:.2f}", "Slope in vol direction"),
        ("Gamma", f"{current_greeks[2]:.6f}", "Curvature"),
    ]
    render_stats_row(greeks_stats, ["blue", "teal", "purple", "amber"])

    # Educational note
    st.markdown("""
    <div style="background: #f0fdf4; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <h5 style="color: #166534; margin-top: 0;">Reading the Surface</h5>
        <ul style="margin-bottom: 0;">
            <li><strong>Delta</strong>: The slope when moving along the spot axis (left-right)</li>
            <li><strong>Vega</strong>: The slope when moving along the volatility axis (front-back)</li>
            <li><strong>Gamma</strong>: How the delta slope changes (curvature)</li>
            <li>Notice how ATM options have the highest Vega (steepest vol sensitivity)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
