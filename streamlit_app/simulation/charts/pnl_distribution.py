"""
P&L Distribution Charts for Monte Carlo Option P&L Simulation.

Provides histogram, CDF, and risk metrics visualizations for
P&L distributions from Monte Carlo simulation.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, Optional
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from config.constants import (
    CHART_HEIGHT_STANDARD,
    CHART_HEIGHT_LARGE,
    PNL_COLORS,
    RISK_METRICS,
    PNL_HISTOGRAM_BINS,
    PNL_KDE_POINTS
)
from config.styles import render_stats_row
from backend.portfolio.pnl import RiskMetrics


def render_pnl_distribution_tab(
    pnl_values: np.ndarray,
    risk_metrics: RiskMetrics,
    params: Dict[str, Any]
) -> None:
    """
    Render the P&L distribution visualization tab.

    Parameters
    ----------
    pnl_values : np.ndarray
        Array of P&L values from simulation
    risk_metrics : RiskMetrics
        Pre-computed risk metrics
    params : dict
        Simulation parameters for display
    """
    if pnl_values is None or len(pnl_values) == 0:
        st.info("Run a P&L simulation to see distribution analysis.")
        return

    # Section header with native help
    st.markdown("### P&L Distribution Analysis")
    st.caption(
        "Probability distribution of strategy P&L outcomes. VaR indicates worst expected loss at confidence level."
    )

    # Summary metrics row
    _render_summary_metrics(risk_metrics)

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Histogram", "CDF & Percentiles", "P&L Drivers"])

    with tab1:
        _render_histogram(pnl_values, risk_metrics)

    with tab2:
        _render_cdf(pnl_values, risk_metrics)

    with tab3:
        _render_pnl_drivers(pnl_values, risk_metrics, params)


def _render_summary_metrics(metrics: RiskMetrics) -> None:
    """Render summary metrics cards using styled components."""
    mean_variant = "green" if metrics.mean_pnl > 0 else "red"
    prob_variant = "green" if metrics.prob_profit > 0.5 else "red"

    summary_stats = [
        ("Mean P&L", f"${metrics.mean_pnl:,.2f}", "Profit" if metrics.mean_pnl > 0 else "Loss"),
        ("VaR 95%", f"${metrics.var_95:,.2f}", "5% worst case"),
        ("P(Profit)", f"{metrics.prob_profit:.1%}", "Favorable" if metrics.prob_profit > 0.5 else "Unfavorable"),
        ("Std Dev", f"${metrics.std_pnl:,.2f}", "P&L volatility"),
    ]
    render_stats_row(summary_stats, [mean_variant, "amber", prob_variant, "blue"])


def _render_histogram(pnl_values: np.ndarray, metrics: RiskMetrics) -> None:
    """Render P&L histogram with profit/loss coloring."""
    fig = go.Figure()

    # Create histogram data
    hist_values, bin_edges = np.histogram(pnl_values, bins=PNL_HISTOGRAM_BINS)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Separate profit and loss bins
    profit_mask = bin_centers > 0
    loss_mask = bin_centers <= 0

    # Loss bars (red)
    if np.any(loss_mask):
        fig.add_trace(go.Bar(
            x=bin_centers[loss_mask],
            y=hist_values[loss_mask],
            name='Loss',
            marker_color=PNL_COLORS['loss'],
            opacity=0.7,
            showlegend=True
        ))

    # Profit bars (green)
    if np.any(profit_mask):
        fig.add_trace(go.Bar(
            x=bin_centers[profit_mask],
            y=hist_values[profit_mask],
            name='Profit',
            marker_color=PNL_COLORS['profit'],
            opacity=0.7,
            showlegend=True
        ))

    # Add KDE overlay
    try:
        kde = stats.gaussian_kde(pnl_values)
        x_kde = np.linspace(pnl_values.min(), pnl_values.max(), PNL_KDE_POINTS)
        y_kde = kde(x_kde)
        # Scale KDE to match histogram
        y_kde_scaled = y_kde * len(pnl_values) * (bin_edges[1] - bin_edges[0])

        fig.add_trace(go.Scatter(
            x=x_kde,
            y=y_kde_scaled,
            mode='lines',
            name='KDE',
            line=dict(color='#1e293b', width=2)
        ))
    except Exception:
        pass  # KDE may fail for certain distributions

    # Add VaR lines
    fig.add_vline(
        x=metrics.var_95,
        line_dash="dash",
        line_color=PNL_COLORS['var_line'],
        annotation_text=f"VaR 95%: ${metrics.var_95:,.0f}",
        annotation_position="top left"
    )

    fig.add_vline(
        x=metrics.var_99,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text=f"VaR 99%: ${metrics.var_99:,.0f}",
        annotation_position="top left"
    )

    # Add zero line
    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color="#475569",
        line_width=1
    )

    # Add mean line
    fig.add_vline(
        x=metrics.mean_pnl,
        line_dash="dash",
        line_color="#3b82f6",
        annotation_text=f"Mean: ${metrics.mean_pnl:,.0f}",
        annotation_position="top right"
    )

    fig.update_layout(
        title=dict(
            text=f"P&L Distribution ({len(pnl_values):,} scenarios)",
            font=dict(size=16)
        ),
        xaxis_title="P&L ($)",
        yaxis_title="Frequency",
        height=CHART_HEIGHT_STANDARD,
        barmode='overlay',
        hovermode='x unified',
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="right", x=0.99,
            bgcolor='rgba(255,255,255,0.9)'
        ),
        margin=dict(l=60, r=40, t=60, b=60)
    )

    st.plotly_chart(fig, width="stretch")

    # Additional stats below chart with styled cards
    skew_variant = "green" if metrics.skewness > 0 else "red"
    kurt_variant = "amber" if metrics.kurtosis > 3 else "slate"

    hist_stats = [
        ("Skewness", f"{metrics.skewness:+.3f}", "Right-skewed" if metrics.skewness > 0 else "Left-skewed"),
        ("Kurtosis", f"{metrics.kurtosis:.3f}", "Fat tails" if metrics.kurtosis > 3 else "Normal tails"),
        ("Max Profit", f"${metrics.max_profit:,.0f}", "Best scenario"),
        ("Max Loss", f"${metrics.max_loss:,.0f}", "Worst scenario"),
    ]
    render_stats_row(hist_stats, [skew_variant, kurt_variant, "green", "red"])


def _render_cdf(pnl_values: np.ndarray, metrics: RiskMetrics) -> None:
    """Render cumulative distribution function with percentile annotations."""
    # Sort P&L values for CDF
    sorted_pnl = np.sort(pnl_values)
    n = len(sorted_pnl)
    cdf = np.arange(1, n + 1) / n

    fig = go.Figure()

    # CDF line
    fig.add_trace(go.Scatter(
        x=sorted_pnl,
        y=cdf,
        mode='lines',
        name='CDF',
        line=dict(color='#3b82f6', width=2),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))

    # Add percentile markers - key percentiles only to avoid clutter
    key_percentiles = [5, 25, 50, 75, 95]
    text_positions = ['bottom right', 'bottom left', 'top center', 'top right', 'top left']

    for p, text_pos in zip(key_percentiles, text_positions):
        idx = int((p / 100) * (n - 1))
        pnl_at_p = sorted_pnl[idx]

        # Determine color based on percentile
        if p <= 10:
            color = '#ef4444'  # Red for tail risk
        elif p >= 90:
            color = '#10b981'  # Green for upside
        else:
            color = '#6b7280'  # Gray for middle

        fig.add_trace(go.Scatter(
            x=[pnl_at_p],
            y=[p / 100],
            mode='markers+text',
            marker=dict(size=10, color=color),
            text=[f"P{p}: ${pnl_at_p:,.0f}"],
            textposition=text_pos,
            showlegend=False,
            textfont=dict(size=9)
        ))

    # Add subtle horizontal reference lines (no annotations - markers have labels)
    fig.add_hline(y=0.05, line_dash="dot", line_color="#f59e0b", opacity=0.4)
    fig.add_hline(y=0.50, line_dash="dot", line_color="#6b7280", opacity=0.4)
    fig.add_hline(y=0.95, line_dash="dot", line_color="#10b981", opacity=0.4)

    # Add breakeven line
    fig.add_vline(x=0, line_dash="solid", line_color="#8b5cf6", line_width=2)

    # Find probability of breakeven
    prob_breakeven = np.mean(pnl_values > 0)
    fig.add_annotation(
        x=0,
        y=prob_breakeven,
        text=f"P(Profit) = {prob_breakeven:.1%}",
        showarrow=True,
        arrowhead=2,
        ax=50,
        ay=-30
    )

    fig.update_layout(
        title="Cumulative Distribution Function",
        xaxis_title="P&L ($)",
        yaxis_title="Cumulative Probability",
        height=CHART_HEIGHT_STANDARD,
        yaxis=dict(
            tickformat='.0%',
            range=[0, 1],
            dtick=0.25,  # Tick marks at 0%, 25%, 50%, 75%, 100%
            gridcolor='rgba(0,0,0,0.05)'
        ),
        xaxis=dict(gridcolor='rgba(0,0,0,0.05)'),
        hovermode='x unified',
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white'
    )

    st.plotly_chart(fig, width="stretch")

    # Percentile table with styled design using dataframe
    st.markdown("#### Key Percentiles")

    # Create percentile data
    import pandas as pd

    percentile_data = []
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        idx = int((p / 100) * (n - 1))
        pnl_val = sorted_pnl[idx]

        if p <= 10:
            category = "🔴 Tail Risk"
        elif p >= 90:
            category = "🟢 Upside"
        else:
            category = "⚪ Middle"

        percentile_data.append({
            "Percentile": f"P{p}",
            "P&L Value": f"${pnl_val:,.2f}",
            "Category": category
        })

    df_percentiles = pd.DataFrame(percentile_data)

    # Style the dataframe
    st.dataframe(
        df_percentiles,
        width="stretch",
        hide_index=True,
        column_config={
            "Percentile": st.column_config.TextColumn("Percentile", width="small"),
            "P&L Value": st.column_config.TextColumn("P&L Value", width="medium"),
            "Category": st.column_config.TextColumn("Category", width="medium")
        }
    )


def _render_pnl_drivers(
    pnl_values: np.ndarray,
    metrics: RiskMetrics,
    params: Dict[str, Any]
) -> None:
    """
    Render educational content explaining P&L distribution drivers.

    Explains why different strategies create different distribution shapes.
    """
    st.markdown("#### Understanding Your P&L Distribution Shape")

    # Detect distribution characteristics
    is_bimodal = _detect_bimodality(pnl_values)
    is_bounded = _detect_bounded_distribution(pnl_values, metrics)
    has_fat_tails = metrics.kurtosis > 3.5

    # Get model info
    model_name = params.get('model_name', 'GBM')
    has_stoch_vol = model_name.lower() in ['heston', 'bates', 'garch', 'ngarch', 'gjr-garch']

    # Strategy type detection from position arrays
    position_arrays = params.get('position_arrays', {})
    strategy_type = _detect_strategy_type(position_arrays)

    # Display detected characteristics
    col1, col2, col3 = st.columns(3)
    with col1:
        bimodal_icon = "✅" if is_bimodal else "❌"
        st.markdown(f"**Bimodal:** {bimodal_icon}")
    with col2:
        bounded_icon = "✅" if is_bounded else "❌"
        st.markdown(f"**Bounded:** {bounded_icon}")
    with col3:
        fat_icon = "✅" if has_fat_tails else "❌"
        st.markdown(f"**Fat Tails:** {fat_icon}")

    st.markdown("---")

    # Explanation based on detected characteristics
    if is_bimodal:
        st.markdown("""
        <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h5 style="color: #92400e; margin-top: 0;">📊 Bimodal Distribution Detected</h5>
            <p style="margin-bottom: 0.5rem;">
                Your P&L distribution has two peaks. This is characteristic of <strong>volatility strategies</strong>
                like straddles and strangles.
            </p>
            <p style="margin-bottom: 0;">
                <strong>Why?</strong> These strategies profit from large moves in either direction, but lose money
                when the underlying stays flat. The two modes represent:
            </p>
            <ul style="margin-bottom: 0;">
                <li><strong>Left peak:</strong> Scenarios where the stock doesn't move much (theta decay dominates)</li>
                <li><strong>Right peak:</strong> Scenarios where the stock moves significantly (gamma gains dominate)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if is_bounded:
        st.markdown("""
        <div style="background: #dbeafe; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h5 style="color: #1e40af; margin-top: 0;">📏 Bounded Distribution Detected</h5>
            <p style="margin-bottom: 0.5rem;">
                Your P&L has defined maximum profit and/or loss limits. This is characteristic of
                <strong>spread strategies</strong> like bull/bear spreads, iron condors, and butterflies.
            </p>
            <p style="margin-bottom: 0;">
                <strong>Why?</strong> Spreads combine long and short options at different strikes:
            </p>
            <ul style="margin-bottom: 0;">
                <li>The short options cap your gains but also reduce cost</li>
                <li>The long options limit your losses</li>
                <li>Result: A "box" of possible outcomes with defined edges</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if has_fat_tails and has_stoch_vol:
        st.markdown(f"""
        <div style="background: #fce7f3; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h5 style="color: #9d174d; margin-top: 0;">📈 Fat Tails from Stochastic Volatility ({model_name})</h5>
            <p style="margin-bottom: 0.5rem;">
                Your distribution has fatter tails than a normal distribution (kurtosis = {metrics.kurtosis:.2f} > 3).
                Using a stochastic volatility model explains why:
            </p>
            <ul style="margin-bottom: 0;">
                <li><strong>Volatility clustering:</strong> High vol tends to stay high, low vol stays low</li>
                <li><strong>Correlation effect:</strong> In crashes, both price drops AND vol spikes (double whammy for long positions)</li>
                <li><strong>Result:</strong> More extreme outcomes than GBM would predict</li>
            </ul>
            <p style="margin-top: 0.5rem; margin-bottom: 0;">
                <em>This is why options are priced with volatility "smile" - the market knows tails are fatter!</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif has_fat_tails:
        st.markdown(f"""
        <div style="background: #fce7f3; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h5 style="color: #9d174d; margin-top: 0;">📈 Fat Tails Detected</h5>
            <p style="margin-bottom: 0;">
                Your distribution has higher kurtosis ({metrics.kurtosis:.2f}) than a normal distribution (3.0).
                This means extreme P&L outcomes are more likely than a bell curve would suggest.
                Consider using a stochastic volatility model (Heston, GARCH) for more realistic tail risk.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # General educational content
    st.markdown("#### Key P&L Drivers by Strategy Type")

    strategy_drivers = {
        "Long Call/Put": {
            "main_driver": "Delta (directional exposure)",
            "secondary": "Gamma (acceleration), Vega (vol sensitivity)",
            "risk": "Theta decay works against you every day",
            "distribution": "Asymmetric: limited loss (premium), unlimited gain potential"
        },
        "Short Call/Put": {
            "main_driver": "Theta (time decay collection)",
            "secondary": "Negative Gamma (risk of large moves)",
            "risk": "Unlimited loss potential if wrong",
            "distribution": "Asymmetric: limited gain (premium), large loss potential"
        },
        "Straddle/Strangle": {
            "main_driver": "Gamma (profits from large moves)",
            "secondary": "Vega (benefits from vol increase)",
            "risk": "Theta decay if underlying stays flat",
            "distribution": "Bimodal: peaks at 'no move' and 'big move' scenarios"
        },
        "Vertical Spread": {
            "main_driver": "Delta (directional view with reduced cost)",
            "secondary": "Limited Gamma and Vega exposure",
            "risk": "Both profit and loss are capped",
            "distribution": "Bounded: max profit at one strike, max loss at other"
        },
        "Iron Condor": {
            "main_driver": "Theta (time decay on 4 options)",
            "secondary": "Negative Vega (hurt by vol increase)",
            "risk": "Loss if underlying moves beyond wings",
            "distribution": "Bounded with central plateau of max profit"
        }
    }

    # Display as expandable sections
    for strategy, drivers in strategy_drivers.items():
        with st.expander(f"**{strategy}**"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Main Driver:** {drivers['main_driver']}")
                st.markdown(f"**Secondary:** {drivers['secondary']}")
            with col2:
                st.markdown(f"**Key Risk:** {drivers['risk']}")
                st.markdown(f"**Distribution Shape:** {drivers['distribution']}")

    # Practical insight
    st.markdown("""
    <div style="background: #ecfdf5; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <h5 style="color: #065f46; margin-top: 0;">💡 Practical Insight</h5>
        <p style="margin-bottom: 0;">
            The shape of your P&L distribution tells you what you're betting on:
        </p>
        <ul style="margin-bottom: 0;">
            <li><strong>Wide symmetric distribution:</strong> You're taking directional risk</li>
            <li><strong>Bimodal distribution:</strong> You're betting on movement magnitude</li>
            <li><strong>Narrow bounded distribution:</strong> You're selling insurance (premium collection)</li>
            <li><strong>Skewed distribution:</strong> Asymmetric risk/reward trade-off</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def _detect_bimodality(pnl_values: np.ndarray) -> bool:
    """
    Detect if distribution is bimodal using dip test heuristic.

    Uses histogram peak counting as a simple proxy.
    """
    try:
        hist, bin_edges = np.histogram(pnl_values, bins=30)
        # Smooth histogram
        smoothed = gaussian_filter1d(hist.astype(float), sigma=2)

        # Count local maxima
        peaks = 0
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                # Only count significant peaks (>10% of max)
                if smoothed[i] > 0.1 * smoothed.max():
                    peaks += 1

        return peaks >= 2
    except Exception:
        return False


def _detect_bounded_distribution(pnl_values: np.ndarray, metrics: RiskMetrics) -> bool:
    """
    Detect if distribution has clearly bounded range.

    Looks for sharp cutoffs in the distribution tails.
    """
    # Check if max loss and max profit are "close" to VaR values
    # indicating hard boundaries rather than fat tails

    range_pnl = metrics.max_profit - metrics.max_loss
    if range_pnl == 0:
        return False

    # Calculate how much of the range is outside the 5-95% interval
    tail_range = (metrics.max_profit - np.percentile(pnl_values, 95)) + \
                 (np.percentile(pnl_values, 5) - metrics.max_loss)

    # If tails are very short relative to total range, distribution is bounded
    tail_ratio = tail_range / range_pnl

    return tail_ratio < 0.15  # Tails are less than 15% of total range


def _detect_strategy_type(position_arrays: Dict[str, Any]) -> str:
    """Attempt to detect strategy type from position arrays."""
    if not position_arrays or len(position_arrays.get('strikes', [])) == 0:
        return "Unknown"

    strikes = position_arrays.get('strikes', [])
    option_types = position_arrays.get('option_types', [])
    position_types = position_arrays.get('position_types', [])

    n_legs = len(strikes)
    n_calls = sum(1 for t in option_types if t == 1)
    n_puts = sum(1 for t in option_types if t == -1)
    n_long = sum(1 for p in position_types if p == 1)
    n_short = sum(1 for p in position_types if p == -1)

    unique_strikes = len(set(strikes))

    # Simple heuristics
    if n_legs == 1:
        return "Single Option"
    elif n_legs == 2 and n_calls == 1 and n_puts == 1 and unique_strikes == 1:
        return "Straddle"
    elif n_legs == 2 and n_calls == 1 and n_puts == 1 and unique_strikes == 2:
        return "Strangle"
    elif n_legs == 2 and (n_calls == 2 or n_puts == 2):
        return "Vertical Spread"
    elif n_legs == 4:
        return "Multi-Leg (Condor/Butterfly)"
    else:
        return "Custom Strategy"


def render_risk_metrics_tab(metrics: RiskMetrics, params: Dict[str, Any]) -> None:
    """
    Render comprehensive risk metrics dashboard.

    Parameters
    ----------
    metrics : RiskMetrics
        Pre-computed risk metrics from P&L simulation
    params : dict
        Simulation parameters for context
    """
    st.markdown("### Risk Metrics Dashboard")

    # Primary metrics row with styled cards
    st.markdown("#### Performance & Risk")
    mean_variant = "green" if metrics.mean_pnl >= 0 else "red"
    prob_variant = "green" if metrics.prob_profit > 0.5 else "red"

    primary_stats = [
        ("Expected P&L", f"${metrics.mean_pnl:,.2f}", "Mean outcome"),
        ("VaR 95%", f"${metrics.var_95:,.2f}", "5% worst case"),
        ("CVaR 95%", f"${metrics.cvar_95:,.2f}", "Expected shortfall"),
        ("P(Profit)", f"{metrics.prob_profit:.1%}", "Favorable" if metrics.prob_profit > 0.5 else "Unfavorable"),
    ]
    render_stats_row(primary_stats, [mean_variant, "amber", "red", prob_variant])

    st.markdown("")  # Spacer

    # Secondary metrics with styled cards
    st.markdown("#### Distribution Characteristics")
    skew_variant = "green" if metrics.skewness > 0 else "red"
    kurt_variant = "amber" if metrics.kurtosis > 3 else "slate"

    # Risk-adjusted return
    sharpe_like = metrics.mean_pnl / metrics.std_pnl if metrics.std_pnl > 0 else 0
    sharpe_variant = "green" if sharpe_like > 0 else "red"

    dist_stats = [
        ("Standard Deviation", f"${metrics.std_pnl:,.2f}", "P&L volatility"),
        ("Return/Risk Ratio", f"{sharpe_like:.3f}" if metrics.std_pnl > 0 else "N/A", "Risk-adjusted"),
        ("Skewness", f"{metrics.skewness:.3f}", "Right-skewed" if metrics.skewness > 0 else "Left-skewed"),
        ("Excess Kurtosis", f"{metrics.kurtosis:.3f}", "Fat tails" if metrics.kurtosis > 3 else "Normal tails"),
    ]
    render_stats_row(dist_stats, ["blue", sharpe_variant, skew_variant, kurt_variant])

    st.divider()

    # Extreme outcomes with styled cards
    st.markdown("#### Extreme Outcomes")
    extreme_stats = [
        ("Maximum Profit", f"${metrics.max_profit:,.2f}", "Best scenario"),
        ("Maximum Loss", f"${metrics.max_loss:,.2f}", "Worst scenario"),
        ("VaR 99%", f"${metrics.var_99:,.2f}", "1% worst case"),
        ("CVaR 99%", f"${metrics.cvar_99:,.2f}", "Extreme shortfall"),
    ]
    render_stats_row(extreme_stats, ["green", "red", "amber", "red"])

    # Risk interpretation
    st.divider()
    st.markdown("#### Interpretation")

    interpretation = []

    if metrics.prob_profit > 0.6:
        interpretation.append("**High probability of profit** (>60%)")
    elif metrics.prob_profit < 0.4:
        interpretation.append("**Low probability of profit** (<40%)")

    if metrics.skewness < -0.5:
        interpretation.append("**Negatively skewed** - larger potential losses than gains")
    elif metrics.skewness > 0.5:
        interpretation.append("**Positively skewed** - larger potential gains than losses")

    if metrics.kurtosis > 3:
        interpretation.append("**Fat tails** - extreme outcomes more likely than normal distribution")

    if metrics.cvar_95 < metrics.var_95:
        interpretation.append(f"**Tail risk**: Expected loss in worst 5% of cases is ${abs(metrics.cvar_95):,.0f}")

    if interpretation:
        for item in interpretation:
            st.markdown(f"- {item}")
    else:
        st.markdown("- Distribution appears relatively symmetric with normal tail behavior")
