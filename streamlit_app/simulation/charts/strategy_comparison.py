"""
Strategy Comparison Module.

Pedagogical module for comparing two option strategies side-by-side:
1. Overlaid P&L distributions
2. Comparative risk metrics table
3. Scenario-by-scenario winner analysis

Helps understand trade-offs between different option strategies.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from backend.portfolio.pnl import compute_risk_metrics, RiskMetrics
from config.constants import CHART_HEIGHT_STANDARD, CHART_HEIGHT_LARGE
from config.styles import render_stats_row


@dataclass
class StrategyComparison:
    """Container for strategy comparison results."""
    name_a: str
    name_b: str
    pnl_a: np.ndarray
    pnl_b: np.ndarray
    metrics_a: RiskMetrics
    metrics_b: RiskMetrics
    winner_by_scenario: np.ndarray  # 1 if A wins, -1 if B wins, 0 if tie


def compare_strategies(
    pnl_a: np.ndarray,
    pnl_b: np.ndarray,
    name_a: str = "Strategy A",
    name_b: str = "Strategy B"
) -> StrategyComparison:
    """
    Compare two strategy P&L distributions.

    Parameters
    ----------
    pnl_a, pnl_b : np.ndarray
        P&L arrays for each strategy (must be same length, same scenarios)
    name_a, name_b : str
        Names for the strategies

    Returns
    -------
    StrategyComparison
    """
    metrics_a = compute_risk_metrics(pnl_a)
    metrics_b = compute_risk_metrics(pnl_b)

    # Winner by scenario (same underlying prices)
    winner = np.sign(pnl_a - pnl_b)

    return StrategyComparison(
        name_a=name_a,
        name_b=name_b,
        pnl_a=pnl_a,
        pnl_b=pnl_b,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        winner_by_scenario=winner
    )


def render_strategy_comparison_tab(
    pnl_result_a: Optional[Dict[str, Any]],
    pnl_result_b: Optional[Dict[str, Any]],
    name_a: str = "Strategy A",
    name_b: str = "Strategy B",
    terminal_prices: Optional[np.ndarray] = None
) -> None:
    """
    Render strategy comparison educational tab.

    Shows side-by-side comparison of two strategies.
    """
    if pnl_result_a is None and pnl_result_b is None:
        st.info("Define two strategies to compare them side-by-side.")
        return

    st.markdown("### Strategy Comparison")
    st.caption(
        "Compare two option strategies across the same price scenarios. "
        "Understand when each strategy outperforms."
    )

    # Handle single strategy case
    if pnl_result_a is None or pnl_result_b is None:
        st.warning("Need two strategies to compare. Currently only one is defined.")
        if pnl_result_a is not None:
            st.write(f"**{name_a}** is defined with {len(pnl_result_a['pnl_values']):,} scenarios.")
        if pnl_result_b is not None:
            st.write(f"**{name_b}** is defined with {len(pnl_result_b['pnl_values']):,} scenarios.")
        return

    # Check alignment
    pnl_a = pnl_result_a['pnl_values']
    pnl_b = pnl_result_b['pnl_values']

    if len(pnl_a) != len(pnl_b):
        st.error("Strategies must be evaluated on the same number of scenarios.")
        return

    # Compare
    comparison = compare_strategies(pnl_a, pnl_b, name_a, name_b)

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "Distribution Comparison",
        "Metrics Comparison",
        "Scenario Analysis"
    ])

    with tab1:
        _render_distribution_comparison(comparison, terminal_prices)

    with tab2:
        _render_metrics_comparison(comparison)

    with tab3:
        _render_scenario_analysis(comparison, terminal_prices)


def _render_distribution_comparison(
    comparison: StrategyComparison,
    terminal_prices: Optional[np.ndarray]
) -> None:
    """Render overlaid P&L distributions."""
    st.markdown("#### P&L Distribution Comparison")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "P&L Distributions (Overlaid)",
            "Cumulative Distribution (CDF)"
        )
    )

    # 1. Histogram overlay
    for name, pnl, color in [
        (comparison.name_a, comparison.pnl_a, '#2563eb'),
        (comparison.name_b, comparison.pnl_b, '#dc2626')
    ]:
        fig.add_trace(
            go.Histogram(
                x=pnl,
                name=name,
                marker_color=color,
                opacity=0.5,
                nbinsx=50,
                histnorm='probability density'
            ),
            row=1, col=1
        )

    # 2. CDF comparison
    for name, pnl, color in [
        (comparison.name_a, comparison.pnl_a, '#2563eb'),
        (comparison.name_b, comparison.pnl_b, '#dc2626')
    ]:
        sorted_pnl = np.sort(pnl)
        cdf = np.arange(1, len(pnl) + 1) / len(pnl)

        fig.add_trace(
            go.Scatter(
                x=sorted_pnl,
                y=cdf,
                mode='lines',
                name=f'{name} CDF',
                line=dict(color=color, width=2)
            ),
            row=1, col=2
        )

    # Add breakeven line
    fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=2)

    fig.update_xaxes(title_text="P&L ($)", row=1, col=1)
    fig.update_xaxes(title_text="P&L ($)", row=1, col=2)
    fig.update_yaxes(title_text="Probability Density", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Probability", row=1, col=2)

    fig.update_layout(
        height=CHART_HEIGHT_STANDARD,
        barmode='overlay',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Quick stats comparison
    col1, col2, col3 = st.columns(3)

    with col1:
        diff_mean = comparison.metrics_a.mean_pnl - comparison.metrics_b.mean_pnl
        winner = comparison.name_a if diff_mean > 0 else comparison.name_b
        st.metric(
            "Higher Expected P&L",
            winner,
            f"by ${abs(diff_mean):,.0f}"
        )

    with col2:
        diff_var = comparison.metrics_a.var_95 - comparison.metrics_b.var_95
        # Remember VaR is negative, so less negative is better
        winner = comparison.name_a if diff_var > 0 else comparison.name_b
        st.metric(
            "Better VaR 95%",
            winner,
            f"by ${abs(diff_var):,.0f}"
        )

    with col3:
        a_wins = np.sum(comparison.winner_by_scenario == 1)
        b_wins = np.sum(comparison.winner_by_scenario == -1)
        winner = comparison.name_a if a_wins > b_wins else comparison.name_b
        win_pct = max(a_wins, b_wins) / len(comparison.pnl_a) * 100
        st.metric(
            "Wins More Scenarios",
            winner,
            f"{win_pct:.1f}% of scenarios"
        )


def _render_metrics_comparison(comparison: StrategyComparison) -> None:
    """Render detailed metrics comparison table."""
    st.markdown("#### Risk Metrics Comparison")

    # Create comparison table
    metrics_list = [
        ("Expected P&L", "mean_pnl", "${:,.2f}", True),
        ("Std Dev", "std_pnl", "${:,.2f}", False),
        ("VaR 95%", "var_95", "${:,.2f}", False),  # Less negative is better
        ("VaR 99%", "var_99", "${:,.2f}", False),
        ("CVaR 95%", "cvar_95", "${:,.2f}", False),
        ("CVaR 99%", "cvar_99", "${:,.2f}", False),
        ("P(Profit)", "prob_profit", "{:.1%}", True),
        ("Max Profit", "max_profit", "${:,.2f}", True),
        ("Max Loss", "max_loss", "${:,.2f}", False),
        ("Skewness", "skewness", "{:.3f}", None),  # Context-dependent
        ("Kurtosis", "kurtosis", "{:.3f}", None),
    ]

    import pandas as pd

    rows = []
    for label, attr, fmt, higher_is_better in metrics_list:
        val_a = getattr(comparison.metrics_a, attr)
        val_b = getattr(comparison.metrics_b, attr)

        if higher_is_better is True:
            winner = comparison.name_a if val_a > val_b else comparison.name_b
            better = "A" if val_a > val_b else "B"
        elif higher_is_better is False:
            # For VaR/CVaR, closer to 0 (less negative) is better
            # For std dev, lower is better
            if "var" in attr.lower() or "cvar" in attr.lower():
                winner = comparison.name_a if val_a > val_b else comparison.name_b
            else:
                winner = comparison.name_a if val_a < val_b else comparison.name_b
            better = "A" if winner == comparison.name_a else "B"
        else:
            winner = "-"
            better = "-"

        rows.append({
            "Metric": label,
            comparison.name_a: fmt.format(val_a),
            comparison.name_b: fmt.format(val_b),
            "Better": better
        })

    df = pd.DataFrame(rows)

    # Style the dataframe
    def highlight_winner(row):
        styles = [''] * len(row)
        if row['Better'] == 'A':
            styles[1] = 'background-color: #dcfce7'
        elif row['Better'] == 'B':
            styles[2] = 'background-color: #dcfce7'
        return styles

    styled_df = df.style.apply(highlight_winner, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Risk-adjusted return comparison
    st.markdown("#### Risk-Adjusted Metrics")

    col1, col2 = st.columns(2)

    with col1:
        # Sharpe-like ratio (return per unit risk)
        ratio_a = comparison.metrics_a.mean_pnl / comparison.metrics_a.std_pnl if comparison.metrics_a.std_pnl > 0 else 0
        ratio_b = comparison.metrics_b.mean_pnl / comparison.metrics_b.std_pnl if comparison.metrics_b.std_pnl > 0 else 0

        st.markdown(f"**Return/Risk Ratio:**")
        st.write(f"- {comparison.name_a}: {ratio_a:.3f}")
        st.write(f"- {comparison.name_b}: {ratio_b:.3f}")
        winner = comparison.name_a if ratio_a > ratio_b else comparison.name_b
        st.success(f"**{winner}** has better risk-adjusted return")

    with col2:
        # Sortino-like ratio (return per downside risk)
        downside_a = -comparison.metrics_a.cvar_95
        downside_b = -comparison.metrics_b.cvar_95
        sortino_a = comparison.metrics_a.mean_pnl / downside_a if downside_a > 0 else 0
        sortino_b = comparison.metrics_b.mean_pnl / downside_b if downside_b > 0 else 0

        st.markdown(f"**Return/CVaR Ratio:**")
        st.write(f"- {comparison.name_a}: {sortino_a:.3f}")
        st.write(f"- {comparison.name_b}: {sortino_b:.3f}")
        winner = comparison.name_a if sortino_a > sortino_b else comparison.name_b
        st.success(f"**{winner}** has better return per tail risk")


def _render_scenario_analysis(
    comparison: StrategyComparison,
    terminal_prices: Optional[np.ndarray]
) -> None:
    """Analyze which strategy wins in which scenarios."""
    st.markdown("#### Scenario-by-Scenario Analysis")

    # P&L difference
    pnl_diff = comparison.pnl_a - comparison.pnl_b

    # Winner counts
    a_wins = np.sum(comparison.winner_by_scenario == 1)
    b_wins = np.sum(comparison.winner_by_scenario == -1)
    ties = np.sum(comparison.winner_by_scenario == 0)
    total = len(comparison.pnl_a)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"{comparison.name_a} Wins", f"{a_wins:,}", f"{a_wins/total*100:.1f}%")
    with col2:
        st.metric(f"{comparison.name_b} Wins", f"{b_wins:,}", f"{b_wins/total*100:.1f}%")
    with col3:
        st.metric("Ties", f"{ties:,}", f"{ties/total*100:.1f}%")

    # Create visualization
    if terminal_prices is not None:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"P&L Difference ({comparison.name_a} - {comparison.name_b})",
                "Winner by Price Level"
            )
        )

        # Sample for scatter plot
        sample_size = min(3000, len(terminal_prices))
        idx = np.random.choice(len(terminal_prices), sample_size, replace=False)

        # 1. P&L difference vs terminal price
        fig.add_trace(
            go.Scatter(
                x=terminal_prices[idx],
                y=pnl_diff[idx],
                mode='markers',
                marker=dict(
                    size=4,
                    color=pnl_diff[idx],
                    colorscale='RdBu',
                    cmin=-np.percentile(np.abs(pnl_diff), 95),
                    cmax=np.percentile(np.abs(pnl_diff), 95),
                    opacity=0.5
                ),
                hovertemplate=(
                    'Price: $%{x:.2f}<br>'
                    f'{comparison.name_a} P&L advantage: $%{{y:,.0f}}<extra></extra>'
                )
            ),
            row=1, col=1
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

        # 2. Winner by price bucket
        n_buckets = 20
        price_buckets = np.percentile(terminal_prices, np.linspace(0, 100, n_buckets + 1))
        bucket_centers = (price_buckets[:-1] + price_buckets[1:]) / 2

        a_win_pct = []
        for i in range(len(price_buckets) - 1):
            mask = (terminal_prices >= price_buckets[i]) & (terminal_prices < price_buckets[i + 1])
            if mask.sum() > 0:
                a_win_pct.append(np.mean(comparison.winner_by_scenario[mask] == 1) * 100)
            else:
                a_win_pct.append(50)

        fig.add_trace(
            go.Bar(
                x=bucket_centers,
                y=a_win_pct,
                marker_color=['#2563eb' if p > 50 else '#dc2626' for p in a_win_pct],
                hovertemplate=(
                    'Price: $%{x:.2f}<br>'
                    f'{comparison.name_a} wins: %{{y:.1f}}%<extra></extra>'
                )
            ),
            row=1, col=2
        )

        fig.add_hline(y=50, line_dash="dash", line_color="gray", row=1, col=2)

        fig.update_xaxes(title_text="Terminal Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Terminal Price ($)", row=1, col=2)
        fig.update_yaxes(title_text=f"P&L Difference ($)", row=1, col=1)
        fig.update_yaxes(title_text=f"{comparison.name_a} Win %", row=1, col=2)

        fig.update_layout(height=CHART_HEIGHT_STANDARD, showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

    # P&L difference distribution
    st.markdown("#### P&L Difference Distribution")

    fig2 = go.Figure()

    fig2.add_trace(go.Histogram(
        x=pnl_diff,
        nbinsx=50,
        marker_color='#64748b',
        opacity=0.7,
        name='P&L Difference'
    ))

    fig2.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Equal Performance"
    )

    fig2.add_vline(
        x=np.mean(pnl_diff),
        line_dash="solid",
        line_color="blue",
        annotation_text=f"Mean: ${np.mean(pnl_diff):+,.0f}"
    )

    fig2.update_layout(
        title=f"P&L Difference: {comparison.name_a} minus {comparison.name_b}",
        xaxis_title="P&L Difference ($)",
        yaxis_title="Frequency",
        height=CHART_HEIGHT_STANDARD - 100
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Educational insight
    st.markdown("""
    <div style="background: #e8f4f8; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <h5 style="color: #1e3a5f; margin-top: 0;">Key Insight: No Strategy Dominates Everywhere</h5>
        <p style="margin-bottom: 0;">
            Typically, different strategies excel in different market conditions:
        </p>
        <ul style="margin-bottom: 0;">
            <li><strong>Long straddle/strangle</strong>: Wins when prices move a lot (either direction)</li>
            <li><strong>Short straddle/strangle</strong>: Wins when prices stay range-bound</li>
            <li><strong>Bull/bear spreads</strong>: Win in their directional scenario, capped upside but defined risk</li>
            <li><strong>Iron condor/butterfly</strong>: Wins in narrow ranges, loses on large moves</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
