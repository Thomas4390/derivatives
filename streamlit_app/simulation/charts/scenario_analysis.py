"""
Scenario Analysis Charts for Monte Carlo Option P&L Simulation.

Provides scatter plots and payoff diagrams to analyze the relationship
between terminal prices and P&L outcomes.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, List, Optional

from config.constants import (
    CHART_HEIGHT_STANDARD,
    CHART_HEIGHT_LARGE,
    PNL_COLORS,
    SCENARIO_SCATTER_MAX_POINTS
)
from backend.simulation import (
    compute_payoff_curve,
    find_breakeven_points,
    RiskMetrics
)


def render_scenario_analysis_tab(
    terminal_prices: np.ndarray,
    pnl_values: np.ndarray,
    position_arrays: Dict[str, np.ndarray],
    risk_metrics: RiskMetrics,
    params: Dict[str, Any]
) -> None:
    """
    Render the scenario analysis visualization tab.

    Parameters
    ----------
    terminal_prices : np.ndarray
        Array of terminal underlying prices from simulation
    pnl_values : np.ndarray
        Corresponding P&L values
    position_arrays : dict
        Position arrays for payoff curve calculation
    risk_metrics : RiskMetrics
        Pre-computed risk metrics
    params : dict
        Simulation parameters
    """
    if pnl_values is None or len(pnl_values) == 0:
        st.info("Run a P&L simulation to see scenario analysis.")
        return

    st.markdown("### Scenario Analysis")

    # Create tabs
    tab1, tab2 = st.tabs(["Price vs P&L", "Scenario Breakdown"])

    with tab1:
        _render_scatter_with_payoff(
            terminal_prices, pnl_values, position_arrays, params
        )

    with tab2:
        _render_scenario_breakdown(terminal_prices, pnl_values, params)


def _render_scatter_with_payoff(
    terminal_prices: np.ndarray,
    pnl_values: np.ndarray,
    position_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> None:
    """Render scatter plot of terminal price vs P&L with theoretical payoff."""
    fig = go.Figure()

    # Downsample if too many points
    n_points = len(terminal_prices)
    if n_points > SCENARIO_SCATTER_MAX_POINTS:
        indices = np.random.choice(n_points, SCENARIO_SCATTER_MAX_POINTS, replace=False)
        plot_prices = terminal_prices[indices]
        plot_pnl = pnl_values[indices]
    else:
        plot_prices = terminal_prices
        plot_pnl = pnl_values

    # Color points by profit/loss
    colors = np.where(plot_pnl > 0, PNL_COLORS['profit'], PNL_COLORS['loss'])

    # Scatter plot
    fig.add_trace(go.Scatter(
        x=plot_prices,
        y=plot_pnl,
        mode='markers',
        marker=dict(
            size=4,
            color=colors,
            opacity=0.5
        ),
        name='Simulated Outcomes',
        hovertemplate="Terminal: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>"
    ))

    # Generate theoretical payoff curve
    spot_range = np.linspace(
        terminal_prices.min() * 0.9,
        terminal_prices.max() * 1.1,
        200
    )

    if len(position_arrays.get('strikes', [])) > 0:
        payoff_curve = compute_payoff_curve(
            spot_range,
            position_arrays['strikes'],
            position_arrays['option_types'],
            position_arrays['position_types'],
            position_arrays['quantities'],
            position_arrays['premiums'],
            position_arrays.get('stock_quantity', 0.0),
            position_arrays.get('stock_entry_price', 0.0),
            multiplier=100.0
        )

        fig.add_trace(go.Scatter(
            x=spot_range,
            y=payoff_curve,
            mode='lines',
            name='Theoretical Payoff',
            line=dict(color=PNL_COLORS['payoff_curve'], width=3)
        ))

        # Find and mark breakeven points
        breakevens = find_breakeven_points(payoff_curve, spot_range)
        for be in breakevens:
            fig.add_vline(
                x=be,
                line_dash="dash",
                line_color=PNL_COLORS['breakeven'],
                annotation_text=f"BE: ${be:.2f}",
                annotation_position="top"
            )

    # Add horizontal line at zero
    fig.add_hline(y=0, line_dash="solid", line_color="#475569", line_width=1)

    # Add vertical line at initial spot
    spot_price = params.get('spot_price', 100)
    fig.add_vline(
        x=spot_price,
        line_dash="dot",
        line_color="#1f2937",
        annotation_text=f"S₀: ${spot_price:.2f}",
        annotation_position="bottom right"
    )

    fig.update_layout(
        title=dict(
            text="Terminal Price vs P&L",
            font=dict(size=16)
        ),
        xaxis_title="Terminal Price ($)",
        yaxis_title="P&L ($)",
        height=CHART_HEIGHT_LARGE,
        hovermode='closest',
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.9)'
        ),
        margin=dict(l=60, r=40, t=60, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Min Terminal Price", f"${terminal_prices.min():.2f}")
    with col2:
        st.metric("Max Terminal Price", f"${terminal_prices.max():.2f}")
    with col3:
        st.metric("Initial Spot", f"${spot_price:.2f}")
    with col4:
        avg_move = (terminal_prices.mean() - spot_price) / spot_price * 100
        st.metric("Avg Price Change", f"{avg_move:+.1f}%")


def _render_scenario_breakdown(
    terminal_prices: np.ndarray,
    pnl_values: np.ndarray,
    params: Dict[str, Any]
) -> None:
    """Render scenario breakdown analysis."""
    spot_price = params.get('spot_price', 100)

    # Define scenarios based on price movement
    scenarios = {
        'Large Down (< -15%)': terminal_prices < spot_price * 0.85,
        'Medium Down (-15% to -5%)': (terminal_prices >= spot_price * 0.85) & (terminal_prices < spot_price * 0.95),
        'Small Down (-5% to 0%)': (terminal_prices >= spot_price * 0.95) & (terminal_prices < spot_price),
        'Small Up (0% to +5%)': (terminal_prices >= spot_price) & (terminal_prices < spot_price * 1.05),
        'Medium Up (+5% to +15%)': (terminal_prices >= spot_price * 1.05) & (terminal_prices < spot_price * 1.15),
        'Large Up (> +15%)': terminal_prices >= spot_price * 1.15
    }

    # Calculate statistics for each scenario
    scenario_stats = []
    for name, mask in scenarios.items():
        if mask.sum() > 0:
            pnl_subset = pnl_values[mask]
            scenario_stats.append({
                'Scenario': name,
                'Count': mask.sum(),
                'Probability': mask.sum() / len(terminal_prices) * 100,
                'Mean P&L': pnl_subset.mean(),
                'Min P&L': pnl_subset.min(),
                'Max P&L': pnl_subset.max(),
                'P(Profit)': (pnl_subset > 0).mean() * 100
            })

    # Create bar chart for scenario P&L
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Average P&L by Scenario", "Probability & Win Rate by Scenario"),
        vertical_spacing=0.15
    )

    scenario_names = [s['Scenario'] for s in scenario_stats]
    mean_pnls = [s['Mean P&L'] for s in scenario_stats]
    probabilities = [s['Probability'] for s in scenario_stats]
    win_rates = [s['P(Profit)'] for s in scenario_stats]

    # Color bars by profit/loss
    bar_colors = [PNL_COLORS['profit'] if p > 0 else PNL_COLORS['loss'] for p in mean_pnls]

    # Mean P&L bars
    fig.add_trace(
        go.Bar(
            x=scenario_names,
            y=mean_pnls,
            marker_color=bar_colors,
            name='Mean P&L',
            text=[f"${p:,.0f}" for p in mean_pnls],
            textposition='outside'
        ),
        row=1, col=1
    )

    # Probability bars
    fig.add_trace(
        go.Bar(
            x=scenario_names,
            y=probabilities,
            marker_color='#3b82f6',
            name='Probability (%)',
            text=[f"{p:.1f}%" for p in probabilities],
            textposition='outside',
            opacity=0.7
        ),
        row=2, col=1
    )

    # Win rate line
    fig.add_trace(
        go.Scatter(
            x=scenario_names,
            y=win_rates,
            mode='lines+markers+text',
            name='Win Rate (%)',
            line=dict(color='#10b981', width=2),
            marker=dict(size=10),
            text=[f"{w:.0f}%" for w in win_rates],
            textposition='top center'
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=CHART_HEIGHT_LARGE,
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="right", x=0.99
        )
    )

    # Add zero line on P&L chart
    fig.add_hline(y=0, line_dash="solid", line_color="#475569", row=1, col=1)

    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
    fig.update_yaxes(title_text="Percentage (%)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Detailed scenario table
    st.markdown("#### Scenario Details")

    # Format table data
    table_data = []
    for s in scenario_stats:
        pnl_color = PNL_COLORS['profit'] if s['Mean P&L'] > 0 else PNL_COLORS['loss']
        win_color = '#059669' if s['P(Profit)'] > 50 else '#dc2626'
        table_data.append({
            'Scenario': s['Scenario'],
            'Count': f"{s['Count']:,} ({s['Probability']:.1f}%)",
            'Mean P&L': f"${s['Mean P&L']:,.2f}",
            'Range': f"${s['Min P&L']:,.0f} to ${s['Max P&L']:,.0f}",
            'Win Rate': f"{s['P(Profit)']:.1f}%"
        })

    # Display as columns
    col_headers = st.columns([2, 2, 1.5, 2, 1])
    col_headers[0].markdown("**Scenario**")
    col_headers[1].markdown("**Occurrences**")
    col_headers[2].markdown("**Avg P&L**")
    col_headers[3].markdown("**P&L Range**")
    col_headers[4].markdown("**Win %**")

    for row in table_data:
        cols = st.columns([2, 2, 1.5, 2, 1])
        cols[0].markdown(row['Scenario'])
        cols[1].markdown(row['Count'])
        cols[2].markdown(row['Mean P&L'])
        cols[3].markdown(row['Range'])
        cols[4].markdown(row['Win Rate'])


def render_beneficial_scenarios(
    terminal_prices: np.ndarray,
    pnl_values: np.ndarray,
    params: Dict[str, Any]
) -> None:
    """Highlight which scenarios were beneficial vs detrimental."""
    st.markdown("#### Scenario Impact Analysis")

    # Sort scenarios by contribution to expected P&L
    spot_price = params.get('spot_price', 100)

    # Create price buckets
    price_changes = (terminal_prices - spot_price) / spot_price * 100
    buckets = np.digitize(price_changes, [-20, -10, -5, 0, 5, 10, 20])

    bucket_labels = [
        '< -20%', '-20% to -10%', '-10% to -5%', '-5% to 0%',
        '0% to +5%', '+5% to +10%', '+10% to +20%', '> +20%'
    ]

    # Calculate contribution to expected P&L
    contributions = []
    for i in range(8):
        mask = buckets == i
        if mask.sum() > 0:
            prob = mask.sum() / len(terminal_prices)
            mean_pnl = pnl_values[mask].mean()
            contribution = prob * mean_pnl
            contributions.append({
                'bucket': bucket_labels[i],
                'probability': prob,
                'mean_pnl': mean_pnl,
                'contribution': contribution,
                'is_positive': mean_pnl > 0
            })

    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Beneficial Scenarios** (Positive P&L)")
        beneficial = [c for c in contributions if c['is_positive']]
        for c in beneficial[:4]:
            impact = "high" if abs(c['contribution']) > 10 else "moderate"
            st.markdown(f"""
            - **{c['bucket']}**: Avg P&L ${c['mean_pnl']:,.0f} ({c['probability']:.1%} prob)
            """)

    with col2:
        st.markdown("**Detrimental Scenarios** (Negative P&L)")
        detrimental = [c for c in contributions if not c['is_positive']]
        for c in detrimental[:4]:
            st.markdown(f"""
            - **{c['bucket']}**: Avg P&L ${c['mean_pnl']:,.0f} ({c['probability']:.1%} prob)
            """)
