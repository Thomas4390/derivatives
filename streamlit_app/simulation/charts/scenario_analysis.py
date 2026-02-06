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
from config.styles import render_stats_row, stats_table_html
from backend.simulation import (
    compute_payoff_curve,
    find_breakeven_points,
    RiskMetrics
)
from backend.greeks.analytic import bs_all_greeks


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

    # Section header with native help
    st.markdown("### Scenario Analysis")
    st.caption(
        "Scenario Analysis shows how your strategy performs across different market conditions. "
        "It helps identify which price movements are beneficial or detrimental to your position."
    )

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Price vs P&L", "Scenario Breakdown", "Greeks by Scenario"])

    with tab1:
        _render_scatter_with_payoff(
            terminal_prices, pnl_values, position_arrays, params
        )

    with tab2:
        _render_scenario_breakdown(terminal_prices, pnl_values, params)

    with tab3:
        _render_greeks_by_scenario(terminal_prices, pnl_values, position_arrays, params)


def _render_scatter_with_payoff(
    terminal_prices: np.ndarray,
    pnl_values: np.ndarray,
    position_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> None:
    """Render scatter plot of terminal price vs P&L with theoretical payoff."""
    # Toggle for theoretical payoff
    show_payoff = st.checkbox(
        "Show Theoretical Payoff Curve",
        value=False,
        help="Toggle the theoretical payoff curve overlay. Disable to see simulated outcomes more clearly."
    )

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

    # Show payoff curve if options OR stock position exists
    has_options = len(position_arrays.get('strikes', [])) > 0
    has_stock = position_arrays.get('stock_quantity', 0.0) != 0

    if (has_options or has_stock) and show_payoff:
        payoff_curve = compute_payoff_curve(
            spot_range,
            position_arrays.get('strikes', np.array([])),
            position_arrays.get('option_types', np.array([])),
            position_arrays.get('position_types', np.array([])),
            position_arrays.get('quantities', np.array([])),
            position_arrays.get('premiums', np.array([])),
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

    st.plotly_chart(fig, width="stretch")

    # Summary stats with styled cards
    avg_move = (terminal_prices.mean() - spot_price) / spot_price * 100
    move_variant = "green" if avg_move > 0 else "red"

    summary_stats = [
        ("Min Terminal", f"${terminal_prices.min():.2f}", "Lowest simulated price"),
        ("Max Terminal", f"${terminal_prices.max():.2f}", "Highest simulated price"),
        ("Initial Spot", f"${spot_price:.2f}", "Starting price"),
        ("Avg Price Change", f"{avg_move:+.1f}%", "Mean vs initial"),
    ]
    render_stats_row(summary_stats, ["slate", "slate", "blue", move_variant])


def _render_scenario_breakdown(
    terminal_prices: np.ndarray,
    pnl_values: np.ndarray,
    params: Dict[str, Any]
) -> None:
    """Render scenario breakdown analysis."""
    # Help caption for this section
    st.caption(
        "Scenario Breakdown divides price movements into categories (Large Down, Small Up, etc.) "
        "and shows the average P&L and probability for each. Use this to understand which market "
        "conditions favor your strategy."
    )

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
        vertical_spacing=0.25
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
        height=CHART_HEIGHT_LARGE + 80,
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="right", x=0.99
        ),
        margin=dict(t=80, b=60)
    )

    # Add zero line on P&L chart
    fig.add_hline(y=0, line_dash="solid", line_color="#475569", row=1, col=1)

    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
    fig.update_yaxes(title_text="Percentage (%)", row=2, col=1)

    st.plotly_chart(fig, width="stretch")

    # Detailed scenario table
    st.markdown("#### Scenario Details")

    import pandas as pd

    # Build dataframe with formatted data
    df_scenarios = pd.DataFrame([
        {
            'Scenario': s['Scenario'],
            'Occurrences': f"{s['Count']:,} ({s['Probability']:.1f}%)",
            'Mean P&L': f"${s['Mean P&L']:,.2f}",
            'P&L Range': f"${s['Min P&L']:,.0f} to ${s['Max P&L']:,.0f}",
            'Win Rate': f"{s['P(Profit)']:.1f}%",
            '_mean_pnl': s['Mean P&L'],  # Hidden column for styling
            '_win_rate': s['P(Profit)']  # Hidden column for styling
        }
        for s in scenario_stats
    ])

    # Display with st.dataframe
    st.dataframe(
        df_scenarios[['Scenario', 'Occurrences', 'Mean P&L', 'P&L Range', 'Win Rate']],
        width="stretch",
        hide_index=True,
        column_config={
            "Scenario": st.column_config.TextColumn("Scenario", width="medium"),
            "Occurrences": st.column_config.TextColumn("Occurrences", width="small"),
            "Mean P&L": st.column_config.TextColumn("Avg P&L", width="small"),
            "P&L Range": st.column_config.TextColumn("P&L Range", width="medium"),
            "Win Rate": st.column_config.TextColumn("Win %", width="small")
        }
    )


def _render_greeks_by_scenario(
    terminal_prices: np.ndarray,
    pnl_values: np.ndarray,
    position_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> None:
    """
    Render Greeks at each price level to show which contributes most to P&L.

    This helps users understand the "why" behind P&L changes:
    - Delta: P&L from directional movement
    - Gamma: P&L from acceleration (large moves)
    - Vega: P&L from volatility changes
    - Theta: P&L from time decay
    """
    st.caption(
        "This view shows how Greeks change as the underlying price moves. "
        "Understanding which Greek dominates at each price level explains your P&L."
    )

    strikes = position_arrays.get('strikes', np.array([]))
    option_types = position_arrays.get('option_types', np.array([]))
    position_types = position_arrays.get('position_types', np.array([]))
    quantities = position_arrays.get('quantities', np.array([]))

    if len(strikes) == 0:
        st.info("Add options to your strategy to see Greeks analysis.")
        return

    spot_price = params.get('spot_price', 100)
    time_to_expiry = params.get('time_horizon', 1.0)
    risk_free_rate = params.get('risk_free_rate', 0.05)
    volatility = params.get('volatility', 0.20)

    # Create price range for Greeks calculation
    price_min = terminal_prices.min() * 0.95
    price_max = terminal_prices.max() * 1.05
    price_range = np.linspace(price_min, price_max, 50)

    # Calculate portfolio Greeks at each price level
    deltas = []
    gammas = []
    vegas = []
    thetas = []

    for price in price_range:
        portfolio_delta = 0.0
        portfolio_gamma = 0.0
        portfolio_vega = 0.0
        portfolio_theta = 0.0

        for i in range(len(strikes)):
            is_call = option_types[i] == 1
            sign = position_types[i]
            qty = quantities[i]

            greeks = bs_all_greeks(
                s=price,
                k=strikes[i],
                t=max(time_to_expiry, 0.001),
                r=risk_free_rate,
                q=0.0,
                sigma=volatility,
                is_call=is_call
            )

            # Each contract is for 100 shares
            multiplier = sign * qty * 100
            portfolio_delta += greeks[1] * multiplier
            portfolio_gamma += greeks[2] * multiplier
            portfolio_vega += greeks[3] * multiplier
            portfolio_theta += greeks[4] * multiplier

        deltas.append(portfolio_delta)
        gammas.append(portfolio_gamma)
        vegas.append(portfolio_vega)
        thetas.append(portfolio_theta)

    # Create subplot with Greeks across price range
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Delta (Directional)", "Gamma (Acceleration)",
                       "Vega (Vol Sensitivity)", "Theta (Time Decay)"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # Delta
    fig.add_trace(
        go.Scatter(
            x=price_range, y=deltas,
            mode='lines', name='Delta',
            line=dict(color='#3b82f6', width=2)
        ),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)

    # Gamma
    fig.add_trace(
        go.Scatter(
            x=price_range, y=gammas,
            mode='lines', name='Gamma',
            line=dict(color='#10b981', width=2)
        ),
        row=1, col=2
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2)

    # Vega
    fig.add_trace(
        go.Scatter(
            x=price_range, y=vegas,
            mode='lines', name='Vega',
            line=dict(color='#8b5cf6', width=2)
        ),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    # Theta
    fig.add_trace(
        go.Scatter(
            x=price_range, y=thetas,
            mode='lines', name='Theta',
            line=dict(color='#ef4444', width=2)
        ),
        row=2, col=2
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=2)

    # Add vertical line at current spot
    for row in [1, 2]:
        for col in [1, 2]:
            fig.add_vline(
                x=spot_price, line_dash="dash", line_color="#1f2937",
                annotation_text="S₀" if row == 1 and col == 1 else None,
                row=row, col=col
            )

    fig.update_layout(
        height=CHART_HEIGHT_LARGE,
        showlegend=False,
        margin=dict(t=60, b=40)
    )

    # Update axis labels
    fig.update_xaxes(title_text="Underlying Price ($)", row=2, col=1)
    fig.update_xaxes(title_text="Underlying Price ($)", row=2, col=2)
    fig.update_yaxes(title_text="Shares Eq.", row=1, col=1)
    fig.update_yaxes(title_text="Δ per $1", row=1, col=2)
    fig.update_yaxes(title_text="$ per 1% Vol", row=2, col=1)
    fig.update_yaxes(title_text="$/day", row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # Educational interpretation
    _render_greeks_interpretation(
        deltas, gammas, vegas, thetas, price_range, spot_price
    )


def _render_greeks_interpretation(
    deltas: List[float],
    gammas: List[float],
    vegas: List[float],
    thetas: List[float],
    price_range: np.ndarray,
    spot_price: float
) -> None:
    """Provide educational interpretation of Greeks patterns."""
    import pandas as pd

    # Find current values (at spot)
    idx_spot = np.argmin(np.abs(price_range - spot_price))
    current_delta = deltas[idx_spot]
    current_gamma = gammas[idx_spot]
    current_vega = vegas[idx_spot]
    current_theta = thetas[idx_spot]

    st.markdown("#### Current Greeks at Spot Price")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta_color = "green" if current_delta > 0 else "red" if current_delta < 0 else "gray"
        st.metric("Delta", f"{current_delta:+,.1f}", "Shares equivalent")
    with col2:
        gamma_sign = "Long" if current_gamma > 0 else "Short"
        st.metric("Gamma", f"{current_gamma:,.4f}", f"{gamma_sign} gamma")
    with col3:
        vega_sign = "Long" if current_vega > 0 else "Short"
        st.metric("Vega", f"${current_vega:+,.1f}", f"{vega_sign} vol")
    with col4:
        theta_impact = "Earning" if current_theta > 0 else "Paying"
        st.metric("Theta", f"${current_theta:+,.2f}/day", theta_impact)

    # Dominant Greek analysis
    st.markdown("#### Which Greek Drives Your P&L?")

    interpretations = []

    # Delta interpretation
    if abs(current_delta) > 50:
        direction = "bullish" if current_delta > 0 else "bearish"
        interpretations.append(
            f"**Delta ({current_delta:+,.0f})**: You have significant directional exposure. "
            f"A 1% move = ~${abs(current_delta * spot_price * 0.01):,.0f} P&L change. "
            f"Strategy is {direction}."
        )

    # Gamma interpretation
    if current_gamma > 0.5:
        interpretations.append(
            f"**Gamma (+{current_gamma:.2f})**: You're long gamma - you profit from large moves "
            f"in either direction. The bigger the move, the more you make."
        )
    elif current_gamma < -0.5:
        interpretations.append(
            f"**Gamma ({current_gamma:.2f})**: You're short gamma - large moves hurt you. "
            f"You need the underlying to stay relatively stable."
        )

    # Vega interpretation
    if abs(current_vega) > 50:
        vol_exposure = "long" if current_vega > 0 else "short"
        interpretations.append(
            f"**Vega (${current_vega:+,.0f})**: You're {vol_exposure} volatility. "
            f"A 1% change in implied vol = ~${abs(current_vega):,.0f} P&L change."
        )

    # Theta interpretation
    if current_theta < -10:
        interpretations.append(
            f"**Theta (${current_theta:+,.0f}/day)**: Time decay is working against you. "
            f"You need the underlying to move (or vol to increase) to overcome this."
        )
    elif current_theta > 10:
        interpretations.append(
            f"**Theta (${current_theta:+,.0f}/day)**: Time decay works for you. "
            f"You earn money each day if nothing moves."
        )

    if interpretations:
        for interp in interpretations:
            st.markdown(f"- {interp}")
    else:
        st.markdown("- Position has relatively balanced Greek exposures.")

    # Educational callout
    st.markdown("""
    <div style="background: #e8f4f8; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <h5 style="color: #1e3a5f; margin-top: 0;">💡 Key Insight: P&L Attribution</h5>
        <p style="margin-bottom: 0;">
            Your total P&L at any price level can be approximated as:
        </p>
        <p style="font-family: monospace; background: #f1f5f9; padding: 0.5rem; border-radius: 4px;">
            P&L ≈ Delta × ΔS + ½ × Gamma × (ΔS)² + Vega × Δσ + Theta × Δt
        </p>
        <ul style="margin-bottom: 0;">
            <li><strong>Delta P&L</strong>: Linear with price movement</li>
            <li><strong>Gamma P&L</strong>: Accelerates with large moves (squared term!)</li>
            <li><strong>Vega P&L</strong>: Changes with implied volatility</li>
            <li><strong>Theta P&L</strong>: Accumulates with time passage</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


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
