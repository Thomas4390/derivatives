"""
Scenario Dashboard Module.

Pedagogical "What-If" stress testing dashboard:
1. Price shocks: +/- 10%, +/- 20% instant moves
2. Volatility shocks: IV +25%, +50%, -25%
3. Combined scenarios: Crash (price -15%, vol +100%)

Heat map visualization with P&L at each scenario.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from backend.greeks.analytic import bs_all_greeks
from config.constants import CHART_HEIGHT_STANDARD, CHART_HEIGHT_LARGE
from config.styles import render_stats_row


@dataclass
class ScenarioResult:
    """Result of a stress scenario."""
    name: str
    price_shock: float  # as decimal (0.1 = +10%)
    vol_shock: float    # as decimal (0.25 = +25% of current vol, not +25 vol points)
    new_spot: float
    new_vol: float
    portfolio_value: float
    pnl: float
    delta: float
    gamma: float
    vega: float
    theta: float


def calculate_scenario_pnl(
    spot: float,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    price_shock: float,
    vol_shock: float,
    time_shock: float = 0.0  # fraction of time elapsed
) -> ScenarioResult:
    """
    Calculate portfolio P&L under a stress scenario.

    Parameters
    ----------
    price_shock : float
        Price change as decimal (0.1 = +10%, -0.2 = -20%)
    vol_shock : float
        Volatility change as relative change (0.5 = vol increases by 50%)
    time_shock : float
        Time elapsed as fraction (0.1 = 10% of time passed)

    Returns
    -------
    ScenarioResult
    """
    # Apply shocks
    new_spot = spot * (1 + price_shock)
    new_vol = volatility * (1 + vol_shock)
    new_time = time_to_expiry * (1 - time_shock)

    # Calculate portfolio value under scenario
    portfolio_value = 0.0
    portfolio_delta = 0.0
    portfolio_gamma = 0.0
    portfolio_vega = 0.0
    portfolio_theta = 0.0

    for i in range(len(strikes)):
        is_call = option_types[i] == 1
        sign = position_types[i]
        qty = quantities[i]

        greeks = bs_all_greeks(
            s=new_spot,
            k=strikes[i],
            t=max(new_time, 0.001),  # Avoid t=0
            r=risk_free_rate,
            q=0.0,
            sigma=new_vol,
            is_call=is_call
        )

        # Each contract is for 100 shares
        multiplier = sign * qty * 100
        portfolio_value += greeks[0] * multiplier  # price
        portfolio_delta += greeks[1] * multiplier  # delta
        portfolio_gamma += greeks[2] * multiplier  # gamma
        portfolio_vega += greeks[3] * multiplier   # vega
        portfolio_theta += greeks[4] * multiplier  # theta

    # Initial portfolio value
    initial_value = 0.0
    for i in range(len(strikes)):
        is_call = option_types[i] == 1
        sign = position_types[i]
        qty = quantities[i]

        greeks = bs_all_greeks(
            s=spot,
            k=strikes[i],
            t=time_to_expiry,
            r=risk_free_rate,
            q=0.0,
            sigma=volatility,
            is_call=is_call
        )

        multiplier = sign * qty * 100
        initial_value += greeks[0] * multiplier

    pnl = portfolio_value - initial_value

    # Create scenario name
    parts = []
    if price_shock != 0:
        parts.append(f"S {price_shock:+.0%}")
    if vol_shock != 0:
        parts.append(f"Vol {vol_shock:+.0%}")
    if time_shock != 0:
        parts.append(f"T {time_shock:+.0%}")
    name = ", ".join(parts) if parts else "Base"

    return ScenarioResult(
        name=name,
        price_shock=price_shock,
        vol_shock=vol_shock,
        new_spot=new_spot,
        new_vol=new_vol,
        portfolio_value=portfolio_value,
        pnl=pnl,
        delta=portfolio_delta,
        gamma=portfolio_gamma,
        vega=portfolio_vega,
        theta=portfolio_theta
    )


def render_scenario_dashboard_tab(params: Dict[str, Any]) -> None:
    """
    Render the stress scenario dashboard.

    Allows user to see P&L under various price/volatility shocks.
    """
    position_arrays = params.get('position_arrays', {})

    if len(position_arrays.get('strikes', [])) == 0:
        st.info("Define a strategy to see stress scenario analysis.")
        return

    st.markdown("### Stress Scenario Dashboard")
    st.caption(
        "Test your strategy against various market shocks. "
        "See how P&L changes with price moves, volatility spikes, and combined scenarios."
    )

    # Get parameters
    spot = params.get('spot_price', 100.0)
    time_to_expiry = params.get('time_horizon', 1.0)
    risk_free_rate = params.get('risk_free_rate', 0.05)
    volatility = params.get('volatility', 0.20)

    strikes = position_arrays['strikes']
    option_types = position_arrays['option_types']
    position_types = position_arrays['position_types']
    quantities = position_arrays['quantities']
    premiums = position_arrays['premiums']

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "Scenario Matrix",
        "Custom Scenario",
        "Predefined Scenarios"
    ])

    with tab1:
        _render_scenario_matrix(
            spot, strikes, option_types, position_types, quantities, premiums,
            time_to_expiry, risk_free_rate, volatility
        )

    with tab2:
        _render_custom_scenario(
            spot, strikes, option_types, position_types, quantities, premiums,
            time_to_expiry, risk_free_rate, volatility
        )

    with tab3:
        _render_predefined_scenarios(
            spot, strikes, option_types, position_types, quantities, premiums,
            time_to_expiry, risk_free_rate, volatility
        )


def _render_scenario_matrix(
    spot: float,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float
) -> None:
    """Render heat map of P&L across price/vol shocks."""
    st.markdown("#### P&L Heat Map: Price vs Volatility Shocks")

    # Define shock ranges
    col1, col2 = st.columns(2)
    with col1:
        price_range = st.slider(
            "Price Shock Range (%)",
            min_value=5,
            max_value=50,
            value=25,
            step=5,
            key="scenario_price_range"
        )
    with col2:
        vol_range = st.slider(
            "Vol Shock Range (%)",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            key="scenario_vol_range"
        )

    # Create grid
    price_shocks = np.linspace(-price_range/100, price_range/100, 11)
    vol_shocks = np.linspace(-vol_range/100, vol_range/100, 9)

    # Calculate P&L at each point
    pnl_matrix = np.zeros((len(vol_shocks), len(price_shocks)))

    for i, vol_shock in enumerate(vol_shocks):
        for j, price_shock in enumerate(price_shocks):
            result = calculate_scenario_pnl(
                spot, strikes, option_types, position_types, quantities, premiums,
                time_to_expiry, risk_free_rate, volatility,
                price_shock, vol_shock
            )
            pnl_matrix[i, j] = result.pnl

    # Create heat map
    fig = go.Figure(data=go.Heatmap(
        z=pnl_matrix,
        x=[f"{s*100:+.0f}%" for s in price_shocks],
        y=[f"{s*100:+.0f}%" for s in vol_shocks],
        colorscale='RdYlGn',
        zmid=0,
        text=[[f"${v:,.0f}" for v in row] for row in pnl_matrix],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate=(
            "Price Shock: %{x}<br>"
            "Vol Shock: %{y}<br>"
            "P&L: $%{z:,.0f}<extra></extra>"
        )
    ))

    fig.update_layout(
        title="P&L Under Combined Price and Volatility Shocks",
        xaxis_title="Price Shock",
        yaxis_title="Volatility Shock",
        height=CHART_HEIGHT_LARGE - 50
    )

    st.plotly_chart(fig, use_container_width=True)

    # Key metrics from matrix
    max_profit = np.max(pnl_matrix)
    max_loss = np.min(pnl_matrix)
    max_profit_idx = np.unravel_index(np.argmax(pnl_matrix), pnl_matrix.shape)
    max_loss_idx = np.unravel_index(np.argmin(pnl_matrix), pnl_matrix.shape)

    col1, col2 = st.columns(2)
    with col1:
        st.success(
            f"**Best Scenario:** ${max_profit:,.0f}\n\n"
            f"Price: {price_shocks[max_profit_idx[1]]*100:+.0f}%, "
            f"Vol: {vol_shocks[max_profit_idx[0]]*100:+.0f}%"
        )
    with col2:
        st.error(
            f"**Worst Scenario:** ${max_loss:,.0f}\n\n"
            f"Price: {price_shocks[max_loss_idx[1]]*100:+.0f}%, "
            f"Vol: {vol_shocks[max_loss_idx[0]]*100:+.0f}%"
        )


def _render_custom_scenario(
    spot: float,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float
) -> None:
    """Allow user to define custom stress scenario."""
    st.markdown("#### Custom Scenario Builder")

    col1, col2, col3 = st.columns(3)

    with col1:
        price_shock = st.slider(
            "Price Shock (%)",
            min_value=-50,
            max_value=50,
            value=0,
            step=1,
            key="custom_price_shock"
        ) / 100

    with col2:
        vol_shock = st.slider(
            "Volatility Shock (%)",
            min_value=-50,
            max_value=100,
            value=0,
            step=5,
            key="custom_vol_shock"
        ) / 100

    with col3:
        time_elapsed = st.slider(
            "Time Elapsed (%)",
            min_value=0,
            max_value=90,
            value=0,
            step=10,
            key="custom_time_shock"
        ) / 100

    # Calculate scenario
    result = calculate_scenario_pnl(
        spot, strikes, option_types, position_types, quantities, premiums,
        time_to_expiry, risk_free_rate, volatility,
        price_shock, vol_shock, time_elapsed
    )

    # Display results
    st.markdown("---")
    st.markdown("#### Scenario Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("New Spot", f"${result.new_spot:,.2f}", f"{price_shock*100:+.0f}%")
    with col2:
        st.metric("New Vol", f"{result.new_vol*100:.1f}%", f"{vol_shock*100:+.0f}%")
    with col3:
        pnl_color = "normal" if result.pnl >= 0 else "inverse"
        st.metric("P&L", f"${result.pnl:,.0f}", delta_color=pnl_color)

    # Greeks under scenario
    st.markdown("#### Portfolio Greeks Under Scenario")

    greeks_stats = [
        ("Delta", f"{result.delta:+,.2f}", "Shares equivalent"),
        ("Gamma", f"{result.gamma:,.4f}", "Delta sensitivity"),
        ("Vega", f"${result.vega:,.2f}", "Per 1% vol"),
        ("Theta", f"${result.theta:,.2f}/day", "Time decay"),
    ]
    render_stats_row(greeks_stats, ["blue", "teal", "purple", "red"])


def _render_predefined_scenarios(
    spot: float,
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    premiums: np.ndarray,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float
) -> None:
    """Render predefined stress scenarios."""
    st.markdown("#### Predefined Stress Scenarios")

    # Define scenarios
    scenarios = [
        ("Base Case", 0, 0),
        ("Market Rally (+10%)", 0.10, 0),
        ("Market Selloff (-10%)", -0.10, 0),
        ("Flash Crash (-20%, Vol +100%)", -0.20, 1.00),
        ("Volatility Spike (Vol +50%)", 0, 0.50),
        ("Vol Crush (Vol -30%)", 0, -0.30),
        ("Slow Grind Up (+15%, Vol -20%)", 0.15, -0.20),
        ("Mild Selloff (-10%, Vol +30%)", -0.10, 0.30),
        ("Black Swan (-30%, Vol +150%)", -0.30, 1.50),
        ("Melt-up (+30%, Vol +25%)", 0.30, 0.25),
    ]

    results = []
    for name, price_shock, vol_shock in scenarios:
        result = calculate_scenario_pnl(
            spot, strikes, option_types, position_types, quantities, premiums,
            time_to_expiry, risk_free_rate, volatility,
            price_shock, vol_shock
        )
        results.append({
            "Scenario": name,
            "Price": f"${result.new_spot:,.2f}",
            "Vol": f"{result.new_vol*100:.1f}%",
            "P&L": result.pnl,
            "Delta": result.delta,
            "Gamma": result.gamma
        })

    import pandas as pd
    df = pd.DataFrame(results)

    # Format P&L with color
    def style_pnl(val):
        if isinstance(val, float):
            color = 'green' if val >= 0 else 'red'
            return f'color: {color}; font-weight: bold'
        return ''

    # Create formatted columns
    df_display = df.copy()
    df_display['P&L'] = df_display['P&L'].apply(lambda x: f"${x:+,.0f}")
    df_display['Delta'] = df_display['Delta'].apply(lambda x: f"{x:+,.1f}")
    df_display['Gamma'] = df_display['Gamma'].apply(lambda x: f"{x:,.4f}")

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Bar chart of P&L by scenario
    pnl_values = [r['P&L'] for r in results]
    colors = ['#10b981' if v >= 0 else '#ef4444' for v in pnl_values]

    fig = go.Figure(data=go.Bar(
        x=[r['Scenario'] for r in results],
        y=pnl_values,
        marker_color=colors,
        text=[f"${v:+,.0f}" for v in pnl_values],
        textposition='outside'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="P&L by Stress Scenario",
        xaxis_title="Scenario",
        yaxis_title="P&L ($)",
        height=CHART_HEIGHT_STANDARD,
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig, use_container_width=True)

    # Educational insight
    st.markdown("""
    <div style="background: #fef2f2; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <h5 style="color: #991b1b; margin-top: 0;">Key Insight: Tail Risk Matters</h5>
        <p style="margin-bottom: 0;">
            Notice how extreme scenarios can dominate your P&L distribution. A strategy that
            looks good in normal conditions may have devastating losses in a crash scenario.
            This is why:
        </p>
        <ul style="margin-bottom: 0;">
            <li><strong>Option sellers</strong> often suffer most in tail events (unlimited downside)</li>
            <li><strong>Spreads</strong> limit both gains and losses, reducing tail exposure</li>
            <li><strong>Long volatility</strong> strategies can provide crash protection</li>
            <li>Always stress test before trading with real money</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
