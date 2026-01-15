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

from config.constants import (
    CHART_HEIGHT_STANDARD,
    CHART_HEIGHT_LARGE,
    PNL_COLORS,
    RISK_METRICS,
    PNL_HISTOGRAM_BINS,
    PNL_KDE_POINTS
)
from backend.simulation import RiskMetrics


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

    st.markdown("### P&L Distribution Analysis")

    # Summary metrics row
    _render_summary_metrics(risk_metrics)

    # Create tabs for different views
    tab1, tab2 = st.tabs(["Histogram", "CDF & Percentiles"])

    with tab1:
        _render_histogram(pnl_values, risk_metrics)

    with tab2:
        _render_cdf(pnl_values, risk_metrics)


def _render_summary_metrics(metrics: RiskMetrics) -> None:
    """Render summary metrics cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        _render_metric_card(
            "Mean P&L",
            metrics.mean_pnl,
            is_currency=True,
            is_positive=metrics.mean_pnl > 0
        )

    with col2:
        _render_metric_card(
            "VaR 95%",
            metrics.var_95,
            is_currency=True,
            is_positive=False  # VaR is typically negative
        )

    with col3:
        _render_metric_card(
            "P(Profit)",
            metrics.prob_profit,
            is_percentage=True,
            is_positive=metrics.prob_profit > 0.5
        )

    with col4:
        _render_metric_card(
            "Std Dev",
            metrics.std_pnl,
            is_currency=True,
            is_positive=None  # Neutral
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)


def _render_metric_card(
    label: str,
    value: float,
    is_currency: bool = False,
    is_percentage: bool = False,
    is_positive: Optional[bool] = None
) -> None:
    """Render a single metric card with styling."""
    # Determine color
    if is_positive is None:
        color = "#475569"  # Neutral gray
    elif is_positive:
        color = "#059669"  # Green
    else:
        color = "#dc2626"  # Red

    # Format value
    if is_percentage:
        formatted = f"{value * 100:.1f}%"
    elif is_currency:
        formatted = f"${value:,.2f}"
    else:
        formatted = f"{value:.3f}"

    # Determine background based on positive/negative
    if is_positive is None:
        bg = "linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%)"
    elif is_positive:
        bg = "linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%)"
    else:
        bg = "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"

    st.markdown(f"""
    <div style="background: {bg}; border-radius: 10px; padding: 1rem; text-align: center;">
        <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 600; margin-bottom: 0.25rem;">{label}</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: {color}; font-family: monospace;">{formatted}</div>
    </div>
    """, unsafe_allow_html=True)


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

    st.plotly_chart(fig, use_container_width=True)

    # Additional stats below chart
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Skewness", f"{metrics.skewness:.3f}")
    with col2:
        st.metric("Kurtosis", f"{metrics.kurtosis:.3f}")
    with col3:
        st.metric("Max Profit", f"${metrics.max_profit:,.2f}")
    with col4:
        st.metric("Max Loss", f"${metrics.max_loss:,.2f}")


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

    # Add percentile markers
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
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
            marker=dict(size=8, color=color),
            text=[f"P{p}: ${pnl_at_p:,.0f}"],
            textposition='top center' if p > 50 else 'bottom center',
            showlegend=False,
            textfont=dict(size=10)
        ))

    # Add horizontal lines for key percentiles
    fig.add_hline(y=0.05, line_dash="dot", line_color="#f59e0b",
                  annotation_text="5%", annotation_position="left")
    fig.add_hline(y=0.50, line_dash="dot", line_color="#6b7280",
                  annotation_text="50%", annotation_position="left")
    fig.add_hline(y=0.95, line_dash="dot", line_color="#10b981",
                  annotation_text="95%", annotation_position="left")

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
        yaxis=dict(tickformat='.0%', range=[0, 1]),
        hovermode='x unified',
        margin=dict(l=60, r=40, t=60, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Percentile table
    st.markdown("#### Key Percentiles")
    percentile_data = []
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        idx = int((p / 100) * (n - 1))
        percentile_data.append({
            "Percentile": f"{p}%",
            "P&L": f"${sorted_pnl[idx]:,.2f}"
        })

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Tail Risk (Left)**")
        for item in percentile_data[:3]:
            st.markdown(f"- {item['Percentile']}: {item['P&L']}")
    with col2:
        st.markdown("**Middle Range**")
        for item in percentile_data[3:6]:
            st.markdown(f"- {item['Percentile']}: {item['P&L']}")
    with col3:
        st.markdown("**Upside (Right)**")
        for item in percentile_data[6:]:
            st.markdown(f"- {item['Percentile']}: {item['P&L']}")


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

    # Primary metrics row
    st.markdown("#### Performance & Risk")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 12px; padding: 1.25rem; border-left: 4px solid #3b82f6;">
            <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #1d4ed8; font-weight: 600;">Expected P&L</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b; font-family: monospace; margin-top: 0.25rem;">${metrics.mean_pnl:,.2f}</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">Mean outcome</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 12px; padding: 1.25rem; border-left: 4px solid #f59e0b;">
            <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #b45309; font-weight: 600;">Value at Risk 95%</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b; font-family: monospace; margin-top: 0.25rem;">${metrics.var_95:,.2f}</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">5% worst case</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border-radius: 12px; padding: 1.25rem; border-left: 4px solid #ef4444;">
            <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #b91c1c; font-weight: 600;">CVaR 95%</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b; font-family: monospace; margin-top: 0.25rem;">${metrics.cvar_95:,.2f}</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">Expected shortfall</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        prob_color = "#059669" if metrics.prob_profit > 0.5 else "#dc2626"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {'#ecfdf5' if metrics.prob_profit > 0.5 else '#fef2f2'} 0%, {'#d1fae5' if metrics.prob_profit > 0.5 else '#fee2e2'} 100%); border-radius: 12px; padding: 1.25rem; border-left: 4px solid {prob_color};">
            <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: {prob_color}; font-weight: 600;">Probability of Profit</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {prob_color}; font-family: monospace; margin-top: 0.25rem;">{metrics.prob_profit:.1%}</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">P(P&L > 0)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Secondary metrics
    st.markdown("#### Distribution Characteristics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Standard Deviation", f"${metrics.std_pnl:,.2f}")
    with col2:
        # Risk-adjusted return
        if metrics.std_pnl > 0:
            sharpe_like = metrics.mean_pnl / metrics.std_pnl
            st.metric("Return/Risk Ratio", f"{sharpe_like:.3f}")
        else:
            st.metric("Return/Risk Ratio", "N/A")
    with col3:
        st.metric("Skewness", f"{metrics.skewness:.3f}",
                  delta="Right-skewed" if metrics.skewness > 0 else "Left-skewed",
                  delta_color="normal" if metrics.skewness > 0 else "inverse")
    with col4:
        st.metric("Excess Kurtosis", f"{metrics.kurtosis:.3f}",
                  delta="Fat tails" if metrics.kurtosis > 0 else "Thin tails",
                  delta_color="inverse" if metrics.kurtosis > 3 else "normal")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Extreme outcomes
    st.markdown("#### Extreme Outcomes")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Maximum Profit", f"${metrics.max_profit:,.2f}")
    with col2:
        st.metric("Maximum Loss", f"${metrics.max_loss:,.2f}")
    with col3:
        st.metric("VaR 99%", f"${metrics.var_99:,.2f}")
    with col4:
        st.metric("CVaR 99%", f"${metrics.cvar_99:,.2f}")

    # Risk interpretation
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
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
