"""
Pricing Comparison Chart - Visualize MC vs Analytical/FFT pricing.

Provides:
- Price comparison bar charts
- Multi-strike option price curves
- Convergence analysis
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List

from services.pricing_service import PricingComparison


def render_pricing_comparison_chart(
    comparison: PricingComparison,
    height: int = 400,
):
    """
    Render pricing comparison chart.

    Args:
        comparison: PricingComparison result object
        height: Chart height in pixels
    """
    methods = comparison.available_methods
    n_methods = len(methods)

    # Create figure
    fig = go.Figure()

    # Colors for each method
    colors = {
        "monte_carlo": "#1f77b4",
        "analytical": "#2ca02c",
        "fft": "#ff7f0e",
    }

    # Prices
    prices = []
    labels = []
    bar_colors = []
    errors = []

    for method in methods:
        if method == "monte_carlo":
            prices.append(comparison.mc_price)
            labels.append("Monte Carlo")
            bar_colors.append(colors["monte_carlo"])
            errors.append(comparison.mc_std_error * 1.96)  # 95% CI
        elif method == "analytical":
            prices.append(comparison.analytical_price)
            labels.append("Black-Scholes")
            bar_colors.append(colors["analytical"])
            errors.append(0)
        elif method == "fft":
            prices.append(comparison.fft_price)
            labels.append("FFT (Carr-Madan)")
            bar_colors.append(colors["fft"])
            errors.append(0)

    # Bar chart
    fig.add_trace(go.Bar(
        x=labels,
        y=prices,
        marker_color=bar_colors,
        error_y=dict(
            type='data',
            array=errors,
            visible=True,
            thickness=2,
            width=10,
        ),
        text=[f"${p:.4f}" for p in prices],
        textposition='outside',
    ))

    # Layout
    fig.update_layout(
        height=height,
        title=f"<b>Option Price Comparison</b><br><sup>{comparison.model.upper()} Model</sup>",
        yaxis_title="Option Price ($)",
        showlegend=False,
        bargap=0.3,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_multi_strike_comparison(
    strikes: np.ndarray,
    mc_prices: np.ndarray,
    mc_errors: np.ndarray,
    analytical_prices: Optional[np.ndarray] = None,
    fft_prices: Optional[np.ndarray] = None,
    spot: float = 100.0,
    height: int = 450,
):
    """
    Render multi-strike price comparison chart.

    Args:
        strikes: Array of strike prices
        mc_prices: MC prices for each strike
        mc_errors: MC standard errors
        analytical_prices: BS prices (optional)
        fft_prices: FFT prices (optional)
        spot: Current spot price
        height: Chart height
    """
    fig = go.Figure()

    # Monte Carlo with error bands
    fig.add_trace(go.Scatter(
        x=strikes,
        y=mc_prices + 1.96 * mc_errors,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=strikes,
        y=mc_prices - 1.96 * mc_errors,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.2)',
        showlegend=False,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=strikes,
        y=mc_prices,
        mode='lines+markers',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6),
        name='Monte Carlo',
    ))

    # Analytical (if available)
    if analytical_prices is not None:
        fig.add_trace(go.Scatter(
            x=strikes,
            y=analytical_prices,
            mode='lines+markers',
            line=dict(color='#2ca02c', width=2, dash='dash'),
            marker=dict(size=6, symbol='square'),
            name='Black-Scholes',
        ))

    # FFT (if available)
    if fft_prices is not None:
        fig.add_trace(go.Scatter(
            x=strikes,
            y=fft_prices,
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=2, dash='dot'),
            marker=dict(size=6, symbol='diamond'),
            name='FFT',
        ))

    # ATM line
    fig.add_vline(
        x=spot,
        line_dash="dash",
        line_color="gray",
        annotation_text="ATM",
        annotation_position="top"
    )

    fig.update_layout(
        height=height,
        title="<b>Option Prices Across Strikes</b>",
        xaxis_title="Strike Price ($)",
        yaxis_title="Option Price ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_pricing_error_chart(
    strikes: np.ndarray,
    mc_prices: np.ndarray,
    reference_prices: np.ndarray,
    reference_name: str = "Reference",
    height: int = 350,
):
    """
    Render pricing error chart (MC vs reference).

    Args:
        strikes: Array of strike prices
        mc_prices: MC prices
        reference_prices: Reference prices (BS or FFT)
        reference_name: Name of reference method
        height: Chart height
    """
    errors = mc_prices - reference_prices
    pct_errors = (errors / reference_prices) * 100

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Absolute Error ($)", "Percentage Error (%)")
    )

    # Absolute error
    fig.add_trace(
        go.Bar(
            x=strikes,
            y=errors,
            marker_color=np.where(errors >= 0, '#2ca02c', '#d62728'),
            name='Error',
            showlegend=False,
        ),
        row=1, col=1
    )

    # Percentage error
    fig.add_trace(
        go.Bar(
            x=strikes,
            y=pct_errors,
            marker_color=np.where(pct_errors >= 0, '#2ca02c', '#d62728'),
            name='Error %',
            showlegend=False,
        ),
        row=1, col=2
    )

    # Zero lines
    fig.add_hline(y=0, line_dash="solid", line_color="black", row=1, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="black", row=1, col=2)

    fig.update_layout(
        height=height,
        title=f"<b>MC vs {reference_name} Pricing Error</b>",
    )

    fig.update_xaxes(title_text="Strike ($)", row=1, col=1)
    fig.update_xaxes(title_text="Strike ($)", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)


def render_convergence_chart(
    path_counts: List[int],
    mc_prices: List[float],
    mc_errors: List[float],
    reference_price: Optional[float] = None,
    reference_name: str = "Reference",
    height: int = 400,
):
    """
    Render MC convergence chart.

    Args:
        path_counts: List of path counts
        mc_prices: MC prices at each path count
        mc_errors: MC standard errors
        reference_price: Reference price (BS/FFT)
        reference_name: Name of reference method
        height: Chart height
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Price Convergence", "Standard Error Decay")
    )

    mc_prices = np.array(mc_prices)
    mc_errors = np.array(mc_errors)
    path_counts = np.array(path_counts)

    # Price convergence with CI
    fig.add_trace(
        go.Scatter(
            x=path_counts,
            y=mc_prices + 1.96 * mc_errors,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=path_counts,
            y=mc_prices - 1.96 * mc_errors,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.2)',
            name='95% CI',
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=path_counts,
            y=mc_prices,
            mode='lines+markers',
            line=dict(color='#1f77b4', width=2),
            name='MC Price',
        ),
        row=1, col=1
    )

    # Reference price line
    if reference_price is not None:
        fig.add_hline(
            y=reference_price,
            line_dash="dash",
            line_color="red",
            annotation_text=reference_name,
            row=1, col=1
        )

    # Standard error decay
    fig.add_trace(
        go.Scatter(
            x=path_counts,
            y=mc_errors,
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=2),
            name='Std Error',
        ),
        row=1, col=2
    )

    # Theoretical 1/sqrt(n) decay
    theoretical_errors = mc_errors[0] * np.sqrt(path_counts[0] / path_counts)
    fig.add_trace(
        go.Scatter(
            x=path_counts,
            y=theoretical_errors,
            mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            name='1/√n',
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=height,
        title="<b>Monte Carlo Convergence Analysis</b>",
    )

    fig.update_xaxes(type='log', title_text="Number of Paths", row=1, col=1)
    fig.update_xaxes(type='log', title_text="Number of Paths", row=1, col=2)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(type='log', title_text="Standard Error", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)


# Alias for backward compatibility
render_single_price_comparison = render_pricing_comparison_chart
