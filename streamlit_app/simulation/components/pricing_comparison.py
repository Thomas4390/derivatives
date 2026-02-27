"""
Pricing Comparison Component - Compare MC vs Analytical/FFT prices.

Provides:
- Side-by-side comparison of pricing methods
- Error analysis
- Convergence visualization
"""

import streamlit as st
import numpy as np
from typing import Optional

from services.pricing_service import (
    PricingComparison,
    get_available_pricing_methods,
)


def render_pricing_comparison(
    comparison: PricingComparison,
    show_greeks: bool = True
):
    """
    Render pricing comparison results.

    Args:
        comparison: PricingComparison result object
        show_greeks: Whether to show Greeks (if available)
    """
    st.subheader("💰 Pricing Comparison")

    # Available methods indicator
    methods_str = " | ".join([m.upper() for m in comparison.available_methods])
    st.caption(f"Available methods: {methods_str}")

    # Main price comparison
    _render_price_table(comparison)

    # Error analysis
    if comparison.mc_vs_analytical_error is not None or comparison.mc_vs_fft_error is not None:
        _render_error_analysis(comparison)

    # Greeks (if available)
    if show_greeks and comparison.analytical_delta is not None:
        _render_greeks(comparison)


def _render_price_table(comparison: PricingComparison):
    """Render price comparison table."""
    cols = st.columns(3)

    # Monte Carlo
    with cols[0]:
        st.metric(
            label="Monte Carlo",
            value=f"${comparison.mc_price:.2f}",
            help=f"N = {comparison.mc_n_paths:,} paths"
        )
        ci_low, ci_high = comparison.mc_confidence_interval
        st.caption(f"95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
        st.caption(f"Std Error: {comparison.mc_std_error:.2f}")

    # Analytical (if available)
    with cols[1]:
        if comparison.analytical_price is not None:
            error = comparison.mc_vs_analytical_error
            st.metric(
                label="Black-Scholes",
                value=f"${comparison.analytical_price:.2f}",
                delta=f"{error:+.2f} (MC - BS)" if error else None,
                delta_color="off"
            )
            pct_error = abs(error / comparison.analytical_price) * 100 if comparison.analytical_price > 0 else 0
            st.caption(f"Error: {pct_error:.2f}%")
        else:
            st.metric(
                label="Black-Scholes",
                value="N/A",
                help="Only available for GBM model"
            )
            st.caption("Not available for this model")

    # FFT (if available)
    with cols[2]:
        if comparison.fft_price is not None:
            error = comparison.mc_vs_fft_error
            st.metric(
                label="FFT (Carr-Madan)",
                value=f"${comparison.fft_price:.2f}",
                delta=f"{error:+.2f} (MC - FFT)" if error else None,
                delta_color="off"
            )
            pct_error = abs(error / comparison.fft_price) * 100 if comparison.fft_price > 0 else 0
            st.caption(f"Error: {pct_error:.2f}%")
        else:
            st.metric(
                label="FFT",
                value="N/A",
                help="Only for models with characteristic function"
            )
            st.caption("Not available for GARCH models")


def _render_error_analysis(comparison: PricingComparison):
    """Render error analysis section."""
    with st.expander("📊 Error Analysis", expanded=False):
        st.markdown("### Monte Carlo Convergence")

        # Reference price (prefer analytical, then FFT)
        ref_price = comparison.analytical_price or comparison.fft_price
        ref_name = "BS" if comparison.analytical_price else "FFT"

        if ref_price is not None:
            error = comparison.mc_price - ref_price
            pct_error = abs(error / ref_price) * 100

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Absolute Error (MC - {ref_name}):**")
                st.code(f"{error:+.2f}")

            with col2:
                st.markdown("**Percentage Error:**")
                st.code(f"{pct_error:.2f}%")

            # Standard error bands
            st.markdown("**Standard Error Bands:**")
            se = comparison.mc_std_error
            st.markdown(f"""
            | Band | Range |
            |------|-------|
            | 1σ | [{comparison.mc_price - se:.2f}, {comparison.mc_price + se:.2f}] |
            | 2σ | [{comparison.mc_price - 2*se:.2f}, {comparison.mc_price + 2*se:.2f}] |
            | 3σ | [{comparison.mc_price - 3*se:.2f}, {comparison.mc_price + 3*se:.2f}] |
            """)

            # Interpretation
            if pct_error < 0.1:
                st.success("✓ Excellent convergence (< 0.1% error)")
            elif pct_error < 0.5:
                st.info("✓ Good convergence (< 0.5% error)")
            elif pct_error < 1.0:
                st.warning("⚠️ Moderate convergence. Consider more paths.")
            else:
                st.error("⚠️ Poor convergence. Increase number of paths.")


def _render_greeks(comparison: PricingComparison):
    """Render Greeks section."""
    with st.expander("📐 Greeks (Black-Scholes)", expanded=False):
        cols = st.columns(5)

        with cols[0]:
            st.metric("Delta (Δ)", f"{comparison.analytical_delta:.2f}")

        with cols[1]:
            st.metric("Gamma (Γ)", f"{comparison.analytical_gamma:.2f}")

        with cols[2]:
            if comparison.analytical_vega is not None:
                st.metric("Vega (ν)", f"{comparison.analytical_vega:.2f}")

        with cols[3]:
            if comparison.analytical_theta is not None:
                st.metric("Theta (Θ)", f"{comparison.analytical_theta:.2f}")

        with cols[4]:
            st.caption("Greeks from BS formula")


def render_pricing_methods_info(model_key: str):
    """Render information about available pricing methods for a model."""
    methods = get_available_pricing_methods(model_key)

    st.markdown("### Available Pricing Methods")

    method_info = {
        "analytical": {
            "name": "Black-Scholes (Analytical)",
            "description": "Closed-form solution. Exact and fast.",
            "icon": "🎯"
        },
        "fft": {
            "name": "FFT (Carr-Madan)",
            "description": "Uses characteristic function. Very accurate.",
            "icon": "📊"
        },
        "monte_carlo": {
            "name": "Monte Carlo",
            "description": "Simulation-based. Works for all models.",
            "icon": "🎲"
        }
    }

    for method in methods:
        info = method_info.get(method, {})
        if info:
            st.markdown(f"**{info['icon']} {info['name']}**")
            st.caption(info['description'])


def render_convergence_guide():
    """Render guide for MC convergence."""
    with st.expander("📚 Monte Carlo Convergence Guide", expanded=False):
        st.markdown("""
        ### Understanding MC Convergence

        **Standard Error decreases as:**
        - `SE ∝ 1/√N` where N = number of paths

        **To halve the error:** Quadruple the paths (4× paths → 2× better)

        **Typical path counts:**
        | Accuracy | Paths | Use Case |
        |----------|-------|----------|
        | Quick estimate | 1,000 | Interactive exploration |
        | Good | 10,000 | General pricing |
        | High | 100,000 | Production pricing |
        | Very High | 1,000,000+ | Risk calculations |

        **Variance Reduction Techniques:**
        - Antithetic variates (default: ON)
        - Control variates
        - Importance sampling
        """)


def render_strike_comparison(
    strikes: np.ndarray,
    mc_prices: np.ndarray,
    mc_errors: np.ndarray,
    analytical_prices: Optional[np.ndarray] = None,
    fft_prices: Optional[np.ndarray] = None,
    spot: float = 100.0
):
    """
    Render comparison across multiple strikes.

    Args:
        strikes: Array of strike prices
        mc_prices: MC prices at each strike
        mc_errors: MC standard errors
        analytical_prices: BS prices (optional)
        fft_prices: FFT prices (optional)
        spot: Current spot price
    """
    import pandas as pd

    st.subheader("📈 Price by Strike")

    # Build dataframe
    data = {
        "Strike": strikes,
        "Moneyness": strikes / spot,
        "MC Price": mc_prices,
        "MC Std Err": mc_errors,
    }

    if analytical_prices is not None:
        data["BS Price"] = analytical_prices
        data["MC - BS"] = mc_prices - analytical_prices

    if fft_prices is not None:
        data["FFT Price"] = fft_prices
        data["MC - FFT"] = mc_prices - fft_prices

    df = pd.DataFrame(data)

    # Format columns
    st.dataframe(
        df.style.format({
            "Strike": "${:.2f}",
            "Moneyness": "{:.2f}",
            "MC Price": "${:.2f}",
            "MC Std Err": "{:.2f}",
            "BS Price": "${:.2f}" if analytical_prices is not None else None,
            "MC - BS": "{:+.2f}" if analytical_prices is not None else None,
            "FFT Price": "${:.2f}" if fft_prices is not None else None,
            "MC - FFT": "{:+.2f}" if fft_prices is not None else None,
        }),
        width="stretch",
        hide_index=True
    )
