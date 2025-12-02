"""
Metrics display components for Options Greeks Explorer.

Modern card-based metrics with professional styling.
"""

import streamlit as st
from config.styles import metric_card_html


def render_metrics_row(
    breakeven_count: int,
    max_profit: float,
    max_loss: float,
    breakeven_points: list[float] | None
) -> None:
    """
    Render the main metrics row with breakeven, max profit/loss info.

    Args:
        breakeven_count: Number of breakeven points
        max_profit: Maximum profit value (inf for unlimited)
        max_loss: Maximum loss value (-inf for unlimited)
        breakeven_points: List of breakeven price points
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            metric_card_html(
                label="Breakeven Points",
                value=str(breakeven_count),
                subtext="price levels"
            ),
            unsafe_allow_html=True
        )

    with col2:
        if max_profit == float('inf'):
            profit_text = "Unlimited"
            value_class = "unlimited"
            subtext = "theoretically infinite"
        else:
            profit_text = f"${max_profit:,.2f}"
            value_class = "profit"
            subtext = "maximum gain"

        st.markdown(
            metric_card_html(
                label="Maximum Profit",
                value=profit_text,
                value_class=value_class,
                subtext=subtext
            ),
            unsafe_allow_html=True
        )

    with col3:
        if max_loss == float('-inf'):
            loss_text = "Unlimited"
            value_class = "unlimited-loss"
            subtext = "theoretically infinite"
        else:
            loss_text = f"${abs(max_loss):,.2f}"
            value_class = "loss"
            subtext = "maximum risk"

        st.markdown(
            metric_card_html(
                label="Maximum Loss",
                value=loss_text,
                value_class=value_class,
                subtext=subtext
            ),
            unsafe_allow_html=True
        )

    with col4:
        be_text = _format_breakeven_points(breakeven_points)
        st.markdown(
            metric_card_html(
                label="Breakeven Prices",
                value=be_text,
                subtext="at expiration"
            ),
            unsafe_allow_html=True
        )


def _format_breakeven_points(breakeven_points: list[float] | None) -> str:
    """Format breakeven points for display."""
    if not breakeven_points:
        return "N/A"

    be_points = sorted(breakeven_points)

    if len(be_points) == 1:
        return f"${be_points[0]:,.2f}"
    elif len(be_points) == 2:
        return f"${be_points[0]:,.0f} / ${be_points[1]:,.0f}"
    else:
        return f"{len(be_points)} points"


def render_position_info_banner(
    positions: list,
    stock_position,
    default_premium: float
) -> None:
    """
    Render an info banner about the current position.

    Args:
        positions: List of option positions
        stock_position: Stock position or None
        default_premium: Default premium for display
    """
    if not positions and not stock_position:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%);
            border: 1px solid #7dd3fc;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        ">
            <div style="font-size: 1.25rem;">ℹ️</div>
            <div>
                <div style="font-weight: 600; color: #0c4a6e; font-size: 0.9rem;">Default Position: Long Call ATM</div>
                <div style="color: #0369a1; font-size: 0.8rem;">Add your own positions using the sidebar to analyze custom strategies</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif stock_position and not positions:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%);
            border: 1px solid #7dd3fc;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        ">
            <div style="font-size: 1.25rem;">📈</div>
            <div>
                <div style="font-weight: 600; color: #0c4a6e; font-size: 0.9rem;">
                    {stock_position.quantity:,} shares {stock_position.position_type.upper()} @ ${stock_position.entry_price:,.2f}
                </div>
                <div style="color: #0369a1; font-size: 0.8rem;">Stock position only - add options to create a strategy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_chart_controls(
    slider_key: str,
    default_dte: bool = True
) -> str:
    """
    Render chart parameter controls.

    Args:
        slider_key: Unique key for the radio button
        default_dte: Whether DTE should be selected by default

    Returns:
        Selected parameter type ("DTE" or "IV")
    """
    col1, col2 = st.columns([1, 3])

    with col1:
        slider_type = st.radio(
            "Vary by",
            ["DTE", "IV"],
            index=0 if default_dte else 1,
            key=slider_key,
            horizontal=True
        )

    with col2:
        if slider_type == "DTE":
            st.markdown("""
            <div style="
                background: #f1f5f9;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-size: 0.8rem;
                color: #475569;
            ">
                <strong>Days to Expiration</strong> · Fixed IV: 25%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background: #f1f5f9;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-size: 0.8rem;
                color: #475569;
            ">
                <strong>Implied Volatility</strong> · Fixed DTE: 31 days
            </div>
            """, unsafe_allow_html=True)

    return slider_type


def render_risk_summary(
    unlimited_profit: bool,
    unlimited_loss: bool
) -> None:
    """
    Render a risk summary box.

    Args:
        unlimited_profit: Whether profit is unlimited
        unlimited_loss: Whether loss is unlimited
    """
    if unlimited_loss:
        st.markdown("""
        <div style="
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-left: 4px solid #dc2626;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin: 1rem 0;
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                <span style="font-size: 1rem;">⚠️</span>
                <strong style="color: #991b1b;">Unlimited Risk Position</strong>
            </div>
            <div style="color: #b91c1c; font-size: 0.85rem;">
                This position has theoretically unlimited loss potential. Consider adding protective positions.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif unlimited_profit:
        st.markdown("""
        <div style="
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-left: 4px solid #059669;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin: 1rem 0;
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                <span style="font-size: 1rem;">✓</span>
                <strong style="color: #166534;">Defined Risk Position</strong>
            </div>
            <div style="color: #15803d; font-size: 0.85rem;">
                This position has unlimited profit potential with limited, defined risk.
            </div>
        </div>
        """, unsafe_allow_html=True)
