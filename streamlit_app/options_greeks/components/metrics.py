"""
Metrics display components for Options Greeks Explorer.

Modern card-based metrics with professional styling.
"""

import streamlit as st
from config.templates import metric_card_html


def render_metrics_row(
    breakeven_count: int,
    max_profit: float,
    max_loss: float,
    breakeven_points: list[float] | None,
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
                subtext="price levels",
            ),
            unsafe_allow_html=True,
        )

    with col2:
        if max_profit == float("inf"):
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
                subtext=subtext,
            ),
            unsafe_allow_html=True,
        )

    with col3:
        if max_loss == float("-inf"):
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
                subtext=subtext,
            ),
            unsafe_allow_html=True,
        )

    with col4:
        be_text = _format_breakeven_points(breakeven_points)
        st.markdown(
            metric_card_html(
                label="Breakeven Prices", value=be_text, subtext="at expiration"
            ),
            unsafe_allow_html=True,
        )


def _format_breakeven_points(breakeven_points: list[float] | None) -> str:
    """Format breakeven points for display."""
    if not breakeven_points:
        return "N/A"

    be_points = sorted(breakeven_points)

    if len(be_points) == 1:
        return f"${be_points[0]:,.2f}"
    if len(be_points) == 2:
        return f"${be_points[0]:,.0f} / ${be_points[1]:,.0f}"
    return f"{len(be_points)} points"


def render_position_info_banner(
    positions: list, stock_position, default_premium: float
) -> None:
    """
    Render an info banner about the current position.

    Args:
        positions: List of option positions
        stock_position: Stock position or None
        default_premium: Default premium for display
    """
    if not positions and not stock_position:
        st.markdown(
            """
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
                <div style="font-weight: 600; color: #0c4a6e; font-size: 0.9rem;">No Positions</div>
                <div style="color: #0369a1; font-size: 0.8rem;">Add positions using the sidebar to begin analysis</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    elif stock_position and not positions:
        st.markdown(
            f"""
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
                    {stock_position["quantity"]:,} shares {stock_position["position_type"].upper()} @ ${stock_position["entry_price"]:,.2f}
                </div>
                <div style="color: #0369a1; font-size: 0.8rem;">Stock position only - add options to create a strategy</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_chart_controls(
    slider_key: str,
    default_dte: bool = True,
    is_single_leg: bool = False,
    spot_price: float = 100.0,
) -> str:
    """
    Render chart parameter controls with styled toggle buttons.

    Args:
        slider_key: Unique key for the radio button
        default_dte: Whether DTE should be selected by default
        is_single_leg: Whether this is a single-leg strategy (enables Strike option)
        spot_price: Current spot price for Strike display

    Returns:
        Selected parameter type ("DTE", "IV", or "Strike")
    """
    options = ["DTE", "IV", "Strike"] if is_single_leg else ["DTE", "IV"]

    # Initialize session state for this control
    state_key = f"{slider_key}_selected"
    if state_key not in st.session_state:
        st.session_state[state_key] = "DTE" if default_dte else "IV"

    # Create button-based toggle with professional styling
    st.markdown(
        """
    <div style="
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.5rem;
    ">
        <span style="
            font-size: 0.75rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        ">Vary By</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Create columns for buttons
    cols = st.columns(len(options) + 2)  # Extra columns for spacing

    selected = st.session_state[state_key]

    for idx, option in enumerate(options):
        with cols[idx]:
            # Different styling based on option type
            if option == "DTE":
                icon = "📅"
                label = "DTE"
            elif option == "IV":
                icon = "📊"
                label = "IV"
            else:
                icon = "🎯"
                label = "Strike"

            is_selected = selected == option

            # Use button with custom key
            if st.button(
                f"{icon} {label}",
                key=f"{slider_key}_{option}",
                type="primary" if is_selected else "secondary",
                width="stretch",
            ):
                st.session_state[state_key] = option
                st.rerun()

    # Show info panel for selected option
    selected = st.session_state[state_key]

    if selected == "DTE":
        st.markdown(
            """
        <div style="
            background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            color: #1e40af;
            border: 1px solid #bfdbfe;
            margin-top: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        ">
            <span style="font-size: 1rem;">📅</span>
            <div>
                <strong>Days to Expiration</strong>
                <span style="color: #3b82f6; margin-left: 0.5rem;">· Fixed IV: 25%</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    elif selected == "IV":
        st.markdown(
            """
        <div style="
            background: linear-gradient(135deg, #f5f3ff 0%, #faf5ff 100%);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            color: #6b21a8;
            border: 1px solid #ddd6fe;
            margin-top: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        ">
            <span style="font-size: 1rem;">📊</span>
            <div>
                <strong>Implied Volatility</strong>
                <span style="color: #8b5cf6; margin-left: 0.5rem;">· Fixed DTE: 31 days</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:  # Strike
        st.markdown(
            f"""
        <div style="
            background: linear-gradient(135deg, #fef3c7 0%, #fffbeb 100%);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            color: #92400e;
            border: 1px solid #fcd34d;
            margin-top: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        ">
            <span style="font-size: 1rem;">🎯</span>
            <div>
                <strong>Strike Price</strong>
                <span style="color: #d97706; margin-left: 0.5rem;">· Range: ${spot_price * 0.8:.0f} - ${spot_price * 1.2:.0f}</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    return selected


def render_risk_summary(unlimited_profit: bool, unlimited_loss: bool) -> None:
    """
    Render a risk summary box.

    Args:
        unlimited_profit: Whether profit is unlimited
        unlimited_loss: Whether loss is unlimited
    """
    if unlimited_loss:
        st.markdown(
            """
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
        """,
            unsafe_allow_html=True,
        )
    elif unlimited_profit:
        st.markdown(
            """
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
        """,
            unsafe_allow_html=True,
        )
