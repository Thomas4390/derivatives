"""
HTML component templates for Options Greeks Explorer.

Reusable HTML-generating functions for headers, metric cards,
position cards, info boxes, and footer elements.
"""

import streamlit as st


# =============================================================================
# HEADER
# =============================================================================


def render_header(title: str, subtitle: str, badge: str = None) -> None:
    """Render the main application header."""
    badge_html = f'<span class="header-badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""
    <div class="main-header animate-in">
        {badge_html}
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# =============================================================================
# METRIC CARDS
# =============================================================================


def metric_card_html(
    label: str, value: str, value_class: str = "", subtext: str = None
) -> str:
    """Generate HTML for a metric card."""
    subtext_html = f'<div class="subtext">{subtext}</div>' if subtext else ""
    return f"""
    <div class="metric-card animate-in">
        <div class="label">{label}</div>
        <div class="value {value_class}">{value}</div>
        {subtext_html}
    </div>
    """


# =============================================================================
# POSITION CARDS
# =============================================================================


def net_position_card_html(position_type: str, amount: float) -> str:
    """Generate HTML for the net position banner."""
    if amount > 0:
        type_class = "credit"
        label = "Net Credit"
        formatted = f"+${amount:,.2f}"
    elif amount < 0:
        type_class = "debit"
        label = "Net Debit"
        formatted = f"-${abs(amount):,.2f}"
    else:
        type_class = ""
        label = "Net Position"
        formatted = "$0.00"

    return f"""
    <div class="net-position-banner {type_class}">
        <span class="label">{label}</span>
        <span class="value">{formatted}</span>
    </div>
    """


def position_item_html(
    index: int,
    quantity: int,
    position_type: str,
    option_type: str,
    strike: float,
    premium: float,
    total_amount: float,
    shares_controlled: int,
    is_long: bool,
) -> str:
    """Generate HTML for an option position card."""
    pos_class = "long" if is_long else "short"
    type_label = "LONG" if is_long else "SHORT"
    amount_label = "Debit" if is_long else "Credit"

    return f"""
    <div class="position-card {pos_class}">
        <div class="position-header">
            <span class="position-details">{quantity}x {option_type.upper()} @ ${strike:,.2f}</span>
            <span class="position-type {pos_class}">{type_label}</span>
        </div>
        <div class="position-meta">
            Premium: <span class="position-value">${premium:,.2f}</span> per contract
            &nbsp;·&nbsp;
            {amount_label}: <span class="position-value">${total_amount:,.2f}</span>
        </div>
    </div>
    """


def exotic_position_item_html(
    index: int,
    quantity: int,
    position_type: str,
    option_type: str,
    strike: float,
    premium: float,
    total_amount: float,
    shares_controlled: int,
    is_long: bool,
    exotic_type_name: str,
) -> str:
    """Generate HTML for an exotic option position card."""
    type_label = "LONG" if is_long else "SHORT"
    amount_label = "Debit" if is_long else "Credit"

    return f"""
    <div class="position-card exotic">
        <div class="position-header">
            <span class="position-details">{quantity}x {option_type.upper()} @ ${strike:,.2f}</span>
            <span class="position-type exotic-badge">{type_label}</span>
        </div>
        <div class="position-meta">
            <span class="exotic-type-label">{exotic_type_name}</span>
            &nbsp;·&nbsp;
            Premium: <span class="position-value">${premium:,.2f}</span> per contract
            &nbsp;·&nbsp;
            {amount_label}: <span class="position-value">${total_amount:,.2f}</span>
        </div>
    </div>
    """


def stock_position_html(
    quantity: int, position_type: str, entry_price: float, stock_cost: float
) -> str:
    """Generate HTML for a stock position card."""
    is_long = position_type == "long"
    pos_class = "long" if is_long else "short"
    type_label = "LONG" if is_long else "SHORT"
    amount_label = "Cost" if is_long else "Credit"

    return f"""
    <div class="position-card stock">
        <div class="position-header">
            <span class="position-details">{quantity:,} shares @ ${entry_price:,.2f}</span>
            <span class="position-type {pos_class}">{type_label} STOCK</span>
        </div>
        <div class="position-meta">
            {amount_label}: <span class="position-value">${stock_cost:,.2f}</span>
        </div>
    </div>
    """


# =============================================================================
# SECTION ELEMENTS
# =============================================================================


def section_header_html(icon: str, title: str) -> str:
    """Generate HTML for a section header."""
    return f"""
    <div class="section-header">
        <div class="icon">{icon}</div>
        <h3>{title}</h3>
    </div>
    """


def info_box_html(message: str, box_type: str = "info") -> str:
    """Generate HTML for an info/warning/success box."""
    icons = {"info": "ℹ️", "success": "✓", "warning": "⚠️"}
    return f"""
    <div class="{box_type}-box">
        <span class="icon">{icons.get(box_type, "ℹ️")}</span>
        <span class="content">{message}</span>
    </div>
    """


def footer_html() -> str:
    """Generate HTML for the application footer."""
    return """
    <div class="app-footer">
        <p>Options Greeks Explorer · Black-Scholes Model</p>
        <p>Educational tool for quantitative finance</p>
    </div>
    """
