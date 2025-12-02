"""
CSS Styles for Options Greeks Explorer.

Modern, professional design for academic/educational use.
Inspired by leading financial institutions and top universities.
"""

import streamlit as st

# =============================================================================
# COLOR PALETTE - Academic & Professional
# =============================================================================

COLORS = {
    # Primary - Deep navy (trust, professionalism)
    'primary': '#1a365d',
    'primary_light': '#2c5282',
    'primary_dark': '#0f2942',

    # Secondary - Warm gold (excellence, prestige)
    'secondary': '#c9a227',
    'secondary_light': '#d4b84a',
    'secondary_dark': '#a68921',

    # Accent - Teal (innovation, clarity)
    'accent': '#0d9488',
    'accent_light': '#14b8a6',
    'accent_dark': '#0f766e',

    # Semantic colors
    'success': '#059669',
    'success_bg': '#d1fae5',
    'warning': '#d97706',
    'warning_bg': '#fef3c7',
    'danger': '#dc2626',
    'danger_bg': '#fee2e2',
    'info': '#0284c7',
    'info_bg': '#e0f2fe',

    # Neutrals
    'text_primary': '#1e293b',
    'text_secondary': '#475569',
    'text_muted': '#94a3b8',
    'border': '#e2e8f0',
    'border_light': '#f1f5f9',
    'background': '#ffffff',
    'background_alt': '#f8fafc',
    'background_dark': '#0f172a',
}

# =============================================================================
# CUSTOM CSS STYLES
# =============================================================================

CUSTOM_CSS = """
<style>
    /* ========== Global Styles ========== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ========== Header Styles ========== */
    .main-header {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2.25rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.025em;
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1.1rem;
        margin: 0;
        font-weight: 400;
    }

    .header-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        color: #c9a227;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }

    /* ========== Metric Cards ========== */
    .metric-card {
        background: #ffffff;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
        height: 100%;
    }

    .metric-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transform: translateY(-2px);
    }

    .metric-card .label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .metric-card .value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1e293b;
        line-height: 1.2;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-card .value.profit {
        color: #059669;
    }

    .metric-card .value.loss {
        color: #dc2626;
    }

    .metric-card .value.unlimited {
        background: linear-gradient(135deg, #059669, #0d9488);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-card .value.unlimited-loss {
        background: linear-gradient(135deg, #dc2626, #ea580c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-card .subtext {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }

    /* ========== Position Cards ========== */
    .position-card {
        background: #ffffff;
        padding: 1rem 1.25rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
        transition: all 0.15s ease;
    }

    .position-card:hover {
        border-color: #cbd5e1;
        background: #f8fafc;
    }

    .position-card.long {
        border-left: 4px solid #059669;
    }

    .position-card.short {
        border-left: 4px solid #dc2626;
    }

    .position-card.stock {
        border-left: 4px solid #0284c7;
    }

    .position-card .position-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .position-card .position-type {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }

    .position-card .position-type.long {
        background: #d1fae5;
        color: #047857;
    }

    .position-card .position-type.short {
        background: #fee2e2;
        color: #b91c1c;
    }

    .position-card .position-details {
        font-size: 0.9rem;
        color: #334155;
        font-weight: 500;
    }

    .position-card .position-meta {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.5rem;
    }

    .position-card .position-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }

    /* ========== Net Position Banner ========== */
    .net-position-banner {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .net-position-banner .label {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .net-position-banner .value {
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .net-position-banner.credit .value {
        color: #6ee7b7;
    }

    .net-position-banner.debit .value {
        color: #fca5a5;
    }

    /* ========== Section Headers ========== */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e2e8f0;
    }

    .section-header h3 {
        margin: 0;
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
    }

    .section-header .icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1rem;
    }

    /* ========== Tabs Styling ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #f1f5f9;
        padding: 0.25rem;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.25rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.875rem;
        color: #64748b;
        background: transparent;
    }

    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #1a365d !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    /* ========== Sidebar Styling ========== */
    section[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    .sidebar-header {
        font-size: 0.7rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }

    /* ========== Buttons ========== */
    .stButton > button {
        font-weight: 500;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.15s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
        border: none;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2c5282 0%, #3b6ba5 100%);
        transform: translateY(-1px);
    }

    /* ========== Input Fields ========== */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 8px;
        border-color: #e2e8f0;
    }

    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div:focus {
        border-color: #1a365d;
        box-shadow: 0 0 0 3px rgba(26, 54, 93, 0.1);
    }

    /* ========== Info/Warning/Success Boxes ========== */
    .info-box {
        background: #e0f2fe;
        border: 1px solid #7dd3fc;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }

    .info-box .icon {
        color: #0284c7;
        font-size: 1.1rem;
    }

    .info-box .content {
        color: #0c4a6e;
        font-size: 0.9rem;
    }

    .success-box {
        background: #d1fae5;
        border: 1px solid #6ee7b7;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }

    .warning-box {
        background: #fef3c7;
        border: 1px solid #fcd34d;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }

    /* ========== Footer ========== */
    .app-footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
        color: #94a3b8;
        font-size: 0.85rem;
    }

    .app-footer a {
        color: #1a365d;
        text-decoration: none;
        font-weight: 500;
    }

    /* ========== Charts Container ========== */
    .chart-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }

    .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #f1f5f9;
    }

    .chart-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
    }

    /* ========== Radio Buttons ========== */
    .stRadio > div {
        gap: 0.5rem;
    }

    .stRadio > div > label {
        background: #f1f5f9;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.15s ease;
    }

    .stRadio > div > label:hover {
        background: #e2e8f0;
    }

    .stRadio > div > label[data-checked="true"] {
        background: #1a365d;
        color: white;
    }

    /* ========== Tooltip/Help Text ========== */
    .help-text {
        font-size: 0.75rem;
        color: #94a3b8;
        font-style: italic;
        margin-top: 0.25rem;
    }

    /* ========== Dividers ========== */
    .divider {
        height: 1px;
        background: #e2e8f0;
        margin: 1.5rem 0;
    }

    /* ========== Animation ========== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-in {
        animation: fadeIn 0.3s ease-out;
    }
</style>
"""


def inject_styles() -> None:
    """Inject custom CSS styles into the Streamlit application."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# HTML COMPONENT TEMPLATES
# =============================================================================

def render_header(title: str, subtitle: str, badge: str = None) -> None:
    """Render the main application header."""
    badge_html = f'<span class="header-badge">{badge}</span>' if badge else ''
    st.markdown(f"""
    <div class="main-header animate-in">
        {badge_html}
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def metric_card_html(
    label: str,
    value: str,
    value_class: str = "",
    subtext: str = None
) -> str:
    """Generate HTML for a metric card."""
    subtext_html = f'<div class="subtext">{subtext}</div>' if subtext else ''
    return f"""
    <div class="metric-card animate-in">
        <div class="label">{label}</div>
        <div class="value {value_class}">{value}</div>
        {subtext_html}
    </div>
    """


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
    is_long: bool
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


def stock_position_html(
    quantity: int,
    position_type: str,
    entry_price: float,
    stock_cost: float
) -> str:
    """Generate HTML for a stock position card."""
    is_long = position_type == 'long'
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
        <span class="icon">{icons.get(box_type, 'ℹ️')}</span>
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
