"""
CSS Styles for Monte Carlo Simulation Explorer.

Modern, professional design for academic/educational use.
Adapted from Options Greeks Explorer.
"""

import streamlit as st

# =============================================================================
# COLOR PALETTE - Academic & Professional
# =============================================================================

COLORS = {
    # Primary - Deep teal (innovation, data science)
    'primary': '#0d9488',
    'primary_light': '#14b8a6',
    'primary_dark': '#0f766e',

    # Secondary - Deep navy (trust, professionalism)
    'secondary': '#1a365d',
    'secondary_light': '#2c5282',
    'secondary_dark': '#0f2942',

    # Accent - Warm amber (energy, insights)
    'accent': '#d97706',
    'accent_light': '#f59e0b',
    'accent_dark': '#b45309',

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
        background-color: #f1f5f9;
    }

    .main .block-container {
        background-color: #f1f5f9;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ========== Header Styles ========== */
    .main-header {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 50%, #1a365d 100%);
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
        color: #fbbf24;
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

    .metric-card .value.positive {
        color: #059669;
    }

    .metric-card .value.negative {
        color: #dc2626;
    }

    .metric-card .subtext {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }

    /* ========== Model Cards ========== */
    .model-card {
        background: #ffffff;
        padding: 1rem 1.25rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
        transition: all 0.15s ease;
    }

    .model-card:hover {
        border-color: #0d9488;
        background: #f0fdfa;
    }

    .model-card.selected {
        border-color: #0d9488;
        border-width: 2px;
        background: #f0fdfa;
    }

    .model-card .model-name {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1e293b;
    }

    .model-card .model-equation {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* ========== Parameter Groups ========== */
    .param-group {
        background: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin: 0.75rem 0;
    }

    .param-group-header {
        font-size: 0.75rem;
        font-weight: 700;
        color: #0d9488;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
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
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
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
        color: #0d9488 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    /* ========== Sidebar Styling ========== */
    section[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
        min-width: 380px !important;
        width: 380px !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        width: 380px !important;
    }

    .sidebar-header {
        font-size: 0.7rem;
        font-weight: 700;
        color: #0d9488;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }

    /* ========== Buttons ========== */
    .stButton > button {
        font-weight: 600;
        font-size: 0.875rem;
        border-radius: 10px;
        padding: 0.625rem 1.25rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.01em;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.06);
        border: 1px solid transparent;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15), 0 2px 4px rgba(0, 0, 0, 0.08);
    }

    /* Primary buttons - Teal gradient */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
        color: #ffffff;
        border: none;
        box-shadow: 0 2px 8px rgba(13, 148, 136, 0.25), 0 1px 3px rgba(13, 148, 136, 0.15);
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0f766e 0%, #115e59 100%);
        box-shadow: 0 6px 20px rgba(13, 148, 136, 0.35), 0 3px 8px rgba(13, 148, 136, 0.2);
    }

    /* Secondary buttons */
    .stButton > button[kind="secondary"],
    .stButton > button:not([kind="primary"]) {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        color: #1e293b;
        border: 1px solid #e2e8f0;
    }

    .stButton > button[kind="secondary"]:hover,
    .stButton > button:not([kind="primary"]):hover {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-color: #cbd5e1;
        color: #0d9488;
    }

    /* Sidebar buttons styling */
    section[data-testid="stSidebar"] .stButton > button {
        font-size: 0.8rem !important;
        padding: 0.5rem 0.75rem !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(13, 148, 136, 0.25) !important;
    }

    /* ========== Input Fields ========== */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 8px;
        border-color: #e2e8f0;
    }

    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div:focus {
        border-color: #0d9488;
        box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.1);
    }

    /* ========== Slider Styling ========== */
    .stSlider > div > div > div > div {
        background-color: #0d9488;
    }

    /* ========== Info Boxes ========== */
    .info-box {
        background: #f0fdfa;
        border: 1px solid #99f6e4;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }

    .info-box .content {
        color: #0f766e;
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

    /* ========== Statistics Table ========== */
    .stats-table {
        width: 100%;
        border-collapse: collapse;
    }

    .stats-table th {
        background: #f8fafc;
        padding: 0.75rem 1rem;
        text-align: left;
        font-weight: 600;
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 2px solid #e2e8f0;
    }

    .stats-table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #f1f5f9;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
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
        color: #0d9488;
        text-decoration: none;
        font-weight: 500;
    }

    /* ========== Animation ========== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-in {
        animation: fadeIn 0.3s ease-out;
    }

    /* ========== Expander Styling ========== */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1e293b;
    }

    .streamlit-expanderContent {
        background: #f8fafc;
        border-radius: 0 0 10px 10px;
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
    icons = {"info": "i", "success": "check", "warning": "!"}
    return f"""
    <div class="{box_type}-box">
        <span class="content">{message}</span>
    </div>
    """


def param_group_html(title: str, icon: str = None) -> str:
    """Generate HTML for a parameter group header."""
    icon_html = f"<span>{icon}</span>" if icon else ""
    return f"""
    <div class="param-group-header">
        {icon_html}
        {title}
    </div>
    """


def footer_html() -> str:
    """Generate HTML for the application footer."""
    return """
    <div class="app-footer">
        <p>Monte Carlo Simulation Explorer</p>
        <p>Educational tool for stochastic processes and volatility modeling</p>
    </div>
    """
