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
    "primary": "#1a365d",
    "primary_light": "#2c5282",
    "primary_dark": "#0f2942",
    # Secondary - Warm gold (excellence, prestige)
    "secondary": "#c9a227",
    "secondary_light": "#d4b84a",
    "secondary_dark": "#a68921",
    # Accent - Teal (innovation, clarity)
    "accent": "#0d9488",
    "accent_light": "#14b8a6",
    "accent_dark": "#0f766e",
    # Semantic colors
    "success": "#059669",
    "success_bg": "#d1fae5",
    "warning": "#d97706",
    "warning_bg": "#fef3c7",
    "danger": "#dc2626",
    "danger_bg": "#fee2e2",
    "info": "#0284c7",
    "info_bg": "#e0f2fe",
    # Neutrals
    "text_primary": "#1e293b",
    "text_secondary": "#475569",
    "text_muted": "#94a3b8",
    "border": "#e2e8f0",
    "border_light": "#f1f5f9",
    "background": "#ffffff",
    "background_alt": "#f8fafc",
    "background_dark": "#0f172a",
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

    /* Main content area background */
    .main .block-container {
        background-color: #f1f5f9;
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

    .position-card.exotic {
        border-left: 4px solid #8b5cf6;
        background: linear-gradient(135deg, #faf5ff 0%, #f5f3ff 100%);
    }

    .position-card.exotic:hover {
        border-color: #c4b5fd;
        background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
    }

    .position-card .position-type.exotic-badge {
        background: #ddd6fe;
        color: #6d28d9;
    }

    .position-card .exotic-type-label {
        font-size: 0.7rem;
        color: #7c3aed;
        font-weight: 500;
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
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(13, 148, 136, 0.25);
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

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Sidebar input styling */
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        font-size: 0.9rem;
    }

    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stSelectbox label {
        font-size: 0.8rem;
        color: #64748b;
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

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }

    /* Primary buttons - Navy gradient */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 50%, #1a365d 100%);
        background-size: 200% 200%;
        color: #ffffff;
        border: none;
        box-shadow: 0 2px 8px rgba(26, 54, 93, 0.25), 0 1px 3px rgba(26, 54, 93, 0.15);
    }

    .stButton > button[kind="primary"]:hover {
        background-position: 100% 100%;
        box-shadow: 0 6px 20px rgba(26, 54, 93, 0.35), 0 3px 8px rgba(26, 54, 93, 0.2);
    }

    .stButton > button[kind="primary"]:active {
        box-shadow: 0 2px 4px rgba(26, 54, 93, 0.2);
    }

    /* Secondary buttons - Clean white with border */
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
        color: #1a365d;
    }

    /* Sidebar buttons styling */
    section[data-testid="stSidebar"] .stButton > button {
        font-size: 0.8rem !important;
        padding: 0.5rem 0.75rem !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    /* Apply Strategy button - Green style */
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #059669 0%, #0d9488 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(5, 150, 105, 0.25) !important;
    }

    section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #047857 0%, #0f766e 100%) !important;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.35) !important;
    }

    /* Clear button - Light with red on hover */
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
        color: #64748b !important;
        border: 1px solid #e2e8f0 !important;
    }

    section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%) !important;
        color: #dc2626 !important;
        border-color: #fecaca !important;
    }

    /* Strategy dropdown styling */
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        padding: 0.125rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.2s ease !important;
    }

    section[data-testid="stSidebar"] .stSelectbox > div > div:hover {
        border-color: #1a365d !important;
        box-shadow: 0 2px 8px rgba(26, 54, 93, 0.15) !important;
    }

    section[data-testid="stSidebar"] .stSelectbox > div > div:focus-within {
        border-color: #1a365d !important;
        box-shadow: 0 0 0 3px rgba(26, 54, 93, 0.1) !important;
    }

    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #1e293b !important;
    }

    /* Dropdown menu items */
    [data-baseweb="popover"] [role="listbox"] {
        background: #ffffff !important;
        border-radius: 10px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15) !important;
        border: 1px solid #e2e8f0 !important;
        padding: 0.5rem !important;
    }

    [data-baseweb="popover"] [role="option"] {
        border-radius: 6px !important;
        padding: 0.5rem 0.75rem !important;
        font-size: 0.85rem !important;
        margin: 0.125rem 0 !important;
    }

    [data-baseweb="popover"] [role="option"]:hover {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important;
    }

    [data-baseweb="popover"] [role="option"][aria-selected="true"] {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%) !important;
        color: #ffffff !important;
    }

    /* Button icons and text alignment */
    .stButton > button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
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

    /* ========== Toggle Button Group (Vary By controls) ========== */
    .main .stButton > button {
        min-width: 90px;
        font-size: 0.85rem;
        padding: 0.5rem 0.75rem;
    }

    /* Vary by toggle - primary selected state */
    .main .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 2px 6px rgba(26, 54, 93, 0.3) !important;
    }

    /* Vary by toggle - secondary unselected state */
    .main .stButton > button[kind="secondary"],
    .main .stButton > button:not([kind="primary"]) {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
        color: #64748b !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }

    .main .stButton > button[kind="secondary"]:hover,
    .main .stButton > button:not([kind="primary"]):hover {
        background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%) !important;
        color: #1a365d !important;
        border-color: #cbd5e1 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
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
