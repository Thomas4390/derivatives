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
    "primary": "#0d9488",
    "primary_light": "#14b8a6",
    "primary_dark": "#0f766e",
    # Secondary - Deep navy (trust, professionalism)
    "secondary": "#1a365d",
    "secondary_light": "#2c5282",
    "secondary_dark": "#0f2942",
    # Accent - Warm amber (energy, insights)
    "accent": "#d97706",
    "accent_light": "#f59e0b",
    "accent_dark": "#b45309",
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
        font-size: clamp(0.9rem, 2vw, 1.5rem);
        font-weight: 700;
        color: #1e293b;
        line-height: 1.2;
        font-family: 'JetBrains Mono', monospace;
        overflow-wrap: break-word;
        word-break: break-word;
        hyphens: auto;
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
    /* Slider colour is driven by the theme primaryColor (.streamlit/config.toml).
       Do NOT target .stSlider inner divs: that descendant selector drifts
       across Streamlit versions and ends up painting the value bubble
       (the "green box" regression). */

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

    /* ========== P-Measure Badge ========== */
    .p-measure-badge {
        background: linear-gradient(135deg, #0d9488, #0f766e);
        color: white;
        font-size: 0.65rem;
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: 4px;
        font-weight: 600;
        letter-spacing: 0.02em;
        vertical-align: middle;
        display: inline-block;
        cursor: help;
    }

    .p-measure-badge:hover {
        background: linear-gradient(135deg, #14b8a6, #0d9488);
        box-shadow: 0 2px 4px rgba(13, 148, 136, 0.3);
    }

    /* ========== Q-Measure Badge (risk-neutral) ========== */
    .q-measure-badge {
        background: linear-gradient(135deg, #6366f1, #4338ca);
        color: white;
        font-size: 0.65rem;
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: 4px;
        font-weight: 600;
        letter-spacing: 0.02em;
        vertical-align: middle;
        display: inline-block;
        cursor: help;
    }

    .q-measure-badge:hover {
        background: linear-gradient(135deg, #818cf8, #6366f1);
        box-shadow: 0 2px 4px rgba(99, 102, 241, 0.3);
    }

    /* ========== Configuration Cards ========== */
    .config-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }

    .config-card-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #f1f5f9;
    }

    .config-card-header-icon {
        width: 28px;
        height: 28px;
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.85rem;
    }

    .config-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #1e293b;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    /* ========== Model Selection Cards ========== */
    .model-select-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }

    .model-select-card:hover {
        border-color: #0d9488;
        background: #f0fdfa;
    }

    .model-select-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .model-select-name {
        font-weight: 600;
        color: #1e293b;
        font-size: 0.9rem;
    }

    .model-params-inline {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
    }

    /* ========== Strategy Builder Horizontal Layout ========== */
    .strategy-legs-horizontal {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }

    .strategy-leg-card {
        flex: 1;
        min-width: 200px;
        max-width: 280px;
        background: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }

    .strategy-leg-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
    }

    .strategy-leg-card.long {
        border-left: 4px solid #10b981;
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
    }

    .strategy-leg-card.short {
        border-left: 4px solid #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    }

    .add-leg-card {
        flex: 1;
        min-width: 200px;
        max-width: 280px;
        background: #f8fafc;
        border: 2px dashed #cbd5e1;
        border-radius: 10px;
        padding: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s ease;
        color: #64748b;
        font-weight: 500;
    }

    .add-leg-card:hover {
        border-color: #0d9488;
        color: #0d9488;
        background: #f0fdfa;
    }

    /* ========== Compact Summary Bar ========== */
    .summary-bar {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: white;
        margin: 1rem 0;
    }

    .summary-item {
        text-align: center;
    }

    .summary-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: rgba(255, 255, 255, 0.7);
        margin-bottom: 0.25rem;
    }

    .summary-value {
        font-size: 1.1rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    .summary-value.profit {
        color: #4ade80;
    }

    .summary-value.loss {
        color: #f87171;
    }

    /* ========== Stale Results Warning ========== */
    .stale-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 1rem 0;
    }

    .stale-warning-icon {
        font-size: 1.25rem;
    }

    .stale-warning-text {
        color: #92400e;
        font-size: 0.875rem;
        font-weight: 500;
    }

    /* ========== Full Width Layout (No Sidebar) ========== */
    .full-width-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 1rem;
    }

    /* ========== Compact Header ========== */
    .compact-header {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 50%, #1a365d 100%);
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .compact-header-left h1 {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.025em;
    }

    .compact-header-left p {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        margin: 0.25rem 0 0 0;
    }

    .compact-header-badge {
        background: rgba(255, 255, 255, 0.15);
        color: #fbbf24;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ========== Main Tab Navigation ==========
       The teal-filled active-tab style now lives on the generic
       `.stTabs [aria-selected="true"]` rule above, shared by every app
       (options_greeks / simulation / calibration). The old
       `.main-tabs-container` wrapper was never applied in any app.py, so
       its rules were dead code and have been removed. */

    /* ========== Visualization Options Compact ========== */
    .viz-options-row {
        display: flex;
        gap: 1.5rem;
        align-items: center;
        padding: 0.75rem 1rem;
        background: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
    }

    .viz-option-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        color: #475569;
    }

    /* ========== Strategy Collapsed Summary ========== */
    .strategy-collapsed {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .strategy-collapsed:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(26, 54, 93, 0.2);
    }

    .strategy-collapsed-left {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .strategy-collapsed-name {
        color: #ffffff;
        font-weight: 600;
        font-size: 0.95rem;
    }

    .strategy-collapsed-legs {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.8rem;
    }

    .strategy-collapsed-cost {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.1rem;
    }

    .strategy-collapsed-cost.debit {
        color: #fca5a5;
    }

    .strategy-collapsed-cost.credit {
        color: #86efac;
    }

    /* ========== Statistics Cards - Colored Variants ========== */
    .stats-card {
        background: #ffffff;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0d9488;
        height: 100%;
        transition: all 0.2s ease;
    }

    .stats-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }

    .stats-card .stats-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.375rem;
    }

    .stats-card .stats-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        line-height: 1.2;
        font-family: 'JetBrains Mono', monospace;
    }

    .stats-card .stats-subtitle {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }

    /* Teal variant (default) */
    .stats-card.teal { border-left-color: #0d9488; background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%); }
    .stats-card.teal .stats-label { color: #0f766e; }
    .stats-card.teal .stats-value { color: #0d9488; }

    /* Blue variant */
    .stats-card.blue { border-left-color: #3b82f6; background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%); }
    .stats-card.blue .stats-label { color: #1d4ed8; }
    .stats-card.blue .stats-value { color: #3b82f6; }

    /* Amber variant */
    .stats-card.amber { border-left-color: #f59e0b; background: linear-gradient(135deg, #ffffff 0%, #fffbeb 100%); }
    .stats-card.amber .stats-label { color: #b45309; }
    .stats-card.amber .stats-value { color: #d97706; }

    /* Red variant */
    .stats-card.red { border-left-color: #ef4444; background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%); }
    .stats-card.red .stats-label { color: #b91c1c; }
    .stats-card.red .stats-value { color: #dc2626; }

    /* Green variant */
    .stats-card.green { border-left-color: #22c55e; background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%); }
    .stats-card.green .stats-label { color: #15803d; }
    .stats-card.green .stats-value { color: #16a34a; }

    /* Purple variant */
    .stats-card.purple { border-left-color: #a855f7; background: linear-gradient(135deg, #ffffff 0%, #faf5ff 100%); }
    .stats-card.purple .stats-label { color: #7e22ce; }
    .stats-card.purple .stats-value { color: #9333ea; }

    /* Slate variant (neutral) */
    .stats-card.slate { border-left-color: #64748b; background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); }
    .stats-card.slate .stats-label { color: #475569; }
    .stats-card.slate .stats-value { color: #334155; }

    /* Stats row container */
    .stats-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }

    /* Stats table styling */
    .stats-table {
        width: 100%;
        border-collapse: collapse;
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }

    .stats-table th {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
        color: white;
        padding: 0.875rem 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        text-align: left;
    }

    .stats-table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #e2e8f0;
        font-size: 0.9rem;
    }

    .stats-table tr:last-child td {
        border-bottom: none;
    }

    .stats-table tr:nth-child(even) {
        background: #f8fafc;
    }

    .stats-table tr:hover {
        background: #f0fdfa;
    }

    .stats-table .value-cell {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        text-align: right;
    }

    .stats-table .positive { color: #16a34a; }
    .stats-table .negative { color: #dc2626; }

    /* Comparison table */
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }

    .comparison-table th {
        background: #f1f5f9;
        color: #475569;
        padding: 0.75rem 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .comparison-table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #e2e8f0;
    }

    .comparison-table .label-cell {
        font-weight: 500;
        color: #374151;
    }

    .comparison-table .value-cell {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        text-align: center;
    }

    .comparison-table .error-cell {
        font-size: 0.85rem;
        text-align: center;
    }

    .comparison-table .error-cell.good { color: #16a34a; background: #f0fdf4; }
    .comparison-table .error-cell.warning { color: #d97706; background: #fffbeb; }
    .comparison-table .error-cell.bad { color: #dc2626; background: #fef2f2; }
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


def render_metric_row(
    metrics: list[tuple[str, str] | tuple[str, str, str]], columns: int | None = None
) -> None:
    """Render a row of metric cards.

    Each item in *metrics* is ``(label, value)`` or ``(label, value, subtext)``.
    *columns* defaults to ``len(metrics)`` when not provided.
    """
    if not metrics:
        return
    n_cols = columns or len(metrics)
    cols = st.columns(n_cols)
    for col, item in zip(cols, metrics):
        label, value = item[0], item[1]
        subtext = item[2] if len(item) > 2 else None
        with col:
            st.markdown(
                metric_card_html(label, value, subtext=subtext), unsafe_allow_html=True
            )


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


def render_compact_header(title: str, subtitle: str, badge: str = None) -> None:
    """Render a compact application header for full-width layout."""
    badge_html = f'<span class="compact-header-badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""
    <div class="compact-header animate-in">
        <div class="compact-header-left">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        {badge_html}
    </div>
    """,
        unsafe_allow_html=True,
    )


def p_measure_badge_html() -> str:
    """Generate HTML for the P-measure badge."""
    return '<span class="p-measure-badge" title="Physical measure - uses expected return as drift">P</span>'


def q_measure_badge_html() -> str:
    """Generate HTML for the Q-measure (risk-neutral) badge."""
    return (
        '<span class="q-measure-badge" title="Risk-neutral measure - '
        'drift is the risk-free rate; calibrated to option prices">Q</span>'
    )


def config_card_html(icon: str, title: str) -> str:
    """Generate HTML for a configuration card header."""
    return f"""
    <div class="config-card-header">
        <div class="config-card-header-icon">{icon}</div>
        <span class="config-card-title">{title}</span>
    </div>
    """


def stale_results_warning_html() -> str:
    """Generate HTML for stale results warning."""
    return """
    <div class="stale-warning">
        <span class="stale-warning-icon">⚠️</span>
        <span class="stale-warning-text">Parameters changed. Re-run simulation to see updated results.</span>
    </div>
    """


def strategy_collapsed_html(
    strategy_name: str, num_legs: int, has_stock: bool, net_cost: float
) -> str:
    """Generate HTML for collapsed strategy summary."""
    is_debit = net_cost < 0
    cost_class = "debit" if is_debit else "credit"
    cost_display = f"-${abs(net_cost):,.2f}" if is_debit else f"+${abs(net_cost):,.2f}"
    legs_text = f"{num_legs} leg{'s' if num_legs > 1 else ''}{' + 100 shares' if has_stock else ''}"

    return f"""
    <div class="strategy-collapsed">
        <div class="strategy-collapsed-left">
            <span style="font-size: 1.25rem;">📊</span>
            <div>
                <div class="strategy-collapsed-name">{strategy_name}</div>
                <div class="strategy-collapsed-legs">{legs_text}</div>
            </div>
        </div>
        <span class="strategy-collapsed-cost {cost_class}">{cost_display}</span>
    </div>
    """


# =============================================================================
# STYLED STATISTICS COMPONENTS
# =============================================================================


def stats_card_html(
    label: str, value: str, subtitle: str = None, variant: str = "teal"
) -> str:
    """
    Generate HTML for a styled statistics card.

    Args:
        label: Card label (appears at top)
        value: Main value to display
        subtitle: Optional subtitle text
        variant: Color variant ('teal', 'blue', 'amber', 'red', 'green', 'purple', 'slate')
    """
    subtitle_html = f'<div class="stats-subtitle">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="stats-card {variant}">
        <div class="stats-label">{label}</div>
        <div class="stats-value">{value}</div>
        {subtitle_html}
    </div>
    """


def render_stats_row(stats: list, variants: list = None) -> None:
    """
    Render a row of statistics cards.

    Args:
        stats: List of tuples (label, value, subtitle) or (label, value)
        variants: Optional list of color variants for each card
    """
    if variants is None:
        variants = ["teal", "blue", "amber", "green", "purple", "red", "slate"]

    cols = st.columns(len(stats))

    for i, (col, stat) in enumerate(zip(cols, stats)):
        variant = variants[i % len(variants)]
        label, value = stat[0], stat[1]
        subtitle = stat[2] if len(stat) > 2 else None

        with col:
            st.markdown(
                stats_card_html(label, value, subtitle, variant), unsafe_allow_html=True
            )


def stats_table_html(headers: list, rows: list) -> str:
    """
    Generate HTML for a styled statistics table.

    Args:
        headers: List of header strings
        rows: List of row tuples/lists
    """
    header_html = "".join(f"<th>{h}</th>" for h in headers)

    rows_html = ""
    for row in rows:
        cells_html = ""
        for i, cell in enumerate(row):
            cell_class = "value-cell" if i > 0 else ""
            # Check for positive/negative values
            if isinstance(cell, str) and cell.startswith("+"):
                cell_class += " positive"
            elif isinstance(cell, str) and cell.startswith("-") and "$" in cell:
                cell_class += " negative"
            cells_html += f'<td class="{cell_class}">{cell}</td>'
        rows_html += f"<tr>{cells_html}</tr>"

    return f"""
    <div style="overflow-x: auto;">
        <table class="stats-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """


def comparison_table_html(title: str, rows: list, headers: list = None) -> str:
    """
    Generate HTML for a comparison table (e.g., Theoretical vs Simulated).

    Args:
        title: Table title
        rows: List of tuples (metric_name, theoretical, simulated, error)
        headers: Optional custom headers (default: Metric, Theoretical, Simulated, Error)
    """
    if headers is None:
        headers = ["Metric", "Theoretical", "Simulated", "Error"]

    header_html = "".join(f"<th>{h}</th>" for h in headers)

    rows_html = ""
    for row in rows:
        cells_html = f'<td class="label-cell">{row[0]}</td>'
        cells_html += f'<td class="value-cell">{row[1]}</td>'
        cells_html += f'<td class="value-cell">{row[2]}</td>'

        # Determine error class based on value
        if len(row) > 3:
            error_val = row[3]
            try:
                error_num = float(error_val.replace("%", ""))
                if error_num < 1:
                    error_class = "good"
                elif error_num < 5:
                    error_class = "warning"
                else:
                    error_class = "bad"
            except Exception:
                error_class = ""
            cells_html += f'<td class="error-cell {error_class}">{error_val}</td>'

        rows_html += f"<tr>{cells_html}</tr>"

    return f"""
    <div style="margin: 1rem 0;">
        <table class="comparison-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """


def var_table_html(var_data: list, cvar_data: list, unit: str = "$") -> str:
    """
    Generate HTML for VaR/CVaR risk metrics table.

    Args:
        var_data: List of (confidence_level, var_value) tuples
        cvar_data: List of (confidence_level, cvar_value) tuples
        unit: Currency/unit symbol
    """
    rows_html = ""
    for (cl_var, var_val), (cl_cvar, cvar_val) in zip(var_data, cvar_data):
        rows_html += f"""
        <tr>
            <td class="label-cell">{int(cl_var * 100)}%</td>
            <td class="value-cell">{unit}{var_val:,.2f}</td>
            <td class="value-cell">{unit}{cvar_val:,.2f}</td>
        </tr>
        """

    return f"""
    <div style="overflow-x: auto; margin: 1rem 0;">
        <table class="stats-table">
            <thead>
                <tr>
                    <th>Confidence</th>
                    <th>VaR</th>
                    <th>CVaR (ES)</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """
