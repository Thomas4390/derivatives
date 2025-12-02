"""
Constants for Options Greeks Explorer.

This module centralizes all configuration constants used throughout the application.
"""

# =============================================================================
# CONTRACT CONFIGURATION
# =============================================================================

# Options contract multiplier (1 contract = 100 shares)
CONTRACT_MULTIPLIER = 100

# =============================================================================
# GREEK DEFINITIONS
# =============================================================================

# All Greek names in calculation order
GREEK_NAMES = [
    'price', 'delta', 'gamma', 'vega', 'theta', 'rho',
    'vanna', 'volga', 'charm', 'veta', 'speed', 'zomma', 'color', 'ultima'
]

# Greeks by order
FIRST_ORDER = ['delta', 'gamma', 'vega', 'theta', 'rho']
SECOND_ORDER = ['vanna', 'volga', 'charm', 'veta']
THIRD_ORDER = ['speed', 'zomma', 'color', 'ultima']

# Greek display titles with mathematical notation
GREEK_TITLES = {
    'price': 'Option Price (V)',
    'delta': 'Delta (∂V/∂S)',
    'gamma': 'Gamma (∂²V/∂S²)',
    'vega': 'Vega (∂V/∂σ)',
    'theta': 'Theta (∂V/∂t)',
    'rho': 'Rho (∂V/∂r)',
    'vanna': 'Vanna (∂²V/∂S∂σ)',
    'volga': 'Volga/Vomma (∂²V/∂σ²)',
    'charm': 'Charm (∂²V/∂S∂t)',
    'veta': 'Veta (∂²V/∂σ∂t)',
    'speed': 'Speed (∂³V/∂S³)',
    'zomma': 'Zomma (∂³V/∂S²∂σ)',
    'color': 'Color (∂³V/∂S²∂t)',
    'ultima': 'Ultima (∂³V/∂σ³)'
}

# Greek chart colors (consistent color scheme)
GREEK_COLORS = {
    'price': '#1f77b4',
    'delta': '#1f77b4',
    'gamma': '#ff7f0e',
    'vega': '#2ca02c',
    'theta': '#d62728',
    'rho': '#9467bd',
    'vanna': '#e377c2',
    'volga': '#17becf',
    'charm': '#bcbd22',
    'veta': '#ff9896',
    'speed': '#ff9896',
    'zomma': '#8c564b',
    'color': '#7f7f7f',
    'ultima': '#c5b0d5'
}

# =============================================================================
# DEFAULT VALUES
# =============================================================================

DEFAULT_SPOT_PRICE = 100.0
DEFAULT_RISK_FREE_RATE = 0.05
DEFAULT_IV = 25  # percentage
DEFAULT_DTE = 31  # days

# =============================================================================
# CALCULATION RANGES
# =============================================================================

# Spot price range as factor of current spot
SPOT_RANGE_FACTOR = 0.3  # ±30% of spot price
SPOT_RANGE_POINTS = 200  # number of points in the range

# DTE range for calculations (every 3 days from 1 to 90)
DTE_RANGE = list(range(1, 91, 3))

# IV range for calculations (every 2% from 5% to 50%)
IV_RANGE = list(range(5, 51, 2))

# =============================================================================
# PREDEFINED STRATEGIES
# =============================================================================

AVAILABLE_STRATEGIES = [
    "",  # Empty option
    "long_straddle",
    "iron_condor",
    "butterfly",
    "covered_call",
    "protective_put",
    "bull_call_spread",
    "bear_put_spread",
    "collar"
]

# Strategy display names
STRATEGY_DISPLAY_NAMES = {
    "": "",
    "long_straddle": "Long Straddle",
    "iron_condor": "Iron Condor",
    "butterfly": "Butterfly",
    "covered_call": "Covered Call",
    "protective_put": "Protective Put",
    "bull_call_spread": "Bull Call Spread",
    "bear_put_spread": "Bear Put Spread",
    "collar": "Collar"
}

# =============================================================================
# UI CONFIGURATION
# =============================================================================

# Chart heights
CHART_HEIGHT_STANDARD = 650
CHART_HEIGHT_LARGE = 700
CHART_HEIGHT_3D = 700

# Slider configuration
SLIDER_POSITION_Y = -0.10
SLIDER_LENGTH = 0.9
SLIDER_POSITION_X = 0.05
