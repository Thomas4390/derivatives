"""
Constants for Options Greeks Explorer.

Core configuration constants used across the application. Domain-specific
constants live in their own modules (strategies, exotic_config, structured_config)
and are re-exported here for backward compatibility.
"""

# Re-export domain-specific constants so existing
# `from config.constants import X` statements continue to work.
from .exotic_config import *  # noqa: F401, F403
from .strategies import *  # noqa: F401, F403
from .structured_config import *  # noqa: F401, F403

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
    "price",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "vanna",
    "volga",
    "charm",
    "veta",
    "speed",
    "zomma",
    "color",
    "ultima",
]

# Greeks by order
FIRST_ORDER = ["delta", "gamma", "vega", "theta", "rho"]
SECOND_ORDER = ["vanna", "volga", "charm", "veta"]
THIRD_ORDER = ["speed", "zomma", "color", "ultima"]

# Greek display titles with mathematical notation
GREEK_TITLES = {
    "price": "Option Price (V)",
    "delta": "Delta (∂V/∂S)",
    "gamma": "Gamma (∂²V/∂S²)",
    "vega": "Vega (∂V/∂σ)",
    "theta": "Theta (∂V/∂t)",
    "rho": "Rho (∂V/∂r)",
    "vanna": "Vanna (∂²V/∂S∂σ)",
    "volga": "Volga/Vomma (∂²V/∂σ²)",
    "charm": "Charm (∂²V/∂S∂t)",
    "veta": "Veta (∂²V/∂σ∂t)",
    "speed": "Speed (∂³V/∂S³)",
    "zomma": "Zomma (∂³V/∂S²∂σ)",
    "color": "Color (∂³V/∂S²∂t)",
    "ultima": "Ultima (∂³V/∂σ³)",
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

# DTE range for calculations (0 = expiry, then every 3 days from 1 to 90)
DTE_RANGE = [0] + list(range(1, 91, 3))

# IV range for calculations (every 2% from 5% to 50%)
IV_RANGE = list(range(5, 51, 2))

# Strike range for single-leg strategies (percentage of spot: 80% to 120%, step 2%)
STRIKE_RANGE_FACTORS = [round(0.80 + i * 0.02, 2) for i in range(21)]  # 0.80 to 1.20

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
