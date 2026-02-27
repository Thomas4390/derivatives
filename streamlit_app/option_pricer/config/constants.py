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

# Strike range for single-leg strategies (percentage of spot: 80% to 120%, step 2%)
STRIKE_RANGE_FACTORS = [round(0.80 + i * 0.02, 2) for i in range(21)]  # 0.80 to 1.20

# =============================================================================
# PREDEFINED STRATEGIES
# =============================================================================

AVAILABLE_STRATEGIES = [
    "",  # Empty option
    # Vanilla strategies
    "long_call",
    "short_call",
    "long_put",
    "short_put",
    # Spread strategies
    "bull_call_spread",
    "bear_put_spread",
    "bull_put_spread",
    "bear_call_spread",
    # Volatility strategies
    "long_straddle",
    "short_straddle",
    "long_strangle",
    "short_strangle",
    # Complex strategies
    "iron_condor",
    "butterfly",
    "covered_call",
    "protective_put",
    "collar",
    # Exotic strategies
    "barrier_up_out_call",
    "barrier_down_out_put",
    "barrier_knock_in_call",
    "digital_call",
    "digital_put",
    "digital_range_bet",
    "asian_call",
    "lookback_floating_call",
    "lookback_fixed_call",
]

# Strategy display names
STRATEGY_DISPLAY_NAMES = {
    "": "",
    # Vanilla
    "long_call": "Long Call",
    "short_call": "Short Call",
    "long_put": "Long Put",
    "short_put": "Short Put",
    # Spreads
    "bull_call_spread": "Bull Call Spread",
    "bear_put_spread": "Bear Put Spread",
    "bull_put_spread": "Bull Put Spread",
    "bear_call_spread": "Bear Call Spread",
    # Volatility
    "long_straddle": "Long Straddle",
    "short_straddle": "Short Straddle",
    "long_strangle": "Long Strangle",
    "short_strangle": "Short Strangle",
    # Complex
    "iron_condor": "Iron Condor",
    "butterfly": "Butterfly",
    "covered_call": "Covered Call",
    "protective_put": "Protective Put",
    "collar": "Collar",
    # Exotic
    "barrier_up_out_call": "Up-and-Out Call",
    "barrier_down_out_put": "Down-and-Out Put",
    "barrier_knock_in_call": "Down-and-In Call",
    "digital_call": "Digital Call",
    "digital_put": "Digital Put",
    "digital_range_bet": "Digital Range Bet",
    "asian_call": "Asian Call (Geometric)",
    "lookback_floating_call": "Lookback Call (Floating)",
    "lookback_fixed_call": "Lookback Call (Fixed Strike)",
}

# Strategy leg definitions (option_type, position_type, strike_offset from spot)
# Standard strike intervals for clear payoff diagrams:
# - ATM = 1.0 (100% of spot)
# - 5% OTM/ITM intervals for spreads and strangles
# - 10% intervals for iron condor wings
# - Butterfly uses 5% equal spacing for symmetry
STRATEGY_LEGS = {
    # ==========================================================================
    # VANILLA STRATEGIES (Single leg, ATM)
    # ==========================================================================
    "long_call": [
        {"option_type": "call", "position_type": "long", "strike_factor": 1.0, "quantity": 1}
    ],
    "short_call": [
        {"option_type": "call", "position_type": "short", "strike_factor": 1.0, "quantity": 1}
    ],
    "long_put": [
        {"option_type": "put", "position_type": "long", "strike_factor": 1.0, "quantity": 1}
    ],
    "short_put": [
        {"option_type": "put", "position_type": "short", "strike_factor": 1.0, "quantity": 1}
    ],

    # ==========================================================================
    # VERTICAL SPREADS (5% width between strikes)
    # ==========================================================================
    # Bull Call Spread: Buy lower strike call, Sell higher strike call
    # Profit when underlying rises, limited risk/reward
    "bull_call_spread": [
        {"option_type": "call", "position_type": "long", "strike_factor": 0.975, "quantity": 1},   # Slightly ITM
        {"option_type": "call", "position_type": "short", "strike_factor": 1.025, "quantity": 1}   # Slightly OTM
    ],

    # Bear Put Spread: Buy higher strike put, Sell lower strike put
    # Profit when underlying falls, limited risk/reward
    "bear_put_spread": [
        {"option_type": "put", "position_type": "long", "strike_factor": 1.025, "quantity": 1},    # Slightly ITM
        {"option_type": "put", "position_type": "short", "strike_factor": 0.975, "quantity": 1}    # Slightly OTM
    ],

    # Bull Put Spread (Credit Spread): Sell higher strike put, Buy lower strike put
    # Profit when underlying stays above short strike
    "bull_put_spread": [
        {"option_type": "put", "position_type": "short", "strike_factor": 0.975, "quantity": 1},   # ATM-ish
        {"option_type": "put", "position_type": "long", "strike_factor": 0.925, "quantity": 1}     # OTM protection
    ],

    # Bear Call Spread (Credit Spread): Sell lower strike call, Buy higher strike call
    # Profit when underlying stays below short strike
    "bear_call_spread": [
        {"option_type": "call", "position_type": "short", "strike_factor": 1.025, "quantity": 1},  # ATM-ish
        {"option_type": "call", "position_type": "long", "strike_factor": 1.075, "quantity": 1}    # OTM protection
    ],

    # ==========================================================================
    # VOLATILITY STRATEGIES
    # ==========================================================================
    # Straddle: ATM Call + ATM Put (same strike)
    "long_straddle": [
        {"option_type": "call", "position_type": "long", "strike_factor": 1.0, "quantity": 1},
        {"option_type": "put", "position_type": "long", "strike_factor": 1.0, "quantity": 1}
    ],
    "short_straddle": [
        {"option_type": "call", "position_type": "short", "strike_factor": 1.0, "quantity": 1},
        {"option_type": "put", "position_type": "short", "strike_factor": 1.0, "quantity": 1}
    ],

    # Strangle: OTM Call + OTM Put (different strikes, 5% OTM each side)
    "long_strangle": [
        {"option_type": "put", "position_type": "long", "strike_factor": 0.95, "quantity": 1},     # 5% OTM put
        {"option_type": "call", "position_type": "long", "strike_factor": 1.05, "quantity": 1}     # 5% OTM call
    ],
    "short_strangle": [
        {"option_type": "put", "position_type": "short", "strike_factor": 0.95, "quantity": 1},    # 5% OTM put
        {"option_type": "call", "position_type": "short", "strike_factor": 1.05, "quantity": 1}    # 5% OTM call
    ],

    # ==========================================================================
    # COMPLEX STRATEGIES
    # ==========================================================================
    # Iron Condor: Short strangle + long wings for protection
    # Symmetric around spot: 90/95 put spread + 105/110 call spread
    "iron_condor": [
        {"option_type": "put", "position_type": "long", "strike_factor": 0.90, "quantity": 1},     # Long put wing
        {"option_type": "put", "position_type": "short", "strike_factor": 0.95, "quantity": 1},    # Short put
        {"option_type": "call", "position_type": "short", "strike_factor": 1.05, "quantity": 1},   # Short call
        {"option_type": "call", "position_type": "long", "strike_factor": 1.10, "quantity": 1}     # Long call wing
    ],

    # Butterfly: Long wings + 2x short middle (5% equal spacing)
    # Maximum profit at middle strike at expiration
    "butterfly": [
        {"option_type": "call", "position_type": "long", "strike_factor": 0.95, "quantity": 1},    # Lower wing
        {"option_type": "call", "position_type": "short", "strike_factor": 1.0, "quantity": 2},    # Body (ATM)
        {"option_type": "call", "position_type": "long", "strike_factor": 1.05, "quantity": 1}     # Upper wing
    ],

    # ==========================================================================
    # STOCK + OPTIONS STRATEGIES
    # ==========================================================================
    # Covered Call: Long stock + Short OTM call
    "covered_call": [
        {"option_type": "call", "position_type": "short", "strike_factor": 1.05, "quantity": 1}    # 5% OTM
    ],

    # Protective Put: Long stock + Long OTM put
    "protective_put": [
        {"option_type": "put", "position_type": "long", "strike_factor": 0.95, "quantity": 1}      # 5% OTM
    ],

    # Collar: Long stock + Long OTM put + Short OTM call
    "collar": [
        {"option_type": "put", "position_type": "long", "strike_factor": 0.95, "quantity": 1},     # 5% OTM put
        {"option_type": "call", "position_type": "short", "strike_factor": 1.05, "quantity": 1}    # 5% OTM call
    ],

    # ==========================================================================
    # EXOTIC STRATEGIES
    # ==========================================================================
    # Up-and-Out Call: cheap directional bet, knocked out if spot rises too high
    "barrier_up_out_call": [
        {"option_type": "call", "position_type": "long", "strike_factor": 1.0, "quantity": 1,
         "instrument_class": "barrier", "barrier_factor": 1.10, "is_up": True, "is_knock_in": False}
    ],

    # Down-and-Out Put: cheap downside protection, knocked out if spot drops too far
    "barrier_down_out_put": [
        {"option_type": "put", "position_type": "long", "strike_factor": 1.0, "quantity": 1,
         "instrument_class": "barrier", "barrier_factor": 0.90, "is_up": False, "is_knock_in": False}
    ],

    # Down-and-In Call: activates only if spot dips below barrier first
    "barrier_knock_in_call": [
        {"option_type": "call", "position_type": "long", "strike_factor": 1.0, "quantity": 1,
         "instrument_class": "barrier", "barrier_factor": 0.90, "is_up": False, "is_knock_in": True}
    ],

    # Digital Call: pays fixed amount if spot > strike at expiry
    "digital_call": [
        {"option_type": "call", "position_type": "long", "strike_factor": 1.0, "quantity": 1,
         "instrument_class": "digital", "payout": 1.0}
    ],

    # Digital Put: pays fixed amount if spot < strike at expiry
    "digital_put": [
        {"option_type": "put", "position_type": "long", "strike_factor": 1.0, "quantity": 1,
         "instrument_class": "digital", "payout": 1.0}
    ],

    # Digital Range Bet: pays if spot ends between two strikes
    # Long digital call at lower strike + Short digital call at upper strike
    "digital_range_bet": [
        {"option_type": "call", "position_type": "long", "strike_factor": 0.95, "quantity": 1,
         "instrument_class": "digital", "payout": 1.0},
        {"option_type": "call", "position_type": "short", "strike_factor": 1.05, "quantity": 1,
         "instrument_class": "digital", "payout": 1.0},
    ],

    # Asian Call (Geometric): payoff based on geometric average price
    "asian_call": [
        {"option_type": "call", "position_type": "long", "strike_factor": 1.0, "quantity": 1,
         "instrument_class": "asian"}
    ],

    # Lookback Floating Call: buy at the minimum price observed
    "lookback_floating_call": [
        {"option_type": "call", "position_type": "long", "strike_factor": 1.0, "quantity": 1,
         "instrument_class": "lookback_floating"}
    ],

    # Lookback Fixed Call: payoff based on max price vs fixed strike
    "lookback_fixed_call": [
        {"option_type": "call", "position_type": "long", "strike_factor": 1.0, "quantity": 1,
         "instrument_class": "lookback_fixed"}
    ],
}

# Strategies that include a stock position
STRATEGIES_WITH_STOCK = ["covered_call", "protective_put", "collar"]

# Stock position type for each strategy (default is 'long')
STRATEGY_STOCK_POSITION = {
    "covered_call": "long",
    "protective_put": "long",
    "collar": "long"
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


# =============================================================================
# EXOTIC OPTIONS CONFIGURATION
# =============================================================================

# Instrument classes available as portfolio legs
INSTRUMENT_CLASSES = {
    'barrier': 'Barrier',
    'asian': 'Asian (Geo)',
    'digital': 'Digital',
    'lookback_fixed': 'Lookback (Fixed)',
    'lookback_floating': 'Lookback (Float)',
}

# Exotic type display names
EXOTIC_TYPE_NAMES = {
    "barrier": "Barrier Option",
    "asian": "Asian Option (Geometric)",
    "digital": "Digital Option (Cash-or-Nothing)",
    "lookback_floating": "Lookback (Floating Strike)",
    "lookback_fixed": "Lookback (Fixed Strike)",
}

# Barrier subtypes: all 8 combinations of up/down, in/out, call/put
BARRIER_SUBTYPES = {
    "up_out_call": {"label": "Up-and-Out Call", "is_call": True, "is_up": True, "is_knock_in": False},
    "up_in_call": {"label": "Up-and-In Call", "is_call": True, "is_up": True, "is_knock_in": True},
    "down_out_call": {"label": "Down-and-Out Call", "is_call": True, "is_up": False, "is_knock_in": False},
    "down_in_call": {"label": "Down-and-In Call", "is_call": True, "is_up": False, "is_knock_in": True},
    "up_out_put": {"label": "Up-and-Out Put", "is_call": False, "is_up": True, "is_knock_in": False},
    "up_in_put": {"label": "Up-and-In Put", "is_call": False, "is_up": True, "is_knock_in": True},
    "down_out_put": {"label": "Down-and-Out Put", "is_call": False, "is_up": False, "is_knock_in": False},
    "down_in_put": {"label": "Down-and-In Put", "is_call": False, "is_up": False, "is_knock_in": True},
}

# Educational descriptions for each exotic type
EXOTIC_DESCRIPTIONS = {
    "barrier": (
        "**Barrier options** activate (knock-in) or deactivate (knock-out) when the underlying "
        "hits a specified barrier level. **Key identity:** Knock-In + Knock-Out = Vanilla option "
        "(for same strike and barrier). Barriers are cheaper than vanilla because they can become "
        "worthless (knock-out) or may never activate (knock-in)."
    ),
    "asian": (
        "**Asian (Geometric) options** have a payoff based on the geometric average price of the "
        "underlying over the option's life, rather than the terminal price. They are **always cheaper** "
        "than vanilla options because averaging reduces the effective volatility "
        "(by a factor of 1/sqrt(3) for geometric averaging)."
    ),
    "digital": (
        "**Digital (Cash-or-Nothing) options** pay a fixed amount if the option expires in-the-money, "
        "and zero otherwise. The price equals PV(probability of ITM) x Payout. "
        "**Key identity:** Digital Call + Digital Put = e^(-rT) x Payout."
    ),
    "lookback_floating": (
        "**Floating-strike lookback options** allow the holder to buy at the lowest price (call) "
        "or sell at the highest price (put) observed during the option's life. This \"no-regret\" "
        "feature makes them **always more expensive** than vanilla options."
    ),
    "lookback_fixed": (
        "**Fixed-strike lookback options** have payoff based on the maximum (call) or minimum (put) "
        "of the underlying price vs a fixed strike. Call payoff: max(M_max - K, 0). "
        "Put payoff: max(K - M_min, 0). They are more expensive than vanilla since the "
        "lookback feature can only improve the payoff."
    ),
}

# Default values for exotic options
DEFAULT_EXOTIC_DTE = 90
DEFAULT_DIGITAL_PAYOUT = 1.0
DEFAULT_BARRIER_UP_FACTOR = 1.10   # Barrier at 110% of spot for up-barriers
DEFAULT_BARRIER_DOWN_FACTOR = 0.90  # Barrier at 90% of spot for down-barriers
