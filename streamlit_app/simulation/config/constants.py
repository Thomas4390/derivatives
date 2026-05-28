"""
Constants for Monte Carlo Simulation Explorer.

This module centralizes all configuration constants used throughout the application.
"""

# =============================================================================
# SIMULATION MODELS
# =============================================================================

# Price path models
PRICE_MODELS = {
    "gbm": "Geometric Brownian Motion (GBM)",
    "heston": "Heston Stochastic Volatility",
    "merton": "Merton Jump Diffusion",
    "bates": "Bates Model (Heston + Jumps)",
    "sabr": "SABR Model",
}

# Volatility models
VOLATILITY_MODELS = {
    "garch": "GARCH(1,1)",
    "ngarch": "NGARCH (NAGARCH)",
    "gjr_garch": "GJR-GARCH",
    "egarch": "EGARCH",
}

# Simulation modes
SIMULATION_MODES = {
    "price": "Price Path Simulation",
    "volatility": "Volatility Simulation",
    "option_pnl": "Option P&L Analysis",
}

# Model descriptions for educational purposes
MODEL_DESCRIPTIONS = {
    # Price models
    "gbm": """
**Geometric Brownian Motion (Black-Scholes)**

The foundational model for option pricing. Assumes:
- Constant volatility (σ)
- Log-normal price distribution
- Continuous sample paths

**SDE:** dS = r·S·dt + σ·S·dW
""",
    "heston": """
**Heston Stochastic Volatility Model**

Extends GBM with stochastic variance:
- Mean-reverting variance process
- Correlation between price and volatility (ρ)
- Captures volatility smile

**SDEs:**
- dS = r·S·dt + √V·S·dW_S
- dV = κ(σ² - V)dt + α·√V·dW_V
""",
    "merton": """
**Merton Jump Diffusion Model**

Adds jumps to GBM to capture:
- Sudden price movements (crashes, spikes)
- Fat tails in return distribution
- More realistic risk modeling

**SDE:** dS/S = (r - λk)dt + σdW + (J-1)dN
""",
    "bates": """
**Bates Model (Stochastic Volatility + Jumps)**

Combines Heston stochastic volatility with Merton-style jumps:
- Stochastic variance with mean reversion
- Jump component for sudden price movements
- Most flexible equity model for options

**SDEs:**
- dS = (μ - λk)·S·dt + √V·S·dW_S + (J-1)·S·dN
- dV = κ(σ² - V)dt + α·√V·dW_V
""",
    "sabr": """
**SABR Stochastic Volatility Model**

Popular for interest rate derivatives:
- CEV-like forward dynamics
- Stochastic volatility
- Analytical implied vol approximation

**SDEs:**
- dF = α·F^β·dW_F
- dα = ν·α·dW_α
""",
    # Volatility models
    "garch": """
**GARCH(1,1) - Generalized Autoregressive Conditional Heteroskedasticity**

The workhorse volatility model:
- Volatility clustering
- Mean reversion in variance
- Symmetric response to shocks

**Equation:** σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
""",
    "ngarch": """
**NGARCH (NAGARCH) - Nonlinear Asymmetric GARCH**

Captures the leverage effect:
- Negative returns increase volatility more
- More realistic for equity markets
- Engle & Ng (1993)

**Equation:** σ²_t = ω + α(ε_{t-1} - γσ_{t-1})² + βσ²_{t-1}
""",
    "gjr_garch": """
**GJR-GARCH - Asymmetric GARCH**

Alternative asymmetric model:
- Uses indicator function for sign
- Glosten, Jagannathan & Runkle (1993)

**Equation:** σ²_t = ω + (α + γI_{t-1})ε²_{t-1} + βσ²_{t-1}
""",
    "egarch": """
**EGARCH - Exponential GARCH**

Log-volatility model:
- No positivity constraints needed
- Nelson (1991)
- Asymmetric response

**Equation:** ln(σ²_t) = ω + α(|z_{t-1}| - E[|z|]) + γz_{t-1} + βln(σ²_{t-1})
""",
}

# =============================================================================
# DEFAULT SIMULATION PARAMETERS
# =============================================================================

# Price model defaults
DEFAULT_SPOT_PRICE = 100.0
DEFAULT_RISK_FREE_RATE = 0.05
DEFAULT_VOLATILITY = 0.20
DEFAULT_TIME_HORIZON = 1.0  # years
DEFAULT_NUM_PATHS = 1000
DEFAULT_NUM_STEPS = 252  # daily steps for 1 year

# Heston model defaults
DEFAULT_HESTON_V0 = 0.04  # Initial variance (20% vol)
DEFAULT_HESTON_KAPPA = 2.0  # Mean reversion speed
DEFAULT_HESTON_THETA = 0.04  # Long-term variance
DEFAULT_HESTON_XI = 0.3  # Vol of vol
DEFAULT_HESTON_RHO = -0.7  # Correlation

# Merton jump defaults
DEFAULT_MERTON_LAMBDA = 0.5  # Jump intensity
DEFAULT_MERTON_MU_J = -0.1  # Mean log-jump
DEFAULT_MERTON_SIGMA_J = 0.2  # Jump vol

# SABR defaults
DEFAULT_SABR_BETA = 0.5  # CEV exponent
DEFAULT_SABR_NU = 0.4  # Vol of vol
DEFAULT_SABR_RHO = -0.3  # Correlation

# Bates model defaults (combines Heston + Merton parameters)
DEFAULT_BATES_V0 = 0.04  # Initial variance (20% vol)
DEFAULT_BATES_KAPPA = 2.0  # Mean reversion speed
DEFAULT_BATES_THETA = 0.04  # Long-term variance
DEFAULT_BATES_XI = 0.3  # Vol of vol
DEFAULT_BATES_RHO = -0.7  # Correlation
DEFAULT_BATES_LAMBDA = 0.5  # Jump intensity
DEFAULT_BATES_MU_J = -0.1  # Mean log-jump
DEFAULT_BATES_SIGMA_J = 0.2  # Jump vol

# Volatility model defaults
DEFAULT_GARCH_OMEGA = 0.000002  # Base variance
DEFAULT_GARCH_ALPHA = 0.05  # ARCH coefficient
DEFAULT_GARCH_BETA = 0.90  # GARCH coefficient
DEFAULT_NGARCH_THETA = 0.5  # Leverage parameter
DEFAULT_GJR_GAMMA = 0.05  # Asymmetry coefficient
DEFAULT_EGARCH_GAMMA = -0.1  # Asymmetry coefficient

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

# Number of paths to display in path charts
MAX_DISPLAY_PATHS = 100

# Chart heights
CHART_HEIGHT_STANDARD = 500
CHART_HEIGHT_LARGE = 600
CHART_HEIGHT_SMALL = 400

# Percentile bands for path visualization
PERCENTILE_LOWER = 5
PERCENTILE_UPPER = 95

# =============================================================================
# COLOR SCHEMES FOR MODELS
# =============================================================================

MODEL_COLORS = {
    "gbm": "#1f77b4",
    "heston": "#ff7f0e",
    "merton": "#2ca02c",
    "bates": "#7f3c8d",
    "sabr": "#d62728",
    "garch": "#9467bd",
    "ngarch": "#8c564b",
    "gjr_garch": "#e377c2",
    "egarch": "#17becf",
}

# Path visualization colors
PATH_COLORS = {
    "sample_paths": "rgba(26, 54, 93, 0.15)",
    "mean_path": "#1a365d",
    "percentile_band": "rgba(13, 148, 136, 0.3)",
    "initial_price": "#dc2626",
    "terminal_dist": "#059669",
}

# =============================================================================
# EDUCATIONAL CONTENT
# =============================================================================

# Greek-style parameter symbols
PARAMETER_SYMBOLS = {
    "s0": "S₀",
    "r": "r",
    "y": "y",
    "sigma": "σ",
    "t": "T",
    "v0": "V₀",
    "kappa": "κ",
    "theta": "σ²",  # Heston/Bates long-run variance
    "rho": "ρ",
    "lam": "λ",
    "alpha_j": "α_J",
    "sigma_j": "σ_J",
    "beta": "β",
    "nu": "ν",
    "omega": "ω",
    "alpha": "α",  # GARCH ARCH effect / Heston vol-of-vol
    "gamma": "γ",  # NGARCH & GJR leverage
}

# Stationarity conditions
STATIONARITY_CONDITIONS = {
    "garch": "α + β < 1",
    "ngarch": "α(1 + γ²) + β < 1",
    "gjr_garch": "α + β + γ/2 < 1",
    "egarch": "|β| < 1",
}

# =============================================================================
# OPTION P&L SIMULATION CONSTANTS
# =============================================================================

# Contract multiplier (shares per contract)
CONTRACT_MULTIPLIER = 100

# Default expected return for P-measure simulation
DEFAULT_EXPECTED_RETURN = 0.08

# P&L visualization colors
PNL_COLORS = {
    "profit": "#10b981",
    "loss": "#ef4444",
    "neutral": "#6b7280",
    "var_line": "#f59e0b",
    "cvar_region": "rgba(239, 68, 68, 0.2)",
    "breakeven": "#8b5cf6",
    "payoff_curve": "#3b82f6",
}

# Risk metric display settings
RISK_METRICS = {
    "mean_pnl": {"label": "Mean P&L", "format": "${:.2f}", "color": "#1e293b"},
    "std_pnl": {"label": "Std Dev", "format": "${:.2f}", "color": "#475569"},
    "var_95": {"label": "VaR 95%", "format": "${:.2f}", "color": "#f59e0b"},
    "var_99": {"label": "VaR 99%", "format": "${:.2f}", "color": "#ef4444"},
    "cvar_95": {"label": "CVaR 95%", "format": "${:.2f}", "color": "#dc2626"},
    "prob_profit": {"label": "P(Profit)", "format": "{:.1%}", "color": "#10b981"},
    "max_profit": {"label": "Max Profit", "format": "${:.2f}", "color": "#059669"},
    "max_loss": {"label": "Max Loss", "format": "${:.2f}", "color": "#b91c1c"},
    "skewness": {"label": "Skewness", "format": "{:.3f}", "color": "#6366f1"},
    "kurtosis": {"label": "Kurtosis", "format": "{:.3f}", "color": "#8b5cf6"},
}

# P&L histogram settings
PNL_HISTOGRAM_BINS = 50
PNL_KDE_POINTS = 200

# Scenario analysis settings
SCENARIO_SCATTER_MAX_POINTS = 5000  # Downsample if more paths

# =============================================================================
# UI LAYOUT CONSTANTS
# =============================================================================

# Main tab configuration
MAIN_TABS = {
    "config": {"icon": "⚙️", "label": "Configuration"},
    "price": {"icon": "📈", "label": "Price Paths"},
    "volatility": {"icon": "📊", "label": "Volatility"},
    "pnl": {"icon": "💰", "label": "Option P&L"},
}

# Analysis sub-tabs for each mode
PRICE_ANALYSIS_TABS = [
    "🎛️ Interactive Path",
    "📈 Sample Paths",
    "📊 Terminal Distribution",
    "📋 Statistics",
]

VOLATILITY_ANALYSIS_TABS = [
    "🎛️ Interactive Path",
    "📈 Volatility Paths",
    "📊 Terminal Distribution",
    "📋 Statistics",
]

PNL_ANALYSIS_TABS = ["📊 P&L Distribution", "📋 Risk Metrics", "🎯 Scenario Analysis"]

# Configuration section groupings
CONFIG_SECTIONS = {
    "market": "Market Parameters",
    "simulation": "Simulation Settings",
    "price_model": "Price Model",
    "vol_model": "Volatility Model",
    "strategy": "Option Strategy",
    "visualization": "Visualization Options",
}

# =============================================================================
# STRUCTURED PRODUCT CONSTANTS
# =============================================================================

SP_DEFAULT_NOTIONAL = 1000.0
SP_DEFAULT_PATHS_SCENARIO = 10_000

SP_PRODUCT_COLORS = {
    "bond_floor": "#0d9488",  # teal
    "option_value": "#f59e0b",  # amber
    "coupon_pv": "#6366f1",  # indigo
    "autocall": "#10b981",  # green
    "barrier": "#ef4444",  # red
    "trigger": "#f97316",  # orange
}

SP_PRODUCT_DESCRIPTIONS = {
    "cpn": "Capital Protected Note: 100% capital protection + capped upside participation",
    "reverse_convertible": "Reverse Convertible: High fixed coupon with capital at risk via knock-in put",
    "autocallable": "Autocallable: Conditional coupons with early redemption trigger and capital protection barrier",
    "phoenix": "Phoenix Autocallable: Monthly conditional coupons with memory + annual autocall + knock-in put",
    "shark_note": "Shark Note: Capital protected + capped upside with knock-out barrier and rebate",
    "twin_win": "Twin Win: Profit from both up and down moves, capital at risk if knock-in barrier breached",
    "snowball": "Snowball Autocallable: Autocall + growing snowball coupon + knock-in put",
}

SP_PRODUCT_COLORS_NEW = {
    "phoenix": "#8b5cf6",  # purple
    "shark_note": "#06b6d4",  # cyan
    "twin_win": "#ec4899",  # pink
    "snowball": "#84cc16",  # lime
}
