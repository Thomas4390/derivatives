"""
Structured products configuration for Options Greeks Explorer.

Contains product type definitions, model types, observation frequencies,
default parameters, educational descriptions, and Monte Carlo settings.
"""

# =============================================================================
# PRODUCT TYPES & MODELS
# =============================================================================

STRUCTURED_PRODUCT_TYPES = {
    "cpn": "Capital Protected Note",
    "reverse_convertible": "Reverse Convertible",
    "autocallable": "Autocallable",
}

STRUCTURED_MODEL_TYPES = {
    "gbm": "GBM (Black-Scholes)",
    "heston": "Heston (Stochastic Vol)",
}

STRUCTURED_OBSERVATION_FREQUENCIES = ["monthly", "quarterly", "semi_annual", "annual"]

# Model parameter input specs for the structured-product pricing sidebar —
# slider/number_input bounds + defaults, centralised so the ranges live in one
# place instead of being hard-coded in the rendering function.
SP_GBM_VOL_SPEC = {
    "label": "Volatility", "default": 0.20, "min": 0.05, "max": 0.80, "step": 0.01,
    "format": "%.2f",
}
SP_HESTON_PARAM_SPECS = [
    {"name": "v0", "key": "sp_v0", "label": "v₀", "default": 0.04, "min": 0.001, "max": 0.50, "step": 0.01, "format": "%.3f", "col": 0},
    {"name": "kappa", "key": "sp_kappa", "label": "κ", "default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1, "format": "%.1f", "col": 0},
    {"name": "rho", "key": "sp_rho", "label": "ρ", "default": -0.7, "min": -0.99, "max": 0.99, "step": 0.05, "format": "%.2f", "col": 0},
    {"name": "theta", "key": "sp_theta", "label": "σ²", "default": 0.04, "min": 0.001, "max": 0.50, "step": 0.01, "format": "%.3f", "col": 1},
    {"name": "alpha", "key": "sp_xi", "label": "α", "default": 0.3, "min": 0.01, "max": 1.5, "step": 0.05, "format": "%.2f", "col": 1},
]

# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

STRUCTURED_PRODUCT_DEFAULTS = {
    "cpn": {
        "notional": 1000.0,
        "maturity": 3.0,
        "participation_rate": 0.80,
        "cap": 1.50,
        "protection_level": 1.0,
        "observation_frequency": "annual",
    },
    "reverse_convertible": {
        "notional": 1000.0,
        "maturity": 1.0,
        "coupon_rate": 0.10,
        "barrier": 0.60,
        "barrier_monitoring": "continuous",
        "observation_frequency": "quarterly",
    },
    "autocallable": {
        "notional": 1000.0,
        "maturity": 3.0,
        "coupon_rate": 0.07,
        "autocall_trigger": 1.0,
        "coupon_barrier": 0.70,
        "ki_barrier": 0.60,
        "memory_coupon": True,
        "barrier_monitoring": "continuous",
        "observation_frequency": "quarterly",
    },
}

# =============================================================================
# EDUCATIONAL DESCRIPTIONS
# =============================================================================

STRUCTURED_PRODUCT_DESCRIPTIONS = {
    "cpn": "A **Capital Protected Note** guarantees return of the notional at maturity (bond floor) while offering participation in the upside. The upside may be capped. Decomposition: zero-coupon bond + call option spread.",
    "reverse_convertible": "A **Reverse Convertible** pays an above-market fixed coupon in exchange for the investor bearing downside risk via a knock-in put. If the underlying breaches the barrier, the investor receives depreciated shares at maturity.",
    "autocallable": "An **Autocallable** pays conditional coupons if the underlying stays above a coupon barrier. If above the autocall trigger at any observation date, it terminates early returning notional + coupon. Capital is at risk via a knock-in put.",
}

# =============================================================================
# MONTE CARLO SETTINGS
# =============================================================================

DEFAULT_MC_PATHS = 50_000
DEFAULT_SCENARIO_PATHS = 10_000
DEFAULT_SCENARIO_POINTS = 15

# Surface calculation (reduced grid for MC feasibility)
SP_SPOT_RANGE_FACTOR = 0.55  # ±55% to capture barriers at 60% of spot
SP_SPOT_RANGE_POINTS = 15
SP_DTE_RANGE = [1, 7, 30, 90, 180, 365, 545, 730, 1095]
SP_IV_RANGE = [10, 20, 30, 40, 50]
SP_SURFACE_PATHS = 15_000
