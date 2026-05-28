"""
Utilities Module
================

Shared utilities used across the backend.
All functions are Numba-optimized for performance.

Contents:
- math: Mathematical primitives, Black-Scholes formulas, all Greeks
- validation: Input parameter validation

This module is the SINGLE SOURCE OF TRUTH for:
- Normal distribution functions (CDF, PDF, inverse CDF)
- Black-Scholes pricing and all Greeks (1st, 2nd, 3rd order)
- Implied volatility calculation
- Discount factors and moneyness utilities

Author: Thomas Vaudescal
Created: 2026
"""

# Core constants re-exported here for convenience.
# For scaling factors (VEGA_SCALE, etc.), classification tuples (FIRST_ORDER_GREEKS, etc.),
# and calibration bounds, import from backend.utils.constants directly.
from backend.utils.constants import (
    # Time
    CALENDAR_DAYS_PER_YEAR,
    DAYS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
    # Numerical
    DEFAULT_TOLERANCE,
    GARCH_CALIBRATION_VARIANCE_FLOOR,
    GARCH_VARIANCE_FLOOR,
    LOG_2PI,
    MIN_MATURITY,
    SMOOTHING_EPS,
    VAR_FLOOR,
    # Greeks
    GREEK_CHARM,
    GREEK_COLOR,
    GREEK_DELTA,
    GREEK_GAMMA,
    GREEK_PRICE,
    GREEK_RHO,
    GREEK_SPEED,
    GREEK_THETA,
    GREEK_ULTIMA,
    GREEK_VANNA,
    GREEK_VEGA,
    GREEK_VETA,
    GREEK_VOLGA,
    GREEK_ZOMMA,
    NUM_GREEKS,
    VALID_GREEKS,
    # Monte Carlo
    DEFAULT_MC_PATHS,
    DEFAULT_MC_STEPS_PER_YEAR,
    DEFAULT_RATE_BUMP,
    DEFAULT_SPOT_BUMP,
    DEFAULT_TIME_BUMP_DAYS,
    DEFAULT_VOL_BUMP,
)
from backend.utils.math import (
    SQRT_2,
    SQRT_2PI,
    # First-order Greeks
    bs_delta,
    bs_gamma,
    bs_greeks,
    # Black-Scholes pricing
    bs_price,
    bs_rho,
    # Second-order Greeks
    bs_second_order_greeks,
    bs_theta,
    # Third-order Greeks
    bs_third_order_greeks,
    bs_vega,
    # Black-Scholes parameters
    d1_d2,
    delta_to_strike,
    # Discount factors
    discount_factor,
    forward_log_moneyness,
    forward_price,
    # Implied volatility
    implied_volatility,
    # Moneyness
    log_moneyness,
    # Normal distribution
    norm_cdf,
    norm_cdf_vec,
    norm_inv_cdf,
    norm_pdf,
    norm_pdf_vec,
)
from backend.utils.logging import (
    configure_root,
    get_logger,
)
from backend.utils.validation import (
    ArbitrageViolationError,
    FellerConditionError,
    ParameterOutOfRangeError,
    # Exceptions
    ValidationError,
    check_put_call_parity,
    # Feller helpers
    feller_ratio,
    feller_satisfied,
    validate_correlation,
    validate_finite,
    # Model validation
    validate_heston_parameters,
    validate_in_range,
    validate_maturity,
    # Arbitrage
    validate_no_arbitrage,
    # Basic validation
    validate_positive,
    validate_rate,
    # Market parameters
    validate_spot,
    validate_strike,
    validate_vanilla_option,
    validate_volatility,
)

__all__ = [
    # Time constants
    "CALENDAR_DAYS_PER_YEAR",
    "DAYS_PER_YEAR",
    "TRADING_DAYS_PER_YEAR",
    # Numerical constants
    "VAR_FLOOR",
    "GARCH_VARIANCE_FLOOR",
    "GARCH_CALIBRATION_VARIANCE_FLOOR",
    "SMOOTHING_EPS",
    "MIN_MATURITY",
    "DEFAULT_TOLERANCE",
    "LOG_2PI",
    # Greek constants
    "GREEK_PRICE",
    "GREEK_DELTA",
    "GREEK_GAMMA",
    "GREEK_VEGA",
    "GREEK_THETA",
    "GREEK_RHO",
    "GREEK_VANNA",
    "GREEK_VOLGA",
    "GREEK_CHARM",
    "GREEK_VETA",
    "GREEK_SPEED",
    "GREEK_ZOMMA",
    "GREEK_COLOR",
    "GREEK_ULTIMA",
    "NUM_GREEKS",
    "VALID_GREEKS",
    # Monte Carlo constants
    "DEFAULT_MC_PATHS",
    "DEFAULT_MC_STEPS_PER_YEAR",
    "DEFAULT_SPOT_BUMP",
    "DEFAULT_VOL_BUMP",
    "DEFAULT_RATE_BUMP",
    "DEFAULT_TIME_BUMP_DAYS",
    # Math constants
    "SQRT_2PI",
    "SQRT_2",
    # Normal distribution
    "norm_cdf",
    "norm_pdf",
    "norm_inv_cdf",
    "norm_cdf_vec",
    "norm_pdf_vec",
    # Black-Scholes parameters
    "d1_d2",
    # Black-Scholes pricing
    "bs_price",
    # First-order Greeks
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "bs_theta",
    "bs_rho",
    "bs_greeks",
    # Second-order Greeks
    "bs_second_order_greeks",
    # Third-order Greeks
    "bs_third_order_greeks",
    # Implied volatility
    "implied_volatility",
    # Discount factors
    "discount_factor",
    "forward_price",
    # Moneyness
    "log_moneyness",
    "forward_log_moneyness",
    "delta_to_strike",
    # Validation exceptions
    "ValidationError",
    "ParameterOutOfRangeError",
    "ArbitrageViolationError",
    "FellerConditionError",
    # Validation functions
    "validate_positive",
    "validate_in_range",
    "validate_finite",
    "validate_spot",
    "validate_strike",
    "validate_maturity",
    "validate_rate",
    "validate_volatility",
    "validate_vanilla_option",
    "validate_heston_parameters",
    "validate_correlation",
    "validate_no_arbitrage",
    "check_put_call_parity",
    # Feller helpers
    "feller_ratio",
    "feller_satisfied",
    # Logging
    "get_logger",
    "configure_root",
]
