"""
Utilities Module
================

Shared utilities used across the backend.
All functions are Numba-optimized for performance.

Contents:
- math: Normal distribution functions, Black-Scholes parameters, higher-order Greeks
- validation: Input parameter validation

Author: Thomas
Created: 2025
"""

from .math import (
    # Normal distribution
    norm_cdf,
    norm_pdf,
    norm_cdf_vec,
    norm_pdf_vec,
    # Black-Scholes parameters
    d1_d2,
    # Higher-order Greeks (Numba-optimized)
    bs_second_order_greeks,
    bs_third_order_greeks,
    DAYS_PER_YEAR,
)

from .validation import (
    # Exceptions
    ValidationError,
    ParameterOutOfRangeError,
    ArbitrageViolationError,
    FellerConditionError,
    # Basic validation
    validate_positive,
    validate_in_range,
    validate_finite,
    # Market parameters
    validate_spot,
    validate_strike,
    validate_maturity,
    validate_rate,
    validate_volatility,
    validate_vanilla_option,
    # Model validation
    validate_heston_parameters,
    validate_correlation,
    # Arbitrage
    validate_no_arbitrage,
    check_put_call_parity,
)

__all__ = [
    # Math
    "norm_cdf",
    "norm_pdf",
    "norm_cdf_vec",
    "norm_pdf_vec",
    "d1_d2",
    "bs_second_order_greeks",
    "bs_third_order_greeks",
    "DAYS_PER_YEAR",
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
]
