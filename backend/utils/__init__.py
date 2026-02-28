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

Author: Thomas
Created: 2025
"""

from backend.utils.math import (
    # Constants
    DAYS_PER_YEAR,
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
from backend.utils.validation import (
    ArbitrageViolationError,
    FellerConditionError,
    ParameterOutOfRangeError,
    # Exceptions
    ValidationError,
    check_put_call_parity,
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
    # Constants
    "DAYS_PER_YEAR",
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
]
