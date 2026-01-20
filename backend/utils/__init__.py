"""
Utilities Module
================

Shared utilities used across the backend.
All functions are Numba-optimized for performance.

Contents:
- math: Normal distribution functions, Black-Scholes parameters, higher-order Greeks
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

__all__ = [
    "norm_cdf",
    "norm_pdf",
    "norm_cdf_vec",
    "norm_pdf_vec",
    "d1_d2",
    "bs_second_order_greeks",
    "bs_third_order_greeks",
    "DAYS_PER_YEAR",
]
