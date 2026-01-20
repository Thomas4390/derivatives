"""
Characteristic Functions Module
================================

Shared characteristic function implementations for FFT pricing.

These are the SINGLE source of truth for characteristic functions,
used by both:
    - FFT pricers (Carr-Madan method)
    - Model calibration routines

All implementations are Numba-optimized for performance.

Author: Derivatives Pricing Project
"""

from .heston_cf import (
    heston_characteristic_function,
    heston_cf_vectorized,
)
from .bates_cf import (
    bates_characteristic_function,
    bates_cf_vectorized,
)
from .merton_cf import (
    merton_characteristic_function,
    merton_cf_vectorized,
    create_merton_cf,
)

__all__ = [
    "heston_characteristic_function",
    "heston_cf_vectorized",
    "bates_characteristic_function",
    "bates_cf_vectorized",
    "merton_characteristic_function",
    "merton_cf_vectorized",
    "create_merton_cf",
]
