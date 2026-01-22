"""
Characteristic Functions Module
================================

Shared characteristic function implementations for FFT pricing.

These are the SINGLE source of truth for characteristic functions,
used by both:
    - FFT pricers (Carr-Madan method)
    - Model calibration routines

All implementations are Numba-optimized for performance.

Author: Thomas
Created: 2025
"""

from backend.models.characteristic_functions.heston_cf import (
    heston_characteristic_function,
    heston_cf_vectorized,
)
from backend.models.characteristic_functions.bates_cf import (
    bates_characteristic_function,
    bates_cf_vectorized,
)
from backend.models.characteristic_functions.merton_cf import (
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
