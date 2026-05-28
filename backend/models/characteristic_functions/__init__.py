"""
Characteristic Functions Module
================================

Shared characteristic function implementations for FFT pricing.

These are the SINGLE source of truth for characteristic functions,
used by both:
    - FFT pricers (Carr-Madan method)
    - Model calibration routines

All implementations are Numba-optimized for performance.

Author: Thomas Vaudescal
Created: 2026
"""

from backend.models.characteristic_functions.bates_cf import (
    bates_cf_vectorized,
    bates_characteristic_function,
)
from backend.models.characteristic_functions.heston_cf import (
    heston_cf_vectorized,
    heston_characteristic_function,
)
from backend.models.characteristic_functions.heston_nandi_cf import (
    heston_nandi_cf_vectorized,
    heston_nandi_characteristic_function,
)
from backend.models.characteristic_functions.merton_cf import (
    create_merton_cf,
    merton_cf_vectorized,
    merton_characteristic_function,
)

__all__ = [
    "heston_characteristic_function",
    "heston_cf_vectorized",
    "heston_nandi_characteristic_function",
    "heston_nandi_cf_vectorized",
    "bates_characteristic_function",
    "bates_cf_vectorized",
    "merton_characteristic_function",
    "merton_cf_vectorized",
    "create_merton_cf",
]
