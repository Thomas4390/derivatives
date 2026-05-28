"""
Fourier Pricing Engines
=======================

FFT-based option pricing engines for models with
known characteristic functions.

Contains:
- CarrMadanFFTEngine: Generic Carr-Madan FFT engine
- FFTConfig: FFT configuration dataclass
- Characteristic functions for all supported models

Note: The high-level FFTEngine wrapper is in backend.engines.fft_engine
and is exported from backend.engines directly.

Author: Thomas Vaudescal
Created: 2026
"""

# Import low-level engine and config from carr_madan
from backend.engines.fourier.carr_madan import (
    CarrMadanFFTEngine,
    FFTConfig,
    fft_price,
)

# Re-export characteristic functions for convenience
# (Primary location remains in models.characteristic_functions)
from backend.models.characteristic_functions import (
    bates_cf_vectorized,
    bates_characteristic_function,
    heston_cf_vectorized,
    heston_characteristic_function,
    merton_cf_vectorized,
    merton_characteristic_function,
)

__all__ = [
    # Engines
    "CarrMadanFFTEngine",
    "FFTConfig",
    "fft_price",
    # Characteristic functions
    "heston_characteristic_function",
    "heston_cf_vectorized",
    "bates_characteristic_function",
    "bates_cf_vectorized",
    "merton_characteristic_function",
    "merton_cf_vectorized",
]
