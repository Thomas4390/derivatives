"""
Models Module
=============

Financial models for derivatives pricing.

Architecture:
- Immutable frozen dataclasses
- Parameters merged directly into model class
- Provides characteristic_function() for FFT pricing
- Provides drift()/diffusion() for Monte Carlo

Available Models:
- GBMModel: Geometric Brownian Motion (Black-Scholes)
- HestonModel: Heston stochastic volatility
- MertonModel: Merton jump-diffusion
- BatesModel: Bates (Heston + jumps)
- GARCHModel: GARCH(1,1) volatility model
- NGARCHModel: Nonlinear Asymmetric GARCH
- GJRGARCHModel: GJR-GARCH with leverage effect

Author: Thomas
Created: 2025
Version: 3.0.0
"""

# =============================================================================
# NEW ARCHITECTURE (Model ABC from core)
# =============================================================================

# Core model classes (new architecture)
from backend.models.gbm import GBMModel
from backend.models.heston import HestonModel
from backend.models.merton import MertonModel
from backend.models.bates import BatesModel

# GARCH family models (now also using merged parameters)
from backend.models.garch import (
    GARCHModel,
    NGARCHModel,
    GJRGARCHModel,
    # Backward compatibility aliases (params classes now alias model classes)
    GARCHParams,
    NGARCHParams,
    GJRGARCHParams,
)

# Characteristic functions (Numba-optimized)
from backend.models.characteristic_functions import (
    heston_characteristic_function,
    heston_cf_vectorized,
    bates_characteristic_function,
    bates_cf_vectorized,
    merton_characteristic_function,
    merton_cf_vectorized,
)

# =============================================================================
# Base Classes and Registry
# =============================================================================

from backend.models.base import BaseModel, Measure, PricingCapability
from backend.models.registry import registry, ModelRegistry


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # New architecture models
    "GBMModel",
    "HestonModel",
    "MertonModel",
    "BatesModel",
    # GARCH family
    "GARCHModel",
    "NGARCHModel",
    "GJRGARCHModel",
    # GARCH params (aliases)
    "GARCHParams",
    "NGARCHParams",
    "GJRGARCHParams",
    # Characteristic functions
    "heston_characteristic_function",
    "heston_cf_vectorized",
    "bates_characteristic_function",
    "bates_cf_vectorized",
    "merton_characteristic_function",
    "merton_cf_vectorized",
    # Base classes
    "BaseModel",
    "Measure",
    "PricingCapability",
    # Registry
    "registry",
    "ModelRegistry",
]

__version__ = "3.0.0"
