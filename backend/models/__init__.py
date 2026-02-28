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
# PricingCapability is canonical in backend.core.result_types
from backend.core.result_types import PricingCapability

# =============================================================================
# Base Classes and Registry
# =============================================================================
from backend.models.base import Measure
from backend.models.bates import BatesModel

# Characteristic functions (Numba-optimized)
from backend.models.characteristic_functions import (
    bates_cf_vectorized,
    bates_characteristic_function,
    heston_cf_vectorized,
    heston_characteristic_function,
    merton_cf_vectorized,
    merton_characteristic_function,
)

# GARCH family models (now also using merged parameters)
from backend.models.garch import (
    BaseGARCHModel,
    GARCHModel,
    # Backward compatibility aliases (params classes now alias model classes)
    GARCHParams,
    GJRGARCHModel,
    GJRGARCHParams,
    NGARCHModel,
    NGARCHParams,
)
from backend.models.gbm import GBMModel
from backend.models.heston import HestonModel
from backend.models.merton import MertonModel
from backend.models.registry import ModelRegistry, registry

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
    "BaseGARCHModel",
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
    "Measure",
    "PricingCapability",
    # Registry
    "registry",
    "ModelRegistry",
]

__version__ = "3.0.0"
