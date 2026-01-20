"""
Unified Models Module
=====================

Central module for unified financial models.

This module provides:
    - Immutable parameter containers for all models
    - Unified model classes that create both simulators and pricers
    - Registry pattern for model discovery and creation

Usage:
    # Direct model creation
    from backend.models import HestonModel
    model = HestonModel.from_params(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    simulator = model.create_simulator()
    pricer = model.create_pricer(r=0.05)

    # Registry-based creation
    from backend.models import registry
    simulator = registry.create_simulator("heston", v0=0.04, kappa=2.0, ...)
    pricer = registry.create_pricer("heston", v0=0.04, kappa=2.0, ...)

    # List available models
    print(registry.list_models())

Author: Derivatives Pricing Project
Version: 1.0.0
"""

# Base classes and enums
from .base import BaseModel, Measure, PricingCapability
from .registry import registry, ModelRegistry

# Parameter containers
from .parameters import (
    BaseParams,
    GBMParams,
    HestonParams,
    MertonParams,
    BatesParams,
    JumpParams,
    GARCHParams,
    NGARCHParams,
    GJRGARCHParams,
)

# Model classes
from .gbm import GBMModel
from .heston import HestonModel
from .merton import MertonModel
from .bates import BatesModel
from .garch import GARCHModel, NGARCHModel, GJRGARCHModel

# Characteristic functions
from .characteristic_functions import (
    heston_characteristic_function,
    heston_cf_vectorized,
    bates_characteristic_function,
    bates_cf_vectorized,
)

# =============================================================================
# Register all models
# =============================================================================

registry.register("gbm", GBMModel, aliases=["bs", "black_scholes", "blackscholes"])
registry.register("heston", HestonModel, aliases=["sv", "stochvol"])
registry.register("merton", MertonModel, aliases=["jump_diffusion", "jd"])
registry.register("bates", BatesModel, aliases=["sv_jumps", "heston_jumps"])
registry.register("garch", GARCHModel, aliases=["garch11"])
registry.register("ngarch", NGARCHModel, aliases=["nonlinear_garch"])
registry.register("gjr_garch", GJRGARCHModel, aliases=["gjrgarch", "tgarch"])


__all__ = [
    # Base
    "BaseModel",
    "Measure",
    "PricingCapability",
    "registry",
    "ModelRegistry",
    # Parameters
    "BaseParams",
    "GBMParams",
    "HestonParams",
    "MertonParams",
    "BatesParams",
    "JumpParams",
    "GARCHParams",
    "NGARCHParams",
    "GJRGARCHParams",
    # Models
    "GBMModel",
    "HestonModel",
    "MertonModel",
    "BatesModel",
    "GARCHModel",
    "NGARCHModel",
    "GJRGARCHModel",
    # Characteristic functions
    "heston_characteristic_function",
    "heston_cf_vectorized",
    "bates_characteristic_function",
    "bates_cf_vectorized",
]

__version__ = "1.0.0"
