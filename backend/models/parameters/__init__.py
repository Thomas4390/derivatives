"""
Model Parameters Module
=======================

Immutable parameter containers for all financial models.

All parameter classes use frozen dataclasses ensuring:
    - Immutability (thread-safe, hashable)
    - Validation at creation time
    - Single source of truth

Usage:
    from backend.models.parameters import HestonParams, GARCHParams

    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    print(params.feller_satisfied)  # True

Author: Derivatives Pricing Project
"""

from .base import BaseParams
from .gbm import GBMParams
from .heston import HestonParams
from .merton import JumpParams, MertonParams
from .bates import BatesParams
from .garch import GARCHParams, NGARCHParams, GJRGARCHParams

__all__ = [
    # Base
    "BaseParams",
    # Models
    "GBMParams",
    "HestonParams",
    "MertonParams",
    "BatesParams",
    "JumpParams",
    # GARCH family
    "GARCHParams",
    "NGARCHParams",
    "GJRGARCHParams",
]
