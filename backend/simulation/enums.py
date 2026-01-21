"""
Enumerations for Simulation Models
==================================

This module defines enumerations for model types used throughout
the simulation framework.

Author: Thomas
Created: 2025
"""

from enum import Enum, auto


class ModelType(Enum):
    """
    Enumeration of available simulation models.

    Categories:
    - Diffusion: GBM (constant volatility)
    - Stochastic Volatility: Heston, Bates
    - Jump Diffusion: Merton
    - GARCH Family: GARCH, NGARCH, GJR_GARCH
    """

    # Diffusion models
    GBM = "Geometric Brownian Motion"

    # Stochastic volatility models
    HESTON = "Heston Stochastic Volatility"
    BATES = "Bates (Heston + Jumps)"

    # Jump diffusion models
    MERTON = "Merton Jump Diffusion"

    # GARCH family models
    GARCH = "GARCH(1,1)"
    NGARCH = "NGARCH (Nonlinear Asymmetric)"
    GJR_GARCH = "GJR-GARCH"

    @classmethod
    def continuous_time_models(cls) -> list:
        """Returns list of continuous-time models."""
        return [cls.GBM, cls.HESTON, cls.BATES, cls.MERTON]

    @classmethod
    def discrete_time_models(cls) -> list:
        """Returns list of discrete-time (GARCH) models."""
        return [cls.GARCH, cls.NGARCH, cls.GJR_GARCH]

    @classmethod
    def stochastic_vol_models(cls) -> list:
        """Returns list of models with stochastic volatility output."""
        return [cls.HESTON, cls.BATES, cls.GARCH, cls.NGARCH, cls.GJR_GARCH]

    @classmethod
    def jump_models(cls) -> list:
        """Returns list of models with jump components."""
        return [cls.MERTON, cls.BATES]


class DiscretizationScheme(Enum):
    """
    Discretization schemes for stochastic volatility models.

    Used primarily for Heston and Bates models where variance
    can become negative under naive Euler discretization.
    """

    EULER = auto()           # Simple Euler (can have negative variance)
    FULL_TRUNCATION = auto()  # Variance floored at 0
    REFLECTION = auto()       # Negative variance reflected
    QE = auto()              # Quadratic Exponential (most accurate)

    @classmethod
    def default(cls) -> "DiscretizationScheme":
        """Returns the recommended default scheme."""
        return cls.FULL_TRUNCATION


class Measure(Enum):
    """
    Probability measure for simulation.

    P-measure (physical/real-world) uses expected return mu as drift.
    Q-measure (risk-neutral) uses risk-free rate r as drift.
    """

    P_MEASURE = "Physical (Real-World)"
    Q_MEASURE = "Risk-Neutral"
