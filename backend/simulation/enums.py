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


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Simulation Enums Smoke Test")
    print("=" * 50)

    # Test ModelType
    print("\n--- ModelType Enumeration ---")
    print("All models:")
    for model in ModelType:
        print(f"  {model.name}: {model.value}")

    # Test classification methods
    print("\n--- Model Classifications ---")

    continuous = ModelType.continuous_time_models()
    print(f"Continuous-time models: {[m.name for m in continuous]}")
    assert ModelType.GBM in continuous
    assert ModelType.HESTON in continuous
    assert ModelType.BATES in continuous
    assert ModelType.MERTON in continuous

    discrete = ModelType.discrete_time_models()
    print(f"Discrete-time models: {[m.name for m in discrete]}")
    assert ModelType.GARCH in discrete
    assert ModelType.NGARCH in discrete
    assert ModelType.GJR_GARCH in discrete

    stoch_vol = ModelType.stochastic_vol_models()
    print(f"Stochastic volatility models: {[m.name for m in stoch_vol]}")
    assert ModelType.HESTON in stoch_vol
    assert ModelType.BATES in stoch_vol
    assert ModelType.GARCH in stoch_vol
    assert ModelType.GBM not in stoch_vol

    jump = ModelType.jump_models()
    print(f"Jump models: {[m.name for m in jump]}")
    assert ModelType.MERTON in jump
    assert ModelType.BATES in jump
    assert ModelType.GBM not in jump

    # Test DiscretizationScheme
    print("\n--- DiscretizationScheme Enumeration ---")
    print("All schemes:")
    for scheme in DiscretizationScheme:
        print(f"  {scheme.name}")

    default_scheme = DiscretizationScheme.default()
    print(f"Default scheme: {default_scheme.name}")
    assert default_scheme == DiscretizationScheme.FULL_TRUNCATION

    # Test Measure
    print("\n--- Measure Enumeration ---")
    print("All measures:")
    for measure in Measure:
        print(f"  {measure.name}: {measure.value}")

    # Verify expected values
    print("\n--- Consistency Checks ---")

    # Bates should be in both stochastic vol AND jump models
    assert ModelType.BATES in stoch_vol and ModelType.BATES in jump
    print("Bates is stochastic vol AND has jumps: ✓")

    # GBM should be only continuous, no stoch vol, no jumps
    assert ModelType.GBM in continuous
    assert ModelType.GBM not in stoch_vol
    assert ModelType.GBM not in jump
    print("GBM is continuous only: ✓")

    # All models should be either continuous or discrete
    all_models = set(ModelType)
    classified = set(continuous) | set(discrete)
    assert all_models == classified
    print("All models classified: ✓")

    print("\n" + "=" * 50)
    print("Simulation Enums smoke test passed")
    print("=" * 50)
