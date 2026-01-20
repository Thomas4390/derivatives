"""
Monte Carlo Simulation Framework
================================

High-performance Monte Carlo simulation for derivatives pricing and risk analysis.

This package provides:
- Multiple stochastic process models (GBM, Heston, Merton, Bates, GARCH family)
- Numba-optimized implementations for maximum performance
- Clean OOP interface with Strategy pattern
- Unified API across all models

P-Measure (Physical Measure) Implementation:
--------------------------------------------
All simulations use the REAL-WORLD (P) measure with drift = μ (expected return).
This is appropriate for:
    - Scenario analysis and risk management
    - VaR and stress testing
    - Backtesting trading strategies
    - Generating realistic price trajectories

For DERIVATIVES PRICING, use the Q-measure (risk-neutral) where drift = r.
Simply pass mu=r (risk-free rate) to use risk-neutral dynamics.

Models Available:
-----------------
Continuous-Time:
    - GBMSimulator: Geometric Brownian Motion
    - HestonSimulator: Heston Stochastic Volatility
    - MertonSimulator: Merton Jump Diffusion
    - BatesSimulator: Bates (Heston + Jumps)

Discrete-Time (GARCH Family):
    - GARCHSimulator: GARCH(1,1)
    - NGARCHSimulator: NGARCH with leverage effect
    - GJRGARCHSimulator: GJR-GARCH with asymmetry

Quick Start:
------------
>>> from backend.simulation import GBMSimulator, HestonSimulator, create_simulator
>>>
>>> # Direct instantiation
>>> sim = GBMSimulator(sigma=0.20)
>>> result = sim.simulate_paths(s0=100, mu=0.08, t=1.0, n_paths=10000, n_steps=252)
>>>
>>> # Using factory
>>> sim = create_simulator("heston", v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7)
>>> result = sim.simulate_paths(s0=100, mu=0.08, t=1.0, n_paths=10000, n_steps=252)

Author: Derivatives Pricing Project
"""

# =============================================================================
# Core Classes
# =============================================================================

from .base import BaseSimulator, SimulationResult, StochasticVolatilityMixin

# =============================================================================
# Enumerations
# =============================================================================

from .enums import ModelType, DiscretizationScheme, Measure

# =============================================================================
# Model Implementations
# =============================================================================

from .models import (
    GBMSimulator,
    HestonSimulator,
    MertonSimulator,
    BatesSimulator,
    GARCHSimulator,
    NGARCHSimulator,
    GJRGARCHSimulator,
)

# =============================================================================
# Convenience Functions
# =============================================================================

from .models.gbm import simulate_gbm
from .models.heston import simulate_heston
from .models.merton import simulate_merton
from .models.bates import simulate_bates
from .models.garch import simulate_garch, estimate_garch_params
from .models.ngarch import simulate_ngarch
from .models.gjr_garch import simulate_gjr_garch

# =============================================================================
# Factory Functions
# =============================================================================

from .factory import (
    create_simulator,
    create_gbm,
    create_heston,
    create_merton,
    create_bates,
    create_garch,
    create_ngarch,
    create_gjr_garch,
    list_models,
    get_model_info,
)

# =============================================================================
# P&L Engine (Kept Separate)
# =============================================================================

from .pnl_engine import (
    # Data classes
    RiskMetrics,
    # Core P&L calculation (Numba-optimized)
    calculate_portfolio_pnl_vectorized,
    calculate_portfolio_pnl_with_stock,
    # Risk metrics
    compute_risk_metrics,
    compute_risk_metrics_core,
    compute_skewness_kurtosis,
    compute_percentiles,
    # Payoff analysis
    compute_payoff_curve,
    find_breakeven_points,
    # Utilities
    prepare_position_arrays,
    warm_up_jit,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Core classes
    "BaseSimulator",
    "SimulationResult",
    "StochasticVolatilityMixin",
    # Enums
    "ModelType",
    "DiscretizationScheme",
    "Measure",
    # Simulator classes
    "GBMSimulator",
    "HestonSimulator",
    "MertonSimulator",
    "BatesSimulator",
    "GARCHSimulator",
    "NGARCHSimulator",
    "GJRGARCHSimulator",
    # Convenience functions
    "simulate_gbm",
    "simulate_heston",
    "simulate_merton",
    "simulate_bates",
    "simulate_garch",
    "simulate_ngarch",
    "simulate_gjr_garch",
    "estimate_garch_params",
    # Factory functions
    "create_simulator",
    "create_gbm",
    "create_heston",
    "create_merton",
    "create_bates",
    "create_garch",
    "create_ngarch",
    "create_gjr_garch",
    "list_models",
    "get_model_info",
    # P&L Engine
    "RiskMetrics",
    "calculate_portfolio_pnl_vectorized",
    "calculate_portfolio_pnl_with_stock",
    "compute_risk_metrics",
    "compute_risk_metrics_core",
    "compute_skewness_kurtosis",
    "compute_percentiles",
    "compute_payoff_curve",
    "find_breakeven_points",
    "prepare_position_arrays",
    "warm_up_jit",
]

__version__ = "3.0.0"  # Major refactor: OOP architecture with Strategy pattern
