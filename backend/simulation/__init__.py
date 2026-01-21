"""
Monte Carlo Simulation Framework
================================

High-performance Monte Carlo simulation for derivatives pricing and risk analysis.

This module provides:
- Multiple stochastic process models (GBM, Heston, Merton, Bates, GARCH family)
- Numba-optimized implementations for maximum performance
- Clean OOP interface with Strategy pattern
- Unified API across all models

Continuous-Time Models:
- GBMSimulator: Geometric Brownian Motion
- HestonSimulator: Heston Stochastic Volatility
- MertonSimulator: Merton Jump Diffusion
- BatesSimulator: Bates (Heston + Jumps)

Discrete-Time (GARCH Family):
- GARCHSimulator: GARCH(1,1)
- NGARCHSimulator: NGARCH with leverage effect
- GJRGARCHSimulator: GJR-GARCH with asymmetry

P-Measure vs Q-Measure:
Default simulations use the physical (P) measure with drift = μ.
For risk-neutral (Q) pricing, pass mu=r (risk-free rate).

Author: Thomas
Created: 2025
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
