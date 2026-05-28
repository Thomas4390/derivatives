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

Author: Thomas Vaudescal
Created: 2026
"""

# =============================================================================
# Core Classes (always loaded)
# =============================================================================

from backend.simulation.base import (
    BaseSimulator,
    SimulationResult,
    StochasticVolatilityMixin,
)

# =============================================================================
# Enumerations (always loaded)
# =============================================================================
from backend.simulation.enums import DiscretizationScheme, Measure, ModelType

# =============================================================================
# Lazy imports to avoid conflicts when running modules directly with -m
# =============================================================================


def __getattr__(name: str):
    """Lazy import for simulators and convenience functions."""
    # Model Implementations
    if name == "GBMSimulator":
        from backend.simulation.models.gbm import GBMSimulator

        return GBMSimulator
    if name == "HestonSimulator":
        from backend.simulation.models.heston import HestonSimulator

        return HestonSimulator
    if name == "MertonSimulator":
        from backend.simulation.models.merton import MertonSimulator

        return MertonSimulator
    if name == "BatesSimulator":
        from backend.simulation.models.bates import BatesSimulator

        return BatesSimulator
    if name == "GARCHSimulator":
        from backend.simulation.models.garch import GARCHSimulator

        return GARCHSimulator
    if name == "NGARCHSimulator":
        from backend.simulation.models.ngarch import NGARCHSimulator

        return NGARCHSimulator
    if name == "GJRGARCHSimulator":
        from backend.simulation.models.gjr_garch import GJRGARCHSimulator

        return GJRGARCHSimulator

    # Convenience Functions
    if name == "simulate_gbm":
        from backend.simulation.models.gbm import simulate_gbm

        return simulate_gbm
    if name == "simulate_heston":
        from backend.simulation.models.heston import simulate_heston

        return simulate_heston
    if name == "simulate_merton":
        from backend.simulation.models.merton import simulate_merton

        return simulate_merton
    if name == "simulate_bates":
        from backend.simulation.models.bates import simulate_bates

        return simulate_bates
    if name == "simulate_garch":
        from backend.simulation.models.garch import simulate_garch

        return simulate_garch
    if name == "estimate_garch_params":
        from backend.simulation.models.garch import estimate_garch_params

        return estimate_garch_params
    if name == "simulate_ngarch":
        from backend.simulation.models.ngarch import simulate_ngarch

        return simulate_ngarch
    if name == "simulate_gjr_garch":
        from backend.simulation.models.gjr_garch import simulate_gjr_garch

        return simulate_gjr_garch

    # Factory Functions
    if name == "create_simulator":
        from backend.simulation.factory import create_simulator

        return create_simulator
    if name == "create_gbm":
        from backend.simulation.factory import create_gbm

        return create_gbm
    if name == "create_heston":
        from backend.simulation.factory import create_heston

        return create_heston
    if name == "create_merton":
        from backend.simulation.factory import create_merton

        return create_merton
    if name == "create_bates":
        from backend.simulation.factory import create_bates

        return create_bates
    if name == "create_garch":
        from backend.simulation.factory import create_garch

        return create_garch
    if name == "create_ngarch":
        from backend.simulation.factory import create_ngarch

        return create_ngarch
    if name == "create_gjr_garch":
        from backend.simulation.factory import create_gjr_garch

        return create_gjr_garch
    if name == "list_models":
        from backend.simulation.factory import list_models

        return list_models
    if name == "get_model_info":
        from backend.simulation.factory import get_model_info

        return get_model_info

    # P&L Engine (Re-exported from portfolio.pnl for backward compatibility)
    if name == "RiskMetrics":
        from backend.portfolio.pnl import RiskMetrics

        return RiskMetrics
    if name == "calculate_portfolio_pnl_vectorized":
        from backend.portfolio.pnl import calculate_portfolio_pnl_vectorized

        return calculate_portfolio_pnl_vectorized
    if name == "calculate_portfolio_pnl_with_stock":
        from backend.portfolio.pnl import calculate_portfolio_pnl_with_stock

        return calculate_portfolio_pnl_with_stock
    if name == "compute_risk_metrics":
        from backend.portfolio.pnl import compute_risk_metrics

        return compute_risk_metrics
    if name == "compute_risk_metrics_core":
        from backend.portfolio.pnl import compute_risk_metrics_core

        return compute_risk_metrics_core
    if name == "compute_skewness_kurtosis":
        from backend.portfolio.pnl import compute_skewness_kurtosis

        return compute_skewness_kurtosis
    if name == "compute_percentiles":
        from backend.portfolio.pnl import compute_percentiles

        return compute_percentiles
    if name == "compute_payoff_curve":
        from backend.portfolio.pnl import compute_payoff_curve

        return compute_payoff_curve
    if name == "find_breakeven_points":
        from backend.portfolio.pnl import find_breakeven_points

        return find_breakeven_points
    if name == "prepare_position_arrays":
        from backend.portfolio.pnl import prepare_position_arrays

        return prepare_position_arrays
    if name == "warm_up_jit":
        from backend.portfolio.pnl import warm_up_jit

        return warm_up_jit

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
