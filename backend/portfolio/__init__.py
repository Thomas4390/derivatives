"""
Portfolio Module
================

Portfolio management with positions, Greeks calculation, and breakeven analysis.

This module provides:
- OptionsPortfolio: Main portfolio class
- PortfolioPosition, StockPosition: Position classes
- Factory functions: long_call, short_call, long_put, short_put, long_stock, short_stock
- Breakeven analysis: BreakevenResult, BreakevenCalculator, find_breakevens

Uses the Model/Engine/Market architecture for pricing.

Author: Thomas Vaudescal
Created: 2026
"""

# =============================================================================
# Position Classes and Factory Functions
# =============================================================================

# =============================================================================
# Re-export GreeksResult from core (for convenience)
# =============================================================================
from backend.core.result_types import GreeksResult

# =============================================================================
# Breakeven Analysis
# =============================================================================
from backend.portfolio.breakeven import (
    BreakevenCalculator,
    BreakevenResult,
    calculate_portfolio_pnl_at_expiry,
    find_breakevens,
    find_breakevens_from_portfolio,
)

# =============================================================================
# Greeks Surface Calculations (Numba-parallel)
# =============================================================================
from backend.portfolio.greeks_surfaces import (
    GREEK_CHARM,
    GREEK_COLOR,
    GREEK_DELTA,
    GREEK_GAMMA,
    # Greek indices
    GREEK_PRICE,
    GREEK_RHO,
    GREEK_SPEED,
    GREEK_THETA,
    GREEK_ULTIMA,
    GREEK_VANNA,
    GREEK_VEGA,
    GREEK_VETA,
    GREEK_VOLGA,
    GREEK_ZOMMA,
    # P&L functions (Numba-optimized)
    calculate_pnl_curve,
    calculate_portfolio_pnl_at_expiry_arrays,
    get_greek_name,
    portfolio_greeks_surface_dte,
    portfolio_greeks_surface_iv,
    single_option_greeks_surface_strike,
)

# =============================================================================
# P&L Engine (Numba-Optimized)
# =============================================================================
from backend.portfolio.pnl import (
    RiskMetrics,
    calculate_portfolio_pnl_vectorized,
    calculate_portfolio_pnl_with_stock,
    compute_payoff_curve,
    compute_percentiles,
    compute_risk_metrics,
    compute_risk_metrics_core,
    compute_skewness_kurtosis,
    prepare_position_arrays,
    warm_up_jit,
)

# =============================================================================
# Main Portfolio Class
# =============================================================================
from backend.portfolio.portfolio import OptionsPortfolio
from backend.portfolio.positions import (
    # Classes
    PortfolioPosition,
    StockPosition,
    StructuredProductPosition,
    # Factory functions
    long_call,
    long_put,
    long_stock,
    short_call,
    short_put,
    short_stock,
)

# =============================================================================
# Risk Analysis
# =============================================================================
from backend.portfolio.risk_analysis import (
    RiskProfile,
    analyze_portfolio_risk,
    analyze_portfolio_risk_from_portfolio,
    check_unlimited_risk,
    check_unlimited_risk_from_portfolio,
    get_risk_summary,
)

__all__ = [
    # Main class
    "OptionsPortfolio",
    # Position classes
    "PortfolioPosition",
    "StockPosition",
    "StructuredProductPosition",
    # Factory functions
    "long_call",
    "short_call",
    "long_put",
    "short_put",
    "long_stock",
    "short_stock",
    # Greeks (from core)
    "GreeksResult",
    # P&L Engine (from pnl.py)
    "RiskMetrics",
    "calculate_portfolio_pnl_vectorized",
    "calculate_portfolio_pnl_with_stock",
    "compute_risk_metrics",
    "compute_risk_metrics_core",
    "compute_skewness_kurtosis",
    "compute_percentiles",
    "compute_payoff_curve",
    "prepare_position_arrays",
    "warm_up_jit",
    # Breakeven
    "BreakevenResult",
    "BreakevenCalculator",
    "find_breakevens",
    "find_breakevens_from_portfolio",
    "calculate_portfolio_pnl_at_expiry",
    # Risk Analysis
    "RiskProfile",
    "check_unlimited_risk",
    "check_unlimited_risk_from_portfolio",
    "analyze_portfolio_risk",
    "analyze_portfolio_risk_from_portfolio",
    "get_risk_summary",
    # Greeks surfaces
    "portfolio_greeks_surface_dte",
    "portfolio_greeks_surface_iv",
    "single_option_greeks_surface_strike",
    "get_greek_name",
    # P&L functions (from greeks_surfaces.py)
    "calculate_pnl_curve",
    "calculate_portfolio_pnl_at_expiry_arrays",
    # Greek indices
    "GREEK_PRICE",
    "GREEK_DELTA",
    "GREEK_GAMMA",
    "GREEK_VEGA",
    "GREEK_THETA",
    "GREEK_RHO",
    "GREEK_VANNA",
    "GREEK_VOLGA",
    "GREEK_CHARM",
    "GREEK_VETA",
    "GREEK_SPEED",
    "GREEK_ZOMMA",
    "GREEK_COLOR",
    "GREEK_ULTIMA",
]

__version__ = "2.1.0"  # Added pnl.py module
