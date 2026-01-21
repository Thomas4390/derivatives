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

Author: Thomas
Created: 2025
"""

# =============================================================================
# Position Classes and Factory Functions
# =============================================================================

from .positions import (
    # Classes
    PortfolioPosition,
    StockPosition,
    # Factory functions
    long_call,
    short_call,
    long_put,
    short_put,
    long_stock,
    short_stock,
)

# =============================================================================
# Breakeven Analysis
# =============================================================================

from .breakeven import (
    BreakevenResult,
    BreakevenCalculator,
    find_breakevens,
    find_breakevens_from_portfolio,
    calculate_portfolio_pnl_at_expiry,
)

# =============================================================================
# Main Portfolio Class
# =============================================================================

from .portfolio import OptionsPortfolio

# =============================================================================
# Re-export GreeksResult from core (for convenience)
# =============================================================================

from backend.core.result_types import GreeksResult

# =============================================================================
# Numba-Optimized Risk Metrics
# =============================================================================

from backend.simulation.pnl_engine import RiskMetrics


__all__ = [
    # Main class
    "OptionsPortfolio",
    # Position classes
    "PortfolioPosition",
    "StockPosition",
    # Factory functions
    "long_call",
    "short_call",
    "long_put",
    "short_put",
    "long_stock",
    "short_stock",
    # Greeks (from core)
    "GreeksResult",
    # Risk metrics (Numba-optimized)
    "RiskMetrics",
    # Breakeven
    "BreakevenResult",
    "BreakevenCalculator",
    "find_breakevens",
    "find_breakevens_from_portfolio",
    "calculate_portfolio_pnl_at_expiry",
]

__version__ = "2.0.0"
