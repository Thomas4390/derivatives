"""
Portfolio Module
================

Portfolio management with positions, Greeks calculation, and breakeven analysis.

Quick Start
-----------
    from backend.portfolio import (
        # Main class
        OptionsPortfolio,
        # Position classes
        OptionPosition, StockPosition,
        OptionType, PositionType,
        # Greeks
        GreeksResult, GreeksCalculator,
        # Breakeven
        BreakevenResult,
    )

Example
-------
    >>> portfolio = OptionsPortfolio(sigma=0.20)
    >>> portfolio.add_option(OptionPosition('call', 'long', strike=100, premium=5.0))
    >>> greeks = portfolio.calculate_greeks(spot=100, rate=0.05, time_to_expiry=0.5)
    >>> print(f"Delta: {greeks.delta:.4f}")

Author: Derivatives Pricing Project
"""

# Position classes and enums
from .positions import (
    OptionPosition,
    StockPosition,
    OptionType,
    PositionType,
)

# Greeks calculation
from .greeks import (
    GreeksResult,
    GreeksStrategy,
    GreeksCalculator,
    AnalyticalGreeksStrategy,
    FiniteDiffGreeksStrategy,
    # Higher-order Greeks (Black-Scholes analytical)
    calculate_bs_second_order_greeks,
    calculate_bs_third_order_greeks,
    DAYS_PER_YEAR,
    # Standalone functions for frontend
    calculate_all_greeks,
    calculate_portfolio_greeks_3d_dte,
    calculate_portfolio_greeks_3d_iv,
    calculate_greeks_3d_strike,
)

# Breakeven analysis
from .breakeven import (
    BreakevenResult,
    BreakevenCalculator,
    find_breakevens,
    find_breakeven_points,  # Legacy-compatible array-based interface
    calculate_portfolio_pnl_at_expiry,
    calculate_pnl_at_expiry_arrays,  # Array-based interface for frontend
)

# Main portfolio class
from .portfolio import OptionsPortfolio


__all__ = [
    # Main class
    "OptionsPortfolio",
    # Positions
    "OptionPosition",
    "StockPosition",
    "OptionType",
    "PositionType",
    # Greeks
    "GreeksResult",
    "GreeksStrategy",
    "GreeksCalculator",
    "AnalyticalGreeksStrategy",
    "FiniteDiffGreeksStrategy",
    "calculate_bs_second_order_greeks",
    "calculate_bs_third_order_greeks",
    "DAYS_PER_YEAR",
    # Standalone functions for frontend
    "calculate_all_greeks",
    "calculate_portfolio_greeks_3d_dte",
    "calculate_portfolio_greeks_3d_iv",
    "calculate_greeks_3d_strike",
    # Breakeven
    "BreakevenResult",
    "BreakevenCalculator",
    "find_breakevens",
    "find_breakeven_points",
    "calculate_portfolio_pnl_at_expiry",
    "calculate_pnl_at_expiry_arrays",
]
