"""
Breakeven Analysis
==================

Find breakeven points and profit/loss zones for option portfolios.

This module provides:
- BreakevenResult: Container for breakeven analysis results
- BreakevenCalculator: Grid search calculator with Numba-optimized functions
- Convenience functions: find_breakevens, find_breakevens_from_portfolio

Uses Numba-optimized functions from pnl module for maximum performance.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

# Import Numba-optimized functions from portfolio.pnl (moved from simulation)
from backend.portfolio.pnl import (
    compute_payoff_curve as _compute_payoff_curve_numba,
)
from backend.portfolio.pnl import (
    find_breakeven_points as _find_breakeven_points_numba,
)
from backend.portfolio.positions import PortfolioPosition, StockPosition

if TYPE_CHECKING:
    from backend.portfolio.portfolio import OptionsPortfolio


# =============================================================================
# BREAKEVEN RESULT CONTAINER
# =============================================================================


@dataclass
class BreakevenResult:
    """
    Container for breakeven analysis results.

    Attributes
    ----------
    breakeven_points : List[float]
        Spot prices where P&L equals zero
    max_profit : float
        Maximum profit achievable
    max_profit_spot : float
        Spot price at maximum profit
    max_loss : float
        Maximum loss (negative value)
    max_loss_spot : float
        Spot price at maximum loss
    profit_zones : List[Tuple[float, float]]
        Ranges where portfolio is profitable
    loss_zones : List[Tuple[float, float]]
        Ranges where portfolio is at a loss
    """

    breakeven_points: list[float]
    max_profit: float
    max_profit_spot: float
    max_loss: float
    max_loss_spot: float
    profit_zones: list[tuple[float, float]]
    loss_zones: list[tuple[float, float]]

    def summary(self) -> str:
        """Generate a formatted summary of breakeven analysis."""
        lines = ["=== Breakeven Analysis ==="]

        if self.breakeven_points:
            formatted = [f"${b:.2f}" for b in self.breakeven_points]
            lines.append(f"Breakeven points: {formatted}")
        else:
            lines.append("No breakeven points (always profit or always loss)")

        lines.append(
            f"Max Profit: ${self.max_profit:.2f} at ${self.max_profit_spot:.2f}"
        )
        lines.append(f"Max Loss: ${self.max_loss:.2f} at ${self.max_loss_spot:.2f}")

        if self.profit_zones:
            lines.append("\nProfit zones:")
            for start, end in self.profit_zones:
                if start == -np.inf:
                    lines.append(f"  Below ${end:.2f}")
                elif end == np.inf:
                    lines.append(f"  Above ${start:.2f}")
                else:
                    lines.append(f"  ${start:.2f} - ${end:.2f}")

        if self.loss_zones:
            lines.append("\nLoss zones:")
            for start, end in self.loss_zones:
                if start == -np.inf:
                    lines.append(f"  Below ${end:.2f}")
                elif end == np.inf:
                    lines.append(f"  Above ${start:.2f}")
                else:
                    lines.append(f"  ${start:.2f} - ${end:.2f}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "breakeven_points": self.breakeven_points,
            "max_profit": self.max_profit,
            "max_profit_spot": self.max_profit_spot,
            "max_loss": self.max_loss,
            "max_loss_spot": self.max_loss_spot,
            "profit_zones": self.profit_zones,
            "loss_zones": self.loss_zones,
        }


# =============================================================================
# POSITION TO ARRAY CONVERSION (for Numba functions)
# =============================================================================


def _positions_to_arrays(
    positions: list[PortfolioPosition],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert portfolio positions to numpy arrays for Numba functions.

    Parameters
    ----------
    positions : List[PortfolioPosition]
        List of option positions

    Returns
    -------
    tuple
        (strikes, option_types, position_types, quantities, premiums)
        option_types: 1.0 = call, -1.0 = put
        position_types: 1.0 = long, -1.0 = short
    """
    n_legs = len(positions)

    strikes = np.zeros(n_legs, dtype=np.float64)
    option_types = np.zeros(n_legs, dtype=np.float64)
    position_types = np.zeros(n_legs, dtype=np.float64)
    quantities = np.zeros(n_legs, dtype=np.float64)
    premiums = np.zeros(n_legs, dtype=np.float64)

    for i, pos in enumerate(positions):
        strikes[i] = pos.strike
        option_types[i] = 1.0 if pos.is_call else -1.0
        position_types[i] = 1.0 if pos.is_long else -1.0
        quantities[i] = abs(pos.quantity)
        premiums[i] = pos.premium

    return strikes, option_types, position_types, quantities, premiums


# =============================================================================
# P&L CALCULATION (Numba-optimized)
# =============================================================================


def calculate_portfolio_pnl_at_expiry(
    spot: float,
    positions: list[PortfolioPosition],
    stock: StockPosition | None = None,
) -> float:
    """
    Calculate portfolio P&L at expiration for a given spot price.

    Uses Numba-optimized function for single spot evaluation.

    Parameters
    ----------
    spot : float
        Spot price at expiration
    positions : List[PortfolioPosition]
        List of option positions (new architecture)
    stock : StockPosition, optional
        Stock position if any

    Returns
    -------
    float
        Total P&L at expiration
    """
    if len(positions) == 0 and stock is None:
        return 0.0

    # Convert to arrays for Numba
    strikes, option_types, position_types, quantities, premiums = _positions_to_arrays(
        positions
    )

    # Stock parameters
    stock_qty = float(stock.quantity) if stock else 0.0
    stock_entry = float(stock.entry_price) if stock else 0.0

    # Use Numba-optimized function (single point)
    spot_arr = np.array([spot], dtype=np.float64)
    pnl_arr = _compute_payoff_curve_numba(
        spot_arr,
        strikes,
        option_types,
        position_types,
        quantities,
        premiums,
        stock_quantity=stock_qty,
        stock_entry_price=stock_entry,
        multiplier=1.0,  # No multiplier at position level
    )

    return float(pnl_arr[0])


# =============================================================================
# BREAKEVEN CALCULATOR
# =============================================================================


class BreakevenCalculator:
    """
    Calculator for breakeven analysis of option portfolios.

    Uses grid search with linear interpolation for precision.

    Parameters
    ----------
    spot_min : float
        Minimum spot price to search (default 0.1)
    spot_max : float
        Maximum spot price to search (default 1000.0)
    precision : int
        Number of grid points for search (default 10000)
    """

    def __init__(
        self,
        spot_min: float = 0.1,
        spot_max: float = 1000.0,
        precision: int = 10000,
    ) -> None:
        """Initialize breakeven calculator."""
        self._spot_min = spot_min
        self._spot_max = spot_max
        self._precision = precision

    def calculate(
        self,
        positions: list[PortfolioPosition],
        stock: StockPosition | None = None,
        spot_min: float | None = None,
        spot_max: float | None = None,
    ) -> BreakevenResult:
        """
        Find all breakeven points for a portfolio.

        Parameters
        ----------
        positions : List[PortfolioPosition]
            List of option positions
        stock : StockPosition, optional
            Stock position if any
        spot_min : float, optional
            Override minimum spot price
        spot_max : float, optional
            Override maximum spot price

        Returns
        -------
        BreakevenResult
            Complete breakeven analysis
        """
        min_spot = spot_min if spot_min is not None else self._spot_min
        max_spot = spot_max if spot_max is not None else self._spot_max

        # Generate price grid
        spot_range = np.linspace(min_spot, max_spot, self._precision)

        # Vectorized P&L calculation
        pnl_values = self._calculate_pnl_array(spot_range, positions, stock)

        # Find breakeven points (where P&L changes sign)
        breakeven_points = self._find_sign_changes(spot_range, pnl_values)

        # Find max profit and max loss
        max_profit_idx = np.argmax(pnl_values)
        max_loss_idx = np.argmin(pnl_values)

        max_profit = float(pnl_values[max_profit_idx])
        max_profit_spot = float(spot_range[max_profit_idx])
        max_loss = float(pnl_values[max_loss_idx])
        max_loss_spot = float(spot_range[max_loss_idx])

        # Identify profit and loss zones
        profit_zones, loss_zones = self._identify_zones(
            breakeven_points, pnl_values[0], positions, stock, min_spot, max_spot
        )

        return BreakevenResult(
            breakeven_points=breakeven_points,
            max_profit=max_profit,
            max_profit_spot=max_profit_spot,
            max_loss=max_loss,
            max_loss_spot=max_loss_spot,
            profit_zones=profit_zones,
            loss_zones=loss_zones,
        )

    def calculate_from_portfolio(
        self,
        portfolio: "OptionsPortfolio",
        spot_min: float | None = None,
        spot_max: float | None = None,
    ) -> BreakevenResult:
        """
        Find breakeven points directly from an OptionsPortfolio.

        Parameters
        ----------
        portfolio : OptionsPortfolio
            Portfolio to analyze
        spot_min : float, optional
            Override minimum spot price
        spot_max : float, optional
            Override maximum spot price

        Returns
        -------
        BreakevenResult
            Complete breakeven analysis
        """
        return self.calculate(
            positions=portfolio.positions,
            stock=portfolio.stock,
            spot_min=spot_min,
            spot_max=spot_max,
        )

    def _calculate_pnl_array(
        self,
        spot_range: np.ndarray,
        positions: list[PortfolioPosition],
        stock: StockPosition | None,
    ) -> np.ndarray:
        """
        Calculate P&L for entire spot array using Numba-optimized function.

        This method is significantly faster than iterating over Python objects.
        """
        if len(positions) == 0 and stock is None:
            return np.zeros_like(spot_range)

        # Convert positions to arrays for Numba
        strikes, option_types, position_types, quantities, premiums = (
            _positions_to_arrays(positions)
        )

        # Stock parameters
        stock_qty = float(stock.quantity) if stock else 0.0
        stock_entry = float(stock.entry_price) if stock else 0.0

        # Use Numba-optimized payoff curve
        return _compute_payoff_curve_numba(
            spot_range.astype(np.float64),
            strikes,
            option_types,
            position_types,
            quantities,
            premiums,
            stock_quantity=stock_qty,
            stock_entry_price=stock_entry,
            multiplier=1.0,  # No multiplier at position level
        )

    def _find_sign_changes(
        self,
        spot_range: np.ndarray,
        pnl_values: np.ndarray,
    ) -> list[float]:
        """
        Find breakeven points using Numba-optimized function.

        Uses linear interpolation at sign changes for precision.
        """
        # Use Numba-optimized breakeven finder
        breakevens = _find_breakeven_points_numba(pnl_values, spot_range)
        return breakevens.tolist()

    def _identify_zones(
        self,
        breakeven_points: list[float],
        first_pnl: float,
        positions: list[PortfolioPosition],
        stock: StockPosition | None,
        spot_min: float,
        spot_max: float,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Identify profit and loss zones based on breakeven points."""
        profit_zones = []
        loss_zones = []

        if len(breakeven_points) == 0:
            # No breakeven: either always profit or always loss
            if first_pnl >= 0:
                profit_zones.append((-np.inf, np.inf))
            else:
                loss_zones.append((-np.inf, np.inf))
        else:
            # Analyze each zone between breakeven points
            breakeven_extended = [-np.inf] + breakeven_points + [np.inf]

            for i in range(len(breakeven_extended) - 1):
                # Determine test spot in the middle of the zone
                if i == 0:
                    test_spot = breakeven_extended[1] - 1
                elif i == len(breakeven_extended) - 2:
                    test_spot = breakeven_extended[-2] + 1
                else:
                    test_spot = (breakeven_extended[i] + breakeven_extended[i + 1]) / 2

                # Clamp to valid range
                test_spot = max(spot_min, min(spot_max, test_spot))

                # Calculate P&L at test point
                test_pnl = calculate_portfolio_pnl_at_expiry(
                    test_spot, positions, stock
                )

                if test_pnl >= 0:
                    profit_zones.append(
                        (breakeven_extended[i], breakeven_extended[i + 1])
                    )
                else:
                    loss_zones.append(
                        (breakeven_extended[i], breakeven_extended[i + 1])
                    )

        return profit_zones, loss_zones


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def find_breakevens(
    positions: list[PortfolioPosition],
    stock: StockPosition | None = None,
    spot_min: float = 0.1,
    spot_max: float = 1000.0,
    precision: int = 10000,
) -> BreakevenResult:
    """
    Convenience function to find breakeven points.

    Parameters
    ----------
    positions : List[PortfolioPosition]
        List of option positions
    stock : StockPosition, optional
        Stock position if any
    spot_min : float
        Minimum spot price to search
    spot_max : float
        Maximum spot price to search
    precision : int
        Number of grid points for search

    Returns
    -------
    BreakevenResult
        Complete breakeven analysis
    """
    calculator = BreakevenCalculator(
        spot_min=spot_min,
        spot_max=spot_max,
        precision=precision,
    )
    return calculator.calculate(positions, stock)


def find_breakevens_from_portfolio(
    portfolio: "OptionsPortfolio",
    spot_min: float = 0.1,
    spot_max: float = 1000.0,
    precision: int = 10000,
) -> BreakevenResult:
    """
    Convenience function to find breakeven points from a portfolio.

    Parameters
    ----------
    portfolio : OptionsPortfolio
        Portfolio to analyze
    spot_min : float
        Minimum spot price to search
    spot_max : float
        Maximum spot price to search
    precision : int
        Number of grid points for search

    Returns
    -------
    BreakevenResult
        Complete breakeven analysis
    """
    calculator = BreakevenCalculator(
        spot_min=spot_min,
        spot_max=spot_max,
        precision=precision,
    )
    return calculator.calculate_from_portfolio(portfolio, spot_min, spot_max)


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    from backend.portfolio.positions import long_call, long_put, long_stock, short_call

    print("=" * 50)
    print("Breakeven Module Smoke Test")
    print("=" * 50)

    # Test 1: Bull Call Spread
    print("\n--- Bull Call Spread ---")
    positions = [
        long_call(strike=95, maturity=0.5, premium=8.0),
        short_call(strike=105, maturity=0.5, premium=3.0),
    ]

    result = find_breakevens(positions, spot_min=80, spot_max=120)
    print(result.summary())

    # Test 2: Long Straddle
    print("\n--- Long Straddle ---")
    positions = [
        long_call(strike=100, maturity=0.5, premium=5.0),
        long_put(strike=100, maturity=0.5, premium=4.5),
    ]

    result = find_breakevens(positions, spot_min=80, spot_max=120)
    print(result.summary())

    # Test 3: Covered Call
    print("\n--- Covered Call ---")
    positions = [
        short_call(strike=105, maturity=0.5, premium=3.0),
    ]
    stock = long_stock(quantity=100, entry_price=100.0)

    result = find_breakevens(positions, stock=stock, spot_min=80, spot_max=120)
    print(result.summary())

    # Test 4: Direct P&L calculation
    print("\n--- P&L at Expiry ---")
    test_positions = [long_call(strike=100, maturity=0.5, premium=5.0)]
    spots = [90.0, 100.0, 105.0, 110.0]
    for spot in spots:
        pnl = calculate_portfolio_pnl_at_expiry(spot, test_positions)
        print(f"  Spot={spot}: P&L=${pnl:.2f}")

    print("\n" + "=" * 50)
    print("Breakeven smoke test passed")
    print("=" * 50)
