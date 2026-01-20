"""
Breakeven Analysis
==================

Find breakeven points and profit/loss zones for option portfolios.

Author: Derivatives Pricing Project
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .positions import OptionPosition, StockPosition


# =============================================================================
# Breakeven Result Container
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
    breakeven_points: List[float]
    max_profit: float
    max_profit_spot: float
    max_loss: float
    max_loss_spot: float
    profit_zones: List[Tuple[float, float]]
    loss_zones: List[Tuple[float, float]]

    def summary(self) -> str:
        """Generate a formatted summary of breakeven analysis."""
        lines = ["=== Breakeven Analysis ==="]

        if self.breakeven_points:
            formatted = [f"${b:.2f}" for b in self.breakeven_points]
            lines.append(f"Breakeven points: {formatted}")
        else:
            lines.append("No breakeven points (always profit or always loss)")

        lines.append(f"Max Profit: ${self.max_profit:.2f} at ${self.max_profit_spot:.2f}")
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
# P&L Calculation
# =============================================================================

def calculate_portfolio_pnl_at_expiry(
    spot: float,
    options: List[OptionPosition],
    stock: Optional[StockPosition] = None,
) -> float:
    """
    Calculate portfolio P&L at expiration for a given spot price.

    Parameters
    ----------
    spot : float
        Spot price at expiration
    options : List[OptionPosition]
        List of option positions
    stock : StockPosition, optional
        Stock position if any

    Returns
    -------
    float
        Total P&L at expiration
    """
    total_pnl = 0.0

    # Sum option payoffs at expiry
    for pos in options:
        total_pnl += pos.payoff_at_expiry(spot)

    # Add stock P&L if present
    if stock is not None:
        total_pnl += stock.pnl(spot)

    return total_pnl


def calculate_pnl_at_expiry_arrays(
    spot: float,
    strikes: np.ndarray,
    option_types: np.ndarray,  # 1 for call, 0 for put
    position_types: np.ndarray,  # 1 for long, -1 for short
    quantities: np.ndarray,
    premiums: np.ndarray,
    stock_quantity: int = 0,
    stock_entry_price: float = 0.0,
) -> float:
    """
    Calculate portfolio P&L at expiration using array-based parameters (legacy-compatible).

    Parameters
    ----------
    spot : float
        Spot price at expiration
    strikes : np.ndarray
        Strike prices
    option_types : np.ndarray
        1 for call, 0 for put
    position_types : np.ndarray
        1 for long, -1 for short
    quantities : np.ndarray
        Position quantities
    premiums : np.ndarray
        Premiums paid/received
    stock_quantity : int
        Stock position quantity (positive for long, negative for short)
    stock_entry_price : float
        Stock entry price

    Returns
    -------
    float
        Total P&L at expiration
    """
    total_pnl = 0.0

    # Option P&L
    for i in range(len(strikes)):
        opt_type = 'call' if option_types[i] == 1 else 'put'

        # Calculate intrinsic value
        if opt_type == 'call':
            intrinsic = max(0, spot - strikes[i])
        else:
            intrinsic = max(0, strikes[i] - spot)

        # P&L based on position type
        if position_types[i] == 1:  # Long
            pnl = (intrinsic - premiums[i]) * quantities[i]
        else:  # Short
            pnl = (premiums[i] - intrinsic) * quantities[i]

        total_pnl += pnl

    # Stock P&L
    if stock_quantity != 0:
        stock_pnl = (spot - stock_entry_price) * stock_quantity
        total_pnl += stock_pnl

    return total_pnl


# =============================================================================
# Breakeven Calculator
# =============================================================================

class BreakevenCalculator:
    """
    Calculator for breakeven analysis of option portfolios.

    Uses grid search with linear interpolation for precision.
    """

    def __init__(
        self,
        spot_min: float = 0.1,
        spot_max: float = 1000.0,
        precision: int = 10000,
    ):
        """
        Initialize breakeven calculator.

        Parameters
        ----------
        spot_min : float
            Minimum spot price to search
        spot_max : float
            Maximum spot price to search
        precision : int
            Number of grid points for search
        """
        self._spot_min = spot_min
        self._spot_max = spot_max
        self._precision = precision

    def calculate(
        self,
        options: List[OptionPosition],
        stock: Optional[StockPosition] = None,
        spot_min: Optional[float] = None,
        spot_max: Optional[float] = None,
    ) -> BreakevenResult:
        """
        Find all breakeven points for a portfolio.

        Parameters
        ----------
        options : List[OptionPosition]
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
        pnl_values = np.array([
            calculate_portfolio_pnl_at_expiry(spot, options, stock)
            for spot in spot_range
        ])

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
            breakeven_points, pnl_values[0], options, stock, min_spot, max_spot
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

    def _find_sign_changes(
        self,
        spot_range: np.ndarray,
        pnl_values: np.ndarray,
    ) -> List[float]:
        """Find breakeven points using linear interpolation at sign changes."""
        breakeven_points = []

        for i in range(len(pnl_values) - 1):
            # Check for sign change
            if pnl_values[i] * pnl_values[i + 1] < 0:
                # Linear interpolation for precise breakeven
                spot1, spot2 = spot_range[i], spot_range[i + 1]
                pnl1, pnl2 = pnl_values[i], pnl_values[i + 1]

                breakeven = spot1 - pnl1 * (spot2 - spot1) / (pnl2 - pnl1)
                breakeven_points.append(float(breakeven))

        return breakeven_points

    def _identify_zones(
        self,
        breakeven_points: List[float],
        first_pnl: float,
        options: List[OptionPosition],
        stock: Optional[StockPosition],
        spot_min: float,
        spot_max: float,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Identify profit and loss zones based on breakeven points."""
        profit_zones = []
        loss_zones = []

        if len(breakeven_points) == 0:
            # No breakeven: either always profit or always loss
            if first_pnl > 0:
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
                test_pnl = calculate_portfolio_pnl_at_expiry(test_spot, options, stock)

                if test_pnl > 0:
                    profit_zones.append((breakeven_extended[i], breakeven_extended[i + 1]))
                else:
                    loss_zones.append((breakeven_extended[i], breakeven_extended[i + 1]))

        return profit_zones, loss_zones


# =============================================================================
# Convenience Function
# =============================================================================

def find_breakevens(
    options: List[OptionPosition],
    stock: Optional[StockPosition] = None,
    spot_min: float = 0.1,
    spot_max: float = 1000.0,
    precision: int = 10000,
) -> BreakevenResult:
    """
    Convenience function to find breakeven points.

    Parameters
    ----------
    options : List[OptionPosition]
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
    return calculator.calculate(options, stock)


def find_breakeven_points(
    strikes: np.ndarray,
    option_types: np.ndarray,  # 1 for call, 0 for put
    position_types: np.ndarray,  # 1 for long, -1 for short
    quantities: np.ndarray,
    premiums: np.ndarray,
    stock_quantity: int = 0,
    stock_entry_price: float = 0.0,
    spot_min: float = 0.01,
    spot_max: float = 1000.0,
    precision: int = 10000,
) -> Optional[BreakevenResult]:
    """
    Find breakeven points using array-based parameters (legacy-compatible).

    Parameters
    ----------
    strikes : np.ndarray
        Strike prices
    option_types : np.ndarray
        1 for call, 0 for put
    position_types : np.ndarray
        1 for long, -1 for short
    quantities : np.ndarray
        Position quantities
    premiums : np.ndarray
        Premiums paid/received
    stock_quantity : int
        Stock position quantity
    stock_entry_price : float
        Stock entry price
    spot_min : float
        Minimum spot price to search
    spot_max : float
        Maximum spot price to search
    precision : int
        Number of grid points

    Returns
    -------
    BreakevenResult or None
        Breakeven analysis results
    """
    if len(strikes) == 0 and stock_quantity == 0:
        return None

    # Convert arrays to OptionPosition objects
    options = []
    for i in range(len(strikes)):
        opt_type = 'call' if option_types[i] == 1 else 'put'
        pos_type = 'long' if position_types[i] == 1 else 'short'

        options.append(OptionPosition(
            option_type=opt_type,
            position_type=pos_type,
            strike=float(strikes[i]),
            premium=float(premiums[i]),
            quantity=int(quantities[i]),
        ))

    stock = None
    if stock_quantity != 0:
        stock = StockPosition(
            position_type='long' if stock_quantity > 0 else 'short',
            quantity=abs(stock_quantity),
            entry_price=stock_entry_price,
        )

    return find_breakevens(
        options=options,
        stock=stock,
        spot_min=spot_min,
        spot_max=spot_max,
        precision=precision,
    )
