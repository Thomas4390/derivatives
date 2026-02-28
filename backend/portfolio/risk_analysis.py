"""
Risk Analysis Module
====================

Portfolio risk analysis for unlimited profit/loss detection.

This module provides:
- RiskProfile: Named tuple containing risk analysis results
- check_unlimited_risk: Detect unlimited profit/loss potential (from positions)
- check_unlimited_risk_from_portfolio: Same from OptionsPortfolio
- check_unlimited_risk_arrays: Numba-optimized version for array inputs
- analyze_portfolio_risk: Full risk analysis
- get_risk_summary: Human-readable risk summary

Designed to work with the backend architecture:
- Uses PortfolioPosition and StockPosition dataclasses
- Consistent with find_breakevens_from_portfolio pattern
- Numba JIT compilation for performance-critical paths

Author: Thomas
Created: 2025
"""

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from backend.portfolio.portfolio import OptionsPortfolio

from backend.portfolio.positions import PortfolioPosition, StockPosition

# =============================================================================
# RESULT TYPES
# =============================================================================

class RiskProfile(NamedTuple):
    """Risk profile for a portfolio."""
    has_unlimited_profit: bool
    has_unlimited_loss: bool
    max_profit: float | None
    max_loss: float | None
    max_profit_spot: float | None
    max_loss_spot: float | None


# =============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# =============================================================================

@njit(cache=True)
def _check_unlimited_risk_numba(
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    has_stock: bool,
    stock_is_long: bool
) -> tuple[bool, bool]:
    """
    Numba-optimized unlimited risk detection.

    Args:
        option_types: Array of option types (1.0 = call, -1.0 = put)
        position_types: Array of position types (1.0 = long, -1.0 = short)
        quantities: Array of quantities (absolute values)
        has_stock: Whether portfolio has stock position
        stock_is_long: True if stock position is long

    Returns:
        Tuple of (has_unlimited_profit, has_unlimited_loss)
    """
    # Stock positions always have unlimited potential in one direction
    if has_stock:
        if stock_is_long:
            return True, False  # Unlimited profit, limited loss
        return False, True  # Limited profit, unlimited loss

    n = len(option_types)
    if n == 0:
        return False, False

    # Count long and short calls
    total_long_calls = 0.0
    total_short_calls = 0.0

    for i in range(n):
        if option_types[i] > 0:  # Call option
            if position_types[i] > 0:  # Long
                total_long_calls += quantities[i]
            else:  # Short
                total_short_calls += quantities[i]

    # Unlimited profit: more long calls than short calls
    unlimited_profit = total_long_calls > total_short_calls

    # Unlimited loss: more short calls than long calls (naked shorts)
    unlimited_loss = total_short_calls > total_long_calls

    return unlimited_profit, unlimited_loss


@njit(cache=True)
def _determine_max_profit_numba(
    unlimited_profit: bool,
    max_profit_from_breakeven: float,
    expiry_pnl: np.ndarray
) -> float:
    """Numba-optimized max profit determination."""
    if not unlimited_profit:
        return max_profit_from_breakeven

    n = len(expiry_pnl)
    if n > 10:
        high_end_trend = expiry_pnl[n - 1] - expiry_pnl[n - 10]
        if high_end_trend > 0:
            return np.inf

    return max_profit_from_breakeven


@njit(cache=True)
def _determine_max_loss_numba(
    unlimited_loss: bool,
    max_loss_from_breakeven: float,
    expiry_pnl: np.ndarray
) -> float:
    """Numba-optimized max loss determination."""
    if not unlimited_loss:
        return max_loss_from_breakeven

    n = len(expiry_pnl)
    if n > 10:
        high_end_trend = expiry_pnl[n - 1] - expiry_pnl[n - 10]
        if high_end_trend < 0 and expiry_pnl[n - 1] < 0:
            return -np.inf

    return max_loss_from_breakeven


# =============================================================================
# ARRAY-BASED API (for direct Numba usage)
# =============================================================================

def check_unlimited_risk_arrays(
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    has_stock: bool = False,
    stock_is_long: bool = True
) -> tuple[bool, bool]:
    """
    Check unlimited risk using array inputs (Numba-optimized).

    This is the fastest way to check risk when you already have arrays.

    Args:
        option_types: Array of option types (1.0 = call, -1.0 = put)
        position_types: Array of position types (1.0 = long, -1.0 = short)
        quantities: Array of absolute quantities
        has_stock: Whether portfolio has stock position
        stock_is_long: True if stock position is long

    Returns:
        Tuple of (has_unlimited_profit, has_unlimited_loss)
    """
    return _check_unlimited_risk_numba(
        option_types, position_types, quantities, has_stock, stock_is_long
    )


# =============================================================================
# HIGH-LEVEL API (works with position classes)
# =============================================================================

def _positions_to_arrays(
    positions: list[PortfolioPosition]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert position objects to numpy arrays for Numba functions.

    Returns:
        Tuple of (option_types, position_types, quantities)
    """
    n = len(positions)
    option_types = np.zeros(n, dtype=np.float64)
    position_types = np.zeros(n, dtype=np.float64)
    quantities = np.zeros(n, dtype=np.float64)

    for i, pos in enumerate(positions):
        option_types[i] = 1.0 if pos.is_call else -1.0
        position_types[i] = 1.0 if pos.is_long else -1.0
        quantities[i] = abs(pos.quantity)

    return option_types, position_types, quantities


def check_unlimited_risk(
    positions: list[PortfolioPosition],
    stock: StockPosition | None = None
) -> tuple[bool, bool]:
    """
    Check if portfolio has unlimited profit or loss potential.

    Works with the backend position classes (PortfolioPosition, StockPosition).
    Uses Numba-optimized core function for performance.

    Args:
        positions: List of PortfolioPosition objects
        stock: Optional StockPosition

    Returns:
        Tuple of (has_unlimited_profit, has_unlimited_loss)
    """
    # Convert positions to arrays
    option_types, position_types, quantities = _positions_to_arrays(positions)

    # Handle stock
    has_stock = stock is not None
    stock_is_long = stock.is_long if stock else True

    # Call Numba function
    return _check_unlimited_risk_numba(
        option_types, position_types, quantities, has_stock, stock_is_long
    )


def check_unlimited_risk_from_portfolio(portfolio: 'OptionsPortfolio') -> tuple[bool, bool]:
    """
    Check if portfolio has unlimited profit or loss potential.

    Convenience function that extracts positions from OptionsPortfolio.

    Args:
        portfolio: OptionsPortfolio object

    Returns:
        Tuple of (has_unlimited_profit, has_unlimited_loss)
    """
    return check_unlimited_risk(portfolio.positions, portfolio.stock)


# =============================================================================
# FULL RISK ANALYSIS
# =============================================================================

def analyze_portfolio_risk(
    positions: list[PortfolioPosition],
    stock: StockPosition | None,
    breakeven_result,  # BreakevenResult or None
    expiry_pnl: np.ndarray
) -> RiskProfile:
    """
    Perform complete risk analysis on a portfolio.

    Uses Numba-optimized functions for performance.

    Args:
        positions: List of PortfolioPosition objects
        stock: Optional StockPosition
        breakeven_result: BreakevenResult from calculations (or None)
        expiry_pnl: P&L values at expiration

    Returns:
        RiskProfile with complete risk analysis
    """
    unlimited_profit, unlimited_loss = check_unlimited_risk(positions, stock)

    if breakeven_result is None:
        return RiskProfile(
            has_unlimited_profit=unlimited_profit,
            has_unlimited_loss=unlimited_loss,
            max_profit=None,
            max_loss=None,
            max_profit_spot=None,
            max_loss_spot=None
        )

    # Use Numba functions for max profit/loss determination
    max_profit = _determine_max_profit_numba(
        unlimited_profit, breakeven_result.max_profit, expiry_pnl
    )
    max_loss = _determine_max_loss_numba(
        unlimited_loss, breakeven_result.max_loss, expiry_pnl
    )

    return RiskProfile(
        has_unlimited_profit=unlimited_profit,
        has_unlimited_loss=unlimited_loss,
        max_profit=float(max_profit),
        max_loss=float(max_loss),
        max_profit_spot=breakeven_result.max_profit_spot,
        max_loss_spot=breakeven_result.max_loss_spot
    )


def analyze_portfolio_risk_from_portfolio(
    portfolio: 'OptionsPortfolio',
    breakeven_result,
    expiry_pnl: np.ndarray
) -> RiskProfile:
    """
    Perform complete risk analysis on an OptionsPortfolio.

    Convenience function that extracts positions from OptionsPortfolio.

    Args:
        portfolio: OptionsPortfolio object
        breakeven_result: BreakevenResult from calculations (or None)
        expiry_pnl: P&L values at expiration

    Returns:
        RiskProfile with complete risk analysis
    """
    return analyze_portfolio_risk(
        portfolio.positions, portfolio.stock, breakeven_result, expiry_pnl
    )


# =============================================================================
# HUMAN-READABLE SUMMARY
# =============================================================================

def get_risk_summary(risk_profile: RiskProfile) -> dict:
    """
    Get a human-readable risk summary.

    Args:
        risk_profile: RiskProfile object

    Returns:
        Dictionary with risk summary information
    """
    summary = {
        'risk_level': 'Unknown',
        'profit_potential': 'Unknown',
        'loss_potential': 'Unknown',
        'warnings': []
    }

    # Profit potential
    if risk_profile.has_unlimited_profit:
        summary['profit_potential'] = 'Unlimited'
    elif risk_profile.max_profit is not None:
        summary['profit_potential'] = f'${risk_profile.max_profit:.2f}'

    # Loss potential
    if risk_profile.has_unlimited_loss:
        summary['loss_potential'] = 'Unlimited'
        summary['warnings'].append('Position has unlimited loss potential')
        summary['risk_level'] = 'High'
    elif risk_profile.max_loss is not None:
        summary['loss_potential'] = f'${abs(risk_profile.max_loss):.2f}'
        summary['risk_level'] = 'Defined'

    return summary


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    from backend.portfolio.positions import (
        long_call,
        long_put,
        long_stock,
        short_call,
        short_put,
        short_stock,
    )

    print("=" * 50)
    print("Risk Analysis Module Smoke Test")
    print("=" * 50)

    # Test 1: Long call (unlimited profit, limited loss)
    print("\n--- Test 1: Long Call ---")
    positions_long_call = [long_call(strike=100, maturity=0.5)]
    up, ul = check_unlimited_risk(positions_long_call)
    assert up is True, f"Expected True, got {up}"
    assert ul is False, f"Expected False, got {ul}"
    print(f"Unlimited Profit: {up} (expected: True) ✓")
    print(f"Unlimited Loss: {ul} (expected: False) ✓")

    # Test 2: Short call (limited profit, unlimited loss)
    print("\n--- Test 2: Short Call ---")
    positions_short_call = [short_call(strike=100, maturity=0.5)]
    up, ul = check_unlimited_risk(positions_short_call)
    assert up is False, f"Expected False, got {up}"
    assert ul is True, f"Expected True, got {ul}"
    print(f"Unlimited Profit: {up} (expected: False) ✓")
    print(f"Unlimited Loss: {ul} (expected: True) ✓")

    # Test 3: Long stock (unlimited profit, limited loss)
    print("\n--- Test 3: Long Stock ---")
    stock_long = long_stock(quantity=100, entry_price=100)
    up, ul = check_unlimited_risk([], stock_long)
    assert up is True, f"Expected True, got {up}"
    assert ul is False, f"Expected False, got {ul}"
    print(f"Unlimited Profit: {up} (expected: True) ✓")
    print(f"Unlimited Loss: {ul} (expected: False) ✓")

    # Test 4: Short stock (limited profit, unlimited loss)
    print("\n--- Test 4: Short Stock ---")
    stock_short = short_stock(quantity=100, entry_price=100)
    up, ul = check_unlimited_risk([], stock_short)
    assert up is False, f"Expected False, got {up}"
    assert ul is True, f"Expected True, got {ul}"
    print(f"Unlimited Profit: {up} (expected: False) ✓")
    print(f"Unlimited Loss: {ul} (expected: True) ✓")

    # Test 5: Call spread (limited profit, limited loss)
    print("\n--- Test 5: Bull Call Spread ---")
    positions_spread = [
        long_call(strike=100, maturity=0.5),
        short_call(strike=110, maturity=0.5)
    ]
    up, ul = check_unlimited_risk(positions_spread)
    assert up is False, f"Expected False, got {up}"
    assert ul is False, f"Expected False, got {ul}"
    print(f"Unlimited Profit: {up} (expected: False) ✓")
    print(f"Unlimited Loss: {ul} (expected: False) ✓")

    # Test 6: Iron Condor (limited profit, limited loss)
    print("\n--- Test 6: Iron Condor ---")
    positions_condor = [
        long_put(strike=90, maturity=0.5),
        short_put(strike=95, maturity=0.5),
        short_call(strike=105, maturity=0.5),
        long_call(strike=110, maturity=0.5)
    ]
    up, ul = check_unlimited_risk(positions_condor)
    assert up is False, f"Expected False, got {up}"
    assert ul is False, f"Expected False, got {ul}"
    print(f"Unlimited Profit: {up} (expected: False) ✓")
    print(f"Unlimited Loss: {ul} (expected: False) ✓")

    # Test 7: Straddle (unlimited profit, limited loss)
    print("\n--- Test 7: Long Straddle ---")
    positions_straddle = [
        long_call(strike=100, maturity=0.5),
        long_put(strike=100, maturity=0.5)
    ]
    up, ul = check_unlimited_risk(positions_straddle)
    assert up is True, f"Expected True, got {up}"
    assert ul is False, f"Expected False, got {ul}"
    print(f"Unlimited Profit: {up} (expected: True) ✓")
    print(f"Unlimited Loss: {ul} (expected: False) ✓")

    # Test 8: Naked short strangle (limited profit, unlimited loss)
    print("\n--- Test 8: Short Strangle ---")
    positions_strangle = [
        short_call(strike=110, maturity=0.5),
        short_put(strike=90, maturity=0.5)
    ]
    up, ul = check_unlimited_risk(positions_strangle)
    assert up is False, f"Expected False, got {up}"
    assert ul is True, f"Expected True, got {ul}"
    print(f"Unlimited Profit: {up} (expected: False) ✓")
    print(f"Unlimited Loss: {ul} (expected: True) ✓")

    # Test 9: Using OptionsPortfolio
    print("\n--- Test 9: Using OptionsPortfolio ---")
    from backend.models.gbm import GBMModel
    from backend.portfolio.portfolio import OptionsPortfolio

    portfolio = OptionsPortfolio(GBMModel(sigma=0.2))
    portfolio.add(long_call(strike=100, maturity=0.5))
    portfolio.add(short_call(strike=110, maturity=0.5))

    up, ul = check_unlimited_risk_from_portfolio(portfolio)
    assert up is False, f"Expected False, got {up}"
    assert ul is False, f"Expected False, got {ul}"
    print(f"Unlimited Profit: {up} (expected: False) ✓")
    print(f"Unlimited Loss: {ul} (expected: False) ✓")

    # Test 10: Array-based API
    print("\n--- Test 10: Array-based API (Numba direct) ---")
    option_types = np.array([1.0, 1.0])  # Both calls
    position_types = np.array([1.0, -1.0])  # Long, Short
    quantities = np.array([1.0, 1.0])

    up, ul = check_unlimited_risk_arrays(option_types, position_types, quantities)
    assert up is False, f"Expected False, got {up}"
    assert ul is False, f"Expected False, got {ul}"
    print(f"Unlimited Profit: {up} (expected: False) ✓")
    print(f"Unlimited Loss: {ul} (expected: False) ✓")

    print("\n" + "=" * 50)
    print("Risk Analysis smoke test passed! ✓")
    print("=" * 50)
