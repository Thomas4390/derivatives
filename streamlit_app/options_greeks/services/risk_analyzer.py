"""
Risk Analyzer Adapter for Streamlit App
========================================

This module provides a frontend adapter that converts dict-based portfolio data
to the backend position classes and calls the backend risk analysis functions.

It acts as a bridge between:
- Frontend: Dict-based positions stored in Streamlit session state
- Backend: PortfolioPosition/StockPosition dataclasses

Author: Thomas
Created: 2025
"""

import numpy as np

from backend.instruments.options import VanillaOption
from backend.portfolio.positions import PortfolioPosition, StockPosition
from backend.portfolio.risk_analysis import (
    RiskProfile,
    get_risk_summary,
)
from backend.portfolio.risk_analysis import (
    analyze_portfolio_risk as _analyze_portfolio_risk_backend,
)
from backend.portfolio.risk_analysis import (
    check_unlimited_risk as _check_unlimited_risk_backend,
)

# Re-export for backwards compatibility
__all__ = [
    "RiskProfile",
    "check_unlimited_risk",
    "check_unlimited_risk_from_dict",
    "analyze_portfolio_risk",
    "analyze_portfolio_risk_from_dict",
    "get_risk_summary",
    "convert_dict_positions",
    "convert_dict_stock",
]


# =============================================================================
# CONVERSION FUNCTIONS: Dict -> Backend Position Classes
# =============================================================================


def convert_dict_positions(positions: list[dict]) -> list[PortfolioPosition]:
    """
    Convert dict-based positions to PortfolioPosition objects.

    Args:
        positions: List of position dicts with keys:
            - option_type: 'call' | 'put'
            - position_type: 'long' | 'short'
            - strike: float
            - quantity: int (positive)
            - premium_paid: float (per contract)
            - dte_days: int (optional, default 30)
            - volatility: float (optional, default 0.2)

    Returns:
        List of PortfolioPosition objects
    """
    result = []
    for pos in positions:
        # Skip exotic legs - they cannot be converted to VanillaOption
        if pos.get("instrument_class", "vanilla") != "vanilla":
            continue
        is_call = pos["option_type"] == "call"
        is_long = pos["position_type"] == "long"

        # Get maturity (convert DTE to years)
        dte_days = pos.get("dte_days", 30)
        maturity = dte_days / 365.0

        # Create VanillaOption instrument
        instrument = VanillaOption(
            strike=pos["strike"], maturity=maturity, is_call=is_call
        )

        # Quantity is positive for long, negative for short
        quantity = abs(pos["quantity"])
        if not is_long:
            quantity = -quantity

        # Premium paid per contract
        premium = pos.get("premium_paid", 0.0)

        result.append(
            PortfolioPosition(instrument=instrument, quantity=quantity, premium=premium)
        )

    return result


def convert_dict_stock(stock_dict: dict | None) -> StockPosition | None:
    """
    Convert dict-based stock position to StockPosition object.

    Args:
        stock_dict: Stock position dict with keys:
            - position_type: 'long' | 'short'
            - quantity: int (positive)
            - entry_price: float

    Returns:
        StockPosition object or None
    """
    if stock_dict is None:
        return None

    is_long = stock_dict["position_type"] == "long"
    quantity = abs(stock_dict["quantity"])
    if not is_long:
        quantity = -quantity

    return StockPosition(
        quantity=quantity, entry_price=stock_dict.get("entry_price", 0.0)
    )


# =============================================================================
# ADAPTER FUNCTIONS: Dict-based -> Backend
# =============================================================================


def check_unlimited_risk_from_dict(portfolio_data: dict) -> tuple[bool, bool]:
    """
    Check if portfolio has unlimited profit or loss potential.

    Works with portfolio data in dict format (for Streamlit frontend).

    Args:
        portfolio_data: Dictionary containing:
            - 'options': List of position dicts
            - 'stock': Stock position dict or None

    Returns:
        Tuple of (has_unlimited_profit, has_unlimited_loss)
    """
    # Convert to backend types
    positions = convert_dict_positions(portfolio_data.get("options", []))
    stock = convert_dict_stock(portfolio_data.get("stock"))

    # Call backend function
    return _check_unlimited_risk_backend(positions, stock)


def analyze_portfolio_risk_from_dict(
    portfolio_data: dict, breakeven_result, expiry_pnl: np.ndarray
) -> RiskProfile:
    """
    Perform complete risk analysis on a dict-based portfolio.

    Args:
        portfolio_data: Dictionary containing portfolio information
        breakeven_result: BreakevenResult from calculations (or None)
        expiry_pnl: P&L values at expiration

    Returns:
        RiskProfile with complete risk analysis
    """
    # Convert to backend types
    positions = convert_dict_positions(portfolio_data.get("options", []))
    stock = convert_dict_stock(portfolio_data.get("stock"))

    # Call backend function
    return _analyze_portfolio_risk_backend(
        positions, stock, breakeven_result, expiry_pnl
    )


# =============================================================================
# BACKWARDS COMPATIBLE ALIASES
# =============================================================================

# These maintain compatibility with existing frontend code that uses the old API


def check_unlimited_risk(portfolio_data: dict) -> tuple[bool, bool]:
    """
    Check if portfolio has unlimited profit or loss potential.

    Backwards-compatible alias for check_unlimited_risk_from_dict.

    Args:
        portfolio_data: Dictionary containing portfolio information

    Returns:
        Tuple of (has_unlimited_profit, has_unlimited_loss)
    """
    return check_unlimited_risk_from_dict(portfolio_data)


def analyze_portfolio_risk(
    portfolio_data: dict, breakeven_result, expiry_pnl: np.ndarray
) -> RiskProfile:
    """
    Perform complete risk analysis on a portfolio.

    Backwards-compatible alias for analyze_portfolio_risk_from_dict.

    Args:
        portfolio_data: Dictionary containing portfolio information
        breakeven_result: BreakevenResult from calculations (or None)
        expiry_pnl: P&L values at expiration

    Returns:
        RiskProfile with complete risk analysis
    """
    return analyze_portfolio_risk_from_dict(
        portfolio_data, breakeven_result, expiry_pnl
    )


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Risk Analyzer Adapter Smoke Test")
    print("=" * 50)

    # Test 1: Long call (unlimited profit, limited loss)
    print("\n--- Test 1: Long Call (dict format) ---")
    portfolio_long_call = {
        "options": [
            {
                "option_type": "call",
                "position_type": "long",
                "strike": 100,
                "quantity": 1,
                "premium_paid": 5.0,
                "dte_days": 30,
            }
        ],
        "stock": None,
    }
    up, ul = check_unlimited_risk(portfolio_long_call)
    assert up is True, f"Expected True, got {up}"
    assert ul is False, f"Expected False, got {ul}"
    print(f"Unlimited Profit: {up} (expected: True) ✓")
    print(f"Unlimited Loss: {ul} (expected: False) ✓")

    # Test 2: Short call (limited profit, unlimited loss)
    print("\n--- Test 2: Short Call (dict format) ---")
    portfolio_short_call = {
        "options": [
            {
                "option_type": "call",
                "position_type": "short",
                "strike": 100,
                "quantity": 1,
                "premium_paid": 5.0,
            }
        ],
        "stock": None,
    }
    up, ul = check_unlimited_risk(portfolio_short_call)
    assert up is False, f"Expected False, got {up}"
    assert ul is True, f"Expected True, got {ul}"
    print(f"Unlimited Profit: {up} (expected: False) ✓")
    print(f"Unlimited Loss: {ul} (expected: True) ✓")

    # Test 3: Long stock (unlimited profit, limited loss)
    print("\n--- Test 3: Long Stock (dict format) ---")
    portfolio_long_stock = {
        "options": [],
        "stock": {"position_type": "long", "quantity": 100, "entry_price": 100},
    }
    up, ul = check_unlimited_risk(portfolio_long_stock)
    assert up is True, f"Expected True, got {up}"
    assert ul is False, f"Expected False, got {ul}"
    print(f"Unlimited Profit: {up} (expected: True) ✓")
    print(f"Unlimited Loss: {ul} (expected: False) ✓")

    # Test 4: Bull call spread (limited profit, limited loss)
    print("\n--- Test 4: Bull Call Spread (dict format) ---")
    portfolio_spread = {
        "options": [
            {
                "option_type": "call",
                "position_type": "long",
                "strike": 100,
                "quantity": 1,
                "premium_paid": 5.0,
            },
            {
                "option_type": "call",
                "position_type": "short",
                "strike": 110,
                "quantity": 1,
                "premium_paid": 2.0,
            },
        ],
        "stock": None,
    }
    up, ul = check_unlimited_risk(portfolio_spread)
    assert up is False, f"Expected False, got {up}"
    assert ul is False, f"Expected False, got {ul}"
    print(f"Unlimited Profit: {up} (expected: False) ✓")
    print(f"Unlimited Loss: {ul} (expected: False) ✓")

    # Test 5: Conversion functions
    print("\n--- Test 5: Conversion Functions ---")
    positions = convert_dict_positions(portfolio_spread["options"])
    assert len(positions) == 2, f"Expected 2 positions, got {len(positions)}"
    assert positions[0].is_long is True, "First position should be long"
    assert positions[1].is_short is True, "Second position should be short"
    print(f"Converted {len(positions)} positions ✓")
    print(
        f"Position 1: long={positions[0].is_long}, call={positions[0].is_call}, K={positions[0].strike} ✓"
    )
    print(
        f"Position 2: long={positions[1].is_long}, call={positions[1].is_call}, K={positions[1].strike} ✓"
    )

    stock = convert_dict_stock(portfolio_long_stock["stock"])
    assert stock is not None, "Stock should not be None"
    assert stock.is_long is True, "Stock should be long"
    assert stock.quantity == 100, f"Expected 100, got {stock.quantity}"
    print(f"Converted stock: long={stock.is_long}, qty={stock.quantity} ✓")

    print("\n" + "=" * 50)
    print("Risk Analyzer Adapter smoke test passed! ✓")
    print("=" * 50)
