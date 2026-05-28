"""
Position Classes
================

Position classes for option portfolio management.

This module provides:
- PortfolioPosition: Wraps a VanillaOption with quantity and premium
- StockPosition: Represents stock/underlying positions
- Factory functions: long_call, short_call, long_put, short_put, long_stock, short_stock

The Instrument (VanillaOption) encapsulates the payoff structure.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from backend.instruments.options import VanillaOption

if TYPE_CHECKING:
    from backend.core.structured_product import StructuredProduct

# =============================================================================
# POSITION CLASSES
# =============================================================================


@dataclass(frozen=True)
class PortfolioPosition:
    """
    Position in a portfolio = Instrument + quantity + premium.

    Immutable. Use the instrument directly for pricing via engines.

    Parameters
    ----------
    instrument : VanillaOption
        The underlying option instrument
    quantity : int
        Position size: positive = long, negative = short
    premium : float
        Premium paid (>0) or received (<0) per unit
    """

    instrument: VanillaOption
    quantity: int
    premium: float = 0.0

    def __post_init__(self) -> None:
        """Validate position."""
        if self.quantity == 0:
            raise ValueError("quantity cannot be zero")

    @property
    def sign(self) -> int:
        """Position direction: +1 long, -1 short."""
        return 1 if self.quantity > 0 else -1

    @property
    def is_long(self) -> bool:
        """True if long position."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """True if short position."""
        return self.quantity < 0

    @property
    def strike(self) -> float:
        """Strike price (from instrument)."""
        return self.instrument.strike

    @property
    def maturity(self) -> float:
        """Time to maturity (from instrument)."""
        return self.instrument.maturity

    @property
    def is_call(self) -> bool:
        """True if call option."""
        return self.instrument.is_call

    @property
    def is_put(self) -> bool:
        """True if put option."""
        return not self.instrument.is_call

    def payoff_at_expiry(self, spot: float | np.ndarray) -> float | np.ndarray:
        """
        Calculate P&L at expiry.

        P&L = quantity * (intrinsic_value - premium)

        Parameters
        ----------
        spot : float or np.ndarray
            Spot price(s) at expiry

        Returns
        -------
        float or np.ndarray
            P&L including premium
        """
        spot_arr = np.atleast_1d(np.asarray(spot, dtype=float))
        intrinsic = self.instrument.payoff(spot_arr)
        pnl = self.quantity * (intrinsic - self.premium)
        return float(pnl[0]) if np.isscalar(spot) else pnl

    def intrinsic_value(self, spot: float) -> float:
        """
        Calculate intrinsic value at a given spot price.

        Parameters
        ----------
        spot : float
            Current spot price

        Returns
        -------
        float
            Intrinsic value (>=0)
        """
        return float(self.instrument.payoff(np.array([spot]))[0])


@dataclass(frozen=True)
class StockPosition:
    """
    Stock/underlying position.

    Linear P&L - no model needed for valuation.

    Parameters
    ----------
    quantity : int
        Number of shares: positive = long, negative = short
    entry_price : float
        Average entry price per share
    """

    quantity: int
    entry_price: float = 0.0

    def __post_init__(self) -> None:
        """Validate position."""
        if self.quantity == 0:
            raise ValueError("quantity cannot be zero")
        if self.entry_price < 0:
            raise ValueError(f"entry_price cannot be negative, got {self.entry_price}")

    @property
    def sign(self) -> int:
        """Position direction: +1 long, -1 short."""
        return 1 if self.quantity > 0 else -1

    @property
    def is_long(self) -> bool:
        """True if long position."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """True if short position."""
        return self.quantity < 0

    @property
    def delta(self) -> float:
        """Stock delta = quantity."""
        return float(self.quantity)

    def pnl(self, spot: float | np.ndarray) -> float | np.ndarray:
        """
        P&L at given spot price(s).

        Parameters
        ----------
        spot : float or np.ndarray
            Current spot price(s)

        Returns
        -------
        float or np.ndarray
            P&L
        """
        return self.quantity * (np.asarray(spot) - self.entry_price)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def long_call(
    strike: float,
    maturity: float,
    quantity: int = 1,
    premium: float = 0.0,
) -> PortfolioPosition:
    """
    Create a long call position.

    Parameters
    ----------
    strike : float
        Strike price
    maturity : float
        Time to maturity in years
    quantity : int
        Number of contracts (must be positive)
    premium : float
        Premium paid per contract

    Returns
    -------
    PortfolioPosition
        Long call position
    """
    instrument = VanillaOption(strike=strike, maturity=maturity, is_call=True)
    return PortfolioPosition(
        instrument=instrument, quantity=abs(quantity), premium=premium
    )


def short_call(
    strike: float,
    maturity: float,
    quantity: int = 1,
    premium: float = 0.0,
) -> PortfolioPosition:
    """
    Create a short call position.

    Parameters
    ----------
    strike : float
        Strike price
    maturity : float
        Time to maturity in years
    quantity : int
        Number of contracts (must be positive, will be negated)
    premium : float
        Premium received per contract

    Returns
    -------
    PortfolioPosition
        Short call position
    """
    instrument = VanillaOption(strike=strike, maturity=maturity, is_call=True)
    return PortfolioPosition(
        instrument=instrument, quantity=-abs(quantity), premium=premium
    )


def long_put(
    strike: float,
    maturity: float,
    quantity: int = 1,
    premium: float = 0.0,
) -> PortfolioPosition:
    """
    Create a long put position.

    Parameters
    ----------
    strike : float
        Strike price
    maturity : float
        Time to maturity in years
    quantity : int
        Number of contracts (must be positive)
    premium : float
        Premium paid per contract

    Returns
    -------
    PortfolioPosition
        Long put position
    """
    instrument = VanillaOption(strike=strike, maturity=maturity, is_call=False)
    return PortfolioPosition(
        instrument=instrument, quantity=abs(quantity), premium=premium
    )


def short_put(
    strike: float,
    maturity: float,
    quantity: int = 1,
    premium: float = 0.0,
) -> PortfolioPosition:
    """
    Create a short put position.

    Parameters
    ----------
    strike : float
        Strike price
    maturity : float
        Time to maturity in years
    quantity : int
        Number of contracts (must be positive, will be negated)
    premium : float
        Premium received per contract

    Returns
    -------
    PortfolioPosition
        Short put position
    """
    instrument = VanillaOption(strike=strike, maturity=maturity, is_call=False)
    return PortfolioPosition(
        instrument=instrument, quantity=-abs(quantity), premium=premium
    )


def long_stock(quantity: int = 100, entry_price: float = 0.0) -> StockPosition:
    """
    Create a long stock position.

    Parameters
    ----------
    quantity : int
        Number of shares (must be positive)
    entry_price : float
        Average entry price

    Returns
    -------
    StockPosition
        Long stock position
    """
    return StockPosition(quantity=abs(quantity), entry_price=entry_price)


@dataclass(frozen=True)
class StructuredProductPosition:
    """
    Position in a structured product.

    Parameters
    ----------
    product : StructuredProduct
        The structured product.
    quantity : int
        Number of units (positive = long, negative = short).
    entry_price : float
        Price paid as % of notional (e.g., 100.0 = par).
    """

    product: StructuredProduct
    quantity: int
    entry_price: float = 0.0

    def __post_init__(self) -> None:
        if self.quantity == 0:
            raise ValueError("quantity cannot be zero")

    @property
    def sign(self) -> int:
        """Position direction: +1 long, -1 short."""
        return 1 if self.quantity > 0 else -1

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def notional_exposure(self) -> float:
        """Total notional exposure = |quantity| * product notional."""
        return abs(self.quantity) * self.product.notional


def short_stock(quantity: int = 100, entry_price: float = 0.0) -> StockPosition:
    """
    Create a short stock position.

    Parameters
    ----------
    quantity : int
        Number of shares (must be positive, will be negated)
    entry_price : float
        Average entry price

    Returns
    -------
    StockPosition
        Short stock position
    """
    return StockPosition(quantity=-abs(quantity), entry_price=entry_price)


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    import numpy as np

    print("=" * 50)
    print("Positions Module Smoke Test")
    print("=" * 50)

    # Test factory functions
    print("\n--- Factory Functions ---")
    call = long_call(strike=100, maturity=0.5, quantity=1, premium=5.0)
    print(
        f"Long Call: strike={call.strike}, is_long={call.is_long}, is_call={call.is_call}"
    )

    put = short_put(strike=95, maturity=0.5, quantity=2, premium=3.0)
    print(
        f"Short Put: strike={put.strike}, quantity={put.quantity}, is_short={put.is_short}"
    )

    stock = long_stock(quantity=100, entry_price=100.0)
    print(f"Long Stock: quantity={stock.quantity}, entry={stock.entry_price}")

    # Test payoff calculations
    print("\n--- Payoff at Expiry ---")
    spots = np.array([90.0, 100.0, 110.0])

    call_pnl = call.payoff_at_expiry(spots)
    print(f"Long Call (K=100, premium=5) at spots {spots}:")
    print(f"  P&L: {call_pnl}")  # Expected: [-5, -5, 5]

    put_pnl = put.payoff_at_expiry(spots)
    print(f"Short Put (K=95, premium=3, qty=-2) at spots {spots}:")
    print(f"  P&L: {put_pnl}")  # Short put profits when spot > strike

    stock_pnl = stock.pnl(spots)
    print(f"Long Stock (100 shares at 100) at spots {spots}:")
    print(f"  P&L: {stock_pnl}")  # Expected: [-1000, 0, 1000]

    # Test single spot
    print("\n--- Single Spot ---")
    single_pnl = call.payoff_at_expiry(105.0)
    print(f"Long Call P&L at spot=105: {single_pnl}")  # Expected: 0

    # Test immutability
    print("\n--- Immutability Test ---")
    try:
        call.quantity = 10  # type: ignore
        print("ERROR: Mutation should have failed!")
    except Exception as e:
        print(f"Correctly prevented mutation: {type(e).__name__}")

    print("\n" + "=" * 50)
    print("Positions smoke test passed")
    print("=" * 50)
