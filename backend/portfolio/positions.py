"""
Position Classes
=================

Dataclasses representing option and stock positions in a portfolio.

Author: Derivatives Pricing Project
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OptionType(Enum):
    """Type of option contract."""
    CALL = "call"
    PUT = "put"

    def __eq__(self, other):
        """Support comparison with strings for backward compatibility."""
        if isinstance(other, str):
            return self.value == other.lower()
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return self.value

    def upper(self):
        """Return uppercase string for frontend compatibility."""
        return self.value.upper()

    def lower(self):
        """Return lowercase string for frontend compatibility."""
        return self.value.lower()


class PositionType(Enum):
    """Direction of the position."""
    LONG = "long"
    SHORT = "short"

    def __eq__(self, other):
        """Support comparison with strings for backward compatibility."""
        if isinstance(other, str):
            return self.value == other.lower()
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return self.value

    def upper(self):
        """Return uppercase string for frontend compatibility."""
        return self.value.upper()

    def lower(self):
        """Return lowercase string for frontend compatibility."""
        return self.value.lower()

    @property
    def sign(self) -> int:
        """Return +1 for long, -1 for short."""
        return 1 if self.value == "long" else -1


@dataclass
class OptionPosition:
    """
    Represents an option position in a portfolio.

    Parameters
    ----------
    option_type : str or OptionType
        Type of option: 'call' or 'put'
    position_type : str or PositionType
        Direction: 'long' or 'short'
    strike : float
        Strike price of the option
    quantity : int, default=1
        Number of contracts
    premium : float, default=0.0
        Premium paid (positive) or received (negative) per contract

    Examples
    --------
    >>> pos = OptionPosition('call', 'long', strike=100, quantity=10, premium=5.0)
    >>> pos.option_type
    <OptionType.CALL: 'call'>
    >>> pos.sign
    1
    """
    option_type: OptionType
    position_type: PositionType
    strike: float
    quantity: int = 1
    premium: float = 0.0

    def __post_init__(self):
        """Validate and convert string inputs to enums."""
        # Convert strings to enums if necessary
        if isinstance(self.option_type, str):
            try:
                self.option_type = OptionType(self.option_type.lower())
            except ValueError:
                raise ValueError(f"option_type must be 'call' or 'put', got '{self.option_type}'")

        if isinstance(self.position_type, str):
            try:
                self.position_type = PositionType(self.position_type.lower())
            except ValueError:
                raise ValueError(f"position_type must be 'long' or 'short', got '{self.position_type}'")

        # Validate numeric fields
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")

    @property
    def sign(self) -> int:
        """Return position sign: +1 for long, -1 for short."""
        return self.position_type.sign

    @property
    def is_call(self) -> bool:
        """True if this is a call option."""
        return self.option_type == OptionType.CALL

    @property
    def is_put(self) -> bool:
        """True if this is a put option."""
        return self.option_type == OptionType.PUT

    @property
    def is_long(self) -> bool:
        """True if this is a long position."""
        return self.position_type == PositionType.LONG

    @property
    def is_short(self) -> bool:
        """True if this is a short position."""
        return self.position_type == PositionType.SHORT

    @property
    def premium_paid(self) -> float:
        """Alias for premium (backward compatibility with frontend)."""
        return self.premium

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
        if self.is_call:
            return max(spot - self.strike, 0.0)
        else:
            return max(self.strike - spot, 0.0)

    def payoff_at_expiry(self, spot: float) -> float:
        """
        Calculate P&L at expiry for this position.

        Parameters
        ----------
        spot : float
            Spot price at expiry

        Returns
        -------
        float
            Total P&L including premium
        """
        intrinsic = self.intrinsic_value(spot)
        # Long pays premium, short receives it
        return self.sign * self.quantity * (intrinsic - self.premium)


@dataclass
class StockPosition:
    """
    Represents a stock/underlying position in a portfolio.

    Parameters
    ----------
    position_type : str or PositionType
        Direction: 'long' or 'short'
    quantity : int, default=100
        Number of shares
    entry_price : float, default=0.0
        Average entry price per share

    Examples
    --------
    >>> stock = StockPosition('long', quantity=100, entry_price=50.0)
    >>> stock.sign
    1
    >>> stock.pnl(55.0)
    500.0
    """
    position_type: PositionType
    quantity: int = 100
    entry_price: float = 0.0

    def __post_init__(self):
        """Validate and convert string inputs to enums."""
        if isinstance(self.position_type, str):
            try:
                self.position_type = PositionType(self.position_type.lower())
            except ValueError:
                raise ValueError(f"position_type must be 'long' or 'short', got '{self.position_type}'")

        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")
        if self.entry_price < 0:
            raise ValueError(f"entry_price cannot be negative, got {self.entry_price}")

    @property
    def sign(self) -> int:
        """Return position sign: +1 for long, -1 for short."""
        return self.position_type.sign

    @property
    def is_long(self) -> bool:
        """True if this is a long position."""
        return self.position_type == PositionType.LONG

    @property
    def is_short(self) -> bool:
        """True if this is a short position."""
        return self.position_type == PositionType.SHORT

    def pnl(self, spot: float) -> float:
        """
        Calculate P&L at a given spot price.

        Parameters
        ----------
        spot : float
            Current spot price

        Returns
        -------
        float
            Total P&L
        """
        return self.sign * self.quantity * (spot - self.entry_price)

    @property
    def delta(self) -> float:
        """Delta of stock position (always +/- quantity)."""
        return self.sign * self.quantity
