"""
Market Default Constants
=========================

Default market environment values used across fixtures, smoke tests, and
example scripts. Do not reuse these in production pricing code — real
valuations should always take market data from the caller.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import Final

# Default spot price for examples and fixtures (100 is the ATM convention).
DEFAULT_SPOT: Final[float] = 100.0

# Default risk-free rate (continuously compounded, annualized).
DEFAULT_RATE: Final[float] = 0.05

# Default dividend yield (continuous).
DEFAULT_DIV: Final[float] = 0.0

# Default time-to-maturity (in years) for one-off examples.
DEFAULT_MATURITY: Final[float] = 1.0
