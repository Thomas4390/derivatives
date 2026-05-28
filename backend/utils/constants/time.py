"""
Time Convention Constants
=========================

Calendar and trading day conventions used across the backend.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import Final

# Calendar days per year (used in DTE-to-years conversion, theta scaling)
CALENDAR_DAYS_PER_YEAR: Final[float] = 365.0

# Alias for backward compatibility (utils/math.py historically used this name)
DAYS_PER_YEAR: Final[float] = CALENDAR_DAYS_PER_YEAR

# Trading days per year (used for annualization, MC step sizing)
TRADING_DAYS_PER_YEAR: Final[int] = 252

# Observation period mappings (fraction of year)
OBSERVATION_PERIODS: Final[dict[str, float]] = {
    "monthly": 1 / 12,
    "quarterly": 0.25,
    "semi_annual": 0.5,
    "annual": 1.0,
}
