"""
Core Constants (backward-compatible re-exports)
================================================

All constants now live in backend.utils.constants.
This file re-exports them for backward compatibility.

Author: Thomas Vaudescal
Created: 2026
"""

from backend.utils.constants.monte_carlo import (
    DEFAULT_MC_PATHS,
    DEFAULT_MC_STEPS_PER_YEAR,
    DEFAULT_RATE_BUMP,
    DEFAULT_SPOT_BUMP,
    DEFAULT_TIME_BUMP_DAYS,
    DEFAULT_VOL_BUMP,
)
from backend.utils.constants.time import (
    CALENDAR_DAYS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
)

__all__ = [
    "CALENDAR_DAYS_PER_YEAR",
    "DEFAULT_MC_PATHS",
    "DEFAULT_MC_STEPS_PER_YEAR",
    "DEFAULT_RATE_BUMP",
    "DEFAULT_SPOT_BUMP",
    "DEFAULT_TIME_BUMP_DAYS",
    "DEFAULT_VOL_BUMP",
    "TRADING_DAYS_PER_YEAR",
]
