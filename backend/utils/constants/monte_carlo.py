"""
Monte Carlo Constants
======================

Default simulation parameters and bump sizes for numerical Greeks.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Default Monte Carlo simulation parameters
# ---------------------------------------------------------------------------

DEFAULT_MC_PATHS: Final[int] = 100_000
DEFAULT_MC_STEPS_PER_YEAR: Final[int] = 252

# Default seed for the model-dependent exotic Monte-Carlo engine. A fixed seed
# keeps a given (instrument, model, market) price reproducible across calls and
# is the prerequisite for stable bump-and-reprice Greeks (CRN).
DEFAULT_MC_SEED: Final[int] = 12345

# ---------------------------------------------------------------------------
# Default bump sizes for Greeks finite differences
# ---------------------------------------------------------------------------

DEFAULT_SPOT_BUMP: Final[float] = 0.01  # 1% relative spot bump
DEFAULT_VOL_BUMP: Final[float] = 0.01  # 1% absolute vol bump
DEFAULT_RATE_BUMP: Final[float] = 0.0001  # 1bp absolute rate bump
DEFAULT_TIME_BUMP_DAYS: Final[float] = 1.0  # 1 calendar day for theta
