"""
Value objects and type aliases for numerical Greeks: the pricing-function
signature alias, the frozen bump-size config + its module-level default, and the
result NamedTuple.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple


# Pricing function signature: (spot, strike, time, rate, vol, ...) -> price
PricingFunc = Callable[..., float]


@dataclass(frozen=True)
class GreeksBumpConfig:
    """
    Configuration for finite difference bump sizes.

    Centralizes the default perturbation sizes used in numerical Greeks
    calculations. All values are industry-standard defaults.

    Parameters
    ----------
    spot_bump : float
        Relative spot bump (default 1%). Applied as: h = spot * spot_bump
    vol_bump : float
        Absolute volatility bump (default 1% = 0.01).
        Applied as: vol ± vol_bump
    time_bump_days : float
        Time decay bump in calendar days (default 1 day).
        Converted to years internally: h = time_bump_days / 365
    rate_bump : float
        Absolute rate bump in basis points (default 1bp = 0.0001).
        Applied as: rate ± rate_bump

    Examples
    --------
    config = GreeksBumpConfig()  # Use defaults
    # config.spot_bump == 0.01

    custom = GreeksBumpConfig(spot_bump=0.005, vol_bump=0.001)
    # custom.spot_bump == 0.005

    Notes
    -----
    Default values are chosen for numerical stability and practical relevance:
    - 1% spot bump: Standard for equity delta hedging
    - 1% vol bump: Standard vega reporting convention
    - 1 day theta: Daily P&L relevance
    - 1bp rate bump: Typical rate sensitivity measure
    """

    spot_bump: float = 0.01  # 1% relative
    vol_bump: float = 0.01  # 1% absolute
    time_bump_days: float = 1.0  # 1 calendar day
    rate_bump: float = 0.0001  # 1 basis point


# Module-level default configuration
DEFAULT_BUMP_CONFIG: GreeksBumpConfig = GreeksBumpConfig()


class NumericalGreeks(NamedTuple):
    """Result from numerical Greeks calculation."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
