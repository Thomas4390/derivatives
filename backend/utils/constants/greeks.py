"""
Greek Constants
================

Greek array indices, scaling factors, and classification constants.

The 14-element Greek array order is used by vectorized_bs.calculate_all_greeks()
and referenced throughout the portfolio and Greeks modules.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import Final

# =============================================================================
# Greek Array Indices
# =============================================================================
# All vectorized Greeks functions return values in this order (14 total).
#
# First-order (indices 0-5):
#   0: price   - Option price
#   1: delta   - ∂V/∂S (spot sensitivity)
#   2: gamma   - ∂²V/∂S² (delta sensitivity to spot)
#   3: vega    - ∂V/∂σ per 1% vol (volatility sensitivity)
#   4: theta   - ∂V/∂t per day (time decay)
#   5: rho     - ∂V/∂r per 1% rate (rate sensitivity)
#
# Second-order (indices 6-9):
#   6: vanna   - ∂²V/∂S∂σ per 1% vol (delta-vol cross)
#   7: volga   - ∂²V/∂σ² per 1%² vol (vega convexity)
#   8: charm   - ∂²V/∂S∂t per day (delta decay)
#   9: veta    - ∂²V/∂σ∂t per day per 1% vol (vega decay)
#
# Third-order (indices 10-13):
#   10: speed  - ∂³V/∂S³ (gamma sensitivity to spot)
#   11: zomma  - ∂³V/∂S²∂σ per 1% vol (gamma-vol cross)
#   12: color  - ∂³V/∂S²∂t per day (gamma decay)
#   13: ultima - ∂³V/∂σ³ per 1%³ vol (volga sensitivity to vol)

GREEK_PRICE: Final[int] = 0
GREEK_DELTA: Final[int] = 1
GREEK_GAMMA: Final[int] = 2
GREEK_VEGA: Final[int] = 3
GREEK_THETA: Final[int] = 4
GREEK_RHO: Final[int] = 5
GREEK_VANNA: Final[int] = 6
GREEK_VOLGA: Final[int] = 7
GREEK_CHARM: Final[int] = 8
GREEK_VETA: Final[int] = 9
GREEK_SPEED: Final[int] = 10
GREEK_ZOMMA: Final[int] = 11
GREEK_COLOR: Final[int] = 12
GREEK_ULTIMA: Final[int] = 13

NUM_GREEKS: Final[int] = 14

# =============================================================================
# Greek Scaling Factors (raw mathematical → market convention)
# =============================================================================
# These factors convert raw mathematical derivatives to market-standard units.
# Scaled value = raw value / SCALE_FACTOR

VEGA_SCALE: Final[float] = 100.0  # Vega per 1% vol (0.01 decimal)
RHO_SCALE: Final[float] = 100.0  # Rho per 1% rate (0.01 decimal)
THETA_SCALE: Final[float] = 365.0  # Theta per calendar day

VANNA_SCALE: Final[float] = 100.0  # Per 1% vol
VOLGA_SCALE: Final[float] = 10000.0  # Per 1%² vol (100 × 100)
CHARM_SCALE: Final[float] = 365.0  # Per calendar day
VETA_SCALE: Final[float] = 365.0 * 100.0  # Per day per 1% vol

ZOMMA_SCALE: Final[float] = 100.0  # Per 1% vol
COLOR_SCALE: Final[float] = 365.0  # Per calendar day
ULTIMA_SCALE: Final[float] = 1000000.0  # Per 1%³ vol (100³)

# =============================================================================
# Greek Classification
# =============================================================================

GREEK_NAMES: Final[tuple[str, ...]] = (
    "price",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "vanna",
    "volga",
    "charm",
    "veta",
    "speed",
    "zomma",
    "color",
    "ultima",
)

VALID_GREEKS: Final[frozenset[str]] = frozenset(GREEK_NAMES)

FIRST_ORDER_GREEKS: Final[tuple[str, ...]] = (
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
)

SECOND_ORDER_GREEKS: Final[tuple[str, ...]] = (
    "vanna",
    "volga",
    "charm",
    "veta",
)

THIRD_ORDER_GREEKS: Final[tuple[str, ...]] = (
    "speed",
    "zomma",
    "color",
    "ultima",
)
