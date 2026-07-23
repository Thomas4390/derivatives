"""
Exotic Option Constants
=======================

Finite-difference Greek bump sizes and series-truncation bounds for the
registry-dispatched closed-form exotic pricers (Haug catalog).

The finite-difference bumps deliberately match
``backend/engines/exotic/engine.py::exotic_greeks_batch`` so that the new
registry-based exotics and the legacy njit exotics share one Greek convention.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import Final

from backend.utils.constants.time import DAYS_PER_YEAR

# --- Finite-difference Greek bumps (mirror exotic_greeks_batch exactly) ---
FD_SPOT_REL_BUMP: Final[float] = 0.01  # dS = 1% of spot
FD_VOL_BUMP: Final[float] = 0.01  # d(sigma) = 1 vol point
FD_RATE_BUMP: Final[float] = 1e-4  # dr = 1 basis point
FD_TIME_BUMP: Final[float] = 1.0 / DAYS_PER_YEAR  # one calendar day
FD_SIGMA_FLOOR: Final[float] = 1e-8  # floor on the bumped-down volatility

# --- Discrete-barrier continuity correction (Broadie-Glasserman-Kou 1997) ---
# beta = zeta(1/2) / sqrt(2*pi). The continuous-barrier formula applied to a
# barrier shifted OUTWARD by exp(+/- beta*sigma*sqrt(dt)) approximates the
# discretely-monitored option.
BGK_BETA: Final[float] = 0.5826

# --- Series truncation ---
# Ikeda-Kunitomo double-barrier: terms summed for n = -N .. N. Haug's published
# VBA uses N=5; the series converges so fast (the n=0, +-1 terms dominate) that
# this reproduces his Table 4-15 to all printed digits.
DOUBLE_BARRIER_SERIES_N: Final[int] = 5

# --- Critical-value root finder (complex chooser / compound / extendible) ---
# Safeguarded bisection on a monotone Black-Scholes-Merton combination. The
# tolerance applies to both the price residual and the absolute bracket width;
# the iteration cap is only a backstop (a 1e8-wide bracket converges to
# ROOTFIND_TOL in well under 60 bisection steps).
ROOTFIND_TOL: Final[float] = 1e-8
ROOTFIND_MAX_ITER: Final[int] = 200
