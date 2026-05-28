"""
Reparametrization and Numerical-Stability Constants
====================================================

Clamp bounds and stability thresholds used by the JAX calibration
reparametrization layer (softplus/tanh transforms) and by Numba-optimized
pricing paths where the default ``MIN_MATURITY`` floor is too coarse.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Reparametrization clamps (JAX calibration constraints layer)
# ---------------------------------------------------------------------------

# Clamp magnitude for the Heston/Bates correlation rho so that tanh remains
# invertible (atanh(rho) finite). Applied as (-RHO_CLAMP, +RHO_CLAMP).
RHO_CLAMP: Final[float] = 0.999

# Threshold above which softplus(x) = log(1 + exp(x)) is replaced by x to
# avoid floating-point overflow in the inverse (log(exp(x) - 1)) direction.
SOFTPLUS_STABILITY_THRESHOLD: Final[float] = 20.0

# ---------------------------------------------------------------------------
# Numba-path numerical floors
# ---------------------------------------------------------------------------

# Minimum time-to-maturity used in Numba-compiled residual functions where
# ``MIN_MATURITY = 1e-3`` is too large (Numba paths run with longer surfaces).
NUMBA_EPS_MATURITY: Final[float] = 1e-10

# Minimum vega used as a denominator when normalizing IV-space residuals.
NUMBA_EPS_VEGA: Final[float] = 1e-6
