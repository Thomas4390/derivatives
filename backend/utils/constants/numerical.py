"""
Numerical Constants
====================

Tolerances, variance floors, and numerical limits used across the backend.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Variance floors
# ---------------------------------------------------------------------------

# Minimum variance for full-truncation scheme (Heston, Bates, GARCH in AAD)
VAR_FLOOR: Final[float] = 1e-8

# Minimum variance for traditional GARCH Monte Carlo pricers (Numba path)
GARCH_VARIANCE_FLOOR: Final[float] = 1e-10

# Minimum variance for GARCH calibration log-likelihood
GARCH_CALIBRATION_VARIANCE_FLOOR: Final[float] = 1e-12

# ---------------------------------------------------------------------------
# Smoothing and approximation
# ---------------------------------------------------------------------------

# Smoothing width for log-normal indicator approximation (AAD backward pass)
SMOOTHING_EPS: Final[float] = 0.02

# ---------------------------------------------------------------------------
# Convergence and tolerances
# ---------------------------------------------------------------------------

# Minimum time-to-maturity to avoid division-by-zero in Greeks
MIN_MATURITY: Final[float] = 0.001

# General-purpose numerical convergence tolerance
DEFAULT_TOLERANCE: Final[float] = 1e-8

# ---------------------------------------------------------------------------
# Pre-computed mathematical constants
# ---------------------------------------------------------------------------

# log(2π) — used in Gaussian log-likelihood functions
LOG_2PI: Final[float] = 1.8378770664093453  # np.log(2.0 * np.pi)

# ---------------------------------------------------------------------------
# Fast Fourier Transform (Carr-Madan) defaults
# ---------------------------------------------------------------------------

# Damping factor for the Carr-Madan integrand (1.0 to 2.0 is typical).
FFT_DEFAULT_ALPHA: Final[float] = 1.5

# Number of FFT points (must be a power of 2).
FFT_DEFAULT_N: Final[int] = 4096

# Integration step size in Fourier space.
FFT_DEFAULT_ETA: Final[float] = 0.25

# ---------------------------------------------------------------------------
# Heston QE (Andersen 2008) scheme thresholds
# ---------------------------------------------------------------------------

# Threshold separating the moment-matched (QE-low) and exponential (QE-high)
# branches in the Andersen Quadratic-Exponential variance discretisation.
QE_PSI_THRESHOLD: Final[float] = 1.5

# Floor used for the QE drift term ``m`` to avoid division by zero.
QE_M_FLOOR: Final[float] = 1e-10

# ---------------------------------------------------------------------------
# Market input validation ranges
# ---------------------------------------------------------------------------

# Acceptable interval for the (annualised continuously compounded) interest
# rate. Negative rates are allowed since they appear in some markets.
RATE_MIN: Final[float] = -0.10
RATE_MAX: Final[float] = 0.50

# Maximum annualised volatility accepted by validation (500%). Anything above
# this is almost certainly a unit error rather than a real market input.
VOLATILITY_MAX: Final[float] = 5.0

# Maximum dividend yield accepted by validation (50%).
DIVIDEND_YIELD_MAX: Final[float] = 0.50

# Tolerance for the put–call parity sanity check (absolute, in price units).
PUT_CALL_PARITY_TOLERANCE: Final[float] = 0.01

# ---------------------------------------------------------------------------
# Implied-volatility solver (Newton–Raphson with bisection fallback)
# ---------------------------------------------------------------------------

# Sigma is clamped to [IV_SIGMA_MIN, VOLATILITY_MAX] on every iteration so the
# Newton iterate cannot wander into non-physical volatilities.
IV_SIGMA_MIN: Final[float] = 0.001

# Below this raw vega the Newton step (diff / vega) is ill-conditioned; the
# solver falls back to a multiplicative bisection nudge instead.
IV_VEGA_FLOOR: Final[float] = 1e-10

# Multiplicative bisection factors used when vega < IV_VEGA_FLOOR.
IV_BISECTION_DOWN: Final[float] = 0.9
IV_BISECTION_UP: Final[float] = 1.1

# ---------------------------------------------------------------------------
# Yield curve defaults
# ---------------------------------------------------------------------------

# Default tenor grid used by ``YieldCurve.flat`` and example fixtures.
DEFAULT_YIELD_CURVE_TENORS: Final[tuple[float, ...]] = (
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
)
