"""
Univariate standard-normal functions (scalar + vectorized).

Numba-compiled CDF/PDF, the Acklam/A&S inverse-CDF, and the parallel vectorized
CDF/PDF. Part of the ``backend.utils.math`` single-source-of-truth package.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from backend.utils.math._constants import SQRT_2, SQRT_2PI  # noqa: F401


@njit(fastmath=True, cache=True)
def norm_cdf(x: float) -> float:
    """
    Cumulative distribution function for standard normal distribution.

    Parameters
    ----------
    x : float
        Value at which to evaluate the CDF

    Returns
    -------
    float
        P(X <= x) where X ~ N(0, 1)
    """
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


@njit(fastmath=True, cache=True)
def norm_pdf(x: float) -> float:
    """
    Probability density function for standard normal distribution.

    Parameters
    ----------
    x : float
        Value at which to evaluate the PDF

    Returns
    -------
    float
        Density at x for X ~ N(0, 1)
    """
    return math.exp(-0.5 * x * x) / SQRT_2PI


@njit(fastmath=True, cache=True)
def norm_inv_cdf(p: float) -> float:
    """
    Inverse cumulative distribution function (quantile function) for standard normal.

    Uses rational approximation (Abramowitz and Stegun approximation).

    Parameters
    ----------
    p : float
        Probability (0 < p < 1)

    Returns
    -------
    float
        x such that P(X <= x) = p where X ~ N(0, 1)
    """
    if p <= 0.0:
        return -1e10
    if p >= 1.0:
        return 1e10

    # Rational approximation constants
    a1 = -3.969683028665376e01
    a2 = 2.209460984245205e02
    a3 = -2.759285104469687e02
    a4 = 1.383577518672690e02
    a5 = -3.066479806614716e01
    a6 = 2.506628277459239e00

    b1 = -5.447609879822406e01
    b2 = 1.615858368580409e02
    b3 = -1.556989798598866e02
    b4 = 6.680131188771972e01
    b5 = -1.328068155288572e01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e00
    c4 = -2.549732539343734e00
    c5 = 4.374664141464968e00
    c6 = 2.938163982698783e00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e00
    d4 = 3.754408661907416e00

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        # Lower tail
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
            (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0
        )
    if p <= p_high:
        # Central region
        q = p - 0.5
        r = q * q
        return (
            (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
            * q
            / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        )
    # Upper tail
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
        (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0
    )


@njit(fastmath=True, cache=True, parallel=True)
def norm_cdf_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized normal CDF."""
    # float64 result regardless of input dtype: np.empty_like(x) on an
    # integer array would truncate the [0, 1] CDF values to 0/1.
    result = np.empty(x.shape, dtype=np.float64)
    for i in prange(len(x)):
        result[i] = 0.5 * (1.0 + math.erf(x[i] / SQRT_2))
    return result


@njit(fastmath=True, cache=True, parallel=True)
def norm_pdf_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized normal PDF."""
    # float64 result regardless of input dtype (see norm_cdf_vec).
    result = np.empty(x.shape, dtype=np.float64)
    for i in prange(len(x)):
        result[i] = math.exp(-0.5 * x[i] * x[i]) / SQRT_2PI
    return result
