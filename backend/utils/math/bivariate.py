"""
Bivariate normal CDF (Genz/Haug CBND) and its Gauss-Legendre node tables.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from backend.utils.math._constants import SQRT_2PI
from backend.utils.math.distributions import norm_cdf


# Gauss-Legendre abscissae/weights for the Genz (2004) bivariate normal CDF.
# Three half-rules (3/6/10 nodes) selected by |rho|, exactly as in Haug's
# published VBA CBND (which itself wraps Genz's double-precision algorithm).
_CBND_W3 = np.array([0.17132449237917, 0.360761573048138, 0.46791393457269])
_CBND_X3 = np.array([-0.932469514203152, -0.661209386466265, -0.238619186083197])
_CBND_W6 = np.array(
    [
        0.0471753363865118,
        0.106939325995318,
        0.160078328543346,
        0.203167426723066,
        0.233492536538355,
        0.249147045813403,
    ]
)
_CBND_X6 = np.array(
    [
        -0.981560634246719,
        -0.904117256370475,
        -0.769902674194305,
        -0.587317954286617,
        -0.36783149899818,
        -0.125233408511469,
    ]
)
_CBND_W10 = np.array(
    [
        0.0176140071391521,
        0.0406014298003869,
        0.0626720483341091,
        0.0832767415767048,
        0.10193011981724,
        0.118194531961518,
        0.131688638449177,
        0.142096109318382,
        0.149172986472604,
        0.152753387130726,
    ]
)
_CBND_X10 = np.array(
    [
        -0.993128599185095,
        -0.963971927277914,
        -0.912234428251326,
        -0.839116971822219,
        -0.746331906460151,
        -0.636053680726515,
        -0.510867001950827,
        -0.37370608871542,
        -0.227785851141645,
        -0.0765265211334973,
    ]
)


@njit(fastmath=True, cache=True)
def cbnd(a: float, b: float, rho: float) -> float:
    """
    Cumulative bivariate normal distribution P(X <= a, Y <= b).

    Computes the joint CDF of a standard bivariate normal with correlation
    ``rho`` -- Haug's ``M(a, b, rho)``. Uses the Genz (2004) double-precision
    algorithm (Gauss-Legendre quadrature with a ``|rho|``-dependent node count
    and a dedicated near-unit-correlation branch), ported verbatim from Haug's
    published VBA ``CBND``. The univariate ``CND`` calls are routed through the
    ``erf``-based :func:`norm_cdf` for maximal accuracy.

    Parameters
    ----------
    a : float
        Upper integration limit for the first variate.
    b : float
        Upper integration limit for the second variate.
    rho : float
        Correlation coefficient in [-1, 1].

    Returns
    -------
    float
        P(X <= a, Y <= b) where (X, Y) ~ N(0, 0, 1, 1, rho).
    """
    abs_rho = abs(rho)
    if abs_rho < 0.3:
        w = _CBND_W3
        xx = _CBND_X3
    elif abs_rho < 0.75:
        w = _CBND_W6
        xx = _CBND_X6
    else:
        w = _CBND_W10
        xx = _CBND_X10
    lg = w.shape[0]

    h = -a
    k = -b
    hk = h * k
    bvn = 0.0

    if abs_rho < 0.925:
        if abs_rho > 0.0:
            hs = (h * h + k * k) / 2.0
            asr = math.asin(rho)
            for i in range(lg):
                for iss in (-1, 1):
                    sn = math.sin(asr * (iss * xx[i] + 1.0) / 2.0)
                    bvn += w[i] * math.exp((sn * hk - hs) / (1.0 - sn * sn))
            bvn = bvn * asr / (4.0 * math.pi)
        bvn = bvn + norm_cdf(-h) * norm_cdf(-k)
    else:
        if rho < 0.0:
            k = -k
            hk = -hk
        if abs_rho < 1.0:
            ass = (1.0 - rho) * (1.0 + rho)
            a_ = math.sqrt(ass)
            bs = (h - k) * (h - k)
            c = (4.0 - hk) / 8.0
            d = (12.0 - hk) / 16.0
            asr = -(bs / ass + hk) / 2.0
            if asr > -100.0:
                bvn = (
                    a_
                    * math.exp(asr)
                    * (
                        1.0
                        - c * (bs - ass) * (1.0 - d * bs / 5.0) / 3.0
                        + c * d * ass * ass / 5.0
                    )
                )
            if -hk < 100.0:
                bb = math.sqrt(bs)
                bvn = bvn - math.exp(-hk / 2.0) * SQRT_2PI * norm_cdf(-bb / a_) * bb * (
                    1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0
                )
            a_ = a_ / 2.0
            for i in range(lg):
                for iss in (-1, 1):
                    xs = (a_ * (iss * xx[i] + 1.0)) ** 2
                    rs = math.sqrt(1.0 - xs)
                    asr = -(bs / xs + hk) / 2.0
                    if asr > -100.0:
                        bvn = bvn + a_ * w[i] * math.exp(asr) * (
                            math.exp(-hk * (1.0 - rs) / (2.0 * (1.0 + rs))) / rs
                            - (1.0 + c * xs * (1.0 + d * xs))
                        )
            bvn = -bvn / (2.0 * math.pi)
        if rho > 0.0:
            bvn = bvn + norm_cdf(-max(h, k))
        else:
            bvn = -bvn
            if k > h:
                bvn = bvn + norm_cdf(k) - norm_cdf(h)
    return bvn
