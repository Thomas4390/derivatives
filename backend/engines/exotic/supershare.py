"""
Supershare option kernel (Hakansson 1976, Haug 4.19.4).

A supershare option pays ``S_T / X_L`` if ``X_L < S_T < X_H`` at maturity and 0
otherwise. Portfolios of supershares build the "superfund" / SuperShares traded
products. The value is (Haug 4.88), with cost of carry ``b = r - q``:

    w = (S e^{(b-r)T} / X_L) [ N(d1) - N(d2) ]
    d1 = [ ln(S/X_L) + (b + sigma^2/2) T ] / (sigma sqrt(T))
    d2 = [ ln(S/X_H) + (b + sigma^2/2) T ] / (sigma sqrt(T))

Both ``d1`` and ``d2`` carry the ``+sigma^2/2`` (asset-or-nothing) drift: the
supershare is exactly ``(1/X_L)`` times the difference of two asset-or-nothing
calls struck at ``X_L`` and ``X_H``. Transcribed from Haug 4.88 and validated
against the book's worked example (0.7389). Univariate -- ``norm_cdf`` only, no
bivariate normal, no root-finding.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.math import norm_cdf


@njit(fastmath=True, cache=True)
def supershare_price(
    S: float,
    X_L: float,
    X_H: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """
    Price a supershare option (Hakansson 1976, Haug 4.88).

    Parameters
    ----------
    S : float
        Spot price (> 0).
    X_L : float
        Lower boundary (> 0); payoff accrues only while ``S_T > X_L``.
    X_H : float
        Upper boundary (> X_L); payoff accrues only while ``S_T < X_H``.
    T : float
        Time to maturity in years (> 0).
    r, q, sigma : float
        Rate, dividend yield, volatility. Cost of carry ``b = r - q``.

    Returns
    -------
    float
        Option price.
    """
    b = r - q
    vst = sigma * math.sqrt(T)
    drift = (b + 0.5 * sigma * sigma) * T
    d1 = (math.log(S / X_L) + drift) / vst
    d2 = (math.log(S / X_H) + drift) / vst
    # Float-typed local keeps mypy's no-any-return at zero (norm_cdf is an
    # untyped njit function that would otherwise widen the expression to Any).
    price: float = (S * math.exp((b - r) * T) / X_L) * (norm_cdf(d1) - norm_cdf(d2))
    return price
