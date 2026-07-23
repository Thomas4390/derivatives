"""
Log contract and log option kernels (Haug 4.5).

A *log contract* (Neuberger 1994/1996) pays ``ln(S_T / X)`` at maturity -- not
strictly an option, but a key building block for variance/volatility
derivatives. Its value is the discounted risk-neutral expected log-return, with
cost of carry ``b = r - q`` (Haug 4.14):

    L = e^{-rT} [ ln(S/X) + (b - sigma^2/2) T ]

The *log(S) contract* (Haug 4.15), whose payoff is simply ``ln(S_T)``, is the
``X = 1`` special case (``ln(S/1) = ln(S)``) and is priced by the same kernel.

A *log option* (Wilmott 2000) pays ``max(ln(S_T / X), 0)`` -- an option on the
asset's log-return, struck at ``ln(X)``. Its value is (Haug 4.16):

    c = e^{-rT} n(d2) sigma sqrt(T) + e^{-rT} [ln(S/X) + (b - sigma^2/2) T] N(d2)
    d2 = [ln(S/X) + (b - sigma^2/2) T] / (sigma sqrt(T))

where ``N`` is the standard normal CDF and ``n`` its density. Both forms are
transcribed directly from Haug 4.14/4.16 and validated against the book's
worked examples (log contract 0.1200, log(S) contract 4.4153) and Table 4-4
(log option). Univariate -- no bivariate normal, no root-finding.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.math import norm_cdf, norm_pdf


@njit(fastmath=True, cache=True)
def log_contract_price(
    S: float,
    X: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """
    Price a log contract (Haug 4.14): payoff ``ln(S_T / X)`` at maturity.

    The ``X = 1`` case recovers the log(S) contract (Haug 4.15, payoff
    ``ln(S_T)``), so the two share one kernel.

    Parameters
    ----------
    S : float
        Spot price (> 0).
    X : float
        Strike (> 0); ``X = 1`` for the log(S) contract.
    T : float
        Time to maturity in years (> 0).
    r, q, sigma : float
        Rate, dividend yield, volatility. Cost of carry ``b = r - q``.

    Returns
    -------
    float
        Contract value (can be negative -- it is a contract, not an option).
    """
    b = r - q
    return math.exp(-r * T) * (math.log(S / X) + (b - 0.5 * sigma * sigma) * T)


@njit(fastmath=True, cache=True)
def log_option_price(
    S: float,
    X: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """
    Price a log option (Haug 4.16): payoff ``max(ln(S_T / X), 0)`` at maturity.

    Parameters
    ----------
    S : float
        Spot price (> 0).
    X : float
        Strike (> 0); the option is on the log-return, struck at ``ln(X)``.
    T : float
        Time to maturity in years (> 0).
    r, q, sigma : float
        Rate, dividend yield, volatility. Cost of carry ``b = r - q``.

    Returns
    -------
    float
        Option price (always >= 0).
    """
    b = r - q
    vst = sigma * math.sqrt(T)
    # m = E^Q[ln(S_T / X)] up to discounting (the log-contract numerator).
    m = math.log(S / X) + (b - 0.5 * sigma * sigma) * T
    d2 = m / vst
    disc = math.exp(-r * T)
    # Float-typed local keeps mypy's no-any-return at zero (norm_pdf/norm_cdf are
    # untyped njit functions that otherwise widen the expression to Any).
    price: float = disc * norm_pdf(d2) * vst + disc * m * norm_cdf(d2)
    return price
