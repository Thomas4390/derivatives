"""
Powered and capped-power option kernels (Esser 2003, Haug 4.4.3/4.4.4).

The legacy ``power.py`` already prices the *standard* power option
(``max(S^i - X, 0)``, Haug 4.4.2). This module adds the two remaining Haug 4.4
members, both ported VERBATIM from Haug's published VBA:

- **Powered option** (4.4.4, eq 4.10/4.11): payoff ``max(eta(S - X), 0)^i`` --
  the *standard* payoff is raised to a power. Valued as a binomial sum over the
  expansion of ``(S - X)^i`` (integer ``i`` only). VBA ``PoweredOption``.

- **Capped power option** (4.4.3, eq 4.8/4.9): payoff
  ``min(max(eta(S^i - X), 0), C)`` -- a standard power option whose payoff is
  capped at ``C``. Esser's closed form with the four ``e1..e4`` thresholds.
  VBA ``CappedPowerOption``.

Cost of carry ``b = r - q``. Univariate -- ``norm_cdf`` only, no bivariate
normal, no root-finding. Validated against Haug Table 4-3 (powered) and, via the
``C -> inf`` limit, Table 4-2 (the standard power option the cap relaxes to).

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.math import norm_cdf


@njit(fastmath=True, cache=True)
def _binom(n: int, k: int) -> float:
    """Binomial coefficient C(n, k) as a float (multiplicative, overflow-safe)."""
    if k < 0 or k > n:
        return 0.0
    kk = k if k < n - k else n - k
    result = 1.0
    for t in range(kk):
        result = result * (n - t) / (t + 1)
    return result


@njit(fastmath=True, cache=True)
def powered_price(
    S: float,
    X: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    i: int,
    is_call: bool,
) -> float:
    """
    Price a powered option (Esser 2003, Haug 4.10/4.11).

    Payoff ``max(S_T - X, 0)^i`` (call) or ``max(X - S_T, 0)^i`` (put); ``i`` is a
    positive integer. Binomial sum over the expansion of the powered payoff.

    Parameters
    ----------
    S, X, T, r, q, sigma : float
        Spot, strike, maturity, rate, dividend yield, volatility (``b = r - q``).
    i : int
        Power (positive integer).
    is_call : bool
        Call or put.

    Returns
    -------
    float
        Option price.
    """
    b = r - q
    vst = sigma * math.sqrt(T)
    sigma2 = sigma * sigma
    total = 0.0
    for j in range(0, i + 1):
        k = i - j  # power on the spot in this term
        d1 = (math.log(S / X) + (b + (k - 0.5) * sigma2) * T) / vst
        comb = _binom(i, j)
        expf = math.exp((k - 1.0) * (r + k * sigma2 / 2.0) * T - k * (r - b) * T)
        if is_call:
            # term = C(i,j) * S^k * (-X)^j * expf * N(d1)
            sign = -1.0 if (j % 2 == 1) else 1.0
            total += (
                comb
                * math.pow(S, float(k))
                * sign
                * math.pow(X, float(j))
                * expf
                * norm_cdf(d1)
            )
        else:
            # term = C(i,j) * (-S)^k * X^j * expf * N(-d1)
            sign = -1.0 if (k % 2 == 1) else 1.0
            total += (
                comb
                * sign
                * math.pow(S, float(k))
                * math.pow(X, float(j))
                * expf
                * norm_cdf(-d1)
            )
    return total


@njit(fastmath=True, cache=True)
def capped_power_price(
    S: float,
    X: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    i: float,
    cap: float,
    is_call: bool,
) -> float:
    """
    Price a capped power option (Esser 2003, Haug 4.8/4.9).

    Payoff ``min(max(S_T^i - X, 0), cap)`` (call) or
    ``min(max(X - S_T^i, 0), cap)`` (put). Relaxing ``cap -> inf`` recovers the
    standard power option (Haug 4.4.2).

    Parameters
    ----------
    S, X, T, r, q, sigma : float
        Spot, strike, maturity, rate, dividend yield, volatility (``b = r - q``).
    i : float
        Power (> 0).
    cap : float
        Maximum payoff ``C`` (> 0; for a put, ``cap < X``).
    is_call : bool
        Call or put.

    Returns
    -------
    float
        Option price.
    """
    b = r - q
    vst = sigma * math.sqrt(T)
    sigma2 = sigma * sigma
    e1 = (math.log(S / math.pow(X, 1.0 / i)) + (b + (i - 0.5) * sigma2) * T) / vst
    e2 = e1 - i * vst
    powf = math.pow(S, i) * math.exp(
        (i - 1.0) * (r + i * sigma2 / 2.0) * T - i * (r - b) * T
    )
    disc = math.exp(-r * T)
    if is_call:
        e3 = (
            math.log(S / math.pow(X + cap, 1.0 / i)) + (b + (i - 0.5) * sigma2) * T
        ) / vst
        e4 = e3 - i * vst
        price: float = powf * (norm_cdf(e1) - norm_cdf(e3)) - disc * (
            X * norm_cdf(e2) - (cap + X) * norm_cdf(e4)
        )
        return price
    e3 = (math.log(S / math.pow(X - cap, 1.0 / i)) + (b + (i - 0.5) * sigma2) * T) / vst
    e4 = e3 - i * vst
    price = disc * (X * norm_cdf(-e2) - (X - cap) * norm_cdf(-e4)) - powf * (
        norm_cdf(-e1) - norm_cdf(-e3)
    )
    return price
