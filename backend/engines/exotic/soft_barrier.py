"""
Soft-barrier option pricing kernel (Hart-Ross 1994).

A soft-barrier option knocks in/out gradually over a *range* ``[L, U]`` rather
than at a single level: if the extreme of the underlying lands inside the band,
a proportional fraction of the option is knocked. Defined (per Hart-Ross) for a
soft-down call and a soft-up put -- the symmetric orientations. Ported verbatim
from Haug's published VBA ``SoftBarrier`` (eq. 4.65).

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.engines.exotic.barrier import barrier_option_price
from backend.utils.math import bs_price as _bs_price_canonical
from backend.utils.math import norm_cdf


@njit(fastmath=True, cache=True)
def soft_barrier_price(
    S: float,
    K: float,
    L: float,
    U: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
    is_knock_in: bool,
) -> float:
    """
    Price a soft-barrier option (Hart-Ross 1994).

    A call uses a soft DOWN barrier band ``[L, U]`` (below spot); a put uses a
    soft UP barrier band. As ``U -> L`` the option collapses to the standard
    (hard) barrier at that level.

    Parameters
    ----------
    S, K : float
        Spot and strike.
    L, U : float
        Lower/upper edges of the soft-barrier band (``0 < L <= U``).
    T, r, q, sigma : float
        Maturity, rate, dividend yield, volatility.
    is_call : bool
        Call (soft-down) or put (soft-up).
    is_knock_in : bool
        Knock-in (True) or knock-out (False, via parity with the vanilla).

    Returns
    -------
    float
        Option price.
    """
    b = r - q
    is_up = not is_call  # call -> down band; put -> up band (Hart-Ross convention)

    # Degenerate cases -> standard (hard) barrier at the band midpoint.
    if T <= 0.0 or sigma <= 0.0 or (U - L) < 1e-12:
        h = 0.5 * (U + L)
        return barrier_option_price(
            S, K, h, T, r, q, sigma, is_call, is_knock_in, is_up, 0.0
        )

    vsq = sigma * sigma
    mu = (b + 0.5 * vsq) / vsq
    # Removable singularities at mu = +/- 0.5 (b = 0 or b = -sigma^2): nudge.
    if abs(mu - 0.5) < 1e-10:
        mu = 0.5 + 1e-10
    if abs(mu + 0.5) < 1e-10:
        mu = -0.5 + 1e-10

    eta = 1.0 if is_call else -1.0
    v_sqrt_T = sigma * math.sqrt(T)
    lam1 = math.exp(-0.5 * vsq * T * (mu + 0.5) * (mu - 0.5))
    lam2 = math.exp(-0.5 * vsq * T * (mu - 0.5) * (mu - 1.5))

    sx = S * K
    ln_u = math.log(U * U / sx) / v_sqrt_T
    ln_l = math.log(L * L / sx) / v_sqrt_T
    d1 = ln_u + mu * v_sqrt_T
    d2 = d1 - (mu + 0.5) * v_sqrt_T
    d3 = ln_u + (mu - 1.0) * v_sqrt_T
    d4 = d3 - (mu - 0.5) * v_sqrt_T
    e1 = ln_l + mu * v_sqrt_T
    e2 = e1 - (mu + 0.5) * v_sqrt_T
    e3 = ln_l + (mu - 1.0) * v_sqrt_T
    e4 = e3 - (mu - 0.5) * v_sqrt_T

    u2 = U * U / sx
    l2 = L * L / sx
    term1 = (
        S
        * math.exp((b - r) * T)
        * math.pow(S, -2.0 * mu)
        * math.pow(sx, mu + 0.5)
        / (2.0 * (mu + 0.5))
        * (
            math.pow(u2, mu + 0.5) * norm_cdf(eta * d1)
            - lam1 * norm_cdf(eta * d2)
            - math.pow(l2, mu + 0.5) * norm_cdf(eta * e1)
            + lam1 * norm_cdf(eta * e2)
        )
    )
    term2 = (
        K
        * math.exp(-r * T)
        * math.pow(S, -2.0 * (mu - 1.0))
        * math.pow(sx, mu - 0.5)
        / (2.0 * (mu - 0.5))
        * (
            math.pow(u2, mu - 0.5) * norm_cdf(eta * d3)
            - lam2 * norm_cdf(eta * d4)
            - math.pow(l2, mu - 0.5) * norm_cdf(eta * e3)
            + lam2 * norm_cdf(eta * e4)
        )
    )
    value = eta / (U - L) * (term1 - term2)

    if is_knock_in:
        return max(value, 0.0)
    vanilla = _bs_price_canonical(S, K, T, r, sigma, is_call, q)
    return max(vanilla - value, 0.0)
