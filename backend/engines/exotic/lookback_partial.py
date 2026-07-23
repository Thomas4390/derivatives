"""
Partial-time floating-strike lookback kernel (Heynen-Kat 1994c, Haug 4.15.3).

The lookback window covers only the *start* of the option's life, ``[0, t1]``;
from ``t1`` to expiry ``T2`` the strike is locked at the extreme observed by
``t1`` (minimum for a call, maximum for a put). As ``t1 -> T2`` the window spans
the whole life and the price collapses to the standard floating-strike lookback
(Goldman-Sosin-Gatto). The factor ``lam`` (lambda) enables "fractional"
lookbacks where the strike is fixed at a fraction of the actual extreme
(``lam >= 1`` for calls, ``0 < lam <= 1`` for puts); ``lam = 1`` is the standard
case.

Transcribed from Haug eq. (4.45) (call) / (4.46) (put) -- Haug's published VBA
ships only the fixed-strike partial lookback, so this floating variant is ported
from the book formula and validated against his Table 4-9. Needs the bivariate
normal CDF ``cbnd``.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.math import cbnd, norm_cdf


@njit(fastmath=True, cache=True)
def partial_float_lookback_price(
    S: float,
    M_min: float,
    M_max: float,
    t1: float,
    T2: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
    lam: float,
) -> float:
    """
    Price a partial-time floating-strike lookback (Heynen-Kat 1994c).

    Parameters
    ----------
    S : float
        Spot price.
    M_min, M_max : float
        Running minimum / maximum observed so far (for a freshly struck option
        both equal ``S``). The call uses ``M_min``, the put uses ``M_max``.
    t1 : float
        End of the lookback window ``[0, t1]`` (``0 < t1 <= T2``).
    T2 : float
        Time to expiration.
    r, q, sigma : float
        Rate, dividend yield, volatility. Cost of carry ``b = r - q``.
    is_call : bool
        Call (lookback on the minimum) or put (on the maximum).
    lam : float
        Fractional-lookback factor lambda (``1.0`` for the standard lookback).

    Returns
    -------
    float
        Option price.
    """
    if T2 <= 0.0 or sigma <= 0.0:
        return 0.0
    b = r - q
    # The closed form divides by b (the sigma^2/(2b) term); nudge a (near-)zero
    # carry off zero so the b -> 0 limit is approximated rather than dividing by 0.
    if -1e-8 < b < 1e-8:
        b = 1e-8
    # Keep t1 strictly inside (0, T2): the formula divides by sqrt(t1) and
    # sqrt(T2 - t1); the endpoints reproduce the degenerate / full-window limits.
    if t1 <= 0.0:
        t1 = T2 * 1e-10
    elif t1 >= T2:
        t1 = T2 * (1.0 - 1e-10)

    m0 = M_min if is_call else M_max
    vsq = sigma * sigma
    v_T2 = sigma * math.sqrt(T2)
    v_t1 = sigma * math.sqrt(t1)
    v_Tt = sigma * math.sqrt(T2 - t1)

    d1 = (math.log(S / m0) + (b + 0.5 * vsq) * T2) / v_T2
    d2 = d1 - v_T2
    e1 = ((b + 0.5 * vsq) * (T2 - t1)) / v_Tt
    e2 = e1 - v_Tt
    f1 = (math.log(S / m0) + (b + 0.5 * vsq) * t1) / v_t1
    f2 = f1 - v_t1
    g1 = math.log(lam) / v_T2
    g2 = math.log(lam) / v_Tt

    rho_tt = math.sqrt(t1 / T2)
    rho_1 = -math.sqrt(1.0 - t1 / T2)
    sqrt_t1 = math.sqrt(t1)
    sqrt_T2 = math.sqrt(T2)
    df_b = math.exp((b - r) * T2)
    df = math.exp(-r * T2)
    carry = math.pow(lam, 2.0 * b / vsq)
    hs = math.pow(S / m0, -2.0 * b / vsq)

    if is_call:
        p1 = S * df_b * norm_cdf(d1 - g1) - lam * m0 * df * norm_cdf(d2 - g1)
        p2 = (
            lam
            * S
            * df
            * vsq
            / (2.0 * b)
            * (
                hs
                * cbnd(
                    -f1 + 2.0 * b * sqrt_t1 / sigma,
                    -d1 + 2.0 * b * sqrt_T2 / sigma - g1,
                    rho_tt,
                )
                - math.exp(b * T2) * carry * cbnd(-d1 - g1, e1 + g2, rho_1)
            )
        )
        p3 = S * df_b * cbnd(-d1 + g1, e1 - g2, rho_1)
        p4 = lam * m0 * df * cbnd(-f2, d2 - g1, -rho_tt)
        p5 = (
            -math.exp(-b * (T2 - t1))
            * (1.0 + vsq / (2.0 * b))
            * lam
            * S
            * df_b
            * norm_cdf(e2 - g2)
            * norm_cdf(-f1)
        )
        return p1 + p2 + p3 + p4 + p5

    p1 = lam * m0 * df * norm_cdf(-d2 + g1) - S * df_b * norm_cdf(-d1 + g1)
    p2 = (
        lam
        * S
        * df
        * vsq
        / (2.0 * b)
        * (
            -hs
            * cbnd(
                f1 - 2.0 * b * sqrt_t1 / sigma,
                d1 - 2.0 * b * sqrt_T2 / sigma + g1,
                rho_tt,
            )
            + math.exp(b * T2) * carry * cbnd(d1 + g1, -e1 - g2, rho_1)
        )
    )
    p3 = -S * df_b * cbnd(d1 - g1, -e1 + g2, rho_1)
    p4 = -lam * m0 * df * cbnd(f2, -d2 + g1, -rho_tt)
    p5 = (
        math.exp(-b * (T2 - t1))
        * (1.0 + vsq / (2.0 * b))
        * lam
        * S
        * df_b
        * norm_cdf(-e2 + g2)
        * norm_cdf(f1)
    )
    return p1 + p2 + p3 + p4 + p5
