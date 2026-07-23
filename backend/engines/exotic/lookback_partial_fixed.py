"""
Partial-time fixed-strike lookback kernel (Heynen-Kat 1994c, Haug 4.15.4).

The lookback period starts at a predetermined date ``t1`` *after* the contract
is initiated and runs to expiry ``T2``; the strike ``X`` is fixed. The call pays
``max(S_max - X, 0)`` and the put ``max(X - S_min, 0)`` where the extreme is
taken over ``[t1, T2]`` only. Because the running extreme is observed over a
shorter window than a standard fixed-strike lookback, this option is cheaper.
As ``t1 -> 0`` the window spans the whole life and the price collapses to the
standard fixed-strike lookback (Conze-Viswanathan).

Ported VERBATIM from Haug's published VBA ``PartialFixedLB`` (Lookback.bas),
which uses cost-of-carry ``b`` and the bivariate normal CDF ``CBND`` (here
``cbnd``) and the univariate ``CND`` (here ``norm_cdf``). Validated against
Haug's Table 4-10.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.math import cbnd, norm_cdf


@njit(fastmath=True, cache=True)
def partial_fixed_lookback_price(
    S: float,
    X: float,
    t1: float,
    T2: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
) -> float:
    """
    Price a partial-time fixed-strike lookback (Heynen-Kat 1994c).

    Parameters
    ----------
    S : float
        Spot price.
    X : float
        Fixed strike price.
    t1 : float
        Start of the lookback window ``[t1, T2]`` (``0 <= t1 < T2``).
    T2 : float
        Time to expiration.
    r, q, sigma : float
        Rate, dividend yield, volatility. Cost of carry ``b = r - q``.
    is_call : bool
        Call (lookback on the maximum) or put (on the minimum).

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

    vsq = sigma * sigma
    sqrt_T2 = math.sqrt(T2)
    sqrt_t1 = math.sqrt(t1)
    sqrt_Tt = math.sqrt(T2 - t1)
    v_T2 = sigma * sqrt_T2
    v_Tt = sigma * sqrt_Tt
    v_t1 = sigma * sqrt_t1

    d1 = (math.log(S / X) + (b + 0.5 * vsq) * T2) / v_T2
    d2 = d1 - v_T2
    e1 = ((b + 0.5 * vsq) * (T2 - t1)) / v_Tt
    e2 = e1 - v_Tt
    f1 = (math.log(S / X) + (b + 0.5 * vsq) * t1) / v_t1
    f2 = f1 - v_t1

    df_b = math.exp((b - r) * T2)
    df = math.exp(-r * T2)
    rho_tt = math.sqrt(t1 / T2)
    rho_1 = math.sqrt(1.0 - t1 / T2)
    hs = math.pow(S / X, -2.0 * b / vsq)
    two_b_T2 = 2.0 * b * sqrt_T2 / sigma
    two_b_t1 = 2.0 * b * sqrt_t1 / sigma

    if is_call:
        return (
            S * df_b * norm_cdf(d1)
            - df * X * norm_cdf(d2)
            + S
            * df
            * vsq
            / (2.0 * b)
            * (
                -hs * cbnd(d1 - two_b_T2, -f1 + two_b_t1, -rho_tt)
                + math.exp(b * T2) * cbnd(e1, d1, rho_1)
            )
            - S * df_b * cbnd(-e1, d1, -rho_1)
            - X * df * cbnd(f2, -d2, -rho_tt)
            + math.exp(-b * (T2 - t1))
            * (1.0 - vsq / (2.0 * b))
            * S
            * df_b
            * norm_cdf(f1)
            * norm_cdf(-e2)
        )

    return (
        X * df * norm_cdf(-d2)
        - S * df_b * norm_cdf(-d1)
        + S
        * df
        * vsq
        / (2.0 * b)
        * (
            hs * cbnd(-d1 + two_b_T2, f1 - two_b_t1, -rho_tt)
            - math.exp(b * T2) * cbnd(-e1, -d1, rho_1)
        )
        + S * df_b * cbnd(e1, -d1, -rho_1)
        + X * df * cbnd(-f2, d2, -rho_tt)
        - math.exp(-b * (T2 - t1))
        * (1.0 - vsq / (2.0 * b))
        * S
        * df_b
        * norm_cdf(-f1)
        * norm_cdf(e2)
    )
