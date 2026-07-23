"""
Partial-time single-asset barrier option kernel (Heynen-Kat 1994).

The barrier is monitored only over part of the option's life. Two families:

- **Type A** (start): the barrier is live over ``[0, t1]`` only.
- **Type B** (end): the barrier is live over ``[t1, T2]`` only. ``B1`` knocks on
  any touch (direction-agnostic); ``B2`` is directional (up- or down-out).

Ported verbatim from Haug's published VBA ``PartialTimeBarrier`` (eq. 4.59-4.63);
needs the bivariate normal CDF ``cbnd``. Puts are obtained from the matching
call plus the in/out correction terms ``z1..z8``.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.math import cbnd, norm_cdf

# --- Barrier type codes ---
PTB_CDOA = 0  # call down-out, type A (start barrier)
PTB_CUOA = 1  # call up-out,   type A
PTB_COB1 = 2  # call out,      type B1 (direction-agnostic end barrier)
PTB_CDOB2 = 3  # call down-out, type B2 (directional end barrier)
PTB_CUOB2 = 4  # call up-out,   type B2
PTB_PDOA = 5  # put down-out,  type A
PTB_PUOA = 6  # put up-out,    type A
PTB_POB1 = 7  # put out,       type B1
PTB_PDOB2 = 8  # put down-out,  type B2
PTB_PUOB2 = 9  # put up-out,    type B2


@njit(fastmath=True, cache=True)
def partial_time_barrier_price(
    S: float,
    K: float,
    H: float,
    t1: float,
    T2: float,
    r: float,
    q: float,
    sigma: float,
    barrier_type: int,
) -> float:
    """
    Price a partial-time single-asset barrier option (Heynen-Kat 1994).

    Parameters
    ----------
    S, K, H : float
        Spot, strike, barrier.
    t1 : float
        Monitoring-window boundary (end of [0,t1] for type A; start of [t1,T2]
        for type B). ``0 < t1 < T2``.
    T2 : float
        Time to expiration.
    r, q, sigma : float
        Rate, dividend yield, volatility.
    barrier_type : int
        One of the ``PTB_*`` constants.

    Returns
    -------
    float
        Option price.
    """
    b = r - q
    # Keep t1 strictly inside (0, T2]: t1=0 would divide by sqrt(t1)=0, and the
    # near-zero limit reproduces the no-monitoring / full-monitoring boundary.
    if t1 <= 0.0:
        t1 = T2 * 1e-10
    elif t1 > T2:
        t1 = T2
    sqrt_t1 = math.sqrt(t1)
    sqrt_T2 = math.sqrt(T2)
    vsq = sigma * sigma

    d1 = (math.log(S / K) + (b + 0.5 * vsq) * T2) / (sigma * sqrt_T2)
    d2 = d1 - sigma * sqrt_T2
    f1 = (math.log(S / K) + 2.0 * math.log(H / S) + (b + 0.5 * vsq) * T2) / (
        sigma * sqrt_T2
    )
    f2 = f1 - sigma * sqrt_T2
    e1 = (math.log(S / H) + (b + 0.5 * vsq) * t1) / (sigma * sqrt_t1)
    e2 = e1 - sigma * sqrt_t1
    e3 = e1 + 2.0 * math.log(H / S) / (sigma * sqrt_t1)
    e4 = e3 - sigma * sqrt_t1
    mu = (b - 0.5 * vsq) / vsq
    rho = math.sqrt(t1 / T2)
    g1 = (math.log(S / H) + (b + 0.5 * vsq) * T2) / (sigma * sqrt_T2)
    g2 = g1 - sigma * sqrt_T2
    g3 = g1 + 2.0 * math.log(H / S) / (sigma * sqrt_T2)
    g4 = g3 - sigma * sqrt_T2

    hs0 = math.pow(H / S, 2.0 * mu)
    hs1 = math.pow(H / S, 2.0 * (mu + 1.0))
    df_b = math.exp((b - r) * T2)
    df = math.exp(-r * T2)

    # In/out correction terms for the put branches.
    z1 = norm_cdf(e2) - hs0 * norm_cdf(e4)
    z2 = norm_cdf(-e2) - hs0 * norm_cdf(-e4)
    z3 = cbnd(g2, e2, rho) - hs0 * cbnd(g4, -e4, -rho)
    z4 = cbnd(-g2, -e2, rho) - hs0 * cbnd(-g4, e4, -rho)
    z5 = norm_cdf(e1) - hs1 * norm_cdf(e3)
    z6 = norm_cdf(-e1) - hs1 * norm_cdf(-e3)
    z7 = cbnd(g1, e1, rho) - hs1 * cbnd(g3, -e3, -rho)
    z8 = cbnd(-g1, -e1, rho) - hs1 * cbnd(-g3, e3, -rho)

    is_put = barrier_type >= PTB_PDOA
    # Map to the base call type.
    if barrier_type == PTB_CDOA or barrier_type == PTB_PDOA:
        base = PTB_CDOA
    elif barrier_type == PTB_CUOA or barrier_type == PTB_PUOA:
        base = PTB_CUOA
    elif barrier_type == PTB_COB1 or barrier_type == PTB_POB1:
        base = PTB_COB1
    elif barrier_type == PTB_CDOB2 or barrier_type == PTB_PDOB2:
        base = PTB_CDOB2
    else:
        base = PTB_CUOB2

    # --- Base call value ---
    if base == PTB_CDOA or base == PTB_CUOA:
        eta = 1.0 if base == PTB_CDOA else -1.0
        call_val = S * df_b * (
            cbnd(d1, eta * e1, eta * rho) - hs1 * cbnd(f1, eta * e3, eta * rho)
        ) - K * df * (
            cbnd(d2, eta * e2, eta * rho) - hs0 * cbnd(f2, eta * e4, eta * rho)
        )
    elif base == PTB_COB1:
        if K > H:
            call_val = S * df_b * (
                cbnd(d1, e1, rho) - hs1 * cbnd(f1, -e3, -rho)
            ) - K * df * (cbnd(d2, e2, rho) - hs0 * cbnd(f2, -e4, -rho))
        else:
            call_val = (
                S * df_b * (cbnd(-g1, -e1, rho) - hs1 * cbnd(-g3, e3, -rho))
                - K * df * (cbnd(-g2, -e2, rho) - hs0 * cbnd(-g4, e4, -rho))
                - S * df_b * (cbnd(-d1, -e1, rho) - hs1 * cbnd(-f1, e3, -rho))
                + K * df * (cbnd(-d2, -e2, rho) - hs0 * cbnd(-f2, e4, -rho))
                + S * df_b * (cbnd(g1, e1, rho) - hs1 * cbnd(g3, -e3, -rho))
                - K * df * (cbnd(g2, e2, rho) - hs0 * cbnd(g4, -e4, -rho))
            )
    elif base == PTB_CDOB2:
        if K < H:
            call_val = S * df_b * (
                cbnd(g1, e1, rho) - hs1 * cbnd(g3, -e3, -rho)
            ) - K * df * (cbnd(g2, e2, rho) - hs0 * cbnd(g4, -e4, -rho))
        else:
            # cdoB2 with X > H reduces to coB1 (X > H).
            call_val = S * df_b * (
                cbnd(d1, e1, rho) - hs1 * cbnd(f1, -e3, -rho)
            ) - K * df * (cbnd(d2, e2, rho) - hs0 * cbnd(f2, -e4, -rho))
    else:  # PTB_CUOB2 (defined for X < H)
        call_val = (
            S * df_b * (cbnd(-g1, -e1, rho) - hs1 * cbnd(-g3, e3, -rho))
            - K * df * (cbnd(-g2, -e2, rho) - hs0 * cbnd(-g4, e4, -rho))
            - S * df_b * (cbnd(-d1, -e1, rho) - hs1 * cbnd(e3, -f1, -rho))
            + K * df * (cbnd(-d2, -e2, rho) - hs0 * cbnd(e4, -f2, -rho))
        )

    if not is_put:
        return call_val

    # --- Put = call + in/out correction ---
    if base == PTB_CDOA:
        return call_val - S * df_b * z5 + K * df * z1
    if base == PTB_CUOA:
        return call_val - S * df_b * z6 + K * df * z2
    if base == PTB_COB1:
        return call_val - S * df_b * z8 + K * df * z4 - S * df_b * z7 + K * df * z3
    if base == PTB_CDOB2:
        return call_val - S * df_b * z7 + K * df * z3
    # PTB_CUOB2
    return call_val - S * df_b * z8 + K * df * z4
