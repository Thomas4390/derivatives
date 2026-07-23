"""
Double-barrier option pricing kernel (Ikeda-Kunitomo 1992).

A double-barrier option is knocked out (or in) if the underlying touches the
lower boundary ``L`` or the upper boundary ``U`` before expiry. Only the
double-knock-OUT is expressed in closed form; the knock-IN is recovered by
parity (vanilla minus knock-out). Curvature parameters ``delta1``/``delta2``
let the flat boundaries grow/decay exponentially (Haug eq. 4.57-4.58):

- delta1 = delta2 = 0 : two flat boundaries.
- delta1 < 0 < delta2 : lower boundary grows, upper boundary decays.
- delta1 > 0 > delta2 : convex-downward lower, convex-upward upper.

Ported from Haug's published VBA ``DoubleBarrier``.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.constants.exotic import DOUBLE_BARRIER_SERIES_N
from backend.utils.math import bs_price as _bs_price_canonical
from backend.utils.math import norm_cdf


@njit(fastmath=True, cache=True)
def double_barrier_price(
    S: float,
    K: float,
    L: float,
    U: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
    is_knock_in: bool = False,
    delta1: float = 0.0,
    delta2: float = 0.0,
) -> float:
    """
    Price a double-barrier option (Ikeda-Kunitomo 1992).

    Parameters
    ----------
    S, K : float
        Spot and strike.
    L, U : float
        Lower and upper barriers (require ``L < S < U`` for an unbreached
        option; ``0 < L < U``).
    T, r, q, sigma : float
        Maturity, risk-free rate, dividend yield, volatility.
    is_call : bool
        Call (True) or put (False).
    is_knock_in : bool
        Knock-in (True) via parity, or knock-out (False).
    delta1, delta2 : float
        Upper/lower boundary curvature (0 = flat).

    Returns
    -------
    float
        Option price.
    """
    b = r - q

    if T <= 0.0:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        breached = S <= L or S >= U
        if is_knock_in:
            return intrinsic if breached else 0.0
        return 0.0 if breached else intrinsic

    # Spot already outside the corridor: knock-out is dead, knock-in is vanilla.
    if S <= L or S >= U:
        if is_knock_in:
            return _bs_price_canonical(S, K, T, r, sigma, is_call, q)
        return 0.0

    if sigma <= 0.0:
        # Deterministic, monotone forward path S*exp(b*t).
        f_terminal = S * math.exp(b * T)
        df = math.exp(-r * T)
        path_max = max(S, f_terminal)
        path_min = min(S, f_terminal)
        breached = path_max >= U or path_min <= L
        intrinsic = max(f_terminal - K, 0.0) if is_call else max(K - f_terminal, 0.0)
        if is_knock_in:
            return intrinsic * df if breached else 0.0
        return 0.0 if breached else intrinsic * df

    sqrt_T = math.sqrt(T)
    vsq = sigma * sigma
    drift = (b + 0.5 * vsq) * T
    df = math.exp(-r * T)
    carry = math.exp((b - r) * T)
    v_sqrt_T = sigma * sqrt_T

    f_up = U * math.exp(delta1 * T)
    e_lo = L * math.exp(delta2 * T)

    sum1 = 0.0
    sum2 = 0.0
    n_max = DOUBLE_BARRIER_SERIES_N
    for n in range(-n_max, n_max + 1):
        # math.pow with float exponents throughout: njit treats `float ** int`
        # with a (possibly negative) integer exponent on a fragile code path.
        nf = float(n)
        l2n = math.pow(L, 2.0 * nf)
        u2n = math.pow(U, 2.0 * nf)
        l2n2 = math.pow(L, 2.0 * nf + 2.0)

        mu1 = 2.0 * (b - delta2 - nf * (delta1 - delta2)) / vsq + 1.0
        mu2 = 2.0 * nf * (delta1 - delta2) / vsq
        mu3 = 2.0 * (b - delta2 + nf * (delta1 - delta2)) / vsq + 1.0

        ul_n = math.pow(U / L, nf)  # (U/L)^n
        ls_mu2 = math.pow(L / S, mu2)
        lus = math.pow(L, nf + 1.0) / (math.pow(U, nf) * S)

        if is_call:
            d1 = (math.log(S * u2n / (K * l2n)) + drift) / v_sqrt_T
            d2 = (math.log(S * u2n / (f_up * l2n)) + drift) / v_sqrt_T
            d3 = (math.log(l2n2 / (K * S * u2n)) + drift) / v_sqrt_T
            d4 = (math.log(l2n2 / (f_up * S * u2n)) + drift) / v_sqrt_T
            sum1 += math.pow(ul_n, mu1) * ls_mu2 * (
                norm_cdf(d1) - norm_cdf(d2)
            ) - math.pow(lus, mu3) * (norm_cdf(d3) - norm_cdf(d4))
            sum2 += math.pow(ul_n, mu1 - 2.0) * ls_mu2 * (
                norm_cdf(d1 - v_sqrt_T) - norm_cdf(d2 - v_sqrt_T)
            ) - math.pow(lus, mu3 - 2.0) * (
                norm_cdf(d3 - v_sqrt_T) - norm_cdf(d4 - v_sqrt_T)
            )
        else:
            d1 = (math.log(S * u2n / (e_lo * l2n)) + drift) / v_sqrt_T
            d2 = (math.log(S * u2n / (K * l2n)) + drift) / v_sqrt_T
            d3 = (math.log(l2n2 / (e_lo * S * u2n)) + drift) / v_sqrt_T
            d4 = (math.log(l2n2 / (K * S * u2n)) + drift) / v_sqrt_T
            sum1 += math.pow(ul_n, mu1 - 2.0) * ls_mu2 * (
                norm_cdf(d1 - v_sqrt_T) - norm_cdf(d2 - v_sqrt_T)
            ) - math.pow(lus, mu3 - 2.0) * (
                norm_cdf(d3 - v_sqrt_T) - norm_cdf(d4 - v_sqrt_T)
            )
            sum2 += math.pow(ul_n, mu1) * ls_mu2 * (
                norm_cdf(d1) - norm_cdf(d2)
            ) - math.pow(lus, mu3) * (norm_cdf(d3) - norm_cdf(d4))

    if is_call:
        # Haug eq. 4.57: Se^{(b-r)T}[mu1, plain] - Xe^{-rT}[mu1-2, discounted].
        out_value = S * carry * sum1 - K * df * sum2
    else:
        # Haug eq. 4.58: Xe^{-rT}[mu1-2, discounted] - Se^{(b-r)T}[mu1, plain].
        out_value = K * df * sum1 - S * carry * sum2

    if is_knock_in:
        vanilla = _bs_price_canonical(S, K, T, r, sigma, is_call, q)
        return max(vanilla - out_value, 0.0)
    return max(out_value, 0.0)
