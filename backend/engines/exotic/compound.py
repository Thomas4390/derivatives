"""
Compound option kernel (options on options, Geske 1979, Haug 4.13).

A compound option is an option whose underlying is itself a vanilla option. At
the compound expiry ``t1`` the holder may pay/receive the compound strike ``K2``
to take a position in the underlying option (strike ``K1``, expiry ``T2 > t1``).
Four types:

* call-on-call (cc): ``max(c_BSM(S, K1, T2) - K2, 0)``
* put-on-call  (pc): ``max(K2 - c_BSM(S, K1, T2), 0)``
* call-on-put  (cp): ``max(p_BSM(S, K1, T2) - K2, 0)``
* put-on-put   (pp): ``max(K2 - p_BSM(S, K1, T2), 0)``

Ported VERBATIM from Haug's published VBA ``OptionsOnOptions`` /
``CriticalValueOptionsOnOptions`` (book section 4.13, eq 4.28-4.31). The critical
asset price ``I`` at ``t1`` where the underlying option is exactly worth ``K2``
is found by the shared safeguarded bisection in
:mod:`backend.engines.exotic._rootfind` (the book uses Newton-Raphson; bisection
on the monotone underlying-BSM value reaches the same root and keeps the kernel
cache-able). Validated against Haug's worked put-on-call example
``= 21.1965`` (critical ``I = 538.3165``) and the §4.13.1 put-call parities.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.engines.exotic._rootfind import critical_value_bsm_combo
from backend.utils.math import cbnd, norm_cdf


@njit(fastmath=True, cache=True)
def compound_option_price(
    S: float,
    K1: float,
    K2: float,
    t1: float,
    T2: float,
    r: float,
    q: float,
    sigma: float,
    is_call_on: bool,
    is_call_under: bool,
) -> float:
    """
    Price a compound option (Geske 1979).

    Parameters
    ----------
    S : float
        Spot price.
    K1 : float
        Strike of the underlying option.
    K2 : float
        Strike of the compound option (paid/received at ``t1``).
    t1 : float
        Compound expiry (``0 < t1 < T2``).
    T2 : float
        Underlying-option expiry.
    r, q, sigma : float
        Rate, dividend yield, volatility. Cost of carry ``b = r - q``.
    is_call_on : bool
        The compound option is a call-on (True) or put-on (False).
    is_call_under : bool
        The underlying option is a call (True) or put (False).

    Returns
    -------
    float
        Option price.
    """
    if sigma <= 0.0 or t1 <= 0.0 or T2 <= t1:
        return 0.0

    b = r - q
    half_vsq = 0.5 * sigma * sigma

    # Critical asset price I at t1: underlying BSM(I, K1, T2 - t1) == K2, monotone
    # in I (increasing for a call underlying, decreasing for a put). A call
    # underlying always has a root in [1e-8, big]. A put underlying has one only
    # when K2 < K1*exp(-r*(T2-t1)) (else the put is worth less than K2 for every
    # spot -> the compound is always/never exercised); there the bisection's
    # no-straddle fallback returns lo, which drives the formula to the correct
    # always-exercise limit (verified: pp -> K2*e^{-r t1} - p_BSM, cp -> 0).
    hi = max(S, K1, K2) * 1000.0
    crit = critical_value_bsm_combo(
        1.0,
        K1,
        T2 - t1,
        is_call_under,
        0.0,
        K1,
        T2 - t1,
        False,
        K2,
        r,
        q,
        sigma,
        1e-8,
        hi,
    )

    v_t1 = sigma * math.sqrt(t1)
    v_T2 = sigma * math.sqrt(T2)
    y1 = (math.log(S / crit) + (b + half_vsq) * t1) / v_t1
    y2 = y1 - v_t1
    z1 = (math.log(S / K1) + (b + half_vsq) * T2) / v_T2
    z2 = z1 - v_T2
    rho = math.sqrt(t1 / T2)

    carry = math.exp((b - r) * T2)
    disc_T2 = math.exp(-r * T2)
    disc_t1 = math.exp(-r * t1)

    if is_call_under:
        if is_call_on:
            # Call on call (eq 4.28).
            return (
                S * carry * cbnd(z1, y1, rho)
                - K1 * disc_T2 * cbnd(z2, y2, rho)
                - K2 * disc_t1 * norm_cdf(y2)
            )
        # Put on call (eq 4.29).
        return (
            K1 * disc_T2 * cbnd(z2, -y2, -rho)
            - S * carry * cbnd(z1, -y1, -rho)
            + K2 * disc_t1 * norm_cdf(-y2)
        )
    if is_call_on:
        # Call on put (eq 4.30).
        return (
            K1 * disc_T2 * cbnd(-z2, -y2, rho)
            - S * carry * cbnd(-z1, -y1, rho)
            - K2 * disc_t1 * norm_cdf(-y2)
        )
    # Put on put (eq 4.31).
    return (
        S * carry * cbnd(-z1, y1, -rho)
        - K1 * disc_T2 * cbnd(-z2, y2, -rho)
        + K2 * disc_t1 * norm_cdf(y2)
    )
