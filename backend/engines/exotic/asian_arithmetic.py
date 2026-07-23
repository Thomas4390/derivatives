"""
Arithmetic average-rate option kernel (Turnbull-Wakeman 1991, Haug 4.20.2).

The arithmetic average of a lognormal asset is not itself lognormal, so there is
no exact closed form. Turnbull and Wakeman (1991) match the exact first two
moments ``M1, M2`` of the arithmetic average to a lognormal, giving an adjusted
cost-of-carry ``bA = ln(M1)/T`` and volatility ``vA = sqrt(ln(M2)/T - 2 bA)``
that are then fed into the generalized BSM formula (Haug 4.97/4.98).

Seasoning: ``T2`` is the (fixed) length of the averaging window and ``T`` the
remaining time to maturity. If the option is already *into* the average period
(``tau = T2 - T > 0``), the strike is replaced by ``Xhat = (T2/T) X - (tau/T) SA``
and the value scaled by ``T/T2`` (``SA`` = the realized average so far). If in
that case ``Xhat < 0`` a call is certain to be exercised (worth the discounted
expected average minus strike) and the put is worthless.

Ported VERBATIM from Haug's published VBA ``TurnbullWakemanAsian``. Validated
against the book put example (5.6093) and Table 4-25 (calls, incl. seasoned
``T < T2`` cells). Univariate -- no bivariate normal, no root-finding. (The
legacy geometric Asian in ``asian.py`` is a different product, untouched.)

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.engines.exotic.barrier import _bs_vanilla_price


@njit(fastmath=True, cache=True)
def asian_arithmetic_tw_price(
    S: float,
    SA: float,
    X: float,
    T: float,
    T2: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
) -> float:
    """
    Price an arithmetic average-rate option (Turnbull-Wakeman, Haug 4.97/4.98).

    Parameters
    ----------
    S : float
        Spot price (> 0).
    SA : float
        Realized arithmetic average so far (only used when seasoned, ``T2 > T``).
    X : float
        Strike (> 0).
    T : float
        Remaining time to maturity in years (> 0).
    T2 : float
        Length of the averaging window in years (> 0; constant over the option's
        life). ``T2 > T`` means the option is into the average period.
    r, q, sigma : float
        Rate, dividend yield, volatility. Cost of carry ``b = r - q``.
    is_call : bool
        Call or put.

    Returns
    -------
    float
        Option price.
    """
    b = r - q
    v = sigma
    t1 = max(0.0, T - T2)
    tau = T2 - T
    dtt = T - t1  # length of the remaining-future averaging span
    # Route a near-zero carry to the stable closed form: the verbatim 1/b and
    # 1/(2b+v^2) terms suffer catastrophic cancellation as b -> 0 (they can even
    # produce a negative vA^2). |b| < 1e-7 is well below any realistic carry and
    # the b->0 limit is M1=1, M2 = the v^2-only expression below.
    b_zero = abs(b) < 1e-7

    # First moment of the arithmetic average.
    if b_zero:
        m1 = 1.0
    else:
        m1 = (math.exp(b * T) - math.exp(b * t1)) / (b * dtt)

    # Into the average period: a call may be certain to exercise, a put worthless.
    # ``<= 0`` (not ``< 0``) so the Xhat == 0 knife-edge takes the closed-form
    # exercise-certain value instead of calling BSM with a zero strike.
    if tau > 0.0:
        if (T2 / T * X - tau / T * SA) <= 0.0:
            if is_call:
                ev = SA * (T2 - T) / T2 + S * m1 * T / T2
                return max(0.0, ev - X) * math.exp(-r * T)
            return 0.0

    # Second moment of the arithmetic average.
    v2 = v * v
    if b_zero:
        m2 = 2.0 * math.exp(v2 * T) / (v2 * v2 * dtt * dtt) - 2.0 * math.exp(
            v2 * t1
        ) * (1.0 + v2 * dtt) / (v2 * v2 * dtt * dtt)
    else:
        m2 = 2.0 * math.exp((2.0 * b + v2) * T) / (
            (b + v2) * (2.0 * b + v2) * dtt * dtt
        ) + 2.0 * math.exp((2.0 * b + v2) * t1) / (b * dtt * dtt) * (
            1.0 / (2.0 * b + v2) - math.exp(b * dtt) / (b + v2)
        )

    bA = math.log(m1) / T
    # Floor the variance argument so a sub-lognormal rounding artifact yields
    # vA = 0 (a finite, degenerate price) rather than a silent NaN.
    vA = math.sqrt(max(0.0, math.log(m2) / T - 2.0 * bA))

    # Generalized BSM with the adjusted carry bA (q_adj = r - bA) and vol vA.
    # Float-typed locals keep mypy's no-any-return at zero (the njit
    # _bs_vanilla_price is untyped and would otherwise widen the result to Any).
    if tau > 0.0:
        x_hat = T2 / T * X - tau / T * SA
        seasoned: float = (
            _bs_vanilla_price(S, x_hat, T, r, r - bA, vA, is_call) * T / T2
        )
        return seasoned
    fresh: float = _bs_vanilla_price(S, X, T, r, r - bA, vA, is_call)
    return fresh
