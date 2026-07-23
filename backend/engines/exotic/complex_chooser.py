"""
Complex chooser option kernel (Rubinstein 1991, Haug 4.12.2).

A *complex* chooser lets the holder decide at the choice date ``t_choice`` whether
the option becomes a call (strike ``Kc``, maturity ``Tc``) or a put (strike
``Kp``, maturity ``Tp``) -- the call and put may differ in BOTH strike and
maturity (the *simple* chooser shares one strike and one maturity). The payoff at
``t_choice`` is ``max(c_BSM(S, Kc, Tc), p_BSM(S, Kp, Tp))``.

Ported VERBATIM from Haug's published VBA ``ComplexChooser`` / ``CriticalValueChooser``
(book section 4.12.2, eq 4.27). The critical asset price ``I`` at ``t_choice``
where the call and put legs are equal is found by the shared safeguarded
bisection in :mod:`backend.engines.exotic._rootfind` (the book uses Newton-
Raphson; bisection on the monotone ``c - p`` reaches the same root and keeps the
kernel cache-able). Validated against Haug's worked example
``ComplexChooser(50, 55, 48, 0.25, 0.5, 0.5833, 0.1, 0.05, 0.35) = 6.0508``.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.engines.exotic._rootfind import critical_value_bsm_combo
from backend.utils.math import cbnd


@njit(fastmath=True, cache=True)
def complex_chooser_price(
    S: float,
    Kc: float,
    Kp: float,
    Tc: float,
    Tp: float,
    t_choice: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """
    Price a complex chooser option (Rubinstein 1991).

    Parameters
    ----------
    S : float
        Spot price.
    Kc, Kp : float
        Call strike and put strike chosen between at ``t_choice``.
    Tc, Tp : float
        Time to maturity of the call leg and the put leg (``> t_choice``).
    t_choice : float
        Choice date (``0 < t_choice < min(Tc, Tp)``).
    r, q, sigma : float
        Rate, dividend yield, volatility. Cost of carry ``b = r - q``.

    Returns
    -------
    float
        Option price.
    """
    if sigma <= 0.0 or t_choice <= 0.0 or Tc <= t_choice or Tp <= t_choice:
        return 0.0

    b = r - q
    half_vsq = 0.5 * sigma * sigma

    # Critical asset price I at t_choice: c_BSM(I, Kc, Tc - t) = p_BSM(I, Kp, Tp - t).
    # c - p is monotone increasing in I, so [1e-8, big] straddles the single root.
    hi = max(S, Kc, Kp) * 1000.0
    crit = critical_value_bsm_combo(
        1.0,
        Kc,
        Tc - t_choice,
        True,
        -1.0,
        Kp,
        Tp - t_choice,
        False,
        0.0,
        r,
        q,
        sigma,
        1e-8,
        hi,
    )

    v_t = sigma * math.sqrt(t_choice)
    v_Tc = sigma * math.sqrt(Tc)
    v_Tp = sigma * math.sqrt(Tp)

    d1 = (math.log(S / crit) + (b + half_vsq) * t_choice) / v_t
    d2 = d1 - v_t
    y1 = (math.log(S / Kc) + (b + half_vsq) * Tc) / v_Tc
    y2 = (math.log(S / Kp) + (b + half_vsq) * Tp) / v_Tp
    rho1 = math.sqrt(t_choice / Tc)
    rho2 = math.sqrt(t_choice / Tp)

    return (
        S * math.exp((b - r) * Tc) * cbnd(d1, y1, rho1)
        - Kc * math.exp(-r * Tc) * cbnd(d2, y1 - v_Tc, rho1)
        - S * math.exp((b - r) * Tp) * cbnd(-d1, -y2, rho2)
        + Kp * math.exp(-r * Tp) * cbnd(-d2, -y2 + v_Tp, rho2)
    )
