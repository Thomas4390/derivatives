"""
Discrete time-switch option kernel (Pechtl 1995, Haug 4.11).

A discrete time-switch call pays a fixed amount ``A * dt`` at maturity for every
monitoring instant ``i * dt`` (``i = 1..n``, ``n = T / dt``) at which the asset
price has exceeded the strike ``X``; the put accrues for every instant the asset
is below ``X``. The value is a discounted sum of digital (cash-or-nothing)
probabilities (Haug 4.24/4.25), with ``Z = +1`` for a call and ``-1`` for a put:

    price = A e^{-rT} sum_{i=1}^{n} N( Z d_i ) dt + dt A e^{-rT} m
    d_i   = [ ln(S/X) + (b - sigma^2/2) i dt ] / (sigma sqrt(i dt))

where ``b = r - q`` is the cost of carry and ``m`` is the number of time units
whose condition has *already* been fulfilled (a seasoned option; ``m = 0`` for a
fresh one). Ported VERBATIM from Haug's published VBA ``TimeSwitchOption`` and
validated against the book's worked example (call 1.3750).

Application (Haug): accrual swaps in rates markets price as a sum of discrete
time-switch options. Univariate -- ``norm_cdf`` only, no bivariate normal, no
root-finding.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.math import norm_cdf


@njit(fastmath=True, cache=True)
def time_switch_price(
    S: float,
    X: float,
    A: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    dt: float,
    m: int,
    is_call: bool,
) -> float:
    """
    Price a discrete time-switch option (Pechtl 1995, Haug 4.24/4.25).

    Parameters
    ----------
    S : float
        Spot price (> 0).
    X : float
        Strike (> 0); accrual is conditioned on ``S_t`` vs ``X``.
    A : float
        Accumulated amount per time unit (the option pays ``A * dt`` per
        in-condition monitoring instant).
    T : float
        Time to maturity in years (> 0).
    r, q, sigma : float
        Rate, dividend yield, volatility. Cost of carry ``b = r - q``.
    dt : float
        Monitoring time step ``Delta t`` (e.g. ``1/365`` for daily).
    m : int
        Number of time units already fulfilling the condition (seasoned
        option); ``0`` for a fresh option.
    is_call : bool
        True accrues while ``S > X`` (call), False while ``S < X`` (put).

    Returns
    -------
    float
        Option price.
    """
    n = int(round(T / dt))
    z = 1.0 if is_call else -1.0
    half_var = 0.5 * sigma * sigma
    b = r - q
    log_sx = math.log(S / X)
    total = 0.0
    for i in range(1, n + 1):
        idt = i * dt
        d = (log_sx + (b - half_var) * idt) / (sigma * math.sqrt(idt))
        total += norm_cdf(z * d) * dt
    disc = math.exp(-r * T)
    return A * disc * total + dt * A * disc * m
