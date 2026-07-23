"""
Forward-start option kernel (Rubinstein 1990, Haug 4.6).

A forward-start option is paid for now but only "starts" at a known future grant
date ``t1``: at that point the strike is set to ``alpha * S(t1)`` (a fixed
moneyness of the then-current spot) and the option then runs to expiry ``T``.
``alpha < 1`` starts the call in-the-money, ``alpha = 1`` at-the-money, ``alpha >
1`` out-of-the-money.

By homogeneity the time-``t1`` value is ``S(t1)`` times the value of a vanilla on
a unit spot struck at ``alpha`` with ``T - t1`` to run; discounting that expected
value to today gives Rubinstein's closed form (Haug 4.17/4.18):

    c = S e^{(b-r)t1} [ e^{(b-r)(T-t1)} N(d1) - alpha e^{-r(T-t1)} N(d2) ]

Ported VERBATIM from Haug's published VBA ``ForwardStartOption`` (``S * Exp((b-r)
* t1) * GBlackScholes(flag, 1, alpha, T - t1, r, b, v)``). Validated against
Haug's worked example (call 4.4064).

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.engines.exotic.barrier import _bs_vanilla_price


@njit(fastmath=True, cache=True)
def forward_start_price(
    S: float,
    alpha: float,
    t1: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
) -> float:
    """
    Price a forward-start option (Rubinstein 1990).

    Parameters
    ----------
    S : float
        Spot price.
    alpha : float
        Moneyness of the forward strike (strike is set to ``alpha * S(t1)`` at
        the grant date).
    t1 : float
        Grant date (``0 <= t1 < T``); the option starts here.
    T : float
        Time to expiration.
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
    # Value at the grant date = S(t1) * vanilla(spot=1, strike=alpha, T-t1);
    # discounting E[S(t1)] = S e^{b t1} back at r gives the S e^{(b-r)t1} factor.
    return (
        S
        * math.exp((b - r) * t1)
        * _bs_vanilla_price(1.0, alpha, T - t1, r, q, sigma, is_call)
    )
