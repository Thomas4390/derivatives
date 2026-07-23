"""
Discrete-barrier option pricing kernel (Broadie-Glasserman-Kou 1997).

Standard barrier formulas assume continuous monitoring. In practice barriers
are monitored at discrete dates (e.g. daily close). Discrete monitoring lowers
the knock-out probability, so a knock-out is worth more. Broadie, Glasserman
and Kou (1995/1997) show the discretely-monitored option is accurately priced
by the *continuous* formula with the barrier shifted away from the underlying
by ``exp(+/- beta * sigma * sqrt(dt))`` (Haug section 4.17.6).

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.constants.exotic import BGK_BETA
from backend.engines.exotic.barrier import barrier_option_price


@njit(fastmath=True, cache=True)
def discrete_barrier_price(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
    is_knock_in: bool,
    is_up: bool,
    m: int,
    rebate: float = 0.0,
) -> float:
    """
    Price a discretely-monitored single barrier (BGK continuity correction).

    Parameters
    ----------
    S, K, H : float
        Spot, strike, barrier.
    T, r, q, sigma : float
        Maturity, rate, dividend yield, volatility.
    is_call, is_knock_in, is_up : bool
        Option flags (as in ``barrier_option_price``).
    m : int
        Number of equally-spaced monitoring dates (``dt = T / m``). As
        ``m -> inf`` the shift vanishes and this returns the continuous price.
    rebate : float
        Rebate (knock-out only).

    Returns
    -------
    float
        Option price.
    """
    if m <= 0 or T <= 0.0 or sigma <= 0.0:
        # Degenerate: fall back to the continuous formula (no correction).
        return barrier_option_price(
            S, K, H, T, r, q, sigma, is_call, is_knock_in, is_up, rebate
        )

    dt = T / m
    shift = BGK_BETA * sigma * math.sqrt(dt)
    # Shift the barrier OUTWARD (away from spot): up-barrier up, down-barrier down.
    h_adj = H * math.exp(shift) if is_up else H * math.exp(-shift)

    return barrier_option_price(
        S, K, h_adj, T, r, q, sigma, is_call, is_knock_in, is_up, rebate
    )
