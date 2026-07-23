"""
Chooser option pricing kernel (Rubinstein 1991).

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.engines.exotic.barrier import _bs_vanilla_price


@njit(fastmath=True, cache=True)
def chooser_price(
    S: float, K: float, T: float, t_c: float, r: float, q: float, sigma: float
) -> float:
    """
    Simple chooser option price (Rubinstein 1991).

    V = BS_call(S, K, T) + exp(-q*(T-t_c)) * BS_put(S, K*exp(-(r-q)*(T-t_c)), t_c)

    The ``exp(-q*(T-t_c))`` factor on the put leg follows from put-call parity at
    the choice date (``max(c, p) = c + exp(-q*tau) * max(0, K*exp(-b*tau) - S_tc)``
    with ``b = r - q``, ``tau = T - t_c``). It equals 1 only when ``q = 0``;
    omitting it overprices the chooser whenever there is a dividend yield.

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to expiry (final maturity)
    t_c : float
        Choice time
    r : float
        Risk-free rate
    q : float
        Continuous dividend yield
    sigma : float
        Volatility

    Returns
    -------
    float
        Option price
    """
    if T <= 0:
        # At expiry, chooser = max(call, put) = max(max(S-K,0), max(K-S,0))
        call_payoff = max(S - K, 0.0)
        put_payoff = max(K - S, 0.0)
        return max(call_payoff, put_payoff)

    if sigma <= 0:
        F = S * math.exp((r - q) * T)
        df = math.exp(-r * T)
        call_payoff = max(F - K, 0.0)
        put_payoff = max(K - F, 0.0)
        return max(call_payoff, put_payoff) * df

    # Rubinstein decomposition: a call to maturity plus a put (struck at the
    # carry-adjusted strike, expiring at the choice date) discounted for the
    # dividend yield accrued over the remaining life of the chosen leg.
    call_part = _bs_vanilla_price(S, K, T, r, q, sigma, True)
    K_adj = K * math.exp(-(r - q) * (T - t_c))
    put_part = math.exp(-q * (T - t_c)) * _bs_vanilla_price(
        S, K_adj, t_c, r, q, sigma, False
    )

    return call_part + put_part
