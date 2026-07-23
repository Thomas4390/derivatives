"""
Safeguarded-Newton (rtsafe) implied-volatility inversion.
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.constants.numerical import (
    IV_SIGMA_MIN,
    IV_VEGA_FLOOR,
    VOLATILITY_MAX,
)
from backend.utils.math.black_scholes import bs_price, d1_d2
from backend.utils.math.distributions import norm_pdf


@njit(fastmath=True, cache=True)
def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    is_call: bool,
    dividend_yield: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.

    Parameters
    ----------
    price : float
        Market price of the option
    spot : float
        Current spot price
    strike : float
        Strike price
    time_to_expiry : float
        Time to expiration in years
    rate : float
        Risk-free rate
    is_call : bool
        True for call, False for put
    dividend_yield : float
        Continuous dividend yield
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations

    Returns
    -------
    float
        Implied volatility. Returns ``nan`` only when there is no real IV —
        i.e. ``time_to_expiry <= 0`` or the price violates the no-arbitrage
        bounds. Any in-bounds price always inverts to a finite volatility:
        the solver is a safeguarded Newton (rtsafe) bracketed on
        ``[IV_SIGMA_MIN, VOLATILITY_MAX]``, which converges for deep
        ITM/OTM inputs where a bare Newton step overshoots and oscillates.
    """
    if time_to_expiry <= 0:
        return math.nan

    # No-arbitrage bounds: a European price outside [intrinsic, forward
    # bound] has no real implied volatility. Reject early instead of letting
    # Newton flail and return a misleading (previously negative) sentinel.
    disc_spot = spot * math.exp(-dividend_yield * time_to_expiry)
    disc_strike = strike * math.exp(-rate * time_to_expiry)
    if is_call:
        lower_bound = disc_spot - disc_strike
        upper_bound = disc_spot
    else:
        lower_bound = disc_strike - disc_spot
        upper_bound = disc_strike
    if lower_bound < 0.0:
        lower_bound = 0.0
    if price <= 0.0 or price < lower_bound - tol or price > upper_bound + tol:
        return math.nan

    # Safeguarded Newton (rtsafe). ``f(sigma) = bs_price(sigma) - price`` is
    # increasing in sigma, and the no-arbitrage check above guarantees a root in
    # ``[IV_SIGMA_MIN, VOLATILITY_MAX]``. Take the Newton step only when it lands
    # strictly inside the current bracket (and vega is usable); otherwise bisect.
    # This converges for deep ITM/OTM inputs where a bare Newton step has a tiny
    # vega, overshoots to the clamp and oscillates — the regime that previously
    # returned NaN and tore holes in the model-implied IV surfaces.
    lo = IV_SIGMA_MIN
    hi = VOLATILITY_MAX
    # Brenner-Subrahmanyam ATM seed, kept inside the bracket.
    sigma = math.sqrt(2 * math.pi / time_to_expiry) * price / spot
    if sigma <= lo or sigma >= hi:
        sigma = 0.5 * (lo + hi)

    for _ in range(max_iter):
        bs_p = bs_price(
            spot, strike, time_to_expiry, rate, sigma, is_call, dividend_yield
        )
        diff = bs_p - price

        if abs(diff) < tol:
            return sigma

        # Narrow the bracket from the sign of the residual (f is increasing).
        if diff < 0.0:
            lo = sigma
        else:
            hi = sigma

        # Newton step, accepted only if it stays strictly inside the bracket.
        d1, _ = d1_d2(spot, strike, time_to_expiry, rate, sigma, dividend_yield)
        vega_raw = (
            spot
            * math.exp(-dividend_yield * time_to_expiry)
            * norm_pdf(d1)
            * math.sqrt(time_to_expiry)
        )
        if vega_raw > IV_VEGA_FLOOR:
            candidate = sigma - diff / vega_raw
        else:
            candidate = lo - 1.0  # unusable vega -> force a bisection step
        if candidate <= lo or candidate >= hi:
            sigma = 0.5 * (lo + hi)
        else:
            sigma = candidate

    # Bracketed root, iterations exhausted: the midpoint is the best estimate
    # (the bracket has narrowed far below any practical tolerance by now).
    return 0.5 * (lo + hi)
