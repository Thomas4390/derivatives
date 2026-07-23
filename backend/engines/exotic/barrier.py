"""
Barrier option pricing kernel (Reiner-Rubinstein 1991).

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.math import bs_price as _bs_price_canonical, norm_cdf


@njit(fastmath=True, cache=True)
def _bs_vanilla_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool
) -> float:
    """Black-Scholes vanilla price with dividend yield.  Delegates to canonical ``bs_price``."""
    return _bs_price_canonical(S, K, T, r, sigma, is_call, q)


@njit(fastmath=True, cache=True)
def barrier_option_price(
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
    rebate: float = 0.0,
) -> float:
    """
    Barrier option pricing using Reiner-Rubinstein (1991) formulas.

    Supports all 8 barrier types (up/down, in/out, call/put).

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    H : float
        Barrier level
    T : float
        Time to expiry
    r : float
        Risk-free rate
    q : float
        Continuous dividend yield
    sigma : float
        Volatility
    is_call : bool
        True for call, False for put
    is_knock_in : bool
        True for knock-in, False for knock-out
    is_up : bool
        True for up-barrier, False for down-barrier
    rebate : float
        Rebate paid at knockout (default 0).
        Note: rebate only applies to knock-out options. For knock-in options,
        the rebate parameter is ignored (per Reiner-Rubinstein convention).

    Returns
    -------
    float
        Option price
    """
    # Rebate conventions:
    # - T=0: rebate is returned undiscounted (df=exp(0)=1, immediate payment)
    # - sigma=0: rebate is discounted by exp(-rT) (deferred payment at T)
    # - S past barrier: rebate is returned undiscounted (immediate payment convention
    #   consistent with front-end F-term formulation in Reiner-Rubinstein)
    if T <= 0:
        payoff = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        if is_up:
            breached = S >= H
        else:
            breached = S <= H
        if is_knock_in:
            return payoff if breached else 0.0
        return payoff if not breached else rebate

    if sigma <= 0:
        # Deterministic forward: S * exp((r-q)*T)
        b = r - q
        F = S * math.exp(b * T)
        df = math.exp(-r * T)
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)

        # Determine if deterministic path breaches barrier
        if is_up:
            # Path max: S*exp(b*T) if b>0, else S
            breached = (F >= H) if b > 0 else False
        else:
            # Path min: S*exp(b*T) if b<0, else S
            breached = (F <= H) if b < 0 else False

        if is_knock_in:
            return (intrinsic * df) if breached else 0.0
        return (rebate * df) if breached else (intrinsic * df)

    # Check barrier breach (spot already past barrier)
    if is_up and S >= H:
        return (
            rebate
            if not is_knock_in
            else _bs_vanilla_price(S, K, T, r, q, sigma, is_call)
        )
    if not is_up and S <= H:
        return (
            rebate
            if not is_knock_in
            else _bs_vanilla_price(S, K, T, r, q, sigma, is_call)
        )

    sqrt_T = math.sqrt(T)

    # Cost of carry
    b = r - q
    qd = math.exp(-q * T)
    df = math.exp(-r * T)

    # Helper parameters (mu uses cost-of-carry b, lambda uses discount rate r)
    mu = (b - 0.5 * sigma * sigma) / (sigma * sigma)
    lambda_val = math.sqrt(mu * mu + 2.0 * r / (sigma * sigma))

    # d-parameters
    x1 = math.log(S / K) / (sigma * sqrt_T) + (1.0 + mu) * sigma * sqrt_T
    x2 = math.log(S / H) / (sigma * sqrt_T) + (1.0 + mu) * sigma * sqrt_T
    y1 = math.log(H * H / (S * K)) / (sigma * sqrt_T) + (1.0 + mu) * sigma * sqrt_T
    y2 = math.log(H / S) / (sigma * sqrt_T) + (1.0 + mu) * sigma * sqrt_T
    z = math.log(H / S) / (sigma * sqrt_T) + lambda_val * sigma * sqrt_T

    # A, B, C, D terms from Reiner-Rubinstein (with dividend discount on S terms)
    if is_call:
        A = S * qd * norm_cdf(x1) - K * df * norm_cdf(x1 - sigma * sqrt_T)
        B = S * qd * norm_cdf(x2) - K * df * norm_cdf(x2 - sigma * sqrt_T)

        # Haug/Reiner-Rubinstein convention: eta=+1 for down, -1 for up
        if is_up:
            eta = -1.0
        else:
            eta = 1.0

        C = S * qd * math.pow(H / S, 2.0 * (mu + 1.0)) * norm_cdf(
            eta * y1
        ) - K * df * math.pow(H / S, 2.0 * mu) * norm_cdf(eta * (y1 - sigma * sqrt_T))
        D = S * qd * math.pow(H / S, 2.0 * (mu + 1.0)) * norm_cdf(
            eta * y2
        ) - K * df * math.pow(H / S, 2.0 * mu) * norm_cdf(eta * (y2 - sigma * sqrt_T))
    else:
        # Put formulas: phi = -1 applied to outer S/K terms
        A = K * df * norm_cdf(-x1 + sigma * sqrt_T) - S * qd * norm_cdf(-x1)
        B = K * df * norm_cdf(-x2 + sigma * sqrt_T) - S * qd * norm_cdf(-x2)

        if is_up:
            eta = -1.0
        else:
            eta = 1.0

        C = K * df * math.pow(H / S, 2.0 * mu) * norm_cdf(
            eta * (y1 - sigma * sqrt_T)
        ) - S * qd * math.pow(H / S, 2.0 * (mu + 1.0)) * norm_cdf(eta * y1)
        D = K * df * math.pow(H / S, 2.0 * mu) * norm_cdf(
            eta * (y2 - sigma * sqrt_T)
        ) - S * qd * math.pow(H / S, 2.0 * (mu + 1.0)) * norm_cdf(eta * y2)

    # Rebate terms
    E = (
        rebate
        * df
        * (
            norm_cdf(eta * (x2 - sigma * sqrt_T))
            - math.pow(H / S, 2.0 * mu) * norm_cdf(eta * (y2 - sigma * sqrt_T))
        )
    )
    F = rebate * (
        math.pow(H / S, mu + lambda_val) * norm_cdf(eta * z)
        + math.pow(H / S, mu - lambda_val)
        * norm_cdf(eta * (z - 2.0 * lambda_val * sigma * sqrt_T))
    )

    # Vanilla price for parity
    vanilla = A

    # Determine core knock-out price (WITHOUT rebate) per Reiner-Rubinstein table
    if is_up:
        if is_call:
            if K >= H:
                core_out = 0.0  # Worthless (barrier below strike)
            else:
                core_out = A - B + C - D
        else:  # put
            if K >= H:
                core_out = B - D
            else:
                core_out = A - C
    else:  # down
        if is_call:
            if K >= H:
                core_out = A - C
            else:
                core_out = B - D
        else:  # put
            if K <= H:
                core_out = 0.0  # Worthless (barrier above strike)
            else:
                core_out = A - B + C - D

    # Knock-out gets F (immediate rebate), knock-in gets E (deferred rebate)
    # Parity: KI + KO = vanilla + E + F
    if is_knock_in:
        price = vanilla - core_out + E
    else:
        price = core_out + F

    return max(price, 0.0)
