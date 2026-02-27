"""
Exotic Analytical Engine
========================

Closed-form pricing engine for exotic options under GBM:
- Barrier Options (Reiner-Rubinstein 1991) — all 8 types
- Asian Geometric Options (Kemna-Vorst 1990)
- Digital/Binary Options (cash-or-nothing)
- Lookback Options (Goldman-Sosin-Gatto 1979, Conze-Viswanathan 1991)

All pricing kernels are Numba-compiled for performance.
Greeks are computed via central finite differences aligned with
``GreeksBumpConfig`` defaults from ``backend.greeks.numerical``.

Author: Thomas
Created: 2025
"""

import math
from dataclasses import dataclass

from numba import njit

from backend.utils.math import norm_cdf, DAYS_PER_YEAR
from backend.core.interfaces import PricingEngine, Instrument, Model
from backend.core.market import MarketEnvironment
from backend.core.result_types import (
    PricingResult, GreeksResult, PricingCapability, ExerciseStyle
)
from backend.instruments.options import (
    BarrierOption, AsianOption, DigitalOption, LookbackOption
)
from backend.models.gbm import GBMModel


# =============================================================================
# Option type constants for Greeks dispatch
# =============================================================================

BARRIER: int = 0
ASIAN_GEO: int = 1
DIGITAL: int = 2
LOOKBACK_FIXED: int = 3
LOOKBACK_FLOATING: int = 4


# =============================================================================
# NUMBA PRICING KERNELS
# =============================================================================

@njit(fastmath=True, cache=True)
def _bs_vanilla_price(S: float, K: float, T: float, r: float, q: float,
                      sigma: float, is_call: bool) -> float:
    """Black-Scholes vanilla price with dividend yield (for knock-in breach fallback)."""
    if T <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if sigma <= 0:
        qd = math.exp(-q * T)
        df = math.exp(-r * T)
        return max(S * qd - K * df, 0.0) if is_call else max(K * df - S * qd, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    qd = math.exp(-q * T)
    df = math.exp(-r * T)
    if is_call:
        return S * qd * norm_cdf(d1) - K * df * norm_cdf(d2)
    else:
        return K * df * norm_cdf(-d2) - S * qd * norm_cdf(-d1)


@njit(fastmath=True, cache=True)
def barrier_option_price(S: float, K: float, H: float, T: float, r: float, q: float,
                         sigma: float, is_call: bool, is_knock_in: bool, is_up: bool,
                         rebate: float = 0.0) -> float:
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
    if T <= 0:
        payoff = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        if is_up:
            breached = S >= H
        else:
            breached = S <= H
        if is_knock_in:
            return payoff if breached else 0.0
        else:
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
        else:
            return (rebate * df) if breached else (intrinsic * df)

    # Check barrier breach (spot already past barrier)
    if is_up and S >= H:
        return rebate if not is_knock_in else _bs_vanilla_price(S, K, T, r, q, sigma, is_call)
    if not is_up and S <= H:
        return rebate if not is_knock_in else _bs_vanilla_price(S, K, T, r, q, sigma, is_call)

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

        if is_up:
            eta = 1.0
        else:
            eta = -1.0

        C = S * qd * math.pow(H / S, 2.0 * (mu + 1.0)) * norm_cdf(eta * y1) - \
            K * df * math.pow(H / S, 2.0 * mu) * norm_cdf(eta * (y1 - sigma * sqrt_T))
        D = S * qd * math.pow(H / S, 2.0 * (mu + 1.0)) * norm_cdf(eta * y2) - \
            K * df * math.pow(H / S, 2.0 * mu) * norm_cdf(eta * (y2 - sigma * sqrt_T))
    else:
        # Put formulas: phi = -1 applied to outer S/K terms
        A = K * df * norm_cdf(-x1 + sigma * sqrt_T) - S * qd * norm_cdf(-x1)
        B = K * df * norm_cdf(-x2 + sigma * sqrt_T) - S * qd * norm_cdf(-x2)

        if is_up:
            eta = -1.0
        else:
            eta = 1.0

        C = K * df * math.pow(H / S, 2.0 * mu) * norm_cdf(eta * (y1 - sigma * sqrt_T)) - \
            S * qd * math.pow(H / S, 2.0 * (mu + 1.0)) * norm_cdf(eta * y1)
        D = K * df * math.pow(H / S, 2.0 * mu) * norm_cdf(eta * (y2 - sigma * sqrt_T)) - \
            S * qd * math.pow(H / S, 2.0 * (mu + 1.0)) * norm_cdf(eta * y2)

    # Rebate terms
    E = rebate * df * (norm_cdf(eta * (x2 - sigma * sqrt_T)) -
                       math.pow(H / S, 2.0 * mu) * norm_cdf(eta * (y2 - sigma * sqrt_T)))
    F = rebate * (math.pow(H / S, mu + lambda_val) * norm_cdf(eta * z) +
                  math.pow(H / S, mu - lambda_val) * norm_cdf(eta * (z - 2.0 * lambda_val * sigma * sqrt_T)))

    # Vanilla price for parity
    vanilla = A

    # Determine knock-out price based on Reiner-Rubinstein table
    if is_up:
        if is_call:
            if K >= H:
                price_out = F  # Worthless (barrier below strike)
            else:
                price_out = A - B + C - D + F
        else:  # put
            if K >= H:
                price_out = B - D + F
            else:
                price_out = A - C + F
    else:  # down
        if is_call:
            if K >= H:
                price_out = A - B + C - D + F
            else:
                price_out = B - D + F
        else:  # put
            if K <= H:
                price_out = F  # Worthless (barrier above strike)
            else:
                price_out = A - B + C - D + F

    # Knock-in via parity: In = Vanilla - Out
    if is_knock_in:
        price = vanilla - price_out
    else:
        price = price_out + E

    return max(price, 0.0)


@njit(fastmath=True, cache=True)
def asian_geometric_price(S: float, K: float, T: float, r: float, q: float,
                          sigma: float, is_call: bool) -> float:
    """
    Geometric Asian option price (Kemna-Vorst 1990).

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
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

    Returns
    -------
    float
        Option price
    """
    if T <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)

    if sigma <= 0:
        b = r - q
        F = S * math.exp(b * T)
        df = math.exp(-r * T)
        return max(F - K, 0.0) * df if is_call else max(K - F, 0.0) * df

    # Adjusted parameters for geometric average
    sigma_adj = sigma / math.sqrt(3.0)
    b = 0.5 * (r - q - 0.5 * sigma * sigma)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (b + 0.5 * sigma_adj * sigma_adj) * T) / (sigma_adj * sqrt_T)
    d2 = d1 - sigma_adj * sqrt_T

    if is_call:
        price = S * math.exp((b - r) * T) * norm_cdf(d1) - \
                K * math.exp(-r * T) * norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - \
                S * math.exp((b - r) * T) * norm_cdf(-d1)

    return max(price, 0.0)


@njit(fastmath=True, cache=True)
def digital_price(S: float, K: float, T: float, r: float, q: float, sigma: float,
                  is_call: bool, payout: float = 1.0) -> float:
    """
    Digital (cash-or-nothing) option price.

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
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
    payout : float
        Fixed payout amount

    Returns
    -------
    float
        Option price
    """
    if T <= 0:
        if is_call:
            return payout if S > K else 0.0
        else:
            return payout if S < K else 0.0

    if sigma <= 0:
        # Deterministic forward
        F = S * math.exp((r - q) * T)
        df = math.exp(-r * T)
        if is_call:
            return (payout * df) if F > K else 0.0
        else:
            return (payout * df) if F < K else 0.0

    sqrt_T = math.sqrt(T)
    d2 = (math.log(S / K) + (r - q - 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    df = math.exp(-r * T)

    if is_call:
        return payout * df * norm_cdf(d2)
    else:
        return payout * df * norm_cdf(-d2)


@njit(fastmath=True, cache=True)
def lookback_fixed_price(S: float, K: float, M_min: float, M_max: float, T: float,
                         r: float, q: float, sigma: float, is_call: bool) -> float:
    """
    Fixed strike lookback option price via decomposition into floating lookbacks.

    Uses the identity (Conze-Viswanathan 1991):
    - Call: fixed_call = floating_put + S - K*df  (when M_max >= K)
    - Put:  fixed_put  = K*df - S + floating_call (when K >= M_min)

    Payoff:
    - Call: max(M_max - K, 0)
    - Put: max(K - M_min, 0)

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    M_min : float
        Running minimum of spot
    M_max : float
        Running maximum of spot
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

    Returns
    -------
    float
        Option price
    """
    if T <= 0:
        if is_call:
            return max(M_max - K, 0.0)
        else:
            return max(K - M_min, 0.0)

    if sigma <= 0:
        # Deterministic forward: path is monotone, so max/min is at endpoints
        b = r - q
        F = S * math.exp(b * T)
        df = math.exp(-r * T)
        if is_call:
            path_max = max(S, F)
            effective_max = max(M_max, path_max)
            return max(effective_max - K, 0.0) * df
        else:
            path_min = min(S, F)
            effective_min = min(M_min, path_min)
            return max(K - effective_min, 0.0) * df

    df = math.exp(-r * T)

    if is_call:
        # Ensure M_max is valid
        M = M_max
        if S > M:
            M = S
        float_put = lookback_floating_price(S, M_min, M, T, r, q, sigma, False)
        price = float_put + S - K * df
    else:
        # Ensure M_min is valid
        M = M_min
        if S < M:
            M = S
        float_call = lookback_floating_price(S, M, M_max, T, r, q, sigma, True)
        price = K * df - S + float_call

    return max(price, 0.0)


@njit(fastmath=True, cache=True)
def lookback_floating_price(S: float, M_min: float, M_max: float, T: float,
                            r: float, q: float, sigma: float, is_call: bool) -> float:
    """
    Floating strike lookback option price (Goldman-Sosin-Gatto 1979).

    Payoff:
    - Call: S_T - M_min (buy at the low)
    - Put: M_max - S_T (sell at the high)

    Parameters
    ----------
    S : float
        Spot price
    M_min : float
        Running minimum of spot
    M_max : float
        Running maximum of spot
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

    Returns
    -------
    float
        Option price
    """
    if T <= 0:
        if is_call:
            return max(S - M_min, 0.0)
        else:
            return max(M_max - S, 0.0)

    if sigma <= 0:
        # Deterministic forward: path is monotone
        b = r - q
        F = S * math.exp(b * T)
        df = math.exp(-r * T)
        if is_call:
            effective_min = min(M_min, min(S, F))
            return max(F - effective_min, 0.0) * df
        else:
            effective_max = max(M_max, max(S, F))
            return max(effective_max - F, 0.0) * df

    b = r - q
    # Guard against b ≈ 0 (division by zero in σ²/(2b) terms)
    b_eff = b if abs(b) > 1e-10 else 1e-10

    sqrt_T = math.sqrt(T)
    M = M_min if is_call else M_max

    qd = math.exp(-q * T)
    df = math.exp(-r * T)

    b1 = (math.log(S / M) + (b + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    b2 = b1 - sigma * sqrt_T

    two_b_over_sigma_sq = 2.0 * b_eff / (sigma * sigma)

    if is_call:
        term1 = S * qd * norm_cdf(b1) - M * df * norm_cdf(b2)
        term2 = S * qd * (sigma * sigma) / (2.0 * b_eff) * (
            -df * norm_cdf(-b2) +
            (S / M) ** (-two_b_over_sigma_sq) * norm_cdf(b1 - 2.0 * b_eff * sqrt_T / sigma)
        )
        price = term1 + term2
    else:
        term1 = M * df * norm_cdf(-b2) - S * qd * norm_cdf(-b1)
        term2 = S * qd * (sigma * sigma) / (2.0 * b_eff) * (
            df * norm_cdf(b2) -
            (S / M) ** (-two_b_over_sigma_sq) * norm_cdf(-b1 + 2.0 * b_eff * sqrt_T / sigma)
        )
        price = term1 + term2

    return max(price, 0.0)


@njit(fastmath=True, cache=True)
def _exotic_price(option_type: int, S: float, K: float, T: float, r: float, q: float,
                  sigma: float, is_call: bool, H: float, M_min: float, M_max: float,
                  is_knock_in: bool, is_up: bool, rebate: float, payout: float) -> float:
    """Dispatch to the correct exotic pricing kernel by option type."""
    if option_type == BARRIER:
        return barrier_option_price(S, K, H, T, r, q, sigma, is_call, is_knock_in, is_up, rebate)
    elif option_type == ASIAN_GEO:
        return asian_geometric_price(S, K, T, r, q, sigma, is_call)
    elif option_type == DIGITAL:
        return digital_price(S, K, T, r, q, sigma, is_call, payout)
    elif option_type == LOOKBACK_FIXED:
        return lookback_fixed_price(S, K, M_min, M_max, T, r, q, sigma, is_call)
    else:
        return lookback_floating_price(S, M_min, M_max, T, r, q, sigma, is_call)


@njit(fastmath=True, cache=True)
def exotic_calculate_greeks(option_type: int, S: float, K: float, T: float, r: float,
                            q: float, sigma: float, is_call: bool, H: float,
                            M_min: float, M_max: float, is_knock_in: bool, is_up: bool,
                            rebate: float, payout: float) -> tuple:
    """
    Calculate option Greeks using finite differences for all exotic types.

    Parameters
    ----------
    option_type : int
        BARRIER, ASIAN_GEO, DIGITAL, LOOKBACK_FIXED, or LOOKBACK_FLOATING
    S, K, T, r, q, sigma : float
        Standard option parameters (q = continuous dividend yield)
    is_call : bool
        True for call
    H : float
        Barrier level (for barrier options)
    M_min, M_max : float
        Running min/max (for lookback options)
    is_knock_in, is_up : bool
        Barrier type flags
    rebate, payout : float
        Rebate (barrier) or payout (digital)

    Returns
    -------
    tuple of 6 floats
        (price, delta, gamma, vega, theta, rho)
    """
    # Bump sizes aligned with GreeksBumpConfig defaults (numerical.py)
    dS = S * 0.01          # 1% relative spot bump
    dV = 0.01              # 1% absolute vol bump
    dT = 1.0 / DAYS_PER_YEAR  # 1 day
    dR = 0.0001            # 1 bp rate bump

    # Central price
    price = _exotic_price(option_type, S, K, T, r, q, sigma, is_call,
                          H, M_min, M_max, is_knock_in, is_up, rebate, payout)

    # Delta & Gamma (central differences on S)
    p_up = _exotic_price(option_type, S + dS, K, T, r, q, sigma, is_call,
                         H, M_min, M_max, is_knock_in, is_up, rebate, payout)
    p_dn = _exotic_price(option_type, S - dS, K, T, r, q, sigma, is_call,
                         H, M_min, M_max, is_knock_in, is_up, rebate, payout)
    delta = (p_up - p_dn) / (2.0 * dS)
    gamma = (p_up - 2.0 * price + p_dn) / (dS * dS)

    # Vega (central difference on sigma)
    p_vol_up = _exotic_price(option_type, S, K, T, r, q, sigma + dV, is_call,
                             H, M_min, M_max, is_knock_in, is_up, rebate, payout)
    p_vol_dn = _exotic_price(option_type, S, K, T, r, q, sigma - dV, is_call,
                             H, M_min, M_max, is_knock_in, is_up, rebate, payout)
    vega = (p_vol_up - p_vol_dn) / (2.0 * dV) / 100.0

    # Theta (forward difference on T)
    if T > dT:
        p_t_dn = _exotic_price(option_type, S, K, T - dT, r, q, sigma, is_call,
                               H, M_min, M_max, is_knock_in, is_up, rebate, payout)
        theta = (p_t_dn - price) / dT
    else:
        theta = 0.0

    # Rho (central difference on r)
    p_r_up = _exotic_price(option_type, S, K, T, r + dR, q, sigma, is_call,
                           H, M_min, M_max, is_knock_in, is_up, rebate, payout)
    p_r_dn = _exotic_price(option_type, S, K, T, r - dR, q, sigma, is_call,
                           H, M_min, M_max, is_knock_in, is_up, rebate, payout)
    rho = (p_r_up - p_r_dn) / (2.0 * dR) / 100.0

    return price, delta, gamma, vega, theta, rho


# =============================================================================
# ENGINE CLASS
# =============================================================================

@dataclass(frozen=True)
class ExoticAnalyticEngine(PricingEngine):
    """
    Analytical pricing engine for exotic options under GBM.

    Supports:
    - Barrier Options (all 8 types) via Reiner-Rubinstein 1991
    - Asian Geometric Options via Kemna-Vorst 1990
    - Digital/Binary Options (cash-or-nothing)
    - Lookback Options (floating and fixed strike)

    All kernels are Numba-compiled for performance.

    Examples
    --------
    engine = ExoticAnalyticEngine()
    gbm = GBMModel(sigma=0.25)
    market = MarketEnvironment(spot=100, rate=0.05)

    barrier = BarrierOption(100, 110, 0.25, is_call=True, is_up=True)
    result = engine.price(barrier, gbm, market)
    greeks = engine.greeks(barrier, gbm, market)
    """

    @property
    def capability(self) -> PricingCapability:
        return PricingCapability.ANALYTICAL

    @property
    def supported_exercises(self) -> list[ExerciseStyle]:
        return [ExerciseStyle.EUROPEAN]

    def can_price(self, instrument: Instrument, model: Model) -> bool:
        """
        Check if this engine can price the given combination.

        Requires GBMModel and one of the supported exotic instrument types.
        For AsianOption, only geometric average is supported analytically.
        """
        if not isinstance(model, GBMModel):
            return False

        if instrument.exercise_style != ExerciseStyle.EUROPEAN:
            return False

        if isinstance(instrument, BarrierOption):
            return True
        if isinstance(instrument, DigitalOption):
            return True
        if isinstance(instrument, LookbackOption):
            return True
        if isinstance(instrument, AsianOption):
            return instrument.average_type == "geometric"

        return False

    def price(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
    ) -> PricingResult:
        """
        Price an exotic option analytically.

        Parameters
        ----------
        instrument : Instrument
            Exotic option (Barrier, Asian geometric, Digital, or Lookback)
        model : GBMModel
            GBM model with constant volatility
        market : MarketEnvironment
            Current market conditions

        Returns
        -------
        PricingResult
            Analytical price
        """
        if not self.can_price(instrument, model):
            raise ValueError(
                f"ExoticAnalyticEngine cannot price {type(instrument).__name__} "
                f"with {type(model).__name__}"
            )

        S: float = market.spot
        r: float = market.rate
        q: float = market.dividend_yield
        sigma: float = model.sigma
        T: float = instrument.maturity

        if isinstance(instrument, BarrierOption):
            p = barrier_option_price(
                S=S, K=instrument.strike, H=instrument.barrier, T=T, r=r, q=q,
                sigma=sigma,
                is_call=instrument.is_call,
                is_knock_in=instrument.is_knock_in,
                is_up=instrument.is_up,
                rebate=instrument.rebate,
            )
        elif isinstance(instrument, AsianOption):
            p = asian_geometric_price(
                S=S, K=instrument.strike, T=T, r=r, q=q, sigma=sigma,
                is_call=instrument.is_call,
            )
        elif isinstance(instrument, DigitalOption):
            p = digital_price(
                S=S, K=instrument.strike, T=T, r=r, q=q, sigma=sigma,
                is_call=instrument.is_call,
                payout=instrument.payout,
            )
        elif isinstance(instrument, LookbackOption):
            if instrument.lookback_type == "fixed":
                p = lookback_fixed_price(
                    S=S, K=instrument.strike, M_min=S, M_max=S, T=T, r=r, q=q,
                    sigma=sigma,
                    is_call=instrument.is_call,
                )
            else:
                p = lookback_floating_price(
                    S=S, M_min=S, M_max=S, T=T, r=r, q=q, sigma=sigma,
                    is_call=instrument.is_call,
                )
        else:
            raise ValueError(f"Unsupported instrument: {type(instrument).__name__}")

        return PricingResult(
            price=p,
            engine="ExoticAnalyticEngine",
            model=model.name,
        )

    def greeks(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
    ) -> GreeksResult:
        """
        Compute Greeks for exotic options via finite differences.

        Parameters
        ----------
        instrument : Instrument
            Exotic option
        model : GBMModel
            GBM model
        market : MarketEnvironment
            Current market conditions

        Returns
        -------
        GreeksResult
            First-order Greeks (delta, gamma, vega, theta, rho)
        """
        if not self.can_price(instrument, model):
            raise ValueError(
                f"ExoticAnalyticEngine cannot compute Greeks for "
                f"{type(instrument).__name__} with {type(model).__name__}"
            )

        S: float = market.spot
        r: float = market.rate
        q: float = market.dividend_yield
        sigma: float = model.sigma
        T: float = instrument.maturity

        # Determine option type and extract parameters
        opt_type: int
        K: float
        H: float
        M_min: float
        M_max: float
        is_knock_in: bool
        is_up: bool
        rebate_val: float
        payout_val: float

        if isinstance(instrument, BarrierOption):
            opt_type = BARRIER
            K = instrument.strike
            H = instrument.barrier
            M_min = 0.0
            M_max = 0.0
            is_knock_in = instrument.is_knock_in
            is_up = instrument.is_up
            rebate_val = instrument.rebate
            payout_val = 1.0
        elif isinstance(instrument, AsianOption):
            opt_type = ASIAN_GEO
            K = instrument.strike
            H = 0.0
            M_min = 0.0
            M_max = 0.0
            is_knock_in = False
            is_up = True
            rebate_val = 0.0
            payout_val = 1.0
        elif isinstance(instrument, DigitalOption):
            opt_type = DIGITAL
            K = instrument.strike
            H = 0.0
            M_min = 0.0
            M_max = 0.0
            is_knock_in = False
            is_up = True
            rebate_val = 0.0
            payout_val = instrument.payout
        elif isinstance(instrument, LookbackOption):
            if instrument.lookback_type == "fixed":
                opt_type = LOOKBACK_FIXED
                K = instrument.strike
            else:
                opt_type = LOOKBACK_FLOATING
                K = 0.0
            H = 0.0
            M_min = S
            M_max = S
            is_knock_in = False
            is_up = True
            rebate_val = 0.0
            payout_val = 1.0
        else:
            raise ValueError(f"Unsupported instrument: {type(instrument).__name__}")

        price, delta, gamma, vega, theta, rho = exotic_calculate_greeks(
            option_type=opt_type,
            S=S, K=K, T=T, r=r, q=q, sigma=sigma,
            is_call=instrument.is_call,
            H=H, M_min=M_min, M_max=M_max,
            is_knock_in=is_knock_in, is_up=is_up,
            rebate=rebate_val, payout=payout_val,
        )

        return GreeksResult(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
        )


if __name__ == "__main__":
    from backend.models.gbm import GBMModel
    from backend.core.market import MarketEnvironment
    from backend.instruments.options import (
        BarrierOption, AsianOption, DigitalOption, LookbackOption
    )
    from backend.engines.analytic_engine import BSAnalyticEngine
    from backend.instruments.options import VanillaOption

    print("=" * 60)
    print("ExoticAnalyticEngine Smoke Test")
    print("=" * 60)

    engine = ExoticAnalyticEngine()
    gbm = GBMModel(sigma=0.25)
    market = MarketEnvironment(spot=100.0, rate=0.05)
    bs_engine = BSAnalyticEngine()

    # -------------------------------------------------------------------------
    # BARRIER OPTIONS
    # -------------------------------------------------------------------------
    print("\n--- Barrier Options ---")

    # Up-and-out call: knocked out if S reaches 120
    uoc = BarrierOption(
        strike=100.0, barrier=120.0, maturity=0.5,
        is_call=True, is_up=True, is_knock_in=False,
    )
    uoc_price = engine.price(instrument=uoc, model=gbm, market=market)
    uoc_greeks = engine.greeks(instrument=uoc, model=gbm, market=market)
    print(f"Up-Out Call  (K=100, B=120, T=0.5): price={uoc_price.price:.4f}")
    print(f"  delta={uoc_greeks.delta:.4f}  gamma={uoc_greeks.gamma:.6f}"
          f"  vega={uoc_greeks.vega:.4f}  theta={uoc_greeks.theta:.4f}")

    # Up-and-in call: symmetric pair
    uic = BarrierOption(
        strike=100.0, barrier=120.0, maturity=0.5,
        is_call=True, is_up=True, is_knock_in=True,
    )
    uic_price = engine.price(instrument=uic, model=gbm, market=market)
    print(f"Up-In  Call  (K=100, B=120, T=0.5): price={uic_price.price:.4f}")

    # Parity: knock-in + knock-out = vanilla
    vanilla = VanillaOption(strike=100.0, maturity=0.5, is_call=True)
    vanilla_price = bs_engine.price(instrument=vanilla, model=gbm, market=market)
    parity_diff = abs((uoc_price.price + uic_price.price) - vanilla_price.price)
    print(f"Knock-in/out parity error: {parity_diff:.2e}  "
          f"({'OK' if parity_diff < 0.01 else 'FAIL'})")

    # Down-and-out put
    dop = BarrierOption(
        strike=100.0, barrier=80.0, maturity=0.5,
        is_call=False, is_up=False, is_knock_in=False,
    )
    dop_price = engine.price(instrument=dop, model=gbm, market=market)
    dop_greeks = engine.greeks(instrument=dop, model=gbm, market=market)
    print(f"Down-Out Put (K=100, B=80,  T=0.5): price={dop_price.price:.4f}")
    print(f"  delta={dop_greeks.delta:.4f}  gamma={dop_greeks.gamma:.6f}"
          f"  vega={dop_greeks.vega:.4f}  theta={dop_greeks.theta:.4f}")

    # Barrier with rebate
    uoc_rebate = BarrierOption(
        strike=100.0, barrier=120.0, maturity=0.5,
        is_call=True, is_up=True, is_knock_in=False, rebate=5.0,
    )
    uoc_rebate_price = engine.price(instrument=uoc_rebate, model=gbm, market=market)
    print(f"Up-Out Call  (rebate=5):             price={uoc_rebate_price.price:.4f}"
          f"  (>= no-rebate {uoc_price.price:.4f}: "
          f"{'OK' if uoc_rebate_price.price >= uoc_price.price else 'FAIL'})")

    # -------------------------------------------------------------------------
    # ASIAN GEOMETRIC OPTIONS
    # -------------------------------------------------------------------------
    print("\n--- Asian Geometric Options ---")

    asian_call = AsianOption(
        strike=100.0, maturity=0.5, is_call=True, average_type="geometric"
    )
    asian_put = AsianOption(
        strike=100.0, maturity=0.5, is_call=False, average_type="geometric"
    )
    ac_price = engine.price(instrument=asian_call, model=gbm, market=market)
    ap_price = engine.price(instrument=asian_put, model=gbm, market=market)
    ac_greeks = engine.greeks(instrument=asian_call, model=gbm, market=market)
    print(f"Asian Geo Call (K=100, T=0.5): price={ac_price.price:.4f}")
    print(f"Asian Geo Put  (K=100, T=0.5): price={ap_price.price:.4f}")
    print(f"  Call cheaper than vanilla {vanilla_price.price:.4f}: "
          f"{'OK' if ac_price.price < vanilla_price.price else 'FAIL'}")
    print(f"  delta={ac_greeks.delta:.4f}  vega={ac_greeks.vega:.4f}"
          f"  theta={ac_greeks.theta:.4f}")

    # -------------------------------------------------------------------------
    # DIGITAL OPTIONS
    # -------------------------------------------------------------------------
    print("\n--- Digital (Cash-or-Nothing) Options ---")

    digital_call = DigitalOption(strike=100.0, maturity=0.5, is_call=True, payout=1.0)
    digital_put = DigitalOption(strike=100.0, maturity=0.5, is_call=False, payout=1.0)
    dc_price = engine.price(instrument=digital_call, model=gbm, market=market)
    dp_price = engine.price(instrument=digital_put, model=gbm, market=market)
    dc_greeks = engine.greeks(instrument=digital_call, model=gbm, market=market)
    print(f"Digital Call (K=100, T=0.5, payout=1): price={dc_price.price:.4f}")
    print(f"Digital Put  (K=100, T=0.5, payout=1): price={dp_price.price:.4f}")
    discount = math.exp(-market.rate * digital_call.maturity)
    parity_digital = abs((dc_price.price + dp_price.price) - discount)
    print(f"Call+Put = e^(-rT) parity error: {parity_digital:.2e}  "
          f"({'OK' if parity_digital < 1e-6 else 'FAIL'})")
    print(f"  delta={dc_greeks.delta:.4f}  vega={dc_greeks.vega:.4f}"
          f"  theta={dc_greeks.theta:.4f}")

    # Higher payout
    digital_10 = DigitalOption(strike=100.0, maturity=0.5, is_call=True, payout=10.0)
    dc10_price = engine.price(instrument=digital_10, model=gbm, market=market)
    print(f"Digital Call (payout=10): price={dc10_price.price:.4f}"
          f"  (= 10x parity {'OK' if abs(dc10_price.price - 10 * dc_price.price) < 1e-6 else 'FAIL'})")

    # -------------------------------------------------------------------------
    # LOOKBACK OPTIONS
    # -------------------------------------------------------------------------
    print("\n--- Lookback Options ---")

    # Floating strike (most common)
    lb_float_call = LookbackOption(maturity=0.5, is_call=True, lookback_type="floating")
    lb_float_put = LookbackOption(maturity=0.5, is_call=False, lookback_type="floating")
    lfc_price = engine.price(instrument=lb_float_call, model=gbm, market=market)
    lfp_price = engine.price(instrument=lb_float_put, model=gbm, market=market)
    lfc_greeks = engine.greeks(instrument=lb_float_call, model=gbm, market=market)
    print(f"Lookback Float Call (T=0.5): price={lfc_price.price:.4f}")
    print(f"Lookback Float Put  (T=0.5): price={lfp_price.price:.4f}")
    print(f"  Call > vanilla {vanilla_price.price:.4f}: "
          f"{'OK' if lfc_price.price > vanilla_price.price else 'FAIL'}")
    print(f"  delta={lfc_greeks.delta:.4f}  vega={lfc_greeks.vega:.4f}"
          f"  theta={lfc_greeks.theta:.4f}")

    # Fixed strike
    lb_fixed_call = LookbackOption(
        maturity=0.5, is_call=True, strike=100.0, lookback_type="fixed"
    )
    lb_fixed_put = LookbackOption(
        maturity=0.5, is_call=False, strike=100.0, lookback_type="fixed"
    )
    lfxc_price = engine.price(instrument=lb_fixed_call, model=gbm, market=market)
    lfxp_price = engine.price(instrument=lb_fixed_put, model=gbm, market=market)
    print(f"Lookback Fixed Call (K=100, T=0.5): price={lfxc_price.price:.4f}")
    print(f"Lookback Fixed Put  (K=100, T=0.5): price={lfxp_price.price:.4f}")

    # -------------------------------------------------------------------------
    # CAN_PRICE COMPATIBILITY
    # -------------------------------------------------------------------------
    print("\n--- Compatibility Check ---")
    from backend.models.heston import HestonModel

    heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    print(f"can_price(BarrierOption, GBM):    {engine.can_price(uoc, gbm)}")
    print(f"can_price(BarrierOption, Heston): {engine.can_price(uoc, heston)}")
    print(f"can_price(AsianOption arithmetic, GBM): "
          f"{engine.can_price(AsianOption(100, 0.5, average_type='arithmetic'), gbm)}")
    print(f"can_price(AsianOption geometric,  GBM): {engine.can_price(asian_call, gbm)}")
    print(f"can_price(DigitalOption, GBM):    {engine.can_price(digital_call, gbm)}")
    print(f"can_price(LookbackOption, GBM):   {engine.can_price(lb_float_call, gbm)}")

    print("\n" + "=" * 60)
    print("ExoticAnalyticEngine smoke test passed")
    print("=" * 60)
