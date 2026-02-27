"""
Exotic Options Pricing Adapter for Streamlit App

Bridges the Streamlit option pricer to the Numba-compiled exotic pricing kernels
in backend.engines.exotic_engine. Provides vectorized surface calculations for
charting exotic vs vanilla price/Greeks comparisons.

Author: Thomas Vaudescal
"""

import numpy as np
import math

from backend.engines.exotic_engine import (
    exotic_calculate_greeks,
    _exotic_price,
    _bs_vanilla_price,
    ExoticAnalyticEngine,
    BARRIER,
    ASIAN_GEO,
    DIGITAL,
    LOOKBACK_FIXED,
    LOOKBACK_FLOATING,
)
from backend.engines.vectorized_bs import calculate_all_greeks as _calculate_all_greeks_numba
from backend.instruments.options import BarrierOption, AsianOption, DigitalOption, LookbackOption
from backend.models.gbm import GBMModel
from backend.core.market import MarketEnvironment
from backend.greeks.calculator import GreeksCalculator


# Mapping from string type names to integer constants
EXOTIC_TYPE_MAP = {
    "barrier": BARRIER,
    "asian": ASIAN_GEO,
    "digital": DIGITAL,
    "lookback_fixed": LOOKBACK_FIXED,
    "lookback_floating": LOOKBACK_FLOATING,
}

# No dividend yield in the Streamlit app
_Q = 0.0


def calculate_exotic_greeks(
    exotic_type: str,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    is_call: bool,
    barrier: float = 0.0,
    is_knock_in: bool = False,
    is_up: bool = True,
    rebate: float = 0.0,
    payout: float = 1.0,
) -> tuple[float, float, float, float, float, float]:
    """Calculate (price, delta, gamma, vega, theta, rho) for an exotic option."""
    opt_type = EXOTIC_TYPE_MAP[exotic_type]

    # For lookbacks, fresh option: M_min = M_max = spot
    M_min = spot if exotic_type in ("lookback_fixed", "lookback_floating") else 0.0
    M_max = spot if exotic_type in ("lookback_fixed", "lookback_floating") else 0.0
    H = barrier if exotic_type == "barrier" else 0.0

    return exotic_calculate_greeks(
        option_type=opt_type,
        S=spot, K=strike, T=maturity, r=rate, q=_Q, sigma=sigma,
        is_call=is_call, H=H, M_min=M_min, M_max=M_max,
        is_knock_in=is_knock_in, is_up=is_up,
        rebate=rebate, payout=payout,
    )


def calculate_exotic_price(
    exotic_type: str,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    is_call: bool,
    barrier: float = 0.0,
    is_knock_in: bool = False,
    is_up: bool = True,
    rebate: float = 0.0,
    payout: float = 1.0,
) -> float:
    """Calculate exotic option price only."""
    opt_type = EXOTIC_TYPE_MAP[exotic_type]
    M_min = spot if exotic_type in ("lookback_fixed", "lookback_floating") else 0.0
    M_max = spot if exotic_type in ("lookback_fixed", "lookback_floating") else 0.0
    H = barrier if exotic_type == "barrier" else 0.0

    return _exotic_price(
        option_type=opt_type,
        S=spot, K=strike, T=maturity, r=rate, q=_Q, sigma=sigma,
        is_call=is_call, H=H, M_min=M_min, M_max=M_max,
        is_knock_in=is_knock_in, is_up=is_up,
        rebate=rebate, payout=payout,
    )


def calculate_vanilla_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    is_call: bool,
) -> float:
    """Calculate vanilla BS price for comparison."""
    return _bs_vanilla_price(spot, strike, maturity, rate, _Q, sigma, is_call)


def _create_exotic_instrument(exotic_type, strike, maturity, is_call, **kwargs):
    """Map adapter params to backend Instrument objects."""
    if exotic_type == 'barrier':
        return BarrierOption(
            strike=strike, barrier=kwargs['barrier'], maturity=maturity,
            is_call=is_call, is_up=kwargs['is_up'],
            is_knock_in=kwargs['is_knock_in'], rebate=kwargs.get('rebate', 0.0),
        )
    elif exotic_type == 'digital':
        return DigitalOption(
            strike=strike, maturity=maturity, is_call=is_call,
            payout=kwargs.get('payout', 1.0),
        )
    elif exotic_type == 'asian':
        return AsianOption(
            strike=strike, maturity=maturity, is_call=is_call,
            average_type="geometric",
        )
    elif exotic_type == 'lookback_floating':
        return LookbackOption(
            maturity=maturity, is_call=is_call, lookback_type="floating",
        )
    elif exotic_type == 'lookback_fixed':
        return LookbackOption(
            maturity=maturity, is_call=is_call, strike=strike,
            lookback_type="fixed",
        )
    else:
        raise ValueError(f"Unknown exotic type: {exotic_type}")


def calculate_exotic_all_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type_int: int,
    exotic_type: str,
    barrier: float = 0.0,
    is_up: bool = True,
    is_knock_in: bool = False,
    rebate: float = 0.0,
    payout: float = 1.0,
) -> np.ndarray:
    """Calculate all 14 Greeks for an exotic option via backend GreeksCalculator.

    Returns array of 14 values matching the vanilla convention:
    [price, delta, gamma, vega, theta, rho,
     vanna, volga, charm, veta, speed, zomma, color, ultima]

    Delegates to GreeksCalculator which uses ExoticAnalyticEngine for first-order
    Greeks and numerical cross finite differences for higher-order Greeks.
    """
    is_call = (option_type_int == 1)

    # Edge cases
    if time_to_expiry <= 0 or volatility <= 0 or spot <= 0:
        return np.zeros(14)

    # Build backend objects
    instrument = _create_exotic_instrument(
        exotic_type, strike, time_to_expiry, is_call,
        barrier=barrier, is_up=is_up, is_knock_in=is_knock_in,
        rebate=rebate, payout=payout,
    )
    model = GBMModel(sigma=volatility)
    market = MarketEnvironment(spot=spot, rate=risk_free_rate)
    engine = ExoticAnalyticEngine()

    # Full 14 Greeks via backend
    calc = GreeksCalculator()
    greeks = calc.calculate(engine, instrument, model, market, include_higher_order=True)

    # Convert AllGreeksResult to 14-element array
    result = np.array([
        greeks.price, greeks.delta, greeks.gamma, greeks.vega, greeks.theta, greeks.rho,
        greeks.vanna, greeks.volga, greeks.charm, greeks.veta,
        greeks.speed, greeks.zomma, greeks.color, greeks.ultima,
    ])

    # Scale higher-order Greeks to match frontend market conventions
    # (GreeksCalculator returns raw derivatives; frontend expects market-scaled)
    result[6] /= 100.0      # vanna: per 1% vol
    result[7] /= 10000.0    # volga: per 1%² vol
    # result[8] unchanged   # charm: already per day
    result[9] /= 100.0      # veta: per 1% vol (already per day)
    # result[10] unchanged  # speed: raw
    result[11] /= 100.0     # zomma: per 1% vol
    # result[12] unchanged  # color: already per day
    result[13] /= 1e6       # ultima: per 1%³ vol

    return result


def calculate_exotic_premium(
    spot: float,
    strike: float,
    dte_days: int,
    risk_free_rate: float,
    volatility: float,
    option_type: str,
    exotic_type: str,
    barrier: float = 0.0,
    is_up: bool = True,
    is_knock_in: bool = False,
    rebate: float = 0.0,
    payout: float = 1.0,
) -> float:
    """Calculate exotic option premium (per share).

    Mirrors calculate_option_premium() from pricing_adapter but for exotics.
    """
    maturity = dte_days / 365.0
    is_call = (option_type == 'call')
    return calculate_exotic_price(
        exotic_type=exotic_type,
        spot=spot, strike=strike, maturity=maturity,
        rate=risk_free_rate, sigma=volatility, is_call=is_call,
        barrier=barrier, is_knock_in=is_knock_in, is_up=is_up,
        rebate=rebate, payout=payout,
    )


def calculate_exotic_payoff_at_expiry(spot: float, position: dict) -> float:
    """Calculate per-share payoff at expiry for an exotic leg.

    Args:
        spot: Spot price at expiry.
        position: Position dict with instrument_class and exotic params.

    Returns:
        Payoff per share (before position sign / quantity scaling).
    """
    exotic_type = position.get('instrument_class', 'vanilla')
    is_call = (position['option_type'] == 'call')
    strike = position['strike']
    barrier = position.get('barrier', 0.0)
    is_up = position.get('is_up', True)
    is_knock_in = position.get('is_knock_in', False)
    payout_amount = position.get('payout', 1.0)

    if exotic_type == 'digital':
        if is_call:
            return payout_amount if spot > strike else 0.0
        else:
            return payout_amount if spot < strike else 0.0

    if exotic_type == 'barrier':
        # Simplified terminal barrier check
        vanilla_payoff = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        if is_up:
            barrier_hit = (spot >= barrier)
        else:
            barrier_hit = (spot <= barrier)

        if is_knock_in:
            return vanilla_payoff if barrier_hit else 0.0
        else:  # knock-out
            return 0.0 if barrier_hit else vanilla_payoff

    if exotic_type == 'lookback_floating':
        # Floating lookback payoff depends on path extremes unavailable at expiry.
        return 0.0

    # Asian / lookback fixed: at expiry, path-dependent features collapse to vanilla intrinsic
    # (without actual path history, vanilla intrinsic is the best deterministic proxy)
    if is_call:
        return max(spot - strike, 0.0)
    else:
        return max(strike - spot, 0.0)


def calculate_exotic_greeks_surface(
    exotic_type: str,
    spot_range: np.ndarray,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    is_call: bool,
    barrier: float = 0.0,
    is_knock_in: bool = False,
    is_up: bool = True,
    rebate: float = 0.0,
    payout: float = 1.0,
) -> dict:
    """Calculate exotic Greeks over a range of spot prices.

    Returns dict with keys: price, delta, gamma, vega, theta, rho (each np.ndarray).
    """
    n = len(spot_range)
    result = {k: np.zeros(n) for k in ("price", "delta", "gamma", "vega", "theta", "rho")}

    for i, s in enumerate(spot_range):
        p, d, g, v, t, r = calculate_exotic_greeks(
            exotic_type, s, strike, maturity, rate, sigma, is_call,
            barrier=barrier, is_knock_in=is_knock_in, is_up=is_up,
            rebate=rebate, payout=payout,
        )
        result["price"][i] = p
        result["delta"][i] = d
        result["gamma"][i] = g
        result["vega"][i] = v
        result["theta"][i] = t
        result["rho"][i] = r

    return result


def calculate_vanilla_greeks_surface(
    spot_range: np.ndarray,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    is_call: bool,
) -> dict:
    """Calculate vanilla BS Greeks over a range of spot prices.

    Returns dict with keys: price, delta, gamma, vega, theta, rho (each np.ndarray).
    """
    n = len(spot_range)
    result = {k: np.zeros(n) for k in ("price", "delta", "gamma", "vega", "theta", "rho")}
    opt_type_int = 1 if is_call else 0

    for i, s in enumerate(spot_range):
        greeks = _calculate_all_greeks_numba(s, strike, maturity, rate, sigma, opt_type_int)
        result["price"][i] = greeks[0]
        result["delta"][i] = greeks[1]
        result["gamma"][i] = greeks[2]
        result["vega"][i] = greeks[3]
        result["theta"][i] = greeks[4]
        result["rho"][i] = greeks[5]

    return result


def calculate_barrier_parity_surface(
    spot_range: np.ndarray,
    strike: float,
    barrier: float,
    maturity: float,
    rate: float,
    sigma: float,
    is_call: bool,
    is_up: bool,
) -> dict:
    """Calculate knock-in, knock-out, and vanilla prices for barrier parity chart.

    Returns dict with keys: knock_in, knock_out, vanilla (each np.ndarray).
    """
    n = len(spot_range)
    ki = np.zeros(n)
    ko = np.zeros(n)
    van = np.zeros(n)

    for i, s in enumerate(spot_range):
        ki[i] = calculate_exotic_price(
            "barrier", s, strike, maturity, rate, sigma, is_call,
            barrier=barrier, is_knock_in=True, is_up=is_up,
        )
        ko[i] = calculate_exotic_price(
            "barrier", s, strike, maturity, rate, sigma, is_call,
            barrier=barrier, is_knock_in=False, is_up=is_up,
        )
        van[i] = calculate_vanilla_price(s, strike, maturity, rate, sigma, is_call)

    return {"knock_in": ki, "knock_out": ko, "vanilla": van}


def calculate_digital_parity_surface(
    spot_range: np.ndarray,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    payout: float,
) -> dict:
    """Calculate digital call + put parity surface.

    Returns dict with keys: digital_call, digital_put, discount_factor.
    """
    n = len(spot_range)
    dc = np.zeros(n)
    dp = np.zeros(n)

    for i, s in enumerate(spot_range):
        dc[i] = calculate_exotic_price(
            "digital", s, strike, maturity, rate, sigma,
            is_call=True, payout=payout,
        )
        dp[i] = calculate_exotic_price(
            "digital", s, strike, maturity, rate, sigma,
            is_call=False, payout=payout,
        )

    return {
        "digital_call": dc,
        "digital_put": dp,
        "discount_factor": math.exp(-rate * maturity),
    }


def calculate_price_vs_param(
    exotic_type: str,
    param_name: str,
    param_range: np.ndarray,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    is_call: bool,
    barrier: float = 0.0,
    is_knock_in: bool = False,
    is_up: bool = True,
    rebate: float = 0.0,
    payout: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate exotic and vanilla prices over a parameter sweep.

    param_name: 'volatility' or 'maturity'
    Returns (exotic_prices, vanilla_prices).
    """
    n = len(param_range)
    exotic_prices = np.zeros(n)
    vanilla_prices = np.zeros(n)

    for i, val in enumerate(param_range):
        s_val, k_val, t_val, r_val, sig_val = spot, strike, maturity, rate, sigma
        if param_name == "volatility":
            sig_val = val
        elif param_name == "maturity":
            t_val = val

        exotic_prices[i] = calculate_exotic_price(
            exotic_type, s_val, k_val, t_val, r_val, sig_val, is_call,
            barrier=barrier, is_knock_in=is_knock_in, is_up=is_up,
            rebate=rebate, payout=payout,
        )
        vanilla_prices[i] = calculate_vanilla_price(
            s_val, k_val, t_val, r_val, sig_val, is_call,
        )

    return exotic_prices, vanilla_prices
