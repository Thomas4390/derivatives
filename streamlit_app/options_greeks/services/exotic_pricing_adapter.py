"""
Exotic Options Pricing Adapter for Streamlit App

Bridges the Streamlit option pricer to the Numba-compiled exotic pricing kernels
in backend.engines.exotic_engine. Provides vectorized surface calculations for
charting exotic vs vanilla price/Greeks comparisons.

Author: Thomas Vaudescal
"""

import math
from collections.abc import Callable

import numpy as np

from backend.core.market import MarketEnvironment
from backend.engines.exotic_engine import (
    ASIAN_GEO,
    ASSET_OR_NOTHING,
    BARRIER,
    CHOOSER,
    DIGITAL,
    GAP,
    LOOKBACK_FIXED,
    LOOKBACK_FLOATING,
    POWER,
    ExoticAnalyticEngine,
    _bs_vanilla_price,
    _exotic_price,
    exotic_calculate_greeks,
    exotic_greeks_surface,
    exotic_price_param_sweep,
    exotic_price_surface,
)
from backend.engines.vectorized_bs import (
    calculate_greeks_vectorized as _calculate_greeks_vectorized_numba,
)
from backend.greeks.calculator import GreeksCalculator
from backend.instruments.exotic_advanced import (
    ArithmeticAsianOption,
    BinaryBarrierOption,
    CappedPowerOption,
    DiscreteBarrierOption,
    DoubleBarrierOption,
    LogContract,
    LogOption,
    PartialTimeBarrierOption,
    PoweredOption,
    SupershareOption,
)
from backend.instruments.options import (
    AsianOption,
    AssetOrNothingOption,
    BarrierOption,
    ChooserOption,
    DigitalOption,
    GapOption,
    LookbackOption,
    PowerOption,
)
from backend.models.gbm import GBMModel

# extra1 semantics per exotic type (Numba kernels use a single float param):
#   CHOOSER         → t_c  (choice time in years)
#   POWER           → n    (power exponent, e.g. 2.0 for S^2)
#   GAP             → K2   (trigger strike)
#   All other types → unused (should be 0.0)

# Mapping from string type names to integer constants
EXOTIC_TYPE_MAP = {
    "barrier": BARRIER,
    "asian": ASIAN_GEO,
    "digital": DIGITAL,
    "lookback_floating": LOOKBACK_FLOATING,
    "lookback_fixed": LOOKBACK_FIXED,
    "chooser": CHOOSER,
    "asset_or_nothing": ASSET_OR_NOTHING,
    "power": POWER,
    "gap": GAP,
}


def _lookback_defaults(exotic_type: str, spot: float) -> tuple[float, float]:
    """Return (M_min, M_max) defaults for lookback options (fresh option: extremes = spot)."""
    if exotic_type in ("lookback_fixed", "lookback_floating"):
        return spot, spot
    return 0.0, 0.0


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
    extra1: float = 0.0,
    dividend_yield: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    """Calculate (price, delta, gamma, vega, theta, rho) for an exotic option.

    extra1 encodes type-specific params: t_c (chooser), n (power), K2 (gap).
    """
    opt_type = EXOTIC_TYPE_MAP[exotic_type]

    M_min, M_max = _lookback_defaults(exotic_type, spot)
    H = barrier if exotic_type == "barrier" else 0.0

    return exotic_calculate_greeks(
        option_type=opt_type,
        S=spot,
        K=strike,
        T=maturity,
        r=rate,
        q=dividend_yield,
        sigma=sigma,
        is_call=is_call,
        H=H,
        M_min=M_min,
        M_max=M_max,
        is_knock_in=is_knock_in,
        is_up=is_up,
        rebate=rebate,
        payout=payout,
        extra1=extra1,
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
    extra1: float = 0.0,
    dividend_yield: float = 0.0,
    cap: float = 0.0,
    params: dict | None = None,
) -> float:
    """Calculate exotic option price only.

    extra1 encodes type-specific params: t_c (chooser), n (power), K2 (gap),
    or the exponent for the Haug powered/capped-power members. ``cap`` is the
    capped-power payoff ceiling. ``params`` carries any other family-specific
    named parameters (e.g. ``accrual``/``step`` for time-switch,
    ``lower_strike``/``upper_strike`` for supershare, ``average_period`` for the
    arithmetic Asian). Legacy basic-8 types price through the integer Numba
    kernels; Haug-catalog types route through the registry.
    """
    if exotic_type not in EXOTIC_TYPE_MAP:
        return _price_via_registry(
            exotic_type,
            spot,
            strike,
            maturity,
            rate,
            sigma,
            is_call,
            power=extra1,
            cap=cap,
            dividend_yield=dividend_yield,
            params=params,
        )

    opt_type = EXOTIC_TYPE_MAP[exotic_type]
    M_min, M_max = _lookback_defaults(exotic_type, spot)
    H = barrier if exotic_type == "barrier" else 0.0

    return _exotic_price(
        option_type=opt_type,
        S=spot,
        K=strike,
        T=maturity,
        r=rate,
        q=dividend_yield,
        sigma=sigma,
        is_call=is_call,
        H=H,
        M_min=M_min,
        M_max=M_max,
        is_knock_in=is_knock_in,
        is_up=is_up,
        rebate=rebate,
        payout=payout,
        extra1=extra1,
    )


def calculate_vanilla_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    is_call: bool,
    dividend_yield: float = 0.0,
) -> float:
    """Calculate vanilla BS price for comparison."""
    return _bs_vanilla_price(
        spot, strike, maturity, rate, dividend_yield, sigma, is_call
    )


def _safe_cap(cap: float, strike: float, is_call: bool) -> float:
    """Return a valid ``CappedPowerOption`` cap (> 0; ``< strike`` for a put).

    Falls back to ``0.5 * strike`` when the requested cap is missing or violates
    the constructor's invariant, so the Greeks / surface paths (which do not
    thread the user's cap) still build a valid instrument.
    """
    cap = float(cap)
    if cap <= 0.0:
        return 0.5 * strike
    if not is_call and cap >= strike:
        return 0.5 * strike
    return cap


def _make_supershare(strike, maturity, is_call, kw):
    """Build a SupershareOption with a valid positive ``lower < upper`` band.

    Falls back to a +/-10% band around ``strike`` when the requested boundaries
    are missing or violate the constructor invariant (so the Greeks / surface
    paths, which do not thread the band, still build a valid instrument).
    """
    lower = float(kw.get("lower_strike") or 0.9 * strike)
    upper = float(kw.get("upper_strike") or 1.1 * strike)
    if lower <= 0:
        lower = 0.9 * strike
    if upper <= lower:
        upper = lower * 1.1
    return SupershareOption(lower_strike=lower, upper_strike=upper, maturity=maturity)


# The 5 Heynen-Kat partial-time barrier monitoring types (mirrors the backend's
# private _PARTIAL_BARRIER_TYPES); the order drives the UI selector.
_PARTIAL_BARRIER_TYPES_UI = (
    "down_out_A",
    "up_out_A",
    "out_B1",
    "down_out_B2",
    "up_out_B2",
)


def _make_partial_barrier(strike, maturity, is_call, kw):
    """Build a PartialTimeBarrierOption with a valid ``0 < t1 <= maturity`` and a
    known monitoring type (falls back to ``out_B1``)."""
    barrier = float(kw.get("barrier") or 1.1 * strike)
    t1 = float(kw.get("t1") or 0.5 * maturity)
    t1 = min(max(t1, 1e-6), maturity)
    btype = kw.get("barrier_type", "out_B1")
    if btype not in _PARTIAL_BARRIER_TYPES_UI:
        btype = "out_B1"
    return PartialTimeBarrierOption(
        strike=strike,
        barrier=barrier,
        t1=t1,
        maturity=maturity,
        barrier_type=btype,
        is_call=is_call,
    )


def _clamp_band(
    lower, upper, strike: float, *, strict: bool, default_lo: float, default_hi: float
) -> tuple[float, float]:
    """Return a valid ``(lower, upper)`` band (positive; ``upper>lower`` if strict,
    else ``upper>=lower``), defaulting to fractions of ``strike`` when missing."""
    lo = float(lower) if lower and lower > 0 else default_lo * strike
    hi = float(upper) if upper and upper > 0 else default_hi * strike
    if strict and hi <= lo:
        hi = lo * 1.2
    elif not strict and hi < lo:
        hi = lo
    return lo, hi


def _make_double_barrier(strike, maturity, is_call, kw):
    """Build a DoubleBarrierOption with a valid ``lower < upper`` band."""
    lo, hi = _clamp_band(
        kw.get("lower"),
        kw.get("upper"),
        strike,
        strict=True,
        default_lo=0.8,
        default_hi=1.2,
    )
    return DoubleBarrierOption(
        strike=strike,
        lower=lo,
        upper=hi,
        maturity=maturity,
        is_call=is_call,
        is_knock_in=bool(kw.get("is_knock_in", False)),
    )


# Map each exotic type to a factory that builds the backend Instrument from the
# adapter params (strike, maturity, is_call, kwargs-dict). Replaces a 9-branch
# if/elif chain: the type → constructor mapping is now one extensible table.
# Haug-catalog (exotic_advanced) members are priced through the Open/Closed
# registry behind ExoticAnalyticEngine, not the integer EXOTIC_TYPE_MAP kernels.
_EXOTIC_INSTRUMENT_FACTORIES: dict[
    str, Callable[[float, float, bool, dict], object]
] = {
    "barrier": lambda strike, maturity, is_call, kw: BarrierOption(
        strike=strike,
        barrier=kw["barrier"],
        maturity=maturity,
        is_call=is_call,
        is_up=kw["is_up"],
        is_knock_in=kw["is_knock_in"],
        rebate=kw.get("rebate", 0.0),
    ),
    "digital": lambda strike, maturity, is_call, kw: DigitalOption(
        strike=strike, maturity=maturity, is_call=is_call, payout=kw.get("payout", 1.0)
    ),
    "asian": lambda strike, maturity, is_call, kw: AsianOption(
        strike=strike, maturity=maturity, is_call=is_call, average_type="geometric"
    ),
    "arithmetic_asian": lambda strike, maturity, is_call, kw: ArithmeticAsianOption(
        strike=strike,
        maturity=maturity,
        # Fresh option when no window info: averaging window == remaining life.
        average_period=float(kw.get("average_period") or maturity),
        realized_average=float(kw.get("realized_average", 0.0)),
        is_call=is_call,
    ),
    "lookback_floating": lambda strike, maturity, is_call, kw: LookbackOption(
        maturity=maturity, is_call=is_call, strike=None, lookback_type="floating"
    ),
    "lookback_fixed": lambda strike, maturity, is_call, kw: LookbackOption(
        maturity=maturity, is_call=is_call, strike=strike, lookback_type="fixed"
    ),
    "chooser": lambda strike, maturity, is_call, kw: ChooserOption(
        strike=strike,
        maturity=maturity,
        choice_time=kw.get("choice_time", maturity * 0.5),
    ),
    "asset_or_nothing": lambda strike, maturity, is_call, kw: AssetOrNothingOption(
        strike=strike, maturity=maturity, is_call=is_call
    ),
    "power": lambda strike, maturity, is_call, kw: PowerOption(
        strike=strike, maturity=maturity, is_call=is_call, power=kw.get("power_n", 2.0)
    ),
    "gap": lambda strike, maturity, is_call, kw: GapOption(
        strike=strike,
        trigger=kw.get("gap_trigger", strike * 1.05),
        maturity=maturity,
        is_call=is_call,
    ),
    # ── Haug advanced (registry-priced) ──
    "powered": lambda strike, maturity, is_call, kw: PoweredOption(
        strike=strike,
        maturity=maturity,
        power=int(kw.get("power_n", 2)),
        is_call=is_call,
    ),
    "capped_power": lambda strike, maturity, is_call, kw: CappedPowerOption(
        strike=strike,
        maturity=maturity,
        power=float(kw.get("power_n", 2.0)),
        cap=_safe_cap(kw.get("cap", 0.0), strike, is_call),
        is_call=is_call,
    ),
    "log_contract": lambda strike, maturity, is_call, kw: LogContract(
        strike=strike, maturity=maturity
    ),
    "log_option": lambda strike, maturity, is_call, kw: LogOption(
        strike=strike, maturity=maturity
    ),
    "supershare": _make_supershare,
    "double_barrier": _make_double_barrier,
    "discrete_barrier": lambda strike, maturity, is_call, kw: DiscreteBarrierOption(
        strike=strike,
        # `barrier or default`: the Greeks path passes barrier=0.0 explicitly.
        barrier=float(kw.get("barrier") or 1.1 * strike),
        maturity=maturity,
        is_call=is_call,
        is_up=bool(kw.get("is_up", True)),
        is_knock_in=bool(kw.get("is_knock_in", False)),
        monitoring_points=int(kw.get("monitoring_points") or 252),
        rebate=float(kw.get("rebate", 0.0)),
    ),
    "partial_barrier": _make_partial_barrier,
    "binary_barrier": lambda strike, maturity, is_call, kw: BinaryBarrierOption(
        strike=strike,
        # `barrier or default`: the Greeks path passes barrier=0.0 explicitly.
        barrier=float(kw.get("barrier") or 1.1 * strike),
        cash=float(kw.get("cash", 10.0)),
        maturity=maturity,
        binary_type=int(kw.get("binary_type") or 13),
    ),
}


def _create_exotic_instrument(exotic_type, strike, maturity, is_call, **kwargs):
    """Map adapter params to backend Instrument objects (table-driven dispatch)."""
    factory = _EXOTIC_INSTRUMENT_FACTORIES.get(exotic_type)
    if factory is None:
        raise ValueError(f"Unknown exotic type: {exotic_type}")
    return factory(strike, maturity, is_call, kwargs)


def _resolve_pct_markers(kwargs: dict, time_to_expiry: float) -> dict:
    """Resolve maturity-relative marker keys (t1_pct, avg_elapsed_pct) into
    absolute factory kwargs against the CURRENT time to expiry, so DTE-varying
    callers stay correct. Mutates and returns ``kwargs``."""
    t1_pct = kwargs.pop("t1_pct", None)
    if t1_pct is not None:
        kwargs.setdefault("t1", float(t1_pct) * time_to_expiry)
    avg_elapsed = kwargs.pop("avg_elapsed_pct", None)
    if avg_elapsed is not None:
        w = min(max(float(avg_elapsed), 0.0), 0.95)
        kwargs.setdefault("average_period", time_to_expiry / (1.0 - w))
    return kwargs


def _price_via_registry(
    exotic_type: str,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    is_call: bool,
    *,
    power: float = 0.0,
    cap: float = 0.0,
    dividend_yield: float = 0.0,
    params: dict | None = None,
) -> float:
    """Price a Haug-catalog exotic through the Open/Closed registry.

    Builds the ``exotic_advanced`` instrument and prices it with
    :class:`ExoticAnalyticEngine` (``lookup`` → registry adapter) — the same
    seam :func:`calculate_exotic_all_greeks` already uses for Greeks. ``params``
    supplies family-specific factory kwargs (explicit values win over the
    ``power``/``cap`` shortcuts); maturity-relative markers (``t1_pct``,
    ``avg_elapsed_pct``) are resolved against ``maturity`` via
    :func:`_resolve_pct_markers`, the same resolution
    :func:`calculate_exotic_all_greeks` applies for Greeks. Raises
    ``ValueError`` for an unknown type (never a bare ``KeyError``).
    """
    kwargs: dict = dict(params) if params else {}
    kwargs = _resolve_pct_markers(kwargs, maturity)
    if power and power > 0:
        kwargs.setdefault("power_n", power)
    if cap and cap > 0:
        kwargs.setdefault("cap", cap)
    instrument = _create_exotic_instrument(
        exotic_type, strike, maturity, is_call, **kwargs
    )
    model = GBMModel(sigma=sigma)
    market = MarketEnvironment(spot=spot, rate=rate, dividend_yield=dividend_yield)
    return float(ExoticAnalyticEngine().price(instrument, model, market).price)


def haug_factory_params(position: dict) -> dict:
    """Family-specific factory kwargs of an advanced (registry-priced) leg.

    Translates the UI leg keys into the ``_EXOTIC_INSTRUMENT_FACTORIES``
    vocabulary (mirrors the leg editor's ``new_params`` mapping). Tolerant to
    missing keys — the factories keep their documented safe defaults. The
    partial-barrier window is kept as ``t1_pct`` (fraction of maturity) so a
    DTE-slider consumer can resolve ``t1 = t1_pct * T`` per call.
    """
    inst = position.get("instrument_class")
    keymaps: dict[str, tuple[tuple[str, str], ...]] = {
        "supershare": (
            ("lower_strike", "lower_strike"),
            ("upper_strike", "upper_strike"),
        ),
        "double_barrier": (
            ("lower", "dbl_lower"),
            ("upper", "dbl_upper"),
            ("is_knock_in", "adv_in"),
        ),
        "discrete_barrier": (
            ("barrier", "adv_barrier"),
            ("is_up", "adv_is_up"),
            ("is_knock_in", "adv_in"),
            ("monitoring_points", "monitoring_points"),
        ),
        "partial_barrier": (
            ("barrier", "adv_barrier"),
            ("t1_pct", "t1_pct"),
            ("barrier_type", "partial_type"),
        ),
        "binary_barrier": (
            ("barrier", "adv_barrier"),
            ("cash", "cash"),
            ("binary_type", "binary_type"),
        ),
        "capped_power": (("cap", "cap"),),
        "arithmetic_asian": (
            ("avg_elapsed_pct", "avg_elapsed_pct"),
            ("realized_average", "avg_realized"),
        ),
    }
    return {
        factory_key: position[pos_key]
        for factory_key, pos_key in keymaps.get(inst or "", ())
        if position.get(pos_key) is not None
    }


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
    extra1: float = 0.0,
    dividend_yield: float = 0.0,
    cap: float = 0.0,
    params: dict | None = None,
    include_higher_order: bool = True,
) -> np.ndarray:
    """Calculate all 14 Greeks for an exotic option via backend GreeksCalculator.

    Returns array of 14 values matching the vanilla convention:
    [price, delta, gamma, vega, theta, rho,
     vanna, volga, charm, veta, speed, zomma, color, ultima]

    Delegates to GreeksCalculator which uses ExoticAnalyticEngine for first-order
    Greeks and numerical cross finite differences for higher-order Greeks.
    ``params`` supplies family-specific factory kwargs (advanced families:
    corridor, adv_* barrier flags, cash, ...) and wins over the basic
    shortcuts; a ``t1_pct`` entry is resolved to ``t1 = t1_pct * T`` here so
    DTE-slider consumers stay correct as T varies. With
    ``include_higher_order=False`` only indices 0-5 are computed (indices 6-13
    stay 0.0) — ~5x cheaper, for per-cell grid sweeps whose higher orders are
    masked anyway.
    """
    is_call = option_type_int == 1

    # Edge cases
    if time_to_expiry <= 0 or volatility <= 0 or spot <= 0:
        return np.zeros(14)

    # Map extra1 to type-specific kwargs
    extra_kwargs = {}
    if exotic_type == "chooser" and extra1 > 0:
        extra_kwargs["choice_time"] = extra1
    elif exotic_type == "power" and extra1 > 0:
        extra_kwargs["power_n"] = extra1
    elif exotic_type == "gap" and extra1 > 0:
        extra_kwargs["gap_trigger"] = extra1
    elif exotic_type == "powered" and extra1 > 0:
        extra_kwargs["power_n"] = extra1
    elif exotic_type == "capped_power":
        if extra1 > 0:
            extra_kwargs["power_n"] = extra1
        if cap > 0:
            extra_kwargs["cap"] = cap
    if params:
        extra_kwargs.update(params)
        extra_kwargs = _resolve_pct_markers(extra_kwargs, time_to_expiry)

    # Build backend objects
    instrument = _create_exotic_instrument(
        exotic_type,
        strike,
        time_to_expiry,
        is_call,
        **{
            "barrier": barrier,
            "is_up": is_up,
            "is_knock_in": is_knock_in,
            "rebate": rebate,
            "payout": payout,
            **extra_kwargs,
        },
    )
    model = GBMModel(sigma=volatility)
    market = MarketEnvironment(
        spot=spot, rate=risk_free_rate, dividend_yield=dividend_yield
    )
    engine = ExoticAnalyticEngine()

    calc = GreeksCalculator()
    greeks = calc.calculate(
        engine, instrument, model, market, include_higher_order=include_higher_order
    )

    result = np.zeros(14)
    result[0] = greeks.price
    result[1] = greeks.delta
    result[2] = greeks.gamma
    result[3] = greeks.vega
    result[4] = greeks.theta
    result[5] = greeks.rho
    if not include_higher_order:
        return result

    result[6:14] = [
        greeks.vanna,
        greeks.volga,
        greeks.charm,
        greeks.veta,
        greeks.speed,
        greeks.zomma,
        greeks.color,
        greeks.ultima,
    ]
    # Scale higher-order Greeks to match frontend market conventions
    # (GreeksCalculator returns raw derivatives; frontend expects market-scaled)
    result[6] /= 100.0  # vanna: per 1% vol
    result[7] /= 10000.0  # volga: per 1%² vol
    # result[8] unchanged   # charm: already per day
    result[9] /= 100.0  # veta: per 1% vol (already per day)
    # result[10] unchanged  # speed: raw
    result[11] /= 100.0  # zomma: per 1% vol
    # result[12] unchanged  # color: already per day
    result[13] /= 1e6  # ultima: per 1%³ vol

    return result


def calculate_exotic_greeks_curve(
    spot_range: np.ndarray,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type_int: int,
    exotic_type: str,
    *,
    barrier: float = 0.0,
    is_up: bool = True,
    is_knock_in: bool = False,
    rebate: float = 0.0,
    payout: float = 1.0,
    extra1: float = 0.0,
    cap: float = 0.0,
    dividend_yield: float = 0.0,
    params: dict | None = None,
) -> np.ndarray:
    """First-order Greeks of one exotic leg over a spot grid, shape ``(n, 14)``.

    Per-spot loop over :func:`calculate_exotic_all_greeks` with
    ``include_higher_order=False`` (indices 6-13 stay 0.0 — the same contract
    as the native kernel branch, whose higher orders are masked by
    GREEK_AVAILABILITY anyway). This is the registry-priced path the 3D
    surface tab already uses; it makes the 2D chain agree with it.
    """
    spots = np.asarray(spot_range, dtype=np.float64)
    out = np.zeros((len(spots), 14))
    if time_to_expiry <= 0 or volatility <= 0:
        return out
    for i, spot in enumerate(spots):
        out[i] = calculate_exotic_all_greeks(
            float(spot),
            strike,
            time_to_expiry,
            risk_free_rate,
            volatility,
            option_type_int,
            exotic_type=exotic_type,
            barrier=barrier,
            is_up=is_up,
            is_knock_in=is_knock_in,
            rebate=rebate,
            payout=payout,
            extra1=extra1,
            cap=cap,
            dividend_yield=dividend_yield,
            params=params,
            include_higher_order=False,
        )
    return out


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
    extra1: float = 0.0,
    dividend_yield: float = 0.0,
    cap: float = 0.0,
    params: dict | None = None,
) -> float:
    """Calculate exotic option premium (per share).

    Mirrors calculate_option_premium() from pricing_adapter but for exotics.
    """
    maturity = dte_days / 365.0
    is_call = option_type == "call"
    return calculate_exotic_price(
        exotic_type=exotic_type,
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=risk_free_rate,
        sigma=volatility,
        is_call=is_call,
        barrier=barrier,
        is_knock_in=is_knock_in,
        is_up=is_up,
        rebate=rebate,
        payout=payout,
        extra1=extra1,
        dividend_yield=dividend_yield,
        cap=cap,
        params=params,
    )


def calculate_exotic_payoff_at_expiry(spot: float, position: dict) -> float:
    """Calculate per-share payoff at expiry for an exotic leg.

    Args:
        spot: Spot price at expiry.
        position: Position dict with instrument_class and exotic params.

    Returns:
        Payoff per share (before position sign / quantity scaling).
    """
    exotic_type = position.get("instrument_class", "vanilla")
    is_call = position["option_type"] == "call"
    strike = position["strike"]
    barrier = position.get("barrier", 0.0)
    is_up = position.get("is_up", True)
    is_knock_in = position.get("is_knock_in", False)
    payout_amount = position.get("payout", 1.0)

    if exotic_type == "digital":
        if is_call:
            return payout_amount if spot > strike else 0.0
        return payout_amount if spot < strike else 0.0

    if exotic_type == "barrier":
        vanilla_payoff = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)

        if is_knock_in:
            # At T→0, knock-in only has value if barrier was breached.
            # This matches the pricing engine's convergence and preserves
            # barrier parity: KI_payoff + KO_payoff = Vanilla_payoff.
            if is_up:
                return vanilla_payoff if spot >= barrier else 0.0
            return vanilla_payoff if spot <= barrier else 0.0
        # Knock-out: option dies if barrier is hit. At expiry, approximate by
        # zeroing payoff in the region past the barrier.
        if is_up:
            return 0.0 if spot >= barrier else vanilla_payoff
        return 0.0 if spot <= barrier else vanilla_payoff

    if exotic_type == "asset_or_nothing":
        if is_call:
            return spot if spot > strike else 0.0
        return spot if spot < strike else 0.0

    if exotic_type == "chooser":
        # At expiry, chooser = max(call payoff, put payoff)
        call_payoff = max(spot - strike, 0.0)
        put_payoff = max(strike - spot, 0.0)
        return max(call_payoff, put_payoff)

    if exotic_type == "power":
        n = position.get("power_n", 2.0)
        s_n = spot**n
        if is_call:
            return max(s_n - strike, 0.0)
        return max(strike - s_n, 0.0)

    if exotic_type == "gap":
        trigger = position.get("gap_trigger", strike)
        if is_call:
            return (spot - strike) if spot > trigger else 0.0
        return (strike - spot) if spot < trigger else 0.0

    if exotic_type == "powered":
        i = int(position.get("power_n", 2))
        base = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        return base**i

    if exotic_type == "capped_power":
        i = position.get("power_n", 2.0)
        cap_amt = _safe_cap(position.get("cap", 0.0), strike, is_call)
        s_i = spot**i
        intrinsic = max(s_i - strike, 0.0) if is_call else max(strike - s_i, 0.0)
        return min(intrinsic, cap_amt)

    if exotic_type == "log_contract":
        return math.log(spot / strike)

    if exotic_type == "log_option":
        return max(math.log(spot / strike), 0.0)

    if exotic_type == "supershare":
        lower = position.get("lower_strike", 0.9 * strike)
        upper = position.get("upper_strike", 1.1 * strike)
        return (spot / lower) if (lower < spot < upper) else 0.0

    if exotic_type == "double_barrier":
        lower = position.get("dbl_lower", 0.8 * strike)
        upper = position.get("dbl_upper", 1.2 * strike)
        vanilla = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        inside = lower < spot < upper
        if position.get("adv_in", False):
            return 0.0 if inside else vanilla
        return vanilla if inside else 0.0

    if exotic_type == "discrete_barrier":
        bar = position.get("adv_barrier", 1.1 * strike)
        is_up = position.get("adv_is_up", True)
        vanilla = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        breached = spot >= bar if is_up else spot <= bar
        if position.get("adv_in", False):
            return vanilla if breached else 0.0
        return 0.0 if breached else vanilla

    if exotic_type == "arithmetic_asian":
        # Drawing an Asian payoff against S_T needs an assumption on the
        # remaining averaging path; convention: it sits at S_T, so
        # A_T = w*SA + (1-w)*S_T and the ITM slope is exactly (1-w).
        w = min(max(float(position.get("avg_elapsed_pct", 0.0)), 0.0), 0.95)
        a_t = w * float(position.get("avg_realized", 0.0)) + (1.0 - w) * spot
        if is_call:
            return max(a_t - strike, 0.0)
        return max(strike - a_t, 0.0)

    # Asian / lookback fixed / time-switch / soft- & partial-time- &
    # binary-barrier: at expiry, path- (or type-) dependent features collapse
    # to the vanilla terminal intrinsic (approximation).
    if is_call:
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def calculate_exotic_payoff_at_expiry_vec(
    spots: np.ndarray, position: dict
) -> np.ndarray:
    """Vectorized terminal exotic payoff over an array of spots.

    Element-for-element identical to :func:`calculate_exotic_payoff_at_expiry`
    (same per-type branches, same strict/loose comparisons), but evaluated once
    over the whole ``spots`` array with NumPy instead of a Python loop — used to
    price exotic legs from a batch of MC terminals / over a spot grid.
    """
    spots = np.asarray(spots, dtype=np.float64)
    exotic_type = position.get("instrument_class", "vanilla")
    is_call = position["option_type"] == "call"
    strike = position["strike"]
    barrier = position.get("barrier", 0.0)
    is_up = position.get("is_up", True)
    is_knock_in = position.get("is_knock_in", False)
    payout_amount = position.get("payout", 1.0)

    if exotic_type == "digital":
        cond = spots > strike if is_call else spots < strike
        return np.where(cond, payout_amount, 0.0)

    if exotic_type == "barrier":
        vanilla = (
            np.maximum(spots - strike, 0.0)
            if is_call
            else np.maximum(strike - spots, 0.0)
        )
        breached = spots >= barrier if is_up else spots <= barrier
        if is_knock_in:
            return np.where(breached, vanilla, 0.0)
        return np.where(breached, 0.0, vanilla)

    if exotic_type == "asset_or_nothing":
        cond = spots > strike if is_call else spots < strike
        return np.where(cond, spots, 0.0)

    if exotic_type == "chooser":
        return np.maximum(
            np.maximum(spots - strike, 0.0), np.maximum(strike - spots, 0.0)
        )

    if exotic_type == "power":
        n = position.get("power_n", 2.0)
        s_n = spots**n
        if is_call:
            return np.maximum(s_n - strike, 0.0)
        return np.maximum(strike - s_n, 0.0)

    if exotic_type == "gap":
        trigger = position.get("gap_trigger", strike)
        if is_call:
            return np.where(spots > trigger, spots - strike, 0.0)
        return np.where(spots < trigger, strike - spots, 0.0)

    if exotic_type == "powered":
        i = int(position.get("power_n", 2))
        base = (
            np.maximum(spots - strike, 0.0)
            if is_call
            else np.maximum(strike - spots, 0.0)
        )
        return base**i

    if exotic_type == "capped_power":
        i = position.get("power_n", 2.0)
        cap_amt = _safe_cap(position.get("cap", 0.0), strike, is_call)
        s_i = spots**i
        intrinsic = (
            np.maximum(s_i - strike, 0.0) if is_call else np.maximum(strike - s_i, 0.0)
        )
        return np.minimum(intrinsic, cap_amt)

    if exotic_type == "log_contract":
        return np.log(spots / strike)

    if exotic_type == "log_option":
        return np.maximum(np.log(spots / strike), 0.0)

    if exotic_type == "supershare":
        lower = position.get("lower_strike", 0.9 * strike)
        upper = position.get("upper_strike", 1.1 * strike)
        return np.where((spots > lower) & (spots < upper), spots / lower, 0.0)

    if exotic_type == "double_barrier":
        lower = position.get("dbl_lower", 0.8 * strike)
        upper = position.get("dbl_upper", 1.2 * strike)
        vanilla = (
            np.maximum(spots - strike, 0.0)
            if is_call
            else np.maximum(strike - spots, 0.0)
        )
        inside = (spots > lower) & (spots < upper)
        if position.get("adv_in", False):
            return np.where(inside, 0.0, vanilla)
        return np.where(inside, vanilla, 0.0)

    if exotic_type == "discrete_barrier":
        bar = position.get("adv_barrier", 1.1 * strike)
        is_up = position.get("adv_is_up", True)
        vanilla = (
            np.maximum(spots - strike, 0.0)
            if is_call
            else np.maximum(strike - spots, 0.0)
        )
        breached = spots >= bar if is_up else spots <= bar
        if position.get("adv_in", False):
            return np.where(breached, vanilla, 0.0)
        return np.where(breached, 0.0, vanilla)

    if exotic_type == "arithmetic_asian":
        w = min(max(float(position.get("avg_elapsed_pct", 0.0)), 0.0), 0.95)
        a_t = w * float(position.get("avg_realized", 0.0)) + (1.0 - w) * spots
        if is_call:
            return np.maximum(a_t - strike, 0.0)
        return np.maximum(strike - a_t, 0.0)

    # Asian / lookback fixed / time-switch / soft- & partial-time- &
    # binary-barrier: terminal intrinsic (approximation).
    if is_call:
        return np.maximum(spots - strike, 0.0)
    return np.maximum(strike - spots, 0.0)


# Reiner-Rubinstein 28 binary-barrier types, parsed into
# (is_down, is_in, is_asset, gate) where gate in {"none","call","put"}.
_BINARY_LABELS = (
    "down_in_cash_athit",
    "up_in_cash_athit",
    "down_in_asset_athit",
    "up_in_asset_athit",
    "down_in_cash_atexp",
    "up_in_cash_atexp",
    "down_in_asset_atexp",
    "up_in_asset_atexp",
    "down_out_cash",
    "up_out_cash",
    "down_out_asset",
    "up_out_asset",
    "down_in_cash_call",
    "up_in_cash_call",
    "down_in_asset_call",
    "up_in_asset_call",
    "down_in_cash_put",
    "up_in_cash_put",
    "down_in_asset_put",
    "up_in_asset_put",
    "down_out_cash_call",
    "up_out_cash_call",
    "down_out_asset_call",
    "up_out_asset_call",
    "down_out_cash_put",
    "up_out_cash_put",
    "down_out_asset_put",
    "up_out_asset_put",
)


def _parse_binary(label: str) -> tuple[bool, bool, bool, str]:
    is_down = "down" in label
    is_in = "_in_" in label
    is_asset = "asset" in label
    gate = (
        "call"
        if label.endswith("call")
        else ("put" if label.endswith("put") else "none")
    )
    return is_down, is_in, is_asset, gate


_BINARY_PARSE = {i + 1: _parse_binary(lbl) for i, lbl in enumerate(_BINARY_LABELS)}


def _binary_conditional_payoff(
    spots: np.ndarray, position: dict, scenario: str
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional terminal payoff for a binary-barrier (1..28), given hit/not-hit.

    Approximates the "at hit" types by their contingent terminal amount (the
    pedagogical point is the hit/not-hit branch + feasibility, not the payment
    timing).
    """
    strike = float(position["strike"])
    h = float(position.get("adv_barrier") or position.get("barrier") or strike)
    cash = float(position.get("cash", 10.0))
    is_down, is_in, is_asset, gate = _BINARY_PARSE[int(position.get("binary_type", 13))]

    amount = spots.astype(np.float64) if is_asset else np.full(spots.shape, cash)
    if gate == "call":
        amount = np.where(spots > strike, amount, 0.0)
    elif gate == "put":
        amount = np.where(spots < strike, amount, 0.0)

    feasible = np.ones(spots.shape, dtype=bool)
    hit = scenario == "touched"
    if is_in:
        payoff = amount if hit else np.zeros(spots.shape)
    else:  # out
        payoff = np.zeros(spots.shape) if hit else amount
    if not hit:
        feasible = spots > h if is_down else spots < h
    return payoff, feasible


def conditional_exotic_payoff_vec(
    spots: np.ndarray, position: dict, scenario: str
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional terminal payoff of a discrete-event path-dependent leg.

    Given a fixed binary path-event ``scenario`` the payoff is a clean function
    of the terminal spot over the FEASIBLE region. Returns ``(payoff, feasible)``
    where ``feasible`` is a boolean mask (False where the scenario cannot occur
    for that terminal spot, e.g. an up-and-out that is "not touched" cannot end
    above the barrier).
    """
    spots = np.asarray(spots, dtype=np.float64)
    inst = position.get("instrument_class", "vanilla")
    is_call = position["option_type"] == "call"
    strike = float(position["strike"])
    vanilla = (
        np.maximum(spots - strike, 0.0) if is_call else np.maximum(strike - spots, 0.0)
    )
    feasible = np.ones(spots.shape, dtype=bool)

    if inst in ("barrier", "discrete_barrier"):
        # Resolve direction / knock by instrument class, NOT by key presence: a
        # discrete-barrier leg stores its real flags under the adv_* keys, while
        # the basic ``is_up`` / ``is_knock_in`` keys are ALWAYS written (with
        # their True/False defaults) by the leg editor — so a
        # ``get(basic, adv_fallback)`` would silently keep the wrong basic
        # default. This mirrors the terminal-payoff resolution above.
        if inst == "discrete_barrier":
            h = float(position.get("adv_barrier") or strike)
            is_up = bool(position.get("adv_is_up", True))
            is_ki = bool(position.get("adv_in", False))
        else:
            h = float(position.get("barrier") or strike)
            is_up = bool(position.get("is_up", True))
            is_ki = bool(position.get("is_knock_in", False))
        rebate = float(position.get("rebate", 0.0))
        live = spots < h if is_up else spots > h
        if scenario == "touched":
            payoff = vanilla if is_ki else np.full(spots.shape, rebate)
        else:
            payoff = np.zeros(spots.shape) if is_ki else vanilla
            feasible = live
        return payoff, feasible

    if inst == "double_barrier":
        lo = float(position.get("dbl_lower", 0.8 * strike))
        up = float(position.get("dbl_upper", 1.2 * strike))
        is_ki = bool(position.get("adv_in", False))
        rebate = float(position.get("rebate", 0.0))
        inside = (spots > lo) & (spots < up)
        if scenario == "touched":
            payoff = vanilla if is_ki else np.full(spots.shape, rebate)
        else:
            payoff = np.zeros(spots.shape) if is_ki else vanilla
            feasible = inside
        return payoff, feasible

    if inst == "partial_barrier":
        payoff = np.zeros(spots.shape) if scenario == "touched" else vanilla
        return payoff, feasible

    if inst == "chooser":
        payoff = (
            np.maximum(strike - spots, 0.0)
            if scenario == "put"
            else np.maximum(spots - strike, 0.0)
        )
        return payoff, feasible

    if inst == "binary_barrier":
        return _binary_conditional_payoff(spots, position, scenario)

    # Not a discrete-event family: terminal intrinsic, fully feasible.
    return vanilla, feasible


def payoff_jump_points(position: dict) -> list[float]:
    """Spot levels where the terminal payoff of ``position`` is discontinuous.

    Used to draw true breaks (double point + NaN separator) instead of a fake
    diagonal ramp between the two sides of a jump. Families whose terminal
    payoff is continuous return an empty list.
    """
    inst = position.get("instrument_class", "vanilla")
    strike = float(position["strike"])

    if inst in ("digital", "asset_or_nothing"):
        return [strike]
    if inst == "gap":
        trigger = float(position.get("gap_trigger", strike))
        # Jump size at the trigger is |trigger - strike|; equal strikes make
        # the payoff continuous there.
        return [trigger] if trigger != strike else []
    if inst == "supershare":
        return [
            float(position.get("lower_strike", 0.9 * strike)),
            float(position.get("upper_strike", 1.1 * strike)),
        ]
    if inst == "barrier":
        return [float(position.get("barrier") or strike)]
    if inst == "double_barrier":
        return [
            float(position.get("dbl_lower", 0.8 * strike)),
            float(position.get("dbl_upper", 1.2 * strike)),
        ]
    if inst == "discrete_barrier":
        return [float(position.get("adv_barrier") or strike)]
    if inst == "binary_barrier":
        # Only the call/put-gated Reiner-Rubinstein types jump at the strike;
        # plain cash/asset types have no terminal discontinuity.
        _, _, _, gate = _BINARY_PARSE[int(position.get("binary_type", 13))]
        return [strike] if gate != "none" else []
    return []


def payoff_curve_with_gaps(
    spot_range: np.ndarray,
    position: dict,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    jumps: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``payoff_fn`` on ``spot_range``, breaking the curve at jumps.

    For each jump ``j`` strictly inside the grid, the returned arrays carry
    the two one-sided limits ``(j - eps, f(j-))`` / ``(j + eps, f(j+))`` with a
    ``(j, NaN)`` separator between them (any exact on-jump grid sample is
    replaced by the separator). Plotted with ``connectgaps=False`` this shows
    a clean break instead of a connecting segment.
    """
    spots = np.asarray(spot_range, dtype=np.float64)
    if jumps is None:
        jumps = payoff_jump_points(position)
    lo, hi = float(spots.min()), float(spots.max())
    inner_jumps = sorted({float(j) for j in jumps if lo < j < hi})
    if not inner_jumps:
        return spots, payoff_fn(spots)

    eps = 1e-9 * max(abs(lo), abs(hi), 1.0)
    sided = [v for j in inner_jumps for v in (j - eps, j + eps)]
    x = np.unique(np.concatenate([spots, np.asarray(sided)]))
    # Drop exact on-jump samples: their one-sided value would sit visually on
    # the break the NaN separator is about to open.
    x = x[~np.isin(x, inner_jumps)]
    y = payoff_fn(x)
    for j in inner_jumps:
        idx = int(np.searchsorted(x, j))
        x = np.insert(x, idx, j)
        y = np.insert(y, idx, np.nan)
    return x, y


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
    extra1: float = 0.0,
    ref_spot: float = 0.0,
    dividend_yield: float = 0.0,
    params: dict | None = None,
) -> dict:
    """Calculate exotic Greeks over a range of spot prices.

    Returns dict with keys: price, delta, gamma, vega, theta, rho (each np.ndarray).
    Native kernel families go through the parallel Numba surface; the
    registry-priced (Haug advanced) families use the per-cell backend path —
    this used to raise a bare ``KeyError`` and blank the whole Exotic tab.
    """
    if exotic_type not in EXOTIC_TYPE_MAP:
        matrix = calculate_exotic_greeks_curve(
            spot_range,
            strike,
            maturity,
            rate,
            sigma,
            1 if is_call else 0,
            exotic_type,
            barrier=barrier,
            is_up=is_up,
            is_knock_in=is_knock_in,
            rebate=rebate,
            payout=payout,
            extra1=extra1,
            dividend_yield=dividend_yield,
            params=params,
        )
        return {
            "price": matrix[:, 0],
            "delta": matrix[:, 1],
            "gamma": matrix[:, 2],
            "vega": matrix[:, 3],
            "theta": matrix[:, 4],
            "rho": matrix[:, 5],
        }

    opt_type = EXOTIC_TYPE_MAP[exotic_type]
    H = barrier if exotic_type == "barrier" else 0.0
    # For lookback_floating, pass ref_spot so the surface tracks extremes properly
    m_min = ref_spot if exotic_type == "lookback_floating" and ref_spot > 0 else 0.0
    m_max = ref_spot if exotic_type == "lookback_floating" and ref_spot > 0 else 0.0
    matrix = exotic_greeks_surface(
        opt_type,
        spot_range,
        strike,
        maturity,
        rate,
        dividend_yield,
        sigma,
        is_call,
        H,
        m_min,
        m_max,
        is_knock_in,
        is_up,
        rebate,
        payout,
        extra1,
    )
    return {
        "price": matrix[:, 0],
        "delta": matrix[:, 1],
        "gamma": matrix[:, 2],
        "vega": matrix[:, 3],
        "theta": matrix[:, 4],
        "rho": matrix[:, 5],
    }


def calculate_vanilla_greeks_surface(
    spot_range: np.ndarray,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    is_call: bool,
    dividend_yield: float = 0.0,
) -> dict:
    """Calculate vanilla BS Greeks over a range of spot prices.

    Returns dict with keys: price, delta, gamma, vega, theta, rho (each np.ndarray).
    """
    opt_type_int = 1 if is_call else 0
    matrix = _calculate_greeks_vectorized_numba(
        spot_range,
        strike,
        maturity,
        rate,
        sigma,
        opt_type_int,
        dividend_yield,
    )
    return {
        "price": matrix[:, 0],
        "delta": matrix[:, 1],
        "gamma": matrix[:, 2],
        "vega": matrix[:, 3],
        "theta": matrix[:, 4],
        "rho": matrix[:, 5],
    }


def calculate_barrier_parity_surface(
    spot_range: np.ndarray,
    strike: float,
    barrier: float,
    maturity: float,
    rate: float,
    sigma: float,
    is_call: bool,
    is_up: bool,
    dividend_yield: float = 0.0,
) -> dict:
    """Calculate knock-in, knock-out, and vanilla prices for barrier parity chart.

    Returns dict with keys: knock_in, knock_out, vanilla (each np.ndarray).
    """
    ki = exotic_price_surface(
        BARRIER,
        spot_range,
        strike,
        maturity,
        rate,
        dividend_yield,
        sigma,
        is_call,
        barrier,
        0.0,
        0.0,
        True,
        is_up,
        0.0,
        1.0,
    )
    ko = exotic_price_surface(
        BARRIER,
        spot_range,
        strike,
        maturity,
        rate,
        dividend_yield,
        sigma,
        is_call,
        barrier,
        0.0,
        0.0,
        False,
        is_up,
        0.0,
        1.0,
    )
    n = len(spot_range)
    van = np.empty(n)
    for i in range(n):
        van[i] = _bs_vanilla_price(
            spot_range[i], strike, maturity, rate, dividend_yield, sigma, is_call
        )

    return {"knock_in": ki, "knock_out": ko, "vanilla": van}


def calculate_digital_parity_surface(
    spot_range: np.ndarray,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    payout: float,
    dividend_yield: float = 0.0,
) -> dict:
    """Calculate digital call + put parity surface.

    Returns dict with keys: digital_call, digital_put, discount_factor.
    """
    dc = exotic_price_surface(
        DIGITAL,
        spot_range,
        strike,
        maturity,
        rate,
        dividend_yield,
        sigma,
        True,
        0.0,
        0.0,
        0.0,
        False,
        True,
        0.0,
        payout,
    )
    dp = exotic_price_surface(
        DIGITAL,
        spot_range,
        strike,
        maturity,
        rate,
        dividend_yield,
        sigma,
        False,
        0.0,
        0.0,
        0.0,
        False,
        True,
        0.0,
        payout,
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
    extra1: float = 0.0,
    dividend_yield: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate exotic and vanilla prices over a parameter sweep.

    param_name: 'volatility' or 'maturity'
    Returns (exotic_prices, vanilla_prices).
    """
    opt_type = EXOTIC_TYPE_MAP[exotic_type]
    H = barrier if exotic_type == "barrier" else 0.0
    M_min, M_max = _lookback_defaults(exotic_type, spot)
    param_is_vol = param_name == "volatility"

    return exotic_price_param_sweep(
        opt_type,
        param_range,
        param_is_vol,
        spot,
        strike,
        maturity,
        rate,
        dividend_yield,
        sigma,
        is_call,
        H,
        M_min,
        M_max,
        is_knock_in,
        is_up,
        rebate,
        payout,
        extra1,
    )
