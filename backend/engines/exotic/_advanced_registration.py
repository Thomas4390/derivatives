"""
Registration of the advanced (Haug-catalog) exotic pricers.

Imported for its side effects by ``backend.engines.exotic.__init__`` so that the
``EXOTIC_PRICER_REGISTRY`` is populated before ``ExoticAnalyticEngine`` is used.
Each pricer contributes one adapter (instrument -> ``f(S, sigma, T, r)`` closure)
and one ``register(PricerSpec(...))`` call. Adding a pricer is purely additive
here -- no engine code changes (Open-Closed).

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import cast

from backend.core.interfaces import Instrument, Model
from backend.core.market import MarketEnvironment
from backend.engines.exotic._registry import PriceFn, PricerSpec, register
from backend.engines.exotic.asian_arithmetic import asian_arithmetic_tw_price
from backend.engines.exotic.binary_barrier import binary_barrier_price
from backend.engines.exotic.discrete_barrier import discrete_barrier_price
from backend.engines.exotic.complex_chooser import complex_chooser_price
from backend.engines.exotic.compound import compound_option_price
from backend.engines.exotic.double_barrier import double_barrier_price
from backend.engines.exotic.extendible import extendible_price
from backend.engines.exotic.extreme_spread import extreme_spread_price
from backend.engines.exotic.forward_start import forward_start_price
from backend.engines.exotic.log_contract import log_contract_price, log_option_price
from backend.engines.exotic.lookback_partial import partial_float_lookback_price
from backend.engines.exotic.power_ext import capped_power_price, powered_price
from backend.engines.exotic.lookback_partial_fixed import partial_fixed_lookback_price
from backend.engines.exotic.partial_barrier import (
    PTB_CDOA,
    PTB_CDOB2,
    PTB_COB1,
    PTB_CUOA,
    PTB_CUOB2,
    PTB_PDOA,
    PTB_PDOB2,
    PTB_POB1,
    PTB_PUOA,
    PTB_PUOB2,
    partial_time_barrier_price,
)
from backend.engines.exotic.soft_barrier import soft_barrier_price
from backend.engines.exotic.supershare import supershare_price
from backend.engines.exotic.time_switch import time_switch_price
from backend.instruments.exotic_advanced import (
    ArithmeticAsianOption,
    BinaryBarrierOption,
    ComplexChooserOption,
    CompoundOption,
    DiscreteBarrierOption,
    DoubleBarrierOption,
    ExtendibleOption,
    ExtremeSpreadOption,
    ForwardStartOption,
    CappedPowerOption,
    LogContract,
    LogOption,
    PartialFixedLookbackOption,
    PartialFloatLookbackOption,
    PartialTimeBarrierOption,
    PoweredOption,
    SoftBarrierOption,
    SupershareOption,
    TimeSwitchOption,
)

# (barrier_type, is_call) -> integer PTB_* code.
_PTB_CODE: dict[tuple[str, bool], int] = {
    ("down_out_A", True): PTB_CDOA,
    ("down_out_A", False): PTB_PDOA,
    ("up_out_A", True): PTB_CUOA,
    ("up_out_A", False): PTB_PUOA,
    ("out_B1", True): PTB_COB1,
    ("out_B1", False): PTB_POB1,
    ("down_out_B2", True): PTB_CDOB2,
    ("down_out_B2", False): PTB_PDOB2,
    ("up_out_B2", True): PTB_CUOB2,
    ("up_out_B2", False): PTB_PUOB2,
}

# =============================================================================
# Double-barrier (Ikeda-Kunitomo 1992)
# =============================================================================


def _double_barrier_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    db = cast(DoubleBarrierOption, inst)
    k = db.strike
    lower = db.lower
    upper = db.upper
    is_call = db.is_call
    is_knock_in = db.is_knock_in
    delta1, delta2 = db.curvature
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        return float(
            double_barrier_price(
                s, k, lower, upper, t, r, q, sigma, is_call, is_knock_in, delta1, delta2
            )
        )

    return f


register(PricerSpec(DoubleBarrierOption, _double_barrier_adapter, "double-barrier"))


# =============================================================================
# Discrete-barrier (Broadie-Glasserman-Kou 1997)
# =============================================================================


def _discrete_barrier_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    dbo = cast(DiscreteBarrierOption, inst)
    k = dbo.strike
    barrier = dbo.barrier
    is_call = dbo.is_call
    is_up = dbo.is_up
    is_knock_in = dbo.is_knock_in
    m = dbo.monitoring_points
    rebate = dbo.rebate
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        return float(
            discrete_barrier_price(
                s, k, barrier, t, r, q, sigma, is_call, is_knock_in, is_up, m, rebate
            )
        )

    return f


register(
    PricerSpec(DiscreteBarrierOption, _discrete_barrier_adapter, "discrete-barrier")
)


# =============================================================================
# Soft-barrier (Hart-Ross 1994)
# =============================================================================


def _soft_barrier_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    sbo = cast(SoftBarrierOption, inst)
    k = sbo.strike
    lower = sbo.lower
    upper = sbo.upper
    is_call = sbo.is_call
    is_knock_in = sbo.is_knock_in
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        return float(
            soft_barrier_price(s, k, lower, upper, t, r, q, sigma, is_call, is_knock_in)
        )

    return f


register(PricerSpec(SoftBarrierOption, _soft_barrier_adapter, "soft-barrier"))


# =============================================================================
# Partial-time barrier (Heynen-Kat 1994)
# =============================================================================


def _partial_barrier_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    pbo = cast(PartialTimeBarrierOption, inst)
    k = pbo.strike
    barrier = pbo.barrier
    t1 = pbo.t1
    code = _PTB_CODE[(pbo.barrier_type, pbo.is_call)]
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        return float(
            partial_time_barrier_price(s, k, barrier, t1, t, r, q, sigma, code)
        )

    return f


register(
    PricerSpec(
        PartialTimeBarrierOption, _partial_barrier_adapter, "partial-time-barrier"
    )
)


# =============================================================================
# Binary-barrier (Reiner-Rubinstein 1991, 28 types)
# =============================================================================


def _binary_barrier_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    bb = cast(BinaryBarrierOption, inst)
    x = bb.strike
    barrier = bb.barrier
    cash = bb.cash
    binary_type = bb.binary_type
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        return float(
            binary_barrier_price(s, x, barrier, cash, t, r, q, sigma, binary_type)
        )

    return f


register(PricerSpec(BinaryBarrierOption, _binary_barrier_adapter, "binary-barrier"))


# =============================================================================
# Partial-time floating-strike lookback (Heynen-Kat 1994c)
# =============================================================================


def _partial_float_lookback_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    plb = cast(PartialFloatLookbackOption, inst)
    t1 = plb.t1
    is_call = plb.is_call
    weight = plb.weight
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        # Fresh option: the running extreme equals the (bumped) spot, so spot,
        # min and max move together -- the economically correct fresh-lookback
        # Greeks (mirrors the legacy floating-lookback bump convention).
        return float(
            partial_float_lookback_price(s, s, s, t1, t, r, q, sigma, is_call, weight)
        )

    return f


register(
    PricerSpec(
        PartialFloatLookbackOption,
        _partial_float_lookback_adapter,
        "partial-float-lookback",
    )
)


# =============================================================================
# Partial-time fixed-strike lookback (Heynen-Kat 1994c)
# =============================================================================


def _partial_fixed_lookback_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    pflb = cast(PartialFixedLookbackOption, inst)
    x = pflb.strike
    t1 = pflb.t1
    is_call = pflb.is_call
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        return float(partial_fixed_lookback_price(s, x, t1, t, r, q, sigma, is_call))

    return f


register(
    PricerSpec(
        PartialFixedLookbackOption,
        _partial_fixed_lookback_adapter,
        "partial-fixed-lookback",
    )
)


# =============================================================================
# Extreme-spread (Bermin 1996b)
# =============================================================================


def _extreme_spread_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    eso = cast(ExtremeSpreadOption, inst)
    t1 = eso.t1
    is_call = eso.is_call
    is_reverse = eso.is_reverse
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        # Fresh option: the carried first-period extreme equals the (bumped)
        # spot, so the relevant baseline tracks spot and the Greeks stay
        # economically sensible (mirrors the floating-lookback convention).
        return float(
            extreme_spread_price(s, s, s, t1, t, r, q, sigma, is_call, is_reverse)
        )

    return f


register(PricerSpec(ExtremeSpreadOption, _extreme_spread_adapter, "extreme-spread"))


# =============================================================================
# Complex chooser (Rubinstein 1991)
# =============================================================================


def _complex_chooser_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    cco = cast(ComplexChooserOption, inst)
    kc = cco.call_strike
    kp = cco.put_strike
    tc0 = cco.call_maturity
    tp0 = cco.put_maturity
    tch0 = cco.choice_time
    t_ref = max(tc0, tp0)
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        # The registry drives a single time axis t = instrument.maturity (= the
        # longer leg). A theta bump shifts calendar time, so shift ALL three
        # contract times (Tc, Tp, choice) by the same delta; spot/vol/rate bumps
        # leave the times untouched (dt = 0). The leg windows Tc - choice and
        # Tp - choice are translation-invariant under this shift, so only the
        # absolute choice date can cross zero -- and only for a one-day theta
        # bump on a sub-one-day choice date. Fall back to the unshifted times
        # there (a benign ~0 theta) rather than letting the kernel see a
        # non-positive choice date and return a spurious 0.
        dt = t - t_ref
        if tch0 + dt <= 0.0:
            dt = 0.0
        return float(
            complex_chooser_price(s, kc, kp, tc0 + dt, tp0 + dt, tch0 + dt, r, q, sigma)
        )

    return f


register(PricerSpec(ComplexChooserOption, _complex_chooser_adapter, "complex-chooser"))


# =============================================================================
# Compound option / option on option (Geske 1979)
# =============================================================================


def _compound_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    co = cast(CompoundOption, inst)
    k1 = co.strike1
    k2 = co.strike2
    t1_0 = co.t1
    t2_0 = co.maturity
    is_call_on = co.is_call_on
    is_call_under = co.is_call_underlying
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        # Single registry time axis t = instrument.maturity (= T2). A theta bump
        # is calendar time passing, so shift both t1 and T2 by the same delta
        # (the underlying option's remaining life T2 - t1 is a fixed contract
        # spec and stays invariant). Only the absolute t1 can cross zero, and
        # only for a one-day bump on a sub-one-day compound expiry -> fall back
        # to the unshifted times (a benign ~0 theta) there.
        dt = t - t2_0
        if t1_0 + dt <= 0.0:
            dt = 0.0
        return float(
            compound_option_price(
                s, k1, k2, t1_0 + dt, t2_0 + dt, r, q, sigma, is_call_on, is_call_under
            )
        )

    return f


register(PricerSpec(CompoundOption, _compound_adapter, "compound"))


# =============================================================================
# Extendible-maturity option (Longstaff 1990)
# =============================================================================


def _extendible_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    eo = cast(ExtendibleOption, inst)
    x1 = eo.strike1
    x2 = eo.strike2
    t1_0 = eo.t1
    t2_0 = eo.maturity
    a = eo.extension_fee
    is_call = eo.is_call
    is_holder = eo.holder_extendible
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        # Single registry time axis t = instrument.maturity (= T2); shift both t1
        # and T2 by the same delta under a theta bump (the initial-to-extended
        # gap T2 - t1 is a fixed contract spec), with a sub-one-day-t1 guard.
        dt = t - t2_0
        if t1_0 + dt <= 0.0:
            dt = 0.0
        return float(
            extendible_price(
                s, x1, x2, t1_0 + dt, t2_0 + dt, r, q, sigma, a, is_call, is_holder
            )
        )

    return f


register(PricerSpec(ExtendibleOption, _extendible_adapter, "extendible"))


# =============================================================================
# Forward-start option (Rubinstein 1990)
# =============================================================================


def _forward_start_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    fso = cast(ForwardStartOption, inst)
    alpha = fso.alpha
    t1_0 = fso.grant_time
    t_ref = fso.maturity
    is_call = fso.is_call
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        # Single registry time axis t = instrument.maturity (= T); a theta bump
        # shifts calendar time, so move both the grant date and T by dt (the
        # pre-grant t1 and post-grant T - t1 are fixed contract specs), with a
        # sub-one-day-grant guard.
        dt = t - t_ref
        if t1_0 + dt <= 0.0:
            dt = 0.0
        # Use t_ref + dt (not the raw t) so the sub-one-day-grant guard reverts
        # BOTH times to unshifted -- keeping T - t1 invariant, like the sibling
        # adapters. In the normal path t_ref + dt == t.
        return float(
            forward_start_price(s, alpha, t1_0 + dt, t_ref + dt, r, q, sigma, is_call)
        )

    return f


register(PricerSpec(ForwardStartOption, _forward_start_adapter, "forward-start"))


# =============================================================================
# Log contract (Neuberger 1994) / log(S) contract (Haug 4.14/4.15)
# =============================================================================


def _log_contract_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    lc = cast(LogContract, inst)
    strike = lc.strike
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        return float(log_contract_price(s, strike, t, r, q, sigma))

    return f


register(PricerSpec(LogContract, _log_contract_adapter, "log-contract"))


# =============================================================================
# Log option (Wilmott 2000, Haug 4.16)
# =============================================================================


def _log_option_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    lo = cast(LogOption, inst)
    strike = lo.strike
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        return float(log_option_price(s, strike, t, r, q, sigma))

    return f


register(PricerSpec(LogOption, _log_option_adapter, "log-option"))


# =============================================================================
# Discrete time-switch option (Pechtl 1995, Haug 4.11)
# =============================================================================


def _time_switch_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    tso = cast(TimeSwitchOption, inst)
    strike = tso.strike
    accrual = tso.accrual
    step = tso.step
    m = tso.units_filled
    is_call = tso.is_call
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        # Single registry time axis t = maturity. The monitoring step `step` is a
        # fixed contract spec, so a theta bump shrinks t (and hence the number of
        # monitored instants n = round(t/step)) -- the correct discrete semantics.
        return float(
            time_switch_price(s, strike, accrual, t, r, q, sigma, step, m, is_call)
        )

    return f


register(PricerSpec(TimeSwitchOption, _time_switch_adapter, "time-switch"))


# =============================================================================
# Supershare option (Hakansson 1976, Haug 4.19.4)
# =============================================================================


def _supershare_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    sso = cast(SupershareOption, inst)
    x_l = sso.lower_strike
    x_h = sso.upper_strike
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        return float(supershare_price(s, x_l, x_h, t, r, q, sigma))

    return f


register(PricerSpec(SupershareOption, _supershare_adapter, "supershare"))


# =============================================================================
# Powered option (Esser 2003, Haug 4.4.4)
# =============================================================================


def _powered_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    po = cast(PoweredOption, inst)
    x = po.strike
    i = po.power
    is_call = po.is_call
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        return float(powered_price(s, x, t, r, q, sigma, i, is_call))

    return f


register(PricerSpec(PoweredOption, _powered_adapter, "powered"))


# =============================================================================
# Capped power option (Esser 2003, Haug 4.4.3)
# =============================================================================


def _capped_power_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    cpo = cast(CappedPowerOption, inst)
    x = cpo.strike
    i = cpo.power
    cap = cpo.cap
    is_call = cpo.is_call
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        return float(capped_power_price(s, x, t, r, q, sigma, i, cap, is_call))

    return f


register(PricerSpec(CappedPowerOption, _capped_power_adapter, "capped-power"))


# =============================================================================
# Arithmetic average-rate option (Turnbull-Wakeman 1991, Haug 4.20.2)
# =============================================================================


def _arithmetic_asian_adapter(
    inst: Instrument, model: Model, market: MarketEnvironment
) -> PriceFn:
    aao = cast(ArithmeticAsianOption, inst)
    x = aao.strike
    t2 = aao.average_period
    sa = aao.realized_average
    is_call = aao.is_call
    q = market.dividend_yield

    def f(s: float, sigma: float, t: float, r: float) -> float:
        # A fresh option's realized average is taken as spot (a just-started
        # average is ~ spot); a seasoned option carries its own SA. The averaging
        # window t2 is a fixed contract spec, so a theta bump shrinks t (and may
        # cross into the seasoned regime) -- the correct semantics.
        sa_used = sa if sa > 0.0 else s
        return float(
            asian_arithmetic_tw_price(s, sa_used, x, t, t2, r, q, sigma, is_call)
        )

    return f


register(
    PricerSpec(ArithmeticAsianOption, _arithmetic_asian_adapter, "arithmetic-asian")
)
