"""
Monte-Carlo batch payoffs for the path-dependent ADVANCED (Haug) exotics.

These extend the model-dependent ``ExoticMonteCarloEngine`` to the Haug-catalog
path-dependent instruments (double / discrete / soft / partial-time barriers,
partial lookbacks, extreme-spread, arithmetic Asian, discrete time-switch and
forward-start). Each adapter maps the simulated ``(n_paths, n_steps + 1)`` path
matrix (column 0 = S0, uniform grid over ``[0, T]``) to one payoff per path and
self-registers in ``EXOTIC_MC_PAYOFF_REGISTRY``.

Implementation notes
--------------------
- Pure NumPy (vectorised reductions over the path matrix). Sub-window and
  two-period logic is far clearer -- and less bug-prone -- here than in Numba,
  and these are not the vanilla hot path.
- Continuously-monitored exotics (double/soft/partial barriers, partial
  lookbacks, extreme-spread) are monitored *discretely* on the simulation grid:
  the result carries a discrete-monitoring bias versus the continuous closed
  form (shrinks as ``n_steps`` grows). Use a fine ``steps_per_year`` for these.
- Calendar instants ``t1`` are mapped to a grid column via :func:`_window_index`.

Deferred (NOT registered): compound, complex-chooser and extendible options.
Their value depends on a *continuation option value* at an intermediate date,
which under a general (non-GBM) model needs nested simulation -- out of scope.
They remain GBM closed-form only and raise for non-GBM via the engine dispatch.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import cast

import numpy as np

from backend.core.interfaces import Instrument
from backend.engines.exotic._mc_payoff_registry import register_mc_payoff
from backend.instruments.exotic_advanced import (
    ArithmeticAsianOption,
    DiscreteBarrierOption,
    DoubleBarrierOption,
    ExtremeSpreadOption,
    ForwardStartOption,
    PartialFixedLookbackOption,
    PartialFloatLookbackOption,
    PartialTimeBarrierOption,
    SoftBarrierOption,
    TimeSwitchOption,
)
from backend.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _arr(x: object) -> np.ndarray:
    """Pin a (possibly Any-typed) NumPy reduction to a float64 ndarray."""
    return np.asarray(x, dtype=np.float64)


def _window_index(t: float, maturity: float, n_cols: int) -> int:
    """Grid column closest to calendar time ``t`` on the uniform ``[0, T]`` grid.

    Clamped to ``[1, n_cols - 1]`` so a window always spans at least one step.
    """
    idx = int(round(t / maturity * (n_cols - 1)))
    return max(1, min(n_cols - 1, idx))


def _vanilla(s_t: np.ndarray, strike: float, is_call: bool) -> np.ndarray:
    """Terminal vanilla payoff against ``strike``."""
    return _arr(
        np.maximum(s_t - strike, 0.0) if is_call else np.maximum(strike - s_t, 0.0)
    )


# ---------------------------------------------------------------------------
# Double barrier (Ikeda-Kunitomo) -- knock-out/in, lower & upper, full life
# ---------------------------------------------------------------------------


def _double_barrier_payoff(paths: np.ndarray, inst: Instrument) -> np.ndarray:
    db = cast(DoubleBarrierOption, inst)
    # Curvature (exponential boundaries) is a closed-form feature; the MC uses
    # flat barriers, which is exact for the default curvature (0, 0).
    breached = (paths.min(axis=1) <= db.lower) | (paths.max(axis=1) >= db.upper)
    vanilla = _vanilla(paths[:, -1], db.strike, db.is_call)
    if db.is_knock_in:
        return _arr(np.where(breached, vanilla, 0.0))
    return _arr(np.where(breached, 0.0, vanilla))


register_mc_payoff(DoubleBarrierOption, _double_barrier_payoff)


# ---------------------------------------------------------------------------
# Discrete barrier (m equally-spaced monitoring dates) -- single barrier
# ---------------------------------------------------------------------------


def _discrete_barrier_payoff(paths: np.ndarray, inst: Instrument) -> np.ndarray:
    dbo = cast(DiscreteBarrierOption, inst)
    n_cols = paths.shape[1]
    m = dbo.monitoring_points
    # Monitoring instants i*T/m, i=1..m, mapped onto the grid (deduplicated).
    cols = np.unique(np.rint(np.arange(1, m + 1) * (n_cols - 1) / m).astype(np.int64))
    cols = cols[cols >= 1]
    monitored = paths[:, cols]
    if dbo.is_up:
        breached = (monitored >= dbo.barrier).any(axis=1)
    else:
        breached = (monitored <= dbo.barrier).any(axis=1)
    vanilla = _vanilla(paths[:, -1], dbo.strike, dbo.is_call)
    rebate = dbo.rebate
    if dbo.is_knock_in:
        return _arr(np.where(breached, vanilla, rebate))
    return _arr(np.where(breached, rebate, vanilla))


register_mc_payoff(DiscreteBarrierOption, _discrete_barrier_payoff)


# ---------------------------------------------------------------------------
# Soft barrier (Hart-Ross) -- probabilistic survival across a [lower, upper] band
# ---------------------------------------------------------------------------


def _soft_barrier_payoff(paths: np.ndarray, inst: Instrument) -> np.ndarray:
    sb = cast(SoftBarrierOption, inst)
    lower, upper = sb.lower, sb.upper
    width = upper - lower
    vanilla = _vanilla(paths[:, -1], sb.strike, sb.is_call)
    if sb.is_call:  # soft DOWN band: survival rises as the running min clears U
        extreme = paths.min(axis=1)
        survival = (
            np.clip((extreme - lower) / width, 0.0, 1.0)
            if width > 0
            else (extreme > lower)
        )
    else:  # soft UP band: survival rises as the running max stays below L
        extreme = paths.max(axis=1)
        survival = (
            np.clip((upper - extreme) / width, 0.0, 1.0)
            if width > 0
            else (extreme < upper)
        )
    weight = _arr(survival) if not sb.is_knock_in else (1.0 - _arr(survival))
    return _arr(weight * vanilla)


register_mc_payoff(SoftBarrierOption, _soft_barrier_payoff)


# ---------------------------------------------------------------------------
# Partial-time barrier (Heynen-Kat) -- barrier live over a sub-window
# ---------------------------------------------------------------------------


def _partial_time_barrier_payoff(paths: np.ndarray, inst: Instrument) -> np.ndarray:
    pb = cast(PartialTimeBarrierOption, inst)
    n_cols = paths.shape[1]
    idx1 = _window_index(pb.t1, pb.maturity, n_cols)
    btype = pb.barrier_type
    if btype.endswith("_A"):  # monitor [0, t1]
        window = paths[:, : idx1 + 1]
    else:  # B1 / B2 -- monitor [t1, T2]
        window = paths[:, idx1:]
    h = pb.barrier
    if btype.startswith("down"):
        knocked = window.min(axis=1) <= h
    elif btype.startswith("up"):
        knocked = window.max(axis=1) >= h
    else:  # out_B1: knock on any touch of H (H within [min, max] of the window)
        knocked = (window.min(axis=1) <= h) & (window.max(axis=1) >= h)
    vanilla = _vanilla(paths[:, -1], pb.strike, pb.is_call)
    return _arr(np.where(knocked, 0.0, vanilla))


register_mc_payoff(PartialTimeBarrierOption, _partial_time_barrier_payoff)


# ---------------------------------------------------------------------------
# Arithmetic average-rate Asian (Turnbull-Wakeman) -- average over the window
# ---------------------------------------------------------------------------


def _arithmetic_asian_payoff(paths: np.ndarray, inst: Instrument) -> np.ndarray:
    aa = cast(ArithmeticAsianOption, inst)
    # Average over the post-inception grid points (excludes S0, matching the
    # continuous-window TW moments more closely than including the seed point).
    future_avg = paths[:, 1:].mean(axis=1)
    sa, period, mat = aa.realized_average, aa.average_period, aa.maturity
    if sa > 0.0 and period > mat:  # seasoned: blend realized + future averages
        w_past = (period - mat) / period
        average = w_past * sa + (1.0 - w_past) * future_avg
    else:  # fresh: the whole window is simulated
        average = future_avg
    return _vanilla(_arr(average), aa.strike, aa.is_call)


register_mc_payoff(ArithmeticAsianOption, _arithmetic_asian_payoff)


# ---------------------------------------------------------------------------
# Partial-time floating lookback (Heynen-Kat) -- extremes over [0, t1]
# ---------------------------------------------------------------------------


def _partial_float_lookback_payoff(paths: np.ndarray, inst: Instrument) -> np.ndarray:
    pl = cast(PartialFloatLookbackOption, inst)
    n_cols = paths.shape[1]
    idx1 = _window_index(pl.t1, pl.maturity, n_cols)
    window = paths[:, : idx1 + 1]
    s_t = paths[:, -1]
    lam = pl.weight
    if pl.is_call:
        return _arr(np.maximum(s_t - lam * window.min(axis=1), 0.0))
    return _arr(np.maximum(lam * window.max(axis=1) - s_t, 0.0))


register_mc_payoff(PartialFloatLookbackOption, _partial_float_lookback_payoff)


# ---------------------------------------------------------------------------
# Partial-time fixed lookback (Heynen-Kat) -- extremes over [t1, T2]
# ---------------------------------------------------------------------------


def _partial_fixed_lookback_payoff(paths: np.ndarray, inst: Instrument) -> np.ndarray:
    pf = cast(PartialFixedLookbackOption, inst)
    n_cols = paths.shape[1]
    idx1 = _window_index(pf.t1, pf.maturity, n_cols)
    window = paths[:, idx1:]
    if pf.is_call:
        return _arr(np.maximum(window.max(axis=1) - pf.strike, 0.0))
    return _arr(np.maximum(pf.strike - window.min(axis=1), 0.0))


register_mc_payoff(PartialFixedLookbackOption, _partial_fixed_lookback_payoff)


# ---------------------------------------------------------------------------
# Extreme-spread (Bermin) -- spread of two-period extremes
# ---------------------------------------------------------------------------


def _extreme_spread_payoff(paths: np.ndarray, inst: Instrument) -> np.ndarray:
    es = cast(ExtremeSpreadOption, inst)
    n_cols = paths.shape[1]
    idx1 = _window_index(es.t1, es.maturity, n_cols)
    p1 = paths[:, : idx1 + 1]  # period 1 [0, t1] (fresh: seeded by spot at col 0)
    p2 = paths[:, idx1:]  # period 2 [t1, T2]
    smax1, smin1 = p1.max(axis=1), p1.min(axis=1)
    smax2, smin2 = p2.max(axis=1), p2.min(axis=1)
    if not es.is_reverse:
        if es.is_call:
            return _arr(np.maximum(smax2 - smax1, 0.0))
        return _arr(np.maximum(smin1 - smin2, 0.0))
    if es.is_call:
        return _arr(np.maximum(smin2 - smin1, 0.0))
    return _arr(np.maximum(smax1 - smax2, 0.0))


register_mc_payoff(ExtremeSpreadOption, _extreme_spread_payoff)


# ---------------------------------------------------------------------------
# Discrete time-switch (Pechtl) -- accrue A*step at each in-condition instant
# ---------------------------------------------------------------------------


def _time_switch_payoff(paths: np.ndarray, inst: Instrument) -> np.ndarray:
    ts = cast(TimeSwitchOption, inst)
    # Monitored on the simulation grid (excluding t=0). For an exact match to the
    # contract's `step` grid, price with steps_per_year = 1/step so that the grid
    # instants coincide with i*step, i=1..T/step.
    monitored = paths[:, 1:]
    if ts.is_call:
        hits = (monitored > ts.strike).sum(axis=1)
    else:
        hits = (monitored < ts.strike).sum(axis=1)
    total_units = _arr(hits) + ts.units_filled
    return _arr(ts.accrual * ts.step * total_units)


register_mc_payoff(TimeSwitchOption, _time_switch_payoff)


# ---------------------------------------------------------------------------
# Forward-start (Rubinstein) -- strike set to alpha*S(t1), payoff at T
# ---------------------------------------------------------------------------


def _forward_start_payoff(paths: np.ndarray, inst: Instrument) -> np.ndarray:
    fs = cast(ForwardStartOption, inst)
    n_cols = paths.shape[1]
    idx1 = _window_index(fs.grant_time, fs.maturity, n_cols)
    strike = fs.alpha * paths[:, idx1]
    s_t = paths[:, -1]
    if fs.is_call:
        return _arr(np.maximum(s_t - strike, 0.0))
    return _arr(np.maximum(strike - s_t, 0.0))


register_mc_payoff(ForwardStartOption, _forward_start_payoff)
