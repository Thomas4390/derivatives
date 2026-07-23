"""
Monte-Carlo payoff registry for the model-dependent exotic engine.

``ExoticMonteCarloEngine`` simulates price paths under the chosen model and then
turns each instrument into a *batch payoff* ``f(price_paths, instrument) ->
payoffs[n_paths]``. This registry maps ``type(instrument)`` to that batch payoff,
mirroring the Open-Closed dispatch of ``EXOTIC_PRICER_REGISTRY`` (closed-form
adapters) so that adding a new MC-priceable exotic is purely additive -- no
engine code changes -- and the instrument classes stay pure value objects (no
payoff method baked onto each ``@dataclass``).

The closure receives the full ``(n_paths, n_steps + 1)`` path matrix (column 0 is
the initial spot) so path-dependent exotics (barriers, lookbacks, Asians) see the
whole trajectory; terminal-only exotics simply read ``paths[:, -1]``.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import Callable, cast

import numpy as np

from backend.core.interfaces import Instrument
from backend.instruments.options import (
    AsianOption,
    BarrierOption,
    DigitalOption,
    LookbackOption,
)
from backend.math_kernels.payoff_kernels import (
    asian_arithmetic_payoff_batch,
    asian_geometric_payoff_batch,
    barrier_payoff_batch,
    digital_call_payoff_vec,
    digital_put_payoff_vec,
    lookback_fixed_payoff_batch,
    lookback_floating_payoff_batch,
)

# A batch payoff turns the simulated path matrix into one payoff per path.
MCPayoffFn = Callable[[np.ndarray, Instrument], np.ndarray]

EXOTIC_MC_PAYOFF_REGISTRY: dict[type, MCPayoffFn] = {}


def register_mc_payoff(instrument_type: type, fn: MCPayoffFn) -> None:
    """Register (or replace) the batch payoff for an instrument type."""
    EXOTIC_MC_PAYOFF_REGISTRY[instrument_type] = fn


def lookup_mc_payoff(instrument: Instrument) -> MCPayoffFn | None:
    """Return the batch payoff registered for ``type(instrument)``, or ``None``."""
    return EXOTIC_MC_PAYOFF_REGISTRY.get(type(instrument))


# =============================================================================
# Barrier (single, knock-in/out, up/down, call/put) -- path-dependent
# =============================================================================


def _barrier_payoff(paths: np.ndarray, inst: Instrument) -> np.ndarray:
    bo = cast(BarrierOption, inst)
    # np.asarray pins the Numba kernel's untyped (Any) return to ndarray (no copy).
    return np.asarray(
        barrier_payoff_batch(
            paths,
            bo.strike,
            bo.barrier,
            bo.is_call,
            bo.is_up,
            bo.is_knock_in,
            bo.rebate,
        )
    )


register_mc_payoff(BarrierOption, _barrier_payoff)


# =============================================================================
# Asian (arithmetic OR geometric average) -- path-dependent
# =============================================================================


def _asian_payoff(paths: np.ndarray, inst: Instrument) -> np.ndarray:
    ao = cast(AsianOption, inst)
    if ao.average_type == "geometric":
        return np.asarray(asian_geometric_payoff_batch(paths, ao.strike, ao.is_call))
    if ao.average_type == "arithmetic":
        return np.asarray(asian_arithmetic_payoff_batch(paths, ao.strike, ao.is_call))
    raise ValueError(
        f"ExoticMonteCarloEngine has no payoff for Asian average_type "
        f"'{ao.average_type}' (supported: geometric, arithmetic)"
    )


register_mc_payoff(AsianOption, _asian_payoff)


# =============================================================================
# Lookback (floating or fixed strike) -- path-dependent
# =============================================================================


def _lookback_payoff(paths: np.ndarray, inst: Instrument) -> np.ndarray:
    lo = cast(LookbackOption, inst)
    if lo.lookback_type == "fixed":
        return np.asarray(lookback_fixed_payoff_batch(paths, lo.strike, lo.is_call))
    return np.asarray(lookback_floating_payoff_batch(paths, lo.is_call))


register_mc_payoff(LookbackOption, _lookback_payoff)


# =============================================================================
# Digital / binary (cash-or-nothing) -- terminal only
# =============================================================================


def _digital_payoff(paths: np.ndarray, inst: Instrument) -> np.ndarray:
    do = cast(DigitalOption, inst)
    terminal = np.ascontiguousarray(paths[:, -1])
    if do.is_call:
        return np.asarray(digital_call_payoff_vec(terminal, do.strike, do.payout))
    return np.asarray(digital_put_payoff_vec(terminal, do.strike, do.payout))


register_mc_payoff(DigitalOption, _digital_payoff)


# Register the path-dependent advanced (Haug-catalog) exotic MC payoffs for
# their side effects. Imported at the bottom -- after `register_mc_payoff` is
# defined -- so the advanced module can import it back without a cycle.
from backend.engines.exotic import _mc_payoff_advanced  # noqa: E402, F401
from backend.engines.exotic import _mc_payoff_terminal  # noqa: E402, F401
