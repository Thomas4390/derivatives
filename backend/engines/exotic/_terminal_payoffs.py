"""
Shared terminal payoffs for the genuinely *terminal* exotics (payoff = g(S_T)).

These exotics depend only on the terminal underlying value, so a single payoff
function ``g(s_t) -> payoff`` serves both pricing routes:
- the Monte-Carlo engine applies ``g`` to the simulated ``price_paths[:, -1]``;
- the Fourier (COS) engine integrates ``g(exp(x))`` against the model-implied
  log-price density.

``TERMINAL_PAYOFF_BUILDERS`` maps ``type(instrument) -> builder``; each builder
closes over the instrument's parameters and returns the vectorised ``g``.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from backend.core.interfaces import Instrument
from backend.instruments.exotic_advanced import (
    CappedPowerOption,
    LogContract,
    LogOption,
    PoweredOption,
    SupershareOption,
)
from backend.instruments.options import AssetOrNothingOption, GapOption

# g(s_t) -> payoff per terminal value; a builder closes over the instrument.
TerminalPayoff = Callable[[np.ndarray], np.ndarray]
PayoffBuilder = Callable[[Instrument], TerminalPayoff]


def _log_contract_g(inst: Instrument) -> TerminalPayoff:
    x = inst.strike  # type: ignore[attr-defined]
    return lambda s: np.asarray(np.log(s / x), dtype=np.float64)


def _log_option_g(inst: Instrument) -> TerminalPayoff:
    x = inst.strike  # type: ignore[attr-defined]
    return lambda s: np.asarray(np.maximum(np.log(s / x), 0.0), dtype=np.float64)


def _powered_g(inst: Instrument) -> TerminalPayoff:
    po = inst  # PoweredOption: max(eta*(S - X), 0)**i
    x, i = po.strike, po.power  # type: ignore[attr-defined]
    eta = 1.0 if po.is_call else -1.0  # type: ignore[attr-defined]
    return lambda s: np.asarray(np.maximum(eta * (s - x), 0.0) ** i, dtype=np.float64)


def _capped_power_g(inst: Instrument) -> TerminalPayoff:
    cp = inst  # CappedPowerOption: min(max(eta*(S**i - X), 0), cap)
    x, i, cap = cp.strike, cp.power, cp.cap  # type: ignore[attr-defined]
    eta = 1.0 if cp.is_call else -1.0  # type: ignore[attr-defined]
    return lambda s: np.asarray(
        np.minimum(np.maximum(eta * (s**i - x), 0.0), cap), dtype=np.float64
    )


def _supershare_g(inst: Instrument) -> TerminalPayoff:
    xl, xh = inst.lower_strike, inst.upper_strike  # type: ignore[attr-defined]
    return lambda s: np.asarray(
        np.where((s > xl) & (s < xh), s / xl, 0.0), dtype=np.float64
    )


def _gap_g(inst: Instrument) -> TerminalPayoff:
    # GapOption: payment strike K1 = .strike, trigger K2 = .trigger.
    k1, k2 = inst.strike, inst.trigger  # type: ignore[attr-defined]
    if inst.is_call:  # type: ignore[attr-defined]
        return lambda s: np.asarray(np.where(s > k2, s - k1, 0.0), dtype=np.float64)
    return lambda s: np.asarray(np.where(s < k2, k1 - s, 0.0), dtype=np.float64)


def _asset_or_nothing_g(inst: Instrument) -> TerminalPayoff:
    k = inst.strike  # type: ignore[attr-defined]
    if inst.is_call:  # type: ignore[attr-defined]
        return lambda s: np.asarray(np.where(s > k, s, 0.0), dtype=np.float64)
    return lambda s: np.asarray(np.where(s < k, s, 0.0), dtype=np.float64)


TERMINAL_PAYOFF_BUILDERS: dict[type, PayoffBuilder] = {
    LogContract: _log_contract_g,
    LogOption: _log_option_g,
    PoweredOption: _powered_g,
    CappedPowerOption: _capped_power_g,
    SupershareOption: _supershare_g,
    GapOption: _gap_g,
    AssetOrNothingOption: _asset_or_nothing_g,
}


def build_terminal_payoff(instrument: Instrument) -> TerminalPayoff | None:
    """Return the terminal payoff ``g`` for ``instrument``, or ``None``."""
    builder = TERMINAL_PAYOFF_BUILDERS.get(type(instrument))
    return builder(instrument) if builder is not None else None
