"""
Monte-Carlo batch payoffs for the terminal exotics (payoff = g(S_T)).

Registers each terminal exotic (log contract/option, powered, capped power,
supershare, gap, asset-or-nothing) into ``EXOTIC_MC_PAYOFF_REGISTRY`` so it can
be priced under ANY MC-capable model (including GARCH, which has no
characteristic function for the Fourier route). The payoff itself is the shared
``g`` from :mod:`_terminal_payoffs`, applied to the simulated terminal column.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import numpy as np

from backend.core.interfaces import Instrument
from backend.engines.exotic._mc_payoff_registry import MCPayoffFn, register_mc_payoff
from backend.engines.exotic._terminal_payoffs import (
    TERMINAL_PAYOFF_BUILDERS,
    PayoffBuilder,
)


def _make_mc_payoff(builder: PayoffBuilder) -> MCPayoffFn:
    """Wrap a terminal payoff builder into a batch MC payoff over price paths."""

    def payoff(paths: np.ndarray, inst: Instrument) -> np.ndarray:
        g = builder(inst)
        s_t = np.ascontiguousarray(paths[:, -1])
        return np.asarray(g(s_t), dtype=np.float64)

    return payoff


for _cls, _builder in TERMINAL_PAYOFF_BUILDERS.items():
    register_mc_payoff(_cls, _make_mc_payoff(_builder))
