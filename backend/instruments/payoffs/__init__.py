"""
Payoff Functions
================

Atomic payoff value-objects for option contracts. Payoffs know the contractual
rules but nothing about market data, dynamics, or pricing method.

The implementation lives in cohesive sub-modules (_internals validation/kernels,
vanilla, exotic, markers); this package re-exports every public payoff class so
``from backend.instruments.payoffs import X`` resolves identically.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from backend.instruments.payoffs._internals import (  # noqa: F401
    _call_payoff,
    _digital_call_payoff,
    _digital_put_payoff,
    _path_validity_code,
    _put_payoff,
    _validate_path_array,
    _validate_spot_array,
)
from backend.instruments.payoffs.exotic import (  # noqa: F401
    AsianCallPayoff,
    AsianPutPayoff,
    BarrierDownOutPutPayoff,
    BarrierUpOutCallPayoff,
    LookbackDiscountedCallPayoff,
    LookbackFloatingCallPayoff,
    LookbackFloatingPutPayoff,
    LowPointForwardPayoff,
)
from backend.instruments.payoffs.markers import AnalyticalOnlyPayoff  # noqa: F401
from backend.instruments.payoffs.vanilla import (  # noqa: F401
    BondPayoff,
    CompositePayoff,
    DigitalCallPayoff,
    DigitalPutPayoff,
    SpotPayoff,
    VanillaCallPayoff,
    VanillaPutPayoff,
)

__all__ = [
    "VanillaCallPayoff",
    "VanillaPutPayoff",
    "DigitalCallPayoff",
    "DigitalPutPayoff",
    "SpotPayoff",
    "BondPayoff",
    "CompositePayoff",
    "AsianCallPayoff",
    "AsianPutPayoff",
    "BarrierUpOutCallPayoff",
    "BarrierDownOutPutPayoff",
    "LookbackFloatingCallPayoff",
    "LookbackFloatingPutPayoff",
    "LookbackDiscountedCallPayoff",
    "LowPointForwardPayoff",
    "AnalyticalOnlyPayoff",
]
