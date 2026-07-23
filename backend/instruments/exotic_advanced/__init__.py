"""Advanced Exotic Option Instruments (Haug catalog).

Immutable instrument classes for the closed-form exotic pricers ported from
Espen Haug, *The Complete Guide to Option Pricing Formulas* (2nd ed.). These
are additive to :mod:`backend.instruments.exotic_options` (the legacy 8 are
left untouched) and dispatch through the registry in
``backend.engines.exotic._registry``.

Split into a sub-package by family (ADR-A, 2026-06-21); the public surface is
unchanged. All instruments are IMMUTABLE after construction.

Author: Thomas Vaudescal
"""

from __future__ import annotations

from backend.instruments.exotic_advanced.barriers import (
    DoubleBarrierOption,
    DoubleBarrierKnockOutCall,
    DoubleBarrierKnockOutPut,
    DoubleBarrierKnockInCall,
    DoubleBarrierKnockInPut,
    DiscreteBarrierOption,
)
from backend.instruments.exotic_advanced.barriers_advanced import (
    SoftBarrierOption,
    PartialTimeBarrierOption,
    BinaryBarrierOption,
)
from backend.instruments.exotic_advanced.lookbacks import (
    PartialFloatLookbackOption,
    PartialFixedLookbackOption,
    ExtremeSpreadOption,
)
from backend.instruments.exotic_advanced.complex_chooser import ComplexChooserOption
from backend.instruments.exotic_advanced.compound import CompoundOption
from backend.instruments.exotic_advanced.extendible import ExtendibleOption
from backend.instruments.exotic_advanced.forward_start import ForwardStartOption
from backend.instruments.exotic_advanced.power import (
    PoweredOption,
    CappedPowerOption,
)
from backend.instruments.exotic_advanced.analytic import (
    LogContract,
    LogOption,
    TimeSwitchOption,
    SupershareOption,
    ArithmeticAsianOption,
)

__all__ = [
    "DoubleBarrierOption",
    "DoubleBarrierKnockOutCall",
    "DoubleBarrierKnockOutPut",
    "DoubleBarrierKnockInCall",
    "DoubleBarrierKnockInPut",
    "DiscreteBarrierOption",
    "SoftBarrierOption",
    "PartialTimeBarrierOption",
    "BinaryBarrierOption",
    "PartialFloatLookbackOption",
    "PartialFixedLookbackOption",
    "ExtremeSpreadOption",
    "ComplexChooserOption",
    "CompoundOption",
    "ExtendibleOption",
    "ForwardStartOption",
    "PoweredOption",
    "CappedPowerOption",
    "LogContract",
    "LogOption",
    "TimeSwitchOption",
    "SupershareOption",
    "ArithmeticAsianOption",
]
