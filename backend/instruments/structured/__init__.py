"""
Structured Products Package
============================

Structured financial products: Autocallables, Capital Protected Notes,
Reverse Convertibles, Phoenix, Shark Notes, Twin Win, Snowball.

Components:
- BondFloor, UpsideParticipation, FixedCoupon, ConditionalCoupon
- AutocallTrigger, KnockInPut
- KnockOutParticipation, TwinWinParticipation, SnowballCoupon

Products:
- Autocallable, CapitalProtectedNote, ReverseConvertible
- PhoenixAutocallable, SharkNote, TwinWin, SnowballAutocallable

Author: Thomas Vaudescal
Created: 2026
"""

from backend.instruments.structured.components import (
    AutocallTrigger,
    BondFloor,
    ConditionalCoupon,
    FixedCoupon,
    KnockInPut,
    KnockOutParticipation,
    SnowballCoupon,
    TwinWinParticipation,
    UpsideParticipation,
)
from backend.instruments.structured.products import (
    Autocallable,
    CapitalProtectedNote,
    PhoenixAutocallable,
    ReverseConvertible,
    SharkNote,
    SnowballAutocallable,
    TwinWin,
)

__all__ = [
    # Components
    "BondFloor",
    "UpsideParticipation",
    "FixedCoupon",
    "ConditionalCoupon",
    "AutocallTrigger",
    "KnockInPut",
    "KnockOutParticipation",
    "TwinWinParticipation",
    "SnowballCoupon",
    # Products
    "Autocallable",
    "CapitalProtectedNote",
    "ReverseConvertible",
    "PhoenixAutocallable",
    "SharkNote",
    "TwinWin",
    "SnowballAutocallable",
]
