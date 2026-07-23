"""
Structured Products
===================

Immutable structured-product assemblies, grouped by economic structure
(protected notes, autocallables, capital-at-risk). This package re-exports all
11 products so ``from backend.instruments.structured.products import X`` resolves
identically.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from backend.instruments.structured.products.autocallables import (  # noqa: F401
    Autocallable,
    PhoenixAutocallable,
    SnowballAutocallable,
)
from backend.instruments.structured.products.capital_at_risk import (  # noqa: F401
    ReverseConvertible,
    TwinWin,
)
from backend.instruments.structured.products.protected_notes import (  # noqa: F401
    AsianNote,
    CapitalProtectedNote,
    CliquetNote,
    LookbackNote,
    RangeAccrualNote,
    SharkNote,
)

__all__ = [
    "CapitalProtectedNote",
    "ReverseConvertible",
    "Autocallable",
    "PhoenixAutocallable",
    "SharkNote",
    "TwinWin",
    "SnowballAutocallable",
    "CliquetNote",
    "AsianNote",
    "LookbackNote",
    "RangeAccrualNote",
]
