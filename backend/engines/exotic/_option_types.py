"""
Option-type integer constants for exotic Greeks dispatch.

Single source of truth shared by the Numba kernels (``_greeks_kernels``), the
engine dispatch helpers, and ``ExoticAnalyticEngine``.
"""

from __future__ import annotations

BARRIER: int = 0
ASIAN_GEO: int = 1
DIGITAL: int = 2
LOOKBACK_FIXED: int = 3
LOOKBACK_FLOATING: int = 4
CHOOSER: int = 5
ASSET_OR_NOTHING: int = 6
POWER: int = 7
GAP: int = 8
