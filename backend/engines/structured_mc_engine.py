"""
Backward-compatibility shim -- use backend.engines.structured instead.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from backend.engines.structured.engine import StructuredProductMCEngine  # noqa: F401

__all__ = ["StructuredProductMCEngine"]
