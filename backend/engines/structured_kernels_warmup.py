"""
Backward-compatibility shim -- use backend.engines.structured.warmup instead.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from backend.engines.structured.warmup import precompile_structured_kernels  # noqa: F401

__all__ = ["precompile_structured_kernels"]
