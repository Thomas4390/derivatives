"""
Structured-product Monte-Carlo engine + Numba cashflow kernels.

Regroups the structured-product MC engine (:mod:`engine`), its Numba cashflow
kernels (:mod:`kernels`) and the kernel warm-up helper (:mod:`warmup`) under a
single package, mirroring ``engines/exotic/``. The legacy import paths
``structured_mc_engine``, ``structured_kernels`` and
``structured_kernels_warmup`` keep resolving through back-compat shims.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from backend.engines.structured.engine import StructuredProductMCEngine  # noqa: F401
from backend.engines.structured.kernels import *  # noqa: F401,F403
