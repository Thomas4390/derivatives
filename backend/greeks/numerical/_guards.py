"""
Defensive numerical guards shared by the finite-difference aggregator and the
model-aware Greeks class: divide-by-zero floor, positive-spot precondition,
finite-price check, and safe division with a WARNING log.
"""

from __future__ import annotations

import math

from backend.utils.logging import get_logger

_logger = get_logger(__name__)


# Defensive divide-by-zero floor for FD denominators (h_s, 2*h_r, ...).
# Tighter than SMOOTHING_EPS (0.02) which is a moneyness window, and tighter
# than DEFAULT_TOLERANCE (1e-8) so legitimate rate bumps (1bp = 1e-4) pass.
_DIV_EPS: float = 1e-15


def _require_positive_spot(spot: float) -> None:
    """Reject zero/negative spot before computing relative bumps.

    Finite-difference Greeks scale ``h`` by ``spot``; if ``spot <= 0`` the
    bump is degenerate and any subsequent division blows up silently.
    """
    if not math.isfinite(spot) or spot <= 0.0:
        raise ValueError(f"finite-difference Greeks require spot > 0 (got {spot!r})")


def _check_finite(value: float, *, label: str) -> float:
    """Raise if a pricer returned NaN/Inf — keeps Greek pipelines fail-fast."""
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite price evaluation at {label}: {value!r}")
    return value


def _safe_div(numerator: float, denominator: float, *, label: str) -> float:
    """Divide with an explicit guard against degenerate denominators.

    Falls back to ``nan`` and logs at WARNING when ``|denominator| < eps``
    so callers can detect the situation without the silent ``inf`` that a
    raw ``/`` would produce.
    """
    if abs(denominator) < _DIV_EPS:
        _logger.warning(
            "finite-difference division skipped at %s: |denom|=%.3e < eps=%.3e",
            label,
            denominator,
            _DIV_EPS,
        )
        return float("nan")
    return numerator / denominator
