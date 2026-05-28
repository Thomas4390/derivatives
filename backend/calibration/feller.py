"""
Feller-condition control for stochastic-variance calibration
============================================================

The Feller condition ``2*kappa*theta > alpha**2`` guarantees that the CIR
variance process of Heston / Bates stays strictly positive. This module
centralises *how* the calibrators treat that condition, exposing three
explicit modes instead of the previous hard-coded soft penalty:

``FellerMode.OFF``
    No penalty — the optimiser is free to land in the Feller-violating
    region. This is the empirically realistic regime for equity-index
    surfaces, which almost always calibrate to a Feller-violating Heston.
``FellerMode.SOFT`` (default — preserves the legacy behaviour)
    The violation ``(alpha**2 - 2*kappa*theta)_+`` is appended to the
    least-squares residual with weight :data:`DEFAULT_FELLER_WEIGHT`. The
    optimiser is *discouraged* from violating Feller but not prevented.
``FellerMode.HARD``
    Feller is guaranteed *by construction*: the vol-of-vol ``alpha`` is
    reparametrised so that ``alpha <= sqrt(2*kappa*theta)`` always holds.
    This is solver-agnostic (works for every ``OptimizerStrategy``) and
    keeps the analytical JAX Jacobian intact, unlike a nonlinear
    constraint which ``scipy.optimize.least_squares`` cannot consume.

The reparametrisation helpers are pure functions duck-typed over the
array module (``numpy`` for seeding, ``jax.numpy`` for the traced
forward pass), so they are unit-testable in isolation.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Final


class FellerMode(str, Enum):
    """How the Feller condition is enforced during calibration."""

    OFF = "off"
    SOFT = "soft"
    HARD = "hard"

    @classmethod
    def coerce(cls, value: "FellerMode | str | None") -> "FellerMode":
        """Best-effort coercion from a UI string / enum, defaulting to SOFT.

        Unknown strings fall back to :attr:`SOFT` (the legacy behaviour)
        rather than raising, so a stale session-state value can never
        crash a calibration run.
        """
        if value is None:
            return cls.SOFT
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.SOFT


# Legacy soft-penalty weight — unchanged so SOFT mode reproduces the
# pre-existing calibration results bit-for-bit.
DEFAULT_FELLER_WEIGHT: Final[float] = 1000.0

# Shrink factor applied to the Feller cap in HARD mode so the *strict*
# inequality ``2*kappa*theta > alpha**2`` holds at the boundary (the model's
# ``feller_satisfied`` property uses a strict comparison).
FELLER_STRICT_FACTOR: Final[float] = 1.0 - 1e-6


def penalty_weight(mode: FellerMode, weight: float) -> float:
    """Effective soft-penalty weight for a given mode.

    Only :attr:`FellerMode.SOFT` carries a non-zero penalty. OFF disables
    it; HARD guarantees Feller via reparametrisation, so the appended
    residual stays at zero (kept in the vector purely for a stable shape).
    """
    return float(weight) if mode is FellerMode.SOFT else 0.0


def feller_capped_xi(
    kappa: Any,
    theta: Any,
    unit: Any,
    xi_lo: float,
    xi_hi: float,
    *,
    xp: Any,
) -> Any:
    """Map a unit variable ``u in (0, 1)`` to ``alpha`` with ``alpha**2 <= 2*kappa*theta``.

    Used by the HARD-mode forward reparametrisation. ``alpha`` is confined to
    ``[xi_lower, xi_upper]`` where ``xi_upper = min(xi_hi, FELLER_STRICT_FACTOR
    * sqrt(2*kappa*theta))`` and ``xi_lower = min(xi_lo, xi_upper)``. The
    ``min`` with ``xi_upper`` on the lower bound gives Feller precedence over
    the numerical box floor in the negligible ``kappa*theta -> 0`` corner.

    Parameters
    ----------
    kappa, theta, unit :
        Scalars or arrays from ``xp`` (``numpy`` or ``jax.numpy``).
    xi_lo, xi_hi :
        Box bounds on the vol-of-vol.
    xp :
        Array module providing ``sqrt`` and ``minimum`` (``numpy`` / ``jnp``).
    """
    xi_feller_max = FELLER_STRICT_FACTOR * xp.sqrt(2.0 * kappa * theta)
    xi_upper = xp.minimum(xi_hi, xi_feller_max)
    xi_lower = xp.minimum(xi_lo, xi_upper)
    return xi_lower + (xi_upper - xi_lower) * unit


def feller_xi_to_unit(
    kappa: float,
    theta: float,
    alpha: float,
    xi_lo: float,
    xi_hi: float,
) -> float:
    """Inverse of :func:`feller_capped_xi` for HARD-mode seed construction.

    Returns the unit variable ``u in (0, 1)`` whose forward image matches a
    (possibly Feller-violating) seed ``alpha``, clamped into the admissible
    capped interval. NumPy-only — used to seed the unconstrained start point.
    """
    xi_feller_max = FELLER_STRICT_FACTOR * float((2.0 * kappa * theta) ** 0.5)
    xi_upper = min(xi_hi, xi_feller_max)
    xi_lower = min(xi_lo, xi_upper)
    span = xi_upper - xi_lower
    if span <= 0.0:
        return 0.5
    clamped = min(max(float(alpha), xi_lower), xi_upper)
    return (clamped - xi_lower) / span
