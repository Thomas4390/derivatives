"""
Stationarity control for Heston-Nandi GARCH calibration
=======================================================

The risk-neutral Heston-Nandi GARCH variance recursion

    h_{t+1} = omega + beta h_t + alpha (z_t - gamma sqrt(h_t))^2

is variance-stationary (finite unconditional variance) iff the *persistence*

    pi = beta + alpha * gamma^2  <  1.

This module centralises *how* the calibrator treats that condition — mirroring
:mod:`backend.calibration.feller` so the UI and backend share one mental model:

``StationarityMode.OFF``
    No penalty — the optimiser may land in the non-stationary region. Finite-
    horizon option prices remain well defined; only the unconditional variance
    diverges.
``StationarityMode.SOFT`` (default)
    The violation ``(beta + alpha*gamma^2 - 1)_+`` is appended to the
    least-squares residual with weight :data:`DEFAULT_STATIONARITY_WEIGHT`.
``StationarityMode.HARD``
    Stationarity is guaranteed *by construction*: ``gamma`` is reparametrised so
    that ``gamma <= sqrt((1 - beta) / alpha)`` always holds. Solver-agnostic and
    keeps the analytical JAX Jacobian intact (unlike a nonlinear constraint).

The reparametrisation helpers are pure functions duck-typed over the array
module (``numpy`` for seeding, ``jax.numpy`` for the traced forward pass).

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Final


class StationarityMode(str, Enum):
    """How the persistence condition ``beta + alpha*gamma^2 < 1`` is enforced."""

    OFF = "off"
    SOFT = "soft"
    HARD = "hard"

    @classmethod
    def coerce(cls, value: "StationarityMode | str | None") -> "StationarityMode":
        """Best-effort coercion from a UI string / enum, defaulting to SOFT.

        Unknown strings fall back to :attr:`SOFT` rather than raising, so a stale
        session-state value can never crash a calibration run.
        """
        if value is None:
            return cls.SOFT
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.SOFT


# Soft-penalty weight default — mirrors DEFAULT_FELLER_WEIGHT.
DEFAULT_STATIONARITY_WEIGHT: Final[float] = 1000.0

# Shrink factor applied to the cap in HARD mode so the *strict* inequality
# ``beta + alpha*gamma^2 < 1`` holds at the boundary.
STATIONARITY_STRICT_FACTOR: Final[float] = 1.0 - 1e-6

# Floor on alpha used when dividing by it in the cap (alpha -> 0 means the
# leverage term vanishes and stationarity is governed by beta alone).
_ALPHA_FLOOR: Final[float] = 1e-12


def penalty_weight(mode: StationarityMode, weight: float) -> float:
    """Effective soft-penalty weight for a given mode.

    Only :attr:`StationarityMode.SOFT` carries a non-zero penalty. OFF disables
    it; HARD guarantees stationarity via reparametrisation, so the appended
    residual stays at zero (kept in the vector purely for a stable shape).
    """
    return float(weight) if mode is StationarityMode.SOFT else 0.0


def stationarity_capped_gamma(
    alpha: Any,
    beta: Any,
    unit: Any,
    gamma_lo: float,
    gamma_hi: float,
    *,
    xp: Any,
) -> Any:
    """Map a unit variable ``u in (0, 1)`` to ``gamma`` with ``beta + alpha*gamma^2 < 1``.

    Used by the HARD-mode forward reparametrisation. ``gamma`` is confined to
    ``[gamma_lower, gamma_upper]`` where
    ``gamma_upper = min(gamma_hi, STATIONARITY_STRICT_FACTOR * sqrt((1-beta)/alpha))``
    and ``gamma_lower = min(gamma_lo, gamma_upper)``.

    Parameters
    ----------
    alpha, beta, unit :
        Scalars or arrays from ``xp`` (``numpy`` or ``jax.numpy``).
    gamma_lo, gamma_hi :
        Box bounds on the leverage parameter.
    xp :
        Array module providing ``sqrt``, ``maximum`` and ``minimum``.
    """
    alpha_safe = xp.maximum(alpha, _ALPHA_FLOOR)
    headroom = xp.maximum(1.0 - beta, 0.0)
    gamma_stat_max = STATIONARITY_STRICT_FACTOR * xp.sqrt(headroom / alpha_safe)
    gamma_upper = xp.minimum(gamma_hi, gamma_stat_max)
    gamma_lower = xp.minimum(gamma_lo, gamma_upper)
    return gamma_lower + (gamma_upper - gamma_lower) * unit


def ngarch_capped_gamma(alpha: float, beta: float, gamma_hi: float) -> float:
    """Largest gamma keeping the *nonaffine* NGARCH persistence below 1.

    The Duan NGARCH-Q persistence ``pi = beta + alpha*(1 + gamma^2)`` differs from
    the affine Heston-Nandi form (``beta + alpha*gamma^2``): ``alpha`` also enters
    the constant term, so the cap solves ``alpha*gamma^2 < 1 - beta - alpha`` ⇒
    ``gamma < sqrt((1 - beta - alpha) / alpha)``. Used by the NGARCH-Q calibrator's
    HARD stationarity mode (its MC objective has no analytical Jacobian to preserve,
    so a direct clamp is enough — no sigmoid reparametrisation needed).
    """
    headroom = max(1.0 - float(beta) - float(alpha), 0.0)
    cap = STATIONARITY_STRICT_FACTOR * float(
        (headroom / max(float(alpha), _ALPHA_FLOOR)) ** 0.5
    )
    return min(float(gamma_hi), cap)


def gjr_capped_gamma(alpha: float, beta: float, gamma_hi: float) -> float:
    """Largest gamma keeping the *GJR* risk-neutral persistence below 1.

    With a symmetric innovation ``E[z^2 1{z<0}] = 1/2``, the GJR persistence is
    ``pi = beta + alpha + gamma/2``, so the cap solves ``gamma/2 < 1 - beta - alpha``
    ⇒ ``gamma < 2(1 - beta - alpha)``. Direct clamp for the GJR-Q calibrator's HARD
    mode (MC objective — no analytical Jacobian to preserve).
    """
    cap = STATIONARITY_STRICT_FACTOR * 2.0 * max(1.0 - float(beta) - float(alpha), 0.0)
    return min(float(gamma_hi), cap)


def garch_capped_alpha(beta: float, alpha_hi: float) -> float:
    """Largest alpha keeping the *symmetric* GARCH persistence below 1.

    Plain GARCH has no leverage parameter; its persistence is ``pi = beta + alpha``,
    so HARD stationarity caps ``alpha < 1 - beta``. Used by the GARCH-Q calibrator's
    HARD mode (there is no gamma to reparametrise).
    """
    cap = STATIONARITY_STRICT_FACTOR * max(1.0 - float(beta), 0.0)
    return min(float(alpha_hi), cap)


def garch_q_persistence(
    garch_type: str, alpha: float, beta: float, gamma: float = 0.0
) -> float:
    """Variance persistence of a risk-neutral GARCH variant (< 1 ⇒ stationary).

    ``garch``  → ``beta + alpha``            (symmetric)
    ``ngarch`` → ``beta + alpha*(1 + gamma^2)``  (Duan nonaffine)
    ``gjr_garch`` → ``beta + alpha + gamma/2``   (GJR, E[z^2 1{z<0}]=1/2)
    """
    a, b, g = float(alpha), float(beta), float(gamma)
    if garch_type == "ngarch":
        return b + a * (1.0 + g * g)
    if garch_type == "gjr_garch":
        return b + a + 0.5 * g
    return b + a


def stationarity_gamma_to_unit(
    alpha: float,
    beta: float,
    gamma: float,
    gamma_lo: float,
    gamma_hi: float,
) -> float:
    """Inverse of :func:`stationarity_capped_gamma` for HARD-mode seed construction.

    Returns the unit variable ``u in (0, 1)`` whose forward image matches a
    (possibly non-stationary) seed ``gamma``, clamped into the admissible capped
    interval. NumPy-only — used to seed the unconstrained start point.
    """
    alpha_safe = max(float(alpha), _ALPHA_FLOOR)
    headroom = max(1.0 - float(beta), 0.0)
    gamma_stat_max = STATIONARITY_STRICT_FACTOR * float((headroom / alpha_safe) ** 0.5)
    gamma_upper = min(gamma_hi, gamma_stat_max)
    gamma_lower = min(gamma_lo, gamma_upper)
    span = gamma_upper - gamma_lower
    if span <= 0.0:
        return 0.5
    clamped = min(max(float(gamma), gamma_lower), gamma_upper)
    return (clamped - gamma_lower) / span
