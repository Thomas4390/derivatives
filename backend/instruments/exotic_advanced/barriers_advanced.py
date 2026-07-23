"""Soft-, partial-time- and binary-barrier options (Hart-Ross, Heynen-Kat, Reiner-Rubinstein).

Part of the Haug exotic-options catalog; see the package overview in
``backend.instruments.exotic_advanced``.

Author: Thomas Vaudescal
"""

from __future__ import annotations

from backend.core.interfaces import Instrument, Payoff
from backend.instruments._frozen import FrozenInstrument
from backend.core.result_types import ExerciseStyle
from backend.instruments.payoffs import AnalyticalOnlyPayoff

__all__ = [
    "SoftBarrierOption",
    "PartialTimeBarrierOption",
    "BinaryBarrierOption",
]


# =============================================================================
# SOFT-BARRIER OPTION (Hart-Ross 1994)
# =============================================================================


class SoftBarrierOption(FrozenInstrument, Instrument):
    """
    Soft-barrier option (Hart-Ross 1994).

    The barrier is a *range* ``[lower, upper]`` rather than a single level; the
    option knocks in/out gradually as the underlying's extreme traverses the
    band. A call uses a soft DOWN band (below spot), a put a soft UP band. As
    ``upper -> lower`` it collapses to a standard barrier.

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike price (> 0).
    lower : float
        Lower edge L of the soft band (> 0).
    upper : float
        Upper edge U of the soft band (>= lower).
    maturity : float
        Time to expiration in years (> 0).
    is_call : bool
        True for call (soft-down), False for put (soft-up).
    is_knock_in : bool
        True for knock-in, False for knock-out (default).
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).
    """

    __slots__ = (
        "_strike",
        "_lower",
        "_upper",
        "_maturity",
        "_is_call",
        "_is_knock_in",
        "_exercise",
        "_payoff",
    )

    def __init__(
        self,
        strike: float,
        lower: float,
        upper: float,
        maturity: float,
        is_call: bool = True,
        is_knock_in: bool = False,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if lower <= 0:
            raise ValueError(f"Lower edge must be positive, got {lower}")
        if upper < lower:
            raise ValueError(
                f"Upper edge must be >= lower, got upper={upper}, lower={lower}"
            )
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_lower", lower)
        object.__setattr__(self, "_upper", upper)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_is_knock_in", is_knock_in)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "SoftBarrierOption",
                "soft-barrier uses the Hart-Ross closed-form",
            ),
        )

    @property
    def strike(self) -> float:
        """Strike price."""
        return self._strike

    @property
    def lower(self) -> float:
        """Lower edge L of the soft band."""
        return self._lower

    @property
    def upper(self) -> float:
        """Upper edge U of the soft band."""
        return self._upper

    @property
    def maturity(self) -> float:
        """Time to expiration in years."""
        return self._maturity

    @property
    def is_call(self) -> bool:
        """True for call (soft-down), False for put (soft-up)."""
        return self._is_call

    @property
    def is_knock_in(self) -> bool:
        """True for knock-in, False for knock-out."""
        return self._is_knock_in

    @property
    def payoff(self) -> Payoff:
        """An :class:`AnalyticalOnlyPayoff` (closed-form pricing only)."""
        return self._payoff

    @property
    def exercise_style(self) -> ExerciseStyle:
        """Exercise style (Instrument interface)."""
        return self._exercise

    @property
    def option_type(self) -> str:
        """String representation of option type."""
        direction = "down" if self._is_call else "up"
        knock = "in" if self._is_knock_in else "out"
        opt = "call" if self._is_call else "put"
        return f"soft_barrier_{direction}_{knock}_{opt}"

    def __repr__(self) -> str:
        opt_type = "Call" if self._is_call else "Put"
        knock = "In" if self._is_knock_in else "Out"
        return (
            f"SoftBarrierOption({knock}{opt_type}, K={self._strike}, "
            f"L={self._lower}, U={self._upper}, T={self._maturity})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SoftBarrierOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._lower == other._lower
            and self._upper == other._upper
            and self._maturity == other._maturity
            and self._is_call == other._is_call
            and self._is_knock_in == other._is_knock_in
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._strike,
                self._lower,
                self._upper,
                self._maturity,
                self._is_call,
                self._is_knock_in,
                self._exercise,
            )
        )


# =============================================================================
# PARTIAL-TIME BARRIER OPTION (Heynen-Kat 1994)
# =============================================================================

_PARTIAL_BARRIER_TYPES = frozenset(
    {"down_out_A", "up_out_A", "out_B1", "down_out_B2", "up_out_B2"}
)


class PartialTimeBarrierOption(FrozenInstrument, Instrument):
    """
    Partial-time single-asset barrier option (Heynen-Kat 1994).

    The barrier ``barrier`` is only live over part of the option's life:

    - ``*_A`` (start) types monitor over ``[0, t1]``;
    - ``*_B1`` / ``*_B2`` (end) types monitor over ``[t1, T2]``. ``B1`` knocks on
      any touch (direction-agnostic); ``B2`` is directional (down/up-out).

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike price (> 0).
    barrier : float
        Barrier level H (> 0).
    t1 : float
        Monitoring-window boundary (``0 < t1 <= maturity``).
    maturity : float
        Time to expiration T2 in years (> 0).
    barrier_type : str
        One of ``down_out_A``, ``up_out_A``, ``out_B1``, ``down_out_B2``,
        ``up_out_B2``.
    is_call : bool
        True for call, False for put.
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).
    """

    __slots__ = (
        "_strike",
        "_barrier",
        "_t1",
        "_maturity",
        "_barrier_type",
        "_is_call",
        "_exercise",
        "_payoff",
    )

    def __init__(
        self,
        strike: float,
        barrier: float,
        t1: float,
        maturity: float,
        barrier_type: str = "out_B1",
        is_call: bool = True,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if barrier <= 0:
            raise ValueError(f"Barrier must be positive, got {barrier}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if not (0 < t1 <= maturity):
            raise ValueError(f"t1 must satisfy 0 < t1 <= maturity, got {t1}")
        if barrier_type not in _PARTIAL_BARRIER_TYPES:
            raise ValueError(
                f"barrier_type must be one of {sorted(_PARTIAL_BARRIER_TYPES)}, "
                f"got '{barrier_type}'"
            )

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_barrier", barrier)
        object.__setattr__(self, "_t1", t1)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_barrier_type", barrier_type)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "PartialTimeBarrierOption",
                "partial-time barrier uses the Heynen-Kat closed-form",
            ),
        )

    @property
    def strike(self) -> float:
        """Strike price."""
        return self._strike

    @property
    def barrier(self) -> float:
        """Barrier level H."""
        return self._barrier

    @property
    def t1(self) -> float:
        """Monitoring-window boundary."""
        return self._t1

    @property
    def maturity(self) -> float:
        """Time to expiration T2 in years."""
        return self._maturity

    @property
    def barrier_type(self) -> str:
        """Partial-time barrier type code."""
        return self._barrier_type

    @property
    def is_call(self) -> bool:
        """True for call, False for put."""
        return self._is_call

    @property
    def payoff(self) -> Payoff:
        """An :class:`AnalyticalOnlyPayoff` (closed-form pricing only)."""
        return self._payoff

    @property
    def exercise_style(self) -> ExerciseStyle:
        """Exercise style (Instrument interface)."""
        return self._exercise

    @property
    def option_type(self) -> str:
        """String representation of option type."""
        opt = "call" if self._is_call else "put"
        return f"partial_barrier_{self._barrier_type}_{opt}"

    def __repr__(self) -> str:
        opt_type = "Call" if self._is_call else "Put"
        return (
            f"PartialTimeBarrierOption({self._barrier_type}{opt_type}, "
            f"K={self._strike}, H={self._barrier}, t1={self._t1}, "
            f"T2={self._maturity})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PartialTimeBarrierOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._barrier == other._barrier
            and self._t1 == other._t1
            and self._maturity == other._maturity
            and self._barrier_type == other._barrier_type
            and self._is_call == other._is_call
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._strike,
                self._barrier,
                self._t1,
                self._maturity,
                self._barrier_type,
                self._is_call,
                self._exercise,
            )
        )


# =============================================================================
# BINARY-BARRIER OPTION (Reiner-Rubinstein 1991)
# =============================================================================

# Short labels for the 28 binary-barrier types (Haug 4.19.5, 1-indexed).
_BINARY_BARRIER_LABELS = (
    "down_in_cash_athit",
    "up_in_cash_athit",
    "down_in_asset_athit",
    "up_in_asset_athit",
    "down_in_cash_atexp",
    "up_in_cash_atexp",
    "down_in_asset_atexp",
    "up_in_asset_atexp",
    "down_out_cash",
    "up_out_cash",
    "down_out_asset",
    "up_out_asset",
    "down_in_cash_call",
    "up_in_cash_call",
    "down_in_asset_call",
    "up_in_asset_call",
    "down_in_cash_put",
    "up_in_cash_put",
    "down_in_asset_put",
    "up_in_asset_put",
    "down_out_cash_call",
    "up_out_cash_call",
    "down_out_asset_call",
    "up_out_asset_call",
    "down_out_cash_put",
    "up_out_cash_put",
    "down_out_asset_put",
    "up_out_asset_put",
)


class BinaryBarrierOption(FrozenInstrument, Instrument):
    """
    Binary-barrier option (Reiner-Rubinstein 1991, one of 28 types).

    A barrier digital: pays a fixed ``cash`` amount (cash-or-nothing types) or
    one unit of the asset (asset-or-nothing types), contingent on the barrier
    ``barrier`` being hit (in) or not hit (out) and, for the strike-gated types,
    on the terminal spot relative to ``strike``. The ``binary_type`` integer
    (1..28) selects the exact variant per Haug's catalogue (4.19.5); see the
    ``BB_*`` constants in :mod:`backend.engines.exotic.binary_barrier`.

    For the asset-or-nothing types the ``cash`` field is ignored (the payout is
    the asset). For the asset-(at-hit) types 3 & 4 the payout equals the barrier
    level by construction.

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike price X (> 0). Parameterises every type; the strike-gated types
        (13..28) compare the terminal spot against it.
    barrier : float
        Barrier level H (> 0).
    cash : float
        Cash payout K for the cash-or-nothing types (>= 0; ignored otherwise).
    maturity : float
        Time to expiration in years (> 0).
    binary_type : int
        Binary-barrier type code, 1..28 (Haug 4.19.5).
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).

    Examples
    --------
    # Down-and-in cash-or-nothing call paying 15 if S_T > 102, given H=100 hit.
    bb = BinaryBarrierOption(102, 100, 15, 0.5, binary_type=13)
    """

    __slots__ = (
        "_strike",
        "_barrier",
        "_cash",
        "_maturity",
        "_binary_type",
        "_exercise",
        "_payoff",
    )

    def __init__(
        self,
        strike: float,
        barrier: float,
        cash: float,
        maturity: float,
        binary_type: int,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if barrier <= 0:
            raise ValueError(f"Barrier must be positive, got {barrier}")
        if cash < 0:
            raise ValueError(f"Cash payout must be non-negative, got {cash}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if not (1 <= binary_type <= 28):
            raise ValueError(
                f"binary_type must be an integer in 1..28, got {binary_type}"
            )

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_barrier", barrier)
        object.__setattr__(self, "_cash", cash)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_binary_type", int(binary_type))
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "BinaryBarrierOption",
                "binary-barrier uses the Reiner-Rubinstein closed-form",
            ),
        )

    @property
    def strike(self) -> float:
        """Strike price X."""
        return self._strike

    @property
    def barrier(self) -> float:
        """Barrier level H."""
        return self._barrier

    @property
    def cash(self) -> float:
        """Cash payout K (cash-or-nothing types only)."""
        return self._cash

    @property
    def maturity(self) -> float:
        """Time to expiration in years."""
        return self._maturity

    @property
    def binary_type(self) -> int:
        """Binary-barrier type code (1..28)."""
        return self._binary_type

    @property
    def payoff(self) -> Payoff:
        """An :class:`AnalyticalOnlyPayoff` (closed-form pricing only)."""
        return self._payoff

    @property
    def exercise_style(self) -> ExerciseStyle:
        """Exercise style (Instrument interface)."""
        return self._exercise

    @property
    def option_type(self) -> str:
        """String representation of option type."""
        return f"binary_barrier_{_BINARY_BARRIER_LABELS[self._binary_type - 1]}"

    def __repr__(self) -> str:
        label = _BINARY_BARRIER_LABELS[self._binary_type - 1]
        return (
            f"BinaryBarrierOption(#{self._binary_type} {label}, K={self._strike}, "
            f"H={self._barrier}, cash={self._cash}, T={self._maturity})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BinaryBarrierOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._barrier == other._barrier
            and self._cash == other._cash
            and self._maturity == other._maturity
            and self._binary_type == other._binary_type
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._strike,
                self._barrier,
                self._cash,
                self._maturity,
                self._binary_type,
                self._exercise,
            )
        )
