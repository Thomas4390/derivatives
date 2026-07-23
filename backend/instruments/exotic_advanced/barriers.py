"""Double- and discrete-monitored barrier options (Ikeda-Kunitomo, BGK).

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
    "DoubleBarrierOption",
    "DoubleBarrierKnockOutCall",
    "DoubleBarrierKnockOutPut",
    "DoubleBarrierKnockInCall",
    "DoubleBarrierKnockInPut",
    "DiscreteBarrierOption",
]

# =============================================================================
# DOUBLE-BARRIER OPTION (Ikeda-Kunitomo 1992)
# =============================================================================


class DoubleBarrierOption(FrozenInstrument, Instrument):
    """
    Double-barrier option (knock-out or knock-in) on a single asset.

    Priced by the Ikeda-Kunitomo (1992) infinite-series formula. The option is
    knocked out (or in) if the underlying touches the lower barrier ``lower``
    or the upper barrier ``upper`` before expiry. Optional curvature
    parameters let the flat boundaries grow/decay exponentially.

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike price (> 0).
    lower : float
        Lower barrier L (> 0).
    upper : float
        Upper barrier U (> lower).
    maturity : float
        Time to expiration in years (> 0).
    is_call : bool
        True for call, False for put.
    is_knock_in : bool
        True for knock-in (priced by parity), False for knock-out (default).
    curvature : tuple[float, float]
        ``(delta1, delta2)`` upper/lower boundary curvature (default flat
        ``(0.0, 0.0)``).
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).

    Examples
    --------
    dko = DoubleBarrierOption(100, 80, 120, 0.5, is_call=True)
    """

    __slots__ = (
        "_strike",
        "_lower",
        "_upper",
        "_maturity",
        "_is_call",
        "_is_knock_in",
        "_curvature",
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
        curvature: tuple[float, float] = (0.0, 0.0),
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if lower <= 0:
            raise ValueError(f"Lower barrier must be positive, got {lower}")
        if upper <= lower:
            raise ValueError(
                f"Upper barrier must exceed lower, got upper={upper}, lower={lower}"
            )
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if len(curvature) != 2:
            raise ValueError(f"curvature must be a 2-tuple, got {curvature!r}")

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_lower", lower)
        object.__setattr__(self, "_upper", upper)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_is_knock_in", is_knock_in)
        object.__setattr__(
            self, "_curvature", (float(curvature[0]), float(curvature[1]))
        )
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "DoubleBarrierOption",
                "double-barrier uses the Ikeda-Kunitomo closed-form series",
            ),
        )

    @property
    def strike(self) -> float:
        """Strike price."""
        return self._strike

    @property
    def lower(self) -> float:
        """Lower barrier L."""
        return self._lower

    @property
    def upper(self) -> float:
        """Upper barrier U."""
        return self._upper

    @property
    def maturity(self) -> float:
        """Time to expiration in years."""
        return self._maturity

    @property
    def is_call(self) -> bool:
        """True for call, False for put."""
        return self._is_call

    @property
    def is_knock_in(self) -> bool:
        """True for knock-in, False for knock-out."""
        return self._is_knock_in

    @property
    def curvature(self) -> tuple[float, float]:
        """Boundary curvature ``(delta1, delta2)``."""
        return self._curvature

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
        knock = "in" if self._is_knock_in else "out"
        opt = "call" if self._is_call else "put"
        return f"double_barrier_{knock}_{opt}"

    def __repr__(self) -> str:
        opt_type = "Call" if self._is_call else "Put"
        knock = "In" if self._is_knock_in else "Out"
        curv = f", curv={self._curvature}" if self._curvature != (0.0, 0.0) else ""
        return (
            f"DoubleBarrierOption({knock}{opt_type}, K={self._strike}, "
            f"L={self._lower}, U={self._upper}, T={self._maturity}{curv})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DoubleBarrierOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._lower == other._lower
            and self._upper == other._upper
            and self._maturity == other._maturity
            and self._is_call == other._is_call
            and self._is_knock_in == other._is_knock_in
            and self._curvature == other._curvature
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
                self._curvature,
                self._exercise,
            )
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def DoubleBarrierKnockOutCall(
    strike: float, lower: float, upper: float, maturity: float
) -> DoubleBarrierOption:
    """Create a double knock-out call (up-and-out-down-and-out)."""
    return DoubleBarrierOption(strike, lower, upper, maturity, is_call=True)


def DoubleBarrierKnockOutPut(
    strike: float, lower: float, upper: float, maturity: float
) -> DoubleBarrierOption:
    """Create a double knock-out put."""
    return DoubleBarrierOption(strike, lower, upper, maturity, is_call=False)


def DoubleBarrierKnockInCall(
    strike: float, lower: float, upper: float, maturity: float
) -> DoubleBarrierOption:
    """Create a double knock-in call."""
    return DoubleBarrierOption(
        strike, lower, upper, maturity, is_call=True, is_knock_in=True
    )


def DoubleBarrierKnockInPut(
    strike: float, lower: float, upper: float, maturity: float
) -> DoubleBarrierOption:
    """Create a double knock-in put."""
    return DoubleBarrierOption(
        strike, lower, upper, maturity, is_call=False, is_knock_in=True
    )


# =============================================================================
# DISCRETE-BARRIER OPTION (Broadie-Glasserman-Kou 1997 continuity correction)
# =============================================================================


class DiscreteBarrierOption(FrozenInstrument, Instrument):
    """
    Discretely-monitored single-barrier option.

    Priced by the continuous Reiner-Rubinstein formula with the barrier shifted
    away from spot by the Broadie-Glasserman-Kou correction
    (``H * exp(+/- 0.5826 * sigma * sqrt(T / m))``), an accurate approximation
    to the option monitored at ``monitoring_points`` equally-spaced dates.

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike price (> 0).
    barrier : float
        Barrier level H (> 0).
    maturity : float
        Time to expiration in years (> 0).
    is_call : bool
        True for call, False for put.
    is_up : bool
        True for up-barrier, False for down-barrier.
    is_knock_in : bool
        True for knock-in, False for knock-out (default).
    monitoring_points : int
        Number of equally-spaced monitoring dates (> 0).
    rebate : float
        Rebate paid at knockout (default 0.0, >= 0).
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).
    """

    __slots__ = (
        "_strike",
        "_barrier",
        "_maturity",
        "_is_call",
        "_is_up",
        "_is_knock_in",
        "_monitoring_points",
        "_rebate",
        "_exercise",
        "_payoff",
    )

    def __init__(
        self,
        strike: float,
        barrier: float,
        maturity: float,
        is_call: bool = True,
        is_up: bool = True,
        is_knock_in: bool = False,
        monitoring_points: int = 252,
        rebate: float = 0.0,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if barrier <= 0:
            raise ValueError(f"Barrier must be positive, got {barrier}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if monitoring_points <= 0:
            raise ValueError(
                f"monitoring_points must be positive, got {monitoring_points}"
            )
        if rebate < 0:
            raise ValueError(f"Rebate must be non-negative, got {rebate}")

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_barrier", barrier)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_is_up", is_up)
        object.__setattr__(self, "_is_knock_in", is_knock_in)
        object.__setattr__(self, "_monitoring_points", monitoring_points)
        object.__setattr__(self, "_rebate", rebate)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "DiscreteBarrierOption",
                "discrete barrier uses the BGK-corrected continuous formula",
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
    def maturity(self) -> float:
        """Time to expiration in years."""
        return self._maturity

    @property
    def is_call(self) -> bool:
        """True for call, False for put."""
        return self._is_call

    @property
    def is_up(self) -> bool:
        """True for up-barrier, False for down-barrier."""
        return self._is_up

    @property
    def is_knock_in(self) -> bool:
        """True for knock-in, False for knock-out."""
        return self._is_knock_in

    @property
    def monitoring_points(self) -> int:
        """Number of equally-spaced monitoring dates."""
        return self._monitoring_points

    @property
    def rebate(self) -> float:
        """Rebate paid at knockout."""
        return self._rebate

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
        direction = "up" if self._is_up else "down"
        knock = "in" if self._is_knock_in else "out"
        opt = "call" if self._is_call else "put"
        return f"discrete_barrier_{direction}_{knock}_{opt}"

    def __repr__(self) -> str:
        opt_type = "Call" if self._is_call else "Put"
        direction = "Up" if self._is_up else "Down"
        knock = "In" if self._is_knock_in else "Out"
        return (
            f"DiscreteBarrierOption({direction}{knock}{opt_type}, K={self._strike}, "
            f"H={self._barrier}, T={self._maturity}, m={self._monitoring_points})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DiscreteBarrierOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._barrier == other._barrier
            and self._maturity == other._maturity
            and self._is_call == other._is_call
            and self._is_up == other._is_up
            and self._is_knock_in == other._is_knock_in
            and self._monitoring_points == other._monitoring_points
            and self._rebate == other._rebate
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._strike,
                self._barrier,
                self._maturity,
                self._is_call,
                self._is_up,
                self._is_knock_in,
                self._monitoring_points,
                self._rebate,
                self._exercise,
            )
        )
