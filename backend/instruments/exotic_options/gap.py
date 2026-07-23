"""
Gap option with separate payment/trigger strikes.
"""

from __future__ import annotations

from backend.core.interfaces import Instrument, Payoff
from backend.instruments._frozen import FrozenInstrument
from backend.core.result_types import ExerciseStyle
from backend.instruments.payoffs import (
    AnalyticalOnlyPayoff,
)


class GapOption(FrozenInstrument, Instrument):
    """
    Gap option.

    Has separate trigger strike (K2) and payment strike (K1).
    Call payoff: (S_T - K1) if S_T > K2, else 0.
    Note: payoff can be negative when K1 > K2 and K2 < S_T < K1.

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Payment strike K1 (must be positive)
    trigger : float
        Trigger strike K2 (must be positive)
    maturity : float
        Time to expiration in years (must be positive)
    is_call : bool
        True for call, False for put
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN)

    Examples
    --------
    gap_call = GapOption(strike=105, trigger=100, maturity=0.5, is_call=True)
    """

    __slots__ = ("_strike", "_trigger", "_maturity", "_is_call", "_exercise", "_payoff")

    def __init__(
        self,
        strike: float,
        trigger: float,
        maturity: float,
        is_call: bool,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if trigger <= 0:
            raise ValueError(f"Trigger must be positive, got {trigger}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_trigger", trigger)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "GapOption",
                "gap option uses dual-strike Black-Scholes, no MC payoff",
            ),
        )

    @property
    def strike(self) -> float:
        """Payment strike K1."""
        return self._strike

    @property
    def trigger(self) -> float:
        """Trigger strike K2."""
        return self._trigger

    @property
    def maturity(self) -> float:
        """Time to expiration in years."""
        return self._maturity

    @property
    def is_call(self) -> bool:
        """True for call, False for put."""
        return self._is_call

    @property
    def payoff(self) -> Payoff:
        """The payoff function (an :class:`AnalyticalOnlyPayoff` for this
        instrument since pricing is analytical only)."""
        return self._payoff

    @property
    def exercise_style(self) -> ExerciseStyle:
        """Exercise style (Instrument interface)."""
        return self._exercise

    @property
    def option_type(self) -> str:
        """String representation of option type."""
        return "gap_call" if self._is_call else "gap_put"

    def __repr__(self) -> str:
        opt_type = "Call" if self._is_call else "Put"
        return (
            f"GapOption({opt_type}, K1={self._strike}, K2={self._trigger}, "
            f"T={self._maturity})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GapOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._trigger == other._trigger
            and self._maturity == other._maturity
            and self._is_call == other._is_call
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (self._strike, self._trigger, self._maturity, self._is_call, self._exercise)
        )


def GapCall(strike: float, trigger: float, maturity: float) -> GapOption:
    """Create a gap call option."""
    return GapOption(strike=strike, trigger=trigger, maturity=maturity, is_call=True)


def GapPut(strike: float, trigger: float, maturity: float) -> GapOption:
    """Create a gap put option."""
    return GapOption(strike=strike, trigger=trigger, maturity=maturity, is_call=False)
