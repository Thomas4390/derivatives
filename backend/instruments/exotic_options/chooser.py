"""
Simple chooser option (Rubinstein 1991).
"""

from __future__ import annotations

from backend.core.interfaces import Instrument, Payoff
from backend.instruments._frozen import FrozenInstrument
from backend.core.result_types import ExerciseStyle
from backend.instruments.payoffs import (
    AnalyticalOnlyPayoff,
)


class ChooserOption(FrozenInstrument, Instrument):
    """
    Simple chooser option (Rubinstein 1991).

    At choice time t_c, the holder chooses max(Call, Put).
    Uses the identity: V = BS_call(S, K, T) + BS_put(S, K*exp(-(r-q)*(T-t_c)), t_c)

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike price (must be positive)
    maturity : float
        Time to expiration in years (must be positive)
    choice_time : float
        Time at which the holder chooses call or put (must be 0 < choice_time <= maturity)
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN)

    Examples
    --------
    chooser = ChooserOption(strike=100, maturity=1.0, choice_time=0.5)
    """

    __slots__ = ("_strike", "_maturity", "_choice_time", "_exercise", "_payoff")

    def __init__(
        self,
        strike: float,
        maturity: float,
        choice_time: float,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if choice_time <= 0 or choice_time > maturity:
            raise ValueError(
                f"Choice time must be in (0, maturity], got {choice_time} "
                f"with maturity={maturity}"
            )

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_choice_time", choice_time)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "ChooserOption",
                "Rubinstein (1991) closed-form via call/put combination",
            ),
        )

    @property
    def strike(self) -> float:
        """Strike price."""
        return self._strike

    @property
    def maturity(self) -> float:
        """Time to expiration in years."""
        return self._maturity

    @property
    def choice_time(self) -> float:
        """Time at which the holder chooses call or put."""
        return self._choice_time

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
        return "chooser"

    def __repr__(self) -> str:
        return (
            f"ChooserOption(K={self._strike}, T={self._maturity}, "
            f"t_c={self._choice_time})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChooserOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._maturity == other._maturity
            and self._choice_time == other._choice_time
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash((self._strike, self._maturity, self._choice_time, self._exercise))


def Chooser(strike: float, maturity: float, choice_time: float) -> ChooserOption:
    """Create a chooser option."""
    return ChooserOption(strike=strike, maturity=maturity, choice_time=choice_time)
