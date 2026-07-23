"""Complex chooser option (Rubinstein 1991).

Part of the Haug exotic-options catalog; see the package overview in
``backend.instruments.exotic_advanced``.

Author: Thomas Vaudescal
"""

from __future__ import annotations

from backend.core.interfaces import Instrument, Payoff
from backend.instruments._frozen import FrozenInstrument
from backend.core.result_types import ExerciseStyle
from backend.instruments.payoffs import AnalyticalOnlyPayoff

__all__ = ["ComplexChooserOption"]


class ComplexChooserOption(FrozenInstrument, Instrument):
    """
    Complex chooser option (Rubinstein 1991) on a single asset.

    At the choice date ``choice_time`` the holder decides whether the option
    becomes a call (strike ``call_strike``, maturity ``call_maturity``) or a put
    (strike ``put_strike``, maturity ``put_maturity``); the call and put may
    differ in BOTH strike and maturity. Priced by the Rubinstein (1991) closed
    form (Haug 4.12.2), whose critical asset price at the choice date is found by
    a safeguarded bisection. When the two legs share a strike and a maturity the
    price collapses to the simple chooser.

    Immutable after construction.

    Parameters
    ----------
    call_strike : float
        Strike of the call leg ``Kc`` (> 0).
    put_strike : float
        Strike of the put leg ``Kp`` (> 0).
    call_maturity : float
        Time to maturity of the call leg ``Tc`` (> choice_time).
    put_maturity : float
        Time to maturity of the put leg ``Tp`` (> choice_time).
    choice_time : float
        Choice date ``t`` (``0 < t < min(call_maturity, put_maturity)``).
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).

    Examples
    --------
    cc = ComplexChooserOption(55, 48, 0.5, 0.5833, 0.25)
    """

    __slots__ = (
        "_call_strike",
        "_put_strike",
        "_call_maturity",
        "_put_maturity",
        "_choice_time",
        "_exercise",
        "_payoff",
    )

    def __init__(
        self,
        call_strike: float,
        put_strike: float,
        call_maturity: float,
        put_maturity: float,
        choice_time: float,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if call_strike <= 0:
            raise ValueError(f"Call strike must be positive, got {call_strike}")
        if put_strike <= 0:
            raise ValueError(f"Put strike must be positive, got {put_strike}")
        if call_maturity <= 0:
            raise ValueError(f"Call maturity must be positive, got {call_maturity}")
        if put_maturity <= 0:
            raise ValueError(f"Put maturity must be positive, got {put_maturity}")
        if not (0 < choice_time < min(call_maturity, put_maturity)):
            raise ValueError(
                "choice_time must satisfy 0 < choice_time < min(call_maturity, "
                f"put_maturity), got {choice_time}"
            )

        object.__setattr__(self, "_call_strike", call_strike)
        object.__setattr__(self, "_put_strike", put_strike)
        object.__setattr__(self, "_call_maturity", call_maturity)
        object.__setattr__(self, "_put_maturity", put_maturity)
        object.__setattr__(self, "_choice_time", choice_time)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "ComplexChooserOption",
                "complex chooser uses the Rubinstein closed-form",
            ),
        )

    @property
    def call_strike(self) -> float:
        """Strike of the call leg Kc."""
        return self._call_strike

    @property
    def put_strike(self) -> float:
        """Strike of the put leg Kp."""
        return self._put_strike

    @property
    def call_maturity(self) -> float:
        """Time to maturity of the call leg Tc."""
        return self._call_maturity

    @property
    def put_maturity(self) -> float:
        """Time to maturity of the put leg Tp."""
        return self._put_maturity

    @property
    def choice_time(self) -> float:
        """Choice date."""
        return self._choice_time

    @property
    def maturity(self) -> float:
        """Time to expiration (the longer of the two legs) -- Instrument interface."""
        return max(self._call_maturity, self._put_maturity)

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
        return "complex_chooser"

    def __repr__(self) -> str:
        return (
            f"ComplexChooserOption(Kc={self._call_strike}, Kp={self._put_strike}, "
            f"Tc={self._call_maturity}, Tp={self._put_maturity}, "
            f"t={self._choice_time})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ComplexChooserOption):
            return NotImplemented
        return (
            self._call_strike == other._call_strike
            and self._put_strike == other._put_strike
            and self._call_maturity == other._call_maturity
            and self._put_maturity == other._put_maturity
            and self._choice_time == other._choice_time
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._call_strike,
                self._put_strike,
                self._call_maturity,
                self._put_maturity,
                self._choice_time,
                self._exercise,
            )
        )
