"""Compound option / option on an option (Geske 1979).

Part of the Haug exotic-options catalog; see the package overview in
``backend.instruments.exotic_advanced``.

Author: Thomas Vaudescal
"""

from __future__ import annotations

from backend.core.interfaces import Instrument, Payoff
from backend.instruments._frozen import FrozenInstrument
from backend.core.result_types import ExerciseStyle
from backend.instruments.payoffs import AnalyticalOnlyPayoff

__all__ = ["CompoundOption"]


class CompoundOption(FrozenInstrument, Instrument):
    """
    Compound option (option on an option, Geske 1979) on a single asset.

    The underlying is itself a vanilla option (strike ``strike1``, expiry
    ``maturity``). At the compound expiry ``t1`` the holder may take a position
    in that underlying option for the compound strike ``strike2``. Four types via
    ``is_call_on`` (the compound) and ``is_call_underlying`` (the underlying):
    call-on-call, put-on-call, call-on-put, put-on-put. Priced by the Geske
    (1979) closed form (Haug 4.13), whose critical asset price at ``t1`` is found
    by a safeguarded bisection.

    Immutable after construction.

    Parameters
    ----------
    strike1 : float
        Strike of the underlying option ``K1`` (> 0).
    strike2 : float
        Strike of the compound option ``K2`` (> 0), paid/received at ``t1``.
    t1 : float
        Compound expiry (``0 < t1 < maturity``).
    maturity : float
        Underlying-option expiry ``T2`` (> 0).
    is_call_on : bool
        The compound option is a call-on (True) or put-on (False).
    is_call_underlying : bool
        The underlying option is a call (True) or put (False).
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).

    Examples
    --------
    poc = CompoundOption(520, 50, 0.25, 0.5, is_call_on=False, is_call_underlying=True)
    """

    __slots__ = (
        "_strike1",
        "_strike2",
        "_t1",
        "_maturity",
        "_is_call_on",
        "_is_call_underlying",
        "_exercise",
        "_payoff",
    )

    def __init__(
        self,
        strike1: float,
        strike2: float,
        t1: float,
        maturity: float,
        is_call_on: bool = True,
        is_call_underlying: bool = True,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike1 <= 0:
            raise ValueError(f"Underlying strike must be positive, got {strike1}")
        if strike2 <= 0:
            raise ValueError(f"Compound strike must be positive, got {strike2}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if not (0 < t1 < maturity):
            raise ValueError(f"t1 must satisfy 0 < t1 < maturity, got {t1}")

        object.__setattr__(self, "_strike1", strike1)
        object.__setattr__(self, "_strike2", strike2)
        object.__setattr__(self, "_t1", t1)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_is_call_on", is_call_on)
        object.__setattr__(self, "_is_call_underlying", is_call_underlying)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "CompoundOption",
                "compound option uses the Geske closed-form",
            ),
        )

    @property
    def strike1(self) -> float:
        """Strike of the underlying option K1."""
        return self._strike1

    @property
    def strike2(self) -> float:
        """Strike of the compound option K2."""
        return self._strike2

    @property
    def t1(self) -> float:
        """Compound expiry."""
        return self._t1

    @property
    def maturity(self) -> float:
        """Underlying-option expiry T2 -- Instrument interface."""
        return self._maturity

    @property
    def is_call_on(self) -> bool:
        """True if the compound option is a call-on, False if put-on."""
        return self._is_call_on

    @property
    def is_call_underlying(self) -> bool:
        """True if the underlying option is a call, False if a put."""
        return self._is_call_underlying

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
        on = "call" if self._is_call_on else "put"
        under = "call" if self._is_call_underlying else "put"
        return f"compound_{on}_on_{under}"

    def __repr__(self) -> str:
        on = "Call" if self._is_call_on else "Put"
        under = "Call" if self._is_call_underlying else "Put"
        return (
            f"CompoundOption({on}On{under}, K1={self._strike1}, "
            f"K2={self._strike2}, t1={self._t1}, T2={self._maturity})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompoundOption):
            return NotImplemented
        return (
            self._strike1 == other._strike1
            and self._strike2 == other._strike2
            and self._t1 == other._t1
            and self._maturity == other._maturity
            and self._is_call_on == other._is_call_on
            and self._is_call_underlying == other._is_call_underlying
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._strike1,
                self._strike2,
                self._t1,
                self._maturity,
                self._is_call_on,
                self._is_call_underlying,
                self._exercise,
            )
        )
