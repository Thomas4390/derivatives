"""
Power option on S^n (Esser 2003).
"""

from __future__ import annotations

from backend.core.interfaces import Instrument, Payoff
from backend.instruments._frozen import FrozenInstrument
from backend.core.result_types import ExerciseStyle
from backend.instruments.payoffs import (
    AnalyticalOnlyPayoff,
)


class PowerOption(FrozenInstrument, Instrument):
    """
    Power option.

    Option on S^n with payoff max(S_T^n - K, 0) for calls.
    Uses adjusted drift and volatility for pricing.

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike price (must be positive)
    maturity : float
        Time to expiration in years (must be positive)
    is_call : bool
        True for call, False for put
    power : float
        Power exponent n (must be positive)
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN)

    Examples
    --------
    power_call = PowerOption(strike=10000, maturity=0.5, is_call=True, power=2)
    """

    __slots__ = ("_strike", "_maturity", "_is_call", "_power", "_exercise", "_payoff")

    def __init__(
        self,
        strike: float,
        maturity: float,
        is_call: bool,
        power: float,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if power <= 0:
            raise ValueError(f"Power must be positive, got {power}")

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_power", power)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "PowerOption",
                "power option uses adjusted-drift closed-form, no MC payoff",
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
    def is_call(self) -> bool:
        """True for call, False for put."""
        return self._is_call

    @property
    def power(self) -> float:
        """Power exponent n."""
        return self._power

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
        return "power_call" if self._is_call else "power_put"

    def __repr__(self) -> str:
        opt_type = "Call" if self._is_call else "Put"
        return (
            f"PowerOption({opt_type}, K={self._strike}, n={self._power}, "
            f"T={self._maturity})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PowerOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._maturity == other._maturity
            and self._is_call == other._is_call
            and self._power == other._power
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (self._strike, self._maturity, self._is_call, self._power, self._exercise)
        )


def PowerCall(strike: float, maturity: float, power: float) -> PowerOption:
    """Create a power call option."""
    return PowerOption(strike=strike, maturity=maturity, is_call=True, power=power)


def PowerPut(strike: float, maturity: float, power: float) -> PowerOption:
    """Create a power put option."""
    return PowerOption(strike=strike, maturity=maturity, is_call=False, power=power)
