"""
Asian (average-price) options: arithmetic (MC) + geometric (analytical).
"""

from __future__ import annotations

from backend.core.interfaces import Instrument, Payoff
from backend.instruments._frozen import FrozenInstrument
from backend.core.result_types import ExerciseStyle
from backend.instruments.payoffs import (
    AnalyticalOnlyPayoff,
    AsianCallPayoff,
    AsianPutPayoff,
)


class AsianOption(FrozenInstrument, Instrument):
    """
    Asian option based on average price.

    Supports arithmetic average (MC pricing) and geometric average
    (closed-form analytical pricing via Kemna-Vorst 1990).

    The payoff depends on the average price over the option's life,
    not just the terminal price.

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike price (must be positive)
    maturity : float
        Time to expiration in years (must be positive)
    is_call : bool
        True for call, False for put
    average_type : str
        "arithmetic" or "geometric" (default "arithmetic")
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN)

    Examples
    --------
    asian_call = AsianOption(strike=100, maturity=0.5, is_call=True)
    geo_call = AsianOption(strike=100, maturity=0.5, is_call=True, average_type="geometric")
    """

    __slots__ = (
        "_strike",
        "_maturity",
        "_is_call",
        "_average_type",
        "_exercise",
        "_payoff",
    )

    def __init__(
        self,
        strike: float,
        maturity: float,
        is_call: bool = True,
        average_type: str = "arithmetic",
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if average_type not in ("arithmetic", "geometric"):
            raise ValueError(
                f"average_type must be 'arithmetic' or 'geometric', got '{average_type}'"
            )

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_average_type", average_type)
        object.__setattr__(self, "_exercise", exercise)

        # Cache the payoff object for MC-supported types
        if average_type == "arithmetic":
            if is_call:
                cached_payoff: Payoff = AsianCallPayoff(strike)
            else:
                cached_payoff = AsianPutPayoff(strike)
        else:
            # Geometric average: analytical only, no MC payoff class
            cached_payoff = AnalyticalOnlyPayoff(
                "AsianOption(geometric)",
                "geometric average has closed-form (Kemna-Vorst) but no MC payoff",
            )
        object.__setattr__(self, "_payoff", cached_payoff)

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
    def average_type(self) -> str:
        """Average type: 'arithmetic' or 'geometric'."""
        return self._average_type

    @property
    def payoff(self) -> Payoff:
        """The payoff function.

        For geometric average (analytical-only), returns an
        :class:`AnalyticalOnlyPayoff` whose ``__call__`` raises
        ``NotImplementedError`` with a clear message.
        """
        return self._payoff

    @property
    def exercise_style(self) -> ExerciseStyle:
        """Exercise style (Instrument interface)."""
        return self._exercise

    @property
    def option_type(self) -> str:
        """String representation of option type."""
        avg = "geometric_" if self._average_type == "geometric" else ""
        opt = "call" if self._is_call else "put"
        return f"asian_{avg}{opt}"

    def __repr__(self) -> str:
        opt_type = "Call" if self._is_call else "Put"
        avg_str = "Geometric" if self._average_type == "geometric" else "Arithmetic"
        return f"AsianOption({avg_str}{opt_type}, K={self._strike}, T={self._maturity})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AsianOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._maturity == other._maturity
            and self._is_call == other._is_call
            and self._average_type == other._average_type
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._strike,
                self._maturity,
                self._is_call,
                self._average_type,
                self._exercise,
            )
        )


def AsianCall(strike: float, maturity: float) -> AsianOption:
    """Create an Asian call option."""
    return AsianOption(strike=strike, maturity=maturity, is_call=True)


def AsianPut(strike: float, maturity: float) -> AsianOption:
    """Create an Asian put option."""
    return AsianOption(strike=strike, maturity=maturity, is_call=False)


def AsianGeometricCall(strike: float, maturity: float) -> AsianOption:
    """Create a geometric Asian call option."""
    return AsianOption(
        strike=strike, maturity=maturity, is_call=True, average_type="geometric"
    )


def AsianGeometricPut(strike: float, maturity: float) -> AsianOption:
    """Create a geometric Asian put option."""
    return AsianOption(
        strike=strike, maturity=maturity, is_call=False, average_type="geometric"
    )
