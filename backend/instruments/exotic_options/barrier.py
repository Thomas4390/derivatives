"""
Single-barrier knock-in/knock-out options (all 8 orientations).
"""

from __future__ import annotations

from backend.core.interfaces import Instrument, Payoff
from backend.instruments._frozen import FrozenInstrument
from backend.core.result_types import ExerciseStyle
from backend.instruments.payoffs import (
    AnalyticalOnlyPayoff,
    BarrierDownOutPutPayoff,
    BarrierUpOutCallPayoff,
)


class BarrierOption(FrozenInstrument, Instrument):
    """
    Barrier option (knock-in or knock-out).

    Knock-out: becomes worthless if the price touches the barrier.
    Knock-in: only activates if the price touches the barrier.

    Supports all 8 barrier types:
    - Up-and-Out Call/Put, Down-and-Out Call/Put
    - Up-and-In Call/Put, Down-and-In Call/Put

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike price (must be positive)
    barrier : float
        Barrier level (must be positive)
    maturity : float
        Time to expiration in years (must be positive)
    is_call : bool
        True for call, False for put
    is_up : bool
        True for up barrier, False for down barrier
    is_knock_in : bool
        True for knock-in, False for knock-out (default False)
    rebate : float
        Rebate paid at knockout (default 0.0, must be >= 0)
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN)

    Examples
    --------
    barrier_call = BarrierOption(strike=100, barrier=120, maturity=0.5,
                                  is_call=True, is_up=True)
    knock_in = BarrierOption(strike=100, barrier=90, maturity=0.5,
                              is_call=True, is_up=False, is_knock_in=True)
    """

    __slots__ = (
        "_strike",
        "_barrier",
        "_maturity",
        "_is_call",
        "_is_up",
        "_is_knock_in",
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
        rebate: float = 0.0,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if barrier <= 0:
            raise ValueError(f"Barrier must be positive, got {barrier}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if rebate < 0:
            raise ValueError(f"Rebate must be non-negative, got {rebate}")

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_barrier", barrier)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_is_up", is_up)
        object.__setattr__(self, "_is_knock_in", is_knock_in)
        object.__setattr__(self, "_rebate", rebate)
        object.__setattr__(self, "_exercise", exercise)

        # Cache the payoff object for MC-supported combinations; use
        # AnalyticalOnlyPayoff for types only supported by the analytical engine.
        cached_payoff: Payoff
        if is_knock_in:
            direction = "up" if is_up else "down"
            opt = "call" if is_call else "put"
            cached_payoff = AnalyticalOnlyPayoff(
                f"BarrierOption({direction}-and-in-{opt})",
                "knock-in barriers are priced analytically (reflection principle)",
            )
        elif is_up and is_call:
            cached_payoff = BarrierUpOutCallPayoff(strike, barrier)
        elif not is_up and not is_call:
            cached_payoff = BarrierDownOutPutPayoff(strike, barrier)
        else:
            # Knock-out types without MC payoff (up-out put, down-out call)
            direction = "up" if is_up else "down"
            opt = "call" if is_call else "put"
            cached_payoff = AnalyticalOnlyPayoff(
                f"BarrierOption({direction}-and-out-{opt})",
                "this barrier orientation has no MC payoff implementation",
            )
        object.__setattr__(self, "_payoff", cached_payoff)

    @property
    def strike(self) -> float:
        """Strike price."""
        return self._strike

    @property
    def barrier(self) -> float:
        """Barrier level."""
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
        """True for up barrier, False for down barrier."""
        return self._is_up

    @property
    def is_knock_in(self) -> bool:
        """True for knock-in, False for knock-out."""
        return self._is_knock_in

    @property
    def rebate(self) -> float:
        """Rebate paid at knockout."""
        return self._rebate

    @property
    def payoff(self) -> Payoff:
        """The payoff function.

        For analytical-only barrier orientations (knock-in, up-out put,
        down-out call), returns an :class:`AnalyticalOnlyPayoff`.
        """
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
        return f"barrier_{direction}_{knock}_{opt}"

    def __repr__(self) -> str:
        opt_type = "Call" if self._is_call else "Put"
        direction = "Up" if self._is_up else "Down"
        knock = "In" if self._is_knock_in else "Out"
        rebate_str = f", R={self._rebate}" if self._rebate > 0 else ""
        return f"BarrierOption({direction}{knock}{opt_type}, K={self._strike}, B={self._barrier}, T={self._maturity}{rebate_str})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BarrierOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._barrier == other._barrier
            and self._maturity == other._maturity
            and self._is_call == other._is_call
            and self._is_up == other._is_up
            and self._is_knock_in == other._is_knock_in
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
                self._rebate,
                self._exercise,
            )
        )


def BarrierUpOutCall(
    strike: float, barrier: float, maturity: float, rebate: float = 0.0
) -> BarrierOption:
    """Create an up-and-out call option."""
    return BarrierOption(
        strike=strike,
        barrier=barrier,
        maturity=maturity,
        is_call=True,
        is_up=True,
        is_knock_in=False,
        rebate=rebate,
    )


def BarrierUpInCall(strike: float, barrier: float, maturity: float) -> BarrierOption:
    """Create an up-and-in call option."""
    return BarrierOption(
        strike=strike,
        barrier=barrier,
        maturity=maturity,
        is_call=True,
        is_up=True,
        is_knock_in=True,
    )


def BarrierDownOutCall(
    strike: float, barrier: float, maturity: float, rebate: float = 0.0
) -> BarrierOption:
    """Create a down-and-out call option."""
    return BarrierOption(
        strike=strike,
        barrier=barrier,
        maturity=maturity,
        is_call=True,
        is_up=False,
        is_knock_in=False,
        rebate=rebate,
    )


def BarrierDownInCall(strike: float, barrier: float, maturity: float) -> BarrierOption:
    """Create a down-and-in call option."""
    return BarrierOption(
        strike=strike,
        barrier=barrier,
        maturity=maturity,
        is_call=True,
        is_up=False,
        is_knock_in=True,
    )


def BarrierUpOutPut(
    strike: float, barrier: float, maturity: float, rebate: float = 0.0
) -> BarrierOption:
    """Create an up-and-out put option."""
    return BarrierOption(
        strike=strike,
        barrier=barrier,
        maturity=maturity,
        is_call=False,
        is_up=True,
        is_knock_in=False,
        rebate=rebate,
    )


def BarrierUpInPut(strike: float, barrier: float, maturity: float) -> BarrierOption:
    """Create an up-and-in put option."""
    return BarrierOption(
        strike=strike,
        barrier=barrier,
        maturity=maturity,
        is_call=False,
        is_up=True,
        is_knock_in=True,
    )


def BarrierDownOutPut(
    strike: float, barrier: float, maturity: float, rebate: float = 0.0
) -> BarrierOption:
    """Create a down-and-out put option."""
    return BarrierOption(
        strike=strike,
        barrier=barrier,
        maturity=maturity,
        is_call=False,
        is_up=False,
        is_knock_in=False,
        rebate=rebate,
    )


def BarrierDownInPut(strike: float, barrier: float, maturity: float) -> BarrierOption:
    """Create a down-and-in put option."""
    return BarrierOption(
        strike=strike,
        barrier=barrier,
        maturity=maturity,
        is_call=False,
        is_up=False,
        is_knock_in=True,
    )
