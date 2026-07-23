"""
Asset-or-nothing binary option (Cox-Ross).
"""

from __future__ import annotations

from backend.core.interfaces import Instrument, Payoff
from backend.instruments._frozen import FrozenInstrument
from backend.core.result_types import ExerciseStyle
from backend.instruments.payoffs import (
    AnalyticalOnlyPayoff,
)


class AssetOrNothingOption(FrozenInstrument, Instrument):
    """
    Asset-or-nothing option.

    Pays S_T if the option expires ITM (vs cash-or-nothing which pays a fixed amount).
    Call: S_T if S_T > K, else 0. Put: S_T if S_T < K, else 0.

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike price (must be positive)
    maturity : float
        Time to expiration in years (must be positive)
    is_call : bool
        True for call, False for put
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN)

    Examples
    --------
    aon_call = AssetOrNothingOption(strike=100, maturity=0.5, is_call=True)
    """

    __slots__ = ("_strike", "_maturity", "_is_call", "_exercise", "_payoff")

    def __init__(
        self,
        strike: float,
        maturity: float,
        is_call: bool,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "AssetOrNothingOption",
                "asset-or-nothing has Cox-Ross closed-form, no MC payoff",
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
        return "asset_or_nothing_call" if self._is_call else "asset_or_nothing_put"

    def __repr__(self) -> str:
        opt_type = "Call" if self._is_call else "Put"
        return f"AssetOrNothingOption({opt_type}, K={self._strike}, T={self._maturity})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AssetOrNothingOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._maturity == other._maturity
            and self._is_call == other._is_call
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash((self._strike, self._maturity, self._is_call, self._exercise))


def AssetOrNothingCall(strike: float, maturity: float) -> AssetOrNothingOption:
    """Create an asset-or-nothing call option."""
    return AssetOrNothingOption(strike=strike, maturity=maturity, is_call=True)


def AssetOrNothingPut(strike: float, maturity: float) -> AssetOrNothingOption:
    """Create an asset-or-nothing put option."""
    return AssetOrNothingOption(strike=strike, maturity=maturity, is_call=False)
