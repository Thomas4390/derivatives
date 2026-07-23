"""
Lookback options: floating-strike (MC) + fixed-strike (analytical).
"""

from __future__ import annotations

from backend.core.interfaces import Instrument, Payoff
from backend.instruments._frozen import FrozenInstrument
from backend.core.result_types import ExerciseStyle
from backend.instruments.payoffs import (
    AnalyticalOnlyPayoff,
    LookbackFloatingCallPayoff,
    LookbackFloatingPutPayoff,
)


class LookbackOption(FrozenInstrument, Instrument):
    """
    Lookback option (floating or fixed strike).

    Floating strike:
        Call: S_T - min(S_t) (buy at the lowest price)
        Put: max(S_t) - S_T (sell at the highest price)

    Fixed strike:
        Call: max(max(S_t) - K, 0)
        Put: max(K - min(S_t), 0)

    Immutable after construction.

    Parameters
    ----------
    maturity : float
        Time to expiration in years (must be positive)
    is_call : bool
        True for call, False for put
    strike : float, optional
        Strike price for fixed-strike lookbacks (must be positive if provided)
    lookback_type : str
        "floating" or "fixed" (default "floating")
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN)

    Examples
    --------
    floating_call = LookbackOption(maturity=0.5, is_call=True)
    fixed_call = LookbackOption(maturity=0.5, is_call=True, strike=100, lookback_type="fixed")
    """

    __slots__ = (
        "_maturity",
        "_is_call",
        "_strike",
        "_lookback_type",
        "_exercise",
        "_payoff",
    )

    def __init__(
        self,
        maturity: float,
        is_call: bool = True,
        strike: float | None = None,
        lookback_type: str = "floating",
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if lookback_type not in ("floating", "fixed"):
            raise ValueError(
                f"lookback_type must be 'floating' or 'fixed', got '{lookback_type}'"
            )
        if lookback_type == "fixed" and (strike is None or strike <= 0):
            raise ValueError(
                f"Fixed-strike lookback requires positive strike, got {strike}"
            )

        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_lookback_type", lookback_type)
        object.__setattr__(self, "_exercise", exercise)

        # Cache the payoff object for MC-supported types
        cached_payoff: Payoff
        if lookback_type == "floating":
            if is_call:
                cached_payoff = LookbackFloatingCallPayoff()
            else:
                cached_payoff = LookbackFloatingPutPayoff()
        else:
            # Fixed-strike lookback: analytical only (Goldman-Sosin-Gatto)
            cached_payoff = AnalyticalOnlyPayoff(
                "LookbackOption(fixed-strike)",
                "fixed-strike lookback uses Goldman-Sosin-Gatto closed-form",
            )
        object.__setattr__(self, "_payoff", cached_payoff)

    @property
    def maturity(self) -> float:
        """Time to expiration in years."""
        return self._maturity

    @property
    def is_call(self) -> bool:
        """True for call, False for put."""
        return self._is_call

    @property
    def strike(self) -> float | None:
        """Strike price (None for floating-strike lookbacks)."""
        return self._strike

    @property
    def lookback_type(self) -> str:
        """Lookback type: 'floating' or 'fixed'."""
        return self._lookback_type

    @property
    def payoff(self) -> Payoff:
        """The payoff function.

        For fixed-strike lookbacks (analytical only), returns an
        :class:`AnalyticalOnlyPayoff`.
        """
        return self._payoff

    @property
    def exercise_style(self) -> ExerciseStyle:
        """Exercise style (Instrument interface)."""
        return self._exercise

    @property
    def option_type(self) -> str:
        """String representation of option type."""
        opt = "call" if self._is_call else "put"
        return f"lookback_{self._lookback_type}_{opt}"

    def __repr__(self) -> str:
        opt_type = "Call" if self._is_call else "Put"
        lb_type = self._lookback_type.capitalize()
        strike_str = f", K={self._strike}" if self._strike is not None else ""
        return f"LookbackOption({lb_type}{opt_type}{strike_str}, T={self._maturity})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LookbackOption):
            return NotImplemented
        return (
            self._maturity == other._maturity
            and self._is_call == other._is_call
            and self._strike == other._strike
            and self._lookback_type == other._lookback_type
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._maturity,
                self._is_call,
                self._strike,
                self._lookback_type,
                self._exercise,
            )
        )


def LookbackCall(maturity: float) -> LookbackOption:
    """Create a lookback call option (floating strike)."""
    return LookbackOption(maturity=maturity, is_call=True)


def LookbackPut(maturity: float) -> LookbackOption:
    """Create a lookback put option (floating strike)."""
    return LookbackOption(maturity=maturity, is_call=False)


def LookbackFixedCall(strike: float, maturity: float) -> LookbackOption:
    """Create a fixed-strike lookback call option."""
    return LookbackOption(
        maturity=maturity, is_call=True, strike=strike, lookback_type="fixed"
    )


def LookbackFixedPut(strike: float, maturity: float) -> LookbackOption:
    """Create a fixed-strike lookback put option."""
    return LookbackOption(
        maturity=maturity, is_call=False, strike=strike, lookback_type="fixed"
    )
