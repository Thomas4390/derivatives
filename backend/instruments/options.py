"""
Option Instruments
==================

Concrete option classes wrapping payoffs with exercise style and maturity.

Instruments are IMMUTABLE after construction.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from backend.core.interfaces import Instrument, Payoff
from backend.instruments._frozen import FrozenInstrument
from backend.core.result_types import ExerciseStyle
from backend.instruments.payoffs import (
    DigitalCallPayoff,
    DigitalPutPayoff,
    VanillaCallPayoff,
    VanillaPutPayoff,
)

# Re-export exotic options for backward compatibility
from backend.instruments.exotic_options import (  # noqa: F401
    AsianCall,
    AsianGeometricCall,
    AsianGeometricPut,
    AsianOption,
    AsianPut,
    AssetOrNothingCall,
    AssetOrNothingOption,
    AssetOrNothingPut,
    BarrierDownInCall,
    BarrierDownInPut,
    BarrierDownOutCall,
    BarrierDownOutPut,
    BarrierOption,
    BarrierUpInCall,
    BarrierUpInPut,
    BarrierUpOutCall,
    BarrierUpOutPut,
    Chooser,
    ChooserOption,
    GapCall,
    GapOption,
    GapPut,
    LookbackCall,
    LookbackFixedCall,
    LookbackFixedPut,
    LookbackOption,
    LookbackPut,
    PowerCall,
    PowerOption,
    PowerPut,
)

# =============================================================================
# VANILLA OPTIONS
# =============================================================================


class VanillaOption(FrozenInstrument, Instrument):
    """
    Vanilla European/American option.

    This is the most common option type. It wraps a vanilla payoff
    with exercise style and maturity.

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
    call = VanillaOption(strike=100, maturity=0.5, is_call=True)
    put = VanillaOption(strike=100, maturity=0.5, is_call=False, exercise=ExerciseStyle.AMERICAN)
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

        # Cache the payoff object for immutability
        if is_call:
            cached_payoff = VanillaCallPayoff(strike)
        else:
            cached_payoff = VanillaPutPayoff(strike)
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
    def exercise(self) -> ExerciseStyle:
        """Exercise style."""
        return self._exercise

    @property
    def payoff(self) -> Payoff:
        """The payoff function (cached for immutability)."""
        return self._payoff

    @property
    def exercise_style(self) -> ExerciseStyle:
        """Exercise style (Instrument interface)."""
        return self._exercise

    @property
    def option_type(self) -> str:
        """String representation of option type."""
        return "call" if self._is_call else "put"

    def __repr__(self) -> str:
        return (
            f"VanillaOption({self.option_type}, K={self._strike}, "
            f"T={self._maturity}, {self._exercise.name})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VanillaOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._maturity == other._maturity
            and self._is_call == other._is_call
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash((self._strike, self._maturity, self._is_call, self._exercise))


# =============================================================================
# DIGITAL OPTIONS
# =============================================================================


class DigitalOption(FrozenInstrument, Instrument):
    """
    Digital (binary) option.

    Pays a fixed amount if the option expires in-the-money.
    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike price
    maturity : float
        Time to expiration in years
    is_call : bool
        True for call, False for put
    payout : float
        Fixed payout amount (default 1.0)
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN)

    Examples
    --------
    digital = DigitalOption(strike=100, maturity=0.5, is_call=True, payout=10)
    """

    __slots__ = ("_strike", "_maturity", "_is_call", "_payout", "_exercise", "_payoff")

    def __init__(
        self,
        strike: float,
        maturity: float,
        is_call: bool,
        payout: float = 1.0,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if payout <= 0:
            raise ValueError(f"Payout must be positive, got {payout}")

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_payout", payout)
        object.__setattr__(self, "_exercise", exercise)

        # Cache the payoff object for immutability
        if is_call:
            cached_payoff = DigitalCallPayoff(strike, payout)
        else:
            cached_payoff = DigitalPutPayoff(strike, payout)
        object.__setattr__(self, "_payoff", cached_payoff)

    @property
    def strike(self) -> float:
        return self._strike

    @property
    def maturity(self) -> float:
        return self._maturity

    @property
    def is_call(self) -> bool:
        return self._is_call

    @property
    def payout(self) -> float:
        return self._payout

    @property
    def exercise(self) -> ExerciseStyle:
        return self._exercise

    @property
    def payoff(self) -> Payoff:
        """The payoff function (cached for immutability)."""
        return self._payoff

    @property
    def exercise_style(self) -> ExerciseStyle:
        """Exercise style (Instrument interface)."""
        return self._exercise

    @property
    def option_type(self) -> str:
        """String representation of option type."""
        return "digital_call" if self._is_call else "digital_put"

    def __repr__(self) -> str:
        return (
            f"DigitalOption({self.option_type}, K={self._strike}, "
            f"T={self._maturity}, payout={self._payout})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DigitalOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._maturity == other._maturity
            and self._is_call == other._is_call
            and self._payout == other._payout
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (self._strike, self._maturity, self._is_call, self._payout, self._exercise)
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_vanilla_option(
    strike: float,
    maturity: float,
    is_call: bool = True,
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
) -> VanillaOption:
    """
    Generic factory for vanilla options.

    This is the primary factory function. Use the convenience aliases
    (EuropeanCall, EuropeanPut, etc.) for simpler usage.

    Parameters
    ----------
    strike : float
        Strike price
    maturity : float
        Time to expiration in years
    is_call : bool, default True
        True for call, False for put
    exercise : ExerciseStyle, default EUROPEAN
        Exercise style (EUROPEAN, AMERICAN, BERMUDAN)

    Returns
    -------
    VanillaOption
        Configured option instrument

    Examples
    --------
    call = create_vanilla_option(100, 0.5, is_call=True)
    american_put = create_vanilla_option(100, 0.5, False, ExerciseStyle.AMERICAN)
    """
    return VanillaOption(
        strike=strike,
        maturity=maturity,
        is_call=is_call,
        exercise=exercise,
    )


# -----------------------------------------------------------------------------
# Convenience Aliases
# -----------------------------------------------------------------------------
# These provide backward-compatible, expressive names for common option types.


def EuropeanCall(strike: float, maturity: float) -> VanillaOption:
    """Create a European call option."""
    return create_vanilla_option(strike, maturity, True, ExerciseStyle.EUROPEAN)


def EuropeanPut(strike: float, maturity: float) -> VanillaOption:
    """Create a European put option."""
    return create_vanilla_option(strike, maturity, False, ExerciseStyle.EUROPEAN)


def AmericanCall(strike: float, maturity: float) -> VanillaOption:
    """Create an American call option."""
    return create_vanilla_option(strike, maturity, True, ExerciseStyle.AMERICAN)


def AmericanPut(strike: float, maturity: float) -> VanillaOption:
    """Create an American put option."""
    return create_vanilla_option(strike, maturity, False, ExerciseStyle.AMERICAN)


def BermudanCall(strike: float, maturity: float) -> VanillaOption:
    """Create a Bermudan call option."""
    return create_vanilla_option(strike, maturity, True, ExerciseStyle.BERMUDAN)


def BermudanPut(strike: float, maturity: float) -> VanillaOption:
    """Create a Bermudan put option."""
    return create_vanilla_option(strike, maturity, False, ExerciseStyle.BERMUDAN)
