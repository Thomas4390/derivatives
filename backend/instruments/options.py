"""
Option Instruments
==================

Concrete option classes wrapping payoffs with exercise style and maturity.

Instruments are IMMUTABLE after construction.

Author: Thomas
Created: 2025
"""

from backend.core.interfaces import Instrument, Payoff
from backend.core.result_types import ExerciseStyle
from backend.instruments.payoffs import (
    VanillaCallPayoff,
    VanillaPutPayoff,
    DigitalCallPayoff,
    DigitalPutPayoff,
)


# =============================================================================
# VANILLA OPTIONS
# =============================================================================

class VanillaOption(Instrument):
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

    __slots__ = ('_strike', '_maturity', '_is_call', '_exercise')

    def __init__(
        self,
        strike: float,
        maturity: float,
        is_call: bool,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")

        object.__setattr__(self, '_strike', strike)
        object.__setattr__(self, '_maturity', maturity)
        object.__setattr__(self, '_is_call', is_call)
        object.__setattr__(self, '_exercise', exercise)

    def __setattr__(self, name, value):
        raise AttributeError("VanillaOption is immutable")

    def __delattr__(self, name):
        raise AttributeError("VanillaOption is immutable")

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
        """The payoff function."""
        if self._is_call:
            return VanillaCallPayoff(self._strike)
        return VanillaPutPayoff(self._strike)

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

    def __eq__(self, other) -> bool:
        if not isinstance(other, VanillaOption):
            return NotImplemented
        return (
            self._strike == other._strike and
            self._maturity == other._maturity and
            self._is_call == other._is_call and
            self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash((self._strike, self._maturity, self._is_call, self._exercise))


# =============================================================================
# DIGITAL OPTIONS
# =============================================================================

class DigitalOption(Instrument):
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

    __slots__ = ('_strike', '_maturity', '_is_call', '_payout', '_exercise')

    def __init__(
        self,
        strike: float,
        maturity: float,
        is_call: bool,
        payout: float = 1.0,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if payout <= 0:
            raise ValueError(f"Payout must be positive, got {payout}")

        object.__setattr__(self, '_strike', strike)
        object.__setattr__(self, '_maturity', maturity)
        object.__setattr__(self, '_is_call', is_call)
        object.__setattr__(self, '_payout', payout)
        object.__setattr__(self, '_exercise', exercise)

    def __setattr__(self, name, value):
        raise AttributeError("DigitalOption is immutable")

    def __delattr__(self, name):
        raise AttributeError("DigitalOption is immutable")

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
        """The payoff function."""
        if self._is_call:
            return DigitalCallPayoff(self._strike, self._payout)
        return DigitalPutPayoff(self._strike, self._payout)

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

    def __eq__(self, other) -> bool:
        if not isinstance(other, DigitalOption):
            return NotImplemented
        return (
            self._strike == other._strike and
            self._maturity == other._maturity and
            self._is_call == other._is_call and
            self._payout == other._payout and
            self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash((self._strike, self._maturity, self._is_call, self._payout, self._exercise))


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def EuropeanCall(strike: float, maturity: float) -> VanillaOption:
    """
    Factory for European call option.

    Parameters
    ----------
    strike : float
        Strike price
    maturity : float
        Time to expiration in years

    Returns
    -------
    VanillaOption
        European call option

    Examples
    --------
    call = EuropeanCall(strike=100, maturity=0.5)
    """
    return VanillaOption(strike=strike, maturity=maturity, is_call=True)


def EuropeanPut(strike: float, maturity: float) -> VanillaOption:
    """
    Factory for European put option.

    Parameters
    ----------
    strike : float
        Strike price
    maturity : float
        Time to expiration in years

    Returns
    -------
    VanillaOption
        European put option

    Examples
    --------
    put = EuropeanPut(strike=100, maturity=0.5)
    """
    return VanillaOption(strike=strike, maturity=maturity, is_call=False)


def AmericanCall(strike: float, maturity: float) -> VanillaOption:
    """
    Factory for American call option.

    Parameters
    ----------
    strike : float
        Strike price
    maturity : float
        Time to expiration in years

    Returns
    -------
    VanillaOption
        American call option

    Examples
    --------
    call = AmericanCall(strike=100, maturity=0.5)
    """
    return VanillaOption(
        strike=strike,
        maturity=maturity,
        is_call=True,
        exercise=ExerciseStyle.AMERICAN,
    )


def AmericanPut(strike: float, maturity: float) -> VanillaOption:
    """
    Factory for American put option.

    Parameters
    ----------
    strike : float
        Strike price
    maturity : float
        Time to expiration in years

    Returns
    -------
    VanillaOption
        American put option

    Examples
    --------
    put = AmericanPut(strike=100, maturity=0.5)
    """
    return VanillaOption(
        strike=strike,
        maturity=maturity,
        is_call=False,
        exercise=ExerciseStyle.AMERICAN,
    )


def BermudanCall(strike: float, maturity: float) -> VanillaOption:
    """
    Factory for Bermudan call option.

    Parameters
    ----------
    strike : float
        Strike price
    maturity : float
        Time to expiration in years

    Returns
    -------
    VanillaOption
        Bermudan call option
    """
    return VanillaOption(
        strike=strike,
        maturity=maturity,
        is_call=True,
        exercise=ExerciseStyle.BERMUDAN,
    )


def BermudanPut(strike: float, maturity: float) -> VanillaOption:
    """
    Factory for Bermudan put option.

    Parameters
    ----------
    strike : float
        Strike price
    maturity : float
        Time to expiration in years

    Returns
    -------
    VanillaOption
        Bermudan put option
    """
    return VanillaOption(
        strike=strike,
        maturity=maturity,
        is_call=False,
        exercise=ExerciseStyle.BERMUDAN,
    )


if __name__ == "__main__":
    import numpy as np

    print("=" * 50)
    print("Options Module Smoke Test")
    print("=" * 50)

    # European options
    euro_call = EuropeanCall(strike=100.0, maturity=0.5)
    euro_put = EuropeanPut(strike=100.0, maturity=0.5)

    print(f"\nEuropean Call: {euro_call}")
    print(f"European Put: {euro_put}")

    # American options
    amer_call = AmericanCall(strike=100.0, maturity=0.5)
    amer_put = AmericanPut(strike=100.0, maturity=0.5)

    print(f"\nAmerican Call: {amer_call}")
    print(f"American Put: {amer_put}")

    # Bermudan options
    berm_call = BermudanCall(strike=100.0, maturity=0.5)
    berm_put = BermudanPut(strike=100.0, maturity=0.5)

    print(f"\nBermudan Call: {berm_call}")
    print(f"Bermudan Put: {berm_put}")

    # Test payoff evaluation
    spots = np.array([90.0, 100.0, 110.0])
    print(f"\nPayoff evaluation at spots {spots}:")
    print(f"  Euro Call payoffs: {euro_call.payoff(spots)}")
    print(f"  Euro Put payoffs: {euro_put.payoff(spots)}")

    # Test exercise style checks
    print(f"\nExercise style checks:")
    print(f"  Euro Call is_european: {euro_call.is_european}")
    print(f"  Amer Put is_american: {amer_put.is_american}")
    print(f"  Berm Call is_bermudan: {berm_call.is_bermudan}")

    # Digital options
    digital_call = DigitalOption(strike=100.0, maturity=0.5, is_call=True, payout=10.0)
    print(f"\nDigital Call: {digital_call}")
    print(f"  Payoffs: {digital_call.payoff(spots)}")

    # Test immutability
    print("\nTesting immutability...")
    try:
        euro_call.strike = 110  # type: ignore
        print("  ERROR: Mutation should have failed!")
    except AttributeError as e:
        print(f"  Correctly prevented mutation: {e}")

    # Test equality and hashing
    print("\nTesting equality and hashing...")
    call1 = EuropeanCall(strike=100, maturity=0.5)
    call2 = EuropeanCall(strike=100, maturity=0.5)
    call3 = EuropeanCall(strike=105, maturity=0.5)
    print(f"  call1 == call2: {call1 == call2}")
    print(f"  call1 == call3: {call1 == call3}")
    print(f"  hash(call1) == hash(call2): {hash(call1) == hash(call2)}")

    print("\n" + "=" * 50)
    print("Options smoke test passed")
    print("=" * 50)
