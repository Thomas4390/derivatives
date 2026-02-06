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
    AsianCallPayoff,
    AsianPutPayoff,
    BarrierUpOutCallPayoff,
    BarrierDownOutPutPayoff,
    LookbackFloatingCallPayoff,
    LookbackFloatingPutPayoff,
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

    __slots__ = ('_strike', '_maturity', '_is_call', '_exercise', '_payoff')

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

        # Cache the payoff object for immutability
        if is_call:
            cached_payoff = VanillaCallPayoff(strike)
        else:
            cached_payoff = VanillaPutPayoff(strike)
        object.__setattr__(self, '_payoff', cached_payoff)

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

    __slots__ = ('_strike', '_maturity', '_is_call', '_payout', '_exercise', '_payoff')

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

        # Cache the payoff object for immutability
        if is_call:
            cached_payoff = DigitalCallPayoff(strike, payout)
        else:
            cached_payoff = DigitalPutPayoff(strike, payout)
        object.__setattr__(self, '_payoff', cached_payoff)

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
# EXOTIC OPTIONS
# =============================================================================

class AsianOption(Instrument):
    """
    Asian option based on arithmetic average price.

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
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN)

    Examples
    --------
    asian_call = AsianOption(strike=100, maturity=0.5, is_call=True)
    """

    __slots__ = ('_strike', '_maturity', '_is_call', '_exercise', '_payoff')

    def __init__(
        self,
        strike: float,
        maturity: float,
        is_call: bool = True,
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

        # Cache the payoff object for immutability
        if is_call:
            cached_payoff = AsianCallPayoff(strike)
        else:
            cached_payoff = AsianPutPayoff(strike)
        object.__setattr__(self, '_payoff', cached_payoff)

    def __setattr__(self, name, value):
        raise AttributeError("AsianOption is immutable")

    def __delattr__(self, name):
        raise AttributeError("AsianOption is immutable")

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
        """The payoff function (cached for immutability)."""
        return self._payoff

    @property
    def exercise_style(self) -> ExerciseStyle:
        """Exercise style (Instrument interface)."""
        return self._exercise

    @property
    def option_type(self) -> str:
        """String representation of option type."""
        return "asian_call" if self._is_call else "asian_put"

    def __repr__(self) -> str:
        opt_type = "Call" if self._is_call else "Put"
        return f"AsianOption({opt_type}, K={self._strike}, T={self._maturity})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, AsianOption):
            return NotImplemented
        return (
            self._strike == other._strike and
            self._maturity == other._maturity and
            self._is_call == other._is_call and
            self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash((self._strike, self._maturity, self._is_call, self._exercise))


class BarrierOption(Instrument):
    """
    Barrier option (knock-out).

    The option is knocked out (becomes worthless) if the price
    touches the barrier level during the option's life.

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
        True for up-and-out, False for down-and-out
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN)

    Examples
    --------
    barrier_call = BarrierOption(strike=100, barrier=120, maturity=0.5,
                                  is_call=True, is_up=True)
    """

    __slots__ = ('_strike', '_barrier', '_maturity', '_is_call', '_is_up', '_exercise', '_payoff')

    def __init__(
        self,
        strike: float,
        barrier: float,
        maturity: float,
        is_call: bool = True,
        is_up: bool = True,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if barrier <= 0:
            raise ValueError(f"Barrier must be positive, got {barrier}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")

        object.__setattr__(self, '_strike', strike)
        object.__setattr__(self, '_barrier', barrier)
        object.__setattr__(self, '_maturity', maturity)
        object.__setattr__(self, '_is_call', is_call)
        object.__setattr__(self, '_is_up', is_up)
        object.__setattr__(self, '_exercise', exercise)

        # Cache the payoff object for immutability
        # Fail fast for unsupported combinations
        if is_up and is_call:
            cached_payoff = BarrierUpOutCallPayoff(strike, barrier)
        elif not is_up and not is_call:
            cached_payoff = BarrierDownOutPutPayoff(strike, barrier)
        else:
            direction = "up" if is_up else "down"
            opt_type = "call" if is_call else "put"
            raise ValueError(
                f"Unsupported barrier option combination: {direction}-out {opt_type}. "
                f"Only up-out call and down-out put are currently supported."
            )
        object.__setattr__(self, '_payoff', cached_payoff)

    def __setattr__(self, name, value):
        raise AttributeError("BarrierOption is immutable")

    def __delattr__(self, name):
        raise AttributeError("BarrierOption is immutable")

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
        direction = "up" if self._is_up else "down"
        opt = "call" if self._is_call else "put"
        return f"barrier_{direction}_out_{opt}"

    def __repr__(self) -> str:
        opt_type = "Call" if self._is_call else "Put"
        direction = "Up" if self._is_up else "Down"
        return f"BarrierOption({direction}Out{opt_type}, K={self._strike}, B={self._barrier}, T={self._maturity})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, BarrierOption):
            return NotImplemented
        return (
            self._strike == other._strike and
            self._barrier == other._barrier and
            self._maturity == other._maturity and
            self._is_call == other._is_call and
            self._is_up == other._is_up and
            self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash((self._strike, self._barrier, self._maturity, self._is_call, self._is_up, self._exercise))


class LookbackOption(Instrument):
    """
    Lookback option (floating strike).

    Call: S_T - min(S_t) (buy at the lowest price)
    Put: max(S_t) - S_T (sell at the highest price)

    Immutable after construction.

    Parameters
    ----------
    maturity : float
        Time to expiration in years (must be positive)
    is_call : bool
        True for call, False for put
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN)

    Examples
    --------
    lookback_call = LookbackOption(maturity=0.5, is_call=True)
    """

    __slots__ = ('_maturity', '_is_call', '_exercise', '_payoff')

    def __init__(
        self,
        maturity: float,
        is_call: bool = True,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ):
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")

        object.__setattr__(self, '_maturity', maturity)
        object.__setattr__(self, '_is_call', is_call)
        object.__setattr__(self, '_exercise', exercise)

        # Cache the payoff object for immutability
        if is_call:
            cached_payoff = LookbackFloatingCallPayoff()
        else:
            cached_payoff = LookbackFloatingPutPayoff()
        object.__setattr__(self, '_payoff', cached_payoff)

    def __setattr__(self, name, value):
        raise AttributeError("LookbackOption is immutable")

    def __delattr__(self, name):
        raise AttributeError("LookbackOption is immutable")

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
        """The payoff function (cached for immutability)."""
        return self._payoff

    @property
    def exercise_style(self) -> ExerciseStyle:
        """Exercise style (Instrument interface)."""
        return self._exercise

    @property
    def option_type(self) -> str:
        """String representation of option type."""
        return "lookback_call" if self._is_call else "lookback_put"

    def __repr__(self) -> str:
        opt_type = "Call" if self._is_call else "Put"
        return f"LookbackOption({opt_type}, T={self._maturity})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, LookbackOption):
            return NotImplemented
        return (
            self._maturity == other._maturity and
            self._is_call == other._is_call and
            self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash((self._maturity, self._is_call, self._exercise))


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
    >>> call = create_vanilla_option(100, 0.5, is_call=True)
    >>> american_put = create_vanilla_option(100, 0.5, False, ExerciseStyle.AMERICAN)
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


# -----------------------------------------------------------------------------
# Exotic Option Factories
# -----------------------------------------------------------------------------

def AsianCall(strike: float, maturity: float) -> AsianOption:
    """Create an Asian call option."""
    return AsianOption(strike=strike, maturity=maturity, is_call=True)


def AsianPut(strike: float, maturity: float) -> AsianOption:
    """Create an Asian put option."""
    return AsianOption(strike=strike, maturity=maturity, is_call=False)


def BarrierUpOutCall(strike: float, barrier: float, maturity: float) -> BarrierOption:
    """Create an up-and-out call option."""
    return BarrierOption(strike=strike, barrier=barrier, maturity=maturity,
                         is_call=True, is_up=True)


def BarrierDownOutPut(strike: float, barrier: float, maturity: float) -> BarrierOption:
    """Create a down-and-out put option."""
    return BarrierOption(strike=strike, barrier=barrier, maturity=maturity,
                         is_call=False, is_up=False)


def LookbackCall(maturity: float) -> LookbackOption:
    """Create a lookback call option (floating strike)."""
    return LookbackOption(maturity=maturity, is_call=True)


def LookbackPut(maturity: float) -> LookbackOption:
    """Create a lookback put option (floating strike)."""
    return LookbackOption(maturity=maturity, is_call=False)


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
