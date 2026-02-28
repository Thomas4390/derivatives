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

    __slots__ = ('_strike', '_maturity', '_is_call', '_average_type', '_exercise', '_payoff')

    def __init__(
        self,
        strike: float,
        maturity: float,
        is_call: bool = True,
        average_type: str = "arithmetic",
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if average_type not in ("arithmetic", "geometric"):
            raise ValueError(f"average_type must be 'arithmetic' or 'geometric', got '{average_type}'")

        object.__setattr__(self, '_strike', strike)
        object.__setattr__(self, '_maturity', maturity)
        object.__setattr__(self, '_is_call', is_call)
        object.__setattr__(self, '_average_type', average_type)
        object.__setattr__(self, '_exercise', exercise)

        # Cache the payoff object for MC-supported types
        if average_type == "arithmetic":
            if is_call:
                cached_payoff = AsianCallPayoff(strike)
            else:
                cached_payoff = AsianPutPayoff(strike)
            object.__setattr__(self, '_payoff', cached_payoff)
        else:
            # Geometric average: analytical only, no MC payoff class
            object.__setattr__(self, '_payoff', None)

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
    def average_type(self) -> str:
        """Average type: 'arithmetic' or 'geometric'."""
        return self._average_type

    @property
    def payoff(self):
        """The payoff function (None for geometric analytical-only types)."""
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

    def __eq__(self, other) -> bool:
        if not isinstance(other, AsianOption):
            return NotImplemented
        return (
            self._strike == other._strike and
            self._maturity == other._maturity and
            self._is_call == other._is_call and
            self._average_type == other._average_type and
            self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash((self._strike, self._maturity, self._is_call, self._average_type, self._exercise))


class BarrierOption(Instrument):
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

    __slots__ = ('_strike', '_barrier', '_maturity', '_is_call', '_is_up',
                 '_is_knock_in', '_rebate', '_exercise', '_payoff')

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
    ):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if barrier <= 0:
            raise ValueError(f"Barrier must be positive, got {barrier}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if rebate < 0:
            raise ValueError(f"Rebate must be non-negative, got {rebate}")

        object.__setattr__(self, '_strike', strike)
        object.__setattr__(self, '_barrier', barrier)
        object.__setattr__(self, '_maturity', maturity)
        object.__setattr__(self, '_is_call', is_call)
        object.__setattr__(self, '_is_up', is_up)
        object.__setattr__(self, '_is_knock_in', is_knock_in)
        object.__setattr__(self, '_rebate', rebate)
        object.__setattr__(self, '_exercise', exercise)

        # Cache the payoff object for MC-supported combinations;
        # set None for types only supported by the analytical engine.
        if is_knock_in:
            # Knock-in types have no MC payoff class - analytical only
            object.__setattr__(self, '_payoff', None)
        elif is_up and is_call:
            object.__setattr__(self, '_payoff', BarrierUpOutCallPayoff(strike, barrier))
        elif not is_up and not is_call:
            object.__setattr__(self, '_payoff', BarrierDownOutPutPayoff(strike, barrier))
        else:
            # Knock-out types without MC payoff (up-out put, down-out call)
            object.__setattr__(self, '_payoff', None)

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
    def is_knock_in(self) -> bool:
        """True for knock-in, False for knock-out."""
        return self._is_knock_in

    @property
    def rebate(self) -> float:
        """Rebate paid at knockout."""
        return self._rebate

    @property
    def payoff(self):
        """The payoff function (None for analytical-only types)."""
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

    def __eq__(self, other) -> bool:
        if not isinstance(other, BarrierOption):
            return NotImplemented
        return (
            self._strike == other._strike and
            self._barrier == other._barrier and
            self._maturity == other._maturity and
            self._is_call == other._is_call and
            self._is_up == other._is_up and
            self._is_knock_in == other._is_knock_in and
            self._rebate == other._rebate and
            self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash((self._strike, self._barrier, self._maturity, self._is_call,
                     self._is_up, self._is_knock_in, self._rebate, self._exercise))


class ChooserOption(Instrument):
    """
    Simple chooser option (Rubinstein 1991).

    At choice time t_c, the holder chooses max(Call, Put).
    Uses the identity: V = BS_call(S, K, T) + BS_put(S, K*exp(-(r-q)*(T-t_c)), t_c)

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike price (must be positive)
    maturity : float
        Time to expiration in years (must be positive)
    choice_time : float
        Time at which the holder chooses call or put (must be 0 < choice_time <= maturity)
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN)

    Examples
    --------
    chooser = ChooserOption(strike=100, maturity=1.0, choice_time=0.5)
    """

    __slots__ = ('_strike', '_maturity', '_choice_time', '_exercise', '_payoff')

    def __init__(
        self,
        strike: float,
        maturity: float,
        choice_time: float,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if choice_time <= 0 or choice_time > maturity:
            raise ValueError(
                f"Choice time must be in (0, maturity], got {choice_time} "
                f"with maturity={maturity}"
            )

        object.__setattr__(self, '_strike', strike)
        object.__setattr__(self, '_maturity', maturity)
        object.__setattr__(self, '_choice_time', choice_time)
        object.__setattr__(self, '_exercise', exercise)
        object.__setattr__(self, '_payoff', None)  # analytical only

    def __setattr__(self, name, value):
        raise AttributeError("ChooserOption is immutable")

    def __delattr__(self, name):
        raise AttributeError("ChooserOption is immutable")

    @property
    def strike(self) -> float:
        """Strike price."""
        return self._strike

    @property
    def maturity(self) -> float:
        """Time to expiration in years."""
        return self._maturity

    @property
    def choice_time(self) -> float:
        """Time at which the holder chooses call or put."""
        return self._choice_time

    @property
    def payoff(self):
        """The payoff function (None — analytical only)."""
        return self._payoff

    @property
    def exercise_style(self) -> ExerciseStyle:
        """Exercise style (Instrument interface)."""
        return self._exercise

    @property
    def option_type(self) -> str:
        """String representation of option type."""
        return "chooser"

    def __repr__(self) -> str:
        return (
            f"ChooserOption(K={self._strike}, T={self._maturity}, "
            f"t_c={self._choice_time})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, ChooserOption):
            return NotImplemented
        return (
            self._strike == other._strike and
            self._maturity == other._maturity and
            self._choice_time == other._choice_time and
            self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash((self._strike, self._maturity, self._choice_time, self._exercise))


class AssetOrNothingOption(Instrument):
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
        object.__setattr__(self, '_payoff', None)  # analytical only

    def __setattr__(self, name, value):
        raise AttributeError("AssetOrNothingOption is immutable")

    def __delattr__(self, name):
        raise AttributeError("AssetOrNothingOption is immutable")

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
    def payoff(self):
        """The payoff function (None — analytical only)."""
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

    def __eq__(self, other) -> bool:
        if not isinstance(other, AssetOrNothingOption):
            return NotImplemented
        return (
            self._strike == other._strike and
            self._maturity == other._maturity and
            self._is_call == other._is_call and
            self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash((self._strike, self._maturity, self._is_call, self._exercise))


class PowerOption(Instrument):
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

    __slots__ = ('_strike', '_maturity', '_is_call', '_power', '_exercise', '_payoff')

    def __init__(
        self,
        strike: float,
        maturity: float,
        is_call: bool,
        power: float,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if power <= 0:
            raise ValueError(f"Power must be positive, got {power}")

        object.__setattr__(self, '_strike', strike)
        object.__setattr__(self, '_maturity', maturity)
        object.__setattr__(self, '_is_call', is_call)
        object.__setattr__(self, '_power', power)
        object.__setattr__(self, '_exercise', exercise)
        object.__setattr__(self, '_payoff', None)  # analytical only

    def __setattr__(self, name, value):
        raise AttributeError("PowerOption is immutable")

    def __delattr__(self, name):
        raise AttributeError("PowerOption is immutable")

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
    def payoff(self):
        """The payoff function (None — analytical only)."""
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

    def __eq__(self, other) -> bool:
        if not isinstance(other, PowerOption):
            return NotImplemented
        return (
            self._strike == other._strike and
            self._maturity == other._maturity and
            self._is_call == other._is_call and
            self._power == other._power and
            self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash((self._strike, self._maturity, self._is_call, self._power, self._exercise))


class GapOption(Instrument):
    """
    Gap option.

    Has separate trigger strike (K2) and payment strike (K1).
    Call payoff: (S_T - K1) if S_T > K2, else 0.
    Note: payoff can be negative when K1 > K2 and K2 < S_T < K1.

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Payment strike K1 (must be positive)
    trigger : float
        Trigger strike K2 (must be positive)
    maturity : float
        Time to expiration in years (must be positive)
    is_call : bool
        True for call, False for put
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN)

    Examples
    --------
    gap_call = GapOption(strike=105, trigger=100, maturity=0.5, is_call=True)
    """

    __slots__ = ('_strike', '_trigger', '_maturity', '_is_call', '_exercise', '_payoff')

    def __init__(
        self,
        strike: float,
        trigger: float,
        maturity: float,
        is_call: bool,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if trigger <= 0:
            raise ValueError(f"Trigger must be positive, got {trigger}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")

        object.__setattr__(self, '_strike', strike)
        object.__setattr__(self, '_trigger', trigger)
        object.__setattr__(self, '_maturity', maturity)
        object.__setattr__(self, '_is_call', is_call)
        object.__setattr__(self, '_exercise', exercise)
        object.__setattr__(self, '_payoff', None)  # analytical only

    def __setattr__(self, name, value):
        raise AttributeError("GapOption is immutable")

    def __delattr__(self, name):
        raise AttributeError("GapOption is immutable")

    @property
    def strike(self) -> float:
        """Payment strike K1."""
        return self._strike

    @property
    def trigger(self) -> float:
        """Trigger strike K2."""
        return self._trigger

    @property
    def maturity(self) -> float:
        """Time to expiration in years."""
        return self._maturity

    @property
    def is_call(self) -> bool:
        """True for call, False for put."""
        return self._is_call

    @property
    def payoff(self):
        """The payoff function (None — analytical only)."""
        return self._payoff

    @property
    def exercise_style(self) -> ExerciseStyle:
        """Exercise style (Instrument interface)."""
        return self._exercise

    @property
    def option_type(self) -> str:
        """String representation of option type."""
        return "gap_call" if self._is_call else "gap_put"

    def __repr__(self) -> str:
        opt_type = "Call" if self._is_call else "Put"
        return (
            f"GapOption({opt_type}, K1={self._strike}, K2={self._trigger}, "
            f"T={self._maturity})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, GapOption):
            return NotImplemented
        return (
            self._strike == other._strike and
            self._trigger == other._trigger and
            self._maturity == other._maturity and
            self._is_call == other._is_call and
            self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash((self._strike, self._trigger, self._maturity, self._is_call, self._exercise))


class LookbackOption(Instrument):
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

    __slots__ = ('_maturity', '_is_call', '_strike', '_lookback_type', '_exercise', '_payoff')

    def __init__(
        self,
        maturity: float,
        is_call: bool = True,
        strike: float = None,
        lookback_type: str = "floating",
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ):
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if lookback_type not in ("floating", "fixed"):
            raise ValueError(f"lookback_type must be 'floating' or 'fixed', got '{lookback_type}'")
        if lookback_type == "fixed" and (strike is None or strike <= 0):
            raise ValueError(f"Fixed-strike lookback requires positive strike, got {strike}")

        object.__setattr__(self, '_maturity', maturity)
        object.__setattr__(self, '_is_call', is_call)
        object.__setattr__(self, '_strike', strike)
        object.__setattr__(self, '_lookback_type', lookback_type)
        object.__setattr__(self, '_exercise', exercise)

        # Cache the payoff object for MC-supported types
        if lookback_type == "floating":
            if is_call:
                cached_payoff = LookbackFloatingCallPayoff()
            else:
                cached_payoff = LookbackFloatingPutPayoff()
            object.__setattr__(self, '_payoff', cached_payoff)
        else:
            # Fixed-strike lookback: analytical only, no MC payoff class
            object.__setattr__(self, '_payoff', None)

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
    def strike(self):
        """Strike price (None for floating-strike lookbacks)."""
        return self._strike

    @property
    def lookback_type(self) -> str:
        """Lookback type: 'floating' or 'fixed'."""
        return self._lookback_type

    @property
    def payoff(self):
        """The payoff function (None for fixed-strike analytical-only types)."""
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

    def __eq__(self, other) -> bool:
        if not isinstance(other, LookbackOption):
            return NotImplemented
        return (
            self._maturity == other._maturity and
            self._is_call == other._is_call and
            self._strike == other._strike and
            self._lookback_type == other._lookback_type and
            self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash((self._maturity, self._is_call, self._strike, self._lookback_type, self._exercise))


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


# -----------------------------------------------------------------------------
# Exotic Option Factories
# -----------------------------------------------------------------------------

def AsianCall(strike: float, maturity: float) -> AsianOption:
    """Create an Asian call option."""
    return AsianOption(strike=strike, maturity=maturity, is_call=True)


def AsianPut(strike: float, maturity: float) -> AsianOption:
    """Create an Asian put option."""
    return AsianOption(strike=strike, maturity=maturity, is_call=False)


def BarrierUpOutCall(strike: float, barrier: float, maturity: float, rebate: float = 0.0) -> BarrierOption:
    """Create an up-and-out call option."""
    return BarrierOption(strike=strike, barrier=barrier, maturity=maturity,
                         is_call=True, is_up=True, is_knock_in=False, rebate=rebate)


def BarrierUpInCall(strike: float, barrier: float, maturity: float) -> BarrierOption:
    """Create an up-and-in call option."""
    return BarrierOption(strike=strike, barrier=barrier, maturity=maturity,
                         is_call=True, is_up=True, is_knock_in=True)


def BarrierDownOutCall(strike: float, barrier: float, maturity: float, rebate: float = 0.0) -> BarrierOption:
    """Create a down-and-out call option."""
    return BarrierOption(strike=strike, barrier=barrier, maturity=maturity,
                         is_call=True, is_up=False, is_knock_in=False, rebate=rebate)


def BarrierDownInCall(strike: float, barrier: float, maturity: float) -> BarrierOption:
    """Create a down-and-in call option."""
    return BarrierOption(strike=strike, barrier=barrier, maturity=maturity,
                         is_call=True, is_up=False, is_knock_in=True)


def BarrierUpOutPut(strike: float, barrier: float, maturity: float, rebate: float = 0.0) -> BarrierOption:
    """Create an up-and-out put option."""
    return BarrierOption(strike=strike, barrier=barrier, maturity=maturity,
                         is_call=False, is_up=True, is_knock_in=False, rebate=rebate)


def BarrierUpInPut(strike: float, barrier: float, maturity: float) -> BarrierOption:
    """Create an up-and-in put option."""
    return BarrierOption(strike=strike, barrier=barrier, maturity=maturity,
                         is_call=False, is_up=True, is_knock_in=True)


def BarrierDownOutPut(strike: float, barrier: float, maturity: float, rebate: float = 0.0) -> BarrierOption:
    """Create a down-and-out put option."""
    return BarrierOption(strike=strike, barrier=barrier, maturity=maturity,
                         is_call=False, is_up=False, is_knock_in=False, rebate=rebate)


def BarrierDownInPut(strike: float, barrier: float, maturity: float) -> BarrierOption:
    """Create a down-and-in put option."""
    return BarrierOption(strike=strike, barrier=barrier, maturity=maturity,
                         is_call=False, is_up=False, is_knock_in=True)


def AsianGeometricCall(strike: float, maturity: float) -> AsianOption:
    """Create a geometric Asian call option."""
    return AsianOption(strike=strike, maturity=maturity, is_call=True, average_type="geometric")


def AsianGeometricPut(strike: float, maturity: float) -> AsianOption:
    """Create a geometric Asian put option."""
    return AsianOption(strike=strike, maturity=maturity, is_call=False, average_type="geometric")


def LookbackCall(maturity: float) -> LookbackOption:
    """Create a lookback call option (floating strike)."""
    return LookbackOption(maturity=maturity, is_call=True)


def LookbackPut(maturity: float) -> LookbackOption:
    """Create a lookback put option (floating strike)."""
    return LookbackOption(maturity=maturity, is_call=False)


def LookbackFixedCall(strike: float, maturity: float) -> LookbackOption:
    """Create a fixed-strike lookback call option."""
    return LookbackOption(maturity=maturity, is_call=True, strike=strike, lookback_type="fixed")


def LookbackFixedPut(strike: float, maturity: float) -> LookbackOption:
    """Create a fixed-strike lookback put option."""
    return LookbackOption(maturity=maturity, is_call=False, strike=strike, lookback_type="fixed")


def Chooser(strike: float, maturity: float, choice_time: float) -> ChooserOption:
    """Create a chooser option."""
    return ChooserOption(strike=strike, maturity=maturity, choice_time=choice_time)


def AssetOrNothingCall(strike: float, maturity: float) -> AssetOrNothingOption:
    """Create an asset-or-nothing call option."""
    return AssetOrNothingOption(strike=strike, maturity=maturity, is_call=True)


def AssetOrNothingPut(strike: float, maturity: float) -> AssetOrNothingOption:
    """Create an asset-or-nothing put option."""
    return AssetOrNothingOption(strike=strike, maturity=maturity, is_call=False)


def PowerCall(strike: float, maturity: float, power: float) -> PowerOption:
    """Create a power call option."""
    return PowerOption(strike=strike, maturity=maturity, is_call=True, power=power)


def PowerPut(strike: float, maturity: float, power: float) -> PowerOption:
    """Create a power put option."""
    return PowerOption(strike=strike, maturity=maturity, is_call=False, power=power)


def GapCall(strike: float, trigger: float, maturity: float) -> GapOption:
    """Create a gap call option."""
    return GapOption(strike=strike, trigger=trigger, maturity=maturity, is_call=True)


def GapPut(strike: float, trigger: float, maturity: float) -> GapOption:
    """Create a gap put option."""
    return GapOption(strike=strike, trigger=trigger, maturity=maturity, is_call=False)


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
    print("\nExercise style checks:")
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
