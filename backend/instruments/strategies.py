"""
Option Strategies
=================

Multi-leg option strategies with semantic validation.

Strategies provide:
- Named constructors for common combinations
- Input validation (strike ordering, etc.)
- CompositePayoff for pricing engines

All strategies are EUROPEAN exercise only.

Author: Thomas
Created: 2025
"""

from dataclasses import dataclass

import numpy as np

from backend.core.interfaces import Instrument, Payoff
from backend.core.result_types import ExerciseStyle
from backend.instruments.payoffs import (
    CompositePayoff,
    VanillaCallPayoff,
    VanillaPutPayoff,
)

# =============================================================================
# STRATEGY LEG
# =============================================================================

@dataclass(frozen=True)
class StrategyLeg:
    """
    Single leg of a multi-leg strategy.

    Parameters
    ----------
    strike : float
        Strike price
    is_call : bool
        True for call, False for put
    quantity : int
        Position quantity (positive = long, negative = short)
    """
    strike: float
    is_call: bool
    quantity: int

    def __post_init__(self):
        """Validate leg parameters."""
        if self.quantity == 0:
            raise ValueError("quantity cannot be zero")
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")

    @property
    def payoff(self) -> Payoff:
        """Payoff for this leg (unweighted)."""
        if self.is_call:
            return VanillaCallPayoff(self.strike)
        return VanillaPutPayoff(self.strike)

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    def __repr__(self) -> str:
        direction = "long" if self.is_long else "short"
        opt_type = "call" if self.is_call else "put"
        return f"StrategyLeg({direction} {abs(self.quantity)} {opt_type} K={self.strike})"


# =============================================================================
# BASE STRATEGY CLASS
# =============================================================================

class OptionStrategy(Instrument):
    """
    Base class for multi-leg option strategies.

    Provides semantic validation while exposing a CompositePayoff
    to the pricing engine.

    All strategies are European exercise only.
    """

    def __init__(self, legs: list[StrategyLeg], maturity: float):
        """
        Initialize strategy.

        Parameters
        ----------
        legs : List[StrategyLeg]
            List of strategy legs
        maturity : float
            Time to expiration in years
        """
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if not legs:
            raise ValueError("Strategy requires at least one leg")

        self._legs = legs
        self._maturity = maturity
        self._validate()

    def _validate(self):
        """Override in subclasses for strategy-specific validation."""
        pass

    @property
    def legs(self) -> list[StrategyLeg]:
        """Strategy legs (read-only)."""
        return list(self._legs)

    @property
    def payoff(self) -> Payoff:
        """Combined payoff for all legs."""
        weighted_legs = [(leg.quantity, leg.payoff) for leg in self._legs]
        return CompositePayoff(weighted_legs)

    @property
    def exercise_style(self) -> ExerciseStyle:
        """Multi-leg strategies are European only."""
        return ExerciseStyle.EUROPEAN

    @property
    def maturity(self) -> float:
        """Time to expiration in years."""
        return self._maturity

    @property
    def net_contracts(self) -> int:
        """Net number of contracts (positive = net long, negative = net short)."""
        return sum(leg.quantity for leg in self._legs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(legs={len(self._legs)}, T={self._maturity})"


# =============================================================================
# COMMON STRATEGIES
# =============================================================================

class Straddle(OptionStrategy):
    """
    Straddle: Long call + long put at same strike.

    Profits from large moves in either direction.
    Max loss = total premium paid (at strike).

    Parameters
    ----------
    strike : float
        Strike price for both legs
    maturity : float
        Time to expiration in years
    is_long : bool
        True for long straddle, False for short (default True)

    Examples
    --------
    straddle = Straddle(strike=100, maturity=0.5)
    """

    def __init__(self, strike: float, maturity: float, is_long: bool = True):
        self._strike = strike
        qty = 1 if is_long else -1
        legs = [
            StrategyLeg(strike=strike, is_call=True, quantity=qty),
            StrategyLeg(strike=strike, is_call=False, quantity=qty),
        ]
        super().__init__(legs, maturity)

    @property
    def strike(self) -> float:
        return self._strike

    def __repr__(self) -> str:
        direction = "Long" if self._legs[0].is_long else "Short"
        return f"{direction}Straddle(K={self._strike}, T={self._maturity})"


class Strangle(OptionStrategy):
    """
    Strangle: Long OTM call + long OTM put at different strikes.

    Similar to straddle but cheaper (lower premium).

    Parameters
    ----------
    put_strike : float
        Strike for put leg (lower)
    call_strike : float
        Strike for call leg (higher)
    maturity : float
        Time to expiration in years
    is_long : bool
        True for long strangle, False for short (default True)

    Examples
    --------
    strangle = Strangle(put_strike=95, call_strike=105, maturity=0.5)
    """

    def __init__(
        self,
        put_strike: float,
        call_strike: float,
        maturity: float,
        is_long: bool = True,
    ):
        self._put_strike = put_strike
        self._call_strike = call_strike
        qty = 1 if is_long else -1
        legs = [
            StrategyLeg(strike=put_strike, is_call=False, quantity=qty),
            StrategyLeg(strike=call_strike, is_call=True, quantity=qty),
        ]
        super().__init__(legs, maturity)

    def _validate(self):
        if self._put_strike >= self._call_strike:
            raise ValueError(
                f"Put strike must be less than call strike, "
                f"got put={self._put_strike}, call={self._call_strike}"
            )

    def __repr__(self) -> str:
        direction = "Long" if self._legs[0].is_long else "Short"
        return (
            f"{direction}Strangle(K_put={self._put_strike}, "
            f"K_call={self._call_strike}, T={self._maturity})"
        )


class Butterfly(OptionStrategy):
    """
    Butterfly spread: Long wings, short body.

    Structure (call butterfly):
    - Long 1 call at K1 (lower wing)
    - Short 2 calls at K2 (body)
    - Long 1 call at K3 (upper wing)

    Max profit at K2 (body strike) at expiry.
    Limited risk, limited reward.

    Parameters
    ----------
    k1 : float
        Lower wing strike
    k2 : float
        Body strike (middle)
    k3 : float
        Upper wing strike
    maturity : float
        Time to expiration in years
    is_call : bool
        True for call butterfly, False for put (default True)

    Examples
    --------
    butterfly = Butterfly(k1=90, k2=100, k3=110, maturity=0.5)
    """

    def __init__(
        self,
        k1: float,
        k2: float,
        k3: float,
        maturity: float,
        is_call: bool = True,
    ):
        self._k1, self._k2, self._k3 = k1, k2, k3
        legs = [
            StrategyLeg(strike=k1, is_call=is_call, quantity=1),
            StrategyLeg(strike=k2, is_call=is_call, quantity=-2),
            StrategyLeg(strike=k3, is_call=is_call, quantity=1),
        ]
        super().__init__(legs, maturity)

    def _validate(self):
        if not (self._k1 < self._k2 < self._k3):
            raise ValueError(
                f"Butterfly strikes must satisfy K1 < K2 < K3, "
                f"got {self._k1}, {self._k2}, {self._k3}"
            )
        # Check equidistant (standard butterfly)
        lower_spread = self._k2 - self._k1
        upper_spread = self._k3 - self._k2
        if abs(lower_spread - upper_spread) > 1e-6:
            raise ValueError(
                f"Butterfly strikes should be equidistant, "
                f"got spreads {lower_spread} and {upper_spread}"
            )

    @property
    def lower_strike(self) -> float:
        return self._k1

    @property
    def middle_strike(self) -> float:
        return self._k2

    @property
    def upper_strike(self) -> float:
        return self._k3

    def __repr__(self) -> str:
        opt_type = "Call" if self._legs[0].is_call else "Put"
        return (
            f"{opt_type}Butterfly(K1={self._k1}, K2={self._k2}, "
            f"K3={self._k3}, T={self._maturity})"
        )


class IronCondor(OptionStrategy):
    """
    Iron Condor: Sell strangle + buy wings for protection.

    Structure:
    - Long put at K1 (lowest, protection)
    - Short put at K2
    - Short call at K3
    - Long call at K4 (highest, protection)

    Profits when underlying stays between K2 and K3.
    Max profit = net premium received.
    Max loss = K2 - K1 - premium (or K4 - K3 - premium).

    Parameters
    ----------
    k1 : float
        Long put strike (lowest)
    k2 : float
        Short put strike
    k3 : float
        Short call strike
    k4 : float
        Long call strike (highest)
    maturity : float
        Time to expiration in years

    Examples
    --------
    ic = IronCondor(k1=85, k2=95, k3=105, k4=115, maturity=0.5)
    """

    def __init__(
        self,
        k1: float,
        k2: float,
        k3: float,
        k4: float,
        maturity: float,
    ):
        self._k1, self._k2, self._k3, self._k4 = k1, k2, k3, k4
        legs = [
            StrategyLeg(strike=k1, is_call=False, quantity=1),   # Long put (wing)
            StrategyLeg(strike=k2, is_call=False, quantity=-1),  # Short put
            StrategyLeg(strike=k3, is_call=True, quantity=-1),   # Short call
            StrategyLeg(strike=k4, is_call=True, quantity=1),    # Long call (wing)
        ]
        super().__init__(legs, maturity)

    def _validate(self):
        if not (self._k1 < self._k2 < self._k3 < self._k4):
            raise ValueError(
                f"Iron Condor strikes must satisfy K1 < K2 < K3 < K4, "
                f"got {self._k1}, {self._k2}, {self._k3}, {self._k4}"
            )

    @property
    def put_spread_strikes(self) -> tuple:
        """(long put strike, short put strike)"""
        return (self._k1, self._k2)

    @property
    def call_spread_strikes(self) -> tuple:
        """(short call strike, long call strike)"""
        return (self._k3, self._k4)

    def __repr__(self) -> str:
        return (
            f"IronCondor(K1={self._k1}, K2={self._k2}, "
            f"K3={self._k3}, K4={self._k4}, T={self._maturity})"
        )


class IronButterfly(OptionStrategy):
    """
    Iron Butterfly: ATM short straddle + OTM long strangle protection.

    Structure:
    - Long put at K1 (OTM, protection)
    - Short put at K2 (ATM)
    - Short call at K2 (ATM, same strike)
    - Long call at K3 (OTM, protection)

    Profits when underlying stays near K2.
    More aggressive than iron condor (ATM short options).

    Parameters
    ----------
    k1 : float
        Long put strike (lower wing)
    k2 : float
        Short straddle strike (ATM)
    k3 : float
        Long call strike (upper wing)
    maturity : float
        Time to expiration in years

    Examples
    --------
    ib = IronButterfly(k1=90, k2=100, k3=110, maturity=0.5)
    """

    def __init__(self, k1: float, k2: float, k3: float, maturity: float):
        self._k1, self._k2, self._k3 = k1, k2, k3
        legs = [
            StrategyLeg(strike=k1, is_call=False, quantity=1),   # Long put (wing)
            StrategyLeg(strike=k2, is_call=False, quantity=-1),  # Short put (body)
            StrategyLeg(strike=k2, is_call=True, quantity=-1),   # Short call (body)
            StrategyLeg(strike=k3, is_call=True, quantity=1),    # Long call (wing)
        ]
        super().__init__(legs, maturity)

    def _validate(self):
        if not (self._k1 < self._k2 < self._k3):
            raise ValueError(
                f"Iron Butterfly strikes must satisfy K1 < K2 < K3, "
                f"got {self._k1}, {self._k2}, {self._k3}"
            )

    def __repr__(self) -> str:
        return (
            f"IronButterfly(K1={self._k1}, K2={self._k2}, "
            f"K3={self._k3}, T={self._maturity})"
        )


class CallSpread(OptionStrategy):
    """
    Bull Call Spread: Long lower strike call, short higher strike call.

    Bullish strategy with limited risk and limited reward.

    Parameters
    ----------
    k_long : float
        Strike of long call (lower)
    k_short : float
        Strike of short call (higher)
    maturity : float
        Time to expiration in years

    Examples
    --------
    spread = CallSpread(k_long=95, k_short=105, maturity=0.5)
    """

    def __init__(self, k_long: float, k_short: float, maturity: float):
        self._k_long, self._k_short = k_long, k_short
        legs = [
            StrategyLeg(strike=k_long, is_call=True, quantity=1),
            StrategyLeg(strike=k_short, is_call=True, quantity=-1),
        ]
        super().__init__(legs, maturity)

    def _validate(self):
        if self._k_long >= self._k_short:
            raise ValueError(
                f"Long strike must be less than short strike, "
                f"got long={self._k_long}, short={self._k_short}"
            )

    @property
    def max_profit(self) -> float:
        """Maximum profit (at expiry if S > K_short)."""
        return self._k_short - self._k_long

    def __repr__(self) -> str:
        return f"BullCallSpread(K_long={self._k_long}, K_short={self._k_short}, T={self._maturity})"


class PutSpread(OptionStrategy):
    """
    Bear Put Spread: Long higher strike put, short lower strike put.

    Bearish strategy with limited risk and limited reward.

    Parameters
    ----------
    k_long : float
        Strike of long put (higher)
    k_short : float
        Strike of short put (lower)
    maturity : float
        Time to expiration in years

    Examples
    --------
    spread = PutSpread(k_long=105, k_short=95, maturity=0.5)
    """

    def __init__(self, k_long: float, k_short: float, maturity: float):
        self._k_long, self._k_short = k_long, k_short
        legs = [
            StrategyLeg(strike=k_long, is_call=False, quantity=1),
            StrategyLeg(strike=k_short, is_call=False, quantity=-1),
        ]
        super().__init__(legs, maturity)

    def _validate(self):
        if self._k_long <= self._k_short:
            raise ValueError(
                f"Long strike must be greater than short strike, "
                f"got long={self._k_long}, short={self._k_short}"
            )

    @property
    def max_profit(self) -> float:
        """Maximum profit (at expiry if S < K_short)."""
        return self._k_long - self._k_short

    def __repr__(self) -> str:
        return f"BearPutSpread(K_long={self._k_long}, K_short={self._k_short}, T={self._maturity})"


if __name__ == "__main__":
    print("=" * 50)
    print("Strategies Module Smoke Test")
    print("=" * 50)

    spots = np.linspace(80, 120, 9)
    print(f"Spot prices: {spots}")

    # Straddle
    straddle = Straddle(strike=100, maturity=0.5)
    print(f"\n{straddle}")
    print(f"  Payoffs: {straddle.payoff(spots)}")

    # Strangle
    strangle = Strangle(put_strike=95, call_strike=105, maturity=0.5)
    print(f"\n{strangle}")
    print(f"  Payoffs: {strangle.payoff(spots)}")

    # Butterfly
    butterfly = Butterfly(k1=90, k2=100, k3=110, maturity=0.5)
    print(f"\n{butterfly}")
    print(f"  Payoffs: {butterfly.payoff(spots)}")

    # Iron Condor
    ic = IronCondor(k1=85, k2=95, k3=105, k4=115, maturity=0.5)
    print(f"\n{ic}")
    print(f"  Payoffs: {ic.payoff(spots)}")

    # Iron Butterfly
    ib = IronButterfly(k1=90, k2=100, k3=110, maturity=0.5)
    print(f"\n{ib}")
    print(f"  Payoffs: {ib.payoff(spots)}")

    # Call Spread
    call_spread = CallSpread(k_long=95, k_short=105, maturity=0.5)
    print(f"\n{call_spread}")
    print(f"  Max profit: {call_spread.max_profit}")
    print(f"  Payoffs: {call_spread.payoff(spots)}")

    # Put Spread
    put_spread = PutSpread(k_long=105, k_short=95, maturity=0.5)
    print(f"\n{put_spread}")
    print(f"  Max profit: {put_spread.max_profit}")
    print(f"  Payoffs: {put_spread.payoff(spots)}")

    # Test validation
    print("\nTesting validation...")
    try:
        bad_ic = IronCondor(k1=100, k2=95, k3=105, k4=110, maturity=0.5)
        print("  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  Correctly caught: {e}")

    try:
        bad_butterfly = Butterfly(k1=90, k2=100, k3=115, maturity=0.5)
        print("  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  Correctly caught: {e}")

    print("\n" + "=" * 50)
    print("Strategies smoke test passed")
    print("=" * 50)
