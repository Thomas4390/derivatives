"""
Payoff Functions
================

Atomic payoff functions for option contracts.

Payoffs know the contractual rules but NOTHING about:
- Market data (spot, rates)
- Stochastic dynamics
- Pricing method

Author: Thomas
Created: 2025
"""

from typing import List
import numpy as np
from numba import njit

from backend.core.interfaces import Payoff


# =============================================================================
# NUMBA KERNELS (Hot Path)
# =============================================================================

@njit(cache=True, fastmath=True)
def _call_payoff(spots: np.ndarray, strike: float) -> np.ndarray:
    """Vectorized call payoff: max(S - K, 0)."""
    result = np.empty_like(spots)
    for i in range(len(spots)):
        result[i] = max(spots[i] - strike, 0.0)
    return result


@njit(cache=True, fastmath=True)
def _put_payoff(spots: np.ndarray, strike: float) -> np.ndarray:
    """Vectorized put payoff: max(K - S, 0)."""
    result = np.empty_like(spots)
    for i in range(len(spots)):
        result[i] = max(strike - spots[i], 0.0)
    return result


@njit(cache=True, fastmath=True)
def _digital_call_payoff(spots: np.ndarray, strike: float, payout: float) -> np.ndarray:
    """Vectorized digital call payoff: payout if S > K, else 0."""
    result = np.empty_like(spots)
    for i in range(len(spots)):
        result[i] = payout if spots[i] > strike else 0.0
    return result


@njit(cache=True, fastmath=True)
def _digital_put_payoff(spots: np.ndarray, strike: float, payout: float) -> np.ndarray:
    """Vectorized digital put payoff: payout if S < K, else 0."""
    result = np.empty_like(spots)
    for i in range(len(spots)):
        result[i] = payout if spots[i] < strike else 0.0
    return result


# =============================================================================
# PAYOFF CLASSES
# =============================================================================

class VanillaCallPayoff(Payoff):
    """
    Vanilla call payoff: max(S - K, 0).

    Parameters
    ----------
    strike : float
        Strike price (must be positive)

    Examples
    --------
    call = VanillaCallPayoff(strike=100.0)
    call(np.array([90, 100, 110]))  # array([0, 0, 10])
    """

    def __init__(self, strike: float):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        self._strike = strike

    @property
    def strike(self) -> float:
        """Strike price."""
        return self._strike

    @property
    def is_path_dependent(self) -> bool:
        return False

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        return _call_payoff(np.atleast_1d(np.asarray(spot, dtype=np.float64)), self._strike)

    def __repr__(self) -> str:
        return f"VanillaCallPayoff(strike={self._strike})"


class VanillaPutPayoff(Payoff):
    """
    Vanilla put payoff: max(K - S, 0).

    Parameters
    ----------
    strike : float
        Strike price (must be positive)

    Examples
    --------
    put = VanillaPutPayoff(strike=100.0)
    put(np.array([90, 100, 110]))  # array([10, 0, 0])
    """

    def __init__(self, strike: float):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        self._strike = strike

    @property
    def strike(self) -> float:
        """Strike price."""
        return self._strike

    @property
    def is_path_dependent(self) -> bool:
        return False

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        return _put_payoff(np.atleast_1d(np.asarray(spot, dtype=np.float64)), self._strike)

    def __repr__(self) -> str:
        return f"VanillaPutPayoff(strike={self._strike})"


class DigitalCallPayoff(Payoff):
    """
    Digital (binary) call payoff: pays fixed amount if S > K.

    Parameters
    ----------
    strike : float
        Strike price
    payout : float
        Fixed payout amount (default 1.0)

    Examples
    --------
    digital = DigitalCallPayoff(strike=100.0, payout=10.0)
    digital(np.array([99, 100, 101]))  # array([0, 0, 10])
    """

    def __init__(self, strike: float, payout: float = 1.0):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if payout <= 0:
            raise ValueError(f"Payout must be positive, got {payout}")
        self._strike = strike
        self._payout = payout

    @property
    def strike(self) -> float:
        return self._strike

    @property
    def payout(self) -> float:
        return self._payout

    @property
    def is_path_dependent(self) -> bool:
        return False

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        return _digital_call_payoff(
            np.atleast_1d(np.asarray(spot, dtype=np.float64)),
            self._strike,
            self._payout
        )

    def __repr__(self) -> str:
        return f"DigitalCallPayoff(strike={self._strike}, payout={self._payout})"


class DigitalPutPayoff(Payoff):
    """
    Digital (binary) put payoff: pays fixed amount if S < K.

    Parameters
    ----------
    strike : float
        Strike price
    payout : float
        Fixed payout amount (default 1.0)

    Examples
    --------
    digital = DigitalPutPayoff(strike=100.0, payout=10.0)
    digital(np.array([99, 100, 101]))  # array([10, 0, 0])
    """

    def __init__(self, strike: float, payout: float = 1.0):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if payout <= 0:
            raise ValueError(f"Payout must be positive, got {payout}")
        self._strike = strike
        self._payout = payout

    @property
    def strike(self) -> float:
        return self._strike

    @property
    def payout(self) -> float:
        return self._payout

    @property
    def is_path_dependent(self) -> bool:
        return False

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        return _digital_put_payoff(
            np.atleast_1d(np.asarray(spot, dtype=np.float64)),
            self._strike,
            self._payout
        )

    def __repr__(self) -> str:
        return f"DigitalPutPayoff(strike={self._strike}, payout={self._payout})"


class CompositePayoff(Payoff):
    """
    Weighted sum of payoffs for multi-leg strategies.

    Used internally by strategy classes (IronCondor, Butterfly, etc.)
    The engine sees this as a single payoff.

    Parameters
    ----------
    legs : List[tuple]
        List of (weight, Payoff) tuples.
        Weight > 0 for long, < 0 for short.

    Examples
    --------
    # Straddle = long call + long put
    call = VanillaCallPayoff(strike=100)
    put = VanillaPutPayoff(strike=100)
    straddle = CompositePayoff([(1.0, call), (1.0, put)])
    straddle(np.array([90, 100, 110]))  # array([10, 0, 10])
    """

    def __init__(self, legs: List[tuple]):
        if not legs:
            raise ValueError("CompositePayoff requires at least one leg")
        self._legs = legs

    @property
    def legs(self) -> List[tuple]:
        """List of (weight, payoff) tuples."""
        return self._legs

    @property
    def is_path_dependent(self) -> bool:
        return any(payoff.is_path_dependent for _, payoff in self._legs)

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        spot = np.atleast_1d(np.asarray(spot, dtype=np.float64))
        result = np.zeros_like(spot)
        for weight, payoff in self._legs:
            result += weight * payoff(spot)
        return result

    def __repr__(self) -> str:
        legs_str = ", ".join(f"{w}*{p}" for w, p in self._legs)
        return f"CompositePayoff([{legs_str}])"


if __name__ == "__main__":
    # Smoke test
    print("=" * 50)
    print("Payoffs Module Smoke Test")
    print("=" * 50)

    spots = np.array([90.0, 100.0, 110.0])

    # Vanilla payoffs
    call = VanillaCallPayoff(strike=100.0)
    put = VanillaPutPayoff(strike=100.0)

    print(f"\nVanilla Call (K=100):")
    print(f"  Spots: {spots}")
    print(f"  Payoffs: {call(spots)}")

    print(f"\nVanilla Put (K=100):")
    print(f"  Spots: {spots}")
    print(f"  Payoffs: {put(spots)}")

    # Digital payoffs
    digital_call = DigitalCallPayoff(strike=100.0, payout=10.0)
    digital_put = DigitalPutPayoff(strike=100.0, payout=10.0)

    print(f"\nDigital Call (K=100, payout=10):")
    print(f"  Payoffs: {digital_call(spots)}")

    print(f"\nDigital Put (K=100, payout=10):")
    print(f"  Payoffs: {digital_put(spots)}")

    # Composite payoff (straddle)
    straddle = CompositePayoff([(1.0, call), (1.0, put)])
    print(f"\nStraddle (Call + Put at K=100):")
    print(f"  Payoffs: {straddle(spots)}")

    # Test path dependency
    print(f"\nPath dependency:")
    print(f"  Call: {call.is_path_dependent}")
    print(f"  Straddle: {straddle.is_path_dependent}")

    print("\n" + "=" * 50)
    print("Payoffs smoke test passed")
    print("=" * 50)
