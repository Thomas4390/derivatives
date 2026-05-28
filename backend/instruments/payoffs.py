"""
Payoff Functions
================

Atomic payoff functions for option contracts.

Payoffs know the contractual rules but NOTHING about:
- Market data (spot, rates)
- Stochastic dynamics
- Pricing method

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import numpy as np
from numba import njit

from backend.core.interfaces import Payoff

# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def _validate_spot_array(spot: np.ndarray | float, name: str = "Spot") -> np.ndarray:
    """
    Validate spot price array for payoff computation.

    Parameters
    ----------
    spot : np.ndarray
        Spot price(s) to validate
    name : str
        Variable name for error messages

    Returns
    -------
    np.ndarray
        Validated 1D float64 array

    Raises
    ------
    ValueError
        If array contains NaN, Inf, or negative values
    """
    spot_arr = np.atleast_1d(np.asarray(spot, dtype=np.float64))
    if np.any(~np.isfinite(spot_arr)):
        raise ValueError(f"{name} prices must be finite (no NaN or Inf)")
    if np.any(spot_arr < 0):
        raise ValueError(f"{name} prices must be non-negative")
    return spot_arr


def _validate_path_array(path: np.ndarray | list) -> np.ndarray:
    """
    Validate path array for path-dependent payoff computation.

    Parameters
    ----------
    path : np.ndarray
        Price path(s) to validate

    Returns
    -------
    np.ndarray
        Validated 2D float64 array with shape (n_paths, n_steps)

    Raises
    ------
    ValueError
        If array contains NaN, Inf, negative values, or has wrong dimensions
    """
    path_arr = np.atleast_2d(np.asarray(path, dtype=np.float64))
    if path_arr.ndim != 2:
        raise ValueError(f"Path must be 2D array, got {path_arr.ndim}D")
    if path_arr.shape[1] < 2:
        raise ValueError(
            f"Path must have at least 2 time steps, got {path_arr.shape[1]}"
        )
    if np.any(~np.isfinite(path_arr)):
        raise ValueError("Path prices must be finite (no NaN or Inf)")
    if np.any(path_arr < 0):
        raise ValueError("Path prices must be non-negative")
    return path_arr


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
    """Vectorized digital call payoff: payout if S >= K, else 0.

    Note: Standard convention is digital call pays at spot >= strike.
    """
    result = np.empty_like(spots)
    for i in range(len(spots)):
        result[i] = payout if spots[i] >= strike else 0.0
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

    _strike: float

    def __init__(self, strike: float) -> None:
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
        spot_arr = _validate_spot_array(spot)
        return _call_payoff(spot_arr, self._strike)

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

    _strike: float

    def __init__(self, strike: float) -> None:
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
        spot_arr = _validate_spot_array(spot)
        return _put_payoff(spot_arr, self._strike)

    def __repr__(self) -> str:
        return f"VanillaPutPayoff(strike={self._strike})"


class DigitalCallPayoff(Payoff):
    """
    Digital (binary) call payoff: pays fixed amount if S >= K.

    Note: Standard convention is that digital call pays at spot >= strike.
    This means at-the-money (S == K), the call pays.

    Parameters
    ----------
    strike : float
        Strike price
    payout : float
        Fixed payout amount (default 1.0)

    Examples
    --------
    digital = DigitalCallPayoff(strike=100.0, payout=10.0)
    digital(np.array([99, 100, 101]))  # array([0, 10, 10])
    """

    _strike: float
    _payout: float

    def __init__(self, strike: float, payout: float = 1.0) -> None:
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
        spot_arr = _validate_spot_array(spot)
        return _digital_call_payoff(spot_arr, self._strike, self._payout)

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

    _strike: float
    _payout: float

    def __init__(self, strike: float, payout: float = 1.0) -> None:
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
        spot_arr = _validate_spot_array(spot)
        return _digital_put_payoff(spot_arr, self._strike, self._payout)

    def __repr__(self) -> str:
        return f"DigitalPutPayoff(strike={self._strike}, payout={self._payout})"


class SpotPayoff(Payoff):
    """
    Identity payoff: returns the terminal spot price.

    Equivalent to holding the underlying asset. Used as a building block
    in portfolio algebra for constructing structured product payoffs.

    Examples
    --------
    spot = SpotPayoff()
    spot(np.array([90, 100, 110]))  # array([90, 100, 110])
    """

    @property
    def is_path_dependent(self) -> bool:
        return False

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        return _validate_spot_array(spot).copy()

    def __repr__(self) -> str:
        return "SpotPayoff()"


class BondPayoff(Payoff):
    """
    Constant payoff: returns a fixed amount regardless of spot.

    Represents a zero-coupon bond paying `notional` at maturity.

    Parameters
    ----------
    notional : float
        Fixed payout amount (default 1.0)

    Examples
    --------
    bond = BondPayoff(notional=100.0)
    bond(np.array([90, 100, 110]))  # array([100, 100, 100])
    """

    _notional: float

    def __init__(self, notional: float = 1.0) -> None:
        self._notional = notional

    @property
    def notional(self) -> float:
        return self._notional

    @property
    def is_path_dependent(self) -> bool:
        return False

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        spot_arr = _validate_spot_array(spot)
        return np.full_like(spot_arr, self._notional)

    def __repr__(self) -> str:
        return f"BondPayoff(notional={self._notional})"


class CompositePayoff(Payoff):
    """
    Weighted sum of payoffs for multi-leg strategies.

    Used internally by strategy classes (IronCondor, Butterfly, etc.)
    and by the portfolio algebra operators (+, -, *).
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

    # Portfolio algebra (equivalent)
    straddle = call + put
    straddle(np.array([90, 100, 110]))  # array([10, 0, 10])
    """

    _legs: list[tuple[int | float, Payoff]]

    def __init__(self, legs: list[tuple[int | float, Payoff]]) -> None:
        if not legs:
            raise ValueError("CompositePayoff requires at least one leg")
        self._legs = legs

    @property
    def legs(self) -> list[tuple[int | float, Payoff]]:
        """List of (weight, payoff) tuples."""
        return self._legs

    @property
    def is_path_dependent(self) -> bool:
        return any(payoff.is_path_dependent for _, payoff in self._legs)

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        spot_arr = _validate_spot_array(spot)
        result = np.zeros_like(spot_arr)
        for weight, payoff in self._legs:
            result += weight * payoff(spot_arr)
        return result

    def __repr__(self) -> str:
        legs_str = ", ".join(f"{w}*{p}" for w, p in self._legs)
        return f"CompositePayoff([{legs_str}])"


class LowPointForwardPayoff(Payoff):
    """
    Low-point forward payoff (MALP family).

    Payoff: S_0 * (S_T / min(S) - 1)

    Forward contract at the floating strike (path minimum), multiplied by
    the discount factor S_0/K_float.

    Examples
    --------
    payoff = LowPointForwardPayoff()
    path = np.array([[100, 80, 110]])  # min=80, S_T=110, S_0=100
    payoff(path)  # 100 * (110/80 - 1) = 37.5
    """

    @property
    def is_path_dependent(self) -> bool:
        return True

    def __call__(self, path: np.ndarray) -> np.ndarray:
        path_arr = _validate_path_array(path)
        s0 = path_arr[:, 0]
        s_t = path_arr[:, -1]
        path_min = np.min(path_arr, axis=1)
        path_min_safe = np.maximum(path_min, 1e-10)
        return s0 * (s_t / path_min_safe - 1.0)

    def __repr__(self) -> str:
        return "LowPointForwardPayoff()"


if __name__ == "__main__":
    # Smoke test
    print("=" * 50)
    print("Payoffs Module Smoke Test")
    print("=" * 50)

    spots = np.array([90.0, 100.0, 110.0])

    # Vanilla payoffs
    call = VanillaCallPayoff(strike=100.0)
    put = VanillaPutPayoff(strike=100.0)

    print("\nVanilla Call (K=100):")
    print(f"  Spots: {spots}")
    print(f"  Payoffs: {call(spots)}")

    print("\nVanilla Put (K=100):")
    print(f"  Spots: {spots}")
    print(f"  Payoffs: {put(spots)}")

    # Digital payoffs
    digital_call = DigitalCallPayoff(strike=100.0, payout=10.0)
    digital_put = DigitalPutPayoff(strike=100.0, payout=10.0)

    print("\nDigital Call (K=100, payout=10):")
    print(f"  Payoffs: {digital_call(spots)}")

    print("\nDigital Put (K=100, payout=10):")
    print(f"  Payoffs: {digital_put(spots)}")

    # Composite payoff (straddle)
    straddle = CompositePayoff([(1.0, call), (1.0, put)])
    print("\nStraddle (Call + Put at K=100):")
    print(f"  Payoffs: {straddle(spots)}")

    # Test path dependency
    print("\nPath dependency:")
    print(f"  Call: {call.is_path_dependent}")
    print(f"  Straddle: {straddle.is_path_dependent}")

    print("\n" + "=" * 50)
    print("Payoffs smoke test passed")
    print("=" * 50)
