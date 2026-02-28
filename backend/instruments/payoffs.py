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


import numpy as np
from numba import njit

from backend.core.interfaces import Payoff

# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def _validate_spot_array(spot: np.ndarray, name: str = "Spot") -> np.ndarray:
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


def _validate_path_array(path: np.ndarray) -> np.ndarray:
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
        raise ValueError(f"Path must have at least 2 time steps, got {path_arr.shape[1]}")
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
        spot_arr = _validate_spot_array(spot)
        return _digital_put_payoff(spot_arr, self._strike, self._payout)

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

    def __init__(self, legs: list[tuple]):
        if not legs:
            raise ValueError("CompositePayoff requires at least one leg")
        self._legs = legs

    @property
    def legs(self) -> list[tuple]:
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


# =============================================================================
# EXOTIC PAYOFF CLASSES (Path-Dependent)
# =============================================================================

class AsianCallPayoff(Payoff):
    """
    Asian call payoff: max(avg(S) - K, 0).

    Uses arithmetic average of the entire price path.

    Parameters
    ----------
    strike : float
        Strike price (must be positive)

    Examples
    --------
    asian_call = AsianCallPayoff(strike=100.0)
    path = np.array([[100, 105, 110]])  # avg = 105
    asian_call(path)  # array([5.0])
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
        return True

    def __call__(self, path: np.ndarray) -> np.ndarray:
        from backend.math_kernels.payoff_kernels import asian_arithmetic_payoff
        path_arr = _validate_path_array(path)
        result = np.empty(len(path_arr))
        for i in range(len(path_arr)):
            result[i] = asian_arithmetic_payoff(path_arr[i], self._strike, True)
        return result

    def __repr__(self) -> str:
        return f"AsianCallPayoff(strike={self._strike})"


class AsianPutPayoff(Payoff):
    """
    Asian put payoff: max(K - avg(S), 0).

    Uses arithmetic average of the entire price path.

    Parameters
    ----------
    strike : float
        Strike price (must be positive)

    Examples
    --------
    asian_put = AsianPutPayoff(strike=100.0)
    path = np.array([[100, 95, 90]])  # avg = 95
    asian_put(path)  # array([5.0])
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
        return True

    def __call__(self, path: np.ndarray) -> np.ndarray:
        from backend.math_kernels.payoff_kernels import asian_arithmetic_payoff
        path_arr = _validate_path_array(path)
        result = np.empty(len(path_arr))
        for i in range(len(path_arr)):
            result[i] = asian_arithmetic_payoff(path_arr[i], self._strike, False)
        return result

    def __repr__(self) -> str:
        return f"AsianPutPayoff(strike={self._strike})"


class BarrierUpOutCallPayoff(Payoff):
    """
    Up-and-out call payoff: knocked out if S >= barrier.

    Returns vanilla call payoff if barrier is never touched,
    otherwise returns 0.

    Parameters
    ----------
    strike : float
        Strike price (must be positive)
    barrier : float
        Upper barrier level (must be above strike)

    Examples
    --------
    barrier_call = BarrierUpOutCallPayoff(strike=100.0, barrier=120.0)
    path = np.array([[100, 105, 110]])  # Never hits 120
    barrier_call(path)  # array([10.0])  # max(110-100, 0)
    """

    def __init__(self, strike: float, barrier: float):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if barrier <= strike:
            raise ValueError(f"Barrier must be above strike for up-out call, got barrier={barrier}, strike={strike}")
        self._strike = strike
        self._barrier = barrier

    @property
    def strike(self) -> float:
        """Strike price."""
        return self._strike

    @property
    def barrier(self) -> float:
        """Barrier level."""
        return self._barrier

    @property
    def is_path_dependent(self) -> bool:
        return True

    def __call__(self, path: np.ndarray) -> np.ndarray:
        from backend.math_kernels.payoff_kernels import barrier_up_out_call_payoff
        path_arr = _validate_path_array(path)
        result = np.empty(len(path_arr))
        for i in range(len(path_arr)):
            result[i] = barrier_up_out_call_payoff(path_arr[i], self._strike, self._barrier)
        return result

    def __repr__(self) -> str:
        return f"BarrierUpOutCallPayoff(strike={self._strike}, barrier={self._barrier})"


class BarrierDownOutPutPayoff(Payoff):
    """
    Down-and-out put payoff: knocked out if S <= barrier.

    Returns vanilla put payoff if barrier is never touched,
    otherwise returns 0.

    Parameters
    ----------
    strike : float
        Strike price (must be positive)
    barrier : float
        Lower barrier level (must be below strike)

    Examples
    --------
    barrier_put = BarrierDownOutPutPayoff(strike=100.0, barrier=80.0)
    path = np.array([[100, 95, 90]])  # Never hits 80
    barrier_put(path)  # array([10.0])  # max(100-90, 0)
    """

    def __init__(self, strike: float, barrier: float):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if barrier >= strike:
            raise ValueError(f"Barrier must be below strike for down-out put, got barrier={barrier}, strike={strike}")
        self._strike = strike
        self._barrier = barrier

    @property
    def strike(self) -> float:
        """Strike price."""
        return self._strike

    @property
    def barrier(self) -> float:
        """Barrier level."""
        return self._barrier

    @property
    def is_path_dependent(self) -> bool:
        return True

    def __call__(self, path: np.ndarray) -> np.ndarray:
        from backend.math_kernels.payoff_kernels import barrier_down_out_put_payoff
        path_arr = _validate_path_array(path)
        result = np.empty(len(path_arr))
        for i in range(len(path_arr)):
            result[i] = barrier_down_out_put_payoff(path_arr[i], self._strike, self._barrier)
        return result

    def __repr__(self) -> str:
        return f"BarrierDownOutPutPayoff(strike={self._strike}, barrier={self._barrier})"


class LookbackFloatingCallPayoff(Payoff):
    """
    Lookback call payoff (floating strike): S_T - min(S_t).

    The effective strike is the minimum price over the path.

    Examples
    --------
    lookback_call = LookbackFloatingCallPayoff()
    path = np.array([[100, 90, 110]])  # min=90, terminal=110
    lookback_call(path)  # array([20.0])  # 110 - 90
    """

    @property
    def is_path_dependent(self) -> bool:
        return True

    def __call__(self, path: np.ndarray) -> np.ndarray:
        from backend.math_kernels.payoff_kernels import lookback_floating_payoff
        path_arr = _validate_path_array(path)
        result = np.empty(len(path_arr))
        for i in range(len(path_arr)):
            result[i] = lookback_floating_payoff(path_arr[i], True)
        return result

    def __repr__(self) -> str:
        return "LookbackFloatingCallPayoff()"


class LookbackFloatingPutPayoff(Payoff):
    """
    Lookback put payoff (floating strike): max(S_t) - S_T.

    The effective strike is the maximum price over the path.

    Examples
    --------
    lookback_put = LookbackFloatingPutPayoff()
    path = np.array([[100, 110, 90]])  # max=110, terminal=90
    lookback_put(path)  # array([20.0])  # 110 - 90
    """

    @property
    def is_path_dependent(self) -> bool:
        return True

    def __call__(self, path: np.ndarray) -> np.ndarray:
        from backend.math_kernels.payoff_kernels import lookback_floating_payoff
        path_arr = _validate_path_array(path)
        result = np.empty(len(path_arr))
        for i in range(len(path_arr)):
            result[i] = lookback_floating_payoff(path_arr[i], False)
        return result

    def __repr__(self) -> str:
        return "LookbackFloatingPutPayoff()"


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
