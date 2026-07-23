"""
Path-dependent exotic payoff value-objects (Asian, barrier, lookback,
low-point-forward).
"""

from __future__ import annotations

import numpy as np

from backend.core.interfaces import Payoff
from backend.instruments.payoffs._internals import _validate_path_array
from backend.math_kernels.payoff_kernels import (
    asian_arithmetic_payoff_batch,
    barrier_down_out_put_payoff_batch,
    barrier_up_out_call_payoff_batch,
    lookback_discounted_call_payoff_batch,
    lookback_floating_payoff_batch,
    low_point_forward_payoff_batch,
)


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
        return True

    def __call__(self, path: np.ndarray) -> np.ndarray:
        path_arr = _validate_path_array(path)
        return asian_arithmetic_payoff_batch(path_arr, self._strike, True)

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
        return True

    def __call__(self, path: np.ndarray) -> np.ndarray:
        path_arr = _validate_path_array(path)
        return asian_arithmetic_payoff_batch(path_arr, self._strike, False)

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

    _strike: float
    _barrier: float

    def __init__(self, strike: float, barrier: float) -> None:
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if barrier <= strike:
            raise ValueError(
                f"Barrier must be above strike for up-out call, got barrier={barrier}, strike={strike}"
            )
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
        path_arr = _validate_path_array(path)
        return barrier_up_out_call_payoff_batch(path_arr, self._strike, self._barrier)

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

    _strike: float
    _barrier: float

    def __init__(self, strike: float, barrier: float) -> None:
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if barrier >= strike:
            raise ValueError(
                f"Barrier must be below strike for down-out put, got barrier={barrier}, strike={strike}"
            )
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
        path_arr = _validate_path_array(path)
        return barrier_down_out_put_payoff_batch(path_arr, self._strike, self._barrier)

    def __repr__(self) -> str:
        return (
            f"BarrierDownOutPutPayoff(strike={self._strike}, barrier={self._barrier})"
        )


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
        path_arr = _validate_path_array(path)
        return lookback_floating_payoff_batch(path_arr, True)

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
        path_arr = _validate_path_array(path)
        return lookback_floating_payoff_batch(path_arr, False)

    def __repr__(self) -> str:
        return "LookbackFloatingPutPayoff()"


class LookbackDiscountedCallPayoff(Payoff):
    """
    Lookback call payoff with S0/K_float discount (Globe Trotter Discounted).

    Payoff: (S_T - min(S)) * (S_0 / min(S))

    This is the standard floating-strike lookback call multiplied by the
    discount factor S_0/K_float, where K_float = min(S_t) over observation dates.

    Examples
    --------
    payoff = LookbackDiscountedCallPayoff()
    path = np.array([[100, 80, 110]])  # min=80, S_T=110, S_0=100
    payoff(path)  # (110-80) * (100/80) = 37.5
    """

    @property
    def is_path_dependent(self) -> bool:
        return True

    def __call__(self, path: np.ndarray) -> np.ndarray:
        path_arr = _validate_path_array(path)
        return lookback_discounted_call_payoff_batch(path_arr)

    def __repr__(self) -> str:
        return "LookbackDiscountedCallPayoff()"


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
        return low_point_forward_payoff_batch(path_arr)

    def __repr__(self) -> str:
        return "LowPointForwardPayoff()"
