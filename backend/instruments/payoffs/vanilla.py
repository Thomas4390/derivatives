"""
Terminal / spot-dependent payoff value-objects (vanilla, digital, spot, bond,
composite).
"""

from __future__ import annotations

import numpy as np

from backend.core.interfaces import Payoff
from backend.instruments.payoffs._internals import (
    _call_payoff,
    _digital_call_payoff,
    _digital_put_payoff,
    _put_payoff,
    _validate_spot_array,
)


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
        # __call__ evaluates every leg on one shared terminal-spot array; it has
        # no per-leg path dispatch, so a path-dependent leg cannot be honoured.
        # Reject it explicitly instead of silently mis-evaluating it on 1D spot.
        if any(payoff.is_path_dependent for _, payoff in legs):
            raise ValueError(
                "CompositePayoff supports only terminal (spot-dependent) legs; "
                "a path-dependent leg has no terminal-spot payoff here."
            )
        self._legs = legs

    @property
    def legs(self) -> list[tuple[int | float, Payoff]]:
        """List of (weight, payoff) tuples."""
        return self._legs

    @property
    def is_path_dependent(self) -> bool:
        # Always terminal: path-dependent legs are rejected in __init__.
        return False

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        spot_arr = _validate_spot_array(spot)
        result = np.zeros_like(spot_arr)
        for weight, payoff in self._legs:
            result += weight * payoff(spot_arr)
        return result

    def __repr__(self) -> str:
        legs_str = ", ".join(f"{w}*{p}" for w, p in self._legs)
        return f"CompositePayoff([{legs_str}])"
