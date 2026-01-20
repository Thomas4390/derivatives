"""
Option Pricing Base Classes
===========================

Base classes and interfaces for all option pricing methods.

Author: Derivatives Pricing Project
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from enum import Enum


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


class PricingMethod(Enum):
    """Pricing method enumeration."""
    ANALYTICAL = "analytical"
    FFT = "fft"
    MONTE_CARLO = "monte_carlo"
    LATTICE = "lattice"


@dataclass
class PricingResult:
    """
    Container for option pricing results.

    Attributes
    ----------
    price : float
        Option price
    delta : float, optional
        First derivative w.r.t. spot
    gamma : float, optional
        Second derivative w.r.t. spot
    vega : float, optional
        Derivative w.r.t. volatility
    theta : float, optional
        Derivative w.r.t. time
    rho : float, optional
        Derivative w.r.t. interest rate
    method : PricingMethod
        Pricing method used
    computation_time : float
        Time to compute in seconds
    parameters : dict
        Model parameters used
    std_error : float, optional
        Standard error (for Monte Carlo)
    n_paths : int, optional
        Number of paths (for Monte Carlo)
    """
    price: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    method: PricingMethod = PricingMethod.ANALYTICAL
    computation_time: float = 0.0
    parameters: Optional[Dict[str, Any]] = None
    std_error: Optional[float] = None
    n_paths: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "price": self.price,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
            "method": self.method.value,
            "computation_time": self.computation_time,
            "std_error": self.std_error,
            "n_paths": self.n_paths,
        }


class BasePricer(ABC):
    """
    Abstract base class for all option pricers.

    All pricing implementations must inherit from this class
    and implement the required methods.
    """

    def __init__(self):
        self._model_name: str = "Base"
        self._method: PricingMethod = PricingMethod.ANALYTICAL

    @property
    def model_name(self) -> str:
        """Returns the model name."""
        return self._model_name

    @property
    def method(self) -> PricingMethod:
        """Returns the pricing method."""
        return self._method

    @abstractmethod
    def price(
        self,
        s0: float,
        k: float,
        t: float,
        r: float,
        option_type: Union[str, OptionType] = OptionType.CALL,
        **kwargs
    ) -> PricingResult:
        """
        Price a European option.

        Parameters
        ----------
        s0 : float
            Current spot price
        k : float
            Strike price
        t : float
            Time to maturity in years
        r : float
            Risk-free interest rate
        option_type : str or OptionType
            'call' or 'put'
        **kwargs
            Additional model-specific parameters

        Returns
        -------
        PricingResult
            Pricing result with price and Greeks
        """
        pass

    @abstractmethod
    def price_surface(
        self,
        s0: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        r: float,
        option_type: Union[str, OptionType] = OptionType.CALL,
        **kwargs
    ) -> np.ndarray:
        """
        Price options across a strike-maturity surface.

        Parameters
        ----------
        s0 : float
            Current spot price
        strikes : np.ndarray
            Array of strike prices
        maturities : np.ndarray
            Array of maturities
        r : float
            Risk-free interest rate
        option_type : str or OptionType
            'call' or 'put'

        Returns
        -------
        np.ndarray
            2D array of prices [strikes x maturities]
        """
        pass

    def _parse_option_type(self, option_type: Union[str, OptionType]) -> OptionType:
        """Convert string to OptionType enum."""
        if isinstance(option_type, str):
            return OptionType(option_type.lower())
        return option_type

    def _validate_inputs(self, s0: float, k: float, t: float, r: float) -> None:
        """Validate common pricing inputs."""
        if s0 <= 0:
            raise ValueError(f"Spot price must be positive, got {s0}")
        if k <= 0:
            raise ValueError(f"Strike price must be positive, got {k}")
        if t < 0:
            raise ValueError(f"Time to maturity must be non-negative, got {t}")
        if r < -1:
            raise ValueError(f"Interest rate seems invalid, got {r}")


class AnalyticalPricer(BasePricer):
    """Base class for analytical (closed-form) pricing methods."""

    def __init__(self):
        super().__init__()
        self._method = PricingMethod.ANALYTICAL


class FFTPricer(BasePricer):
    """Base class for FFT-based pricing methods."""

    def __init__(self):
        super().__init__()
        self._method = PricingMethod.FFT

    @abstractmethod
    def characteristic_function(
        self,
        u: np.ndarray,
        s0: float,
        t: float,
        r: float,
        **kwargs
    ) -> np.ndarray:
        """
        Compute the characteristic function phi(u).

        Parameters
        ----------
        u : np.ndarray
            Complex frequency values
        s0 : float
            Current spot price
        t : float
            Time to maturity
        r : float
            Risk-free rate

        Returns
        -------
        np.ndarray
            Characteristic function values
        """
        pass


class MonteCarloPricer(BasePricer):
    """Base class for Monte Carlo pricing methods."""

    def __init__(self):
        super().__init__()
        self._method = PricingMethod.MONTE_CARLO

    @abstractmethod
    def simulate_terminal(
        self,
        s0: float,
        t: float,
        r: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Simulate terminal stock prices under risk-neutral measure.

        Parameters
        ----------
        s0 : float
            Initial spot price
        t : float
            Time to maturity
        r : float
            Risk-free rate
        n_paths : int
            Number of simulation paths
        n_steps : int
            Number of time steps
        seed : int, optional
            Random seed

        Returns
        -------
        np.ndarray
            Terminal prices, shape (n_paths,)
        """
        pass
