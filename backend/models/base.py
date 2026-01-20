"""
Base Model Classes
==================

Abstract base class for unified financial models.

A Model owns its parameters and can create both:
    - Simulators (for path generation under P or Q measure)
    - Pricers (for option valuation)

This is the single source of truth for model configuration.

Author: Derivatives Pricing Project
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List, Dict, Any, TYPE_CHECKING, TypeVar, Generic

from .parameters.base import BaseParams

# Type variable for parameter classes
P = TypeVar('P', bound=BaseParams)

if TYPE_CHECKING:
    from backend.simulation.base import BaseSimulator
    from backend.option_pricing.base import BasePricer


class Measure(Enum):
    """
    Probability measure for simulation/pricing.

    P (Physical): Real-world measure, uses expected return mu as drift
    Q (Risk-Neutral): Risk-neutral measure, uses risk-free rate r as drift
    """
    P = "physical"
    Q = "risk_neutral"


class PricingCapability(Enum):
    """
    Pricing methods a model supports.

    ANALYTICAL: Closed-form solution (e.g., Black-Scholes)
    FFT: Fast Fourier Transform via characteristic function
    MONTE_CARLO: Monte Carlo simulation
    """
    ANALYTICAL = "analytical"
    FFT = "fft"
    MONTE_CARLO = "monte_carlo"


class BaseModel(ABC, Generic[P]):
    """
    Abstract base class for unified financial models.

    A Model is the single source of truth for a financial model:
        - Owns the model parameters (immutable)
        - Can create simulators for path generation
        - Can create pricers for option valuation
        - Provides characteristic function if available

    This design ensures consistency between simulation and pricing,
    as both share the same underlying parameters.

    Example
    -------
    model = HestonModel.from_params(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    simulator = model.create_simulator()
    pricer = model.create_pricer()  # Uses same parameters
    """

    def __init__(self, params: P):
        """
        Initialize model with parameters.

        Parameters
        ----------
        params : P
            Immutable parameter object for this model
        """
        self._params: P = params

    @property
    def params(self) -> P:
        """Immutable model parameters."""
        return self._params

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model name."""
        pass

    @property
    @abstractmethod
    def supported_pricing_methods(self) -> List[PricingCapability]:
        """List of pricing methods this model supports."""
        pass

    @classmethod
    @abstractmethod
    def from_params(cls, **kwargs) -> "BaseModel":
        """
        Create model from individual parameters.

        This is a convenience method that creates the appropriate
        parameter object and model instance.
        """
        pass

    @abstractmethod
    def create_simulator(self, **kwargs) -> "BaseSimulator":
        """
        Create a simulator for this model.

        The simulator uses the model's parameters and can simulate
        paths for price and (if applicable) volatility.

        Parameters
        ----------
        **kwargs
            Simulator-specific options (e.g., discretization scheme)

        Returns
        -------
        BaseSimulator
            Configured simulator instance
        """
        pass

    @abstractmethod
    def create_pricer(
        self,
        method: Optional[PricingCapability] = None,
        **kwargs
    ) -> "BasePricer":
        """
        Create an option pricer for this model.

        Parameters
        ----------
        method : PricingCapability, optional
            Preferred pricing method. If not specified, uses the
            most efficient available method.
        **kwargs
            Pricer-specific options

        Returns
        -------
        BasePricer
            Configured pricer instance
        """
        pass

    def characteristic_function(self, u, s0: float, t: float, r: float):
        """
        Characteristic function phi(u) = E^Q[exp(i*u*ln(S_T))].

        Override in models that support FFT pricing.

        Parameters
        ----------
        u : complex or array
            Frequency argument
        s0 : float
            Initial spot price
        t : float
            Time to maturity
        r : float
            Risk-free rate

        Returns
        -------
        complex or array
            Characteristic function value(s)
        """
        raise NotImplementedError(
            f"{self.model_name} does not have a characteristic function"
        )

    def get_parameters(self) -> Dict[str, Any]:
        """Return parameters as dictionary."""
        return self._params.to_dict()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._params})"
