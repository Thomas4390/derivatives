"""
Base Model Classes
==================

Abstract base class for unified financial models.

Provides:
- BaseModel: Abstract base class for financial models
- PricingCapability: Enum for supported pricing methods
- Measure: Enum for probability measure (P/Q)

Author: Thomas
Created: 2025
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.simulation.base import BaseSimulator


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


class BaseModel(ABC):
    """
    Abstract base class for financial models.

    Defines the interface for financial models used in pricing.
    Models should implement model_name, supported_pricing_methods,
    create_simulator, create_pricer, and get_parameters.
    """

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

    @abstractmethod
    def create_simulator(self, **kwargs) -> "BaseSimulator":
        """
        Create a simulator for this model.

        Parameters
        ----------
        **kwargs
            Simulator-specific options

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
    ):
        """
        Create an option pricer for this model.

        Parameters
        ----------
        method : PricingCapability, optional
            Preferred pricing method
        **kwargs
            Pricer-specific options

        Returns
        -------
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

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return parameters as dictionary."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.get_parameters()})"
