"""
Core Interfaces
===============

Abstract base classes defining the three pillars of the pricing architecture:
- Instrument (the "What"): Financial contracts and payoffs
- Model (the "Physics"): Stochastic dynamics
- Engine (the "How"): Numerical methods

Author: Thomas
Created: 2025
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from backend.core.market import MarketEnvironment
from backend.core.result_types import ExerciseStyle, PricingCapability, PricingResult

# =============================================================================
# PILLAR 1: PAYOFF (The Contract Terms)
# =============================================================================

class Payoff(ABC):
    """
    Atomic payoff function.

    The Payoff knows the contractual rules but NOTHING about:
    - Market data (spot, rates)
    - Stochastic dynamics
    - Pricing method

    Examples
    --------
    call = VanillaCallPayoff(strike=100)
    call(np.array([90, 100, 110]))  # array([ 0.,  0., 10.])
    """

    @abstractmethod
    def __call__(self, spot: np.ndarray) -> np.ndarray:
        """
        Evaluate payoff at terminal spot prices.

        Parameters
        ----------
        spot : np.ndarray
            Terminal spot prices (can be 1D array)

        Returns
        -------
        np.ndarray
            Payoff values at each spot price
        """
        pass

    @property
    @abstractmethod
    def is_path_dependent(self) -> bool:
        """Whether payoff depends on full path (vs terminal only)."""
        pass


# =============================================================================
# PILLAR 1: INSTRUMENT (The "What")
# =============================================================================

class Instrument(ABC):
    """
    Financial contract = Payoff + Exercise Style + Maturity.

    Instruments are IMMUTABLE after construction.

    Examples
    --------
    option = VanillaOption(strike=100, maturity=0.5, is_call=True)
    option.payoff(np.array([90, 100, 110]))  # array([ 0.,  0., 10.])
    """

    @property
    @abstractmethod
    def payoff(self) -> Payoff:
        """The payoff function."""
        pass

    @property
    @abstractmethod
    def exercise_style(self) -> ExerciseStyle:
        """When the payoff can be triggered."""
        pass

    @property
    @abstractmethod
    def maturity(self) -> float:
        """Time to expiration in years."""
        pass

    @property
    def is_european(self) -> bool:
        """True if European exercise."""
        return self.exercise_style == ExerciseStyle.EUROPEAN

    @property
    def is_american(self) -> bool:
        """True if American exercise."""
        return self.exercise_style == ExerciseStyle.AMERICAN

    @property
    def is_bermudan(self) -> bool:
        """True if Bermudan exercise."""
        return self.exercise_style == ExerciseStyle.BERMUDAN


# =============================================================================
# PILLAR 2: MODEL (The "Physics")
# =============================================================================

class Model(ABC):
    """
    Stochastic model for asset dynamics.

    The Model knows the mathematics of randomness but NOTHING about:
    - What a "Call" or "Put" is
    - How to price an option

    Models are IMMUTABLE after construction (params frozen).

    Examples
    --------
    model = GBMModel(sigma=0.20)
    model.characteristic_function(u=1.0, s0=100, t=0.5, r=0.05)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        pass

    @property
    @abstractmethod
    def supported_engines(self) -> list[PricingCapability]:
        """Which pricing methods this model supports."""
        pass

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return model parameters as dictionary."""
        pass

    def characteristic_function(
        self, u: complex, s0: float, t: float, r: float, q: float = 0.0
    ) -> complex:
        """
        Characteristic function phi(u) = E^Q[exp(i*u*ln(S_T))].

        Override in models that support FFT pricing.

        Parameters
        ----------
        u : complex
            Fourier transform variable
        s0 : float
            Initial spot price
        t : float
            Time to maturity
        r : float
            Risk-free rate
        q : float
            Dividend yield

        Returns
        -------
        complex
            Value of characteristic function at u
        """
        raise NotImplementedError(f"{self.name} has no characteristic function")

    def characteristic_function_vectorized(
        self, u: np.ndarray, s0: float, t: float, r: float, q: float = 0.0
    ) -> np.ndarray:
        """
        Vectorized characteristic function for FFT pricing.

        Override in models that support efficient FFT pricing.
        Default implementation delegates to characteristic_function(),
        which typically works because numpy operations vectorize naturally.

        Parameters
        ----------
        u : np.ndarray
            Array of Fourier transform variables
        s0 : float
            Initial spot price
        t : float
            Time to maturity
        r : float
            Risk-free rate
        q : float
            Dividend yield

        Returns
        -------
        np.ndarray
            Array of characteristic function values
        """
        return np.asarray(self.characteristic_function(u, s0, t, r, q))

    def drift(self, s: float, v: float, t: float, r: float, q: float) -> float:
        """
        Drift coefficient for SDE discretization.

        Override in models that support Monte Carlo.

        Parameters
        ----------
        s : float
            Current spot price
        v : float
            Current variance state. For stochastic volatility models (Heston,
            Bates), this is the instantaneous variance v_t. For GBM/Merton,
            this is sigma^2 (constant variance). Named 'v' for brevity in
            SDE discretization code.
        t : float
            Current time
        r : float
            Risk-free rate
        q : float
            Dividend yield

        Returns
        -------
        float
            Drift value
        """
        raise NotImplementedError(f"{self.name} has no SDE drift")

    def diffusion(self, s: float, v: float, t: float) -> float:
        """
        Diffusion coefficient for SDE discretization.

        Parameters
        ----------
        s : float
            Current spot price
        v : float
            Current variance state. For stochastic volatility models (Heston,
            Bates), this is the instantaneous variance v_t. For GBM/Merton,
            this is sigma^2 (constant variance). Named 'v' for brevity in
            SDE discretization code.
        t : float
            Current time

        Returns
        -------
        float
            Diffusion value
        """
        raise NotImplementedError(f"{self.name} has no SDE diffusion")

    def create_simulator(self, **kwargs):
        """
        Create a simulator for Monte Carlo pricing.

        This method provides a clean bridge between Model parameters
        and the simulation infrastructure. Each model knows how to
        create its own simulator with the correct parameters.

        Parameters
        ----------
        **kwargs
            Simulator-specific parameters (e.g., antithetic, scheme)

        Returns
        -------
        BaseSimulator
            Configured simulator instance

        Raises
        ------
        NotImplementedError
            If the model does not support Monte Carlo simulation
        """
        raise NotImplementedError(
            f"{self.name} does not support Monte Carlo simulation. "
            "Implement create_simulator() to enable MC pricing."
        )


# =============================================================================
# PILLAR 3: ENGINE (The "How")
# =============================================================================

class PricingEngine(ABC):
    """
    Numerical method for option valuation.

    The Engine is the ONLY component that bridges:
    - Instrument (asks: what's the payoff? what's the strike?)
    - Model (asks: what's the characteristic function? what's the SDE?)

    Engines are STATELESS calculators.

    Examples
    --------
    engine = BSAnalyticEngine()
    result = engine.price(option, model, market)
    print(result.price)
    """

    @property
    @abstractmethod
    def capability(self) -> PricingCapability:
        """What type of engine this is."""
        pass

    @property
    @abstractmethod
    def supported_exercises(self) -> list[ExerciseStyle]:
        """Which exercise styles this engine can handle."""
        pass

    @abstractmethod
    def price(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
    ) -> PricingResult:
        """
        Price an instrument under a model.

        Parameters
        ----------
        instrument : Instrument
            The financial contract to price
        model : Model
            The stochastic model for dynamics
        market : MarketEnvironment
            Current market conditions (spot, rate, dividend)

        Returns
        -------
        PricingResult
            Price and optional metadata
        """
        pass

    def can_price(self, instrument: Instrument, model: Model) -> bool:
        """
        Check if this engine can price the given instrument/model pair.

        Parameters
        ----------
        instrument : Instrument
            The financial contract
        model : Model
            The stochastic model

        Returns
        -------
        bool
            True if this engine can handle the combination
        """
        return (
            self.capability in model.supported_engines and
            instrument.exercise_style in self.supported_exercises
        )


if __name__ == "__main__":
    # Smoke test - just verify imports work
    print("Interfaces module loaded successfully")
    print(f"ExerciseStyle values: {list(ExerciseStyle)}")
    print(f"PricingCapability values: {list(PricingCapability)}")
    print("✓ Interfaces smoke test passed")
