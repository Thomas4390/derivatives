"""
Core Interfaces
===============

Abstract base classes defining the three pillars of the pricing architecture:
- Instrument (the "What"): Financial contracts and payoffs
- Model (the "Physics"): Stochastic dynamics
- Engine (the "How"): Numerical methods

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np


# =============================================================================
# MEASURE ENUM (canonical definition)
# =============================================================================


class Measure(Enum):
    """Probability measure for simulation/pricing.

    Canonical members: PHYSICAL, RISK_NEUTRAL
    Backward-compat aliases: P, Q, P_MEASURE, Q_MEASURE
    """

    PHYSICAL = "physical"
    RISK_NEUTRAL = "risk_neutral"
    P = "physical"
    Q = "risk_neutral"
    P_MEASURE = "physical"
    Q_MEASURE = "risk_neutral"


# =============================================================================
# PROTOCOL: Priceable (shared structural type)
# =============================================================================


@runtime_checkable
class Priceable(Protocol):
    """Structural type for anything that can be priced.

    Both ``Instrument`` and ``StructuredProduct`` satisfy this protocol.
    Used by ``EngineRegistry``, ``GreeksCalculator``, and ``Portfolio``
    to accept any priceable contract without coupling to a specific ABC.
    """

    @property
    def maturity(self) -> float: ...


# =============================================================================
# CAPABILITY PROTOCOLS (for Model optional features)
# =============================================================================


@runtime_checkable
class CharacteristicFunctionCapable(Protocol):
    """Model that supports FFT pricing via characteristic function."""

    def characteristic_function(
        self, u: complex, s0: float, t: float, r: float, q: float = 0.0
    ) -> complex: ...

    def characteristic_function_vectorized(
        self, u: np.ndarray, s0: float, t: float, r: float, q: float = 0.0
    ) -> np.ndarray: ...


@runtime_checkable
class SDECapable(Protocol):
    """Model that supports Monte Carlo via SDE discretization."""

    def drift(self, s: float, v: float, t: float, r: float, q: float) -> float: ...

    def diffusion(self, s: float, v: float, t: float) -> float: ...


@runtime_checkable
class SimulatorCapable(Protocol):
    """Model that can create its own optimized simulator."""

    def create_simulator(self, **kwargs: Any) -> Any: ...


from backend.core.market import MarketEnvironment
from backend.core.result_types import (
    ExerciseStyle,
    PricingCapability,
    PricingResult,
)

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

    # ----- Portfolio algebra operators -----

    def __add__(self, other: object) -> Any:
        """Payoff + Payoff → CompositePayoff."""
        if isinstance(other, Payoff):
            CompositePayoff = _composite_payoff_cls()
            left = self._legs if isinstance(self, CompositePayoff) else [(1.0, self)]
            right = (
                other._legs if isinstance(other, CompositePayoff) else [(1.0, other)]
            )
            return CompositePayoff(list(left) + list(right))
        return NotImplemented

    def __radd__(self, other: object) -> Any:
        if other == 0:
            return self
        return NotImplemented

    def __sub__(self, other: object) -> Any:
        """Payoff - Payoff → CompositePayoff."""
        if isinstance(other, Payoff):
            CompositePayoff = _composite_payoff_cls()
            left = self._legs if isinstance(self, CompositePayoff) else [(1.0, self)]
            right = [
                (-weight, payoff)
                for weight, payoff in (
                    other._legs
                    if isinstance(other, CompositePayoff)
                    else [(1.0, other)]
                )
            ]
            return CompositePayoff(list(left) + right)
        return NotImplemented

    def __mul__(self, scalar: object) -> Any:
        """scalar * Payoff → CompositePayoff."""
        if isinstance(scalar, (int, float)):
            CompositePayoff = _composite_payoff_cls()
            if isinstance(self, CompositePayoff):
                return CompositePayoff(
                    [(weight * scalar, payoff) for weight, payoff in self._legs]
                )
            return CompositePayoff([(float(scalar), self)])
        return NotImplemented

    def __rmul__(self, scalar: object) -> Any:
        """Payoff * scalar → CompositePayoff."""
        return self.__mul__(scalar)

    def __neg__(self) -> Payoff:
        """-Payoff → CompositePayoff with weight -1."""
        return self.__mul__(-1.0)


# Module-level cache for the CompositePayoff class. We cannot import it at
# module load time because backend.instruments.payoffs depends on Payoff defined
# above (circular import). The first algebra operation memoizes the class.
_COMPOSITE_PAYOFF_CLS: type | None = None


def _composite_payoff_cls() -> type:
    global _COMPOSITE_PAYOFF_CLS
    if _COMPOSITE_PAYOFF_CLS is None:
        from backend.instruments.payoffs import CompositePayoff

        _COMPOSITE_PAYOFF_CLS = CompositePayoff
    return _COMPOSITE_PAYOFF_CLS


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

    def create_simulator(self, **kwargs: Any) -> Any:
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
            self.capability in model.supported_engines
            and instrument.exercise_style in self.supported_exercises
        )
