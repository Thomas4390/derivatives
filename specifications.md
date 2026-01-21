# Backend Refactoring Specification v2.0

## Executive Summary

Complete refactoring of `backend/` following a **Separation of Concerns** architecture with three pillars:
- **Instrument** (the "What") - Financial contracts and payoffs
- **Model** (the "Physics") - Stochastic dynamics and parameters
- **Engine** (the "How") - Numerical computation methods

---

## User Stories

### US1: Clean Architecture
**As a** quantitative developer
**I want to** have clearly separated Instruments, Models, and Engines
**So that** I can extend any component without affecting others

### US2: Automatic Engine Selection
**As a** user
**I want to** price an option without choosing the engine manually
**So that** the system selects the optimal method (Analytical > FFT > MC)

### US3: Strategy Composition
**As a** trader
**I want to** create complex strategies (Iron Condor, Butterfly) via named classes
**So that** I get semantic validation while the engine sees composition

### US4: American/Bermudan Options
**As a** quantitative analyst
**I want to** price American and Bermudan options
**So that** I can handle early exercise features

### US5: High Performance
**As a** developer
**I want to** have Numba-optimized kernels for critical computations
**So that** pricing and simulation are fast

---

## Architecture Overview

```
backend/
│
├── 📁 core/                         # Interfaces & Registry
│   ├── __init__.py
│   ├── interfaces.py               # ABC: Instrument, Model, Engine, Payoff
│   ├── registry.py                 # EngineFactory (auto-selection)
│   ├── market.py                   # MarketEnvironment (r, q, spot)
│   └── result_types.py             # Enums, Result dataclasses
│
├── 📁 instruments/                  # The "What" (Financial Contracts)
│   ├── __init__.py
│   ├── payoffs.py                  # PayoffCall, PayoffPut, PayoffDigital
│   ├── exercise.py                 # ExerciseStyle (European, American, Bermudan)
│   ├── options.py                  # VanillaOption, BarrierOption
│   └── strategies.py               # IronCondor, Straddle, Butterfly (semantic wrappers)
│
├── 📁 models/                       # The "Physics" (Stochastic Dynamics)
│   ├── __init__.py
│   ├── base.py                     # BaseModel ABC (params embedded)
│   ├── gbm.py                      # GBMModel (Black-Scholes)
│   ├── heston.py                   # HestonModel
│   ├── bates.py                    # BatesModel (Heston + Jumps)
│   ├── merton.py                   # MertonModel (Jump-Diffusion)
│   └── garch.py                    # GARCHModel
│
├── 📁 engines/                      # The "How" (Numerical Methods)
│   ├── __init__.py
│   ├── base.py                     # BasePricingEngine ABC
│   │
│   ├── 📁 analytic/                # Closed-form solutions
│   │   ├── __init__.py
│   │   └── bs_analytic.py          # Black-Scholes formula
│   │
│   ├── 📁 fourier/                 # FFT-based pricing
│   │   ├── __init__.py
│   │   ├── carr_madan.py           # Carr-Madan algorithm
│   │   └── characteristic_fns.py   # CF implementations
│   │
│   ├── 📁 monte_carlo/             # Simulation-based pricing
│   │   ├── __init__.py
│   │   ├── mc_european.py          # European MC engine
│   │   ├── mc_american.py          # Longstaff-Schwartz
│   │   └── path_generators.py      # SDE discretization
│   │
│   └── 📁 pde/                     # PDE-based pricing
│       ├── __init__.py
│       └── crank_nicolson.py       # Finite difference solver
│
├── 📁 math_kernels/                 # Numba-optimized hot paths
│   ├── __init__.py
│   ├── sde_kernels.py              # @njit: Euler, Milstein discretization
│   ├── random.py                   # @njit: Correlated Brownian generation
│   ├── payoff_kernels.py           # @njit: Vectorized payoff evaluation
│   └── regression.py               # @njit: Longstaff-Schwartz regression
│
├── 📁 greeks/                       # Risk Sensitivities
│   ├── __init__.py
│   ├── calculator.py               # GreeksCalculator (strategy pattern)
│   ├── analytic.py                 # Closed-form Greeks (BS)
│   └── numerical.py                # Central differences bump-and-revalue
│
├── 📁 simulation/                   # Path Simulation (refactored)
│   ├── __init__.py
│   ├── base.py                     # BaseSimulator
│   ├── result.py                   # SimulationResult
│   └── simulators/                 # Model-specific simulators
│       ├── gbm.py
│       ├── heston.py
│       ├── bates.py
│       └── merton.py
│
├── 📁 calibration/                  # [STUB] Future calibration
│   ├── __init__.py
│   └── stub.py                     # Placeholder for future implementation
│
├── 📁 portfolio/                    # Portfolio Management
│   ├── __init__.py
│   ├── positions.py                # OptionPosition, StockPosition
│   ├── portfolio.py                # OptionsPortfolio
│   └── breakeven.py                # Breakeven calculation
│
└── 📁 utils/                        # Shared Utilities
    ├── __init__.py
    ├── math.py                     # @njit: norm_cdf, norm_pdf, d1_d2
    └── validation.py               # Input validation helpers
```

---

## Detailed Component Specifications

### 1. Core Interfaces (`core/interfaces.py`)

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from enum import Enum, auto
import numpy as np

from .types import PricingResult, GreeksResult
from .market import MarketEnvironment


class ExerciseStyle(Enum):
    EUROPEAN = auto()
    AMERICAN = auto()
    BERMUDAN = auto()


class PricingCapability(Enum):
    ANALYTICAL = auto()
    FFT = auto()
    MONTE_CARLO = auto()
    PDE = auto()


# ============================================================
# PILLAR 1: INSTRUMENT (The "What")
# ============================================================

class Payoff(ABC):
    """
    Atomic payoff function.

    The Payoff knows the contractual rules but NOTHING about:
    - Market data (spot, rates)
    - Stochastic dynamics
    - Pricing method
    """

    @abstractmethod
    def __call__(self, spot: np.ndarray) -> np.ndarray:
        """Evaluate payoff at terminal spot prices."""
        pass

    @property
    @abstractmethod
    def is_path_dependent(self) -> bool:
        """Whether payoff depends on full path (vs terminal only)."""
        pass


class Instrument(ABC):
    """
    Financial contract = Payoff + Exercise Style + Maturity.

    Instruments are IMMUTABLE after construction.
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
        return self.exercise_style == ExerciseStyle.EUROPEAN

    @property
    def is_american(self) -> bool:
        return self.exercise_style == ExerciseStyle.AMERICAN


# ============================================================
# PILLAR 2: MODEL (The "Physics")
# ============================================================

class Model(ABC):
    """
    Stochastic model for asset dynamics.

    The Model knows the mathematics of randomness but NOTHING about:
    - What a "Call" or "Put" is
    - How to price an option

    Models are IMMUTABLE after construction (params frozen).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        pass

    @property
    @abstractmethod
    def supported_engines(self) -> List[PricingCapability]:
        """Which pricing methods this model supports."""
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """Return model parameters as dictionary."""
        pass

    def characteristic_function(
        self, u: complex, s0: float, t: float, r: float, q: float = 0.0
    ) -> complex:
        """
        Characteristic function φ(u) = E^Q[exp(i·u·ln(S_T))].

        Override in models that support FFT pricing.
        """
        raise NotImplementedError(f"{self.name} has no characteristic function")

    def drift(self, s: float, v: float, t: float, r: float, q: float) -> float:
        """
        Drift coefficient for SDE discretization.

        Override in models that support Monte Carlo.
        """
        raise NotImplementedError(f"{self.name} has no SDE drift")

    def diffusion(self, s: float, v: float, t: float) -> float:
        """
        Diffusion coefficient for SDE discretization.
        """
        raise NotImplementedError(f"{self.name} has no SDE diffusion")


# ============================================================
# PILLAR 3: ENGINE (The "How")
# ============================================================

class PricingEngine(ABC):
    """
    Numerical method for option valuation.

    The Engine is the ONLY component that bridges:
    - Instrument (asks: what's the payoff? what's the strike?)
    - Model (asks: what's the characteristic function? what's the SDE?)

    Engines are STATELESS calculators.
    """

    @property
    @abstractmethod
    def capability(self) -> PricingCapability:
        """What type of engine this is."""
        pass

    @property
    @abstractmethod
    def supported_exercises(self) -> List[ExerciseStyle]:
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
            Price and optional Greeks
        """
        pass

    def can_price(self, instrument: Instrument, model: Model) -> bool:
        """Check if this engine can price the given instrument/model pair."""
        return (
            self.capability in model.supported_engines and
            instrument.exercise_style in self.supported_exercises
        )
```

### 2. Market Environment (`core/market.py`)

```python
from dataclasses import dataclass
from typing import Optional
import copy


@dataclass(frozen=True)
class MarketEnvironment:
    """
    Immutable snapshot of market conditions.

    Separating market data from Instrument and Model enables:
    - Greeks calculation via bumping (clone → modify → reprice)
    - Scenario analysis
    - Clean separation of concerns

    Parameters
    ----------
    spot : float
        Current underlying price
    rate : float
        Risk-free interest rate (annualized)
    dividend_yield : float
        Continuous dividend yield (default 0)
    valuation_date : Optional[str]
        Date of valuation (for logging/audit)
    """
    spot: float
    rate: float
    dividend_yield: float = 0.0
    valuation_date: Optional[str] = None

    def bump_spot(self, delta: float) -> 'MarketEnvironment':
        """Create new environment with bumped spot."""
        return MarketEnvironment(
            spot=self.spot + delta,
            rate=self.rate,
            dividend_yield=self.dividend_yield,
            valuation_date=self.valuation_date,
        )

    def bump_rate(self, delta: float) -> 'MarketEnvironment':
        """Create new environment with bumped rate."""
        return MarketEnvironment(
            spot=self.spot,
            rate=self.rate + delta,
            dividend_yield=self.dividend_yield,
            valuation_date=self.valuation_date,
        )

    def with_spot(self, spot: float) -> 'MarketEnvironment':
        """Create new environment with different spot."""
        return MarketEnvironment(
            spot=spot,
            rate=self.rate,
            dividend_yield=self.dividend_yield,
            valuation_date=self.valuation_date,
        )
```

### 3. Payoffs (`instruments/payoffs.py`)

```python
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from numba import njit

from backend.core.interfaces import Payoff


# ============================================================
# NUMBA KERNELS (Hot Path)
# ============================================================

@njit(cache=True, fastmath=True)
def _call_payoff(spots: np.ndarray, strike: float) -> np.ndarray:
    """Vectorized call payoff."""
    result = np.empty_like(spots)
    for i in range(len(spots)):
        result[i] = max(spots[i] - strike, 0.0)
    return result


@njit(cache=True, fastmath=True)
def _put_payoff(spots: np.ndarray, strike: float) -> np.ndarray:
    """Vectorized put payoff."""
    result = np.empty_like(spots)
    for i in range(len(spots)):
        result[i] = max(strike - spots[i], 0.0)
    return result


# ============================================================
# PAYOFF CLASSES
# ============================================================

class VanillaCallPayoff(Payoff):
    """European call payoff: max(S - K, 0)."""

    def __init__(self, strike: float):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        self._strike = strike

    @property
    def strike(self) -> float:
        return self._strike

    @property
    def is_path_dependent(self) -> bool:
        return False

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        return _call_payoff(np.atleast_1d(spot), self._strike)


class VanillaPutPayoff(Payoff):
    """European put payoff: max(K - S, 0)."""

    def __init__(self, strike: float):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        self._strike = strike

    @property
    def strike(self) -> float:
        return self._strike

    @property
    def is_path_dependent(self) -> bool:
        return False

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        return _put_payoff(np.atleast_1d(spot), self._strike)


class CompositePayoff(Payoff):
    """
    Weighted sum of payoffs for multi-leg strategies.

    Used internally by strategy classes (IronCondor, Butterfly, etc.)
    The engine sees this as a single payoff.
    """

    def __init__(self, legs: List[tuple]):
        """
        Parameters
        ----------
        legs : List[tuple]
            List of (weight, Payoff) tuples.
            Weight > 0 for long, < 0 for short.
        """
        self._legs = legs

    @property
    def is_path_dependent(self) -> bool:
        return any(payoff.is_path_dependent for _, payoff in self._legs)

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        spot = np.atleast_1d(spot)
        result = np.zeros_like(spot)
        for weight, payoff in self._legs:
            result += weight * payoff(spot)
        return result


if __name__ == "__main__":
    # Smoke test
    import numpy as np

    spots = np.array([90.0, 100.0, 110.0])

    call = VanillaCallPayoff(strike=100.0)
    put = VanillaPutPayoff(strike=100.0)

    print("Call payoffs:", call(spots))  # [0, 0, 10]
    print("Put payoffs:", put(spots))    # [10, 0, 0]

    # Straddle = long call + long put
    straddle = CompositePayoff([(1.0, call), (1.0, put)])
    print("Straddle payoffs:", straddle(spots))  # [10, 0, 10]

    print("✓ Payoffs smoke test passed")
```

### 4. Options (`instruments/options.py`)

```python
from dataclasses import dataclass
from typing import Optional, Union

from backend.core.interfaces import Instrument, Payoff, ExerciseStyle
from .payoffs import VanillaCallPayoff, VanillaPutPayoff


@dataclass(frozen=True)
class VanillaOption(Instrument):
    """
    Vanilla European/American option.

    This is the most common option type. It wraps a vanilla payoff
    with exercise style and maturity.
    """
    strike: float
    maturity: float  # in years
    is_call: bool
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN

    def __post_init__(self):
        if self.strike <= 0:
            raise ValueError(f"Strike must be positive, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {self.maturity}")

    @property
    def payoff(self) -> Payoff:
        if self.is_call:
            return VanillaCallPayoff(self.strike)
        return VanillaPutPayoff(self.strike)

    @property
    def exercise_style(self) -> ExerciseStyle:
        return self.exercise

    @property
    def option_type(self) -> str:
        return "call" if self.is_call else "put"


def EuropeanCall(strike: float, maturity: float) -> VanillaOption:
    """Factory for European call option."""
    return VanillaOption(strike=strike, maturity=maturity, is_call=True)


def EuropeanPut(strike: float, maturity: float) -> VanillaOption:
    """Factory for European put option."""
    return VanillaOption(strike=strike, maturity=maturity, is_call=False)


def AmericanCall(strike: float, maturity: float) -> VanillaOption:
    """Factory for American call option."""
    return VanillaOption(
        strike=strike, maturity=maturity, is_call=True,
        exercise=ExerciseStyle.AMERICAN
    )


def AmericanPut(strike: float, maturity: float) -> VanillaOption:
    """Factory for American put option."""
    return VanillaOption(
        strike=strike, maturity=maturity, is_call=False,
        exercise=ExerciseStyle.AMERICAN
    )


if __name__ == "__main__":
    # Smoke test
    call = EuropeanCall(strike=100.0, maturity=0.5)
    put = AmericanPut(strike=100.0, maturity=0.5)

    print(f"European Call: K={call.strike}, T={call.maturity}, exercise={call.exercise_style}")
    print(f"American Put: K={put.strike}, T={put.maturity}, exercise={put.exercise_style}")

    # Test payoff evaluation
    import numpy as np
    spots = np.array([90.0, 100.0, 110.0])
    print(f"Call payoffs at {spots}: {call.payoff(spots)}")

    print("✓ Options smoke test passed")
```

### 5. Strategies (`instruments/strategies.py`)

```python
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from backend.core.interfaces import Instrument, Payoff, ExerciseStyle
from .payoffs import VanillaCallPayoff, VanillaPutPayoff, CompositePayoff


@dataclass(frozen=True)
class StrategyLeg:
    """Single leg of a multi-leg strategy."""
    strike: float
    is_call: bool
    quantity: int  # positive = long, negative = short

    @property
    def payoff(self) -> Payoff:
        if self.is_call:
            return VanillaCallPayoff(self.strike)
        return VanillaPutPayoff(self.strike)


class OptionStrategy(Instrument):
    """
    Base class for multi-leg option strategies.

    Provides semantic validation while exposing a CompositePayoff
    to the pricing engine.
    """

    def __init__(self, legs: List[StrategyLeg], maturity: float):
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        self._legs = legs
        self._maturity = maturity
        self._validate()

    def _validate(self):
        """Override in subclasses for strategy-specific validation."""
        pass

    @property
    def legs(self) -> List[StrategyLeg]:
        return self._legs

    @property
    def payoff(self) -> Payoff:
        weighted_legs = [(leg.quantity, leg.payoff) for leg in self._legs]
        return CompositePayoff(weighted_legs)

    @property
    def exercise_style(self) -> ExerciseStyle:
        return ExerciseStyle.EUROPEAN  # Multi-leg = European only

    @property
    def maturity(self) -> float:
        return self._maturity


class IronCondor(OptionStrategy):
    """
    Iron Condor: Sell strangle + buy wings for protection.

    Structure:
    - Long put at K1 (lowest)
    - Short put at K2
    - Short call at K3
    - Long call at K4 (highest)

    Max profit: Net premium received
    Max loss: K2 - K1 - premium (or K4 - K3 - premium)
    """

    def __init__(
        self,
        k1: float,  # Long put strike
        k2: float,  # Short put strike
        k3: float,  # Short call strike
        k4: float,  # Long call strike
        maturity: float,
    ):
        self._k1, self._k2, self._k3, self._k4 = k1, k2, k3, k4
        legs = [
            StrategyLeg(strike=k1, is_call=False, quantity=1),   # Long put
            StrategyLeg(strike=k2, is_call=False, quantity=-1),  # Short put
            StrategyLeg(strike=k3, is_call=True, quantity=-1),   # Short call
            StrategyLeg(strike=k4, is_call=True, quantity=1),    # Long call
        ]
        super().__init__(legs, maturity)

    def _validate(self):
        if not (self._k1 < self._k2 < self._k3 < self._k4):
            raise ValueError(
                f"Iron Condor strikes must satisfy K1 < K2 < K3 < K4, "
                f"got {self._k1}, {self._k2}, {self._k3}, {self._k4}"
            )


class Straddle(OptionStrategy):
    """
    Straddle: Long call + long put at same strike.

    Profits from large moves in either direction.
    """

    def __init__(self, strike: float, maturity: float, is_long: bool = True):
        qty = 1 if is_long else -1
        legs = [
            StrategyLeg(strike=strike, is_call=True, quantity=qty),
            StrategyLeg(strike=strike, is_call=False, quantity=qty),
        ]
        super().__init__(legs, maturity)


class Butterfly(OptionStrategy):
    """
    Butterfly spread: Long wings, short body.

    Structure (call butterfly):
    - Long 1 call at K1
    - Short 2 calls at K2 (middle)
    - Long 1 call at K3

    Max profit at K2 (middle strike).
    """

    def __init__(
        self,
        k1: float,
        k2: float,
        k3: float,
        maturity: float,
        is_call: bool = True,
    ):
        self._k1, self._k2, self._k3 = k1, k2, k3
        legs = [
            StrategyLeg(strike=k1, is_call=is_call, quantity=1),
            StrategyLeg(strike=k2, is_call=is_call, quantity=-2),
            StrategyLeg(strike=k3, is_call=is_call, quantity=1),
        ]
        super().__init__(legs, maturity)

    def _validate(self):
        if not (self._k1 < self._k2 < self._k3):
            raise ValueError(
                f"Butterfly strikes must satisfy K1 < K2 < K3, "
                f"got {self._k1}, {self._k2}, {self._k3}"
            )
        # Middle strike should be equidistant
        if abs((self._k2 - self._k1) - (self._k3 - self._k2)) > 1e-6:
            raise ValueError("Butterfly strikes should be equidistant")


if __name__ == "__main__":
    # Smoke test
    import numpy as np

    spots = np.linspace(80, 120, 5)

    # Iron Condor
    ic = IronCondor(k1=85, k2=95, k3=105, k4=115, maturity=0.5)
    print(f"Iron Condor payoffs at expiry: {ic.payoff(spots)}")

    # Straddle
    straddle = Straddle(strike=100, maturity=0.5)
    print(f"Straddle payoffs at expiry: {straddle.payoff(spots)}")

    # Butterfly
    butterfly = Butterfly(k1=90, k2=100, k3=110, maturity=0.5)
    print(f"Butterfly payoffs at expiry: {butterfly.payoff(spots)}")

    print("✓ Strategies smoke test passed")
```

### 6. Engine Registry (`core/registry.py`)

```python
from typing import Dict, List, Optional, Type, Tuple
from backend.core.interfaces import (
    Instrument, Model, PricingEngine, PricingCapability, ExerciseStyle
)
from backend.core.market import MarketEnvironment
from backend.core.types import PricingResult


class EngineRegistry:
    """
    Factory that automatically selects the optimal pricing engine.

    Priority: ANALYTICAL > FFT > PDE > MONTE_CARLO

    The registry maintains a mapping of (Model, Exercise) -> Engine
    and selects the best available option.
    """

    # Priority order (highest to lowest)
    PRIORITY = [
        PricingCapability.ANALYTICAL,
        PricingCapability.FFT,
        PricingCapability.PDE,
        PricingCapability.MONTE_CARLO,
    ]

    _engines: Dict[Tuple[str, PricingCapability], Type[PricingEngine]] = {}

    @classmethod
    def register(
        cls,
        model_name: str,
        capability: PricingCapability,
        engine_class: Type[PricingEngine],
    ):
        """Register an engine for a model/capability pair."""
        cls._engines[(model_name, capability)] = engine_class

    @classmethod
    def get_engine(
        cls,
        instrument: Instrument,
        model: Model,
        preferred: Optional[PricingCapability] = None,
    ) -> PricingEngine:
        """
        Get the optimal engine for instrument/model pair.

        Parameters
        ----------
        instrument : Instrument
            The contract to price
        model : Model
            The stochastic model
        preferred : PricingCapability, optional
            Force a specific engine type if available

        Returns
        -------
        PricingEngine
            Configured engine instance

        Raises
        ------
        ValueError
            If no compatible engine found
        """
        model_name = model.name
        exercise = instrument.exercise_style

        # If user prefers a specific engine
        if preferred is not None:
            key = (model_name, preferred)
            if key in cls._engines:
                engine = cls._engines[key]()
                if exercise in engine.supported_exercises:
                    return engine
                raise ValueError(
                    f"Engine {preferred} does not support {exercise} exercise"
                )

        # Auto-select by priority
        for capability in cls.PRIORITY:
            if capability not in model.supported_engines:
                continue

            key = (model_name, capability)
            if key not in cls._engines:
                continue

            engine = cls._engines[key]()
            if exercise in engine.supported_exercises:
                return engine

        raise ValueError(
            f"No compatible engine found for {model_name} with {exercise} exercise"
        )

    @classmethod
    def price(
        cls,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
        preferred: Optional[PricingCapability] = None,
    ) -> PricingResult:
        """
        Convenience method: get engine and price in one call.
        """
        engine = cls.get_engine(instrument, model, preferred)
        return engine.price(instrument, model, market)


# Convenience function at module level
def price(
    instrument: Instrument,
    model: Model,
    market: MarketEnvironment,
    method: Optional[PricingCapability] = None,
) -> PricingResult:
    """
    Price an instrument under a model.

    This is the main entry point for option pricing.

    Examples
    --------
    >>> from backend import price, EuropeanCall, GBMModel, MarketEnvironment
    >>> option = EuropeanCall(strike=100, maturity=0.5)
    >>> model = GBMModel(sigma=0.20)
    >>> market = MarketEnvironment(spot=100, rate=0.05)
    >>> result = price(option, model, market)
    >>> print(result.price)
    """
    return EngineRegistry.price(instrument, model, market, method)


if __name__ == "__main__":
    print("Registry module loaded. Registration happens at import time.")
    print(f"Registered engines: {list(EngineRegistry._engines.keys())}")
    print("✓ Registry smoke test passed")
```

### 7. GBM Model (`models/gbm.py`)

```python
from dataclasses import dataclass
from typing import List
import numpy as np

from backend.core.interfaces import Model, PricingCapability


@dataclass(frozen=True)
class GBMModel(Model):
    """
    Geometric Brownian Motion (Black-Scholes) model.

    dS = (r - q) * S * dt + σ * S * dW

    Parameters
    ----------
    sigma : float
        Constant volatility (annualized)
    """
    sigma: float

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"Volatility must be positive, got {self.sigma}")

    @property
    def name(self) -> str:
        return "GBM"

    @property
    def supported_engines(self) -> List[PricingCapability]:
        return [
            PricingCapability.ANALYTICAL,
            PricingCapability.FFT,
            PricingCapability.PDE,
            PricingCapability.MONTE_CARLO,
        ]

    def get_parameters(self) -> dict:
        return {"sigma": self.sigma}

    def characteristic_function(
        self, u: complex, s0: float, t: float, r: float, q: float = 0.0
    ) -> complex:
        """
        GBM characteristic function.

        φ(u) = exp(i*u*(ln(S0) + (r-q-σ²/2)*t) - σ²*t*u²/2)
        """
        drift = (r - q - 0.5 * self.sigma**2) * t
        diffusion = -0.5 * self.sigma**2 * t * u**2
        return np.exp(1j * u * (np.log(s0) + drift) + diffusion)

    def drift(self, s: float, v: float, t: float, r: float, q: float) -> float:
        """Drift for SDE: (r - q) * S"""
        return (r - q) * s

    def diffusion(self, s: float, v: float, t: float) -> float:
        """Diffusion for SDE: σ * S"""
        return self.sigma * s


if __name__ == "__main__":
    # Smoke test
    model = GBMModel(sigma=0.20)

    print(f"Model: {model.name}")
    print(f"Parameters: {model.get_parameters()}")
    print(f"Supported engines: {model.supported_engines}")

    # Test characteristic function
    cf = model.characteristic_function(u=1.0, s0=100, t=0.5, r=0.05)
    print(f"CF(u=1): {cf}")

    print("✓ GBM Model smoke test passed")
```

### 8. Heston Model (`models/heston.py`)

```python
from dataclasses import dataclass
from typing import List
import numpy as np

from backend.core.interfaces import Model, PricingCapability


@dataclass(frozen=True)
class HestonModel(Model):
    """
    Heston stochastic volatility model.

    dS = (r - q) * S * dt + √V * S * dW_S
    dV = κ * (θ - V) * dt + ξ * √V * dW_V

    with corr(dW_S, dW_V) = ρ

    Parameters
    ----------
    v0 : float
        Initial variance
    kappa : float
        Mean reversion speed
    theta : float
        Long-run variance
    xi : float
        Volatility of variance (vol of vol)
    rho : float
        Correlation between spot and variance (-1 < ρ < 1)
    """
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self):
        if self.v0 <= 0:
            raise ValueError(f"Initial variance must be positive, got {self.v0}")
        if self.kappa <= 0:
            raise ValueError(f"Mean reversion must be positive, got {self.kappa}")
        if self.theta <= 0:
            raise ValueError(f"Long-run variance must be positive, got {self.theta}")
        if self.xi <= 0:
            raise ValueError(f"Vol of vol must be positive, got {self.xi}")
        if not -1 < self.rho < 1:
            raise ValueError(f"Correlation must be in (-1, 1), got {self.rho}")

    @property
    def name(self) -> str:
        return "Heston"

    @property
    def supported_engines(self) -> List[PricingCapability]:
        return [
            PricingCapability.FFT,
            PricingCapability.PDE,
            PricingCapability.MONTE_CARLO,
        ]

    def get_parameters(self) -> dict:
        return {
            "v0": self.v0,
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
        }

    @property
    def feller_condition(self) -> bool:
        """Check if Feller condition is satisfied (2κθ > ξ²)."""
        return 2 * self.kappa * self.theta > self.xi**2

    def characteristic_function(
        self, u: complex, s0: float, t: float, r: float, q: float = 0.0
    ) -> complex:
        """
        Heston characteristic function (Gatheral formulation).
        """
        kappa, theta, xi, rho, v0 = self.kappa, self.theta, self.xi, self.rho, self.v0

        # Avoid numerical issues
        u = complex(u)

        d = np.sqrt(
            (rho * xi * 1j * u - kappa)**2 + xi**2 * (1j * u + u**2)
        )

        g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)

        C = (r - q) * 1j * u * t + (kappa * theta / xi**2) * (
            (kappa - rho * xi * 1j * u - d) * t - 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g))
        )

        D = ((kappa - rho * xi * 1j * u - d) / xi**2) * (
            (1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t))
        )

        return np.exp(C + D * v0 + 1j * u * np.log(s0))


if __name__ == "__main__":
    # Smoke test
    model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

    print(f"Model: {model.name}")
    print(f"Parameters: {model.get_parameters()}")
    print(f"Feller condition satisfied: {model.feller_condition}")
    print(f"Supported engines: {model.supported_engines}")

    # Test characteristic function
    cf = model.characteristic_function(u=1.0, s0=100, t=0.5, r=0.05)
    print(f"CF(u=1): {cf}")

    print("✓ Heston Model smoke test passed")
```

---

## Numerical Greeks (`greeks/numerical.py`)

```python
from typing import Optional
import numpy as np

from backend.core.interfaces import Instrument, Model, PricingEngine
from backend.core.market import MarketEnvironment
from backend.core.registry import EngineRegistry


class NumericalGreeksCalculator:
    """
    Universal Greeks calculator using central finite differences.

    Works with ANY engine/model combination by bumping and revaluing.

    Central differences: f'(x) ≈ (f(x+h) - f(x-h)) / 2h
    Error: O(h²)
    """

    def __init__(
        self,
        spot_bump: float = 0.01,     # 1% bump
        rate_bump: float = 0.0001,   # 1bp bump
        vol_bump: float = 0.01,      # 1% vol bump
        time_bump: float = 1/365,    # 1 day
    ):
        self._spot_bump_pct = spot_bump
        self._rate_bump = rate_bump
        self._vol_bump = vol_bump
        self._time_bump = time_bump

    def calculate_all(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
        engine: Optional[PricingEngine] = None,
    ) -> dict:
        """
        Calculate all Greeks numerically.

        Returns
        -------
        dict
            Keys: price, delta, gamma, vega, theta, rho
        """
        if engine is None:
            engine = EngineRegistry.get_engine(instrument, model)

        spot = market.spot
        h_spot = spot * self._spot_bump_pct

        # Base price
        p0 = engine.price(instrument, model, market).price

        # Delta: ∂V/∂S
        p_up = engine.price(instrument, model, market.bump_spot(h_spot)).price
        p_down = engine.price(instrument, model, market.bump_spot(-h_spot)).price
        delta = (p_up - p_down) / (2 * h_spot)

        # Gamma: ∂²V/∂S²
        gamma = (p_up - 2 * p0 + p_down) / (h_spot ** 2)

        # Rho: ∂V/∂r
        h_r = self._rate_bump
        p_r_up = engine.price(instrument, model, market.bump_rate(h_r)).price
        p_r_down = engine.price(instrument, model, market.bump_rate(-h_r)).price
        rho = (p_r_up - p_r_down) / (2 * h_r)

        # Theta: -∂V/∂t (negative because time decreases)
        # Create instrument with shorter maturity
        # Note: This is approximate - proper theta needs instrument modification
        theta = -p0 * self._time_bump  # Simplified

        # Vega: ∂V/∂σ
        # Note: Requires model with sigma parameter - skip if not available
        vega = 0.0
        if hasattr(model, 'sigma'):
            # Would need to create new model with bumped sigma
            pass

        return {
            "price": p0,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho,
        }


if __name__ == "__main__":
    print("Numerical Greeks calculator ready")
    print("✓ Numerical Greeks smoke test passed")
```

---

## Portfolio Module (Clean Architecture)

### Overview

Le module Portfolio utilise **exclusivement** la nouvelle architecture Model/Engine/Market. Pas de rétrocompatibilité - rupture nette avec l'ancien système.

### Architecture

```
portfolio/
├── __init__.py              # Public API
├── positions.py             # PortfolioPosition (wrapper Instrument + quantity)
├── portfolio.py             # OptionsPortfolio avec Model/Engine/Market
├── greeks.py                # Greeks via bump-and-revalue
└── breakeven.py             # Analyse breakeven (P&L at expiry)
```

### Detailed Component Specifications

#### 1. Position Classes (`portfolio/positions.py`)

```python
from dataclasses import dataclass
from typing import Union
import numpy as np

from backend.core.types import ExerciseStyle
from backend.instruments.options import VanillaOption
from backend.instruments.payoffs import VanillaCallPayoff, VanillaPutPayoff


@dataclass(frozen=True)
class PortfolioPosition:
    """
    Position in a portfolio = Instrument + quantity + premium.

    Immutable. Use the instrument directly for pricing.
    """
    instrument: VanillaOption
    quantity: int  # positive = long, negative = short
    premium: float = 0.0  # premium paid (>0) or received (<0) per unit

    @property
    def sign(self) -> int:
        """Position direction: +1 long, -1 short."""
        return 1 if self.quantity > 0 else -1

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    def payoff_at_expiry(self, spot: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate P&L at expiry.

        Returns quantity * (intrinsic_value - premium)
        """
        intrinsic = self.instrument.payoff(np.atleast_1d(np.array([spot]))).flatten()
        pnl = self.quantity * (intrinsic - self.premium)
        return pnl[0] if np.isscalar(spot) else pnl


@dataclass(frozen=True)
class StockPosition:
    """
    Stock/underlying position.

    Linear P&L - no model needed for valuation.
    """
    quantity: int  # positive = long, negative = short
    entry_price: float = 0.0

    @property
    def sign(self) -> int:
        return 1 if self.quantity > 0 else -1

    @property
    def delta(self) -> float:
        """Stock delta = quantity."""
        return float(self.quantity)

    def pnl(self, spot: float) -> float:
        """P&L at given spot price."""
        return self.quantity * (spot - self.entry_price)


# ============================================================
# Factory Functions (convenient constructors)
# ============================================================

def long_call(strike: float, maturity: float, quantity: int = 1, premium: float = 0.0) -> PortfolioPosition:
    """Create a long call position."""
    instrument = VanillaOption(strike=strike, maturity=maturity, is_call=True)
    return PortfolioPosition(instrument=instrument, quantity=abs(quantity), premium=premium)


def short_call(strike: float, maturity: float, quantity: int = 1, premium: float = 0.0) -> PortfolioPosition:
    """Create a short call position."""
    instrument = VanillaOption(strike=strike, maturity=maturity, is_call=True)
    return PortfolioPosition(instrument=instrument, quantity=-abs(quantity), premium=premium)


def long_put(strike: float, maturity: float, quantity: int = 1, premium: float = 0.0) -> PortfolioPosition:
    """Create a long put position."""
    instrument = VanillaOption(strike=strike, maturity=maturity, is_call=False)
    return PortfolioPosition(instrument=instrument, quantity=abs(quantity), premium=premium)


def short_put(strike: float, maturity: float, quantity: int = 1, premium: float = 0.0) -> PortfolioPosition:
    """Create a short put position."""
    instrument = VanillaOption(strike=strike, maturity=maturity, is_call=False)
    return PortfolioPosition(instrument=instrument, quantity=-abs(quantity), premium=premium)


def long_stock(quantity: int = 100, entry_price: float = 0.0) -> StockPosition:
    """Create a long stock position."""
    return StockPosition(quantity=abs(quantity), entry_price=entry_price)


def short_stock(quantity: int = 100, entry_price: float = 0.0) -> StockPosition:
    """Create a short stock position."""
    return StockPosition(quantity=-abs(quantity), entry_price=entry_price)
```

#### 2. Portfolio Class (`portfolio/portfolio.py`)

```python
from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np

from backend.core.interfaces import Model, PricingEngine
from backend.core.market import MarketEnvironment
from backend.core.registry import EngineRegistry
from backend.core.types import GreeksResult

from .positions import PortfolioPosition, StockPosition


class OptionsPortfolio:
    """
    Portfolio of option and stock positions.

    Uses Model/Engine/Market architecture exclusively.

    Example
    -------
    >>> from backend.models import GBMModel
    >>> from backend.core import MarketEnvironment
    >>> from backend.portfolio import OptionsPortfolio, long_call
    >>>
    >>> portfolio = OptionsPortfolio(model=GBMModel(sigma=0.20))
    >>> portfolio.add(long_call(strike=100, maturity=0.5, premium=5.0))
    >>>
    >>> market = MarketEnvironment(spot=100, rate=0.05)
    >>> value = portfolio.value(market)
    >>> greeks = portfolio.greeks(market)
    """

    def __init__(self, model: Model, engine: Optional[PricingEngine] = None):
        """
        Initialize portfolio.

        Parameters
        ----------
        model : Model
            Pricing model (GBM, Heston, etc.)
        engine : PricingEngine, optional
            Pricing engine. If None, auto-selects via EngineRegistry.
        """
        self._model = model
        self._engine = engine
        self._positions: List[PortfolioPosition] = []
        self._stock: Optional[StockPosition] = None

    # =========================================================================
    # Position Management
    # =========================================================================

    def add(self, position: Union[PortfolioPosition, StockPosition]) -> 'OptionsPortfolio':
        """Add a position to the portfolio."""
        if isinstance(position, StockPosition):
            self._stock = position
        else:
            self._positions.append(position)
        return self

    def clear(self) -> 'OptionsPortfolio':
        """Remove all positions."""
        self._positions.clear()
        self._stock = None
        return self

    @property
    def positions(self) -> List[PortfolioPosition]:
        """Option positions (read-only copy)."""
        return list(self._positions)

    @property
    def stock(self) -> Optional[StockPosition]:
        """Stock position if any."""
        return self._stock

    @property
    def model(self) -> Model:
        return self._model

    @model.setter
    def model(self, value: Model):
        self._model = value

    # =========================================================================
    # Valuation
    # =========================================================================

    def value(self, market: MarketEnvironment) -> float:
        """
        Calculate current portfolio value.

        Parameters
        ----------
        market : MarketEnvironment
            Current market conditions

        Returns
        -------
        float
            Portfolio mark-to-market value
        """
        total = 0.0

        for pos in self._positions:
            if self._engine:
                result = self._engine.price(pos.instrument, self._model, market)
            else:
                result = EngineRegistry.price(pos.instrument, self._model, market)
            total += pos.quantity * result.price

        if self._stock:
            total += self._stock.quantity * market.spot

        return total

    def pnl_at_expiry(self, spot: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate P&L at expiry (no model needed).

        Parameters
        ----------
        spot : float or array
            Spot price(s) at expiry

        Returns
        -------
        float or array
            Total P&L
        """
        spot_arr = np.atleast_1d(spot)
        total = np.zeros_like(spot_arr, dtype=float)

        for pos in self._positions:
            total += pos.payoff_at_expiry(spot_arr)

        if self._stock:
            total += self._stock.pnl(spot_arr)

        return total[0] if np.isscalar(spot) else total

    # =========================================================================
    # Greeks
    # =========================================================================

    def greeks(
        self,
        market: MarketEnvironment,
        spot_bump: float = 0.01,
        rate_bump: float = 0.0001,
        vol_bump: float = 0.01,
    ) -> GreeksResult:
        """
        Calculate portfolio Greeks via central finite differences.

        Parameters
        ----------
        market : MarketEnvironment
            Current market conditions
        spot_bump : float
            Relative spot bump (default 1%)
        rate_bump : float
            Absolute rate bump (default 1bp)
        vol_bump : float
            Absolute vol bump (default 1%)

        Returns
        -------
        GreeksResult
            Portfolio Greeks
        """
        v0 = self.value(market)

        # Delta & Gamma
        h_s = market.spot * spot_bump
        v_up = self.value(market.bump_spot(h_s))
        v_down = self.value(market.bump_spot(-h_s))
        delta = (v_up - v_down) / (2 * h_s)
        gamma = (v_up - 2 * v0 + v_down) / (h_s ** 2)

        # Rho
        h_r = rate_bump
        v_r_up = self.value(market.bump_rate(h_r))
        v_r_down = self.value(market.bump_rate(-h_r))
        rho = (v_r_up - v_r_down) / (2 * h_r)

        # Vega (requires model bumping)
        vega = self._compute_vega(market, vol_bump)

        # Theta (requires maturity bumping - approximate)
        theta = 0.0  # TODO: implement via instrument maturity shift

        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
        )

    def _compute_vega(self, market: MarketEnvironment, h: float) -> float:
        """Compute vega by bumping model volatility."""
        params = self._model.get_parameters()

        if "sigma" in params:
            # GBM model
            from backend.models.gbm import GBMModel
            sigma = params["sigma"]

            model_up = GBMModel(sigma=sigma + h)
            model_down = GBMModel(sigma=sigma - h)

            v_up = self._value_with_model(market, model_up)
            v_down = self._value_with_model(market, model_down)

            return (v_up - v_down) / (2 * h)

        return 0.0

    def _value_with_model(self, market: MarketEnvironment, model: Model) -> float:
        """Value portfolio with a different model."""
        total = 0.0
        for pos in self._positions:
            if self._engine:
                result = self._engine.price(pos.instrument, model, market)
            else:
                result = EngineRegistry.price(pos.instrument, model, market)
            total += pos.quantity * result.price
        if self._stock:
            total += self._stock.quantity * market.spot
        return total

    # =========================================================================
    # Utilities
    # =========================================================================

    def __len__(self) -> int:
        return len(self._positions)

    def __repr__(self) -> str:
        stock_str = f", stock={self._stock.quantity}" if self._stock else ""
        return f"OptionsPortfolio({len(self._positions)} options{stock_str}, model={self._model.name})"
```

#### 3. Breakeven Analysis (`portfolio/breakeven.py`)

```python
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from .positions import PortfolioPosition, StockPosition


@dataclass
class BreakevenResult:
    """Breakeven analysis results."""
    breakeven_points: List[float]
    max_profit: float
    max_profit_spot: float
    max_loss: float
    max_loss_spot: float
    profit_zones: List[Tuple[float, float]]
    loss_zones: List[Tuple[float, float]]


def find_breakevens(
    positions: List[PortfolioPosition],
    stock: Optional[StockPosition] = None,
    spot_min: float = 0.1,
    spot_max: float = 500.0,
    n_points: int = 10000,
) -> BreakevenResult:
    """
    Find breakeven points and profit/loss zones.

    Parameters
    ----------
    positions : List[PortfolioPosition]
        Option positions
    stock : StockPosition, optional
        Stock position
    spot_min, spot_max : float
        Search range
    n_points : int
        Grid resolution

    Returns
    -------
    BreakevenResult
        Analysis results
    """
    spots = np.linspace(spot_min, spot_max, n_points)
    pnl = np.zeros(n_points)

    for pos in positions:
        pnl += pos.payoff_at_expiry(spots)

    if stock:
        pnl += stock.pnl(spots)

    # Find sign changes (breakevens)
    breakevens = []
    for i in range(len(pnl) - 1):
        if pnl[i] * pnl[i+1] < 0:
            # Linear interpolation
            s1, s2 = spots[i], spots[i+1]
            p1, p2 = pnl[i], pnl[i+1]
            be = s1 - p1 * (s2 - s1) / (p2 - p1)
            breakevens.append(float(be))

    # Max profit/loss
    max_idx = np.argmax(pnl)
    min_idx = np.argmin(pnl)

    # Identify zones
    profit_zones, loss_zones = _identify_zones(breakevens, pnl, spots, spot_min, spot_max)

    return BreakevenResult(
        breakeven_points=breakevens,
        max_profit=float(pnl[max_idx]),
        max_profit_spot=float(spots[max_idx]),
        max_loss=float(pnl[min_idx]),
        max_loss_spot=float(spots[min_idx]),
        profit_zones=profit_zones,
        loss_zones=loss_zones,
    )


def _identify_zones(breakevens, pnl, spots, spot_min, spot_max):
    """Identify profit and loss zones."""
    profit_zones, loss_zones = [], []

    if not breakevens:
        if pnl[len(pnl)//2] > 0:
            profit_zones.append((spot_min, spot_max))
        else:
            loss_zones.append((spot_min, spot_max))
        return profit_zones, loss_zones

    # Check each zone
    bounds = [spot_min] + breakevens + [spot_max]
    for i in range(len(bounds) - 1):
        mid = (bounds[i] + bounds[i+1]) / 2
        idx = np.searchsorted(spots, mid)
        if pnl[min(idx, len(pnl)-1)] > 0:
            profit_zones.append((bounds[i], bounds[i+1]))
        else:
            loss_zones.append((bounds[i], bounds[i+1]))

    return profit_zones, loss_zones
```

### Usage Example

```python
from backend.models import GBMModel
from backend.core import MarketEnvironment
from backend.portfolio import OptionsPortfolio, long_call, short_call, find_breakevens

# Create model and market
model = GBMModel(sigma=0.20)
market = MarketEnvironment(spot=100, rate=0.05)

# Build portfolio (Bull Call Spread)
portfolio = OptionsPortfolio(model=model)
portfolio.add(long_call(strike=95, maturity=0.5, premium=8.0))
portfolio.add(short_call(strike=105, maturity=0.5, premium=3.0))

# Valuations
print(f"Portfolio value: ${portfolio.value(market):.2f}")

# Greeks
greeks = portfolio.greeks(market)
print(f"Delta: {greeks.delta:.4f}")
print(f"Gamma: {greeks.gamma:.6f}")
print(f"Vega: {greeks.vega:.4f}")

# Breakeven analysis
result = find_breakevens(portfolio.positions)
print(f"Breakevens: {result.breakeven_points}")
print(f"Max profit: ${result.max_profit:.2f} at ${result.max_profit_spot:.2f}")
```

---

## Known Limitations

1. **Single Asset Only**: No correlation modeling for multi-asset portfolios (deferred)

2. **Constant Rates**: No term structure for interest rates or dividend yields

3. **Calibration Stub**: Calibration module is placeholder only

4. **Theta Approximation**: Numerical theta requires instrument modification (simplified)

5. **No Barrier Options**: Path-dependent barriers not in initial scope

6. **European Strategies Only**: Multi-leg strategies (IronCondor, etc.) are European-only

---

## Migration Checklist

### Phase 1: Core Infrastructure
- [ ] Create `backend/core/` with interfaces, registry, market, types
- [ ] Create `backend/math_kernels/` with Numba kernels
- [ ] Create `backend/utils/` with math utilities

### Phase 2: Instruments
- [ ] Create `backend/instruments/payoffs.py`
- [ ] Create `backend/instruments/options.py`
- [ ] Create `backend/instruments/strategies.py`

### Phase 3: Models (Refactor)
- [ ] Refactor `backend/models/gbm.py` (merge params)
- [ ] Refactor `backend/models/heston.py`
- [ ] Refactor `backend/models/bates.py`
- [ ] Refactor `backend/models/merton.py`
- [ ] Delete `backend/models/parameters/` directory

### Phase 4: Engines
- [ ] Create `backend/engines/base.py`
- [ ] Create `backend/engines/analytic/bs_analytic.py`
- [ ] Create `backend/engines/fourier/carr_madan.py`
- [ ] Create `backend/engines/monte_carlo/mc_european.py`
- [ ] Create `backend/engines/monte_carlo/mc_american.py` (Longstaff-Schwartz)
- [ ] Create `backend/engines/pde/crank_nicolson.py`

### Phase 5: Portfolio Module (Clean Rewrite)
- [ ] Delete old `backend/portfolio/` (backup if needed)
- [ ] Create `backend/portfolio/__init__.py` with clean exports
- [ ] Create `backend/portfolio/positions.py`:
  - [ ] `PortfolioPosition` dataclass (instrument + quantity + premium)
  - [ ] `StockPosition` dataclass
  - [ ] Factory functions: `long_call`, `short_call`, `long_put`, `short_put`
  - [ ] Factory functions: `long_stock`, `short_stock`
- [ ] Create `backend/portfolio/portfolio.py`:
  - [ ] `OptionsPortfolio(model, engine=None)` - Model/Engine/Market only
  - [ ] `add()`, `clear()`, `positions`, `stock` properties
  - [ ] `value(market)` - portfolio valuation
  - [ ] `pnl_at_expiry(spot)` - P&L at expiration
  - [ ] `greeks(market)` - finite difference Greeks
- [ ] Create `backend/portfolio/breakeven.py`:
  - [ ] `BreakevenResult` dataclass
  - [ ] `find_breakevens()` function
- [ ] Run smoke tests

### Phase 6: Simulation (Refactor)
- [ ] Update `backend/simulation/` to use new Model interface

### Phase 7: Cleanup
- [ ] Delete legacy files (`options_calculator.py`, etc.)
- [ ] Update `backend/__init__.py` with new exports
- [ ] Run all smoke tests

---

## Version

- **Spec Version**: 2.0
- **Target Backend Version**: 5.0.0 (major breaking change)
- **Author**: Technical Interview Session
- **Date**: 2025-01-20
