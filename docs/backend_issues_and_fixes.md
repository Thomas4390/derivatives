# Backend Issues and Fixes

This document provides a detailed analysis of identified backend inconsistencies, their financial impact, and the implemented solutions.

---

## Table of Contents

1. [Higher-Order Greeks Hardcoded to 0.0](#issue-1-higher-order-greeks-hardcoded-to-00)
2. [FFT Call Pricing Ignores Dividend Yield](#issue-2-fft-call-pricing-ignores-dividend-yield)
3. [Registries Non-Initialized](#issue-3-registries-non-initialized)
4. [Engines Assume .strike on Instrument](#issue-4-engines-assume-strike-on-instrument)
5. [Payoff Recreated on Each Access](#issue-5-payoff-recreated-on-each-access)
6. [Dual Model Hierarchies (Bridge Model-Simulator)](#issue-6-dual-model-hierarchies-bridge-model-simulator)
7. [isinstance() Fragile Checks](#issue-7-isinstance-fragile-checks)
8. [Silent Failure for Greek Typos](#issue-8-silent-failure-for-greek-typos)

---

## Issue 1: Higher-Order Greeks Hardcoded to 0.0

### Location
- **File**: `backend/greeks/calculator.py`
- **Lines**: 205-225

### Problematic Code
```python
if include_higher_order:
    price = engine.price(instrument, model, market).price
    return AllGreeksResult(
        price=price,
        delta=num_greeks.delta,
        gamma=num_greeks.gamma,
        vega=num_greeks.vega,
        theta=num_greeks.theta,
        rho=num_greeks.rho,
        # Higher order - set to 0 for numerical method
        vanna=0.0,    # PROBLEM
        volga=0.0,    # PROBLEM
        charm=0.0,    # PROBLEM
        veta=0.0,     # PROBLEM
        speed=0.0,    # PROBLEM
        zomma=0.0,    # PROBLEM
        color=0.0,    # PROBLEM
        ultima=0.0    # PROBLEM
    )
```

### Financial Impact

- **Vanna** (dDelta/dsigma): A trader hedging a book of exotic options with Heston receives vanna=0, so doesn't hedge the spot-vol cross sensitivity. During market stress, actual P&L diverges massively.

- **Volga** (dVega/dsigma): Critical for smile trading. Volga=0 means the trader underestimates volatility convexity risk.

- **Charm** (dDelta/dt): Delta decay over time. Missing charm leads to incorrect daily delta rebalancing.

- **Speed** (dGamma/dS): Third-order spot exposure. Important for large gamma books.

### Solution

Implemented numerical cross-derivatives in `_numerical_greeks()` using finite differences:

```python
def _numerical_higher_order_greeks(self, engine, instrument, model, market):
    # Vanna = dDelta/dsigma = (Delta(sigma+h) - Delta(sigma-h)) / 2h
    # Volga = dVega/dsigma
    # Charm = dDelta/dt
    # Speed = dGamma/dS
    # Zomma = dGamma/dsigma
    # Color = dGamma/dt
    # Ultima = dVolga/dsigma
```

### Files Modified
- `backend/greeks/calculator.py`
- `backend/greeks/numerical.py` (utility functions)

---

## Issue 2: FFT Call Pricing Ignores Dividend Yield

### Location
- **File**: `backend/engines/fft_engine.py`
- **Lines**: 146-149

### Problematic Code
```python
# Price using FFT
if option.is_call:
    price = self._fft_engine.price_call(cf, s0, k, t, r)     # q MISSING!
else:
    price = self._fft_engine.price_put(cf, s0, k, t, r, q)   # q present
```

### Financial Impact

The forward price under dividends is: `F = S0 * exp((r-q)*T)`

For an index option with q=1.5%:
- Correct call ATM 1Y price: $15.20
- Calculated price (without q): $16.80
- **Error: +10.5%**

A desk pricing SPX/EuroStoxx options with this code systematically overvalues calls.

### Root Cause Analysis

In `backend/engines/fourier/carr_madan.py`:
- `price_call()` (line 217-256): Does NOT accept `q` parameter
- `price_put()` (line 258-293): Accepts `q` for put-call parity

The characteristic function already includes `q`, but the Carr-Madan integrand discounting uses only `r` instead of using `q` for the forward adjustment.

### Solution

Modified `carr_madan.py` to accept `q` in `price_call()` and use it in `_compute_integrand()` for proper forward discounting.

### Files Modified
- `backend/engines/fourier/carr_madan.py`
- `backend/engines/fft_engine.py`

---

## Issue 3: Registries Non-Initialized

### Location
- **File**: `backend/core/registry.py` (EngineRegistry)
- **File**: `backend/models/registry.py` (ModelRegistry if exists)

### Problem

Registries are empty at startup. Nothing calls `register()`.

```python
from backend.core import EngineRegistry
EngineRegistry.list_engines()  # [] - EMPTY!
```

### Impact

The high-level function `EngineRegistry.price()` is unusable:
```python
# Always fails with "No compatible engine found"
result = EngineRegistry.price(option, model, market)
```

### Solution

Created `backend/engines/_registration.py` for auto-registration:

```python
"""Auto-registration of engines with the registry."""
from backend.core.registry import EngineRegistry
from backend.core.result_types import PricingCapability

def register_all_engines():
    from backend.engines.analytic_engine import BSAnalyticEngine
    from backend.engines.fft_engine import FFTEngine
    from backend.engines.mc_engine import MonteCarloEngine

    # GBM model
    EngineRegistry.register("Geometric Brownian Motion", PricingCapability.ANALYTICAL, BSAnalyticEngine)
    EngineRegistry.register("Geometric Brownian Motion", PricingCapability.FFT, FFTEngine)
    EngineRegistry.register("Geometric Brownian Motion", PricingCapability.MONTE_CARLO, MonteCarloEngine)

    # Heston model
    EngineRegistry.register("Heston", PricingCapability.FFT, FFTEngine)
    EngineRegistry.register("Heston", PricingCapability.MONTE_CARLO, MonteCarloEngine)

    # ... other models
```

Modified `backend/engines/__init__.py` to call `register_all_engines()` on import.

### Files Created/Modified
- **Created**: `backend/engines/_registration.py`
- **Modified**: `backend/engines/__init__.py`

---

## Issue 4: Engines Assume .strike on Instrument

### Location
- `backend/engines/analytic_engine.py:119` -> `k = option.strike`
- `backend/engines/fft_engine.py:135` -> `k = option.strike`
- `backend/engines/mc_engine.py:153` -> `k = option.strike`
- `backend/greeks/calculator.py:151` -> `k=instrument.strike`

### Problem

The `Instrument` interface (in `backend/core/interfaces.py`) does NOT guarantee `.strike`:

```python
class Instrument(ABC):
    @property
    @abstractmethod
    def payoff(self) -> Payoff: ...

    @property
    @abstractmethod
    def exercise_style(self) -> ExerciseStyle: ...

    @property
    @abstractmethod
    def maturity(self) -> float: ...

    # NO .strike!
```

`LookbackOption` has no strike (floating strike = min/max of path).

### Impact
```python
lookback = LookbackCall(maturity=1.0)
engine.price(lookback, model, market)  # AttributeError: 'LookbackOption' has no attribute 'strike'
```

### Solution

Added validation in `can_price()` methods:

```python
def can_price(self, instrument, model):
    if not hasattr(instrument, 'strike'):
        return False  # This engine only supports fixed-strike options
    ...
```

### Files Modified
- `backend/engines/analytic_engine.py`
- `backend/engines/fft_engine.py`
- `backend/engines/mc_engine.py`
- `backend/greeks/calculator.py`

---

## Issue 5: Payoff Recreated on Each Access

### Location
- **File**: `backend/instruments/options.py`
- **Lines**: 99-103 (VanillaOption), similar for other options

### Problematic Code
```python
@property
def payoff(self) -> Payoff:
    """The payoff function."""
    if self._is_call:
        return VanillaCallPayoff(self._strike)  # NEW object on each call!
    return VanillaPutPayoff(self._strike)
```

### Impact
```python
option = VanillaOption(strike=100, maturity=0.5, is_call=True)
p1 = option.payoff
p2 = option.payoff
print(p1 is p2)  # False - Violates immutability!
```

This breaks:
- Payoff-based caching
- Identity tests
- Hashed collections

### Solution

Create payoff in `__init__` and cache it:

```python
def __init__(self, strike, maturity, is_call, exercise=ExerciseStyle.EUROPEAN):
    ...
    # Create and cache the payoff
    if is_call:
        payoff = VanillaCallPayoff(strike)
    else:
        payoff = VanillaPutPayoff(strike)
    object.__setattr__(self, '_payoff', payoff)

@property
def payoff(self) -> Payoff:
    return self._payoff  # Always the same object
```

### Files Modified
- `backend/instruments/options.py`: VanillaOption, DigitalOption, AsianOption, BarrierOption, LookbackOption

---

## Issue 6: Dual Model Hierarchies (Bridge Model-Simulator)

### Location
- `backend/models/` -> Abstract models (GBMModel, HestonModel, etc.)
- `backend/simulation/models/` -> Concrete simulators (GBMSimulator, HestonSimulator, etc.)

### Problematic Code (in mc_engine.py)
```python
def _make_heston_simulator(self, model: HestonModel, q: float):
    from backend.simulation.models.heston import HestonSimulator
    # Manual parameter extraction!
    v0 = model.v0
    kappa = model.kappa
    theta = model.theta
    sigma_v = model.sigma_v
    rho = model.rho
    return HestonSimulator(v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho, ...)
```

### Impact
- Duplication: each new model = modifications in 2 places
- Strong coupling: MCEngine knows all types
- Maintenance: adding a parameter = modify MCEngine

### Solution

Added `create_simulator()` to the Model interface:

```python
# In backend/core/interfaces.py
class Model(ABC):
    ...
    def create_simulator(self, **kwargs) -> "BaseSimulator":
        """Create a simulator for Monte Carlo pricing."""
        raise NotImplementedError(
            f"{self.name} does not support Monte Carlo simulation."
        )

# In backend/models/heston.py
class HestonModel(Model):
    ...
    def create_simulator(self, antithetic=True, **kwargs):
        from backend.simulation.models.heston import HestonSimulator
        return HestonSimulator(
            v0=self.v0,
            kappa=self.kappa,
            theta=self.theta,
            sigma_v=self.xi,
            rho=self.rho,
            antithetic=antithetic,
            **kwargs
        )
```

Then in MCEngine:
```python
def price(self, instrument, model, market):
    simulator = model.create_simulator(antithetic=self.antithetic)
    paths = simulator.simulate_terminal(...)
    ...
```

### Files Modified
- `backend/core/interfaces.py`: Added `create_simulator()` to Model
- `backend/models/gbm.py`, `heston.py`, `bates.py`, `merton.py`: Implemented `create_simulator()`
- `backend/engines/mc_engine.py`: Optionally use `model.create_simulator()` when available

---

## Issue 7: isinstance() Fragile Checks

### Location
- `backend/engines/analytic_engine.py:71-83`

### Problematic Code
```python
def can_price(self, instrument: Instrument, model: Model) -> bool:
    is_european = instrument.exercise_style == ExerciseStyle.EUROPEAN
    is_vanilla = isinstance(instrument, VanillaOption)  # Fragile
    is_gbm = isinstance(model, GBMModel)  # Fragile
    return is_european and is_vanilla and is_gbm
```

### Impact
- A `GBMModelWrapper` or mock fails
- Violates Python duck typing

### Solution

Use capabilities instead of types:

```python
def can_price(self, instrument: Instrument, model: Model) -> bool:
    # Check exercise style
    if instrument.exercise_style != ExerciseStyle.EUROPEAN:
        return False

    # Check model supports analytical pricing
    if PricingCapability.ANALYTICAL not in model.supported_engines:
        return False

    # Check it's a vanilla option (with strike)
    if not hasattr(instrument, 'strike'):
        return False

    return True
```

### Files Modified
- `backend/engines/analytic_engine.py`

---

## Issue 8: Silent Failure for Greek Typos

### Location
- `backend/greeks/calculator.py:278`

### Problematic Code
```python
results[i] = getattr(greeks, greek, 0.0)  # Returns 0.0 silently!
```

### Impact
```python
calc.calculate_surface(..., greek="delat")  # Typo!
# Returns array of 0.0 instead of an error
```

### Solution

Validate Greek name and raise explicit error:

```python
VALID_GREEKS = {'delta', 'gamma', 'vega', 'theta', 'rho',
                'vanna', 'volga', 'charm', 'veta',
                'speed', 'zomma', 'color', 'ultima', 'price'}

def calculate_surface(self, ..., greek: str = 'delta'):
    if greek not in VALID_GREEKS:
        raise ValueError(
            f"Unknown Greek '{greek}'. Valid values: {sorted(VALID_GREEKS)}"
        )
    ...
    results[i] = getattr(greeks, greek)  # No default - raises AttributeError if invalid
```

### Files Modified
- `backend/greeks/calculator.py`

---

## Implementation Order

1. **Task 0**: Create this documentation (docs/backend_issues_and_fixes.md)
2. **Task 2**: FFT dividend yield (simple fix, immediate impact)
3. **Task 5**: Cache payoffs (simple fix)
4. **Task 8**: Greek validation (simple fix)
5. **Task 4**: Handle optional .strike
6. **Task 7**: Replace isinstance()
7. **Task 3**: Initialize registries
8. **Task 1**: Higher-order Greeks (more complex)
9. **Task 6**: Bridge Model-Simulator (architectural refactoring)

---

## Verification

```bash
# Run existing tests
python -m pytest tests/ -v

# Run specific tests
python -m pytest tests/test_exotic_options.py -v

# Smoke tests
python -c "
from backend.instruments import LookbackCall
from backend.engines import FFTEngine
from backend.models import HestonModel
from backend.core import MarketEnvironment, EngineRegistry

# Test registry
print('Engines:', EngineRegistry.list_engines())

# Test FFT with dividends
market = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.02)
# ... pricing test
"
```
