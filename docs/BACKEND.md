# Backend Module Documentation

**Version**: 5.3.0
**Author**: Thomas
**Created**: 2025

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Architecture: Three Pillars](#architecture-three-pillars)
4. [Module Structure](#module-structure)
5. [Core Module](#core-module)
6. [Models Module](#models-module)
7. [Engines Module](#engines-module)
8. [Simulation Module](#simulation-module)
9. [Instruments Module](#instruments-module)
10. [Greeks Module](#greeks-module)
11. [Portfolio Module](#portfolio-module)
12. [Utils Module](#utils-module)
13. [Math Kernels Module](#math-kernels-module)
14. [Cross-Cutting Conventions](#cross-cutting-conventions)
15. [Inter-Module Relationships](#inter-module-relationships)
16. [Complete Examples](#complete-examples)

---

## Overview

The `backend` package is a high-performance derivatives pricing and portfolio management library. It implements a clean separation of concerns through the **Three Pillars Architecture**:

- **Instrument** (What): Defines the contract being priced (calls, puts, strategies)
- **Model** (Physics): Captures market dynamics (GBM, Heston, GARCH, etc.)
- **Engine** (How): Computes the price (Analytic, FFT, Monte Carlo)

### Key Features

- Numba JIT compilation for high performance
- Support for multiple pricing models (GBM, Heston, Bates, Merton, GARCH family)
- Multiple pricing engines (Black-Scholes analytic, Carr-Madan FFT, Monte Carlo)
- Full Greeks computation (analytic and numerical)
- Portfolio management with P&L analysis
- Strategy builders (Iron Condor, Straddle, Butterfly, etc.)

---

## Installation

### From Source (Development Mode)

```bash
# Navigate to the project root
cd /path/to/derivatives

# Install in editable mode
pip install -e .

# Or with uv (recommended)
uv pip install -e .
```

### Dependencies

The package requires Python 3.13+ and the following dependencies (from `pyproject.toml`):

```toml
[project]
name = "derivatives"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "numpy>=2.0,<2.4",
    "numba>=0.62.0",
    "scipy>=1.15.0",
    "streamlit>=1.40.0",
    "plotly>=5.0.0",
    "scipy-stubs~=1.17.0",
]
```

### Build System

The project uses **Hatchling** as the build backend:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["backend"]
```

### Verification

After installation, verify the package:

```python
import backend
print(backend.__version__)  # Should print "5.0.0"

# Quick test
from backend import GBMModel, VanillaOption, BSAnalyticEngine, MarketEnvironment

market = MarketEnvironment(spot=100, rate=0.05)
option = VanillaOption(strike=100, maturity=0.5, is_call=True)
model = GBMModel(sigma=0.2)
engine = BSAnalyticEngine()

price = engine.price(option, model, market)
print(f"Price: ${price.price:.4f}")
```

---

## Architecture: Three Pillars

The framework follows a clean separation of concerns:

```
+---------------------------------------------------------+
|                    PRICING WORKFLOW                       |
+---------------------------------------------------------+
|                                                           |
|   Instrument          Model              Engine           |
|   ----------         -----              ------           |
|   (What)             (Physics)          (How)            |
|                                                           |
|   VanillaOption  +   GBMModel      +   BSAnalyticEngine  |
|   EuropeanCall       HestonModel       FFTEngine         |
|   AmericanPut        BatesModel        MonteCarloEngine  |
|   OptionStrategy     MertonModel                         |
|                      GARCHModel                          |
|                                                           |
|   -----------------------------------------------        |
|                           |                               |
|                    PricingResult                          |
|                    (price, greeks, metadata)              |
|                                                           |
+---------------------------------------------------------+
```

### Example: Three Pillars in Action

```python
from backend import (
    VanillaOption,      # Instrument
    HestonModel,        # Model
    FFTEngine,          # Engine
    MarketEnvironment,  # Market data
)

# 1. Define the instrument (WHAT we're pricing)
option = VanillaOption(
    strike=100.0,
    maturity=0.5,
    is_call=True
)

# 2. Define the model (PHYSICS of the underlying)
model = HestonModel(
    v0=0.04,      # Initial variance
    kappa=2.0,    # Mean reversion speed
    theta=0.04,   # Long-run variance
    xi=0.3,       # Vol of vol
    rho=-0.7      # Correlation
)

# 3. Define market conditions
market = MarketEnvironment(
    spot=100.0,
    rate=0.05,
    dividend_yield=0.0
)

# 4. Choose pricing engine (HOW to compute)
engine = FFTEngine()

# 5. Price the option
result = engine.price(option, model, market)
print(f"Price: ${result.price:.4f}")
```

---

## Module Structure

```
backend/
+-- __init__.py              # Unified API exports
+-- core/                    # Abstract interfaces and base types
|   +-- __init__.py
|   +-- interfaces.py        # Instrument, Model, PricingEngine protocols
|   +-- market.py            # MarketEnvironment
|   +-- result_types.py      # PricingResult, GreeksResult, ExerciseStyle
|   +-- registry.py          # EngineRegistry, price() convenience
+-- models/                  # Pricing models
|   +-- __init__.py
|   +-- base.py              # BaseModel, Measure
|   +-- gbm.py               # GBMModel
|   +-- heston.py            # HestonModel
|   +-- merton.py            # MertonModel
|   +-- bates.py             # BatesModel
|   +-- garch.py             # GARCHModel, NGARCHModel, GJRGARCHModel
|   +-- vol_bump.py          # create_vol_bumped_model, create_vol_bumped_pair
|   +-- registry.py          # ModelRegistry
|   +-- characteristic_functions/  # CFs for FFT pricing
|       +-- heston_cf.py
|       +-- merton_cf.py
|       +-- bates_cf.py
+-- engines/                 # Pricing engines
|   +-- __init__.py
|   +-- analytic_engine.py   # BSAnalyticEngine (Black-Scholes)
|   +-- fft_engine.py        # FFTEngine (Carr-Madan)
|   +-- mc_engine.py         # MonteCarloEngine (Monte Carlo)
|   +-- exotic_engine.py     # ExoticAnalyticEngine (analytic exotic pricing)
|   +-- _registration.py     # Engine registry setup
|   +-- vectorized_bs.py     # Numba-vectorized BS Greeks
|   +-- fourier/
|   |   +-- carr_madan.py    # Carr-Madan FFT algorithm
|   +-- monte_carlo/
|       +-- mc_base.py       # Generic MC engine
|       +-- garch_pricer.py  # GARCH-specific pricer with LRNVR
+-- simulation/              # Monte Carlo simulators
|   +-- __init__.py
|   +-- base.py              # BaseSimulator, SimulationResult
|   +-- enums.py             # ModelType, DiscretizationScheme, Measure
|   +-- factory.py           # create_simulator factory
|   +-- risk_engine.py       # RiskMetrics, compute_risk_metrics
|   +-- models/              # Concrete simulators
|       +-- gbm.py
|       +-- heston.py
|       +-- merton.py
|       +-- bates.py
|       +-- garch.py         # GARCH, NGARCH, GJR-GARCH
+-- instruments/             # Financial instruments
|   +-- __init__.py
|   +-- options.py           # VanillaOption, DigitalOption, exotics
|   +-- payoffs.py           # VanillaCallPayoff, exotic payoffs
|   +-- strategies.py        # OptionStrategy, IronCondor, Straddle
|   +-- exercise.py          # EuropeanExercise, AmericanExercise, BermudanExercise
+-- greeks/                  # Greeks calculation
|   +-- __init__.py
|   +-- calculator.py        # GreeksCalculator, calculate_greeks
|   +-- analytic.py          # bs_greeks_*, analytical formulas
|   +-- numerical.py         # finite_difference_greeks, ModelNumericalGreeks
|   +-- _instrument_utils.py # create_decayed_instrument (time-bump helper)
+-- portfolio/               # Portfolio management
|   +-- __init__.py
|   +-- portfolio.py         # OptionsPortfolio
|   +-- positions.py         # PortfolioPosition, StockPosition
|   +-- breakeven.py         # BreakevenCalculator
|   +-- risk_analysis.py     # RiskProfile, check_unlimited_risk
|   +-- greeks_surfaces.py   # 3D Greeks surfaces (Numba-optimized)
|   +-- pnl.py               # P&L calculations, RiskMetrics
|   +-- factory.py           # long_call, short_put, etc.
+-- utils/                   # Utility functions
|   +-- __init__.py
|   +-- math.py              # bs_price, implied_vol, Greeks
|   +-- distributions.py     # norm_cdf, norm_pdf (Numba)
+-- math_kernels/            # Low-level numerical kernels
    +-- __init__.py
    +-- sde_kernels.py       # Euler, Milstein, QE discretization
    +-- payoff_kernels.py    # Vectorized payoff computation
    +-- regression.py        # Longstaff-Schwartz for Americans
    +-- random.py            # Correlated Brownian, antithetic
```

---

## Core Module

**Location**: `backend/core/`

The core module defines abstract interfaces and base types used throughout the framework.

### Key Components

| Component | Description |
|-----------|-------------|
| `Instrument` | Protocol for tradeable instruments |
| `Model` | Protocol for pricing models |
| `PricingEngine` | Protocol for pricing engines |
| `Payoff` | Protocol for option payoffs |
| `MarketEnvironment` | Container for market data |
| `PricingResult` | Result container with price and metadata |
| `GreeksResult` | Result container for all 13 Greeks |
| `ExerciseStyle` | Enum for European/American/Bermudan exercise |
| `PricingCapability` | Enum for engine types |
| `EngineRegistry` | Factory for auto-selecting optimal engine |

### MarketEnvironment

Immutable snapshot of market conditions. Frozen dataclass enabling clean bumping for Greeks.

```python
@dataclass(frozen=True)
class MarketEnvironment:
    spot: float                          # Current underlying price
    rate: float                          # Risk-free rate (annualized)
    dividend_yield: float = 0.0          # Continuous dividend yield
    valuation_date: Optional[str] = None # Date for logging/audit
```

**Bump methods** (return new instances with modified values):

| Method | Signature | Description |
|--------|-----------|-------------|
| `bump_spot(delta)` | `(float) -> MarketEnvironment` | `spot + delta` |
| `bump_rate(delta, validate=True)` | `(float, bool) -> MarketEnvironment` | `rate + delta` |
| `bump_dividend(delta, validate=True)` | `(float, bool) -> MarketEnvironment` | `dividend_yield + delta` |
| `with_spot(spot)` | `(float) -> MarketEnvironment` | Replace spot |
| `with_rate(rate, validate=False)` | `(float, bool) -> MarketEnvironment` | Replace rate (bypasses validation) |
| `with_dividend(dividend_yield, validate=False)` | `(float, bool) -> MarketEnvironment` | Replace dividend (bypasses validation) |

```python
from backend import MarketEnvironment

market = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.02)

# Bump pattern for numerical delta
market_up = market.bump_spot(+0.5)    # spot = 100.5
market_dn = market.bump_spot(-0.5)    # spot =  99.5
# delta ~ (price_up - price_dn) / (2 * 0.5)

# Replace spot entirely
market_new = market.with_spot(110.0)  # spot = 110

# Stress testing with extreme rates (bypass validation)
market_stress = market.with_rate(2.0)  # No ValueError
```

### PricingResult

```python
@dataclass(frozen=True)
class PricingResult:
    price: float                   # Option price (premium)
    engine: str = ""               # Name of engine that produced result
    model: str = ""                # Name of model used
    error: Optional[float] = None  # Standard error (MC methods)
```

### GreeksResult

Full 13-Greek container with arithmetic support for portfolio aggregation.

```python
@dataclass(frozen=True)
class GreeksResult:
    # First order
    delta: float = 0.0   # dV/dS
    theta: float = 0.0   # dV/dt
    vega: float = 0.0    # dV/dsigma
    rho: float = 0.0     # dV/dr
    # Second order
    gamma: float = 0.0   # d2V/dS2
    vanna: float = 0.0   # d2V/dS dsigma
    volga: float = 0.0   # d2V/dsigma2
    charm: float = 0.0   # d2V/dS dt
    veta: float = 0.0    # d2V/dsigma dt
    # Third order
    speed: float = 0.0   # d3V/dS3
    zomma: float = 0.0   # d3V/dS2 dsigma
    color: float = 0.0   # d3V/dS2 dt
    ultima: float = 0.0  # d3V/dsigma3
```

**Operators** (all return new `GreeksResult` instances):

| Operator | Usage | Description |
|----------|-------|-------------|
| `+` | `g1 + g2` | Sum Greeks for portfolio aggregation |
| `-` | `g1 - g2` | Subtract Greeks for hedging |
| `*` | `g * 10` or `10 * g` | Scale by scalar (position size) |
| `/` | `g / 2` | Divide by scalar (normalization) |
| `-` (unary) | `-g` | Negate (short positions) |

**Properties and methods**:

| Member | Returns | Description |
|--------|---------|-------------|
| `.vomma` | `float` | Alias for `.volga` |
| `.delta_decay` | `float` | Alias for `.charm` |
| `.first_order()` | `dict` | `{delta, theta, vega, rho}` |
| `.second_order()` | `dict` | `{gamma, vanna, volga, charm, veta}` |
| `.third_order()` | `dict` | `{speed, zomma, color, ultima}` |
| `.to_dict()` | `dict` | All 13 Greeks as dictionary |

```python
from backend import GreeksResult

# Portfolio aggregation
call_greeks = GreeksResult(delta=0.55, gamma=0.02, vega=0.20, theta=-0.05)
put_greeks = GreeksResult(delta=-0.45, gamma=0.02, vega=0.20, theta=-0.05)
portfolio = call_greeks + put_greeks  # delta = 0.10

# Scale by position size
scaled = call_greeks * 100  # 100 contracts

# Short position
short = -call_greeks  # delta = -0.55
```

### ExerciseStyle Enum

```python
class ExerciseStyle(Enum):
    EUROPEAN = auto()
    AMERICAN = auto()
    BERMUDAN = auto()
```

### PricingCapability Enum

```python
class PricingCapability(Enum):
    ANALYTICAL = auto()
    FFT = auto()
    MONTE_CARLO = auto()
```

### EngineRegistry

Factory that auto-selects the optimal pricing engine. Priority: `ANALYTICAL > FFT > MONTE_CARLO`.

```python
class EngineRegistry:
    PRIORITY = [PricingCapability.ANALYTICAL, PricingCapability.FFT, PricingCapability.MONTE_CARLO]
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `register()` | `(model_name: str, capability: PricingCapability, engine_provider: EngineProvider) -> None` | Register engine for model/capability |
| `unregister()` | `(model_name: str, capability: PricingCapability) -> bool` | Remove registration |
| `clear()` | `() -> None` | Remove all registrations |
| `get_engine()` | `(instrument, model, preferred=None) -> PricingEngine` | Get optimal engine |
| `price()` | `(instrument, model, market, preferred=None) -> PricingResult` | Get engine + price in one call |
| `list_engines()` | `() -> List[Tuple[str, str]]` | List all registered (model, capability) pairs |

**Module-level convenience function**:

```python
from backend.core import price, MarketEnvironment
from backend.instruments import EuropeanCall
from backend.models import GBMModel

result = price(EuropeanCall(strike=100, maturity=0.5), GBMModel(sigma=0.20),
               MarketEnvironment(spot=100, rate=0.05))
```

### Model ABC

Abstract base class for all pricing models.

```python
class Model(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def supported_engines(self) -> List[PricingCapability]: ...

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]: ...

    def characteristic_function(self, u: complex, s0: float, t: float, r: float, q: float = 0.0) -> complex: ...
    def characteristic_function_vectorized(self, u_arr: np.ndarray, s0: float, t: float, r: float, q: float = 0.0) -> np.ndarray: ...
    def drift(self, s: float, v: float, t: float, r: float, q: float) -> float: ...
    def diffusion(self, s: float, v: float, t: float) -> float: ...
    def create_simulator(self, **kwargs) -> BaseSimulator: ...
```

---

## Models Module

**Location**: `backend/models/`

Pricing models define the stochastic dynamics of the underlying asset. All models are frozen dataclasses implementing the `Model` ABC.

### Available Models

| Model | Description | Parameters |
|-------|-------------|------------|
| `GBMModel` | Geometric Brownian Motion | `sigma` |
| `HestonModel` | Stochastic volatility | `v0`, `kappa`, `theta`, `xi`, `rho` |
| `BatesModel` | Heston + Jumps | Heston params + `lambda_j`, `mu_j`, `sigma_j` |
| `MertonModel` | Jump Diffusion | `sigma`, `lambda_j`, `mu_j`, `sigma_j` |
| `GARCHModel` | GARCH(1,1) | `sigma0`, `omega`, `alpha`, `beta` |
| `NGARCHModel` | Nonlinear GARCH | GARCH + `theta` (asymmetry) |
| `GJRGARCHModel` | GJR-GARCH | GARCH + `gamma` (leverage) |

### HestonModel

Heston (1993) stochastic volatility: `dS = (r-q)S dt + sqrt(V)S dW_S`, `dV = kappa(theta-V)dt + xi sqrt(V) dW_V`.

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `feller_satisfied` | `bool` | `True` if `2*kappa*theta > xi^2` |
| `feller_ratio` | `float` | `2*kappa*theta / xi^2` (>1 means satisfied) |
| `long_run_volatility` | `float` | `sqrt(theta)` |
| `initial_volatility` | `float` | `sqrt(v0)` |

**Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `variance_drift(v)` | `(float) -> float` | `kappa * (theta - v)` |
| `variance_diffusion(v)` | `(float) -> float` | `xi * sqrt(max(v, 0))` |
| `mean_variance(t)` | `(float) -> float` | `E[V_t] = theta + (v0-theta)*exp(-kappa*t)` |
| `expected_variance(t)` | `(float) -> float` | Alias for `mean_variance` |
| `total_variance(t=1.0)` | `(float) -> float` | Integrated expected variance over `[0, t]` |
| `total_volatility(t=1.0)` | `(float) -> float` | `sqrt(total_variance(t) / t)` |
| `create_simulator(**kwargs)` | `(**kwargs) -> HestonSimulator` | Create MC simulator |

```python
from backend import HestonModel

model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

print(f"Feller satisfied: {model.feller_satisfied}")   # True
print(f"Feller ratio: {model.feller_ratio:.2f}")        # 1.78
print(f"Initial vol: {model.initial_volatility:.1%}")   # 20.0%
print(f"Long-run vol: {model.long_run_volatility:.1%}") # 20.0%
print(f"E[V_1]: {model.mean_variance(1.0):.4f}")        # 0.04
```

### MertonModel

Merton (1976) jump-diffusion: `dS = (r-q-lambda_j*k)S dt + sigma S dW + (J-1)S dN`.

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `expected_jump_size` | `float` | `E[J-1] = exp(mu_j + 0.5*sigma_j^2) - 1` |
| `expected_jump_return` | `float` | `expected_jump_size * 100` (percentage) |
| `variance` | `float` | `sigma^2` (annualized diffusion variance) |

**Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `expected_jumps_per_year()` | `() -> float` | Returns `lambda_j` |
| `jump_contribution_to_variance()` | `() -> float` | `lambda_j * (mu_j^2 + sigma_j^2)` |
| `total_variance(t=1.0)` | `(float) -> float` | Diffusion + jump variance |
| `total_volatility(t=1.0)` | `(float) -> float` | `sqrt(total_variance(t) / t)` |
| `to_gbm()` | `() -> GBMModel` | Drop jumps, keep diffusion `sigma` |
| `create_simulator(**kwargs)` | `(**kwargs) -> MertonSimulator` | Create MC simulator |

```python
from backend import MertonModel

model = MertonModel(sigma=0.20, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)

print(f"Expected jump size: {model.expected_jump_size:.2%}")
print(f"Jump variance: {model.jump_contribution_to_variance():.4f}")
print(f"Total vol: {model.total_volatility():.1%}")

gbm = model.to_gbm()  # GBMModel(sigma=0.2)
```

### BatesModel

Bates (1996): Heston stochastic volatility + Merton-style jumps.

Has all Heston properties (`feller_satisfied`, `feller_ratio`, `long_run_volatility`, `initial_volatility`, `mean_variance`, `total_variance`, `total_volatility`) plus all Merton jump properties (`expected_jump_size`, `expected_jump_return`, `expected_jumps_per_year`, `jump_contribution_to_variance`).

**Additional methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `to_heston()` | `() -> HestonModel` | Drop jumps, keep SV params |
| `variance_drift(v)` | `(float) -> float` | `kappa * (theta - v)` |
| `variance_diffusion(v)` | `(float) -> float` | `xi * sqrt(max(v, 0))` |
| `create_simulator(**kwargs)` | `(**kwargs) -> BatesSimulator` | Create MC simulator |

```python
from backend import BatesModel

model = BatesModel(
    v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
    lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
)
heston = model.to_heston()  # Drop jumps
```

### GARCH Family

All GARCH models inherit from `BaseGARCHModel` and share common properties:

**Shared properties** (`BaseGARCHModel`):

| Property | Type | Description |
|----------|------|-------------|
| `persistence` | `float` | Model-specific (abstract), must be < 1 |
| `long_run_variance` | `float` | `omega / (1 - persistence)` |
| `long_run_volatility` | `float` | `sqrt(long_run_variance)` |
| `half_life` | `float` | `ln(2) / (-ln(persistence))` time steps |

**Shared methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `create_simulator(**kwargs)` | `() -> Simulator` | Model-specific simulator |
| `create_pricer(n_paths=100000, n_steps=252)` | `(int, int) -> GARCHMCPricer` | MC pricer with LRNVR |

**Model-specific persistence formulas**:

| Model | Persistence | Stationarity Condition |
|-------|-------------|----------------------|
| `GARCHModel` | `alpha + beta` | `alpha + beta < 1` |
| `NGARCHModel` | `alpha*(1 + theta^2) + beta` | `alpha*(1+theta^2) + beta < 1` |
| `GJRGARCHModel` | `alpha + 0.5*gamma + beta` | `alpha + 0.5*gamma + beta < 1` |

```python
from backend import GARCHModel, NGARCHModel, GJRGARCHModel

garch = GARCHModel(sigma0=0.20, omega=0.002, alpha=0.05, beta=0.90)
print(f"Persistence: {garch.persistence:.3f}")          # 0.950
print(f"Long-run vol: {garch.long_run_volatility:.1%}") # 20.0%
print(f"Half-life: {garch.half_life:.1f} steps")         # 13.5

# Create pricer with LRNVR measure change
pricer = garch.create_pricer(n_paths=100_000)
result = pricer.price(s0=100, k=100, t=0.25, r=0.05)
```

### Characteristic Functions

Numba-optimized characteristic functions for FFT pricing.

| Function | Signature |
|----------|-----------|
| `heston_characteristic_function(u, s0, v0, t, r, kappa, theta, xi, rho)` | Scalar CF |
| `heston_cf_vectorized(u_arr, s0, v0, t, r, kappa, theta, xi, rho)` | Vectorized CF |
| `merton_characteristic_function(u, s0, t, r, sigma, lambda_j, mu_j, sigma_j)` | Scalar CF |
| `merton_cf_vectorized(u_arr, s0, t, r, sigma, lambda_j, mu_j, sigma_j)` | Vectorized CF |
| `bates_characteristic_function(u, s0, v0, t, r, kappa, theta, xi, rho, lambda_j, mu_j, sigma_j)` | Scalar CF |
| `bates_cf_vectorized(u_arr, s0, v0, t, r, kappa, theta, xi, rho, lambda_j, mu_j, sigma_j)` | Vectorized CF |

### ModelRegistry

Singleton pattern for model lookup by name.

```python
from backend.models import registry, ModelRegistry

models = registry.list_models()  # All registered model names
model = registry.create("heston", v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
```

### Vol Bump Utilities

**Location**: `backend/models/vol_bump.py`

Utilities for creating volatility-bumped model copies (used for numerical vega).

```python
from backend.models.vol_bump import create_vol_bumped_model, create_vol_bumped_pair

model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

# Single bump: for Heston, bumps in vol space: v0_new = (sqrt(v0) + h)^2
model_up = create_vol_bumped_model(model, vol_bump=0.01)

# Pair for central difference
model_up, model_down = create_vol_bumped_pair(model, vol_bump=0.01)
```

**Bump semantics by model type**:

| Model | Bump Target | Formula |
|-------|-------------|---------|
| GBM | `sigma` | `sigma + h` |
| Merton | `sigma` | `sigma + h` (preserves jump params) |
| Heston/Bates | `v0` | `v0_new = (sqrt(v0) + h)^2` (bump in vol space) |
| GARCH family | `sigma0` | `sigma0 + h` |

---

## Engines Module

**Location**: `backend/engines/`

Engines implement the pricing algorithms.

### High-Level Engines

| Engine | Method | Best For |
|--------|--------|----------|
| `BSAnalyticEngine` | Black-Scholes formula | GBM, fast pricing |
| `FFTEngine` | Carr-Madan FFT | Stochastic vol models |
| `MonteCarloEngine` | Monte Carlo simulation | All models, path-dependent |
| `ExoticAnalyticEngine` | Closed-form exotic formulas | Barrier, Asian geometric, Digital, Lookback under GBM |

### Low-Level Engines

| Engine | Description |
|--------|-------------|
| `CarrMadanFFTEngine` | Generic FFT with characteristic function |
| `GenericMCEngine` | Generic MC with terminal simulator |
| `GARCHMCPricer` | GARCH-specific with LRNVR measure change |

### BSAnalyticEngine

```python
class BSAnalyticEngine(PricingEngine):
    def price(self, instrument, model, market) -> PricingResult: ...
    def implied_volatility(self, price, instrument, market) -> float: ...
```

The `implied_volatility()` method inverts the BS formula using Newton-Raphson.

### FFTConfig

Configuration for FFT pricing:

```python
@dataclass
class FFTConfig:
    alpha: float = 1.5     # Dampening factor (must be > 0)
    n_fft: int = 4096      # FFT grid size (must be power of 2)
    eta: float = 0.25      # Grid spacing in frequency domain (must be > 0)
```

**Validation constraints**: `alpha > 0`, `n_fft` must be a power of 2, `eta > 0`.

**Log-strike spacing**: `lambda_spacing = 2 * pi / (n_fft * eta)`.

### CarrMadanFFTEngine

```python
class CarrMadanFFTEngine:
    def __init__(self, config: FFTConfig = FFTConfig()): ...
    def price_call(self, characteristic_fn, s0, k, t, r, q=0.0) -> float: ...
    def price_put(self, characteristic_fn, s0, k, t, r, q=0.0) -> float: ...
    def price_strikes(self, characteristic_fn, s0, strikes, t, r, is_call=True, q=0.0) -> np.ndarray: ...
    def price_surface(self, characteristic_fn_factory, s0, strikes, maturities, r, is_call=True, q=0.0) -> np.ndarray: ...
```

### MCConfig

```python
@dataclass
class MCConfig:
    n_paths: int = 100_000   # Number of Monte Carlo paths
    n_steps: int = 252       # Number of time steps
    seed: Optional[int] = None  # Random seed for reproducibility
    antithetic: bool = True  # Antithetic variance reduction
```

### MCResult

```python
@dataclass
class MCResult:
    price: float           # Estimated option price
    std_error: float       # Standard error of estimate
    n_paths: int           # Number of paths used
```

### GARCHMCPricer

Monte Carlo pricer for GARCH family with LRNVR (Locally Risk-Neutral Valuation Relationship) measure change.

```python
class GARCHMCPricer:
    def __init__(
        self,
        garch_type: GARCHType,  # GARCH, NGARCH, or GJR_GARCH
        sigma0: float, omega: float, alpha: float, beta: float,
        theta: float = 0.0, gamma: float = 0.0,
        n_paths: int = 100_000, n_steps: int = 252
    ): ...

    def price(self, s0, k, t, r, option_type='call', n_paths=None, n_steps=None, seed=None) -> GARCHPricingResult: ...
    def price_surface(self, s0, strikes, maturities, r, option_type='call') -> np.ndarray: ...

    @property
    def persistence(self) -> float: ...
    def long_run_variance(self) -> float: ...
    def long_run_volatility(self) -> float: ...
```

**Convenience factories**:

```python
from backend.engines.monte_carlo.garch_pricer import (
    create_garch_pricer, create_ngarch_pricer, create_gjr_garch_pricer
)

pricer = create_garch_pricer(sigma0=0.20, omega=0.002, alpha=0.05, beta=0.90)
result = pricer.price(s0=100, k=100, t=0.25, r=0.05)
```

### Vectorized Black-Scholes

**Location**: `backend/engines/vectorized_bs.py`

Numba-compiled vectorized Greeks. Uses `option_type: int` convention (1=call, 0=put).

```python
def calculate_first_order_greeks(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type: int) -> tuple:
    """Returns (price, delta, gamma, vega, theta, rho)."""

def calculate_all_greeks(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type: int) -> np.ndarray:
    """Returns array of 14 Greeks."""

def calculate_greeks_vectorized(spot_range: np.ndarray, strike, time_to_expiry, risk_free_rate, volatility, option_type: int) -> np.ndarray:
    """Returns (n_spots, 14) array. Numba-parallel over spots."""
```

**Greek index constants** (for indexing into the 14-element arrays):

```python
GREEK_PRICE  = 0   # Option price
GREEK_DELTA  = 1   # dV/dS
GREEK_GAMMA  = 2   # d2V/dS2
GREEK_VEGA   = 3   # dV/dsigma (per 1% vol)
GREEK_THETA  = 4   # dV/dt (per day)
GREEK_RHO    = 5   # dV/dr (per 1% rate)
GREEK_VANNA  = 6   # d2V/dS dsigma
GREEK_VOLGA  = 7   # d2V/dsigma2
GREEK_CHARM  = 8   # d2V/dS dt
GREEK_VETA   = 9   # d2V/dsigma dt
GREEK_SPEED  = 10  # d3V/dS3
GREEK_ZOMMA  = 11  # d3V/dS2 dsigma
GREEK_COLOR  = 12  # d3V/dS2 dt
GREEK_ULTIMA = 13  # d3V/dsigma3
```

### Example: Black-Scholes Engine

```python
from backend import VanillaOption, GBMModel, BSAnalyticEngine, MarketEnvironment

option = VanillaOption(strike=100, maturity=0.5, is_call=True)
model = GBMModel(sigma=0.2)
market = MarketEnvironment(spot=100, rate=0.05)

engine = BSAnalyticEngine()
result = engine.price(option, model, market)

print(f"Call Price: ${result.price:.4f}")
print(f"Greeks: delta={result.greeks.delta:.4f}, gamma={result.greeks.gamma:.6f}")
```

### Example: FFT Engine for Heston

```python
from backend import VanillaOption, HestonModel, FFTEngine, MarketEnvironment, FFTConfig

option = VanillaOption(strike=100, maturity=0.5, is_call=True)
model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
market = MarketEnvironment(spot=100, rate=0.05)

# Configure FFT
config = FFTConfig(alpha=1.5, n_fft=4096, eta=0.25)
engine = FFTEngine(config=config)

result = engine.price(option, model, market)
print(f"Heston Call Price: ${result.price:.4f}")
```

### Example: Monte Carlo Engine

```python
from backend import VanillaOption, BatesModel, MonteCarloEngine, MarketEnvironment, MCConfig

option = VanillaOption(strike=100, maturity=0.5, is_call=True)
model = BatesModel(
    v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
    lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
)
market = MarketEnvironment(spot=100, rate=0.05)

# Configure Monte Carlo
config = MCConfig(n_paths=100_000, n_steps=252, seed=42)
engine = MonteCarloEngine(config=config)

result = engine.price(option, model, market)
print(f"Bates Call Price: ${result.price:.4f} +/- ${result.std_error:.4f}")
```

### ExoticAnalyticEngine

Closed-form pricing engine for exotic options under GBM. All kernels are Numba-compiled.

```python
@dataclass(frozen=True)
class ExoticAnalyticEngine(PricingEngine):
    def can_price(self, instrument, model) -> bool: ...
    def price(self, instrument, model, market) -> PricingResult: ...
    def greeks(self, instrument, model, market) -> GreeksResult: ...
```

**Supported instrument types** (requires `GBMModel` + `ExerciseStyle.EUROPEAN`):

| Instrument | Formula | Reference |
|------------|---------|-----------|
| `BarrierOption` | All 8 types (up/down × in/out × call/put) | Reiner-Rubinstein 1991 |
| `AsianOption` (geometric only) | Kemna-Vorst 1990 reduced volatility | Kemna-Vorst 1990 |
| `DigitalOption` | Cash-or-nothing closed form | Black-Scholes |
| `LookbackOption` | Floating & fixed strike | Conze-Viswanathan 1991 |

> **Note**: `ExoticAnalyticEngine` is not registered in `EngineRegistry`. Instantiate it directly.

**`greeks()` method** returns first-order Greeks (delta, gamma, vega, theta, rho) computed via central finite differences on the analytic prices (aligned with `GreeksBumpConfig` defaults).

**Key parity relationships** (useful for validation):
- Knock-in + knock-out = vanilla: `barrier_in + barrier_out == vanilla_price`
- Digital parity: `digital_call + digital_put = exp(-r*T)`

### Example: Exotic Engine

```python
from backend.engines.exotic_engine import ExoticAnalyticEngine
from backend.instruments.options import (
    BarrierUpOutCall, AsianGeometricCall, DigitalOption, LookbackCall
)
from backend.models.gbm import GBMModel
from backend.core.market import MarketEnvironment

engine = ExoticAnalyticEngine()
model = GBMModel(sigma=0.25)
market = MarketEnvironment(spot=100, rate=0.05)

# Barrier option
barrier = BarrierUpOutCall(strike=100, barrier=120, maturity=0.5)
result = engine.price(barrier, model, market)
greeks = engine.greeks(barrier, model, market)
print(f"Up-Out Call: ${result.price:.4f}  delta={greeks.delta:.4f}")

# Asian geometric
asian = AsianGeometricCall(strike=100, maturity=0.5)
result = engine.price(asian, model, market)
print(f"Asian Geometric Call: ${result.price:.4f}")

# Digital
digital = DigitalOption(strike=100, maturity=0.5, is_call=True, payout=1.0)
result = engine.price(digital, model, market)
print(f"Digital Call: ${result.price:.4f}")

# Floating lookback
lookback = LookbackCall(maturity=0.5)
result = engine.price(lookback, model, market)
print(f"Floating Lookback Call: ${result.price:.4f}")
```

---

## Simulation Module

**Location**: `backend/simulation/`

The simulation module provides Monte Carlo simulators for path generation.

### Available Simulators

| Simulator | Model | Output |
|-----------|-------|--------|
| `GBMSimulator` | GBM | Price paths |
| `HestonSimulator` | Heston | Price + variance paths |
| `MertonSimulator` | Merton Jump | Price paths |
| `BatesSimulator` | Bates | Price + variance paths |
| `GARCHSimulator` | GARCH(1,1) | Price + volatility paths |
| `NGARCHSimulator` | NGARCH | Price + volatility paths |
| `GJRGARCHSimulator` | GJR-GARCH | Price + volatility paths |

### SimulationResult

Full API for simulation output analysis.

**Core fields**:

```python
class SimulationResult:
    price_paths: np.ndarray      # Shape (n_paths, n_steps+1)
    time_grid: np.ndarray        # Shape (n_steps+1,)
    model_name: str
    computation_time: float
    n_paths: int
    n_steps: int
    volatility_paths: Optional[np.ndarray] = None  # For SV models
```

**Computed properties**:

| Property | Type | Description |
|----------|------|-------------|
| `terminal_prices` | `np.ndarray` | `price_paths[:, -1]` |
| `mean_path` | `np.ndarray` | Mean across paths at each step |
| `std_path` | `np.ndarray` | Std dev across paths at each step |
| `terminal_mean` | `float` | Mean of terminal prices |
| `terminal_std` | `float` | Std of terminal prices |
| `has_volatility` | `bool` | Whether volatility paths exist |
| `terminal_volatility` | `Optional[np.ndarray]` | Terminal vol values |
| `mean_volatility_path` | `Optional[np.ndarray]` | Mean vol path |

**Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `percentile_paths(pcts)` | `(List[float]) -> np.ndarray` | Shape `(len(pcts), n_steps+1)` |
| `log_returns()` | `() -> np.ndarray` | Period-by-period log returns |
| `realized_volatility()` | `() -> np.ndarray` | Per-path realized vol |

### StochasticVolatilityMixin

Mixin providing volatility analysis for SV simulators.

| Method | Returns | Description |
|--------|---------|-------------|
| `long_run_variance()` | `float` | Theoretical long-run variance |
| `long_run_volatility()` | `float` | `sqrt(long_run_variance())` |
| `feller_condition_satisfied()` | `bool` | `2*kappa*theta > xi^2` |

### GBMSimulator

```python
class GBMSimulator(BaseSimulator):
    def __init__(self, sigma: float, antithetic: bool = True): ...
```

The `antithetic` parameter enables antithetic variance reduction (default: `True`).

### HestonSimulator

```python
class HestonSimulator(BaseSimulator, StochasticVolatilityMixin):
    def __init__(
        self,
        v0: float, kappa: float, theta: float, xi: float, rho: float,
        scheme: DiscretizationScheme = DiscretizationScheme.FULL_TRUNCATION
    ): ...
```

**Discretization schemes comparison**:

| Scheme | Variance Handling | Accuracy | Speed |
|--------|------------------|----------|-------|
| `EULER` | No protection (can go negative) | Low | Fast |
| `FULL_TRUNCATION` | `V = max(V, 0)` in drift+diffusion | Good | Fast |
| `REFLECTION` | `V = abs(V)` if negative | Good | Fast |
| `QE` | Quadratic-Exponential (Andersen 2008) | Best | Moderate |

### DiscretizationScheme Enum

```python
class DiscretizationScheme(Enum):
    EULER = auto()
    FULL_TRUNCATION = auto()
    REFLECTION = auto()
    QE = auto()

    @classmethod
    def default(cls) -> DiscretizationScheme:
        return cls.FULL_TRUNCATION
```

### ModelType Enum

```python
class ModelType(Enum):
    GBM = "Geometric Brownian Motion"
    HESTON = "Heston Stochastic Volatility"
    BATES = "Bates (Heston + Jumps)"
    MERTON = "Merton Jump Diffusion"
    GARCH = "GARCH(1,1)"
    NGARCH = "NGARCH (Nonlinear Asymmetric)"
    GJR_GARCH = "GJR-GARCH"
```

**Classification methods**:

| Method | Returns |
|--------|---------|
| `ModelType.continuous_time_models()` | `[GBM, HESTON, BATES, MERTON]` |
| `ModelType.discrete_time_models()` | `[GARCH, NGARCH, GJR_GARCH]` |
| `ModelType.stochastic_vol_models()` | `[HESTON, BATES, GARCH, NGARCH, GJR_GARCH]` |
| `ModelType.jump_models()` | `[MERTON, BATES]` |

### Measure Enum

```python
class Measure(Enum):
    P_MEASURE = "Physical (Real-World)"
    Q_MEASURE = "Risk-Neutral"
```

### Factory Functions

```python
from backend import create_simulator, ModelType

# By string (case-insensitive, supports aliases)
sim = create_simulator("heston", v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

# By enum
sim = create_simulator(ModelType.GARCH, sigma0=0.2, omega=1e-6, alpha=0.1, beta=0.85)
```

**Convenience factory functions**:

| Function | Signature |
|----------|-----------|
| `create_gbm(sigma, antithetic=True)` | `-> GBMSimulator` |
| `create_heston(v0, kappa, theta, xi, rho, scheme=FULL_TRUNCATION)` | `-> HestonSimulator` |
| `create_merton(sigma, lambda_j, mu_j, sigma_j)` | `-> MertonSimulator` |
| `create_bates(v0, kappa, theta, xi, rho, lambda_j, mu_j, sigma_j)` | `-> BatesSimulator` |
| `create_garch(sigma0, omega, alpha, beta)` | `-> GARCHSimulator` |
| `create_ngarch(sigma0, omega, alpha, beta, theta)` | `-> NGARCHSimulator` |
| `create_gjr_garch(sigma0, omega, alpha, beta, gamma)` | `-> GJRGARCHSimulator` |

**Name alias map** (all resolve via `create_simulator()`):

| Alias | Maps To |
|-------|---------|
| `"gbm"`, `"geometric_brownian_motion"` | `ModelType.GBM` |
| `"heston"` | `ModelType.HESTON` |
| `"merton"`, `"merton_jump"`, `"jump_diffusion"` | `ModelType.MERTON` |
| `"bates"`, `"heston_jump"` | `ModelType.BATES` |
| `"garch"`, `"garch11"` | `ModelType.GARCH` |
| `"ngarch"`, `"nagarch"` | `ModelType.NGARCH` |
| `"gjr"`, `"gjr_garch"`, `"tgarch"` | `ModelType.GJR_GARCH` |

**Factory helpers**:

```python
from backend.simulation.factory import list_models, get_model_info

# List all available models
models = list_models()  # {"GBM": "Geometric Brownian Motion", ...}

# Get detailed info
info = get_model_info("heston")
# {'name': 'HESTON', 'description': '...', 'class': 'HestonSimulator',
#  'is_stochastic_vol': True, 'has_jumps': False, 'is_continuous_time': True,
#  'parameters': {...}}
```

---

## Instruments Module

**Location**: `backend/instruments/`

The instruments module defines tradeable contracts.

### Options

| Class | Description |
|-------|-------------|
| `VanillaOption` | Generic vanilla option |
| `DigitalOption` | Digital (binary) option |
| `AsianOption` | Asian (average price) option |
| `BarrierOption` | Barrier option |
| `LookbackOption` | Lookback option |

**Convenience factory functions** (return configured `VanillaOption` instances):

| Function | Exercise | Type |
|----------|----------|------|
| `EuropeanCall(strike, maturity)` | European | Call |
| `EuropeanPut(strike, maturity)` | European | Put |
| `AmericanCall(strike, maturity)` | American | Call |
| `AmericanPut(strike, maturity)` | American | Put |
| `BermudanCall(strike, maturity)` | Bermudan | Call |
| `BermudanPut(strike, maturity)` | Bermudan | Put |

**Exotic option factories**:

| Function | Description |
|----------|-------------|
| `AsianCall(strike, maturity)` | Asian arithmetic call |
| `AsianPut(strike, maturity)` | Asian arithmetic put |
| `AsianGeometricCall(strike, maturity)` | Asian geometric call (analytically priceable by `ExoticAnalyticEngine`) |
| `AsianGeometricPut(strike, maturity)` | Asian geometric put (analytically priceable) |
| `BarrierUpOutCall(strike, barrier, maturity, rebate=0)` | Up-and-out call |
| `BarrierUpInCall(strike, barrier, maturity)` | Up-and-in call |
| `BarrierDownOutCall(strike, barrier, maturity, rebate=0)` | Down-and-out call |
| `BarrierDownInCall(strike, barrier, maturity)` | Down-and-in call |
| `BarrierUpOutPut(strike, barrier, maturity, rebate=0)` | Up-and-out put |
| `BarrierUpInPut(strike, barrier, maturity)` | Up-and-in put |
| `BarrierDownOutPut(strike, barrier, maturity, rebate=0)` | Down-and-out put |
| `BarrierDownInPut(strike, barrier, maturity)` | Down-and-in put |
| `LookbackCall(maturity)` | Floating lookback call (payoff: `S_T - S_min`) |
| `LookbackPut(maturity)` | Floating lookback put (payoff: `S_max - S_T`) |
| `LookbackFixedCall(strike, maturity)` | Fixed strike lookback call (payoff: `max(S_max - K, 0)`) |
| `LookbackFixedPut(strike, maturity)` | Fixed strike lookback put (payoff: `max(K - S_min, 0)`) |

### Payoffs

| Class | Formula |
|-------|---------|
| `VanillaCallPayoff(strike)` | `max(S - K, 0)` |
| `VanillaPutPayoff(strike)` | `max(K - S, 0)` |
| `DigitalCallPayoff(strike)` | `1 if S > K else 0` |
| `DigitalPutPayoff(strike)` | `1 if S < K else 0` |
| `CompositePayoff(payoffs, weights)` | Weighted sum |
| `AsianCallPayoff(strike)` | `max(S_avg - K, 0)` |
| `AsianPutPayoff(strike)` | `max(K - S_avg, 0)` |
| `BarrierUpOutCallPayoff(strike, barrier)` | Call that dies if S > barrier |
| `BarrierDownOutPutPayoff(strike, barrier)` | Put that dies if S < barrier |
| `LookbackFloatingCallPayoff()` | `S_T - S_min` |
| `LookbackFloatingPutPayoff()` | `S_max - S_T` |

### Strategies

| Class | Description | Legs |
|-------|-------------|------|
| `OptionStrategy` | Generic multi-leg strategy | Any |
| `Straddle(strike, maturity)` | ATM call + ATM put | 2 |
| `Strangle(put_strike, call_strike, maturity)` | OTM call + OTM put | 2 |
| `Butterfly(k1, k2, k3, maturity)` | Bull + bear spread | 3 |
| `IronCondor(k1, k2, k3, k4, maturity)` | Short strangle + long wings | 4 |
| `IronButterfly(k1, k2, k3, maturity)` | Short straddle + long wings | 4 |
| `CallSpread(k_long, k_short, maturity)` | Bull call spread | 2 |
| `PutSpread(k_long, k_short, maturity)` | Bear put spread | 2 |

**StrategyLeg dataclass**:

```python
@dataclass
class StrategyLeg:
    strike: float
    is_call: bool
    quantity: int  # positive=long, negative=short
```

`CallSpread` and `PutSpread` expose a `max_profit` property.

### Exercise Schedules

| Class | Description |
|-------|-------------|
| `EuropeanExercise(maturity)` | Exercise only at maturity |
| `AmericanExercise(maturity, start_time=0.0)` | Exercise any time |
| `BermudanExercise(exercise_dates)` | Exercise at discrete dates |

**Common interface** (all exercise classes):

| Method | Description |
|--------|-------------|
| `exercise_type` | Returns `ExerciseStyle` enum |
| `maturity` | Final exercise date |
| `can_exercise(t, tol=1e-8)` | Whether exercise is allowed at time `t` |
| `get_exercise_times(...)` | Array of exercise times |

**BermudanExercise.from_schedule() factory**:

```python
bermudan = BermudanExercise.from_schedule(
    start=0.25, end=1.0, frequency='monthly'
)
# frequency: 'daily', 'weekly', 'monthly', 'quarterly', 'semiannual', 'annual'
```

---

## Greeks Module

**Location**: `backend/greeks/`

Comprehensive Greeks calculation with analytic and numerical methods.

### Scaling Conventions

All Greeks are scaled to market-standard values:

| Greek | Scale Factor | Meaning |
|-------|-------------|---------|
| Delta | 1 (raw) | Sensitivity to $1 spot move |
| Gamma | 1 (raw) | Change in delta per $1 spot move |
| **Vega** | **/ 100** | Per 1% vol change |
| **Theta** | **/ 365** | Per calendar day |
| **Rho** | **/ 100** | Per 1% rate change |
| **Vanna** | **/ 100** | Per 1% vol change |
| **Volga** | **/ 10,000** | Per 1%^2 vol change |
| **Charm** | **/ 365** | Per calendar day |
| **Veta** | **/ 36,500** | Per day per 1% vol |
| Speed | 1 (raw) | d3V/dS3 |
| **Zomma** | **/ 100** | Per 1% vol |
| **Color** | **/ 365** | Per calendar day |
| **Ultima** | **/ 1,000,000** | Per 1%^3 vol |

**Constants** (from `backend/greeks/analytic.py`):

```python
VEGA_SCALE = 100.0
RHO_SCALE = 100.0
THETA_SCALE = 365.0  # DAYS_PER_YEAR
VANNA_SCALE = 100.0
VOLGA_SCALE = 10000.0
CHARM_SCALE = 365.0
VETA_SCALE = 36500.0  # 365 * 100
ZOMMA_SCALE = 100.0
COLOR_SCALE = 365.0
ULTIMA_SCALE = 1000000.0
```

### Result NamedTuples

```python
class FirstOrderGreeks(NamedTuple):
    delta: float; gamma: float; vega: float; theta: float; rho: float

class SecondOrderGreeks(NamedTuple):
    vanna: float; volga: float; charm: float; veta: float

class ThirdOrderGreeks(NamedTuple):
    speed: float; zomma: float; color: float; ultima: float

class AllGreeksResult(NamedTuple):
    price: float
    delta: float; gamma: float; vega: float; theta: float; rho: float
    vanna: float; volga: float; charm: float; veta: float
    speed: float; zomma: float; color: float; ultima: float
```

### Analytic Greeks Functions

Individual BS Greeks (from `backend/greeks/analytic.py`, delegate to `backend/utils/math`):

| Function | Signature | Returns |
|----------|-----------|---------|
| `bs_delta(s, k, t, r, sigma, is_call)` | `-> float` | Delta |
| `bs_gamma(s, k, t, r, sigma)` | `-> float` | Gamma (same for calls/puts) |
| `bs_vega(s, k, t, r, sigma)` | `-> float` | Vega (scaled) |
| `bs_theta(s, k, t, r, sigma, is_call)` | `-> float` | Theta (scaled) |
| `bs_rho(s, k, t, r, sigma, is_call)` | `-> float` | Rho (scaled) |
| `bs_first_order_greeks(s, k, t, r, sigma, is_call)` | `-> FirstOrderGreeks` | All first-order |
| `bs_second_order_greeks(s, k, t, r, sigma)` | `-> SecondOrderGreeks` | All second-order |
| `bs_third_order_greeks(s, k, t, r, sigma)` | `-> ThirdOrderGreeks` | All third-order |
| `bs_all_greeks(s, k, t, r, sigma, is_call)` | `-> AllGreeksResult` | All 14 values |

### GreeksCalculator

```python
class GreeksCalculator:
    def __init__(
        self,
        prefer_analytic: bool = True,
        spot_bump: float = 0.01,       # 1% relative
        vol_bump: float = 0.01,        # 1% absolute
        time_bump_days: float = 1.0,   # 1 calendar day
        rate_bump: float = 0.0001      # 1 basis point
    ): ...

    def calculate(
        self,
        engine: PricingEngine,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
        include_higher_order: bool = True
    ) -> Union[GreeksResult, AllGreeksResult]: ...

    def calculate_surface(
        self,
        engine: PricingEngine,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
        spot_range: np.ndarray,
        greek: str = 'delta'
    ) -> np.ndarray: ...
```

**Dispatch priority** in `calculate()`:
1. `ExoticAnalyticEngine.greeks()` — when engine provides its own first-order Greeks
2. Analytic BS Greeks — when `BSAnalyticEngine` + `GBMModel` + `VanillaOption`
3. Numerical finite differences — all other combinations

**Convenience function**:

```python
from backend.greeks.calculator import calculate_greeks

greeks = calculate_greeks(engine, instrument, model, market, include_higher_order=False)
```

### Numerical Greeks

**GreeksBumpConfig**:

```python
@dataclass(frozen=True)
class GreeksBumpConfig:
    spot_bump: float = 0.01        # 1% relative
    vol_bump: float = 0.01         # 1% absolute
    time_bump_days: float = 1.0    # 1 calendar day
    rate_bump: float = 0.0001      # 1 basis point
```

**Finite difference functions** (all take `price_func: Callable, ...` and bump config):

| Function | Description |
|----------|-------------|
| `finite_difference_delta(price_func, spot, bump, **kwargs)` | Central diff delta |
| `finite_difference_gamma(price_func, spot, bump, **kwargs)` | Central diff gamma |
| `finite_difference_vega(price_func, vol, bump, **kwargs)` | Central diff vega |
| `finite_difference_theta(price_func, t, bump_days, **kwargs)` | Forward diff theta |
| `finite_difference_rho(price_func, r, bump, **kwargs)` | Central diff rho |
| `finite_difference_vanna(price_func, spot, vol, ...)` | Cross partial |
| `finite_difference_volga(price_func, vol, bump, ...)` | Vol-vol sensitivity |
| `finite_difference_charm(price_func, spot, t, ...)` | Spot-time cross |
| `finite_difference_speed(price_func, spot, bump, ...)` | Third order spot |
| `finite_difference_zomma(price_func, spot, vol, ...)` | Gamma-vol cross |
| `finite_difference_color(price_func, spot, t, ...)` | Gamma-time cross |
| `finite_difference_ultima(price_func, vol, bump, ...)` | Third order vol |

**ModelNumericalGreeks class**:

```python
class ModelNumericalGreeks:
    def __init__(self, config: GreeksBumpConfig = GreeksBumpConfig()): ...

    def calculate(
        self,
        engine: PricingEngine,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment
    ) -> NumericalGreeks: ...
```

`NumericalGreeks` is a `NamedTuple` with `(delta, gamma, vega, theta, rho)` fields.

### _instrument_utils

**Location**: `backend/greeks/_instrument_utils.py`

Utility for creating maturity-bumped copies of immutable instruments. Used internally by `GreeksCalculator` for theta, charm, color, and veta calculations.

```python
def create_decayed_instrument(
    instrument: Instrument,
    new_maturity: float
) -> Optional[Instrument]:
    """
    Create a copy of the instrument with a different maturity.

    Supports VanillaOption, BarrierOption, AsianOption, DigitalOption,
    and LookbackOption. Returns None for unrecognized types.
    """
```

---

## Portfolio Module

**Location**: `backend/portfolio/`

Portfolio management with P&L analysis and breakeven calculation.

### OptionsPortfolio

Main portfolio container.

```python
class OptionsPortfolio:
    def __init__(self, model=None): ...
```

**Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `add(position)` | `(PortfolioPosition) -> None` | Add position |
| `add_position(position)` | `(PortfolioPosition) -> None` | Alias for `add` |
| `remove_position(index)` | `(int) -> None` | Remove by index |
| `clear()` | `() -> None` | Remove all positions |
| `value(market)` | `(MarketEnvironment) -> float` | Current portfolio value |
| `pnl_at_expiry(spots)` | `(np.ndarray) -> np.ndarray` | P&L at expiry |
| `pnl_at_expiry_fast(spots)` | `(np.ndarray) -> np.ndarray` | Numba-optimized P&L |
| `payoff_curve(spots)` | `(np.ndarray) -> np.ndarray` | Raw payoff (no premium) |
| `find_breakevens(spot_range)` | `(tuple) -> BreakevenResult` | Find breakeven prices |
| `risk_metrics_from_simulation(sim_result)` | `(SimulationResult) -> RiskMetrics` | Risk from MC |
| `greeks(market)` | `(MarketEnvironment) -> GreeksResult` | Aggregate Greeks |
| `calculate_greeks_surface(market, ...)` | `(...) -> np.ndarray` | Greeks surface |
| `summary()` | `() -> dict` | Portfolio summary |
| `total_premium()` | `() -> float` | Net premium paid/received |

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `positions` | `List[PortfolioPosition]` | All option positions |
| `stock` | `Optional[StockPosition]` | Stock position if any |
| `n_positions` | `int` | Number of positions |

### PortfolioPosition

```python
class PortfolioPosition:
    def __init__(self, strike, maturity, is_call, quantity, premium=0.0): ...
```

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `sign` | `int` | `+1` for long, `-1` for short |
| `is_long` | `bool` | `quantity > 0` |
| `strike` | `float` | Strike price |
| `maturity` | `float` | Time to expiry |
| `is_call` | `bool` | Call or put |

**Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `payoff_at_expiry(spot)` | `(float) -> float` | Single-point payoff |
| `intrinsic_value(spot)` | `(float) -> float` | `max(S-K, 0)` or `max(K-S, 0)` |

### StockPosition

```python
class StockPosition:
    def __init__(self, quantity, entry_price): ...
```

| Member | Type | Description |
|--------|------|-------------|
| `quantity` | `int` | Positive=long, negative=short |
| `entry_price` | `float` | Entry price |
| `is_long` | `bool` | `quantity > 0` |
| `delta` | `float` | Always `quantity` (stock delta = 1 per share) |
| `pnl(current_price)` | `float` | `quantity * (current_price - entry_price)` |

### Factory Functions

| Function | Parameters | Position |
|----------|------------|----------|
| `long_call(strike, maturity, premium=0, quantity=1)` | | +1 call |
| `short_call(strike, maturity, premium=0, quantity=1)` | | -1 call |
| `long_put(strike, maturity, premium=0, quantity=1)` | | +1 put |
| `short_put(strike, maturity, premium=0, quantity=1)` | | -1 put |
| `long_stock(quantity, entry_price)` | | +N shares |
| `short_stock(quantity, entry_price)` | | -N shares |

### P&L Engine (`pnl.py`)

**RiskMetrics NamedTuple** (11 fields):

```python
class RiskMetrics(NamedTuple):
    mean_return: float
    std_return: float
    var_95: float           # Value at Risk (95%)
    cvar_95: float          # Conditional VaR (95%)
    var_99: float           # Value at Risk (99%)
    cvar_99: float          # Conditional VaR (99%)
    max_drawdown: float
    skewness: float
    kurtosis: float
    percentile_5: float
    percentile_95: float
```

**P&L functions**:

| Function | Signature | Description |
|----------|-----------|-------------|
| `calculate_portfolio_pnl_vectorized(...)` | Numba arrays | Core vectorized P&L |
| `calculate_portfolio_pnl_with_stock(...)` | Numba arrays | P&L including stock |
| `compute_risk_metrics(sim_result, confidence)` | `-> RiskMetrics` | Risk from simulation |
| `compute_payoff_curve(spot_range, ...)` | `-> np.ndarray` | Payoff across spots |
| `prepare_position_arrays(portfolio)` | `-> tuple` | Convert positions to arrays |

### Breakeven Analysis (`breakeven.py`)

**BreakevenResult**:

```python
class BreakevenResult:
    breakevens: List[float]     # Breakeven spot prices
    max_profit: float           # Maximum profit
    max_loss: float             # Maximum loss (negative)
    max_profit_spot: float      # Spot at max profit
    max_loss_spot: float        # Spot at max loss
    profit_zones: List[Tuple[float, float]]  # (low, high) ranges
    loss_zones: List[Tuple[float, float]]    # (low, high) ranges
```

**BreakevenCalculator class**:

```python
class BreakevenCalculator:
    def __init__(self, portfolio): ...
    def calculate(self, spot_range, n_points=1000) -> BreakevenResult: ...
```

**Convenience functions**:

```python
from backend.portfolio import find_breakevens, find_breakevens_from_portfolio

result = find_breakevens_from_portfolio(portfolio, spot_range=(80, 120))
```

### Greeks Surfaces (`greeks_surfaces.py`)

Numba-parallel 3D Greeks surface calculations.

| Function | Signature | Description |
|----------|-----------|-------------|
| `portfolio_greeks_surface_dte(...)` | `-> np.ndarray` | Greeks surface vs DTE |
| `portfolio_greeks_surface_iv(...)` | `-> np.ndarray` | Greeks surface vs IV |
| `single_option_greeks_surface_strike(...)` | `-> np.ndarray` | Single option vs strike |
| `calculate_pnl_curve(...)` | `-> np.ndarray` | P&L curve |
| `calculate_portfolio_pnl_at_expiry_arrays(...)` | `-> np.ndarray` | Expiry P&L (arrays) |

Same Greek index constants as `vectorized_bs.py` (`GREEK_PRICE`, `GREEK_DELTA`, etc.).

### Risk Analysis

**Location**: `backend/portfolio/risk_analysis.py`

```python
class RiskProfile(NamedTuple):
    has_unlimited_profit: bool
    has_unlimited_loss: bool
    max_profit: Optional[float]
    max_loss: Optional[float]
    max_profit_spot: Optional[float]
    max_loss_spot: Optional[float]
```

| Function | Description |
|----------|-------------|
| `check_unlimited_risk(positions, stock=None)` | `-> (bool, bool)` profit/loss |
| `check_unlimited_risk_from_portfolio(portfolio)` | Convenience wrapper |
| `check_unlimited_risk_arrays(option_types, position_types, quantities, ...)` | Numba-optimized |
| `analyze_portfolio_risk(positions, stock, breakeven_result, expiry_pnl)` | `-> RiskProfile` |
| `analyze_portfolio_risk_from_portfolio(portfolio, breakeven_result, expiry_pnl)` | Convenience |
| `get_risk_summary(risk_profile)` | `-> dict` with `risk_level`, `profit_potential`, etc. |

---

## Utils Module

**Location**: `backend/utils/`

Low-level utility functions with Numba acceleration. This module is the **single source of truth** for all BS formulas.

### Distribution Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `norm_cdf(x)` | `(float) -> float` | Standard normal CDF |
| `norm_pdf(x)` | `(float) -> float` | Standard normal PDF |
| `norm_inv_cdf(p)` | `(float) -> float` | Inverse CDF (quantile) |
| `norm_cdf_vec(x)` | `(np.ndarray) -> np.ndarray` | Vectorized CDF |
| `norm_pdf_vec(x)` | `(np.ndarray) -> np.ndarray` | Vectorized PDF |

### Black-Scholes Pricing

| Function | Signature | Description |
|----------|-----------|-------------|
| `d1_d2(s, k, t, r, sigma)` | `-> (float, float)` | BS d1 and d2 |
| `bs_price(s, k, t, r, sigma, is_call)` | `-> float` | BS option price |
| `implied_vol(price, s, k, t, r, is_call)` | `-> float` | Newton-Raphson IV |

### Financial Utilities

| Function | Signature | Description |
|----------|-----------|-------------|
| `discount_factor(r, t)` | `(float, float) -> float` | `exp(-r*t)` |
| `forward_price(s, r, q, t)` | `(float, float, float, float) -> float` | `S * exp((r-q)*t)` |
| `log_moneyness(s, k)` | `(float, float) -> float` | `ln(S/K)` |
| `forward_log_moneyness(s, k, r, q, t)` | `(...) -> float` | `ln(F/K)` |
| `delta_to_strike(delta, s, t, r, sigma, is_call)` | `(...) -> float` | Convert delta to strike |

### Individual BS Greeks

All Numba-compiled:

| Function | Signature | Returns |
|----------|-----------|---------|
| `bs_delta(s, k, t, r, sigma, is_call)` | `-> float` | Raw delta |
| `bs_gamma(s, k, t, r, sigma)` | `-> float` | Raw gamma |
| `bs_vega(s, k, t, r, sigma)` | `-> float` | Scaled per 1% vol |
| `bs_theta(s, k, t, r, sigma, is_call)` | `-> float` | Scaled per day |
| `bs_rho(s, k, t, r, sigma, is_call)` | `-> float` | Scaled per 1% rate |
| `bs_greeks(s, k, t, r, sigma, is_call)` | `-> tuple` | `(price, delta, gamma, vega, theta, rho)` |
| `bs_second_order_greeks(s, k, t, r, sigma)` | `-> tuple` | `(vanna, volga, charm, veta)` |
| `bs_third_order_greeks(s, k, t, r, sigma)` | `-> tuple` | `(speed, zomma, color, ultima)` |

```python
from backend.utils.math import bs_price, implied_vol, bs_greeks

# Price
price = bs_price(s=100, k=100, t=0.5, r=0.05, sigma=0.2, is_call=True)

# Implied vol
iv = implied_vol(price=price, s=100, k=100, t=0.5, r=0.05, is_call=True)

# All first-order Greeks in one call
price, delta, gamma, vega, theta, rho = bs_greeks(100, 100, 0.5, 0.05, 0.2, True)
```

---

## Math Kernels Module

**Location**: `backend/math_kernels/`

Low-level numerical kernels. These are **standalone reference implementations** separate from the production simulation pipeline. Useful for custom simulations, research, and validation.

### SDE Kernels (`sde_kernels.py`)

**Generic schemes**:

| Function | Signature | Description |
|----------|-----------|-------------|
| `euler_step(x, drift, diffusion, dt, dw)` | `-> float` | Euler-Maruyama: `x + drift*dt + diffusion*dw` |
| `milstein_step(x, drift, diffusion, diffusion_prime, dt, dw)` | `-> float` | Milstein with Ito correction |

**GBM kernels**:

| Function | Signature | Description |
|----------|-----------|-------------|
| `gbm_exact_step(s, r, sigma, dt, dw)` | `-> float` | Exact GBM: `S * exp((r-0.5*sigma^2)*dt + sigma*dw)` |
| `gbm_euler_step(s, r, sigma, dt, dw)` | `-> float` | Euler approximation |

**Heston variance kernels**:

| Function | Signature | Description |
|----------|-----------|-------------|
| `heston_euler_step(v, kappa, theta, xi, dt, dw)` | `-> float` | Naive Euler (can go negative) |
| `heston_truncation_step(v, kappa, theta, xi, dt, dw)` | `-> float` | `v = max(v, 0)` in drift/diffusion |
| `heston_reflection_step(v, kappa, theta, xi, dt, dw)` | `-> float` | `v = abs(v)` if negative |
| `heston_qe_step(v, kappa, theta, xi, dt, u1, u2)` | `-> float` | Quadratic-Exponential (Andersen) |
| `heston_spot_step(s, r, v, rho, dt, dw_s)` | `-> float` | Spot process update |

**Jump kernel**:

| Function | Signature | Description |
|----------|-----------|-------------|
| `merton_jump_step(s, r, sigma, lambda_j, mu_j, sigma_j, dt, dw, n_jumps, jump_sizes)` | `-> float` | GBM + Poisson jumps |

### Random Number Generation (`random.py`)

| Function | Signature | Description |
|----------|-----------|-------------|
| `generate_normal(n, seed=None)` | `-> np.ndarray` | Standard normals |
| `generate_normal_2d(n_paths, n_steps, seed=None)` | `-> np.ndarray` | 2D normal array |
| `generate_correlated_normals(n, rho, seed=None)` | `-> (np.ndarray, np.ndarray)` | Correlated pair |
| `generate_correlated_brownian(n_paths, n_steps, rho, dt, seed=None)` | `-> (np.ndarray, np.ndarray)` | Correlated BM increments |
| `generate_antithetic_normals(n, seed=None)` | `-> np.ndarray` | `[Z, -Z]` stacked |
| `generate_antithetic_brownian(n_paths, n_steps, dt, seed=None)` | `-> np.ndarray` | Antithetic BM |
| `compute_cholesky(rho)` | `-> np.ndarray` | 2x2 Cholesky factor |
| `cholesky_transform(z1, z2, rho)` | `-> (np.ndarray, np.ndarray)` | Apply correlation |
| `box_muller_transform(u1, u2)` | `-> (np.ndarray, np.ndarray)` | Uniform to normal |

### Payoff Kernels (`payoff_kernels.py`)

**Vanilla (scalar + vectorized)**:

| Function | Description |
|----------|-------------|
| `call_payoff(s, k)` | `max(s - k, 0)` |
| `put_payoff(s, k)` | `max(k - s, 0)` |
| `call_payoff_vec(s_arr, k)` | Vectorized call |
| `put_payoff_vec(s_arr, k)` | Vectorized put |

**Digital**:

| Function | Description |
|----------|-------------|
| `digital_call_payoff(s, k)` | `1.0 if s > k else 0.0` |
| `digital_put_payoff(s, k)` | `1.0 if s < k else 0.0` |
| `digital_call_payoff_vec(s_arr, k)` | Vectorized |
| `digital_put_payoff_vec(s_arr, k)` | Vectorized |

**Strategy payoffs**:

| Function | Description |
|----------|-------------|
| `straddle_payoff(s_arr, k)` | `call + put` at same strike |
| `strangle_payoff(s_arr, k_put, k_call)` | OTM call + OTM put |
| `butterfly_payoff(s_arr, k1, k2, k3)` | Butterfly spread |

**Exotic payoffs**:

| Function | Description |
|----------|-------------|
| `asian_arithmetic_payoff(paths, k, is_call)` | Average price vs strike |
| `barrier_up_out_call_payoff(paths, k, barrier)` | Knocked out if max > barrier |
| `barrier_down_out_put_payoff(paths, k, barrier)` | Knocked out if min < barrier |

### Regression (`regression.py`)

For Longstaff-Schwartz American option pricing.

**Basis functions**:

| Function | Signature | Description |
|----------|-----------|-------------|
| `laguerre_basis(x, n_basis)` | `-> np.ndarray` | Laguerre polynomials |
| `polynomial_basis(x, n_basis)` | `-> np.ndarray` | Standard polynomials |

**Regression functions**:

| Function | Signature | Description |
|----------|-----------|-------------|
| `lstsq_regression(X, y)` | `-> np.ndarray` | Least squares coefficients |
| `continuation_value(X, coeffs)` | `-> np.ndarray` | Predicted continuation values |

---

## Cross-Cutting Conventions

### Option Type Convention Warning

Different modules use different conventions for `option_type`. Be careful when mixing:

| Module | Convention | Call | Put |
|--------|-----------|------|-----|
| `portfolio/pnl.py` | `int` | `1` | `-1` |
| `portfolio/greeks_surfaces.py` | `int` | `1` | `0` |
| `engines/vectorized_bs.py` | `int` | `1` | `0` |
| Model/Engine layer | `bool` | `is_call=True` | `is_call=False` |

When converting between these, ensure the mapping is correct to avoid sign errors.

---

## Inter-Module Relationships

```
+-----------------------------------------------------------------------+
|                         MODULE DEPENDENCY GRAPH                        |
+-----------------------------------------------------------------------+
|                                                                         |
|                            +---------+                                  |
|                            |  core   | <-- Interfaces & Types           |
|                            +----+----+                                  |
|                                 |                                       |
|         +-----------------------+-----------------------+              |
|         |                       |                       |              |
|         v                       v                       v              |
|    +---------+            +---------+            +-------------+       |
|    | models  |            | engines |            | instruments |       |
|    +----+----+            +----+----+            +------+------+       |
|         |                      |                        |              |
|         |   +------------------+----------------+       |              |
|         |   |                  |                |       |              |
|         v   v                  v                v       v              |
|    +------------+        +----------+     +-------------+             |
|    | simulation |        |  greeks  |     |  portfolio  |             |
|    +-----+------+        +----------+     +-------------+             |
|          |                                                             |
|          v                                                             |
|    +-------------+       +---------+                                   |
|    |math_kernels | <---- |  utils  | <-- Low-level utilities           |
|    +-------------+       +---------+                                   |
|                                                                         |
+-----------------------------------------------------------------------+

Arrows indicate "depends on" relationships.
```

### Relationship Details

| Module | Depends On | Provides To |
|--------|------------|-------------|
| `core` | - | All modules |
| `utils` | NumPy, Numba | All modules |
| `math_kernels` | `utils`, NumPy, Numba | `simulation`, `engines` |
| `models` | `core`, `utils` | `engines` |
| `simulation` | `core`, `math_kernels` | `models`, `engines` |
| `engines` | `core`, `models`, `simulation` | `greeks`, `portfolio` |
| `instruments` | `core` | `engines`, `portfolio` |
| `greeks` | `core`, `utils`, `models` | `portfolio` |
| `portfolio` | `core`, `instruments`, `greeks` | User code |

---

## Complete Examples

### Example 1: Price European Options Across Models

```python
from backend import (
    VanillaOption, MarketEnvironment,
    GBMModel, HestonModel, BatesModel, MertonModel,
    BSAnalyticEngine, FFTEngine, MonteCarloEngine
)

# Common parameters
option = VanillaOption(strike=100, maturity=0.5, is_call=True)
market = MarketEnvironment(spot=100, rate=0.05)

# Define models
models = {
    "GBM": GBMModel(sigma=0.2),
    "Heston": HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7),
    "Bates": BatesModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                        lambda_j=0.1, mu_j=-0.1, sigma_j=0.15),
    "Merton": MertonModel(sigma=0.2, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2),
}

# Price with appropriate engines
bs_engine = BSAnalyticEngine()
fft_engine = FFTEngine()

print(f"{'Model':<10} {'Price':>10} {'Engine':<15}")
print("-" * 40)

for name, model in models.items():
    if name == "GBM":
        result = bs_engine.price(option, model, market)
        engine_name = "BS Analytic"
    else:
        result = fft_engine.price(option, model, market)
        engine_name = "FFT"
    print(f"{name:<10} ${result.price:>9.4f} {engine_name:<15}")
```

### Example 2: Numerical Greeks via Bumping

```python
from backend import MarketEnvironment, VanillaOption, GBMModel, BSAnalyticEngine

option = VanillaOption(strike=100, maturity=0.5, is_call=True)
model = GBMModel(sigma=0.2)
market = MarketEnvironment(spot=100, rate=0.05)
engine = BSAnalyticEngine()

# Numerical delta via market bumping
h = 0.5
price_up = engine.price(option, model, market.bump_spot(+h)).price
price_dn = engine.price(option, model, market.bump_spot(-h)).price
numerical_delta = (price_up - price_dn) / (2 * h)

# Analytic delta for comparison
result = engine.price(option, model, market)
print(f"Numerical delta: {numerical_delta:.6f}")
print(f"Analytic delta:  {result.greeks.delta:.6f}")
```

### Example 3: Monte Carlo Path Analysis

```python
from backend import HestonSimulator, DiscretizationScheme, compute_risk_metrics
import numpy as np

# Simulator
simulator = HestonSimulator(
    v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
    scheme=DiscretizationScheme.QE
)

# Simulate paths
result = simulator.simulate_paths(
    s0=100.0, mu=0.08, t=1.0,
    n_paths=10_000, n_steps=252, seed=42
)

# Analyze results
print(f"Paths: {result.n_paths:,}")
print(f"Terminal mean: ${result.terminal_mean:.2f}")
print(f"Terminal std: ${result.terminal_std:.2f}")
print(f"Has volatility: {result.has_volatility}")

# Percentiles
pcts = result.percentile_paths([5, 50, 95])
print(f"5th percentile: ${pcts[0, -1]:.2f}")
print(f"50th percentile: ${pcts[1, -1]:.2f}")
print(f"95th percentile: ${pcts[2, -1]:.2f}")

# Risk metrics
metrics = compute_risk_metrics(result, confidence=0.95)
print(f"VaR (95%): ${metrics.var_95:.2f}")
print(f"CVaR (95%): ${metrics.cvar_95:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown*100:.1f}%")
```

### Example 4: Full Greeks Analysis

```python
from backend import (
    VanillaOption, GBMModel, MarketEnvironment,
    BSAnalyticEngine, GreeksCalculator
)

# Setup
option = VanillaOption(strike=100, maturity=0.5, is_call=True)
model = GBMModel(sigma=0.2)
market = MarketEnvironment(spot=100, rate=0.05)

# Price with Greeks
engine = BSAnalyticEngine()
result = engine.price(option, model, market)

print(f"Option Price: ${result.price:.4f}")
print(f"\nFirst Order Greeks:")
print(f"  Delta: {result.greeks.delta:.4f}")
print(f"  Gamma: {result.greeks.gamma:.6f}")
print(f"  Vega: {result.greeks.vega:.4f}")
print(f"  Theta: {result.greeks.theta:.4f}")
print(f"  Rho: {result.greeks.rho:.4f}")
```

### Example 5: Iron Condor Portfolio

```python
from backend import (
    OptionsPortfolio,
    long_put, short_put, short_call, long_call,
    find_breakevens_from_portfolio
)
import numpy as np

# Build iron condor
portfolio = OptionsPortfolio()
portfolio.add_position(long_put(strike=90, maturity=0.25, premium=1.00))
portfolio.add_position(short_put(strike=95, maturity=0.25, premium=2.50))
portfolio.add_position(short_call(strike=105, maturity=0.25, premium=2.50))
portfolio.add_position(long_call(strike=110, maturity=0.25, premium=1.00))

# Analysis
net_credit = portfolio.total_premium()
print(f"Iron Condor Analysis")
print(f"=" * 40)
print(f"Net Credit Received: ${net_credit:.2f}")

# P&L at various spots
spots = np.array([85, 90, 95, 100, 105, 110, 115])
pnls = portfolio.compute_pnl_at_expiry(spots)

print(f"\nP&L at Expiry:")
print(f"{'Spot':>8} {'P&L':>10}")
for spot, pnl in zip(spots, pnls):
    print(f"${spot:>7.0f} ${pnl:>9.2f}")

# Breakevens
result = find_breakevens_from_portfolio(portfolio, spot_range=(85, 115))
print(f"\nBreakeven Points: {[f'${b:.2f}' for b in result.breakevens]}")
print(f"Max Profit: ${result.max_profit:.2f}")
print(f"Max Loss: ${result.max_loss:.2f}")
```

### Example 6: GARCH Pricing

```python
from backend import GARCHModel

model = GARCHModel(sigma0=0.20, omega=0.002, alpha=0.05, beta=0.90)
print(f"Persistence: {model.persistence:.3f}")
print(f"Long-run vol: {model.long_run_volatility:.1%}")
print(f"Half-life: {model.half_life:.1f} steps")

# Price via LRNVR Monte Carlo
pricer = model.create_pricer(n_paths=100_000)
result = pricer.price(s0=100, k=100, t=0.25, r=0.05)
print(f"Call price: ${result.price:.4f} +/- ${result.std_error:.4f}")
```

---

## Running Smoke Tests

Each module includes a smoke test that can be run standalone:

```bash
# Run individual module smoke tests
python -m backend.utils.math
python -m backend.simulation.base
python -m backend.engines.fourier.carr_madan
python -m backend.engines.exotic_engine
python -m backend.greeks.analytic
python -m backend.greeks.calculator

# Run all tests with pytest
pytest tests/ -v
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 5.3.0 | 2025 | ExoticAnalyticEngine, type annotations, named params, _instrument_utils |
| 5.2.0 | 2025 | Comprehensive API documentation enrichment |
| 5.1.0 | 2025 | Added risk_analysis module with RiskProfile, check_unlimited_risk |
| 5.0.0 | 2025 | Three Pillars Architecture, unified API |
| 4.0.0 | 2025 | Added GARCH family models |
| 3.0.0 | 2025 | Added portfolio management |
| 2.0.0 | 2025 | FFT engine, Heston/Bates models |
| 1.0.0 | 2025 | Initial release with GBM |

---

## License

Copyright 2025 Thomas. All rights reserved.
