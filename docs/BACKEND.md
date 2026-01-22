# Backend Module Documentation

**Version**: 5.0.0
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
14. [Inter-Module Relationships](#inter-module-relationships)
15. [Complete Examples](#complete-examples)

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
option = VanillaOption(strike=100, expiry=0.5, is_call=True)
model = GBMModel(sigma=0.2)
engine = BSAnalyticEngine()

price = engine.price(option, model, market)
print(f"Price: ${price.price:.4f}")
```

---

## Architecture: Three Pillars

The framework follows a clean separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    PRICING WORKFLOW                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Instrument          Model              Engine             │
│   ──────────         ─────              ──────             │
│   (What)             (Physics)          (How)              │
│                                                             │
│   VanillaOption  +   GBMModel      +   BSAnalyticEngine    │
│   EuropeanCall       HestonModel       FFTEngine           │
│   AmericanPut        BatesModel        MonteCarloEngine    │
│   OptionStrategy     MertonModel                           │
│                      GARCHModel                            │
│                                                             │
│   ─────────────────────────────────────────────────────    │
│                           ↓                                 │
│                    PricingResult                            │
│                    (price, greeks, metadata)                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
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
    expiry=0.5,
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
├── __init__.py              # Unified API exports
├── core/                    # Abstract interfaces and base types
│   ├── __init__.py
│   ├── interfaces.py        # Instrument, Model, PricingEngine protocols
│   ├── market.py            # MarketEnvironment
│   └── results.py           # PricingResult, ExerciseStyle
├── models/                  # Pricing models
│   ├── __init__.py
│   ├── base.py              # BaseModel, PricingCapability
│   ├── gbm.py               # GBMModel
│   ├── heston.py            # HestonModel
│   ├── merton.py            # MertonModel
│   ├── bates.py             # BatesModel
│   ├── garch.py             # GARCHModel, NGARCHModel, GJRGARCHModel
│   └── characteristic_functions/  # CFs for FFT pricing
│       ├── heston_cf.py
│       ├── merton_cf.py
│       └── bates_cf.py
├── engines/                 # Pricing engines
│   ├── __init__.py
│   ├── unified.py           # BSAnalyticEngine, FFTEngine, MonteCarloEngine
│   ├── fourier/
│   │   └── carr_madan.py    # Carr-Madan FFT algorithm
│   └── monte_carlo/
│       ├── mc_base.py       # Generic MC engine
│       └── garch_pricer.py  # GARCH-specific pricer with LRNVR
├── simulation/              # Monte Carlo simulators
│   ├── __init__.py
│   ├── base.py              # BaseSimulator, SimulationResult
│   ├── enums.py             # ModelType, DiscretizationScheme, Measure
│   ├── factory.py           # create_simulator factory
│   ├── risk_engine.py       # RiskMetrics, compute_risk_metrics
│   └── models/              # Concrete simulators
│       ├── gbm.py
│       ├── heston.py
│       ├── merton.py
│       ├── bates.py
│       └── garch.py         # GARCH, NGARCH, GJR-GARCH
├── instruments/             # Financial instruments
│   ├── __init__.py
│   ├── options.py           # VanillaOption, EuropeanCall, etc.
│   ├── payoffs.py           # VanillaCallPayoff, VanillaPutPayoff
│   └── strategies.py        # OptionStrategy, IronCondor, Straddle
├── greeks/                  # Greeks calculation
│   ├── __init__.py
│   ├── calculator.py        # GreeksCalculator, calculate_greeks
│   ├── analytic.py          # bs_greeks_*, analytical formulas
│   └── numerical.py         # finite_difference_greeks
├── portfolio/               # Portfolio management
│   ├── __init__.py
│   ├── portfolio.py         # OptionsPortfolio
│   ├── positions.py         # PortfolioPosition, StockPosition
│   ├── breakeven.py         # BreakevenCalculator
│   └── factory.py           # long_call, short_put, etc.
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── math.py              # bs_price, implied_vol
│   └── distributions.py     # norm_cdf, norm_pdf (Numba)
└── math_kernels/            # Low-level numerical kernels
    ├── __init__.py
    ├── sde_kernels.py       # Euler, Milstein, QE discretization
    ├── payoff_kernels.py    # Vectorized payoff computation
    ├── regression.py        # Longstaff-Schwartz for Americans
    └── random.py            # Correlated Brownian, antithetic
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
| `ExerciseStyle` | Enum for European/American exercise |

### Example

```python
from backend import MarketEnvironment, PricingResult, ExerciseStyle

# Create market environment
market = MarketEnvironment(
    spot=100.0,
    rate=0.05,
    dividend_yield=0.02
)

# Check exercise style
print(ExerciseStyle.EUROPEAN)  # ExerciseStyle.EUROPEAN
print(ExerciseStyle.AMERICAN)  # ExerciseStyle.AMERICAN
```

---

## Models Module

**Location**: `backend/models/`

Pricing models define the stochastic dynamics of the underlying asset.

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

### Model Capabilities

Each model declares its pricing capabilities:

```python
from backend import HestonModel, PricingCapability

model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

# Check what the model supports
print(model.capabilities)
# {PricingCapability.ANALYTIC_CALL, PricingCapability.FFT,
#  PricingCapability.MONTE_CARLO, PricingCapability.SIMULATION}
```

### Example: GBM Model

```python
from backend import GBMModel

model = GBMModel(sigma=0.2)

# Model provides characteristic function for FFT
cf = model.characteristic_function(s0=100, t=0.5, r=0.05)
u = 1.0 + 0.5j
phi = cf(u)

# Model provides terminal simulator for MC
simulator = model.terminal_simulator(s0=100, t=0.5, r=0.05)
terminals = simulator(n_paths=10000, n_steps=100, seed=42)
```

### Example: Heston Model

```python
from backend import HestonModel

model = HestonModel(
    v0=0.04,      # Initial variance (20% vol)
    kappa=2.0,    # Mean reversion speed
    theta=0.04,   # Long-run variance
    xi=0.3,       # Volatility of variance
    rho=-0.7      # Correlation (negative = leverage effect)
)

# Check Feller condition: 2*kappa*theta > xi^2
feller = 2 * model.kappa * model.theta > model.xi ** 2
print(f"Feller condition satisfied: {feller}")  # True
```

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

### Low-Level Engines

| Engine | Description |
|--------|-------------|
| `CarrMadanFFTEngine` | Generic FFT with characteristic function |
| `GenericMCEngine` | Generic MC with terminal simulator |
| `GARCHMCPricer` | GARCH-specific with LRNVR measure change |

### Example: Black-Scholes Engine

```python
from backend import VanillaOption, GBMModel, BSAnalyticEngine, MarketEnvironment

option = VanillaOption(strike=100, expiry=0.5, is_call=True)
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

option = VanillaOption(strike=100, expiry=0.5, is_call=True)
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

option = VanillaOption(strike=100, expiry=0.5, is_call=True)
model = BatesModel(
    v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
    lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
)
market = MarketEnvironment(spot=100, rate=0.05)

# Configure Monte Carlo
config = MCConfig(n_paths=100_000, n_steps=252, seed=42)
engine = MonteCarloEngine(config=config)

result = engine.price(option, model, market)
print(f"Bates Call Price: ${result.price:.4f} ± ${result.std_error:.4f}")
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

### Discretization Schemes

For stochastic volatility models:

| Scheme | Description |
|--------|-------------|
| `EULER` | Simple Euler (can have negative variance) |
| `FULL_TRUNCATION` | Variance floored at 0 |
| `REFLECTION` | Negative variance reflected |
| `QE` | Quadratic Exponential (most accurate) |

### Example: Path Simulation

```python
from backend import HestonSimulator, DiscretizationScheme

# Create simulator
simulator = HestonSimulator(
    v0=0.04,
    kappa=2.0,
    theta=0.04,
    xi=0.3,
    rho=-0.7,
    scheme=DiscretizationScheme.QE  # Best accuracy
)

# Simulate paths
result = simulator.simulate_paths(
    s0=100.0,
    mu=0.08,        # Expected return (P-measure)
    t=1.0,          # 1 year
    n_paths=10_000,
    n_steps=252,    # Daily
    seed=42
)

print(f"Paths shape: {result.price_paths.shape}")
print(f"Terminal mean: ${result.terminal_mean:.2f}")
print(f"Has volatility: {result.has_volatility}")
```

### Example: Factory Pattern

```python
from backend import create_simulator, ModelType

# Create by string name
sim = create_simulator("heston", v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

# Create by enum
sim = create_simulator(ModelType.GARCH, sigma0=0.2, omega=1e-6, alpha=0.1, beta=0.85)
```

### Example: Risk Metrics

```python
from backend import GBMSimulator, compute_risk_metrics

simulator = GBMSimulator(sigma=0.2)
result = simulator.simulate_paths(s0=100, mu=0.08, t=1.0, n_paths=10_000, n_steps=252)

# Compute risk metrics
metrics = compute_risk_metrics(result, confidence=0.95)

print(f"VaR (95%): ${metrics.var_95:.2f}")
print(f"CVaR (95%): ${metrics.cvar_95:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown*100:.1f}%")
```

---

## Instruments Module

**Location**: `backend/instruments/`

The instruments module defines tradeable contracts.

### Options

| Class | Description |
|-------|-------------|
| `VanillaOption` | Generic vanilla option |
| `EuropeanCall` | European call (convenience) |
| `EuropeanPut` | European put (convenience) |
| `AmericanCall` | American call |
| `AmericanPut` | American put |

### Payoffs

| Class | Description |
|-------|-------------|
| `VanillaCallPayoff` | max(S - K, 0) |
| `VanillaPutPayoff` | max(K - S, 0) |
| `CompositePayoff` | Combination of payoffs |

### Strategies

| Class | Description | Legs |
|-------|-------------|------|
| `OptionStrategy` | Generic multi-leg strategy | Any |
| `IronCondor` | Short strangle + long wings | 4 |
| `Straddle` | ATM call + ATM put | 2 |
| `Butterfly` | Bull + bear spread | 3 |

### Example: Vanilla Option

```python
from backend import VanillaOption, EuropeanCall

# Generic creation
option = VanillaOption(
    strike=100.0,
    expiry=0.5,
    is_call=True,
    style="european"
)

# Convenience class
call = EuropeanCall(strike=100.0, expiry=0.5)
put = EuropeanPut(strike=100.0, expiry=0.5)
```

### Example: Iron Condor Strategy

```python
from backend import IronCondor, GBMModel, BSAnalyticEngine, MarketEnvironment

# Create iron condor
strategy = IronCondor(
    expiry=0.25,
    put_long_strike=90,
    put_short_strike=95,
    call_short_strike=105,
    call_long_strike=110
)

# Price the strategy
model = GBMModel(sigma=0.2)
market = MarketEnvironment(spot=100, rate=0.05)
engine = BSAnalyticEngine()

result = engine.price(strategy, model, market)
print(f"Iron Condor Price: ${result.price:.4f}")
```

---

## Greeks Module

**Location**: `backend/greeks/`

Comprehensive Greeks calculation with analytic and numerical methods.

### First Order Greeks

| Greek | Formula | Interpretation |
|-------|---------|----------------|
| Delta (Δ) | ∂V/∂S | Price sensitivity to spot |
| Vega (ν) | ∂V/∂σ | Price sensitivity to volatility |
| Theta (Θ) | ∂V/∂t | Time decay |
| Rho (ρ) | ∂V/∂r | Interest rate sensitivity |

### Second Order Greeks

| Greek | Formula | Interpretation |
|-------|---------|----------------|
| Gamma (Γ) | ∂²V/∂S² | Delta sensitivity |
| Vanna | ∂²V/∂S∂σ | Delta-vol cross |
| Volga | ∂²V/∂σ² | Vega convexity |
| Charm | ∂²V/∂S∂t | Delta decay |

### Third Order Greeks

| Greek | Formula |
|-------|---------|
| Speed | ∂³V/∂S³ |
| Zomma | ∂³V/∂S²∂σ |
| Color | ∂³V/∂S²∂t |
| Ultima | ∂³V/∂σ³ |

### Example: Analytic Greeks

```python
from backend import bs_all_greeks

greeks = bs_all_greeks(
    s=100.0,
    k=100.0,
    t=0.5,
    r=0.05,
    sigma=0.2,
    is_call=True
)

print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.6f}")
print(f"Vega: {greeks['vega']:.4f}")
print(f"Theta: {greeks['theta']:.4f}")
print(f"Rho: {greeks['rho']:.4f}")
```

### Example: Calculator Interface

```python
from backend import GreeksCalculator, VanillaOption, GBMModel, MarketEnvironment

calculator = GreeksCalculator()

option = VanillaOption(strike=100, expiry=0.5, is_call=True)
model = GBMModel(sigma=0.2)
market = MarketEnvironment(spot=100, rate=0.05)

greeks = calculator.calculate(option, model, market)
print(f"Delta: {greeks.delta:.4f}")
print(f"Gamma: {greeks.gamma:.6f}")
```

### Example: Numerical Greeks

```python
from backend import finite_difference_greeks

greeks = finite_difference_greeks(
    s=100.0,
    k=100.0,
    t=0.5,
    r=0.05,
    sigma=0.2,
    is_call=True,
    h_spot=0.01,
    h_vol=0.001,
    h_time=1/252
)
```

---

## Portfolio Module

**Location**: `backend/portfolio/`

Portfolio management with P&L analysis and breakeven calculation.

### Components

| Class | Description |
|-------|-------------|
| `OptionsPortfolio` | Main portfolio container |
| `PortfolioPosition` | Option position |
| `StockPosition` | Stock/underlying position |
| `BreakevenCalculator` | Find breakeven prices |

### Factory Functions

| Function | Position |
|----------|----------|
| `long_call(strike, expiry, premium, quantity)` | +1 call |
| `short_call(strike, expiry, premium, quantity)` | -1 call |
| `long_put(strike, expiry, premium, quantity)` | +1 put |
| `short_put(strike, expiry, premium, quantity)` | -1 put |
| `long_stock(price, quantity)` | +N shares |
| `short_stock(price, quantity)` | -N shares |

### Example: Building a Portfolio

```python
from backend import (
    OptionsPortfolio,
    long_call, short_call, long_put, short_put,
    find_breakevens_from_portfolio
)

# Build a bull call spread
portfolio = OptionsPortfolio()
portfolio.add_position(long_call(strike=100, expiry=0.25, premium=5.50))
portfolio.add_position(short_call(strike=110, expiry=0.25, premium=2.00))

# Analyze P&L at various prices
import numpy as np
spot_range = np.linspace(80, 130, 100)
pnl = portfolio.compute_pnl_at_expiry(spot_range)

# Find breakeven points
breakevens = find_breakevens_from_portfolio(portfolio, spot_range=(80, 130))
print(f"Breakeven: ${breakevens.breakevens[0]:.2f}")
```

### Example: Complex Strategy

```python
from backend import OptionsPortfolio, long_call, short_call, long_put, short_put

# Iron Condor
portfolio = OptionsPortfolio()
portfolio.add_position(long_put(strike=90, expiry=0.25, premium=1.00))
portfolio.add_position(short_put(strike=95, expiry=0.25, premium=2.50))
portfolio.add_position(short_call(strike=105, expiry=0.25, premium=2.50))
portfolio.add_position(long_call(strike=110, expiry=0.25, premium=1.00))

# Net credit received
net_premium = portfolio.total_premium()
print(f"Net credit: ${net_premium:.2f}")
```

---

## Utils Module

**Location**: `backend/utils/`

Low-level utility functions with Numba acceleration.

### Functions

| Function | Description |
|----------|-------------|
| `norm_cdf(x)` | Standard normal CDF |
| `norm_pdf(x)` | Standard normal PDF |
| `norm_cdf_vec(x)` | Vectorized CDF |
| `norm_pdf_vec(x)` | Vectorized PDF |
| `d1_d2(s, k, t, r, sigma)` | Black-Scholes d1, d2 |
| `bs_price(s, k, t, r, sigma, is_call)` | Black-Scholes price |
| `implied_vol(price, s, k, t, r, is_call)` | Newton-Raphson IV |

### Example

```python
from backend import norm_cdf, d1_d2
from backend.utils import bs_price, implied_vol

# Normal distribution
print(f"N(0) = {norm_cdf(0):.4f}")      # 0.5
print(f"N(1.96) = {norm_cdf(1.96):.4f}")  # ~0.975

# Black-Scholes d1, d2
d1, d2 = d1_d2(s=100, k=100, t=0.5, r=0.05, sigma=0.2)
print(f"d1 = {d1:.4f}, d2 = {d2:.4f}")

# Price calculation
price = bs_price(s=100, k=100, t=0.5, r=0.05, sigma=0.2, is_call=True)
print(f"BS Price: ${price:.4f}")

# Implied volatility
iv = implied_vol(price=price, s=100, k=100, t=0.5, r=0.05, is_call=True)
print(f"Implied Vol: {iv*100:.1f}%")
```

---

## Math Kernels Module

**Location**: `backend/math_kernels/`

Low-level numerical kernels for advanced users.

### Components

| File | Contents |
|------|----------|
| `sde_kernels.py` | Euler, Milstein, GBM, Heston QE |
| `payoff_kernels.py` | Vectorized payoff evaluation |
| `regression.py` | Longstaff-Schwartz for American options |
| `random.py` | Correlated Brownian, antithetic variates |

### Example: SDE Discretization

```python
from backend.math_kernels.sde_kernels import (
    euler_step,
    milstein_step,
    gbm_exact_step,
    heston_qe_step
)
import numpy as np

# GBM exact step
s0 = 100.0
dt = 1/252
r, sigma = 0.05, 0.2
z = np.random.standard_normal()
s1 = gbm_exact_step(s0, r, sigma, dt, z)
```

### Example: Longstaff-Schwartz

```python
from backend.math_kernels.regression import longstaff_schwartz_price
import numpy as np

# Price American put
terminals = np.random.lognormal(np.log(100), 0.2, 10000)
price = longstaff_schwartz_price(
    paths=terminals.reshape(-1, 1),
    k=100.0,
    r=0.05,
    dt=1.0,
    is_call=False
)
```

---

## Inter-Module Relationships

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MODULE DEPENDENCY GRAPH                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                            ┌─────────┐                                   │
│                            │  core   │ ◄── Interfaces & Types            │
│                            └────┬────┘                                   │
│                                 │                                        │
│         ┌───────────────────────┼───────────────────────┐               │
│         │                       │                       │               │
│         ▼                       ▼                       ▼               │
│    ┌─────────┐            ┌─────────┐            ┌─────────────┐        │
│    │ models  │            │ engines │            │ instruments │        │
│    └────┬────┘            └────┬────┘            └──────┬──────┘        │
│         │                      │                        │               │
│         │   ┌──────────────────┼────────────────┐      │               │
│         │   │                  │                │      │               │
│         ▼   ▼                  ▼                ▼      ▼               │
│    ┌────────────┐        ┌──────────┐     ┌─────────────┐              │
│    │ simulation │        │  greeks  │     │  portfolio  │              │
│    └─────┬──────┘        └──────────┘     └─────────────┘              │
│          │                                                              │
│          ▼                                                              │
│    ┌─────────────┐       ┌─────────┐                                   │
│    │math_kernels │ ◄──── │  utils  │ ◄── Low-level utilities           │
│    └─────────────┘       └─────────┘                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

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
option = VanillaOption(strike=100, expiry=0.5, is_call=True)
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

### Example 2: Volatility Surface Generation

```python
from backend import HestonModel, FFTEngine, VanillaOption, MarketEnvironment
import numpy as np

# Model and market
model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
market = MarketEnvironment(spot=100, rate=0.05)
engine = FFTEngine()

# Grid
strikes = np.array([80, 90, 95, 100, 105, 110, 120])
maturities = np.array([0.1, 0.25, 0.5, 1.0])

# Build surface
surface = np.zeros((len(strikes), len(maturities)))
for j, T in enumerate(maturities):
    for i, K in enumerate(strikes):
        option = VanillaOption(strike=K, expiry=T, is_call=True)
        result = engine.price(option, model, market)
        surface[i, j] = result.price

print("Volatility Surface (Prices):")
print(f"{'K/T':<8}", end="")
for T in maturities:
    print(f"{T:>8.2f}", end="")
print()
for i, K in enumerate(strikes):
    print(f"{K:<8.0f}", end="")
    for j in range(len(maturities)):
        print(f"${surface[i,j]:>7.2f}", end="")
    print()
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
    s0=100.0,
    mu=0.08,
    t=1.0,
    n_paths=10_000,
    n_steps=252,
    seed=42
)

# Analyze results
print(f"Simulation Summary")
print(f"=" * 40)
print(f"Paths: {result.n_paths:,}")
print(f"Steps: {result.n_steps}")
print(f"Terminal mean: ${result.terminal_mean:.2f}")
print(f"Terminal std: ${result.terminal_std:.2f}")

# Percentiles
pcts = result.percentile_paths([5, 50, 95])
print(f"\nTerminal Distribution:")
print(f"  5th percentile: ${pcts[0, -1]:.2f}")
print(f" 50th percentile: ${pcts[1, -1]:.2f}")
print(f" 95th percentile: ${pcts[2, -1]:.2f}")

# Risk metrics
metrics = compute_risk_metrics(result, confidence=0.95)
print(f"\nRisk Metrics (95% confidence):")
print(f"  VaR: ${metrics.var_95:.2f}")
print(f"  CVaR: ${metrics.cvar_95:.2f}")
print(f"  Max Drawdown: {metrics.max_drawdown*100:.1f}%")
```

### Example 4: Full Greeks Analysis

```python
from backend import (
    VanillaOption, GBMModel, MarketEnvironment,
    BSAnalyticEngine, GreeksCalculator
)

# Setup
option = VanillaOption(strike=100, expiry=0.5, is_call=True)
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
portfolio.add_position(long_put(strike=90, expiry=0.25, premium=1.00))
portfolio.add_position(short_put(strike=95, expiry=0.25, premium=2.50))
portfolio.add_position(short_call(strike=105, expiry=0.25, premium=2.50))
portfolio.add_position(long_call(strike=110, expiry=0.25, premium=1.00))

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

---

## Running Smoke Tests

Each module includes a smoke test that can be run standalone:

```bash
# Run individual module smoke tests
python -m backend.utils.math
python -m backend.simulation.base
python -m backend.engines.fourier.carr_madan
python -m backend.greeks.analytic

# Run all tests with pytest
pytest tests/ -v
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 5.0.0 | 2025 | Three Pillars Architecture, unified API |
| 4.0.0 | 2025 | Added GARCH family models |
| 3.0.0 | 2025 | Added portfolio management |
| 2.0.0 | 2025 | FFT engine, Heston/Bates models |
| 1.0.0 | 2025 | Initial release with GBM |

---

## License

Copyright 2025 Thomas. All rights reserved.
