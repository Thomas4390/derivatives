"""
Engines Module
==============

Pricing engines implementing the three-pillar architecture.

Architecture
------------
Engines are the "How" - numerical methods that bridge:
- Instrument (the contract/payoff)
- Model (the stochastic dynamics)
- Market (current conditions)

Available Engines
-----------------
- BSAnalyticEngine: Black-Scholes analytical pricing (GBM only)
- FFTEngine: Carr-Madan FFT pricing (any model with characteristic function)
- MonteCarloEngine: Monte Carlo simulation (any model with SDE)

Usage
-----
    from backend.engines import BSAnalyticEngine, FFTEngine, MonteCarloEngine
    from backend.instruments.options import VanillaOption
    from backend.models import GBMModel, HestonModel
    from backend.core.market import MarketEnvironment

    # Create components
    option = VanillaOption(strike=100, maturity=0.5, is_call=True)
    market = MarketEnvironment(spot=100, rate=0.05)

    # Analytical pricing (GBM only)
    bs_engine = BSAnalyticEngine()
    gbm = GBMModel(sigma=0.20)
    result = bs_engine.price(option, gbm, market)

    # FFT pricing (any model with characteristic function)
    fft_engine = FFTEngine()
    heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    result = fft_engine.price(option, heston, market)

    # Monte Carlo pricing (any model with SDE)
    mc_engine = MonteCarloEngine(n_paths=100000, seed=42)
    result = mc_engine.price(option, heston, market)

Compatibility Matrix
--------------------
                        | Analytical | FFT | Monte Carlo |
    --------------------|------------|-----|-------------|
    GBMModel            |     ✓      |  ✓  |      ✓      |
    HestonModel         |     ✗      |  ✓  |      ✓      |
    BatesModel          |     ✗      |  ✓  |      ✓      |
    MertonModel         |     ✗      |  ✓  |      ✓      |

Author: Thomas
Created: 2025
Version: 2.0.0
"""

# Engine classes - import from renamed module files
from backend.engines.analytic_engine import BSAnalyticEngine
from backend.engines.fft_engine import FFTEngine
from backend.engines.mc_engine import MonteCarloEngine

# Configuration classes from underlying engines
from backend.engines.fourier.carr_madan import FFTConfig
from backend.engines.monte_carlo.mc_base import MCConfig, MCResult

# Vectorized Numba functions for high-performance array computations
from backend.engines.vectorized_bs import (
    calculate_first_order_greeks,
    calculate_all_greeks,
    calculate_portfolio_greeks_3d_dte_vectorized,
    calculate_portfolio_greeks_3d_iv_vectorized,
    calculate_greeks_3d_strike_vectorized,
    calculate_portfolio_pnl_at_expiry,
    calculate_pnl_curve,
    calculate_greeks_vectorized,
)


__all__ = [
    # Analytic engines
    "BSAnalyticEngine",
    # Fourier engines
    "FFTEngine",
    # Monte Carlo engines
    "MonteCarloEngine",
    # Configuration
    "FFTConfig",
    "MCConfig",
    "MCResult",
    # Vectorized functions (Numba-optimized)
    "calculate_first_order_greeks",
    "calculate_all_greeks",
    "calculate_portfolio_greeks_3d_dte_vectorized",
    "calculate_portfolio_greeks_3d_iv_vectorized",
    "calculate_greeks_3d_strike_vectorized",
    "calculate_portfolio_pnl_at_expiry",
    "calculate_pnl_curve",
    "calculate_greeks_vectorized",
]

__version__ = "2.0.0"
