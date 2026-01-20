"""
Option Pricing Package
======================

High-performance option pricing for multiple models.

Pricing Methods:
- Analytical (Black-Scholes): For GBM models
- FFT (Carr-Madan): For Heston and Bates stochastic volatility
- Monte Carlo: For all models (validation, path-dependent options)

Architecture:
- Each model has ONE unified pricer class with multiple pricing methods
- Generic engines (CarrMadanFFTEngine, MonteCarloEngine) are shared
- Pricing method is selected via `method=PricingMethod.XXX` parameter

Quick Start:
------------
# Method 1: Create pricer directly
from backend.option_pricing import BlackScholesPricer, HestonPricer, BatesPricer

# Black-Scholes (analytical by default)
bs_pricer = BlackScholesPricer(sigma=0.20)
result = bs_pricer.price(s0=100, k=100, t=0.25, r=0.05)

# Heston (FFT by default)
heston_pricer = HestonPricer(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
result = heston_pricer.price(s0=100, k=100, t=0.25, r=0.05)

# Heston with Monte Carlo
result_mc = heston_pricer.price(s0=100, k=100, t=0.25, r=0.05,
                                 method=PricingMethod.MONTE_CARLO)

# Method 2 (Recommended): Use unified models
from backend.models import HestonModel

model = HestonModel.from_params(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
pricer = model.create_pricer()  # FFT by default
result = pricer.price(s0=100, k=100, t=0.25, r=0.05)

Author: Derivatives Pricing Project
"""

# =============================================================================
# Core Classes
# =============================================================================

from .base import (
    BasePricer,
    AnalyticalPricer,
    FFTPricer,
    MonteCarloPricer,
    PricingResult,
    PricingMethod,
    OptionType,
)

# =============================================================================
# Generic Pricing Engines
# =============================================================================

from .engines import (
    CarrMadanFFTEngine,
    FFTConfig,
    MonteCarloEngine,
    MCConfig,
    MCResult,
)

# =============================================================================
# Unified Pricers (recommended)
# =============================================================================

# Black-Scholes / GBM
from .black_scholes import (
    BlackScholesPricer,
    GBMPricer,  # Alias for BlackScholesPricer
    GBMParams,
    bs_call_price,
    bs_put_price,
    bs_greeks,
    implied_volatility,
)

# Heston Stochastic Volatility
from .heston import (
    HestonPricer,
    HestonParams,
    heston_call_price,
    heston_put_price,
)

# Bates (Heston + Jumps)
from .bates import (
    BatesPricer,
    BatesParams,
    bates_call_price,
    bates_put_price,
)

# Merton Jump-Diffusion
from .merton import (
    MertonPricer,
    MertonParams,
    merton_call_price,
    merton_put_price,
)

# =============================================================================
# GARCH Monte Carlo
# =============================================================================

from .garch import (
    GARCHMCPricer,
    GARCHType,
    create_garch_pricer,
    create_ngarch_pricer,
    create_gjr_garch_pricer,
)

# =============================================================================
# Legacy Calculator (for backward compatibility)
# =============================================================================

from .options_calculator import (
    OptionsPortfolio,
    OptionPosition,
    StockPosition,
    BreakevenResult,
    # Core functions
    calculate_all_greeks,
    calculate_first_order_greeks,
    calculate_second_order_greeks,
    calculate_third_order_greeks,
    black_scholes_call_price,
    black_scholes_put_price,
    # 3D matrices
    calculate_greeks_3d_dte_matrix,
    calculate_greeks_3d_iv_matrix,
    calculate_greeks_3d_strike,
    calculate_portfolio_greeks_3d_dte,
    calculate_portfolio_greeks_3d_iv,
    # Utilities
    find_breakeven_points,
    norm_cdf,
    norm_pdf,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Core classes
    "BasePricer",
    "AnalyticalPricer",
    "FFTPricer",
    "MonteCarloPricer",
    "PricingResult",
    "PricingMethod",
    "OptionType",
    # Generic engines
    "CarrMadanFFTEngine",
    "FFTConfig",
    "MonteCarloEngine",
    "MCConfig",
    "MCResult",
    # Black-Scholes / GBM
    "BlackScholesPricer",
    "GBMPricer",
    "GBMParams",
    "bs_call_price",
    "bs_put_price",
    "bs_greeks",
    "implied_volatility",
    # Heston
    "HestonPricer",
    "HestonParams",
    "heston_call_price",
    "heston_put_price",
    # Bates
    "BatesPricer",
    "BatesParams",
    "bates_call_price",
    "bates_put_price",
    # Merton
    "MertonPricer",
    "MertonParams",
    "merton_call_price",
    "merton_put_price",
    # GARCH
    "GARCHMCPricer",
    "GARCHType",
    "create_garch_pricer",
    "create_ngarch_pricer",
    "create_gjr_garch_pricer",
    # Legacy (portfolio calculator)
    "OptionsPortfolio",
    "OptionPosition",
    "StockPosition",
    "BreakevenResult",
    "calculate_all_greeks",
    "calculate_first_order_greeks",
    "calculate_second_order_greeks",
    "calculate_third_order_greeks",
    "black_scholes_call_price",
    "black_scholes_put_price",
    "calculate_greeks_3d_dte_matrix",
    "calculate_greeks_3d_iv_matrix",
    "calculate_greeks_3d_strike",
    "calculate_portfolio_greeks_3d_dte",
    "calculate_portfolio_greeks_3d_iv",
    "find_breakeven_points",
    "norm_cdf",
    "norm_pdf",
]

__version__ = "3.0.0"
