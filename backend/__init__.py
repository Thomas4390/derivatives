"""
Backend Package - Unified API
=============================

High-performance derivatives pricing and portfolio management.

Quick Start
-----------
    from backend import (
        # Portfolio management (NEW)
        OptionsPortfolio, OptionPosition, StockPosition,
        GreeksResult, BreakevenResult,
        # Option pricers
        BlackScholesPricer, HestonPricer, BatesPricer, MertonPricer,
        PricingMethod, PricingResult,
        # Simulators
        GBMSimulator, HestonSimulator, MertonSimulator, BatesSimulator,
        SimulationResult,
        # Unified models
        GBMModel, HestonModel, BatesModel, MertonModel,
    )

Examples
--------
Portfolio Greeks (New API):
    >>> portfolio = OptionsPortfolio(sigma=0.20)
    >>> portfolio.add_option(OptionPosition('call', 'long', strike=100, premium=5.0))
    >>> greeks = portfolio.calculate_greeks(spot=100, rate=0.05, time_to_expiry=0.5)
    >>> print(f"Delta: {greeks.delta:.4f}")

With custom pricer:
    >>> pricer = HestonPricer(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    >>> portfolio = OptionsPortfolio(pricer=pricer)

Single option pricing:
    >>> pricer = BlackScholesPricer(sigma=0.20)
    >>> result = pricer.price(s0=100, k=100, t=0.25, r=0.05)
    >>> print(f"Price: {result.price:.4f}")

Simulation:
    >>> sim = HestonSimulator(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    >>> result = sim.simulate_paths(s0=100, mu=0.08, t=1.0, n_paths=10000)

Author: Derivatives Pricing Project
Version: 4.0.0
"""

# =============================================================================
# Portfolio Management (NEW - Primary API)
# =============================================================================

from .portfolio import (
    # Main portfolio class
    OptionsPortfolio,
    # Position classes
    OptionPosition,
    StockPosition,
    OptionType,
    PositionType,
    # Greeks
    GreeksResult,
    GreeksCalculator,
    AnalyticalGreeksStrategy,
    FiniteDiffGreeksStrategy,
    # Breakeven
    BreakevenResult,
    BreakevenCalculator,
    find_breakevens,
)

# =============================================================================
# Utilities
# =============================================================================

from .utils import (
    norm_cdf,
    norm_pdf,
    norm_cdf_vec,
    norm_pdf_vec,
    d1_d2,
)

# =============================================================================
# Option Pricers
# =============================================================================

from .option_pricing import (
    # Base classes
    BasePricer,
    PricingResult,
    PricingMethod,
    # Black-Scholes / GBM
    BlackScholesPricer,
    GBMPricer,
    bs_call_price,
    bs_put_price,
    bs_greeks,
    implied_volatility,
    # Heston
    HestonPricer,
    heston_call_price,
    heston_put_price,
    # Bates
    BatesPricer,
    bates_call_price,
    bates_put_price,
    # Merton
    MertonPricer,
    merton_call_price,
    merton_put_price,
    # GARCH
    GARCHMCPricer,
    GARCHType,
    # Engines
    CarrMadanFFTEngine,
    FFTConfig,
    MonteCarloEngine,
    MCConfig,
)

# =============================================================================
# Simulation
# =============================================================================

from .simulation import (
    # Base
    BaseSimulator,
    SimulationResult,
    # Enums
    ModelType,
    DiscretizationScheme,
    Measure,
    # Simulators
    GBMSimulator,
    HestonSimulator,
    MertonSimulator,
    BatesSimulator,
    GARCHSimulator,
    NGARCHSimulator,
    GJRGARCHSimulator,
    # Factory
    create_simulator,
    # P&L engine
    RiskMetrics,
    compute_risk_metrics,
)

# =============================================================================
# Unified Models
# =============================================================================

from .models import (
    # Base
    BaseModel,
    PricingCapability,
    registry,
    # Models
    GBMModel,
    HestonModel,
    MertonModel,
    BatesModel,
    GARCHModel,
    NGARCHModel,
    GJRGARCHModel,
    # Parameters
    GBMParams,
    HestonParams,
    MertonParams,
    BatesParams,
    GARCHParams,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # =========================================================================
    # Portfolio Management (Primary API)
    # =========================================================================
    "OptionsPortfolio",
    "OptionPosition",
    "StockPosition",
    "OptionType",
    "PositionType",
    "GreeksResult",
    "GreeksCalculator",
    "AnalyticalGreeksStrategy",
    "FiniteDiffGreeksStrategy",
    "BreakevenResult",
    "BreakevenCalculator",
    "find_breakevens",
    # =========================================================================
    # Utilities
    # =========================================================================
    "norm_cdf",
    "norm_pdf",
    "norm_cdf_vec",
    "norm_pdf_vec",
    "d1_d2",
    # =========================================================================
    # Option Pricers
    # =========================================================================
    "BasePricer",
    "PricingResult",
    "PricingMethod",
    # Black-Scholes
    "BlackScholesPricer",
    "GBMPricer",
    "bs_call_price",
    "bs_put_price",
    "bs_greeks",
    "implied_volatility",
    # Heston
    "HestonPricer",
    "heston_call_price",
    "heston_put_price",
    # Bates
    "BatesPricer",
    "bates_call_price",
    "bates_put_price",
    # Merton
    "MertonPricer",
    "merton_call_price",
    "merton_put_price",
    # GARCH
    "GARCHMCPricer",
    "GARCHType",
    # Engines
    "CarrMadanFFTEngine",
    "FFTConfig",
    "MonteCarloEngine",
    "MCConfig",
    # =========================================================================
    # Simulation
    # =========================================================================
    "BaseSimulator",
    "SimulationResult",
    "ModelType",
    "DiscretizationScheme",
    "Measure",
    "GBMSimulator",
    "HestonSimulator",
    "MertonSimulator",
    "BatesSimulator",
    "GARCHSimulator",
    "NGARCHSimulator",
    "GJRGARCHSimulator",
    "create_simulator",
    "RiskMetrics",
    "compute_risk_metrics",
    # =========================================================================
    # Unified Models
    # =========================================================================
    "BaseModel",
    "PricingCapability",
    "registry",
    "GBMModel",
    "HestonModel",
    "MertonModel",
    "BatesModel",
    "GARCHModel",
    "NGARCHModel",
    "GJRGARCHModel",
    "GBMParams",
    "HestonParams",
    "MertonParams",
    "BatesParams",
    "GARCHParams",
]

__version__ = "4.0.0"
