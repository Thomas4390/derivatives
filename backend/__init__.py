"""
Backend Package - Unified API
=============================

High-performance derivatives pricing and portfolio management.

Three Pillars Architecture:
- Instrument (What): VanillaOption, OptionStrategy
- Model (Physics): GBMModel, HestonModel, BatesModel, MertonModel
- Engine (How): BSAnalyticEngine, FFTEngine, MonteCarloEngine

Main Components:
- Portfolio management: OptionsPortfolio, PortfolioPosition, StockPosition
- Pricing engines: BSAnalyticEngine, FFTEngine, MonteCarloEngine
- Models: GBMModel, HestonModel, BatesModel, MertonModel, GARCH variants
- Simulation: GBMSimulator, HestonSimulator, MertonSimulator, BatesSimulator

Author: Thomas
Created: 2025
Version: 5.0.0
"""

# =============================================================================
# Portfolio Management (NEW Architecture)
# =============================================================================

from backend.portfolio import (
    # Main portfolio class
    OptionsPortfolio,
    # Position classes
    PortfolioPosition,
    StockPosition,
    # Factory functions
    long_call,
    short_call,
    long_put,
    short_put,
    long_stock,
    short_stock,
    # Greeks (from core)
    GreeksResult,
    # Breakeven
    BreakevenResult,
    BreakevenCalculator,
    find_breakevens,
    find_breakevens_from_portfolio,
)

# =============================================================================
# Utilities
# =============================================================================

from backend.utils import (
    norm_cdf,
    norm_pdf,
    norm_cdf_vec,
    norm_pdf_vec,
    d1_d2,
)

# =============================================================================
# GARCH Pricer (Standalone, uses LRNVR)
# =============================================================================

from backend.engines.monte_carlo.garch_pricer import (
    GARCHMCPricer,
    GARCHType,
    GARCHPricingResult,
)

# =============================================================================
# Low-level Engines (for advanced use)
# =============================================================================

from backend.engines.fourier.carr_madan import (
    CarrMadanFFTEngine,
    FFTConfig,
)

from backend.engines.monte_carlo.mc_base import (
    GenericMCEngine,
    MCConfig,
    MCResult,
)

# =============================================================================
# Simulation
# =============================================================================

from backend.simulation import (
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

from backend.models import (
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
    # GARCH params (aliases for backward compatibility)
    GARCHParams,
    NGARCHParams,
    GJRGARCHParams,
)

# =============================================================================
# New Architecture - Core
# =============================================================================

from backend.core import (
    # Interfaces
    Instrument,
    Model,
    PricingEngine,
    Payoff,
    # Market
    MarketEnvironment,
    # Result types
    PricingResult,
    ExerciseStyle,
)

# =============================================================================
# New Architecture - Instruments
# =============================================================================

from backend.instruments import (
    # Options
    VanillaOption,
    EuropeanCall,
    EuropeanPut,
    AmericanCall,
    AmericanPut,
    # Payoffs
    VanillaCallPayoff,
    VanillaPutPayoff,
    CompositePayoff,
    # Strategies
    OptionStrategy,
    IronCondor,
    Straddle,
    Butterfly,
    StrategyLeg,
)

# =============================================================================
# New Architecture - Engines
# =============================================================================

from backend.engines import (
    BSAnalyticEngine,
    FFTEngine,
    MonteCarloEngine,
)

# =============================================================================
# Greeks Calculation
# =============================================================================

from backend.greeks import (
    GreeksCalculator,
    calculate_greeks,
    # Analytic Greeks
    bs_greeks_first_order,
    bs_greeks_second_order,
    bs_greeks_third_order,
    bs_all_greeks,
    # Numerical Greeks
    finite_difference_greeks,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # =========================================================================
    # Portfolio Management (New Architecture)
    # =========================================================================
    "OptionsPortfolio",
    "PortfolioPosition",
    "StockPosition",
    # Factory functions
    "long_call",
    "short_call",
    "long_put",
    "short_put",
    "long_stock",
    "short_stock",
    # Greeks & Breakeven
    "GreeksResult",
    "BreakevenResult",
    "BreakevenCalculator",
    "find_breakevens",
    "find_breakevens_from_portfolio",
    # =========================================================================
    # Utilities
    # =========================================================================
    "norm_cdf",
    "norm_pdf",
    "norm_cdf_vec",
    "norm_pdf_vec",
    "d1_d2",
    # =========================================================================
    # GARCH Pricer
    # =========================================================================
    "GARCHMCPricer",
    "GARCHType",
    "GARCHPricingResult",
    # =========================================================================
    # Low-level Engines
    # =========================================================================
    "CarrMadanFFTEngine",
    "FFTConfig",
    "GenericMCEngine",
    "MCConfig",
    "MCResult",
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
    # GARCH params (aliases for backward compatibility)
    "GARCHParams",
    "NGARCHParams",
    "GJRGARCHParams",
    # =========================================================================
    # Core Interfaces and Types
    # =========================================================================
    "Instrument",
    "Model",
    "PricingEngine",
    "Payoff",
    "MarketEnvironment",
    "PricingResult",
    "ExerciseStyle",
    # =========================================================================
    # Instruments
    # =========================================================================
    "VanillaOption",
    "EuropeanCall",
    "EuropeanPut",
    "AmericanCall",
    "AmericanPut",
    "VanillaCallPayoff",
    "VanillaPutPayoff",
    "CompositePayoff",
    "OptionStrategy",
    "IronCondor",
    "Straddle",
    "Butterfly",
    "StrategyLeg",
    # =========================================================================
    # Engines
    # =========================================================================
    "BSAnalyticEngine",
    "FFTEngine",
    "MonteCarloEngine",
    # =========================================================================
    # Greeks Calculation
    # =========================================================================
    "GreeksCalculator",
    "calculate_greeks",
    "bs_greeks_first_order",
    "bs_greeks_second_order",
    "bs_greeks_third_order",
    "bs_all_greeks",
    "finite_difference_greeks",
]

__version__ = "5.0.0"
