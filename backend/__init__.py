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

Author: Thomas Vaudescal
Created: 2026
Version: 5.0.0
"""

# =============================================================================
# Portfolio Management (NEW Architecture)
# =============================================================================

# =============================================================================
# New Architecture - Core
# =============================================================================
from backend.core import (
    ExerciseStyle,
    # Interfaces
    Instrument,
    # Market
    MarketEnvironment,
    Model,
    Payoff,
    PricingEngine,
    # Result types
    PricingResult,
)

# Structured product infrastructure
from backend.core.market import EnrichedMarketEnvironment, YieldCurve
from backend.core.result_types import StructuredProductResult
from backend.core.structured_product import (
    ObservationSchedule,
    ProductComponent,
    StructuredProduct,
)

# BaseModel is an alias for Model from core.interfaces
from backend.core.interfaces import Model as BaseModel

# =============================================================================
# New Architecture - Engines
# =============================================================================
from backend.engines import (
    BSAnalyticEngine,
    FFTEngine,
    MonteCarloEngine,
)

# =============================================================================
# Low-level Engines (for advanced use)
# =============================================================================
from backend.engines.fourier.carr_madan import (
    CarrMadanFFTEngine,
    FFTConfig,
)

# =============================================================================
# GARCH Pricer (Standalone, uses LRNVR)
# =============================================================================
from backend.engines.monte_carlo.garch_pricer import (
    GARCHMCPricer,
    GARCHPricingResult,
    GARCHType,
)
from backend.engines.monte_carlo.mc_base import (
    GenericMCEngine,
    MCConfig,
    MCResult,
)

# =============================================================================
# Greeks Calculation
# =============================================================================
from backend.greeks import (
    GreeksCalculator,
    bs_all_greeks,
    # Analytic Greeks
    bs_greeks_first_order,
    bs_greeks_second_order,
    bs_greeks_third_order,
    calculate_greeks,
    # Numerical Greeks
    finite_difference_greeks,
)

# =============================================================================
# New Architecture - Instruments
# =============================================================================
from backend.instruments import (
    AmericanCall,
    AmericanPut,
    Butterfly,
    CompositePayoff,
    DigitalOption,
    EuropeanCall,
    EuropeanPut,
    IronCondor,
    # Strategies
    OptionStrategy,
    Straddle,
    StrategyLeg,
    # Payoffs
    VanillaCallPayoff,
    # Options
    VanillaOption,
    VanillaPutPayoff,
)

# =============================================================================
# Unified Models
# =============================================================================
from backend.models import (
    BatesModel,
    GARCHModel,
    # GARCH params (aliases for backward compatibility)
    GARCHParams,
    # Models
    GBMModel,
    GJRGARCHModel,
    GJRGARCHParams,
    HestonModel,
    MertonModel,
    NGARCHModel,
    NGARCHParams,
    # Base
    PricingCapability,
    registry,
)
from backend.portfolio import (
    BreakevenCalculator,
    # Breakeven
    BreakevenResult,
    # Greeks (from core)
    GreeksResult,
    # Main portfolio class
    OptionsPortfolio,
    # Position classes
    PortfolioPosition,
    StockPosition,
    StructuredProductPosition,
    find_breakevens,
    find_breakevens_from_portfolio,
    # Factory functions
    long_call,
    long_put,
    long_stock,
    short_call,
    short_put,
    short_stock,
)

# =============================================================================
# Structured Products
# =============================================================================
from backend.instruments.structured import (
    # Components
    AutocallTrigger,
    BondFloor,
    ConditionalCoupon,
    FixedCoupon,
    KnockInPut,
    UpsideParticipation,
    # Products
    Autocallable,
    CapitalProtectedNote,
    ReverseConvertible,
)

# Structured product engine
from backend.engines.structured_mc_engine import StructuredProductMCEngine

# =============================================================================
# Simulation
# =============================================================================
from backend.simulation import (
    # Base
    BaseSimulator,
    BatesSimulator,
    DiscretizationScheme,
    GARCHSimulator,
    # Simulators
    GBMSimulator,
    GJRGARCHSimulator,
    HestonSimulator,
    Measure,
    MertonSimulator,
    # Enums
    ModelType,
    NGARCHSimulator,
    # P&L engine
    RiskMetrics,
    SimulationResult,
    compute_risk_metrics,
    # Factory
    create_simulator,
)

# =============================================================================
# Utilities
# =============================================================================
from backend.utils import (
    d1_d2,
    norm_cdf,
    norm_cdf_vec,
    norm_pdf,
    norm_pdf_vec,
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
    "DigitalOption",
    # Payoffs
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
    "StructuredProductMCEngine",
    # =========================================================================
    # Structured Products
    # =========================================================================
    # Core
    "StructuredProduct",
    "ProductComponent",
    "ObservationSchedule",
    "StructuredProductResult",
    "StructuredProductPosition",
    # Market extensions
    "YieldCurve",
    "EnrichedMarketEnvironment",
    # Components
    "BondFloor",
    "UpsideParticipation",
    "FixedCoupon",
    "ConditionalCoupon",
    "AutocallTrigger",
    "KnockInPut",
    # Products
    "Autocallable",
    "CapitalProtectedNote",
    "ReverseConvertible",
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
