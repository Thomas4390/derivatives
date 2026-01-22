"""
Core Module
===========

Interfaces, types, and registry for the option pricing system.

Quick Start
-----------
    from backend.core import (
        # Market data
        MarketEnvironment,
        # Types and enums
        ExerciseStyle, PricingCapability,
        PricingResult, GreeksResult,
        # Interfaces (for implementing new components)
        Payoff, Instrument, Model, PricingEngine,
        # Registry
        EngineRegistry, price,
    )

Example
-------
    market = MarketEnvironment(spot=100, rate=0.05)
    bumped = market.bump_spot(5.0)  # spot = 105
    print(bumped.spot)  # 105.0

Author: Thomas
Created: 2025
"""

# Types and enums
from backend.core.result_types import (
    ExerciseStyle,
    PricingCapability,
    PricingResult,
    GreeksResult,
)

# Market environment
from backend.core.market import MarketEnvironment

# Interfaces (ABCs)
from backend.core.interfaces import (
    Payoff,
    Instrument,
    Model,
    PricingEngine,
)

# Registry and pricing function
from backend.core.registry import (
    EngineRegistry,
    price,
)

# Black-Scholes formulas - imported from utils (single source of truth)
from backend.utils import (
    norm_cdf,
    norm_pdf,
    d1_d2,
    bs_price,
    bs_delta,
    bs_gamma,
    bs_vega,
    bs_theta,
    bs_rho,
    bs_greeks,
)


__all__ = [
    # Types
    "ExerciseStyle",
    "PricingCapability",
    "PricingResult",
    "GreeksResult",
    # Market
    "MarketEnvironment",
    # Interfaces
    "Payoff",
    "Instrument",
    "Model",
    "PricingEngine",
    # Registry
    "EngineRegistry",
    "price",
    # Black-Scholes formulas
    "norm_cdf",
    "norm_pdf",
    "d1_d2",
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "bs_theta",
    "bs_rho",
    "bs_greeks",
]


if __name__ == "__main__":
    # Smoke test
    print("=" * 50)
    print("Core Module Smoke Test")
    print("=" * 50)

    # Test enums
    print(f"\nExerciseStyle: {list(ExerciseStyle)}")
    print(f"PricingCapability: {list(PricingCapability)}")

    # Test MarketEnvironment
    market = MarketEnvironment(spot=100.0, rate=0.05)
    bumped = market.bump_spot(5.0)
    print(f"\nMarket: spot={market.spot}, rate={market.rate}")
    print(f"Bumped: spot={bumped.spot}")

    # Test PricingResult
    result = PricingResult(price=5.0, engine="Test", model="GBM")
    print(f"\nPricingResult: {result}")

    # Test GreeksResult
    greeks = GreeksResult(delta=0.5, gamma=0.02, theta=-0.05, vega=0.2, rho=0.1)
    print(f"GreeksResult: {greeks}")

    # Test EngineRegistry
    print(f"\nEngineRegistry priority: {[c.name for c in EngineRegistry.PRIORITY]}")
    print(f"Registered engines: {EngineRegistry.list_engines()}")

    print("\n" + "=" * 50)
    print("✓ Core module smoke test passed")
    print("=" * 50)
