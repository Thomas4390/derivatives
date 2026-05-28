"""
Monte Carlo Pricing Engines
===========================

Monte Carlo simulation engines for option pricing.

Contains:
- GenericMCEngine: Low-level MC engine for terminal simulation
- MCConfig: Monte Carlo configuration dataclass
- MCResult: Monte Carlo result namedtuple
- GARCHMCPricer: GARCH family MC pricer with LRNVR

Note: The high-level MonteCarloEngine wrapper is in backend.engines.mc_engine
and is exported from backend.engines directly.

Author: Thomas Vaudescal
Created: 2026
"""

# Import low-level engine and config from mc_base
# Import GARCH pricer
from backend.engines.monte_carlo.garch_pricer import (
    GARCHMCPricer,
    GARCHPricingResult,
    GARCHType,
    create_garch_pricer,
    create_gjr_garch_pricer,
    create_ngarch_pricer,
)
from backend.engines.monte_carlo.mc_base import (
    GenericMCEngine,
    MCConfig,
    MCResult,
    mc_price,
)

__all__ = [
    "GenericMCEngine",
    "MCConfig",
    "MCResult",
    "mc_price",
    # GARCH
    "GARCHMCPricer",
    "GARCHType",
    "GARCHPricingResult",
    "create_garch_pricer",
    "create_ngarch_pricer",
    "create_gjr_garch_pricer",
]
