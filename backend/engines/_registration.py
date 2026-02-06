"""
Engine Registration
===================

Auto-registration of engines with the EngineRegistry.

This module is imported by backend/engines/__init__.py to ensure
all engines are registered when the engines package is imported.

Author: Thomas
Created: 2025
"""

from backend.core.registry import EngineRegistry
from backend.core.result_types import PricingCapability


def register_all_engines():
    """
    Register all engines with the EngineRegistry.

    This function registers each engine for each model it supports.
    Called automatically when the engines package is imported.
    """
    # Import engines (deferred to avoid circular imports)
    from backend.engines.analytic_engine import BSAnalyticEngine
    from backend.engines.fft_engine import FFTEngine
    from backend.engines.mc_engine import MonteCarloEngine

    # GBM Model (Geometric Brownian Motion)
    EngineRegistry.register(
        "Geometric Brownian Motion",
        PricingCapability.ANALYTICAL,
        BSAnalyticEngine
    )
    EngineRegistry.register(
        "Geometric Brownian Motion",
        PricingCapability.FFT,
        FFTEngine
    )
    EngineRegistry.register(
        "Geometric Brownian Motion",
        PricingCapability.MONTE_CARLO,
        MonteCarloEngine
    )

    # Heston Model
    EngineRegistry.register(
        "Heston Stochastic Volatility",
        PricingCapability.FFT,
        FFTEngine
    )
    EngineRegistry.register(
        "Heston Stochastic Volatility",
        PricingCapability.MONTE_CARLO,
        MonteCarloEngine
    )

    # Bates Model (Heston + Jumps)
    EngineRegistry.register(
        "Bates (Heston + Jumps)",
        PricingCapability.FFT,
        FFTEngine
    )
    EngineRegistry.register(
        "Bates (Heston + Jumps)",
        PricingCapability.MONTE_CARLO,
        MonteCarloEngine
    )

    # Merton Jump-Diffusion Model
    EngineRegistry.register(
        "Merton Jump-Diffusion",
        PricingCapability.FFT,
        FFTEngine
    )
    EngineRegistry.register(
        "Merton Jump-Diffusion",
        PricingCapability.MONTE_CARLO,
        MonteCarloEngine
    )

    # GARCH Family Models (Monte Carlo only)
    EngineRegistry.register(
        "GARCH(1,1)",
        PricingCapability.MONTE_CARLO,
        MonteCarloEngine
    )
    EngineRegistry.register(
        "NGARCH (Nonlinear Asymmetric)",
        PricingCapability.MONTE_CARLO,
        MonteCarloEngine
    )
    EngineRegistry.register(
        "GJR-GARCH",
        PricingCapability.MONTE_CARLO,
        MonteCarloEngine
    )


# Track if registration has been done
_registered = False


def ensure_registered():
    """
    Ensure engines are registered (idempotent).

    This function can be called multiple times safely.
    Only the first call will perform registration.
    """
    global _registered
    if not _registered:
        register_all_engines()
        _registered = True
