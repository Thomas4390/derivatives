"""
Simulation Models Package
=========================

This package contains all simulation model implementations.

Models:
- GBMSimulator: Geometric Brownian Motion
- HestonSimulator: Heston Stochastic Volatility
- MertonSimulator: Merton Jump Diffusion
- BatesSimulator: Bates (Heston + Jumps)
- GARCHSimulator: GARCH(1,1)
- NGARCHSimulator: NGARCH
- GJRGARCHSimulator: GJR-GARCH

Author: Thomas
Created: 2025
"""


def __getattr__(name: str):
    """Lazy import to avoid conflicts when running modules directly."""
    if name == "GBMSimulator":
        from backend.simulation.models.gbm import GBMSimulator
        return GBMSimulator
    elif name == "HestonSimulator":
        from backend.simulation.models.heston import HestonSimulator
        return HestonSimulator
    elif name == "MertonSimulator":
        from backend.simulation.models.merton import MertonSimulator
        return MertonSimulator
    elif name == "BatesSimulator":
        from backend.simulation.models.bates import BatesSimulator
        return BatesSimulator
    elif name == "GARCHSimulator":
        from backend.simulation.models.garch import GARCHSimulator
        return GARCHSimulator
    elif name == "NGARCHSimulator":
        from backend.simulation.models.ngarch import NGARCHSimulator
        return NGARCHSimulator
    elif name == "GJRGARCHSimulator":
        from backend.simulation.models.gjr_garch import GJRGARCHSimulator
        return GJRGARCHSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GBMSimulator",
    "HestonSimulator",
    "MertonSimulator",
    "BatesSimulator",
    "GARCHSimulator",
    "NGARCHSimulator",
    "GJRGARCHSimulator",
]
