"""
Finite Difference Engine
========================

PDE-based option pricing using finite difference methods.

Supports:
- European and American options
- Implicit, explicit, and Crank-Nicolson schemes
- Various boundary conditions

Author: Thomas
Created: 2025
"""

from typing import Optional, Union
from enum import Enum
import numpy as np

from backend.core.interfaces import PricingEngine, Model, Instrument
from backend.core.market import MarketEnvironment
from backend.core.result_types import PricingResult


# =============================================================================
# FD Schemes
# =============================================================================

class FDScheme(Enum):
    """Finite difference scheme."""
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    CRANK_NICOLSON = "crank_nicolson"


class BoundaryCondition(Enum):
    """Boundary condition type."""
    DIRICHLET = "dirichlet"  # Fixed value
    NEUMANN = "neumann"      # Fixed derivative
    LINEAR = "linear"        # Linear extrapolation


# =============================================================================
# Finite Difference Engine (Stub)
# =============================================================================

class FDEngine(PricingEngine):
    """
    Finite difference pricing engine.

    STUB: This engine is not yet implemented.

    Future implementation will support:
    - Explicit, Implicit, Crank-Nicolson schemes
    - American option pricing via PSOR
    - Barrier options
    - Local volatility models

    Parameters
    ----------
    n_spots : int
        Number of spot grid points (default 200)
    n_times : int
        Number of time steps (default 200)
    spot_min_mult : float
        Spot grid minimum as fraction of strike (default 0.2)
    spot_max_mult : float
        Spot grid maximum as multiple of strike (default 3.0)
    scheme : FDScheme
        Finite difference scheme (default Crank-Nicolson)
    """

    def __init__(
        self,
        n_spots: int = 200,
        n_times: int = 200,
        spot_min_mult: float = 0.2,
        spot_max_mult: float = 3.0,
        scheme: Union[str, FDScheme] = FDScheme.CRANK_NICOLSON
    ):
        self.n_spots = n_spots
        self.n_times = n_times
        self.spot_min_mult = spot_min_mult
        self.spot_max_mult = spot_max_mult

        if isinstance(scheme, str):
            scheme = FDScheme(scheme.lower())
        self.scheme = scheme

    def price(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment
    ) -> PricingResult:
        """
        Price an instrument using finite differences.

        STUB: Not yet implemented.

        Parameters
        ----------
        instrument : Instrument
            Option to price
        model : Model
            Pricing model
        market : MarketEnvironment
            Market environment

        Returns
        -------
        PricingResult
            Pricing result

        Raises
        ------
        NotImplementedError
            Always (stub implementation)
        """
        raise NotImplementedError(
            "Finite difference engine not yet implemented. "
            "This is a placeholder for future development. "
            "Use BSAnalyticEngine for European options or "
            "MonteCarloEngine for American options."
        )

    def _build_spot_grid(self, strike: float) -> np.ndarray:
        """Build the spatial grid for spot prices."""
        s_min = strike * self.spot_min_mult
        s_max = strike * self.spot_max_mult
        return np.linspace(s_min, s_max, self.n_spots)

    def _build_time_grid(self, maturity: float) -> np.ndarray:
        """Build the time grid."""
        return np.linspace(0, maturity, self.n_times + 1)

    def __repr__(self) -> str:
        return (
            f"FDEngine(scheme={self.scheme.value}, "
            f"n_spots={self.n_spots}, n_times={self.n_times})"
        )


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Finite Difference Engine Smoke Test")
    print("=" * 50)

    # Create engine
    engine = FDEngine(
        n_spots=100,
        n_times=100,
        scheme="crank_nicolson"
    )
    print(f"\nCreated: {engine}")

    # Test that pricing raises NotImplementedError
    from backend.instruments.options import VanillaOption
    from backend.models.gbm import GBMModel
    from backend.core.market import MarketEnvironment

    option = VanillaOption(strike=100, maturity=0.5, is_call=True)
    model = GBMModel(sigma=0.20)
    market = MarketEnvironment(spot=100, rate=0.05)

    print("\nTesting price() method...")
    try:
        engine.price(option, model, market)
        print("  ERROR: Should have raised NotImplementedError")
    except NotImplementedError as e:
        print(f"  Correctly raised NotImplementedError")

    # Test grid construction
    print("\nTesting grid construction...")
    spot_grid = engine._build_spot_grid(strike=100)
    time_grid = engine._build_time_grid(maturity=0.5)
    print(f"  Spot grid: {len(spot_grid)} points, [{spot_grid[0]:.1f}, {spot_grid[-1]:.1f}]")
    print(f"  Time grid: {len(time_grid)} points, [{time_grid[0]:.3f}, {time_grid[-1]:.3f}]")

    print("\n" + "=" * 50)
    print("FD Engine stub verification complete")
    print("=" * 50)
