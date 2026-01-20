"""
GBM Model
=========

Unified Geometric Brownian Motion model.

Model:
    dS = mu * S * dt + sigma * S * dW

Author: Derivatives Pricing Project
"""

from typing import Optional, List
import sys
from pathlib import Path

# Handle imports for both package and direct execution
try:
    from .base import BaseModel, PricingCapability
    from .parameters.gbm import GBMParams
except ImportError:
    _project_root = Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from backend.models.base import BaseModel, PricingCapability
    from backend.models.parameters.gbm import GBMParams

try:
    from backend.simulation.models.gbm import GBMSimulator
    from backend.option_pricing.black_scholes import BlackScholesPricer
except ImportError:
    from backend.simulation.models.gbm import GBMSimulator
    from backend.option_pricing.black_scholes import BlackScholesPricer


class GBMModel(BaseModel[GBMParams]):
    """
    Geometric Brownian Motion (Black-Scholes) Model.

    Single source of truth for GBM configuration.
    Can create both simulators (for path generation) and
    pricers (Black-Scholes analytical formulas).

    Parameters
    ----------
    params : GBMParams
        Model parameters (sigma)

    Example
    -------
    model = GBMModel.from_params(sigma=0.20)
    simulator = model.create_simulator()
    pricer = model.create_pricer()

    # Simulate paths
    result = simulator.simulate_paths(s0=100, mu=0.08, t=1.0, n_paths=10000, n_steps=252)

    # Price option
    price = pricer.price(s0=100, k=105, t=1.0, r=0.05, option_type="call")
    """

    def __init__(self, params: GBMParams):
        super().__init__(params)

    @classmethod
    def from_params(cls, sigma: float) -> "GBMModel":
        """Create model from volatility."""
        return cls(GBMParams(sigma=sigma))

    @property
    def model_name(self) -> str:
        return "Geometric Brownian Motion"

    @property
    def supported_pricing_methods(self) -> List[PricingCapability]:
        return [PricingCapability.ANALYTICAL, PricingCapability.MONTE_CARLO]

    def create_simulator(self, **kwargs) -> GBMSimulator:
        """
        Create GBM simulator.

        Parameters
        ----------
        **kwargs
            Additional simulator options

        Returns
        -------
        GBMSimulator
            Configured simulator
        """
        return GBMSimulator(sigma=self.params.sigma)

    def create_pricer(
        self,
        method: Optional[PricingCapability] = None,
        n_paths: int = 100000,
        n_steps: int = 252,
        **kwargs
    ):
        """
        Create GBM pricer.

        Parameters
        ----------
        method : PricingCapability, optional
            ANALYTICAL (default, Black-Scholes) or MONTE_CARLO
        n_paths : int
            Number of MC paths (only for MONTE_CARLO)
        n_steps : int
            Number of time steps (only for MONTE_CARLO)
        **kwargs
            Additional options

        Returns
        -------
        BasePricer
            BlackScholesPricer (analytical) or GBMMCPricer (Monte Carlo)

        Notes
        -----
        The risk-free rate is passed at pricing time, not construction time.
        """
        method = method or PricingCapability.ANALYTICAL

        if method == PricingCapability.ANALYTICAL:
            return BlackScholesPricer(sigma=self.params.sigma)
        elif method == PricingCapability.MONTE_CARLO:
            return GBMMCPricer(
                model=self,
                n_paths=n_paths,
                n_steps=n_steps,
            )
        else:
            raise ValueError(f"GBM does not support {method}")


# =============================================================================
# GBM Monte Carlo Pricer
# =============================================================================

class GBMMCPricer:
    """
    GBM option pricer using Monte Carlo simulation.

    This pricer uses the GBMSimulator to generate paths under
    the risk-neutral measure and prices options via discounted payoffs.

    Parameters
    ----------
    model : GBMModel
        The GBM model instance
    n_paths : int
        Number of Monte Carlo paths
    n_steps : int
        Number of time steps per path

    Notes
    -----
    Put-call parity (C - P = S - K*e^(-rT)) may not be exactly satisfied when
    calls and puts are priced with different random seeds due to Monte Carlo
    sampling error. To enforce parity, price the call and compute the put via:
    put_price = call_price - s0 + k * exp(-r * t)

    Examples
    --------
    model = GBMModel.from_params(sigma=0.20)
    pricer = model.create_pricer(method=PricingCapability.MONTE_CARLO, n_paths=50000)
    result = pricer.price(s0=100, k=100, t=0.25, r=0.05)
    """

    def __init__(self, model: GBMModel, n_paths: int = 100000, n_steps: int = 252):
        self._model = model
        self._n_paths = n_paths
        self._n_steps = n_steps
        self._model_name = "GBM (Monte Carlo)"

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_parameters(self):
        return self._model.get_parameters()

    def price(self, s0: float, k: float, t: float, r: float, option_type: str = "call", seed=None):
        """
        Price a European option using Monte Carlo simulation.

        Parameters
        ----------
        s0 : float
            Spot price
        k : float
            Strike price
        t : float
            Time to maturity
        r : float
            Risk-free rate
        option_type : str
            'call' or 'put'
        seed : int, optional
            Random seed

        Returns
        -------
        PricingResult
            Result with price, std_error, computation_time
        """
        import time
        import numpy as np
        from backend.option_pricing.base import PricingResult, PricingMethod

        start_time = time.perf_counter()

        # Create simulator and simulate under risk-neutral measure (r instead of mu)
        simulator = self._model.create_simulator()
        terminals = simulator.simulate_terminal(
            s0=s0, mu=r, t=t,
            n_paths=self._n_paths, n_steps=self._n_steps, seed=seed
        )

        # Compute payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(terminals - k, 0.0)
        else:
            payoffs = np.maximum(k - terminals, 0.0)

        # Discounted expected value
        discount = np.exp(-r * t)
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(self._n_paths)

        computation_time = time.perf_counter() - start_time

        return PricingResult(
            price=price,
            method=PricingMethod.MONTE_CARLO,
            computation_time=computation_time,
            std_error=std_error,
            n_paths=self._n_paths,
            parameters=self.get_parameters() | {"s0": s0, "k": k, "t": t, "r": r}
        )


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == "__main__":
    import time
    import numpy as np

    print("=" * 60)
    print("GBM Unified Model Benchmark")
    print("=" * 60)

    # Test parameters
    sigma = 0.20
    s0, k, t, r = 100.0, 100.0, 0.25, 0.05

    # Create model
    print("\n1. Creating GBMModel")
    print("-" * 40)
    model = GBMModel.from_params(sigma=sigma)
    print(f"Model: {model}")
    print(f"Parameters: {model.get_parameters()}")

    # Test simulator
    print("\n2. Simulator Test")
    print("-" * 40)
    simulator = model.create_simulator()
    print(f"Simulator: {type(simulator).__name__}")

    start = time.perf_counter()
    result = simulator.simulate_paths(s0=s0, mu=0.08, t=1.0, n_paths=10000, n_steps=252)
    elapsed = time.perf_counter() - start
    print(f"Simulated 10,000 paths in {elapsed*1000:.2f} ms")
    print(f"Final price mean: ${result.price_paths[:, -1].mean():.2f}")
    print(f"Final price std: ${result.price_paths[:, -1].std():.2f}")

    # Test Black-Scholes pricer
    print("\n3. Black-Scholes Pricer Test")
    print("-" * 40)
    pricer = model.create_pricer()

    start = time.perf_counter()
    call_result = pricer.price(s0=s0, k=k, t=t, r=r, option_type="call")
    put_result = pricer.price(s0=s0, k=k, t=t, r=r, option_type="put")
    elapsed = time.perf_counter() - start

    call_price = call_result.price
    put_price = put_result.price
    print(f"Call Price: ${call_price:.4f}")
    print(f"Put Price: ${put_price:.4f}")
    print(f"Computation Time: {elapsed*1000:.4f} ms")
    print(f"Greeks (call): delta={call_result.delta:.4f}, gamma={call_result.gamma:.4f}, vega={call_result.vega:.4f}")

    # Verify put-call parity
    print("\n4. Put-Call Parity Check")
    print("-" * 40)
    parity_lhs = call_price - put_price
    parity_rhs = s0 - k * np.exp(-r * t)
    print(f"C - P = ${parity_lhs:.4f}")
    print(f"S - K*e^(-rT) = ${parity_rhs:.4f}")
    print(f"Difference: ${abs(parity_lhs - parity_rhs):.8f}")

    # Monte Carlo pricer
    print("\n5. Monte Carlo Pricer Test")
    print("-" * 40)
    pricer_mc = model.create_pricer(method=PricingCapability.MONTE_CARLO, n_paths=100000)
    result_mc = pricer_mc.price(s0=s0, k=k, t=t, r=r)
    print(f"MC Call Price (100,000 paths): ${result_mc.price:.4f} +/- ${result_mc.std_error:.4f}")
    print(f"MC Time: {result_mc.computation_time*1000:.1f} ms")

    # Method comparison
    print("\n6. Method Comparison")
    print("-" * 40)
    print(f"Black-Scholes (analytical): ${call_price:.4f}")
    print(f"Monte Carlo (simulated):    ${result_mc.price:.4f}")
    print(f"Difference: ${abs(call_price - result_mc.price):.4f}")
    print(f"BS is ~{result_mc.computation_time/elapsed:.0f}x faster")

    print()
