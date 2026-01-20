"""
Merton Jump-Diffusion Model
===========================

Unified Merton (1976) jump-diffusion model.

Model:
    dS = (mu - lambda*k) * S * dt + sigma * S * dW + (J - 1) * S * dN

Where:
    - dN is Poisson with intensity lambda
    - J is lognormal: ln(J) ~ N(mu_j, sigma_j^2)

Author: Derivatives Pricing Project
"""

from typing import Optional, List
import sys
from pathlib import Path

# Handle imports for both package and direct execution
try:
    from .base import BaseModel, PricingCapability
    from .parameters.merton import MertonParams
except ImportError:
    _project_root = Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from backend.models.base import BaseModel, PricingCapability
    from backend.models.parameters.merton import MertonParams

try:
    from backend.simulation.models.merton import MertonSimulator
    from backend.option_pricing.garch_mc import GARCHMCPricer
except ImportError:
    from backend.simulation.models.merton import MertonSimulator
    from backend.option_pricing.garch_mc import GARCHMCPricer


class MertonModel(BaseModel[MertonParams]):
    """
    Merton (1976) Jump-Diffusion Model.

    Single source of truth for Merton configuration.
    Combines GBM with Poisson-driven lognormal jumps.

    Parameters
    ----------
    params : MertonParams
        Model parameters (sigma, lambda_j, mu_j, sigma_j)

    Example
    -------
    model = MertonModel.from_params(sigma=0.20, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)
    simulator = model.create_simulator()
    pricer = model.create_pricer(r=0.05)  # MC pricing
    """

    def __init__(self, params: MertonParams):
        super().__init__(params)

    @classmethod
    def from_params(
        cls,
        sigma: float,
        lambda_j: float,
        mu_j: float,
        sigma_j: float
    ) -> "MertonModel":
        """Create model from individual parameters."""
        return cls(MertonParams(
            sigma=sigma,
            lambda_j=lambda_j,
            mu_j=mu_j,
            sigma_j=sigma_j
        ))

    @property
    def model_name(self) -> str:
        return "Merton Jump-Diffusion"

    @property
    def supported_pricing_methods(self) -> List[PricingCapability]:
        return [PricingCapability.MONTE_CARLO]

    def create_simulator(self, **kwargs) -> MertonSimulator:
        """
        Create Merton simulator.

        Returns
        -------
        MertonSimulator
            Configured simulator
        """
        return MertonSimulator(
            sigma=self.params.sigma,
            lambda_j=self.params.lambda_j,
            mu_j=self.params.mu_j,
            sigma_j=self.params.sigma_j,
        )

    def create_pricer(
        self,
        method: Optional[PricingCapability] = None,
        n_paths: int = 100000,
        n_steps: int = 252,
        **kwargs
    ):
        """
        Create Merton pricer (Monte Carlo).

        Parameters
        ----------
        method : PricingCapability
            Only MONTE_CARLO supported
        n_paths : int
            Number of MC paths
        n_steps : int
            Number of time steps

        Returns
        -------
        MertonMCPricer
            MC-based pricer using simulation

        Notes
        -----
        The risk-free rate is passed at pricing time, not construction time.
        """
        method = method or PricingCapability.MONTE_CARLO

        if method != PricingCapability.MONTE_CARLO:
            raise ValueError(f"Merton only supports MONTE_CARLO pricing, got {method}")

        return MertonMCPricer(
            model=self,
            n_paths=n_paths,
            n_steps=n_steps,
        )


# =============================================================================
# Merton Monte Carlo Pricer
# =============================================================================

class MertonMCPricer:
    """
    Merton option pricer using Monte Carlo simulation.

    This pricer uses the MertonSimulator to generate paths under
    the risk-neutral measure and prices options via discounted payoffs.

    Parameters
    ----------
    model : MertonModel
        The Merton model instance
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
    model = MertonModel.from_params(sigma=0.20, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)
    pricer = model.create_pricer(n_paths=50000)
    result = pricer.price(s0=100, k=100, t=0.25, r=0.05)
    """

    def __init__(self, model: "MertonModel", n_paths: int = 100000, n_steps: int = 252):
        self._model = model
        self._n_paths = n_paths
        self._n_steps = n_steps
        self._model_name = "Merton (Monte Carlo)"

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
    print("Merton Jump-Diffusion Unified Model Benchmark")
    print("=" * 60)

    # Test parameters
    sigma = 0.20
    lambda_j, mu_j, sigma_j = 0.5, -0.1, 0.2  # Jump parameters
    s0, k, t, r = 100.0, 100.0, 0.25, 0.05

    # Create model
    print("\n1. Creating MertonModel")
    print("-" * 40)
    model = MertonModel.from_params(
        sigma=sigma, lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j
    )
    print(f"Model: {model}")
    print(f"Parameters: {model.get_parameters()}")
    print(f"Jump intensity: {lambda_j} jumps/year")
    print(f"Mean jump size: {mu_j} ({(np.exp(mu_j) - 1)*100:.1f}%)")

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

    # Test Monte Carlo pricer
    print("\n3. Monte Carlo Pricer Test")
    print("-" * 40)
    pricer = model.create_pricer(n_paths=50000)
    result_call = pricer.price(s0=s0, k=k, t=t, r=r, option_type="call", seed=42)
    result_put = pricer.price(s0=s0, k=k, t=t, r=r, option_type="put", seed=42)

    print(f"MC Call Price (50,000 paths): ${result_call.price:.4f} +/- ${result_call.std_error:.4f}")
    print(f"MC Put Price: ${result_put.price:.4f} +/- ${result_put.std_error:.4f}")
    print(f"MC Time (call): {result_call.computation_time*1000:.1f} ms")

    # Compare Merton vs GBM (effect of jumps)
    print("\n4. Jump Impact Analysis")
    print("-" * 40)
    from backend.models.gbm import GBMModel
    gbm_model = GBMModel.from_params(sigma=sigma)
    gbm_pricer = gbm_model.create_pricer()
    gbm_call = gbm_pricer.price(s0=s0, k=k, t=t, r=r, option_type="call").price

    print(f"GBM Call Price (BS): ${gbm_call:.4f}")
    print(f"Merton MC Call Price: ${result_call.price:.4f}")
    print(f"Difference (jump impact): ${result_call.price - gbm_call:.4f}")

    # Put-call parity check for MC
    print("\n5. Put-Call Parity Check (MC)")
    print("-" * 40)
    parity_lhs = result_call.price - result_put.price
    parity_rhs = s0 - k * np.exp(-r * t)
    print(f"C - P (MC) = ${parity_lhs:.4f}")
    print(f"S - K*e^(-rT) = ${parity_rhs:.4f}")
    print(f"Parity error: ${abs(parity_lhs - parity_rhs):.4f}")

    print()
