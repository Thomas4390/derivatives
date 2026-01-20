"""
Bates Model
===========

Unified Bates (1996) model combining Heston stochastic volatility
with Merton-style jumps.

Model:
    dS = (mu - lambda*k) * S * dt + sqrt(V) * S * dW_S + (J - 1) * S * dN
    dV = kappa * (theta - V) * dt + xi * sqrt(V) * dW_V
    Corr(dW_S, dW_V) = rho

Author: Derivatives Pricing Project
"""

from typing import Optional, List
import sys
from pathlib import Path

# Handle imports for both package and direct execution
try:
    from .base import BaseModel, PricingCapability
    from .parameters.bates import BatesParams
    from .characteristic_functions.bates_cf import (
        bates_characteristic_function,
        bates_cf_vectorized,
    )
except ImportError:
    _project_root = Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from backend.models.base import BaseModel, PricingCapability
    from backend.models.parameters.bates import BatesParams
    from backend.models.characteristic_functions.bates_cf import (
        bates_characteristic_function,
        bates_cf_vectorized,
    )

from backend.simulation.models.bates import BatesSimulator


class BatesModel(BaseModel[BatesParams]):
    """
    Bates (1996) Stochastic Volatility with Jumps Model.

    Combines Heston stochastic volatility with Merton-style jumps.
    Single source of truth for Bates configuration.

    Parameters
    ----------
    params : BatesParams
        Model parameters (v0, kappa, theta, xi, rho, lambda_j, mu_j, sigma_j)

    Example
    -------
    model = BatesModel.from_params(
        v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
    )
    simulator = model.create_simulator()
    pricer = model.create_pricer(r=0.05)
    """

    def __init__(self, params: BatesParams):
        super().__init__(params)

    @classmethod
    def from_params(
        cls,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        lambda_j: float,
        mu_j: float,
        sigma_j: float
    ) -> "BatesModel":
        """Create model from individual parameters."""
        return cls(BatesParams(
            v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
            lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j
        ))

    @property
    def model_name(self) -> str:
        return "Bates (Heston + Jumps)"

    @property
    def supported_pricing_methods(self) -> List[PricingCapability]:
        return [PricingCapability.FFT, PricingCapability.MONTE_CARLO]

    def create_simulator(self, **kwargs) -> BatesSimulator:
        """
        Create Bates simulator.

        Returns
        -------
        BatesSimulator
            Configured simulator
        """
        return BatesSimulator(
            v0=self.params.v0,
            kappa=self.params.kappa,
            theta=self.params.theta,
            xi=self.params.xi,
            rho=self.params.rho,
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
        Create Bates pricer.

        Parameters
        ----------
        method : PricingCapability
            FFT (default, fast) or MONTE_CARLO (slower but more flexible)
        n_paths : int
            Number of MC paths (only for MONTE_CARLO method)
        n_steps : int
            Number of time steps (only for MONTE_CARLO method)
        **kwargs
            Additional pricer options (alpha, n_fft for FFT)

        Returns
        -------
        BasePricer
            Configured pricer (BatesPricer or BatesMCPricer)

        Notes
        -----
        The risk-free rate is passed at pricing time, not construction time.
        """
        method = method or PricingCapability.FFT

        if method == PricingCapability.FFT:
            # Lazy import to avoid circular dependency
            from backend.option_pricing.bates import BatesPricer
            return BatesPricer(
                v0=self.params.v0,
                kappa=self.params.kappa,
                theta=self.params.theta,
                xi=self.params.xi,
                rho=self.params.rho,
                lambda_j=self.params.lambda_j,
                mu_j=self.params.mu_j,
                sigma_j=self.params.sigma_j,
                **kwargs
            )
        elif method == PricingCapability.MONTE_CARLO:
            return BatesMCPricer(
                model=self,
                n_paths=n_paths,
                n_steps=n_steps,
            )
        else:
            raise ValueError(f"Bates does not support {method}")

    def characteristic_function(self, u, s0: float, t: float, r: float):
        """
        Bates characteristic function.

        Uses shared implementation from characteristic_functions module.
        """
        return bates_characteristic_function(
            u, s0, self.params.v0, t, r,
            self.params.kappa, self.params.theta,
            self.params.xi, self.params.rho,
            self.params.lambda_j, self.params.mu_j, self.params.sigma_j
        )

    def characteristic_function_vectorized(self, u_arr, s0: float, t: float, r: float):
        """Vectorized characteristic function for FFT."""
        return bates_cf_vectorized(
            u_arr, s0, self.params.v0, t, r,
            self.params.kappa, self.params.theta,
            self.params.xi, self.params.rho,
            self.params.lambda_j, self.params.mu_j, self.params.sigma_j
        )


# =============================================================================
# Bates Monte Carlo Pricer
# =============================================================================

class BatesMCPricer:
    """
    Bates option pricer using Monte Carlo simulation.

    This pricer uses the BatesSimulator to generate paths under
    the risk-neutral measure and prices options via discounted payoffs.

    Parameters
    ----------
    model : BatesModel
        The Bates model instance
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
    model = BatesModel.from_params(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                                   lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)
    pricer = model.create_pricer(method=PricingCapability.MONTE_CARLO, n_paths=50000)
    result = pricer.price(s0=100, k=100, t=0.25, r=0.05)
    """

    def __init__(self, model: BatesModel, n_paths: int = 100000, n_steps: int = 252):
        self._model = model
        self._n_paths = n_paths
        self._n_steps = n_steps
        self._model_name = "Bates (Monte Carlo)"

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
    print("Bates Unified Model Benchmark")
    print("=" * 60)

    # Test parameters (Heston + jumps)
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7
    lambda_j, mu_j, sigma_j = 0.5, -0.1, 0.2  # Jump parameters
    s0, k, t, r = 100.0, 100.0, 0.25, 0.05

    # Create model
    print("\n1. Creating BatesModel")
    print("-" * 40)
    model = BatesModel.from_params(
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j
    )
    print(f"Model: {model}")
    print(f"Feller satisfied: {model.params.feller_satisfied}")
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

    # Test FFT pricer
    print("\n3. FFT Pricer Test")
    print("-" * 40)
    pricer_fft = model.create_pricer(method=PricingCapability.FFT)
    result_fft = pricer_fft.price(s0=s0, k=k, t=t, r=r)
    print(f"FFT Call Price: ${result_fft.price:.4f}")
    print(f"FFT Time: {result_fft.computation_time*1000:.2f} ms")

    # Monte Carlo pricer
    print("\n4. Monte Carlo Pricer Test")
    print("-" * 40)
    pricer_mc = model.create_pricer(method=PricingCapability.MONTE_CARLO, n_paths=50000)
    result_mc = pricer_mc.price(s0=s0, k=k, t=t, r=r, seed=42)
    print(f"MC Call Price (50,000 paths): ${result_mc.price:.4f} +/- ${result_mc.std_error:.4f}")
    print(f"MC Time: {result_mc.computation_time*1000:.1f} ms")
    print(f"FFT vs MC difference: ${abs(result_fft.price - result_mc.price):.4f}")

    # Characteristic function test
    print("\n5. Characteristic Function Test")
    print("-" * 40)
    u = 1.0 + 0.5j
    cf = model.characteristic_function(u, s0=s0, t=t, r=r)
    print(f"CF(1+0.5i) = {cf:.6f}")

    # Compare Bates vs Heston (effect of jumps)
    print("\n6. Jump Impact Analysis")
    print("-" * 40)
    from backend.models.heston import HestonModel
    heston_model = HestonModel.from_params(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
    heston_pricer = heston_model.create_pricer()
    heston_result = heston_pricer.price(s0=s0, k=k, t=t, r=r)
    print(f"Heston Call Price: ${heston_result.price:.4f}")
    print(f"Bates Call Price: ${result_fft.price:.4f}")
    print(f"Jump premium: ${result_fft.price - heston_result.price:.4f}")

    print()
