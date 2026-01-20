"""
Heston Model
============

Unified Heston (1993) stochastic volatility model.

Model:
    dS = mu * S * dt + sqrt(V) * S * dW_S
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
    from .parameters.heston import HestonParams
    from .characteristic_functions.heston_cf import (
        heston_characteristic_function,
        heston_cf_vectorized,
    )
except ImportError:
    _project_root = Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from backend.models.base import BaseModel, PricingCapability
    from backend.models.parameters.heston import HestonParams
    from backend.models.characteristic_functions.heston_cf import (
        heston_characteristic_function,
        heston_cf_vectorized,
    )

from backend.simulation.models.heston import HestonSimulator
from backend.simulation.enums import DiscretizationScheme


class HestonModel(BaseModel[HestonParams]):
    """
    Heston (1993) Stochastic Volatility Model.

    Single source of truth for Heston configuration.
    Can create both simulators and pricers.

    Parameters
    ----------
    params : HestonParams
        Model parameters (v0, kappa, theta, xi, rho)

    Example
    -------
    model = HestonModel.from_params(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    simulator = model.create_simulator(scheme="qe")
    pricer = model.create_pricer()

    # Same parameters used for both
    result = simulator.simulate_paths(s0=100, mu=0.08, t=1.0, n_paths=10000, n_steps=252)
    price = pricer.price(s0=100, k=105, t=1.0, r=0.05, option_type="call")
    """

    def __init__(self, params: HestonParams):
        super().__init__(params)

    @classmethod
    def from_params(
        cls,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float
    ) -> "HestonModel":
        """Create model from individual parameters."""
        return cls(HestonParams(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho))

    @property
    def model_name(self) -> str:
        return "Heston Stochastic Volatility"

    @property
    def supported_pricing_methods(self) -> List[PricingCapability]:
        return [PricingCapability.FFT, PricingCapability.MONTE_CARLO]

    def create_simulator(
        self,
        scheme: str = "full_truncation",
        **kwargs
    ) -> HestonSimulator:
        """
        Create Heston simulator.

        Parameters
        ----------
        scheme : str
            Discretization scheme: 'euler', 'full_truncation', 'reflection', 'qe'

        Returns
        -------
        HestonSimulator
            Configured simulator
        """
        scheme_map = {
            "euler": DiscretizationScheme.EULER,
            "full_truncation": DiscretizationScheme.FULL_TRUNCATION,
            "reflection": DiscretizationScheme.REFLECTION,
            "qe": DiscretizationScheme.QE,
        }

        return HestonSimulator(
            v0=self.params.v0,
            kappa=self.params.kappa,
            theta=self.params.theta,
            xi=self.params.xi,
            rho=self.params.rho,
            scheme=scheme_map.get(scheme.lower(), DiscretizationScheme.FULL_TRUNCATION),
        )

    def create_pricer(
        self,
        method: Optional[PricingCapability] = None,
        n_paths: int = 100000,
        n_steps: int = 252,
        **kwargs
    ):
        """
        Create Heston pricer.

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
            Configured pricer (HestonPricer or HestonMCPricer)

        Notes
        -----
        The risk-free rate is passed at pricing time, not construction time.

        Examples
        --------
        # FFT pricer (fast, default)
        pricer = model.create_pricer()  # or method=PricingCapability.FFT

        # Monte Carlo pricer (flexible)
        pricer = model.create_pricer(method=PricingCapability.MONTE_CARLO, n_paths=50000)
        """
        method = method or PricingCapability.FFT

        if method == PricingCapability.FFT:
            # Lazy import to avoid circular dependency
            from backend.option_pricing.heston import HestonPricer
            return HestonPricer(
                v0=self.params.v0,
                kappa=self.params.kappa,
                theta=self.params.theta,
                xi=self.params.xi,
                rho=self.params.rho,
                **kwargs
            )
        elif method == PricingCapability.MONTE_CARLO:
            # Monte Carlo pricer using simulator
            scheme = kwargs.pop("scheme", "full_truncation")
            return HestonMCPricer(
                model=self,
                n_paths=n_paths,
                n_steps=n_steps,
                scheme=scheme,
            )
        else:
            raise ValueError(f"Heston does not support {method}")

    def characteristic_function(self, u, s0: float, t: float, r: float):
        """
        Heston characteristic function.

        Uses shared implementation from characteristic_functions module.
        """
        return heston_characteristic_function(
            u, s0, self.params.v0, t, r,
            self.params.kappa, self.params.theta,
            self.params.xi, self.params.rho
        )

    def characteristic_function_vectorized(self, u_arr, s0: float, t: float, r: float):
        """Vectorized characteristic function for FFT."""
        return heston_cf_vectorized(
            u_arr, s0, self.params.v0, t, r,
            self.params.kappa, self.params.theta,
            self.params.xi, self.params.rho
        )


# =============================================================================
# Heston Monte Carlo Pricer
# =============================================================================

class HestonMCPricer:
    """
    Heston option pricer using Monte Carlo simulation.

    This pricer uses the HestonSimulator to generate paths under
    the risk-neutral measure and prices options via discounted payoffs.

    Parameters
    ----------
    model : HestonModel
        The Heston model instance
    n_paths : int
        Number of Monte Carlo paths
    n_steps : int
        Number of time steps per path
    scheme : str
        Discretization scheme: 'euler', 'full_truncation', 'reflection', 'qe'

    Notes
    -----
    Put-call parity (C - P = S - K*e^(-rT)) may not be exactly satisfied when
    calls and puts are priced with different random seeds due to Monte Carlo
    sampling error. To enforce parity, price the call and compute the put via:
    put_price = call_price - s0 + k * exp(-r * t)

    Examples
    --------
    model = HestonModel.from_params(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    pricer = model.create_pricer(method=PricingCapability.MONTE_CARLO, n_paths=50000, scheme="qe")
    result = pricer.price(s0=100, k=100, t=0.25, r=0.05)
    """

    def __init__(self, model: HestonModel, n_paths: int = 100000, n_steps: int = 252, scheme: str = "full_truncation"):
        self._model = model
        self._n_paths = n_paths
        self._n_steps = n_steps
        self._scheme = scheme
        self._model_name = "Heston (Monte Carlo)"

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
        simulator = self._model.create_simulator(scheme=self._scheme)
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
    print("Heston Unified Model Benchmark")
    print("=" * 60)

    # Test parameters
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7
    s0, k, t, r = 100.0, 100.0, 0.25, 0.05

    # Create model
    print("\n1. Creating HestonModel")
    print("-" * 40)
    model = HestonModel.from_params(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
    print(f"Model: {model}")
    print(f"Feller satisfied: {model.params.feller_satisfied}")

    # Test simulator
    print("\n2. Simulator Test")
    print("-" * 40)
    simulator = model.create_simulator(scheme="qe")
    print(f"Simulator: {type(simulator).__name__}")

    start = time.perf_counter()
    result = simulator.simulate_paths(s0=s0, mu=0.08, t=1.0, n_paths=10000, n_steps=252)
    elapsed = time.perf_counter() - start
    print(f"Simulated 10,000 paths in {elapsed*1000:.2f} ms")
    print(f"Final price mean: ${result.price_paths[:, -1].mean():.2f}")

    # Test FFT pricer
    print("\n3. FFT Pricer Test")
    print("-" * 40)
    pricer_fft = model.create_pricer(method=PricingCapability.FFT)
    result_fft = pricer_fft.price(s0=s0, k=k, t=t, r=r)
    print(f"FFT Call Price: ${result_fft.price:.4f}")
    print(f"FFT Time: {result_fft.computation_time*1000:.2f} ms")

    # Test Monte Carlo pricer
    print("\n4. Monte Carlo Pricer Test")
    print("-" * 40)
    pricer_mc = model.create_pricer(method=PricingCapability.MONTE_CARLO, n_paths=50000)
    result_mc = pricer_mc.price(s0=s0, k=k, t=t, r=r)
    print(f"MC Call Price: ${result_mc.price:.4f} +/- ${result_mc.std_error:.4f}")
    print(f"MC Time: {result_mc.computation_time*1000:.1f} ms")

    # Compare methods
    print("\n5. Method Comparison")
    print("-" * 40)
    print(f"FFT vs MC difference: ${abs(result_fft.price - result_mc.price):.4f}")
    print(f"FFT is ~{result_mc.computation_time/result_fft.computation_time:.0f}x faster")

    # Characteristic function test
    print("\n6. Characteristic Function Test")
    print("-" * 40)
    u = 1.0 + 0.5j
    cf = model.characteristic_function(u, s0=s0, t=t, r=r)
    print(f"CF(1+0.5i) = {cf:.6f}")

    print()
