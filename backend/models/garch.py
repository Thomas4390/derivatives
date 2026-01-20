"""
GARCH Family Models
===================

Unified GARCH(1,1), NGARCH, and GJR-GARCH models.

Models:
    GARCH(1,1):   sigma^2_t = omega + alpha * sigma^2_{t-1} * z^2_{t-1} + beta * sigma^2_{t-1}
    NGARCH:       sigma^2_t = omega + alpha * sigma^2_{t-1} * (z_{t-1} - theta)^2 + beta * sigma^2_{t-1}
    GJR-GARCH:    sigma^2_t = omega + (alpha + gamma * I_{t-1}) * sigma^2_{t-1} * z^2_{t-1} + beta * sigma^2_{t-1}

Author: Derivatives Pricing Project
"""

from typing import Optional, List, Union
import sys
from pathlib import Path

# Handle imports for both package and direct execution
try:
    from .base import BaseModel, PricingCapability
    from .parameters.garch import GARCHParams, NGARCHParams, GJRGARCHParams
except ImportError:
    _project_root = Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from backend.models.base import BaseModel, PricingCapability
    from backend.models.parameters.garch import GARCHParams, NGARCHParams, GJRGARCHParams

try:
    from backend.simulation.models.garch import GARCHSimulator
    from backend.simulation.models.ngarch import NGARCHSimulator
    from backend.simulation.models.gjr_garch import GJRGARCHSimulator
    from backend.option_pricing.garch import GARCHMCPricer, GARCHType
except ImportError:
    from backend.simulation.models.garch import GARCHSimulator
    from backend.simulation.models.ngarch import NGARCHSimulator
    from backend.simulation.models.gjr_garch import GJRGARCHSimulator
    from backend.option_pricing.garch import GARCHMCPricer, GARCHType


class GARCHModel(BaseModel[GARCHParams]):
    """
    GARCH(1,1) Model.

    Variance follows: sigma^2_t = omega + alpha * sigma^2_{t-1} * z^2_{t-1} + beta * sigma^2_{t-1}

    Example
    -------
    model = GARCHModel.from_params(sigma0=0.20, omega=0.000002, alpha=0.05, beta=0.90)
    simulator = model.create_simulator()
    pricer = model.create_pricer(r=0.05)
    """

    def __init__(self, params: GARCHParams):
        super().__init__(params)

    @classmethod
    def from_params(
        cls,
        sigma0: float,
        omega: float,
        alpha: float,
        beta: float
    ) -> "GARCHModel":
        """Create model from individual parameters."""
        return cls(GARCHParams(sigma0=sigma0, omega=omega, alpha=alpha, beta=beta))

    @property
    def model_name(self) -> str:
        return "GARCH(1,1)"

    @property
    def supported_pricing_methods(self) -> List[PricingCapability]:
        return [PricingCapability.MONTE_CARLO]

    def create_simulator(self, **kwargs) -> GARCHSimulator:
        """Create GARCH simulator."""
        return GARCHSimulator(
            sigma0=self.params.sigma0,
            omega=self.params.omega,
            alpha=self.params.alpha,
            beta=self.params.beta,
        )

    def create_pricer(
        self,
        method: Optional[PricingCapability] = None,
        n_paths: int = 100000,
        n_steps: int = 252,
        **kwargs
    ) -> GARCHMCPricer:
        """
        Create GARCH pricer (Monte Carlo with LRNVR).

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
        GARCHMCPricer
            Configured MC pricer

        Notes
        -----
        The risk-free rate is passed at pricing time, not construction time.
        """
        method = method or PricingCapability.MONTE_CARLO

        if method != PricingCapability.MONTE_CARLO:
            raise ValueError(f"GARCH only supports MONTE_CARLO pricing, got {method}")

        return GARCHMCPricer(
            garch_type=GARCHType.GARCH,
            sigma0=self.params.sigma0,
            omega=self.params.omega,
            alpha=self.params.alpha,
            beta=self.params.beta,
            n_paths=n_paths,
            n_steps=n_steps,
        )


class NGARCHModel(BaseModel[NGARCHParams]):
    """
    NGARCH (Nonlinear Asymmetric GARCH) Model.

    Variance follows: sigma^2_t = omega + alpha * sigma^2_{t-1} * (z_{t-1} - theta)^2 + beta * sigma^2_{t-1}

    Example
    -------
    model = NGARCHModel.from_params(sigma0=0.20, omega=0.000002, alpha=0.05, beta=0.90, theta=0.5)
    simulator = model.create_simulator()
    pricer = model.create_pricer(r=0.05)
    """

    def __init__(self, params: NGARCHParams):
        super().__init__(params)

    @classmethod
    def from_params(
        cls,
        sigma0: float,
        omega: float,
        alpha: float,
        beta: float,
        theta: float
    ) -> "NGARCHModel":
        """Create model from individual parameters."""
        return cls(NGARCHParams(
            sigma0=sigma0, omega=omega, alpha=alpha, beta=beta, theta=theta
        ))

    @property
    def model_name(self) -> str:
        return "NGARCH (Nonlinear Asymmetric)"

    @property
    def supported_pricing_methods(self) -> List[PricingCapability]:
        return [PricingCapability.MONTE_CARLO]

    def create_simulator(self, **kwargs) -> NGARCHSimulator:
        """Create NGARCH simulator."""
        return NGARCHSimulator(
            sigma0=self.params.sigma0,
            omega=self.params.omega,
            alpha=self.params.alpha,
            beta=self.params.beta,
            theta=self.params.theta,
        )

    def create_pricer(
        self,
        method: Optional[PricingCapability] = None,
        n_paths: int = 100000,
        n_steps: int = 252,
        **kwargs
    ) -> GARCHMCPricer:
        """Create NGARCH pricer (Monte Carlo)."""
        method = method or PricingCapability.MONTE_CARLO

        if method != PricingCapability.MONTE_CARLO:
            raise ValueError(f"NGARCH only supports MONTE_CARLO pricing, got {method}")

        return GARCHMCPricer(
            garch_type=GARCHType.NGARCH,
            sigma0=self.params.sigma0,
            omega=self.params.omega,
            alpha=self.params.alpha,
            beta=self.params.beta,
            theta=self.params.theta,
            n_paths=n_paths,
            n_steps=n_steps,
        )


class GJRGARCHModel(BaseModel[GJRGARCHParams]):
    """
    GJR-GARCH Model.

    Variance follows: sigma^2_t = omega + (alpha + gamma * I_{t-1}) * sigma^2_{t-1} * z^2_{t-1} + beta * sigma^2_{t-1}

    Example
    -------
    model = GJRGARCHModel.from_params(sigma0=0.20, omega=0.000002, alpha=0.03, beta=0.90, gamma=0.07)
    simulator = model.create_simulator()
    pricer = model.create_pricer(r=0.05)
    """

    def __init__(self, params: GJRGARCHParams):
        super().__init__(params)

    @classmethod
    def from_params(
        cls,
        sigma0: float,
        omega: float,
        alpha: float,
        beta: float,
        gamma: float
    ) -> "GJRGARCHModel":
        """Create model from individual parameters."""
        return cls(GJRGARCHParams(
            sigma0=sigma0, omega=omega, alpha=alpha, beta=beta, gamma=gamma
        ))

    @property
    def model_name(self) -> str:
        return "GJR-GARCH"

    @property
    def supported_pricing_methods(self) -> List[PricingCapability]:
        return [PricingCapability.MONTE_CARLO]

    def create_simulator(self, **kwargs) -> GJRGARCHSimulator:
        """Create GJR-GARCH simulator."""
        return GJRGARCHSimulator(
            sigma0=self.params.sigma0,
            omega=self.params.omega,
            alpha=self.params.alpha,
            beta=self.params.beta,
            gamma=self.params.gamma,
        )

    def create_pricer(
        self,
        method: Optional[PricingCapability] = None,
        n_paths: int = 100000,
        n_steps: int = 252,
        **kwargs
    ) -> GARCHMCPricer:
        """Create GJR-GARCH pricer (Monte Carlo)."""
        method = method or PricingCapability.MONTE_CARLO

        if method != PricingCapability.MONTE_CARLO:
            raise ValueError(f"GJR-GARCH only supports MONTE_CARLO pricing, got {method}")

        return GARCHMCPricer(
            garch_type=GARCHType.GJR_GARCH,
            sigma0=self.params.sigma0,
            omega=self.params.omega,
            alpha=self.params.alpha,
            beta=self.params.beta,
            gamma=self.params.gamma,
            n_paths=n_paths,
            n_steps=n_steps,
        )


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == "__main__":
    import time
    import numpy as np

    print("=" * 60)
    print("GARCH Family Unified Models Benchmark")
    print("=" * 60)

    # Common test parameters
    s0, k, t, r = 100.0, 100.0, 0.25, 0.05

    # =========================================================================
    # 1. GARCH(1,1) Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. GARCH(1,1) Model")
    print("=" * 60)

    # omega calibrated for 20% long-run vol: omega = sigma_lr^2 * (1 - alpha - beta)
    garch_params = {"sigma0": 0.20, "omega": 0.002, "alpha": 0.05, "beta": 0.90}
    garch_model = GARCHModel.from_params(**garch_params)
    print(f"\nModel: {garch_model}")
    print(f"Persistence (alpha + beta): {garch_params['alpha'] + garch_params['beta']:.3f}")
    print(f"Stationary: {garch_params['alpha'] + garch_params['beta'] < 1}")

    # Simulator test
    print("\n  Simulator Test:")
    print("  " + "-" * 38)
    simulator = garch_model.create_simulator()
    start = time.perf_counter()
    result = simulator.simulate_paths(s0=s0, mu=0.08, t=1.0, n_paths=10000, n_steps=252)
    elapsed = time.perf_counter() - start
    print(f"  Simulated 10,000 paths in {elapsed*1000:.2f} ms")
    print(f"  Final price mean: ${result.price_paths[:, -1].mean():.2f}")

    # MC Pricer test
    print("\n  MC Pricer Test:")
    print("  " + "-" * 38)
    pricer = garch_model.create_pricer(n_paths=50000, n_steps=int(252*t))
    result_mc = pricer.price(s0=s0, k=k, t=t, r=r)
    print(f"  MC Call Price: ${result_mc.price:.4f} +/- ${result_mc.std_error:.4f}")
    print(f"  MC Time: {result_mc.computation_time*1000:.1f} ms")

    # =========================================================================
    # 2. NGARCH Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. NGARCH (Nonlinear Asymmetric) Model")
    print("=" * 60)

    # omega calibrated for 20% long-run vol (approximately, theta affects this slightly)
    ngarch_params = {"sigma0": 0.20, "omega": 0.002, "alpha": 0.05, "beta": 0.90, "theta": 0.5}
    ngarch_model = NGARCHModel.from_params(**ngarch_params)
    print(f"\nModel: {ngarch_model}")
    print(f"Leverage parameter (theta): {ngarch_params['theta']}")

    # Simulator test
    print("\n  Simulator Test:")
    print("  " + "-" * 38)
    simulator = ngarch_model.create_simulator()
    start = time.perf_counter()
    result = simulator.simulate_paths(s0=s0, mu=0.08, t=1.0, n_paths=10000, n_steps=252)
    elapsed = time.perf_counter() - start
    print(f"  Simulated 10,000 paths in {elapsed*1000:.2f} ms")
    print(f"  Final price mean: ${result.price_paths[:, -1].mean():.2f}")

    # MC Pricer test
    print("\n  MC Pricer Test:")
    print("  " + "-" * 38)
    pricer = ngarch_model.create_pricer(n_paths=50000, n_steps=int(252*t))
    result_mc = pricer.price(s0=s0, k=k, t=t, r=r)
    print(f"  MC Call Price: ${result_mc.price:.4f} +/- ${result_mc.std_error:.4f}")
    print(f"  MC Time: {result_mc.computation_time*1000:.1f} ms")

    # =========================================================================
    # 3. GJR-GARCH Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. GJR-GARCH Model")
    print("=" * 60)

    # omega calibrated for 20% long-run vol (approximately, gamma affects this slightly)
    gjr_params = {"sigma0": 0.20, "omega": 0.002, "alpha": 0.03, "beta": 0.90, "gamma": 0.07}
    gjr_model = GJRGARCHModel.from_params(**gjr_params)
    print(f"\nModel: {gjr_model}")
    print(f"Asymmetry parameter (gamma): {gjr_params['gamma']}")
    print(f"Effective alpha for neg. shocks: {gjr_params['alpha'] + gjr_params['gamma']:.3f}")

    # Simulator test
    print("\n  Simulator Test:")
    print("  " + "-" * 38)
    simulator = gjr_model.create_simulator()
    start = time.perf_counter()
    result = simulator.simulate_paths(s0=s0, mu=0.08, t=1.0, n_paths=10000, n_steps=252)
    elapsed = time.perf_counter() - start
    print(f"  Simulated 10,000 paths in {elapsed*1000:.2f} ms")
    print(f"  Final price mean: ${result.price_paths[:, -1].mean():.2f}")

    # MC Pricer test
    print("\n  MC Pricer Test:")
    print("  " + "-" * 38)
    pricer = gjr_model.create_pricer(n_paths=50000, n_steps=int(252*t))
    result_mc = pricer.price(s0=s0, k=k, t=t, r=r)
    print(f"  MC Call Price: ${result_mc.price:.4f} +/- ${result_mc.std_error:.4f}")
    print(f"  MC Time: {result_mc.computation_time*1000:.1f} ms")

    # =========================================================================
    # 4. Model Comparison
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. GARCH Family Comparison")
    print("=" * 60)

    print("\n  Model Prices (Call, K=100, T=0.25):")
    print("  " + "-" * 38)

    models = [
        ("GARCH(1,1)", garch_model),
        ("NGARCH", ngarch_model),
        ("GJR-GARCH", gjr_model),
    ]

    for name, model in models:
        pricer = model.create_pricer(n_paths=50000, n_steps=int(252*t))
        result = pricer.price(s0=s0, k=k, t=t, r=r, seed=42)
        print(f"  {name:12s}: ${result.price:.4f} +/- ${result.std_error:.4f}")

    # Compare with Black-Scholes
    from backend.models.gbm import GBMModel
    bs_model = GBMModel.from_params(sigma=0.20)
    bs_price = bs_model.create_pricer().price(s0=s0, k=k, t=t, r=r, option_type="call").price
    print(f"  {'Black-Scholes':12s}: ${bs_price:.4f} (analytical)")

    print()
