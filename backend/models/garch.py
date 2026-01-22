"""
GARCH Family Models
===================

Unified GARCH(1,1), NGARCH, and GJR-GARCH models.

Models:
    GARCH(1,1):   sigma^2_t = omega + alpha * sigma^2_{t-1} * z^2_{t-1} + beta * sigma^2_{t-1}
    NGARCH:       sigma^2_t = omega + alpha * sigma^2_{t-1} * (z_{t-1} - theta)^2 + beta * sigma^2_{t-1}
    GJR-GARCH:    sigma^2_t = omega + (alpha + gamma * I_{t-1}) * sigma^2_{t-1} * z^2_{t-1} + beta * sigma^2_{t-1}

Used with specialized MC pricing (LRNVR - Locally Risk-Neutral Valuation Relationship).

Author: Thomas
Created: 2025
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability
from backend.simulation.models.garch import GARCHSimulator
from backend.simulation.models.ngarch import NGARCHSimulator
from backend.simulation.models.gjr_garch import GJRGARCHSimulator
from backend.engines.monte_carlo.garch_pricer import GARCHMCPricer, GARCHType


# =============================================================================
# BASE GARCH MODEL
# =============================================================================

@dataclass(frozen=True)
class BaseGARCHModel(Model):
    """
    Base class for GARCH family models.

    All GARCH variants share these common parameters and must implement
    their specific persistence calculation.

    Parameters
    ----------
    sigma0 : float
        Initial volatility (annualized), e.g., 0.20 for 20%
    omega : float
        Constant term in variance equation (omega > 0)
    alpha : float
        ARCH coefficient - reaction to past shocks (alpha >= 0)
    beta : float
        GARCH coefficient - persistence (beta >= 0)

    Notes
    -----
    - All GARCH models only support Monte Carlo pricing
    - Long-run variance = omega / (1 - persistence)
    - persistence property must be implemented by subclasses
    """

    sigma0: float
    omega: float
    alpha: float
    beta: float

    def _validate_base_params(self) -> None:
        """Validate common GARCH parameters."""
        if self.sigma0 <= 0:
            raise ValueError(f"sigma0 must be positive, got {self.sigma0}")
        if self.omega <= 0:
            raise ValueError(f"omega must be positive, got {self.omega}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")

    @property
    def supported_engines(self) -> List[PricingCapability]:
        """GARCH models only support Monte Carlo pricing."""
        return [PricingCapability.MONTE_CARLO]

    @property
    @abstractmethod
    def persistence(self) -> float:
        """
        Model-specific persistence measure.

        Must be < 1 for stationarity.
        """
        ...

    @property
    def long_run_variance(self) -> float:
        """Returns omega / (1 - persistence)."""
        return self.omega / (1 - self.persistence)

    @property
    def long_run_volatility(self) -> float:
        """Returns sqrt of long-run variance."""
        return np.sqrt(self.long_run_variance)

    @property
    def half_life(self) -> float:
        """Half-life of variance shocks in time steps."""
        if self.persistence <= 0 or self.persistence >= 1:
            return np.inf
        return np.log(2) / (-np.log(self.persistence))


# =============================================================================
# GARCH(1,1) MODEL
# =============================================================================

@dataclass(frozen=True)
class GARCHModel(BaseGARCHModel):
    """
    GARCH(1,1) Model.

    Variance dynamics:
        sigma^2_t = omega + alpha * sigma^2_{t-1} * z^2_{t-1} + beta * sigma^2_{t-1}

    Notes
    -----
    - Stationarity requires: alpha + beta < 1
    - Long-run variance: omega / (1 - alpha - beta)
    - Typical values: alpha ~ 0.05-0.10, beta ~ 0.85-0.95

    Examples
    --------
    model = GARCHModel(sigma0=0.20, omega=0.000002, alpha=0.05, beta=0.90)
    simulator = model.create_simulator()
    pricer = model.create_pricer()
    price = pricer.price(s0=100, k=100, t=0.25, r=0.05)
    """

    def __post_init__(self):
        """Validate parameters."""
        self._validate_base_params()
        if self.persistence >= 1:
            raise ValueError(
                f"Process is not stationary: alpha + beta = {self.persistence:.4f} >= 1. "
                f"Reduce alpha or beta."
            )

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return "GARCH(1,1)"

    def get_parameters(self) -> Dict[str, Any]:
        """Return model parameters as dictionary."""
        return {
            "sigma0": self.sigma0,
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    @property
    def persistence(self) -> float:
        """Returns alpha + beta, the persistence of shocks."""
        return self.alpha + self.beta

    def create_simulator(self, **kwargs) -> GARCHSimulator:
        """Create GARCH simulator."""
        return GARCHSimulator(
            sigma0=self.sigma0,
            omega=self.omega,
            alpha=self.alpha,
            beta=self.beta,
        )

    def create_pricer(
        self,
        n_paths: int = 100000,
        n_steps: int = 252,
        **kwargs
    ) -> GARCHMCPricer:
        """
        Create GARCH pricer (Monte Carlo with LRNVR).

        Parameters
        ----------
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
        return GARCHMCPricer(
            garch_type=GARCHType.GARCH,
            sigma0=self.sigma0,
            omega=self.omega,
            alpha=self.alpha,
            beta=self.beta,
            n_paths=n_paths,
            n_steps=n_steps,
        )

    def __repr__(self) -> str:
        return (
            f"GARCHModel(sigma0={self.sigma0}, omega={self.omega}, "
            f"alpha={self.alpha}, beta={self.beta})"
        )


# =============================================================================
# NGARCH MODEL
# =============================================================================

@dataclass(frozen=True)
class NGARCHModel(BaseGARCHModel):
    """
    NGARCH (Nonlinear Asymmetric GARCH) Model.

    Variance dynamics:
        sigma^2_t = omega + alpha * sigma^2_{t-1} * (z_{t-1} - theta)^2 + beta * sigma^2_{t-1}

    Parameters
    ----------
    theta : float
        Leverage parameter (theta > 0 for leverage effect)

    Notes
    -----
    - Stationarity requires: alpha * (1 + theta^2) + beta < 1
    - theta > 0: Bad news increases volatility more than good news
    - theta = 0: Reduces to GARCH(1,1)

    Examples
    --------
    model = NGARCHModel(sigma0=0.20, omega=0.000002, alpha=0.05, beta=0.90, theta=0.5)
    simulator = model.create_simulator()
    pricer = model.create_pricer()
    """

    theta: float = 0.0

    def __post_init__(self):
        """Validate parameters."""
        self._validate_base_params()
        if self.persistence >= 1:
            raise ValueError(
                f"Process is not stationary: alpha*(1+theta^2) + beta = {self.persistence:.4f} >= 1. "
                f"Reduce alpha, beta, or theta."
            )

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return "NGARCH (Nonlinear Asymmetric)"

    def get_parameters(self) -> Dict[str, Any]:
        """Return model parameters as dictionary."""
        return {
            "sigma0": self.sigma0,
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "theta": self.theta,
        }

    @property
    def persistence(self) -> float:
        """Returns alpha * (1 + theta^2) + beta."""
        return self.alpha * (1 + self.theta ** 2) + self.beta

    def create_simulator(self, **kwargs) -> NGARCHSimulator:
        """Create NGARCH simulator."""
        return NGARCHSimulator(
            sigma0=self.sigma0,
            omega=self.omega,
            alpha=self.alpha,
            beta=self.beta,
            theta=self.theta,
        )

    def create_pricer(
        self,
        n_paths: int = 100000,
        n_steps: int = 252,
        **kwargs
    ) -> GARCHMCPricer:
        """Create NGARCH pricer (Monte Carlo)."""
        return GARCHMCPricer(
            garch_type=GARCHType.NGARCH,
            sigma0=self.sigma0,
            omega=self.omega,
            alpha=self.alpha,
            beta=self.beta,
            theta=self.theta,
            n_paths=n_paths,
            n_steps=n_steps,
        )

    def __repr__(self) -> str:
        return (
            f"NGARCHModel(sigma0={self.sigma0}, omega={self.omega}, "
            f"alpha={self.alpha}, beta={self.beta}, theta={self.theta})"
        )


# =============================================================================
# GJR-GARCH MODEL
# =============================================================================

@dataclass(frozen=True)
class GJRGARCHModel(BaseGARCHModel):
    """
    GJR-GARCH Model.

    Variance dynamics:
        sigma^2_t = omega + (alpha + gamma * I_{t-1}) * sigma^2_{t-1} * z^2_{t-1} + beta * sigma^2_{t-1}

    Where I_{t-1} = 1 if z_{t-1} < 0 (negative return indicator).

    Parameters
    ----------
    gamma : float
        Asymmetry coefficient (gamma > 0 for leverage effect)

    Notes
    -----
    - Stationarity requires: alpha + 0.5*gamma + beta < 1
    - gamma > 0: Negative returns add extra volatility
    - gamma = 0: Reduces to GARCH(1,1)

    Examples
    --------
    model = GJRGARCHModel(sigma0=0.20, omega=0.000002, alpha=0.03, beta=0.90, gamma=0.07)
    simulator = model.create_simulator()
    pricer = model.create_pricer()
    """

    gamma: float = 0.0

    def __post_init__(self):
        """Validate parameters."""
        self._validate_base_params()
        if self.gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {self.gamma}")
        if self.persistence >= 1:
            raise ValueError(
                f"Process is not stationary: alpha + 0.5*gamma + beta = {self.persistence:.4f} >= 1. "
                f"Reduce alpha, beta, or gamma."
            )

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return "GJR-GARCH"

    def get_parameters(self) -> Dict[str, Any]:
        """Return model parameters as dictionary."""
        return {
            "sigma0": self.sigma0,
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }

    @property
    def persistence(self) -> float:
        """Returns alpha + 0.5*gamma + beta."""
        return self.alpha + 0.5 * self.gamma + self.beta

    def create_simulator(self, **kwargs) -> GJRGARCHSimulator:
        """Create GJR-GARCH simulator."""
        return GJRGARCHSimulator(
            sigma0=self.sigma0,
            omega=self.omega,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )

    def create_pricer(
        self,
        n_paths: int = 100000,
        n_steps: int = 252,
        **kwargs
    ) -> GARCHMCPricer:
        """Create GJR-GARCH pricer (Monte Carlo)."""
        return GARCHMCPricer(
            garch_type=GARCHType.GJR_GARCH,
            sigma0=self.sigma0,
            omega=self.omega,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            n_paths=n_paths,
            n_steps=n_steps,
        )

    def __repr__(self) -> str:
        return (
            f"GJRGARCHModel(sigma0={self.sigma0}, omega={self.omega}, "
            f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma})"
        )


# =============================================================================
# PARAMETER ALIASES
# =============================================================================

# Aliases for convenience
GARCHParams = GARCHModel
NGARCHParams = NGARCHModel
GJRGARCHParams = GJRGARCHModel


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("GARCH Family Models Test")
    print("=" * 60)

    # Common test parameters
    s0, k, t, r = 100.0, 100.0, 0.25, 0.05

    # =========================================================================
    # 1. GARCH(1,1) Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. GARCH(1,1) Model")
    print("=" * 60)

    # omega calibrated for ~20% long-run vol: omega = sigma_lr^2 * (1 - alpha - beta)
    garch_model = GARCHModel(sigma0=0.20, omega=0.002, alpha=0.05, beta=0.90)
    print(f"\nModel: {garch_model}")
    print(f"Persistence: {garch_model.persistence:.3f}")
    print(f"Long-run volatility: {garch_model.long_run_volatility:.1%}")
    print(f"Half-life: {garch_model.half_life:.1f} time steps")

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

    ngarch_model = NGARCHModel(sigma0=0.20, omega=0.002, alpha=0.05, beta=0.90, theta=0.5)
    print(f"\nModel: {ngarch_model}")
    print(f"Persistence: {ngarch_model.persistence:.3f}")
    print(f"Long-run volatility: {ngarch_model.long_run_volatility:.1%}")

    # Simulator test
    print("\n  Simulator Test:")
    print("  " + "-" * 38)
    simulator = ngarch_model.create_simulator()
    start = time.perf_counter()
    result = simulator.simulate_paths(s0=s0, mu=0.08, t=1.0, n_paths=10000, n_steps=252)
    elapsed = time.perf_counter() - start
    print(f"  Simulated 10,000 paths in {elapsed*1000:.2f} ms")

    # =========================================================================
    # 3. GJR-GARCH Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. GJR-GARCH Model")
    print("=" * 60)

    gjr_model = GJRGARCHModel(sigma0=0.20, omega=0.002, alpha=0.03, beta=0.90, gamma=0.07)
    print(f"\nModel: {gjr_model}")
    print(f"Persistence: {gjr_model.persistence:.3f}")
    print(f"Effective alpha for neg. shocks: {gjr_model.alpha + gjr_model.gamma:.3f}")

    # Simulator test
    print("\n  Simulator Test:")
    print("  " + "-" * 38)
    simulator = gjr_model.create_simulator()
    start = time.perf_counter()
    result = simulator.simulate_paths(s0=s0, mu=0.08, t=1.0, n_paths=10000, n_steps=252)
    elapsed = time.perf_counter() - start
    print(f"  Simulated 10,000 paths in {elapsed*1000:.2f} ms")

    # =========================================================================
    # 4. Validation Test
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. Validation Test")
    print("=" * 60)

    try:
        bad_model = GARCHModel(sigma0=0.20, omega=0.002, alpha=0.5, beta=0.6)
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"\nCorrectly rejected non-stationary GARCH: {e}")

    try:
        bad_model = GARCHModel(sigma0=-0.1, omega=0.002, alpha=0.05, beta=0.90)
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"Correctly rejected negative sigma0: {e}")

    print("\n" + "=" * 60)
    print("All GARCH tests passed!")
    print("=" * 60)
