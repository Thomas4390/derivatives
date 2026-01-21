"""
NGARCH (Nonlinear Asymmetric GARCH) Simulator
==============================================

Implementation of the NGARCH model (Engle & Ng, 1993) with joint
price-volatility dynamics.

Model Specification:
    Return dynamics:
        r_t = (μ - 0.5·σ²_t)·dt + σ_t·√dt·z_t,  where z_t ~ N(0,1)
        S_t = S_{t-1}·exp(r_t)

    Variance dynamics (NGARCH):
        σ²_t = ω + α·σ²_{t-1}·(z_{t-1} - θ)² + β·σ²_{t-1}

The θ parameter captures the leverage effect:
    - θ > 0: Negative returns (z < 0) increase future volatility more
    - θ = 0: Reduces to standard GARCH(1,1)

Stationarity condition: α·(1 + θ²) + β < 1
Long-run variance: σ²_∞ = ω / (1 - α·(1 + θ²) - β)

References:
    Engle, R.F. and Ng, V.K. (1993). "Measuring and Testing the Impact
    of News on Volatility." Journal of Finance, 48(5), 1749-1778.

Author: Thomas
Created: 2025
"""

import numpy as np
from numba import njit, prange
import time
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Handle both package import and direct script execution
try:
    from ..base import BaseSimulator, SimulationResult, StochasticVolatilityMixin
except ImportError:
    _project_root = Path(__file__).resolve().parents[3]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from backend.simulation.base import BaseSimulator, SimulationResult, StochasticVolatilityMixin


# =============================================================================
# Numba-Optimized Kernels
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _simulate_ngarch_paths(
    s0: float,
    mu: float,
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    theta: float,
    t: float,
    n_paths: int,
    n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba kernel for NGARCH joint price-volatility simulation.

    The shifted squared term (z - θ)² captures asymmetric response to news:
        - When θ > 0 and z < 0 (negative return), (z - θ)² > z²
        - This creates leverage effect: bad news increases vol more than good news
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    var0 = sigma0 * sigma0

    price_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    vol_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)

    for i in prange(n_paths):
        price_paths[i, 0] = s0
        vol_paths[i, 0] = sigma0

        var_t = var0

        for j in range(n_steps):
            z = np.random.standard_normal()

            # Current volatility
            sigma_t = np.sqrt(var_t)
            vol_paths[i, j] = sigma_t

            # Log return with volatility adjustment
            log_return = (mu - 0.5 * var_t) * dt + sigma_t * sqrt_dt * z

            # Price update
            price_paths[i, j + 1] = price_paths[i, j] * np.exp(log_return)

            # NGARCH variance update: shifted squared term
            shifted = z - theta
            var_next = omega + alpha * var_t * shifted * shifted + beta * var_t
            var_t = max(var_next, 1e-10)

        # Store final volatility
        vol_paths[i, n_steps] = np.sqrt(var_t)

    return price_paths, vol_paths


@njit(parallel=True, cache=True, fastmath=True)
def _simulate_ngarch_terminal(
    s0: float,
    mu: float,
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    theta: float,
    t: float,
    n_paths: int,
    n_steps: int
) -> np.ndarray:
    """
    Numba kernel for terminal-only NGARCH simulation.
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    var0 = sigma0 * sigma0

    terminals = np.empty(n_paths, dtype=np.float64)

    for i in prange(n_paths):
        s = s0
        var_t = var0

        for j in range(n_steps):
            z = np.random.standard_normal()

            sigma_t = np.sqrt(var_t)
            log_return = (mu - 0.5 * var_t) * dt + sigma_t * sqrt_dt * z
            s = s * np.exp(log_return)

            shifted = z - theta
            var_next = omega + alpha * var_t * shifted * shifted + beta * var_t
            var_t = max(var_next, 1e-10)

        terminals[i] = s

    return terminals


# =============================================================================
# NGARCH Simulator Class
# =============================================================================

class NGARCHSimulator(BaseSimulator, StochasticVolatilityMixin):
    """
    NGARCH (Nonlinear Asymmetric GARCH) model simulator.

    The variance follows the NGARCH recursion with leverage effect:
        σ²_t = ω + α·σ²_{t-1}·(z_{t-1} - θ)² + β·σ²_{t-1}

    The θ parameter introduces asymmetry (leverage effect):
        - θ > 0: Bad news (negative returns) increases volatility more
        - θ = 0: Standard GARCH(1,1)

    Parameters
    ----------
    sigma0 : float
        Initial volatility (annualized)
    omega : float
        Constant term in variance equation (ω > 0)
    alpha : float
        ARCH coefficient (α ≥ 0)
    beta : float
        GARCH coefficient (β ≥ 0)
    theta : float
        Leverage parameter (typically θ > 0)

    Notes
    -----
    - Stationarity requires: α·(1 + θ²) + β < 1
    - Long-run variance: ω / (1 - α·(1 + θ²) - β)
    - Typical values: θ ≈ 0.3-0.8 for equity indices

    Examples
    --------
    simulator = NGARCHSimulator(
        sigma0=0.20, omega=0.002, alpha=0.05, beta=0.90, theta=0.5
    )
    result = simulator.simulate_paths(s0=100, mu=0.08, t=1.0, n_paths=10000, n_steps=252)
    """

    def __init__(
        self,
        sigma0: float,
        omega: float,
        alpha: float,
        beta: float,
        theta: float
    ):
        super().__init__()
        self._model_name = "NGARCH (Nonlinear Asymmetric)"
        self._sigma0 = sigma0
        self._omega = omega
        self._alpha = alpha
        self._beta = beta
        self._theta = theta

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate NGARCH parameters."""
        if self._sigma0 <= 0:
            raise ValueError(f"Initial volatility sigma0 must be positive, got {self._sigma0}")
        if self._omega <= 0:
            raise ValueError(f"Omega must be positive, got {self._omega}")
        if self._alpha < 0:
            raise ValueError(f"Alpha must be non-negative, got {self._alpha}")
        if self._beta < 0:
            raise ValueError(f"Beta must be non-negative, got {self._beta}")

        persistence = self._alpha * (1 + self._theta ** 2) + self._beta
        if persistence >= 1:
            raise ValueError(
                f"Process is not stationary: α·(1+θ²) + β = {persistence:.4f} ≥ 1. "
                f"Reduce α, β, or θ."
            )

    # Properties
    @property
    def sigma0(self) -> float:
        return self._sigma0

    @property
    def omega(self) -> float:
        return self._omega

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def persistence(self) -> float:
        """Returns α·(1 + θ²) + β, the persistence of shocks."""
        return self._alpha * (1 + self._theta ** 2) + self._beta

    def get_parameters(self) -> Dict[str, Any]:
        """Returns model parameters."""
        return {
            "sigma0": self._sigma0,
            "omega": self._omega,
            "alpha": self._alpha,
            "beta": self._beta,
            "theta": self._theta,
        }

    def long_run_variance(self) -> float:
        """Returns the theoretical long-run variance."""
        return self._omega / (1 - self.persistence)

    def long_run_volatility(self) -> float:
        """Returns the theoretical long-run volatility."""
        return np.sqrt(self.long_run_variance())

    def feller_condition_satisfied(self) -> bool:
        """Returns whether process is stationary."""
        return self.persistence < 1

    def news_impact_curve(self, z_range: np.ndarray) -> np.ndarray:
        """
        Compute the news impact curve (NIC).

        The NIC shows how the variance responds to different shock sizes.

        Parameters
        ----------
        z_range : np.ndarray
            Range of standardized shocks to evaluate

        Returns
        -------
        np.ndarray
            Conditional variance values for each shock
        """
        var = self.long_run_variance()
        shifted = z_range - self._theta
        return self._omega + self._alpha * var * shifted ** 2 + self._beta * var

    def simulate_paths(
        self,
        s0: float,
        mu: float,
        t: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None
    ) -> SimulationResult:
        """
        Simulate NGARCH price and volatility paths.

        Returns
        -------
        SimulationResult
            Result with both price_paths and volatility_paths
        """
        self.validate_inputs(s0, mu, t, n_paths, n_steps)

        if seed is not None:
            np.random.seed(seed)

        start_time = time.perf_counter()

        price_paths, vol_paths = _simulate_ngarch_paths(
            s0, mu, self._sigma0,
            self._omega, self._alpha, self._beta, self._theta,
            t, n_paths, n_steps
        )

        computation_time = time.perf_counter() - start_time
        time_grid = np.linspace(0, t, n_steps + 1)

        return SimulationResult(
            price_paths=price_paths,
            time_grid=time_grid,
            model_name=self._model_name,
            computation_time=computation_time,
            n_paths=n_paths,
            n_steps=n_steps,
            volatility_paths=vol_paths,
            parameters=self.get_parameters() | {"s0": s0, "mu": mu, "t": t}
        )

    def simulate_terminal(
        self,
        s0: float,
        mu: float,
        t: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate only terminal prices S(T).

        Returns
        -------
        np.ndarray
            Terminal prices, shape (n_paths,)
        """
        self.validate_inputs(s0, mu, t, n_paths, n_steps)

        if seed is not None:
            np.random.seed(seed)

        return _simulate_ngarch_terminal(
            s0, mu, self._sigma0,
            self._omega, self._alpha, self._beta, self._theta,
            t, n_paths, n_steps
        )


# =============================================================================
# Convenience Function
# =============================================================================

def simulate_ngarch(
    s0: float,
    mu: float,
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    theta: float,
    t: float,
    n_paths: int = 100000,
    n_steps: int = 252,
    seed: Optional[int] = None,
    terminal_only: bool = False
):
    """
    Convenience function for NGARCH simulation.

    Parameters
    ----------
    s0 : float
        Initial stock price
    mu : float
        Expected return (annualized)
    sigma0 : float
        Initial volatility
    omega : float
        NGARCH constant term
    alpha : float
        ARCH coefficient
    beta : float
        GARCH coefficient
    theta : float
        Leverage parameter
    t : float
        Time horizon in years
    n_paths : int
        Number of paths
    n_steps : int
        Number of steps
    seed : int, optional
        Random seed
    terminal_only : bool
        Return only terminal values

    Returns
    -------
    SimulationResult or np.ndarray
    """
    simulator = NGARCHSimulator(
        sigma0=sigma0, omega=omega, alpha=alpha, beta=beta, theta=theta
    )

    if terminal_only:
        return simulator.simulate_terminal(s0, mu, t, n_paths, n_steps, seed)
    else:
        return simulator.simulate_paths(s0, mu, t, n_paths, n_steps, seed)


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NGARCH (Nonlinear Asymmetric) Benchmark")
    print("=" * 60)

    # Test parameters
    s0, mu, t, n_steps = 100.0, 0.08, 1.0, 252
    # omega calibrated for ~20% long-run vol
    sigma0, omega, alpha, beta, theta = 0.20, 0.002, 0.05, 0.90, 0.5

    # Warmup JIT compilation
    print("\nWarming up JIT compilation...")
    _ = simulate_ngarch(s0, mu, sigma0, omega, alpha, beta, theta, t, n_paths=1000, n_steps=10, terminal_only=True)

    # Benchmark different path counts
    path_counts = [10_000, 50_000, 100_000, 500_000, 1_000_000]

    print(f"\n{'Paths':>12} {'Time (ms)':>12} {'Paths/sec':>15}")
    print("-" * 42)

    for n_paths in path_counts:
        start = time.perf_counter()
        result = simulate_ngarch(s0, mu, sigma0, omega, alpha, beta, theta, t, n_paths=n_paths, n_steps=n_steps)
        elapsed = time.perf_counter() - start

        paths_per_sec = n_paths / elapsed
        print(f"{n_paths:>12,} {elapsed*1000:>12.2f} {paths_per_sec:>15,.0f}")

    print()
