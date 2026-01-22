"""
Bates Stochastic Volatility with Jumps Simulator
=================================================

Implementation of the Bates (1996) model combining Heston stochastic
volatility with Merton-style jumps, with Numba optimization.

Model Specification (P-measure):
    dS = μ·S·dt + √V·S·dW_S + (J - 1)·S·dN
    dV = κ·(θ - V)·dt + ξ·√V·dW_V

    Corr(dW_S, dW_V) = ρ

Where:
    - dN is a Poisson process with intensity λ
    - J is lognormally distributed: ln(J) ~ N(μ_J, σ²_J)

Parameters:
    Heston: κ, θ, ξ, ρ, v0
    Jumps: λ (lambda_j), μ_J (mu_j), σ_J (sigma_j)

Author: Thomas
Created: 2025
"""

import numpy as np
from numba import njit, prange
import time
from typing import Optional, Dict, Any, Tuple

from ..base import BaseSimulator, SimulationResult, StochasticVolatilityMixin


# =============================================================================
# Numba-Optimized Kernels
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _simulate_bates_paths(
    s0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    t: float,
    n_paths: int,
    n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba kernel for Bates model path simulation.

    Combines Heston variance dynamics with Merton-style jumps.
    Uses full truncation scheme for variance.
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    # Pre-compute correlation coefficients
    sqrt_one_minus_rho2 = np.sqrt(1.0 - rho * rho)

    # Jump compensator
    k = np.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
    lambda_dt = lambda_j * dt

    # Allocate output arrays
    s_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    v_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)

    for i in prange(n_paths):
        s_paths[i, 0] = s0
        v_paths[i, 0] = v0

        for j in range(n_steps):
            # Generate correlated Brownian increments
            z1 = np.random.standard_normal()
            z2 = np.random.standard_normal()

            dw_s = sqrt_dt * z1
            dw_v = sqrt_dt * (rho * z1 + sqrt_one_minus_rho2 * z2)

            v_curr = v_paths[i, j]
            s_curr = s_paths[i, j]

            # Full truncation for variance
            v_plus = max(v_curr, 0.0)
            sqrt_v = np.sqrt(v_plus)

            # Variance update (Heston dynamics)
            v_next = v_curr + kappa * (theta - v_plus) * dt + xi * sqrt_v * dw_v
            v_next = max(v_next, 0.0)

            # Price update with diffusion
            # Adjusted drift for jump compensator
            drift = (mu - lambda_j * k - 0.5 * v_plus) * dt
            log_return = drift + sqrt_v * dw_s

            # Jump component
            n_jumps = np.random.poisson(lambda_dt)
            if n_jumps > 0:
                for _ in range(n_jumps):
                    log_return += mu_j + sigma_j * np.random.standard_normal()

            s_next = s_curr * np.exp(log_return)

            v_paths[i, j + 1] = v_next
            s_paths[i, j + 1] = s_next

    return s_paths, v_paths


@njit(parallel=True, cache=True, fastmath=True)
def _simulate_bates_terminal(
    s0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    t: float,
    n_paths: int,
    n_steps: int
) -> np.ndarray:
    """
    Numba kernel for terminal-only Bates simulation.
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)
    sqrt_one_minus_rho2 = np.sqrt(1.0 - rho * rho)

    k = np.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
    lambda_dt = lambda_j * dt

    terminals = np.empty(n_paths, dtype=np.float64)

    for i in prange(n_paths):
        s = s0
        v = v0

        for j in range(n_steps):
            z1 = np.random.standard_normal()
            z2 = np.random.standard_normal()

            dw_s = sqrt_dt * z1
            dw_v = sqrt_dt * (rho * z1 + sqrt_one_minus_rho2 * z2)

            v_plus = max(v, 0.0)
            sqrt_v = np.sqrt(v_plus)

            v = max(v + kappa * (theta - v_plus) * dt + xi * sqrt_v * dw_v, 0.0)

            drift = (mu - lambda_j * k - 0.5 * v_plus) * dt
            log_return = drift + sqrt_v * dw_s

            n_jumps = np.random.poisson(lambda_dt)
            if n_jumps > 0:
                for _ in range(n_jumps):
                    log_return += mu_j + sigma_j * np.random.standard_normal()

            s = s * np.exp(log_return)

        terminals[i] = s

    return terminals


# =============================================================================
# Bates Simulator Class
# =============================================================================

class BatesSimulator(BaseSimulator, StochasticVolatilityMixin):
    """
    Bates stochastic volatility with jumps model simulator.

    Combines Heston variance dynamics with Merton-style price jumps.

    Parameters
    ----------
    v0 : float
        Initial variance
    kappa : float
        Mean reversion speed of variance
    theta : float
        Long-run variance level
    xi : float
        Volatility of volatility
    rho : float
        Correlation between price and variance
    lambda_j : float
        Jump intensity (expected jumps per year)
    mu_j : float
        Mean of log-jump size
    sigma_j : float
        Standard deviation of log-jump size

    Examples
    --------
    simulator = BatesSimulator(
        v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
    )
    result = simulator.simulate_paths(s0=100, mu=0.08, t=1.0, n_paths=10000, n_steps=252)
    """

    def __init__(
        self,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        lambda_j: float,
        mu_j: float,
        sigma_j: float
    ):
        super().__init__()
        self._model_name = "Bates (Heston + Jumps)"

        # Heston parameters
        self._v0 = v0
        self._kappa = kappa
        self._theta = theta
        self._xi = xi
        self._rho = rho

        # Jump parameters
        self._lambda_j = lambda_j
        self._mu_j = mu_j
        self._sigma_j = sigma_j

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate model parameters."""
        # Heston parameter validation
        if self._v0 < 0:
            raise ValueError(f"Initial variance v0 must be non-negative, got {self._v0}")
        if self._kappa <= 0:
            raise ValueError(f"Mean reversion kappa must be positive, got {self._kappa}")
        if self._theta < 0:
            raise ValueError(f"Long-run variance theta must be non-negative, got {self._theta}")
        if self._xi <= 0:
            raise ValueError(f"Vol of vol xi must be positive, got {self._xi}")
        if not -1 <= self._rho <= 1:
            raise ValueError(f"Correlation rho must be in [-1, 1], got {self._rho}")

        # Jump parameter validation
        if self._lambda_j < 0:
            raise ValueError(f"Jump intensity lambda_j must be non-negative, got {self._lambda_j}")
        if self._sigma_j < 0:
            raise ValueError(f"Jump volatility sigma_j must be non-negative, got {self._sigma_j}")

    # Properties
    @property
    def v0(self) -> float:
        return self._v0

    @property
    def kappa(self) -> float:
        return self._kappa

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def xi(self) -> float:
        return self._xi

    @property
    def rho(self) -> float:
        return self._rho

    @property
    def lambda_j(self) -> float:
        return self._lambda_j

    @property
    def mu_j(self) -> float:
        return self._mu_j

    @property
    def sigma_j(self) -> float:
        return self._sigma_j

    @property
    def expected_jump_size(self) -> float:
        """Returns E[J - 1], the expected percentage jump."""
        return np.exp(self._mu_j + 0.5 * self._sigma_j ** 2) - 1

    def get_parameters(self) -> Dict[str, Any]:
        """Returns model parameters."""
        return {
            # Heston
            "v0": self._v0,
            "kappa": self._kappa,
            "theta": self._theta,
            "xi": self._xi,
            "rho": self._rho,
            # Jumps
            "lambda_j": self._lambda_j,
            "mu_j": self._mu_j,
            "sigma_j": self._sigma_j,
        }

    def long_run_variance(self) -> float:
        """Returns the theoretical long-run variance θ."""
        return self._theta

    def feller_condition_satisfied(self) -> bool:
        """Check if Feller condition (2κθ > ξ²) is satisfied."""
        return 2 * self._kappa * self._theta > self._xi ** 2

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
        Simulate Bates price and variance paths.

        Returns
        -------
        SimulationResult
            Result with both price_paths and volatility_paths
        """
        self.validate_inputs(s0, mu, t, n_paths, n_steps)

        if seed is not None:
            np.random.seed(seed)

        start_time = time.perf_counter()

        s_paths, v_paths = _simulate_bates_paths(
            s0, self._v0, mu,
            self._kappa, self._theta, self._xi, self._rho,
            self._lambda_j, self._mu_j, self._sigma_j,
            t, n_paths, n_steps
        )

        computation_time = time.perf_counter() - start_time
        time_grid = np.linspace(0, t, n_steps + 1)

        # Convert variance paths to volatility paths
        vol_paths = np.sqrt(v_paths)

        return SimulationResult(
            price_paths=s_paths,
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

        return _simulate_bates_terminal(
            s0, self._v0, mu,
            self._kappa, self._theta, self._xi, self._rho,
            self._lambda_j, self._mu_j, self._sigma_j,
            t, n_paths, n_steps
        )


# =============================================================================
# Convenience Function
# =============================================================================

def simulate_bates(
    s0: float,
    mu: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    t: float,
    n_paths: int = 100000,
    n_steps: int = 252,
    seed: Optional[int] = None,
    terminal_only: bool = False
):
    """
    Convenience function for Bates model simulation.

    Parameters
    ----------
    s0 : float
        Initial stock price
    mu : float
        Expected return (annualized)
    v0 : float
        Initial variance
    kappa : float
        Mean reversion speed
    theta : float
        Long-run variance
    xi : float
        Volatility of volatility
    rho : float
        Correlation
    lambda_j : float
        Jump intensity
    mu_j : float
        Mean of log-jump
    sigma_j : float
        Std of log-jump
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
    simulator = BatesSimulator(
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j
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
    print("Bates (Heston + Jumps) Benchmark")
    print("=" * 60)

    # Test parameters
    s0, mu, t, n_steps = 100.0, 0.08, 1.0, 252
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7
    lambda_j, mu_j, sigma_j = 0.5, -0.1, 0.2

    # Warmup JIT compilation
    print("\nWarming up JIT compilation...")
    _ = simulate_bates(s0, mu, v0, kappa, theta, xi, rho, lambda_j, mu_j, sigma_j, t, n_paths=1000, n_steps=10, terminal_only=True)

    # Benchmark different path counts
    path_counts = [10_000, 50_000, 100_000, 500_000]

    print(f"\n{'Paths':>12} {'Time (ms)':>12} {'Paths/sec':>15}")
    print("-" * 42)

    for n_paths in path_counts:
        start = time.perf_counter()
        result = simulate_bates(s0, mu, v0, kappa, theta, xi, rho, lambda_j, mu_j, sigma_j, t, n_paths=n_paths, n_steps=n_steps)
        elapsed = time.perf_counter() - start

        paths_per_sec = n_paths / elapsed
        print(f"{n_paths:>12,} {elapsed*1000:>12.2f} {paths_per_sec:>15,.0f}")

    print()
