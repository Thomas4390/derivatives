"""
Merton Jump Diffusion Simulator
================================

Implementation of the Merton (1976) jump-diffusion model with Numba optimization.

Model Specification (P-measure):
    dS/S = (μ - λ·k)·dt + σ·dW + (J - 1)·dN

Where:
    - dN is a Poisson process with intensity λ
    - J is lognormally distributed: ln(J) ~ N(μ_J, σ²_J)
    - k = E[J - 1] = exp(μ_J + 0.5·σ²_J) - 1 (compensator)

Parameters:
    λ (lambda_j): Jump intensity (expected jumps per year)
    μ_J (mu_j): Mean of log-jump size
    σ_J (sigma_j): Std dev of log-jump size

Author: Thomas
Created: 2025
"""

import numpy as np
from numba import njit, prange
import time
from typing import Optional, Dict, Any

from ..base import BaseSimulator, SimulationResult


# =============================================================================
# Numba-Optimized Kernels
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _simulate_merton_paths(
    s0: float,
    mu: float,
    sigma: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    t: float,
    n_paths: int,
    n_steps: int
) -> np.ndarray:
    """
    Numba kernel for Merton jump diffusion path simulation.
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    # Compensator for drift
    k = np.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0

    # Adjusted drift under P-measure
    drift = (mu - lambda_j * k - 0.5 * sigma * sigma) * dt
    diffusion = sigma * sqrt_dt

    # Jump intensity per time step
    lambda_dt = lambda_j * dt

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)

    for i in prange(n_paths):
        paths[i, 0] = s0

        for j in range(n_steps):
            # Diffusion component
            z = np.random.standard_normal()
            log_return = drift + diffusion * z

            # Jump component (Poisson number of jumps)
            n_jumps = np.random.poisson(lambda_dt)

            if n_jumps > 0:
                # Sum of log-normal jump sizes
                jump_sum = 0.0
                for _ in range(n_jumps):
                    jump_sum += mu_j + sigma_j * np.random.standard_normal()
                log_return += jump_sum

            paths[i, j + 1] = paths[i, j] * np.exp(log_return)

    return paths


@njit(parallel=True, cache=True, fastmath=True)
def _simulate_merton_terminal(
    s0: float,
    mu: float,
    sigma: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    t: float,
    n_paths: int,
    n_steps: int
) -> np.ndarray:
    """
    Numba kernel for terminal-only Merton simulation.
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    k = np.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
    drift = (mu - lambda_j * k - 0.5 * sigma * sigma) * dt
    diffusion = sigma * sqrt_dt
    lambda_dt = lambda_j * dt

    terminals = np.empty(n_paths, dtype=np.float64)

    for i in prange(n_paths):
        s = s0

        for j in range(n_steps):
            z = np.random.standard_normal()
            log_return = drift + diffusion * z

            n_jumps = np.random.poisson(lambda_dt)
            if n_jumps > 0:
                for _ in range(n_jumps):
                    log_return += mu_j + sigma_j * np.random.standard_normal()

            s = s * np.exp(log_return)

        terminals[i] = s

    return terminals


# =============================================================================
# Merton Simulator Class
# =============================================================================

class MertonSimulator(BaseSimulator):
    """
    Merton jump diffusion model simulator.

    Combines GBM diffusion with Poisson-driven lognormal jumps.

    Parameters
    ----------
    sigma : float
        Diffusion volatility (annualized)
    lambda_j : float
        Jump intensity (expected number of jumps per year)
    mu_j : float
        Mean of log-jump size
    sigma_j : float
        Standard deviation of log-jump size

    Examples
    --------
    simulator = MertonSimulator(
        sigma=0.15, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
    )
    result = simulator.simulate_paths(s0=100, mu=0.08, t=1.0, n_paths=10000, n_steps=252)
    """

    def __init__(
        self,
        sigma: float,
        lambda_j: float,
        mu_j: float,
        sigma_j: float
    ):
        super().__init__()
        self._model_name = "Merton Jump Diffusion"
        self._sigma = sigma
        self._lambda_j = lambda_j
        self._mu_j = mu_j
        self._sigma_j = sigma_j

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate model parameters."""
        if self._sigma <= 0:
            raise ValueError(f"Diffusion volatility sigma must be positive, got {self._sigma}")
        if self._lambda_j < 0:
            raise ValueError(f"Jump intensity lambda_j must be non-negative, got {self._lambda_j}")
        if self._sigma_j < 0:
            raise ValueError(f"Jump volatility sigma_j must be non-negative, got {self._sigma_j}")

    @property
    def sigma(self) -> float:
        return self._sigma

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
            "sigma": self._sigma,
            "lambda_j": self._lambda_j,
            "mu_j": self._mu_j,
            "sigma_j": self._sigma_j,
        }

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
        Simulate Merton jump diffusion paths.

        Returns
        -------
        SimulationResult
            Result with price paths (no volatility paths for this model)
        """
        self.validate_inputs(s0, mu, t, n_paths, n_steps)

        if seed is not None:
            np.random.seed(seed)

        start_time = time.perf_counter()

        paths = _simulate_merton_paths(
            s0, mu, self._sigma,
            self._lambda_j, self._mu_j, self._sigma_j,
            t, n_paths, n_steps
        )

        computation_time = time.perf_counter() - start_time
        time_grid = np.linspace(0, t, n_steps + 1)

        return SimulationResult(
            price_paths=paths,
            time_grid=time_grid,
            model_name=self._model_name,
            computation_time=computation_time,
            n_paths=n_paths,
            n_steps=n_steps,
            volatility_paths=None,
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

        return _simulate_merton_terminal(
            s0, mu, self._sigma,
            self._lambda_j, self._mu_j, self._sigma_j,
            t, n_paths, n_steps
        )


# =============================================================================
# Convenience Function
# =============================================================================

def simulate_merton(
    s0: float,
    mu: float,
    sigma: float,
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
    Convenience function for Merton jump diffusion simulation.

    Parameters
    ----------
    s0 : float
        Initial stock price
    mu : float
        Expected return (annualized)
    sigma : float
        Diffusion volatility
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
    simulator = MertonSimulator(
        sigma=sigma, lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j
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
    print("Merton Jump Diffusion Benchmark")
    print("=" * 60)

    # Test parameters
    s0, mu, t, n_steps = 100.0, 0.08, 1.0, 252
    sigma, lambda_j, mu_j, sigma_j = 0.15, 0.5, -0.1, 0.2

    # Warmup JIT compilation
    print("\nWarming up JIT compilation...")
    _ = simulate_merton(s0, mu, sigma, lambda_j, mu_j, sigma_j, t, n_paths=1000, n_steps=10, terminal_only=True)

    # Benchmark different path counts
    path_counts = [10_000, 50_000, 100_000, 500_000, 1_000_000]

    print(f"\n{'Paths':>12} {'Time (ms)':>12} {'Paths/sec':>15}")
    print("-" * 42)

    for n_paths in path_counts:
        start = time.perf_counter()
        result = simulate_merton(s0, mu, sigma, lambda_j, mu_j, sigma_j, t, n_paths=n_paths, n_steps=n_steps)
        elapsed = time.perf_counter() - start

        paths_per_sec = n_paths / elapsed
        print(f"{n_paths:>12,} {elapsed*1000:>12.2f} {paths_per_sec:>15,.0f}")

    print()
