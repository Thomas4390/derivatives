"""
Geometric Brownian Motion (GBM) Simulator
==========================================

Implementation of the classic GBM model with Numba optimization.

Model Specification (P-measure):
    dS = μ·S·dt + σ·S·dW

Exact solution:
    S(t) = S(0) · exp((μ - 0.5·σ²)·t + σ·W(t))

For Q-measure (risk-neutral pricing), set μ = r (risk-free rate).

Author: Thomas
Created: 2025
"""

import time
import warnings
from typing import Any

import numpy as np
from numba import njit, prange

from backend.simulation.base import BaseSimulator, SimulationResult

# =============================================================================
# Numba-Optimized Kernels
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _simulate_gbm_paths(
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_paths: int,
    n_steps: int,
    antithetic: bool = True
) -> np.ndarray:
    """
    Numba kernel for GBM path simulation.

    Uses exact solution with antithetic variates for variance reduction.
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    # Pre-compute drift and diffusion coefficients
    drift = (mu - 0.5 * sigma * sigma) * dt
    diffusion = sigma * sqrt_dt

    if antithetic:
        half_paths = n_paths // 2
        paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)

        for i in prange(half_paths):
            paths[i, 0] = s0
            paths[i + half_paths, 0] = s0

            for j in range(n_steps):
                z = np.random.standard_normal()

                # Original path
                log_return = drift + diffusion * z
                paths[i, j + 1] = paths[i, j] * np.exp(log_return)

                # Antithetic path (use -z)
                log_return_anti = drift - diffusion * z
                paths[i + half_paths, j + 1] = paths[i + half_paths, j] * np.exp(log_return_anti)
    else:
        paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)

        for i in prange(n_paths):
            paths[i, 0] = s0

            for j in range(n_steps):
                z = np.random.standard_normal()
                log_return = drift + diffusion * z
                paths[i, j + 1] = paths[i, j] * np.exp(log_return)

    return paths


@njit(parallel=True, cache=True, fastmath=True)
def _simulate_gbm_terminal(
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_paths: int,
    n_steps: int,
    antithetic: bool = True
) -> np.ndarray:
    """
    Numba kernel for terminal-only GBM simulation.

    Memory efficient: only returns S(T), not full paths.
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    drift = (mu - 0.5 * sigma * sigma) * dt
    diffusion = sigma * sqrt_dt

    terminals = np.empty(n_paths, dtype=np.float64)

    if antithetic:
        half_paths = n_paths // 2

        for i in prange(half_paths):
            s = s0
            s_anti = s0

            for j in range(n_steps):
                z = np.random.standard_normal()
                s = s * np.exp(drift + diffusion * z)
                s_anti = s_anti * np.exp(drift - diffusion * z)

            terminals[i] = s
            terminals[i + half_paths] = s_anti
    else:
        for i in prange(n_paths):
            s = s0
            for j in range(n_steps):
                z = np.random.standard_normal()
                s = s * np.exp(drift + diffusion * z)
            terminals[i] = s

    return terminals


# =============================================================================
# GBM Simulator Class
# =============================================================================

class GBMSimulator(BaseSimulator):
    """
    Geometric Brownian Motion simulator.

    The classic Black-Scholes model with constant volatility.

    Parameters
    ----------
    sigma : float
        Volatility (annualized), e.g., 0.20 for 20%
    antithetic : bool
        Use antithetic variates for variance reduction (default True)

    Examples
    --------
    simulator = GBMSimulator(sigma=0.20)
    result = simulator.simulate_paths(s0=100, mu=0.08, t=1.0, n_paths=10000, n_steps=252)
    print(f"Terminal mean: ${result.terminal_mean:.2f}")
    """

    def __init__(self, sigma: float, antithetic: bool = True):
        super().__init__()
        self._model_name = "Geometric Brownian Motion"
        self._sigma = sigma
        self._antithetic = antithetic

        if sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {sigma}")

    @property
    def sigma(self) -> float:
        """Returns the volatility parameter."""
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {value}")
        self._sigma = value

    def get_parameters(self) -> dict[str, Any]:
        """Returns model parameters."""
        return {
            "sigma": self._sigma,
            "antithetic": self._antithetic,
        }

    def simulate_paths(
        self,
        s0: float,
        mu: float,
        t: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None
    ) -> SimulationResult:
        """
        Simulate full GBM price paths.

        Parameters
        ----------
        s0 : float
            Initial stock price
        mu : float
            Expected return (annualized)
        t : float
            Time horizon in years
        n_paths : int
            Number of Monte Carlo paths
        n_steps : int
            Number of time steps per path
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        SimulationResult
            Simulation results with price paths and metadata
        """
        self.validate_inputs(s0, mu, t, n_paths, n_steps)

        # Warn and adjust if antithetic is on with odd n_paths (but keep at least 2)
        if self._antithetic and n_paths % 2 != 0:
            adjusted_paths = max(n_paths - 1, 2)
            if adjusted_paths != n_paths:
                warnings.warn(
                    f"Antithetic sampling requires even n_paths. "
                    f"Using {adjusted_paths} instead of {n_paths}.",
                    UserWarning
                )
                n_paths = adjusted_paths

        if seed is not None:
            np.random.seed(seed)

        start_time = time.perf_counter()

        paths = _simulate_gbm_paths(
            s0, mu, self._sigma, t, n_paths, n_steps, self._antithetic
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
        seed: int | None = None
    ) -> np.ndarray:
        """
        Simulate only terminal values S(T).

        Memory efficient version for European option pricing.

        Parameters
        ----------
        s0 : float
            Initial stock price
        mu : float
            Expected return (annualized)
        t : float
            Time horizon in years
        n_paths : int
            Number of Monte Carlo paths
        n_steps : int
            Number of time steps per path
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        np.ndarray
            Terminal prices S(T), shape (n_paths,)
        """
        self.validate_inputs(s0, mu, t, n_paths, n_steps)

        # Warn and adjust if antithetic is on with odd n_paths (but keep at least 2)
        if self._antithetic and n_paths % 2 != 0:
            adjusted_paths = max(n_paths - 1, 2)
            if adjusted_paths != n_paths:
                warnings.warn(
                    f"Antithetic sampling requires even n_paths. "
                    f"Using {adjusted_paths} instead of {n_paths}.",
                    UserWarning
                )
                n_paths = adjusted_paths

        if seed is not None:
            np.random.seed(seed)

        return _simulate_gbm_terminal(
            s0, mu, self._sigma, t, n_paths, n_steps, self._antithetic
        )


# =============================================================================
# Convenience Function
# =============================================================================

def simulate_gbm(
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_paths: int = 100000,
    n_steps: int = 252,
    seed: int | None = None,
    antithetic: bool = True,
    terminal_only: bool = False
):
    """
    Convenience function for GBM simulation.

    Parameters
    ----------
    s0 : float
        Initial stock price
    mu : float
        Expected return (annualized)
    sigma : float
        Volatility (annualized)
    t : float
        Time horizon in years
    n_paths : int
        Number of paths (default 100,000)
    n_steps : int
        Number of steps (default 252)
    seed : int, optional
        Random seed
    antithetic : bool
        Use antithetic variates (default True)
    terminal_only : bool
        Return only terminal values (default False)

    Returns
    -------
    SimulationResult or np.ndarray
        Full result or terminal values if terminal_only=True
    """
    simulator = GBMSimulator(sigma=sigma, antithetic=antithetic)

    if terminal_only:
        return simulator.simulate_terminal(s0, mu, t, n_paths, n_steps, seed)
    return simulator.simulate_paths(s0, mu, t, n_paths, n_steps, seed)


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GBM (Geometric Brownian Motion) Benchmark")
    print("=" * 60)

    # Test parameters
    s0, mu, sigma, t, n_steps = 100.0, 0.08, 0.20, 1.0, 252

    # Warmup JIT compilation
    print("\nWarming up JIT compilation...")
    _ = simulate_gbm(s0, mu, sigma, t, n_paths=1000, n_steps=10, terminal_only=True)

    # Benchmark different path counts
    path_counts = [10_000, 50_000, 100_000, 500_000]

    print(f"\n{'Paths':>12} {'Time (ms)':>12} {'Paths/sec':>15}")
    print("-" * 42)

    for n_paths in path_counts:
        start = time.perf_counter()
        result = simulate_gbm(s0, mu, sigma, t, n_paths=n_paths, n_steps=n_steps)
        elapsed = time.perf_counter() - start

        paths_per_sec = n_paths / elapsed
        print(f"{n_paths:>12,} {elapsed*1000:>12.2f} {paths_per_sec:>15,.0f}")

    print()
