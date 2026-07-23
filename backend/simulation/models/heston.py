"""
Heston Stochastic Volatility Simulator
======================================

Implementation of the Heston (1993) model with Numba optimization.

Model Specification (P-measure):
    dS = μ·S·dt + √V·S·dW_S
    dV = κ·(θ - V)·dt + ξ·√V·dW_V

    Corr(dW_S, dW_V) = ρ

Parameters:
    κ (kappa): Mean reversion speed of variance
    θ (theta): Long-run variance level
    ξ (alpha/volvol): Volatility of volatility
    ρ (rho): Correlation between price and variance

Feller condition: 2·κ·θ > ξ² ensures variance stays positive.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from backend.simulation.base import (
    BaseSimulator,
    SimulationResult,
    StochasticVolatilityMixin,
)
from backend.simulation.enums import DiscretizationScheme
from backend.simulation.models._heston_kernels import (
    heston_paths_euler,
    heston_paths_full_truncation,
    heston_paths_qe,
    heston_paths_reflection,
    heston_terminal_full_truncation,
)

# =============================================================================
# Heston Simulator Class
# =============================================================================


class HestonSimulator(BaseSimulator, StochasticVolatilityMixin):
    """
    Heston stochastic volatility model simulator.

    Parameters
    ----------
    v0 : float
        Initial variance (σ²_0), e.g., 0.04 for 20% vol
    kappa : float
        Mean reversion speed (typical: 1-5)
    theta : float
        Long-run variance level
    alpha : float
        Volatility of volatility (vol of vol)
    rho : float
        Correlation between price and variance (typically negative)
    scheme : DiscretizationScheme
        Discretization scheme for variance process

    Examples
    --------
    simulator = HestonSimulator(
        v0=0.04, kappa=2.0, theta=0.04, alpha=0.3, rho=-0.7
    )
    result = simulator.simulate_paths(s0=100, mu=0.08, t=1.0, n_paths=10000, n_steps=252)
    print(f"Terminal vol mean: {np.sqrt(result.volatility_paths[:, -1]).mean()*100:.1f}%")
    """

    def __init__(
        self,
        v0: float,
        kappa: float,
        theta: float,
        alpha: float,
        rho: float,
        scheme: DiscretizationScheme = DiscretizationScheme.FULL_TRUNCATION,
        antithetic: bool = True,
    ) -> None:
        super().__init__()
        self._model_name = "Heston Stochastic Volatility"
        self._v0 = v0
        self._kappa = kappa
        self._theta = theta
        self._xi = alpha
        self._rho = rho
        self._scheme = scheme
        self._antithetic = antithetic

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate model parameters."""
        if self._v0 < 0:
            raise ValueError(
                f"Initial variance v0 must be non-negative, got {self._v0}"
            )
        if self._kappa <= 0:
            raise ValueError(
                f"Mean reversion kappa must be positive, got {self._kappa}"
            )
        if self._theta < 0:
            raise ValueError(
                f"Long-run variance theta must be non-negative, got {self._theta}"
            )
        if self._xi <= 0:
            raise ValueError(f"Vol of vol alpha must be positive, got {self._xi}")
        if not -1 <= self._rho <= 1:
            raise ValueError(f"Correlation rho must be in [-1, 1], got {self._rho}")

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
    def alpha(self) -> float:
        return self._xi

    @property
    def rho(self) -> float:
        return self._rho

    def get_parameters(self) -> dict[str, Any]:
        """Returns model parameters."""
        return {
            "v0": self._v0,
            "kappa": self._kappa,
            "theta": self._theta,
            "alpha": self._xi,
            "rho": self._rho,
            "scheme": self._scheme.name,
        }

    def long_run_variance(self) -> float:
        """Returns the theoretical long-run variance θ."""
        return self._theta

    def feller_condition_satisfied(self) -> bool:
        """Check if Feller condition (2κθ > ξ²) is satisfied."""
        return 2 * self._kappa * self._theta > self._xi**2

    def _get_scheme_code(self) -> int:
        """Convert scheme enum to integer code for Numba kernel."""
        scheme_map = {
            DiscretizationScheme.EULER: 0,
            DiscretizationScheme.FULL_TRUNCATION: 1,
            DiscretizationScheme.REFLECTION: 2,
            DiscretizationScheme.QE: 3,
        }
        return scheme_map.get(self._scheme, 1)

    def simulate_paths(
        self,
        s0: float,
        mu: float,
        t: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
    ) -> SimulationResult:
        """
        Simulate Heston price and variance paths.

        Returns
        -------
        SimulationResult
            Result with both price_paths and volatility_paths
        """
        self.validate_inputs(s0, mu, t, n_paths, n_steps)

        rng = np.random.default_rng(seed)
        z1_noise = rng.standard_normal((n_paths, n_steps))
        z2_noise = rng.standard_normal((n_paths, n_steps))

        scheme_code = self._get_scheme_code()
        kernel_args = (
            s0,
            self._v0,
            mu,
            self._kappa,
            self._theta,
            self._xi,
            self._rho,
            t,
            n_paths,
            n_steps,
            z1_noise,
            z2_noise,
        )

        # Only QE draws its two extra noise arrays (outside the timed region,
        # like the z draws above).
        qe_noise: tuple[np.ndarray, np.ndarray] | None = None
        if scheme_code == 3:
            qe_noise = (
                rng.standard_normal((n_paths, n_steps)),
                rng.random((n_paths, n_steps)),
            )

        start_time = time.perf_counter()

        # One specialized kernel per scheme (top-level njit functions, so each
        # compiles once).
        if qe_noise is not None:
            s_paths, v_paths = heston_paths_qe(*kernel_args, *qe_noise)
        elif scheme_code == 0:
            s_paths, v_paths = heston_paths_euler(*kernel_args)
        elif scheme_code == 2:
            s_paths, v_paths = heston_paths_reflection(*kernel_args)
        else:
            s_paths, v_paths = heston_paths_full_truncation(*kernel_args)

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
            parameters=self.get_parameters() | {"s0": s0, "mu": mu, "t": t},
        )

    def simulate_terminal(
        self,
        s0: float,
        mu: float,
        t: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Simulate only terminal prices S(T).

        Returns
        -------
        np.ndarray
            Terminal prices, shape (n_paths,)
        """
        self.validate_inputs(s0, mu, t, n_paths, n_steps)

        rng = np.random.default_rng(seed)
        z1_noise = rng.standard_normal((n_paths, n_steps))
        z2_noise = rng.standard_normal((n_paths, n_steps))

        return heston_terminal_full_truncation(
            s0,
            self._v0,
            mu,
            self._kappa,
            self._theta,
            self._xi,
            self._rho,
            t,
            n_paths,
            n_steps,
            z1_noise,
            z2_noise,
        )


# =============================================================================
# Convenience Function
# =============================================================================


def simulate_heston(
    s0: float,
    mu: float,
    v0: float,
    kappa: float,
    theta: float,
    alpha: float,
    rho: float,
    t: float,
    n_paths: int = 100000,
    n_steps: int = 252,
    seed: int | None = None,
    scheme: DiscretizationScheme = DiscretizationScheme.FULL_TRUNCATION,
    terminal_only: bool = False,
) -> SimulationResult | np.ndarray:
    """
    Convenience function for Heston simulation.

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
    alpha : float
        Volatility of volatility
    rho : float
        Correlation
    t : float
        Time horizon in years
    n_paths : int
        Number of paths
    n_steps : int
        Number of steps
    seed : int, optional
        Random seed
    scheme : DiscretizationScheme
        Variance discretization scheme
    terminal_only : bool
        Return only terminal values

    Returns
    -------
    SimulationResult or np.ndarray
        Full result or terminal values if terminal_only=True
    """
    simulator = HestonSimulator(
        v0=v0, kappa=kappa, theta=theta, alpha=alpha, rho=rho, scheme=scheme
    )

    if terminal_only:
        return simulator.simulate_terminal(s0, mu, t, n_paths, n_steps, seed)
    return simulator.simulate_paths(s0, mu, t, n_paths, n_steps, seed)


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Heston Stochastic Volatility Benchmark")
    print("=" * 60)

    # Test parameters
    s0, mu, t, n_steps = 100.0, 0.08, 1.0, 252
    v0, kappa, theta, alpha, rho = 0.04, 2.0, 0.04, 0.3, -0.7

    # Warmup JIT compilation
    print("\nWarming up JIT compilation...")
    _ = simulate_heston(
        s0,
        mu,
        v0,
        kappa,
        theta,
        alpha,
        rho,
        t,
        n_paths=1000,
        n_steps=10,
        terminal_only=True,
    )

    # Benchmark different path counts
    path_counts = [10_000, 50_000, 100_000, 500_000]

    print(f"\n{'Paths':>12} {'Time (ms)':>12} {'Paths/sec':>15}")
    print("-" * 42)

    for n_paths in path_counts:
        start = time.perf_counter()
        result = simulate_heston(
            s0, mu, v0, kappa, theta, alpha, rho, t, n_paths=n_paths, n_steps=n_steps
        )
        elapsed = time.perf_counter() - start

        paths_per_sec = n_paths / elapsed
        print(f"{n_paths:>12,} {elapsed * 1000:>12.2f} {paths_per_sec:>15,.0f}")

    print()
