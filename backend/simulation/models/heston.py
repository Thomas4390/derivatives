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
    ξ (xi/volvol): Volatility of volatility
    ρ (rho): Correlation between price and variance

Feller condition: 2·κ·θ > ξ² ensures variance stays positive.

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
    from ..enums import DiscretizationScheme
except ImportError:
    _project_root = Path(__file__).resolve().parents[3]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from backend.simulation.base import BaseSimulator, SimulationResult, StochasticVolatilityMixin
    from backend.simulation.enums import DiscretizationScheme


# =============================================================================
# Numba-Optimized Kernels
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _simulate_heston_paths(
    s0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    t: float,
    n_paths: int,
    n_steps: int,
    scheme: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba kernel for Heston path simulation.

    Scheme codes:
        0 = Euler (simple, can have negative variance)
        1 = Full truncation (variance floored at 0)
        2 = Reflection (negative variance reflected)
        3 = QE scheme (Quadratic Exponential - most accurate)
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    # Pre-compute correlation coefficients
    sqrt_one_minus_rho2 = np.sqrt(1.0 - rho * rho)

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

            if scheme == 0:
                # Simple Euler scheme
                sqrt_v = np.sqrt(max(v_curr, 0.0))
                v_next = v_curr + kappa * (theta - v_curr) * dt + xi * sqrt_v * dw_v
                s_next = s_curr * np.exp((mu - 0.5 * v_curr) * dt + sqrt_v * dw_s)

            elif scheme == 1:
                # Full truncation scheme
                v_plus = max(v_curr, 0.0)
                sqrt_v = np.sqrt(v_plus)
                v_next = v_curr + kappa * (theta - v_plus) * dt + xi * sqrt_v * dw_v
                v_next = max(v_next, 0.0)
                s_next = s_curr * np.exp((mu - 0.5 * v_plus) * dt + sqrt_v * dw_s)

            elif scheme == 2:
                # Reflection scheme
                v_plus = abs(v_curr)
                sqrt_v = np.sqrt(v_plus)
                v_next = v_curr + kappa * (theta - v_plus) * dt + xi * sqrt_v * dw_v
                v_next = abs(v_next)
                s_next = s_curr * np.exp((mu - 0.5 * v_plus) * dt + sqrt_v * dw_s)

            else:  # scheme == 3: QE scheme (simplified)
                v_plus = max(v_curr, 0.0)

                # Compute m (drift term) and s^2 (variance term)
                exp_kappa_dt = np.exp(-kappa * dt)
                m = theta + (v_plus - theta) * exp_kappa_dt
                s2 = (v_plus * xi * xi * exp_kappa_dt / kappa * (1.0 - exp_kappa_dt) +
                      theta * xi * xi / (2.0 * kappa) * (1.0 - exp_kappa_dt) ** 2)

                psi = s2 / (m * m) if m > 1e-10 else 1000.0

                if psi <= 1.5:
                    # Use moment-matched approximation
                    b2 = 2.0 / psi - 1.0 + np.sqrt(2.0 / psi * (2.0 / psi - 1.0))
                    a = m / (1.0 + b2)
                    z_v = np.random.standard_normal()
                    v_next = a * (np.sqrt(b2) + z_v) ** 2
                else:
                    # Use exponential approximation
                    p = (psi - 1.0) / (psi + 1.0)
                    beta = (1.0 - p) / m if m > 1e-10 else 1.0
                    u = np.random.random()
                    if u <= p:
                        v_next = 0.0
                    else:
                        v_next = np.log((1.0 - p) / (1.0 - u)) / beta

                sqrt_v = np.sqrt(v_plus)
                s_next = s_curr * np.exp((mu - 0.5 * v_plus) * dt + sqrt_v * dw_s)

            v_paths[i, j + 1] = v_next
            s_paths[i, j + 1] = s_next

    return s_paths, v_paths


@njit(parallel=True, cache=True, fastmath=True)
def _simulate_heston_terminal(
    s0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    t: float,
    n_paths: int,
    n_steps: int
) -> np.ndarray:
    """
    Numba kernel for terminal-only Heston simulation.

    Uses full truncation scheme. Memory efficient.
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)
    sqrt_one_minus_rho2 = np.sqrt(1.0 - rho * rho)

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
            s = s * np.exp((mu - 0.5 * v_plus) * dt + sqrt_v * dw_s)

        terminals[i] = s

    return terminals


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
    xi : float
        Volatility of volatility (vol of vol)
    rho : float
        Correlation between price and variance (typically negative)
    scheme : DiscretizationScheme
        Discretization scheme for variance process

    Examples
    --------
    simulator = HestonSimulator(
        v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7
    )
    result = simulator.simulate_paths(s0=100, mu=0.08, t=1.0, n_paths=10000, n_steps=252)
    print(f"Terminal vol mean: {np.sqrt(result.volatility_paths[:, -1]).mean()*100:.1f}%")
    """

    def __init__(
        self,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        scheme: DiscretizationScheme = DiscretizationScheme.FULL_TRUNCATION
    ):
        super().__init__()
        self._model_name = "Heston Stochastic Volatility"
        self._v0 = v0
        self._kappa = kappa
        self._theta = theta
        self._xi = xi
        self._rho = rho
        self._scheme = scheme

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate model parameters."""
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

    def get_parameters(self) -> Dict[str, Any]:
        """Returns model parameters."""
        return {
            "v0": self._v0,
            "kappa": self._kappa,
            "theta": self._theta,
            "xi": self._xi,
            "rho": self._rho,
            "scheme": self._scheme.name,
        }

    def long_run_variance(self) -> float:
        """Returns the theoretical long-run variance θ."""
        return self._theta

    def feller_condition_satisfied(self) -> bool:
        """Check if Feller condition (2κθ > ξ²) is satisfied."""
        return 2 * self._kappa * self._theta > self._xi ** 2

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
        seed: Optional[int] = None
    ) -> SimulationResult:
        """
        Simulate Heston price and variance paths.

        Returns
        -------
        SimulationResult
            Result with both price_paths and volatility_paths
        """
        self.validate_inputs(s0, mu, t, n_paths, n_steps)

        if seed is not None:
            np.random.seed(seed)

        start_time = time.perf_counter()

        s_paths, v_paths = _simulate_heston_paths(
            s0, self._v0, mu,
            self._kappa, self._theta, self._xi, self._rho,
            t, n_paths, n_steps,
            self._get_scheme_code()
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

        return _simulate_heston_terminal(
            s0, self._v0, mu,
            self._kappa, self._theta, self._xi, self._rho,
            t, n_paths, n_steps
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
    xi: float,
    rho: float,
    t: float,
    n_paths: int = 100000,
    n_steps: int = 252,
    seed: Optional[int] = None,
    scheme: DiscretizationScheme = DiscretizationScheme.FULL_TRUNCATION,
    terminal_only: bool = False
):
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
    xi : float
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
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, scheme=scheme
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
    print("Heston Stochastic Volatility Benchmark")
    print("=" * 60)

    # Test parameters
    s0, mu, t, n_steps = 100.0, 0.08, 1.0, 252
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7

    # Warmup JIT compilation
    print("\nWarming up JIT compilation...")
    _ = simulate_heston(s0, mu, v0, kappa, theta, xi, rho, t, n_paths=1000, n_steps=10, terminal_only=True)

    # Benchmark different path counts
    path_counts = [10_000, 50_000, 100_000, 500_000, 1_000_000]

    print(f"\n{'Paths':>12} {'Time (ms)':>12} {'Paths/sec':>15}")
    print("-" * 42)

    for n_paths in path_counts:
        start = time.perf_counter()
        result = simulate_heston(s0, mu, v0, kappa, theta, xi, rho, t, n_paths=n_paths, n_steps=n_steps)
        elapsed = time.perf_counter() - start

        paths_per_sec = n_paths / elapsed
        print(f"{n_paths:>12,} {elapsed*1000:>12.2f} {paths_per_sec:>15,.0f}")

    print()
