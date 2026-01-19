"""
Monte Carlo Path Simulation Module - Ultra-Optimized with Numba
===============================================================

This module provides high-performance Monte Carlo simulation for various
stochastic processes under the PHYSICAL MEASURE (P-measure).

P-Measure vs Q-Measure:
-----------------------
This module simulates under the REAL-WORLD (P) measure, which is appropriate for:
    - Scenario analysis and risk management
    - VaR and stress testing
    - Backtesting trading strategies
    - Generating realistic price trajectories

For DERIVATIVES PRICING, use the Q-measure (risk-neutral) where drift = r.
The P-measure uses drift = μ (expected return), capturing real-world dynamics.

Reference: https://quantnet.com/threads/p-measure-vs-q-measure.3227/

Models Implemented:
    1. Geometric Brownian Motion (GBM)
    2. Heston Stochastic Volatility
    3. Merton Jump Diffusion
    4. SABR Stochastic Volatility

Performance Optimizations:
    - Numba JIT compilation with nopython mode
    - Parallel execution with prange
    - Cache-friendly memory access patterns
    - Pre-allocated arrays to avoid memory allocation overhead
    - Vectorized random number generation

Author: Derivatives Pricing Project
"""

import numpy as np
from numba import njit, prange
import time
import warnings
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Configuration and Constants
# =============================================================================

class ModelType(Enum):
    """Enumeration of available stochastic models."""
    GBM = "Geometric Brownian Motion"
    HESTON = "Heston Stochastic Volatility"
    MERTON_JUMP = "Merton Jump Diffusion"
    BATES = "Bates Model"
    SABR = "SABR Model"


@dataclass
class SimulationResult:
    """Container for simulation results with metadata."""
    paths: np.ndarray
    time_grid: np.ndarray
    model: str
    computation_time: float
    num_paths: int
    num_steps: int

    @property
    def terminal_values(self) -> np.ndarray:
        """Returns the terminal values of all paths."""
        return self.paths[:, -1]

    @property
    def mean_path(self) -> np.ndarray:
        """Returns the mean path across all simulations."""
        return np.mean(self.paths, axis=0)

    @property
    def std_path(self) -> np.ndarray:
        """Returns the standard deviation path across all simulations."""
        return np.std(self.paths, axis=0)

    def percentile_paths(self, percentiles: list) -> np.ndarray:
        """Returns percentile paths (e.g., [5, 50, 95])."""
        return np.percentile(self.paths, percentiles, axis=0)


# =============================================================================
# Random Number Generation Utilities
# =============================================================================

@njit(cache=True)
def _set_seed(seed: int) -> None:
    """Set the random seed for reproducibility within Numba context."""
    np.random.seed(seed)


@njit(cache=True, fastmath=True)
def _generate_standard_normals(n_paths: int, n_steps: int) -> np.ndarray:
    """
    Generate standard normal random variates efficiently.

    Uses pre-allocation for optimal memory usage.
    """
    return np.random.standard_normal((n_paths, n_steps))


@njit(cache=True, fastmath=True)
def _generate_correlated_normals(
    n_paths: int,
    n_steps: int,
    rho: float
) -> np.ndarray:
    """
    Generate two correlated standard normal sequences.

    Returns array of shape (n_paths, n_steps, 2) where the two sequences
    have correlation coefficient rho.
    """
    z1 = np.random.standard_normal((n_paths, n_steps))
    z2 = np.random.standard_normal((n_paths, n_steps))

    # Cholesky decomposition for correlation
    z2_correlated = rho * z1 + np.sqrt(1.0 - rho * rho) * z2

    result = np.empty((n_paths, n_steps, 2), dtype=np.float64)
    result[:, :, 0] = z1
    result[:, :, 1] = z2_correlated

    return result


# =============================================================================
# Model 1: Geometric Brownian Motion (Black-Scholes)
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def simulate_gbm_paths(
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_paths: int,
    n_steps: int,
    antithetic: bool = True
) -> np.ndarray:
    """
    Simulate price paths using Geometric Brownian Motion under P-measure.

    P-MEASURE (Real World):
        dS = μ*S*dt + σ*S*dW

    Uses exact solution: S(t) = S(0) * exp((μ - 0.5*σ²)*t + σ*W(t))

    Note: For risk-neutral pricing (Q-measure), set μ = r (risk-free rate).

    Parameters
    ----------
    s0 : float
        Initial stock price
    mu : float
        Expected return / drift (annualized) - use risk-free rate r for Q-measure
    sigma : float
        Volatility (annualized)
    t : float
        Time to maturity (in years)
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps per path
    antithetic : bool
        Use antithetic variates for variance reduction

    Returns
    -------
    np.ndarray
        Array of shape (n_paths, n_steps + 1) containing simulated paths
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    # Pre-compute drift and diffusion coefficients
    drift = (mu - 0.5 * sigma * sigma) * dt
    diffusion = sigma * sqrt_dt

    if antithetic:
        # Generate half the paths, then use antithetic variates
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
def simulate_gbm_paths_vectorized(
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_paths: int,
    n_steps: int
) -> np.ndarray:
    """
    Vectorized version of GBM simulation under P-measure.

    Optimized for very large n_paths. Generates all random numbers at once
    and uses cumulative sum for better memory locality.

    Parameters
    ----------
    mu : float
        Expected return / drift (annualized) - use risk-free rate r for Q-measure
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    drift = (mu - 0.5 * sigma * sigma) * dt
    diffusion = sigma * sqrt_dt

    # Generate all random numbers at once
    z = np.random.standard_normal((n_paths, n_steps))

    # Compute log returns
    log_returns = drift + diffusion * z

    # Build paths using cumulative sum
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = s0

    # Cumulative sum of log returns
    for i in prange(n_paths):
        cumsum = 0.0
        for j in range(n_steps):
            cumsum += log_returns[i, j]
            paths[i, j + 1] = s0 * np.exp(cumsum)

    return paths


# =============================================================================
# Model 2: Heston Stochastic Volatility Model
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def simulate_heston_paths(
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
    scheme: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate price paths using the Heston stochastic volatility model under P-measure.

    P-MEASURE (Real World):
        dS = μ*S*dt + sqrt(V)*S*dW_S
        dV = κ*(θ - V)*dt + ξ*sqrt(V)*dW_V

    Corr(dW_S, dW_V) = ρ

    Note: For risk-neutral pricing (Q-measure), set μ = r (risk-free rate).

    Parameters
    ----------
    s0 : float
        Initial stock price
    v0 : float
        Initial variance (sigma^2, not sigma)
    mu : float
        Expected return / drift (annualized) - use risk-free rate r for Q-measure
    kappa : float
        Mean reversion speed of variance
    theta : float
        Long-term variance level
    xi : float
        Volatility of volatility (vol of vol)
    rho : float
        Correlation between stock and variance Brownian motions
    t : float
        Time to maturity (in years)
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps
    scheme : int
        Discretization scheme:
        0 = Euler (simple, can have negative variance)
        1 = Full truncation (variance floored at 0)
        2 = Reflection (negative variance reflected)
        3 = QE scheme (Quadratic Exponential - most accurate)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (price_paths, variance_paths) each of shape (n_paths, n_steps + 1)
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
                # Quadratic-Exponential scheme for variance
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


@njit(cache=True, fastmath=True)
def simulate_heston_single_path(
    s0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    t: float,
    n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single Heston path under P-measure - useful for visualization.

    Parameters
    ----------
    mu : float
        Expected return / drift (annualized) - use risk-free rate r for Q-measure
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)
    sqrt_one_minus_rho2 = np.sqrt(1.0 - rho * rho)

    s_path = np.empty(n_steps + 1, dtype=np.float64)
    v_path = np.empty(n_steps + 1, dtype=np.float64)

    s_path[0] = s0
    v_path[0] = v0

    for j in range(n_steps):
        z1 = np.random.standard_normal()
        z2 = np.random.standard_normal()

        dw_s = sqrt_dt * z1
        dw_v = sqrt_dt * (rho * z1 + sqrt_one_minus_rho2 * z2)

        v_curr = max(v_path[j], 0.0)
        sqrt_v = np.sqrt(v_curr)

        v_path[j + 1] = max(v_path[j] + kappa * (theta - v_curr) * dt + xi * sqrt_v * dw_v, 0.0)
        s_path[j + 1] = s_path[j] * np.exp((mu - 0.5 * v_curr) * dt + sqrt_v * dw_s)

    return s_path, v_path


# =============================================================================
# Model 3: Merton Jump Diffusion Model
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def simulate_merton_jump_paths(
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
    Simulate price paths using the Merton Jump Diffusion model under P-measure.

    P-MEASURE (Real World):
        dS/S = (μ - λ*k)*dt + σ*dW + (J-1)*dN

    Where:
    - dN is a Poisson process with intensity λ
    - J is lognormally distributed: ln(J) ~ N(mu_j, sigma_j²)
    - k = E[J-1] = exp(mu_j + 0.5*sigma_j²) - 1

    Note: For risk-neutral pricing (Q-measure), set μ = r (risk-free rate).

    Parameters
    ----------
    s0 : float
        Initial stock price
    mu : float
        Expected return / drift (annualized) - use risk-free rate r for Q-measure
    sigma : float
        Diffusion volatility
    lambda_j : float
        Jump intensity (expected number of jumps per year)
    mu_j : float
        Mean of log-jump size
    sigma_j : float
        Std dev of log-jump size
    t : float
        Time to maturity (in years)
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps

    Returns
    -------
    np.ndarray
        Array of shape (n_paths, n_steps + 1) containing simulated paths
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


# =============================================================================
# Model 4: Bates Model (Heston + Jumps)
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def simulate_bates_paths(
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
    n_paths: int,
    n_steps: int,
    antithetic: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate price paths using the Bates model (Heston + Jump Diffusion).

    The Bates model combines Heston stochastic volatility with Merton-style jumps:

    dS = (μ - λ·k)·S·dt + √V·S·dW_S + (J-1)·S·dN
    dV = κ(θ - V)dt + ξ·√V·dW_V

    Where:
    - k = E[J-1] = exp(μ_j + σ_j²/2) - 1 (compensator)
    - J = exp(μ_j + σ_j·Z) is the log-normal jump size
    - N is a Poisson process with intensity λ
    - Corr(dW_S, dW_V) = ρ

    Parameters
    ----------
    s0 : float
        Initial stock price
    mu : float
        Expected return (drift under P-measure)
    v0 : float
        Initial variance
    kappa : float
        Mean reversion speed for variance
    theta : float
        Long-term variance level
    xi : float
        Volatility of variance (vol of vol)
    rho : float
        Correlation between price and variance Brownian motions
    lambda_j : float
        Jump intensity (expected number of jumps per year)
    mu_j : float
        Mean of log-jump size
    sigma_j : float
        Standard deviation of log-jump size
    t : float
        Time horizon (in years)
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps
    antithetic : bool
        Use antithetic variates for variance reduction

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (price_paths, variance_paths) each of shape (n_paths, n_steps + 1)
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    # Jump compensator: k = E[J-1] = exp(mu_j + sigma_j^2/2) - 1
    k = np.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0

    # Adjusted drift for jump compensation
    drift_adj = mu - lambda_j * k

    sqrt_one_minus_rho2 = np.sqrt(1.0 - rho * rho)

    if antithetic:
        half_paths = n_paths // 2
        actual_paths = half_paths * 2
    else:
        actual_paths = n_paths

    paths = np.empty((actual_paths, n_steps + 1), dtype=np.float64)
    var_paths = np.empty((actual_paths, n_steps + 1), dtype=np.float64)

    for i in prange(actual_paths // 2 if antithetic else actual_paths):
        # Initialize
        paths[i, 0] = s0
        var_paths[i, 0] = v0

        if antithetic:
            paths[i + half_paths, 0] = s0
            var_paths[i + half_paths, 0] = v0

        for j in range(n_steps):
            # Generate correlated Brownian increments
            z1 = np.random.standard_normal()
            z2 = np.random.standard_normal()

            dw_s = sqrt_dt * z1
            dw_v = sqrt_dt * (rho * z1 + sqrt_one_minus_rho2 * z2)

            # Current values (truncated scheme for variance)
            v_curr = max(var_paths[i, j], 0.0)
            sqrt_v = np.sqrt(v_curr)

            # Variance process (Euler-Maruyama with truncation)
            v_new = v_curr + kappa * (theta - v_curr) * dt + xi * sqrt_v * dw_v
            var_paths[i, j + 1] = max(v_new, 0.0)

            # Price process with diffusion
            log_return = (drift_adj - 0.5 * v_curr) * dt + sqrt_v * dw_s

            # Jump component (Poisson)
            n_jumps = np.random.poisson(lambda_j * dt)
            if n_jumps > 0:
                jump_sum = 0.0
                for _ in range(n_jumps):
                    jump_sum += mu_j + sigma_j * np.random.standard_normal()
                log_return += jump_sum

            paths[i, j + 1] = paths[i, j] * np.exp(log_return)

            # Antithetic path
            if antithetic:
                idx_anti = i + half_paths

                dw_s_anti = -dw_s
                dw_v_anti = -dw_v

                v_curr_anti = max(var_paths[idx_anti, j], 0.0)
                sqrt_v_anti = np.sqrt(v_curr_anti)

                v_new_anti = v_curr_anti + kappa * (theta - v_curr_anti) * dt + xi * sqrt_v_anti * dw_v_anti
                var_paths[idx_anti, j + 1] = max(v_new_anti, 0.0)

                log_return_anti = (drift_adj - 0.5 * v_curr_anti) * dt + sqrt_v_anti * dw_s_anti

                # Same jump realization for antithetic (to maintain correlation)
                if n_jumps > 0:
                    log_return_anti += jump_sum

                paths[idx_anti, j + 1] = paths[idx_anti, j] * np.exp(log_return_anti)

    return paths, var_paths


@njit(parallel=True, cache=True, fastmath=True)
def simulate_bates_terminal(
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
    n_paths: int,
    n_steps: int,
    antithetic: bool = True
) -> np.ndarray:
    """
    Simulate terminal prices only using the Bates model.

    This is optimized for P&L calculations where only final prices are needed.

    Parameters
    ----------
    s0 : float
        Initial stock price
    mu : float
        Expected return (drift under P-measure)
    v0 : float
        Initial variance
    kappa : float
        Mean reversion speed for variance
    theta : float
        Long-term variance level
    xi : float
        Volatility of variance (vol of vol)
    rho : float
        Correlation between price and variance Brownian motions
    lambda_j : float
        Jump intensity (expected number of jumps per year)
    mu_j : float
        Mean of log-jump size
    sigma_j : float
        Standard deviation of log-jump size
    t : float
        Time horizon (in years)
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps
    antithetic : bool
        Use antithetic variates for variance reduction

    Returns
    -------
    np.ndarray
        Terminal prices of shape (n_paths,)
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    # Jump compensator
    k = np.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
    drift_adj = mu - lambda_j * k

    sqrt_one_minus_rho2 = np.sqrt(1.0 - rho * rho)

    if antithetic:
        half_paths = n_paths // 2
        actual_paths = half_paths * 2
    else:
        actual_paths = n_paths

    terminal_prices = np.empty(actual_paths, dtype=np.float64)

    for i in prange(actual_paths // 2 if antithetic else actual_paths):
        # Path 1
        s = s0
        v = v0

        # Path 2 (antithetic)
        if antithetic:
            s_anti = s0
            v_anti = v0

        for j in range(n_steps):
            z1 = np.random.standard_normal()
            z2 = np.random.standard_normal()

            dw_s = sqrt_dt * z1
            dw_v = sqrt_dt * (rho * z1 + sqrt_one_minus_rho2 * z2)

            # Path 1
            v_curr = max(v, 0.0)
            sqrt_v = np.sqrt(v_curr)

            v = v_curr + kappa * (theta - v_curr) * dt + xi * sqrt_v * dw_v
            v = max(v, 0.0)

            log_return = (drift_adj - 0.5 * v_curr) * dt + sqrt_v * dw_s

            n_jumps = np.random.poisson(lambda_j * dt)
            jump_sum = 0.0
            if n_jumps > 0:
                for _ in range(n_jumps):
                    jump_sum += mu_j + sigma_j * np.random.standard_normal()
                log_return += jump_sum

            s = s * np.exp(log_return)

            # Antithetic path
            if antithetic:
                v_curr_anti = max(v_anti, 0.0)
                sqrt_v_anti = np.sqrt(v_curr_anti)

                v_anti = v_curr_anti + kappa * (theta - v_curr_anti) * dt + xi * sqrt_v_anti * (-dw_v)
                v_anti = max(v_anti, 0.0)

                log_return_anti = (drift_adj - 0.5 * v_curr_anti) * dt + sqrt_v_anti * (-dw_s)
                if n_jumps > 0:
                    log_return_anti += jump_sum

                s_anti = s_anti * np.exp(log_return_anti)

        terminal_prices[i] = s
        if antithetic:
            terminal_prices[i + half_paths] = s_anti

    return terminal_prices


# =============================================================================
# Model 5: SABR Model
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def simulate_sabr_paths(
    f0: float,
    alpha0: float,
    beta: float,
    rho: float,
    nu: float,
    t: float,
    n_paths: int,
    n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate forward price paths using the SABR model.

    dF = alpha * F^beta * dW_F
    d(alpha) = nu * alpha * dW_alpha

    Corr(dW_F, dW_alpha) = rho

    Parameters
    ----------
    f0 : float
        Initial forward price
    alpha0 : float
        Initial volatility level
    beta : float
        CEV exponent (0 = normal, 1 = lognormal)
    rho : float
        Correlation between forward and volatility
    nu : float
        Volatility of volatility
    t : float
        Time to maturity (in years)
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (forward_paths, volatility_paths) each of shape (n_paths, n_steps + 1)
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    sqrt_one_minus_rho2 = np.sqrt(1.0 - rho * rho)

    f_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    alpha_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)

    for i in prange(n_paths):
        f_paths[i, 0] = f0
        alpha_paths[i, 0] = alpha0

        for j in range(n_steps):
            z1 = np.random.standard_normal()
            z2 = np.random.standard_normal()

            dw_f = sqrt_dt * z1
            dw_alpha = sqrt_dt * (rho * z1 + sqrt_one_minus_rho2 * z2)

            f_curr = max(f_paths[i, j], 1e-10)  # Avoid zero/negative
            alpha_curr = max(alpha_paths[i, j], 1e-10)

            # Euler discretization
            f_beta = f_curr ** beta

            f_next = f_curr + alpha_curr * f_beta * dw_f
            alpha_next = alpha_curr * np.exp(-0.5 * nu * nu * dt + nu * dw_alpha)

            f_paths[i, j + 1] = max(f_next, 1e-10)
            alpha_paths[i, j + 1] = alpha_next

    return f_paths, alpha_paths


# =============================================================================
# Multi-Asset Simulation (Correlated GBM)
# =============================================================================

def validate_correlation_matrix(corr_matrix: np.ndarray) -> None:
    """
    Validate that a correlation matrix is valid for Cholesky decomposition.

    A valid correlation matrix must:
    1. Have diagonal elements equal to 1.0
    2. Be symmetric
    3. Be positive semi-definite (all eigenvalues >= 0)

    Parameters
    ----------
    corr_matrix : np.ndarray
        Correlation matrix to validate, shape (n, n)

    Raises
    ------
    ValueError
        If the correlation matrix is invalid

    Notes
    -----
    This validation should be called BEFORE using Cholesky decomposition
    to provide clear error messages instead of numerical failures.
    """
    # Check square matrix
    if corr_matrix.ndim != 2 or corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError(
            f"Correlation matrix must be square, got shape {corr_matrix.shape}"
        )

    n = corr_matrix.shape[0]

    # Check diagonal elements are 1.0
    diag = np.diag(corr_matrix)
    if not np.allclose(diag, 1.0, atol=1e-10):
        raise ValueError(
            f"Correlation matrix diagonal must be 1.0, got {diag}"
        )

    # Check symmetry
    if not np.allclose(corr_matrix, corr_matrix.T, atol=1e-10):
        max_diff = np.max(np.abs(corr_matrix - corr_matrix.T))
        raise ValueError(
            f"Correlation matrix must be symmetric, max asymmetry: {max_diff:.2e}"
        )

    # Check positive semi-definite (eigenvalues >= 0)
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    min_eigenvalue = np.min(eigenvalues)
    if min_eigenvalue < -1e-10:
        raise ValueError(
            f"Correlation matrix must be positive semi-definite. "
            f"Minimum eigenvalue: {min_eigenvalue:.6f}. "
            f"Consider using nearest correlation matrix approximation."
        )

    # Check off-diagonal elements in valid range
    for i in range(n):
        for j in range(i + 1, n):
            if corr_matrix[i, j] < -1.0 or corr_matrix[i, j] > 1.0:
                raise ValueError(
                    f"Correlation coefficient must be in [-1, 1], "
                    f"got corr[{i},{j}] = {corr_matrix[i, j]}"
                )


@njit(cache=True, fastmath=True)
def _cholesky_decomposition(corr_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of correlation matrix.
    Returns lower triangular matrix L such that corr_matrix = L @ L.T
    """
    n = corr_matrix.shape[0]
    L = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                sum_sq = 0.0
                for k in range(j):
                    sum_sq += L[j, k] ** 2
                L[i, j] = np.sqrt(corr_matrix[i, i] - sum_sq)
            else:
                sum_prod = 0.0
                for k in range(j):
                    sum_prod += L[i, k] * L[j, k]
                L[i, j] = (corr_matrix[i, j] - sum_prod) / L[j, j]

    return L


@njit(parallel=True, cache=True, fastmath=True)
def simulate_correlated_gbm_paths(
    s0: np.ndarray,
    mu: float,
    sigmas: np.ndarray,
    corr_matrix: np.ndarray,
    t: float,
    n_paths: int,
    n_steps: int
) -> np.ndarray:
    """
    Simulate correlated GBM paths for multiple assets under P-measure.

    IMPORTANT: Call validate_correlation_matrix(corr_matrix) before this function
    to ensure the correlation matrix is valid for Cholesky decomposition.
    Invalid matrices will cause numerical errors or incorrect results.

    Parameters
    ----------
    s0 : np.ndarray
        Initial prices for each asset, shape (n_assets,)
    mu : float
        Expected return / drift (annualized) - use risk-free rate r for Q-measure
    sigmas : np.ndarray
        Volatilities for each asset, shape (n_assets,)
    corr_matrix : np.ndarray
        Correlation matrix, shape (n_assets, n_assets).
        Must be symmetric, positive semi-definite, with diagonal = 1.0.
    t : float
        Time to maturity
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps

    Returns
    -------
    np.ndarray
        Array of shape (n_paths, n_steps + 1, n_assets)

    See Also
    --------
    validate_correlation_matrix : Validate correlation matrix before simulation
    """
    n_assets = len(s0)
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    # Cholesky decomposition for correlation
    L = _cholesky_decomposition(corr_matrix)

    # Pre-compute drifts
    drifts = np.empty(n_assets, dtype=np.float64)
    for k in range(n_assets):
        drifts[k] = (mu - 0.5 * sigmas[k] * sigmas[k]) * dt

    paths = np.empty((n_paths, n_steps + 1, n_assets), dtype=np.float64)

    for i in prange(n_paths):
        # Initialize
        for k in range(n_assets):
            paths[i, 0, k] = s0[k]

        for j in range(n_steps):
            # Generate independent standard normals
            z_indep = np.empty(n_assets, dtype=np.float64)
            for k in range(n_assets):
                z_indep[k] = np.random.standard_normal()

            # Correlate using Cholesky
            z_corr = np.zeros(n_assets, dtype=np.float64)
            for k in range(n_assets):
                for m in range(k + 1):
                    z_corr[k] += L[k, m] * z_indep[m]

            # Update each asset
            for k in range(n_assets):
                log_return = drifts[k] + sigmas[k] * sqrt_dt * z_corr[k]
                paths[i, j + 1, k] = paths[i, j, k] * np.exp(log_return)

    return paths


# =============================================================================
# Variance Reduction Techniques
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def simulate_gbm_with_control_variate(
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_paths: int,
    n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GBM simulation with geometric average as control variate under P-measure.

    Returns both the price paths and the geometric average paths
    for use in Asian option pricing with variance reduction.

    Parameters
    ----------
    mu : float
        Expected return / drift (annualized) - use risk-free rate r for Q-measure
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    drift = (mu - 0.5 * sigma * sigma) * dt
    diffusion = sigma * sqrt_dt

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    geo_avg = np.empty(n_paths, dtype=np.float64)

    for i in prange(n_paths):
        paths[i, 0] = s0
        log_sum = np.log(s0)

        for j in range(n_steps):
            z = np.random.standard_normal()
            log_return = drift + diffusion * z
            paths[i, j + 1] = paths[i, j] * np.exp(log_return)
            log_sum += np.log(paths[i, j + 1])

        # Geometric average
        geo_avg[i] = np.exp(log_sum / (n_steps + 1))

    return paths, geo_avg


# =============================================================================
# Terminal-Only Simulations (Optimized for European Option Pricing)
# =============================================================================
# These functions only return terminal values S(T), reducing memory by ~99%
# and providing ~1.8x speedup. Use these for European option pricing.

@njit(parallel=True, cache=True, fastmath=True)
def simulate_gbm_terminal(
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_paths: int,
    n_steps: int,
    antithetic: bool = True
) -> np.ndarray:
    """
    GBM simulation returning ONLY terminal values under P-measure.

    Optimized for scenario analysis and risk management:
    - 1.8x faster than full path simulation
    - 99.6% less memory (8 MB vs 2 GB for 1M paths)

    Parameters
    ----------
    s0 : float
        Initial stock price
    mu : float
        Expected return / drift (annualized) - use risk-free rate r for Q-measure
    sigma : float
        Volatility (annualized)
    t : float
        Time to maturity (in years)
    n_paths : int
        Number of simulation paths (should be even for antithetic)
    n_steps : int
        Number of time steps per path
    antithetic : bool
        Use antithetic variates for variance reduction (recommended)

    Returns
    -------
    np.ndarray
        Array of shape (n_paths,) containing terminal prices S(T)

    Example
    -------
    # P-measure (scenario analysis with expected return μ=8%)
    terminals = simulate_gbm_terminal(100, 0.08, 0.2, 1.0, 1_000_000, 252)

    # Q-measure (pricing with risk-free rate r=5%)
    terminals = simulate_gbm_terminal(100, 0.05, 0.2, 1.0, 1_000_000, 252)
    call_price = price_european_call_mc(terminals, 100, 0.05, 1.0)
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


@njit(parallel=True, cache=True, fastmath=True)
def simulate_heston_terminal(
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
    Heston model simulation returning ONLY terminal values under P-measure.

    Uses full truncation scheme. Optimized for scenario analysis.

    Parameters
    ----------
    s0 : float
        Initial stock price
    v0 : float
        Initial variance (sigma^2, not sigma)
    mu : float
        Expected return / drift (annualized) - use risk-free rate r for Q-measure
    kappa : float
        Mean reversion speed of variance
    theta : float
        Long-term variance level
    xi : float
        Volatility of volatility (vol of vol)
    rho : float
        Correlation between stock and variance Brownian motions
    t : float
        Time to maturity (in years)
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps

    Returns
    -------
    np.ndarray
        Array of shape (n_paths,) containing terminal prices S(T)
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


@njit(parallel=True, cache=True, fastmath=True)
def simulate_merton_terminal(
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
    Merton Jump Diffusion simulation returning ONLY terminal values under P-measure.

    Optimized for scenario analysis.

    Parameters
    ----------
    s0 : float
        Initial stock price
    mu : float
        Expected return / drift (annualized) - use risk-free rate r for Q-measure
    sigma : float
        Diffusion volatility
    lambda_j : float
        Jump intensity (expected number of jumps per year)
    mu_j : float
        Mean of log-jump size
    sigma_j : float
        Std dev of log-jump size
    t : float
        Time to maturity (in years)
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps

    Returns
    -------
    np.ndarray
        Array of shape (n_paths,) containing terminal prices S(T)
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


@njit(parallel=True, cache=True, fastmath=True)
def simulate_sabr_terminal(
    f0: float,
    alpha0: float,
    beta: float,
    rho: float,
    nu: float,
    t: float,
    n_paths: int,
    n_steps: int
) -> np.ndarray:
    """
    SABR model simulation returning ONLY terminal values.

    Optimized for European option pricing on forwards.

    Parameters
    ----------
    f0 : float
        Initial forward price
    alpha0 : float
        Initial volatility level
    beta : float
        CEV exponent (0 = normal, 1 = lognormal)
    rho : float
        Correlation between forward and volatility
    nu : float
        Volatility of volatility
    t : float
        Time to maturity (in years)
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps

    Returns
    -------
    np.ndarray
        Array of shape (n_paths,) containing terminal forward prices F(T)
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)
    sqrt_one_minus_rho2 = np.sqrt(1.0 - rho * rho)

    terminals = np.empty(n_paths, dtype=np.float64)

    for i in prange(n_paths):
        f = f0
        alpha = alpha0

        for j in range(n_steps):
            z1 = np.random.standard_normal()
            z2 = np.random.standard_normal()

            dw_f = sqrt_dt * z1
            dw_alpha = sqrt_dt * (rho * z1 + sqrt_one_minus_rho2 * z2)

            f_curr = max(f, 1e-10)
            alpha_curr = max(alpha, 1e-10)

            f_beta = f_curr ** beta
            f = f_curr + alpha_curr * f_beta * dw_f
            alpha = alpha_curr * np.exp(-0.5 * nu * nu * dt + nu * dw_alpha)

            f = max(f, 1e-10)

        terminals[i] = f

    return terminals


def simulate_terminal(
    model: str,
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_paths: int = 100000,
    n_steps: int = 252,
    seed: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    High-level interface for terminal-only simulation under P-measure.

    This is ~1.8x faster and uses ~99% less memory than full path simulation.
    Use this for scenario analysis and risk management.

    Note: For derivatives pricing under Q-measure, use drift = r (risk-free rate).
    This function uses P-measure with drift = mu (expected return).

    Parameters
    ----------
    model : str
        One of 'gbm', 'heston', 'merton', 'bates', 'sabr'
    s0 : float
        Initial price
    mu : float
        Expected return (drift under P-measure)
    sigma : float
        Volatility (or initial vol for stochastic vol models)
    t : float
        Time to maturity in years
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps
    seed : int, optional
        Random seed for reproducibility
    **kwargs : dict
        Model-specific parameters

    Returns
    -------
    np.ndarray
        Array of terminal values S(T), shape (n_paths,)

    Example
    -------
    # Simulate with 8% expected annual return
    terminals = simulate_terminal('gbm', 100, 0.08, 0.2, 1.0, n_paths=1_000_000)
    """
    if seed is not None:
        np.random.seed(seed)

    model_lower = model.lower()

    if model_lower == 'gbm':
        antithetic = kwargs.get('antithetic', True)
        return simulate_gbm_terminal(s0, mu, sigma, t, n_paths, n_steps, antithetic)

    elif model_lower == 'heston':
        v0 = kwargs.get('v0', sigma * sigma)
        kappa = kwargs.get('kappa', 2.0)
        theta = kwargs.get('theta', sigma * sigma)
        xi = kwargs.get('xi', 0.3)
        rho = kwargs.get('rho', -0.7)
        return simulate_heston_terminal(s0, v0, mu, kappa, theta, xi, rho, t, n_paths, n_steps)

    elif model_lower == 'merton':
        lambda_j = kwargs.get('lambda_j', 0.5)
        mu_j = kwargs.get('mu_j', -0.1)
        sigma_j = kwargs.get('sigma_j', 0.2)
        return simulate_merton_terminal(s0, mu, sigma, lambda_j, mu_j, sigma_j, t, n_paths, n_steps)

    elif model_lower == 'bates':
        # Heston parameters
        v0 = kwargs.get('v0', sigma * sigma)
        kappa = kwargs.get('kappa', 2.0)
        theta = kwargs.get('theta', sigma * sigma)
        xi = kwargs.get('xi', 0.3)
        rho = kwargs.get('rho', -0.7)
        # Jump parameters
        lambda_j = kwargs.get('lambda_j', 0.5)
        mu_j = kwargs.get('mu_j', -0.1)
        sigma_j = kwargs.get('sigma_j', 0.2)
        antithetic = kwargs.get('antithetic', True)
        return simulate_bates_terminal(
            s0, mu, v0, kappa, theta, xi, rho,
            lambda_j, mu_j, sigma_j, t, n_paths, n_steps, antithetic
        )

    elif model_lower == 'sabr':
        alpha0 = kwargs.get('alpha0', sigma)
        beta = kwargs.get('beta', 0.5)
        rho = kwargs.get('rho', -0.3)
        nu = kwargs.get('nu', 0.4)
        return simulate_sabr_terminal(s0, alpha0, beta, rho, nu, t, n_paths, n_steps)

    else:
        raise ValueError(f"Unknown model: {model}. Choose from 'gbm', 'heston', 'merton', 'bates', 'sabr'")


# =============================================================================
# High-Level Wrapper Functions
# =============================================================================

def simulate_paths(
    model: str,
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_paths: int = 100000,
    n_steps: int = 252,
    seed: Optional[int] = None,
    **kwargs
) -> SimulationResult:
    """
    High-level interface for simulating price paths under P-measure.

    This function simulates under the REAL-WORLD (P) measure, which is appropriate for:
        - Scenario analysis and risk management
        - VaR and stress testing
        - Backtesting trading strategies
        - Generating realistic price trajectories

    Note: For derivatives pricing under Q-measure, use drift = r (risk-free rate).

    Parameters
    ----------
    model : str
        One of 'gbm', 'heston', 'merton', 'bates', 'sabr'
    s0 : float
        Initial price
    mu : float
        Expected return (drift under P-measure)
    sigma : float
        Volatility (or initial vol for stochastic vol models)
    t : float
        Time to maturity in years
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps
    seed : int, optional
        Random seed for reproducibility
    **kwargs : dict
        Model-specific parameters

    Returns
    -------
    SimulationResult
        Container with paths and metadata
    """
    if seed is not None:
        np.random.seed(seed)

    time_grid = np.linspace(0, t, n_steps + 1)

    start_time = time.perf_counter()

    model_lower = model.lower()

    if model_lower == 'gbm':
        antithetic = kwargs.get('antithetic', True)
        if antithetic and n_paths % 2 != 0:
            actual_paths = 2 * (n_paths // 2)
            warnings.warn(
                f"n_paths={n_paths} is odd; antithetic variates will use {actual_paths} paths. "
                f"Consider using an even number for exact path count.",
                UserWarning
            )
        paths = simulate_gbm_paths(s0, mu, sigma, t, n_paths, n_steps, antithetic)
        model_name = "Geometric Brownian Motion"

    elif model_lower == 'heston':
        v0 = kwargs.get('v0', sigma * sigma)
        kappa = kwargs.get('kappa', 2.0)
        theta = kwargs.get('theta', sigma * sigma)
        xi = kwargs.get('xi', 0.3)
        rho = kwargs.get('rho', -0.7)
        scheme = kwargs.get('scheme', 1)

        paths, _ = simulate_heston_paths(
            s0, v0, mu, kappa, theta, xi, rho, t, n_paths, n_steps, scheme
        )
        model_name = "Heston Stochastic Volatility"

    elif model_lower == 'merton':
        lambda_j = kwargs.get('lambda_j', 0.5)
        mu_j = kwargs.get('mu_j', -0.1)
        sigma_j = kwargs.get('sigma_j', 0.2)

        paths = simulate_merton_jump_paths(
            s0, mu, sigma, lambda_j, mu_j, sigma_j, t, n_paths, n_steps
        )
        model_name = "Merton Jump Diffusion"

    elif model_lower == 'bates':
        # Heston parameters
        v0 = kwargs.get('v0', sigma * sigma)
        kappa = kwargs.get('kappa', 2.0)
        theta = kwargs.get('theta', sigma * sigma)
        xi = kwargs.get('xi', 0.3)
        rho = kwargs.get('rho', -0.7)
        # Jump parameters
        lambda_j = kwargs.get('lambda_j', 0.5)
        mu_j = kwargs.get('mu_j', -0.1)
        sigma_j = kwargs.get('sigma_j', 0.2)
        antithetic = kwargs.get('antithetic', True)

        paths, _ = simulate_bates_paths(
            s0, mu, v0, kappa, theta, xi, rho,
            lambda_j, mu_j, sigma_j, t, n_paths, n_steps, antithetic
        )
        model_name = "Bates Model"

    elif model_lower == 'sabr':
        alpha0 = kwargs.get('alpha0', sigma)
        beta = kwargs.get('beta', 0.5)
        rho = kwargs.get('rho', -0.3)
        nu = kwargs.get('nu', 0.4)

        paths, _ = simulate_sabr_paths(
            s0, alpha0, beta, rho, nu, t, n_paths, n_steps
        )
        model_name = "SABR Model"

    else:
        raise ValueError(f"Unknown model: {model}. Choose from 'gbm', 'heston', 'merton', 'bates', 'sabr'")

    computation_time = time.perf_counter() - start_time

    return SimulationResult(
        paths=paths,
        time_grid=time_grid,
        model=model_name,
        computation_time=computation_time,
        num_paths=n_paths,
        num_steps=n_steps
    )


# =============================================================================
# Option Pricing via Monte Carlo
# =============================================================================

@njit(cache=True, fastmath=True)
def price_european_call_mc(terminal_prices: np.ndarray, k: float, r: float, t: float) -> float:
    """Price European call option from simulated terminal prices."""
    n = len(terminal_prices)
    payoff_sum = 0.0
    for i in range(n):
        payoff_sum += max(terminal_prices[i] - k, 0.0)
    return np.exp(-r * t) * payoff_sum / n


@njit(cache=True, fastmath=True)
def price_european_put_mc(terminal_prices: np.ndarray, k: float, r: float, t: float) -> float:
    """Price European put option from simulated terminal prices."""
    n = len(terminal_prices)
    payoff_sum = 0.0
    for i in range(n):
        payoff_sum += max(k - terminal_prices[i], 0.0)
    return np.exp(-r * t) * payoff_sum / n


@njit(cache=True, fastmath=True)
def price_asian_arithmetic_call_mc(paths: np.ndarray, k: float, r: float, t: float) -> float:
    """Price arithmetic average Asian call option."""
    n_paths = paths.shape[0]
    n_steps = paths.shape[1]

    payoff_sum = 0.0
    for i in range(n_paths):
        avg = 0.0
        for j in range(n_steps):
            avg += paths[i, j]
        avg /= n_steps
        payoff_sum += max(avg - k, 0.0)

    return np.exp(-r * t) * payoff_sum / n_paths


@njit(cache=True, fastmath=True)
def price_lookback_call_mc(paths: np.ndarray, r: float, t: float) -> float:
    """Price floating strike lookback call option (payoff = S_T - min(S))."""
    n_paths = paths.shape[0]
    n_steps = paths.shape[1]

    payoff_sum = 0.0
    for i in range(n_paths):
        min_price = paths[i, 0]
        for j in range(1, n_steps):
            if paths[i, j] < min_price:
                min_price = paths[i, j]
        payoff_sum += paths[i, n_steps - 1] - min_price

    return np.exp(-r * t) * payoff_sum / n_paths


@njit(cache=True, fastmath=True)
def price_barrier_down_out_call_mc(
    paths: np.ndarray,
    k: float,
    barrier: float,
    r: float,
    t: float
) -> float:
    """Price down-and-out barrier call option."""
    n_paths = paths.shape[0]
    n_steps = paths.shape[1]

    payoff_sum = 0.0
    for i in range(n_paths):
        knocked_out = False
        for j in range(n_steps):
            if paths[i, j] <= barrier:
                knocked_out = True
                break

        if not knocked_out:
            payoff_sum += max(paths[i, n_steps - 1] - k, 0.0)

    return np.exp(-r * t) * payoff_sum / n_paths


# =============================================================================
# Benchmarking Utilities
# =============================================================================

def benchmark_simulation(
    model_func: Callable,
    args: tuple,
    n_runs: int = 5,
    warmup_runs: int = 2
) -> dict:
    """
    Benchmark a simulation function.

    Parameters
    ----------
    model_func : Callable
        The simulation function to benchmark
    args : tuple
        Arguments to pass to the function
    n_runs : int
        Number of timed runs
    warmup_runs : int
        Number of warmup runs (for JIT compilation)

    Returns
    -------
    dict
        Benchmark results
    """
    # Warmup runs (JIT compilation)
    for _ in range(warmup_runs):
        _ = model_func(*args)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = model_func(*args)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)

    # Get result shape for throughput calculation
    if isinstance(result, tuple):
        paths = result[0]
    else:
        paths = result

    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1
    total_samples = n_paths * n_steps

    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'throughput_paths_per_sec': n_paths / np.mean(times),
        'throughput_samples_per_sec': total_samples / np.mean(times),
        'n_paths': n_paths,
        'n_steps': n_steps
    }


def run_full_benchmark(n_paths: int = 100000, n_steps: int = 252) -> dict:
    """
    Run comprehensive benchmarks on all models.

    Parameters
    ----------
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps

    Returns
    -------
    dict
        Benchmark results for all models
    """
    # Common parameters
    s0 = 100.0
    r = 0.05
    sigma = 0.2
    t = 1.0

    results = {}

    # GBM
    print(f"Benchmarking GBM ({n_paths:,} paths, {n_steps} steps)...")
    results['GBM'] = benchmark_simulation(
        simulate_gbm_paths,
        (s0, r, sigma, t, n_paths, n_steps, True)
    )

    # GBM Vectorized
    print(f"Benchmarking GBM Vectorized...")
    results['GBM_Vectorized'] = benchmark_simulation(
        simulate_gbm_paths_vectorized,
        (s0, r, sigma, t, n_paths, n_steps)
    )

    # Heston
    print(f"Benchmarking Heston...")
    v0 = sigma * sigma
    kappa = 2.0
    theta = sigma * sigma
    xi = 0.3
    rho = -0.7
    results['Heston'] = benchmark_simulation(
        simulate_heston_paths,
        (s0, v0, r, kappa, theta, xi, rho, t, n_paths, n_steps, 1)
    )

    # Merton Jump
    print(f"Benchmarking Merton Jump Diffusion...")
    lambda_j = 0.5
    mu_j = -0.1
    sigma_j = 0.2
    results['Merton_Jump'] = benchmark_simulation(
        simulate_merton_jump_paths,
        (s0, r, sigma, lambda_j, mu_j, sigma_j, t, n_paths, n_steps)
    )

    # SABR
    print(f"Benchmarking SABR...")
    alpha0 = sigma
    beta = 0.5
    nu = 0.4
    results['SABR'] = benchmark_simulation(
        simulate_sabr_paths,
        (s0, alpha0, beta, rho, nu, t, n_paths, n_steps)
    )

    return results


def print_benchmark_results(results: dict) -> None:
    """Pretty print benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    for model, stats in results.items():
        print(f"\n{model}:")
        print(f"  Time: {stats['mean_time']*1000:.2f} ms +/- {stats['std_time']*1000:.2f} ms")
        print(f"  Throughput: {stats['throughput_paths_per_sec']:,.0f} paths/sec")
        print(f"  Throughput: {stats['throughput_samples_per_sec']/1e6:.2f} M samples/sec")

    print("\n" + "=" * 80)
