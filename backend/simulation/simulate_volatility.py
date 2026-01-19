"""
Volatility Path Simulation Module - Ultra-Optimized with Numba
===============================================================

This module provides high-performance Monte Carlo simulation for various
volatility models used in scenario analysis and risk management.

P-Measure (Physical Measure) Implementation:
--------------------------------------------
The GARCH-family models simulate volatility dynamics under the REAL-WORLD (P)
measure. This is appropriate because:
    - GARCH models are estimated from historical data (observed returns)
    - The volatility clustering patterns reflect real-world dynamics
    - Used for VaR, stress testing, and scenario analysis

For joint price-volatility simulations, the price dynamics also use P-measure
with drift = mu (expected return), NOT the risk-free rate r.

Models Implemented:
    1. GARCH(1,1) - Generalized Autoregressive Conditional Heteroskedasticity
    2. NGARCH (NAGARCH) - Nonlinear Asymmetric GARCH (Engle & Ng, 1993)
    3. GJR-GARCH - Glosten-Jagannathan-Runkle GARCH with asymmetric effects
    4. EGARCH - Exponential GARCH (Nelson, 1991)

Mathematical Specifications:
----------------------------
GARCH(1,1):
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    where ε_t = σ_t·z_t, z_t ~ N(0,1)

NGARCH (NAGARCH):
    σ²_t = ω + α·(ε_{t-1} - θ·σ_{t-1})² + β·σ²_{t-1}
    θ > 0 captures the leverage effect (negative returns → higher volatility)

GJR-GARCH:
    σ²_t = ω + (α + γ·I_{t-1})·ε²_{t-1} + β·σ²_{t-1}
    I_{t-1} = 1 if ε_{t-1} < 0, else 0

EGARCH:
    ln(σ²_t) = ω + α·(|z_{t-1}| - E[|z|]) + γ·z_{t-1} + β·ln(σ²_{t-1})
    where E[|z|] = √(2/π) for standard normal

Performance Optimizations:
    - Numba JIT compilation with nopython mode
    - Parallel execution with prange
    - Cache-friendly memory access patterns
    - Pre-allocated arrays

References:
    - Bollerslev (1986): GARCH
    - Engle & Ng (1993): NGARCH/NAGARCH, News impact curves
    - Glosten, Jagannathan & Runkle (1993): GJR-GARCH
    - Nelson (1991): EGARCH

Author: Derivatives Pricing Project
"""

import numpy as np
from numba import njit, prange
import time
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


# =============================================================================
# Configuration and Enums
# =============================================================================

class VolatilityModelType(Enum):
    """Enumeration of available volatility models."""
    GARCH = "GARCH(1,1)"
    NGARCH = "NGARCH (NAGARCH)"
    GJR_GARCH = "GJR-GARCH"
    EGARCH = "EGARCH"


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class VolatilitySimulationResult:
    """Container for volatility simulation results with metadata."""
    variance_paths: np.ndarray  # Shape: (n_paths, n_steps + 1)
    return_paths: np.ndarray    # Shape: (n_paths, n_steps)
    time_grid: np.ndarray
    model: str
    computation_time: float
    num_paths: int
    num_steps: int
    parameters: Dict[str, float] = field(default_factory=dict)

    @property
    def volatility_paths(self) -> np.ndarray:
        """Returns volatility paths (sqrt of variance)."""
        return np.sqrt(self.variance_paths)

    @property
    def terminal_variance(self) -> np.ndarray:
        """Returns terminal variance values."""
        return self.variance_paths[:, -1]

    @property
    def terminal_volatility(self) -> np.ndarray:
        """Returns terminal volatility values."""
        return np.sqrt(self.variance_paths[:, -1])

    @property
    def mean_variance_path(self) -> np.ndarray:
        """Returns the mean variance path across all simulations."""
        return np.mean(self.variance_paths, axis=0)

    @property
    def mean_volatility_path(self) -> np.ndarray:
        """Returns the mean volatility path across all simulations."""
        return np.mean(np.sqrt(self.variance_paths), axis=0)

    @property
    def std_volatility_path(self) -> np.ndarray:
        """Returns the standard deviation of volatility across simulations."""
        return np.std(np.sqrt(self.variance_paths), axis=0)

    def percentile_volatility_paths(self, percentiles: list) -> np.ndarray:
        """Returns percentile volatility paths (e.g., [5, 50, 95])."""
        return np.percentile(np.sqrt(self.variance_paths), percentiles, axis=0)

    @property
    def realized_volatility(self) -> np.ndarray:
        """
        Computes realized volatility for each path from variance paths.

        Returns the time-averaged volatility along each path, which is already
        in annualized form since variance_paths stores annualized variance.

        This avoids the issue of assuming a fixed annualization factor (e.g., 252 days)
        which would be incorrect for arbitrary time grids.

        Returns
        -------
        np.ndarray
            Realized volatility for each path, shape (n_paths,)
        """
        # Use variance paths which are already in annualized form
        # Compute time-averaged volatility along each path
        return np.mean(np.sqrt(self.variance_paths), axis=1)

    @property
    def long_run_variance(self) -> float:
        """Computes the theoretical long-run variance from parameters."""
        if self.model == "GARCH(1,1)":
            omega = self.parameters.get('omega', 0)
            alpha = self.parameters.get('alpha', 0)
            beta = self.parameters.get('beta', 0)
            if alpha + beta < 1:
                return omega / (1 - alpha - beta)
        return np.nan


@dataclass
class JointSimulationResult:
    """Container for joint price-volatility simulation results."""
    price_paths: np.ndarray      # Shape: (n_paths, n_steps + 1)
    variance_paths: np.ndarray   # Shape: (n_paths, n_steps + 1)
    return_paths: np.ndarray     # Shape: (n_paths, n_steps)
    time_grid: np.ndarray
    price_model: str
    volatility_model: str
    computation_time: float
    num_paths: int
    num_steps: int

    @property
    def volatility_paths(self) -> np.ndarray:
        """Returns volatility paths (sqrt of variance)."""
        return np.sqrt(self.variance_paths)

    @property
    def terminal_prices(self) -> np.ndarray:
        """Returns terminal price values."""
        return self.price_paths[:, -1]

    @property
    def terminal_volatility(self) -> np.ndarray:
        """Returns terminal volatility values."""
        return np.sqrt(self.variance_paths[:, -1])


# =============================================================================
# GARCH(1,1) Model Implementation
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def simulate_garch_paths(
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    n_paths: int,
    n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate volatility paths using GARCH(1,1) model.

    Model Specification:
        ε_t = σ_t · z_t,  where z_t ~ N(0,1)
        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    Stationarity requires: α + β < 1
    Long-run variance: ω / (1 - α - β)

    Parameters
    ----------
    sigma0 : float
        Initial volatility (NOT variance, i.e., σ_0)
    omega : float
        Constant term (ω > 0)
    alpha : float
        ARCH coefficient - reaction to past shocks (α >= 0)
    beta : float
        GARCH coefficient - persistence of variance (β >= 0)
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (variance_paths, return_paths)
        variance_paths: shape (n_paths, n_steps + 1)
        return_paths: shape (n_paths, n_steps) - standardized returns (ε_t)
    """
    # Initial variance
    var0 = sigma0 * sigma0

    # Allocate arrays
    variance_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    return_paths = np.empty((n_paths, n_steps), dtype=np.float64)

    for i in prange(n_paths):
        variance_paths[i, 0] = var0

        for t in range(n_steps):
            # Generate shock
            z = np.random.standard_normal()

            # Current volatility
            sigma_t = np.sqrt(variance_paths[i, t])

            # Return (shock scaled by volatility)
            epsilon_t = sigma_t * z
            return_paths[i, t] = epsilon_t

            # Update variance using GARCH(1,1) equation
            variance_next = omega + alpha * epsilon_t * epsilon_t + beta * variance_paths[i, t]

            # Ensure variance stays positive
            variance_paths[i, t + 1] = max(variance_next, 1e-10)

    return variance_paths, return_paths


@njit(cache=True, fastmath=True)
def simulate_garch_single_path(
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single GARCH(1,1) path - useful for visualization.
    """
    var0 = sigma0 * sigma0

    variance_path = np.empty(n_steps + 1, dtype=np.float64)
    return_path = np.empty(n_steps, dtype=np.float64)

    variance_path[0] = var0

    for t in range(n_steps):
        z = np.random.standard_normal()
        sigma_t = np.sqrt(variance_path[t])
        epsilon_t = sigma_t * z
        return_path[t] = epsilon_t

        variance_next = omega + alpha * epsilon_t * epsilon_t + beta * variance_path[t]
        variance_path[t + 1] = max(variance_next, 1e-10)

    return variance_path, return_path


# =============================================================================
# NGARCH (NAGARCH) Model Implementation - Engle & Ng (1993)
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def simulate_ngarch_paths(
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    theta: float,
    n_paths: int,
    n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate volatility paths using NGARCH (NAGARCH) model.

    Model Specification (Engle & Ng, 1993):
        ε_t = σ_t · z_t,  where z_t ~ N(0,1)
        σ²_t = ω + α·(ε_{t-1} - θ·σ_{t-1})² + β·σ²_{t-1}

    The θ parameter captures the leverage effect:
        - θ > 0: negative returns increase future volatility more than positive
        - θ = 0: reduces to standard GARCH(1,1)

    Stationarity requires: α(1 + θ²) + β < 1

    Parameters
    ----------
    sigma0 : float
        Initial volatility (σ_0)
    omega : float
        Constant term (ω > 0)
    alpha : float
        ARCH coefficient (α >= 0)
    beta : float
        GARCH coefficient (β >= 0)
    theta : float
        Leverage/asymmetry parameter (typically θ > 0)
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (variance_paths, return_paths)
        variance_paths: shape (n_paths, n_steps + 1)
        return_paths: shape (n_paths, n_steps)
    """
    var0 = sigma0 * sigma0

    variance_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    return_paths = np.empty((n_paths, n_steps), dtype=np.float64)

    for i in prange(n_paths):
        variance_paths[i, 0] = var0

        for t in range(n_steps):
            z = np.random.standard_normal()

            sigma_t = np.sqrt(variance_paths[i, t])
            epsilon_t = sigma_t * z
            return_paths[i, t] = epsilon_t

            # NGARCH update: shifted squared term captures asymmetry
            shifted_term = epsilon_t - theta * sigma_t
            variance_next = omega + alpha * shifted_term * shifted_term + beta * variance_paths[i, t]

            variance_paths[i, t + 1] = max(variance_next, 1e-10)

    return variance_paths, return_paths


@njit(cache=True, fastmath=True)
def simulate_ngarch_single_path(
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    theta: float,
    n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single NGARCH path - useful for visualization.
    """
    var0 = sigma0 * sigma0

    variance_path = np.empty(n_steps + 1, dtype=np.float64)
    return_path = np.empty(n_steps, dtype=np.float64)

    variance_path[0] = var0

    for t in range(n_steps):
        z = np.random.standard_normal()
        sigma_t = np.sqrt(variance_path[t])
        epsilon_t = sigma_t * z
        return_path[t] = epsilon_t

        shifted_term = epsilon_t - theta * sigma_t
        variance_next = omega + alpha * shifted_term * shifted_term + beta * variance_path[t]
        variance_path[t + 1] = max(variance_next, 1e-10)

    return variance_path, return_path


# =============================================================================
# GJR-GARCH Model Implementation - Glosten, Jagannathan & Runkle (1993)
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def simulate_gjr_garch_paths(
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    n_paths: int,
    n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate volatility paths using GJR-GARCH model.

    Model Specification:
        ε_t = σ_t · z_t,  where z_t ~ N(0,1)
        σ²_t = ω + (α + γ·I_{t-1})·ε²_{t-1} + β·σ²_{t-1}
        where I_{t-1} = 1 if ε_{t-1} < 0, else 0

    The γ parameter captures the leverage effect via an indicator function.

    Parameters
    ----------
    sigma0 : float
        Initial volatility
    omega : float
        Constant term (ω > 0)
    alpha : float
        ARCH coefficient (α >= 0)
    beta : float
        GARCH coefficient (β >= 0)
    gamma : float
        Asymmetry coefficient (γ >= 0 for leverage effect)
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (variance_paths, return_paths)
    """
    var0 = sigma0 * sigma0

    variance_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    return_paths = np.empty((n_paths, n_steps), dtype=np.float64)

    for i in prange(n_paths):
        variance_paths[i, 0] = var0

        for t in range(n_steps):
            z = np.random.standard_normal()

            sigma_t = np.sqrt(variance_paths[i, t])
            epsilon_t = sigma_t * z
            return_paths[i, t] = epsilon_t

            # GJR-GARCH: indicator for negative shocks
            indicator = 1.0 if epsilon_t < 0.0 else 0.0
            epsilon_sq = epsilon_t * epsilon_t

            variance_next = (omega +
                           (alpha + gamma * indicator) * epsilon_sq +
                           beta * variance_paths[i, t])

            variance_paths[i, t + 1] = max(variance_next, 1e-10)

    return variance_paths, return_paths


# =============================================================================
# EGARCH Model Implementation - Nelson (1991)
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def simulate_egarch_paths(
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    n_paths: int,
    n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate volatility paths using EGARCH model.

    Model Specification (Nelson, 1991):
        ε_t = σ_t · z_t,  where z_t ~ N(0,1)
        ln(σ²_t) = ω + α·(|z_{t-1}| - E[|z|]) + γ·z_{t-1} + β·ln(σ²_{t-1})

        where E[|z|] = √(2/π) for standard normal

    Key advantage: No non-negativity constraints needed as variance is
    modeled in log-space.

    Parameters
    ----------
    sigma0 : float
        Initial volatility
    omega : float
        Constant term in log-variance
    alpha : float
        Magnitude effect coefficient
    beta : float
        Persistence coefficient
    gamma : float
        Asymmetry/leverage coefficient (γ < 0 for leverage effect)
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (variance_paths, return_paths)
    """
    # E[|z|] for standard normal
    expected_abs_z = np.sqrt(2.0 / np.pi)

    var0 = sigma0 * sigma0
    log_var0 = np.log(var0)

    variance_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    return_paths = np.empty((n_paths, n_steps), dtype=np.float64)

    for i in prange(n_paths):
        variance_paths[i, 0] = var0
        log_var = log_var0

        for t in range(n_steps):
            z = np.random.standard_normal()

            sigma_t = np.sqrt(variance_paths[i, t])
            epsilon_t = sigma_t * z
            return_paths[i, t] = epsilon_t

            # EGARCH update in log-space
            log_var_next = (omega +
                          alpha * (np.abs(z) - expected_abs_z) +
                          gamma * z +
                          beta * log_var)

            log_var = log_var_next
            variance_paths[i, t + 1] = np.exp(log_var_next)

    return variance_paths, return_paths


# =============================================================================
# Terminal-Only Simulations (Optimized for Performance)
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def simulate_garch_terminal(
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    n_paths: int,
    n_steps: int
) -> np.ndarray:
    """
    GARCH(1,1) simulation returning ONLY terminal variance values.

    More memory efficient for applications only needing final volatility.
    """
    var0 = sigma0 * sigma0
    terminal_variance = np.empty(n_paths, dtype=np.float64)

    for i in prange(n_paths):
        var = var0

        for t in range(n_steps):
            z = np.random.standard_normal()
            sigma = np.sqrt(var)
            epsilon = sigma * z
            var = max(omega + alpha * epsilon * epsilon + beta * var, 1e-10)

        terminal_variance[i] = var

    return terminal_variance


@njit(parallel=True, cache=True, fastmath=True)
def simulate_ngarch_terminal(
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    theta: float,
    n_paths: int,
    n_steps: int
) -> np.ndarray:
    """
    NGARCH simulation returning ONLY terminal variance values.
    """
    var0 = sigma0 * sigma0
    terminal_variance = np.empty(n_paths, dtype=np.float64)

    for i in prange(n_paths):
        var = var0

        for t in range(n_steps):
            z = np.random.standard_normal()
            sigma = np.sqrt(var)
            epsilon = sigma * z
            shifted = epsilon - theta * sigma
            var = max(omega + alpha * shifted * shifted + beta * var, 1e-10)

        terminal_variance[i] = var

    return terminal_variance


# =============================================================================
# Joint Price-Volatility Simulation (GBM with GARCH Volatility)
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def simulate_gbm_garch_paths(
    s0: float,
    mu: float,
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    t: float,
    n_paths: int,
    n_steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate joint price and volatility paths using GBM with GARCH(1,1) volatility.

    Model (P-measure):
        dS/S = μ·dt + σ_t·dW
        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    The GARCH equation operates on standardized returns (z_t), so we use:
        h_t = ω + α·h_{t-1}·z²_{t-1} + β·h_{t-1}

    where h_t is the conditional variance and z_t ~ N(0,1).

    Parameters
    ----------
    s0 : float
        Initial stock price
    mu : float
        Expected return (drift under P-measure, annualized)
    sigma0 : float
        Initial volatility (annualized)
    omega : float
        GARCH constant term (in annualized variance terms)
    alpha : float
        GARCH ARCH coefficient
    beta : float
        GARCH persistence coefficient
    t : float
        Time horizon in years
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (price_paths, variance_paths, return_paths)
        price_paths: shape (n_paths, n_steps + 1)
        variance_paths: shape (n_paths, n_steps + 1) - annualized variance
        return_paths: shape (n_paths, n_steps) - log returns
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    var0 = sigma0 * sigma0

    price_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    variance_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    return_paths = np.empty((n_paths, n_steps), dtype=np.float64)

    for i in prange(n_paths):
        price_paths[i, 0] = s0
        variance_paths[i, 0] = var0

        for t_idx in range(n_steps):
            z = np.random.standard_normal()

            # Current annualized variance and volatility
            var_t = variance_paths[i, t_idx]
            sigma_t = np.sqrt(var_t)

            # Log return for price evolution (P-measure: drift = mu)
            log_return = (mu - 0.5 * var_t) * dt + sigma_t * sqrt_dt * z
            price_paths[i, t_idx + 1] = price_paths[i, t_idx] * np.exp(log_return)
            return_paths[i, t_idx] = log_return

            # GARCH update using standardized innovation z
            # The variance equation: h_t = omega + alpha * h_{t-1} * z^2 + beta * h_{t-1}
            variance_next = omega + alpha * var_t * z * z + beta * var_t
            variance_paths[i, t_idx + 1] = max(variance_next, 1e-10)

    return price_paths, variance_paths, return_paths


@njit(parallel=True, cache=True, fastmath=True)
def simulate_gbm_ngarch_paths(
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate joint price and volatility paths using GBM with NGARCH volatility.

    Model (P-measure):
        dS/S = μ·dt + σ_t·dW
        h_t = ω + α·h_{t-1}·(z_{t-1} - θ)² + β·h_{t-1}

    where h_t is conditional variance and z_t ~ N(0,1).
    The θ parameter creates asymmetric volatility response to returns.

    Parameters
    ----------
    s0 : float
        Initial stock price
    mu : float
        Expected return (drift under P-measure, annualized)
    sigma0 : float
        Initial volatility (annualized)
    omega : float
        NGARCH constant term (in annualized variance terms)
    alpha : float
        NGARCH ARCH coefficient
    beta : float
        NGARCH persistence coefficient
    theta : float
        Leverage parameter (typically > 0)
    t : float
        Time horizon in years
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (price_paths, variance_paths, return_paths)
        variance_paths contains annualized variance
    """
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    var0 = sigma0 * sigma0

    price_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    variance_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    return_paths = np.empty((n_paths, n_steps), dtype=np.float64)

    for i in prange(n_paths):
        price_paths[i, 0] = s0
        variance_paths[i, 0] = var0

        for t_idx in range(n_steps):
            z = np.random.standard_normal()

            # Current annualized variance and volatility
            var_t = variance_paths[i, t_idx]
            sigma_t = np.sqrt(var_t)

            # Price evolution (P-measure: drift = mu)
            log_return = (mu - 0.5 * var_t) * dt + sigma_t * sqrt_dt * z
            price_paths[i, t_idx + 1] = price_paths[i, t_idx] * np.exp(log_return)
            return_paths[i, t_idx] = log_return

            # NGARCH update with leverage using standardized innovation
            # h_t = omega + alpha * h_{t-1} * (z - theta)^2 + beta * h_{t-1}
            shifted = z - theta
            variance_next = omega + alpha * var_t * shifted * shifted + beta * var_t
            variance_paths[i, t_idx + 1] = max(variance_next, 1e-10)

    return price_paths, variance_paths, return_paths


# =============================================================================
# Parameter Utilities
# =============================================================================

@njit(cache=True)
def compute_garch_long_run_variance(omega: float, alpha: float, beta: float) -> float:
    """
    Compute long-run (unconditional) variance for GARCH(1,1).

    σ²_∞ = ω / (1 - α - β)
    """
    persistence = alpha + beta
    if persistence >= 1.0:
        return np.inf
    return omega / (1.0 - persistence)


@njit(cache=True)
def compute_ngarch_long_run_variance(omega: float, alpha: float, beta: float, theta: float) -> float:
    """
    Compute long-run variance for NGARCH.

    For NGARCH: σ²_∞ = ω / (1 - α(1 + θ²) - β)
    """
    persistence = alpha * (1.0 + theta * theta) + beta
    if persistence >= 1.0:
        return np.inf
    return omega / (1.0 - persistence)


def validate_garch_params(omega: float, alpha: float, beta: float) -> Tuple[bool, str]:
    """
    Validate GARCH(1,1) parameters.

    Returns
    -------
    Tuple[bool, str]
        (is_valid, message)
    """
    if omega <= 0:
        return False, "omega must be positive"
    if alpha < 0:
        return False, "alpha must be non-negative"
    if beta < 0:
        return False, "beta must be non-negative"
    if alpha + beta >= 1:
        return False, f"alpha + beta = {alpha + beta:.4f} >= 1: process is not stationary"
    return True, "Parameters are valid"


def validate_ngarch_params(omega: float, alpha: float, beta: float, theta: float) -> Tuple[bool, str]:
    """
    Validate NGARCH parameters.

    Returns
    -------
    Tuple[bool, str]
        (is_valid, message)
    """
    if omega <= 0:
        return False, "omega must be positive"
    if alpha < 0:
        return False, "alpha must be non-negative"
    if beta < 0:
        return False, "beta must be non-negative"

    persistence = alpha * (1 + theta * theta) + beta
    if persistence >= 1:
        return False, f"alpha*(1+theta^2) + beta = {persistence:.4f} >= 1: process is not stationary"
    return True, "Parameters are valid"


def validate_gjr_garch_params(omega: float, alpha: float, beta: float, gamma: float) -> Tuple[bool, str]:
    """
    Validate GJR-GARCH parameters.

    Stationarity condition: alpha + beta + gamma/2 < 1

    Returns
    -------
    Tuple[bool, str]
        (is_valid, message)
    """
    if omega <= 0:
        return False, "omega must be positive"
    if alpha < 0:
        return False, "alpha must be non-negative"
    if beta < 0:
        return False, "beta must be non-negative"
    if gamma < 0:
        return False, "gamma must be non-negative for leverage effect"

    persistence = alpha + beta + 0.5 * gamma
    if persistence >= 1:
        return False, f"alpha + beta + gamma/2 = {persistence:.4f} >= 1: process is not stationary"
    return True, "Parameters are valid"


def validate_egarch_params(omega: float, alpha: float, beta: float, gamma: float) -> Tuple[bool, str]:
    """
    Validate EGARCH parameters.

    Stationarity condition: |beta| < 1

    Returns
    -------
    Tuple[bool, str]
        (is_valid, message)
    """
    # omega can be any real value in EGARCH (log-space)
    if alpha < 0:
        return False, "alpha must be non-negative"

    if abs(beta) >= 1:
        return False, f"|beta| = {abs(beta):.4f} >= 1: process is not stationary"
    return True, "Parameters are valid"


def estimate_garch_params_from_volatility(
    target_long_run_vol: float,
    half_life_days: int = 20,
    alpha_ratio: float = 0.1
) -> Dict[str, float]:
    """
    Estimate reasonable GARCH parameters from target volatility and half-life.

    Parameters
    ----------
    target_long_run_vol : float
        Target long-run annualized volatility (e.g., 0.20 for 20%)
    half_life_days : int
        Half-life of volatility shocks in days
    alpha_ratio : float
        Ratio of alpha to (alpha + beta), typically 0.05-0.15

    Returns
    -------
    Dict[str, float]
        Dictionary with 'omega', 'alpha', 'beta' keys
    """
    # Convert half-life to persistence
    # persistence^half_life = 0.5 => persistence = 0.5^(1/half_life)
    persistence = 0.5 ** (1.0 / half_life_days)

    # Split persistence into alpha and beta
    alpha = alpha_ratio * persistence
    beta = (1 - alpha_ratio) * persistence

    # Compute omega from long-run variance
    target_variance = target_long_run_vol ** 2
    omega = target_variance * (1 - alpha - beta)

    return {
        'omega': omega,
        'alpha': alpha,
        'beta': beta,
        'long_run_variance': target_variance,
        'half_life_days': half_life_days
    }


# =============================================================================
# High-Level Wrapper Functions
# =============================================================================

def simulate_volatility_paths(
    model: str,
    sigma0: float,
    n_paths: int = 10000,
    n_steps: int = 252,
    seed: Optional[int] = None,
    **kwargs
) -> VolatilitySimulationResult:
    """
    High-level interface for simulating volatility paths.

    Parameters
    ----------
    model : str
        One of 'garch', 'ngarch', 'gjr_garch', 'egarch'
    sigma0 : float
        Initial volatility (annualized, e.g., 0.20 for 20%)
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps
    seed : int, optional
        Random seed for reproducibility
    **kwargs : dict
        Model-specific parameters

    GARCH kwargs:
        omega : float - constant term (default: estimated from sigma0)
        alpha : float - ARCH coefficient (default: 0.05)
        beta : float - GARCH coefficient (default: 0.90)

    NGARCH kwargs:
        omega : float - constant term
        alpha : float - ARCH coefficient (default: 0.05)
        beta : float - GARCH coefficient (default: 0.90)
        theta : float - leverage parameter (default: 0.5)

    GJR-GARCH kwargs:
        omega : float - constant term
        alpha : float - ARCH coefficient (default: 0.05)
        beta : float - GARCH coefficient (default: 0.90)
        gamma : float - asymmetry coefficient (default: 0.05)

    EGARCH kwargs:
        omega : float - constant term in log-variance
        alpha : float - magnitude effect (default: 0.1)
        beta : float - persistence (default: 0.95)
        gamma : float - asymmetry/leverage (default: -0.1)

    Returns
    -------
    VolatilitySimulationResult
        Container with variance paths, return paths, and metadata
    """
    if seed is not None:
        np.random.seed(seed)

    time_grid = np.linspace(0, 1, n_steps + 1)  # Normalized time

    start_time = time.perf_counter()

    model_lower = model.lower().replace('-', '_')
    parameters = {}

    if model_lower == 'garch':
        # Default parameters for GARCH(1,1)
        alpha = kwargs.get('alpha', 0.05)
        beta = kwargs.get('beta', 0.90)
        # Estimate omega to match initial volatility as long-run vol
        omega = kwargs.get('omega', sigma0**2 * (1 - alpha - beta))

        # Validate parameters
        is_valid, msg = validate_garch_params(omega, alpha, beta)
        if not is_valid:
            raise ValueError(f"Invalid GARCH parameters: {msg}")

        parameters = {'omega': omega, 'alpha': alpha, 'beta': beta}

        variance_paths, return_paths = simulate_garch_paths(
            sigma0, omega, alpha, beta, n_paths, n_steps
        )
        model_name = "GARCH(1,1)"

    elif model_lower == 'ngarch':
        alpha = kwargs.get('alpha', 0.05)
        beta = kwargs.get('beta', 0.90)
        theta = kwargs.get('theta', 0.5)
        # For NGARCH: long-run var = omega / (1 - alpha*(1+theta^2) - beta)
        persistence = alpha * (1 + theta**2) + beta
        omega = kwargs.get('omega', sigma0**2 * (1 - persistence))

        # Validate parameters
        is_valid, msg = validate_ngarch_params(omega, alpha, beta, theta)
        if not is_valid:
            raise ValueError(f"Invalid NGARCH parameters: {msg}")

        parameters = {'omega': omega, 'alpha': alpha, 'beta': beta, 'theta': theta}

        variance_paths, return_paths = simulate_ngarch_paths(
            sigma0, omega, alpha, beta, theta, n_paths, n_steps
        )
        model_name = "NGARCH (NAGARCH)"

    elif model_lower == 'gjr_garch':
        alpha = kwargs.get('alpha', 0.05)
        beta = kwargs.get('beta', 0.90)
        gamma = kwargs.get('gamma', 0.05)
        omega = kwargs.get('omega', sigma0**2 * (1 - alpha - 0.5*gamma - beta))

        # Validate parameters
        is_valid, msg = validate_gjr_garch_params(omega, alpha, beta, gamma)
        if not is_valid:
            raise ValueError(f"Invalid GJR-GARCH parameters: {msg}")

        parameters = {'omega': omega, 'alpha': alpha, 'beta': beta, 'gamma': gamma}

        variance_paths, return_paths = simulate_gjr_garch_paths(
            sigma0, omega, alpha, beta, gamma, n_paths, n_steps
        )
        model_name = "GJR-GARCH"

    elif model_lower == 'egarch':
        alpha = kwargs.get('alpha', 0.1)
        beta = kwargs.get('beta', 0.95)
        gamma = kwargs.get('gamma', -0.1)
        # For EGARCH, omega is in log-space
        omega = kwargs.get('omega', np.log(sigma0**2) * (1 - beta))

        # Validate parameters
        is_valid, msg = validate_egarch_params(omega, alpha, beta, gamma)
        if not is_valid:
            raise ValueError(f"Invalid EGARCH parameters: {msg}")

        parameters = {'omega': omega, 'alpha': alpha, 'beta': beta, 'gamma': gamma}

        variance_paths, return_paths = simulate_egarch_paths(
            sigma0, omega, alpha, beta, gamma, n_paths, n_steps
        )
        model_name = "EGARCH"

    else:
        raise ValueError(f"Unknown model: {model}. Choose from 'garch', 'ngarch', 'gjr_garch', 'egarch'")

    computation_time = time.perf_counter() - start_time

    return VolatilitySimulationResult(
        variance_paths=variance_paths,
        return_paths=return_paths,
        time_grid=time_grid,
        model=model_name,
        computation_time=computation_time,
        num_paths=n_paths,
        num_steps=n_steps,
        parameters=parameters
    )


def simulate_terminal_volatility(
    model: str,
    sigma0: float,
    n_paths: int = 100000,
    n_steps: int = 252,
    seed: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    High-level interface for terminal-only volatility simulation.

    More memory efficient - returns only terminal variance values.

    Parameters
    ----------
    model : str
        One of 'garch', 'ngarch'
    sigma0 : float
        Initial volatility
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps
    seed : int, optional
        Random seed
    **kwargs : dict
        Model-specific parameters

    Returns
    -------
    np.ndarray
        Terminal variance values, shape (n_paths,)
    """
    if seed is not None:
        np.random.seed(seed)

    model_lower = model.lower()

    if model_lower == 'garch':
        alpha = kwargs.get('alpha', 0.05)
        beta = kwargs.get('beta', 0.90)
        omega = kwargs.get('omega', sigma0**2 * (1 - alpha - beta))

        return simulate_garch_terminal(sigma0, omega, alpha, beta, n_paths, n_steps)

    elif model_lower == 'ngarch':
        alpha = kwargs.get('alpha', 0.05)
        beta = kwargs.get('beta', 0.90)
        theta = kwargs.get('theta', 0.5)
        persistence = alpha * (1 + theta**2) + beta
        omega = kwargs.get('omega', sigma0**2 * (1 - persistence))

        return simulate_ngarch_terminal(sigma0, omega, alpha, beta, theta, n_paths, n_steps)

    else:
        raise ValueError(f"Unknown model: {model}. Choose from 'garch', 'ngarch'")


def simulate_joint_paths(
    volatility_model: str,
    s0: float,
    mu: float,
    sigma0: float,
    t: float,
    n_paths: int = 10000,
    n_steps: int = 252,
    seed: Optional[int] = None,
    **kwargs
) -> JointSimulationResult:
    """
    Simulate joint price and volatility paths under P-measure.

    Combines GBM price dynamics with GARCH-family volatility.
    Uses drift = mu (expected return) for realistic scenario analysis.

    Parameters
    ----------
    volatility_model : str
        Volatility model: 'garch' or 'ngarch'
    s0 : float
        Initial stock price
    mu : float
        Expected return (drift under P-measure, annualized)
    sigma0 : float
        Initial volatility (annualized)
    t : float
        Time horizon in years
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps
    seed : int, optional
        Random seed
    **kwargs : dict
        Model-specific parameters

    Returns
    -------
    JointSimulationResult
        Container with price paths, variance paths, and metadata
    """
    if seed is not None:
        np.random.seed(seed)

    time_grid = np.linspace(0, t, n_steps + 1)

    start_time = time.perf_counter()

    model_lower = volatility_model.lower()

    if model_lower == 'garch':
        alpha = kwargs.get('alpha', 0.05)
        beta = kwargs.get('beta', 0.90)
        omega = kwargs.get('omega', sigma0**2 * (1 - alpha - beta))

        price_paths, variance_paths, return_paths = simulate_gbm_garch_paths(
            s0, mu, sigma0, omega, alpha, beta, t, n_paths, n_steps
        )
        vol_model_name = "GARCH(1,1)"

    elif model_lower == 'ngarch':
        alpha = kwargs.get('alpha', 0.05)
        beta = kwargs.get('beta', 0.90)
        theta = kwargs.get('theta', 0.5)
        persistence = alpha * (1 + theta**2) + beta
        omega = kwargs.get('omega', sigma0**2 * (1 - persistence))

        price_paths, variance_paths, return_paths = simulate_gbm_ngarch_paths(
            s0, mu, sigma0, omega, alpha, beta, theta, t, n_paths, n_steps
        )
        vol_model_name = "NGARCH"

    else:
        raise ValueError(f"Unknown volatility model: {volatility_model}")

    computation_time = time.perf_counter() - start_time

    return JointSimulationResult(
        price_paths=price_paths,
        variance_paths=variance_paths,
        return_paths=return_paths,
        time_grid=time_grid,
        price_model="GBM",
        volatility_model=vol_model_name,
        computation_time=computation_time,
        num_paths=n_paths,
        num_steps=n_steps
    )


# =============================================================================
# Benchmarking Utilities
# =============================================================================

def benchmark_volatility_simulation(
    model_func,
    args: tuple,
    n_runs: int = 5,
    warmup_runs: int = 2
) -> dict:
    """
    Benchmark a volatility simulation function.
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

    if isinstance(result, tuple):
        paths = result[0]
    else:
        paths = result

    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1 if len(paths.shape) > 1 else 1
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


def run_volatility_benchmark(n_paths: int = 100000, n_steps: int = 252) -> dict:
    """
    Run comprehensive benchmarks on all volatility models.
    """
    sigma0 = 0.20
    omega = 0.0001
    alpha = 0.05
    beta = 0.90
    theta = 0.5
    gamma = 0.05

    results = {}

    print(f"Benchmarking volatility models ({n_paths:,} paths, {n_steps} steps)...")

    # GARCH
    print("  GARCH(1,1)...")
    results['GARCH'] = benchmark_volatility_simulation(
        simulate_garch_paths,
        (sigma0, omega, alpha, beta, n_paths, n_steps)
    )

    # NGARCH
    print("  NGARCH...")
    results['NGARCH'] = benchmark_volatility_simulation(
        simulate_ngarch_paths,
        (sigma0, omega, alpha, beta, theta, n_paths, n_steps)
    )

    # GJR-GARCH
    print("  GJR-GARCH...")
    results['GJR_GARCH'] = benchmark_volatility_simulation(
        simulate_gjr_garch_paths,
        (sigma0, omega, alpha, beta, gamma, n_paths, n_steps)
    )

    # EGARCH
    print("  EGARCH...")
    omega_egarch = np.log(sigma0**2) * (1 - beta)
    results['EGARCH'] = benchmark_volatility_simulation(
        simulate_egarch_paths,
        (sigma0, omega_egarch, alpha, beta, -gamma, n_paths, n_steps)
    )

    return results


def print_volatility_benchmark_results(results: dict) -> None:
    """Pretty print benchmark results."""
    print("\n" + "=" * 70)
    print("VOLATILITY SIMULATION BENCHMARK RESULTS")
    print("=" * 70)

    for model, stats in results.items():
        print(f"\n{model}:")
        print(f"  Time: {stats['mean_time']*1000:.2f} ms +/- {stats['std_time']*1000:.2f} ms")
        print(f"  Throughput: {stats['throughput_paths_per_sec']:,.0f} paths/sec")
        print(f"  Throughput: {stats['throughput_samples_per_sec']/1e6:.2f} M samples/sec")

    print("\n" + "=" * 70)
