"""
SDE Discretization Kernels
==========================

Numba-optimized kernels for stochastic differential equation discretization.

Includes:
- Euler-Maruyama scheme
- Milstein scheme
- GBM exact solution
- Heston variance process schemes (Euler, Truncation, Reflection, QE)

Author: Thomas
Created: 2025
"""

import math

import numpy as np
from numba import njit

# =============================================================================
# Generic SDE Schemes
# =============================================================================

@njit(fastmath=True, cache=True)
def euler_step(
    x: float,
    drift: float,
    diffusion: float,
    dt: float,
    dw: float
) -> float:
    """
    Euler-Maruyama scheme for SDE: dX = μdt + σdW.

    Parameters
    ----------
    x : float
        Current value
    drift : float
        Drift coefficient (μ)
    diffusion : float
        Diffusion coefficient (σ)
    dt : float
        Time step
    dw : float
        Brownian increment (√dt * Z, where Z ~ N(0,1))

    Returns
    -------
    float
        Updated value
    """
    return x + drift * dt + diffusion * dw


@njit(fastmath=True, cache=True)
def milstein_step(
    x: float,
    drift: float,
    diffusion: float,
    diffusion_prime: float,
    dt: float,
    dw: float
) -> float:
    """
    Milstein scheme for SDE: dX = μdt + σdW.

    Higher-order scheme that accounts for diffusion gradient.

    Parameters
    ----------
    x : float
        Current value
    drift : float
        Drift coefficient (μ)
    diffusion : float
        Diffusion coefficient (σ)
    diffusion_prime : float
        Derivative of diffusion w.r.t. x (dσ/dx)
    dt : float
        Time step
    dw : float
        Brownian increment

    Returns
    -------
    float
        Updated value
    """
    dw_sq = dw * dw - dt  # ∫dW² - dt for Ito correction
    return (x + drift * dt + diffusion * dw +
            0.5 * diffusion * diffusion_prime * dw_sq)


# =============================================================================
# GBM Kernels
# =============================================================================

@njit(fastmath=True, cache=True)
def gbm_exact_step(
    s: float,
    mu: float,
    sigma: float,
    dt: float,
    z: float
) -> float:
    """
    Exact GBM solution step.

    Uses log-normal exact solution:
    S(t+dt) = S(t) * exp((μ - σ²/2)dt + σ√dt·Z)

    Parameters
    ----------
    s : float
        Current spot price
    mu : float
        Drift rate (under P: expected return, under Q: risk-free rate)
    sigma : float
        Volatility
    dt : float
        Time step
    z : float
        Standard normal random variable

    Returns
    -------
    float
        Updated spot price
    """
    log_return = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z
    return s * math.exp(log_return)


@njit(fastmath=True, cache=True)
def gbm_euler_step(
    s: float,
    mu: float,
    sigma: float,
    dt: float,
    dw: float
) -> float:
    """
    Euler scheme for GBM: dS = μSdt + σSdW.

    Parameters
    ----------
    s : float
        Current spot price
    mu : float
        Drift rate
    sigma : float
        Volatility
    dt : float
        Time step
    dw : float
        Brownian increment (√dt * Z)

    Returns
    -------
    float
        Updated spot price
    """
    return s * (1.0 + mu * dt + sigma * dw)


# =============================================================================
# Heston Variance Process Kernels
# =============================================================================

@njit(fastmath=True, cache=True)
def heston_euler_step(
    v: float,
    kappa: float,
    theta: float,
    xi: float,
    dt: float,
    dw_v: float
) -> float:
    """
    Simple Euler scheme for Heston variance: dV = κ(θ-V)dt + ξ√VdW.

    Warning: Can produce negative variance.

    Parameters
    ----------
    v : float
        Current variance
    kappa : float
        Mean reversion speed
    theta : float
        Long-run variance
    xi : float
        Volatility of volatility
    dt : float
        Time step
    dw_v : float
        Brownian increment for variance

    Returns
    -------
    float
        Updated variance (may be negative!)
    """
    sqrt_v = math.sqrt(max(v, 0.0))
    return v + kappa * (theta - v) * dt + xi * sqrt_v * dw_v


@njit(fastmath=True, cache=True)
def heston_truncation_step(
    v: float,
    kappa: float,
    theta: float,
    xi: float,
    dt: float,
    dw_v: float
) -> float:
    """
    Full truncation scheme for Heston variance.

    Floors variance at zero and uses V+ in drift term.

    Parameters
    ----------
    v : float
        Current variance
    kappa : float
        Mean reversion speed
    theta : float
        Long-run variance
    xi : float
        Volatility of volatility
    dt : float
        Time step
    dw_v : float
        Brownian increment for variance

    Returns
    -------
    float
        Updated variance (always >= 0)
    """
    v_plus = max(v, 0.0)
    sqrt_v = math.sqrt(v_plus)
    v_next = v + kappa * (theta - v_plus) * dt + xi * sqrt_v * dw_v
    return max(v_next, 0.0)


@njit(fastmath=True, cache=True)
def heston_reflection_step(
    v: float,
    kappa: float,
    theta: float,
    xi: float,
    dt: float,
    dw_v: float
) -> float:
    """
    Reflection scheme for Heston variance.

    Takes absolute value of negative variance.

    Parameters
    ----------
    v : float
        Current variance
    kappa : float
        Mean reversion speed
    theta : float
        Long-run variance
    xi : float
        Volatility of volatility
    dt : float
        Time step
    dw_v : float
        Brownian increment for variance

    Returns
    -------
    float
        Updated variance (always >= 0)
    """
    v_abs = abs(v)
    sqrt_v = math.sqrt(v_abs)
    v_next = v + kappa * (theta - v_abs) * dt + xi * sqrt_v * dw_v
    return abs(v_next)


@njit(fastmath=True, cache=True)
def heston_qe_step(
    v: float,
    kappa: float,
    theta: float,
    xi: float,
    dt: float,
    u1: float,
    u2: float
) -> float:
    """
    Quadratic Exponential (QE) scheme for Heston variance.

    Most accurate scheme from Andersen (2008).
    Uses moment matching with different approximations based on psi.

    Parameters
    ----------
    v : float
        Current variance
    kappa : float
        Mean reversion speed
    theta : float
        Long-run variance
    xi : float
        Volatility of volatility
    dt : float
        Time step
    u1 : float
        Uniform random for exponential approximation
    u2 : float
        Standard normal for moment matching

    Returns
    -------
    float
        Updated variance (always >= 0)
    """
    v_plus = max(v, 0.0)

    # Compute m (mean) and s² (variance) of V(t+dt)
    exp_kappa_dt = math.exp(-kappa * dt)
    m = theta + (v_plus - theta) * exp_kappa_dt

    s2 = (v_plus * xi * xi * exp_kappa_dt / kappa * (1.0 - exp_kappa_dt) +
          theta * xi * xi / (2.0 * kappa) * (1.0 - exp_kappa_dt) ** 2)

    psi = s2 / (m * m) if m > 1e-10 else 1000.0

    if psi <= 1.5:
        # Moment-matched quadratic approximation
        b2 = 2.0 / psi - 1.0 + math.sqrt(2.0 / psi * (2.0 / psi - 1.0))
        a = m / (1.0 + b2)
        v_next = a * (math.sqrt(b2) + u2) ** 2
    else:
        # Exponential approximation
        p = (psi - 1.0) / (psi + 1.0)
        beta = (1.0 - p) / m if m > 1e-10 else 1.0

        if u1 <= p:
            v_next = 0.0
        else:
            v_next = math.log((1.0 - p) / (1.0 - u1)) / beta

    return v_next


@njit(fastmath=True, cache=True)
def heston_spot_step(
    s: float,
    v: float,
    mu: float,
    dt: float,
    dw_s: float
) -> float:
    """
    Spot price step in Heston model: dS = μSdt + √VSdW.

    Parameters
    ----------
    s : float
        Current spot price
    v : float
        Current variance (will be floored at 0)
    mu : float
        Drift rate
    dt : float
        Time step
    dw_s : float
        Brownian increment for spot

    Returns
    -------
    float
        Updated spot price
    """
    v_plus = max(v, 0.0)
    sqrt_v = math.sqrt(v_plus)
    log_return = (mu - 0.5 * v_plus) * dt + sqrt_v * dw_s
    return s * math.exp(log_return)


# =============================================================================
# Jump Process Kernels
# =============================================================================

@njit(fastmath=True, cache=True)
def merton_jump_step(
    s: float,
    mu: float,
    sigma: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    dt: float,
    z_diff: float,
    n_jumps: int,
    z_jumps: np.ndarray
) -> float:
    """
    Merton jump-diffusion step.

    dS/S = (μ - λκ)dt + σdW + (e^J - 1)dN

    Parameters
    ----------
    s : float
        Current spot price
    mu : float
        Drift rate
    sigma : float
        Diffusion volatility
    lambda_j : float
        Jump intensity (jumps per year)
    mu_j : float
        Mean jump size (log scale)
    sigma_j : float
        Jump size volatility
    dt : float
        Time step
    z_diff : float
        Standard normal for diffusion
    n_jumps : int
        Number of jumps in this step (Poisson draw)
    z_jumps : np.ndarray
        Standard normals for jump sizes

    Returns
    -------
    float
        Updated spot price
    """
    # Compensator
    kappa = math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0

    # Diffusion component
    drift = (mu - lambda_j * kappa - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt) * z_diff

    # Jump component
    jump_sum = 0.0
    for i in range(n_jumps):
        jump_sum += mu_j + sigma_j * z_jumps[i]

    log_return = drift + diffusion + jump_sum
    return s * math.exp(log_return)


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("SDE Kernels Smoke Test")
    print("=" * 50)

    np.random.seed(42)

    # Test GBM exact
    s0, mu, sigma, dt = 100.0, 0.05, 0.20, 1/252
    z = np.random.standard_normal()
    s1 = gbm_exact_step(s0, mu, sigma, dt, z)
    print(f"\nGBM Exact: S0={s0:.2f} -> S1={s1:.4f}")

    # Test Heston truncation
    v0, kappa, theta, xi = 0.04, 2.0, 0.04, 0.3
    dw_v = np.sqrt(dt) * np.random.standard_normal()
    v1 = heston_truncation_step(v0, kappa, theta, xi, dt, dw_v)
    print(f"Heston Truncation: V0={v0:.4f} -> V1={v1:.6f}")

    # Test Heston QE
    u1, u2 = np.random.random(), np.random.standard_normal()
    v1_qe = heston_qe_step(v0, kappa, theta, xi, dt, u1, u2)
    print(f"Heston QE: V0={v0:.4f} -> V1={v1_qe:.6f}")

    print("\n" + "=" * 50)
    print("SDE Kernels smoke test passed")
    print("=" * 50)
