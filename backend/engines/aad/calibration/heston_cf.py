"""
Heston Characteristic Function (JAX)
=====================================

Pure JAX port of the Heston (1993) characteristic function.
Uses the Gatheral (2006) formulation for numerical stability.

Compatible with jax.grad, jax.jit, and jax.vmap.
All Python-level branching replaced with jnp.where for JIT tracing.

Reference implementation: backend/models/characteristic_functions/heston_cf.py

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


def heston_cf_jax(
    u: jax.Array,
    v0: float,
    kappa: float,
    theta: float,
    alpha: float,
    rho: float,
    tau: float,
    r: float,
    q: float = 0.0,
) -> jax.Array:
    """Heston characteristic function phi(u) = E^Q[exp(i*u*ln(S_T/S_0))].

    Uses the Gatheral (2006) formulation which avoids branch-cut
    discontinuities in the complex logarithm.

    Parameters
    ----------
    u : jnp.ndarray
        Frequency points, shape (N,). Real-valued (cast to complex internally).
    v0 : float
        Initial variance.
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run variance.
    alpha : float
        Volatility of volatility.
    rho : float
        Spot-vol correlation.
    tau : float
        Time to maturity.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.

    Returns
    -------
    jnp.ndarray
        Complex CF values, shape (N,).
    """
    u_c = u * 1.0 + 0.0j  # promote to complex
    i = 1.0j
    eps = 1e-12
    safe_xi = jnp.maximum(alpha, 1e-6)

    # Gatheral (2006) intermediate quantities
    d = jnp.sqrt(
        (rho * safe_xi * i * u_c - kappa) ** 2 + safe_xi**2 * (i * u_c + u_c**2)
    )

    numerator = kappa - rho * safe_xi * i * u_c - d
    denominator = kappa - rho * safe_xi * i * u_c + d

    # Guard against division by zero (JIT-safe)
    safe_denom = jnp.where(jnp.abs(denominator) < eps, eps + 0.0j, denominator)
    g = numerator / safe_denom

    exp_dt = jnp.exp(-d * tau)

    one_minus_g = 1.0 - g
    one_minus_g_exp = 1.0 - g * exp_dt

    # Guard log singularity
    safe_omg = jnp.where(jnp.abs(one_minus_g) < eps, eps + 0.0j, one_minus_g)
    safe_omge = jnp.where(jnp.abs(one_minus_g_exp) < eps, eps + 0.0j, one_minus_g_exp)

    # C coefficient
    C = (r - q) * i * u_c * tau + kappa * theta / safe_xi**2 * (
        numerator * tau - 2.0 * jnp.log(safe_omge / safe_omg)
    )

    # D coefficient
    D = numerator / safe_xi**2 * ((1.0 - exp_dt) / safe_omge)

    return jnp.exp(C + D * v0)


def bates_cf_jax(
    u: jax.Array,
    v0: float,
    kappa: float,
    theta: float,
    alpha: float,
    rho: float,
    tau: float,
    r: float,
    q: float = 0.0,
    lam: float = 0.0,
    alpha_j: float = 0.0,
    sigma_j: float = 0.0,
) -> jax.Array:
    """Bates (1996) characteristic function: Heston + Merton jumps.

    phi_Bates(u) = phi_Heston(u) * exp(jump_correction)

    The jump component adds a compensated Poisson process with
    log-normal jump sizes to the Heston stochastic volatility.

    Parameters
    ----------
    u : jnp.ndarray
        Frequency points, shape (N,).
    v0, kappa, theta, alpha, rho : float
        Heston parameters.
    tau, r, q : float
        Time, rate, dividend yield.
    lam : float
        Jump intensity (jumps per year).
    alpha_j : float
        Mean log-jump size (negative for equity).
    sigma_j : float
        Jump size volatility.

    Returns
    -------
    jnp.ndarray
        Complex CF values, shape (N,).
    """
    # Heston component
    cf_heston = heston_cf_jax(u, v0, kappa, theta, alpha, rho, tau, r, q)

    # Jump component: Merton (1976) jump factor
    u_c = u * 1.0 + 0.0j
    i = 1.0j

    # E[exp(i*u*J)] where J ~ N(alpha_j, sigma_j^2)
    jump_cf = jnp.exp(i * u_c * alpha_j - 0.5 * sigma_j**2 * u_c**2)

    # Compensator: -lam * (E[exp(J)] - 1) * tau ensures martingale
    mean_jump = jnp.exp(alpha_j + 0.5 * sigma_j**2) - 1.0
    compensator = -lam * mean_jump * i * u_c * tau

    # Jump contribution
    jump_factor = jnp.exp(lam * tau * (jump_cf - 1.0) + compensator)

    return cf_heston * jump_factor


@functools.partial(jax.jit, static_argnums=())
def heston_cf_jax_jit(
    u: jax.Array,
    v0: float,
    kappa: float,
    theta: float,
    alpha: float,
    rho: float,
    tau: float,
    r: float,
    q: float = 0.0,
) -> jax.Array:
    """JIT-compiled version of heston_cf_jax."""
    return heston_cf_jax(u, v0, kappa, theta, alpha, rho, tau, r, q)


if __name__ == "__main__":
    print("=" * 50)
    print("Heston CF JAX Smoke Test")
    print("=" * 50)

    # Test params
    v0, kappa, theta, alpha, rho = 0.04, 2.0, 0.04, 0.3, -0.7
    tau, r, q = 1.0, 0.05, 0.0

    u = jnp.linspace(0.1, 20.0, 20)
    cf_vals = heston_cf_jax(u, v0, kappa, theta, alpha, rho, tau, r, q)

    print(f"  u shape: {u.shape}")
    print(f"  CF shape: {cf_vals.shape}")
    assert cf_vals.shape == u.shape
    assert jnp.all(jnp.isfinite(jnp.abs(cf_vals)))
    print(f"  |CF(0.1)| = {float(jnp.abs(cf_vals[0])):.6f}")
    print(f"  |CF(20)| = {float(jnp.abs(cf_vals[-1])):.10f}")

    # CF at u=0 should be 1 (normalization)
    cf_zero = heston_cf_jax(jnp.array([0.0]), v0, kappa, theta, alpha, rho, tau, r, q)
    print(f"  CF(0) = {cf_zero[0]} (should be ~1.0)")
    assert jnp.abs(cf_zero[0] - 1.0) < 1e-10

    # Differentiability w.r.t. v0
    def mean_abs_cf(v0_val):
        vals = heston_cf_jax(u, v0_val, kappa, theta, alpha, rho, tau, r, q)
        return jnp.mean(jnp.abs(vals))

    grad_v0 = jax.grad(mean_abs_cf)(v0)
    print(f"  d|CF|/dv0 = {float(grad_v0):.6f}")
    assert jnp.isfinite(grad_v0)

    print("\n  All smoke tests passed")
