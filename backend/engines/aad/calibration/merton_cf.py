"""
Merton Jump-Diffusion Characteristic Function (JAX)
====================================================

Pure JAX port of the Merton (1976) characteristic function, matching
the contract used by `fft_jax.price_call_strikes_jax` — i.e. returns
phi_log_R(u) = E[exp(i*u*log(S_T/S_0))] (the log-RETURN CF), so the
log-spot phase shift is applied by the FFT pricer itself.

Compatible with jax.grad / jax.jacfwd / jax.vmap for use in
Levenberg-Marquardt calibration with analytical Jacobians.

Reference: backend/models/characteristic_functions/merton_cf.py

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def merton_cf_jax(
    u: jax.Array,
    sigma: float,
    lam: float,
    alpha_j: float,
    sigma_j: float,
    tau: float,
    r: float,
    q: float = 0.0,
) -> jax.Array:
    """Merton jump-diffusion CF of log-returns.

    Parameters
    ----------
    u : jnp.ndarray
        Frequency points (real or complex), shape (N,).
    sigma : float
        Diffusion volatility.
    lam : float
        Jump intensity (expected jumps per unit time).
    alpha_j : float
        Mean log-jump size.
    sigma_j : float
        Std dev of log-jump size.
    tau : float
        Time to maturity.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.

    Returns
    -------
    jnp.ndarray
        Complex CF values phi_log_R(u), shape (N,).

    Notes
    -----
    phi_log_R(u) = phi_GBM_R(u) * phi_Jump(u)
      phi_GBM_R(u) = exp(i*u*(r - q - 0.5*sigma^2 - lambda*k)*tau - 0.5*sigma^2*u^2*tau)
      phi_Jump(u) = exp(lambda*tau*(exp(i*u*alpha_j - 0.5*sigma_j^2*u^2) - 1))
      k = exp(alpha_j + 0.5*sigma_j^2) - 1   (jump compensator for Q-martingale)
    """
    u_c = u * 1.0 + 0.0j
    i = 1.0j

    # Jump compensator
    k = jnp.exp(alpha_j + 0.5 * sigma_j**2) - 1.0

    # Risk-neutral drift (log-return drift)
    drift = r - q - 0.5 * sigma**2 - lam * k

    # Diffusion (log-return) exponent
    diffusion = i * u_c * drift * tau - 0.5 * sigma**2 * u_c**2 * tau

    # Jump CF exponent: clamp real part to avoid overflow
    jump_exp_raw = i * u_c * alpha_j - 0.5 * sigma_j**2 * u_c**2
    jump_exp_clamped = jnp.where(
        jump_exp_raw.real > 700.0,
        700.0 + jump_exp_raw.imag * 1.0j,
        jump_exp_raw,
    )
    jump_cf = jnp.exp(jump_exp_clamped)

    # Aggregate jump contribution
    jump_component = lam * tau * (jump_cf - 1.0)

    return jnp.exp(diffusion + jump_component)


if __name__ == "__main__":
    print("Merton JAX CF smoke test")
    u = jnp.linspace(0.1, 20.0, 20)
    cf = merton_cf_jax(u, 0.2, 0.5, -0.1, 0.2, 1.0, 0.05, 0.0)
    print(f"  CF shape: {cf.shape}")
    cf0 = merton_cf_jax(jnp.array([0.0]), 0.2, 0.5, -0.1, 0.2, 1.0, 0.05, 0.0)
    print(f"  CF(0) = {cf0[0]} (expected ~1.0)")
    assert jnp.abs(cf0[0] - 1.0) < 1e-10

    def mean_cf(sigma):
        return jnp.mean(
            jnp.abs(merton_cf_jax(u, sigma, 0.5, -0.1, 0.2, 1.0, 0.05, 0.0))
        )

    g = jax.grad(mean_cf)(0.2)
    print(f"  d|CF|/dsigma = {float(g):.6f} (must be finite)")
    assert jnp.isfinite(g)
    print("  OK")
