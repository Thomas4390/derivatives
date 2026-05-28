"""
Heston-Nandi GARCH Characteristic Function (JAX)
=================================================

Pure JAX port of the Heston & Nandi (2000) closed-form GARCH option-pricing
characteristic function. End-to-end differentiable via ``jax.grad`` /
``jax.jacfwd`` with respect to the GARCH parameters — enabling
Levenberg-Marquardt surface calibration with an analytical Jacobian, exactly
like ``heston_cf_jax``.

Model (risk-neutral, per-period / daily step, lambda* = -1/2):

    R_t = ln(S_t/S_{t-1}) = r_step - 0.5 h_t + sqrt(h_t) z_t,  z_t ~ N(0,1)
    h_{t+1} = omega + beta h_t + alpha (z_t - gamma sqrt(h_t))^2

The conditional generating function of the log return X_T = ln(S_T/S_0) is
log-affine, ``phi_X(u) = exp(A_t + B_t h0)`` with ``phi = i u`` and the backward
recursion (Heston-Nandi 2000, eq. A.12-A.13) run over
``N = round(tau * steps_per_year)`` steps from the terminal ``A_T = B_T = 0``:

    denom = 1 - 2 alpha B
    B_new = phi (gamma - 1/2) - 1/2 gamma^2 + beta B + 1/2 (phi - gamma)^2 / denom
    A_new = A + phi r_step + omega B - 1/2 ln(denom)

This module returns the LOG-RETURN CF (spot-agnostic), matching the contract of
``price_call_strikes_jax`` which applies the log-spot shift internally. The
Numba reference implementation (log-PRICE form) lives in
``backend/models/characteristic_functions/heston_nandi_cf.py``.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from backend.utils.constants.calibration import HESTON_NANDI_STEPS_PER_YEAR


def heston_nandi_cf_jax(
    u: jax.Array,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    h0: float,
    tau: float,
    r: float,
    steps_per_year: int = HESTON_NANDI_STEPS_PER_YEAR,
) -> jax.Array:
    """Heston-Nandi log-return CF ``phi_X(u) = E^Q[exp(i u ln(S_T/S_0))]``.

    Parameters
    ----------
    u : jnp.ndarray
        Frequency points, shape (N,). May be complex (the Carr-Madan grid is
        shifted into the complex plane); promoted to complex internally.
    omega, alpha, beta, gamma : float
        Risk-neutral GARCH(1,1) parameters (per period).
    h0 : float
        Initial conditional variance ``h_1`` (per period).
    tau : float
        Time to maturity in years. The discrete horizon is
        ``N = round(tau * steps_per_year)`` steps.
    r : float
        Annual risk-free rate; the per-step rate is ``r / steps_per_year``.
    steps_per_year : int
        Trading-day discretization (default 252).

    Returns
    -------
    jnp.ndarray
        Complex CF values, shape (N,).
    """
    n_steps = max(int(round(float(tau) * steps_per_year)), 1)
    r_step = r / steps_per_year

    phi = 1j * (u * 1.0 + 0.0j)  # promote to complex, phi = i u
    a0 = jnp.zeros_like(phi)
    b0 = jnp.zeros_like(phi)

    def body(_: int, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        a, b = carry  # coefficients of the *next* time step (A_{s+1}, B_{s+1})
        denom = 1.0 - 2.0 * alpha * b
        b_new = (
            phi * (gamma - 0.5)
            - 0.5 * gamma**2
            + beta * b
            + 0.5 * (phi - gamma) ** 2 / denom
        )
        a_new = a + phi * r_step + omega * b - 0.5 * jnp.log(denom)
        return a_new, b_new

    a, b = jax.lax.fori_loop(0, n_steps, body, (a0, b0))
    return jnp.exp(a + b * h0)


if __name__ == "__main__":
    print("=" * 56)
    print("Heston-Nandi GARCH CF (JAX) Smoke Test")
    print("=" * 56)

    # Risk-neutral daily params (a benign, stationary regime).
    omega, alpha, beta, gamma, h0 = 1.0e-6, 2.0e-6, 0.80, 150.0, 4.0e-5
    r = 0.05
    spy = 252
    # Use integer-step maturities so N/spy == tau exactly (clean invariants).
    tau = 126 / spy  # ~0.5y

    # --- Invariant 1: phi_X(0) = 1 (normalization) ---
    cf0 = heston_nandi_cf_jax(
        jnp.array([0.0 + 0.0j]), omega, alpha, beta, gamma, h0, tau, r, spy
    )
    print(f"\nphi_X(0)   = {complex(cf0[0]):.10f}  (expect 1)")
    assert abs(complex(cf0[0]) - 1.0) < 1e-9

    # --- Invariant 2: phi_X(-i) = e^{r tau} (risk-neutral martingale) ---
    cf_m = heston_nandi_cf_jax(
        jnp.array([-1.0j]), omega, alpha, beta, gamma, h0, tau, r, spy
    )
    target = jnp.exp(r * tau)
    print(f"phi_X(-i)  = {complex(cf_m[0]):.10f}  (expect e^(r·tau) = {float(target):.10f})")
    assert abs(complex(cf_m[0]) - complex(target)) < 1e-8

    # --- Differentiability w.r.t. omega via jacfwd ---
    u_grid = jnp.linspace(0.1, 20.0, 16)

    def mean_abs(w: float) -> jax.Array:
        return jnp.mean(jnp.abs(heston_nandi_cf_jax(u_grid, w, alpha, beta, gamma, h0, tau, r, spy)))

    g = jax.grad(mean_abs)(omega)
    print(f"d|CF|/d_omega = {float(g):.4e}  (finite)")
    assert jnp.isfinite(g)

    print("\nAll JAX CF smoke tests passed")
