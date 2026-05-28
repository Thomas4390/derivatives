"""
JAX port of the Carr-Madan FFT option pricer.
==============================================

Reference: backend/engines/fourier/carr_madan.py (numpy+numba version).

This port is end-to-end differentiable via ``jax.grad`` / ``jax.jacfwd``
/ ``jax.jacrev`` with respect to the characteristic function parameters
— enabling Levenberg-Marquardt calibration with analytical Jacobians.

The pre-computed FFT grids (v_grid, Simpson weights, lambda_spacing) are
constructed at build time (static args) and kept outside the traced
function so the JIT compiles once per (n_fft, eta, alpha) tuple.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from backend.utils.constants import (
    FFT_DEFAULT_ALPHA as DEFAULT_ALPHA,
    FFT_DEFAULT_ETA as DEFAULT_ETA,
    FFT_DEFAULT_N as DEFAULT_N_FFT,
)


# --------------------------------------------------------------------------- #
# Static grid builders (numpy — evaluated once at construction time)
# --------------------------------------------------------------------------- #


def _simpson_weights(n: int, eta: float) -> jnp.ndarray:
    w = np.ones(n, dtype=np.float64)
    w[1::2] = 4.0
    w[2::2] = 2.0
    w[0] = 1.0
    w[-1] = 1.0
    return jnp.asarray(w * eta / 3.0)


@dataclass(frozen=True)
class JaxFFTGrids:
    """Immutable FFT grids cached once per configuration."""

    alpha: float
    n_fft: int
    eta: float
    v_grid: jnp.ndarray  # shape (n_fft,), float64
    simpson: jnp.ndarray  # shape (n_fft,), float64
    lambda_spacing: float  # 2*pi / (n_fft * eta)

    @classmethod
    def build(
        cls,
        alpha: float = DEFAULT_ALPHA,
        n_fft: int = DEFAULT_N_FFT,
        eta: float = DEFAULT_ETA,
    ) -> "JaxFFTGrids":
        if n_fft & (n_fft - 1):
            raise ValueError(f"n_fft must be a power of 2, got {n_fft}")
        v = jnp.arange(n_fft, dtype=jnp.float64) * eta
        return cls(
            alpha=alpha,
            n_fft=n_fft,
            eta=eta,
            v_grid=v,
            simpson=_simpson_weights(n_fft, eta),
            lambda_spacing=2.0 * np.pi / (n_fft * eta),
        )


# --------------------------------------------------------------------------- #
# Core pricing routines (pure, jit-able)
# --------------------------------------------------------------------------- #


def carr_madan_call_prices(
    cf_values: jnp.ndarray,  # phi(u) evaluated on the shifted grid
    v_grid: jnp.ndarray,
    simpson: jnp.ndarray,
    lambda_spacing: float,
    alpha: float,
    spot: float,
    rate: float,
    tau: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return damped call prices on the log-strike grid.

    Outputs
    -------
    damped_prices : (n_fft,) — call prices at the equispaced log-strikes
    log_strikes   : (n_fft,) — log-strike grid aligned with damped_prices
    """
    denom = alpha**2 + alpha - v_grid**2 + 1j * (2.0 * alpha + 1.0) * v_grid
    integrand = jnp.exp(-rate * tau) * cf_values / denom

    log_spot = jnp.log(spot)
    # Phase shift equivalent to numpy engine: exp(-1j * v * (-log_s0)) = exp(1j*v*log_s0)
    x = integrand * simpson * jnp.exp(1j * v_grid * log_spot)
    fft_result = jnp.fft.fft(x)

    log_strikes = -log_spot + lambda_spacing * jnp.arange(v_grid.shape[0])
    damped = jnp.exp(-alpha * log_strikes) / jnp.pi * jnp.real(fft_result)
    return damped, log_strikes


def interpolate_strikes(
    damped_prices: jnp.ndarray,
    log_strikes_grid: jnp.ndarray,
    log_strikes_target: jnp.ndarray,
    lambda_spacing: float,
    spot: float,
    n_fft: int,
) -> jnp.ndarray:
    """Linearly interpolate damped prices at target log-strikes.

    Mirrors the numpy engine's linear interpolation scheme:
        idx = floor((log_k + log_s0) / lambda_spacing)
    """
    log_spot = jnp.log(spot)
    raw_idx = (log_strikes_target + log_spot) / lambda_spacing
    idx = jnp.clip(jnp.floor(raw_idx).astype(jnp.int32), 0, n_fft - 2)

    lower = damped_prices[idx]
    upper = damped_prices[idx + 1]

    w = (log_strikes_target - log_strikes_grid[idx]) / lambda_spacing
    w = jnp.clip(w, 0.0, 1.0)

    price = (1.0 - w) * lower + w * upper
    return jnp.maximum(price, 0.0)


def price_call_strikes_jax(
    cf_fn: Callable[[jnp.ndarray], jnp.ndarray],
    spot: float,
    strikes: jnp.ndarray,
    tau: float,
    rate: float,
    grids: JaxFFTGrids,
) -> jnp.ndarray:
    """Price a vector of European call strikes under one maturity.

    ``cf_fn(u)`` must return the characteristic function of the LOG-RETURN
    process X_T = log(S_T / S_0), i.e. phi_log_R(u) = E[exp(i*u*X_T)].
    The log-spot shift needed by Carr-Madan (which expects the log-PRICE
    CF) is applied internally here so the JAX CF modules
    (``heston_cf_jax``, ``merton_cf_jax``, ``bates_cf_jax``) can stay
    spot-agnostic and therefore independent of any market environment.
    """
    u = grids.v_grid - (grids.alpha + 1.0) * 1j
    log_spot = jnp.log(spot)
    # phi_log_S(u) = phi_log_R(u) * exp(i * u * log_spot)
    cf_vals = cf_fn(u) * jnp.exp(1j * u * log_spot)
    damped, log_strikes_grid = carr_madan_call_prices(
        cf_vals,
        grids.v_grid,
        grids.simpson,
        grids.lambda_spacing,
        grids.alpha,
        spot,
        rate,
        tau,
    )
    return interpolate_strikes(
        damped,
        log_strikes_grid,
        jnp.log(strikes),
        grids.lambda_spacing,
        spot,
        grids.n_fft,
    )


# --------------------------------------------------------------------------- #
# Convenience: price_call wrapper matching the numpy engine API
# --------------------------------------------------------------------------- #


@partial(jax.jit, static_argnames=("grids",))
def _price_call_strikes_jit(
    cf_vals: jnp.ndarray,
    spot: float,
    strikes: jnp.ndarray,
    tau: float,
    rate: float,
    grids: JaxFFTGrids,
) -> jnp.ndarray:
    """JIT-friendly inner helper: CF values already evaluated upstream."""
    damped, log_strikes_grid = carr_madan_call_prices(
        cf_vals,
        grids.v_grid,
        grids.simpson,
        grids.lambda_spacing,
        grids.alpha,
        spot,
        rate,
        tau,
    )
    return interpolate_strikes(
        damped,
        log_strikes_grid,
        jnp.log(strikes),
        grids.lambda_spacing,
        spot,
        grids.n_fft,
    )
