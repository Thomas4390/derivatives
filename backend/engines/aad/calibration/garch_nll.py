"""
JAX implementations of GARCH/NGARCH/GJR-GARCH negative log-likelihoods.
========================================================================

Ported from backend.calibration.garch_calibrator (Numba) to JAX, using
``jax.lax.scan`` for the variance recursion. Fully differentiable via
``jax.grad`` / ``jax.jacfwd`` — enables exact gradient-based MLE and
per-observation score vectors for BHHH covariance.

All three functions return the scalar NLL (with stationarity penalty)
and expose a sibling that yields the per-observation log-likelihood
contributions, used for BHHH standard errors.

Reference: backend/calibration/garch_calibrator.py

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations


import jax
import jax.numpy as jnp

from backend.utils.constants.numerical import (
    GARCH_CALIBRATION_VARIANCE_FLOOR as _VARIANCE_FLOOR,
    LOG_2PI as _LOG2PI,
)


# --------------------------------------------------------------------------- #
# GARCH(1,1)
# --------------------------------------------------------------------------- #


def _garch_step(var_t: jnp.ndarray, r_t: jnp.ndarray, omega, alpha, beta):
    """One step of the GARCH variance recursion, returning (new_var, logL_t)."""
    var_t = jnp.maximum(var_t, _VARIANCE_FLOOR)
    logL_t = -0.5 * (_LOG2PI + jnp.log(var_t) + r_t * r_t / var_t)
    z = r_t / jnp.sqrt(var_t)
    var_next = omega + alpha * var_t * z * z + beta * var_t
    return var_next, logL_t


def garch_per_obs_logL(params: jnp.ndarray, returns: jnp.ndarray) -> jnp.ndarray:
    """Per-observation log-likelihood vector for GARCH(1,1).

    Parameters
    ----------
    params : jnp.ndarray, shape (3,)
        [omega, alpha, beta]
    returns : jnp.ndarray, shape (T,)

    Returns
    -------
    jnp.ndarray, shape (T,)
        ``log L_t`` for each observation.
    """
    omega, alpha, beta = params[0], params[1], params[2]
    var0 = jnp.var(returns)

    def _scan(var_t, r_t):
        return _garch_step(var_t, r_t, omega, alpha, beta)

    _, logLs = jax.lax.scan(_scan, var0, returns)
    return logLs


def neg_log_likelihood_garch(params, returns) -> jnp.ndarray:
    """GARCH(1,1) negative log-likelihood with stationarity penalty."""
    alpha, beta = params[1], params[2]
    persistence = alpha + beta
    # Continuous stationarity barrier: a soft quadratic that already has a
    # non-zero gradient as persistence approaches 1 (so L-BFGS-B is pushed
    # back below ~0.995 *before* reaching the non-stationary region), plus a
    # hard wall once persistence >= 1. Replaces the previous discontinuous
    # ``jnp.where(p >= 1, 1e6*(p-0.999)**2, 0)`` whose ~1.0 value at p=1 was
    # negligible vs. an NLL of order 1e3-1e4.
    penalty = 1e4 * jnp.maximum(persistence - 0.995, 0.0) ** 2 + jnp.where(
        persistence >= 1.0, 1e8 * (persistence - 0.995) ** 2, 0.0
    )
    return -jnp.sum(garch_per_obs_logL(params, returns)) + penalty


# --------------------------------------------------------------------------- #
# NGARCH
# --------------------------------------------------------------------------- #


def _ngarch_step(var_t, r_t, omega, alpha, beta, gamma):
    var_t = jnp.maximum(var_t, _VARIANCE_FLOOR)
    logL_t = -0.5 * (_LOG2PI + jnp.log(var_t) + r_t * r_t / var_t)
    z = r_t / jnp.sqrt(var_t)
    var_next = omega + alpha * var_t * (z - gamma) ** 2 + beta * var_t
    return var_next, logL_t


def ngarch_per_obs_logL(params, returns):
    omega, alpha, beta, gamma = params[0], params[1], params[2], params[3]
    var0 = jnp.var(returns)

    def _scan(var_t, r_t):
        return _ngarch_step(var_t, r_t, omega, alpha, beta, gamma)

    _, logLs = jax.lax.scan(_scan, var0, returns)
    return logLs


def neg_log_likelihood_ngarch(params, returns):
    alpha, beta, gamma = params[1], params[2], params[3]
    persistence = alpha * (1.0 + gamma * gamma) + beta
    # Continuous stationarity barrier: a soft quadratic that already has a
    # non-zero gradient as persistence approaches 1 (so L-BFGS-B is pushed
    # back below ~0.995 *before* reaching the non-stationary region), plus a
    # hard wall once persistence >= 1. Replaces the previous discontinuous
    # ``jnp.where(p >= 1, 1e6*(p-0.999)**2, 0)`` whose ~1.0 value at p=1 was
    # negligible vs. an NLL of order 1e3-1e4.
    penalty = 1e4 * jnp.maximum(persistence - 0.995, 0.0) ** 2 + jnp.where(
        persistence >= 1.0, 1e8 * (persistence - 0.995) ** 2, 0.0
    )
    return -jnp.sum(ngarch_per_obs_logL(params, returns)) + penalty


# --------------------------------------------------------------------------- #
# GJR-GARCH
# --------------------------------------------------------------------------- #


def _gjr_step(var_t, r_t, omega, alpha, beta, gamma):
    var_t = jnp.maximum(var_t, _VARIANCE_FLOOR)
    logL_t = -0.5 * (_LOG2PI + jnp.log(var_t) + r_t * r_t / var_t)
    z = r_t / jnp.sqrt(var_t)
    indicator = jnp.where(z < 0.0, 1.0, 0.0)
    var_next = omega + (alpha + gamma * indicator) * var_t * z * z + beta * var_t
    return var_next, logL_t


def gjr_per_obs_logL(params, returns):
    omega, alpha, beta, gamma = params[0], params[1], params[2], params[3]
    var0 = jnp.var(returns)

    def _scan(var_t, r_t):
        return _gjr_step(var_t, r_t, omega, alpha, beta, gamma)

    _, logLs = jax.lax.scan(_scan, var0, returns)
    return logLs


def neg_log_likelihood_gjr(params, returns):
    alpha, beta, gamma = params[1], params[2], params[3]
    persistence = alpha + 0.5 * gamma + beta
    # Continuous stationarity barrier: a soft quadratic that already has a
    # non-zero gradient as persistence approaches 1 (so L-BFGS-B is pushed
    # back below ~0.995 *before* reaching the non-stationary region), plus a
    # hard wall once persistence >= 1. Replaces the previous discontinuous
    # ``jnp.where(p >= 1, 1e6*(p-0.999)**2, 0)`` whose ~1.0 value at p=1 was
    # negligible vs. an NLL of order 1e3-1e4.
    penalty = 1e4 * jnp.maximum(persistence - 0.995, 0.0) ** 2 + jnp.where(
        persistence >= 1.0, 1e8 * (persistence - 0.995) ** 2, 0.0
    )
    return -jnp.sum(gjr_per_obs_logL(params, returns)) + penalty


# --------------------------------------------------------------------------- #
# JIT-compiled versions with grad
# --------------------------------------------------------------------------- #


nll_garch_jit = jax.jit(neg_log_likelihood_garch)
nll_ngarch_jit = jax.jit(neg_log_likelihood_ngarch)
nll_gjr_jit = jax.jit(neg_log_likelihood_gjr)

nll_garch_grad = jax.jit(jax.grad(neg_log_likelihood_garch))
nll_ngarch_grad = jax.jit(jax.grad(neg_log_likelihood_ngarch))
nll_gjr_grad = jax.jit(jax.grad(neg_log_likelihood_gjr))

# Per-observation score (Jacobian of log-likelihood vector wrt params)
scores_garch_jit = jax.jit(jax.jacfwd(garch_per_obs_logL))
scores_ngarch_jit = jax.jit(jax.jacfwd(ngarch_per_obs_logL))
scores_gjr_jit = jax.jit(jax.jacfwd(gjr_per_obs_logL))
