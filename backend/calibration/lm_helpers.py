"""
Shared Levenberg-Marquardt helpers for V2 JAX-based calibrators
================================================================

The four V2 calibrators (Heston, Merton, Bates, GARCH) all wrap
``scipy.optimize.least_squares`` around a JIT-compiled JAX residual + Jacobian
pair. This module hosts the strictly identical pieces so each calibrator only
keeps its model-specific reparametrisation, residual builder, and post-fit
reporting.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import Any, Protocol

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares


class CompiledResiduals(Protocol):
    """Structural type expected by :func:`run_lm`.

    Calibrator-specific bundles already expose ``residual`` and ``jacobian``
    JAX-JIT closures with the right signature; we only require these two
    attributes here so the helper stays oblivious to model dimensions or
    penalty terms.
    """

    residual: Any
    jacobian: Any


def run_lm(
    compiled: CompiledResiduals,
    x0: np.ndarray,
    max_nfev: int,
    ftol: float,
    xtol: float,
    gtol: float,
):
    """Run scipy ``least_squares`` (TRF) on JAX-compiled residuals.

    The wrapper bridges numpy <-> JAX at the boundary so the optimiser sees
    pure ``np.ndarray`` while the residual / Jacobian themselves remain
    JIT-compiled. This is identical for every V2 LM calibrator and was
    duplicated verbatim in three modules before this extraction.
    """

    def f(x: np.ndarray) -> np.ndarray:
        return np.asarray(compiled.residual(jnp.asarray(x)))

    def jac(x: np.ndarray) -> np.ndarray:
        return np.asarray(compiled.jacobian(jnp.asarray(x)))

    return least_squares(
        f,
        x0,
        jac=jac,
        method="trf",
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
    )


def make_multi_starts(
    x0_base: np.ndarray,
    n_restarts: int,
    near_scale: float,
    far_scale: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Build the list of LM initial guesses for a multi-start run.

    The first start is always ``x0_base`` (the seeded initial guess from ATM
    IV / market priors). The remaining ``n_restarts - 1`` starts are split
    between near-basin perturbations (exploit, scale ``near_scale``) and
    far-field jumps (explore, scale ``far_scale``) so the optimiser has a
    chance to escape local minima on skew-heavy regimes.
    """
    starts: list[np.ndarray] = [x0_base]
    if n_restarts <= 1:
        return starts

    dim = int(x0_base.size)
    n_near = max((n_restarts - 1) // 2, 0)
    n_far = (n_restarts - 1) - n_near
    for _ in range(n_near):
        starts.append(x0_base + rng.normal(scale=near_scale, size=dim))
    for _ in range(n_far):
        starts.append(rng.normal(scale=far_scale, size=dim))
    return starts


__all__ = ["CompiledResiduals", "run_lm", "make_multi_starts"]
