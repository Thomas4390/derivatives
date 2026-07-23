"""
Parameter Transforms for Calibration
=====================================

Bijective maps between constrained parameter spaces and unconstrained
real space. Enables gradient-based optimizers (L-BFGS, Levenberg-Marquardt)
to operate without explicit bounds — all constraints are encoded in the
transform.

Why:
    Bounded optimizers (L-BFGS-B, trust-region with bounds) can stall
    or oscillate near the boundary. Reparameterizing with smooth
    bijections moves the constraint into the objective, letting the
    optimizer work in R^n where standard convergence theory applies.

Transforms
----------
sigmoid(p)  : (lo, hi) <-> R      -- bounded interval (e.g. rho in [-1, 1])
softplus(p) : (lo, +inf) <-> R    -- positive params with floor (e.g. kappa >= 0.01)
identity(p) : same space          -- unbounded params

All transforms expose:
    forward(raw_param) -> unconstrained
    inverse(u)         -> raw_param
    log_abs_det_jac(u) -> Jacobian correction for probabilistic uses

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Transform(Protocol):
    """Bijection between a constrained parameter and R."""

    def forward(self, p: float) -> float:
        """Map constrained -> unconstrained."""
        ...

    def inverse(self, u: float) -> float:
        """Map unconstrained -> constrained."""
        ...


@dataclass(frozen=True)
class SigmoidTransform:
    """Map (lo, hi) <-> R via scaled logit/sigmoid.

    forward(p) = log((p - lo) / (hi - p))
    inverse(u) = lo + (hi - lo) / (1 + exp(-u))

    Clipped internally to avoid overflow near the endpoints.
    """

    lo: float
    hi: float
    clip: float = 1e-8

    def __post_init__(self) -> None:
        if self.hi <= self.lo:
            raise ValueError(f"hi ({self.hi}) must exceed lo ({self.lo})")

    def forward(self, p: float) -> float:
        # Clip the FRACTION (not the param) so forward/inverse share one clip
        # space and round-trip to machine precision; clipping the param in
        # absolute units left a span-dependent ~clip boundary gap vs inverse.
        span = self.hi - self.lo
        frac = float(np.clip((p - self.lo) / span, self.clip, 1.0 - self.clip))
        return float(np.log(frac / (1.0 - frac)))

    def inverse(self, u: float) -> float:
        # Stable sigmoid — avoids overflow for |u| large.
        u_arr = np.asarray(u, dtype=float)
        pos = u_arr >= 0
        out = np.empty_like(u_arr)
        ex = np.exp(-np.abs(u_arr))
        out[pos] = 1.0 / (1.0 + ex[pos])
        out[~pos] = ex[~pos] / (1.0 + ex[~pos])
        # Clamp strictly inside (lo, hi) to avoid degenerate CF evaluations
        # when the optimizer explores extreme unconstrained values.
        out = np.clip(out, self.clip, 1.0 - self.clip)
        return float(self.lo + (self.hi - self.lo) * out)


@dataclass(frozen=True)
class SoftplusTransform:
    """Map (lo, +inf) <-> R via shifted softplus.

    forward(p) = log(exp(p - lo) - 1)
    inverse(u) = lo + log(1 + exp(u))      (stable form)
    """

    lo: float = 0.0
    clip: float = 1e-8

    def forward(self, p: float) -> float:
        shifted = max(float(p) - self.lo, self.clip)
        # log(exp(shifted) - 1) stable for large shifted
        if shifted > 30.0:
            return float(shifted)
        return float(np.log(np.expm1(shifted)))

    def inverse(self, u: float) -> float:
        u_f = float(u)
        if u_f > 30.0:
            return self.lo + u_f
        # Floor at `clip` so extreme negative u (underflow of exp) still
        # yields a strictly positive offset above `lo`.
        return float(self.lo + max(float(np.log1p(np.exp(u_f))), self.clip))


@dataclass(frozen=True)
class IdentityTransform:
    """Pass-through for unbounded parameters."""

    def forward(self, p: float) -> float:
        return float(p)

    def inverse(self, u: float) -> float:
        return float(u)


@dataclass(frozen=True)
class ParameterTransform:
    """Composition of per-parameter transforms.

    Holds an ordered tuple of Transform objects matching a parameter
    vector, providing vectorized forward/inverse over numpy arrays.
    """

    transforms: tuple[Transform, ...]

    def forward(self, params: np.ndarray) -> np.ndarray:
        """Constrained -> unconstrained (for seeding the optimizer)."""
        params = np.asarray(params, dtype=float)
        if params.shape != (len(self.transforms),):
            raise ValueError(
                f"expected length {len(self.transforms)}, got {params.shape}"
            )
        return np.array(
            [t.forward(float(p)) for t, p in zip(self.transforms, params)],
            dtype=float,
        )

    def inverse(self, u: np.ndarray) -> np.ndarray:
        """Unconstrained -> constrained (for evaluating the model)."""
        u = np.asarray(u, dtype=float)
        if u.shape != (len(self.transforms),):
            raise ValueError(f"expected length {len(self.transforms)}, got {u.shape}")
        return np.array(
            [t.inverse(float(x)) for t, x in zip(self.transforms, u)],
            dtype=float,
        )


# =============================================================================
# Model-specific transform factories
# =============================================================================


def heston_transform() -> ParameterTransform:
    """Transform for Heston [v0, kappa, theta, alpha, rho].

    v0, theta : (1e-5, 1.0)    -- variance
    kappa     : (0.01, 20.0)   -- mean-reversion speed
    alpha        : (1e-3, 2.0)    -- vol-of-vol
    rho       : (-0.999, 0.999)-- correlation
    """
    return ParameterTransform(
        (
            SigmoidTransform(1e-5, 1.0),
            SigmoidTransform(0.01, 20.0),
            SigmoidTransform(1e-5, 1.0),
            SigmoidTransform(1e-3, 2.0),
            SigmoidTransform(-0.999, 0.999),
        )
    )


def merton_transform() -> ParameterTransform:
    """Transform for Merton [sigma, lam, alpha_j, sigma_j].

    sigma     : (1e-3, 2.0)   -- diffusion vol
    lam  : (0.0, 10.0)   -- jump intensity
    alpha_j      : (-2.0, 2.0)   -- jump mean (unbounded in theory, reasonable clip)
    sigma_j   : (1e-3, 2.0)   -- jump std
    """
    return ParameterTransform(
        (
            SigmoidTransform(1e-3, 2.0),
            SigmoidTransform(1e-6, 10.0),
            SigmoidTransform(-2.0, 2.0),
            SigmoidTransform(1e-3, 2.0),
        )
    )


def bates_transform() -> ParameterTransform:
    """Transform for Bates [v0, kappa, theta, alpha, rho, lam, alpha_j, sigma_j]."""
    return ParameterTransform(
        (
            SigmoidTransform(1e-5, 1.0),
            SigmoidTransform(0.01, 20.0),
            SigmoidTransform(1e-5, 1.0),
            SigmoidTransform(1e-3, 2.0),
            SigmoidTransform(-0.999, 0.999),
            SigmoidTransform(1e-6, 10.0),
            SigmoidTransform(-2.0, 2.0),
            SigmoidTransform(1e-3, 2.0),
        )
    )
