"""
Levenberg-Marquardt strategy (Trust-Region-Reflective + JAX Jacobian)
======================================================================

Wraps :func:`scipy.optimize.least_squares(method='trf')` so the V2
calibrators (which already supply JIT-compiled JAX residuals + analytic
Jacobians) can drive it through the unified Strategy interface.

This is the *production default*: it is the same algorithm + tolerances
that all V2 calibrators have been calling directly via ``run_lm()``.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from backend.calibration.optimizers.base import (
    CalibrationProblem,
    IterationLogger,
    OptimizationResult,
    StrategyMetadata,
)


@dataclass
class LMJaxStrategy:
    """Trust-Region-Reflective LM with analytical JAX Jacobian.

    The strategy operates on the *least-squares* view of the problem
    (``residual_fn`` + ``jacobian_fn``).  Calling it on a problem
    without residuals raises ``ValueError``.

    Parameters
    ----------
    ftol, xtol, gtol : float
        scipy LM convergence tolerances (default ``1e-10``).
    use_bounds : bool
        If True, pass the problem bounds to scipy (TRF supports them).
        The V2 calibrators use unconstrained internal coordinates so
        the default is False — bounds are enforced by reparametrisation.
    """

    ftol: float = 1e-10
    xtol: float = 1e-10
    gtol: float = 1e-10
    use_bounds: bool = False

    metadata: StrategyMetadata = field(
        default_factory=lambda: StrategyMetadata(
            name="LM-JAX",
            is_global=False,
            requires_gradient=False,  # uses Jacobian directly
            requires_residuals=True,
            description=(
                "Levenberg-Marquardt (Trust-Region-Reflective) with analytical "
                "JAX Jacobian. Industry-standard local solver for option-surface "
                "calibration. Fast convergence, requires good initial guess."
            ),
        )
    )

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def is_global(self) -> bool:
        return self.metadata.is_global

    @property
    def requires_gradient(self) -> bool:
        return self.metadata.requires_gradient

    @property
    def requires_residuals(self) -> bool:
        return self.metadata.requires_residuals

    # ------------------------------------------------------------------ #

    def solve(
        self,
        problem: CalibrationProblem,
        logger: IterationLogger | None = None,
        *,
        max_nfev: int = 200,
        tol: float | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        if not problem.has_residuals():
            raise ValueError(
                "LMJaxStrategy requires a least-squares problem (residual_fn). "
                "GARCH-style scalar problems must use a different strategy."
            )

        ftol = float(tol) if tol is not None else self.ftol
        xtol = float(tol) if tol is not None else self.xtol
        gtol = float(tol) if tol is not None else self.gtol

        # Wire the logger into the residual evaluations so the history
        # captures every Jacobian / residual pair.
        residual_fn = problem.residual_fn
        jacobian_fn = problem.jacobian_fn
        if logger is not None:
            residual_fn = logger.wrap_residual(residual_fn)

        bounds: tuple[Any, Any]
        if self.use_bounds:
            bounds = (problem.bounds_lo, problem.bounds_hi)
        else:
            bounds = (-np.inf, np.inf)

        t0 = time.perf_counter()
        if jacobian_fn is None:
            res = least_squares(
                residual_fn,
                problem.x0,
                method="trf",
                bounds=bounds,
                max_nfev=int(max_nfev),
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
            )
        else:
            res = least_squares(
                residual_fn,
                problem.x0,
                jac=jacobian_fn,
                method="trf",
                bounds=bounds,
                max_nfev=int(max_nfev),
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
            )
        elapsed = time.perf_counter() - t0

        # scipy stores 0.5 * ||r||^2 in res.cost
        objective_value = float(res.cost)
        grad_norm: float | None
        try:
            grad_norm = float(np.linalg.norm(res.grad))
        except (AttributeError, ValueError):
            grad_norm = None

        history = logger.history if logger is not None else ()

        return OptimizationResult(
            x_optimal=np.asarray(res.x, dtype=float),
            objective_value=objective_value,
            n_iterations=int(getattr(res, "njev", res.nfev)),
            n_function_evals=int(res.nfev),
            success=bool(res.success),
            method=self.name,
            elapsed_seconds=elapsed,
            grad_norm=grad_norm,
            iteration_history=history,
            raw_result=res,
            message=str(res.message),
        )


__all__ = ["LMJaxStrategy"]
