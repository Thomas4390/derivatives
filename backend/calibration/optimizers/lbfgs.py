"""
L-BFGS-B strategy (scipy quasi-Newton with bounds & gradient)
==============================================================

Limited-memory BFGS with box constraints — gradient-based local
optimiser.  Pedagogically interesting as a *gradient* counterpoint to
LM (which uses a Jacobian) and as a *local-only* counterpoint to DE.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from backend.calibration.optimizers.base import (
    CalibrationProblem,
    IterationLogger,
    OptimizationResult,
    StrategyMetadata,
)


@dataclass
class LBFGSStrategy:
    """L-BFGS-B with analytical gradient when available, FD otherwise.

    Parameters
    ----------
    ftol : float
        Tolerance on objective relative change.
    gtol : float
        Tolerance on projected gradient infinity-norm.
    maxcor : int
        Limited-memory pair count (history length).
    """

    ftol: float = 1e-10
    gtol: float = 1e-8
    maxcor: int = 10

    metadata: StrategyMetadata = field(
        default_factory=lambda: StrategyMetadata(
            name="L-BFGS-B",
            is_global=False,
            requires_gradient=False,  # falls back to FD
            requires_residuals=False,
            description=(
                "L-BFGS-B: limited-memory BFGS quasi-Newton with box bounds "
                "and analytical gradient (when available). Local gradient-"
                "based solver — fast and accurate near a basin, but needs a "
                "good initial guess. Counter-example to LM: scalar objective "
                "instead of residual vector."
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
        if not problem.has_objective():
            raise ValueError("LBFGSStrategy needs a scalar objective.")

        objective: Any = problem.evaluate_objective
        if logger is not None:
            objective = logger.wrap_objective(objective)

        # Use analytical gradient if the problem can derive one
        # (either explicit gradient_fn or derived from J^T r).
        gradient: Any = None
        if problem.has_gradient():
            gradient = problem.evaluate_gradient

        bounds_arg: Any = None
        if np.all(np.isfinite(problem.bounds_lo)) and np.all(
            np.isfinite(problem.bounds_hi)
        ):
            bounds_arg = list(zip(problem.bounds_lo, problem.bounds_hi))

        ftol = float(tol) if tol is not None else self.ftol
        gtol = float(tol) if tol is not None else self.gtol

        options: dict[str, Any] = {
            "ftol": ftol,
            "gtol": gtol,
            "maxcor": int(self.maxcor),
            "maxfun": int(max_nfev),
            "maxiter": int(max_nfev),
        }

        def _callback(xk: np.ndarray) -> None:
            if logger is None:
                return
            logger.record_callback(np.asarray(xk, dtype=float))

        t0 = time.perf_counter()
        res = minimize(
            objective,
            problem.x0.astype(float),
            method="L-BFGS-B",
            jac=gradient,
            bounds=bounds_arg,
            options=options,
            callback=_callback,
        )
        elapsed = time.perf_counter() - t0

        grad_norm: float | None = None
        if hasattr(res, "jac") and res.jac is not None:
            try:
                grad_norm = float(np.linalg.norm(res.jac))
            except (TypeError, ValueError):
                grad_norm = None

        return OptimizationResult(
            x_optimal=np.asarray(res.x, dtype=float),
            objective_value=float(res.fun),
            n_iterations=int(res.nit),
            n_function_evals=int(res.nfev),
            success=bool(res.success),
            method=self.name,
            elapsed_seconds=elapsed,
            grad_norm=grad_norm,
            iteration_history=logger.history if logger is not None else (),
            raw_result=res,
            message=str(res.message),
        )


__all__ = ["LBFGSStrategy"]
