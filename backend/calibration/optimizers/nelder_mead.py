"""
Nelder-Mead simplex strategy (scipy local derivative-free)
=============================================================

Classic simplex algorithm — re-introduced from the legacy V1 stack.
Pedagogically valuable: visually intuitive (geometric simplex moves),
no derivatives required, but converges slowly on >5-dim problems and
gets trapped in local minima.

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
class NelderMeadStrategy:
    """Nelder-Mead simplex method through scipy.optimize.minimize.

    Bounds are passed to the (newer) scipy implementation that supports
    them; for older scipy versions, parameters can wander outside the
    box but the calibrators always reparametrise with sigmoid bijections
    so this is harmless in practice.

    Parameters
    ----------
    xatol : float
        Absolute tolerance on parameter changes.
    fatol : float
        Absolute tolerance on objective change.
    initial_simplex_scale : float
        Scale of the initial simplex relative to ``x0``.
    """

    xatol: float = 1e-6
    fatol: float = 1e-8
    initial_simplex_scale: float = 0.05
    adaptive: bool = True

    metadata: StrategyMetadata = field(
        default_factory=lambda: StrategyMetadata(
            name="NM",
            is_global=False,
            requires_gradient=False,
            requires_residuals=False,
            description=(
                "Nelder-Mead simplex: derivative-free local optimiser using "
                "geometric reflections / expansions / contractions. Slow on "
                ">5-dim problems but very intuitive — great pedagogy for "
                "showing derivative-free search."
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
            raise ValueError("NelderMeadStrategy needs a scalar objective.")

        objective: Any = problem.evaluate_objective
        if logger is not None:
            objective = logger.wrap_objective(objective)

        # NM accepts bounds in modern scipy; pass them only if finite
        bounds_arg: Any = None
        if np.all(np.isfinite(problem.bounds_lo)) and np.all(
            np.isfinite(problem.bounds_hi)
        ):
            bounds_arg = list(zip(problem.bounds_lo, problem.bounds_hi))

        # Build an initial simplex around x0 to give NM a chance on
        # multi-dim problems: dim+1 vertices in a small box around x0.
        n = problem.x0.size
        scale = self.initial_simplex_scale
        simplex = np.tile(problem.x0, (n + 1, 1)).astype(float)
        for i in range(n):
            step = scale * (
                abs(problem.x0[i]) if problem.x0[i] != 0.0 else 1.0
            )
            simplex[i + 1, i] += step

        options: dict[str, Any] = {
            "xatol": float(tol) if tol is not None else self.xatol,
            "fatol": float(tol) if tol is not None else self.fatol,
            "maxfev": int(max_nfev),
            "adaptive": bool(self.adaptive),
            "initial_simplex": simplex,
        }

        # scipy minimize callback: called after each iteration with current xk
        def _callback(xk: np.ndarray) -> None:
            if logger is None:
                return
            logger.record_callback(np.asarray(xk, dtype=float))

        t0 = time.perf_counter()
        res = minimize(
            objective,
            problem.x0.astype(float),
            method="Nelder-Mead",
            bounds=bounds_arg,
            options=options,
            callback=_callback,
        )
        elapsed = time.perf_counter() - t0

        return OptimizationResult(
            x_optimal=np.asarray(res.x, dtype=float),
            objective_value=float(res.fun),
            n_iterations=int(res.nit),
            n_function_evals=int(res.nfev),
            success=bool(res.success),
            method=self.name,
            elapsed_seconds=elapsed,
            grad_norm=None,
            iteration_history=logger.history if logger is not None else (),
            raw_result=res,
            message=str(res.message),
        )


__all__ = ["NelderMeadStrategy"]
