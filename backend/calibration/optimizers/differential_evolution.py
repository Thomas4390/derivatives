"""
Differential Evolution strategy (scipy global derivative-free)
================================================================

Population-based stochastic global optimiser — re-introduced from the
legacy V1 calibration stack.  Pedagogically interesting because it
explores the parameter space globally and converges from random starts
without a gradient, at the cost of being 5-50x slower than LM.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from backend.calibration.optimizers.base import (
    CalibrationProblem,
    IterationLogger,
    OptimizationResult,
    StrategyMetadata,
)


@dataclass
class DifferentialEvolutionStrategy:
    """scipy.optimize.differential_evolution wrapped as an OptimizerStrategy.

    Parameters
    ----------
    population_size : int
        Multiplier for ``popsize * len(x)`` — total population size.
    mutation : tuple[float, float]
        Mutation constant range (dithering).
    recombination : float
        Crossover probability.
    seed : int | None
        Base reproducibility seed. The actual seed passed to scipy is
        ``seed + restart_index`` where ``restart_index`` is incremented on
        each call to :meth:`solve`. This keeps each call reproducible while
        ensuring multi-start loops produce diverse trajectories instead of
        identical runs.
    polish : bool
        If True, scipy runs a final L-BFGS-B polish on the best member.
        Disabled by default to keep the comparison "pure DE".
    workers : int
        Parallel workers (1 = serial).  Streamlit + JAX is not always
        thread-safe so keep the default.
    """

    population_size: int = 15
    mutation: tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    seed: int | None = 42
    polish: bool = False
    workers: int = 1
    init: str = "sobol"  # "latinhypercube" / "sobol" / "halton"
    # Auto-incremented per .solve() call so n_restarts produces diverse runs.
    _restart_counter: int = field(default=0, init=False, repr=False)

    metadata: StrategyMetadata = field(
        default_factory=lambda: StrategyMetadata(
            name="DE",
            is_global=True,
            requires_gradient=False,
            requires_residuals=False,
            description=(
                "Differential Evolution: stochastic population-based global "
                "optimiser. Robust against local minima but ~10x slower than "
                "LM on smooth problems. Ideal pedagogy for showing global vs "
                "local search."
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
        tol: float = 1e-7,
        **kwargs: Any,
    ) -> OptimizationResult:
        if not problem.has_objective():
            raise ValueError("DifferentialEvolutionStrategy needs a scalar objective.")

        # Wrap scalar objective for logging, then count evaluations so the
        # callback can stop DE the moment the max_nfev budget is exhausted.
        base_objective: Any = problem.evaluate_objective
        if logger is not None:
            base_objective = logger.wrap_objective(base_objective)
        nfev = {"n": 0}

        def objective(x: np.ndarray) -> float:
            nfev["n"] += 1
            return base_objective(x)

        bounds = list(zip(problem.bounds_lo, problem.bounds_hi))
        if any(not np.isfinite(lo) or not np.isfinite(hi) for lo, hi in bounds):
            raise ValueError(
                "DifferentialEvolution requires finite bounds on every parameter."
            )

        # Approximate maxiter from max_nfev: each generation costs ~popsize*N
        # evals. The _callback below enforces the exact budget as a hard cap, so
        # the worst-case overshoot is a single (already-evaluated) generation.
        N = problem.x0.size
        approx_evals_per_gen = max(self.population_size * N, 1)
        maxiter = max(1, max_nfev // approx_evals_per_gen)
        # Diverse seed per call so multi-start loops produce different
        # trajectories. Reproducible because the offset is deterministic.
        if self.seed is not None:
            effective_seed: int | None = int(self.seed) + int(self._restart_counter)
        else:
            effective_seed = None
        self._restart_counter += 1

        # Native iteration callback: record a per-generation mark, then stop the
        # run as soon as the max_nfev budget is reached. DE has no native nfev
        # cap, and the callback fires at generation boundaries, so the overshoot
        # is bounded by one already-evaluated generation.
        def _callback(intermediate_result):
            if logger is not None:
                try:
                    x = np.asarray(intermediate_result.x, dtype=float)
                    fval = float(intermediate_result.fun)
                    logger.record_callback(x, objective=fval)
                except (AttributeError, TypeError):
                    pass
            return nfev["n"] >= max_nfev  # True → stop DE

        t0 = time.perf_counter()
        res = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=int(maxiter),
            popsize=int(self.population_size),
            mutation=self.mutation,
            recombination=float(self.recombination),
            seed=effective_seed,
            polish=bool(self.polish),
            workers=int(self.workers),
            tol=float(tol),
            init=self.init,
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


__all__ = ["DifferentialEvolutionStrategy"]
