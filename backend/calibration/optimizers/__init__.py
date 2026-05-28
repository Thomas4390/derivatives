"""
Optimizer Strategies for Calibration
======================================

Strategy Pattern for the calibration solvers.  Each strategy
implements the :class:`OptimizerStrategy` protocol and accepts a
unified :class:`CalibrationProblem` describing the objective, bounds,
and initial guess.

Available strategies
--------------------

============== ============== ================== ============================
Name           Type           Needs              Best for
============== ============== ================== ============================
LMJaxStrategy  Local LS       residuals + Jac    Production V2 calibration
DE...Strategy  Global scalar  scalar objective   Avoiding local minima
NelderMead...  Local scalar   scalar objective   Pedagogy / no-derivative
LBFGSStrategy  Local scalar   scalar objective   Gradient-based scalar
============== ============== ================== ============================

Quick example
-------------

>>> from backend.calibration.optimizers import (
...     CalibrationProblem, IterationLogger, LMJaxStrategy
... )
>>> problem = CalibrationProblem(
...     x0=x0, bounds_lo=lo, bounds_hi=hi,
...     param_names=("v0", "kappa", "theta", "alpha", "rho"),
...     residual_fn=residual, jacobian_fn=jacobian,
... )
>>> logger = IterationLogger(problem)
>>> result = LMJaxStrategy().solve(problem, logger=logger, max_nfev=200)
>>> for snap in result.iteration_history: ...

Author: Thomas Vaudescal
Created: 2026
"""

from backend.calibration.optimizers.base import (
    CalibrationProblem,
    GradientFn,
    IterationLogger,
    IterationSnapshot,
    JacobianFn,
    ObjectiveFn,
    OptimizationResult,
    OptimizerStrategy,
    ResidualFn,
    StrategyMetadata,
)
from backend.calibration.optimizers.differential_evolution import (
    DifferentialEvolutionStrategy,
)
from backend.calibration.optimizers.lbfgs import LBFGSStrategy
from backend.calibration.optimizers.lm_jax import LMJaxStrategy
from backend.calibration.optimizers.nelder_mead import NelderMeadStrategy

DEFAULT_STRATEGIES: dict[str, type] = {
    "LM-JAX": LMJaxStrategy,
    "DE": DifferentialEvolutionStrategy,
    "NM": NelderMeadStrategy,
    "L-BFGS-B": LBFGSStrategy,
}


def make_strategy(name: str, **kwargs) -> OptimizerStrategy:
    """Factory: build a strategy by short name."""
    try:
        cls = DEFAULT_STRATEGIES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown strategy '{name}'. Available: {list(DEFAULT_STRATEGIES)}"
        ) from exc
    return cls(**kwargs)


__all__ = [
    # Base contracts
    "CalibrationProblem",
    "IterationLogger",
    "IterationSnapshot",
    "OptimizationResult",
    "OptimizerStrategy",
    "StrategyMetadata",
    # Type aliases
    "ResidualFn",
    "JacobianFn",
    "ObjectiveFn",
    "GradientFn",
    # Concrete strategies
    "LMJaxStrategy",
    "DifferentialEvolutionStrategy",
    "NelderMeadStrategy",
    "LBFGSStrategy",
    # Factory
    "DEFAULT_STRATEGIES",
    "make_strategy",
]
