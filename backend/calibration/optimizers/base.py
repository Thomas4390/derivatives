"""
Optimizer Strategy — Base Protocol & IterationLogger
=====================================================

Strategy Pattern for the calibration solvers:

* :class:`OptimizerStrategy` is the structural interface every concrete
  algorithm (LM, DE, Nelder-Mead, L-BFGS-B, ...) must implement.
* :class:`CalibrationProblem` is the unified problem description shared
  across solvers — it carries the residual / Jacobian (least-squares
  view) and / or objective / gradient (scalar view) plus bounds and an
  initial guess.
* :class:`IterationLogger` instruments the user-supplied callables so
  each solver records a per-evaluation snapshot of the optimisation
  state.  Strategies may *additionally* hook into the solver-native
  iteration callback (when one exists) to obtain per-iteration entries
  on top of the per-evaluation log.

The module deliberately does **not** import scipy or JAX so it can be
used as the lightweight contract for the Streamlit layer.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

ResidualFn = Callable[[np.ndarray], np.ndarray]
JacobianFn = Callable[[np.ndarray], np.ndarray]
ObjectiveFn = Callable[[np.ndarray], float]
GradientFn = Callable[[np.ndarray], np.ndarray]
ParamMapper = Callable[[np.ndarray], dict[str, float]]


# --------------------------------------------------------------------------- #
# Snapshots & results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IterationSnapshot:
    """One record of optimiser state captured during a calibration run.

    The ``iteration`` field is a monotonically-increasing counter that
    matches the *evaluation* index — for derivative-free solvers (DE,
    NM) one optimiser iteration spans many evaluations, so this is in
    practice "evaluation #".  Strategies that expose a native iteration
    callback may emit a parallel sequence of snapshots; consumers can
    distinguish the two via ``source``.
    """

    iteration: int
    x: np.ndarray
    params_natural: dict[str, float]
    objective: float
    grad_norm: float | None
    elapsed_seconds: float
    source: str = "evaluation"  # or "callback"


@dataclass(frozen=True)
class OptimizationResult:
    """Outcome of one strategy invocation, independent of the calibrator."""

    x_optimal: np.ndarray
    objective_value: float
    n_iterations: int
    n_function_evals: int
    success: bool
    method: str
    elapsed_seconds: float
    grad_norm: float | None = None
    iteration_history: tuple[IterationSnapshot, ...] = ()
    raw_result: Any = None
    message: str = ""


# --------------------------------------------------------------------------- #
# Unified problem description
# --------------------------------------------------------------------------- #


@dataclass
class CalibrationProblem:
    """Unified optimisation problem shared by all solver strategies.

    Either the *least-squares* interface (``residual_fn``) or the
    *scalar* interface (``objective_fn``) MUST be supplied — the other
    is auto-derived if missing.  When residuals are available, the
    scalar objective defaults to ``0.5 * ||r||^2``; when a Jacobian is
    available, the gradient defaults to ``J^T r``.

    ``param_mapper`` converts the optimiser's working vector ``x`` to
    natural parameter values so iteration snapshots stay
    human-readable.  Identity by default.
    """

    x0: np.ndarray
    bounds_lo: np.ndarray
    bounds_hi: np.ndarray
    param_names: tuple[str, ...]

    residual_fn: ResidualFn | None = None
    jacobian_fn: JacobianFn | None = None
    objective_fn: ObjectiveFn | None = None
    gradient_fn: GradientFn | None = None
    param_mapper: ParamMapper | None = None

    def __post_init__(self) -> None:
        if self.residual_fn is None and self.objective_fn is None:
            raise ValueError(
                "CalibrationProblem requires either residual_fn or objective_fn."
            )
        if self.x0.shape != self.bounds_lo.shape or self.x0.shape != self.bounds_hi.shape:
            raise ValueError("x0 / bounds_lo / bounds_hi must share the same shape.")
        if len(self.param_names) != self.x0.size:
            raise ValueError(
                f"param_names length ({len(self.param_names)}) "
                f"!= x0 size ({self.x0.size})."
            )

    # -- resolved callables ------------------------------------------------- #

    def has_residuals(self) -> bool:
        return self.residual_fn is not None

    def has_objective(self) -> bool:
        return self.objective_fn is not None or self.residual_fn is not None

    def has_jacobian(self) -> bool:
        return self.jacobian_fn is not None

    def has_gradient(self) -> bool:
        return self.gradient_fn is not None or self.jacobian_fn is not None

    def evaluate_residuals(self, x: np.ndarray) -> np.ndarray:
        if self.residual_fn is None:
            raise RuntimeError("Problem has no residual_fn (scalar-only).")
        return np.asarray(self.residual_fn(x))

    def evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        if self.jacobian_fn is None:
            raise RuntimeError("Problem has no jacobian_fn.")
        return np.asarray(self.jacobian_fn(x))

    def evaluate_objective(self, x: np.ndarray) -> float:
        if self.objective_fn is not None:
            return float(self.objective_fn(x))
        r = self.evaluate_residuals(x)
        return 0.5 * float(r @ r)

    def evaluate_gradient(self, x: np.ndarray) -> np.ndarray:
        if self.gradient_fn is not None:
            return np.asarray(self.gradient_fn(x))
        if self.jacobian_fn is None or self.residual_fn is None:
            raise RuntimeError("Problem has no gradient_fn nor jacobian_fn.")
        J = self.evaluate_jacobian(x)
        r = self.evaluate_residuals(x)
        return np.asarray(J.T @ r)

    def map_params(self, x: np.ndarray) -> dict[str, float]:
        if self.param_mapper is not None:
            return self.param_mapper(x)
        return {name: float(v) for name, v in zip(self.param_names, x)}


# --------------------------------------------------------------------------- #
# IterationLogger
# --------------------------------------------------------------------------- #


class IterationLogger:
    """Captures a snapshot per residual / objective evaluation.

    Usage::

        logger = IterationLogger(problem)
        wrapped_residual = logger.wrap_residual(problem.residual_fn)
        wrapped_objective = logger.wrap_objective(problem.objective_fn)
        # ... pass wrapped functions to the solver ...
        for snap in logger.history: ...

    The logger is *strategy-agnostic*: it simply intercepts the user
    callables and records ``IterationSnapshot`` values into ``history``.
    Strategies may also call :meth:`record_callback` from a native
    iteration callback to flag certain evaluations as iteration
    boundaries.
    """

    def __init__(
        self,
        problem: CalibrationProblem,
        *,
        compute_grad_norm: bool = False,
        max_snapshots: int | None = None,
        on_snapshot: Callable[[IterationSnapshot], None] | None = None,
    ) -> None:
        self._problem = problem
        self._compute_grad_norm = compute_grad_norm
        self._max_snapshots = max_snapshots
        self._history: list[IterationSnapshot] = []
        self._t0 = time.perf_counter()
        self._on_snapshot = on_snapshot

    # ----- public API ------------------------------------------------------ #

    @property
    def history(self) -> tuple[IterationSnapshot, ...]:
        return tuple(self._history)

    @property
    def n_evaluations(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        self._history.clear()
        self._t0 = time.perf_counter()

    def wrap_residual(self, fn: ResidualFn) -> ResidualFn:
        """Wrap a residual function to log each evaluation."""

        def wrapped(x: np.ndarray) -> np.ndarray:
            r = np.asarray(fn(x))
            obj = 0.5 * float(r @ r)
            self._record(x, obj, source="evaluation")
            return r

        return wrapped

    def wrap_objective(self, fn: ObjectiveFn) -> ObjectiveFn:
        """Wrap a scalar objective function to log each evaluation."""

        def wrapped(x: np.ndarray) -> float:
            obj = float(fn(x))
            self._record(x, obj, source="evaluation")
            return obj

        return wrapped

    def record_callback(self, x: np.ndarray, objective: float | None = None) -> None:
        """Manually record a snapshot tagged with ``source='callback'``.

        Use this from a solver-native iteration callback (scipy.minimize
        ``callback=`` or differential_evolution ``callback=``) so the
        per-iteration boundary is visible in the history alongside the
        per-evaluation log.
        """
        if objective is None:
            try:
                objective = self._problem.evaluate_objective(x)
            except RuntimeError:
                objective = float("nan")
        self._record(np.asarray(x), float(objective), source="callback")

    # ----- internals ------------------------------------------------------- #

    def _record(self, x: np.ndarray, objective: float, *, source: str) -> None:
        if self._max_snapshots is not None and len(self._history) >= self._max_snapshots:
            return
        grad_norm: float | None = None
        if self._compute_grad_norm and self._problem.has_gradient():
            try:
                g = self._problem.evaluate_gradient(x)
                grad_norm = float(np.linalg.norm(g))
            except RuntimeError:
                grad_norm = None
        snap = IterationSnapshot(
            iteration=len(self._history),
            x=np.asarray(x, dtype=float).copy(),
            params_natural=self._problem.map_params(np.asarray(x, dtype=float)),
            objective=float(objective),
            grad_norm=grad_norm,
            elapsed_seconds=time.perf_counter() - self._t0,
            source=source,
        )
        self._history.append(snap)
        if self._on_snapshot is not None:
            try:
                self._on_snapshot(snap)
            except Exception:
                # Never let UI / consumer errors crash the optimisation
                pass


# --------------------------------------------------------------------------- #
# Strategy protocol
# --------------------------------------------------------------------------- #


@runtime_checkable
class OptimizerStrategy(Protocol):
    """Structural type for any concrete optimisation algorithm."""

    name: str
    is_global: bool
    requires_gradient: bool
    requires_residuals: bool

    def solve(
        self,
        problem: CalibrationProblem,
        logger: IterationLogger | None = None,
        *,
        max_nfev: int = 200,
        tol: float = 1e-10,
        **kwargs: Any,
    ) -> OptimizationResult: ...


# --------------------------------------------------------------------------- #
# Helper: simple base class strategies can inherit (optional)
# --------------------------------------------------------------------------- #


@dataclass
class StrategyMetadata:
    """Static metadata describing a strategy's capabilities."""

    name: str
    is_global: bool = False
    requires_gradient: bool = False
    requires_residuals: bool = False
    description: str = ""


__all__ = [
    "CalibrationProblem",
    "IterationLogger",
    "IterationSnapshot",
    "OptimizationResult",
    "OptimizerStrategy",
    "StrategyMetadata",
    "ResidualFn",
    "JacobianFn",
    "ObjectiveFn",
    "GradientFn",
]
