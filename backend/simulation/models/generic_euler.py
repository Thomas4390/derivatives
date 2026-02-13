"""
Generic Euler-Maruyama Simulator
================================

A generic Euler-Maruyama simulator that wraps any Model with drift()/diffusion()
methods. Users only define SDE coefficients, and this simulator handles the rest.

Supports an optional jump(s, dt) method for compound Poisson jump-diffusion models.

No Numba — uses vectorized NumPy for flexibility with arbitrary Python callables.

Author: Thomas
Created: 2025
"""

import time
import numpy as np
from typing import Optional, Dict, Any

from backend.simulation.base import BaseSimulator, SimulationResult


class GenericEulerSimulator(BaseSimulator):
    """
    Generic Euler-Maruyama simulator for any Model with drift()/diffusion().

    Wraps a Model instance and performs vectorized Euler-Maruyama discretization.
    Slower than Numba-optimized simulators but works with arbitrary Python code.

    If the model defines a ``jump(s, dt)`` method, a compound Poisson jump
    term is added at each timestep: S_{t+dt} += jump(S_t, dt).

    Parameters
    ----------
    model : Model
        Any model implementing drift(s, v, t, r, q) and diffusion(s, v, t).
        Optionally implements jump(s, dt) for jump-diffusion dynamics.
    """

    def __init__(self, model):
        super().__init__()
        self._model = model
        self._model_name = model.name
        self._has_jump = hasattr(model, 'jump') and callable(getattr(model, 'jump', None))

    def get_parameters(self) -> Dict[str, Any]:
        return self._model.get_parameters()

    def simulate_paths(
        self,
        s0: float,
        mu: float,
        t: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        self.validate_inputs(s0, mu, t, n_paths, n_steps)

        if seed is not None:
            np.random.seed(seed)

        start_time = time.perf_counter()

        dt = t / n_steps
        sqrt_dt = np.sqrt(dt)
        time_grid = np.linspace(0, t, n_steps + 1)

        paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        paths[:, 0] = s0

        params = self._model.get_parameters()
        v = params.get("v0", params.get("sigma", 0.20) ** 2)

        drift_fn = self._model.drift
        diff_fn = self._model.diffusion
        has_jump = self._has_jump

        for j in range(n_steps):
            s = paths[:, j]
            t_j = j * dt
            z = np.random.standard_normal(n_paths)

            # Try vectorized call first; fall back to np.vectorize
            try:
                drift_val = drift_fn(s, v, t_j, mu, 0.0)
                diff_val = diff_fn(s, v, t_j)
                # Ensure array output
                if np.isscalar(drift_val):
                    drift_val = np.full(n_paths, drift_val)
                if np.isscalar(diff_val):
                    diff_val = np.full(n_paths, diff_val)
            except (TypeError, ValueError):
                drift_vec = np.vectorize(lambda si: drift_fn(si, v, t_j, mu, 0.0))
                diff_vec = np.vectorize(lambda si: diff_fn(si, v, t_j))
                drift_val = drift_vec(s)
                diff_val = diff_vec(s)

            paths[:, j + 1] = s + drift_val * dt + diff_val * sqrt_dt * z

            # Add jump component if model supports it
            if has_jump:
                paths[:, j + 1] += self._model.jump(s, dt)

            np.maximum(paths[:, j + 1], 1e-10, out=paths[:, j + 1])

        computation_time = time.perf_counter() - start_time

        return SimulationResult(
            price_paths=paths,
            time_grid=time_grid,
            model_name=self._model_name,
            computation_time=computation_time,
            n_paths=n_paths,
            n_steps=n_steps,
            volatility_paths=None,
            parameters=self.get_parameters() | {"s0": s0, "mu": mu, "t": t},
        )

    def simulate_terminal(
        self,
        s0: float,
        mu: float,
        t: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        self.validate_inputs(s0, mu, t, n_paths, n_steps)

        if seed is not None:
            np.random.seed(seed)

        dt = t / n_steps
        sqrt_dt = np.sqrt(dt)

        s = np.full(n_paths, s0, dtype=np.float64)

        params = self._model.get_parameters()
        v = params.get("v0", params.get("sigma", 0.20) ** 2)

        drift_fn = self._model.drift
        diff_fn = self._model.diffusion
        has_jump = self._has_jump

        for j in range(n_steps):
            t_j = j * dt
            z = np.random.standard_normal(n_paths)

            try:
                drift_val = drift_fn(s, v, t_j, mu, 0.0)
                diff_val = diff_fn(s, v, t_j)
                if np.isscalar(drift_val):
                    drift_val = np.full(n_paths, drift_val)
                if np.isscalar(diff_val):
                    diff_val = np.full(n_paths, diff_val)
            except (TypeError, ValueError):
                drift_vec = np.vectorize(lambda si: drift_fn(si, v, t_j, mu, 0.0))
                diff_vec = np.vectorize(lambda si: diff_fn(si, v, t_j))
                drift_val = drift_vec(s)
                diff_val = diff_vec(s)

            s = s + drift_val * dt + diff_val * sqrt_dt * z

            # Add jump component if model supports it
            if has_jump:
                s += self._model.jump(s, dt)

            np.maximum(s, 1e-10, out=s)

        return s
