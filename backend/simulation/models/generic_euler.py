"""
Generic Euler-Maruyama Simulator
================================

A generic Euler-Maruyama simulator that wraps any Model with drift()/diffusion()
methods. Users only define SDE coefficients, and this simulator handles the rest.

Supports:
- Optional jump(s, dt) method for compound Poisson jump-diffusion models
- Optional stochastic volatility via variance_drift/variance_diffusion/get_correlation

No Numba — uses vectorized NumPy for flexibility with arbitrary Python callables.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from backend.simulation.base import BaseSimulator, SimulationResult


class GenericEulerSimulator(BaseSimulator):
    """
    Generic Euler-Maruyama simulator for any Model with drift()/diffusion().

    Wraps a Model instance and performs vectorized Euler-Maruyama discretization.
    Slower than Numba-optimized simulators but works with arbitrary Python code.

    Optional model methods:
    - ``jump(s, dt)``: compound Poisson jump term added at each timestep
    - ``variance_drift(v, s, t)``, ``variance_diffusion(v, s, t)``,
      ``get_correlation()``: stochastic variance dynamics with correlated BMs

    Parameters
    ----------
    model : Model
        Any model implementing drift(s, v, t, r, q) and diffusion(s, v, t).
    """

    def __init__(self, model: Any) -> None:
        super().__init__()
        self._model: Any = model
        self._model_name: str = model.name
        self._has_jump: bool = hasattr(model, "jump") and callable(
            getattr(model, "jump", None)
        )
        self._has_stoch_vol: bool = (
            hasattr(model, "variance_drift")
            and callable(getattr(model, "variance_drift", None))
            and hasattr(model, "variance_diffusion")
            and callable(getattr(model, "variance_diffusion", None))
        )

    def get_parameters(self) -> dict[str, Any]:
        return self._model.get_parameters()

    def _get_initial_variance(self, params: dict) -> float:
        """Get initial variance from model parameters."""
        return params.get("v0", params.get("sigma", 0.20) ** 2)

    def _get_correlation(self) -> float:
        """Get price-variance correlation (0 if not specified)."""
        if hasattr(self._model, "get_correlation") and callable(
            getattr(self._model, "get_correlation", None)
        ):
            return self._model.get_correlation()
        return 0.0

    def simulate_paths(
        self,
        s0: float,
        mu: float,
        t: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
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
        v0 = self._get_initial_variance(params)

        drift_fn = self._model.drift
        diff_fn = self._model.diffusion
        has_jump = self._has_jump
        has_stoch_vol = self._has_stoch_vol

        # Stochastic vol setup
        vol_paths = None
        if has_stoch_vol:
            rho = self._get_correlation()
            sqrt_1mrho2 = np.sqrt(1.0 - rho**2)
            var_drift_fn = self._model.variance_drift
            var_diff_fn = self._model.variance_diffusion
            v = np.full(n_paths, v0, dtype=np.float64)
            vol_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
            vol_paths[:, 0] = np.sqrt(np.maximum(v0, 0.0))
        else:
            v = v0  # scalar, constant

        for j in range(n_steps):
            s = paths[:, j]
            t_j = j * dt
            z1 = np.random.standard_normal(n_paths)

            # Price SDE
            try:
                drift_val = drift_fn(s, v, t_j, mu, 0.0)
                diff_val = diff_fn(s, v, t_j)
                if np.isscalar(drift_val):
                    drift_val = np.full(n_paths, drift_val)
                if np.isscalar(diff_val):
                    diff_val = np.full(n_paths, diff_val)
            except (TypeError, ValueError):
                drift_vec = np.vectorize(lambda si, vi: drift_fn(si, vi, t_j, mu, 0.0))
                diff_vec = np.vectorize(lambda si, vi: diff_fn(si, vi, t_j))
                drift_val = drift_vec(s, v)
                diff_val = diff_vec(s, v)

            paths[:, j + 1] = s + drift_val * dt + diff_val * sqrt_dt * z1

            # Jump component
            if has_jump:
                paths[:, j + 1] += self._model.jump(s, dt)

            np.maximum(paths[:, j + 1], 1e-10, out=paths[:, j + 1])

            # Variance SDE (correlated with price)
            if has_stoch_vol:
                z_indep = np.random.standard_normal(n_paths)
                z2 = rho * z1 + sqrt_1mrho2 * z_indep

                vd = var_drift_fn(v, s, t_j)
                vdiff = var_diff_fn(v, s, t_j)
                if np.isscalar(vd):
                    vd = np.full(n_paths, vd)
                if np.isscalar(vdiff):
                    vdiff = np.full(n_paths, vdiff)

                v = v + vd * dt + vdiff * sqrt_dt * z2
                np.maximum(v, 1e-10, out=v)
                vol_paths[:, j + 1] = np.sqrt(v)

        computation_time = time.perf_counter() - start_time

        return SimulationResult(
            price_paths=paths,
            time_grid=time_grid,
            model_name=self._model_name,
            computation_time=computation_time,
            n_paths=n_paths,
            n_steps=n_steps,
            volatility_paths=vol_paths,
            parameters=self.get_parameters() | {"s0": s0, "mu": mu, "t": t},
        )

    def simulate_terminal(
        self,
        s0: float,
        mu: float,
        t: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
    ) -> np.ndarray:
        self.validate_inputs(s0, mu, t, n_paths, n_steps)

        if seed is not None:
            np.random.seed(seed)

        dt = t / n_steps
        sqrt_dt = np.sqrt(dt)

        s = np.full(n_paths, s0, dtype=np.float64)

        params = self._model.get_parameters()
        v0 = self._get_initial_variance(params)

        drift_fn = self._model.drift
        diff_fn = self._model.diffusion
        has_jump = self._has_jump
        has_stoch_vol = self._has_stoch_vol

        if has_stoch_vol:
            rho = self._get_correlation()
            sqrt_1mrho2 = np.sqrt(1.0 - rho**2)
            var_drift_fn = self._model.variance_drift
            var_diff_fn = self._model.variance_diffusion
            v = np.full(n_paths, v0, dtype=np.float64)
        else:
            v = v0

        for j in range(n_steps):
            t_j = j * dt
            z1 = np.random.standard_normal(n_paths)

            try:
                drift_val = drift_fn(s, v, t_j, mu, 0.0)
                diff_val = diff_fn(s, v, t_j)
                if np.isscalar(drift_val):
                    drift_val = np.full(n_paths, drift_val)
                if np.isscalar(diff_val):
                    diff_val = np.full(n_paths, diff_val)
            except (TypeError, ValueError):
                drift_vec = np.vectorize(lambda si, vi: drift_fn(si, vi, t_j, mu, 0.0))
                diff_vec = np.vectorize(lambda si, vi: diff_fn(si, vi, t_j))
                drift_val = drift_vec(s, v)
                diff_val = diff_vec(s, v)

            s = s + drift_val * dt + diff_val * sqrt_dt * z1

            if has_jump:
                s += self._model.jump(s, dt)

            np.maximum(s, 1e-10, out=s)

            if has_stoch_vol:
                z_indep = np.random.standard_normal(n_paths)
                z2 = rho * z1 + sqrt_1mrho2 * z_indep

                vd = var_drift_fn(v, s, t_j)
                vdiff = var_diff_fn(v, s, t_j)
                if np.isscalar(vd):
                    vd = np.full(n_paths, vd)
                if np.isscalar(vdiff):
                    vdiff = np.full(n_paths, vdiff)

                v = v + vd * dt + vdiff * sqrt_dt * z2
                np.maximum(v, 1e-10, out=v)

        return s
