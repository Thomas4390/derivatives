"""
Heston-Nandi GARCH Simulator (risk-neutral)
===========================================

Monte-Carlo simulator for the Heston & Nandi (2000) discrete-time GARCH model
under the risk-neutral measure Q. Its sole structural difference from the
physical-measure ``GARCHSimulator`` is the locally-risk-neutral log-return

    R_t = r_step - 0.5 h_t + sqrt(h_t) z_t,   z_t ~ N(0,1)
    h_{t+1} = omega + beta h_t + alpha (z_t - gamma sqrt(h_t))^2

so that ``E^Q[S_T] = S_0 e^{r T}`` holds by construction.

It exists primarily as an **independent reference** for validating the
closed-form characteristic-function price (``test_heston_nandi``): a round-trip
calibration alone cannot prove theoretical correctness because it reuses the
same CF on both sides, whereas this simulator never touches the CF.

The model is intrinsically defined on a fixed per-period (daily) grid, so the
calendar-consistent step count is ``N = round(t * steps_per_year)``; the
convenience pricer enforces that, while the generic ``BaseSimulator`` methods
honour the caller-supplied ``n_steps`` with a matched per-step drift.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from backend.simulation.base import (
    BaseSimulator,
    SimulationResult,
    StochasticVolatilityMixin,
)
from backend.utils.constants.calibration import HESTON_NANDI_STEPS_PER_YEAR


class HestonNandiGARCHSimulator(BaseSimulator, StochasticVolatilityMixin):
    """Risk-neutral path/terminal simulator for Heston-Nandi GARCH(1,1)."""

    def __init__(
        self,
        omega: float,
        alpha: float,
        beta: float,
        gamma: float,
        h0: float,
        steps_per_year: int = HESTON_NANDI_STEPS_PER_YEAR,
    ) -> None:
        super().__init__()
        self._model_name = "Heston-Nandi GARCH"
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.h0 = float(h0)
        self.steps_per_year = int(steps_per_year)

    # ------------------------------------------------------------------ #
    # Core vectorized recursion
    # ------------------------------------------------------------------ #

    def _simulate_log_terminal(
        self,
        log_s0: float,
        r_step: float,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator,
        antithetic: bool,
    ) -> np.ndarray:
        """Vectorized risk-neutral GARCH recursion → terminal log-prices."""
        log_s = np.full(n_paths, log_s0, dtype=float)
        h = np.full(n_paths, self.h0, dtype=float)
        half = n_paths // 2
        for _ in range(n_steps):
            if antithetic and n_paths % 2 == 0:
                z_half = rng.standard_normal(half)
                z = np.concatenate([z_half, -z_half])
            else:
                z = rng.standard_normal(n_paths)
            sqrt_h = np.sqrt(np.maximum(h, 0.0))
            log_s += r_step - 0.5 * h + sqrt_h * z
            h = self.omega + self.beta * h + self.alpha * (z - self.gamma * sqrt_h) ** 2
        return log_s

    # ------------------------------------------------------------------ #
    # Convenience risk-neutral European pricer (used by the MC cross-check)
    # ------------------------------------------------------------------ #

    def price_european_call(
        self,
        s0: float,
        strike: float,
        r: float,
        t: float,
        n_paths: int = 200_000,
        seed: int | None = None,
        antithetic: bool = True,
    ) -> float:
        """Discounted risk-neutral MC price of a European call.

        Uses the calendar-consistent ``N = round(t * steps_per_year)`` steps and
        per-step rate ``r / steps_per_year`` so the result is directly
        comparable to the closed-form characteristic-function price.
        """
        n_steps = max(int(round(t * self.steps_per_year)), 1)
        r_step = r / self.steps_per_year
        rng = np.random.default_rng(seed)
        log_terminal = self._simulate_log_terminal(
            np.log(s0), r_step, n_paths, n_steps, rng, antithetic
        )
        payoff = np.maximum(np.exp(log_terminal) - strike, 0.0)
        return float(np.exp(-r * t) * np.mean(payoff))

    # ------------------------------------------------------------------ #
    # BaseSimulator contract
    # ------------------------------------------------------------------ #

    def simulate_terminal(
        self,
        s0: float,
        mu: float,
        t: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """Terminal prices S(T). ``mu`` is the annual drift (pass ``r`` for Q).

        For calendar consistency with the model's per-period dynamics, pass
        ``n_steps = round(t * steps_per_year)``; the per-step drift is set to
        ``mu * t / n_steps`` so total drift is ``mu * t`` for any ``n_steps``.
        """
        self.validate_inputs(s0, mu, t, n_paths, n_steps)
        r_step = mu * t / n_steps
        rng = np.random.default_rng(seed)
        log_terminal = self._simulate_log_terminal(
            np.log(s0), r_step, n_paths, n_steps, rng, antithetic=False
        )
        return np.exp(log_terminal)

    def simulate_paths(
        self,
        s0: float,
        mu: float,
        t: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
    ) -> SimulationResult:
        """Full risk-neutral price + volatility paths."""
        self.validate_inputs(s0, mu, t, n_paths, n_steps)
        r_step = mu * t / n_steps
        rng = np.random.default_rng(seed)

        start = time.perf_counter()
        log_s = np.empty((n_paths, n_steps + 1), dtype=float)
        vol = np.empty((n_paths, n_steps + 1), dtype=float)
        log_s[:, 0] = np.log(s0)
        h = np.full(n_paths, self.h0, dtype=float)
        vol[:, 0] = np.sqrt(np.maximum(h, 0.0))
        for step in range(n_steps):
            z = rng.standard_normal(n_paths)
            sqrt_h = np.sqrt(np.maximum(h, 0.0))
            log_s[:, step + 1] = log_s[:, step] + r_step - 0.5 * h + sqrt_h * z
            h = self.omega + self.beta * h + self.alpha * (z - self.gamma * sqrt_h) ** 2
            vol[:, step + 1] = np.sqrt(np.maximum(h, 0.0))
        elapsed = time.perf_counter() - start

        return SimulationResult(
            price_paths=np.exp(log_s),
            time_grid=np.linspace(0.0, t, n_steps + 1),
            model_name=self._model_name,
            computation_time=elapsed,
            n_paths=n_paths,
            n_steps=n_steps,
            volatility_paths=vol,
            parameters=self.get_parameters(),
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "h0": self.h0,
            "steps_per_year": self.steps_per_year,
        }

    # ------------------------------------------------------------------ #
    # StochasticVolatilityMixin
    # ------------------------------------------------------------------ #

    def long_run_variance(self) -> float:
        """Per-period unconditional variance (``inf`` if non-stationary)."""
        gap = 1.0 - (self.beta + self.alpha * self.gamma**2)
        if gap <= 0.0:
            return float("inf")
        return (self.omega + self.alpha) / gap
