"""
Trigger components — autocall / early-redemption logic.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.core.structured_product import EvaluationContext, ProductComponent


# =============================================================================
# Autocall Trigger
# =============================================================================


@dataclass(frozen=True)
class AutocallTrigger(ProductComponent):
    """
    Early redemption trigger.

    If underlying performance >= trigger at an observation date,
    the product terminates and pays back the notional (plus any
    accrued coupons handled by other components).

    Returns the PV of the notional redemption on autocall,
    along with a boolean mask of which paths were called.

    Parameters
    ----------
    trigger_level : float
        Performance level triggering autocall (e.g., 1.0 = 100% of initial).
    notional : float
        Product notional.
    """

    trigger_level: float
    notional: float

    @property
    def name(self) -> str:
        return f"AutocallTrigger({self.trigger_level:.0%})"

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        n_paths = paths.shape[0]
        s0 = paths[:, 0]

        pv = np.zeros(n_paths)
        called = np.zeros(n_paths, dtype=bool)

        for j in range(len(obs_indices)):
            spot_j = paths[:, obs_indices[j]]
            perf = spot_j / s0

            # Trigger autocall for paths not yet called
            trigger_now = (~called) & (perf >= self.trigger_level)
            pv += np.where(trigger_now, self.notional * discount_factors[j], 0.0)
            called |= trigger_now

        return pv

    def evaluate_in_context(self, ctx: EvaluationContext) -> np.ndarray:
        """Evaluate autocall trigger, mutating ctx.alive and ctx.terminated_at_obs."""
        pv = np.zeros(ctx.n_paths)
        for j in range(ctx.n_obs):
            perf = ctx.performance_at(j)
            trigger_now = ctx.alive & (perf >= self.trigger_level)
            pv += np.where(trigger_now, self.notional * ctx.discount_factors[j], 0.0)
            ctx.alive[trigger_now] = False
            ctx.terminated_at_obs[trigger_now] = j
        return pv

    def evaluate_with_call_info(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate with additional autocall information.

        Returns
        -------
        pv : np.ndarray
            PV of notional redemption, shape (n_paths,).
        called : np.ndarray
            Boolean mask of called paths, shape (n_paths,).
        call_time_idx : np.ndarray
            Observation index at which each path was called (-1 if not), shape (n_paths,).
        """
        n_paths = paths.shape[0]
        s0 = paths[:, 0]

        pv = np.zeros(n_paths)
        called = np.zeros(n_paths, dtype=bool)
        call_time_idx = np.full(n_paths, -1, dtype=np.int64)

        for j in range(len(obs_indices)):
            spot_j = paths[:, obs_indices[j]]
            perf = spot_j / s0

            trigger_now = (~called) & (perf >= self.trigger_level)
            pv += np.where(trigger_now, self.notional * discount_factors[j], 0.0)
            call_time_idx = np.where(trigger_now, j, call_time_idx)
            called |= trigger_now

        return pv, called, call_time_idx


__all__ = ["AutocallTrigger"]
