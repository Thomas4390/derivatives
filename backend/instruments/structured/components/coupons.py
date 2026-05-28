"""
Coupon components — fixed, conditional, CMI, variable, snowball, range accrual.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.core.structured_product import EvaluationContext, ProductComponent
from backend.engines.structured_kernels import (
    evaluate_cmi_coupon_paths,
    evaluate_conditional_coupon_paths,
    evaluate_range_accrual_paths,
)


# =============================================================================
# Fixed Coupon
# =============================================================================


@dataclass(frozen=True)
class FixedCoupon(ProductComponent):
    """
    Unconditional fixed coupon paid at each observation date.

    Parameters
    ----------
    coupon_rate : float
        Annualized coupon rate (e.g., 0.08 = 8%).
    notional : float
        Product notional.
    """

    coupon_rate: float
    notional: float

    @property
    def name(self) -> str:
        return f"FixedCoupon({self.coupon_rate:.2%})"

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        n_paths = paths.shape[0]
        obs_times = time_grid[obs_indices]

        # Period lengths between observations
        prev_times = np.concatenate([[0.0], obs_times[:-1]])
        dt = obs_times - prev_times

        # PV of coupons = sum of coupon_rate * dt * notional * df
        coupon_pv = np.sum(self.coupon_rate * dt * self.notional * discount_factors)
        return np.full(n_paths, coupon_pv)


# =============================================================================
# Conditional Coupon
# =============================================================================


@dataclass(frozen=True)
class ConditionalCoupon(ProductComponent):
    """
    Coupon paid only if underlying performance >= barrier at observation.

    Parameters
    ----------
    coupon_rate : float
        Annualized coupon rate.
    barrier : float
        Performance barrier (e.g., 0.7 = 70% of initial).
    notional : float
        Product notional.
    memory : bool
        If True, unpaid coupons accumulate and are paid when barrier is met.
    """

    coupon_rate: float
    barrier: float
    notional: float
    memory: bool = False

    @property
    def name(self) -> str:
        mem_str = ", memory" if self.memory else ""
        return f"ConditionalCoupon({self.coupon_rate:.2%}, barrier={self.barrier:.0%}{mem_str})"

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        obs_times = time_grid[obs_indices]
        prev_times = np.concatenate([[0.0], obs_times[:-1]])
        obs_dt = obs_times - prev_times

        return evaluate_conditional_coupon_paths(
            paths=paths,
            obs_indices=obs_indices,
            discount_factors=discount_factors,
            obs_dt=obs_dt,
            notional=self.notional,
            coupon_rate=self.coupon_rate,
            barrier=self.barrier,
            memory=self.memory,
        )

    def evaluate_in_context(self, ctx: EvaluationContext) -> np.ndarray:
        """Evaluate conditional coupon respecting ctx.alive (autocall kills future coupons)."""
        pv = np.zeros(ctx.n_paths)

        if self.memory:
            unpaid_coupons = np.zeros(ctx.n_paths)
            for j in range(ctx.n_obs):
                perf = ctx.performance_at(j)
                coupon_j = self.coupon_rate * ctx.obs_dt[j] * self.notional
                # Accrue for alive paths + paths terminated at this obs (coupon before autocall)
                eligible = ctx.alive | (ctx.terminated_at_obs == j)
                unpaid_coupons += np.where(eligible, coupon_j, 0.0)
                above_barrier = perf >= self.barrier
                pay_mask = eligible & above_barrier
                pv += np.where(pay_mask, unpaid_coupons * ctx.discount_factors[j], 0.0)
                unpaid_coupons = np.where(pay_mask, 0.0, unpaid_coupons)
        else:
            for j in range(ctx.n_obs):
                perf = ctx.performance_at(j)
                coupon_j = self.coupon_rate * ctx.obs_dt[j] * self.notional
                eligible = ctx.alive | (ctx.terminated_at_obs == j)
                above_barrier = perf >= self.barrier
                pv += np.where(
                    eligible & above_barrier,
                    coupon_j * ctx.discount_factors[j],
                    0.0,
                )

        return pv


# =============================================================================
# CMI Conditional Coupon (CPPF/DPPF)
# =============================================================================


@dataclass(frozen=True)
class CMIConditionalCoupon(ProductComponent):
    """
    Coupon à mémoire contingente with variable return (CPPF/DPPF).

    NBC ACCMI family: conditional coupon with memory AND variable return
    that depends on whether this is a "current" or "deferred" payment.

    At each observation:
    - If S_t/S_0 >= barrier AND no prior missed coupons:
        pays fixed_coupon + cppf * max(S_t/S_0 - 1, 0) * notional
    - If S_t/S_0 >= barrier AND prior missed coupons exist:
        pays fixed_coupon + accumulated_missed + dppf * max(S_t/S_0 - barrier, 0) * notional
    - If S_t/S_0 < barrier:
        accumulate (memory)

    Parameters
    ----------
    coupon_rate : float
        Annualized fixed coupon rate.
    barrier : float
        Performance barrier (e.g., 0.70 = 70%).
    notional : float
        Product notional.
    cppf : float
        Current Participation Performance Factor (applied when no misses).
    dppf : float
        Deferred Participation Performance Factor (applied with misses).
    """

    coupon_rate: float
    barrier: float
    notional: float
    cppf: float = 0.0
    dppf: float = 0.0

    @property
    def name(self) -> str:
        return f"CMIConditionalCoupon({self.coupon_rate:.2%}, barrier={self.barrier:.0%}, cppf={self.cppf}, dppf={self.dppf})"

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        obs_times = time_grid[obs_indices]
        prev_times = np.concatenate([[0.0], obs_times[:-1]])
        obs_dt = obs_times - prev_times

        return evaluate_cmi_coupon_paths(
            paths=paths,
            obs_indices=obs_indices,
            discount_factors=discount_factors,
            obs_dt=obs_dt,
            notional=self.notional,
            coupon_rate=self.coupon_rate,
            barrier=self.barrier,
            cppf=self.cppf,
            dppf=self.dppf,
        )

    def evaluate_in_context(self, ctx: EvaluationContext) -> np.ndarray:
        pv = np.zeros(ctx.n_paths)
        unpaid_coupons = np.zeros(ctx.n_paths)

        for j in range(ctx.n_obs):
            perf = ctx.performance_at(j)
            coupon_j = self.coupon_rate * ctx.obs_dt[j] * self.notional
            eligible = ctx.alive | (ctx.terminated_at_obs == j)
            above_barrier = perf >= self.barrier

            has_misses = ctx.cmi_unpaid_count > 0
            current_var = self.cppf * np.maximum(perf - 1.0, 0.0) * self.notional
            deferred_var = (
                self.dppf * np.maximum(perf - self.barrier, 0.0) * self.notional
            )
            variable_return = np.where(has_misses, deferred_var, current_var)

            pay_mask = eligible & above_barrier
            pv += np.where(
                pay_mask,
                (coupon_j + unpaid_coupons + variable_return) * ctx.discount_factors[j],
                0.0,
            )

            unpaid_coupons = np.where(
                pay_mask,
                0.0,
                np.where(eligible, unpaid_coupons + coupon_j, unpaid_coupons),
            )
            ctx.cmi_unpaid_count = np.where(
                pay_mask,
                0,
                np.where(eligible, ctx.cmi_unpaid_count + 1, ctx.cmi_unpaid_count),
            )

        return pv


# =============================================================================
# Variable Income Coupon
# =============================================================================


@dataclass(frozen=True)
class VariableIncomeCoupon(ProductComponent):
    """
    Variable income coupon paid based on underlying performance.

    At each observation: pays coupon if performance >= barrier.
    No memory — missed coupons are lost.

    Used in VI (Variable Income) and VIAC (Variable Income Autocallable) families.

    Parameters
    ----------
    coupon_rate : float
        Annualized coupon rate.
    barrier : float
        Performance barrier for coupon trigger.
    notional : float
        Product notional.
    """

    coupon_rate: float
    barrier: float
    notional: float

    @property
    def name(self) -> str:
        return (
            f"VariableIncomeCoupon({self.coupon_rate:.2%}, barrier={self.barrier:.0%})"
        )

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        obs_times = time_grid[obs_indices]
        prev_times = np.concatenate([[0.0], obs_times[:-1]])
        obs_dt = obs_times - prev_times

        return evaluate_conditional_coupon_paths(
            paths=paths,
            obs_indices=obs_indices,
            discount_factors=discount_factors,
            obs_dt=obs_dt,
            notional=self.notional,
            coupon_rate=self.coupon_rate,
            barrier=self.barrier,
            memory=False,
        )

    def evaluate_in_context(self, ctx: EvaluationContext) -> np.ndarray:
        pv = np.zeros(ctx.n_paths)
        for j in range(ctx.n_obs):
            perf = ctx.performance_at(j)
            coupon_j = self.coupon_rate * ctx.obs_dt[j] * self.notional
            eligible = ctx.alive | (ctx.terminated_at_obs == j)
            above_barrier = perf >= self.barrier
            pv += np.where(
                eligible & above_barrier, coupon_j * ctx.discount_factors[j], 0.0
            )
        return pv


# =============================================================================
# Snowball Coupon
# =============================================================================


@dataclass(frozen=True)
class SnowballCoupon(ProductComponent):
    """
    Snowball coupon: coupon that grows with time (coupon(j) = rate * t_j * notional).

    Respects ctx.alive for autocall products.

    Parameters
    ----------
    coupon_rate : float
        Annualized coupon rate.
    notional : float
        Product notional.
    """

    coupon_rate: float
    notional: float

    @property
    def name(self) -> str:
        return f"SnowballCoupon({self.coupon_rate:.2%})"

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        n_paths = paths.shape[0]
        obs_times = time_grid[obs_indices]
        pv = np.zeros(n_paths)
        for j in range(len(obs_indices)):
            coupon_j = self.coupon_rate * obs_times[j] * self.notional
            pv += coupon_j * discount_factors[j]
        return np.full(
            n_paths,
            np.sum(self.coupon_rate * obs_times * self.notional * discount_factors),
        )

    def evaluate_in_context(self, ctx: EvaluationContext) -> np.ndarray:
        """Snowball coupon respecting alive mask."""
        obs_times = ctx.time_grid[ctx.obs_indices]
        pv = np.zeros(ctx.n_paths)
        for j in range(ctx.n_obs):
            # Snowball: coupon grows with time elapsed
            coupon_j = self.coupon_rate * obs_times[j] * self.notional
            # Pay to paths alive at obs j (still alive OR terminated at j or later)
            eligible = ctx.alive | (ctx.terminated_at_obs >= j)
            pv += np.where(eligible, coupon_j * ctx.discount_factors[j], 0.0)
        return pv


# =============================================================================
# Range Accrual Coupon
# =============================================================================


@dataclass(frozen=True)
class RangeAccrualCoupon(ProductComponent):
    """
    Range accrual coupon component.

    Coupon at each period is proportional to the fraction of days the
    underlying spent within the range [lower_barrier, upper_barrier].

    Parameters
    ----------
    coupon_rate : float
        Maximum annualized coupon rate (paid in full if always in range).
    lower_barrier : float
        Lower range barrier as fraction of initial (e.g., 0.80 = 80%).
    upper_barrier : float
        Upper range barrier as fraction of initial (e.g., 1.20 = 120%).
    notional : float
        Product notional.
    """

    coupon_rate: float
    lower_barrier: float
    upper_barrier: float
    notional: float

    def __post_init__(self) -> None:
        if self.lower_barrier >= self.upper_barrier:
            raise ValueError(
                f"lower_barrier ({self.lower_barrier}) must be < upper_barrier ({self.upper_barrier})"
            )

    @property
    def name(self) -> str:
        return f"RangeAccrualCoupon({self.coupon_rate:.2%}, [{self.lower_barrier:.0%}-{self.upper_barrier:.0%}])"

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        obs_times = time_grid[obs_indices]
        prev_times = np.concatenate([[0.0], obs_times[:-1]])
        obs_dt = obs_times - prev_times

        return evaluate_range_accrual_paths(
            paths=paths,
            obs_indices=obs_indices,
            discount_factors=discount_factors,
            obs_dt=obs_dt,
            notional=self.notional,
            coupon_rate=self.coupon_rate,
            lower_barrier=self.lower_barrier,
            upper_barrier=self.upper_barrier,
        )


__all__ = [
    "CMIConditionalCoupon",
    "ConditionalCoupon",
    "FixedCoupon",
    "RangeAccrualCoupon",
    "SnowballCoupon",
    "VariableIncomeCoupon",
]
