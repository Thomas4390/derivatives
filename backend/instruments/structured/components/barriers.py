"""
Barrier components — knock-in puts, digitals, two-step bonuses.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.core.structured_product import EvaluationContext, ProductComponent
from backend.engines.structured_kernels import check_barrier_breach


# =============================================================================
# Knock-In Put
# =============================================================================


@dataclass(frozen=True)
class KnockInPut(ProductComponent):
    """
    Knock-in put component — capital at risk.

    At maturity, if the barrier has been breached:
        pays notional * min(S_T / S_0, 1.0) [i.e., loss if S_T < S_0]
    Otherwise:
        pays notional (full redemption).

    Parameters
    ----------
    barrier : float
        Knock-in barrier as fraction of initial (e.g., 0.6 = 60%).
    notional : float
        Product notional.
    monitoring : str
        'continuous' (min of path) or 'discrete' (only at observation dates).
    """

    barrier: float
    notional: float
    monitoring: str = "continuous"

    def __post_init__(self) -> None:
        if not 0.0 < self.barrier < 1.0:
            raise ValueError(f"barrier must be in (0, 1), got {self.barrier}")
        if self.monitoring not in ("continuous", "discrete"):
            raise ValueError(
                f"monitoring must be 'continuous' or 'discrete', got '{self.monitoring}'"
            )

    @property
    def name(self) -> str:
        return f"KnockInPut(barrier={self.barrier:.0%}, {self.monitoring})"

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        breached = check_barrier_breach(
            paths=paths,
            barrier_fraction=self.barrier,
            obs_indices=obs_indices,
            continuous=(self.monitoring == "continuous"),
            breach_below=True,
        )

        s0 = paths[:, 0]
        s_t = paths[:, -1]
        df_terminal = discount_factors[-1]

        terminal_pv = np.where(
            breached,
            self.notional * np.minimum(s_t / s0, 1.0) * df_terminal,
            self.notional * df_terminal,
        )

        return terminal_pv

    def evaluate_in_context(self, ctx: EvaluationContext) -> np.ndarray:
        """Evaluate knock-in put respecting ctx.alive (only non-autocalled paths get terminal payoff)."""
        s0 = ctx.s0

        # Check barrier breach
        if self.monitoring == "continuous":
            path_min = np.min(ctx.paths, axis=1)
            breached = path_min <= (self.barrier * s0)
        else:
            obs_prices = ctx.paths[:, ctx.obs_indices]
            obs_min = np.min(obs_prices, axis=1)
            breached = obs_min <= (self.barrier * s0)

        ctx.ki_breached = breached

        s_t = ctx.paths[:, -1]
        terminal_perf = s_t / s0

        # Only alive paths (not autocalled) receive the terminal payoff
        terminal_pv = np.where(
            ctx.alive & breached,
            self.notional * np.minimum(terminal_perf, 1.0) * ctx.df_terminal,
            np.where(ctx.alive, self.notional * ctx.df_terminal, 0.0),
        )

        return terminal_pv


# =============================================================================
# Geared Knock-In Put
# =============================================================================


@dataclass(frozen=True)
class GearedKnockInPut(ProductComponent):
    """
    Knock-in put with geared (amplified) losses below the barrier.

    At maturity, if the barrier has been breached:
        pays notional * max(1 - gearing * (1 - S_T/S_0), 0) * df
    Otherwise:
        pays notional * df (full redemption).

    The gearing amplifies losses: with gearing=1.33 (i.e., S0/B for B=75%),
    a 30% drop results in a 40% loss instead of 30%.

    Parameters
    ----------
    barrier : float
        Knock-in barrier as fraction of initial (e.g., 0.75 = 75%).
    notional : float
        Product notional.
    gearing : float or None
        Loss amplification factor. None auto-computes as 1/barrier.
    monitoring : str
        'continuous' or 'discrete'.
    """

    barrier: float
    notional: float
    gearing: float | None = None
    monitoring: str = "continuous"

    def __post_init__(self) -> None:
        if not 0.0 < self.barrier < 1.0:
            raise ValueError(f"barrier must be in (0, 1), got {self.barrier}")
        if self.monitoring not in ("continuous", "discrete"):
            raise ValueError(
                f"monitoring must be 'continuous' or 'discrete', got '{self.monitoring}'"
            )

    @property
    def effective_gearing(self) -> float:
        return self.gearing if self.gearing is not None else 1.0 / self.barrier

    @property
    def name(self) -> str:
        return f"GearedKnockInPut(barrier={self.barrier:.0%}, gearing={self.effective_gearing:.2f})"

    def _check_breach(
        self, paths: np.ndarray, s0: np.ndarray, obs_indices: np.ndarray
    ) -> np.ndarray:
        return check_barrier_breach(
            paths=paths,
            barrier_fraction=self.barrier,
            obs_indices=obs_indices,
            continuous=(self.monitoring == "continuous"),
            breach_below=True,
        )

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        s0 = paths[:, 0]
        s_t = paths[:, -1]
        df_terminal = discount_factors[-1]
        breached = self._check_breach(paths, s0, obs_indices)
        gearing = self.effective_gearing
        loss_fraction = gearing * (1.0 - s_t / s0)
        geared_pv = self.notional * np.maximum(1.0 - loss_fraction, 0.0) * df_terminal
        full_pv = self.notional * df_terminal
        return np.where(breached, geared_pv, full_pv)

    def evaluate_in_context(self, ctx: EvaluationContext) -> np.ndarray:
        s0 = ctx.s0
        s_t = ctx.paths[:, -1]
        breached = self._check_breach(ctx.paths, s0, ctx.obs_indices)
        ctx.ki_breached = breached
        gearing = self.effective_gearing
        loss_fraction = gearing * (1.0 - s_t / s0)
        geared_pv = (
            self.notional * np.maximum(1.0 - loss_fraction, 0.0) * ctx.df_terminal
        )
        full_pv = self.notional * ctx.df_terminal
        return np.where(
            ctx.alive & breached,
            geared_pv,
            np.where(ctx.alive, full_pv, 0.0),
        )


# =============================================================================
# Maturity Barrier Knock-In Put
# =============================================================================


@dataclass(frozen=True)
class MaturityBarrierKnockInPut(ProductComponent):
    """
    Maturity-monitored barrier with payoff discontinuity.

    Checked ONLY at expiry (not daily). If S_T/S_0 < barrier:
        pays notional * (S_T / S_0) * df  (full depreciation to spot)
    Otherwise:
        pays notional * df  (protected)

    This creates a discontinuity (cliff) at the barrier — the holder
    jumps from receiving S_T to receiving notional as S_T crosses
    barrier * S_0 from below.

    Equivalent to: S0*Bond - Put(S0, B) - (S0-B)*DigitalPut(B)

    Parameters
    ----------
    barrier : float
        Barrier as fraction of initial (e.g., 0.70 = 70%).
    notional : float
        Product notional.
    """

    barrier: float
    notional: float

    def __post_init__(self) -> None:
        if not 0.0 < self.barrier < 1.0:
            raise ValueError(f"barrier must be in (0, 1), got {self.barrier}")

    @property
    def name(self) -> str:
        return f"MaturityBarrierKnockInPut(barrier={self.barrier:.0%})"

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        s0 = paths[:, 0]
        s_t = paths[:, -1]
        df_terminal = discount_factors[-1]
        perf = s_t / s0
        below_barrier = perf < self.barrier
        return np.where(
            below_barrier,
            self.notional * perf * df_terminal,
            self.notional * df_terminal,
        )

    def evaluate_in_context(self, ctx: EvaluationContext) -> np.ndarray:
        perf = ctx.terminal_performance()
        below_barrier = perf < self.barrier
        ctx.ki_breached = below_barrier
        return np.where(
            ctx.alive & below_barrier,
            self.notional * perf * ctx.df_terminal,
            np.where(ctx.alive, self.notional * ctx.df_terminal, 0.0),
        )


# =============================================================================
# Bonus Digital
# =============================================================================


@dataclass(frozen=True)
class BonusDigital(ProductComponent):
    """
    Digital bonus paid if terminal performance exceeds a threshold.

    At maturity: pays bonus_return * notional if S_T/S_0 >= threshold.

    Used in BO (Bonus) and ICBO (Issuer Callable Bonus) families.

    Parameters
    ----------
    bonus_return : float
        Bonus amount as fraction of notional (e.g., 0.45 = 45%).
    threshold : float
        Performance threshold (e.g., 0.90 = 90% of initial).
    notional : float
        Product notional.
    """

    bonus_return: float
    threshold: float
    notional: float

    @property
    def name(self) -> str:
        return f"BonusDigital(return={self.bonus_return:.0%}, threshold={self.threshold:.0%})"

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        s0 = paths[:, 0]
        s_t = paths[:, -1]
        df_terminal = discount_factors[-1]
        perf = s_t / s0
        triggered = perf >= self.threshold
        return np.where(triggered, self.bonus_return * self.notional * df_terminal, 0.0)

    def evaluate_in_context(self, ctx: EvaluationContext) -> np.ndarray:
        perf = ctx.terminal_performance()
        triggered = perf >= self.threshold
        return np.where(
            ctx.alive & triggered,
            self.bonus_return * self.notional * ctx.df_terminal,
            0.0,
        )


# =============================================================================
# Two-Step Bonus
# =============================================================================


@dataclass(frozen=True)
class TwoStepBonus(ProductComponent):
    """
    Two-tier bonus based on terminal performance.

    At maturity:
    - If S_T/S_0 >= threshold_high: pays bonus_high * notional
    - Elif S_T/S_0 >= threshold_low: pays bonus_low * notional
    - Else: 0 (protection handled by separate component)

    Used in BTS (Bonus Two-Step) family.

    Parameters
    ----------
    bonus_high : float
        Bonus rate for high performance tier.
    threshold_high : float
        High performance threshold (e.g., 1.0 = 100% of initial).
    bonus_low : float
        Bonus rate for low performance tier.
    threshold_low : float
        Low performance threshold (e.g., 0.90 = 90% of initial).
    notional : float
        Product notional.
    """

    bonus_high: float
    threshold_high: float
    bonus_low: float
    threshold_low: float
    notional: float

    def __post_init__(self) -> None:
        if self.threshold_high <= self.threshold_low:
            raise ValueError(
                f"threshold_high ({self.threshold_high}) must exceed threshold_low ({self.threshold_low})"
            )

    @property
    def name(self) -> str:
        return f"TwoStepBonus(high={self.bonus_high:.0%}@{self.threshold_high:.0%}, low={self.bonus_low:.0%}@{self.threshold_low:.0%})"

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        s0 = paths[:, 0]
        s_t = paths[:, -1]
        df_terminal = discount_factors[-1]
        perf = s_t / s0
        high_tier = perf >= self.threshold_high
        low_tier = (~high_tier) & (perf >= self.threshold_low)
        bonus = np.where(
            high_tier, self.bonus_high, np.where(low_tier, self.bonus_low, 0.0)
        )
        return bonus * self.notional * df_terminal

    def evaluate_in_context(self, ctx: EvaluationContext) -> np.ndarray:
        perf = ctx.terminal_performance()
        high_tier = perf >= self.threshold_high
        low_tier = (~high_tier) & (perf >= self.threshold_low)
        bonus = np.where(
            high_tier, self.bonus_high, np.where(low_tier, self.bonus_low, 0.0)
        )
        return np.where(ctx.alive, bonus * self.notional * ctx.df_terminal, 0.0)


__all__ = [
    "BonusDigital",
    "GearedKnockInPut",
    "KnockInPut",
    "MaturityBarrierKnockInPut",
    "TwoStepBonus",
]
