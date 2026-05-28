"""
Participation components — upside, knock-out, twin-win, cliquet, asian, lookback.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.core.structured_product import EvaluationContext, ProductComponent
from backend.engines.structured_kernels import (
    evaluate_asian_participation_paths,
    evaluate_cliquet_paths,
    evaluate_knock_out_participation_paths,
    evaluate_lookback_participation_paths,
    evaluate_twin_win_paths,
)


# =============================================================================
# Upside Participation
# =============================================================================


@dataclass(frozen=True)
class UpsideParticipation(ProductComponent):
    """
    Participation in the upside of the underlying.

    Payoff at maturity:
        notional * participation * max(performance - 1, 0)
    Capped at:
        notional * participation * (cap - 1) if cap is set.

    Parameters
    ----------
    participation : float
        Participation rate (e.g., 1.0 = 100%).
    notional : float
        Product notional.
    cap : float or None
        Maximum performance level (e.g., 1.5 = 150% of initial).
        None means unlimited upside.
    """

    participation: float
    notional: float
    cap: float | None = None

    def __post_init__(self) -> None:
        if self.participation <= 0:
            raise ValueError(
                f"participation must be positive, got {self.participation}"
            )
        if self.cap is not None and self.cap <= 1.0:
            raise ValueError(f"cap must be > 1.0 (above initial), got {self.cap}")

    @property
    def name(self) -> str:
        cap_str = f", cap={self.cap:.0%}" if self.cap else ""
        return f"UpsideParticipation({self.participation:.0%}{cap_str})"

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        s0 = paths[:, 0]
        s_t = paths[:, -1]
        performance = s_t / s0

        upside = np.maximum(performance - 1.0, 0.0)
        if self.cap is not None:
            upside = np.minimum(upside, self.cap - 1.0)

        df_terminal = discount_factors[-1]
        return self.notional * self.participation * upside * df_terminal


# =============================================================================
# Knock-Out Participation
# =============================================================================


@dataclass(frozen=True)
class KnockOutParticipation(ProductComponent):
    """
    Participation in upside that knocks out if underlying touches upper barrier.

    Used in Shark Notes:
    - If knock-out: pays a fixed rebate
    - If no knock-out: pays capped upside participation

    Parameters
    ----------
    participation : float
        Participation rate (e.g., 1.5 = 150%).
    barrier : float
        Upper knock-out barrier as fraction of initial (e.g., 1.40 = 140%).
    rebate : float
        Fixed rebate paid if knocked out (as fraction of notional, e.g., 0.05 = 5%).
    notional : float
        Product notional.
    cap : float | None
        Maximum performance level. None = barrier acts as cap.
    """

    participation: float
    barrier: float
    rebate: float
    notional: float
    cap: float | None = None

    @property
    def name(self) -> str:
        return f"KnockOutParticipation(barrier={self.barrier:.0%}, rebate={self.rebate:.1%})"

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        effective_cap = self.cap if self.cap is not None else self.barrier
        return evaluate_knock_out_participation_paths(
            paths=paths,
            notional=self.notional,
            participation=self.participation,
            barrier=self.barrier,
            rebate=self.rebate,
            cap=effective_cap,
            has_cap=True,
            df_terminal=discount_factors[-1],
        )


# =============================================================================
# Twin-Win Participation
# =============================================================================


@dataclass(frozen=True)
class TwinWinParticipation(ProductComponent):
    """
    Symmetric participation: profit from both upside and downside moves.

    Payoff at maturity:
    - If NOT knock-in breached: notional * |performance - 1| * participation + notional
    - If knock-in breached: notional * min(performance, 1.0) (capital loss)

    Parameters
    ----------
    participation : float
        Participation rate.
    ki_barrier : float
        Knock-in barrier (e.g., 0.60). If breached, twin-win becomes capital loss.
    notional : float
        Product notional.
    monitoring : str
        'continuous' or 'discrete'.
    """

    participation: float
    ki_barrier: float
    notional: float
    monitoring: str = "continuous"

    @property
    def name(self) -> str:
        return f"TwinWinParticipation(barrier={self.ki_barrier:.0%})"

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        return evaluate_twin_win_paths(
            paths=paths,
            obs_indices=obs_indices,
            notional=self.notional,
            participation=self.participation,
            ki_barrier=self.ki_barrier,
            continuous=(self.monitoring == "continuous"),
            df_terminal=discount_factors[-1],
        )

    def evaluate_in_context(self, ctx: EvaluationContext) -> np.ndarray:
        """Evaluate using ctx.ki_breached if already set (e.g., by a prior KnockInPut)."""
        s0 = ctx.s0
        performance = ctx.terminal_performance()
        df_terminal = ctx.df_terminal

        # Check knock-in breach
        if self.monitoring == "continuous":
            path_min = np.min(ctx.paths, axis=1)
            breached = path_min <= (self.ki_barrier * s0)
        else:
            obs_prices = ctx.paths[:, ctx.obs_indices]
            obs_min = np.min(obs_prices, axis=1)
            breached = obs_min <= (self.ki_barrier * s0)

        ctx.ki_breached = breached

        abs_move = np.abs(performance - 1.0)
        twin_win_pv = (
            (1.0 + self.participation * abs_move) * self.notional * df_terminal
        )
        # Breached twin-win becomes a linear tracker (no upside cap)
        tracker_pv = performance * self.notional * df_terminal

        return np.where(breached, tracker_pv, twin_win_pv)


# =============================================================================
# Cliquet Participation
# =============================================================================


@dataclass(frozen=True)
class CliquetParticipation(ProductComponent):
    """
    Cliquet (ratchet) participation component.

    Accumulates locally capped/floored periodic returns across observation
    dates, then applies a global cap/floor to the total accumulated return.

    Parameters
    ----------
    notional : float
        Product notional.
    local_cap : float
        Maximum return per period (e.g., 0.05 = 5%).
    local_floor : float
        Minimum return per period (e.g., 0.0 = no negative contributions).
    global_cap : float
        Maximum total accumulated return (e.g., 0.50 = 50%).
    global_floor : float
        Minimum total accumulated return (e.g., 0.0 = principal protected).
    """

    notional: float
    local_cap: float = 0.05
    local_floor: float = 0.0
    global_cap: float = 1.0
    global_floor: float = 0.0

    @property
    def name(self) -> str:
        return f"CliquetParticipation(cap={self.local_cap:.1%}, floor={self.local_floor:.1%})"

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        return evaluate_cliquet_paths(
            paths=paths,
            obs_indices=obs_indices,
            discount_factors=discount_factors,
            notional=self.notional,
            local_cap=self.local_cap,
            local_floor=self.local_floor,
            global_cap=self.global_cap,
            global_floor=self.global_floor,
        )


# =============================================================================
# Average (Asian) Participation
# =============================================================================


@dataclass(frozen=True)
class AverageParticipation(ProductComponent):
    """
    Asian (average) participation component.

    Payoff based on the arithmetic average of the underlying's performance
    at observation dates.

    Two modes:
    - Average rate (default): participation * max(avg_perf - 1, 0)
    - Average strike: participation * max(terminal_perf - avg_perf, 0)

    Parameters
    ----------
    participation : float
        Participation rate.
    notional : float
        Product notional.
    cap : float or None
        Maximum return. None = unlimited.
    average_strike : bool
        If True, uses average as strike. If False (default), uses average as rate.
    """

    participation: float
    notional: float
    cap: float | None = None
    average_strike: bool = False

    @property
    def name(self) -> str:
        mode = "avg_strike" if self.average_strike else "avg_rate"
        cap_str = f", cap={self.cap:.0%}" if self.cap else ""
        return f"AverageParticipation({self.participation:.0%}, {mode}{cap_str})"

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        return evaluate_asian_participation_paths(
            paths=paths,
            obs_indices=obs_indices,
            notional=self.notional,
            participation=self.participation,
            cap=self.cap if self.cap is not None else 0.0,
            has_cap=self.cap is not None,
            is_average_strike=self.average_strike,
            df_terminal=discount_factors[-1],
        )


# =============================================================================
# Lookback Participation
# =============================================================================


@dataclass(frozen=True)
class LookbackParticipation(ProductComponent):
    """
    Lookback participation component.

    Payoff based on the maximum or minimum performance of the underlying
    across observation dates.

    Two modes:
    - Call-like (use_max=True): participation * max(max_perf - 1, 0)
    - Put-like (use_max=False): participation * max(1 - min_perf, 0)

    Parameters
    ----------
    participation : float
        Participation rate.
    notional : float
        Product notional.
    cap : float or None
        Maximum return. None = unlimited.
    use_max : bool
        True = lookback on max (call-like), False = lookback on min (put-like).
    """

    participation: float
    notional: float
    cap: float | None = None
    use_max: bool = True

    @property
    def name(self) -> str:
        mode = "max" if self.use_max else "min"
        cap_str = f", cap={self.cap:.0%}" if self.cap else ""
        return f"LookbackParticipation({self.participation:.0%}, {mode}{cap_str})"

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        return evaluate_lookback_participation_paths(
            paths=paths,
            obs_indices=obs_indices,
            notional=self.notional,
            participation=self.participation,
            cap=self.cap if self.cap is not None else 0.0,
            has_cap=self.cap is not None,
            use_max=self.use_max,
            df_terminal=discount_factors[-1],
        )


__all__ = [
    "AverageParticipation",
    "CliquetParticipation",
    "KnockOutParticipation",
    "LookbackParticipation",
    "TwinWinParticipation",
    "UpsideParticipation",
]
