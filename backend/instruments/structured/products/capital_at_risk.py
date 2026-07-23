"""
Capital-at-risk income/participation products (no bond floor, no autocall).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from backend.core.structured_product import (
    ObservationSchedule,
    ProductComponent,
    StructuredProduct,
)
from backend.engines.structured_kernels import (
    evaluate_rc_paths,
)
from backend.instruments.structured.components import (
    FixedCoupon,
    KnockInPut,
    TwinWinParticipation,
)

if TYPE_CHECKING:
    from backend.core.result_types import StructuredProductPricingComponents


@dataclass(frozen=True)
class ReverseConvertible(StructuredProduct):
    """
    Reverse Convertible.

    High fixed coupon + capital at risk via knock-in put.

    Parameters
    ----------
    notional_ : float
        Notional amount.
    maturity_ : float
        Maturity in years.
    coupon_rate : float
        Annualized coupon rate (e.g., 0.10 = 10%).
    barrier : float
        Knock-in barrier as fraction of initial (e.g., 0.6 = 60%).
    barrier_monitoring : str
        'continuous' or 'discrete'.
    observation_frequency : str
        Coupon observation frequency.
    """

    notional_: float
    maturity_: float
    coupon_rate: float = 0.10
    barrier: float = 0.60
    barrier_monitoring: str = "continuous"
    observation_frequency: str = "quarterly"

    def __post_init__(self) -> None:
        if self.notional_ <= 0:
            raise ValueError(f"notional must be positive, got {self.notional_}")
        if self.maturity_ <= 0:
            raise ValueError(f"maturity must be positive, got {self.maturity_}")

    @property
    def notional(self) -> float:
        return self.notional_

    @property
    def maturity(self) -> float:
        return self.maturity_

    @property
    def product_type(self) -> str:
        return "yield_enhancement"

    @property
    def observation_schedule(self) -> ObservationSchedule:
        return ObservationSchedule.from_frequency(
            self.maturity_, self.observation_frequency
        )

    @property
    def components(self) -> list[ProductComponent]:
        return [
            FixedCoupon(
                coupon_rate=self.coupon_rate,
                notional=self.notional_,
            ),
            KnockInPut(
                barrier=self.barrier,
                notional=self.notional_,
                monitoring=self.barrier_monitoring,
            ),
        ]

    def has_early_termination(self) -> bool:
        return False

    def evaluate_paths(
        self,
        paths: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
        obs_dt: np.ndarray,
        df_terminal: float,
        s0_reference: float = -1.0,
    ) -> StructuredProductPricingComponents:
        pv, coupon_pv, terminal_pv = evaluate_rc_paths(
            paths=paths,
            obs_indices=obs_indices,
            discount_factors=discount_factors,
            obs_dt=obs_dt,
            notional=self.notional_,
            coupon_rate=self.coupon_rate,
            ki_barrier=self.barrier,
            continuous_monitoring=(self.barrier_monitoring == "continuous"),
            df_terminal=df_terminal,
            s0_reference=s0_reference,
        )
        return {
            "pv": pv,
            "bond_floor_pv": terminal_pv,
            "option_pv": np.zeros_like(pv),
            "coupon_pv": coupon_pv,
            "autocall_probability": 0.0,
        }


@dataclass(frozen=True)
class TwinWin(StructuredProduct):
    """
    Twin Win Certificate.

    Profit from both upside and downside moves (absolute performance).
    Capital at risk if knock-in barrier is breached.
    """

    notional_: float
    maturity_: float
    participation_rate: float = 1.0
    ki_barrier: float = 0.60
    barrier_monitoring: str = "continuous"
    observation_frequency: str = "quarterly"

    def __post_init__(self) -> None:
        if self.notional_ <= 0:
            raise ValueError(f"notional must be positive, got {self.notional_}")
        if self.maturity_ <= 0:
            raise ValueError(f"maturity must be positive, got {self.maturity_}")

    @property
    def notional(self) -> float:
        return self.notional_

    @property
    def maturity(self) -> float:
        return self.maturity_

    @property
    def product_type(self) -> str:
        return "participation"

    @property
    def observation_schedule(self) -> ObservationSchedule:
        return ObservationSchedule.from_frequency(
            self.maturity_, self.observation_frequency
        )

    @property
    def components(self) -> list[ProductComponent]:
        return [
            TwinWinParticipation(
                participation=self.participation_rate,
                ki_barrier=self.ki_barrier,
                notional=self.notional_,
                monitoring=self.barrier_monitoring,
            ),
        ]

    def has_early_termination(self) -> bool:
        return False
