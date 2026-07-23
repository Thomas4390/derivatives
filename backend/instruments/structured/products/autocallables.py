"""
Autocallable structured products (AutocallTrigger; early termination).
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
    evaluate_autocallable_paths,
    evaluate_snowball_paths,
)
from backend.instruments.structured.components import (
    AutocallTrigger,
    ConditionalCoupon,
    KnockInPut,
    SnowballCoupon,
)

if TYPE_CHECKING:
    from backend.core.result_types import StructuredProductPricingComponents


@dataclass(frozen=True)
class Autocallable(StructuredProduct):
    """
    Autocallable structured product.

    Combines:
    - AutocallTrigger: early redemption if performance >= trigger
    - ConditionalCoupon: coupon paid if performance >= coupon barrier
    - KnockInPut: capital at risk if barrier breached and not autocalled

    Parameters
    ----------
    notional_ : float
        Notional amount.
    maturity_ : float
        Maximum maturity in years.
    coupon_rate : float
        Annualized conditional coupon rate.
    autocall_trigger : float
        Performance level for autocall (e.g., 1.0 = 100%).
    coupon_barrier : float
        Performance level for coupon payment (e.g., 0.7 = 70%).
    ki_barrier : float
        Knock-in put barrier (e.g., 0.6 = 60%).
    memory_coupon : bool
        Whether unpaid coupons accumulate.
    barrier_monitoring : str
        'continuous' or 'discrete' for knock-in put.
    observation_frequency : str
        Observation/coupon frequency.
    """

    notional_: float
    maturity_: float
    coupon_rate: float = 0.07
    autocall_trigger: float = 1.0
    coupon_barrier: float = 0.70
    ki_barrier: float = 0.60
    memory_coupon: bool = True
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
            AutocallTrigger(
                trigger_level=self.autocall_trigger,
                notional=self.notional_,
            ),
            ConditionalCoupon(
                coupon_rate=self.coupon_rate,
                barrier=self.coupon_barrier,
                notional=self.notional_,
                memory=self.memory_coupon,
            ),
            KnockInPut(
                barrier=self.ki_barrier,
                notional=self.notional_,
                monitoring=self.barrier_monitoring,
            ),
        ]

    def has_early_termination(self) -> bool:
        return True

    def evaluate_paths(
        self,
        paths: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
        obs_dt: np.ndarray,
        df_terminal: float,
        s0_reference: float = -1.0,
    ) -> StructuredProductPricingComponents:
        pv, called, coupon_pv = evaluate_autocallable_paths(
            paths=paths,
            obs_indices=obs_indices,
            discount_factors=discount_factors,
            obs_dt=obs_dt,
            notional=self.notional_,
            autocall_trigger=self.autocall_trigger,
            coupon_rate=self.coupon_rate,
            coupon_barrier=self.coupon_barrier,
            ki_barrier=self.ki_barrier,
            memory_coupon=self.memory_coupon,
            continuous_monitoring=(self.barrier_monitoring == "continuous"),
            df_terminal=df_terminal,
            s0_reference=s0_reference,
        )
        return {
            "pv": pv,
            "bond_floor_pv": np.zeros_like(pv),
            "option_pv": np.zeros_like(pv),
            "coupon_pv": coupon_pv,
            "autocall_probability": float(np.mean(called)),
        }


@dataclass(frozen=True)
class PhoenixAutocallable(StructuredProduct):
    """
    Phoenix Autocallable.

    Conditional coupons (monthly/quarterly) + autocall trigger (typically annual)
    + knock-in put for capital protection. Similar to Autocallable but typically
    with more frequent coupon observations and different trigger/barrier structure.

    Uses the generic evaluator via EvaluationContext (no custom kernel needed).
    """

    notional_: float
    maturity_: float
    coupon_rate: float = 0.08
    autocall_trigger: float = 1.0
    coupon_barrier: float = 0.65
    ki_barrier: float = 0.55
    memory_coupon: bool = True
    barrier_monitoring: str = "continuous"
    observation_frequency: str = "monthly"

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
            AutocallTrigger(
                trigger_level=self.autocall_trigger,
                notional=self.notional_,
            ),
            ConditionalCoupon(
                coupon_rate=self.coupon_rate,
                barrier=self.coupon_barrier,
                notional=self.notional_,
                memory=self.memory_coupon,
            ),
            KnockInPut(
                barrier=self.ki_barrier,
                notional=self.notional_,
                monitoring=self.barrier_monitoring,
            ),
        ]

    def has_early_termination(self) -> bool:
        return True

    def evaluate_paths(
        self,
        paths: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
        obs_dt: np.ndarray,
        df_terminal: float,
        s0_reference: float = -1.0,
    ) -> StructuredProductPricingComponents:
        pv, called, coupon_pv = evaluate_autocallable_paths(
            paths=paths,
            obs_indices=obs_indices,
            discount_factors=discount_factors,
            obs_dt=obs_dt,
            notional=self.notional_,
            autocall_trigger=self.autocall_trigger,
            coupon_rate=self.coupon_rate,
            coupon_barrier=self.coupon_barrier,
            ki_barrier=self.ki_barrier,
            memory_coupon=self.memory_coupon,
            continuous_monitoring=(self.barrier_monitoring == "continuous"),
            df_terminal=df_terminal,
            s0_reference=s0_reference,
        )
        return {
            "pv": pv,
            "bond_floor_pv": np.zeros_like(pv),
            "option_pv": np.zeros_like(pv),
            "coupon_pv": coupon_pv,
            "autocall_probability": float(np.mean(called)),
        }


@dataclass(frozen=True)
class SnowballAutocallable(StructuredProduct):
    """
    Snowball Autocallable.

    Autocall + snowball coupon (coupon grows with time) + knock-in put.
    The coupon at observation j is rate * t_j * notional (increasing over time).
    """

    notional_: float
    maturity_: float
    coupon_rate: float = 0.10
    autocall_trigger: float = 1.0
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
        return "yield_enhancement"

    @property
    def observation_schedule(self) -> ObservationSchedule:
        return ObservationSchedule.from_frequency(
            self.maturity_, self.observation_frequency
        )

    @property
    def components(self) -> list[ProductComponent]:
        return [
            AutocallTrigger(
                trigger_level=self.autocall_trigger,
                notional=self.notional_,
            ),
            SnowballCoupon(
                coupon_rate=self.coupon_rate,
                notional=self.notional_,
            ),
            KnockInPut(
                barrier=self.ki_barrier,
                notional=self.notional_,
                monitoring=self.barrier_monitoring,
            ),
        ]

    def has_early_termination(self) -> bool:
        return True

    def evaluate_paths(
        self,
        paths: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
        obs_dt: np.ndarray,
        df_terminal: float,
        s0_reference: float = -1.0,
    ) -> StructuredProductPricingComponents:
        obs_times = np.cumsum(obs_dt)
        pv, called, coupon_pv = evaluate_snowball_paths(
            paths=paths,
            obs_indices=obs_indices,
            discount_factors=discount_factors,
            obs_times=obs_times,
            notional=self.notional_,
            autocall_trigger=self.autocall_trigger,
            coupon_rate=self.coupon_rate,
            ki_barrier=self.ki_barrier,
            continuous_monitoring=(self.barrier_monitoring == "continuous"),
            df_terminal=df_terminal,
            s0_reference=s0_reference,
        )
        return {
            "pv": pv,
            "bond_floor_pv": np.zeros_like(pv),
            "option_pv": np.zeros_like(pv),
            "coupon_pv": coupon_pv,
            "autocall_probability": float(np.mean(called)),
        }
