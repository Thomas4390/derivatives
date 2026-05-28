"""
Structured Products
===================

Three priority structured products built from reusable components:
- CapitalProtectedNote: Bond floor + upside participation
- ReverseConvertible: Fixed coupon + knock-in put
- Autocallable: Autocall trigger + conditional coupons + knock-in put

Author: Thomas Vaudescal
Created: 2026
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

if TYPE_CHECKING:
    from backend.core.result_types import StructuredProductPricingComponents
from backend.engines.structured_kernels import (
    evaluate_autocallable_paths,
    evaluate_cpn_paths,
    evaluate_rc_paths,
    evaluate_snowball_paths,
)
from backend.instruments.structured.components import (
    AutocallTrigger,
    BondFloor,
    ConditionalCoupon,
    FixedCoupon,
    KnockInPut,
    KnockOutParticipation,
    SnowballCoupon,
    TwinWinParticipation,
    UpsideParticipation,
)


# =============================================================================
# Capital Protected Note
# =============================================================================


@dataclass(frozen=True)
class CapitalProtectedNote(StructuredProduct):
    """
    Capital Protected Note (CPN).

    100% capital protection + capped participation in the upside.

    Parameters
    ----------
    notional_ : float
        Notional amount.
    maturity_ : float
        Maturity in years.
    participation_rate : float
        Participation in upside (e.g., 0.8 = 80%).
    cap : float or None
        Maximum performance level (e.g., 1.5 = 150%). None = unlimited.
    observation_frequency : str
        Frequency for observation schedule.
    protection_level : float
        Capital protection level (default 1.0 = 100%).
    """

    notional_: float
    maturity_: float
    participation_rate: float = 0.80
    cap: float | None = 1.50
    observation_frequency: str = "annual"
    protection_level: float = 1.0

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
        return "capital_protected"

    @property
    def observation_schedule(self) -> ObservationSchedule:
        return ObservationSchedule.from_frequency(
            self.maturity_, self.observation_frequency
        )

    @property
    def components(self) -> list[ProductComponent]:
        return [
            BondFloor(
                protection_level=self.protection_level,
                notional=self.notional_,
                maturity=self.maturity_,
            ),
            UpsideParticipation(
                participation=self.participation_rate,
                notional=self.notional_,
                cap=self.cap,
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
        pv, bond_pv, option_pv = evaluate_cpn_paths(
            paths=paths,
            discount_factor_terminal=df_terminal,
            notional=self.notional_,
            protection_level=self.protection_level,
            participation=self.participation_rate,
            cap=self.cap if self.cap is not None else 0.0,
            has_cap=self.cap is not None,
            s0_reference=s0_reference,
        )
        return {
            "pv": pv,
            "bond_floor_pv": bond_pv,
            "option_pv": option_pv,
            "coupon_pv": np.zeros_like(pv),
            "autocall_probability": 0.0,
        }


# =============================================================================
# Reverse Convertible
# =============================================================================


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


# =============================================================================
# Autocallable
# =============================================================================


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


# =============================================================================
# Phoenix Autocallable
# =============================================================================


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


# =============================================================================
# Shark Note
# =============================================================================


@dataclass(frozen=True)
class SharkNote(StructuredProduct):
    """
    Shark Note (Shark Fin).

    Capital protected + participation in upside with knock-out barrier.
    If the underlying touches the upper barrier, participation is replaced
    by a fixed rebate.

    No inter-component dependencies.
    """

    notional_: float
    maturity_: float
    participation_rate: float = 1.50
    ko_barrier: float = 1.40
    rebate: float = 0.05
    protection_level: float = 1.0
    observation_frequency: str = "annual"

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
        return "capital_protected"

    @property
    def observation_schedule(self) -> ObservationSchedule:
        return ObservationSchedule.from_frequency(
            self.maturity_, self.observation_frequency
        )

    @property
    def components(self) -> list[ProductComponent]:
        return [
            BondFloor(
                protection_level=self.protection_level,
                notional=self.notional_,
                maturity=self.maturity_,
            ),
            KnockOutParticipation(
                participation=self.participation_rate,
                barrier=self.ko_barrier,
                rebate=self.rebate,
                notional=self.notional_,
            ),
        ]

    def has_early_termination(self) -> bool:
        return False


# =============================================================================
# Twin Win
# =============================================================================


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


# =============================================================================
# Snowball Autocallable
# =============================================================================


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


# =============================================================================
# Cliquet Note
# =============================================================================


@dataclass(frozen=True)
class CliquetNote(StructuredProduct):
    """
    Cliquet (Ratchet) Note.

    Capital protection + locally capped/floored periodic return accumulation.
    The product accumulates returns at each observation period, each capped
    and floored individually, then applies a global cap/floor.

    Parameters
    ----------
    notional_ : float
        Notional amount.
    maturity_ : float
        Maturity in years.
    local_cap : float
        Maximum return per period (e.g., 0.05 = 5%).
    local_floor : float
        Minimum return per period (e.g., 0.0 = no negative contributions).
    global_cap : float
        Maximum total return (e.g., 0.50 = 50%).
    global_floor : float
        Minimum total return (e.g., 0.0 = principal protected).
    protection_level : float
        Capital protection level.
    observation_frequency : str
        Observation frequency for cliquet resets.
    """

    notional_: float
    maturity_: float
    local_cap: float = 0.05
    local_floor: float = 0.0
    global_cap: float = 1.0
    global_floor: float = 0.0
    protection_level: float = 1.0
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
        return "cliquet"

    @property
    def observation_schedule(self) -> ObservationSchedule:
        return ObservationSchedule.from_frequency(
            self.maturity_, self.observation_frequency
        )

    @property
    def components(self) -> list[ProductComponent]:
        from backend.instruments.structured.components import CliquetParticipation

        return [
            BondFloor(
                protection_level=self.protection_level,
                notional=self.notional_,
                maturity=self.maturity_,
            ),
            CliquetParticipation(
                notional=self.notional_,
                local_cap=self.local_cap,
                local_floor=self.local_floor,
                global_cap=self.global_cap,
                global_floor=self.global_floor,
            ),
        ]

    def has_early_termination(self) -> bool:
        return False


# =============================================================================
# Asian (Average) Note
# =============================================================================


@dataclass(frozen=True)
class AsianNote(StructuredProduct):
    """
    Asian (Average) Note.

    Capital protection + participation in the average performance.
    Averaging reduces volatility exposure, enabling higher participation rates.

    Parameters
    ----------
    notional_ : float
        Notional amount.
    maturity_ : float
        Maturity in years.
    participation_rate : float
        Participation in average performance.
    cap : float or None
        Maximum return. None = unlimited.
    protection_level : float
        Capital protection level.
    average_strike : bool
        If True, uses average as strike (more exotic). Default False.
    observation_frequency : str
        Averaging observation frequency.
    """

    notional_: float
    maturity_: float
    participation_rate: float = 1.50
    cap: float | None = None
    protection_level: float = 1.0
    average_strike: bool = False
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
        return "asian_note"

    @property
    def observation_schedule(self) -> ObservationSchedule:
        return ObservationSchedule.from_frequency(
            self.maturity_, self.observation_frequency
        )

    @property
    def components(self) -> list[ProductComponent]:
        from backend.instruments.structured.components import AverageParticipation

        return [
            BondFloor(
                protection_level=self.protection_level,
                notional=self.notional_,
                maturity=self.maturity_,
            ),
            AverageParticipation(
                participation=self.participation_rate,
                notional=self.notional_,
                cap=self.cap,
                average_strike=self.average_strike,
            ),
        ]

    def has_early_termination(self) -> bool:
        return False


# =============================================================================
# Lookback Note
# =============================================================================


@dataclass(frozen=True)
class LookbackNote(StructuredProduct):
    """
    Lookback Note.

    Capital protection + participation based on the best (or worst)
    performance of the underlying over observation dates.

    Parameters
    ----------
    notional_ : float
        Notional amount.
    maturity_ : float
        Maturity in years.
    participation_rate : float
        Participation rate.
    cap : float or None
        Maximum return. None = unlimited.
    protection_level : float
        Capital protection level.
    use_max : bool
        True = lookback on max (call-like), False = lookback on min (put-like).
    observation_frequency : str
        Observation frequency.
    """

    notional_: float
    maturity_: float
    participation_rate: float = 1.0
    cap: float | None = None
    protection_level: float = 1.0
    use_max: bool = True
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
        return "lookback_note"

    @property
    def observation_schedule(self) -> ObservationSchedule:
        return ObservationSchedule.from_frequency(
            self.maturity_, self.observation_frequency
        )

    @property
    def components(self) -> list[ProductComponent]:
        from backend.instruments.structured.components import LookbackParticipation

        return [
            BondFloor(
                protection_level=self.protection_level,
                notional=self.notional_,
                maturity=self.maturity_,
            ),
            LookbackParticipation(
                participation=self.participation_rate,
                notional=self.notional_,
                cap=self.cap,
                use_max=self.use_max,
            ),
        ]

    def has_early_termination(self) -> bool:
        return False


# =============================================================================
# Range Accrual Note
# =============================================================================


@dataclass(frozen=True)
class RangeAccrualNote(StructuredProduct):
    """
    Range Accrual Note.

    Bond floor + range accrual coupon that accrues proportionally to the
    time the underlying spends within a specified range.

    Parameters
    ----------
    notional_ : float
        Notional amount.
    maturity_ : float
        Maturity in years.
    coupon_rate : float
        Maximum annualized coupon rate.
    lower_barrier : float
        Lower range barrier as fraction of initial.
    upper_barrier : float
        Upper range barrier as fraction of initial.
    protection_level : float
        Capital protection level.
    observation_frequency : str
        Coupon payment frequency.
    """

    notional_: float
    maturity_: float
    coupon_rate: float = 0.10
    lower_barrier: float = 0.80
    upper_barrier: float = 1.20
    protection_level: float = 1.0
    observation_frequency: str = "quarterly"

    def __post_init__(self) -> None:
        if self.notional_ <= 0:
            raise ValueError(f"notional must be positive, got {self.notional_}")
        if self.maturity_ <= 0:
            raise ValueError(f"maturity must be positive, got {self.maturity_}")
        if self.lower_barrier >= self.upper_barrier:
            raise ValueError(
                f"lower_barrier ({self.lower_barrier}) must be < upper_barrier ({self.upper_barrier})"
            )

    @property
    def notional(self) -> float:
        return self.notional_

    @property
    def maturity(self) -> float:
        return self.maturity_

    @property
    def product_type(self) -> str:
        return "range_accrual"

    @property
    def observation_schedule(self) -> ObservationSchedule:
        return ObservationSchedule.from_frequency(
            self.maturity_, self.observation_frequency
        )

    @property
    def components(self) -> list[ProductComponent]:
        from backend.instruments.structured.components import RangeAccrualCoupon

        return [
            BondFloor(
                protection_level=self.protection_level,
                notional=self.notional_,
                maturity=self.maturity_,
            ),
            RangeAccrualCoupon(
                coupon_rate=self.coupon_rate,
                lower_barrier=self.lower_barrier,
                upper_barrier=self.upper_barrier,
                notional=self.notional_,
            ),
        ]

    def has_early_termination(self) -> bool:
        return False


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Structured Products Smoke Test")
    print("=" * 50)

    # Capital Protected Note
    cpn = CapitalProtectedNote(
        notional_=1000,
        maturity_=3.0,
        participation_rate=0.80,
        cap=1.50,
    )
    print(f"\n{cpn.name}: type={cpn.product_type}")
    print(f"  Components: {[c.name for c in cpn.components]}")
    print(f"  Schedule: {cpn.observation_schedule}")
    print(f"  Early termination: {cpn.has_early_termination()}")

    # Reverse Convertible
    rc = ReverseConvertible(
        notional_=1000,
        maturity_=1.0,
        coupon_rate=0.12,
        barrier=0.65,
    )
    print(f"\n{rc.name}: type={rc.product_type}")
    print(f"  Components: {[c.name for c in rc.components]}")
    print(f"  Schedule: {rc.observation_schedule}")

    # Autocallable
    auto = Autocallable(
        notional_=1000,
        maturity_=3.0,
        coupon_rate=0.07,
        autocall_trigger=1.0,
        coupon_barrier=0.70,
        ki_barrier=0.60,
        memory_coupon=True,
    )
    print(f"\n{auto.name}: type={auto.product_type}")
    print(f"  Components: {[c.name for c in auto.components]}")
    print(f"  Schedule: {auto.observation_schedule}")
    print(f"  Early termination: {auto.has_early_termination()}")

    print("\n" + "=" * 50)
    print("Products smoke test passed")
    print("=" * 50)
