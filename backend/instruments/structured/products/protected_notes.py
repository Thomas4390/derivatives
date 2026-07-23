"""
Principal-protected structured products (each leads with a BondFloor).
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
    evaluate_cpn_paths,
)
from backend.instruments.structured.components import (
    BondFloor,
    KnockOutParticipation,
    UpsideParticipation,
)

if TYPE_CHECKING:
    from backend.core.result_types import StructuredProductPricingComponents


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
