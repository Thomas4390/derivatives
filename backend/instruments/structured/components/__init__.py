"""
Structured Product Components
==============================

Reusable building blocks for structured products. Each component implements
the ProductComponent interface and evaluates its cashflow contribution
given simulated paths.

Components:
- BondFloor: Capital protection (0-100%)
- UpsideParticipation: Participation in upside with optional cap
- FixedCoupon: Unconditional fixed coupon
- ConditionalCoupon: Coupon paid if underlying > barrier (optional memory)
- AutocallTrigger: Early redemption if underlying > trigger level
- KnockInPut: Put knock-in = capital at risk if barrier breached

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from backend.instruments.structured.components.barriers import (
    BonusDigital,
    GearedKnockInPut,
    KnockInPut,
    MaturityBarrierKnockInPut,
    TwoStepBonus,
)
from backend.instruments.structured.components.coupons import (
    CMIConditionalCoupon,
    ConditionalCoupon,
    FixedCoupon,
    RangeAccrualCoupon,
    SnowballCoupon,
    VariableIncomeCoupon,
)
from backend.instruments.structured.components.participation import (
    AverageParticipation,
    CliquetParticipation,
    KnockOutParticipation,
    LookbackParticipation,
    TwinWinParticipation,
    UpsideParticipation,
)
from backend.instruments.structured.components.redemptions import BondFloor
from backend.instruments.structured.components.triggers import AutocallTrigger

__all__ = [
    "AutocallTrigger",
    "AverageParticipation",
    "BondFloor",
    "BonusDigital",
    "CMIConditionalCoupon",
    "CliquetParticipation",
    "ConditionalCoupon",
    "FixedCoupon",
    "GearedKnockInPut",
    "KnockInPut",
    "KnockOutParticipation",
    "LookbackParticipation",
    "MaturityBarrierKnockInPut",
    "RangeAccrualCoupon",
    "SnowballCoupon",
    "TwinWinParticipation",
    "TwoStepBonus",
    "UpsideParticipation",
    "VariableIncomeCoupon",
]
