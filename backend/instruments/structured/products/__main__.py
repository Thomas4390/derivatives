"""
Smoke test for ``backend.instruments.structured.products`` — run via
``python -m backend.instruments.structured.products``.
"""

from __future__ import annotations

from backend.instruments.structured.products import (
    Autocallable,
    CapitalProtectedNote,
    ReverseConvertible,
)


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
