"""
Smoke test for ``backend.instruments.payoffs`` — run via
``python -m backend.instruments.payoffs``.
"""

from __future__ import annotations

import numpy as np

from backend.instruments.payoffs import (
    CompositePayoff,
    DigitalCallPayoff,
    DigitalPutPayoff,
    VanillaCallPayoff,
    VanillaPutPayoff,
)


if __name__ == "__main__":
    # Smoke test
    print("=" * 50)
    print("Payoffs Module Smoke Test")
    print("=" * 50)

    spots = np.array([90.0, 100.0, 110.0])

    # Vanilla payoffs
    call = VanillaCallPayoff(strike=100.0)
    put = VanillaPutPayoff(strike=100.0)

    print("\nVanilla Call (K=100):")
    print(f"  Spots: {spots}")
    print(f"  Payoffs: {call(spots)}")

    print("\nVanilla Put (K=100):")
    print(f"  Spots: {spots}")
    print(f"  Payoffs: {put(spots)}")

    # Digital payoffs
    digital_call = DigitalCallPayoff(strike=100.0, payout=10.0)
    digital_put = DigitalPutPayoff(strike=100.0, payout=10.0)

    print("\nDigital Call (K=100, payout=10):")
    print(f"  Payoffs: {digital_call(spots)}")

    print("\nDigital Put (K=100, payout=10):")
    print(f"  Payoffs: {digital_put(spots)}")

    # Composite payoff (straddle)
    straddle = CompositePayoff([(1.0, call), (1.0, put)])
    print("\nStraddle (Call + Put at K=100):")
    print(f"  Payoffs: {straddle(spots)}")

    # Test path dependency
    print("\nPath dependency:")
    print(f"  Call: {call.is_path_dependent}")
    print(f"  Straddle: {straddle.is_path_dependent}")

    print("\n" + "=" * 50)
    print("Payoffs smoke test passed")
    print("=" * 50)
