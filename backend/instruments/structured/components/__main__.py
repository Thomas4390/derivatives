"""
Smoke test entrypoint — preserves `python -m backend.instruments.structured.components`.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import numpy as np

from backend.instruments.structured.components import (
    AutocallTrigger,
    BondFloor,
    ConditionalCoupon,
    FixedCoupon,
    KnockInPut,
    UpsideParticipation,
)

# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Structured Components Smoke Test")
    print("=" * 50)

    # Create synthetic paths
    np.random.seed(42)
    n_paths, n_steps = 10_000, 252
    s0 = 100.0
    dt = 1.0 / n_steps
    time_grid = np.linspace(0, 1.0, n_steps + 1)
    z = np.random.standard_normal((n_paths, n_steps))
    log_ret = (0.05 - 0.5 * 0.2**2) * dt + 0.2 * np.sqrt(dt) * z
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(np.cumsum(log_ret, axis=1))

    # Quarterly observations for 1 year
    obs_indices = np.array([63, 126, 189, 252])  # ~quarterly
    discount_factors = np.exp(-0.05 * time_grid[obs_indices])

    notional = 1000.0

    # Test BondFloor
    bf = BondFloor(protection_level=1.0, notional=notional, maturity=1.0)
    bf_pv = bf.evaluate(paths, time_grid, obs_indices, discount_factors)
    print(
        f"BondFloor(100%): PV = {bf_pv[0]:.2f} (expected ~{notional * np.exp(-0.05):.2f})"
    )

    # Test UpsideParticipation
    up = UpsideParticipation(participation=1.0, notional=notional, cap=1.3)
    up_pv = up.evaluate(paths, time_grid, obs_indices, discount_factors)
    print(f"UpsideParticipation(100%, cap=130%): mean PV = {np.mean(up_pv):.2f}")

    # Test FixedCoupon
    fc = FixedCoupon(coupon_rate=0.08, notional=notional)
    fc_pv = fc.evaluate(paths, time_grid, obs_indices, discount_factors)
    print(f"FixedCoupon(8%): PV = {fc_pv[0]:.2f}")

    # Test ConditionalCoupon
    cc = ConditionalCoupon(
        coupon_rate=0.10, barrier=0.7, notional=notional, memory=True
    )
    cc_pv = cc.evaluate(paths, time_grid, obs_indices, discount_factors)
    print(
        f"ConditionalCoupon(10%, barrier=70%, memory): mean PV = {np.mean(cc_pv):.2f}"
    )

    # Test AutocallTrigger
    at = AutocallTrigger(trigger_level=1.0, notional=notional)
    at_pv, called, call_idx = at.evaluate_with_call_info(
        paths, time_grid, obs_indices, discount_factors
    )
    print(f"AutocallTrigger(100%): P(autocall) = {np.mean(called):.2%}")

    # Test KnockInPut
    kip = KnockInPut(barrier=0.6, notional=notional, monitoring="continuous")
    kip_pv = kip.evaluate(paths, time_grid, obs_indices, discount_factors)
    print(f"KnockInPut(60%, continuous): mean PV = {np.mean(kip_pv):.2f}")

    print("\n" + "=" * 50)
    print("Components smoke test passed")
    print("=" * 50)
