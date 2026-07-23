"""
Smoke test for ``backend.utils.math`` — run via ``python -m backend.utils.math``.
"""

from __future__ import annotations

import math

import numpy as np

from backend.utils.math import (
    bs_greeks,
    bs_price,
    bs_second_order_greeks,
    bs_third_order_greeks,
    d1_d2,
    delta_to_strike,
    discount_factor,
    forward_price,
    implied_volatility,
    log_moneyness,
    norm_cdf,
    norm_cdf_vec,
    norm_inv_cdf,
    norm_pdf,
    norm_pdf_vec,
)


if __name__ == "__main__":
    print("=" * 50)
    print("Math Utilities Smoke Test")
    print("=" * 50)

    # Test parameters
    s, k, t, r, q, sigma = 100.0, 100.0, 0.25, 0.05, 0.02, 0.20

    # --- Test 1: Normal distribution functions ---
    print("\n--- Test 1: Normal Distribution Functions ---")
    print(f"  norm_cdf(0.0) = {norm_cdf(0.0):.6f} (expected: 0.5)")
    print(f"  norm_cdf(1.96) = {norm_cdf(1.96):.6f} (expected: ~0.975)")
    print(f"  norm_pdf(0.0) = {norm_pdf(0.0):.6f} (expected: ~0.3989)")
    print(f"  norm_inv_cdf(0.5) = {norm_inv_cdf(0.5):.6f} (expected: 0.0)")
    print(f"  norm_inv_cdf(0.975) = {norm_inv_cdf(0.975):.4f} (expected: ~1.96)")

    # Verify round-trip
    for p in [0.1, 0.5, 0.9]:
        x = norm_inv_cdf(p)
        p_back = norm_cdf(x)
        print(f"  norm_cdf(norm_inv_cdf({p})) = {p_back:.6f}")
        assert abs(p - p_back) < 1e-6, f"Round-trip failed for p={p}"

    # --- Test 2: d1/d2 parameters ---
    print("\n--- Test 2: d1/d2 Parameters ---")
    d1, d2 = d1_d2(s, k, t, r, sigma)
    print(f"  d1 = {d1:.6f}")
    print(f"  d2 = {d2:.6f}")
    print(
        f"  d1 - d2 = {d1 - d2:.6f} (should equal sigma*sqrt(t) = {sigma * math.sqrt(t):.6f})"
    )

    # --- Test 3: Black-Scholes pricing ---
    print("\n--- Test 3: Black-Scholes Pricing ---")
    call_price = bs_price(s, k, t, r, sigma, is_call=True)
    put_price = bs_price(s, k, t, r, sigma, is_call=False)
    print(f"  ATM Call price: ${call_price:.4f}")
    print(f"  ATM Put price:  ${put_price:.4f}")

    # Put-call parity check
    parity_lhs = call_price - put_price
    parity_rhs = s - k * math.exp(-r * t)
    print(
        f"  Put-call parity: C - P = {parity_lhs:.4f}, S - K*e^(-rT) = {parity_rhs:.4f}"
    )
    assert abs(parity_lhs - parity_rhs) < 0.01, "Put-call parity violated"

    # --- Test 4: First-order Greeks ---
    print("\n--- Test 4: First-Order Greeks ---")
    price, delta, gamma, vega, theta, rho = bs_greeks(s, k, t, r, sigma, is_call=True)
    print(f"  Price: ${price:.4f}")
    print(f"  Delta: {delta:.6f}")
    print(f"  Gamma: {gamma:.6f}")
    print(f"  Vega:  {vega:.6f} (per 1% vol)")
    print(f"  Theta: {theta:.6f} (per day)")
    print(f"  Rho:   {rho:.6f} (per 1% rate)")

    # Delta should be ~0.5 for ATM call
    assert 0.4 < delta < 0.6, f"ATM delta out of expected range: {delta}"

    # --- Test 5: Second-order Greeks ---
    print("\n--- Test 5: Second-Order Greeks ---")
    vanna, volga, charm, veta = bs_second_order_greeks(s, k, t, r, sigma)
    print(f"  Vanna: {vanna:.6f}")
    print(f"  Volga: {volga:.6f}")
    print(f"  Charm: {charm:.6f}")
    print(f"  Veta:  {veta:.6f}")

    # --- Test 6: Third-order Greeks ---
    print("\n--- Test 6: Third-Order Greeks ---")
    speed, zomma, color, ultima = bs_third_order_greeks(s, k, t, r, sigma)
    print(f"  Speed:  {speed:.8f}")
    print(f"  Zomma:  {zomma:.8f}")
    print(f"  Color:  {color:.8f}")
    print(f"  Ultima: {ultima:.10f}")

    # --- Test 7: Implied Volatility ---
    print("\n--- Test 7: Implied Volatility ---")
    # Round-trip: price -> IV -> price
    target_price = call_price
    recovered_iv = implied_volatility(target_price, s, k, t, r, is_call=True)
    print(f"  Original sigma: {sigma:.4f}")
    print(f"  Recovered IV:   {recovered_iv:.4f}")
    assert abs(recovered_iv - sigma) < 0.001, (
        f"IV recovery failed: {recovered_iv} vs {sigma}"
    )

    # --- Test 8: Utility functions ---
    print("\n--- Test 8: Utility Functions ---")
    df = discount_factor(r, t)
    fwd = forward_price(s, r, 0.0, t)
    print(f"  Discount factor: {df:.6f}")
    print(f"  Forward price:   {fwd:.4f}")
    print(f"  Log-moneyness:   {log_moneyness(s, k):.6f}")

    # Delta to strike round-trip
    target_delta = 0.25
    strike_25d = delta_to_strike(s, target_delta, t, r, sigma, is_call=True)
    print(f"  Strike for 25-delta call: ${strike_25d:.2f}")

    # --- Test 9: Vectorized functions ---
    print("\n--- Test 9: Vectorized Functions ---")
    x_vec = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    cdf_vec = norm_cdf_vec(x_vec)
    pdf_vec = norm_pdf_vec(x_vec)
    print(f"  norm_cdf_vec: {cdf_vec}")
    print(f"  norm_pdf_vec: {pdf_vec}")

    # --- Test 10: Edge cases ---
    print("\n--- Test 10: Edge Cases ---")
    # At expiry
    price_exp = bs_price(s, k, 0.0, r, sigma, is_call=True)
    print(f"  Price at expiry (S=K): ${price_exp:.4f} (expected: $0.00)")

    # Zero volatility (deep ITM)
    d1_zv, d2_zv = d1_d2(110, 100, t, r, 0.0)
    print(f"  d1 with zero vol (ITM): {d1_zv:.0f} (expected: large positive)")

    print("\n" + "=" * 50)
    print("Math utilities smoke test passed")
    print("=" * 50)
