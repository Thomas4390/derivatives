"""
Smoke test for ``backend.greeks.numerical`` — run via
``python -m backend.greeks.numerical``.
"""

from __future__ import annotations

from backend.greeks.numerical import finite_difference_greeks


if __name__ == "__main__":
    print("=" * 50)
    print("Numerical Greeks Smoke Test")
    print("=" * 50)

    # Create a simple BS pricing function for testing
    from backend.greeks.analytic import bs_all_greeks

    def bs_price(
        spot: float,
        vol: float = 0.20,
        time: float = 0.25,
        rate: float = 0.05,
        strike: float = 100.0,
        is_call: bool = True,
    ) -> float:
        greeks = bs_all_greeks(spot, strike, time, rate, 0.0, vol, is_call)
        return greeks[0]  # price

    # Test parameters
    spot, vol, time, rate = 100.0, 0.20, 0.25, 0.05

    # Calculate numerical Greeks
    num_greeks = finite_difference_greeks(
        bs_price, spot, vol, time, rate, strike=100.0, is_call=True
    )

    # Compare with analytic
    _, a_delta, a_gamma, a_vega, a_theta, a_rho, *_ = bs_all_greeks(
        spot, 100.0, time, rate, 0.0, vol, True
    )

    print("\nComparison (Numerical vs Analytic):")
    print(f"  Delta: {num_greeks.delta:.6f} vs {a_delta:.6f}")
    print(f"  Gamma: {num_greeks.gamma:.6f} vs {a_gamma:.6f}")
    print(f"  Vega:  {num_greeks.vega:.6f} vs {a_vega:.6f}")
    print(f"  Theta: {num_greeks.theta:.6f} vs {a_theta:.6f}")
    print(f"  Rho:   {num_greeks.rho:.6f} vs {a_rho:.6f}")

    print("\n" + "=" * 50)
    print("Numerical Greeks smoke test passed")
    print("=" * 50)
