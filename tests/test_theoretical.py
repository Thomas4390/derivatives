"""
Theoretical Principles Tests
============================

Tests for fundamental theoretical relationships in option pricing:
- Put-Call Parity
- Greeks Sum Rules
- Boundary Conditions
- Model Consistency
- Convergence Properties

These tests validate that the implementation adheres to financial theory.

Author: Thomas
Created: 2025
"""

import pytest
import numpy as np

from backend.instruments.options import VanillaOption
from backend.models.gbm import GBMModel
from backend.models.heston import HestonModel
from backend.models.bates import BatesModel
from backend.engines import BSAnalyticEngine, MonteCarloEngine
from backend.core.market import MarketEnvironment

from tests.conftest import report


# =============================================================================
# PUT-CALL PARITY TESTS
# =============================================================================

class TestPutCallParity:
    """
    Put-Call Parity: C - P = S*exp(-qT) - K*exp(-rT)

    This fundamental relationship must hold for European options.
    """

    @pytest.mark.parametrize("strike", [80, 90, 100, 110, 120])
    @pytest.mark.parametrize("maturity", [0.1, 0.25, 0.5, 1.0])
    def test_put_call_parity_bs(self, strike, maturity, market_atm, gbm_model, bs_engine):
        """Test put-call parity holds for Black-Scholes pricing."""
        report.header(f"Put-Call Parity BS (K={strike}, T={maturity})")
        report.info("Tests fundamental put-call parity: C - P = S*exp(-qT) - K*exp(-rT)")
        report.info("Must hold for European options to prevent arbitrage")

        call = VanillaOption(strike=strike, maturity=maturity, is_call=True)
        put = VanillaOption(strike=strike, maturity=maturity, is_call=False)

        call_price = bs_engine.price(call, gbm_model, market_atm).price
        put_price = bs_engine.price(put, gbm_model, market_atm).price

        # Put-Call Parity: C - P = S*exp(-qT) - K*exp(-rT)
        S = market_atm.spot
        K = strike
        r = market_atm.rate
        q = market_atm.dividend_yield
        T = maturity

        lhs = call_price - put_price
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)

        report.params(strike=K, maturity=T, spot=S, rate=r, dividend=q)
        report.value("Call Price", call_price, unit="$")
        report.value("Put Price", put_price, unit="$")
        report.value("C - P (LHS)", lhs, expected=rhs, unit="$")
        report.success("Put-Call Parity holds: C - P = S*exp(-qT) - K*exp(-rT)")

        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)

    @pytest.mark.parametrize("strike", [90, 100, 110])
    def test_put_call_parity_fft(self, strike, market_atm, heston_model, fft_engine):
        """Test put-call parity holds for FFT pricing with Heston model."""
        report.header(f"Put-Call Parity FFT/Heston (K={strike})")
        report.info("Tests put-call parity with FFT pricing on Heston model")
        report.info("Numerical pricing must preserve fundamental relationships")

        call = VanillaOption(strike=strike, maturity=0.5, is_call=True)
        put = VanillaOption(strike=strike, maturity=0.5, is_call=False)

        call_price = fft_engine.price(call, heston_model, market_atm).price
        put_price = fft_engine.price(put, heston_model, market_atm).price

        S = market_atm.spot
        K = strike
        r = market_atm.rate
        T = 0.5

        lhs = call_price - put_price
        rhs = S - K * np.exp(-r * T)

        report.params(strike=K, maturity=T, spot=S, rate=r)
        report.value("Call Price (FFT)", call_price, unit="$")
        report.value("Put Price (FFT)", put_price, unit="$")
        report.value("C - P (LHS)", lhs, expected=rhs, unit="$")
        report.success("Put-Call Parity holds with FFT/Heston")

        # FFT has numerical errors, use looser tolerance
        np.testing.assert_allclose(lhs, rhs, rtol=1e-4, atol=0.01)

    def test_put_call_parity_with_dividends(self, market_with_dividend, gbm_model, bs_engine):
        """Test put-call parity with continuous dividend yield."""
        report.header("Put-Call Parity with Dividends")
        report.info("Tests put-call parity with continuous dividend yield q > 0")
        report.info("Dividend adjustment: S*exp(-qT) replaces S in the formula")

        call = VanillaOption(strike=100, maturity=0.5, is_call=True)
        put = VanillaOption(strike=100, maturity=0.5, is_call=False)

        call_price = bs_engine.price(call, gbm_model, market_with_dividend).price
        put_price = bs_engine.price(put, gbm_model, market_with_dividend).price

        S = market_with_dividend.spot
        K = 100
        r = market_with_dividend.rate
        q = market_with_dividend.dividend_yield
        T = 0.5

        lhs = call_price - put_price
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)

        report.params(strike=K, maturity=T, spot=S, rate=r, dividend=q)
        report.value("Call Price", call_price, unit="$")
        report.value("Put Price", put_price, unit="$")
        report.value("S*exp(-qT)", S * np.exp(-q * T), unit="$")
        report.value("K*exp(-rT)", K * np.exp(-r * T), unit="$")
        report.value("C - P (LHS)", lhs, expected=rhs, unit="$")
        report.success("Put-Call Parity holds with dividend yield")

        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


# =============================================================================
# GREEKS SUM RULES
# =============================================================================

class TestGreeksSumRules:
    """
    Test fundamental relationships between Greeks.
    """

    def test_delta_call_put_relationship(self, market_atm, gbm_model, bs_engine):
        """
        For same strike/maturity: delta_call - delta_put = exp(-qT)
        """
        report.header("Delta Call-Put Relationship")
        report.info("Tests delta relationship: delta_call - delta_put = exp(-qT)")
        report.info("Derivative of put-call parity with respect to spot")

        call = VanillaOption(strike=100, maturity=0.25, is_call=True)
        put = VanillaOption(strike=100, maturity=0.25, is_call=False)

        call_greeks = bs_engine.greeks(call, gbm_model, market_atm)
        put_greeks = bs_engine.greeks(put, gbm_model, market_atm)

        q = market_atm.dividend_yield
        T = 0.25
        expected_diff = np.exp(-q * T)

        actual_diff = call_greeks.delta - put_greeks.delta

        report.params(strike=100, maturity=T, dividend=q)
        report.value("Call Delta", call_greeks.delta)
        report.value("Put Delta", put_greeks.delta)
        report.value("Delta_call - Delta_put", actual_diff, expected=expected_diff)
        report.info(f"Expected difference: exp(-qT) = exp(-{q}*{T}) = {expected_diff:.6f}")
        report.success("Delta relationship holds: delta_call - delta_put = exp(-qT)")

        np.testing.assert_allclose(actual_diff, expected_diff, rtol=1e-6)

    def test_gamma_same_for_call_put(self, market_atm, gbm_model, bs_engine):
        """
        Gamma is the same for call and put with same strike/maturity.
        """
        report.header("Gamma Equality for Call and Put")
        report.info("Tests that gamma is identical for call and put at same strike")
        report.info("Second derivative of put-call parity: gamma_C = gamma_P")

        call = VanillaOption(strike=100, maturity=0.25, is_call=True)
        put = VanillaOption(strike=100, maturity=0.25, is_call=False)

        call_greeks = bs_engine.greeks(call, gbm_model, market_atm)
        put_greeks = bs_engine.greeks(put, gbm_model, market_atm)

        report.params(strike=100, maturity=0.25)
        report.comparison("Call Gamma", call_greeks.gamma, "Put Gamma", put_greeks.gamma)
        report.success("Gamma is identical for call and put")

        np.testing.assert_allclose(call_greeks.gamma, put_greeks.gamma, rtol=1e-10)

    def test_vega_same_for_call_put(self, market_atm, gbm_model, bs_engine):
        """
        Vega is the same for call and put with same strike/maturity.
        """
        report.header("Vega Equality for Call and Put")
        report.info("Tests that vega is identical for call and put at same strike")
        report.info("Volatility affects call and put equally (forward price invariant)")

        call = VanillaOption(strike=100, maturity=0.25, is_call=True)
        put = VanillaOption(strike=100, maturity=0.25, is_call=False)

        call_greeks = bs_engine.greeks(call, gbm_model, market_atm)
        put_greeks = bs_engine.greeks(put, gbm_model, market_atm)

        report.params(strike=100, maturity=0.25)
        report.comparison("Call Vega", call_greeks.vega, "Put Vega", put_greeks.vega)
        report.success("Vega is identical for call and put")

        np.testing.assert_allclose(call_greeks.vega, put_greeks.vega, rtol=1e-10)

    def test_theta_rho_parity(self, market_atm, gbm_model, bs_engine):
        """
        Relationship: theta_call - theta_put = r*K*exp(-rT) - q*S*exp(-qT)
        """
        report.header("Theta-Rho Parity Relationship")
        report.info("Tests theta relationship from put-call parity derivative")
        report.info("Time decay differs between calls and puts due to discounting")

        call = VanillaOption(strike=100, maturity=0.25, is_call=True)
        put = VanillaOption(strike=100, maturity=0.25, is_call=False)

        call_greeks = bs_engine.greeks(call, gbm_model, market_atm)
        put_greeks = bs_engine.greeks(put, gbm_model, market_atm)

        S = market_atm.spot
        K = 100
        r = market_atm.rate
        q = market_atm.dividend_yield
        T = 0.25

        theta_diff = call_greeks.theta - put_greeks.theta
        # Theta is now scaled per calendar day (÷365), so scale expected accordingly
        expected_diff = (-r * K * np.exp(-r * T) + q * S * np.exp(-q * T)) / 365

        report.params(strike=K, maturity=T, spot=S, rate=r, dividend=q)
        report.value("Call Theta", call_greeks.theta)
        report.value("Put Theta", put_greeks.theta)
        report.value("Theta_call - Theta_put", theta_diff, expected=expected_diff)
        report.info(f"Expected: -r*K*exp(-rT) + q*S*exp(-qT) = {expected_diff:.6f}")
        report.success("Theta parity relationship holds")

        np.testing.assert_allclose(theta_diff, expected_diff, rtol=1e-4)

    def test_atm_delta_approximately_half(self, market_atm, gbm_model, bs_engine):
        """
        ATM call delta is approximately 0.5 (slightly higher due to drift).
        """
        report.header("ATM Delta Approximation")
        report.info("Tests that ATM call delta is approximately 0.5")
        report.info("Delta > 0.5 due to positive drift (forward > spot)")

        # Use longer maturity for clearer effect
        call = VanillaOption(strike=100, maturity=1.0, is_call=True)
        greeks = bs_engine.greeks(call, gbm_model, market_atm)

        report.params(strike=100, maturity=1.0, spot=market_atm.spot, rate=market_atm.rate)
        report.value("ATM Call Delta", greeks.delta)
        report.info("Expected range: 0.5 < delta < 0.65 (due to positive drift)")
        report.success(f"ATM delta = {greeks.delta:.4f} is in expected range (0.5, 0.65)")

        # Delta should be between 0.5 and 0.6 for ATM call with positive drift
        assert 0.5 < greeks.delta < 0.65


# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

class TestBoundaryConditions:
    """
    Test that option prices satisfy boundary conditions.
    """

    def test_call_lower_bound(self, market_atm, gbm_model, bs_engine):
        """
        Call price >= max(0, S*exp(-qT) - K*exp(-rT))
        """
        report.header("Call Lower Bound Test")
        report.info("Tests that call prices satisfy lower bound: C >= max(0, S*exp(-qT) - K*exp(-rT))")
        report.info("Violating this bound creates arbitrage via synthetic forward")

        S = market_atm.spot
        r = market_atm.rate
        q = market_atm.dividend_yield
        T = 0.25

        results = []
        for strike in [80, 100, 120]:
            call = VanillaOption(strike=strike, maturity=0.25, is_call=True)
            price = bs_engine.price(call, gbm_model, market_atm).price

            K = strike
            lower_bound = max(0, S * np.exp(-q * T) - K * np.exp(-r * T))
            margin = price - lower_bound

            results.append((strike, price, lower_bound, margin))
            assert price >= lower_bound - 1e-10

        report.table(
            headers=["Strike", "Price", "Lower Bound", "Margin"],
            rows=results,
            title="Call Lower Bounds",
            precision=4
        )
        report.success("All call prices satisfy lower bound constraint")

    def test_put_lower_bound(self, market_atm, gbm_model, bs_engine):
        """
        Put price >= max(0, K*exp(-rT) - S*exp(-qT))
        """
        report.header("Put Lower Bound Test")
        report.info("Tests that put prices satisfy lower bound: P >= max(0, K*exp(-rT) - S*exp(-qT))")
        report.info("Symmetric to call bound via put-call parity relationship")

        S = market_atm.spot
        r = market_atm.rate
        q = market_atm.dividend_yield
        T = 0.25

        results = []
        for strike in [80, 100, 120]:
            put = VanillaOption(strike=strike, maturity=0.25, is_call=False)
            price = bs_engine.price(put, gbm_model, market_atm).price

            K = strike
            lower_bound = max(0, K * np.exp(-r * T) - S * np.exp(-q * T))
            margin = price - lower_bound

            results.append((strike, price, lower_bound, margin))
            assert price >= lower_bound - 1e-10

        report.table(
            headers=["Strike", "Price", "Lower Bound", "Margin"],
            rows=results,
            title="Put Lower Bounds",
            precision=4
        )
        report.success("All put prices satisfy lower bound constraint")

    def test_call_upper_bound(self, market_atm, gbm_model, bs_engine):
        """
        Call price <= S*exp(-qT)
        """
        report.header("Call Upper Bound Test")
        report.info("Tests that call prices satisfy upper bound: C <= S*exp(-qT)")
        report.info("A call cannot be worth more than the underlying asset itself")

        call = VanillaOption(strike=100, maturity=0.25, is_call=True)
        price = bs_engine.price(call, gbm_model, market_atm).price

        S = market_atm.spot
        q = market_atm.dividend_yield
        T = 0.25

        upper_bound = S * np.exp(-q * T)

        report.params(strike=100, maturity=T, spot=S)
        report.value("Call Price", price, unit="$")
        report.value("Upper Bound (S*exp(-qT))", upper_bound, unit="$")
        report.value("Margin", upper_bound - price, unit="$")
        report.success(f"Call price {price:.4f} <= upper bound {upper_bound:.4f}")

        assert price <= upper_bound + 1e-10

    def test_put_upper_bound(self, market_atm, gbm_model, bs_engine):
        """
        Put price <= K*exp(-rT)
        """
        report.header("Put Upper Bound Test")
        report.info("Tests that put prices satisfy upper bound: P <= K*exp(-rT)")
        report.info("A put cannot be worth more than discounted strike (max possible payoff)")

        put = VanillaOption(strike=100, maturity=0.25, is_call=False)
        price = bs_engine.price(put, gbm_model, market_atm).price

        K = 100
        r = market_atm.rate
        T = 0.25

        upper_bound = K * np.exp(-r * T)

        report.params(strike=K, maturity=T, rate=r)
        report.value("Put Price", price, unit="$")
        report.value("Upper Bound (K*exp(-rT))", upper_bound, unit="$")
        report.value("Margin", upper_bound - price, unit="$")
        report.success(f"Put price {price:.4f} <= upper bound {upper_bound:.4f}")

        assert price <= upper_bound + 1e-10

    def test_deep_itm_call_approaches_intrinsic(self, gbm_model, bs_engine):
        """
        Deep ITM call approaches discounted intrinsic value as vol -> 0.
        """
        report.header("Deep ITM Call Intrinsic Value Test")
        report.info("Tests that deep ITM call converges to discounted intrinsic as vol -> 0")
        report.info("With zero uncertainty, option value equals deterministic payoff")

        market = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.0)
        low_vol_model = GBMModel(sigma=0.01)  # Very low vol

        call = VanillaOption(strike=50, maturity=0.25, is_call=True)  # Deep ITM
        price = bs_engine.price(call, low_vol_model, market).price

        # Intrinsic value approximation
        S = 100
        K = 50
        r = 0.05
        T = 0.25
        intrinsic_approx = S - K * np.exp(-r * T)

        report.params(spot=S, strike=K, maturity=T, sigma=0.01)
        report.value("Deep ITM Call Price", price, expected=intrinsic_approx, unit="$")
        report.info("With very low volatility, option approaches discounted intrinsic")
        report.success("Deep ITM call converges to intrinsic value")

        np.testing.assert_allclose(price, intrinsic_approx, rtol=0.01)

    def test_option_price_positive(self, market_atm, gbm_model, bs_engine):
        """
        Option prices must always be non-negative.
        """
        report.header("Option Price Positivity Test")
        report.info("Tests that all option prices are non-negative")
        report.info("Options are rights (not obligations), so their value cannot be negative")

        results = []
        for strike in [50, 80, 100, 120, 150]:
            for is_call in [True, False]:
                option = VanillaOption(strike=strike, maturity=0.25, is_call=is_call)
                price = bs_engine.price(option, gbm_model, market_atm).price
                opt_type = "Call" if is_call else "Put"
                results.append((opt_type, strike, price, "Pass" if price >= 0 else "FAIL"))
                assert price >= 0

        report.table(
            headers=["Type", "Strike", "Price", "Status"],
            rows=results,
            title="Positivity Check",
            precision=4
        )
        report.success("All option prices are non-negative")


# =============================================================================
# MODEL CONSISTENCY
# =============================================================================

class TestModelConsistency:
    """
    Test consistency between different pricing methods.
    """

    def test_bs_fft_consistency(self, market_atm, gbm_model, bs_engine, fft_engine):
        """
        BS analytical and FFT should give same prices for GBM model.
        """
        report.header("BS vs FFT Consistency (GBM Model)")
        report.info("Tests that BS analytical and FFT engines produce identical prices for GBM")
        report.info("FFT uses characteristic function - for GBM must match closed-form BS")

        results = []
        for strike in [90, 100, 110]:
            call = VanillaOption(strike=strike, maturity=0.5, is_call=True)

            bs_price = bs_engine.price(call, gbm_model, market_atm).price
            fft_price = fft_engine.price(call, gbm_model, market_atm).price
            diff = abs(fft_price - bs_price)
            rel_diff = diff / bs_price * 100

            results.append((strike, bs_price, fft_price, diff, f"{rel_diff:.4f}%"))
            np.testing.assert_allclose(bs_price, fft_price, rtol=1e-4, atol=0.01)

        report.table(
            headers=["Strike", "BS Price", "FFT Price", "Abs Diff", "Rel Diff"],
            rows=results,
            title="BS vs FFT Comparison",
            precision=4
        )
        report.success("BS and FFT prices are consistent for GBM model")

    def test_bs_mc_consistency(self, market_atm, gbm_model, bs_engine):
        """
        BS analytical and Monte Carlo should converge for GBM model.
        """
        report.header("BS vs Monte Carlo Consistency")
        report.info("Tests that Monte Carlo converges to BS analytical price for GBM")
        report.info("Validates MC simulation correctness with large number of paths")

        mc_engine = MonteCarloEngine(n_paths=100000, seed=42)

        call = VanillaOption(strike=100, maturity=0.5, is_call=True)

        bs_price = bs_engine.price(call, gbm_model, market_atm).price
        mc_result = mc_engine.price(call, gbm_model, market_atm)

        report.params(n_paths=100000, strike=100, maturity=0.5)
        report.value("BS Analytical Price", bs_price, unit="$")
        report.value("Monte Carlo Price", mc_result.price, expected=bs_price, unit="$")
        if hasattr(mc_result, 'std_error'):
            report.value("MC Std Error", mc_result.std_error, unit="$")
        report.success("Monte Carlo converges to BS analytical price")

        # MC should be within 2% of analytical
        np.testing.assert_allclose(mc_result.price, bs_price, rtol=0.02)

    def test_heston_reduces_to_bs(self, market_atm, fft_engine):
        """
        Heston model with zero vol-of-vol should reduce to Black-Scholes.
        """
        report.header("Heston Reduces to BS (Zero Vol-of-Vol)")
        report.info("Tests that Heston with xi~0 degenerates to Black-Scholes")
        report.info("With no stochastic volatility, Heston becomes constant-vol GBM")

        sigma = 0.20
        gbm = GBMModel(sigma=sigma)

        # Heston with xi=0 (no stochastic vol)
        heston_degenerate = HestonModel(
            v0=sigma**2,
            kappa=1.0,
            theta=sigma**2,
            xi=0.001,  # Very small vol-of-vol
            rho=0.0
        )

        bs_engine = BSAnalyticEngine()

        call = VanillaOption(strike=100, maturity=0.5, is_call=True)

        bs_price = bs_engine.price(call, gbm, market_atm).price
        heston_price = fft_engine.price(call, heston_degenerate, market_atm).price

        report.params(sigma=sigma, xi=0.001, v0=sigma**2, theta=sigma**2)
        report.value("BS Price", bs_price, unit="$")
        report.value("Heston Price (xi~0)", heston_price, expected=bs_price, unit="$")
        report.info("With vol-of-vol ~0, Heston degenerates to GBM/BS")
        report.success("Heston reduces to Black-Scholes when xi -> 0")

        np.testing.assert_allclose(heston_price, bs_price, rtol=0.01)


# =============================================================================
# MONOTONICITY
# =============================================================================

class TestMonotonicity:
    """
    Test monotonicity properties of option prices.
    """

    def test_call_price_decreasing_in_strike(self, market_atm, gbm_model, bs_engine):
        """
        Call price decreases as strike increases.
        """
        report.header("Call Price Monotonicity in Strike")
        report.info("Tests that call prices decrease strictly as strike increases")
        report.info("Higher strike = lower probability of exercise = lower call value")

        strikes = [80, 90, 100, 110, 120]
        prices = []

        for K in strikes:
            call = VanillaOption(strike=K, maturity=0.25, is_call=True)
            prices.append(bs_engine.price(call, gbm_model, market_atm).price)

        results = list(zip(strikes, prices))
        report.table(
            headers=["Strike", "Call Price"],
            rows=results,
            title="Call Prices by Strike",
            precision=4
        )

        # Check strictly decreasing
        for i in range(len(prices) - 1):
            report.info(f"C(K={strikes[i]}) = {prices[i]:.4f} > C(K={strikes[i+1]}) = {prices[i+1]:.4f}")
            assert prices[i] > prices[i + 1]

        report.success("Call prices strictly decrease as strike increases")

    def test_put_price_increasing_in_strike(self, market_atm, gbm_model, bs_engine):
        """
        Put price increases as strike increases.
        """
        report.header("Put Price Monotonicity in Strike")
        report.info("Tests that put prices increase strictly as strike increases")
        report.info("Higher strike = higher payoff potential = higher put value")

        strikes = [80, 90, 100, 110, 120]
        prices = []

        for K in strikes:
            put = VanillaOption(strike=K, maturity=0.25, is_call=False)
            prices.append(bs_engine.price(put, gbm_model, market_atm).price)

        results = list(zip(strikes, prices))
        report.table(
            headers=["Strike", "Put Price"],
            rows=results,
            title="Put Prices by Strike",
            precision=4
        )

        # Check strictly increasing
        for i in range(len(prices) - 1):
            report.info(f"P(K={strikes[i]}) = {prices[i]:.4f} < P(K={strikes[i+1]}) = {prices[i+1]:.4f}")
            assert prices[i] < prices[i + 1]

        report.success("Put prices strictly increase as strike increases")

    def test_option_price_increasing_in_maturity(self, market_atm, gbm_model, bs_engine):
        """
        European option prices generally increase with maturity (for calls with no dividends).
        """
        report.header("Option Price Monotonicity in Maturity")
        report.info("Tests that European call prices increase with time to maturity")
        report.info("More time = more optionality value (for calls without dividends)")

        maturities = [0.1, 0.25, 0.5, 1.0]
        prices = []

        for T in maturities:
            call = VanillaOption(strike=100, maturity=T, is_call=True)
            prices.append(bs_engine.price(call, gbm_model, market_atm).price)

        results = list(zip(maturities, prices))
        report.table(
            headers=["Maturity", "Call Price"],
            rows=results,
            title="Call Prices by Maturity",
            precision=4
        )

        # Check increasing
        for i in range(len(prices) - 1):
            report.info(f"C(T={maturities[i]}) = {prices[i]:.4f} <= C(T={maturities[i+1]}) = {prices[i+1]:.4f}")
            assert prices[i] <= prices[i + 1]

        report.success("Call prices increase with maturity")

    def test_option_price_increasing_in_volatility(self, market_atm, bs_engine):
        """
        Option prices increase with volatility.
        """
        report.header("Option Price Monotonicity in Volatility")
        report.info("Tests that option prices increase strictly with volatility")
        report.info("Higher volatility = more uncertainty = higher option value (positive vega)")

        vols = [0.10, 0.20, 0.30, 0.40]
        prices = []

        call = VanillaOption(strike=100, maturity=0.25, is_call=True)

        for sigma in vols:
            model = GBMModel(sigma=sigma)
            prices.append(bs_engine.price(call, model, market_atm).price)

        results = list(zip(vols, prices))
        report.table(
            headers=["Volatility", "Call Price"],
            rows=results,
            title="Call Prices by Volatility",
            precision=4
        )

        # Check strictly increasing
        for i in range(len(prices) - 1):
            report.info(f"C(sigma={vols[i]}) = {prices[i]:.4f} < C(sigma={vols[i+1]}) = {prices[i+1]:.4f}")
            assert prices[i] < prices[i + 1]

        report.success("Option prices strictly increase with volatility (positive vega)")


# =============================================================================
# CONVEXITY
# =============================================================================

class TestConvexity:
    """
    Test convexity properties of option prices.
    """

    def test_call_convex_in_strike(self, market_atm, gbm_model, bs_engine):
        """
        Call price is convex in strike: C(K2) <= (C(K1) + C(K3))/2 for K1 < K2 < K3.
        """
        report.header("Call Convexity in Strike")
        report.info("Tests that call prices are convex in strike: C(K2) <= (C(K1)+C(K3))/2")
        report.info("Convexity ensures no arbitrage via butterfly spreads")

        K1, K2, K3 = 90, 100, 110

        c1 = bs_engine.price(VanillaOption(strike=K1, maturity=0.25, is_call=True), gbm_model, market_atm).price
        c2 = bs_engine.price(VanillaOption(strike=K2, maturity=0.25, is_call=True), gbm_model, market_atm).price
        c3 = bs_engine.price(VanillaOption(strike=K3, maturity=0.25, is_call=True), gbm_model, market_atm).price

        midpoint_price = (c1 + c3) / 2

        report.params(K1=K1, K2=K2, K3=K3)
        report.value(f"C(K1={K1})", c1, unit="$")
        report.value(f"C(K2={K2})", c2, unit="$")
        report.value(f"C(K3={K3})", c3, unit="$")
        report.value("Midpoint (C1+C3)/2", midpoint_price, unit="$")
        report.info(f"Convexity check: C(K2) = {c2:.4f} <= midpoint = {midpoint_price:.4f}")
        report.success("Call prices are convex in strike")

        # Convexity: midpoint should be above the line
        assert c2 <= (c1 + c3) / 2 + 1e-10

    def test_butterfly_spread_positive(self, market_atm, gbm_model, bs_engine):
        """
        Butterfly spread value must be positive (proof of convexity).
        Butterfly = C(K1) - 2*C(K2) + C(K3)
        """
        report.header("Butterfly Spread Positivity (Convexity Proof)")
        report.info("Tests that butterfly spread value is non-negative")
        report.info("Butterfly = C(K1) - 2*C(K2) + C(K3) >= 0 is equivalent to convexity")

        K1, K2, K3 = 90, 100, 110

        c1 = bs_engine.price(VanillaOption(strike=K1, maturity=0.25, is_call=True), gbm_model, market_atm).price
        c2 = bs_engine.price(VanillaOption(strike=K2, maturity=0.25, is_call=True), gbm_model, market_atm).price
        c3 = bs_engine.price(VanillaOption(strike=K3, maturity=0.25, is_call=True), gbm_model, market_atm).price

        butterfly = c1 - 2 * c2 + c3

        report.params(K1=K1, K2=K2, K3=K3)
        report.value(f"C(K1={K1})", c1, unit="$")
        report.value(f"2*C(K2={K2})", 2 * c2, unit="$")
        report.value(f"C(K3={K3})", c3, unit="$")
        report.value("Butterfly Value", butterfly, unit="$")
        report.info("Butterfly = C(K1) - 2*C(K2) + C(K3) >= 0 implies convexity")
        report.success(f"Butterfly spread value = {butterfly:.4f} >= 0")

        assert butterfly >= -1e-10


# =============================================================================
# LIMITING CASES
# =============================================================================

class TestLimitingCases:
    """
    Test behavior in limiting cases.
    """

    def test_zero_time_to_expiry(self, market_atm, gbm_model, bs_engine):
        """
        At expiry, option value equals intrinsic value.
        """
        report.header("Zero Time to Expiry Test")
        report.info("Tests that option value equals intrinsic value at expiry")
        report.info("With no time remaining, option is worth exactly max(S-K, 0) or max(K-S, 0)")

        # Very small time to expiry
        call = VanillaOption(strike=95, maturity=1e-6, is_call=True)
        put = VanillaOption(strike=105, maturity=1e-6, is_call=False)

        call_price = bs_engine.price(call, gbm_model, market_atm).price
        put_price = bs_engine.price(put, gbm_model, market_atm).price

        S = market_atm.spot

        # Intrinsic values
        call_intrinsic = max(S - 95, 0)
        put_intrinsic = max(105 - S, 0)

        report.params(spot=S, T="1e-6 (near expiry)")
        report.value("ITM Call Price (K=95)", call_price, expected=call_intrinsic, unit="$")
        report.value("ITM Put Price (K=105)", put_price, expected=put_intrinsic, unit="$")
        report.info("At expiry, option value = intrinsic value")
        report.success("Options converge to intrinsic value at expiry")

        np.testing.assert_allclose(call_price, call_intrinsic, atol=0.01)
        np.testing.assert_allclose(put_price, put_intrinsic, atol=0.01)

    def test_zero_volatility(self, market_atm, bs_engine):
        """
        With zero volatility, option price equals discounted intrinsic.
        """
        report.header("Zero Volatility Test")
        report.info("Tests that option price equals forward intrinsic with zero volatility")
        report.info("No uncertainty means deterministic outcome - option = discounted payoff")

        model = GBMModel(sigma=0.001)  # Very low vol

        # ITM call
        call = VanillaOption(strike=90, maturity=0.25, is_call=True)
        price = bs_engine.price(call, model, market_atm).price

        S = market_atm.spot
        K = 90
        r = market_atm.rate
        T = 0.25

        # Forward intrinsic
        expected = max(S - K * np.exp(-r * T), 0)

        report.params(spot=S, strike=K, maturity=T, sigma=0.001)
        report.value("ITM Call Price", price, expected=expected, unit="$")
        report.info("With zero vol, option = discounted intrinsic value")
        report.success("Option converges to forward intrinsic with zero vol")

        np.testing.assert_allclose(price, expected, rtol=0.01)

    def test_very_high_volatility(self, market_atm, bs_engine):
        """
        With very high volatility, ATM call approaches spot price.
        """
        report.header("Very High Volatility Test")
        report.info("Tests that ATM call approaches spot price with very high volatility")
        report.info("In the limit sigma -> infinity, call approaches its upper bound S")

        model = GBMModel(sigma=5.0)  # Extremely high vol

        call = VanillaOption(strike=100, maturity=1.0, is_call=True)
        price = bs_engine.price(call, model, market_atm).price

        S = market_atm.spot

        report.params(spot=S, strike=100, maturity=1.0, sigma=5.0)
        report.value("ATM Call Price", price, unit="$")
        report.value("Spot Price (upper bound)", S, unit="$")
        report.value("90% of Spot", 0.9 * S, unit="$")
        report.info(f"Price {price:.2f} should be > 90% of spot ({0.9*S:.2f}) and <= spot ({S})")
        report.success(f"High-vol call approaches spot: {price:.2f} in range [{0.9*S:.2f}, {S}]")

        # Should approach spot (upper bound)
        assert price > 0.9 * S
        assert price <= S


# =============================================================================
# CHARACTERISTIC FUNCTION PROPERTIES
# =============================================================================

class TestCharacteristicFunction:
    """
    Test properties of characteristic functions.
    """

    def test_cf_at_zero(self, market_atm):
        """
        Characteristic function at u=0 equals 1.
        """
        report.header("Characteristic Function at u=0")
        report.info("Tests that characteristic function equals 1 at u=0 for all models")
        report.info("phi(0) = E[exp(0)] = 1 is fundamental property of any CF")

        models = [
            ("GBM", GBMModel(sigma=0.20)),
            ("Heston", HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)),
            ("Bates", BatesModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                      lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)),
        ]

        S = market_atm.spot
        T = 0.5
        r = market_atm.rate
        q = market_atm.dividend_yield

        results = []
        for name, model in models:
            cf = model.characteristic_function(0.0, S, T, r, q)
            cf_abs = np.abs(cf)
            results.append((name, cf.real, cf.imag, cf_abs))
            np.testing.assert_allclose(cf_abs, 1.0, rtol=1e-10)

        report.table(
            headers=["Model", "Re(CF(0))", "Im(CF(0))", "|CF(0)|"],
            rows=results,
            title="CF(0) for each model",
            precision=6
        )
        report.info("Characteristic function property: phi(0) = 1")
        report.success("All models satisfy |CF(0)| = 1")

    def test_cf_conjugate_symmetry(self, market_atm):
        """
        For real-valued processes: phi(-u) = conj(phi(u))
        """
        report.header("Characteristic Function Conjugate Symmetry")
        report.info("Tests that phi(-u) = conj(phi(u)) for real-valued processes")
        report.info("Conjugate symmetry ensures CF corresponds to real-valued random variable")

        gbm = GBMModel(sigma=0.20)

        S = market_atm.spot
        T = 0.5
        r = market_atm.rate
        q = market_atm.dividend_yield

        u_values = [1.0, 2.0, 5.0, 10.0]

        results = []
        for u in u_values:
            cf_pos = gbm.characteristic_function(u, S, T, r, q)
            cf_neg = gbm.characteristic_function(-u, S, T, r, q)
            cf_conj = np.conj(cf_pos)

            diff = np.abs(cf_neg - cf_conj)
            results.append((u, f"{cf_pos.real:.6f}+{cf_pos.imag:.6f}i",
                           f"{cf_neg.real:.6f}+{cf_neg.imag:.6f}i",
                           f"{cf_conj.real:.6f}+{cf_conj.imag:.6f}i",
                           f"{diff:.2e}"))

            np.testing.assert_allclose(cf_neg, np.conj(cf_pos), rtol=1e-10)

        report.table(
            headers=["u", "phi(u)", "phi(-u)", "conj(phi(u))", "Diff"],
            rows=results,
            title="Conjugate Symmetry Test"
        )
        report.info("Property: phi(-u) = conj(phi(u)) for real-valued processes")
        report.success("Conjugate symmetry holds for all test values")


# =============================================================================
# FFT PUT-CALL PARITY WITH DIVIDENDS (Gap 6)
# =============================================================================

class TestFFTPutCallParityWithDividends:
    """Gap 6: FFT put-call parity with non-zero dividend yield."""

    @pytest.mark.parametrize("strike", [90, 100, 110])
    def test_fft_put_call_parity_with_dividends(self, strike, market_with_dividend, heston_model, fft_engine):
        """Test put-call parity holds for FFT pricing with dividends."""
        report.header(f"FFT Put-Call Parity with Dividends (K={strike})")
        report.info("Tests C - P = S*exp(-qT) - K*exp(-rT) with q > 0")

        T = 0.5
        call = VanillaOption(strike=strike, maturity=T, is_call=True)
        put = VanillaOption(strike=strike, maturity=T, is_call=False)

        call_price = fft_engine.price(call, heston_model, market_with_dividend).price
        put_price = fft_engine.price(put, heston_model, market_with_dividend).price

        S = market_with_dividend.spot
        K = strike
        r = market_with_dividend.rate
        q = market_with_dividend.dividend_yield

        lhs = call_price - put_price
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)

        report.params(strike=K, maturity=T, spot=S, rate=r, dividend=q)
        report.value("Call Price (FFT)", call_price, unit="$")
        report.value("Put Price (FFT)", put_price, unit="$")
        report.value("C - P (LHS)", lhs, expected=rhs, unit="$")
        report.success("Put-Call Parity holds with FFT/Heston + dividends")

        np.testing.assert_allclose(lhs, rhs, rtol=1e-3, atol=0.01)
