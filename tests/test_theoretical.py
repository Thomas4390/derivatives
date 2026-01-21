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
from scipy.stats import norm

from backend.instruments.options import VanillaOption
from backend.models.gbm import GBMModel
from backend.models.heston import HestonModel
from backend.models.bates import BatesModel
from backend.engines import BSAnalyticEngine, FFTEngine, MonteCarloEngine
from backend.core.market import MarketEnvironment
from backend.greeks import bs_all_greeks


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

        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)

    @pytest.mark.parametrize("strike", [90, 100, 110])
    def test_put_call_parity_fft(self, strike, market_atm, heston_model, fft_engine):
        """Test put-call parity holds for FFT pricing with Heston model."""
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

        # FFT has numerical errors, use looser tolerance
        np.testing.assert_allclose(lhs, rhs, rtol=1e-4, atol=0.01)

    def test_put_call_parity_with_dividends(self, market_with_dividend, gbm_model, bs_engine):
        """Test put-call parity with continuous dividend yield."""
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
        call = VanillaOption(strike=100, maturity=0.25, is_call=True)
        put = VanillaOption(strike=100, maturity=0.25, is_call=False)

        call_greeks = bs_engine.greeks(call, gbm_model, market_atm)
        put_greeks = bs_engine.greeks(put, gbm_model, market_atm)

        q = market_atm.dividend_yield
        T = 0.25
        expected_diff = np.exp(-q * T)

        actual_diff = call_greeks.delta - put_greeks.delta
        np.testing.assert_allclose(actual_diff, expected_diff, rtol=1e-6)

    def test_gamma_same_for_call_put(self, market_atm, gbm_model, bs_engine):
        """
        Gamma is the same for call and put with same strike/maturity.
        """
        call = VanillaOption(strike=100, maturity=0.25, is_call=True)
        put = VanillaOption(strike=100, maturity=0.25, is_call=False)

        call_greeks = bs_engine.greeks(call, gbm_model, market_atm)
        put_greeks = bs_engine.greeks(put, gbm_model, market_atm)

        np.testing.assert_allclose(call_greeks.gamma, put_greeks.gamma, rtol=1e-10)

    def test_vega_same_for_call_put(self, market_atm, gbm_model, bs_engine):
        """
        Vega is the same for call and put with same strike/maturity.
        """
        call = VanillaOption(strike=100, maturity=0.25, is_call=True)
        put = VanillaOption(strike=100, maturity=0.25, is_call=False)

        call_greeks = bs_engine.greeks(call, gbm_model, market_atm)
        put_greeks = bs_engine.greeks(put, gbm_model, market_atm)

        np.testing.assert_allclose(call_greeks.vega, put_greeks.vega, rtol=1e-10)

    def test_theta_rho_parity(self, market_atm, gbm_model, bs_engine):
        """
        Relationship: theta_call - theta_put = r*K*exp(-rT) - q*S*exp(-qT)
        """
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
        expected_diff = -r * K * np.exp(-r * T) + q * S * np.exp(-q * T)

        np.testing.assert_allclose(theta_diff, expected_diff, rtol=1e-4)

    def test_atm_delta_approximately_half(self, market_atm, gbm_model, bs_engine):
        """
        ATM call delta is approximately 0.5 (slightly higher due to drift).
        """
        # Use longer maturity for clearer effect
        call = VanillaOption(strike=100, maturity=1.0, is_call=True)
        greeks = bs_engine.greeks(call, gbm_model, market_atm)

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
        for strike in [80, 100, 120]:
            call = VanillaOption(strike=strike, maturity=0.25, is_call=True)
            price = bs_engine.price(call, gbm_model, market_atm).price

            S = market_atm.spot
            K = strike
            r = market_atm.rate
            q = market_atm.dividend_yield
            T = 0.25

            lower_bound = max(0, S * np.exp(-q * T) - K * np.exp(-r * T))
            assert price >= lower_bound - 1e-10

    def test_put_lower_bound(self, market_atm, gbm_model, bs_engine):
        """
        Put price >= max(0, K*exp(-rT) - S*exp(-qT))
        """
        for strike in [80, 100, 120]:
            put = VanillaOption(strike=strike, maturity=0.25, is_call=False)
            price = bs_engine.price(put, gbm_model, market_atm).price

            S = market_atm.spot
            K = strike
            r = market_atm.rate
            q = market_atm.dividend_yield
            T = 0.25

            lower_bound = max(0, K * np.exp(-r * T) - S * np.exp(-q * T))
            assert price >= lower_bound - 1e-10

    def test_call_upper_bound(self, market_atm, gbm_model, bs_engine):
        """
        Call price <= S*exp(-qT)
        """
        call = VanillaOption(strike=100, maturity=0.25, is_call=True)
        price = bs_engine.price(call, gbm_model, market_atm).price

        S = market_atm.spot
        q = market_atm.dividend_yield
        T = 0.25

        upper_bound = S * np.exp(-q * T)
        assert price <= upper_bound + 1e-10

    def test_put_upper_bound(self, market_atm, gbm_model, bs_engine):
        """
        Put price <= K*exp(-rT)
        """
        put = VanillaOption(strike=100, maturity=0.25, is_call=False)
        price = bs_engine.price(put, gbm_model, market_atm).price

        K = 100
        r = market_atm.rate
        T = 0.25

        upper_bound = K * np.exp(-r * T)
        assert price <= upper_bound + 1e-10

    def test_deep_itm_call_approaches_intrinsic(self, gbm_model, bs_engine):
        """
        Deep ITM call approaches discounted intrinsic value as vol -> 0.
        """
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

        np.testing.assert_allclose(price, intrinsic_approx, rtol=0.01)

    def test_option_price_positive(self, market_atm, gbm_model, bs_engine):
        """
        Option prices must always be non-negative.
        """
        for strike in [50, 80, 100, 120, 150]:
            for is_call in [True, False]:
                option = VanillaOption(strike=strike, maturity=0.25, is_call=is_call)
                price = bs_engine.price(option, gbm_model, market_atm).price
                assert price >= 0


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
        for strike in [90, 100, 110]:
            call = VanillaOption(strike=strike, maturity=0.5, is_call=True)

            bs_price = bs_engine.price(call, gbm_model, market_atm).price
            fft_price = fft_engine.price(call, gbm_model, market_atm).price

            np.testing.assert_allclose(bs_price, fft_price, rtol=1e-4, atol=0.01)

    def test_bs_mc_consistency(self, market_atm, gbm_model, bs_engine):
        """
        BS analytical and Monte Carlo should converge for GBM model.
        """
        mc_engine = MonteCarloEngine(n_paths=100000, seed=42)

        call = VanillaOption(strike=100, maturity=0.5, is_call=True)

        bs_price = bs_engine.price(call, gbm_model, market_atm).price
        mc_result = mc_engine.price(call, gbm_model, market_atm)

        # MC should be within 2% of analytical
        np.testing.assert_allclose(mc_result.price, bs_price, rtol=0.02)

    def test_heston_reduces_to_bs(self, market_atm, fft_engine):
        """
        Heston model with zero vol-of-vol should reduce to Black-Scholes.
        """
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
        strikes = [80, 90, 100, 110, 120]
        prices = []

        for K in strikes:
            call = VanillaOption(strike=K, maturity=0.25, is_call=True)
            prices.append(bs_engine.price(call, gbm_model, market_atm).price)

        # Check strictly decreasing
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1]

    def test_put_price_increasing_in_strike(self, market_atm, gbm_model, bs_engine):
        """
        Put price increases as strike increases.
        """
        strikes = [80, 90, 100, 110, 120]
        prices = []

        for K in strikes:
            put = VanillaOption(strike=K, maturity=0.25, is_call=False)
            prices.append(bs_engine.price(put, gbm_model, market_atm).price)

        # Check strictly increasing
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1]

    def test_option_price_increasing_in_maturity(self, market_atm, gbm_model, bs_engine):
        """
        European option prices generally increase with maturity (for calls with no dividends).
        """
        maturities = [0.1, 0.25, 0.5, 1.0]
        prices = []

        for T in maturities:
            call = VanillaOption(strike=100, maturity=T, is_call=True)
            prices.append(bs_engine.price(call, gbm_model, market_atm).price)

        # Check increasing
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1]

    def test_option_price_increasing_in_volatility(self, market_atm, bs_engine):
        """
        Option prices increase with volatility.
        """
        vols = [0.10, 0.20, 0.30, 0.40]
        prices = []

        call = VanillaOption(strike=100, maturity=0.25, is_call=True)

        for sigma in vols:
            model = GBMModel(sigma=sigma)
            prices.append(bs_engine.price(call, model, market_atm).price)

        # Check strictly increasing
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1]


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
        K1, K2, K3 = 90, 100, 110

        c1 = bs_engine.price(VanillaOption(strike=K1, maturity=0.25, is_call=True), gbm_model, market_atm).price
        c2 = bs_engine.price(VanillaOption(strike=K2, maturity=0.25, is_call=True), gbm_model, market_atm).price
        c3 = bs_engine.price(VanillaOption(strike=K3, maturity=0.25, is_call=True), gbm_model, market_atm).price

        # Convexity: midpoint should be above the line
        assert c2 <= (c1 + c3) / 2 + 1e-10

    def test_butterfly_spread_positive(self, market_atm, gbm_model, bs_engine):
        """
        Butterfly spread value must be positive (proof of convexity).
        Butterfly = C(K1) - 2*C(K2) + C(K3)
        """
        K1, K2, K3 = 90, 100, 110

        c1 = bs_engine.price(VanillaOption(strike=K1, maturity=0.25, is_call=True), gbm_model, market_atm).price
        c2 = bs_engine.price(VanillaOption(strike=K2, maturity=0.25, is_call=True), gbm_model, market_atm).price
        c3 = bs_engine.price(VanillaOption(strike=K3, maturity=0.25, is_call=True), gbm_model, market_atm).price

        butterfly = c1 - 2 * c2 + c3
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
        # Very small time to expiry
        call = VanillaOption(strike=95, maturity=1e-6, is_call=True)
        put = VanillaOption(strike=105, maturity=1e-6, is_call=False)

        call_price = bs_engine.price(call, gbm_model, market_atm).price
        put_price = bs_engine.price(put, gbm_model, market_atm).price

        S = market_atm.spot

        # Intrinsic values
        call_intrinsic = max(S - 95, 0)
        put_intrinsic = max(105 - S, 0)

        np.testing.assert_allclose(call_price, call_intrinsic, atol=0.01)
        np.testing.assert_allclose(put_price, put_intrinsic, atol=0.01)

    def test_zero_volatility(self, market_atm, bs_engine):
        """
        With zero volatility, option price equals discounted intrinsic.
        """
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
        np.testing.assert_allclose(price, expected, rtol=0.01)

    def test_very_high_volatility(self, market_atm, bs_engine):
        """
        With very high volatility, ATM call approaches spot price.
        """
        model = GBMModel(sigma=5.0)  # Extremely high vol

        call = VanillaOption(strike=100, maturity=1.0, is_call=True)
        price = bs_engine.price(call, model, market_atm).price

        S = market_atm.spot

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
        models = [
            GBMModel(sigma=0.20),
            HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7),
            BatesModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                      lambda_j=0.5, mu_j=-0.1, sigma_j=0.2),
        ]

        S = market_atm.spot
        T = 0.5
        r = market_atm.rate
        q = market_atm.dividend_yield

        for model in models:
            cf = model.characteristic_function(0.0, S, T, r, q)
            np.testing.assert_allclose(np.abs(cf), 1.0, rtol=1e-10)

    def test_cf_conjugate_symmetry(self, market_atm):
        """
        For real-valued processes: phi(-u) = conj(phi(u))
        """
        gbm = GBMModel(sigma=0.20)

        S = market_atm.spot
        T = 0.5
        r = market_atm.rate
        q = market_atm.dividend_yield

        u_values = [1.0, 2.0, 5.0, 10.0]

        for u in u_values:
            cf_pos = gbm.characteristic_function(u, S, T, r, q)
            cf_neg = gbm.characteristic_function(-u, S, T, r, q)

            np.testing.assert_allclose(cf_neg, np.conj(cf_pos), rtol=1e-10)
