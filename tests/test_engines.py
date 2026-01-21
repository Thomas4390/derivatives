"""
Pricing Engines Tests
=====================

Tests for all pricing engines:
- BSAnalyticEngine: Black-Scholes analytical pricing
- FFTEngine: Carr-Madan FFT pricing
- MonteCarloEngine: Monte Carlo simulation pricing

Author: Thomas
Created: 2025
"""

import pytest
import numpy as np

from backend.instruments.options import VanillaOption
from backend.models.gbm import GBMModel
from backend.models.heston import HestonModel
from backend.models.bates import BatesModel
from backend.models.merton import MertonModel
from backend.engines import BSAnalyticEngine, FFTEngine, MonteCarloEngine
from backend.engines.fourier.carr_madan import FFTConfig
from backend.core.market import MarketEnvironment

# Import constants from conftest
from tests.conftest import BS_BENCHMARK, PRICE_ATOL, PRICE_RTOL, MC_RTOL


# =============================================================================
# BLACK-SCHOLES ANALYTICAL ENGINE TESTS
# =============================================================================

class TestBSAnalyticEngine:
    """Tests for Black-Scholes analytical pricing engine."""

    def test_atm_call_price(self, market_atm, gbm_model, bs_engine, call_atm):
        """Test ATM call pricing against benchmark."""
        result = bs_engine.price(call_atm, gbm_model, market_atm)

        np.testing.assert_allclose(
            result.price, BS_BENCHMARK["call_price"],
            rtol=PRICE_RTOL, atol=PRICE_ATOL
        )

    def test_atm_put_price(self, market_atm, gbm_model, bs_engine, put_atm):
        """Test ATM put pricing against benchmark."""
        result = bs_engine.price(put_atm, gbm_model, market_atm)

        np.testing.assert_allclose(
            result.price, BS_BENCHMARK["put_price"],
            rtol=PRICE_RTOL, atol=PRICE_ATOL
        )

    def test_greeks_first_order(self, market_atm, gbm_model, bs_engine, call_atm):
        """Test first-order Greeks against benchmark."""
        greeks = bs_engine.greeks(call_atm, gbm_model, market_atm)

        np.testing.assert_allclose(greeks.delta, BS_BENCHMARK["delta_call"], rtol=0.01)
        np.testing.assert_allclose(greeks.gamma, BS_BENCHMARK["gamma"], rtol=0.01)
        # Vega per 100% volatility move
        np.testing.assert_allclose(greeks.vega, BS_BENCHMARK["vega"], rtol=0.01)

    def test_can_price_gbm_only(self, market_atm, bs_engine, call_atm):
        """BSAnalyticEngine should only work with GBM model."""
        gbm = GBMModel(sigma=0.20)
        heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

        assert bs_engine.can_price(call_atm, gbm) is True
        assert bs_engine.can_price(call_atm, heston) is False

    def test_implied_volatility(self, market_atm, gbm_model, bs_engine):
        """Test implied volatility calculation."""
        call = VanillaOption(strike=100, maturity=0.25, is_call=True)

        # Get the price first
        price = bs_engine.price(call, gbm_model, market_atm).price

        # Calculate IV from price
        iv = bs_engine.implied_volatility(price, call, market_atm)

        # IV should match the original volatility
        np.testing.assert_allclose(iv, 0.20, rtol=1e-4)

    def test_implied_volatility_different_strikes(self, market_atm, gbm_model, bs_engine):
        """Test IV calculation for different moneyness levels."""
        for strike in [90, 100, 110]:
            call = VanillaOption(strike=strike, maturity=0.25, is_call=True)
            price = bs_engine.price(call, gbm_model, market_atm).price
            iv = bs_engine.implied_volatility(price, call, market_atm)

            np.testing.assert_allclose(iv, 0.20, rtol=1e-3)

    @pytest.mark.parametrize("spot", [80, 90, 100, 110, 120])
    def test_price_across_spots(self, gbm_model, bs_engine, spot):
        """Test pricing across different spot prices."""
        market = MarketEnvironment(spot=spot, rate=0.05, dividend_yield=0.0)
        call = VanillaOption(strike=100, maturity=0.25, is_call=True)

        result = bs_engine.price(call, gbm_model, market)

        assert result.price >= 0
        assert result.engine == "BSAnalyticEngine"

    def test_price_with_dividends(self, market_with_dividend, gbm_model, bs_engine, call_atm):
        """Test pricing with continuous dividend yield."""
        result = bs_engine.price(call_atm, gbm_model, market_with_dividend)

        # Price should be lower with dividends
        result_no_div = bs_engine.price(call_atm, gbm_model, MarketEnvironment(spot=100, rate=0.05))
        assert result.price < result_no_div.price


# =============================================================================
# FFT ENGINE TESTS
# =============================================================================

class TestFFTEngine:
    """Tests for FFT pricing engine."""

    def test_gbm_matches_bs(self, market_atm, gbm_model, bs_engine, fft_engine, call_atm):
        """FFT with GBM should match BS analytical."""
        bs_price = bs_engine.price(call_atm, gbm_model, market_atm).price
        fft_price = fft_engine.price(call_atm, gbm_model, market_atm).price

        np.testing.assert_allclose(fft_price, bs_price, rtol=1e-4, atol=0.01)

    def test_heston_pricing(self, market_atm, heston_model, fft_engine, call_atm):
        """Test Heston model pricing via FFT."""
        result = fft_engine.price(call_atm, heston_model, market_atm)

        assert result.price > 0
        # Model name is the full name from model.name property
        assert "Heston" in result.model

    def test_bates_pricing(self, market_atm, bates_model, fft_engine, call_atm):
        """Test Bates model pricing via FFT."""
        result = fft_engine.price(call_atm, bates_model, market_atm)

        assert result.price > 0
        # Model name is the full name from model.name property
        assert "Bates" in result.model

    def test_merton_pricing(self, market_atm, merton_model, fft_engine, call_atm):
        """Test Merton jump-diffusion pricing via FFT."""
        result = fft_engine.price(call_atm, merton_model, market_atm)

        assert result.price > 0
        # Model name is the full name from model.name property
        assert "Merton" in result.model

    def test_price_strikes(self, market_atm, heston_model, fft_engine):
        """Test batch pricing of multiple strikes."""
        call = VanillaOption(strike=100, maturity=0.5, is_call=True)
        strikes = np.array([90, 95, 100, 105, 110], dtype=float)

        prices = fft_engine.price_strikes(call, heston_model, market_atm, strikes)

        assert len(prices) == len(strikes)
        assert all(p > 0 for p in prices)
        # Prices should be decreasing for calls
        assert all(prices[i] >= prices[i + 1] for i in range(len(prices) - 1))

    def test_price_surface(self, market_atm, heston_model, fft_engine):
        """Test pricing surface generation."""
        strikes = np.array([90, 100, 110], dtype=float)
        maturities = np.array([0.25, 0.5, 1.0])

        surface = fft_engine.price_surface(
            heston_model, market_atm, strikes, maturities, is_call=True
        )

        assert surface.shape == (3, 3)
        assert np.all(surface > 0)

    def test_fft_config(self, market_atm, gbm_model):
        """Test custom FFT configuration."""
        config = FFTConfig(alpha=1.5, n_fft=4096, eta=0.25)
        engine = FFTEngine(config=config)

        call = VanillaOption(strike=100, maturity=0.5, is_call=True)
        result = engine.price(call, gbm_model, market_atm)

        assert result.price > 0

    def test_put_pricing(self, market_atm, heston_model, fft_engine):
        """Test put option pricing via FFT."""
        put = VanillaOption(strike=100, maturity=0.5, is_call=False)
        result = fft_engine.price(put, heston_model, market_atm)

        assert result.price > 0

    def test_can_price(self, market_atm, fft_engine, call_atm):
        """Test can_price method."""
        gbm = GBMModel(sigma=0.20)
        heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

        # FFT should work with models that have characteristic functions
        assert fft_engine.can_price(call_atm, gbm) is True
        assert fft_engine.can_price(call_atm, heston) is True


# =============================================================================
# MONTE CARLO ENGINE TESTS
# =============================================================================

class TestMonteCarloEngine:
    """Tests for Monte Carlo pricing engine."""

    def test_gbm_converges_to_bs(self, market_atm, gbm_model, bs_engine, call_atm):
        """MC with GBM should converge to BS analytical."""
        mc_engine = MonteCarloEngine(n_paths=100000, seed=42)

        bs_price = bs_engine.price(call_atm, gbm_model, market_atm).price
        mc_result = mc_engine.price(call_atm, gbm_model, market_atm)

        np.testing.assert_allclose(mc_result.price, bs_price, rtol=MC_RTOL)

    def test_heston_pricing(self, market_atm, heston_model, call_atm):
        """Test Heston model pricing via MC."""
        mc_engine = MonteCarloEngine(n_paths=50000, seed=42)
        result = mc_engine.price(call_atm, heston_model, market_atm)

        assert result.price > 0

    def test_bates_pricing(self, market_atm, bates_model, call_atm):
        """Test Bates model pricing via MC."""
        mc_engine = MonteCarloEngine(n_paths=50000, seed=42)
        result = mc_engine.price(call_atm, bates_model, market_atm)

        assert result.price > 0

    def test_merton_pricing(self, market_atm, merton_model, call_atm):
        """Test Merton model pricing via MC."""
        mc_engine = MonteCarloEngine(n_paths=50000, seed=42)
        result = mc_engine.price(call_atm, merton_model, market_atm)

        assert result.price > 0

    def test_reproducibility(self, market_atm, gbm_model, call_atm):
        """Test that MC results are statistically consistent.

        Note: Exact reproducibility is not guaranteed with parallel numba code
        (@njit(parallel=True)) because thread scheduling is non-deterministic.
        We test that two runs produce results within expected MC variance.
        """
        mc1 = MonteCarloEngine(n_paths=10000, seed=123)
        mc2 = MonteCarloEngine(n_paths=10000, seed=456)

        result1 = mc1.price(call_atm, gbm_model, market_atm)
        result2 = mc2.price(call_atm, gbm_model, market_atm)

        # Results should be within a few standard errors of each other
        combined_se = np.sqrt(result1.error**2 + result2.error**2)
        np.testing.assert_allclose(result1.price, result2.price, atol=4*combined_se)

    def test_different_seeds_different_results(self, market_atm, gbm_model, call_atm):
        """Different seeds should give different results (within MC error)."""
        mc1 = MonteCarloEngine(n_paths=10000, seed=123)
        mc2 = MonteCarloEngine(n_paths=10000, seed=456)

        result1 = mc1.price(call_atm, gbm_model, market_atm)
        result2 = mc2.price(call_atm, gbm_model, market_atm)

        # Should be different but close
        assert result1.price != result2.price
        np.testing.assert_allclose(result1.price, result2.price, rtol=0.05)

    def test_price_strikes(self, market_atm, heston_model):
        """Test batch pricing of multiple strikes."""
        mc_engine = MonteCarloEngine(n_paths=50000, seed=42)

        call = VanillaOption(strike=100, maturity=0.5, is_call=True)
        strikes = np.array([90, 95, 100, 105, 110], dtype=float)

        # MonteCarloEngine.price_strikes returns (prices, std_errors) tuple
        prices, std_errors = mc_engine.price_strikes(call, heston_model, market_atm, strikes)

        assert len(prices) == len(strikes)
        assert len(std_errors) == len(strikes)
        assert all(p > 0 for p in prices)

    def test_put_pricing(self, market_atm, gbm_model, bs_engine):
        """Test put option pricing via MC."""
        mc_engine = MonteCarloEngine(n_paths=100000, seed=42)

        put = VanillaOption(strike=100, maturity=0.25, is_call=False)

        bs_price = bs_engine.price(put, gbm_model, market_atm).price
        mc_result = mc_engine.price(put, gbm_model, market_atm)

        np.testing.assert_allclose(mc_result.price, bs_price, rtol=MC_RTOL)


# =============================================================================
# ENGINE COMPARISON TESTS
# =============================================================================

class TestEngineComparison:
    """Cross-engine comparison tests."""

    @pytest.mark.parametrize("strike", [90, 100, 110])
    @pytest.mark.parametrize("maturity", [0.25, 0.5, 1.0])
    def test_all_engines_consistent_gbm(self, strike, maturity, market_atm, gbm_model):
        """All engines should give consistent prices for GBM."""
        call = VanillaOption(strike=strike, maturity=maturity, is_call=True)

        bs_engine = BSAnalyticEngine()
        fft_engine = FFTEngine()
        mc_engine = MonteCarloEngine(n_paths=100000, seed=42)

        bs_price = bs_engine.price(call, gbm_model, market_atm).price
        fft_price = fft_engine.price(call, gbm_model, market_atm).price
        mc_price = mc_engine.price(call, gbm_model, market_atm).price

        # FFT should be very close to BS (slightly higher tolerance for OTM options)
        np.testing.assert_allclose(fft_price, bs_price, rtol=2e-3)

        # MC should be within 2%
        np.testing.assert_allclose(mc_price, bs_price, rtol=0.02)

    def test_fft_mc_consistent_heston(self, market_atm, heston_model, fft_engine):
        """FFT and MC should be consistent for Heston."""
        mc_engine = MonteCarloEngine(n_paths=100000, seed=42)

        call = VanillaOption(strike=100, maturity=0.5, is_call=True)

        fft_price = fft_engine.price(call, heston_model, market_atm).price
        mc_price = mc_engine.price(call, heston_model, market_atm).price

        # Should be within 3% (MC has higher variance for stochastic vol)
        np.testing.assert_allclose(mc_price, fft_price, rtol=0.03)


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_short_maturity(self, market_atm, gbm_model, bs_engine):
        """Test pricing with very short time to expiry."""
        call = VanillaOption(strike=100, maturity=0.001, is_call=True)
        result = bs_engine.price(call, gbm_model, market_atm)

        # Should be close to intrinsic (S - K) = 0 for ATM
        assert abs(result.price) < 1.0

    def test_very_long_maturity(self, market_atm, gbm_model, bs_engine):
        """Test pricing with long time to expiry."""
        call = VanillaOption(strike=100, maturity=10.0, is_call=True)
        result = bs_engine.price(call, gbm_model, market_atm)

        assert result.price > 0
        assert result.price < market_atm.spot  # Upper bound

    def test_deep_itm_call(self, market_atm, gbm_model, bs_engine):
        """Test deep ITM call."""
        call = VanillaOption(strike=50, maturity=0.25, is_call=True)
        result = bs_engine.price(call, gbm_model, market_atm)

        # Should be close to S - K*exp(-rT) ≈ 100 - 50*0.9876 ≈ 50.6
        assert result.price > 49
        assert result.price < 52

    def test_deep_otm_call(self, market_atm, gbm_model, bs_engine):
        """Test deep OTM call."""
        call = VanillaOption(strike=200, maturity=0.25, is_call=True)
        result = bs_engine.price(call, gbm_model, market_atm)

        # Should be very small but positive
        assert result.price > 0
        assert result.price < 1

    def test_low_volatility(self, market_atm, bs_engine):
        """Test with very low volatility."""
        model = GBMModel(sigma=0.01)
        call = VanillaOption(strike=100, maturity=0.25, is_call=True)

        result = bs_engine.price(call, model, market_atm)

        # Should be close to forward intrinsic
        forward_intrinsic = max(100 - 100 * np.exp(-0.05 * 0.25), 0)
        assert abs(result.price - forward_intrinsic) < 1

    def test_high_volatility(self, market_atm, bs_engine):
        """Test with high volatility."""
        model = GBMModel(sigma=1.0)  # 100% volatility
        call = VanillaOption(strike=100, maturity=0.25, is_call=True)

        result = bs_engine.price(call, model, market_atm)

        assert result.price > 0
        assert result.price < market_atm.spot
