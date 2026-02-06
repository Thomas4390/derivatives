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

# Import constants and reporter from conftest
from tests.conftest import BS_BENCHMARK, PRICE_ATOL, PRICE_RTOL, MC_RTOL, report


# =============================================================================
# BLACK-SCHOLES ANALYTICAL ENGINE TESTS
# =============================================================================

class TestBSAnalyticEngine:
    """Tests for Black-Scholes analytical pricing engine."""

    def test_atm_call_price(self, market_atm, gbm_model, bs_engine, call_atm):
        """Test ATM call pricing against benchmark."""
        report.header("BS Engine: ATM Call Price")
        report.info("Validates BS analytical pricing for an at-the-money call option")
        report.info("Compares computed price against pre-calculated benchmark value")

        result = bs_engine.price(call_atm, gbm_model, market_atm)

        report.value("ATM Call Price", result.price, BS_BENCHMARK["call_price"], unit="$")
        report.success("Price matches benchmark within tolerance")

        np.testing.assert_allclose(
            result.price, BS_BENCHMARK["call_price"],
            rtol=PRICE_RTOL, atol=PRICE_ATOL
        )

    def test_atm_put_price(self, market_atm, gbm_model, bs_engine, put_atm):
        """Test ATM put pricing against benchmark."""
        report.header("BS Engine: ATM Put Price")
        report.info("Validates BS analytical pricing for an at-the-money put option")
        report.info("Compares computed price against pre-calculated benchmark value")

        result = bs_engine.price(put_atm, gbm_model, market_atm)

        report.value("ATM Put Price", result.price, BS_BENCHMARK["put_price"], unit="$")
        report.success("Price matches benchmark within tolerance")

        np.testing.assert_allclose(
            result.price, BS_BENCHMARK["put_price"],
            rtol=PRICE_RTOL, atol=PRICE_ATOL
        )

    def test_greeks_first_order(self, market_atm, gbm_model, bs_engine, call_atm):
        """Test first-order Greeks against benchmark."""
        report.header("BS Engine: First-Order Greeks")
        report.info("Validates analytical Greeks computation (Delta, Gamma, Vega)")
        report.info("Compares computed values against pre-calculated benchmarks")

        greeks = bs_engine.greeks(call_atm, gbm_model, market_atm)

        report.greeks(greeks, "Computed Greeks")
        print("  Expected Greeks (benchmark):")
        print(f"    Delta: {BS_BENCHMARK['delta_call']:>12.6f}")
        print(f"    Gamma: {BS_BENCHMARK['gamma']:>12.6f}")
        print(f"    Vega:  {BS_BENCHMARK['vega']:>12.6f}")
        report.success("All Greeks match benchmarks within tolerance")

        np.testing.assert_allclose(greeks.delta, BS_BENCHMARK["delta_call"], rtol=0.01)
        np.testing.assert_allclose(greeks.gamma, BS_BENCHMARK["gamma"], rtol=0.01)
        # Vega now scaled per 1% vol move (benchmark is per 100% move)
        np.testing.assert_allclose(greeks.vega, BS_BENCHMARK["vega"] / 100, rtol=0.01)

    def test_can_price_gbm_only(self, market_atm, bs_engine, call_atm):
        """BSAnalyticEngine should only work with GBM model."""
        report.header("BS Engine: Model Compatibility Check")
        report.info("Verifies that BS engine only accepts GBM models")
        report.info("BS analytical formula is only valid for constant volatility (GBM)")

        gbm = GBMModel(sigma=0.20)
        heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

        can_price_gbm = bs_engine.can_price(call_atm, gbm)
        can_price_heston = bs_engine.can_price(call_atm, heston)

        print(f"  can_price(GBM):    {can_price_gbm}")
        print(f"  can_price(Heston): {can_price_heston}")
        report.success("BS engine correctly accepts GBM and rejects Heston")

        assert can_price_gbm is True
        assert can_price_heston is False

    def test_implied_volatility(self, market_atm, gbm_model, bs_engine):
        """Test implied volatility calculation."""
        report.header("BS Engine: Implied Volatility")
        report.info("Tests IV extraction from option price using Newton-Raphson")
        report.info("Given a price from sigma=20%, IV solver should recover sigma=20%")

        call = VanillaOption(strike=100, maturity=0.25, is_call=True)

        # Get the price first
        price = bs_engine.price(call, gbm_model, market_atm).price

        # Calculate IV from price
        iv = bs_engine.implied_volatility(price, call, market_atm)

        report.value("Price used", price, unit="$")
        report.value("Computed IV", iv, expected=0.20, precision=6)
        report.success("IV correctly recovered from option price")

        # IV should match the original volatility
        np.testing.assert_allclose(iv, 0.20, rtol=1e-4)

    def test_implied_volatility_different_strikes(self, market_atm, gbm_model, bs_engine):
        """Test IV calculation for different moneyness levels."""
        report.header("BS Engine: IV Across Strikes")
        report.info("Tests IV extraction for ITM, ATM, and OTM options")
        report.info("Under BS model, IV should be constant across all strikes (no smile)")

        rows = []
        for strike in [90, 100, 110]:
            call = VanillaOption(strike=strike, maturity=0.25, is_call=True)
            price = bs_engine.price(call, gbm_model, market_atm).price
            iv = bs_engine.implied_volatility(price, call, market_atm)
            rows.append((strike, price, iv, 0.20, abs(iv - 0.20)))

            np.testing.assert_allclose(iv, 0.20, rtol=1e-3)

        report.table(
            ["Strike", "Price", "IV Calc", "IV Exp", "Diff"],
            rows,
            title="IV by Strike",
            precision=4
        )
        report.success("IV is constant across strikes as expected for GBM")

    @pytest.mark.parametrize("spot", [80, 90, 100, 110, 120])
    def test_price_across_spots(self, gbm_model, bs_engine, spot):
        """Test pricing across different spot prices."""
        report.header(f"BS Engine: Price at Spot={spot}")
        report.info(f"Tests option pricing with varying spot prices (K=100 fixed)")
        report.info(f"Spot < K: OTM call, Spot > K: ITM call")

        market = MarketEnvironment(spot=spot, rate=0.05, dividend_yield=0.0)
        call = VanillaOption(strike=100, maturity=0.25, is_call=True)

        result = bs_engine.price(call, gbm_model, market)

        print(f"  Spot={spot}: Price={result.price:.4f}$, Engine={result.engine}")
        report.success(f"Price computed correctly for spot={spot}")

        assert result.price >= 0
        assert result.engine == "BSAnalyticEngine"

    def test_price_with_dividends(self, market_with_dividend, gbm_model, bs_engine, call_atm):
        """Test pricing with continuous dividend yield."""
        report.header("BS Engine: Dividend Impact")
        report.info("Tests the effect of continuous dividend yield on call prices")
        report.info("Dividends reduce call value (less upside as stock drops ex-div)")

        result = bs_engine.price(call_atm, gbm_model, market_with_dividend)

        # Price should be lower with dividends
        result_no_div = bs_engine.price(call_atm, gbm_model, MarketEnvironment(spot=100, rate=0.05))

        report.comparison(
            "Price with dividend", result.price,
            "Price without dividend", result_no_div.price,
            unit="$"
        )
        report.success("Call price is lower with dividend yield as expected")

        assert result.price < result_no_div.price


# =============================================================================
# FFT ENGINE TESTS
# =============================================================================

class TestFFTEngine:
    """Tests for FFT pricing engine."""

    def test_gbm_matches_bs(self, market_atm, gbm_model, bs_engine, fft_engine, call_atm):
        """FFT with GBM should match BS analytical."""
        report.header("FFT Engine: GBM vs BS Comparison")
        report.info("Validates FFT pricing against analytical BS for GBM model")
        report.info("Both methods should give identical prices (up to numerical precision)")

        bs_price = bs_engine.price(call_atm, gbm_model, market_atm).price
        fft_price = fft_engine.price(call_atm, gbm_model, market_atm).price

        report.comparison("FFT", fft_price, "BS Analytic", bs_price, unit="$")
        report.success("FFT matches BS analytical price")

        np.testing.assert_allclose(fft_price, bs_price, rtol=1e-4, atol=0.01)

    def test_heston_pricing(self, market_atm, heston_model, fft_engine, call_atm):
        """Test Heston model pricing via FFT."""
        report.header("FFT Engine: Heston Model Pricing")
        report.info("Tests Carr-Madan FFT pricing with Heston stochastic volatility")
        report.info("Heston captures volatility clustering and mean reversion")

        result = fft_engine.price(call_atm, heston_model, market_atm)

        report.value("Heston Price (FFT)", result.price, unit="$")
        report.info(f"Model: {result.model}")
        report.success("Heston option successfully priced via FFT")

        assert result.price > 0
        # Model name is the full name from model.name property
        assert "Heston" in result.model

    def test_bates_pricing(self, market_atm, bates_model, fft_engine, call_atm):
        """Test Bates model pricing via FFT."""
        report.header("FFT Engine: Bates Model Pricing")
        report.info("Tests FFT pricing with Bates model (Heston + jumps)")
        report.info("Bates adds jump component to Heston for crash modeling")

        result = fft_engine.price(call_atm, bates_model, market_atm)

        report.value("Bates Price (FFT)", result.price, unit="$")
        report.info(f"Model: {result.model}")
        report.success("Bates option successfully priced via FFT")

        assert result.price > 0
        # Model name is the full name from model.name property
        assert "Bates" in result.model

    def test_merton_pricing(self, market_atm, merton_model, fft_engine, call_atm):
        """Test Merton jump-diffusion pricing via FFT."""
        report.header("FFT Engine: Merton Jump-Diffusion Pricing")
        report.info("Tests FFT pricing with Merton jump-diffusion model")
        report.info("Merton adds log-normal jumps to GBM for fat tails")

        result = fft_engine.price(call_atm, merton_model, market_atm)

        report.value("Merton Price (FFT)", result.price, unit="$")
        report.info(f"Model: {result.model}")
        report.success("Merton option successfully priced via FFT")

        assert result.price > 0
        # Model name is the full name from model.name property
        assert "Merton" in result.model

    def test_price_strikes(self, market_atm, heston_model, fft_engine):
        """Test batch pricing of multiple strikes."""
        report.header("FFT Engine: Batch Strike Pricing")
        report.info("Tests efficient pricing of multiple strikes in one FFT pass")
        report.info("FFT naturally produces prices for a range of strikes")

        call = VanillaOption(strike=100, maturity=0.5, is_call=True)
        strikes = np.array([90, 95, 100, 105, 110], dtype=float)

        prices = fft_engine.price_strikes(call, heston_model, market_atm, strikes)

        rows = [(k, p) for k, p in zip(strikes, prices)]
        report.table(["Strike", "Price"], rows, title="Price by Strike (Heston FFT)", precision=4)
        report.info("Prices should decrease monotonically for calls")
        report.success("Batch pricing completed with correct monotonicity")

        assert len(prices) == len(strikes)
        assert all(p > 0 for p in prices)
        # Prices should be decreasing for calls
        assert all(prices[i] >= prices[i + 1] for i in range(len(prices) - 1))

    def test_price_surface(self, market_atm, heston_model, fft_engine):
        """Test pricing surface generation."""
        report.header("FFT Engine: Volatility Surface Generation")
        report.info("Tests generation of full price surface (strikes x maturities)")
        report.info("Used for calibration and implied volatility surface extraction")

        strikes = np.array([90, 100, 110], dtype=float)
        maturities = np.array([0.25, 0.5, 1.0])

        surface = fft_engine.price_surface(
            heston_model, market_atm, strikes, maturities, is_call=True
        )

        print("  Price Surface (Heston):")
        print(f"    Shape: {surface.shape}")
        print(f"    Strikes: {strikes}")
        print(f"    Maturities: {maturities}")
        print(f"    Surface:\n{surface}")
        report.success("Price surface generated successfully")

        assert surface.shape == (3, 3)
        assert np.all(surface > 0)

    def test_fft_config(self, market_atm, gbm_model):
        """Test custom FFT configuration."""
        report.header("FFT Engine: Custom Configuration")
        report.info("Tests FFT with custom parameters (alpha, N, eta)")
        report.info("Allows fine-tuning of numerical accuracy vs speed tradeoff")

        config = FFTConfig(alpha=1.5, n_fft=4096, eta=0.25)
        engine = FFTEngine(config=config)

        call = VanillaOption(strike=100, maturity=0.5, is_call=True)
        result = engine.price(call, gbm_model, market_atm)

        report.params(alpha=config.alpha, n_fft=config.n_fft, eta=config.eta)
        report.value("Price (custom config)", result.price, unit="$")
        report.success("Custom FFT configuration works correctly")

        assert result.price > 0

    def test_put_pricing(self, market_atm, heston_model, fft_engine):
        """Test put option pricing via FFT."""
        report.header("FFT Engine: Put Option Pricing")
        report.info("Tests FFT pricing for put options using put-call parity transform")
        report.info("FFT computes calls directly, puts via parity")

        put = VanillaOption(strike=100, maturity=0.5, is_call=False)
        result = fft_engine.price(put, heston_model, market_atm)

        report.value("Heston Put Price (FFT)", result.price, unit="$")
        report.success("Put option priced successfully via FFT")

        assert result.price > 0

    def test_can_price(self, market_atm, fft_engine, call_atm):
        """Test can_price method."""
        report.header("FFT Engine: Model Compatibility Check")
        report.info("Verifies FFT engine works with all models having characteristic functions")
        report.info("FFT requires analytical characteristic function, available for GBM, Heston, etc.")

        gbm = GBMModel(sigma=0.20)
        heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

        can_gbm = fft_engine.can_price(call_atm, gbm)
        can_heston = fft_engine.can_price(call_atm, heston)

        print(f"  FFT can_price(GBM):    {can_gbm}")
        print(f"  FFT can_price(Heston): {can_heston}")
        report.success("FFT correctly supports both GBM and Heston models")

        # FFT should work with models that have characteristic functions
        assert can_gbm is True
        assert can_heston is True


# =============================================================================
# MONTE CARLO ENGINE TESTS
# =============================================================================

class TestMonteCarloEngine:
    """Tests for Monte Carlo pricing engine."""

    def test_gbm_converges_to_bs(self, market_atm, gbm_model, bs_engine, call_atm):
        """MC with GBM should converge to BS analytical."""
        report.header("MC Engine: Convergence to BS")
        report.info("Validates Monte Carlo convergence to analytical price for GBM")
        report.info("With 100k paths, MC error should be small relative to price")

        mc_engine = MonteCarloEngine(n_paths=100000, seed=42)

        bs_price = bs_engine.price(call_atm, gbm_model, market_atm).price
        mc_result = mc_engine.price(call_atm, gbm_model, market_atm)

        report.comparison("MC (100k paths)", mc_result.price, "BS Analytic", bs_price, unit="$")
        report.value("MC Std Error", mc_result.error, unit="$", precision=6)
        report.success("MC converges to analytical price within tolerance")

        np.testing.assert_allclose(mc_result.price, bs_price, rtol=MC_RTOL)

    def test_heston_pricing(self, market_atm, heston_model, call_atm):
        """Test Heston model pricing via MC."""
        report.header("MC Engine: Heston Model Pricing")
        report.info("Tests Monte Carlo simulation with Heston stochastic volatility")
        report.info("Simulates correlated stock and variance paths")

        mc_engine = MonteCarloEngine(n_paths=50000, seed=42)
        result = mc_engine.price(call_atm, heston_model, market_atm)

        report.value("Heston Price (MC)", result.price, unit="$")
        report.value("Std Error", result.error, unit="$", precision=6)
        report.success("Heston option priced via Monte Carlo")

        assert result.price > 0

    def test_bates_pricing(self, market_atm, bates_model, call_atm):
        """Test Bates model pricing via MC."""
        report.header("MC Engine: Bates Model Pricing")
        report.info("Tests Monte Carlo with Bates model (stochastic vol + jumps)")
        report.info("Simulates variance process and compound Poisson jumps")

        mc_engine = MonteCarloEngine(n_paths=50000, seed=42)
        result = mc_engine.price(call_atm, bates_model, market_atm)

        report.value("Bates Price (MC)", result.price, unit="$")
        report.value("Std Error", result.error, unit="$", precision=6)
        report.success("Bates option priced via Monte Carlo")

        assert result.price > 0

    def test_merton_pricing(self, market_atm, merton_model, call_atm):
        """Test Merton model pricing via MC."""
        report.header("MC Engine: Merton Jump-Diffusion Pricing")
        report.info("Tests Monte Carlo with Merton jump-diffusion")
        report.info("Simulates GBM with superimposed log-normal jumps")

        mc_engine = MonteCarloEngine(n_paths=50000, seed=42)
        result = mc_engine.price(call_atm, merton_model, market_atm)

        report.value("Merton Price (MC)", result.price, unit="$")
        report.value("Std Error", result.error, unit="$", precision=6)
        report.success("Merton option priced via Monte Carlo")

        assert result.price > 0

    def test_reproducibility(self, market_atm, gbm_model, call_atm):
        """Test that MC results are statistically consistent.

        Note: Exact reproducibility is not guaranteed with parallel numba code
        (@njit(parallel=True)) because thread scheduling is non-deterministic.
        We test that two runs produce results within expected MC variance.
        """
        report.header("MC Engine: Statistical Consistency")
        report.info("Tests that different seeds give statistically consistent results")
        report.info("Results should differ but be within combined standard errors")

        mc1 = MonteCarloEngine(n_paths=10000, seed=123)
        mc2 = MonteCarloEngine(n_paths=10000, seed=456)

        result1 = mc1.price(call_atm, gbm_model, market_atm)
        result2 = mc2.price(call_atm, gbm_model, market_atm)

        # Results should be within a few standard errors of each other
        combined_se = np.sqrt(result1.error**2 + result2.error**2)

        report.comparison("Run 1 (seed=123)", result1.price, "Run 2 (seed=456)", result2.price, unit="$")
        report.value("Combined error", combined_se, unit="$", precision=6)
        report.value("Tolerance (4*SE)", 4*combined_se, unit="$", precision=6)
        report.success("Results are within 4 standard errors of each other")

        np.testing.assert_allclose(result1.price, result2.price, atol=4*combined_se)

    def test_different_seeds_different_results(self, market_atm, gbm_model, call_atm):
        """Different seeds should give different results (within MC error)."""
        report.header("MC Engine: Seed Independence")
        report.info("Verifies different random seeds produce different sample paths")
        report.info("Results should differ due to different random numbers")

        mc1 = MonteCarloEngine(n_paths=10000, seed=123)
        mc2 = MonteCarloEngine(n_paths=10000, seed=456)

        result1 = mc1.price(call_atm, gbm_model, market_atm)
        result2 = mc2.price(call_atm, gbm_model, market_atm)

        report.comparison("Seed 123", result1.price, "Seed 456", result2.price, unit="$")
        report.success("Different seeds produce different (but close) results")

        # Should be different but close
        assert result1.price != result2.price
        np.testing.assert_allclose(result1.price, result2.price, rtol=0.05)

    def test_price_strikes(self, market_atm, heston_model):
        """Test batch pricing of multiple strikes."""
        report.header("MC Engine: Batch Strike Pricing")
        report.info("Tests pricing multiple strikes from same simulation")
        report.info("Efficient: reuses same paths for all strikes")

        mc_engine = MonteCarloEngine(n_paths=50000, seed=42)

        call = VanillaOption(strike=100, maturity=0.5, is_call=True)
        strikes = np.array([90, 95, 100, 105, 110], dtype=float)

        # MonteCarloEngine.price_strikes returns (prices, std_errors) tuple
        prices, std_errors = mc_engine.price_strikes(call, heston_model, market_atm, strikes)

        rows = [(k, p, e) for k, p, e in zip(strikes, prices, std_errors)]
        report.table(
            ["Strike", "Price", "Std Err"],
            rows,
            title="Price by Strike (Heston MC)",
            precision=4
        )
        report.success("Batch pricing completed for all strikes")

        assert len(prices) == len(strikes)
        assert len(std_errors) == len(strikes)
        assert all(p > 0 for p in prices)

    def test_put_pricing(self, market_atm, gbm_model, bs_engine):
        """Test put option pricing via MC."""
        report.header("MC Engine: Put Option Pricing")
        report.info("Tests Monte Carlo pricing for put options")
        report.info("Put payoff max(K-S,0) simulated directly")

        mc_engine = MonteCarloEngine(n_paths=100000, seed=42)

        put = VanillaOption(strike=100, maturity=0.25, is_call=False)

        bs_price = bs_engine.price(put, gbm_model, market_atm).price
        mc_result = mc_engine.price(put, gbm_model, market_atm)

        report.comparison("MC Put", mc_result.price, "BS Put", bs_price, unit="$")
        report.success("MC put price matches BS analytical")

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
        report.header(f"Engine Comparison: K={strike}, T={maturity}")
        report.info("Compares BS, FFT, and MC engines for the same option")
        report.info("All should converge to same price for GBM model")

        call = VanillaOption(strike=strike, maturity=maturity, is_call=True)

        bs_engine = BSAnalyticEngine()
        fft_engine = FFTEngine()
        mc_engine = MonteCarloEngine(n_paths=100000, seed=42)

        bs_price = bs_engine.price(call, gbm_model, market_atm).price
        fft_price = fft_engine.price(call, gbm_model, market_atm).price
        mc_price = mc_engine.price(call, gbm_model, market_atm).price

        print(f"  K={strike}, T={maturity}:")
        print(f"    BS:  {bs_price:.6f}$ ")
        print(f"    FFT: {fft_price:.6f}$ (diff: {abs(fft_price-bs_price):.6f})")
        print(f"    MC:  {mc_price:.6f}$ (diff: {abs(mc_price-bs_price):.6f})")
        report.success("All engines produce consistent prices")

        # FFT should be very close to BS (slightly higher tolerance for OTM options)
        np.testing.assert_allclose(fft_price, bs_price, rtol=2e-3)

        # MC should be within 2%
        np.testing.assert_allclose(mc_price, bs_price, rtol=0.02)

    def test_fft_mc_consistent_heston(self, market_atm, heston_model, fft_engine):
        """FFT and MC should be consistent for Heston."""
        report.header("Engine Comparison: FFT vs MC (Heston)")
        report.info("Compares FFT and Monte Carlo for Heston model")
        report.info("No analytical solution exists, so we cross-validate methods")

        mc_engine = MonteCarloEngine(n_paths=100000, seed=42)

        call = VanillaOption(strike=100, maturity=0.5, is_call=True)

        fft_price = fft_engine.price(call, heston_model, market_atm).price
        mc_price = mc_engine.price(call, heston_model, market_atm).price

        report.comparison("FFT (Heston)", fft_price, "MC (Heston)", mc_price, unit="$")
        report.success("FFT and MC are consistent for Heston model")

        # Should be within 3% (MC has higher variance for stochastic vol)
        np.testing.assert_allclose(mc_price, fft_price, rtol=0.03)


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_short_maturity(self, market_atm, gbm_model, bs_engine):
        """Test pricing with very short time to expiry."""
        report.header("Edge Case: Very Short Maturity (T=0.001)")
        report.info("Tests option pricing near expiry (1 day equivalent)")
        report.info("Price should approach intrinsic value as T->0")

        call = VanillaOption(strike=100, maturity=0.001, is_call=True)
        result = bs_engine.price(call, gbm_model, market_atm)

        report.value("Price (T=0.001)", result.price, unit="$")
        report.info("Should be close to intrinsic value (0 for ATM)")
        report.success("Near-expiry pricing handled correctly")

        # Should be close to intrinsic (S - K) = 0 for ATM
        assert abs(result.price) < 1.0

    def test_very_long_maturity(self, market_atm, gbm_model, bs_engine):
        """Test pricing with long time to expiry."""
        report.header("Edge Case: Very Long Maturity (T=10 years)")
        report.info("Tests option pricing for long-dated options")
        report.info("Price bounded above by spot price")

        call = VanillaOption(strike=100, maturity=10.0, is_call=True)
        result = bs_engine.price(call, gbm_model, market_atm)

        report.value("Price (T=10 years)", result.price, unit="$")
        report.info(f"Should be < Spot ({market_atm.spot}$)")
        report.success("Long-dated option priced correctly")

        assert result.price > 0
        assert result.price < market_atm.spot  # Upper bound

    def test_deep_itm_call(self, market_atm, gbm_model, bs_engine):
        """Test deep ITM call."""
        report.header("Edge Case: Deep ITM Call (K=50)")
        report.info("Tests pricing for deep in-the-money option")
        report.info("Price should approach S - K*exp(-rT) discounted intrinsic")

        call = VanillaOption(strike=50, maturity=0.25, is_call=True)
        result = bs_engine.price(call, gbm_model, market_atm)

        # Should be close to S - K*exp(-rT) ~ 100 - 50*0.9876 ~ 50.6
        expected_approx = 100 - 50 * np.exp(-0.05 * 0.25)
        report.value("Deep ITM Price (K=50)", result.price, expected=expected_approx, unit="$")
        report.success("Deep ITM call priced correctly")

        assert result.price > 49
        assert result.price < 52

    def test_deep_otm_call(self, market_atm, gbm_model, bs_engine):
        """Test deep OTM call."""
        report.header("Edge Case: Deep OTM Call (K=200)")
        report.info("Tests pricing for deep out-of-the-money option")
        report.info("Price should be very small but positive")

        call = VanillaOption(strike=200, maturity=0.25, is_call=True)
        result = bs_engine.price(call, gbm_model, market_atm)

        report.value("Deep OTM Price (K=200)", result.price, unit="$")
        report.info("Should be very small but positive")
        report.success("Deep OTM call priced correctly (small positive value)")

        # Should be very small but positive
        assert result.price > 0
        assert result.price < 1

    def test_low_volatility(self, market_atm, bs_engine):
        """Test with very low volatility."""
        report.header("Edge Case: Very Low Volatility (sigma=1%)")
        report.info("Tests pricing with near-zero volatility")
        report.info("Price should approach discounted forward intrinsic value")

        model = GBMModel(sigma=0.01)
        call = VanillaOption(strike=100, maturity=0.25, is_call=True)

        result = bs_engine.price(call, model, market_atm)

        # Should be close to forward intrinsic
        forward_intrinsic = max(100 - 100 * np.exp(-0.05 * 0.25), 0)
        report.value("Price (vol=1%)", result.price, expected=forward_intrinsic, unit="$")
        report.success("Low volatility case handled correctly")

        assert abs(result.price - forward_intrinsic) < 1

    def test_high_volatility(self, market_atm, bs_engine):
        """Test with high volatility."""
        report.header("Edge Case: Very High Volatility (sigma=100%)")
        report.info("Tests pricing with extreme volatility")
        report.info("Price bounded between 0 and spot price")

        model = GBMModel(sigma=1.0)  # 100% volatility
        call = VanillaOption(strike=100, maturity=0.25, is_call=True)

        result = bs_engine.price(call, model, market_atm)

        report.value("Price (vol=100%)", result.price, unit="$")
        report.info(f"Bounds: 0 < Price < Spot ({market_atm.spot}$)")
        report.success("High volatility case handled correctly")

        assert result.price > 0
        assert result.price < market_atm.spot


# =============================================================================
# EXTENDED CROSS-ENGINE CONSISTENCY TESTS
# =============================================================================

class TestExtendedEngineComparison:
    """Extended cross-engine consistency tests for jump and SV models."""

    def test_fft_mc_consistent_bates(self, market_atm, bates_model, fft_engine):
        """FFT vs MC for Bates model (tolerance 3%)."""
        mc_engine = MonteCarloEngine(n_paths=100000, seed=42)
        call = VanillaOption(strike=100, maturity=0.5, is_call=True)

        fft_price = fft_engine.price(call, bates_model, market_atm).price
        mc_price = mc_engine.price(call, bates_model, market_atm).price

        np.testing.assert_allclose(mc_price, fft_price, rtol=0.03)

    def test_fft_mc_consistent_merton(self, market_atm, merton_model, fft_engine):
        """FFT vs MC for Merton model (tolerance 3%)."""
        mc_engine = MonteCarloEngine(n_paths=100000, seed=42)
        call = VanillaOption(strike=100, maturity=0.5, is_call=True)

        fft_price = fft_engine.price(call, merton_model, market_atm).price
        mc_price = mc_engine.price(call, merton_model, market_atm).price

        np.testing.assert_allclose(mc_price, fft_price, rtol=0.03)

    @pytest.mark.parametrize("engine_factory,model_name", [
        (lambda: BSAnalyticEngine(), "bs"),
        (lambda: FFTEngine(), "fft"),
        (lambda: MonteCarloEngine(n_paths=100000, seed=42), "mc"),
    ])
    def test_put_call_parity_all_engines(self, engine_factory, model_name, market_atm, gbm_model):
        """Put-call parity: C - P = S - K*exp(-rT) for all engines."""
        engine = engine_factory()
        k, t = 100.0, 0.5

        call = VanillaOption(strike=k, maturity=t, is_call=True)
        put = VanillaOption(strike=k, maturity=t, is_call=False)

        c = engine.price(call, gbm_model, market_atm).price
        p = engine.price(put, gbm_model, market_atm).price

        expected = market_atm.spot - k * np.exp(-market_atm.rate * t)
        actual = c - p

        np.testing.assert_allclose(actual, expected, atol=0.5)

    def test_fft_put_call_parity_heston(self, market_atm, heston_model, fft_engine):
        """Put-call parity for Heston/FFT."""
        k, t = 100.0, 0.5

        call = VanillaOption(strike=k, maturity=t, is_call=True)
        put = VanillaOption(strike=k, maturity=t, is_call=False)

        c = fft_engine.price(call, heston_model, market_atm).price
        p = fft_engine.price(put, heston_model, market_atm).price

        expected = market_atm.spot - k * np.exp(-market_atm.rate * t)
        actual = c - p

        np.testing.assert_allclose(actual, expected, atol=0.5)


# =============================================================================
# REGRESSION TESTS
# =============================================================================

class TestRegressions:
    """Regression tests for confirmed bugs."""

    def test_fft_put_with_dividends(self):
        """Bug 3: FFT put price must be correct when q > 0."""
        market = MarketEnvironment(spot=100.0, rate=0.05, dividend_yield=0.03)
        model = GBMModel(sigma=0.20)

        bs_engine = BSAnalyticEngine()
        fft_engine = FFTEngine()

        put = VanillaOption(strike=100, maturity=0.5, is_call=False)

        bs_price = bs_engine.price(put, model, market).price
        fft_price = fft_engine.price(put, model, market).price

        # FFT put with dividends must match BS within tight tolerance
        np.testing.assert_allclose(fft_price, bs_price, rtol=1e-3, atol=0.02)

    def test_bs_engine_greeks_scaled(self):
        """Bug 1: engine.greeks() must return market-scaled values matching bs_all_greeks."""
        from backend.greeks.analytic import bs_all_greeks

        market = MarketEnvironment(spot=100.0, rate=0.05, dividend_yield=0.0)
        model = GBMModel(sigma=0.20)
        engine = BSAnalyticEngine()
        option = VanillaOption(strike=100, maturity=0.5, is_call=True)

        greeks = engine.greeks(option, model, market)
        g = bs_all_greeks(100.0, 100.0, 0.5, 0.05, 0.0, 0.20, True)

        # All Greeks must match exactly (same source of truth)
        np.testing.assert_allclose(greeks.delta, g[1], rtol=1e-10)
        np.testing.assert_allclose(greeks.gamma, g[2], rtol=1e-10)
        np.testing.assert_allclose(greeks.vega, g[3], rtol=1e-10)
        np.testing.assert_allclose(greeks.theta, g[4], rtol=1e-10)
        np.testing.assert_allclose(greeks.rho, g[5], rtol=1e-10)
        np.testing.assert_allclose(greeks.vanna, g[6], rtol=1e-10)
        np.testing.assert_allclose(greeks.volga, g[7], rtol=1e-10)
        np.testing.assert_allclose(greeks.ultima, g[13], rtol=1e-10)


# =============================================================================
# GARCH MC REJECTION TESTS (Group 1)
# =============================================================================

class TestGARCHMCRejection:
    """Tests that MonteCarloEngine correctly rejects GARCH models."""

    def test_mc_engine_rejects_garch(self):
        """can_price() returns False for all 3 GARCH variants."""
        from backend.models.garch import GARCHModel, NGARCHModel, GJRGARCHModel

        mc = MonteCarloEngine(n_paths=1000, seed=42)
        call = VanillaOption(strike=100, maturity=0.5, is_call=True)

        garch = GARCHModel(sigma0=0.20, omega=0.002, alpha=0.05, beta=0.90)
        ngarch = NGARCHModel(sigma0=0.20, omega=0.002, alpha=0.05, beta=0.90, theta=0.3)
        gjr = GJRGARCHModel(sigma0=0.20, omega=0.002, alpha=0.04, beta=0.85, gamma=0.10)

        assert mc.can_price(call, garch) is False
        assert mc.can_price(call, ngarch) is False
        assert mc.can_price(call, gjr) is False

    def test_mc_engine_still_accepts_supported_models(self):
        """can_price() returns True for GBM/Heston/Bates/Merton."""
        mc = MonteCarloEngine(n_paths=1000, seed=42)
        call = VanillaOption(strike=100, maturity=0.5, is_call=True)

        assert mc.can_price(call, GBMModel(sigma=0.20)) is True
        assert mc.can_price(call, HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)) is True
        assert mc.can_price(call, BatesModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)) is True
        assert mc.can_price(call, MertonModel(sigma=0.20, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)) is True

    def test_garch_models_have_own_pricer(self):
        """GARCH models provide create_pricer() as an alternative to MonteCarloEngine."""
        from backend.models.garch import GARCHModel, NGARCHModel, GJRGARCHModel

        garch = GARCHModel(sigma0=0.20, omega=0.002, alpha=0.05, beta=0.90)
        ngarch = NGARCHModel(sigma0=0.20, omega=0.002, alpha=0.05, beta=0.90, theta=0.3)
        gjr = GJRGARCHModel(sigma0=0.20, omega=0.002, alpha=0.04, beta=0.85, gamma=0.10)

        # All GARCH models should have create_pricer
        assert hasattr(garch, 'create_pricer')
        assert hasattr(ngarch, 'create_pricer')
        assert hasattr(gjr, 'create_pricer')

        # create_pricer should return a pricer object with a price method
        assert hasattr(garch.create_pricer(), 'price')
        assert hasattr(ngarch.create_pricer(), 'price')
        assert hasattr(gjr.create_pricer(), 'price')


# =============================================================================
# IMPLIED VOLATILITY EDGE CASES (Group 2)
# =============================================================================

class TestImpliedVolatilityEdgeCases:
    """Tests for implied volatility edge cases and convergence."""

    def test_iv_convergence_failure_raises(self):
        """Impossible price raises ValueError."""
        engine = BSAnalyticEngine()
        call = VanillaOption(strike=100, maturity=0.25, is_call=True)
        market = MarketEnvironment(spot=100, rate=0.05)

        # Price that's too high (above spot) is impossible for a call
        with pytest.raises(ValueError):
            engine.implied_volatility(200.0, call, market)

    @pytest.mark.parametrize("sigma", [0.05, 0.10, 0.20, 0.50, 1.0, 2.0])
    def test_iv_roundtrip_various_vols(self, sigma):
        """Round-trip test: price at sigma -> IV should recover sigma."""
        engine = BSAnalyticEngine()
        model = GBMModel(sigma=sigma)
        call = VanillaOption(strike=100, maturity=0.5, is_call=True)
        market = MarketEnvironment(spot=100, rate=0.05)

        price = engine.price(call, model, market).price
        iv = engine.implied_volatility(price, call, market)

        np.testing.assert_allclose(iv, sigma, rtol=1e-3)

    def test_iv_very_low_vega_early_exit(self):
        """Near-expiry deep OTM raises ValueError (vega too small to iterate)."""
        engine = BSAnalyticEngine()
        # Very short maturity + deep OTM -> negligible vega
        call = VanillaOption(strike=200, maturity=0.001, is_call=True)
        market = MarketEnvironment(spot=100, rate=0.05)

        # Price of this option is essentially 0; IV search should fail
        with pytest.raises(ValueError):
            engine.implied_volatility(0.0001, call, market)
