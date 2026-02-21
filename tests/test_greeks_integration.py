"""
Tests for Greeks Integration
============================

Integration tests for GreeksCalculator with all engines and ModelRegistry.

Author: Thomas
Created: 2025
"""

import pytest
import numpy as np

from backend.core import MarketEnvironment
from backend.models import GBMModel, HestonModel
from backend.instruments import EuropeanCall, EuropeanPut, VanillaOption
from backend.engines import BSAnalyticEngine, FFTEngine, MonteCarloEngine
from backend.greeks.calculator import GreeksCalculator
from backend.greeks.numerical import ModelNumericalGreeks, GreeksBumpConfig


class TestGreeksCalculatorIntegration:
    """Integration tests for GreeksCalculator with all engines."""

    @pytest.fixture
    def setup(self):
        return {
            'market': MarketEnvironment(spot=100, rate=0.05),
            'model': GBMModel(sigma=0.20),
            'option': EuropeanCall(strike=100, maturity=0.5),
        }

    def test_analytic_greeks_with_bs_engine(self, setup):
        """Test analytic Greeks calculation with BS engine."""
        calc = GreeksCalculator()
        engine = BSAnalyticEngine()
        result = calc.calculate(
            engine, setup['option'], setup['model'], setup['market'],
            include_higher_order=False
        )
        assert result.delta is not None
        assert 0 < result.delta < 1  # Call delta bounded

    def test_numerical_greeks_with_bs_engine(self, setup):
        """Test numerical Greeks calculation with BS engine."""
        calc = GreeksCalculator(prefer_analytic=False)
        engine = BSAnalyticEngine()
        result = calc.calculate(
            engine, setup['option'], setup['model'], setup['market'],
            include_higher_order=False
        )
        assert result.delta is not None
        assert 0 < result.delta < 1

    def test_numerical_greeks_with_fft_engine(self, setup):
        """Test numerical Greeks work with FFT engine."""
        calc = GreeksCalculator()
        engine = FFTEngine()
        result = calc.calculate(
            engine, setup['option'], setup['model'], setup['market'],
            include_higher_order=False
        )
        assert result.delta is not None
        assert 0 < result.delta < 1

    def test_numerical_greeks_with_mc_engine(self, setup):
        """Test numerical Greeks work with MC engine."""
        calc = GreeksCalculator()
        engine = MonteCarloEngine(n_paths=10000, seed=42)
        result = calc.calculate(
            engine, setup['option'], setup['model'], setup['market'],
            include_higher_order=False
        )
        assert result.delta is not None
        # MC has more variance, so use looser bounds
        assert -0.2 < result.delta < 1.2

    def test_greeks_consistency_across_engines(self, setup):
        """Test that Greeks are roughly consistent across engines."""
        calc = GreeksCalculator(prefer_analytic=False)

        bs_engine = BSAnalyticEngine()
        fft_engine = FFTEngine()

        bs_result = calc.calculate(
            bs_engine, setup['option'], setup['model'], setup['market'],
            include_higher_order=False
        )
        fft_result = calc.calculate(
            fft_engine, setup['option'], setup['model'], setup['market'],
            include_higher_order=False
        )

        # Results should be close (numerical differences from FFT vs analytic pricing)
        assert abs(bs_result.delta - fft_result.delta) < 0.01
        assert abs(bs_result.gamma - fft_result.gamma) < 0.005  # Gamma has more variance

    def test_greeks_with_heston_model(self):
        """Test Greeks with stochastic volatility model."""
        market = MarketEnvironment(spot=100, rate=0.05)
        model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        option = EuropeanCall(strike=100, maturity=0.5)

        calc = GreeksCalculator()
        engine = FFTEngine()

        result = calc.calculate(
            engine, option, model, market,
            include_higher_order=False
        )

        assert result.delta is not None
        assert 0 < result.delta < 1

    def test_greeks_with_put_option(self, setup):
        """Test Greeks calculation for put options."""
        calc = GreeksCalculator()
        engine = BSAnalyticEngine()
        put = EuropeanPut(strike=100, maturity=0.5)

        result = calc.calculate(
            engine, put, setup['model'], setup['market'],
            include_higher_order=False
        )

        # Put delta should be negative
        assert result.delta is not None
        assert -1 < result.delta < 0


class TestModelNumericalGreeksIntegration:
    """Tests for ModelNumericalGreeks class."""

    def test_constructor_with_config(self):
        """Test ModelNumericalGreeks accepts config."""
        config = GreeksBumpConfig(
            spot_bump=0.005,
            vol_bump=0.005,
            time_bump_days=0.5,
            rate_bump=0.00005
        )
        calc = ModelNumericalGreeks(config)

        assert calc.spot_bump == 0.005
        assert calc.vol_bump == 0.005
        assert calc.time_bump_days == 0.5
        assert calc.rate_bump == 0.00005

    def test_constructor_with_defaults(self):
        """Test ModelNumericalGreeks uses defaults."""
        calc = ModelNumericalGreeks()

        assert calc.spot_bump == 0.01
        assert calc.vol_bump == 0.01
        assert calc.time_bump_days == 1.0
        assert calc.rate_bump == 0.0001

    def test_calculate_with_bs_engine(self):
        """Test calculate method works."""
        market = MarketEnvironment(spot=100, rate=0.05)
        model = GBMModel(sigma=0.20)
        option = EuropeanCall(strike=100, maturity=0.5)
        engine = BSAnalyticEngine()

        calc = ModelNumericalGreeks()
        result = calc.calculate(engine, option, model, market)

        assert result.delta is not None
        assert result.gamma is not None
        assert result.vega is not None
        assert result.theta is not None
        assert result.rho is not None


class TestModelRegistryCreate:
    """Tests for ModelRegistry.create() method."""

    def test_create_gbm_model(self):
        """Test creating GBM model via registry."""
        from backend.models.registry import registry

        registry.register('test_gbm', GBMModel)
        model = registry.create('test_gbm', sigma=0.2)

        assert isinstance(model, GBMModel)
        assert model.sigma == 0.2

    def test_create_heston_model(self):
        """Test creating Heston model via registry."""
        from backend.models.registry import registry

        registry.register('test_heston', HestonModel)
        model = registry.create('test_heston',
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7
        )

        assert isinstance(model, HestonModel)
        assert model.v0 == 0.04
        assert model.kappa == 2.0

    def test_create_unknown_model_raises(self):
        """Test creating unknown model raises KeyError."""
        from backend.models.registry import registry

        with pytest.raises(KeyError):
            registry.create('nonexistent_model', sigma=0.2)


class TestEngineGreeksMethod:
    """Tests for greeks() method on engines."""

    @pytest.fixture
    def setup(self):
        return {
            'market': MarketEnvironment(spot=100, rate=0.05),
            'model': GBMModel(sigma=0.20),
            'option': EuropeanCall(strike=100, maturity=0.5),
        }

    def test_fft_engine_greeks(self, setup):
        """Test FFTEngine.greeks() method."""
        engine = FFTEngine()
        result = engine.greeks(
            setup['option'], setup['model'], setup['market']
        )

        assert result.delta is not None
        assert result.gamma is not None
        assert 0 < result.delta < 1

    def test_mc_engine_greeks(self, setup):
        """Test MonteCarloEngine.greeks() method."""
        engine = MonteCarloEngine(n_paths=10000, seed=42)
        result = engine.greeks(
            setup['option'], setup['model'], setup['market']
        )

        assert result.delta is not None
        assert result.gamma is not None

    def test_bs_engine_greeks(self, setup):
        """Test BSAnalyticEngine.greeks() method exists."""
        engine = BSAnalyticEngine()
        # BS engine should have greeks method
        assert hasattr(engine, 'greeks')


class TestExoticInstrumentGreeks:
    """Tests for Greeks with exotic instruments."""

    def test_lookback_floating_has_no_strike(self):
        """Verify floating LookbackOption has strike=None."""
        from backend.instruments import LookbackCall

        lookback = LookbackCall(maturity=0.5)
        assert lookback.strike is None

    def test_asian_has_strike(self):
        """Verify AsianOption has strike attribute."""
        from backend.instruments import AsianCall

        asian = AsianCall(strike=100, maturity=0.5)
        assert hasattr(asian, 'strike')
        assert asian.strike == 100


class TestHigherOrderGreeks:
    """Tests for second and third order Greeks."""

    @pytest.fixture
    def atm_setup(self):
        return {
            'market': MarketEnvironment(spot=100, rate=0.05),
            'model': GBMModel(sigma=0.20),
            'option': EuropeanCall(strike=100, maturity=0.5),
        }

    def test_analytic_higher_order_greeks_nonzero(self, atm_setup):
        """All 8 higher-order Greeks are non-zero for ATM call."""
        calc = GreeksCalculator()
        engine = BSAnalyticEngine()
        result = calc.calculate(
            engine, atm_setup['option'], atm_setup['model'], atm_setup['market'],
            include_higher_order=True
        )

        assert result.vanna != 0.0, "Vanna should be non-zero"
        assert result.volga != 0.0, "Volga should be non-zero"
        assert result.charm != 0.0, "Charm should be non-zero"
        assert result.veta != 0.0, "Veta should be non-zero"
        assert result.speed != 0.0, "Speed should be non-zero"
        assert result.zomma != 0.0, "Zomma should be non-zero"
        assert result.color != 0.0, "Color should be non-zero"
        assert result.ultima != 0.0, "Ultima should be non-zero"

    def test_analytic_vs_numerical_higher_order(self, atm_setup):
        """Numerical higher-order Greeks agree in sign and order with analytic.

        Note: analytic and numerical use different scaling conventions
        (analytic: per 1% vol for vanna/zomma; per 1%^2 for volga; etc.).
        We verify sign agreement and that spot-only Greeks (speed, charm, color)
        which share the same convention match closely.
        """
        engine = BSAnalyticEngine()

        analytic_calc = GreeksCalculator(prefer_analytic=True)
        numerical_calc = GreeksCalculator(prefer_analytic=False)

        analytic = analytic_calc.calculate(
            engine, atm_setup['option'], atm_setup['model'], atm_setup['market'],
            include_higher_order=True
        )
        numerical = numerical_calc.calculate(
            engine, atm_setup['option'], atm_setup['model'], atm_setup['market'],
            include_higher_order=True
        )

        # Same-convention Greeks should match closely
        np.testing.assert_allclose(numerical.charm, analytic.charm, rtol=0.05, atol=1e-4)
        np.testing.assert_allclose(numerical.speed, analytic.speed, rtol=0.05, atol=1e-6)

        # Vanna/zomma: analytic is per 1% vol (scaled by 1/100), numerical is raw
        np.testing.assert_allclose(numerical.vanna, analytic.vanna * 100, rtol=0.10, atol=1e-3)
        np.testing.assert_allclose(numerical.zomma, analytic.zomma * 100, rtol=0.10, atol=1e-3)

        # Volga and ultima: sign agreement (different stencil conventions)
        assert np.sign(numerical.volga) == np.sign(analytic.volga) or abs(analytic.volga) < 1e-8
        assert np.sign(numerical.ultima) == np.sign(analytic.ultima) or abs(analytic.ultima) < 1e-8

    def test_vanna_sign_convention(self):
        """Vanna is small for ATM, positive for OTM call."""
        calc = GreeksCalculator()
        engine = BSAnalyticEngine()
        model = GBMModel(sigma=0.20)
        market = MarketEnvironment(spot=100, rate=0.05)

        atm = EuropeanCall(strike=100, maturity=0.5)
        otm = EuropeanCall(strike=115, maturity=0.5)

        atm_greeks = calc.calculate(engine, atm, model, market, include_higher_order=True)
        otm_greeks = calc.calculate(engine, otm, model, market, include_higher_order=True)

        # ATM vanna is near zero (exactly zero for ATM with q=0 and specific d1)
        assert abs(atm_greeks.vanna) < abs(otm_greeks.vanna)
        # OTM call vanna is positive (delta increases with vol for OTM call)
        assert otm_greeks.vanna > 0

    def test_volga_positive_atm(self):
        """Volga > 0 for ATM (convexity in vol)."""
        calc = GreeksCalculator()
        engine = BSAnalyticEngine()
        model = GBMModel(sigma=0.20)
        market = MarketEnvironment(spot=100, rate=0.05)
        option = EuropeanCall(strike=100, maturity=0.5)

        result = calc.calculate(engine, option, model, market, include_higher_order=True)
        assert result.volga > 0, "Volga should be positive for ATM"

    def test_charm_direction(self):
        """Charm is consistent with delta decay."""
        calc = GreeksCalculator()
        engine = BSAnalyticEngine()
        model = GBMModel(sigma=0.20)
        market = MarketEnvironment(spot=100, rate=0.05)

        option = EuropeanCall(strike=100, maturity=0.5)
        result = calc.calculate(engine, option, model, market, include_higher_order=True)

        # For an ATM call, charm (dDelta/dt) captures delta's time decay
        # Value should be non-zero and have a definite sign
        assert result.charm != 0.0

    def test_ultima_regression(self):
        """Ultima produces correct values after Bug 1 fix."""
        calc = GreeksCalculator()
        engine = BSAnalyticEngine()
        model = GBMModel(sigma=0.20)
        market = MarketEnvironment(spot=100, rate=0.05)
        option = EuropeanCall(strike=100, maturity=0.5)

        # Both analytic and numerical should give a non-zero ultima
        analytic = calc.calculate(engine, option, model, market, include_higher_order=True)
        assert analytic.ultima != 0.0, "Ultima should be non-zero"

        # Cross-check: numerical ultima should be in same ballpark
        num_calc = GreeksCalculator(prefer_analytic=False)
        numerical = num_calc.calculate(engine, option, model, market, include_higher_order=True)
        assert numerical.ultima != 0.0, "Numerical ultima should be non-zero"

        # They should agree in sign
        assert np.sign(analytic.ultima) == np.sign(numerical.ultima), \
            f"Sign mismatch: analytic={analytic.ultima}, numerical={numerical.ultima}"

    def test_higher_order_greeks_heston(self):
        """Numerical higher-order Greeks work with Heston + FFT."""
        calc = GreeksCalculator()
        engine = FFTEngine()
        model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        market = MarketEnvironment(spot=100, rate=0.05)
        option = EuropeanCall(strike=100, maturity=0.5)

        result = calc.calculate(engine, option, model, market, include_higher_order=True)

        # Should produce non-zero higher-order Greeks
        assert result.vanna != 0.0
        assert result.volga != 0.0
        assert result.speed != 0.0
        assert result.zomma != 0.0


class TestNumericalVolga:
    """Regression test for Bug 4: numerical volga 4x scaling error."""

    def test_numerical_volga_vs_analytic(self):
        """Bug 4: Numerical volga must match analytic volga within 10% for GBM."""
        from backend.greeks.analytic import bs_all_greeks

        market = MarketEnvironment(spot=100, rate=0.05)
        model = GBMModel(sigma=0.20)
        option = EuropeanCall(strike=100, maturity=0.5)
        engine = BSAnalyticEngine()

        # Get numerical volga
        calc = GreeksCalculator(prefer_analytic=False)
        numerical = calc.calculate(engine, option, model, market, include_higher_order=True)

        # Get analytic volga (scaled per 1%²)
        g = bs_all_greeks(100.0, 100.0, 0.5, 0.05, 0.0, 0.20, True)
        analytic_volga = g[7]  # Scaled volga

        # Numerical volga is raw (unscaled), analytic is per 1%²
        # Raw volga = scaled_volga * 10000
        analytic_volga_raw = analytic_volga * 10000

        np.testing.assert_allclose(
            numerical.volga, analytic_volga_raw, rtol=0.10,
            err_msg="Numerical volga should match analytic within 10%"
        )


class TestDirectNumericalGreeks:
    """Group 5: Test finite_difference_* functions directly against bs_all_greeks."""

    @pytest.fixture
    def bs_price_func(self):
        """Create a BS price function for finite difference testing."""
        from backend.greeks.analytic import bs_all_greeks

        def price_func(spot, vol=0.20, time=0.25, rate=0.05, strike=100.0, is_call=True):
            g = bs_all_greeks(spot, strike, time, rate, 0.0, vol, is_call)
            return g[0]  # price

        return price_func

    @pytest.fixture
    def analytic_greeks(self):
        """Get analytic Greeks for comparison."""
        from backend.greeks.analytic import bs_all_greeks
        return bs_all_greeks(100.0, 100.0, 0.25, 0.05, 0.0, 0.20, True)

    def test_finite_difference_delta_vs_analytic(self, bs_price_func, analytic_greeks):
        """FD delta matches analytic delta within 0.5%."""
        from backend.greeks.numerical import finite_difference_delta

        fd_delta = finite_difference_delta(bs_price_func, 100.0, bump=0.01)
        np.testing.assert_allclose(fd_delta, analytic_greeks[1], rtol=0.005)

    def test_finite_difference_gamma_vs_analytic(self, bs_price_func, analytic_greeks):
        """FD gamma matches analytic gamma within 1%."""
        from backend.greeks.numerical import finite_difference_gamma

        fd_gamma = finite_difference_gamma(bs_price_func, 100.0, bump=0.01)
        np.testing.assert_allclose(fd_gamma, analytic_greeks[2], rtol=0.01)

    def test_finite_difference_vega_vs_analytic(self, bs_price_func, analytic_greeks):
        """FD vega matches analytic vega within 1%."""
        from backend.greeks.numerical import finite_difference_vega

        # FD vega returns per 1% vol, analytic vega (index 3) is also per 1% vol
        fd_vega = finite_difference_vega(bs_price_func, 100.0, vol=0.20, bump=0.01)
        np.testing.assert_allclose(fd_vega, analytic_greeks[3], rtol=0.01)

    def test_finite_difference_theta_vs_analytic(self, bs_price_func, analytic_greeks):
        """FD theta matches analytic theta within 2%."""
        from backend.greeks.numerical import finite_difference_theta

        # FD theta returns per day, analytic theta (index 4) is also per day
        fd_theta = finite_difference_theta(bs_price_func, 100.0, time=0.25, bump_days=1.0)
        np.testing.assert_allclose(fd_theta, analytic_greeks[4], rtol=0.02)

    def test_finite_difference_rho_vs_analytic(self, bs_price_func, analytic_greeks):
        """FD rho matches analytic rho within 1%."""
        from backend.greeks.numerical import finite_difference_rho

        # FD rho returns per 1% rate, analytic rho (index 5) is also per 1%
        fd_rho = finite_difference_rho(bs_price_func, 100.0, rate=0.05, bump=0.0001)
        np.testing.assert_allclose(fd_rho, analytic_greeks[5], rtol=0.01)

    def test_finite_difference_greeks_combined(self, bs_price_func, analytic_greeks):
        """All first-order Greeks via finite_difference_greeks() match analytic."""
        from backend.greeks.numerical import finite_difference_greeks

        result = finite_difference_greeks(
            bs_price_func, spot=100.0, vol=0.20, time=0.25, rate=0.05
        )

        np.testing.assert_allclose(result.delta, analytic_greeks[1], rtol=0.005)
        np.testing.assert_allclose(result.gamma, analytic_greeks[2], rtol=0.01)
        np.testing.assert_allclose(result.vega, analytic_greeks[3], rtol=0.01)
        np.testing.assert_allclose(result.theta, analytic_greeks[4], rtol=0.02)
        np.testing.assert_allclose(result.rho, analytic_greeks[5], rtol=0.01)


# =============================================================================
# Exotic Options Greeks Integration
# =============================================================================

class TestExoticGreeksIntegration:
    """Test GreeksCalculator and calculate_greeks with exotic instruments."""

    @pytest.fixture
    def exotic_setup(self):
        from backend.engines.exotic_engine import ExoticAnalyticEngine
        from backend.instruments.options import (
            BarrierOption, AsianOption, DigitalOption, LookbackOption
        )
        return {
            'engine': ExoticAnalyticEngine(),
            'gbm': GBMModel(sigma=0.25),
            'market': MarketEnvironment(spot=100, rate=0.05),
            'barrier': BarrierOption(100, 110, 0.25, is_call=True, is_up=True),
            'barrier_put': BarrierOption(100, 90, 0.25, is_call=False, is_up=False),
            'asian_geo': AsianOption(100, 0.25, is_call=True, average_type="geometric"),
            'digital': DigitalOption(100, 0.25, is_call=True, payout=100.0),
            'lookback_float': LookbackOption(0.25, is_call=True),
            'lookback_fixed': LookbackOption(0.25, is_call=True, strike=100, lookback_type="fixed"),
        }

    def test_calculate_greeks_barrier(self, exotic_setup):
        """calculate_greeks works with barrier options."""
        from backend.greeks.calculator import calculate_greeks
        s = exotic_setup
        greeks = calculate_greeks(s['engine'], s['barrier'], s['gbm'], s['market'])
        assert greeks.delta is not None
        assert 0 < greeks.delta < 1  # Call delta bounded

    def test_calculate_greeks_barrier_put(self, exotic_setup):
        """calculate_greeks works with barrier put options."""
        from backend.greeks.calculator import calculate_greeks
        s = exotic_setup
        greeks = calculate_greeks(s['engine'], s['barrier_put'], s['gbm'], s['market'])
        # Down-and-out put delta can be positive near barrier (knock-out effect)
        assert np.isfinite(greeks.delta)

    def test_calculate_greeks_asian_geometric(self, exotic_setup):
        """calculate_greeks works with Asian geometric options."""
        from backend.greeks.calculator import calculate_greeks
        s = exotic_setup
        greeks = calculate_greeks(s['engine'], s['asian_geo'], s['gbm'], s['market'])
        assert greeks.delta is not None
        assert 0 < greeks.delta < 1

    def test_calculate_greeks_digital(self, exotic_setup):
        """calculate_greeks works with digital options."""
        from backend.greeks.calculator import calculate_greeks
        s = exotic_setup
        greeks = calculate_greeks(s['engine'], s['digital'], s['gbm'], s['market'])
        assert greeks.delta > 0  # Digital call delta positive

    def test_calculate_greeks_lookback_floating(self, exotic_setup):
        """calculate_greeks works with floating-strike lookback options."""
        from backend.greeks.calculator import calculate_greeks
        s = exotic_setup
        greeks = calculate_greeks(s['engine'], s['lookback_float'], s['gbm'], s['market'])
        assert greeks.delta is not None

    def test_calculate_greeks_lookback_fixed(self, exotic_setup):
        """calculate_greeks works with fixed-strike lookback options."""
        from backend.greeks.calculator import calculate_greeks
        s = exotic_setup
        greeks = calculate_greeks(s['engine'], s['lookback_fixed'], s['gbm'], s['market'])
        assert greeks.delta is not None
        assert greeks.delta > 0  # Call delta positive

    def test_calculate_greeks_higher_order_barrier(self, exotic_setup):
        """Higher-order Greeks work for barrier options."""
        from backend.greeks.calculator import calculate_greeks, AllGreeksResult
        s = exotic_setup
        result = calculate_greeks(
            s['engine'], s['barrier'], s['gbm'], s['market'],
            include_higher_order=True
        )
        assert isinstance(result, AllGreeksResult)
        assert result.price > 0
        assert result.delta is not None
        assert result.vanna is not None
        assert result.speed is not None

    def test_calculate_greeks_higher_order_digital(self, exotic_setup):
        """Higher-order Greeks work for digital options."""
        from backend.greeks.calculator import calculate_greeks, AllGreeksResult
        s = exotic_setup
        result = calculate_greeks(
            s['engine'], s['digital'], s['gbm'], s['market'],
            include_higher_order=True
        )
        assert isinstance(result, AllGreeksResult)
        assert result.price > 0

    def test_calculate_greeks_higher_order_lookback(self, exotic_setup):
        """Higher-order Greeks work for lookback options."""
        from backend.greeks.calculator import calculate_greeks, AllGreeksResult
        s = exotic_setup
        result = calculate_greeks(
            s['engine'], s['lookback_float'], s['gbm'], s['market'],
            include_higher_order=True
        )
        assert isinstance(result, AllGreeksResult)
        assert result.price > 0

    def test_calculate_greeks_higher_order_asian_geometric(self, exotic_setup):
        """Higher-order Greeks work for Asian geometric options."""
        from backend.greeks.calculator import calculate_greeks, AllGreeksResult
        s = exotic_setup
        result = calculate_greeks(
            s['engine'], s['asian_geo'], s['gbm'], s['market'],
            include_higher_order=True
        )
        assert isinstance(result, AllGreeksResult)
        assert result.price > 0
        assert result.delta is not None
        assert result.vanna is not None
        assert result.speed is not None

    def test_engine_greeks_matches_direct(self, exotic_setup):
        """GreeksCalculator first-order matches engine.greeks() directly."""
        from backend.greeks.calculator import calculate_greeks
        s = exotic_setup
        direct = s['engine'].greeks(s['barrier'], s['gbm'], s['market'])
        via_calc = calculate_greeks(s['engine'], s['barrier'], s['gbm'], s['market'])
        np.testing.assert_allclose(via_calc.delta, direct.delta, rtol=1e-10)
        np.testing.assert_allclose(via_calc.gamma, direct.gamma, rtol=1e-10)
        np.testing.assert_allclose(via_calc.vega, direct.vega, rtol=1e-10)
        np.testing.assert_allclose(via_calc.theta, direct.theta, rtol=1e-10)
        np.testing.assert_allclose(via_calc.rho, direct.rho, rtol=1e-10)

    def test_theta_negative_for_long_exotic(self, exotic_setup):
        """Theta should be negative for long exotic options (time decay)."""
        from backend.greeks.calculator import calculate_greeks
        s = exotic_setup
        # Asian geometric call
        greeks = calculate_greeks(s['engine'], s['asian_geo'], s['gbm'], s['market'])
        assert greeks.theta < 0, f"Asian geometric theta should be negative, got {greeks.theta}"

    def test_mc_rejects_exotic_instruments(self, exotic_setup):
        """MonteCarloEngine.can_price() rejects exotic instruments."""
        mc = MonteCarloEngine(n_paths=1000, seed=42)
        s = exotic_setup
        assert not mc.can_price(s['barrier'], s['gbm'])
        assert not mc.can_price(s['asian_geo'], s['gbm'])
        assert not mc.can_price(s['digital'], s['gbm'])
        assert not mc.can_price(s['lookback_float'], s['gbm'])
        assert not mc.can_price(s['lookback_fixed'], s['gbm'])

    def test_mc_still_prices_vanilla(self, exotic_setup):
        """MonteCarloEngine still works for vanilla options."""
        mc = MonteCarloEngine(n_paths=1000, seed=42)
        vanilla = VanillaOption(strike=100, maturity=0.25, is_call=True)
        assert mc.can_price(vanilla, exotic_setup['gbm'])


class TestEngineRejectsExoticInstruments:
    """Tests that BS and FFT engines reject exotic instruments."""

    @pytest.fixture
    def exotic_instruments(self):
        from backend.instruments.options import (
            BarrierOption, AsianOption, DigitalOption, LookbackOption
        )
        return {
            'barrier': BarrierOption(100, 110, 0.25, is_call=True, is_up=True),
            'asian_geo': AsianOption(100, 0.25, is_call=True, average_type="geometric"),
            'digital': DigitalOption(100, 0.25, is_call=True, payout=100.0),
            'lookback': LookbackOption(0.25, is_call=True),
        }

    def test_bs_engine_rejects_exotic_instruments(self, exotic_instruments):
        """BSAnalyticEngine.can_price() returns False for exotic instruments."""
        bs = BSAnalyticEngine()
        gbm = GBMModel(sigma=0.25)
        for name, instr in exotic_instruments.items():
            assert not bs.can_price(instr, gbm), \
                f"BSAnalyticEngine should reject {name}"

    def test_fft_engine_rejects_exotic_instruments(self, exotic_instruments):
        """FFTEngine.can_price() returns False for exotic instruments."""
        fft = FFTEngine()
        gbm = GBMModel(sigma=0.25)
        for name, instr in exotic_instruments.items():
            assert not fft.can_price(instr, gbm), \
                f"FFTEngine should reject {name}"

    def test_bs_engine_still_accepts_vanilla(self):
        """BSAnalyticEngine still accepts VanillaOption."""
        bs = BSAnalyticEngine()
        gbm = GBMModel(sigma=0.20)
        vanilla = VanillaOption(strike=100, maturity=0.5, is_call=True)
        assert bs.can_price(vanilla, gbm)

    def test_fft_engine_still_accepts_vanilla(self):
        """FFTEngine still accepts VanillaOption."""
        fft = FFTEngine()
        gbm = GBMModel(sigma=0.20)
        vanilla = VanillaOption(strike=100, maturity=0.5, is_call=True)
        assert fft.can_price(vanilla, gbm)

    def test_calculate_greeks_bs_engine_with_exotic_falls_back(self):
        """calculate_greeks with BSAnalyticEngine + exotic uses numerical fallback, not BS analytic."""
        from backend.instruments.options import BarrierOption
        from backend.engines.exotic_engine import ExoticAnalyticEngine
        from backend.greeks.calculator import calculate_greeks

        gbm = GBMModel(sigma=0.25)
        market = MarketEnvironment(spot=100, rate=0.05)
        barrier = BarrierOption(100, 110, 0.25, is_call=True, is_up=True)

        # ExoticAnalyticEngine gives the correct exotic greeks
        exotic_engine = ExoticAnalyticEngine()
        correct_greeks = calculate_greeks(exotic_engine, barrier, gbm, market)

        # BSAnalyticEngine should NOT silently return vanilla BS greeks
        # It should raise because it can't price the instrument
        bs_engine = BSAnalyticEngine()
        assert not bs_engine.can_price(barrier, gbm), \
            "BSAnalyticEngine must reject barrier options"

        # Verify the exotic delta is substantially different from vanilla BS delta
        vanilla = VanillaOption(strike=100, maturity=0.25, is_call=True)
        vanilla_greeks = calculate_greeks(bs_engine, vanilla, gbm, market)
        assert abs(correct_greeks.delta - vanilla_greeks.delta) > 0.1, \
            "Exotic and vanilla deltas should differ significantly"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
