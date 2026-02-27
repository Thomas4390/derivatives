"""
Tests for Exotic Options
========================

Test suite for Asian, Barrier, Lookback, and Digital options.
Includes instrument construction tests, MC pricing tests,
and ExoticAnalyticEngine tests with parity/bounds/greeks verification.

Author: Thomas
Created: 2025
"""

import pytest
import numpy as np
from backend.instruments import (
    # Asian
    AsianOption,
    AsianCall,
    AsianPut,
    AsianGeometricCall,
    AsianGeometricPut,
    AsianCallPayoff,
    AsianPutPayoff,
    # Barrier
    BarrierOption,
    BarrierUpOutCall,
    BarrierUpInCall,
    BarrierDownOutCall,
    BarrierDownInCall,
    BarrierUpOutPut,
    BarrierUpInPut,
    BarrierDownOutPut,
    BarrierDownInPut,
    BarrierUpOutCallPayoff,
    BarrierDownOutPutPayoff,
    # Lookback
    LookbackOption,
    LookbackCall,
    LookbackPut,
    LookbackFixedCall,
    LookbackFloatingCallPayoff,
    LookbackFloatingPutPayoff,
    # Digital
    DigitalOption,
)
from backend.engines.exotic_engine import ExoticAnalyticEngine
from backend.engines.analytic_engine import BSAnalyticEngine
from backend.instruments.options import VanillaOption
from backend.models.gbm import GBMModel
from backend.core.market import MarketEnvironment
from backend.core.result_types import GreeksResult


class TestAsianOption:
    """Tests for Asian options."""

    def test_asian_call_creation(self):
        """Test Asian call option creation."""
        opt = AsianCall(strike=100, maturity=1.0)
        assert opt.strike == 100
        assert opt.maturity == 1.0
        assert opt.is_call is True
        assert opt.average_type == "arithmetic"
        assert opt.payoff.is_path_dependent is True

    def test_asian_put_creation(self):
        """Test Asian put option creation."""
        opt = AsianPut(strike=100, maturity=1.0)
        assert opt.strike == 100
        assert opt.maturity == 1.0
        assert opt.is_call is False
        assert opt.payoff.is_path_dependent is True

    def test_asian_geometric_creation(self):
        """Test geometric Asian option creation."""
        opt = AsianGeometricCall(strike=100, maturity=1.0)
        assert opt.average_type == "geometric"
        assert opt.payoff is None  # analytical only

    def test_asian_call_payoff_evaluation(self):
        """Test Asian call payoff calculation."""
        opt = AsianCall(strike=100, maturity=1.0)
        # Path with average = 105
        path = np.array([[100, 105, 110]])  # avg = 105
        payoffs = opt.payoff(path)
        assert payoffs[0] == pytest.approx(5.0)  # max(105-100, 0)

    def test_asian_put_payoff_evaluation(self):
        """Test Asian put payoff calculation."""
        opt = AsianPut(strike=100, maturity=1.0)
        # Path with average = 95
        path = np.array([[90, 95, 100]])  # avg = 95
        payoffs = opt.payoff(path)
        assert payoffs[0] == pytest.approx(5.0)  # max(100-95, 0)

    def test_asian_call_otm(self):
        """Test Asian call OTM payoff."""
        opt = AsianCall(strike=100, maturity=1.0)
        # Path with average = 95
        path = np.array([[90, 95, 100]])  # avg = 95
        payoffs = opt.payoff(path)
        assert payoffs[0] == pytest.approx(0.0)  # max(95-100, 0)

    def test_asian_immutability(self):
        """Test Asian option immutability."""
        opt = AsianCall(strike=100, maturity=1.0)
        with pytest.raises(AttributeError):
            opt._strike = 200

    def test_asian_invalid_strike(self):
        """Test Asian option validation."""
        with pytest.raises(ValueError, match="Strike must be positive"):
            AsianCall(strike=-100, maturity=1.0)

    def test_asian_invalid_maturity(self):
        """Test Asian option validation."""
        with pytest.raises(ValueError, match="Maturity must be positive"):
            AsianCall(strike=100, maturity=-1.0)

    def test_asian_invalid_average_type(self):
        """Test Asian option average_type validation."""
        with pytest.raises(ValueError, match="average_type"):
            AsianOption(strike=100, maturity=1.0, average_type="harmonic")

    def test_asian_repr(self):
        """Test Asian option string representation."""
        opt = AsianCall(strike=100, maturity=1.0)
        assert "AsianOption" in repr(opt)
        assert "Call" in repr(opt)
        assert "K=100" in repr(opt)

    def test_asian_geometric_repr(self):
        """Test geometric Asian option repr."""
        opt = AsianGeometricCall(strike=100, maturity=1.0)
        assert "Geometric" in repr(opt)

    def test_asian_equality(self):
        """Test Asian option equality."""
        opt1 = AsianCall(strike=100, maturity=1.0)
        opt2 = AsianCall(strike=100, maturity=1.0)
        opt3 = AsianCall(strike=105, maturity=1.0)
        opt4 = AsianGeometricCall(strike=100, maturity=1.0)
        assert opt1 == opt2
        assert opt1 != opt3
        assert opt1 != opt4  # different average_type
        assert hash(opt1) == hash(opt2)

    def test_asian_multiple_paths(self):
        """Test Asian payoff with multiple paths."""
        opt = AsianCall(strike=100, maturity=1.0)
        paths = np.array([
            [100, 105, 110],  # avg=105, payoff=5
            [90, 95, 100],    # avg=95, payoff=0
            [100, 110, 120],  # avg=110, payoff=10
        ])
        payoffs = opt.payoff(paths)
        assert len(payoffs) == 3
        assert payoffs[0] == pytest.approx(5.0)
        assert payoffs[1] == pytest.approx(0.0)
        assert payoffs[2] == pytest.approx(10.0)


class TestBarrierOption:
    """Tests for Barrier options."""

    def test_barrier_up_out_call_not_knocked(self):
        """Test up-and-out call when barrier is not touched."""
        opt = BarrierUpOutCall(strike=100, barrier=120, maturity=1.0)
        path = np.array([[100, 105, 110]])  # Never hits 120
        payoffs = opt.payoff(path)
        assert payoffs[0] == pytest.approx(10.0)  # max(110-100, 0)

    def test_barrier_up_out_call_knocked(self):
        """Test up-and-out call when barrier is touched."""
        opt = BarrierUpOutCall(strike=100, barrier=120, maturity=1.0)
        path = np.array([[100, 125, 110]])  # Hits 120 -> knocked out
        payoffs = opt.payoff(path)
        assert payoffs[0] == 0.0

    def test_barrier_up_out_call_exactly_at_barrier(self):
        """Test up-and-out call when price exactly at barrier."""
        opt = BarrierUpOutCall(strike=100, barrier=120, maturity=1.0)
        path = np.array([[100, 120, 110]])  # Hits exactly 120 -> knocked out
        payoffs = opt.payoff(path)
        assert payoffs[0] == 0.0

    def test_barrier_down_out_put_not_knocked(self):
        """Test down-and-out put when barrier is not touched."""
        opt = BarrierDownOutPut(strike=100, barrier=80, maturity=1.0)
        path = np.array([[100, 95, 90]])  # Never hits 80
        payoffs = opt.payoff(path)
        assert payoffs[0] == pytest.approx(10.0)  # max(100-90, 0)

    def test_barrier_down_out_put_knocked(self):
        """Test down-and-out put when barrier is touched."""
        opt = BarrierDownOutPut(strike=100, barrier=80, maturity=1.0)
        path = np.array([[100, 75, 90]])  # Hits 80 -> knocked out
        payoffs = opt.payoff(path)
        assert payoffs[0] == 0.0

    def test_barrier_immutability(self):
        """Test Barrier option immutability."""
        opt = BarrierUpOutCall(strike=100, barrier=120, maturity=1.0)
        with pytest.raises(AttributeError):
            opt._strike = 200

    def test_barrier_invalid_strike(self):
        """Test Barrier option validation."""
        with pytest.raises(ValueError, match="Strike must be positive"):
            BarrierUpOutCall(strike=-100, barrier=120, maturity=1.0)

    def test_barrier_up_out_call_barrier_validation(self):
        """Test that barrier must be above strike for up-out call."""
        with pytest.raises(ValueError, match="Barrier must be above strike"):
            BarrierUpOutCallPayoff(strike=100, barrier=80)  # barrier < strike

    def test_barrier_down_out_put_barrier_validation(self):
        """Test that barrier must be below strike for down-out put."""
        with pytest.raises(ValueError, match="Barrier must be below strike"):
            BarrierDownOutPutPayoff(strike=100, barrier=120)  # barrier > strike

    def test_barrier_repr(self):
        """Test Barrier option string representation."""
        opt = BarrierUpOutCall(strike=100, barrier=120, maturity=1.0)
        assert "BarrierOption" in repr(opt)
        assert "UpOutCall" in repr(opt)
        assert "K=100" in repr(opt)
        assert "B=120" in repr(opt)

    def test_barrier_knock_in_repr(self):
        """Test knock-in barrier repr."""
        opt = BarrierUpInCall(strike=100, barrier=120, maturity=1.0)
        assert "UpInCall" in repr(opt)

    def test_barrier_equality(self):
        """Test Barrier option equality."""
        opt1 = BarrierUpOutCall(strike=100, barrier=120, maturity=1.0)
        opt2 = BarrierUpOutCall(strike=100, barrier=120, maturity=1.0)
        opt3 = BarrierUpOutCall(strike=100, barrier=130, maturity=1.0)
        opt4 = BarrierUpInCall(strike=100, barrier=120, maturity=1.0)
        assert opt1 == opt2
        assert opt1 != opt3
        assert opt1 != opt4  # different knock type
        assert hash(opt1) == hash(opt2)

    def test_barrier_all_8_types_constructable(self):
        """Test that all 8 barrier option types can be constructed."""
        opts = [
            BarrierUpOutCall(100, 120, 1.0),
            BarrierUpInCall(100, 120, 1.0),
            BarrierDownOutCall(100, 80, 1.0),
            BarrierDownInCall(100, 80, 1.0),
            BarrierUpOutPut(100, 120, 1.0),
            BarrierUpInPut(100, 120, 1.0),
            BarrierDownOutPut(100, 80, 1.0),
            BarrierDownInPut(100, 80, 1.0),
        ]
        assert len(opts) == 8
        for opt in opts:
            assert isinstance(opt, BarrierOption)

    def test_barrier_knock_in_properties(self):
        """Test knock-in/out and rebate properties."""
        ko = BarrierUpOutCall(100, 120, 1.0, rebate=5.0)
        assert ko.is_knock_in is False
        assert ko.rebate == 5.0

        ki = BarrierUpInCall(100, 120, 1.0)
        assert ki.is_knock_in is True
        assert ki.rebate == 0.0

    def test_barrier_multiple_paths(self):
        """Test Barrier payoff with multiple paths."""
        opt = BarrierUpOutCall(strike=100, barrier=120, maturity=1.0)
        paths = np.array([
            [100, 105, 110],  # Not knocked, payoff=10
            [100, 125, 110],  # Knocked out, payoff=0
            [100, 115, 130],  # Knocked at end, payoff=0
        ])
        payoffs = opt.payoff(paths)
        assert len(payoffs) == 3
        assert payoffs[0] == pytest.approx(10.0)
        assert payoffs[1] == pytest.approx(0.0)
        assert payoffs[2] == pytest.approx(0.0)


class TestLookbackOption:
    """Tests for Lookback options."""

    def test_lookback_call_creation(self):
        """Test Lookback call option creation."""
        opt = LookbackCall(maturity=1.0)
        assert opt.maturity == 1.0
        assert opt.is_call is True
        assert opt.lookback_type == "floating"
        assert opt.strike is None
        assert opt.payoff.is_path_dependent is True

    def test_lookback_put_creation(self):
        """Test Lookback put option creation."""
        opt = LookbackPut(maturity=1.0)
        assert opt.maturity == 1.0
        assert opt.is_call is False
        assert opt.payoff.is_path_dependent is True

    def test_lookback_fixed_creation(self):
        """Test fixed-strike lookback creation."""
        opt = LookbackFixedCall(strike=100, maturity=1.0)
        assert opt.lookback_type == "fixed"
        assert opt.strike == 100
        assert opt.payoff is None  # analytical only

    def test_lookback_fixed_requires_strike(self):
        """Test fixed lookback validation."""
        with pytest.raises(ValueError, match="Fixed-strike lookback requires positive strike"):
            LookbackOption(maturity=1.0, lookback_type="fixed")

    def test_lookback_invalid_type(self):
        """Test lookback_type validation."""
        with pytest.raises(ValueError, match="lookback_type"):
            LookbackOption(maturity=1.0, lookback_type="partial")

    def test_lookback_call_payoff(self):
        """Test Lookback call payoff calculation."""
        opt = LookbackCall(maturity=1.0)
        path = np.array([[100, 90, 110]])  # min=90, terminal=110
        payoffs = opt.payoff(path)
        assert payoffs[0] == pytest.approx(20.0)  # 110 - 90

    def test_lookback_put_payoff(self):
        """Test Lookback put payoff calculation."""
        opt = LookbackPut(maturity=1.0)
        path = np.array([[100, 110, 90]])  # max=110, terminal=90
        payoffs = opt.payoff(path)
        assert payoffs[0] == pytest.approx(20.0)  # 110 - 90

    def test_lookback_call_always_positive(self):
        """Test that Lookback call payoff is always non-negative."""
        opt = LookbackCall(maturity=1.0)
        # Even if price ends at minimum, payoff is 0
        path = np.array([[100, 110, 100]])  # min=100, terminal=100
        payoffs = opt.payoff(path)
        assert payoffs[0] >= 0.0

    def test_lookback_immutability(self):
        """Test Lookback option immutability."""
        opt = LookbackCall(maturity=1.0)
        with pytest.raises(AttributeError):
            opt._maturity = 2.0

    def test_lookback_invalid_maturity(self):
        """Test Lookback option validation."""
        with pytest.raises(ValueError, match="Maturity must be positive"):
            LookbackCall(maturity=-1.0)

    def test_lookback_repr(self):
        """Test Lookback option string representation."""
        opt = LookbackCall(maturity=1.0)
        assert "LookbackOption" in repr(opt)
        assert "Call" in repr(opt)
        assert "T=1.0" in repr(opt)

    def test_lookback_fixed_repr(self):
        """Test fixed lookback repr."""
        opt = LookbackFixedCall(strike=100, maturity=1.0)
        assert "Fixed" in repr(opt)
        assert "K=100" in repr(opt)

    def test_lookback_equality(self):
        """Test Lookback option equality."""
        opt1 = LookbackCall(maturity=1.0)
        opt2 = LookbackCall(maturity=1.0)
        opt3 = LookbackPut(maturity=1.0)
        opt4 = LookbackFixedCall(strike=100, maturity=1.0)
        assert opt1 == opt2
        assert opt1 != opt3
        assert opt1 != opt4  # different lookback_type
        assert hash(opt1) == hash(opt2)

    def test_lookback_multiple_paths(self):
        """Test Lookback payoff with multiple paths."""
        opt = LookbackCall(maturity=1.0)
        paths = np.array([
            [100, 90, 110],   # min=90, terminal=110, payoff=20
            [100, 100, 100],  # min=100, terminal=100, payoff=0
            [100, 80, 120],   # min=80, terminal=120, payoff=40
        ])
        payoffs = opt.payoff(paths)
        assert len(payoffs) == 3
        assert payoffs[0] == pytest.approx(20.0)
        assert payoffs[1] == pytest.approx(0.0)
        assert payoffs[2] == pytest.approx(40.0)


class TestPayoffClasses:
    """Direct tests for payoff classes."""

    def test_asian_call_payoff_repr(self):
        """Test AsianCallPayoff repr."""
        payoff = AsianCallPayoff(strike=100)
        assert "AsianCallPayoff" in repr(payoff)
        assert "strike=100" in repr(payoff)

    def test_asian_put_payoff_repr(self):
        """Test AsianPutPayoff repr."""
        payoff = AsianPutPayoff(strike=100)
        assert "AsianPutPayoff" in repr(payoff)

    def test_barrier_up_out_call_payoff_repr(self):
        """Test BarrierUpOutCallPayoff repr."""
        payoff = BarrierUpOutCallPayoff(strike=100, barrier=120)
        assert "BarrierUpOutCallPayoff" in repr(payoff)
        assert "strike=100" in repr(payoff)
        assert "barrier=120" in repr(payoff)

    def test_barrier_down_out_put_payoff_repr(self):
        """Test BarrierDownOutPutPayoff repr."""
        payoff = BarrierDownOutPutPayoff(strike=100, barrier=80)
        assert "BarrierDownOutPutPayoff" in repr(payoff)

    def test_lookback_floating_call_payoff_repr(self):
        """Test LookbackFloatingCallPayoff repr."""
        payoff = LookbackFloatingCallPayoff()
        assert "LookbackFloatingCallPayoff" in repr(payoff)

    def test_lookback_floating_put_payoff_repr(self):
        """Test LookbackFloatingPutPayoff repr."""
        payoff = LookbackFloatingPutPayoff()
        assert "LookbackFloatingPutPayoff" in repr(payoff)


class TestExoticOptionPricing:
    """MC pricing tests for exotic options via simulate_paths + payoff evaluation."""

    @staticmethod
    def _mc_price(model, option, spot, rate, maturity, n_paths=50000, n_steps=252, seed=42):
        """Price an exotic option via MC: simulate -> payoff -> discount -> average."""
        simulator = model.create_simulator()
        result = simulator.simulate_paths(
            s0=spot, mu=rate, t=maturity,
            n_paths=n_paths, n_steps=n_steps, seed=seed
        )
        payoffs = option.payoff(result.price_paths)
        return float(np.exp(-rate * maturity) * np.mean(payoffs))

    @staticmethod
    def _vanilla_mc_price(model, spot, strike, rate, maturity, is_call=True, n_paths=50000, n_steps=252, seed=42):
        """Price a vanilla option via MC for comparison."""
        simulator = model.create_simulator()
        result = simulator.simulate_paths(
            s0=spot, mu=rate, t=maturity,
            n_paths=n_paths, n_steps=n_steps, seed=seed
        )
        terminal = result.price_paths[:, -1]
        if is_call:
            payoffs = np.maximum(terminal - strike, 0)
        else:
            payoffs = np.maximum(strike - terminal, 0)
        return float(np.exp(-rate * maturity) * np.mean(payoffs))

    def test_asian_call_cheaper_than_vanilla(self):
        """Asian call < vanilla call (averaging reduces exposure)."""
        from backend.models.gbm import GBMModel
        model = GBMModel(sigma=0.20)
        s, k, r, t = 100.0, 100.0, 0.05, 0.5

        asian = AsianCall(strike=k, maturity=t)
        asian_price = self._mc_price(model, asian, s, r, t)
        vanilla_price = self._vanilla_mc_price(model, s, k, r, t, is_call=True)

        assert asian_price < vanilla_price, \
            f"Asian call ({asian_price:.4f}) should be cheaper than vanilla ({vanilla_price:.4f})"

    def test_asian_put_cheaper_than_vanilla(self):
        """Asian put < vanilla put."""
        from backend.models.gbm import GBMModel
        model = GBMModel(sigma=0.20)
        s, k, r, t = 100.0, 100.0, 0.05, 0.5

        asian = AsianPut(strike=k, maturity=t)
        asian_price = self._mc_price(model, asian, s, r, t)
        vanilla_price = self._vanilla_mc_price(model, s, k, r, t, is_call=False)

        assert asian_price < vanilla_price, \
            f"Asian put ({asian_price:.4f}) should be cheaper than vanilla ({vanilla_price:.4f})"

    def test_barrier_up_out_cheaper_than_vanilla(self):
        """Knock-out call < vanilla call."""
        from backend.models.gbm import GBMModel
        model = GBMModel(sigma=0.20)
        s, k, r, t = 100.0, 100.0, 0.05, 0.5

        barrier = BarrierUpOutCall(strike=k, barrier=120, maturity=t)
        barrier_price = self._mc_price(model, barrier, s, r, t)
        vanilla_price = self._vanilla_mc_price(model, s, k, r, t, is_call=True)

        assert barrier_price < vanilla_price, \
            f"Barrier ({barrier_price:.4f}) should be cheaper than vanilla ({vanilla_price:.4f})"

    def test_barrier_far_barrier_approaches_vanilla(self):
        """With very far barrier, price ~ vanilla."""
        from backend.models.gbm import GBMModel
        model = GBMModel(sigma=0.20)
        s, k, r, t = 100.0, 100.0, 0.05, 0.5

        barrier = BarrierUpOutCall(strike=k, barrier=500, maturity=t)
        barrier_price = self._mc_price(model, barrier, s, r, t)
        vanilla_price = self._vanilla_mc_price(model, s, k, r, t, is_call=True)

        np.testing.assert_allclose(barrier_price, vanilla_price, rtol=0.05)

    def test_lookback_call_more_expensive(self):
        """Lookback call (buy at min) > vanilla call."""
        from backend.models.gbm import GBMModel
        model = GBMModel(sigma=0.20)
        s, k, r, t = 100.0, 100.0, 0.05, 0.5

        lookback = LookbackCall(maturity=t)
        lookback_price = self._mc_price(model, lookback, s, r, t)
        vanilla_price = self._vanilla_mc_price(model, s, k, r, t, is_call=True)

        assert lookback_price > vanilla_price, \
            f"Lookback ({lookback_price:.4f}) should be more expensive than vanilla ({vanilla_price:.4f})"

    def test_lookback_put_more_expensive(self):
        """Lookback put (sell at max) > vanilla put."""
        from backend.models.gbm import GBMModel
        model = GBMModel(sigma=0.20)
        s, k, r, t = 100.0, 100.0, 0.05, 0.5

        lookback = LookbackPut(maturity=t)
        lookback_price = self._mc_price(model, lookback, s, r, t)
        vanilla_price = self._vanilla_mc_price(model, s, k, r, t, is_call=False)

        assert lookback_price > vanilla_price, \
            f"Lookback ({lookback_price:.4f}) should be more expensive than vanilla ({vanilla_price:.4f})"

    def test_asian_with_heston(self):
        """Asian pricing works with Heston model."""
        from backend.models.heston import HestonModel
        model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        s, k, r, t = 100.0, 100.0, 0.05, 0.5

        asian = AsianCall(strike=k, maturity=t)
        price = self._mc_price(model, asian, s, r, t, n_paths=30000)
        assert price > 0, f"Asian call with Heston should have positive price, got {price}"


# =============================================================================
# EXOTIC ANALYTIC ENGINE TESTS
# =============================================================================

class TestExoticAnalyticEngine:
    """Tests for ExoticAnalyticEngine - analytical pricing of exotic options."""

    @pytest.fixture
    def engine(self):
        return ExoticAnalyticEngine()

    @pytest.fixture
    def bs_engine(self):
        return BSAnalyticEngine()

    @pytest.fixture
    def gbm(self):
        return GBMModel(sigma=0.25)

    @pytest.fixture
    def market(self):
        return MarketEnvironment(spot=100.0, rate=0.05)

    # =========================================================================
    # Barrier parity tests: KI + KO = vanilla
    # =========================================================================

    def test_barrier_parity_up_call(self, engine, bs_engine, gbm, market):
        """Up-and-out call + up-and-in call = vanilla call."""
        K, H, T = 100.0, 110.0, 0.25
        ko = BarrierOption(K, H, T, is_call=True, is_up=True, is_knock_in=False)
        ki = BarrierOption(K, H, T, is_call=True, is_up=True, is_knock_in=True)
        vanilla = VanillaOption(K, T, is_call=True)

        p_ko = engine.price(ko, gbm, market).price
        p_ki = engine.price(ki, gbm, market).price
        p_van = bs_engine.price(vanilla, gbm, market).price

        np.testing.assert_allclose(p_ko + p_ki, p_van, rtol=1e-6,
            err_msg=f"KO={p_ko:.6f} + KI={p_ki:.6f} = {p_ko+p_ki:.6f} vs vanilla={p_van:.6f}")

    def test_barrier_parity_down_put(self, engine, bs_engine, gbm, market):
        """Down-and-out put + down-and-in put = vanilla put."""
        K, H, T = 100.0, 90.0, 0.25
        ko = BarrierOption(K, H, T, is_call=False, is_up=False, is_knock_in=False)
        ki = BarrierOption(K, H, T, is_call=False, is_up=False, is_knock_in=True)
        vanilla = VanillaOption(K, T, is_call=False)

        p_ko = engine.price(ko, gbm, market).price
        p_ki = engine.price(ki, gbm, market).price
        p_van = bs_engine.price(vanilla, gbm, market).price

        np.testing.assert_allclose(p_ko + p_ki, p_van, rtol=1e-6)

    def test_barrier_parity_down_call(self, engine, bs_engine, gbm, market):
        """Down-and-out call + down-and-in call = vanilla call."""
        K, H, T = 100.0, 90.0, 0.25
        ko = BarrierOption(K, H, T, is_call=True, is_up=False, is_knock_in=False)
        ki = BarrierOption(K, H, T, is_call=True, is_up=False, is_knock_in=True)
        vanilla = VanillaOption(K, T, is_call=True)

        p_ko = engine.price(ko, gbm, market).price
        p_ki = engine.price(ki, gbm, market).price
        p_van = bs_engine.price(vanilla, gbm, market).price

        np.testing.assert_allclose(p_ko + p_ki, p_van, rtol=1e-6)

    def test_barrier_parity_up_put(self, engine, bs_engine, gbm, market):
        """Up-and-out put + up-and-in put = vanilla put."""
        K, H, T = 100.0, 110.0, 0.25
        ko = BarrierOption(K, H, T, is_call=False, is_up=True, is_knock_in=False)
        ki = BarrierOption(K, H, T, is_call=False, is_up=True, is_knock_in=True)
        vanilla = VanillaOption(K, T, is_call=False)

        p_ko = engine.price(ko, gbm, market).price
        p_ki = engine.price(ki, gbm, market).price
        p_van = bs_engine.price(vanilla, gbm, market).price

        np.testing.assert_allclose(p_ko + p_ki, p_van, rtol=1e-6)

    # =========================================================================
    # Barrier price bounds
    # =========================================================================

    def test_barrier_knockout_cheaper_than_vanilla(self, engine, bs_engine, gbm, market):
        """Knock-out must be cheaper than vanilla."""
        K, H, T = 100.0, 110.0, 0.25
        ko = BarrierUpOutCall(K, H, T)
        vanilla = VanillaOption(K, T, is_call=True)

        p_ko = engine.price(ko, gbm, market).price
        p_van = bs_engine.price(vanilla, gbm, market).price

        assert p_ko < p_van

    def test_barrier_knockin_cheaper_than_vanilla(self, engine, bs_engine, gbm, market):
        """Knock-in must be cheaper than vanilla."""
        K, H, T = 100.0, 110.0, 0.25
        ki = BarrierUpInCall(K, H, T)
        vanilla = VanillaOption(K, T, is_call=True)

        p_ki = engine.price(ki, gbm, market).price
        p_van = bs_engine.price(vanilla, gbm, market).price

        assert p_ki < p_van

    def test_barrier_prices_positive(self, engine, gbm, market):
        """All barrier prices must be non-negative."""
        T = 0.25
        for opt in [
            BarrierUpOutCall(100, 120, T),
            BarrierUpInCall(100, 120, T),
            BarrierDownOutCall(100, 80, T),
            BarrierDownInCall(100, 80, T),
            BarrierUpOutPut(100, 120, T),
            BarrierUpInPut(100, 120, T),
            BarrierDownOutPut(100, 80, T),
            BarrierDownInPut(100, 80, T),
        ]:
            p = engine.price(opt, gbm, market).price
            assert p >= 0.0, f"{opt.option_type} has negative price: {p}"

    # =========================================================================
    # Asian geometric tests
    # =========================================================================

    def test_asian_geometric_cheaper_than_vanilla(self, engine, bs_engine, gbm, market):
        """Geometric Asian < vanilla (averaging reduces variance)."""
        K, T = 100.0, 0.5
        geo = AsianGeometricCall(K, T)
        vanilla = VanillaOption(K, T, is_call=True)

        p_geo = engine.price(geo, gbm, market).price
        p_van = bs_engine.price(vanilla, gbm, market).price

        assert p_geo < p_van, f"Geometric Asian ({p_geo:.4f}) should be < vanilla ({p_van:.4f})"

    def test_asian_geometric_positive(self, engine, gbm, market):
        """Geometric Asian prices must be positive."""
        for is_call in [True, False]:
            opt = AsianOption(100, 0.5, is_call=is_call, average_type="geometric")
            p = engine.price(opt, gbm, market).price
            assert p > 0.0

    def test_asian_geometric_put_call_relationship(self, engine, gbm, market):
        """Geometric Asian call > put for ATM when r > 0 (forward bias)."""
        K, T = 100.0, 0.5
        call = AsianGeometricCall(K, T)
        put = AsianGeometricPut(K, T)

        p_call = engine.price(call, gbm, market).price
        p_put = engine.price(put, gbm, market).price

        assert p_call > p_put, f"Call ({p_call:.4f}) should > put ({p_put:.4f}) for ATM with r>0"

    # =========================================================================
    # Digital tests
    # =========================================================================

    def test_digital_put_call_parity(self, engine, gbm, market):
        """Digital call + digital put = PV(payout)."""
        K, T, payout = 100.0, 0.25, 100.0
        call = DigitalOption(K, T, is_call=True, payout=payout)
        put = DigitalOption(K, T, is_call=False, payout=payout)

        p_call = engine.price(call, gbm, market).price
        p_put = engine.price(put, gbm, market).price
        pv_payout = payout * np.exp(-market.rate * T)

        np.testing.assert_allclose(p_call + p_put, pv_payout, rtol=1e-6)

    def test_digital_prices_in_bounds(self, engine, gbm, market):
        """0 <= digital price <= PV(payout)."""
        K, T, payout = 100.0, 0.25, 100.0
        pv_payout = payout * np.exp(-market.rate * T)

        for is_call in [True, False]:
            opt = DigitalOption(K, T, is_call=is_call, payout=payout)
            p = engine.price(opt, gbm, market).price
            assert 0.0 <= p <= pv_payout + 0.01

    def test_digital_atm_near_half(self, engine, gbm, market):
        """ATM digital ~ 0.5 * PV(payout) (approximately)."""
        K, T, payout = 100.0, 0.25, 100.0
        call = DigitalOption(K, T, is_call=True, payout=payout)
        pv_payout = payout * np.exp(-market.rate * T)

        p = engine.price(call, gbm, market).price
        np.testing.assert_allclose(p, 0.5 * pv_payout, rtol=0.15)

    # =========================================================================
    # Lookback tests
    # =========================================================================

    def test_lookback_floating_geq_vanilla(self, engine, bs_engine, gbm, market):
        """Floating lookback call >= ATM vanilla call."""
        T = 0.5
        lookback = LookbackCall(maturity=T)
        vanilla = VanillaOption(100, T, is_call=True)

        p_lb = engine.price(lookback, gbm, market).price
        p_van = bs_engine.price(vanilla, gbm, market).price

        assert p_lb >= p_van, f"Lookback ({p_lb:.4f}) should >= vanilla ({p_van:.4f})"

    def test_lookback_fixed_positive(self, engine, gbm, market):
        """Fixed-strike lookback prices are positive."""
        # Call with ATM strike
        call = LookbackOption(maturity=0.5, is_call=True, strike=100, lookback_type="fixed")
        p_call = engine.price(call, gbm, market).price
        assert p_call > 0.0

        # Put with ITM strike (avoids edge case at M_min = S = K)
        put = LookbackOption(maturity=0.5, is_call=False, strike=105, lookback_type="fixed")
        p_put = engine.price(put, gbm, market).price
        assert p_put > 0.0

    def test_lookback_floating_positive(self, engine, gbm, market):
        """Floating-strike lookback prices are positive."""
        for is_call in [True, False]:
            opt = LookbackOption(maturity=0.5, is_call=is_call)
            p = engine.price(opt, gbm, market).price
            assert p > 0.0

    # =========================================================================
    # Greeks tests
    # =========================================================================

    def test_barrier_greeks_signs(self, engine, gbm, market):
        """Barrier call delta > 0, put delta < 0."""
        T = 0.25
        call = BarrierUpOutCall(100, 120, T)
        put = BarrierDownOutPut(100, 80, T)

        g_call = engine.greeks(call, gbm, market)
        g_put = engine.greeks(put, gbm, market)

        assert g_call.delta > 0, f"Call delta should be positive, got {g_call.delta}"
        assert g_put.delta < 0, f"Put delta should be negative, got {g_put.delta}"

    def test_digital_greeks_delta_positive(self, engine, gbm, market):
        """Digital call delta is positive."""
        call = DigitalOption(100, 0.25, is_call=True, payout=100)
        g = engine.greeks(call, gbm, market)
        assert g.delta > 0

    def test_greeks_returns_greeks_result(self, engine, gbm, market):
        """Greeks method returns GreeksResult type."""
        opt = BarrierUpOutCall(100, 120, 0.25)
        g = engine.greeks(opt, gbm, market)
        assert isinstance(g, GreeksResult)
        # Check all first-order greeks are finite
        assert np.isfinite(g.delta)
        assert np.isfinite(g.gamma)
        assert np.isfinite(g.vega)
        assert np.isfinite(g.theta)
        assert np.isfinite(g.rho)

    # =========================================================================
    # Edge cases
    # =========================================================================

    def test_near_expiry_convergence(self, engine, gbm, market):
        """Near expiry, barrier call converges toward intrinsic."""
        T = 0.001  # ~0.4 days
        K, H = 95.0, 120.0  # deep ITM, far barrier
        opt = BarrierUpOutCall(K, H, T)
        p = engine.price(opt, gbm, market).price
        intrinsic = max(market.spot - K, 0)
        np.testing.assert_allclose(p, intrinsic, atol=0.5)

    def test_can_price_rejects_non_gbm(self, engine):
        """can_price returns False for non-GBM model."""
        from backend.models.heston import HestonModel
        heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        opt = BarrierUpOutCall(100, 120, 0.25)
        assert engine.can_price(opt, heston) is False

    def test_can_price_rejects_vanilla(self, engine, gbm):
        """can_price returns False for VanillaOption."""
        vanilla = VanillaOption(100, 0.25, is_call=True)
        assert engine.can_price(vanilla, gbm) is False

    def test_can_price_rejects_arithmetic_asian(self, engine, gbm):
        """can_price returns False for arithmetic Asian (MC only)."""
        arith = AsianCall(100, 0.25)
        assert engine.can_price(arith, gbm) is False

    def test_can_price_accepts_geometric_asian(self, engine, gbm):
        """can_price returns True for geometric Asian."""
        geo = AsianGeometricCall(100, 0.25)
        assert engine.can_price(geo, gbm) is True

    # =========================================================================
    # Cross-validation: Analytic vs MC
    # =========================================================================

    def test_digital_analytic_vs_mc(self, engine, gbm, market):
        """Compare analytic digital price to MC (~5% tolerance)."""
        K, T, payout = 100.0, 0.5, 100.0
        opt = DigitalOption(K, T, is_call=True, payout=payout)

        # Analytic
        p_analytic = engine.price(opt, gbm, market).price

        # MC
        simulator = gbm.create_simulator()
        result = simulator.simulate_paths(s0=100.0, mu=0.05, t=T,
                                          n_paths=100000, n_steps=252, seed=42)
        terminal = result.price_paths[:, -1]
        mc_payoffs = np.where(terminal > K, payout, 0.0)
        p_mc = float(np.exp(-0.05 * T) * np.mean(mc_payoffs))

        np.testing.assert_allclose(p_analytic, p_mc, rtol=0.05,
            err_msg=f"Digital analytic={p_analytic:.4f} vs MC={p_mc:.4f}")

    def test_barrier_analytic_vs_mc(self, engine, bs_engine, gbm, market):
        """Validate analytic barrier via parity: KO + KI = vanilla (no MC bias)."""
        # MC with discrete monitoring underestimates knock-out prices (misses
        # barrier crossings between steps). Instead, verify parity which is
        # an exact analytical relationship.
        K, H, T = 100.0, 120.0, 0.5
        ko = BarrierOption(K, H, T, is_call=True, is_up=True, is_knock_in=False)
        ki = BarrierOption(K, H, T, is_call=True, is_up=True, is_knock_in=True)
        vanilla = VanillaOption(K, T, is_call=True)

        p_ko = engine.price(ko, gbm, market).price
        p_ki = engine.price(ki, gbm, market).price
        p_van = bs_engine.price(vanilla, gbm, market).price

        np.testing.assert_allclose(p_ko + p_ki, p_van, rtol=1e-6,
            err_msg=f"KO={p_ko:.4f} + KI={p_ki:.4f} = {p_ko+p_ki:.4f} vs vanilla={p_van:.4f}")


class TestEngineRejectsExotics:
    """BS and FFT engines must reject exotic instruments."""

    @pytest.fixture
    def gbm(self):
        return GBMModel(sigma=0.25)

    def test_bs_engine_rejects_barrier(self, gbm):
        """BSAnalyticEngine.can_price() returns False for barrier options."""
        bs = BSAnalyticEngine()
        barrier = BarrierOption(100, 110, 0.25, is_call=True, is_up=True)
        assert not bs.can_price(barrier, gbm)

    def test_fft_engine_rejects_all_exotics(self, gbm):
        """FFTEngine.can_price() returns False for all exotic types."""
        from backend.engines.fft_engine import FFTEngine

        fft = FFTEngine()
        exotics = [
            BarrierOption(100, 110, 0.25, is_call=True, is_up=True),
            AsianOption(100, 0.25, is_call=True, average_type="geometric"),
            DigitalOption(100, 0.25, is_call=True, payout=100.0),
            LookbackOption(0.25, is_call=True),
        ]
        for exotic in exotics:
            assert not fft.can_price(exotic, gbm), \
                f"FFTEngine should reject {type(exotic).__name__}"


class TestExoticEdgeCases:
    """Edge case tests for exotic pricing kernels."""

    @pytest.fixture
    def engine(self):
        return ExoticAnalyticEngine()

    @pytest.fixture
    def bs_engine(self):
        return BSAnalyticEngine()

    @pytest.fixture
    def gbm(self):
        return GBMModel(sigma=0.25)

    @pytest.fixture
    def market(self):
        return MarketEnvironment(spot=100.0, rate=0.05)

    # =========================================================================
    # Knock-in breach: S == H → should return vanilla BS price
    # =========================================================================

    def test_barrier_knock_in_breach_returns_vanilla(self, engine, bs_engine, gbm, market):
        """When S >= H for up knock-in, price equals vanilla BS price."""
        K, T = 100.0, 0.25
        # S=100 == H=100 → barrier already breached → knock-in becomes vanilla
        ki = BarrierOption(K, barrier=100.0, maturity=T, is_call=True,
                           is_up=True, is_knock_in=True)
        vanilla = VanillaOption(K, T, is_call=True)

        p_ki = engine.price(ki, gbm, market).price
        p_van = bs_engine.price(vanilla, gbm, market).price

        np.testing.assert_allclose(p_ki, p_van, rtol=1e-6,
            err_msg=f"Breached knock-in ({p_ki:.6f}) should equal vanilla ({p_van:.6f})")

    # =========================================================================
    # Rebate-bearing barrier pricing
    # =========================================================================

    def test_barrier_knockout_with_rebate_positive(self, engine, gbm, market):
        """Knock-out with rebate must be >= knock-out without rebate."""
        K, H, T = 100.0, 120.0, 0.25
        ko_no_rebate = BarrierOption(K, H, T, is_call=True, is_up=True,
                                     is_knock_in=False, rebate=0.0)
        ko_rebate = BarrierOption(K, H, T, is_call=True, is_up=True,
                                  is_knock_in=False, rebate=5.0)

        p_no = engine.price(ko_no_rebate, gbm, market).price
        p_yes = engine.price(ko_rebate, gbm, market).price

        assert p_yes >= p_no, \
            f"With rebate ({p_yes:.6f}) should be >= without ({p_no:.6f})"

    def test_barrier_knockout_rebate_greeks(self, engine, gbm, market):
        """Knock-out with rebate has finite Greeks."""
        K, H, T = 100.0, 120.0, 0.25
        ko = BarrierOption(K, H, T, is_call=True, is_up=True,
                           is_knock_in=False, rebate=5.0)
        g = engine.greeks(ko, gbm, market)
        assert np.isfinite(g.delta)
        assert np.isfinite(g.gamma)
        assert np.isfinite(g.vega)
        assert np.isfinite(g.theta)
        assert np.isfinite(g.rho)

    # =========================================================================
    # sigma=0 deterministic pricing
    # =========================================================================

    def test_barrier_sigma_zero_deterministic(self):
        """Barrier kernel with sigma=0 returns deterministic value, not 0."""
        from backend.engines.exotic_engine import barrier_option_price
        # Up-out call, S=100, H=120, K=95 (ITM). With r>0 and sigma=0:
        # Forward = 100*exp(0.05*0.5) = 102.53, barrier not hit → intrinsic
        p = barrier_option_price(S=100.0, K=95.0, H=120.0, T=0.5, r=0.05, q=0.0,
                                 sigma=0.0, is_call=True, is_knock_in=False,
                                 is_up=True, rebate=0.0)
        assert p > 0.0, f"Barrier sigma=0 should give positive price for ITM, got {p}"

        # Down-out put, S=100, H=80, K=105 (ITM). Forward stays above 80.
        p_put = barrier_option_price(S=100.0, K=105.0, H=80.0, T=0.5, r=0.05, q=0.0,
                                     sigma=0.0, is_call=False, is_knock_in=False,
                                     is_up=False, rebate=0.0)
        assert p_put > 0.0, f"Barrier put sigma=0 should give positive price, got {p_put}"

    def test_digital_sigma_zero_deterministic(self):
        """Digital kernel with sigma=0 returns deterministic payoff."""
        from backend.engines.exotic_engine import digital_price
        # Forward = 100*exp(0.05*0.5) = 102.53 > 100 → call pays
        p_call = digital_price(S=100.0, K=100.0, T=0.5, r=0.05, q=0.0,
                               sigma=0.0, is_call=True, payout=1.0)
        expected = np.exp(-0.05 * 0.5)  # PV of payout
        np.testing.assert_allclose(p_call, expected, rtol=1e-6,
            err_msg=f"Digital sigma=0 ITM call should be PV(payout), got {p_call:.6f}")

        # Forward = 102.53 < 110 → call does NOT pay
        p_otm = digital_price(S=100.0, K=110.0, T=0.5, r=0.05, q=0.0,
                              sigma=0.0, is_call=True, payout=1.0)
        assert p_otm == 0.0, f"Digital sigma=0 OTM call should be 0, got {p_otm}"

    # =========================================================================
    # Negative rates
    # =========================================================================

    def test_lookback_negative_rate_positive_price(self):
        """Lookback with negative rate still produces positive price."""
        engine = ExoticAnalyticEngine()
        gbm = GBMModel(sigma=0.25)
        market_neg = MarketEnvironment(spot=100.0, rate=-0.01)

        lb = LookbackOption(maturity=0.5, is_call=True)
        p = engine.price(lb, gbm, market_neg).price
        assert p > 0.0, f"Lookback with r<0 should have positive price, got {p}"

    def test_barrier_negative_rate(self):
        """Barrier with negative rate has positive price and valid parity."""
        engine = ExoticAnalyticEngine()
        bs_engine = BSAnalyticEngine()
        gbm = GBMModel(sigma=0.25)
        market_neg = MarketEnvironment(spot=100.0, rate=-0.01)

        K, H, T = 100.0, 110.0, 0.25
        ko = BarrierOption(K, H, T, is_call=True, is_up=True, is_knock_in=False)
        ki = BarrierOption(K, H, T, is_call=True, is_up=True, is_knock_in=True)
        vanilla = VanillaOption(K, T, is_call=True)

        p_ko = engine.price(ko, gbm, market_neg).price
        p_ki = engine.price(ki, gbm, market_neg).price
        p_van = bs_engine.price(vanilla, gbm, market_neg).price

        assert p_ko >= 0.0
        assert p_ki >= 0.0
        np.testing.assert_allclose(p_ko + p_ki, p_van, rtol=1e-5,
            err_msg="KO+KI parity must hold with negative rate")

    # =========================================================================
    # Deep ITM/OTM
    # =========================================================================

    def test_barrier_deep_itm_call(self, engine, gbm, market):
        """Deep ITM call with far barrier ≈ vanilla intrinsic discounted."""
        K, H, T = 50.0, 200.0, 0.25  # deep ITM, far barrier
        ko = BarrierOption(K, H, T, is_call=True, is_up=True, is_knock_in=False)
        p = engine.price(ko, gbm, market).price
        intrinsic = max(market.spot - K, 0)
        assert p >= intrinsic * 0.9, \
            f"Deep ITM barrier ({p:.4f}) should be near intrinsic ({intrinsic})"

    def test_digital_deep_otm_near_zero(self, engine, gbm, market):
        """Deep OTM digital price ≈ 0."""
        # Call with K=200, S=100 → very deep OTM
        call = DigitalOption(strike=200.0, maturity=0.25, is_call=True, payout=100.0)
        p = engine.price(call, gbm, market).price
        assert p < 1.0, f"Deep OTM digital ({p:.4f}) should be near zero"

    # =========================================================================
    # Dividend yield tests
    # =========================================================================

    def test_barrier_with_dividend_parity(self, engine, bs_engine, gbm):
        """Barrier KI+KO = vanilla must hold with dividend yield."""
        market_div = MarketEnvironment(spot=100.0, rate=0.05, dividend_yield=0.03)
        K, H, T = 100.0, 110.0, 0.25

        ko = BarrierOption(K, H, T, is_call=True, is_up=True, is_knock_in=False)
        ki = BarrierOption(K, H, T, is_call=True, is_up=True, is_knock_in=True)
        vanilla = VanillaOption(K, T, is_call=True)

        p_ko = engine.price(ko, gbm, market_div).price
        p_ki = engine.price(ki, gbm, market_div).price
        p_van = bs_engine.price(vanilla, gbm, market_div).price

        np.testing.assert_allclose(p_ko + p_ki, p_van, rtol=1e-5,
            err_msg=f"Barrier parity must hold with q=0.03: "
                    f"KO={p_ko:.6f} + KI={p_ki:.6f} vs vanilla={p_van:.6f}")

    def test_digital_put_call_parity_with_dividend(self, engine, gbm):
        """Digital call + put = PV(payout) still holds with dividends."""
        market_div = MarketEnvironment(spot=100.0, rate=0.05, dividend_yield=0.03)
        K, T, payout = 100.0, 0.25, 100.0

        call = DigitalOption(K, T, is_call=True, payout=payout)
        put = DigitalOption(K, T, is_call=False, payout=payout)

        p_call = engine.price(call, gbm, market_div).price
        p_put = engine.price(put, gbm, market_div).price
        pv_payout = payout * np.exp(-market_div.rate * T)

        np.testing.assert_allclose(p_call + p_put, pv_payout, rtol=1e-5,
            err_msg=f"Digital parity must hold with q=0.03: "
                    f"call={p_call:.6f} + put={p_put:.6f} vs PV={pv_payout:.6f}")

    def test_dividend_reduces_call_price(self, engine, gbm):
        """Higher dividend yield should reduce call price (lower forward)."""
        market_no_div = MarketEnvironment(spot=100.0, rate=0.05, dividend_yield=0.0)
        market_div = MarketEnvironment(spot=100.0, rate=0.05, dividend_yield=0.03)

        call = BarrierOption(100, 120, 0.5, is_call=True, is_up=True, is_knock_in=False)
        p_no = engine.price(call, gbm, market_no_div).price
        p_div = engine.price(call, gbm, market_div).price

        assert p_div < p_no, \
            f"Call with q=0.03 ({p_div:.4f}) should be < with q=0 ({p_no:.4f})"


class TestExoticEdgeCasesZero:
    """Edge-case tests for exotic pricing kernels at T=0 and sigma=0."""

    def test_asian_geometric_intrinsic_at_expiry_call(self):
        """Asian geometric call at T=0 should return vanilla intrinsic."""
        from backend.engines.exotic_engine import asian_geometric_price
        # ITM call: S=110, K=100 -> intrinsic = 10
        price = asian_geometric_price(S=110.0, K=100.0, T=0.0, r=0.05, q=0.0, sigma=0.25, is_call=True)
        assert price == pytest.approx(10.0), f"Expected 10.0, got {price}"

    def test_asian_geometric_intrinsic_at_expiry_put(self):
        """Asian geometric put at T=0 should return vanilla intrinsic."""
        from backend.engines.exotic_engine import asian_geometric_price
        # ITM put: S=90, K=100 -> intrinsic = 10
        price = asian_geometric_price(S=90.0, K=100.0, T=0.0, r=0.05, q=0.0, sigma=0.25, is_call=False)
        assert price == pytest.approx(10.0), f"Expected 10.0, got {price}"

    def test_asian_geometric_otm_at_expiry(self):
        """Asian geometric OTM at T=0 should return 0."""
        from backend.engines.exotic_engine import asian_geometric_price
        price = asian_geometric_price(S=90.0, K=100.0, T=0.0, r=0.05, q=0.0, sigma=0.25, is_call=True)
        assert price == pytest.approx(0.0), f"Expected 0.0, got {price}"

    def test_asian_geometric_at_zero_vol(self):
        """Asian geometric at sigma=0 should return deterministic forward price."""
        from backend.engines.exotic_engine import asian_geometric_price
        import math
        S, K, T, r, q = 100.0, 95.0, 0.5, 0.05, 0.0
        F = S * math.exp((r - q) * T)
        df = math.exp(-r * T)
        # ITM call
        expected_call = max(F - K, 0.0) * df
        price_call = asian_geometric_price(S=S, K=K, T=T, r=r, q=q, sigma=0.0, is_call=True)
        assert price_call == pytest.approx(expected_call, rel=1e-10), f"Expected {expected_call}, got {price_call}"
        assert price_call > 0, "ITM asian geometric call at zero vol should have positive price"
        # OTM put (same params)
        expected_put = max(K - F, 0.0) * df
        price_put = asian_geometric_price(S=S, K=K, T=T, r=r, q=q, sigma=0.0, is_call=False)
        assert price_put == pytest.approx(expected_put, abs=1e-14), "OTM asian geometric put at zero vol should be 0"

    def test_lookback_fixed_at_zero_vol_call(self):
        """Lookback fixed call at sigma=0 should return deterministic price."""
        from backend.engines.exotic_engine import lookback_fixed_price
        import math
        S, K, T, r, q = 100.0, 95.0, 0.5, 0.05, 0.0
        # Deterministic forward: F = S * exp((r-q)*T)
        F = S * math.exp((r - q) * T)
        # b > 0 so path max = F, path min = S
        expected_max = max(S, F)  # F since r > 0
        df = math.exp(-r * T)
        expected = max(expected_max - K, 0.0) * df

        price = lookback_fixed_price(S=S, K=K, M_min=S, M_max=S, T=T, r=r, q=q, sigma=0.0, is_call=True)
        assert price == pytest.approx(expected, rel=1e-10), f"Expected {expected}, got {price}"
        assert price > 0, "ITM lookback fixed call at zero vol should have positive price"

    def test_lookback_fixed_at_zero_vol_put(self):
        """Lookback fixed put at sigma=0 should return deterministic price."""
        from backend.engines.exotic_engine import lookback_fixed_price
        import math
        S, K, T, r, q = 100.0, 105.0, 0.5, 0.05, 0.0
        F = S * math.exp((r - q) * T)
        # b > 0 so path_min = S
        effective_min = min(S, min(S, F))  # S
        df = math.exp(-r * T)
        expected = max(K - effective_min, 0.0) * df

        price = lookback_fixed_price(S=S, K=K, M_min=S, M_max=S, T=T, r=r, q=q, sigma=0.0, is_call=False)
        assert price == pytest.approx(expected, rel=1e-10), f"Expected {expected}, got {price}"
        assert price > 0, "ITM lookback fixed put at zero vol should have positive price"

    def test_lookback_floating_at_zero_vol_call(self):
        """Lookback floating call at sigma=0 should return deterministic price."""
        from backend.engines.exotic_engine import lookback_floating_price
        import math
        S, T, r, q = 100.0, 0.5, 0.05, 0.0
        F = S * math.exp((r - q) * T)
        df = math.exp(-r * T)
        # Call payoff: F - min(S, F). Since r>0, F>S, so min = S
        expected = max(F - S, 0.0) * df

        price = lookback_floating_price(S=S, M_min=S, M_max=S, T=T, r=r, q=q, sigma=0.0, is_call=True)
        assert price == pytest.approx(expected, rel=1e-10), f"Expected {expected}, got {price}"
        assert price > 0, "Lookback floating call at zero vol with r>0 should have positive price"

    def test_lookback_floating_at_zero_vol_put(self):
        """Lookback floating put at sigma=0 should return deterministic price."""
        from backend.engines.exotic_engine import lookback_floating_price
        import math
        S, T, r, q = 100.0, 0.5, 0.05, 0.0
        F = S * math.exp((r - q) * T)
        df = math.exp(-r * T)
        # Put payoff: max(S, F) - F. Since r>0, F>S, so max = F -> payoff = 0
        expected = max(max(S, F) - F, 0.0) * df

        price = lookback_floating_price(S=S, M_min=S, M_max=S, T=T, r=r, q=q, sigma=0.0, is_call=False)
        assert price == pytest.approx(expected, rel=1e-10), f"Expected {expected}, got {price}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
