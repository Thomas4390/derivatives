"""
Tests for Exotic Options
========================

Test suite for Asian, Barrier, and Lookback options.

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
    AsianCallPayoff,
    AsianPutPayoff,
    # Barrier
    BarrierOption,
    BarrierUpOutCall,
    BarrierDownOutPut,
    BarrierUpOutCallPayoff,
    BarrierDownOutPutPayoff,
    # Lookback
    LookbackOption,
    LookbackCall,
    LookbackPut,
    LookbackFloatingCallPayoff,
    LookbackFloatingPutPayoff,
)


class TestAsianOption:
    """Tests for Asian options."""

    def test_asian_call_creation(self):
        """Test Asian call option creation."""
        opt = AsianCall(strike=100, maturity=1.0)
        assert opt.strike == 100
        assert opt.maturity == 1.0
        assert opt.is_call is True
        assert opt.payoff.is_path_dependent is True

    def test_asian_put_creation(self):
        """Test Asian put option creation."""
        opt = AsianPut(strike=100, maturity=1.0)
        assert opt.strike == 100
        assert opt.maturity == 1.0
        assert opt.is_call is False
        assert opt.payoff.is_path_dependent is True

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

    def test_asian_repr(self):
        """Test Asian option string representation."""
        opt = AsianCall(strike=100, maturity=1.0)
        assert "AsianOption" in repr(opt)
        assert "Call" in repr(opt)
        assert "K=100" in repr(opt)

    def test_asian_equality(self):
        """Test Asian option equality."""
        opt1 = AsianCall(strike=100, maturity=1.0)
        opt2 = AsianCall(strike=100, maturity=1.0)
        opt3 = AsianCall(strike=105, maturity=1.0)
        assert opt1 == opt2
        assert opt1 != opt3
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

    def test_barrier_equality(self):
        """Test Barrier option equality."""
        opt1 = BarrierUpOutCall(strike=100, barrier=120, maturity=1.0)
        opt2 = BarrierUpOutCall(strike=100, barrier=120, maturity=1.0)
        opt3 = BarrierUpOutCall(strike=100, barrier=130, maturity=1.0)
        assert opt1 == opt2
        assert opt1 != opt3
        assert hash(opt1) == hash(opt2)

    def test_barrier_unsupported_combination(self):
        """Test that unsupported barrier combinations raise error at construction."""
        # Up-out put is not supported - now fails fast at construction
        with pytest.raises(ValueError, match="Unsupported barrier option combination"):
            BarrierOption(strike=100, barrier=120, maturity=1.0, is_call=False, is_up=True)

        # Down-out call is not supported either
        with pytest.raises(ValueError, match="Unsupported barrier option combination"):
            BarrierOption(strike=100, barrier=80, maturity=1.0, is_call=True, is_up=False)

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
        assert opt.payoff.is_path_dependent is True

    def test_lookback_put_creation(self):
        """Test Lookback put option creation."""
        opt = LookbackPut(maturity=1.0)
        assert opt.maturity == 1.0
        assert opt.is_call is False
        assert opt.payoff.is_path_dependent is True

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

    def test_lookback_equality(self):
        """Test Lookback option equality."""
        opt1 = LookbackCall(maturity=1.0)
        opt2 = LookbackCall(maturity=1.0)
        opt3 = LookbackPut(maturity=1.0)
        assert opt1 == opt2
        assert opt1 != opt3
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
        """Price an exotic option via MC: simulate → payoff → discount → average."""
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
        """With very far barrier, price ≈ vanilla."""
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
