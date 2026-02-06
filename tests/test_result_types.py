"""
Tests for Result Types
======================

Tests for PricingResult and GreeksResult dataclasses.

Author: Thomas
Created: 2025
"""

import pytest

from backend.core.result_types import GreeksResult, PricingResult, PricingCapability


class TestGreeksResult:
    """Tests for GreeksResult dataclass."""

    def test_addition(self):
        """Test adding two GreeksResult objects."""
        g1 = GreeksResult(delta=0.5, gamma=0.02, theta=-0.05, vega=0.20, rho=0.15)
        g2 = GreeksResult(delta=0.3, gamma=0.01, theta=-0.03, vega=0.10, rho=0.10)
        combined = g1 + g2

        assert combined.delta == pytest.approx(0.8)
        assert combined.gamma == pytest.approx(0.03)
        assert combined.theta == pytest.approx(-0.08)
        assert combined.vega == pytest.approx(0.30)
        assert combined.rho == pytest.approx(0.25)

    def test_subtraction(self):
        """Test subtracting GreeksResult objects for hedging."""
        g1 = GreeksResult(delta=0.5, gamma=0.02, theta=-0.05, vega=0.20, rho=0.15)
        g2 = GreeksResult(delta=0.3, gamma=0.01, theta=-0.03, vega=0.10, rho=0.10)
        diff = g1 - g2

        assert diff.delta == pytest.approx(0.2)
        assert diff.gamma == pytest.approx(0.01)
        assert diff.theta == pytest.approx(-0.02)
        assert diff.vega == pytest.approx(0.10)
        assert diff.rho == pytest.approx(0.05)

    def test_multiplication(self):
        """Test scaling GreeksResult by a scalar."""
        g = GreeksResult(delta=0.5, gamma=0.02, theta=-0.05, vega=0.20, rho=0.15)
        scaled = g * 10

        assert scaled.delta == pytest.approx(5.0)
        assert scaled.gamma == pytest.approx(0.2)
        assert scaled.theta == pytest.approx(-0.5)
        assert scaled.vega == pytest.approx(2.0)
        assert scaled.rho == pytest.approx(1.5)

    def test_right_multiplication(self):
        """Test right multiplication (scalar * GreeksResult)."""
        g = GreeksResult(delta=0.5, gamma=0.02)
        scaled = 10 * g

        assert scaled.delta == pytest.approx(5.0)
        assert scaled.gamma == pytest.approx(0.2)

    def test_division(self):
        """Test dividing GreeksResult by a scalar for normalization."""
        g = GreeksResult(delta=0.5, gamma=0.02, theta=-0.05, vega=0.20, rho=0.15)
        normalized = g / 10

        assert normalized.delta == pytest.approx(0.05)
        assert normalized.gamma == pytest.approx(0.002)
        assert normalized.theta == pytest.approx(-0.005)
        assert normalized.vega == pytest.approx(0.02)
        assert normalized.rho == pytest.approx(0.015)

    def test_negation(self):
        """Test negating GreeksResult for short positions."""
        g = GreeksResult(delta=0.5, gamma=0.02, theta=-0.05, vega=0.20, rho=0.15)
        short = -g

        assert short.delta == pytest.approx(-0.5)
        assert short.gamma == pytest.approx(-0.02)
        assert short.theta == pytest.approx(0.05)
        assert short.vega == pytest.approx(-0.20)
        assert short.rho == pytest.approx(-0.15)

    def test_second_order_greeks(self):
        """Test operations include second order Greeks."""
        g1 = GreeksResult(vanna=0.03, volga=0.01, charm=-0.001, veta=-0.02)
        g2 = GreeksResult(vanna=0.02, volga=0.005, charm=-0.0005, veta=-0.01)

        diff = g1 - g2
        assert diff.vanna == pytest.approx(0.01)
        assert diff.volga == pytest.approx(0.005)
        assert diff.charm == pytest.approx(-0.0005)
        assert diff.veta == pytest.approx(-0.01)

        neg = -g1
        assert neg.vanna == pytest.approx(-0.03)
        assert neg.volga == pytest.approx(-0.01)

    def test_third_order_greeks(self):
        """Test operations include third order Greeks."""
        g = GreeksResult(speed=0.001, zomma=0.0005, color=-0.0001, ultima=0.0003)

        neg = -g
        assert neg.speed == pytest.approx(-0.001)
        assert neg.zomma == pytest.approx(-0.0005)
        assert neg.color == pytest.approx(0.0001)
        assert neg.ultima == pytest.approx(-0.0003)

        divided = g / 2
        assert divided.speed == pytest.approx(0.0005)
        assert divided.zomma == pytest.approx(0.00025)

    def test_hedging_example(self):
        """Real-world hedging example: portfolio - hedge = residual."""
        portfolio = GreeksResult(delta=100, gamma=5, vega=50)
        hedge = GreeksResult(delta=95, gamma=4.8, vega=48)

        residual = portfolio - hedge
        assert residual.delta == pytest.approx(5)
        assert residual.gamma == pytest.approx(0.2)
        assert residual.vega == pytest.approx(2)

    def test_per_unit_example(self):
        """Real-world example: total Greeks / notional = per-unit Greeks."""
        total_greeks = GreeksResult(delta=1000, gamma=50, vega=200)
        notional = 100

        per_unit = total_greeks / notional
        assert per_unit.delta == pytest.approx(10)
        assert per_unit.gamma == pytest.approx(0.5)
        assert per_unit.vega == pytest.approx(2)

    def test_short_position_example(self):
        """Real-world example: short position has opposite Greeks."""
        long = GreeksResult(delta=0.5, gamma=0.02, theta=-0.05)
        short = -long

        # Short position: profit when underlying falls
        assert short.delta == pytest.approx(-0.5)
        # Short position: negative gamma (losses accelerate on large moves)
        assert short.gamma == pytest.approx(-0.02)
        # Short position: positive theta (collect premium decay)
        assert short.theta == pytest.approx(0.05)

    def test_aliases(self):
        """Test that aliases work correctly."""
        g = GreeksResult(volga=0.01, charm=-0.001)

        assert g.vomma == g.volga
        assert g.delta_decay == g.charm


class TestPricingResult:
    """Tests for PricingResult dataclass."""

    def test_basic_creation(self):
        """Test basic PricingResult creation."""
        result = PricingResult(price=5.123, engine="BS", model="GBM")

        assert result.price == pytest.approx(5.123)
        assert result.engine == "BS"
        assert result.model == "GBM"
        assert result.error is None

    def test_with_error(self):
        """Test PricingResult with Monte Carlo error."""
        result = PricingResult(price=5.123, engine="MC", model="Heston", error=0.01)

        assert result.price == pytest.approx(5.123)
        assert result.error == pytest.approx(0.01)

    def test_repr(self):
        """Test string representation."""
        result = PricingResult(price=5.123456)
        assert "5.123456" in repr(result)


class TestPricingCapability:
    """Tests for PricingCapability enum."""

    def test_all_capabilities_exist(self):
        """Verify all expected capabilities exist."""
        assert hasattr(PricingCapability, 'ANALYTICAL')
        assert hasattr(PricingCapability, 'FFT')
        assert hasattr(PricingCapability, 'MONTE_CARLO')

    def test_capabilities_are_unique(self):
        """Verify capabilities have unique values."""
        values = [cap.value for cap in PricingCapability]
        assert len(values) == len(set(values))

    def test_single_source_of_truth(self):
        """Ensure there's only one PricingCapability in the codebase."""
        # Import from the canonical location
        from backend.core.result_types import PricingCapability as PC1

        # Import via models (should be re-exported)
        from backend.models import PricingCapability as PC2

        # They should be the same enum
        assert PC1.ANALYTICAL is PC2.ANALYTICAL
        assert PC1.FFT is PC2.FFT
        assert PC1.MONTE_CARLO is PC2.MONTE_CARLO


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
