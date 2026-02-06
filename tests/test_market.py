"""
Tests for Market Environment
============================

Tests for MarketEnvironment validation and functionality.

Author: Thomas
Created: 2025
"""

import pytest

from backend.core import MarketEnvironment


class TestMarketEnvironmentValidation:
    """Tests for MarketEnvironment parameter validation."""

    def test_valid_market(self):
        """Test creation with valid parameters."""
        market = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.02)

        assert market.spot == 100
        assert market.rate == 0.05
        assert market.dividend_yield == 0.02

    def test_spot_must_be_positive(self):
        """Test that spot price must be positive."""
        with pytest.raises(ValueError, match="Spot must be positive"):
            MarketEnvironment(spot=0, rate=0.05)

        with pytest.raises(ValueError, match="Spot must be positive"):
            MarketEnvironment(spot=-100, rate=0.05)

    def test_rate_upper_bound_validation(self):
        """Test that unreasonably high rates are rejected."""
        with pytest.raises(ValueError, match="Rate seems unreasonable"):
            MarketEnvironment(spot=100, rate=5.0)  # 500% rate

        with pytest.raises(ValueError, match="Rate seems unreasonable"):
            MarketEnvironment(spot=100, rate=1.5)  # 150% rate

    def test_rate_lower_bound_validation(self):
        """Test that unreasonably low rates are rejected."""
        with pytest.raises(ValueError, match="Rate seems unreasonable"):
            MarketEnvironment(spot=100, rate=-1.0)  # -100% rate

        with pytest.raises(ValueError, match="Rate seems unreasonable"):
            MarketEnvironment(spot=100, rate=-0.6)  # -60% rate

    def test_rate_at_bounds(self):
        """Test rates at boundary values are accepted."""
        # Upper bound: 100%
        market_high = MarketEnvironment(spot=100, rate=1.0)
        assert market_high.rate == 1.0

        # Lower bound: -50%
        market_low = MarketEnvironment(spot=100, rate=-0.5)
        assert market_low.rate == -0.5

    def test_dividend_upper_bound_validation(self):
        """Test that unreasonably high dividend yields are rejected."""
        with pytest.raises(ValueError, match="Dividend yield seems unreasonable"):
            MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.5)  # 50%

        with pytest.raises(ValueError, match="Dividend yield seems unreasonable"):
            MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.25)  # 25%

    def test_dividend_lower_bound_validation(self):
        """Test that unreasonably negative dividend yields are rejected."""
        with pytest.raises(ValueError, match="Dividend yield seems unreasonable"):
            MarketEnvironment(spot=100, rate=0.05, dividend_yield=-0.5)  # -50%

        with pytest.raises(ValueError, match="Dividend yield seems unreasonable"):
            MarketEnvironment(spot=100, rate=0.05, dividend_yield=-0.15)  # -15%

    def test_dividend_at_bounds(self):
        """Test dividend yields at boundary values are accepted."""
        # Upper bound: 20%
        market_high = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.2)
        assert market_high.dividend_yield == 0.2

        # Lower bound: -10%
        market_low = MarketEnvironment(spot=100, rate=0.05, dividend_yield=-0.1)
        assert market_low.dividend_yield == -0.1


class TestMarketEnvironmentBypassValidation:
    """Tests for bypassing validation with with_rate() and with_dividend()."""

    def test_with_rate_bypasses_validation(self):
        """Test that with_rate() can set extreme values."""
        market = MarketEnvironment(spot=100, rate=0.05)

        # This should NOT raise - with_rate bypasses validation by default
        extreme_market = market.with_rate(5.0)
        assert extreme_market.rate == 5.0

        # Negative extreme
        negative_market = market.with_rate(-1.0)
        assert negative_market.rate == -1.0

    def test_with_rate_can_validate(self):
        """Test that with_rate() can optionally validate."""
        market = MarketEnvironment(spot=100, rate=0.05)

        # With validation, extreme value should raise
        with pytest.raises(ValueError, match="Rate seems unreasonable"):
            market.with_rate(5.0, validate=True)

    def test_with_dividend_bypasses_validation(self):
        """Test that with_dividend() can set extreme values."""
        market = MarketEnvironment(spot=100, rate=0.05)

        # This should NOT raise - with_dividend bypasses validation by default
        extreme_market = market.with_dividend(0.5)
        assert extreme_market.dividend_yield == 0.5

        # Negative extreme
        negative_market = market.with_dividend(-0.5)
        assert negative_market.dividend_yield == -0.5

    def test_with_dividend_can_validate(self):
        """Test that with_dividend() can optionally validate."""
        market = MarketEnvironment(spot=100, rate=0.05)

        # With validation, extreme value should raise
        with pytest.raises(ValueError, match="Dividend yield seems unreasonable"):
            market.with_dividend(0.5, validate=True)


class TestMarketEnvironmentBumping:
    """Tests for bump methods."""

    def test_bump_spot(self):
        """Test bump_spot creates correct new environment."""
        market = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.02)
        bumped = market.bump_spot(5.0)

        assert bumped.spot == 105
        assert bumped.rate == 0.05  # unchanged
        assert bumped.dividend_yield == 0.02  # unchanged

    def test_bump_rate(self):
        """Test bump_rate creates correct new environment."""
        market = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.02)
        bumped = market.bump_rate(0.01)

        assert bumped.spot == 100  # unchanged
        assert bumped.rate == pytest.approx(0.06)
        assert bumped.dividend_yield == 0.02  # unchanged

    def test_bump_rate_validates_by_default(self):
        """Test bump_rate validates new rate by default."""
        market = MarketEnvironment(spot=100, rate=0.90)

        # Should raise because 0.90 + 0.20 = 1.10 > 1.0
        with pytest.raises(ValueError, match="Rate seems unreasonable"):
            market.bump_rate(0.20)

    def test_bump_rate_can_bypass_validation(self):
        """Test bump_rate with validate=False bypasses validation."""
        market = MarketEnvironment(spot=100, rate=0.90)

        # Should NOT raise with validate=False
        extreme = market.bump_rate(0.20, validate=False)
        assert extreme.rate == pytest.approx(1.10)

    def test_bump_dividend(self):
        """Test bump_dividend creates correct new environment."""
        market = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.02)
        bumped = market.bump_dividend(0.01)

        assert bumped.spot == 100  # unchanged
        assert bumped.rate == 0.05  # unchanged
        assert bumped.dividend_yield == 0.03

    def test_bump_dividend_validates_by_default(self):
        """Test bump_dividend validates new yield by default."""
        market = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.15)

        # Should raise because 0.15 + 0.10 = 0.25 > 0.20
        with pytest.raises(ValueError, match="Dividend yield seems unreasonable"):
            market.bump_dividend(0.10)

    def test_bump_dividend_can_bypass_validation(self):
        """Test bump_dividend with validate=False bypasses validation."""
        market = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.15)

        # Should NOT raise with validate=False
        extreme = market.bump_dividend(0.10, validate=False)
        assert extreme.dividend_yield == pytest.approx(0.25)

    def test_immutability(self):
        """Test that original market is unchanged after operations."""
        market = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.02)

        _ = market.bump_spot(5.0)
        _ = market.bump_rate(0.01)
        _ = market.bump_dividend(0.01)
        _ = market.with_spot(200)
        _ = market.with_rate(0.10)
        _ = market.with_dividend(0.05)

        # Original unchanged
        assert market.spot == 100
        assert market.rate == 0.05
        assert market.dividend_yield == 0.02


class TestMarketEnvironmentWith:
    """Tests for with_* methods."""

    def test_with_spot(self):
        """Test with_spot creates correct new environment."""
        market = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.02)
        new_market = market.with_spot(150)

        assert new_market.spot == 150
        assert new_market.rate == 0.05
        assert new_market.dividend_yield == 0.02

    def test_with_spot_validates(self):
        """Test with_spot validates spot price."""
        market = MarketEnvironment(spot=100, rate=0.05)

        with pytest.raises(ValueError, match="Spot must be positive"):
            market.with_spot(-50)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
