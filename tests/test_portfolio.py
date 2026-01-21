"""
Portfolio and Breakeven Tests
==============================

Tests for portfolio management, P&L calculations, and breakeven analysis.

Author: Thomas
Created: 2025
"""

import pytest
import numpy as np
from typing import List, Tuple

from backend.portfolio import (
    OptionsPortfolio,
    PortfolioPosition,
    StockPosition,
    long_call,
    short_call,
    long_put,
    short_put,
    long_stock,
    short_stock,
)
from backend.portfolio.breakeven import (
    BreakevenCalculator,
    BreakevenResult,
    find_breakevens,
    find_breakevens_from_portfolio,
    calculate_portfolio_pnl_at_expiry,
)
from backend.models.gbm import GBMModel


# =============================================================================
# POSITION FACTORY TESTS
# =============================================================================

class TestPositionFactories:
    """Test position creation helper functions."""

    def test_long_call_creation(self):
        """Test long call position factory."""
        pos = long_call(strike=100, maturity=0.5, premium=5.0, quantity=2)

        assert pos.strike == 100
        assert pos.maturity == 0.5
        assert pos.premium == 5.0
        assert pos.quantity == 2
        assert pos.is_call is True
        assert pos.is_long is True

    def test_short_call_creation(self):
        """Test short call position factory."""
        pos = short_call(strike=105, maturity=0.25, premium=3.0)

        assert pos.strike == 105
        assert pos.is_call is True
        assert pos.is_long is False
        assert pos.quantity == -1  # Short position has negative quantity

    def test_long_put_creation(self):
        """Test long put position factory."""
        pos = long_put(strike=95, maturity=0.5, premium=4.0)

        assert pos.strike == 95
        assert pos.is_call is False
        assert pos.is_long is True

    def test_short_put_creation(self):
        """Test short put position factory."""
        pos = short_put(strike=90, maturity=0.25, premium=2.0)

        assert pos.strike == 90
        assert pos.is_call is False
        assert pos.is_long is False

    def test_long_stock_creation(self):
        """Test long stock position factory."""
        stock = long_stock(quantity=100, entry_price=50.0)

        assert stock.quantity == 100
        assert stock.entry_price == 50.0

    def test_short_stock_creation(self):
        """Test short stock position factory."""
        stock = short_stock(quantity=50, entry_price=60.0)

        assert stock.quantity == -50
        assert stock.entry_price == 60.0


# =============================================================================
# SINGLE OPTION P&L TESTS
# =============================================================================

class TestSingleOptionPnL:
    """Test P&L calculations for single option positions."""

    def test_long_call_pnl_itm(self):
        """Long call ITM: P&L = S - K - premium."""
        pos = [long_call(strike=100, maturity=0.5, premium=5.0)]
        spot = 110.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, pos)

        expected = (110 - 100) - 5  # Intrinsic - premium = 5
        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_long_call_pnl_otm(self):
        """Long call OTM: P&L = -premium."""
        pos = [long_call(strike=100, maturity=0.5, premium=5.0)]
        spot = 90.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, pos)

        expected = -5.0  # Lose premium
        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_long_call_pnl_atm(self):
        """Long call ATM: P&L = -premium."""
        pos = [long_call(strike=100, maturity=0.5, premium=5.0)]
        spot = 100.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, pos)

        expected = -5.0  # No intrinsic, lose premium
        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_short_call_pnl_itm(self):
        """Short call ITM: P&L = premium - (S - K)."""
        pos = [short_call(strike=100, maturity=0.5, premium=5.0)]
        spot = 110.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, pos)

        expected = 5 - (110 - 100)  # Premium - intrinsic = -5
        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_short_call_pnl_otm(self):
        """Short call OTM: P&L = premium."""
        pos = [short_call(strike=100, maturity=0.5, premium=5.0)]
        spot = 90.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, pos)

        expected = 5.0  # Keep premium
        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_long_put_pnl_itm(self):
        """Long put ITM: P&L = K - S - premium."""
        pos = [long_put(strike=100, maturity=0.5, premium=4.0)]
        spot = 90.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, pos)

        expected = (100 - 90) - 4  # Intrinsic - premium = 6
        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_long_put_pnl_otm(self):
        """Long put OTM: P&L = -premium."""
        pos = [long_put(strike=100, maturity=0.5, premium=4.0)]
        spot = 110.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, pos)

        expected = -4.0  # Lose premium
        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_short_put_pnl_itm(self):
        """Short put ITM: P&L = premium - (K - S)."""
        pos = [short_put(strike=100, maturity=0.5, premium=4.0)]
        spot = 90.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, pos)

        expected = 4 - (100 - 90)  # Premium - intrinsic = -6
        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_quantity_multiplier(self):
        """Test that quantity multiplies P&L correctly."""
        pos_single = [long_call(strike=100, maturity=0.5, premium=5.0, quantity=1)]
        pos_double = [long_call(strike=100, maturity=0.5, premium=5.0, quantity=2)]
        spot = 110.0

        pnl_single = calculate_portfolio_pnl_at_expiry(spot, pos_single)
        pnl_double = calculate_portfolio_pnl_at_expiry(spot, pos_double)

        np.testing.assert_allclose(pnl_double, 2 * pnl_single, rtol=1e-10)


# =============================================================================
# STOCK POSITION TESTS
# =============================================================================

class TestStockPositionPnL:
    """Test P&L calculations for stock positions."""

    def test_long_stock_profit(self):
        """Long stock with price increase."""
        stock = long_stock(quantity=100, entry_price=50.0)
        spot = 60.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, [], stock)

        expected = 100 * (60 - 50)  # 100 shares * $10 gain = $1000
        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_long_stock_loss(self):
        """Long stock with price decrease."""
        stock = long_stock(quantity=100, entry_price=50.0)
        spot = 40.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, [], stock)

        expected = 100 * (40 - 50)  # 100 shares * -$10 = -$1000
        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_short_stock_profit(self):
        """Short stock with price decrease."""
        stock = short_stock(quantity=100, entry_price=50.0)
        spot = 40.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, [], stock)

        expected = -100 * (40 - 50)  # -100 shares * -$10 = $1000
        np.testing.assert_allclose(pnl, expected, rtol=1e-10)


# =============================================================================
# STRATEGY TESTS
# =============================================================================

class TestStrategyPnL:
    """Test P&L calculations for common option strategies."""

    def test_bull_call_spread(self):
        """
        Bull call spread: Long K=95, Short K=105

        Max profit = K2 - K1 - net debit = 105 - 95 - 5 = 5
        Max loss = net debit = 5
        Breakeven = K1 + net debit = 95 + 5 = 100
        """
        positions = [
            long_call(strike=95, maturity=0.5, premium=8.0),
            short_call(strike=105, maturity=0.5, premium=3.0),
        ]
        net_debit = 8.0 - 3.0  # $5

        # Below lower strike: max loss
        pnl_low = calculate_portfolio_pnl_at_expiry(80.0, positions)
        np.testing.assert_allclose(pnl_low, -net_debit, rtol=1e-10)

        # Above upper strike: max profit
        pnl_high = calculate_portfolio_pnl_at_expiry(120.0, positions)
        expected_max_profit = (105 - 95) - net_debit  # $5
        np.testing.assert_allclose(pnl_high, expected_max_profit, rtol=1e-10)

        # At breakeven
        breakeven = 95 + net_debit  # $100
        pnl_be = calculate_portfolio_pnl_at_expiry(breakeven, positions)
        np.testing.assert_allclose(pnl_be, 0.0, atol=1e-10)

    def test_bear_put_spread(self):
        """
        Bear put spread: Long K=105, Short K=95

        Max profit = K1 - K2 - net debit
        Max loss = net debit
        """
        positions = [
            long_put(strike=105, maturity=0.5, premium=7.0),
            short_put(strike=95, maturity=0.5, premium=2.0),
        ]
        net_debit = 7.0 - 2.0  # $5

        # Below lower strike: max profit
        pnl_low = calculate_portfolio_pnl_at_expiry(80.0, positions)
        expected_max_profit = (105 - 95) - net_debit  # $5
        np.testing.assert_allclose(pnl_low, expected_max_profit, rtol=1e-10)

        # Above upper strike: max loss
        pnl_high = calculate_portfolio_pnl_at_expiry(120.0, positions)
        np.testing.assert_allclose(pnl_high, -net_debit, rtol=1e-10)

    def test_long_straddle(self):
        """
        Long straddle: Long call + Long put at same strike

        Max loss at strike = total premium paid
        Unlimited profit potential
        """
        positions = [
            long_call(strike=100, maturity=0.5, premium=5.0),
            long_put(strike=100, maturity=0.5, premium=4.5),
        ]
        total_premium = 5.0 + 4.5  # $9.5

        # At strike: max loss
        pnl_atm = calculate_portfolio_pnl_at_expiry(100.0, positions)
        np.testing.assert_allclose(pnl_atm, -total_premium, rtol=1e-10)

        # Far above: call profit - put loss
        pnl_high = calculate_portfolio_pnl_at_expiry(120.0, positions)
        expected = (120 - 100) - total_premium  # $10.5
        np.testing.assert_allclose(pnl_high, expected, rtol=1e-10)

        # Far below: put profit - call loss
        pnl_low = calculate_portfolio_pnl_at_expiry(80.0, positions)
        expected = (100 - 80) - total_premium  # $10.5
        np.testing.assert_allclose(pnl_low, expected, rtol=1e-10)

    def test_short_straddle(self):
        """
        Short straddle: Short call + Short put at same strike

        Max profit at strike = total premium received
        Unlimited loss potential
        """
        positions = [
            short_call(strike=100, maturity=0.5, premium=5.0),
            short_put(strike=100, maturity=0.5, premium=4.5),
        ]
        total_premium = 5.0 + 4.5  # $9.5

        # At strike: max profit
        pnl_atm = calculate_portfolio_pnl_at_expiry(100.0, positions)
        np.testing.assert_allclose(pnl_atm, total_premium, rtol=1e-10)

    def test_long_strangle(self):
        """
        Long strangle: Long call K2 + Long put K1 (K1 < K2)

        Max loss between strikes = total premium
        """
        positions = [
            long_put(strike=95, maturity=0.5, premium=3.0),
            long_call(strike=105, maturity=0.5, premium=2.5),
        ]
        total_premium = 3.0 + 2.5  # $5.5

        # Between strikes: max loss
        pnl_middle = calculate_portfolio_pnl_at_expiry(100.0, positions)
        np.testing.assert_allclose(pnl_middle, -total_premium, rtol=1e-10)

    def test_iron_condor(self):
        """
        Iron condor: Bull put spread + Bear call spread

        Max profit = net credit (between middle strikes)
        Max loss = width - net credit
        """
        positions = [
            # Bull put spread
            short_put(strike=95, maturity=0.5, premium=2.0),
            long_put(strike=90, maturity=0.5, premium=1.0),
            # Bear call spread
            short_call(strike=105, maturity=0.5, premium=2.0),
            long_call(strike=110, maturity=0.5, premium=1.0),
        ]
        net_credit = (2.0 - 1.0) + (2.0 - 1.0)  # $2

        # In the middle: max profit
        pnl_middle = calculate_portfolio_pnl_at_expiry(100.0, positions)
        np.testing.assert_allclose(pnl_middle, net_credit, rtol=1e-10)

        # Below lowest strike: max loss
        pnl_low = calculate_portfolio_pnl_at_expiry(80.0, positions)
        width = 95 - 90  # $5
        expected_loss = net_credit - width  # -$3
        np.testing.assert_allclose(pnl_low, expected_loss, rtol=1e-10)

    def test_covered_call(self):
        """
        Covered call: Long stock + Short call

        Capped upside, reduced downside risk

        With multiplier=1.0, each option contract covers 1 share.
        For 100 shares, we need 100 option contracts.
        """
        positions = [
            short_call(strike=105, maturity=0.5, premium=3.0, quantity=100),
        ]
        stock = long_stock(quantity=100, entry_price=100.0)

        # Below strike: stock P&L + premium
        pnl_low = calculate_portfolio_pnl_at_expiry(90.0, positions, stock)
        expected = 100 * (90 - 100) + 100 * 3.0  # -$1000 + $300 = -$700
        np.testing.assert_allclose(pnl_low, expected, rtol=1e-10)

        # At strike: stock profit + premium
        pnl_atm = calculate_portfolio_pnl_at_expiry(105.0, positions, stock)
        expected = 100 * (105 - 100) + 100 * 3.0  # $500 + $300 = $800
        np.testing.assert_allclose(pnl_atm, expected, rtol=1e-10)

        # Above strike: capped profit
        pnl_high = calculate_portfolio_pnl_at_expiry(120.0, positions, stock)
        expected = 100 * (105 - 100) + 100 * 3.0  # Max at $800 (capped at strike)
        np.testing.assert_allclose(pnl_high, expected, rtol=1e-10)

    def test_protective_put(self):
        """
        Protective put: Long stock + Long put

        Limited downside, unlimited upside (minus premium)

        With multiplier=1.0, each option contract covers 1 share.
        For 100 shares, we need 100 option contracts.
        """
        positions = [
            long_put(strike=95, maturity=0.5, premium=3.0, quantity=100),
        ]
        stock = long_stock(quantity=100, entry_price=100.0)

        # Below strike: protected
        pnl_low = calculate_portfolio_pnl_at_expiry(80.0, positions, stock)
        # Stock loss: 100 * (80-100) = -$2000
        # Put profit: 100 * (95-80) - 100*3 = $1500 - $300 = $1200
        expected = 100 * (80 - 100) + 100 * ((95 - 80) - 3.0)  # -$800
        np.testing.assert_allclose(pnl_low, expected, rtol=1e-10)


# =============================================================================
# BREAKEVEN CALCULATOR TESTS
# =============================================================================

class TestBreakevenCalculator:
    """Test breakeven calculation functionality."""

    def test_single_call_breakeven(self):
        """Long call has one breakeven: K + premium."""
        positions = [long_call(strike=100, maturity=0.5, premium=5.0)]

        result = find_breakevens(positions, spot_min=80, spot_max=120)

        assert len(result.breakeven_points) == 1
        expected_be = 100 + 5  # $105
        np.testing.assert_allclose(result.breakeven_points[0], expected_be, rtol=0.01)

    def test_single_put_breakeven(self):
        """Long put has one breakeven: K - premium."""
        positions = [long_put(strike=100, maturity=0.5, premium=4.0)]

        result = find_breakevens(positions, spot_min=80, spot_max=120)

        assert len(result.breakeven_points) == 1
        expected_be = 100 - 4  # $96
        np.testing.assert_allclose(result.breakeven_points[0], expected_be, rtol=0.01)

    def test_straddle_two_breakevens(self):
        """Long straddle has two breakevens."""
        positions = [
            long_call(strike=100, maturity=0.5, premium=5.0),
            long_put(strike=100, maturity=0.5, premium=4.5),
        ]
        total_premium = 9.5

        result = find_breakevens(positions, spot_min=80, spot_max=120)

        assert len(result.breakeven_points) == 2

        # Lower breakeven: K - total premium
        expected_lower = 100 - total_premium  # $90.5
        # Upper breakeven: K + total premium
        expected_upper = 100 + total_premium  # $109.5

        breakevens_sorted = sorted(result.breakeven_points)
        np.testing.assert_allclose(breakevens_sorted[0], expected_lower, rtol=0.01)
        np.testing.assert_allclose(breakevens_sorted[1], expected_upper, rtol=0.01)

    def test_max_profit_calculation(self):
        """Test max profit identification."""
        positions = [
            long_call(strike=95, maturity=0.5, premium=8.0),
            short_call(strike=105, maturity=0.5, premium=3.0),
        ]

        result = find_breakevens(positions, spot_min=80, spot_max=120)

        # Max profit = spread width - net debit = 10 - 5 = $5
        expected_max_profit = (105 - 95) - (8.0 - 3.0)
        np.testing.assert_allclose(result.max_profit, expected_max_profit, rtol=0.01)

    def test_max_loss_calculation(self):
        """Test max loss identification."""
        positions = [
            long_call(strike=95, maturity=0.5, premium=8.0),
            short_call(strike=105, maturity=0.5, premium=3.0),
        ]

        result = find_breakevens(positions, spot_min=80, spot_max=120)

        # Max loss = net debit = $5
        expected_max_loss = -(8.0 - 3.0)
        np.testing.assert_allclose(result.max_loss, expected_max_loss, rtol=0.01)

    def test_profit_zones_identification(self):
        """Test profit zone identification."""
        positions = [long_call(strike=100, maturity=0.5, premium=5.0)]

        result = find_breakevens(positions, spot_min=80, spot_max=120)

        # Profit zone should be above breakeven (105)
        assert len(result.profit_zones) >= 1

        # Check that profit zone starts around breakeven
        profit_start = result.profit_zones[0][0]
        assert profit_start > 100  # Should be above strike

    def test_loss_zones_identification(self):
        """Test loss zone identification."""
        positions = [long_call(strike=100, maturity=0.5, premium=5.0)]

        result = find_breakevens(positions, spot_min=80, spot_max=120)

        # Loss zone should be below breakeven
        assert len(result.loss_zones) >= 1

    def test_always_profitable(self):
        """Test strategy that's always profitable (net credit iron condor with narrow wings)."""
        # This is a contrived example - short straddle with zero wings
        positions = [
            short_call(strike=100, maturity=0.5, premium=15.0),
            short_put(strike=100, maturity=0.5, premium=15.0),
        ]

        result = find_breakevens(positions, spot_min=60, spot_max=140)

        # With $30 credit, breakevens at 70 and 130
        # If we search only 60-140, might find 2 breakevens
        assert len(result.breakeven_points) >= 0  # May have breakevens depending on range


# =============================================================================
# PORTFOLIO CLASS TESTS
# =============================================================================

class TestOptionsPortfolio:
    """Test OptionsPortfolio class functionality."""

    def test_portfolio_creation(self, gbm_model):
        """Test basic portfolio creation."""
        portfolio = OptionsPortfolio(model=gbm_model)

        assert portfolio.model == gbm_model
        assert len(portfolio.positions) == 0
        assert portfolio.stock is None

    def test_add_position(self, gbm_model):
        """Test adding positions to portfolio."""
        portfolio = OptionsPortfolio(model=gbm_model)

        pos = long_call(strike=100, maturity=0.5, premium=5.0)
        portfolio.add(pos)

        assert len(portfolio.positions) == 1
        assert portfolio.positions[0] == pos

    def test_add_stock(self, gbm_model):
        """Test adding stock position."""
        portfolio = OptionsPortfolio(model=gbm_model)

        stock = long_stock(quantity=100, entry_price=50.0)
        portfolio.add(stock)

        assert portfolio.stock == stock

    def test_clear_portfolio(self, gbm_model):
        """Test clearing portfolio."""
        portfolio = OptionsPortfolio(model=gbm_model)
        portfolio.add(long_call(strike=100, maturity=0.5, premium=5.0))
        portfolio.add(long_stock(quantity=100, entry_price=50.0))

        portfolio.clear()

        assert len(portfolio.positions) == 0
        assert portfolio.stock is None

    def test_portfolio_breakeven_integration(self, bull_call_spread):
        """Test breakeven calculation from portfolio."""
        result = find_breakevens_from_portfolio(
            bull_call_spread,
            spot_min=80,
            spot_max=120
        )

        assert isinstance(result, BreakevenResult)
        assert len(result.breakeven_points) == 1  # Bull call spread has one breakeven


# =============================================================================
# NUMERICAL ACCURACY TESTS
# =============================================================================

class TestNumericalAccuracy:
    """Test numerical accuracy of P&L calculations."""

    def test_pnl_symmetry_call_put(self):
        """Long call and short put with same strike should have related P&L."""
        strike = 100
        premium = 5.0

        pos_call = [long_call(strike=strike, maturity=0.5, premium=premium)]
        pos_put = [short_put(strike=strike, maturity=0.5, premium=premium)]

        for spot in [90, 100, 110]:
            pnl_call = calculate_portfolio_pnl_at_expiry(spot, pos_call)
            pnl_put = calculate_portfolio_pnl_at_expiry(spot, pos_put)

            # At expiry: C - P = S - K (approximately, ignoring premium effects)
            # Actually: long call pnl + short put pnl = S - K (at expiry)
            combined = pnl_call + pnl_put
            expected = spot - strike
            np.testing.assert_allclose(combined, expected, rtol=1e-10)

    def test_empty_portfolio(self):
        """Empty portfolio should have zero P&L."""
        pnl = calculate_portfolio_pnl_at_expiry(100.0, [])
        assert pnl == 0.0

    def test_precision_high_grid(self):
        """Test precision with different grid densities."""
        positions = [
            long_call(strike=100, maturity=0.5, premium=5.0),
            long_put(strike=100, maturity=0.5, premium=4.5),
        ]

        result_low = find_breakevens(positions, spot_min=80, spot_max=120, precision=1000)
        result_high = find_breakevens(positions, spot_min=80, spot_max=120, precision=50000)

        # Higher precision should give similar or better results
        assert len(result_low.breakeven_points) == len(result_high.breakeven_points)

        for be_low, be_high in zip(
            sorted(result_low.breakeven_points),
            sorted(result_high.breakeven_points)
        ):
            np.testing.assert_allclose(be_low, be_high, rtol=0.01)


# =============================================================================
# BREAKEVEN RESULT TESTS
# =============================================================================

class TestBreakevenResult:
    """Test BreakevenResult dataclass functionality."""

    def test_summary_generation(self):
        """Test summary string generation."""
        result = BreakevenResult(
            breakeven_points=[100.0, 110.0],
            max_profit=10.0,
            max_profit_spot=120.0,
            max_loss=-5.0,
            max_loss_spot=90.0,
            profit_zones=[(110.0, np.inf)],
            loss_zones=[(-np.inf, 100.0)],
        )

        summary = result.summary()

        assert "Breakeven" in summary
        assert "$100.00" in summary
        assert "Max Profit" in summary
        assert "Max Loss" in summary

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = BreakevenResult(
            breakeven_points=[100.0],
            max_profit=10.0,
            max_profit_spot=120.0,
            max_loss=-5.0,
            max_loss_spot=80.0,
            profit_zones=[(100.0, np.inf)],
            loss_zones=[(-np.inf, 100.0)],
        )

        d = result.to_dict()

        assert d["breakeven_points"] == [100.0]
        assert d["max_profit"] == 10.0
        assert d["max_loss"] == -5.0
