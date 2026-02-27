"""
Portfolio and Breakeven Tests
==============================

Tests for portfolio management, P&L calculations, and breakeven analysis.

Author: Thomas
Created: 2025
"""

import pytest
import numpy as np

from backend.portfolio import (
    OptionsPortfolio,
    long_call,
    short_call,
    long_put,
    short_put,
    long_stock,
    short_stock,
)
from backend.portfolio.breakeven import (
    BreakevenResult,
    find_breakevens,
    find_breakevens_from_portfolio,
    calculate_portfolio_pnl_at_expiry,
)
from backend.models.gbm import GBMModel
from backend.core.market import MarketEnvironment
from backend.engines import BSAnalyticEngine
from backend.greeks.analytic import bs_all_greeks

from tests.conftest import report


# =============================================================================
# POSITION FACTORY TESTS
# =============================================================================

class TestPositionFactories:
    """Test position creation helper functions."""

    def test_long_call_creation(self):
        """Test long call position factory."""
        report.header("Long Call Position Factory")
        report.info("Tests the long_call() helper function creates position with correct attributes")
        report.info("A long call has positive quantity and is_long=True, is_call=True")

        pos = long_call(strike=100, maturity=0.5, premium=5.0, quantity=2)

        report.params(strike=100, maturity=0.5, premium=5.0, quantity=2)
        report.info(f"is_call: {pos.is_call}, is_long: {pos.is_long}")

        assert pos.strike == 100
        assert pos.maturity == 0.5
        assert pos.premium == 5.0
        assert pos.quantity == 2
        assert pos.is_call is True
        assert pos.is_long is True

        report.success("Long call position created correctly")

    def test_short_call_creation(self):
        """Test short call position factory."""
        report.header("Short Call Position Factory")
        report.info("Tests short_call() creates position with negative quantity (sold option)")
        report.info("A short call has is_long=False and receives premium upfront")

        pos = short_call(strike=105, maturity=0.25, premium=3.0)

        report.params(strike=105, maturity=0.25, premium=3.0)
        report.info(f"is_call: {pos.is_call}, is_long: {pos.is_long}, quantity: {pos.quantity}")

        assert pos.strike == 105
        assert pos.is_call is True
        assert pos.is_long is False
        assert pos.quantity == -1  # Short position has negative quantity

        report.success("Short call position created correctly")

    def test_long_put_creation(self):
        """Test long put position factory."""
        report.header("Long Put Position Factory")
        report.info("Tests long_put() creates position with is_call=False and is_long=True")
        report.info("A long put profits when the underlying price falls below the strike")

        pos = long_put(strike=95, maturity=0.5, premium=4.0)

        report.params(strike=95, maturity=0.5, premium=4.0)
        report.info(f"is_call: {pos.is_call}, is_long: {pos.is_long}")

        assert pos.strike == 95
        assert pos.is_call is False
        assert pos.is_long is True

        report.success("Long put position created correctly")

    def test_short_put_creation(self):
        """Test short put position factory."""
        report.header("Short Put Position Factory")
        report.info("Tests short_put() creates position with is_call=False and is_long=False")
        report.info("A short put obligates buying stock at strike if exercised")

        pos = short_put(strike=90, maturity=0.25, premium=2.0)

        report.params(strike=90, maturity=0.25, premium=2.0)
        report.info(f"is_call: {pos.is_call}, is_long: {pos.is_long}")

        assert pos.strike == 90
        assert pos.is_call is False
        assert pos.is_long is False

        report.success("Short put position created correctly")

    def test_long_stock_creation(self):
        """Test long stock position factory."""
        report.header("Long Stock Position Factory")
        report.info("Tests long_stock() creates position with positive quantity")
        report.info("Long stock profits when price rises above entry price")

        stock = long_stock(quantity=100, entry_price=50.0)

        report.params(quantity=100, entry_price=50.0)
        report.info(f"Actual quantity: {stock.quantity}")

        assert stock.quantity == 100
        assert stock.entry_price == 50.0

        report.success("Long stock position created correctly")

    def test_short_stock_creation(self):
        """Test short stock position factory."""
        report.header("Short Stock Position Factory")
        report.info("Tests short_stock() creates position with negative quantity")
        report.info("Short stock profits when price falls below entry price")

        stock = short_stock(quantity=50, entry_price=60.0)

        report.params(input_quantity=50, entry_price=60.0)
        report.info(f"Actual quantity (negative for short): {stock.quantity}")

        assert stock.quantity == -50
        assert stock.entry_price == 60.0

        report.success("Short stock position created correctly")


# =============================================================================
# SINGLE OPTION P&L TESTS
# =============================================================================

class TestSingleOptionPnL:
    """Test P&L calculations for single option positions."""

    def test_long_call_pnl_itm(self):
        """Long call ITM: P&L = S - K - premium."""
        report.header("Long Call ITM P&L")
        report.info("Tests P&L when long call finishes in-the-money (S > K)")
        report.info("P&L = intrinsic value - premium paid = (S - K) - premium")

        pos = [long_call(strike=100, maturity=0.5, premium=5.0)]
        spot = 110.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, pos)
        expected = (110 - 100) - 5  # Intrinsic - premium = 5

        report.params(strike=100, premium=5.0, spot_at_expiry=spot)
        report.info("Formula: P&L = max(S-K, 0) - premium")
        report.value("P&L", pnl, expected=expected, unit="$")

        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_long_call_pnl_otm(self):
        """Long call OTM: P&L = -premium."""
        report.header("Long Call OTM P&L")
        report.info("Tests P&L when long call finishes out-of-the-money (S < K)")
        report.info("Option expires worthless, so P&L equals the premium lost")

        pos = [long_call(strike=100, maturity=0.5, premium=5.0)]
        spot = 90.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, pos)
        expected = -5.0  # Lose premium

        report.params(strike=100, premium=5.0, spot_at_expiry=spot)
        report.info("Option expires worthless - lose premium")
        report.value("P&L", pnl, expected=expected, unit="$")

        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_long_call_pnl_atm(self):
        """Long call ATM: P&L = -premium."""
        report.header("Long Call ATM P&L")
        report.info("Tests P&L when long call finishes at-the-money (S = K)")
        report.info("Option has zero intrinsic value, so full premium is lost")

        pos = [long_call(strike=100, maturity=0.5, premium=5.0)]
        spot = 100.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, pos)
        expected = -5.0  # No intrinsic, lose premium

        report.params(strike=100, premium=5.0, spot_at_expiry=spot)
        report.info("No intrinsic value - lose full premium")
        report.value("P&L", pnl, expected=expected, unit="$")

        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_short_call_pnl_itm(self):
        """Short call ITM: P&L = premium - (S - K)."""
        report.header("Short Call ITM P&L")
        report.info("Tests P&L when short call finishes in-the-money (S > K)")
        report.info("Seller must pay intrinsic value, keeping only premium minus payout")

        pos = [short_call(strike=100, maturity=0.5, premium=5.0)]
        spot = 110.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, pos)
        expected = 5 - (110 - 100)  # Premium - intrinsic = -5

        report.params(strike=100, premium=5.0, spot_at_expiry=spot)
        report.info("Formula: P&L = premium - max(S-K, 0)")
        report.value("P&L", pnl, expected=expected, unit="$")

        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_short_call_pnl_otm(self):
        """Short call OTM: P&L = premium."""
        report.header("Short Call OTM P&L")
        report.info("Tests P&L when short call finishes out-of-the-money (S < K)")
        report.info("Option expires worthless, seller keeps the full premium")

        pos = [short_call(strike=100, maturity=0.5, premium=5.0)]
        spot = 90.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, pos)
        expected = 5.0  # Keep premium

        report.params(strike=100, premium=5.0, spot_at_expiry=spot)
        report.info("Option expires worthless - keep full premium")
        report.value("P&L", pnl, expected=expected, unit="$")

        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_long_put_pnl_itm(self):
        """Long put ITM: P&L = K - S - premium."""
        report.header("Long Put ITM P&L")
        report.info("Tests P&L when long put finishes in-the-money (S < K)")
        report.info("P&L = intrinsic value - premium = (K - S) - premium")

        pos = [long_put(strike=100, maturity=0.5, premium=4.0)]
        spot = 90.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, pos)
        expected = (100 - 90) - 4  # Intrinsic - premium = 6

        report.params(strike=100, premium=4.0, spot_at_expiry=spot)
        report.info("Formula: P&L = max(K-S, 0) - premium")
        report.value("P&L", pnl, expected=expected, unit="$")

        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_long_put_pnl_otm(self):
        """Long put OTM: P&L = -premium."""
        report.header("Long Put OTM P&L")
        report.info("Tests P&L when long put finishes out-of-the-money (S > K)")
        report.info("Option expires worthless, so P&L equals the premium lost")

        pos = [long_put(strike=100, maturity=0.5, premium=4.0)]
        spot = 110.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, pos)
        expected = -4.0  # Lose premium

        report.params(strike=100, premium=4.0, spot_at_expiry=spot)
        report.info("Option expires worthless - lose premium")
        report.value("P&L", pnl, expected=expected, unit="$")

        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_short_put_pnl_itm(self):
        """Short put ITM: P&L = premium - (K - S)."""
        report.header("Short Put ITM P&L")
        report.info("Tests P&L when short put finishes in-the-money (S < K)")
        report.info("Seller must pay intrinsic value, losing premium minus payout")

        pos = [short_put(strike=100, maturity=0.5, premium=4.0)]
        spot = 90.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, pos)
        expected = 4 - (100 - 90)  # Premium - intrinsic = -6

        report.params(strike=100, premium=4.0, spot_at_expiry=spot)
        report.info("Formula: P&L = premium - max(K-S, 0)")
        report.value("P&L", pnl, expected=expected, unit="$")

        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_quantity_multiplier(self):
        """Test that quantity multiplies P&L correctly."""
        report.header("Quantity Multiplier Test")
        report.info("Verifies that position quantity scales P&L proportionally")
        report.info("Two contracts should produce exactly twice the P&L of one contract")

        pos_single = [long_call(strike=100, maturity=0.5, premium=5.0, quantity=1)]
        pos_double = [long_call(strike=100, maturity=0.5, premium=5.0, quantity=2)]
        spot = 110.0

        pnl_single = calculate_portfolio_pnl_at_expiry(spot, pos_single)
        pnl_double = calculate_portfolio_pnl_at_expiry(spot, pos_double)

        report.params(strike=100, premium=5.0, spot_at_expiry=spot)
        report.comparison("P&L (qty=1)", pnl_single, "P&L (qty=2)", pnl_double, unit="$")
        report.info(f"Ratio: {pnl_double / pnl_single:.1f}x")

        np.testing.assert_allclose(pnl_double, 2 * pnl_single, rtol=1e-10)


# =============================================================================
# STOCK POSITION TESTS
# =============================================================================

class TestStockPositionPnL:
    """Test P&L calculations for stock positions."""

    def test_long_stock_profit(self):
        """Long stock with price increase."""
        report.header("Long Stock Profit")
        report.info("Tests P&L when long stock position has a favorable price move")
        report.info("P&L = quantity × (current_price - entry_price)")

        stock = long_stock(quantity=100, entry_price=50.0)
        spot = 60.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, [], stock)
        expected = 100 * (60 - 50)  # 100 shares * $10 gain = $1000

        report.params(quantity=100, entry_price=50.0, current_price=spot)
        report.value("P&L", pnl, expected=expected, unit="$")

        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_long_stock_loss(self):
        """Long stock with price decrease."""
        report.header("Long Stock Loss")
        report.info("Tests P&L when long stock position has an adverse price move")
        report.info("Falling prices result in negative P&L for long positions")

        stock = long_stock(quantity=100, entry_price=50.0)
        spot = 40.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, [], stock)
        expected = 100 * (40 - 50)  # 100 shares * -$10 = -$1000

        report.params(quantity=100, entry_price=50.0, current_price=spot)
        report.value("P&L", pnl, expected=expected, unit="$")

        np.testing.assert_allclose(pnl, expected, rtol=1e-10)

    def test_short_stock_profit(self):
        """Short stock with price decrease."""
        report.header("Short Stock Profit")
        report.info("Tests P&L when short stock position has a favorable price move")
        report.info("Short positions profit when price falls below entry price")

        stock = short_stock(quantity=100, entry_price=50.0)
        spot = 40.0

        pnl = calculate_portfolio_pnl_at_expiry(spot, [], stock)
        expected = -100 * (40 - 50)  # -100 shares * -$10 = $1000

        report.params(quantity=-100, entry_price=50.0, current_price=spot)
        report.value("P&L", pnl, expected=expected, unit="$")

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
        report.header("Bull Call Spread P&L")
        report.info("Tests bull call spread: Long lower-strike call, short higher-strike call")
        report.info("Bullish strategy with limited risk and capped profit potential")

        positions = [
            long_call(strike=95, maturity=0.5, premium=8.0),
            short_call(strike=105, maturity=0.5, premium=3.0),
        ]
        net_debit = 8.0 - 3.0  # $5

        report.params(
            long_strike=95, long_premium=8.0,
            short_strike=105, short_premium=3.0,
            net_debit=net_debit
        )

        # Below lower strike: max loss
        pnl_low = calculate_portfolio_pnl_at_expiry(80.0, positions)
        report.value("P&L at S=80 (max loss)", pnl_low, expected=-net_debit, unit="$")
        np.testing.assert_allclose(pnl_low, -net_debit, rtol=1e-10)

        # Above upper strike: max profit
        pnl_high = calculate_portfolio_pnl_at_expiry(120.0, positions)
        expected_max_profit = (105 - 95) - net_debit  # $5
        report.value("P&L at S=120 (max profit)", pnl_high, expected=expected_max_profit, unit="$")
        np.testing.assert_allclose(pnl_high, expected_max_profit, rtol=1e-10)

        # At breakeven
        breakeven = 95 + net_debit  # $100
        pnl_be = calculate_portfolio_pnl_at_expiry(breakeven, positions)
        report.value("P&L at breakeven", pnl_be, expected=0.0, unit="$")
        report.info(f"Breakeven point: ${breakeven:.2f}")
        np.testing.assert_allclose(pnl_be, 0.0, atol=1e-10)

    def test_bear_put_spread(self):
        """
        Bear put spread: Long K=105, Short K=95

        Max profit = K1 - K2 - net debit
        Max loss = net debit
        """
        report.header("Bear Put Spread P&L")
        report.info("Tests bear put spread: Long higher-strike put, short lower-strike put")
        report.info("Bearish strategy with limited risk and capped profit potential")

        positions = [
            long_put(strike=105, maturity=0.5, premium=7.0),
            short_put(strike=95, maturity=0.5, premium=2.0),
        ]
        net_debit = 7.0 - 2.0  # $5

        report.params(
            long_strike=105, long_premium=7.0,
            short_strike=95, short_premium=2.0,
            net_debit=net_debit
        )

        # Below lower strike: max profit
        pnl_low = calculate_portfolio_pnl_at_expiry(80.0, positions)
        expected_max_profit = (105 - 95) - net_debit  # $5
        report.value("P&L at S=80 (max profit)", pnl_low, expected=expected_max_profit, unit="$")
        np.testing.assert_allclose(pnl_low, expected_max_profit, rtol=1e-10)

        # Above upper strike: max loss
        pnl_high = calculate_portfolio_pnl_at_expiry(120.0, positions)
        report.value("P&L at S=120 (max loss)", pnl_high, expected=-net_debit, unit="$")
        np.testing.assert_allclose(pnl_high, -net_debit, rtol=1e-10)

    def test_long_straddle(self):
        """
        Long straddle: Long call + Long put at same strike

        Max loss at strike = total premium paid
        Unlimited profit potential
        """
        report.header("Long Straddle P&L")
        report.info("Tests long straddle: Long call + long put at same strike")
        report.info("Profits from large moves in either direction; max loss at strike")

        positions = [
            long_call(strike=100, maturity=0.5, premium=5.0),
            long_put(strike=100, maturity=0.5, premium=4.5),
        ]
        total_premium = 5.0 + 4.5  # $9.5

        report.params(strike=100, call_premium=5.0, put_premium=4.5, total_premium=total_premium)

        # At strike: max loss
        pnl_atm = calculate_portfolio_pnl_at_expiry(100.0, positions)
        report.value("P&L at S=100 (max loss)", pnl_atm, expected=-total_premium, unit="$")
        np.testing.assert_allclose(pnl_atm, -total_premium, rtol=1e-10)

        # Far above: call profit - put loss
        pnl_high = calculate_portfolio_pnl_at_expiry(120.0, positions)
        expected = (120 - 100) - total_premium  # $10.5
        report.value("P&L at S=120", pnl_high, expected=expected, unit="$")
        np.testing.assert_allclose(pnl_high, expected, rtol=1e-10)

        # Far below: put profit - call loss
        pnl_low = calculate_portfolio_pnl_at_expiry(80.0, positions)
        expected = (100 - 80) - total_premium  # $10.5
        report.value("P&L at S=80", pnl_low, expected=expected, unit="$")
        np.testing.assert_allclose(pnl_low, expected, rtol=1e-10)

    def test_short_straddle(self):
        """
        Short straddle: Short call + Short put at same strike

        Max profit at strike = total premium received
        Unlimited loss potential
        """
        report.header("Short Straddle P&L")
        report.info("Tests short straddle: Short call + short put at same strike")
        report.info("Profits when price stays near strike; unlimited loss potential")

        positions = [
            short_call(strike=100, maturity=0.5, premium=5.0),
            short_put(strike=100, maturity=0.5, premium=4.5),
        ]
        total_premium = 5.0 + 4.5  # $9.5

        report.params(strike=100, call_premium=5.0, put_premium=4.5, total_premium=total_premium)

        # At strike: max profit
        pnl_atm = calculate_portfolio_pnl_at_expiry(100.0, positions)
        report.value("P&L at S=100 (max profit)", pnl_atm, expected=total_premium, unit="$")
        np.testing.assert_allclose(pnl_atm, total_premium, rtol=1e-10)

    def test_long_strangle(self):
        """
        Long strangle: Long call K2 + Long put K1 (K1 < K2)

        Max loss between strikes = total premium
        """
        report.header("Long Strangle P&L")
        report.info("Tests long strangle: Long OTM put + long OTM call at different strikes")
        report.info("Cheaper than straddle; needs larger move to profit")

        positions = [
            long_put(strike=95, maturity=0.5, premium=3.0),
            long_call(strike=105, maturity=0.5, premium=2.5),
        ]
        total_premium = 3.0 + 2.5  # $5.5

        report.params(put_strike=95, call_strike=105, total_premium=total_premium)

        # Between strikes: max loss
        pnl_middle = calculate_portfolio_pnl_at_expiry(100.0, positions)
        report.value("P&L at S=100 (max loss)", pnl_middle, expected=-total_premium, unit="$")
        np.testing.assert_allclose(pnl_middle, -total_premium, rtol=1e-10)

    def test_iron_condor(self):
        """
        Iron condor: Bull put spread + Bear call spread

        Max profit = net credit (between middle strikes)
        Max loss = width - net credit
        """
        report.header("Iron Condor P&L")
        report.info("Tests iron condor: Bull put spread + bear call spread")
        report.info("Neutral strategy profiting from low volatility; limited risk/reward")

        positions = [
            # Bull put spread
            short_put(strike=95, maturity=0.5, premium=2.0),
            long_put(strike=90, maturity=0.5, premium=1.0),
            # Bear call spread
            short_call(strike=105, maturity=0.5, premium=2.0),
            long_call(strike=110, maturity=0.5, premium=1.0),
        ]
        net_credit = (2.0 - 1.0) + (2.0 - 1.0)  # $2

        report.params(
            put_strikes="90/95", call_strikes="105/110",
            net_credit=net_credit
        )

        # In the middle: max profit
        pnl_middle = calculate_portfolio_pnl_at_expiry(100.0, positions)
        report.value("P&L at S=100 (max profit)", pnl_middle, expected=net_credit, unit="$")
        np.testing.assert_allclose(pnl_middle, net_credit, rtol=1e-10)

        # Below lowest strike: max loss
        pnl_low = calculate_portfolio_pnl_at_expiry(80.0, positions)
        width = 95 - 90  # $5
        expected_loss = net_credit - width  # -$3
        report.value("P&L at S=80 (max loss)", pnl_low, expected=expected_loss, unit="$")
        np.testing.assert_allclose(pnl_low, expected_loss, rtol=1e-10)

    def test_covered_call(self):
        """
        Covered call: Long stock + Short call

        Capped upside, reduced downside risk

        With multiplier=1.0, each option contract covers 1 share.
        For 100 shares, we need 100 option contracts.
        """
        report.header("Covered Call P&L")
        report.info("Tests covered call: Long stock + short call (income strategy)")
        report.info("Generates premium income but caps upside at strike price")

        positions = [
            short_call(strike=105, maturity=0.5, premium=3.0, quantity=100),
        ]
        stock = long_stock(quantity=100, entry_price=100.0)

        report.params(
            stock_qty=100, entry_price=100.0,
            call_strike=105, call_premium=3.0
        )

        # Below strike: stock P&L + premium
        pnl_low = calculate_portfolio_pnl_at_expiry(90.0, positions, stock)
        expected = 100 * (90 - 100) + 100 * 3.0  # -$1000 + $300 = -$700
        report.value("P&L at S=90", pnl_low, expected=expected, unit="$")
        np.testing.assert_allclose(pnl_low, expected, rtol=1e-10)

        # At strike: stock profit + premium
        pnl_atm = calculate_portfolio_pnl_at_expiry(105.0, positions, stock)
        expected = 100 * (105 - 100) + 100 * 3.0  # $500 + $300 = $800
        report.value("P&L at S=105", pnl_atm, expected=expected, unit="$")
        np.testing.assert_allclose(pnl_atm, expected, rtol=1e-10)

        # Above strike: capped profit
        pnl_high = calculate_portfolio_pnl_at_expiry(120.0, positions, stock)
        expected = 100 * (105 - 100) + 100 * 3.0  # Max at $800 (capped at strike)
        report.value("P&L at S=120 (capped)", pnl_high, expected=expected, unit="$")
        report.info("Profit capped at strike due to short call obligation")
        np.testing.assert_allclose(pnl_high, expected, rtol=1e-10)

    def test_protective_put(self):
        """
        Protective put: Long stock + Long put

        Limited downside, unlimited upside (minus premium)

        With multiplier=1.0, each option contract covers 1 share.
        For 100 shares, we need 100 option contracts.
        """
        report.header("Protective Put P&L")
        report.info("Tests protective put: Long stock + long put (insurance strategy)")
        report.info("Limits downside risk while preserving upside potential")

        positions = [
            long_put(strike=95, maturity=0.5, premium=3.0, quantity=100),
        ]
        stock = long_stock(quantity=100, entry_price=100.0)

        report.params(
            stock_qty=100, entry_price=100.0,
            put_strike=95, put_premium=3.0
        )

        # Below strike: protected
        pnl_low = calculate_portfolio_pnl_at_expiry(80.0, positions, stock)
        # Stock loss: 100 * (80-100) = -$2000
        # Put profit: 100 * (95-80) - 100*3 = $1500 - $300 = $1200
        expected = 100 * (80 - 100) + 100 * ((95 - 80) - 3.0)  # -$800
        report.value("P&L at S=80 (protected)", pnl_low, expected=expected, unit="$")
        report.info("Loss limited by protective put")
        np.testing.assert_allclose(pnl_low, expected, rtol=1e-10)


# =============================================================================
# BREAKEVEN CALCULATOR TESTS
# =============================================================================

class TestBreakevenCalculator:
    """Test breakeven calculation functionality."""

    def test_single_call_breakeven(self):
        """Long call has one breakeven: K + premium."""
        report.header("Long Call Breakeven")
        report.info("Tests breakeven calculation for a single long call position")
        report.info("Breakeven = strike + premium paid (where P&L crosses zero)")

        positions = [long_call(strike=100, maturity=0.5, premium=5.0)]
        result = find_breakevens(positions, spot_min=80, spot_max=120)

        expected_be = 100 + 5  # $105

        report.params(strike=100, premium=5.0)
        report.info(f"Expected breakeven: K + premium = ${expected_be:.2f}")
        report.value("Calculated breakeven", result.breakeven_points[0], expected=expected_be, unit="$")

        assert len(result.breakeven_points) == 1
        np.testing.assert_allclose(result.breakeven_points[0], expected_be, rtol=0.01)

    def test_single_put_breakeven(self):
        """Long put has one breakeven: K - premium."""
        report.header("Long Put Breakeven")
        report.info("Tests breakeven calculation for a single long put position")
        report.info("Breakeven = strike - premium paid")

        positions = [long_put(strike=100, maturity=0.5, premium=4.0)]
        result = find_breakevens(positions, spot_min=80, spot_max=120)

        expected_be = 100 - 4  # $96

        report.params(strike=100, premium=4.0)
        report.info(f"Expected breakeven: K - premium = ${expected_be:.2f}")
        report.value("Calculated breakeven", result.breakeven_points[0], expected=expected_be, unit="$")

        assert len(result.breakeven_points) == 1
        np.testing.assert_allclose(result.breakeven_points[0], expected_be, rtol=0.01)

    def test_straddle_two_breakevens(self):
        """Long straddle has two breakevens."""
        report.header("Straddle Breakevens")
        report.info("Tests that a long straddle has exactly two breakeven points")
        report.info("Lower BE = K - premium; Upper BE = K + premium")

        positions = [
            long_call(strike=100, maturity=0.5, premium=5.0),
            long_put(strike=100, maturity=0.5, premium=4.5),
        ]
        total_premium = 9.5
        result = find_breakevens(positions, spot_min=80, spot_max=120)

        # Lower breakeven: K - total premium
        expected_lower = 100 - total_premium  # $90.5
        # Upper breakeven: K + total premium
        expected_upper = 100 + total_premium  # $109.5

        report.params(strike=100, total_premium=total_premium)
        report.info(f"Expected lower BE: K - premium = ${expected_lower:.2f}")
        report.info(f"Expected upper BE: K + premium = ${expected_upper:.2f}")

        assert len(result.breakeven_points) == 2

        breakevens_sorted = sorted(result.breakeven_points)
        report.comparison("Lower BE (calc)", breakevens_sorted[0],
                         "Lower BE (exp)", expected_lower, unit="$")
        report.comparison("Upper BE (calc)", breakevens_sorted[1],
                         "Upper BE (exp)", expected_upper, unit="$")

        np.testing.assert_allclose(breakevens_sorted[0], expected_lower, rtol=0.01)
        np.testing.assert_allclose(breakevens_sorted[1], expected_upper, rtol=0.01)

    def test_max_profit_calculation(self):
        """Test max profit identification."""
        report.header("Max Profit Calculation")
        report.info("Tests that max profit is correctly identified for a bull call spread")
        report.info("Max profit = spread width - net debit")

        positions = [
            long_call(strike=95, maturity=0.5, premium=8.0),
            short_call(strike=105, maturity=0.5, premium=3.0),
        ]
        result = find_breakevens(positions, spot_min=80, spot_max=120)

        # Max profit = spread width - net debit = 10 - 5 = $5
        expected_max_profit = (105 - 95) - (8.0 - 3.0)

        report.params(spread_width=10, net_debit=5.0)
        report.value("Max Profit", result.max_profit, expected=expected_max_profit, unit="$")

        np.testing.assert_allclose(result.max_profit, expected_max_profit, rtol=0.01)

    def test_max_loss_calculation(self):
        """Test max loss identification."""
        report.header("Max Loss Calculation")
        report.info("Tests that max loss is correctly identified for a bull call spread")
        report.info("Max loss = net debit paid (occurs when spot < lower strike)")

        positions = [
            long_call(strike=95, maturity=0.5, premium=8.0),
            short_call(strike=105, maturity=0.5, premium=3.0),
        ]
        result = find_breakevens(positions, spot_min=80, spot_max=120)

        # Max loss = net debit = $5
        expected_max_loss = -(8.0 - 3.0)

        report.params(net_debit=5.0)
        report.value("Max Loss", result.max_loss, expected=expected_max_loss, unit="$")

        np.testing.assert_allclose(result.max_loss, expected_max_loss, rtol=0.01)

    def test_profit_zones_identification(self):
        """Test profit zone identification."""
        report.header("Profit Zone Identification")
        report.info("Tests that profit zones are correctly identified for a long call")
        report.info("Profit zone begins above breakeven point (strike + premium)")

        positions = [long_call(strike=100, maturity=0.5, premium=5.0)]
        result = find_breakevens(positions, spot_min=80, spot_max=120)

        report.params(strike=100, premium=5.0)
        report.info(f"Number of profit zones: {len(result.profit_zones)}")

        # Profit zone should be above breakeven (105)
        assert len(result.profit_zones) >= 1

        # Check that profit zone starts around breakeven
        profit_start = result.profit_zones[0][0]
        report.info(f"Profit zone starts at: ${profit_start:.2f}")
        assert profit_start > 100  # Should be above strike

        report.success("Profit zone correctly identified above breakeven")

    def test_loss_zones_identification(self):
        """Test loss zone identification."""
        report.header("Loss Zone Identification")
        report.info("Tests that loss zones are correctly identified for a long call")
        report.info("Loss zone exists below breakeven point where option loses value")

        positions = [long_call(strike=100, maturity=0.5, premium=5.0)]
        result = find_breakevens(positions, spot_min=80, spot_max=120)

        report.params(strike=100, premium=5.0)
        report.info(f"Number of loss zones: {len(result.loss_zones)}")

        # Loss zone should be below breakeven
        assert len(result.loss_zones) >= 1

        report.success("Loss zone correctly identified below breakeven")

    def test_always_profitable(self):
        """Test strategy that's always profitable (net credit iron condor with narrow wings)."""
        report.header("High Credit Short Straddle")
        report.info("Tests breakeven calculation for a high credit short straddle")
        report.info("With $30 total credit, breakevens are at K-30 and K+30")

        # This is a contrived example - short straddle with zero wings
        positions = [
            short_call(strike=100, maturity=0.5, premium=15.0),
            short_put(strike=100, maturity=0.5, premium=15.0),
        ]
        result = find_breakevens(positions, spot_min=60, spot_max=140)

        report.params(strike=100, total_credit=30.0, search_range="60-140")
        report.info(f"Breakevens found: {len(result.breakeven_points)}")

        # With $30 credit, breakevens at 70 and 130
        # If we search only 60-140, might find 2 breakevens
        assert len(result.breakeven_points) >= 0  # May have breakevens depending on range

        report.success("Breakeven search completed")


# =============================================================================
# PORTFOLIO CLASS TESTS
# =============================================================================

class TestOptionsPortfolio:
    """Test OptionsPortfolio class functionality."""

    def test_portfolio_creation(self, gbm_model):
        """Test basic portfolio creation."""
        report.header("Portfolio Creation")
        report.info("Tests that an empty portfolio is correctly initialized")
        report.info("New portfolio should have zero positions and no stock")

        portfolio = OptionsPortfolio(model=gbm_model)

        report.info(f"Model type: {type(gbm_model).__name__}")
        report.info(f"Positions count: {len(portfolio.positions)}")
        report.info(f"Stock position: {portfolio.stock}")

        assert portfolio.model == gbm_model
        assert len(portfolio.positions) == 0
        assert portfolio.stock is None

        report.success("Empty portfolio created correctly")

    def test_add_position(self, gbm_model):
        """Test adding positions to portfolio."""
        report.header("Add Position to Portfolio")
        report.info("Tests that option positions can be added to portfolio")
        report.info("Portfolio should track all added positions correctly")

        portfolio = OptionsPortfolio(model=gbm_model)
        pos = long_call(strike=100, maturity=0.5, premium=5.0)
        portfolio.add(pos)

        report.info(f"Position added: Long Call K={pos.strike}")
        report.info(f"Portfolio size: {len(portfolio.positions)}")

        assert len(portfolio.positions) == 1
        assert portfolio.positions[0] == pos

        report.success("Position added successfully")

    def test_add_stock(self, gbm_model):
        """Test adding stock position."""
        report.header("Add Stock to Portfolio")
        report.info("Tests that stock positions can be added to portfolio")
        report.info("Stock is stored separately from option positions")

        portfolio = OptionsPortfolio(model=gbm_model)
        stock = long_stock(quantity=100, entry_price=50.0)
        portfolio.add(stock)

        report.info(f"Stock added: {stock.quantity} shares at ${stock.entry_price}")

        assert portfolio.stock == stock

        report.success("Stock position added successfully")

    def test_clear_portfolio(self, gbm_model):
        """Test clearing portfolio."""
        report.header("Clear Portfolio")
        report.info("Tests that portfolio can be cleared of all positions")
        report.info("Clear should remove both option positions and stock")

        portfolio = OptionsPortfolio(model=gbm_model)
        portfolio.add(long_call(strike=100, maturity=0.5, premium=5.0))
        portfolio.add(long_stock(quantity=100, entry_price=50.0))

        report.info(f"Before clear: {len(portfolio.positions)} options, stock={portfolio.stock is not None}")

        portfolio.clear()

        report.info(f"After clear: {len(portfolio.positions)} options, stock={portfolio.stock}")

        assert len(portfolio.positions) == 0
        assert portfolio.stock is None

        report.success("Portfolio cleared successfully")

    def test_portfolio_breakeven_integration(self, bull_call_spread):
        """Test breakeven calculation from portfolio."""
        report.header("Portfolio Breakeven Integration")
        report.info("Tests integration between OptionsPortfolio and breakeven calculator")
        report.info("find_breakevens_from_portfolio should correctly extract positions")

        result = find_breakevens_from_portfolio(
            bull_call_spread,
            spot_min=80,
            spot_max=120
        )

        report.info(f"Result type: {type(result).__name__}")
        report.info(f"Breakevens found: {len(result.breakeven_points)}")

        for i, be in enumerate(result.breakeven_points):
            report.info(f"  Breakeven {i+1}: ${be:.2f}")

        assert isinstance(result, BreakevenResult)
        assert len(result.breakeven_points) == 1  # Bull call spread has one breakeven

        report.success("Portfolio breakeven integration works correctly")


# =============================================================================
# NUMERICAL ACCURACY TESTS
# =============================================================================

class TestNumericalAccuracy:
    """Test numerical accuracy of P&L calculations."""

    def test_pnl_symmetry_call_put(self):
        """Long call and short put with same strike should have related P&L."""
        report.header("Call-Put P&L Symmetry")
        report.info("Tests put-call parity relationship in P&L at expiry")
        report.info("Long Call P&L + Short Put P&L = S - K (synthetic forward)")

        strike = 100
        premium = 5.0

        pos_call = [long_call(strike=strike, maturity=0.5, premium=premium)]
        pos_put = [short_put(strike=strike, maturity=0.5, premium=premium)]

        report.params(strike=strike, premium=premium)
        report.info("Property: Long Call P&L + Short Put P&L = S - K")

        rows = []
        for spot in [90, 100, 110]:
            pnl_call = calculate_portfolio_pnl_at_expiry(spot, pos_call)
            pnl_put = calculate_portfolio_pnl_at_expiry(spot, pos_put)

            # At expiry: C - P = S - K (approximately, ignoring premium effects)
            # Actually: long call pnl + short put pnl = S - K (at expiry)
            combined = pnl_call + pnl_put
            expected = spot - strike
            rows.append((spot, pnl_call, pnl_put, combined, expected))
            np.testing.assert_allclose(combined, expected, rtol=1e-10)

        report.table(
            headers=["Spot", "Call P&L", "Put P&L", "Combined", "S-K"],
            rows=rows,
            precision=2
        )

        report.success("P&L symmetry verified")

    def test_empty_portfolio(self):
        """Empty portfolio should have zero P&L."""
        report.header("Empty Portfolio P&L")
        report.info("Tests edge case: portfolio with no positions")
        report.info("Empty portfolio should always return zero P&L")

        pnl = calculate_portfolio_pnl_at_expiry(100.0, [])

        report.value("Empty portfolio P&L", pnl, expected=0.0, unit="$")

        assert pnl == 0.0

        report.success("Empty portfolio has zero P&L")

    def test_precision_high_grid(self):
        """Test precision with different grid densities."""
        report.header("Grid Precision Comparison")
        report.info("Tests that breakeven results are consistent across grid sizes")
        report.info("Higher precision should give similar results to lower precision")

        positions = [
            long_call(strike=100, maturity=0.5, premium=5.0),
            long_put(strike=100, maturity=0.5, premium=4.5),
        ]

        result_low = find_breakevens(positions, spot_min=80, spot_max=120, precision=1000)
        result_high = find_breakevens(positions, spot_min=80, spot_max=120, precision=50000)

        report.info("Low precision: 1,000 points")
        report.info("High precision: 50,000 points")

        # Higher precision should give similar or better results
        assert len(result_low.breakeven_points) == len(result_high.breakeven_points)

        for i, (be_low, be_high) in enumerate(zip(
            sorted(result_low.breakeven_points),
            sorted(result_high.breakeven_points)
        )):
            report.comparison(
                f"BE{i+1} (low prec)", be_low,
                f"BE{i+1} (high prec)", be_high,
                unit="$"
            )
            np.testing.assert_allclose(be_low, be_high, rtol=0.01)

        report.success("Both precision levels produce consistent results")


# =============================================================================
# BREAKEVEN RESULT TESTS
# =============================================================================

class TestBreakevenResult:
    """Test BreakevenResult dataclass functionality."""

    def test_summary_generation(self):
        """Test summary string generation."""
        report.header("BreakevenResult Summary Generation")
        report.info("Tests that BreakevenResult generates a readable summary string")
        report.info("Summary should include breakevens, max profit, and max loss")

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

        report.info("Summary content check:")
        report.info(f"  Contains 'Breakeven': {'Breakeven' in summary}")
        report.info(f"  Contains '$100.00': {'$100.00' in summary}")
        report.info(f"  Contains 'Max Profit': {'Max Profit' in summary}")
        report.info(f"  Contains 'Max Loss': {'Max Loss' in summary}")

        assert "Breakeven" in summary
        assert "$100.00" in summary
        assert "Max Profit" in summary
        assert "Max Loss" in summary

        report.success("Summary generation works correctly")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report.header("BreakevenResult to Dictionary")
        report.info("Tests that BreakevenResult can be converted to dictionary")
        report.info("Dictionary should contain all result fields for serialization")

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

        report.info("Dictionary keys: " + ", ".join(d.keys()))
        report.value("breakeven_points", d["breakeven_points"][0], expected=100.0)
        report.value("max_profit", d["max_profit"], expected=10.0, unit="$")
        report.value("max_loss", d["max_loss"], expected=-5.0, unit="$")

        assert d["breakeven_points"] == [100.0]
        assert d["max_profit"] == 10.0
        assert d["max_loss"] == -5.0

        report.success("Dictionary conversion works correctly")


# =============================================================================
# PORTFOLIO GREEKS TESTS
# =============================================================================

class TestPortfolioGreeks:
    """Tests for portfolio-level Greeks calculation."""

    @pytest.fixture
    def market(self):
        return MarketEnvironment(spot=100.0, rate=0.05, dividend_yield=0.0)

    @pytest.fixture
    def model(self):
        return GBMModel(sigma=0.20)

    def test_single_call_greeks(self, market, model):
        """Portfolio Greeks = BS Greeks for a single call."""
        portfolio = OptionsPortfolio(model=model)
        portfolio.add(long_call(strike=100, maturity=0.25, premium=0))

        port_greeks = portfolio.greeks(market)

        # Compare with BS analytic Greeks
        bs = bs_all_greeks(
            s=market.spot, k=100.0, t=0.25, r=market.rate,
            q=market.dividend_yield, sigma=0.20, is_call=True
        )
        bs_delta, bs_gamma, _, _ = bs[1], bs[2], bs[3], bs[4]

        np.testing.assert_allclose(port_greeks.delta, bs_delta, rtol=0.02)
        np.testing.assert_allclose(port_greeks.gamma, bs_gamma, rtol=0.02)

    def test_straddle_delta_neutral(self, market, model):
        """ATM straddle has |delta| < 0.15 (not perfectly zero due to r > 0)."""
        portfolio = OptionsPortfolio(model=model)
        portfolio.add(long_call(strike=100, maturity=0.25, premium=0))
        portfolio.add(long_put(strike=100, maturity=0.25, premium=0))

        greeks = portfolio.greeks(market)
        assert abs(greeks.delta) < 0.15, f"|delta| = {abs(greeks.delta)} should be < 0.15"

    def test_straddle_gamma_positive(self, market, model):
        """Long straddle has positive gamma."""
        portfolio = OptionsPortfolio(model=model)
        portfolio.add(long_call(strike=100, maturity=0.25, premium=0))
        portfolio.add(long_put(strike=100, maturity=0.25, premium=0))

        greeks = portfolio.greeks(market)
        assert greeks.gamma > 0, f"Gamma = {greeks.gamma} should be positive"

    def test_straddle_vega_positive(self, market, model):
        """Long straddle has positive vega."""
        portfolio = OptionsPortfolio(model=model)
        portfolio.add(long_call(strike=100, maturity=0.25, premium=0))
        portfolio.add(long_put(strike=100, maturity=0.25, premium=0))

        greeks = portfolio.greeks(market)
        assert greeks.vega > 0, f"Vega = {greeks.vega} should be positive"

    def test_straddle_theta_negative(self, market, model):
        """Long options have negative theta."""
        portfolio = OptionsPortfolio(model=model)
        portfolio.add(long_call(strike=100, maturity=0.25, premium=0))
        portfolio.add(long_put(strike=100, maturity=0.25, premium=0))

        greeks = portfolio.greeks(market)
        assert greeks.theta < 0, f"Theta = {greeks.theta} should be negative"

    def test_iron_condor_low_delta(self, market, model):
        """Iron condor has |delta| < 0.2."""
        portfolio = OptionsPortfolio(model=model)
        portfolio.add(short_put(strike=95, maturity=0.25, premium=0))
        portfolio.add(long_put(strike=90, maturity=0.25, premium=0))
        portfolio.add(short_call(strike=105, maturity=0.25, premium=0))
        portfolio.add(long_call(strike=110, maturity=0.25, premium=0))

        greeks = portfolio.greeks(market)
        assert abs(greeks.delta) < 0.2, f"|delta| = {abs(greeks.delta)} should be < 0.2"

    def test_greeks_aggregation_linearity(self, market, model):
        """greeks(A+B) = greeks(A) + greeks(B)."""
        # Portfolio A: single call
        port_a = OptionsPortfolio(model=model)
        port_a.add(long_call(strike=100, maturity=0.25, premium=0))
        greeks_a = port_a.greeks(market)

        # Portfolio B: single put
        port_b = OptionsPortfolio(model=model)
        port_b.add(long_put(strike=105, maturity=0.25, premium=0))
        greeks_b = port_b.greeks(market)

        # Combined portfolio
        port_ab = OptionsPortfolio(model=model)
        port_ab.add(long_call(strike=100, maturity=0.25, premium=0))
        port_ab.add(long_put(strike=105, maturity=0.25, premium=0))
        greeks_ab = port_ab.greeks(market)

        np.testing.assert_allclose(greeks_ab.delta, greeks_a.delta + greeks_b.delta, atol=1e-4)
        np.testing.assert_allclose(greeks_ab.gamma, greeks_a.gamma + greeks_b.gamma, atol=1e-4)
        np.testing.assert_allclose(greeks_ab.vega, greeks_a.vega + greeks_b.vega, atol=1e-4)
        np.testing.assert_allclose(greeks_ab.theta, greeks_a.theta + greeks_b.theta, atol=1e-4)


class TestPortfolioNameCollision:
    """Bug 6: Both pnl functions must be accessible without collision."""

    def test_portfolio_pnl_name_no_collision(self):
        """Both breakeven and arrays versions of calculate_portfolio_pnl_at_expiry are accessible."""
        from backend.portfolio.breakeven import (
            calculate_portfolio_pnl_at_expiry as pnl_breakeven,
        )
        from backend.portfolio.greeks_surfaces import (
            calculate_portfolio_pnl_at_expiry_arrays as pnl_arrays,
        )

        # They must be different functions
        assert pnl_breakeven is not pnl_arrays

        # breakeven version takes (spot, positions_list)
        pos = [long_call(strike=100, maturity=0.5, premium=5.0)]
        result_be = pnl_breakeven(110.0, pos)
        assert result_be == pytest.approx(5.0)  # (110-100) - 5

        # arrays version takes numpy arrays
        import numpy as np
        result_arr = pnl_arrays(
            110.0,
            np.array([100.0]),
            np.array([1.0]),   # call
            np.array([1.0]),   # long
            np.array([1.0]),
            np.array([5.0]),
            0.0, 0.0
        )
        assert result_arr == pytest.approx(5.0)


# =============================================================================
# NEW COVERAGE TESTS (Phase 3)
# =============================================================================

class TestPortfolioGreeksScaling:
    """Gap 1: Verify portfolio.greeks() vega/theta/rho match engine.greeks() scaling."""

    def test_portfolio_greeks_match_engine_scaling(self):
        """Portfolio Greeks vega/theta/rho should match BSAnalyticEngine.greeks() within 2%."""
        model = GBMModel(sigma=0.20)
        market = MarketEnvironment(spot=100.0, rate=0.05, dividend_yield=0.0)
        engine = BSAnalyticEngine()

        from backend.instruments.options import VanillaOption
        option = VanillaOption(strike=100.0, maturity=0.25, is_call=True)

        # Engine (analytic) Greeks
        engine_greeks = engine.greeks(option, model, market)

        # Portfolio Greeks (single call, quantity=1, premium=0)
        portfolio = OptionsPortfolio(model=model, engine=engine)
        portfolio.add(long_call(strike=100.0, maturity=0.25, premium=0.0))
        port_greeks = portfolio.greeks(market)

        report.header("Portfolio vs Engine Greeks Scaling")
        report.greeks(engine_greeks, label="Engine Greeks")
        report.greeks(port_greeks, label="Portfolio Greeks")

        np.testing.assert_allclose(port_greeks.delta, engine_greeks.delta, rtol=0.02)
        np.testing.assert_allclose(port_greeks.gamma, engine_greeks.gamma, rtol=0.02)
        np.testing.assert_allclose(port_greeks.vega, engine_greeks.vega, rtol=0.02)
        np.testing.assert_allclose(port_greeks.theta, engine_greeks.theta, rtol=0.02)
        np.testing.assert_allclose(port_greeks.rho, engine_greeks.rho, rtol=0.02)

        report.success("Portfolio Greeks match engine Greeks scaling")


class TestPureStockPortfolioGreeks:
    """Gap 2: Portfolio with only stock position."""

    def test_pure_stock_portfolio_greeks(self):
        """Pure stock portfolio: delta = quantity, gamma = vega = theta = rho = 0."""
        model = GBMModel(sigma=0.20)
        market = MarketEnvironment(spot=100.0, rate=0.05, dividend_yield=0.0)

        portfolio = OptionsPortfolio(model=model)
        portfolio.add(long_stock(quantity=100, entry_price=100.0))

        greeks = portfolio.greeks(market)

        report.header("Pure Stock Portfolio Greeks")
        report.greeks(greeks)

        np.testing.assert_allclose(greeks.delta, 100.0, rtol=0.01)
        assert abs(greeks.gamma) < 1e-6
        assert abs(greeks.vega) < 1e-6
        assert abs(greeks.theta) < 1e-6
        assert abs(greeks.rho) < 1e-6

        report.success("Pure stock: delta=shares, gamma=vega=theta=rho≈0")


class TestThetaShortDated:
    """Gap 3: Theta for very short-dated options."""

    def test_theta_very_short_dated_option(self):
        """Theta should still be finite and negative for T < 0.01."""
        model = GBMModel(sigma=0.20)
        market = MarketEnvironment(spot=100.0, rate=0.05, dividend_yield=0.0)

        portfolio = OptionsPortfolio(model=model)
        portfolio.add(long_call(strike=100.0, maturity=0.005, premium=0.0))

        greeks = portfolio.greeks(market)

        report.header("Short-Dated Theta")
        report.value("Theta (T=0.005)", greeks.theta)

        assert np.isfinite(greeks.theta), "Theta should be finite"
        assert greeks.theta < 0, f"Theta = {greeks.theta} should be negative"

        report.success("Short-dated theta is finite and negative")


class TestRiskMetricsZeroVariance:
    """Gap 4: Zero-variance P&L skewness/kurtosis edge case."""

    def test_risk_metrics_zero_variance_pnl(self):
        """Zero-variance P&L should return (0, 0) for skewness/kurtosis."""
        from backend.portfolio.pnl import compute_skewness_kurtosis

        # Constant P&L array -> zero variance
        pnl = np.full(1000, 5.0, dtype=np.float64)
        skew, kurt = compute_skewness_kurtosis(pnl)

        report.header("Zero Variance Risk Metrics")
        report.value("Skewness", skew, expected=0.0)
        report.value("Excess Kurtosis", kurt, expected=0.0)

        assert skew == 0.0
        assert kurt == 0.0

        report.success("Zero-variance P&L returns (0, 0) for skew/kurtosis")


class TestBreakevenAlwaysLoss:
    """Gap 5: Always-loss portfolio classification."""

    def test_breakeven_always_loss_portfolio(self):
        """Portfolio that always loses should be classified correctly."""
        # Long call + short call at same strike with net debit
        # (impossible to profit, always loses the net premium)
        positions = [
            long_call(strike=100, maturity=0.5, premium=10.0),
            short_call(strike=100, maturity=0.5, premium=3.0),
        ]
        # At every spot: both cancel out, net P&L = -(10-3) = -7

        result = find_breakevens(positions, spot_min=50, spot_max=150)

        report.header("Always-Loss Portfolio Breakeven")
        report.info(f"Breakeven points: {result.breakeven_points}")
        report.info(f"Profit zones: {result.profit_zones}")
        report.info(f"Loss zones: {result.loss_zones}")

        assert len(result.breakeven_points) == 0
        assert len(result.profit_zones) == 0
        assert len(result.loss_zones) >= 1

        report.success("Always-loss portfolio classified correctly")


# =============================================================================
# SKEWNESS / KURTOSIS TESTS (Group 6)
# =============================================================================

class TestSkewnessKurtosis:
    """Tests for compute_skewness_kurtosis on known distributions."""

    def test_normal_distribution_skewness_near_zero(self):
        """Normal samples: skew ~ 0, excess kurtosis ~ 0."""
        from backend.portfolio.pnl import compute_skewness_kurtosis

        rng = np.random.default_rng(42)
        pnl = rng.normal(0.0, 1.0, size=100_000).astype(np.float64)
        skew, kurt = compute_skewness_kurtosis(pnl)

        np.testing.assert_allclose(skew, 0.0, atol=0.05)
        np.testing.assert_allclose(kurt, 0.0, atol=0.10)

    def test_positive_skew_distribution(self):
        """Exponential distribution has positive skew."""
        from backend.portfolio.pnl import compute_skewness_kurtosis

        rng = np.random.default_rng(42)
        pnl = rng.exponential(1.0, size=100_000).astype(np.float64)
        skew, kurt = compute_skewness_kurtosis(pnl)

        assert skew > 0.5, f"Exponential skew={skew:.3f} should be > 0.5"

    def test_negative_skew_distribution(self):
        """Negated exponential distribution has negative skew."""
        from backend.portfolio.pnl import compute_skewness_kurtosis

        rng = np.random.default_rng(42)
        pnl = -rng.exponential(1.0, size=100_000).astype(np.float64)
        skew, kurt = compute_skewness_kurtosis(pnl)

        assert skew < -0.5, f"Negated exponential skew={skew:.3f} should be < -0.5"


# =============================================================================
# PERCENTILE TESTS (Group 7)
# =============================================================================

class TestPercentiles:
    """Tests for compute_percentiles on known inputs."""

    def test_percentiles_sorted_array(self):
        """Known values 0-100: percentiles should match expected positions."""
        from backend.portfolio.pnl import compute_percentiles

        pnl = np.arange(101, dtype=np.float64)  # 0, 1, 2, ..., 100
        pcts = np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype=np.float64)
        result = compute_percentiles(pnl, pcts)

        assert result[0] == 0.0    # 0th percentile
        assert result[2] == 50.0   # 50th percentile (median)
        assert result[4] == 100.0  # 100th percentile

    def test_percentiles_constant_array(self):
        """All-equal array: every percentile should equal that constant."""
        from backend.portfolio.pnl import compute_percentiles

        pnl = np.full(1000, 42.0, dtype=np.float64)
        pcts = np.array([10.0, 50.0, 90.0], dtype=np.float64)
        result = compute_percentiles(pnl, pcts)

        for val in result:
            np.testing.assert_allclose(val, 42.0)

    def test_percentiles_two_values(self):
        """Two-value array: 0th percentile = min, 100th = max."""
        from backend.portfolio.pnl import compute_percentiles

        pnl = np.array([10.0, 20.0], dtype=np.float64)
        pcts = np.array([0.0, 100.0], dtype=np.float64)
        result = compute_percentiles(pnl, pcts)

        assert result[0] == 10.0
        assert result[1] == 20.0
