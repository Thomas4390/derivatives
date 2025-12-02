"""
Risk analysis for Options Greeks Explorer.

This module provides functions for analyzing portfolio risk characteristics.
"""

from typing import NamedTuple


class RiskProfile(NamedTuple):
    """Risk profile for a portfolio."""
    has_unlimited_profit: bool
    has_unlimited_loss: bool
    max_profit: float | None
    max_loss: float | None
    max_profit_spot: float | None
    max_loss_spot: float | None


def check_unlimited_risk(portfolio_data: dict) -> tuple[bool, bool]:
    """
    Check if portfolio has unlimited profit or loss potential.

    Args:
        portfolio_data: Dictionary containing portfolio information

    Returns:
        Tuple of (has_unlimited_profit, has_unlimited_loss)
    """
    # Stock positions always have unlimited potential in one direction
    if portfolio_data.get('stock'):
        stock = portfolio_data['stock']
        if stock['position_type'] == 'long':
            # Unlimited profit upside, limited loss (stock price to 0)
            return True, False
        else:  # short stock
            # Limited profit (stock to 0), unlimited loss upside
            return False, True

    # For options-only portfolios
    if not portfolio_data.get('options'):
        return False, False  # No positions = no unlimited risk

    long_calls = []
    short_calls = []
    long_puts = []
    short_puts = []

    for pos in portfolio_data['options']:
        strike = pos['strike']
        quantity = pos['quantity']

        if pos['option_type'] == 'call':
            if pos['position_type'] == 'long':
                long_calls.append((strike, quantity))
            else:
                short_calls.append((strike, quantity))
        else:  # put
            if pos['position_type'] == 'long':
                long_puts.append((strike, quantity))
            else:
                short_puts.append((strike, quantity))

    unlimited_profit = _check_unlimited_profit(long_calls, short_calls)
    unlimited_loss = _check_unlimited_loss(long_calls, short_calls)

    return unlimited_profit, unlimited_loss


def _check_unlimited_profit(
    long_calls: list[tuple[float, int]],
    short_calls: list[tuple[float, int]]
) -> bool:
    """
    Check for unlimited profit potential.

    Net long calls at the highest strike = unlimited profit.
    """
    if not long_calls and not short_calls:
        return False

    # Calculate net position at each strike
    call_positions = {}
    for strike, qty in long_calls:
        call_positions[strike] = call_positions.get(strike, 0) + qty
    for strike, qty in short_calls:
        call_positions[strike] = call_positions.get(strike, 0) - qty

    # Check if we have net long calls at the highest strike
    if call_positions:
        highest_strike = max(call_positions.keys())
        if call_positions[highest_strike] > 0:
            return True

    return False


def _check_unlimited_loss(
    long_calls: list[tuple[float, int]],
    short_calls: list[tuple[float, int]]
) -> bool:
    """
    Check for unlimited loss potential.

    Only naked short calls create unlimited loss.
    Note: Puts can NEVER create unlimited loss as stock can't go below 0.
    """
    if not short_calls:
        return False

    if not long_calls:
        # We have short calls with no long calls at all
        return True

    # Check if all short calls are covered or capped
    total_short = sum(qty for _, qty in short_calls)
    total_long = sum(qty for _, qty in long_calls)

    # More short calls than long calls = potential unlimited loss
    if total_short > total_long:
        return True

    return False


def analyze_portfolio_risk(
    portfolio_data: dict,
    breakeven_result,
    expiry_pnl
) -> RiskProfile:
    """
    Perform complete risk analysis on a portfolio.

    Args:
        portfolio_data: Dictionary containing portfolio information
        breakeven_result: BreakevenResult from calculations
        expiry_pnl: P&L values at expiration

    Returns:
        RiskProfile with complete risk analysis
    """
    unlimited_profit, unlimited_loss = check_unlimited_risk(portfolio_data)

    if not breakeven_result:
        return RiskProfile(
            has_unlimited_profit=unlimited_profit,
            has_unlimited_loss=unlimited_loss,
            max_profit=None,
            max_loss=None,
            max_profit_spot=None,
            max_loss_spot=None
        )

    # Determine actual max profit/loss
    max_profit = _determine_max_profit(
        unlimited_profit, breakeven_result, expiry_pnl
    )
    max_loss = _determine_max_loss(
        unlimited_loss, breakeven_result, expiry_pnl
    )

    return RiskProfile(
        has_unlimited_profit=unlimited_profit,
        has_unlimited_loss=unlimited_loss,
        max_profit=max_profit,
        max_loss=max_loss,
        max_profit_spot=breakeven_result.max_profit_spot,
        max_loss_spot=breakeven_result.max_loss_spot
    )


def _determine_max_profit(
    unlimited_profit: bool,
    breakeven_result,
    expiry_pnl
) -> float:
    """Determine the maximum profit value."""
    if not unlimited_profit:
        return breakeven_result.max_profit

    # For unlimited profit, verify P&L continues to increase at high spot prices
    if len(expiry_pnl) > 10:
        high_end_trend = expiry_pnl[-1] - expiry_pnl[-10]
        if high_end_trend > 0:  # Profit is increasing at high end
            return float('inf')

    return breakeven_result.max_profit


def _determine_max_loss(
    unlimited_loss: bool,
    breakeven_result,
    expiry_pnl
) -> float:
    """Determine the maximum loss value."""
    if not unlimited_loss:
        return breakeven_result.max_loss

    # For unlimited loss, verify P&L continues to decrease at high spot prices
    if len(expiry_pnl) > 10:
        high_end_trend = expiry_pnl[-1] - expiry_pnl[-10]
        if high_end_trend < 0 and expiry_pnl[-1] < 0:
            return float('-inf')

    return breakeven_result.max_loss


def get_risk_summary(risk_profile: RiskProfile) -> dict:
    """
    Get a human-readable risk summary.

    Args:
        risk_profile: RiskProfile object

    Returns:
        Dictionary with risk summary information
    """
    summary = {
        'risk_level': 'Unknown',
        'profit_potential': 'Unknown',
        'loss_potential': 'Unknown',
        'warnings': []
    }

    # Profit potential
    if risk_profile.has_unlimited_profit:
        summary['profit_potential'] = 'Unlimited'
    elif risk_profile.max_profit is not None:
        summary['profit_potential'] = f'${risk_profile.max_profit:.2f}'

    # Loss potential
    if risk_profile.has_unlimited_loss:
        summary['loss_potential'] = 'Unlimited'
        summary['warnings'].append('Position has unlimited loss potential')
        summary['risk_level'] = 'High'
    elif risk_profile.max_loss is not None:
        summary['loss_potential'] = f'${abs(risk_profile.max_loss):.2f}'
        summary['risk_level'] = 'Defined'

    return summary
