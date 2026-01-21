"""
Market Environment
==================

Immutable snapshot of market conditions for option pricing.

Author: Thomas
Created: 2025
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MarketEnvironment:
    """
    Immutable snapshot of market conditions.

    Separating market data from Instrument and Model enables:
    - Greeks calculation via bumping (clone -> modify -> reprice)
    - Scenario analysis
    - Clean separation of concerns

    Parameters
    ----------
    spot : float
        Current underlying price
    rate : float
        Risk-free interest rate (annualized)
    dividend_yield : float
        Continuous dividend yield (default 0)
    valuation_date : Optional[str]
        Date of valuation (for logging/audit)

    Examples
    --------
    market = MarketEnvironment(spot=100, rate=0.05)
    market_bumped = market.bump_spot(1.0)  # spot = 101
    market_bumped.spot  # 101.0
    """
    spot: float
    rate: float
    dividend_yield: float = 0.0
    valuation_date: Optional[str] = None

    def __post_init__(self):
        """Validate market parameters."""
        if self.spot <= 0:
            raise ValueError(f"Spot must be positive, got {self.spot}")

    def bump_spot(self, delta: float) -> 'MarketEnvironment':
        """
        Create new environment with bumped spot.

        Parameters
        ----------
        delta : float
            Amount to add to spot price

        Returns
        -------
        MarketEnvironment
            New environment with spot = spot + delta
        """
        return MarketEnvironment(
            spot=self.spot + delta,
            rate=self.rate,
            dividend_yield=self.dividend_yield,
            valuation_date=self.valuation_date,
        )

    def bump_rate(self, delta: float) -> 'MarketEnvironment':
        """
        Create new environment with bumped rate.

        Parameters
        ----------
        delta : float
            Amount to add to risk-free rate

        Returns
        -------
        MarketEnvironment
            New environment with rate = rate + delta
        """
        return MarketEnvironment(
            spot=self.spot,
            rate=self.rate + delta,
            dividend_yield=self.dividend_yield,
            valuation_date=self.valuation_date,
        )

    def bump_dividend(self, delta: float) -> 'MarketEnvironment':
        """
        Create new environment with bumped dividend yield.

        Parameters
        ----------
        delta : float
            Amount to add to dividend yield

        Returns
        -------
        MarketEnvironment
            New environment with dividend_yield = dividend_yield + delta
        """
        return MarketEnvironment(
            spot=self.spot,
            rate=self.rate,
            dividend_yield=self.dividend_yield + delta,
            valuation_date=self.valuation_date,
        )

    def with_spot(self, spot: float) -> 'MarketEnvironment':
        """
        Create new environment with different spot.

        Parameters
        ----------
        spot : float
            New spot price

        Returns
        -------
        MarketEnvironment
            New environment with the given spot
        """
        return MarketEnvironment(
            spot=spot,
            rate=self.rate,
            dividend_yield=self.dividend_yield,
            valuation_date=self.valuation_date,
        )

    def with_rate(self, rate: float) -> 'MarketEnvironment':
        """
        Create new environment with different rate.

        Parameters
        ----------
        rate : float
            New risk-free rate

        Returns
        -------
        MarketEnvironment
            New environment with the given rate
        """
        return MarketEnvironment(
            spot=self.spot,
            rate=rate,
            dividend_yield=self.dividend_yield,
            valuation_date=self.valuation_date,
        )


if __name__ == "__main__":
    # Smoke test
    market = MarketEnvironment(spot=100.0, rate=0.05, dividend_yield=0.02)
    print(f"Market: spot={market.spot}, rate={market.rate}, q={market.dividend_yield}")

    # Test bumping
    bumped = market.bump_spot(5.0)
    print(f"After bump_spot(5): spot={bumped.spot}")

    bumped_rate = market.bump_rate(0.01)
    print(f"After bump_rate(0.01): rate={bumped_rate.rate}")

    # Test with_spot
    new_spot = market.with_spot(110.0)
    print(f"After with_spot(110): spot={new_spot.spot}")

    # Test immutability (original unchanged)
    print(f"Original market unchanged: spot={market.spot}")

    print("✓ Market smoke test passed")
