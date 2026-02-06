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

        # Rate sanity check: -50% to +100% (extreme but possible)
        if not -0.5 <= self.rate <= 1.0:
            raise ValueError(
                f"Rate seems unreasonable: {self.rate}. "
                "Expected annualized rate in [-0.5, 1.0]. "
                "If intentional, use with_rate() to bypass validation."
            )

        # Dividend yield sanity check: -10% to +20%
        if not -0.1 <= self.dividend_yield <= 0.2:
            raise ValueError(
                f"Dividend yield seems unreasonable: {self.dividend_yield}. "
                "Expected in [-0.1, 0.2]. "
                "If intentional, use with_dividend() to bypass validation."
            )

    @classmethod
    def _create_without_validation(
        cls, spot: float, rate: float, dividend_yield: float, valuation_date: Optional[str]
    ) -> 'MarketEnvironment':
        """
        Create MarketEnvironment bypassing __post_init__ validation.

        Used internally by with_rate() and with_dividend() to allow
        extreme values for stress testing.
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, 'spot', spot)
        object.__setattr__(obj, 'rate', rate)
        object.__setattr__(obj, 'dividend_yield', dividend_yield)
        object.__setattr__(obj, 'valuation_date', valuation_date)
        return obj

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

    def bump_rate(self, delta: float, validate: bool = True) -> 'MarketEnvironment':
        """
        Create new environment with bumped rate.

        Parameters
        ----------
        delta : float
            Amount to add to risk-free rate
        validate : bool, default True
            If True, validate the new rate. If False, bypass validation
            for stress testing or extreme scenario analysis.

        Returns
        -------
        MarketEnvironment
            New environment with rate = rate + delta
        """
        new_rate = self.rate + delta
        if validate:
            return MarketEnvironment(
                spot=self.spot,
                rate=new_rate,
                dividend_yield=self.dividend_yield,
                valuation_date=self.valuation_date,
            )
        return self._create_without_validation(
            spot=self.spot,
            rate=new_rate,
            dividend_yield=self.dividend_yield,
            valuation_date=self.valuation_date,
        )

    def bump_dividend(self, delta: float, validate: bool = True) -> 'MarketEnvironment':
        """
        Create new environment with bumped dividend yield.

        Parameters
        ----------
        delta : float
            Amount to add to dividend yield
        validate : bool, default True
            If True, validate the new dividend yield. If False, bypass validation
            for stress testing or extreme scenario analysis.

        Returns
        -------
        MarketEnvironment
            New environment with dividend_yield = dividend_yield + delta
        """
        new_dividend = self.dividend_yield + delta
        if validate:
            return MarketEnvironment(
                spot=self.spot,
                rate=self.rate,
                dividend_yield=new_dividend,
                valuation_date=self.valuation_date,
            )
        return self._create_without_validation(
            spot=self.spot,
            rate=self.rate,
            dividend_yield=new_dividend,
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

    def with_rate(self, rate: float, validate: bool = False) -> 'MarketEnvironment':
        """
        Create new environment with different rate.

        This method bypasses validation by default, allowing extreme values
        for stress testing or scenario analysis.

        Parameters
        ----------
        rate : float
            New risk-free rate
        validate : bool, default False
            If True, apply validation to the new rate

        Returns
        -------
        MarketEnvironment
            New environment with the given rate
        """
        if validate:
            return MarketEnvironment(
                spot=self.spot,
                rate=rate,
                dividend_yield=self.dividend_yield,
                valuation_date=self.valuation_date,
            )
        # Bypass validation for extreme scenario analysis
        return self._create_without_validation(
            spot=self.spot,
            rate=rate,
            dividend_yield=self.dividend_yield,
            valuation_date=self.valuation_date,
        )

    def with_dividend(self, dividend_yield: float, validate: bool = False) -> 'MarketEnvironment':
        """
        Create new environment with different dividend yield.

        This method bypasses validation by default, allowing extreme values
        for stress testing or scenario analysis.

        Parameters
        ----------
        dividend_yield : float
            New dividend yield
        validate : bool, default False
            If True, apply validation to the new dividend yield

        Returns
        -------
        MarketEnvironment
            New environment with the given dividend yield
        """
        if validate:
            return MarketEnvironment(
                spot=self.spot,
                rate=self.rate,
                dividend_yield=dividend_yield,
                valuation_date=self.valuation_date,
            )
        # Bypass validation for extreme scenario analysis
        return self._create_without_validation(
            spot=self.spot,
            rate=self.rate,
            dividend_yield=dividend_yield,
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

    # Test with_dividend
    new_div = market.with_dividend(0.03)
    print(f"After with_dividend(0.03): dividend_yield={new_div.dividend_yield}")
    assert new_div.dividend_yield == 0.03, "with_dividend() failed"

    # Test immutability (original unchanged)
    print(f"Original market unchanged: spot={market.spot}")

    print("✓ Market smoke test passed")
