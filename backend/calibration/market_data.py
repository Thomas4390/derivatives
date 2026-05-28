"""
Market Data Containers
======================

Immutable data structures for calibration inputs:
- OptionQuote: single observed option price
- OptionMarketData: option surface + market environment
- HistoricalReturns: time series for GARCH MLE

Author: Thomas Vaudescal
Created: 2026
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """A single observed option price from the market."""

    strike: float
    maturity: float
    is_call: bool
    market_price: float
    bid: float | None = None
    ask: float | None = None
    implied_vol: float | None = None
    volume: int | None = None
    open_interest: int | None = None

    def __post_init__(self):
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"maturity must be positive, got {self.maturity}")
        if self.market_price < 0:
            raise ValueError(
                f"market_price must be non-negative, got {self.market_price}"
            )

    @property
    def mid_price(self) -> float:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return self.market_price

    @property
    def spread(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None


@dataclass(frozen=True)
class OptionMarketData:
    """Option market surface: quotes + market environment."""

    spot: float
    rate: float
    dividend_yield: float
    quotes: tuple[OptionQuote, ...]
    # Pre-computed arrays (set in __post_init__, excluded from __init__)
    _strikes: np.ndarray = field(init=False, repr=False, compare=False)
    _maturities: np.ndarray = field(init=False, repr=False, compare=False)
    _market_prices: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.spot <= 0:
            raise ValueError(f"spot must be positive, got {self.spot}")
        if len(self.quotes) == 0:
            raise ValueError("quotes must not be empty")
        # Sort quotes by (maturity, strike) for consistent ordering
        sorted_quotes = tuple(sorted(self.quotes, key=lambda q: (q.maturity, q.strike)))
        object.__setattr__(self, "quotes", sorted_quotes)
        object.__setattr__(
            self, "_strikes", np.array([q.strike for q in sorted_quotes])
        )
        object.__setattr__(
            self, "_maturities", np.array([q.maturity for q in sorted_quotes])
        )
        object.__setattr__(
            self, "_market_prices", np.array([q.market_price for q in sorted_quotes])
        )

    @property
    def strikes(self) -> np.ndarray:
        return self._strikes

    @property
    def maturities(self) -> np.ndarray:
        return self._maturities

    @property
    def unique_maturities(self) -> np.ndarray:
        return np.unique(self._maturities)

    @property
    def market_prices(self) -> np.ndarray:
        return self._market_prices

    @property
    def n_quotes(self) -> int:
        return len(self.quotes)

    def quotes_for_maturity(self, T: float, tol: float = 1e-8) -> list[OptionQuote]:
        return [q for q in self.quotes if abs(q.maturity - T) < tol]

    def filter(
        self,
        min_strike: float | None = None,
        max_strike: float | None = None,
        min_maturity: float | None = None,
        max_maturity: float | None = None,
        calls_only: bool = False,
        puts_only: bool = False,
    ) -> "OptionMarketData":
        filtered = list(self.quotes)
        if min_strike is not None:
            filtered = [q for q in filtered if q.strike >= min_strike]
        if max_strike is not None:
            filtered = [q for q in filtered if q.strike <= max_strike]
        if min_maturity is not None:
            filtered = [q for q in filtered if q.maturity >= min_maturity]
        if max_maturity is not None:
            filtered = [q for q in filtered if q.maturity <= max_maturity]
        if calls_only:
            filtered = [q for q in filtered if q.is_call]
        if puts_only:
            filtered = [q for q in filtered if not q.is_call]
        if not filtered:
            raise ValueError("Filter produced empty quote set")
        return OptionMarketData(
            spot=self.spot,
            rate=self.rate,
            dividend_yield=self.dividend_yield,
            quotes=tuple(filtered),
        )


@dataclass(frozen=True)
class HistoricalReturns:
    """Historical return series for GARCH MLE calibration."""

    log_returns: np.ndarray
    frequency: str = "daily"
    annualization_factor: int = 252

    def __post_init__(self):
        if len(self.log_returns) < 10:
            raise ValueError(
                f"Need at least 10 observations, got {len(self.log_returns)}"
            )
        if self.frequency not in ("daily", "weekly", "monthly"):
            raise ValueError(
                f"frequency must be daily/weekly/monthly, got {self.frequency}"
            )

    @property
    def n_obs(self) -> int:
        return len(self.log_returns)

    @property
    def sample_variance(self) -> float:
        return float(np.var(self.log_returns, ddof=1))

    @property
    def sample_volatility(self) -> float:
        return float(
            np.std(self.log_returns, ddof=1) * np.sqrt(self.annualization_factor)
        )

    @classmethod
    def from_prices(cls, prices: np.ndarray, **kwargs) -> "HistoricalReturns":
        prices = np.asarray(prices, dtype=float)
        if np.any(prices <= 0):
            raise ValueError(
                "All prices must be strictly positive for log-return computation. "
                f"Found {np.sum(prices <= 0)} non-positive values."
            )
        log_returns = np.diff(np.log(prices))
        return cls(log_returns=log_returns, **kwargs)
