"""
Market Environment
==================

Immutable snapshot of market conditions for option pricing.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from backend.utils.constants.numerical import DEFAULT_YIELD_CURVE_TENORS


# =============================================================================
# Yield Curve
# =============================================================================


@dataclass(frozen=True)
class YieldCurve:
    """
    Zero-coupon yield curve for discounting.

    Interpolates linearly between tenors. Extrapolates flat beyond
    the shortest and longest tenors.

    Parameters
    ----------
    tenors : np.ndarray
        Maturities in years (e.g., [0.25, 0.5, 1.0, 2.0, 5.0]).
    rates : np.ndarray
        Continuously compounded zero rates at each tenor.
    """

    tenors: np.ndarray
    rates: np.ndarray

    def __post_init__(self) -> None:
        if len(self.tenors) != len(self.rates):
            raise ValueError(
                f"tenors and rates must have same length, "
                f"got {len(self.tenors)} and {len(self.rates)}"
            )
        if len(self.tenors) == 0:
            raise ValueError("Yield curve must have at least one tenor")
        if any(t <= 0 for t in self.tenors):
            raise ValueError("All tenors must be positive")
        # np.interp requires strictly increasing x-coordinates; out-of-order
        # tenors would otherwise return silently wrong discount factors.
        if np.any(np.diff(np.asarray(self.tenors, dtype=float)) <= 0):
            raise ValueError("Tenors must be strictly increasing")

    def _interpolate_rate(self, t: float) -> float:
        """Linearly interpolate the zero rate at time t."""
        if t <= 0:
            return float(self.rates[0])
        return float(np.interp(t, self.tenors, self.rates))

    def discount_factor(self, t: float) -> float:
        """
        Compute discount factor D(0, t) = exp(-r(t) * t).

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            Discount factor.
        """
        if t <= 0:
            return 1.0
        r = self._interpolate_rate(t)
        return float(np.exp(-r * t))

    def discount_factors(self, times: np.ndarray) -> np.ndarray:
        """
        Vectorized discount factors for an array of times.

        Parameters
        ----------
        times : np.ndarray
            Times in years.

        Returns
        -------
        np.ndarray
            Discount factors D(0, t_i).
        """
        rates = np.interp(times, self.tenors, self.rates)
        return np.exp(-rates * times)

    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Compute continuously compounded forward rate f(t1, t2).

        f(t1,t2) = [r(t2)*t2 - r(t1)*t1] / (t2 - t1)

        Parameters
        ----------
        t1 : float
            Start time.
        t2 : float
            End time (must be > t1).

        Returns
        -------
        float
            Forward rate.
        """
        if t2 <= t1:
            raise ValueError(f"t2 must be > t1, got t1={t1}, t2={t2}")
        r1 = self._interpolate_rate(t1)
        r2 = self._interpolate_rate(t2)
        return (r2 * t2 - r1 * t1) / (t2 - t1)

    @classmethod
    def flat(cls, rate: float) -> YieldCurve:
        """
        Create a flat yield curve.

        Parameters
        ----------
        rate : float
            Flat rate for all maturities.

        Returns
        -------
        YieldCurve
        """
        tenors = np.asarray(DEFAULT_YIELD_CURVE_TENORS, dtype=np.float64)
        return cls(tenors=tenors, rates=np.full(tenors.size, rate))

    def __repr__(self) -> str:
        return (
            f"YieldCurve(tenors=[{self.tenors[0]:.2f}..{self.tenors[-1]:.2f}], "
            f"rates=[{self.rates[0]:.4f}..{self.rates[-1]:.4f}])"
        )


# =============================================================================
# Market Environment
# =============================================================================


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
    valuation_date: str | None = None

    def __post_init__(self) -> None:
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

    def _replace_unvalidated(self, **overrides: object) -> "MarketEnvironment":
        """
        Like ``dataclasses.replace`` but bypassing ``__post_init__`` validation.

        Preserves the concrete subclass and *all* its fields — e.g. an
        ``EnrichedMarketEnvironment``'s ``yield_curve``/``credit_spread`` — which
        a hard-coded ``MarketEnvironment(...)`` would silently drop. Used by the
        bump_*/with_* helpers when validation is bypassed for extreme stress
        scenarios.
        """
        obj = object.__new__(type(self))
        for f in fields(self):
            value = overrides[f.name] if f.name in overrides else getattr(self, f.name)
            object.__setattr__(obj, f.name, value)
        return obj

    def bump_spot(self, delta: float) -> "MarketEnvironment":
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
        return replace(self, spot=self.spot + delta)

    def bump_rate(self, delta: float, validate: bool = True) -> "MarketEnvironment":
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
            return replace(self, rate=new_rate)
        return self._replace_unvalidated(rate=new_rate)

    def bump_dividend(self, delta: float, validate: bool = True) -> "MarketEnvironment":
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
            return replace(self, dividend_yield=new_dividend)
        return self._replace_unvalidated(dividend_yield=new_dividend)

    def with_spot(self, spot: float) -> "MarketEnvironment":
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
        return replace(self, spot=spot)

    def with_rate(self, rate: float, validate: bool = False) -> "MarketEnvironment":
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
            return replace(self, rate=rate)
        # Bypass validation for extreme scenario analysis
        return self._replace_unvalidated(rate=rate)

    def with_dividend(
        self, dividend_yield: float, validate: bool = False
    ) -> "MarketEnvironment":
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
            return replace(self, dividend_yield=dividend_yield)
        # Bypass validation for extreme scenario analysis
        return self._replace_unvalidated(dividend_yield=dividend_yield)


# =============================================================================
# Enriched Market Environment (for structured products)
# =============================================================================


@dataclass(frozen=True)
class EnrichedMarketEnvironment(MarketEnvironment):
    """
    Market environment extended with a yield curve and credit spread.

    Backward-compatible with MarketEnvironment — all existing engines
    continue to work. The `rate` field serves as fallback when
    `yield_curve` is None.

    Parameters
    ----------
    yield_curve : YieldCurve, optional
        Term structure of interest rates. If None, uses flat `rate`.
    credit_spread : float
        Issuer credit spread (annualized, e.g. 0.01 = 100bp).
    """

    yield_curve: YieldCurve | None = None
    credit_spread: float = 0.0

    def discount_factor(self, t: float) -> float:
        """
        Compute risk-free discount factor D(0, t).

        Uses yield curve if available, otherwise exp(-rate * t).

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            Discount factor.
        """
        if self.yield_curve is not None:
            return self.yield_curve.discount_factor(t)
        return float(np.exp(-self.rate * t))

    def risky_discount_factor(self, t: float) -> float:
        """
        Compute risky discount factor including credit spread.

        D_risky(0, t) = D(0, t) * exp(-credit_spread * t)

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            Risky discount factor.
        """
        return self.discount_factor(t) * float(np.exp(-self.credit_spread * t))

    def discount_factors(self, times: np.ndarray) -> np.ndarray:
        """
        Vectorized risk-free discount factors.

        Parameters
        ----------
        times : np.ndarray
            Times in years.

        Returns
        -------
        np.ndarray
            Discount factors D(0, t_i).
        """
        if self.yield_curve is not None:
            return self.yield_curve.discount_factors(times)
        return np.exp(-self.rate * times)
