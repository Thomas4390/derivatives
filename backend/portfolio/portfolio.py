"""
Options Portfolio
=================

Main portfolio class with pluggable pricing models.

Author: Derivatives Pricing Project
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import numpy as np

from .positions import OptionPosition, StockPosition
from .greeks import (
    GreeksResult,
    GreeksCalculator,
    GreeksStrategy,
    AnalyticalGreeksStrategy,
    FiniteDiffGreeksStrategy,
    AnalyticalGreeksPricer,
)
from .breakeven import BreakevenResult, BreakevenCalculator


# =============================================================================
# Options Portfolio Class
# =============================================================================

class OptionsPortfolio:
    """
    Portfolio of options and stock positions with pluggable pricing.

    This class provides a unified interface for:
    - Managing option and stock positions
    - Calculating portfolio Greeks (analytical or finite differences)
    - Computing P&L at expiry
    - Finding breakeven points

    Parameters
    ----------
    pricer : BasePricer, optional
        Pricing model to use. If None, uses BlackScholesPricer with given sigma.
    sigma : float, optional
        Volatility for default BlackScholesPricer (required if pricer is None).

    Examples
    --------
    >>> from backend.portfolio import OptionsPortfolio, OptionPosition
    >>> portfolio = OptionsPortfolio(sigma=0.20)
    >>> portfolio.add_option(OptionPosition('call', 'long', strike=100, premium=5.0))
    >>> greeks = portfolio.calculate_greeks(spot=100, rate=0.05, time_to_expiry=0.5)
    >>> print(f"Delta: {greeks.delta:.4f}")

    With custom pricer:
    >>> from backend.option_pricing import HestonPricer
    >>> pricer = HestonPricer(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    >>> portfolio = OptionsPortfolio(pricer=pricer)
    """

    def __init__(
        self,
        spot_or_pricer: Optional[Any] = None,
        rate_or_sigma: Optional[float] = None,
        sigma: Optional[float] = None,
    ):
        """
        Initialize portfolio with a pricer or spot/rate for backward compatibility.

        Supports two call signatures:
        1. OptionsPortfolio(spot_price, risk_free_rate) - legacy mode
        2. OptionsPortfolio(pricer=pricer) or OptionsPortfolio(sigma=sigma) - new mode

        Parameters
        ----------
        spot_or_pricer : float or BasePricer, optional
            Spot price (legacy) or pricer object (new)
        rate_or_sigma : float, optional
            Risk-free rate (legacy) or sigma for default pricer (new)
        sigma : float, optional
            Volatility (only used if pricer is not provided)
        """
        # Legacy mode detection: if first arg is a number > 1, it's probably spot price
        self._legacy_mode = False
        self._spot_price: Optional[float] = None
        self._risk_free_rate: Optional[float] = None

        if spot_or_pricer is not None and isinstance(spot_or_pricer, (int, float)):
            if spot_or_pricer > 1.0:  # Likely spot price, not sigma
                # Legacy mode: OptionsPortfolio(spot_price, risk_free_rate)
                self._legacy_mode = True
                self._spot_price = float(spot_or_pricer)
                self._risk_free_rate = float(rate_or_sigma) if rate_or_sigma is not None else 0.05
                # Use default volatility for premium calculation
                sigma = sigma if sigma is not None else 0.25
                from backend.option_pricing import BlackScholesPricer
                pricer = BlackScholesPricer(sigma=sigma)
            else:
                # New mode with sigma as first positional arg (unlikely but handle it)
                sigma = float(spot_or_pricer)
                from backend.option_pricing import BlackScholesPricer
                pricer = BlackScholesPricer(sigma=sigma)
        elif spot_or_pricer is not None:
            # Pricer object provided
            pricer = spot_or_pricer
        elif sigma is not None or rate_or_sigma is not None:
            # sigma provided via keyword or second positional
            sigma_val = sigma if sigma is not None else rate_or_sigma
            from backend.option_pricing import BlackScholesPricer
            pricer = BlackScholesPricer(sigma=sigma_val)
        else:
            raise ValueError("Either pricer, sigma, or (spot_price, risk_free_rate) must be provided")

        self._pricer = pricer
        self._options: List[OptionPosition] = []
        self._stock: Optional[StockPosition] = None

        # Select Greeks strategy based on pricer capabilities
        self._greeks_calc = GreeksCalculator(AnalyticalGreeksStrategy(pricer))

    @property
    def pricer(self) -> Any:
        """Current pricing model."""
        return self._pricer

    @pricer.setter
    def pricer(self, value: Any):
        """
        Set a new pricer and update Greeks strategy.

        Parameters
        ----------
        value : BasePricer
            New pricing model
        """
        self._pricer = value
        self._greeks_calc = GreeksCalculator(AnalyticalGreeksStrategy(value))

    @property
    def options(self) -> List[OptionPosition]:
        """List of option positions (read-only copy)."""
        return list(self._options)

    @property
    def stock(self) -> Optional[StockPosition]:
        """Stock position if any."""
        return self._stock

    # =========================================================================
    # Position Management
    # =========================================================================

    def add_option(self, position: OptionPosition) -> 'OptionsPortfolio':
        """
        Add an option position to the portfolio.

        In legacy mode, calculates and sets the premium using Black-Scholes
        if the position's premium is 0.

        Parameters
        ----------
        position : OptionPosition
            Option position to add

        Returns
        -------
        OptionsPortfolio
            Self for method chaining
        """
        # In legacy mode, calculate premium if not already set
        if self._legacy_mode and position.premium == 0.0:
            self._calculate_and_set_premium(position)

        self._options.append(position)
        return self

    def _calculate_and_set_premium(self, position: OptionPosition) -> None:
        """Calculate premium using Black-Scholes and set it on the position."""
        if self._spot_price is None or self._risk_free_rate is None:
            return

        # Default time to expiry (30 days)
        time_to_expiry = 30.0 / 365.0

        # Calculate option price using the pricer
        # API: price(s0, k, t, r, option_type)
        result = self._pricer.price(
            s0=self._spot_price,
            k=position.strike,
            t=time_to_expiry,
            r=self._risk_free_rate,
            option_type='call' if position.is_call else 'put'
        )

        # Set the premium on the position (dataclass field assignment)
        object.__setattr__(position, 'premium', result.price)

    def add_stock(self, position: StockPosition) -> 'OptionsPortfolio':
        """
        Add or replace stock position.

        Parameters
        ----------
        position : StockPosition
            Stock position

        Returns
        -------
        OptionsPortfolio
            Self for method chaining
        """
        self._stock = position
        return self

    def clear(self) -> 'OptionsPortfolio':
        """
        Remove all positions from the portfolio.

        Returns
        -------
        OptionsPortfolio
            Self for method chaining
        """
        self._options.clear()
        self._stock = None
        return self

    def remove_option(self, index: int) -> OptionPosition:
        """
        Remove and return an option position by index.

        Parameters
        ----------
        index : int
            Index of option to remove

        Returns
        -------
        OptionPosition
            Removed option position
        """
        return self._options.pop(index)

    def clear_stock(self) -> Optional[StockPosition]:
        """
        Remove and return stock position.

        Returns
        -------
        StockPosition or None
            Removed stock position if any
        """
        stock = self._stock
        self._stock = None
        return stock

    # =========================================================================
    # Greeks Calculation
    # =========================================================================

    def calculate_greeks(
        self,
        spot: float,
        rate: float,
        time_to_expiry: float,
    ) -> GreeksResult:
        """
        Calculate aggregate Greeks for the portfolio.

        Parameters
        ----------
        spot : float
            Current spot price
        rate : float
            Risk-free interest rate (annualized)
        time_to_expiry : float
            Time to expiry in years

        Returns
        -------
        GreeksResult
            Portfolio Greeks (delta, gamma, theta, vega, rho)
        """
        return self._greeks_calc.calculate(
            options=self._options,
            stock=self._stock,
            spot=spot,
            rate=rate,
            time_to_expiry=time_to_expiry,
        )

    def calculate_greeks_surface(
        self,
        spot: float,
        rate: float,
        dte_range: np.ndarray,
        greek: str = 'delta',
    ) -> np.ndarray:
        """
        Calculate a Greek surface over days to expiry.

        Parameters
        ----------
        spot : float
            Current spot price
        rate : float
            Risk-free interest rate
        dte_range : np.ndarray
            Array of days to expiry values
        greek : str
            Which Greek to compute ('delta', 'gamma', 'theta', 'vega', 'rho')

        Returns
        -------
        np.ndarray
            Greek values corresponding to each DTE
        """
        results = np.zeros(len(dte_range))

        for i, dte in enumerate(dte_range):
            time_to_expiry = dte / 365.0
            greeks = self.calculate_greeks(spot, rate, time_to_expiry)
            results[i] = getattr(greeks, greek)

        return results

    def calculate_greeks_surface_2d(
        self,
        spot_range: np.ndarray,
        rate: float,
        dte_range: np.ndarray,
        greek: str = 'delta',
    ) -> np.ndarray:
        """
        Calculate a 2D Greek surface over spot and DTE.

        Parameters
        ----------
        spot_range : np.ndarray
            Array of spot prices
        rate : float
            Risk-free interest rate
        dte_range : np.ndarray
            Array of days to expiry values
        greek : str
            Which Greek to compute

        Returns
        -------
        np.ndarray
            2D array of shape (len(spot_range), len(dte_range))
        """
        results = np.zeros((len(spot_range), len(dte_range)))

        for i, spot in enumerate(spot_range):
            for j, dte in enumerate(dte_range):
                time_to_expiry = dte / 365.0
                greeks = self.calculate_greeks(spot, rate, time_to_expiry)
                results[i, j] = getattr(greeks, greek)

        return results

    # =========================================================================
    # P&L Calculation
    # =========================================================================

    def calculate_pnl_at_expiry(
        self,
        spot: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Calculate P&L at expiry for given spot price(s).

        Parameters
        ----------
        spot : float or np.ndarray
            Spot price(s) at expiry

        Returns
        -------
        float or np.ndarray
            P&L at expiry
        """
        if isinstance(spot, np.ndarray):
            return np.array([self._pnl_at_spot(s) for s in spot])
        return self._pnl_at_spot(spot)

    def _pnl_at_spot(self, spot: float) -> float:
        """Calculate P&L at a single spot price."""
        total = 0.0

        for pos in self._options:
            total += pos.payoff_at_expiry(spot)

        if self._stock is not None:
            total += self._stock.pnl(spot)

        return total

    # =========================================================================
    # Breakeven Analysis
    # =========================================================================

    def calculate_breakevens(
        self,
        spot_min: Optional[float] = None,
        spot_max: Optional[float] = None,
        precision: int = 10000,
    ) -> BreakevenResult:
        """
        Calculate breakeven points for the portfolio.

        Parameters
        ----------
        spot_min : float, optional
            Minimum spot price to search (default: lowest strike * 0.5)
        spot_max : float, optional
            Maximum spot price to search (default: highest strike * 1.5)
        precision : int
            Number of grid points for search

        Returns
        -------
        BreakevenResult
            Breakeven analysis results
        """
        # Auto-determine search range if not provided
        if spot_min is None or spot_max is None:
            strikes = [pos.strike for pos in self._options]
            if not strikes:
                strikes = [100.0]  # Default if no options

            if spot_min is None:
                spot_min = min(strikes) * 0.5
            if spot_max is None:
                spot_max = max(strikes) * 1.5

        calculator = BreakevenCalculator(
            spot_min=spot_min,
            spot_max=spot_max,
            precision=precision,
        )

        return calculator.calculate(self._options, self._stock)

    # =========================================================================
    # Portfolio Value
    # =========================================================================

    def calculate_value(
        self,
        spot: float,
        rate: float,
        time_to_expiry: float,
    ) -> float:
        """
        Calculate current portfolio value using the pricer.

        Parameters
        ----------
        spot : float
            Current spot price
        rate : float
            Risk-free interest rate
        time_to_expiry : float
            Time to expiry in years

        Returns
        -------
        float
            Current portfolio value
        """
        total = 0.0

        for pos in self._options:
            opt_type = pos.option_type.value
            result = self._pricer.price(
                s0=spot,
                k=pos.strike,
                t=time_to_expiry,
                r=rate,
                option_type=opt_type,
            )
            price = result.price if hasattr(result, 'price') else float(result)
            total += pos.sign * pos.quantity * price

        # Stock position value
        if self._stock is not None:
            total += self._stock.sign * self._stock.quantity * spot

        return total

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def __len__(self) -> int:
        """Number of option positions."""
        return len(self._options)

    def __repr__(self) -> str:
        stock_str = f", stock={self._stock}" if self._stock else ""
        return f"OptionsPortfolio(options={len(self._options)}{stock_str})"

    def summary(self) -> str:
        """Generate a human-readable summary of the portfolio."""
        lines = ["=== Portfolio Summary ==="]

        for i, pos in enumerate(self._options):
            direction = "Long" if pos.is_long else "Short"
            opt_type = "Call" if pos.is_call else "Put"
            lines.append(
                f"{i+1}. {direction} {pos.quantity}x {opt_type} @ K={pos.strike:.2f} "
                f"(premium={pos.premium:.2f})"
            )

        if self._stock:
            direction = "Long" if self._stock.is_long else "Short"
            lines.append(
                f"Stock: {direction} {self._stock.quantity} shares "
                f"@ {self._stock.entry_price:.2f}"
            )

        return "\n".join(lines)
