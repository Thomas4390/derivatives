"""
Options Portfolio
=================

Portfolio class for managing option and stock positions.

This module provides:
- OptionsPortfolio: Main portfolio class with valuation and Greeks calculation

Uses the Model/Engine/Market architecture for pricing.

Author: Thomas
Created: 2025
"""

from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
import numpy as np

from backend.core.interfaces import Model, PricingEngine
from backend.core.market import MarketEnvironment
from backend.core.result_types import GreeksResult

from .positions import PortfolioPosition, StockPosition

# Import Numba-optimized functions from portfolio.pnl
from backend.portfolio.pnl import (
    calculate_portfolio_pnl_vectorized,
    calculate_portfolio_pnl_with_stock,
    compute_payoff_curve,
    find_breakeven_points,
    compute_risk_metrics,
    RiskMetrics,
)


# =============================================================================
# OPTIONS PORTFOLIO CLASS
# =============================================================================

class OptionsPortfolio:
    """
    Portfolio of option and stock positions.

    Uses Model/Engine/Market architecture for pricing.

    Parameters
    ----------
    model : Model
        Pricing model (GBMModel, HestonModel, etc.)
    engine : PricingEngine, optional
        Pricing engine. If None, selects based on model capabilities:
        - GBMModel: BSAnalyticEngine
        - Others: FFTEngine
    """

    def __init__(self, model: Model, engine: Optional[PricingEngine] = None):
        """
        Initialize portfolio.

        Parameters
        ----------
        model : Model
            Pricing model (immutable, contains all model parameters)
        engine : PricingEngine, optional
            Pricing engine. If None, auto-selects based on model.
        """
        self._model = model
        self._engine = engine or self._auto_select_engine(model)
        self._positions: List[PortfolioPosition] = []
        self._stock: Optional[StockPosition] = None

    @staticmethod
    def _auto_select_engine(model: Model) -> PricingEngine:
        """Auto-select appropriate engine for the model."""
        from backend.models.gbm import GBMModel
        from backend.engines import BSAnalyticEngine, FFTEngine

        if isinstance(model, GBMModel):
            return BSAnalyticEngine()
        else:
            # FFT works with any model that has characteristic function
            return FFTEngine()

    # =========================================================================
    # Position Management
    # =========================================================================

    def add(self, position: Union[PortfolioPosition, StockPosition]) -> 'OptionsPortfolio':
        """
        Add a position to the portfolio.

        Parameters
        ----------
        position : PortfolioPosition or StockPosition
            Position to add

        Returns
        -------
        OptionsPortfolio
            Self for method chaining
        """
        if isinstance(position, StockPosition):
            self._stock = position
        else:
            self._positions.append(position)
        return self

    def clear(self) -> 'OptionsPortfolio':
        """
        Remove all positions.

        Returns
        -------
        OptionsPortfolio
            Self for method chaining
        """
        self._positions.clear()
        self._stock = None
        return self

    def remove_position(self, index: int) -> PortfolioPosition:
        """
        Remove and return an option position by index.

        Parameters
        ----------
        index : int
            Index of position to remove

        Returns
        -------
        PortfolioPosition
            Removed position
        """
        return self._positions.pop(index)

    def clear_stock(self) -> Optional[StockPosition]:
        """
        Remove and return stock position.

        Returns
        -------
        StockPosition or None
            Removed stock position
        """
        stock = self._stock
        self._stock = None
        return stock

    @property
    def positions(self) -> List[PortfolioPosition]:
        """Option positions (read-only copy)."""
        return list(self._positions)

    @property
    def stock(self) -> Optional[StockPosition]:
        """Stock position if any."""
        return self._stock

    @property
    def model(self) -> Model:
        """Current pricing model."""
        return self._model

    @model.setter
    def model(self, value: Model):
        """Set new model and auto-select engine."""
        self._model = value
        self._engine = self._auto_select_engine(value)

    @property
    def engine(self) -> PricingEngine:
        """Current pricing engine."""
        return self._engine

    @engine.setter
    def engine(self, value: PricingEngine):
        """Set new engine."""
        self._engine = value

    # =========================================================================
    # Valuation
    # =========================================================================

    def value(self, market: MarketEnvironment) -> float:
        """
        Calculate current portfolio value.

        Parameters
        ----------
        market : MarketEnvironment
            Current market conditions (spot, rate, dividend_yield)

        Returns
        -------
        float
            Portfolio mark-to-market value
        """
        total = 0.0

        for pos in self._positions:
            result = self._engine.price(pos.instrument, self._model, market)
            total += pos.quantity * result.price

        if self._stock:
            total += self._stock.quantity * market.spot

        return total

    def pnl_at_expiry(self, spot: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate P&L at expiry (no model needed).

        Parameters
        ----------
        spot : float or array
            Spot price(s) at expiry

        Returns
        -------
        float or array
            Total P&L
        """
        spot_arr = np.atleast_1d(np.asarray(spot, dtype=float))
        total = np.zeros_like(spot_arr)

        for pos in self._positions:
            total += pos.payoff_at_expiry(spot_arr)

        if self._stock:
            total += self._stock.pnl(spot_arr)

        return float(total[0]) if np.isscalar(spot) else total

    # =========================================================================
    # Numba-Optimized P&L Calculation
    # =========================================================================

    def _positions_to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert portfolio positions to numpy arrays for Numba functions.

        Returns
        -------
        tuple
            (strikes, option_types, position_types, quantities, premiums)
            option_types: 1 = call, -1 = put
            position_types: 1 = long, -1 = short
        """
        n_legs = len(self._positions)

        strikes = np.zeros(n_legs, dtype=np.float64)
        option_types = np.zeros(n_legs, dtype=np.float64)
        position_types = np.zeros(n_legs, dtype=np.float64)
        quantities = np.zeros(n_legs, dtype=np.float64)
        premiums = np.zeros(n_legs, dtype=np.float64)

        for i, pos in enumerate(self._positions):
            strikes[i] = pos.strike
            option_types[i] = 1.0 if pos.is_call else -1.0
            position_types[i] = 1.0 if pos.is_long else -1.0
            quantities[i] = abs(pos.quantity)
            premiums[i] = pos.premium

        return strikes, option_types, position_types, quantities, premiums

    def pnl_at_expiry_fast(
        self,
        spot: np.ndarray,
        multiplier: float = 100.0
    ) -> np.ndarray:
        """
        Calculate P&L at expiry using Numba-optimized functions.

        This method is significantly faster for large arrays (>1000 elements)
        due to parallel execution across all CPU cores.

        Parameters
        ----------
        spot : np.ndarray
            Terminal spot prices, shape (n_paths,)
        multiplier : float
            Contract multiplier (default 100 shares per contract)

        Returns
        -------
        np.ndarray
            P&L for each spot price
        """
        strikes, option_types, position_types, quantities, premiums = self._positions_to_arrays()

        if self._stock:
            return calculate_portfolio_pnl_with_stock(
                spot,
                strikes, option_types, position_types, quantities, premiums,
                stock_quantity=self._stock.quantity,
                stock_entry_price=self._stock.entry_price,
                multiplier=multiplier
            )
        else:
            return calculate_portfolio_pnl_vectorized(
                spot,
                strikes, option_types, position_types, quantities, premiums,
                multiplier=multiplier
            )

    def payoff_curve(
        self,
        spot_range: np.ndarray,
        multiplier: float = 100.0
    ) -> np.ndarray:
        """
        Compute theoretical payoff curve at expiration using Numba.

        Parameters
        ----------
        spot_range : np.ndarray
            Range of spot prices to evaluate
        multiplier : float
            Contract multiplier (default 100)

        Returns
        -------
        np.ndarray
            P&L values for each spot price
        """
        strikes, option_types, position_types, quantities, premiums = self._positions_to_arrays()

        stock_qty = self._stock.quantity if self._stock else 0.0
        stock_entry = self._stock.entry_price if self._stock else 0.0

        return compute_payoff_curve(
            spot_range,
            strikes, option_types, position_types, quantities, premiums,
            stock_quantity=stock_qty,
            stock_entry_price=stock_entry,
            multiplier=multiplier
        )

    def find_breakevens(
        self,
        spot_min: float = 50.0,
        spot_max: float = 150.0,
        n_points: int = 1000,
        multiplier: float = 100.0
    ) -> np.ndarray:
        """
        Find breakeven points using Numba-optimized functions.

        Parameters
        ----------
        spot_min : float
            Minimum spot price in range
        spot_max : float
            Maximum spot price in range
        n_points : int
            Number of points for interpolation
        multiplier : float
            Contract multiplier

        Returns
        -------
        np.ndarray
            Breakeven spot prices (where P&L = 0)
        """
        spot_range = np.linspace(spot_min, spot_max, n_points)
        payoff = self.payoff_curve(spot_range, multiplier)
        return find_breakeven_points(payoff, spot_range)

    def risk_metrics_from_simulation(
        self,
        terminal_prices: np.ndarray,
        multiplier: float = 100.0
    ) -> RiskMetrics:
        """
        Compute risk metrics from simulated terminal prices.

        Parameters
        ----------
        terminal_prices : np.ndarray
            Array of simulated terminal spot prices
        multiplier : float
            Contract multiplier

        Returns
        -------
        RiskMetrics
            Named tuple with VaR, CVaR, probability of profit, etc.
        """
        pnl = self.pnl_at_expiry_fast(terminal_prices, multiplier)
        return compute_risk_metrics(pnl)

    # =========================================================================
    # Greeks (Finite Differences)
    # =========================================================================

    def greeks(
        self,
        market: MarketEnvironment,
        spot_bump: float = 0.01,
        rate_bump: float = 0.0001,
        vol_bump: float = 0.01,
    ) -> GreeksResult:
        """
        Calculate portfolio Greeks via central finite differences.

        Parameters
        ----------
        market : MarketEnvironment
            Current market conditions
        spot_bump : float
            Relative spot bump (default 1%)
        rate_bump : float
            Absolute rate bump (default 1bp)
        vol_bump : float
            Absolute vol bump (default 1%)

        Returns
        -------
        GreeksResult
            Portfolio Greeks
        """
        v0 = self.value(market)

        # Delta & Gamma (via finite differences on full portfolio value)
        # Note: Stock delta is already captured via value() which includes stock position
        h_s = market.spot * spot_bump
        v_up = self.value(market.bump_spot(h_s))
        v_down = self.value(market.bump_spot(-h_s))
        delta = (v_up - v_down) / (2 * h_s)
        gamma = (v_up - 2 * v0 + v_down) / (h_s ** 2)

        # Rho
        h_r = rate_bump
        v_r_up = self.value(market.bump_rate(h_r))
        v_r_down = self.value(market.bump_rate(-h_r))
        rho = (v_r_up - v_r_down) / (2 * h_r)

        # Vega (requires model bumping)
        vega = self._compute_vega(market, vol_bump)

        # Theta (approximate via 1-day time decay)
        theta = self._compute_theta(market)

        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
        )

    def _compute_vega(self, market: MarketEnvironment, h: float) -> float:
        """
        Compute vega by bumping model volatility.

        Handles different volatility parameters:
        - "sigma" for GBM and Merton models
        - "v0" for Heston and Bates models
        - "sigma0" for GARCH family models
        """
        import warnings
        params = self._model.get_parameters()

        if "sigma" in params:
            # GBM/Merton models - bump sigma
            model_class = type(self._model)
            sigma = params["sigma"]

            params_up = dict(params)
            params_down = dict(params)
            params_up["sigma"] = sigma + h
            params_down["sigma"] = max(sigma - h, 0.001)

            model_up = model_class(**params_up)
            model_down = model_class(**params_down)

            v_up = self._value_with_model(market, model_up)
            v_down = self._value_with_model(market, model_down)

            return (v_up - v_down) / (2 * h)

        elif "v0" in params:
            # Heston/Bates - bump v0
            v0 = params["v0"]

            model_class = type(self._model)
            params_up = dict(params)
            params_down = dict(params)
            params_up["v0"] = v0 + h * v0
            params_down["v0"] = max(v0 - h * v0, 0.001)

            model_up = model_class(**params_up)
            model_down = model_class(**params_down)

            v_up = self._value_with_model(market, model_up)
            v_down = self._value_with_model(market, model_down)

            return (v_up - v_down) / (2 * h * v0)

        elif "sigma0" in params:
            # GARCH family - bump sigma0 (initial volatility)
            sigma0 = params["sigma0"]

            model_class = type(self._model)
            params_up = dict(params)
            params_down = dict(params)
            params_up["sigma0"] = sigma0 + h
            params_down["sigma0"] = max(sigma0 - h, 0.001)

            model_up = model_class(**params_up)
            model_down = model_class(**params_down)

            v_up = self._value_with_model(market, model_up)
            v_down = self._value_with_model(market, model_down)

            return (v_up - v_down) / (2 * h)

        # Unknown model type - warn and return 0.0
        warnings.warn(
            f"Cannot compute vega for model '{self._model.name}': "
            f"no recognized volatility parameter (sigma, v0, sigma0). "
            f"Available parameters: {list(params.keys())}",
            UserWarning
        )
        return 0.0

    def _compute_theta(self, market: MarketEnvironment) -> float:
        """
        Compute theta via time decay.

        Approximate by creating positions with reduced maturity.
        """
        dt = 1.0 / 365.0  # 1 day

        v0 = self.value(market)

        # Create portfolio with shorter maturities
        total_decayed = 0.0
        for pos in self._positions:
            new_maturity = max(pos.maturity - dt, 0.001)
            from backend.instruments.options import VanillaOption
            decayed_instrument = VanillaOption(
                strike=pos.strike,
                maturity=new_maturity,
                is_call=pos.is_call,
            )
            result = self._engine.price(decayed_instrument, self._model, market)
            total_decayed += pos.quantity * result.price

        if self._stock:
            total_decayed += self._stock.quantity * market.spot

        # Theta = change in value for 1 day time decay (negative)
        return (total_decayed - v0) / dt

    def _value_with_model(self, market: MarketEnvironment, model: Model) -> float:
        """Value portfolio with a different model."""
        total = 0.0
        for pos in self._positions:
            result = self._engine.price(pos.instrument, model, market)
            total += pos.quantity * result.price

        if self._stock:
            total += self._stock.quantity * market.spot

        return total

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def calculate_greeks_surface(
        self,
        market: MarketEnvironment,
        spot_range: np.ndarray,
        greek: str = 'delta',
    ) -> np.ndarray:
        """
        Calculate a Greek across spot prices.

        Parameters
        ----------
        market : MarketEnvironment
            Base market conditions
        spot_range : np.ndarray
            Array of spot prices
        greek : str
            Which Greek to compute ('delta', 'gamma', 'theta', 'vega', 'rho')

        Returns
        -------
        np.ndarray
            Greek values corresponding to each spot
        """
        results = np.zeros(len(spot_range))

        for i, spot in enumerate(spot_range):
            bumped_market = market.with_spot(spot)
            greeks = self.greeks(bumped_market)
            results[i] = getattr(greeks, greek, 0.0)

        return results

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def __len__(self) -> int:
        """Number of option positions."""
        return len(self._positions)

    def __repr__(self) -> str:
        stock_str = f", stock={self._stock.quantity}" if self._stock else ""
        return f"OptionsPortfolio({len(self._positions)} options{stock_str}, model={self._model.name})"

    def summary(self) -> str:
        """Generate a human-readable summary of the portfolio."""
        lines = [f"=== Portfolio Summary (Model: {self._model.name}) ==="]

        for i, pos in enumerate(self._positions):
            direction = "Long" if pos.is_long else "Short"
            opt_type = "Call" if pos.is_call else "Put"
            lines.append(
                f"{i+1}. {direction} {abs(pos.quantity)}x {opt_type} @ K={pos.strike:.2f} "
                f"T={pos.maturity:.2f}y (premium={pos.premium:.2f})"
            )

        if self._stock:
            direction = "Long" if self._stock.is_long else "Short"
            lines.append(
                f"Stock: {direction} {abs(self._stock.quantity)} shares "
                f"@ {self._stock.entry_price:.2f}"
            )

        return "\n".join(lines)


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    from backend.models.gbm import GBMModel
    from backend.models.heston import HestonModel
    from backend.core.market import MarketEnvironment
    from backend.engines import BSAnalyticEngine, FFTEngine
    from .positions import long_call, short_call, long_put, long_stock

    print("=" * 50)
    print("Portfolio Module Smoke Test")
    print("=" * 50)

    # Create model and market
    gbm = GBMModel(sigma=0.20)
    market = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.02)

    print(f"\n--- Setup ---")
    print(f"Model: {gbm}")
    print(f"Market: S={market.spot}, r={market.rate}, q={market.dividend_yield}")

    # Build portfolio (Bull Call Spread)
    print(f"\n--- Bull Call Spread ---")
    portfolio = OptionsPortfolio(model=gbm)
    portfolio.add(long_call(strike=95, maturity=0.5, premium=8.0))
    portfolio.add(short_call(strike=105, maturity=0.5, premium=3.0))

    print(portfolio.summary())

    # Portfolio value
    value = portfolio.value(market)
    print(f"\nPortfolio Value: ${value:.4f}")

    # Greeks
    greeks = portfolio.greeks(market)
    print(f"\n--- Greeks ---")
    print(f"Delta: {greeks.delta:.4f}")
    print(f"Gamma: {greeks.gamma:.6f}")
    print(f"Theta: {greeks.theta:.4f}")
    print(f"Vega: {greeks.vega:.4f}")
    print(f"Rho: {greeks.rho:.4f}")

    # P&L at expiry
    print(f"\n--- P&L at Expiry ---")
    spots = np.array([85, 90, 95, 100, 105, 110, 115])
    pnls = portfolio.pnl_at_expiry(spots)
    for s, p in zip(spots, pnls):
        print(f"  Spot={s}: P&L=${p:.2f}")

    # Test with Heston model
    print(f"\n--- Heston Model ---")
    heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    portfolio_heston = OptionsPortfolio(model=heston)
    portfolio_heston.add(long_call(strike=100, maturity=0.5, premium=5.0))

    value_heston = portfolio_heston.value(market)
    greeks_heston = portfolio_heston.greeks(market)
    print(f"Value: ${value_heston:.4f}")
    print(f"Delta: {greeks_heston.delta:.4f}")

    # Test with stock
    print(f"\n--- Portfolio with Stock ---")
    covered_call = OptionsPortfolio(model=gbm)
    covered_call.add(long_stock(quantity=100, entry_price=100.0))
    covered_call.add(short_call(strike=105, maturity=0.5, premium=3.0))

    print(covered_call.summary())
    covered_greeks = covered_call.greeks(market)
    print(f"Covered Call Delta: {covered_greeks.delta:.4f}")  # Should be ~0.5-0.6

    print("\n" + "=" * 50)
    print("Portfolio smoke test passed")
    print("=" * 50)
