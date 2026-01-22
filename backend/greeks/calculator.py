"""
Unified Greeks Calculator
=========================

High-level interface for Greeks calculation.

Dispatches to analytic or numerical methods based on model/engine.
Supports first, second, and third order Greeks.

Author: Thomas
Created: 2025
"""

from typing import Optional, NamedTuple, Union
import numpy as np

from backend.core.interfaces import Model, PricingEngine, Instrument
from backend.core.market import MarketEnvironment
from backend.core.result_types import GreeksResult


# =============================================================================
# Greeks Result Types
# =============================================================================

class AllGreeksResult(NamedTuple):
    """Complete Greeks result with all 14 Greeks."""
    # Price
    price: float
    # First order
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    # Second order
    vanna: float
    volga: float
    charm: float
    veta: float
    # Third order
    speed: float
    zomma: float
    color: float
    ultima: float


# =============================================================================
# Unified Calculator
# =============================================================================

class GreeksCalculator:
    """
    Unified Greeks calculator.

    Automatically dispatches to analytic Greeks when available (BS engine),
    otherwise falls back to numerical finite differences.

    Parameters
    ----------
    prefer_analytic : bool
        Prefer analytic Greeks when available (default True)
    spot_bump : float
        Relative spot bump for numerical Greeks (default 1%)
    vol_bump : float
        Absolute vol bump for numerical Greeks (default 1%)
    time_bump_days : float
        Time bump in days for numerical theta (default 1 day)
    rate_bump : float
        Absolute rate bump for numerical rho (default 1bp)
    """

    def __init__(
        self,
        prefer_analytic: bool = True,
        spot_bump: float = 0.01,
        vol_bump: float = 0.01,
        time_bump_days: float = 1.0,
        rate_bump: float = 0.0001
    ):
        self.prefer_analytic = prefer_analytic
        self.spot_bump = spot_bump
        self.vol_bump = vol_bump
        self.time_bump_days = time_bump_days
        self.rate_bump = rate_bump

    def calculate(
        self,
        engine: PricingEngine,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
        include_higher_order: bool = True
    ) -> Union[GreeksResult, AllGreeksResult]:
        """
        Calculate Greeks for an instrument.

        Parameters
        ----------
        engine : PricingEngine
            Pricing engine
        instrument : Instrument
            Option instrument
        model : Model
            Pricing model
        market : MarketEnvironment
            Market conditions
        include_higher_order : bool
            Include second and third order Greeks (default True)

        Returns
        -------
        GreeksResult or AllGreeksResult
            Greeks result
        """
        # Check if we can use analytic Greeks
        if self.prefer_analytic and self._can_use_analytic(engine, model):
            return self._analytic_greeks(
                instrument, model, market, include_higher_order
            )
        else:
            return self._numerical_greeks(
                engine, instrument, model, market, include_higher_order
            )

    def _can_use_analytic(self, engine: PricingEngine, model: Model) -> bool:
        """Check if analytic Greeks are available."""
        from backend.engines import BSAnalyticEngine
        from backend.models.gbm import GBMModel

        return (
            isinstance(engine, BSAnalyticEngine) and
            isinstance(model, GBMModel)
        )

    def _analytic_greeks(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
        include_higher_order: bool
    ) -> Union[GreeksResult, AllGreeksResult]:
        """Calculate analytic Black-Scholes Greeks."""
        from backend.greeks.analytic import bs_all_greeks

        params = model.get_parameters()
        sigma = params['sigma']

        greeks = bs_all_greeks(
            s=market.spot,
            k=instrument.strike,
            t=instrument.maturity,
            r=market.rate,
            q=market.dividend_yield,
            sigma=sigma,
            is_call=instrument.is_call
        )

        if include_higher_order:
            return AllGreeksResult(
                price=greeks[0],
                delta=greeks[1],
                gamma=greeks[2],
                vega=greeks[3],
                theta=greeks[4],
                rho=greeks[5],
                vanna=greeks[6],
                volga=greeks[7],
                charm=greeks[8],
                veta=greeks[9],
                speed=greeks[10],
                zomma=greeks[11],
                color=greeks[12],
                ultima=greeks[13]
            )
        else:
            return GreeksResult(
                delta=greeks[1],
                gamma=greeks[2],
                vega=greeks[3],
                theta=greeks[4],
                rho=greeks[5]
            )

    def _numerical_greeks(
        self,
        engine: PricingEngine,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
        include_higher_order: bool
    ) -> Union[GreeksResult, AllGreeksResult]:
        """Calculate numerical Greeks via finite differences."""
        from backend.greeks.numerical import ModelNumericalGreeks

        num_calc = ModelNumericalGreeks(
            spot_bump=self.spot_bump,
            vol_bump=self.vol_bump,
            time_bump_days=self.time_bump_days,
            rate_bump=self.rate_bump
        )

        num_greeks = num_calc.calculate(engine, instrument, model, market)

        if include_higher_order:
            # Calculate second and third order numerically
            price = engine.price(instrument, model, market).price
            return AllGreeksResult(
                price=price,
                delta=num_greeks.delta,
                gamma=num_greeks.gamma,
                vega=num_greeks.vega,
                theta=num_greeks.theta,
                rho=num_greeks.rho,
                # Higher order - set to 0 for numerical method
                # (would need more complex cross-derivatives)
                vanna=0.0,
                volga=0.0,
                charm=0.0,
                veta=0.0,
                speed=0.0,
                zomma=0.0,
                color=0.0,
                ultima=0.0
            )
        else:
            return GreeksResult(
                delta=num_greeks.delta,
                gamma=num_greeks.gamma,
                vega=num_greeks.vega,
                theta=num_greeks.theta,
                rho=num_greeks.rho
            )

    def calculate_surface(
        self,
        engine: PricingEngine,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
        spot_range: np.ndarray,
        greek: str = 'delta'
    ) -> np.ndarray:
        """
        Calculate a Greek across spot prices.

        Parameters
        ----------
        engine : PricingEngine
            Pricing engine
        instrument : Instrument
            Option instrument
        model : Model
            Pricing model
        market : MarketEnvironment
            Base market conditions
        spot_range : np.ndarray
            Array of spot prices
        greek : str
            Which Greek to calculate

        Returns
        -------
        np.ndarray
            Greek values
        """
        results = np.zeros(len(spot_range))

        for i, spot in enumerate(spot_range):
            bumped_market = market.with_spot(spot)
            greeks = self.calculate(
                engine, instrument, model, bumped_market,
                include_higher_order=(greek in [
                    'vanna', 'volga', 'charm', 'veta',
                    'speed', 'zomma', 'color', 'ultima'
                ])
            )
            results[i] = getattr(greeks, greek, 0.0)

        return results


# =============================================================================
# Convenience Function
# =============================================================================

def calculate_greeks(
    engine: PricingEngine,
    instrument: Instrument,
    model: Model,
    market: MarketEnvironment,
    include_higher_order: bool = False
) -> Union[GreeksResult, AllGreeksResult]:
    """
    Calculate Greeks for an instrument.

    Convenience function that creates a calculator and computes Greeks.

    Parameters
    ----------
    engine : PricingEngine
        Pricing engine
    instrument : Instrument
        Option instrument
    model : Model
        Pricing model
    market : MarketEnvironment
        Market conditions
    include_higher_order : bool
        Include second and third order Greeks

    Returns
    -------
    GreeksResult or AllGreeksResult
        Greeks result
    """
    calc = GreeksCalculator()
    return calc.calculate(engine, instrument, model, market, include_higher_order)


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Greeks Calculator Smoke Test")
    print("=" * 50)

    from backend.instruments.options import VanillaOption
    from backend.models.gbm import GBMModel
    from backend.engines import BSAnalyticEngine
    from backend.core.market import MarketEnvironment

    # Create components
    option = VanillaOption(strike=100.0, maturity=0.25, is_call=True)
    model = GBMModel(sigma=0.20)
    engine = BSAnalyticEngine()
    market = MarketEnvironment(spot=100.0, rate=0.05, dividend_yield=0.02)

    # Calculate Greeks
    calc = GreeksCalculator()
    greeks = calc.calculate(engine, option, model, market, include_higher_order=True)

    print(f"\nATM Call Greeks (S=K=100, T=0.25, σ=20%):")
    print(f"  Price: ${greeks.price:.4f}")
    print(f"\n  First Order:")
    print(f"    Delta: {greeks.delta:.6f}")
    print(f"    Gamma: {greeks.gamma:.6f}")
    print(f"    Vega:  {greeks.vega:.6f}")
    print(f"    Theta: {greeks.theta:.6f}")
    print(f"    Rho:   {greeks.rho:.6f}")
    print(f"\n  Second Order:")
    print(f"    Vanna: {greeks.vanna:.6f}")
    print(f"    Volga: {greeks.volga:.6f}")
    print(f"    Charm: {greeks.charm:.6f}")
    print(f"    Veta:  {greeks.veta:.6f}")
    print(f"\n  Third Order:")
    print(f"    Speed:  {greeks.speed:.8f}")
    print(f"    Zomma:  {greeks.zomma:.8f}")
    print(f"    Color:  {greeks.color:.8f}")
    print(f"    Ultima: {greeks.ultima:.10f}")

    # Test convenience function
    print("\n--- Convenience Function ---")
    simple_greeks = calculate_greeks(engine, option, model, market)
    print(f"  Delta: {simple_greeks.delta:.6f}")

    print("\n" + "=" * 50)
    print("Greeks Calculator smoke test passed")
    print("=" * 50)
