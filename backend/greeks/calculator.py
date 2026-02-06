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


# Valid Greek names for validation
VALID_GREEKS = frozenset({
    'price',
    # First order
    'delta', 'gamma', 'vega', 'theta', 'rho',
    # Second order
    'vanna', 'volga', 'charm', 'veta',
    # Third order
    'speed', 'zomma', 'color', 'ultima'
})


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

        # Validate instrument has strike
        if not hasattr(instrument, 'strike'):
            raise ValueError(
                f"Analytic Greeks require an instrument with fixed strike. "
                f"{type(instrument).__name__} has no strike attribute."
            )

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
        from backend.greeks.numerical import ModelNumericalGreeks, GreeksBumpConfig

        config = GreeksBumpConfig(
            spot_bump=self.spot_bump,
            vol_bump=self.vol_bump,
            time_bump_days=self.time_bump_days,
            rate_bump=self.rate_bump
        )
        num_calc = ModelNumericalGreeks(config)

        num_greeks = num_calc.calculate(engine, instrument, model, market)

        if include_higher_order:
            # Calculate second and third order numerically
            price = engine.price(instrument, model, market).price

            # Calculate higher-order Greeks using cross finite differences
            higher_order = self._numerical_higher_order_greeks(
                engine, instrument, model, market
            )

            return AllGreeksResult(
                price=price,
                delta=num_greeks.delta,
                gamma=num_greeks.gamma,
                vega=num_greeks.vega,
                theta=num_greeks.theta,
                rho=num_greeks.rho,
                # Second order
                vanna=higher_order['vanna'],
                volga=higher_order['volga'],
                charm=higher_order['charm'],
                veta=higher_order['veta'],
                # Third order
                speed=higher_order['speed'],
                zomma=higher_order['zomma'],
                color=higher_order['color'],
                ultima=higher_order['ultima']
            )
        else:
            return GreeksResult(
                delta=num_greeks.delta,
                gamma=num_greeks.gamma,
                vega=num_greeks.vega,
                theta=num_greeks.theta,
                rho=num_greeks.rho
            )

    def _numerical_higher_order_greeks(
        self,
        engine: PricingEngine,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment
    ) -> dict:
        """
        Calculate higher-order Greeks using numerical cross finite differences.

        Note: Volatility-related Greeks (vanna, volga, zomma, veta, ultima) require
        a model with a 'sigma' parameter that can be bumped. For stochastic
        volatility models (Heston, Bates), these are approximated by bumping
        the initial volatility (sqrt(v0)).

        Returns
        -------
        dict
            Dictionary with vanna, volga, charm, veta, speed, zomma, color, ultima
        """
        from backend.instruments.options import VanillaOption
        from backend.models.gbm import GBMModel
        from backend.models.heston import HestonModel
        from backend.models.bates import BatesModel
        from backend.models.merton import MertonModel

        # Get model parameters and determine how to bump volatility
        params = model.get_parameters()
        has_sigma = 'sigma' in params
        has_v0 = 'v0' in params  # For Heston/Bates models

        # Helper to create bumped model preserving model type
        def create_bumped_model(base_model: Model, vol_bump: float) -> Model:
            """Create a model with bumped volatility, preserving model type."""
            p = base_model.get_parameters()
            if isinstance(base_model, GBMModel):
                return GBMModel(sigma=p['sigma'] + vol_bump)
            elif isinstance(base_model, MertonModel):
                return MertonModel(
                    sigma=p['sigma'] + vol_bump,
                    lambda_j=p['lambda_j'],
                    mu_j=p['mu_j'],
                    sigma_j=p['sigma_j']
                )
            elif isinstance(base_model, HestonModel):
                # Bump v0 (initial variance) to approximate vol bump
                # vol_bump^2 approximately equals variance bump for small changes
                new_v0 = max(p['v0'] + 2 * np.sqrt(p['v0']) * vol_bump, 1e-8)
                return HestonModel(
                    v0=new_v0,
                    kappa=p['kappa'],
                    theta=p['theta'],
                    xi=p['xi'],
                    rho=p['rho']
                )
            elif isinstance(base_model, BatesModel):
                new_v0 = max(p['v0'] + 2 * np.sqrt(p['v0']) * vol_bump, 1e-8)
                return BatesModel(
                    v0=new_v0,
                    kappa=p['kappa'],
                    theta=p['theta'],
                    xi=p['xi'],
                    rho=p['rho'],
                    lambda_j=p['lambda_j'],
                    mu_j=p['mu_j'],
                    sigma_j=p['sigma_j']
                )
            else:
                # Fallback: can't bump unknown model type
                return None

        # Bump sizes
        h_s = market.spot * self.spot_bump
        h_t = self.time_bump_days / 365.0
        h_v = self.vol_bump

        # Create bumped models once for efficiency
        mod_up = create_bumped_model(model, h_v)
        mod_down = create_bumped_model(model, -h_v)
        can_bump_vol = mod_up is not None and mod_down is not None

        # Helper to compute delta at a given market
        def get_delta(m):
            """Compute delta at market m."""
            s_up = m.bump_spot(h_s)
            s_down = m.bump_spot(-h_s)
            v_up = engine.price(instrument, model, s_up).price
            v_down = engine.price(instrument, model, s_down).price
            return (v_up - v_down) / (2 * h_s)

        # Helper to compute gamma at a given market
        def get_gamma(m):
            """Compute gamma at market m."""
            s_up = m.bump_spot(h_s)
            s_down = m.bump_spot(-h_s)
            v_up = engine.price(instrument, model, s_up).price
            v_mid = engine.price(instrument, model, m).price
            v_down = engine.price(instrument, model, s_down).price
            return (v_up - 2 * v_mid + v_down) / (h_s ** 2)

        # Helper to compute vega at a given market
        def get_vega(m):
            """Compute vega at market m."""
            if not can_bump_vol:
                return 0.0
            v_up = engine.price(instrument, mod_up, m).price
            v_down = engine.price(instrument, mod_down, m).price
            return (v_up - v_down) / (2 * h_v)

        # Create decayed instrument for time derivatives
        def get_decayed_instrument():
            if not hasattr(instrument, 'strike'):
                return None
            return VanillaOption(
                strike=instrument.strike,
                maturity=max(instrument.maturity - h_t, 0.001),
                is_call=instrument.is_call
            )

        # Initialize results
        result = {
            'vanna': 0.0,
            'volga': 0.0,
            'charm': 0.0,
            'veta': 0.0,
            'speed': 0.0,
            'zomma': 0.0,
            'color': 0.0,
            'ultima': 0.0
        }

        # VANNA = dDelta/dSigma
        if can_bump_vol:
            # Delta at vol + h
            s_up = market.bump_spot(h_s)
            s_down = market.bump_spot(-h_s)
            v_up_vup = engine.price(instrument, mod_up, s_up).price
            v_down_vup = engine.price(instrument, mod_up, s_down).price
            delta_vup = (v_up_vup - v_down_vup) / (2 * h_s)

            # Delta at vol - h
            v_up_vdown = engine.price(instrument, mod_down, s_up).price
            v_down_vdown = engine.price(instrument, mod_down, s_down).price
            delta_vdown = (v_up_vdown - v_down_vdown) / (2 * h_s)

            result['vanna'] = (delta_vup - delta_vdown) / (2 * h_v)

            # VOLGA = dVega/dSigma
            # Need models with double bump for volga
            mod_2up = create_bumped_model(model, 2 * h_v)
            mod_2down = create_bumped_model(model, -2 * h_v)
            if mod_2up is not None and mod_2down is not None:
                v_2up = engine.price(instrument, mod_2up, market).price
                v_mid = engine.price(instrument, model, market).price
                v_2down = engine.price(instrument, mod_2down, market).price
                # Volga = d²V/dσ² using central difference on vega
                result['volga'] = (v_2up - 2 * v_mid + v_2down) / (h_v ** 2)

            # ZOMMA = dGamma/dSigma
            # Compute gamma at two volatility levels
            v_up_up = engine.price(instrument, mod_up, s_up).price
            v_mid_up = engine.price(instrument, mod_up, market).price
            v_down_up = engine.price(instrument, mod_up, s_down).price
            gamma_vol_up = (v_up_up - 2 * v_mid_up + v_down_up) / (h_s ** 2)

            v_up_down = engine.price(instrument, mod_down, s_up).price
            v_mid_down = engine.price(instrument, mod_down, market).price
            v_down_down = engine.price(instrument, mod_down, s_down).price
            gamma_vol_down = (v_up_down - 2 * v_mid_down + v_down_down) / (h_s ** 2)

            result['zomma'] = (gamma_vol_up - gamma_vol_down) / (2 * h_v)

            # ULTIMA = dVolga/dSigma (third derivative wrt sigma)
            # Compute using finite difference on volga
            if mod_2up is not None and mod_2down is not None:
                # Volga at mod_up
                v_2up_up = engine.price(instrument, mod_2up, market).price
                vega_up = engine.price(instrument, mod_up, market).price
                v_0 = engine.price(instrument, model, market).price
                volga_at_up = (v_2up_up - 2 * vega_up + v_0) / (h_v ** 2)

                # Volga at mod_down (requires mod at -2h which we have)
                v_0_for_down = vega_up = engine.price(instrument, model, market).price
                v_down_mid = engine.price(instrument, mod_down, market).price
                v_2down_val = engine.price(instrument, mod_2down, market).price
                volga_at_down = (v_0_for_down - 2 * v_down_mid + v_2down_val) / (h_v ** 2)

                result['ultima'] = (volga_at_up - volga_at_down) / (2 * h_v)

        # CHARM = dDelta/dt (time decay of delta)
        decayed = get_decayed_instrument()
        if decayed is not None:
            delta_now = get_delta(market)
            # Compute delta for decayed instrument
            s_up = market.bump_spot(h_s)
            s_down = market.bump_spot(-h_s)
            v_up_later = engine.price(decayed, model, s_up).price
            v_down_later = engine.price(decayed, model, s_down).price
            delta_later = (v_up_later - v_down_later) / (2 * h_s)

            result['charm'] = (delta_later - delta_now) / self.time_bump_days

            # COLOR = dGamma/dt
            gamma_now = get_gamma(market)
            v_up_later = engine.price(decayed, model, s_up).price
            v_mid_later = engine.price(decayed, model, market).price
            v_down_later = engine.price(decayed, model, s_down).price
            gamma_later = (v_up_later - 2 * v_mid_later + v_down_later) / (h_s ** 2)

            result['color'] = (gamma_later - gamma_now) / self.time_bump_days

        # VETA = dVega/dt
        if can_bump_vol and decayed is not None:
            # Vega now
            vega_now = get_vega(market)

            # Vega later (with decayed instrument)
            v_vup_later = engine.price(decayed, mod_up, market).price
            v_vdown_later = engine.price(decayed, mod_down, market).price
            vega_later = (v_vup_later - v_vdown_later) / (2 * h_v)

            result['veta'] = (vega_later - vega_now) / self.time_bump_days

        # SPEED = dGamma/dS (third derivative wrt spot)
        # Using four-point stencil
        s_2up = market.bump_spot(2 * h_s)
        s_up = market.bump_spot(h_s)
        s_down = market.bump_spot(-h_s)
        s_2down = market.bump_spot(-2 * h_s)

        v_2up = engine.price(instrument, model, s_2up).price
        v_up = engine.price(instrument, model, s_up).price
        v_down = engine.price(instrument, model, s_down).price
        v_2down = engine.price(instrument, model, s_2down).price

        # Third derivative approximation
        result['speed'] = (v_2up - 2 * v_up + 2 * v_down - v_2down) / (2 * h_s ** 3)

        return result

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

        Raises
        ------
        ValueError
            If greek name is not valid
        """
        # Validate Greek name
        if greek not in VALID_GREEKS:
            raise ValueError(
                f"Unknown Greek '{greek}'. Valid values: {sorted(VALID_GREEKS)}"
            )

        results = np.zeros(len(spot_range))

        higher_order_greeks = {'vanna', 'volga', 'charm', 'veta',
                               'speed', 'zomma', 'color', 'ultima'}

        for i, spot in enumerate(spot_range):
            bumped_market = market.with_spot(spot)
            greeks = self.calculate(
                engine, instrument, model, bumped_market,
                include_higher_order=(greek in higher_order_greeks)
            )
            results[i] = getattr(greeks, greek)

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
