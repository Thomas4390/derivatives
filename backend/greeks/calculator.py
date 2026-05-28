"""
Unified Greeks Calculator
=========================

High-level interface for Greeks calculation.

Dispatches to analytic or numerical methods based on model/engine.
Supports first, second, and third order Greeks.

Author: Thomas Vaudescal
Created: 2026
"""

import logging

import numpy as np

from backend.utils.constants.greeks import (
    ULTIMA_SCALE,
    VALID_GREEKS,
    VANNA_SCALE,
    VEGA_SCALE,
    VOLGA_SCALE,
    ZOMMA_SCALE,
)
from backend.utils.constants.monte_carlo import (
    DEFAULT_RATE_BUMP,
    DEFAULT_SPOT_BUMP,
    DEFAULT_TIME_BUMP_DAYS,
    DEFAULT_VOL_BUMP,
)
from backend.utils.constants.time import CALENDAR_DAYS_PER_YEAR
from backend.core.interfaces import Instrument, Model, Priceable, PricingEngine
from backend.core.market import MarketEnvironment
from backend.core.result_types import GreeksResult
from backend.greeks._instrument_utils import create_decayed_instrument
from backend.greeks.strategies import (
    DEFAULT_STRATEGIES,
    GreeksStrategy,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Backward compatibility
# =============================================================================

# AllGreeksResult is now unified into GreeksResult (which includes a price field).
# Keep the alias for any external code that references AllGreeksResult.
AllGreeksResult = GreeksResult


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
        spot_bump: float = DEFAULT_SPOT_BUMP,
        vol_bump: float = DEFAULT_VOL_BUMP,
        time_bump_days: float = DEFAULT_TIME_BUMP_DAYS,
        rate_bump: float = DEFAULT_RATE_BUMP,
    ):
        self.prefer_analytic = prefer_analytic
        self.spot_bump = spot_bump
        self.vol_bump = vol_bump
        self.time_bump_days = time_bump_days
        self.rate_bump = rate_bump

    def calculate(
        self,
        engine: PricingEngine,
        instrument: Priceable,
        model: Model,
        market: MarketEnvironment,
        include_higher_order: bool = True,
        method: str | None = None,
    ) -> GreeksResult:
        """
        Calculate Greeks for any priceable instrument.

        Uses the Strategy Pattern to dispatch to the best available method.
        Priority: analytic > numerical (bump-and-reprice).

        Parameters
        ----------
        engine : PricingEngine
            Pricing engine
        instrument : Priceable
            Any priceable contract (VanillaOption, Autocallable, etc.)
        model : Model
            Pricing model
        market : MarketEnvironment
            Market conditions
        include_higher_order : bool
            Include second and third order Greeks (default True)
        method : str, optional
            Force a specific strategy: "analytic" or "numerical".
            If None, auto-selects the best available strategy.

        Returns
        -------
        GreeksResult
            Greeks result (with price and all orders)

        Raises
        ------
        ValueError
            If the requested method cannot compute Greeks for this pair.
        """
        from backend.greeks.numerical import GreeksBumpConfig

        config = GreeksBumpConfig(
            spot_bump=self.spot_bump,
            vol_bump=self.vol_bump,
            time_bump_days=self.time_bump_days,
            rate_bump=self.rate_bump,
        )

        if method is not None:
            strategy = self._get_strategy_by_name(method)
            # Engine strategy uses engine-aware check
            can = (
                strategy.can_compute_with_engine(instrument, model, engine)
                if hasattr(strategy, "can_compute_with_engine")
                else strategy.can_compute(instrument, model)
            )
            if not can:
                raise ValueError(
                    f"Strategy '{method}' cannot compute Greeks for "
                    f"{type(instrument).__name__} / {model.name}"
                )
            logger.debug(
                "Using %s Greeks (forced) for %s/%s",
                strategy.name,
                type(instrument).__name__,
                model.name,
            )
            return strategy.compute(
                engine, instrument, model, market, config, include_higher_order
            )

        # Auto-select: try strategies in priority order
        strategy = self._auto_select(instrument, model, engine)
        logger.debug(
            "Using %s Greeks (auto) for %s/%s",
            strategy.name,
            type(instrument).__name__,
            model.name,
        )
        return strategy.compute(
            engine, instrument, model, market, config, include_higher_order
        )

    def _auto_select(
        self,
        instrument: Priceable,
        model: Model,
        engine: PricingEngine,
    ) -> GreeksStrategy:
        """Select the best strategy for this instrument/model pair."""
        from backend.greeks.strategies import EngineGreeksStrategy

        for strategy in DEFAULT_STRATEGIES:
            if strategy.name == "analytic" and not self.prefer_analytic:
                continue
            # Engine strategy needs the engine to decide compatibility
            if isinstance(strategy, EngineGreeksStrategy):
                if strategy.can_compute_with_engine(instrument, model, engine):
                    return strategy
                continue
            if strategy.can_compute(instrument, model):
                return strategy
        # Should never reach here — NumericalGreeksStrategy always returns True
        return DEFAULT_STRATEGIES[-1]

    @staticmethod
    def _get_strategy_by_name(name: str) -> GreeksStrategy:
        """Look up a strategy by name."""
        for strategy in DEFAULT_STRATEGIES:
            if strategy.name == name:
                return strategy
        available = [s.name for s in DEFAULT_STRATEGIES]
        raise ValueError(f"Unknown Greeks method '{name}'. Available: {available}")

    def _numerical_higher_order_greeks(
        self,
        engine: PricingEngine,
        instrument: Priceable,
        model: Model,
        market: MarketEnvironment,
    ) -> dict[str, float]:
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
        from backend.models.vol_bump import create_vol_bumped_model

        # Local cache to avoid redundant engine.price() calls.
        # We keep references alive to prevent id() reuse after GC.
        _price_cache: dict[tuple, float] = {}
        _cache_refs: list = []

        def cached_price(inst, mod, mkt):
            key = (id(inst), id(mod), id(mkt))
            if key not in _price_cache:
                _cache_refs.extend([inst, mod, mkt])
                _price_cache[key] = engine.price(inst, mod, mkt).price
            return _price_cache[key]

        # Bump sizes
        h_s = market.spot * self.spot_bump
        h_t = self.time_bump_days / CALENDAR_DAYS_PER_YEAR
        h_v = self.vol_bump

        # Create bumped models once for efficiency
        mod_up = create_vol_bumped_model(model, h_v)
        mod_down = create_vol_bumped_model(model, -h_v)
        can_bump_vol = mod_up is not None and mod_down is not None

        # Helper to compute delta at a given market
        def get_delta(m):
            """Compute delta at market m."""
            s_up = m.bump_spot(h_s)
            s_down = m.bump_spot(-h_s)
            v_up = cached_price(instrument, model, s_up)
            v_down = cached_price(instrument, model, s_down)
            return (v_up - v_down) / (2 * h_s)

        # Helper to compute gamma at a given market
        def get_gamma(m):
            """Compute gamma at market m."""
            s_up = m.bump_spot(h_s)
            s_down = m.bump_spot(-h_s)
            v_up = cached_price(instrument, model, s_up)
            v_mid = cached_price(instrument, model, m)
            v_down = cached_price(instrument, model, s_down)
            return (v_up - 2 * v_mid + v_down) / (h_s**2)

        # Helper to compute vega at a given market
        def get_vega(m):
            """Compute vega at market m."""
            if not can_bump_vol:
                return 0.0
            v_up = cached_price(instrument, mod_up, m)
            v_down = cached_price(instrument, mod_down, m)
            return (v_up - v_down) / (2 * h_v)

        # Create decayed instrument for time derivatives (works for all types)
        def get_decayed_instrument():
            new_T = max(instrument.maturity - h_t, 0.001)
            return create_decayed_instrument(instrument, new_T)

        # Initialize results
        result = {
            "vanna": 0.0,
            "volga": 0.0,
            "charm": 0.0,
            "veta": 0.0,
            "speed": 0.0,
            "zomma": 0.0,
            "color": 0.0,
            "ultima": 0.0,
        }

        # VANNA = dDelta/dSigma
        if can_bump_vol:
            # Delta at vol + h
            s_up = market.bump_spot(h_s)
            s_down = market.bump_spot(-h_s)
            v_up_vup = cached_price(instrument, mod_up, s_up)
            v_down_vup = cached_price(instrument, mod_up, s_down)
            delta_vup = (v_up_vup - v_down_vup) / (2 * h_s)

            # Delta at vol - h
            v_up_vdown = cached_price(instrument, mod_down, s_up)
            v_down_vdown = cached_price(instrument, mod_down, s_down)
            delta_vdown = (v_up_vdown - v_down_vdown) / (2 * h_s)

            result["vanna"] = (delta_vup - delta_vdown) / (2 * h_v) / VANNA_SCALE

            # VOLGA = dVega/dSigma
            # Need models with double bump for volga
            mod_2up = create_vol_bumped_model(model, 2 * h_v)
            mod_2down = create_vol_bumped_model(model, -2 * h_v)
            if mod_2up is not None and mod_2down is not None:
                v_2up = cached_price(instrument, mod_2up, market)
                v_mid = cached_price(instrument, model, market)
                v_2down = cached_price(instrument, mod_2down, market)
                # Volga = d^2V/d sigma^2 using central difference on vega
                result["volga"] = (
                    (v_2up - 2 * v_mid + v_2down) / (2 * h_v) ** 2 / VOLGA_SCALE
                )

            # ZOMMA = dGamma/dSigma
            # Compute gamma at two volatility levels
            v_up_up = cached_price(instrument, mod_up, s_up)
            v_mid_up = cached_price(instrument, mod_up, market)
            v_down_up = cached_price(instrument, mod_up, s_down)
            gamma_vol_up = (v_up_up - 2 * v_mid_up + v_down_up) / (h_s**2)

            v_up_down = cached_price(instrument, mod_down, s_up)
            v_mid_down = cached_price(instrument, mod_down, market)
            v_down_down = cached_price(instrument, mod_down, s_down)
            gamma_vol_down = (v_up_down - 2 * v_mid_down + v_down_down) / (h_s**2)

            result["zomma"] = (gamma_vol_up - gamma_vol_down) / (2 * h_v) / ZOMMA_SCALE

            # ULTIMA = dVolga/dSigma (third derivative wrt sigma)
            # Compute using finite difference on volga
            if mod_2up is not None and mod_2down is not None:
                # Volga at mod_up
                v_2up_up = cached_price(instrument, mod_2up, market)
                v_at_up = cached_price(instrument, mod_up, market)
                v_0 = cached_price(instrument, model, market)
                volga_at_up = (v_2up_up - 2 * v_at_up + v_0) / (h_v**2)

                # Volga at mod_down (requires mod at -2h which we have)
                v_0_for_down = v_0
                v_down_mid = cached_price(instrument, mod_down, market)
                v_2down_val = cached_price(instrument, mod_2down, market)
                volga_at_down = (v_0_for_down - 2 * v_down_mid + v_2down_val) / (h_v**2)

                result["ultima"] = (
                    (volga_at_up - volga_at_down) / (2 * h_v) / ULTIMA_SCALE
                )

        # CHARM = dDelta/dt (time decay of delta)
        decayed = get_decayed_instrument()
        if decayed is not None:
            delta_now = get_delta(market)
            # Compute delta for decayed instrument
            s_up = market.bump_spot(h_s)
            s_down = market.bump_spot(-h_s)
            v_up_later = cached_price(decayed, model, s_up)
            v_down_later = cached_price(decayed, model, s_down)
            delta_later = (v_up_later - v_down_later) / (2 * h_s)

            result["charm"] = (delta_later - delta_now) / self.time_bump_days

            # COLOR = dGamma/dt
            gamma_now = get_gamma(market)
            v_up_later = cached_price(decayed, model, s_up)
            v_mid_later = cached_price(decayed, model, market)
            v_down_later = cached_price(decayed, model, s_down)
            gamma_later = (v_up_later - 2 * v_mid_later + v_down_later) / (h_s**2)

            result["color"] = (gamma_later - gamma_now) / self.time_bump_days

        # VETA = dVega/dt
        if can_bump_vol and decayed is not None:
            # Vega now
            vega_now = get_vega(market)

            # Vega later (with decayed instrument)
            v_vup_later = cached_price(decayed, mod_up, market)
            v_vdown_later = cached_price(decayed, mod_down, market)
            vega_later = (v_vup_later - v_vdown_later) / (2 * h_v)

            result["veta"] = (vega_later - vega_now) / self.time_bump_days / VEGA_SCALE

        # SPEED = dGamma/dS (third derivative wrt spot)
        # Using four-point stencil
        s_2up = market.bump_spot(2 * h_s)
        s_up = market.bump_spot(h_s)
        s_down = market.bump_spot(-h_s)
        s_2down = market.bump_spot(-2 * h_s)

        v_2up = cached_price(instrument, model, s_2up)
        v_up = cached_price(instrument, model, s_up)
        v_down = cached_price(instrument, model, s_down)
        v_2down = cached_price(instrument, model, s_2down)

        # Third derivative approximation
        result["speed"] = (v_2up - 2 * v_up + 2 * v_down - v_2down) / (2 * h_s**3)

        return result

    def calculate_surface(
        self,
        engine: PricingEngine,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
        spot_range: np.ndarray,
        greek: str = "delta",
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

        higher_order_greeks = {
            "vanna",
            "volga",
            "charm",
            "veta",
            "speed",
            "zomma",
            "color",
            "ultima",
        }

        for i, spot in enumerate(spot_range):
            bumped_market = market.with_spot(spot)
            greeks = self.calculate(
                engine,
                instrument,
                model,
                bumped_market,
                include_higher_order=(greek in higher_order_greeks),
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
    include_higher_order: bool = False,
) -> GreeksResult:
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
    GreeksResult
        Greeks result (with price and all orders when include_higher_order=True)
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

    from backend.core.market import MarketEnvironment
    from backend.engines import BSAnalyticEngine
    from backend.instruments.options import VanillaOption
    from backend.models.gbm import GBMModel

    # Create components
    option = VanillaOption(strike=100.0, maturity=0.25, is_call=True)
    model = GBMModel(sigma=0.20)
    engine = BSAnalyticEngine()
    market = MarketEnvironment(spot=100.0, rate=0.05, dividend_yield=0.02)

    # Calculate Greeks
    calc = GreeksCalculator()
    greeks = calc.calculate(engine, option, model, market, include_higher_order=True)

    print("\nATM Call Greeks (S=K=100, T=0.25, sigma=20%):")
    print(f"  Price: ${greeks.price:.4f}")
    print("\n  First Order:")
    print(f"    Delta: {greeks.delta:.6f}")
    print(f"    Gamma: {greeks.gamma:.6f}")
    print(f"    Vega:  {greeks.vega:.6f}")
    print(f"    Theta: {greeks.theta:.6f}")
    print(f"    Rho:   {greeks.rho:.6f}")
    print("\n  Second Order:")
    print(f"    Vanna: {greeks.vanna:.6f}")
    print(f"    Volga: {greeks.volga:.6f}")
    print(f"    Charm: {greeks.charm:.6f}")
    print(f"    Veta:  {greeks.veta:.6f}")
    print("\n  Third Order:")
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
