"""
Greeks Computation Strategies
==============================

Strategy Pattern for pluggable Greeks computation methods.

Each strategy encapsulates a specific numerical technique:
- AnalyticGreeksStrategy: Closed-form BS Greeks (GBM + VanillaOption only)
- NumericalGreeksStrategy: Bump-and-reprice finite difference
- NumericalGreeksStrategy: Bump-and-reprice (universal fallback for any Priceable)

Usage::

    calc = GreeksCalculator()
    # Auto-select best method
    result = calc.calculate(instrument, model, market)
    # Force specific method
    result = calc.calculate(instrument, model, market, method="aad")

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import logging
from typing import Protocol

from backend.core.interfaces import Model, Priceable, PricingEngine
from backend.core.market import MarketEnvironment
from backend.core.result_types import GreeksResult, PricingCapability
from backend.greeks.numerical import GreeksBumpConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Strategy Protocol
# =============================================================================


class GreeksStrategy(Protocol):
    """Protocol for Greeks computation strategies."""

    @property
    def name(self) -> str:
        """Strategy identifier (e.g., 'analytic', 'aad', 'numerical')."""
        ...

    def can_compute(self, instrument: Priceable, model: Model) -> bool:
        """Check if this strategy can compute Greeks for this pair."""
        ...

    def compute(
        self,
        engine: PricingEngine,
        instrument: Priceable,
        model: Model,
        market: MarketEnvironment,
        config: GreeksBumpConfig,
        include_higher_order: bool = True,
    ) -> GreeksResult:
        """Compute Greeks using this strategy."""
        ...


# =============================================================================
# Analytic Strategy (BS closed-form)
# =============================================================================


class AnalyticGreeksStrategy:
    """Closed-form Black-Scholes Greeks.

    Applicable only to VanillaOption + GBM model (constant volatility).
    Highest precision, no numerical error.
    """

    @property
    def name(self) -> str:
        return "analytic"

    def can_compute(self, instrument: Priceable, model: Model) -> bool:
        from backend.instruments.options import VanillaOption

        return (
            isinstance(instrument, VanillaOption) and "sigma" in model.get_parameters()
        )

    def compute(
        self,
        engine: PricingEngine,
        instrument: Priceable,
        model: Model,
        market: MarketEnvironment,
        config: GreeksBumpConfig,
        include_higher_order: bool = True,
    ) -> GreeksResult:
        from backend.greeks.analytic import bs_all_greeks

        params = model.get_parameters()
        sigma = params["sigma"]

        greeks = bs_all_greeks(
            s=market.spot,
            k=instrument.strike,  # type: ignore[attr-defined]
            t=instrument.maturity,
            r=market.rate,
            q=market.dividend_yield,
            sigma=sigma,
            is_call=instrument.is_call,  # type: ignore[attr-defined]
        )

        if include_higher_order:
            return GreeksResult(
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
                ultima=greeks[13],
            )
        return GreeksResult(
            price=greeks[0],
            delta=greeks[1],
            gamma=greeks[2],
            vega=greeks[3],
            theta=greeks[4],
            rho=greeks[5],
        )


# =============================================================================
# Engine-provided Strategy (for engines with their own greeks() method)
# =============================================================================


class EngineGreeksStrategy:
    """Use the engine's own greeks() method.

    Applicable to engines that provide their own Greeks implementation,
    Excludes MC engines (prefer numerical for those).
    """

    @property
    def name(self) -> str:
        return "engine"

    def can_compute(self, instrument: Priceable, model: Model) -> bool:
        # This strategy needs the engine to decide — always return False
        # from can_compute and use can_compute_with_engine instead.
        return False

    def can_compute_with_engine(
        self, instrument: Priceable, model: Model, engine: PricingEngine
    ) -> bool:
        return (
            hasattr(engine, "greeks")
            and hasattr(engine, "can_price")
            and engine.can_price(instrument, model)
            and engine.capability != PricingCapability.MONTE_CARLO
        )

    def compute(
        self,
        engine: PricingEngine,
        instrument: Priceable,
        model: Model,
        market: MarketEnvironment,
        config: GreeksBumpConfig,
        include_higher_order: bool = True,
    ) -> GreeksResult:
        engine_result = engine.greeks(instrument, model, market)  # type: ignore[attr-defined]
        price = (
            engine_result.price
            if engine_result.price != 0.0
            else engine.price(instrument, model, market).price
        )

        if not include_higher_order:
            return GreeksResult(
                price=price,
                delta=engine_result.delta,
                gamma=engine_result.gamma,
                vega=engine_result.vega,
                theta=engine_result.theta,
                rho=engine_result.rho,
            )

        # If the engine already provides full Greeks, use them directly
        if engine_result.has_higher_order:
            return GreeksResult(
                price=price,
                delta=engine_result.delta,
                gamma=engine_result.gamma,
                vega=engine_result.vega,
                theta=engine_result.theta,
                rho=engine_result.rho,
                vanna=engine_result.vanna,
                volga=engine_result.volga,
                charm=engine_result.charm,
                veta=engine_result.veta,
                speed=engine_result.speed,
                zomma=engine_result.zomma,
                color=engine_result.color,
                ultima=engine_result.ultima,
            )

        # Engine only provides first-order — supplement with numerical higher-order
        from backend.greeks.calculator import GreeksCalculator

        calc = GreeksCalculator(
            spot_bump=config.spot_bump,
            vol_bump=config.vol_bump,
            time_bump_days=config.time_bump_days,
            rate_bump=config.rate_bump,
        )
        higher = calc._numerical_higher_order_greeks(engine, instrument, model, market)
        return GreeksResult(
            price=price,
            delta=engine_result.delta,
            gamma=engine_result.gamma,
            vega=engine_result.vega,
            theta=engine_result.theta,
            rho=engine_result.rho,
            vanna=higher["vanna"],
            volga=higher["volga"],
            charm=higher["charm"],
            veta=higher["veta"],
            speed=higher["speed"],
            zomma=higher["zomma"],
            color=higher["color"],
            ultima=higher["ultima"],
        )


# =============================================================================
# Numerical Strategy (Bump-and-reprice, universal fallback)
# =============================================================================


class NumericalGreeksStrategy:
    """Finite difference bump-and-reprice Greeks.

    Universal fallback that works with ANY Priceable (vanilla,
    structured products) and ANY model/engine combination.
    """

    @property
    def name(self) -> str:
        return "numerical"

    def can_compute(self, instrument: Priceable, model: Model) -> bool:
        # Always available as long as we can price the instrument
        return True

    def compute(
        self,
        engine: PricingEngine,
        instrument: Priceable,
        model: Model,
        market: MarketEnvironment,
        config: GreeksBumpConfig,
        include_higher_order: bool = True,
    ) -> GreeksResult:
        from backend.core.structured_product import StructuredProduct
        from backend.greeks.numerical import ModelNumericalGreeks

        # For structured products, delegate to their engine's greeks()
        if isinstance(instrument, StructuredProduct):
            return self._structured_greeks(engine, instrument, model, market)

        num_calc = ModelNumericalGreeks(config)
        num_greeks = num_calc.calculate(engine, instrument, model, market)
        price = engine.price(instrument, model, market).price

        result = GreeksResult(
            price=price,
            delta=num_greeks.delta,
            gamma=num_greeks.gamma,
            vega=num_greeks.vega,
            theta=num_greeks.theta,
            rho=num_greeks.rho,
        )

        if include_higher_order:
            from backend.greeks.calculator import GreeksCalculator

            calc = GreeksCalculator(
                spot_bump=config.spot_bump,
                vol_bump=config.vol_bump,
                time_bump_days=config.time_bump_days,
                rate_bump=config.rate_bump,
            )
            higher = calc._numerical_higher_order_greeks(
                engine, instrument, model, market
            )
            result = GreeksResult(
                price=price,
                delta=num_greeks.delta,
                gamma=num_greeks.gamma,
                vega=num_greeks.vega,
                theta=num_greeks.theta,
                rho=num_greeks.rho,
                vanna=higher["vanna"],
                volga=higher["volga"],
                charm=higher["charm"],
                veta=higher["veta"],
                speed=higher["speed"],
                zomma=higher["zomma"],
                color=higher["color"],
                ultima=higher["ultima"],
            )

        return result

    def _structured_greeks(
        self,
        engine: PricingEngine,
        instrument: Priceable,
        model: Model,
        market: MarketEnvironment,
    ) -> GreeksResult:
        """Greeks for structured products via their dedicated engine."""
        from backend.engines.structured_mc_engine import StructuredProductMCEngine

        if isinstance(engine, StructuredProductMCEngine):
            return engine.greeks(instrument, model, market)
        sp_engine = StructuredProductMCEngine()
        return sp_engine.greeks(instrument, model, market)


# =============================================================================
# Default strategy priority
# =============================================================================

DEFAULT_STRATEGIES: list[GreeksStrategy] = [
    AnalyticGreeksStrategy(),
    EngineGreeksStrategy(),
    NumericalGreeksStrategy(),
]
