"""
Engine Registry
===============

Factory that automatically selects the optimal pricing engine.

Priority: ANALYTICAL > FFT > MONTE_CARLO > AAD

Author: Thomas Vaudescal
Created: 2026
"""

import logging
from collections.abc import Callable

from backend.core.interfaces import Model, Priceable, PricingEngine
from backend.core.market import MarketEnvironment
from backend.core.result_types import PricingCapability, PricingResult

logger = logging.getLogger(__name__)

# Type alias for engine providers
# Supports: Class, pre-configured instance, or factory function
EngineProvider = type[PricingEngine] | Callable[[], PricingEngine] | PricingEngine


class EngineRegistry:
    """
    Factory that automatically selects the optimal pricing engine.

    Priority: ANALYTICAL > FFT > MONTE_CARLO > AAD

    The registry maintains a mapping of (Model, Capability) -> Engine
    and selects the best available option.

    Engine Registration
    -------------------
    The registry supports three types of providers:

    1. **Class** (backward compatible): Instantiated with no arguments
    2. **Pre-configured instance**: Used directly as-is
    3. **Factory function**: Called to create a new instance each time

    Examples
    --------
    # Register a class (backward compatible)
    EngineRegistry.register("GBM", PricingCapability.ANALYTICAL, BSAnalyticEngine)

    # Register a pre-configured instance
    EngineRegistry.register("GBM", PricingCapability.MONTE_CARLO,
                            MonteCarloEngine(n_paths=500_000))

    # Register a factory function
    EngineRegistry.register("GBM", PricingCapability.MONTE_CARLO,
                            lambda: MonteCarloEngine(n_paths=200_000))

    # Auto-select engine and price
    result = EngineRegistry.price(option, model, market)
    """

    # Priority order (highest to lowest)
    PRIORITY = [
        PricingCapability.ANALYTICAL,
        PricingCapability.FFT,
        PricingCapability.MONTE_CARLO,
    ]

    _engines: dict[tuple[str, PricingCapability], EngineProvider] = {}

    @classmethod
    def register(
        cls,
        model_name: str,
        capability: PricingCapability,
        engine_provider: EngineProvider,
    ) -> None:
        """
        Register an engine for a model/capability pair.

        Parameters
        ----------
        model_name : str
            Name of the model (e.g., "GBM", "Heston")
        capability : PricingCapability
            Type of engine
        engine_provider : EngineProvider
            Engine class, pre-configured instance, or factory function
        """
        cls._engines[(model_name, capability)] = engine_provider

    @classmethod
    def _instantiate(cls, provider: EngineProvider) -> PricingEngine:
        """
        Instantiate an engine from a provider.

        Parameters
        ----------
        provider : EngineProvider
            Engine class, pre-configured instance, or factory function

        Returns
        -------
        PricingEngine
            Engine instance ready for pricing

        Raises
        ------
        TypeError
            If provider is not a valid engine provider
        """
        if isinstance(provider, PricingEngine):
            # Pre-configured instance
            return provider
        if callable(provider):
            # Class or factory function
            return provider()
        raise TypeError(f"Invalid engine provider: {provider}")

    @classmethod
    def unregister(cls, model_name: str, capability: PricingCapability) -> bool:
        """
        Remove an engine registration.

        Parameters
        ----------
        model_name : str
            Name of the model
        capability : PricingCapability
            Type of engine

        Returns
        -------
        bool
            True if engine was removed, False if not found
        """
        key = (model_name, capability)
        if key in cls._engines:
            del cls._engines[key]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Remove all registered engines."""
        cls._engines.clear()
        cls._fallback_engines.clear()

    # Fallback engines that use can_price() for dispatch (e.g. StructuredProductMCEngine)
    _fallback_engines: list[EngineProvider] = []

    @classmethod
    def register_fallback(cls, engine_provider: EngineProvider) -> None:
        """
        Register a fallback engine that uses can_price() for dispatch.

        Fallback engines are tried after the primary (model_name, capability)
        lookup fails. They must implement can_price() to determine compatibility.

        Parameters
        ----------
        engine_provider : EngineProvider
            Engine class, instance, or factory
        """
        cls._fallback_engines.append(engine_provider)

    @classmethod
    def get_engine(
        cls,
        instrument: Priceable,
        model: Model,
        preferred: PricingCapability | None = None,
    ) -> PricingEngine:
        """
        Get the optimal engine for instrument/model pair.

        Parameters
        ----------
        instrument : Priceable
            The contract to price (Instrument or StructuredProduct)
        model : Model
            The stochastic model
        preferred : PricingCapability, optional
            Force a specific engine type if available

        Returns
        -------
        PricingEngine
            Configured engine instance

        Raises
        ------
        ValueError
            If no compatible engine found
        """
        # Try primary registry lookup for instruments with exercise_style
        if hasattr(instrument, "exercise_style"):
            model_name = model.name
            exercise = instrument.exercise_style

            # If user prefers a specific engine
            if preferred is not None:
                key = (model_name, preferred)
                if key in cls._engines:
                    engine = cls._instantiate(cls._engines[key])
                    if exercise in engine.supported_exercises:
                        return engine
                    raise ValueError(
                        f"Engine {preferred.name} does not support {exercise.name} exercise"
                    )
                # Don't raise yet — try fallbacks below

            else:
                # Auto-select by priority
                for capability in cls.PRIORITY:
                    if capability not in model.supported_engines:
                        continue

                    key = (model_name, capability)
                    if key not in cls._engines:
                        continue

                    engine = cls._instantiate(cls._engines[key])
                    if exercise in engine.supported_exercises and engine.can_price(
                        instrument, model
                    ):
                        logger.debug(
                            "Selected engine %s for %s/%s",
                            engine.__class__.__name__,
                            model_name,
                            type(instrument).__name__,
                        )
                        return engine

        # Fallback: try engines that use can_price() for dispatch
        for provider in cls._fallback_engines:
            engine = cls._instantiate(provider)
            if engine.can_price(instrument, model):
                logger.debug(
                    "Fallback engine %s matched for %s",
                    engine.__class__.__name__,
                    type(instrument).__name__,
                )
                return engine

        registered = cls.list_engines()
        raise ValueError(
            f"No compatible engine found for {model.name}. "
            f"Registered engines: {registered}"
        )

    @classmethod
    def price(
        cls,
        instrument: Priceable,
        model: Model,
        market: MarketEnvironment,
        preferred: PricingCapability | None = None,
    ) -> PricingResult:
        """
        Convenience method: get engine and price in one call.

        Parameters
        ----------
        instrument : Instrument
            The contract to price
        model : Model
            The stochastic model
        market : MarketEnvironment
            Market conditions
        preferred : PricingCapability, optional
            Force a specific engine type

        Returns
        -------
        PricingResult
            Pricing result with price and metadata
        """
        engine = cls.get_engine(instrument, model, preferred)
        return engine.price(instrument, model, market)

    @classmethod
    def list_engines(cls) -> list[tuple[str, str]]:
        """
        List all registered engines.

        Returns
        -------
        List[Tuple[str, str]]
            List of (model_name, capability_name) pairs
        """
        return [(model, cap.name) for model, cap in cls._engines.keys()]


# =============================================================================
# Module-level convenience function
# =============================================================================


def price(
    instrument: Priceable,
    model: Model,
    market: MarketEnvironment,
    method: PricingCapability | None = None,
) -> PricingResult:
    """
    Price an instrument under a model.

    This is the main entry point for option pricing.

    Parameters
    ----------
    instrument : Instrument
        The contract to price
    model : Model
        The stochastic model
    market : MarketEnvironment
        Market conditions
    method : PricingCapability, optional
        Force a specific pricing method

    Returns
    -------
    PricingResult
        Pricing result

    Examples
    --------
    from backend.core import price, MarketEnvironment
    from backend.instruments import EuropeanCall
    from backend.models import GBMModel

    option = EuropeanCall(strike=100, maturity=0.5)
    model = GBMModel(sigma=0.20)
    market = MarketEnvironment(spot=100, rate=0.05)
    result = price(option, model, market)
    print(f"Price: {result.price:.4f}")
    """
    return EngineRegistry.price(instrument, model, market, method)
