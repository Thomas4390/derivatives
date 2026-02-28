"""
Engine Registry
===============

Factory that automatically selects the optimal pricing engine.

Priority: ANALYTICAL > FFT > PDE > MONTE_CARLO

Author: Thomas
Created: 2025
"""

from collections.abc import Callable

from backend.core.interfaces import Instrument, Model, PricingEngine
from backend.core.market import MarketEnvironment
from backend.core.result_types import PricingCapability, PricingResult

# Type alias for engine providers
# Supports: Class, pre-configured instance, or factory function
EngineProvider = type[PricingEngine] | Callable[[], PricingEngine] | PricingEngine


class EngineRegistry:
    """
    Factory that automatically selects the optimal pricing engine.

    Priority: ANALYTICAL > FFT > MONTE_CARLO

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

    @classmethod
    def get_engine(
        cls,
        instrument: Instrument,
        model: Model,
        preferred: PricingCapability | None = None,
    ) -> PricingEngine:
        """
        Get the optimal engine for instrument/model pair.

        Parameters
        ----------
        instrument : Instrument
            The contract to price
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
            raise ValueError(
                f"No {preferred.name} engine registered for model {model_name}"
            )

        # Auto-select by priority
        for capability in cls.PRIORITY:
            if capability not in model.supported_engines:
                continue

            key = (model_name, capability)
            if key not in cls._engines:
                continue

            engine = cls._instantiate(cls._engines[key])
            if exercise in engine.supported_exercises:
                return engine

        registered = cls.list_engines()
        raise ValueError(
            f"No compatible engine found for {model_name} with {exercise.name} exercise. "
            f"Registered engines: {registered}"
        )

    @classmethod
    def price(
        cls,
        instrument: Instrument,
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
    instrument: Instrument,
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


if __name__ == "__main__":
    # Smoke test
    print("Registry module loaded successfully")
    print(f"Engine priority: {[c.name for c in EngineRegistry.PRIORITY]}")
    print(f"Registered engines: {EngineRegistry.list_engines()}")
    print("✓ Registry smoke test passed")
