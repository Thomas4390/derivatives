"""
Engine Registry
===============

Factory that automatically selects the optimal pricing engine.

Priority: ANALYTICAL > FFT > PDE > MONTE_CARLO

Author: Thomas
Created: 2025
"""

from typing import Dict, List, Optional, Type, Tuple

from backend.core.interfaces import Instrument, Model, PricingEngine
from backend.core.result_types import PricingResult, PricingCapability, ExerciseStyle
from backend.core.market import MarketEnvironment


class EngineRegistry:
    """
    Factory that automatically selects the optimal pricing engine.

    Priority: ANALYTICAL > FFT > MONTE_CARLO

    The registry maintains a mapping of (Model, Capability) -> Engine
    and selects the best available option.

    Important
    ---------
    Registered engines are instantiated with no arguments, so they must
    have sensible default parameters. For engines with specific configurations,
    instantiate them directly rather than using the registry:

        # Direct instantiation with custom parameters
        engine = MonteCarloEngine(n_paths=100000, seed=42)
        result = engine.price(option, model, market)

    Examples
    --------
    # Register an engine
    EngineRegistry.register("GBM", PricingCapability.ANALYTICAL, BSAnalyticEngine)

    # Auto-select engine and price
    result = EngineRegistry.price(option, model, market)
    """

    # Priority order (highest to lowest)
    PRIORITY = [
        PricingCapability.ANALYTICAL,
        PricingCapability.FFT,
        PricingCapability.MONTE_CARLO,
    ]

    _engines: Dict[Tuple[str, PricingCapability], Type[PricingEngine]] = {}

    @classmethod
    def register(
        cls,
        model_name: str,
        capability: PricingCapability,
        engine_class: Type[PricingEngine],
    ) -> None:
        """
        Register an engine for a model/capability pair.

        Parameters
        ----------
        model_name : str
            Name of the model (e.g., "GBM", "Heston")
        capability : PricingCapability
            Type of engine
        engine_class : Type[PricingEngine]
            Engine class to instantiate
        """
        cls._engines[(model_name, capability)] = engine_class

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
        preferred: Optional[PricingCapability] = None,
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
                engine = cls._engines[key]()
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

            engine = cls._engines[key]()
            if exercise in engine.supported_exercises:
                return engine

        raise ValueError(
            f"No compatible engine found for {model_name} with {exercise.name} exercise"
        )

    @classmethod
    def price(
        cls,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
        preferred: Optional[PricingCapability] = None,
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
    def list_engines(cls) -> List[Tuple[str, str]]:
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
    method: Optional[PricingCapability] = None,
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
