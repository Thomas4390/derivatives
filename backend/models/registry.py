"""
Model Registry
==============

Central registry for all financial models using the Singleton pattern.

Replaces string-based factory pattern with explicit registration,
providing type safety and clear model discovery.

Usage:
    from backend.models import registry

    # Create via registry
    simulator = registry.create_simulator("heston", v0=0.04, kappa=2.0, ...)
    pricer = registry.create_pricer("heston", v0=0.04, kappa=2.0, ...)

    # List available models
    print(registry.list_models())

Author: Thomas
Created: 2025
"""

from typing import Dict, Type, Optional, List, Any, TYPE_CHECKING

from backend.core.interfaces import Model as BaseModel
from backend.core.result_types import PricingCapability

if TYPE_CHECKING:
    from backend.simulation.base import BaseSimulator


class ModelRegistry:
    """
    Central registry for all financial models.

    Singleton pattern ensures a single global registry.
    Models register themselves on module import.

    Features:
        - Explicit registration (no string matching)
        - Alias support (e.g., "sv" -> "heston")
        - Type-safe model creation
        - Model discovery via list_models()
    """

    _instance: Optional["ModelRegistry"] = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models: Dict[str, Type[BaseModel]] = {}
            cls._instance._aliases: Dict[str, str] = {}
        return cls._instance

    def register(
        self,
        key: str,
        model_class: Type[BaseModel],
        aliases: Optional[List[str]] = None
    ) -> None:
        """
        Register a model class.

        Parameters
        ----------
        key : str
            Primary model identifier (lowercase)
        model_class : Type[BaseModel]
            The model class to register
        aliases : list[str], optional
            Alternative names for this model

        Example
        -------
        registry.register("heston", HestonModel, aliases=["sv", "stochvol"])
        """
        key = key.lower()
        self._models[key] = model_class

        if aliases:
            for alias in aliases:
                self._aliases[alias.lower()] = key

    def _resolve_key(self, key: str) -> str:
        """Resolve alias to primary key."""
        key = key.lower()
        return self._aliases.get(key, key)

    def get(self, key: str) -> Type[BaseModel]:
        """
        Get model class by key.

        Parameters
        ----------
        key : str
            Model identifier or alias

        Returns
        -------
        Type[BaseModel]
            The registered model class

        Raises
        ------
        KeyError
            If model is not registered
        """
        key = self._resolve_key(key)
        if key not in self._models:
            available = sorted(list(self._models.keys()) + list(self._aliases.keys()))
            raise KeyError(f"Unknown model: '{key}'. Available: {available}")
        return self._models[key]

    def create(self, key: str, **params) -> BaseModel:
        """
        Create model instance from parameters.

        Parameters
        ----------
        key : str
            Model identifier
        **params
            Model parameters

        Returns
        -------
        BaseModel
            Configured model instance
        """
        model_class = self.get(key)
        # Models are dataclasses, so use direct constructor
        return model_class(**params)

    def create_simulator(self, key: str, **params) -> "BaseSimulator":
        """
        Create simulator directly from model key and params.

        Parameters
        ----------
        key : str
            Model identifier
        **params
            Model parameters and simulator options

        Returns
        -------
        BaseSimulator
            Configured simulator
        """
        # Separate simulator options from model params
        simulator_opts = {}
        model_params = {}

        # Known simulator options (use 'antithetic' to match GBMModel.create_simulator)
        sim_keys = {"scheme", "antithetic", "control_variate"}

        for k, v in params.items():
            if k in sim_keys:
                simulator_opts[k] = v
            else:
                model_params[k] = v

        model = self.create(key, **model_params)
        return model.create_simulator(**simulator_opts)

    def create_pricer(
        self,
        key: str,
        method: Optional[PricingCapability] = None,
        **params
    ) -> "BasePricer":
        """
        Create pricer directly from model key and params.

        Parameters
        ----------
        key : str
            Model identifier
        method : PricingCapability, optional
            Preferred pricing method
        **params
            Model parameters and pricer options

        Returns
        -------
        BasePricer
            Configured pricer
        """
        # Separate pricer options from model params
        pricer_opts = {}
        model_params = {}

        # Known pricer options
        pricer_keys = {"r", "n_paths", "n_steps", "n_fft", "alpha"}

        for k, v in params.items():
            if k in pricer_keys:
                pricer_opts[k] = v
            else:
                model_params[k] = v

        model = self.create(key, **model_params)
        return model.create_pricer(method=method, **pricer_opts)

    def list_models(self) -> Dict[str, str]:
        """
        List all registered models.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping model key to model name
        """
        result = {}
        for key, model_class in self._models.items():
            # Instantiate temporarily to get model_name
            # This is a bit hacky but works for now
            try:
                result[key] = model_class.__name__
            except Exception:
                result[key] = key
        return result

    def list_aliases(self) -> Dict[str, str]:
        """
        List all registered aliases.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping alias to primary key
        """
        return dict(self._aliases)

    def is_registered(self, key: str) -> bool:
        """Check if a model is registered."""
        key = self._resolve_key(key)
        return key in self._models


# Global singleton instance
registry = ModelRegistry()


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Model Registry Smoke Test")
    print("=" * 50)

    # Import models for registration
    from backend.models.gbm import GBMModel
    from backend.models.heston import HestonModel

    # Test registration
    print("\n--- Registration Test ---")
    registry.register("gbm", GBMModel, aliases=["bs", "black-scholes"])
    registry.register("heston", HestonModel, aliases=["sv", "stochvol"])
    print(f"Registered models: {list(registry.list_models().keys())}")
    print(f"Registered aliases: {registry.list_aliases()}")

    # Test retrieval
    print("\n--- Retrieval Test ---")
    gbm_class = registry.get("gbm")
    print(f"Retrieved GBM class: {gbm_class.__name__}")

    gbm_class_alias = registry.get("bs")
    print(f"Retrieved via alias 'bs': {gbm_class_alias.__name__}")

    # Test is_registered
    print("\n--- Registration Check ---")
    print(f"'gbm' registered: {registry.is_registered('gbm')}")
    print(f"'bs' registered (alias): {registry.is_registered('bs')}")
    print(f"'unknown' registered: {registry.is_registered('unknown')}")

    # Test error handling
    print("\n--- Error Handling ---")
    try:
        registry.get("nonexistent")
        print("ERROR: Should have raised KeyError!")
    except KeyError as e:
        print(f"Correctly raised KeyError: {e}")

    print("\n" + "=" * 50)
    print("Model Registry smoke test passed")
    print("=" * 50)
