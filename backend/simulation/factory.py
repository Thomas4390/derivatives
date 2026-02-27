"""
Model Factory
=============

Factory functions for creating simulator instances.

Provides a unified interface for creating any simulation model
from configuration dictionaries or enum types.

Author: Thomas
Created: 2025
"""

from typing import Dict, Any, Type, Union

from backend.simulation.base import BaseSimulator
from backend.simulation.enums import ModelType, DiscretizationScheme
from backend.simulation.models import (
    GBMSimulator,
    HestonSimulator,
    MertonSimulator,
    BatesSimulator,
    GARCHSimulator,
    NGARCHSimulator,
    GJRGARCHSimulator,
)


# =============================================================================
# Model Registry
# =============================================================================

_MODEL_REGISTRY: Dict[ModelType, Type[BaseSimulator]] = {
    ModelType.GBM: GBMSimulator,
    ModelType.HESTON: HestonSimulator,
    ModelType.MERTON: MertonSimulator,
    ModelType.BATES: BatesSimulator,
    ModelType.GARCH: GARCHSimulator,
    ModelType.NGARCH: NGARCHSimulator,
    ModelType.GJR_GARCH: GJRGARCHSimulator,
}

_MODEL_NAME_MAP: Dict[str, ModelType] = {
    "gbm": ModelType.GBM,
    "geometric_brownian_motion": ModelType.GBM,
    "heston": ModelType.HESTON,
    "merton": ModelType.MERTON,
    "merton_jump": ModelType.MERTON,
    "jump_diffusion": ModelType.MERTON,
    "bates": ModelType.BATES,
    "heston_jump": ModelType.BATES,
    "garch": ModelType.GARCH,
    "garch11": ModelType.GARCH,
    "ngarch": ModelType.NGARCH,
    "nagarch": ModelType.NGARCH,
    "gjr": ModelType.GJR_GARCH,
    "gjr_garch": ModelType.GJR_GARCH,
    "tgarch": ModelType.GJR_GARCH,
}


# =============================================================================
# Factory Functions
# =============================================================================

def create_simulator(
    model: Union[str, ModelType],
    **kwargs
) -> BaseSimulator:
    """
    Create a simulator instance from model type and parameters.

    Parameters
    ----------
    model : str or ModelType
        Model identifier (e.g., "gbm", "heston", ModelType.GARCH)
    **kwargs
        Model-specific parameters

    Returns
    -------
    BaseSimulator
        Configured simulator instance
    """
    # Resolve model type
    if isinstance(model, str):
        model_key = model.lower().replace("-", "_").replace(" ", "_")
        if model_key not in _MODEL_NAME_MAP:
            available = list(_MODEL_NAME_MAP.keys())
            raise ValueError(f"Unknown model: '{model}'. Available: {available}")
        model_type = _MODEL_NAME_MAP[model_key]
    elif isinstance(model, ModelType):
        model_type = model
    else:
        raise TypeError(f"model must be str or ModelType, got {type(model)}")

    # Get simulator class
    simulator_class = _MODEL_REGISTRY.get(model_type)
    if simulator_class is None:
        raise ValueError(f"No simulator registered for {model_type}")

    # Create instance
    return simulator_class(**kwargs)


def create_gbm(sigma: float, antithetic: bool = True) -> GBMSimulator:
    """Create a GBM simulator."""
    return GBMSimulator(sigma=sigma, antithetic=antithetic)


def create_heston(
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    scheme: DiscretizationScheme = DiscretizationScheme.FULL_TRUNCATION
) -> HestonSimulator:
    """Create a Heston simulator."""
    return HestonSimulator(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, scheme=scheme)


def create_merton(
    sigma: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float
) -> MertonSimulator:
    """Create a Merton jump diffusion simulator."""
    return MertonSimulator(sigma=sigma, lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j)


def create_bates(
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float
) -> BatesSimulator:
    """Create a Bates simulator."""
    return BatesSimulator(
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j
    )


def create_garch(
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float
) -> GARCHSimulator:
    """Create a GARCH(1,1) simulator."""
    return GARCHSimulator(sigma0=sigma0, omega=omega, alpha=alpha, beta=beta)


def create_ngarch(
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    theta: float
) -> NGARCHSimulator:
    """Create an NGARCH simulator."""
    return NGARCHSimulator(sigma0=sigma0, omega=omega, alpha=alpha, beta=beta, theta=theta)


def create_gjr_garch(
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float
) -> GJRGARCHSimulator:
    """Create a GJR-GARCH simulator."""
    return GJRGARCHSimulator(sigma0=sigma0, omega=omega, alpha=alpha, beta=beta, gamma=gamma)


# =============================================================================
# Model Information
# =============================================================================

def list_models() -> Dict[str, str]:
    """
    List all available models with descriptions.

    Returns
    -------
    dict
        Model names and descriptions
    """
    return {model_type.name: model_type.value for model_type in ModelType}


def get_model_info(model: Union[str, ModelType]) -> Dict[str, Any]:
    """
    Get information about a specific model.

    Parameters
    ----------
    model : str or ModelType
        Model identifier

    Returns
    -------
    dict
        Model information including parameters and description
    """
    # Resolve model type
    if isinstance(model, str):
        model_key = model.lower().replace("-", "_")
        model_type = _MODEL_NAME_MAP.get(model_key)
        if model_type is None:
            raise ValueError(f"Unknown model: '{model}'")
    else:
        model_type = model

    simulator_class = _MODEL_REGISTRY[model_type]

    # Get constructor parameters
    import inspect
    sig = inspect.signature(simulator_class.__init__)
    params = {
        name: {
            "annotation": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
            "default": param.default if param.default != inspect.Parameter.empty else "required"
        }
        for name, param in sig.parameters.items()
        if name != "self"
    }

    return {
        "name": model_type.name,
        "description": model_type.value,
        "class": simulator_class.__name__,
        "parameters": params,
        "is_stochastic_vol": model_type in ModelType.stochastic_vol_models(),
        "has_jumps": model_type in ModelType.jump_models(),
        "is_continuous_time": model_type in ModelType.continuous_time_models(),
    }


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Model Factory Smoke Test")
    print("=" * 50)

    # Test list_models
    print("\n--- Available Models ---")
    models = list_models()
    for name, desc in models.items():
        print(f"  {name}: {desc}")

    # Test create_simulator with string names
    print("\n--- Create Simulator (String Names) ---")
    string_names = ["gbm", "heston", "merton", "bates", "garch", "ngarch", "gjr_garch"]
    configs = {
        "gbm": {"sigma": 0.2},
        "heston": {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7},
        "merton": {"sigma": 0.2, "lambda_j": 0.5, "mu_j": -0.1, "sigma_j": 0.2},
        "bates": {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7,
                  "lambda_j": 0.5, "mu_j": -0.1, "sigma_j": 0.2},
        "garch": {"sigma0": 0.2, "omega": 1e-6, "alpha": 0.1, "beta": 0.85},
        "ngarch": {"sigma0": 0.2, "omega": 1e-6, "alpha": 0.1, "beta": 0.85, "theta": 0.5},
        "gjr_garch": {"sigma0": 0.2, "omega": 1e-6, "alpha": 0.05, "beta": 0.85, "gamma": 0.1},
    }

    for name in string_names:
        sim = create_simulator(name, **configs[name])
        print(f"  {name} -> {sim.model_name}")

    # Test create_simulator with ModelType enum
    print("\n--- Create Simulator (ModelType Enum) ---")
    sim = create_simulator(ModelType.GBM, sigma=0.25)
    print(f"  ModelType.GBM -> {sim.model_name}")
    sim = create_simulator(ModelType.HESTON, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    print(f"  ModelType.HESTON -> {sim.model_name}")

    # Test convenience factory functions
    print("\n--- Convenience Factory Functions ---")
    gbm = create_gbm(sigma=0.2)
    print(f"  create_gbm: {gbm.model_name}")

    heston = create_heston(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    print(f"  create_heston: {heston.model_name}")

    merton = create_merton(sigma=0.2, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)
    print(f"  create_merton: {merton.model_name}")

    bates = create_bates(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                         lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)
    print(f"  create_bates: {bates.model_name}")

    garch = create_garch(sigma0=0.2, omega=1e-6, alpha=0.1, beta=0.85)
    print(f"  create_garch: {garch.model_name}")

    ngarch = create_ngarch(sigma0=0.2, omega=1e-6, alpha=0.1, beta=0.85, theta=0.5)
    print(f"  create_ngarch: {ngarch.model_name}")

    gjr = create_gjr_garch(sigma0=0.2, omega=1e-6, alpha=0.05, beta=0.85, gamma=0.1)
    print(f"  create_gjr_garch: {gjr.model_name}")

    # Test name aliases
    print("\n--- Name Aliases ---")
    aliases = [
        ("geometric_brownian_motion", "GBM"),
        ("heston_jump", "Bates"),
        ("jump_diffusion", "Merton"),
        ("garch11", "GARCH"),
        ("tgarch", "GJR-GARCH"),
    ]
    for alias, expected in aliases:
        sim = create_simulator(alias, **configs[alias.replace("geometric_brownian_motion", "gbm")
                                                 .replace("heston_jump", "bates")
                                                 .replace("jump_diffusion", "merton")
                                                 .replace("garch11", "garch")
                                                 .replace("tgarch", "gjr_garch")])
        print(f"  '{alias}' -> {sim.model_name}")

    # Test get_model_info
    print("\n--- Model Information ---")
    info = get_model_info("heston")
    print(f"  Name: {info['name']}")
    print(f"  Description: {info['description']}")
    print(f"  Class: {info['class']}")
    print(f"  Is stochastic vol: {info['is_stochastic_vol']}")
    print(f"  Has jumps: {info['has_jumps']}")
    print(f"  Is continuous time: {info['is_continuous_time']}")
    print(f"  Parameters: {list(info['parameters'].keys())}")

    # Test error handling
    print("\n--- Error Handling ---")
    try:
        create_simulator("unknown_model")
        print("  ERROR: Should have raised ValueError")
    except ValueError:
        print("  Unknown model rejected: ✓")

    try:
        create_simulator(123)  # Invalid type
        print("  ERROR: Should have raised TypeError")
    except TypeError:
        print("  Invalid type rejected: ✓")

    # Test simulation
    print("\n--- Simulation Test ---")
    sim = create_simulator("gbm", sigma=0.2)
    result = sim.simulate_paths(s0=100.0, mu=0.05, t=1.0, n_paths=100, n_steps=252, seed=42)
    print(f"  GBM simulation: {result.n_paths} paths, {result.n_steps} steps")
    print(f"  Terminal mean: {result.terminal_mean:.2f}")

    print("\n" + "=" * 50)
    print("Model Factory smoke test passed")
    print("=" * 50)
