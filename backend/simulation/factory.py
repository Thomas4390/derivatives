"""
Model Factory
=============

Factory functions for creating simulator instances.

Provides a unified interface for creating any simulation model
from configuration dictionaries or enum types.

Author: Derivatives Pricing Project
"""

from typing import Dict, Any, Optional, Type, Union

from .base import BaseSimulator
from .enums import ModelType, DiscretizationScheme
from .models import (
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

    Examples
    --------
    >>> sim = create_simulator("gbm", sigma=0.20)
    >>> sim = create_simulator(ModelType.HESTON, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    >>> sim = create_simulator("garch", sigma0=0.20, omega=0.000002, alpha=0.05, beta=0.90)
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
