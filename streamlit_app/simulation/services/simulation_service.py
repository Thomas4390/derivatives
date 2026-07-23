"""
Unified Simulation Service - Single interface for all simulation models.

Provides:
- Unified simulation execution across all 7 models
- Consistent result handling
- Automatic parameter mapping
- Caching support
"""

from typing import Any

import numpy as np

from backend.simulation.base import SimulationResult

# Backend imports
from backend.simulation.factory import create_simulator
from config.model_registry import get_all_models
from services import model_adapter


def get_spot(sim_params: dict[str, Any]) -> float:
    """Extract spot price from sim_params with standard fallback chain."""
    return float(sim_params.get("spot_price", sim_params.get("spot", 100.0)))


def _extract_model_params(model: str, params: dict[str, Any]) -> dict[str, Any]:
    """Extract model-specific parameters from the unified params dict.

    Delegates the 7 registered models to the registry-backed
    :mod:`services.model_adapter` (single source of parameter names/defaults).
    Only the dynamic custom-model path and the GBM antithetic guard (disable
    antithetic sampling below 2 paths) remain here.
    """
    from services.custom_model_service import get_custom_model_class, is_custom_model

    if is_custom_model(model):
        cls = get_custom_model_class()
        if cls is not None:
            return {
                s["name"]: params.get(s["name"], s["default"])
                for s in cls.PARAMETER_SPECS
            }

    result = model_adapter.extract_params(model, params)
    if model.lower() == "gbm" and params.get("n_paths", 10000) < 2:
        result["antithetic"] = False
    return result


def run_simulation(
    model: str, params: dict[str, Any], return_simulator: bool = False
) -> SimulationResult | tuple:
    """
    Run unified simulation for any model.

    Args:
        model: Model key ('gbm', 'heston', 'merton', 'bates', 'garch', 'ngarch', 'gjr_garch')
        params: Dictionary with all parameters (market + model + simulation)
        return_simulator: If True, also return the simulator instance

    Returns:
        SimulationResult containing price_paths, time_grid, volatility_paths (if applicable)
        If return_simulator=True: (SimulationResult, BaseSimulator)
    """
    # Extract parameters
    spot = params.get("spot", 100.0)
    drift = params.get("drift", 0.08)
    dividend_yield = params.get("dividend_yield", 0.0)
    # Dividends lower the price drift: the asset appreciates at (μ − q).
    effective_drift = drift - dividend_yield
    time_horizon = params.get("time_horizon", 1.0)
    n_paths = int(params.get("n_paths", 10000))
    n_steps = int(params.get("n_steps", 252))
    seed = params.get("seed")

    # Handle seed=0 as random
    if seed == 0:
        seed = None

    # Extract model-specific parameters
    model_params = _extract_model_params(model, params)

    # Create simulator
    from services.custom_model_service import get_custom_model_class, is_custom_model

    if is_custom_model(model):
        cls = get_custom_model_class()
        instance = cls(**model_params)
        from backend.simulation.models.generic_euler import GenericEulerSimulator

        simulator = GenericEulerSimulator(instance)
    else:
        simulator = create_simulator(model, **model_params)

    # Run simulation
    result = simulator.simulate_paths(
        s0=spot,
        mu=effective_drift,
        t=time_horizon,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )

    if return_simulator:
        return result, simulator
    return result


def run_terminal_simulation(model: str, params: dict[str, Any]) -> np.ndarray:
    """
    Run simulation and return only terminal prices (memory efficient).

    Args:
        model: Model key
        params: Dictionary with all parameters

    Returns:
        Array of terminal prices with shape (n_paths,)
    """
    # Extract parameters
    spot = params.get("spot", 100.0)
    drift = params.get("drift", 0.08)
    dividend_yield = params.get("dividend_yield", 0.0)
    # Dividends lower the price drift: the asset appreciates at (μ − q).
    effective_drift = drift - dividend_yield
    time_horizon = params.get("time_horizon", 1.0)
    n_paths = int(params.get("n_paths", 10000))
    n_steps = int(params.get("n_steps", 252))
    seed = params.get("seed")

    if seed == 0:
        seed = None

    model_params = _extract_model_params(model, params)

    from services.custom_model_service import get_custom_model_class, is_custom_model

    if is_custom_model(model):
        cls = get_custom_model_class()
        instance = cls(**model_params)
        from backend.simulation.models.generic_euler import GenericEulerSimulator

        simulator = GenericEulerSimulator(instance)
    else:
        simulator = create_simulator(model, **model_params)

    return simulator.simulate_terminal(
        s0=spot,
        mu=effective_drift,
        t=time_horizon,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )


def get_model_characteristics(model: str) -> dict[str, Any]:
    """
    Get characteristics of a model.

    Returns:
        Dictionary with:
        - has_stochastic_vol: Whether model has time-varying volatility
        - has_jumps: Whether model includes jump components
        - volatility_type: 'constant', 'stochastic', or 'time_varying'
    """
    model_lower = model.lower()

    characteristics = {
        "gbm": {
            "has_stochastic_vol": False,
            "has_jumps": False,
            "volatility_type": "constant",
        },
        "heston": {
            "has_stochastic_vol": True,
            "has_jumps": False,
            "volatility_type": "stochastic",
        },
        "merton": {
            "has_stochastic_vol": False,
            "has_jumps": True,
            "volatility_type": "constant",
        },
        "bates": {
            "has_stochastic_vol": True,
            "has_jumps": True,
            "volatility_type": "stochastic",
        },
        "garch": {
            "has_stochastic_vol": True,
            "has_jumps": False,
            "volatility_type": "time_varying",
        },
        "ngarch": {
            "has_stochastic_vol": True,
            "has_jumps": False,
            "volatility_type": "time_varying",
        },
        "gjr_garch": {
            "has_stochastic_vol": True,
            "has_jumps": False,
            "volatility_type": "time_varying",
        },
    }

    if model_lower not in characteristics:
        # Detect features from custom model
        from services.custom_model_service import (
            get_custom_model_class,
            is_custom_model,
        )

        if is_custom_model(model):
            cls = get_custom_model_class()
            if cls is not None:
                instance = cls(**{s["name"]: s["default"] for s in cls.PARAMETER_SPECS})
                has_sv = hasattr(instance, "variance_drift") and callable(
                    getattr(instance, "variance_drift", None)
                )
                has_j = hasattr(instance, "jump") and callable(
                    getattr(instance, "jump", None)
                )
                return {
                    "has_stochastic_vol": has_sv,
                    "has_jumps": has_j,
                    "volatility_type": "stochastic" if has_sv else "custom",
                }
        return {
            "has_stochastic_vol": False,
            "has_jumps": False,
            "volatility_type": "custom",
        }

    return characteristics[model_lower]


def check_model_conditions(model: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    Check model-specific conditions (stationarity, Feller, etc.).

    Returns:
        Dictionary with:
        - is_valid: Overall validity
        - conditions: List of condition checks with status
    """
    model_lower = model.lower()
    conditions = []

    if model_lower in ["heston", "bates"]:
        # Feller condition: 2*kappa*theta > alpha^2
        kappa = params.get("kappa", 2.0)
        theta = params.get("theta", 0.04)
        alpha = params.get("alpha", 0.3)
        feller_lhs = 2 * kappa * theta
        feller_rhs = alpha**2
        feller_satisfied = feller_lhs > feller_rhs
        conditions.append(
            {
                "name": "Feller Condition",
                "equation": "2κθ > ξ²",
                "lhs": feller_lhs,
                "rhs": feller_rhs,
                "satisfied": feller_satisfied,
                "description": "Ensures variance stays positive"
                if feller_satisfied
                else "Variance may become negative (use Full Truncation scheme)",
            }
        )

    elif model_lower == "garch":
        # Stationarity: alpha + beta < 1
        alpha = params.get("alpha", 0.06)
        beta = params.get("beta", 0.90)
        persistence = alpha + beta
        stationary = persistence < 1
        conditions.append(
            {
                "name": "Stationarity",
                "equation": "α + β < 1",
                "value": persistence,
                "threshold": 1.0,
                "satisfied": stationary,
                "description": "Volatility mean-reverts"
                if stationary
                else "Integrated GARCH - volatility doesn't mean-revert",
            }
        )

    elif model_lower == "ngarch":
        # Stationarity: alpha*(1 + theta^2) + beta < 1
        alpha = params.get("alpha", 0.06)
        beta = params.get("beta", 0.90)
        gamma = params.get("gamma_ngarch", params.get("gamma", 0.5))
        persistence = alpha * (1 + gamma**2) + beta
        stationary = persistence < 1
        conditions.append(
            {
                "name": "Stationarity",
                "equation": "α(1 + θ²) + β < 1",
                "value": persistence,
                "threshold": 1.0,
                "satisfied": stationary,
                "description": "Volatility mean-reverts"
                if stationary
                else "Non-stationary - volatility explodes",
            }
        )

    elif model_lower == "gjr_garch":
        # Stationarity: alpha + beta + gamma/2 < 1
        alpha = params.get("alpha", 0.06)
        beta = params.get("beta", 0.90)
        gamma = params.get("gamma", 0.03)
        persistence = alpha + beta + gamma / 2
        stationary = persistence < 1
        conditions.append(
            {
                "name": "Stationarity",
                "equation": "α + β + γ/2 < 1",
                "value": persistence,
                "threshold": 1.0,
                "satisfied": stationary,
                "description": "Volatility mean-reverts"
                if stationary
                else "Non-stationary - volatility explodes",
            }
        )

    is_valid = all(c["satisfied"] for c in conditions) if conditions else True

    return {"is_valid": is_valid, "conditions": conditions}


def compute_long_run_volatility(model: str, params: dict[str, Any]) -> float | None:
    """
    Compute long-run (unconditional) volatility for models that support it.

    Returns:
        Long-run volatility as decimal (e.g., 0.20 for 20%), or None if not applicable
    """
    model_lower = model.lower()

    if model_lower in ["heston", "bates"]:
        theta = params.get("theta", 0.04)
        return np.sqrt(theta)

    if model_lower == "garch":
        omega = params.get("omega", 0.002)
        alpha = params.get("alpha", 0.06)
        beta = params.get("beta", 0.90)
        persistence = alpha + beta
        if persistence >= 1:
            return None
        # sigma^2_inf = omega / (1 - alpha - beta), already annualized
        # (the GARCH simulator uses sigma_t * sqrt(dt) * z for returns,
        #  so sigma^2_t is annualized variance; no n_steps scaling needed)
        return np.sqrt(omega / (1 - persistence))

    if model_lower == "ngarch":
        omega = params.get("omega", 0.002)
        alpha = params.get("alpha", 0.06)
        beta = params.get("beta", 0.90)
        gamma = params.get("gamma_ngarch", params.get("gamma", 0.5))
        persistence = alpha * (1 + gamma**2) + beta
        if persistence >= 1:
            return None
        return np.sqrt(omega / (1 - persistence))

    if model_lower == "gjr_garch":
        omega = params.get("omega", 0.002)
        alpha = params.get("alpha", 0.06)
        beta = params.get("beta", 0.90)
        gamma = params.get("gamma", 0.03)
        persistence = alpha + beta + gamma / 2
        if persistence >= 1:
            return None
        return np.sqrt(omega / (1 - persistence))

    if model_lower in ["gbm", "merton"]:
        return params.get("sigma", 0.20)

    return None


def get_initial_volatility(model: str, params: dict[str, Any]) -> float:
    """Representative initial volatility for a model.

    The 7 registered models delegate to the registry-backed
    :mod:`services.model_adapter`; the dynamic custom-model heuristic (first
    vol-like parameter, √ when the value looks like a variance) is preserved here.
    """
    from services.custom_model_service import get_custom_model_class, is_custom_model

    if is_custom_model(model):
        cls = get_custom_model_class()
        if cls is not None:
            for s in cls.PARAMETER_SPECS:
                name = s["name"].lower()
                if "sigma" in name or "vol" in name or "v0" in name:
                    val = params.get(s["name"], s["default"])
                    return val if val < 1.0 else np.sqrt(val)
        return 0.20
    return model_adapter.initial_volatility(model, params)


# Convenience mapping for model display — derived from the registry (single
# source of truth) rather than hand-maintained.
MODEL_NAMES = {key: spec.name for key, spec in get_all_models().items()}


def get_model_display_name(key: str) -> str:
    """Get display name for a model, including custom models."""
    if key in MODEL_NAMES:
        return MODEL_NAMES[key]
    try:
        import streamlit as st

        custom = st.session_state.get("custom_model")
        if custom and "spec" in custom:
            return custom["spec"].name
    except Exception:
        pass
    return key
