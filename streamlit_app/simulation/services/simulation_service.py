"""
Unified Simulation Service - Single interface for all simulation models.

Provides:
- Unified simulation execution across all 7 models
- Consistent result handling
- Automatic parameter mapping
- Caching support
"""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np

# Backend imports
from backend.simulation.factory import create_simulator, ModelType
from backend.simulation.base import SimulationResult


@dataclass
class UnifiedSimulationParams:
    """Unified parameters for simulation."""
    # Market parameters
    spot: float = 100.0
    drift: float = 0.08
    risk_free_rate: float = 0.05
    time_horizon: float = 1.0

    # Simulation settings
    n_paths: int = 10000
    n_steps: int = 252
    seed: Optional[int] = 42

    # Model selection
    model: str = "gbm"

    # GBM / Merton diffusion
    sigma: Optional[float] = None

    # Heston / Bates variance
    v0: Optional[float] = None
    kappa: Optional[float] = None
    theta: Optional[float] = None
    xi: Optional[float] = None
    rho: Optional[float] = None

    # Jump parameters (Merton / Bates)
    lambda_j: Optional[float] = None
    mu_j: Optional[float] = None
    sigma_j: Optional[float] = None

    # GARCH family
    sigma0: Optional[float] = None
    omega: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None

    # NGARCH leverage
    theta_ngarch: Optional[float] = None

    # GJR-GARCH asymmetry
    gamma: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _extract_model_params(model: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model-specific parameters from unified params dict."""
    model_lower = model.lower()

    if model_lower == "gbm":
        return {
            "sigma": params.get("sigma", 0.20),
        }

    elif model_lower == "heston":
        return {
            "v0": params.get("v0", 0.04),
            "kappa": params.get("kappa", 2.0),
            "theta": params.get("theta", 0.04),
            "xi": params.get("xi", 0.3),
            "rho": params.get("rho", -0.7),
        }

    elif model_lower == "merton":
        return {
            "sigma": params.get("sigma", 0.20),
            "lambda_j": params.get("lambda_j", 0.5),
            "mu_j": params.get("mu_j", -0.1),
            "sigma_j": params.get("sigma_j", 0.2),
        }

    elif model_lower == "bates":
        return {
            "v0": params.get("v0", 0.04),
            "kappa": params.get("kappa", 2.0),
            "theta": params.get("theta", 0.04),
            "xi": params.get("xi", 0.3),
            "rho": params.get("rho", -0.7),
            "lambda_j": params.get("lambda_j", 0.5),
            "mu_j": params.get("mu_j", -0.1),
            "sigma_j": params.get("sigma_j", 0.2),
        }

    elif model_lower == "garch":
        return {
            "sigma0": params.get("sigma0", 0.20),
            "omega": params.get("omega", 0.000001),
            "alpha": params.get("alpha", 0.05),
            "beta": params.get("beta", 0.90),
        }

    elif model_lower == "ngarch":
        return {
            "sigma0": params.get("sigma0", 0.20),
            "omega": params.get("omega", 0.000001),
            "alpha": params.get("alpha", 0.05),
            "beta": params.get("beta", 0.90),
            "theta": params.get("theta_ngarch", params.get("theta", 0.5)),
        }

    elif model_lower == "gjr_garch":
        return {
            "sigma0": params.get("sigma0", 0.20),
            "omega": params.get("omega", 0.000001),
            "alpha": params.get("alpha", 0.05),
            "beta": params.get("beta", 0.90),
            "gamma": params.get("gamma", 0.05),
        }

    else:
        raise ValueError(f"Unknown model: {model}")


def run_simulation(
    model: str,
    params: Dict[str, Any],
    return_simulator: bool = False
) -> Union[SimulationResult, tuple]:
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
    time_horizon = params.get("time_horizon", 1.0)
    n_paths = int(params.get("n_paths", 10000))
    n_steps = int(params.get("n_steps", 252))
    seed = params.get("seed", None)

    # Handle seed=0 as random
    if seed == 0:
        seed = None

    # Extract model-specific parameters
    model_params = _extract_model_params(model, params)

    # Create simulator
    simulator = create_simulator(model, **model_params)

    # Run simulation
    result = simulator.simulate_paths(
        s0=spot,
        mu=drift,
        t=time_horizon,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed
    )

    if return_simulator:
        return result, simulator
    return result


def run_terminal_simulation(
    model: str,
    params: Dict[str, Any]
) -> np.ndarray:
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
    time_horizon = params.get("time_horizon", 1.0)
    n_paths = int(params.get("n_paths", 10000))
    n_steps = int(params.get("n_steps", 252))
    seed = params.get("seed", None)

    if seed == 0:
        seed = None

    model_params = _extract_model_params(model, params)
    simulator = create_simulator(model, **model_params)

    return simulator.simulate_terminal(
        s0=spot,
        mu=drift,
        t=time_horizon,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed
    )


def get_model_characteristics(model: str) -> Dict[str, Any]:
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
        raise ValueError(f"Unknown model: {model}")

    return characteristics[model_lower]


def check_model_conditions(model: str, params: Dict[str, Any]) -> Dict[str, Any]:
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
        # Feller condition: 2*kappa*theta > xi^2
        kappa = params.get("kappa", 2.0)
        theta = params.get("theta", 0.04)
        xi = params.get("xi", 0.3)
        feller_lhs = 2 * kappa * theta
        feller_rhs = xi ** 2
        feller_satisfied = feller_lhs > feller_rhs
        conditions.append({
            "name": "Feller Condition",
            "equation": "2κθ > ξ²",
            "lhs": feller_lhs,
            "rhs": feller_rhs,
            "satisfied": feller_satisfied,
            "description": "Ensures variance stays positive" if feller_satisfied
                          else "Variance may become negative (use Full Truncation scheme)"
        })

    elif model_lower == "garch":
        # Stationarity: alpha + beta < 1
        alpha = params.get("alpha", 0.05)
        beta = params.get("beta", 0.90)
        persistence = alpha + beta
        stationary = persistence < 1
        conditions.append({
            "name": "Stationarity",
            "equation": "α + β < 1",
            "value": persistence,
            "threshold": 1.0,
            "satisfied": stationary,
            "description": "Volatility mean-reverts" if stationary
                          else "Integrated GARCH - volatility doesn't mean-revert"
        })

    elif model_lower == "ngarch":
        # Stationarity: alpha*(1 + theta^2) + beta < 1
        alpha = params.get("alpha", 0.05)
        beta = params.get("beta", 0.90)
        theta = params.get("theta_ngarch", params.get("theta", 0.5))
        persistence = alpha * (1 + theta ** 2) + beta
        stationary = persistence < 1
        conditions.append({
            "name": "Stationarity",
            "equation": "α(1 + θ²) + β < 1",
            "value": persistence,
            "threshold": 1.0,
            "satisfied": stationary,
            "description": "Volatility mean-reverts" if stationary
                          else "Non-stationary - volatility explodes"
        })

    elif model_lower == "gjr_garch":
        # Stationarity: alpha + beta + gamma/2 < 1
        alpha = params.get("alpha", 0.05)
        beta = params.get("beta", 0.90)
        gamma = params.get("gamma", 0.05)
        persistence = alpha + beta + gamma / 2
        stationary = persistence < 1
        conditions.append({
            "name": "Stationarity",
            "equation": "α + β + γ/2 < 1",
            "value": persistence,
            "threshold": 1.0,
            "satisfied": stationary,
            "description": "Volatility mean-reverts" if stationary
                          else "Non-stationary - volatility explodes"
        })

    is_valid = all(c["satisfied"] for c in conditions) if conditions else True

    return {
        "is_valid": is_valid,
        "conditions": conditions
    }


def compute_long_run_volatility(model: str, params: Dict[str, Any]) -> Optional[float]:
    """
    Compute long-run (unconditional) volatility for models that support it.

    Returns:
        Long-run volatility as decimal (e.g., 0.20 for 20%), or None if not applicable
    """
    model_lower = model.lower()

    if model_lower in ["heston", "bates"]:
        theta = params.get("theta", 0.04)
        return np.sqrt(theta)

    elif model_lower == "garch":
        omega = params.get("omega", 0.000001)
        alpha = params.get("alpha", 0.05)
        beta = params.get("beta", 0.90)
        persistence = alpha + beta
        if persistence >= 1:
            return None
        long_run_var = omega / (1 - persistence)
        # Convert from per-step variance to annualized volatility
        # Assuming daily steps, multiply by sqrt(252)
        n_steps = params.get("n_steps", 252)
        return np.sqrt(long_run_var * n_steps)

    elif model_lower == "ngarch":
        omega = params.get("omega", 0.000001)
        alpha = params.get("alpha", 0.05)
        beta = params.get("beta", 0.90)
        theta = params.get("theta_ngarch", params.get("theta", 0.5))
        persistence = alpha * (1 + theta ** 2) + beta
        if persistence >= 1:
            return None
        long_run_var = omega / (1 - persistence)
        n_steps = params.get("n_steps", 252)
        return np.sqrt(long_run_var * n_steps)

    elif model_lower == "gjr_garch":
        omega = params.get("omega", 0.000001)
        alpha = params.get("alpha", 0.05)
        beta = params.get("beta", 0.90)
        gamma = params.get("gamma", 0.05)
        persistence = alpha + beta + gamma / 2
        if persistence >= 1:
            return None
        long_run_var = omega / (1 - persistence)
        n_steps = params.get("n_steps", 252)
        return np.sqrt(long_run_var * n_steps)

    elif model_lower in ["gbm", "merton"]:
        return params.get("sigma", 0.20)

    return None


def get_initial_volatility(model: str, params: Dict[str, Any]) -> float:
    """Get initial volatility for a model."""
    model_lower = model.lower()

    if model_lower in ["gbm", "merton"]:
        return params.get("sigma", 0.20)

    elif model_lower in ["heston", "bates"]:
        v0 = params.get("v0", 0.04)
        return np.sqrt(v0)

    elif model_lower in ["garch", "ngarch", "gjr_garch"]:
        return params.get("sigma0", 0.20)

    return 0.20


# Convenience mapping for model display
MODEL_NAMES = {
    "gbm": "Geometric Brownian Motion",
    "heston": "Heston Stochastic Volatility",
    "merton": "Merton Jump-Diffusion",
    "bates": "Bates (Heston + Jumps)",
    "garch": "GARCH(1,1)",
    "ngarch": "NGARCH (Nonlinear)",
    "gjr_garch": "GJR-GARCH (Threshold)",
}
