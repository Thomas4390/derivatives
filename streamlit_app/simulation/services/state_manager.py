"""
State management for Monte Carlo Simulation Explorer.

This module provides a centralized interface for managing Streamlit session state.
"""

import streamlit as st
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict

from config.constants import (
    DEFAULT_SPOT_PRICE,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_VOLATILITY,
    DEFAULT_TIME_HORIZON,
    DEFAULT_NUM_PATHS,
    DEFAULT_NUM_STEPS,
    DEFAULT_HESTON_V0,
    DEFAULT_HESTON_KAPPA,
    DEFAULT_HESTON_THETA,
    DEFAULT_HESTON_XI,
    DEFAULT_HESTON_RHO,
    DEFAULT_MERTON_LAMBDA,
    DEFAULT_MERTON_MU_J,
    DEFAULT_MERTON_SIGMA_J,
    DEFAULT_SABR_BETA,
    DEFAULT_SABR_NU,
    DEFAULT_SABR_RHO,
    DEFAULT_GARCH_OMEGA,
    DEFAULT_GARCH_ALPHA,
    DEFAULT_GARCH_BETA,
    DEFAULT_NGARCH_THETA,
    DEFAULT_GJR_GAMMA,
    DEFAULT_EGARCH_GAMMA,
    DEFAULT_EXPECTED_RETURN
)


@dataclass
class SimulationParams:
    """Container for simulation parameters."""
    # Common parameters
    spot_price: float = DEFAULT_SPOT_PRICE
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    volatility: float = DEFAULT_VOLATILITY
    time_horizon: float = DEFAULT_TIME_HORIZON
    num_paths: int = DEFAULT_NUM_PATHS
    num_steps: int = DEFAULT_NUM_STEPS
    seed: Optional[int] = 42

    # Heston parameters
    heston_v0: float = DEFAULT_HESTON_V0
    heston_kappa: float = DEFAULT_HESTON_KAPPA
    heston_theta: float = DEFAULT_HESTON_THETA
    heston_xi: float = DEFAULT_HESTON_XI
    heston_rho: float = DEFAULT_HESTON_RHO

    # Merton jump parameters
    merton_lambda: float = DEFAULT_MERTON_LAMBDA
    merton_mu_j: float = DEFAULT_MERTON_MU_J
    merton_sigma_j: float = DEFAULT_MERTON_SIGMA_J

    # SABR parameters
    sabr_beta: float = DEFAULT_SABR_BETA
    sabr_nu: float = DEFAULT_SABR_NU
    sabr_rho: float = DEFAULT_SABR_RHO

    # GARCH family parameters
    garch_omega: float = DEFAULT_GARCH_OMEGA
    garch_alpha: float = DEFAULT_GARCH_ALPHA
    garch_beta: float = DEFAULT_GARCH_BETA
    ngarch_theta: float = DEFAULT_NGARCH_THETA
    gjr_gamma: float = DEFAULT_GJR_GAMMA
    egarch_gamma: float = DEFAULT_EGARCH_GAMMA

    # Option P&L parameters
    expected_return: float = DEFAULT_EXPECTED_RETURN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def init_session_state() -> None:
    """Initialize all session state variables with defaults."""
    if 'simulation_params' not in st.session_state:
        st.session_state.simulation_params = SimulationParams()

    if 'price_model' not in st.session_state:
        st.session_state.price_model = 'gbm'

    if 'volatility_model' not in st.session_state:
        st.session_state.volatility_model = 'garch'

    if 'simulation_mode' not in st.session_state:
        st.session_state.simulation_mode = 'price_paths'  # or 'volatility_paths' or 'joint'

    if 'last_simulation_result' not in st.session_state:
        st.session_state.last_simulation_result = None

    if 'show_percentile_bands' not in st.session_state:
        st.session_state.show_percentile_bands = True

    if 'show_mean_path' not in st.session_state:
        st.session_state.show_mean_path = True

    # Simulation results by mode
    if 'price_result' not in st.session_state:
        st.session_state.price_result = None

    if 'vol_result' not in st.session_state:
        st.session_state.vol_result = None

    if 'pnl_result' not in st.session_state:
        st.session_state.pnl_result = None

    # Option P&L specific state
    if 'option_positions' not in st.session_state:
        st.session_state.option_positions = []

    if 'stock_position' not in st.session_state:
        st.session_state.stock_position = None


def get_simulation_params() -> SimulationParams:
    """Get current simulation parameters."""
    return st.session_state.get('simulation_params', SimulationParams())


def set_simulation_params(params: SimulationParams) -> None:
    """Set simulation parameters."""
    st.session_state.simulation_params = params


def update_simulation_param(key: str, value: Any) -> None:
    """Update a single simulation parameter."""
    if 'simulation_params' not in st.session_state:
        st.session_state.simulation_params = SimulationParams()
    setattr(st.session_state.simulation_params, key, value)


def get_price_model() -> str:
    """Get current price model selection."""
    return st.session_state.get('price_model', 'gbm')


def set_price_model(model: str) -> None:
    """Set price model selection."""
    st.session_state.price_model = model


def get_volatility_model() -> str:
    """Get current volatility model selection."""
    return st.session_state.get('volatility_model', 'garch')


def set_volatility_model(model: str) -> None:
    """Set volatility model selection."""
    st.session_state.volatility_model = model


def get_simulation_mode() -> str:
    """Get current simulation mode."""
    return st.session_state.get('simulation_mode', 'price_paths')


def set_simulation_mode(mode: str) -> None:
    """Set simulation mode."""
    st.session_state.simulation_mode = mode


def set_simulation_result(result: Any) -> None:
    """Cache the last simulation result."""
    st.session_state.last_simulation_result = result


def get_simulation_result() -> Any:
    """Get the cached simulation result."""
    return st.session_state.get('last_simulation_result', None)


def clear_simulation_cache() -> None:
    """Clear cached simulation results."""
    st.session_state.last_simulation_result = None


def reset_to_defaults() -> None:
    """Reset all parameters to defaults."""
    st.session_state.simulation_params = SimulationParams()
    st.session_state.price_model = 'gbm'
    st.session_state.volatility_model = 'garch'
    st.session_state.simulation_mode = 'price_paths'
    st.session_state.last_simulation_result = None
    st.session_state.price_result = None
    st.session_state.vol_result = None
    st.session_state.pnl_result = None
    st.session_state.option_positions = []
    st.session_state.stock_position = None


def get_pnl_result() -> Any:
    """Get the cached P&L simulation result."""
    return st.session_state.get('pnl_result', None)


def set_pnl_result(result: Any) -> None:
    """Cache the P&L simulation result."""
    st.session_state.pnl_result = result


def clear_pnl_result() -> None:
    """Clear cached P&L simulation result."""
    st.session_state.pnl_result = None


def get_option_positions() -> list:
    """Get current option positions."""
    return st.session_state.get('option_positions', [])


def set_option_positions(positions: list) -> None:
    """Set option positions."""
    st.session_state.option_positions = positions


def get_stock_position() -> Any:
    """Get current stock position."""
    return st.session_state.get('stock_position', None)


def set_stock_position(position: Any) -> None:
    """Set stock position."""
    st.session_state.stock_position = position


def clear_all_positions() -> None:
    """Clear all option and stock positions."""
    st.session_state.option_positions = []
    st.session_state.stock_position = None
