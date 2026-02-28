"""
State management for Monte Carlo Simulation Explorer.

This module provides a centralized interface for managing Streamlit session state.
"""

from dataclasses import asdict, dataclass
from typing import Any

import streamlit as st
from config.constants import (
    DEFAULT_EGARCH_GAMMA,
    DEFAULT_EXPECTED_RETURN,
    DEFAULT_GARCH_ALPHA,
    DEFAULT_GARCH_BETA,
    DEFAULT_GARCH_OMEGA,
    DEFAULT_GJR_GAMMA,
    DEFAULT_HESTON_KAPPA,
    DEFAULT_HESTON_RHO,
    DEFAULT_HESTON_THETA,
    DEFAULT_HESTON_V0,
    DEFAULT_HESTON_XI,
    DEFAULT_MERTON_LAMBDA,
    DEFAULT_MERTON_MU_J,
    DEFAULT_MERTON_SIGMA_J,
    DEFAULT_NGARCH_THETA,
    DEFAULT_NUM_PATHS,
    DEFAULT_NUM_STEPS,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_SABR_BETA,
    DEFAULT_SABR_NU,
    DEFAULT_SABR_RHO,
    DEFAULT_SPOT_PRICE,
    DEFAULT_TIME_HORIZON,
    DEFAULT_VOLATILITY,
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
    seed: int | None = 42

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

    def to_dict(self) -> dict[str, Any]:
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

    # Config versioning for stale result detection
    init_config_versioning()

    # Strategy builder expanded state
    if 'strategy_expanded' not in st.session_state:
        st.session_state.strategy_expanded = True

    # Active main tab
    if 'active_main_tab' not in st.session_state:
        st.session_state.active_main_tab = 'config'


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


# =============================================================================
# CONFIG VERSIONING FOR STALE RESULT DETECTION
# =============================================================================

def init_config_versioning() -> None:
    """Initialize config versioning for detecting parameter changes."""
    if 'config_version' not in st.session_state:
        st.session_state.config_version = 0

    if 'results_versions' not in st.session_state:
        st.session_state.results_versions = {
            'price': -1,
            'volatility': -1,
            'pnl': -1
        }

    # Track last known parameter values for change detection
    if 'last_params_hash' not in st.session_state:
        st.session_state.last_params_hash = None


def increment_config_version() -> None:
    """Increment config version when parameters change."""
    if 'config_version' not in st.session_state:
        st.session_state.config_version = 0
    st.session_state.config_version += 1


def get_config_version() -> int:
    """Get current config version."""
    return st.session_state.get('config_version', 0)


def mark_results_current(result_type: str) -> None:
    """Mark results as current (matching config version)."""
    if 'results_versions' not in st.session_state:
        st.session_state.results_versions = {'price': -1, 'volatility': -1, 'pnl': -1}
    st.session_state.results_versions[result_type] = get_config_version()


def are_results_stale(result_type: str) -> bool:
    """Check if results are stale (config changed since last run)."""
    if 'results_versions' not in st.session_state:
        return True
    return st.session_state.results_versions.get(result_type, -1) != get_config_version()


def compute_params_hash(params: dict) -> str:
    """Compute a hash of relevant parameters for change detection."""
    import hashlib
    import json

    # Keys to track for change detection
    tracked_keys = [
        'spot_price', 'risk_free_rate', 'volatility', 'time_horizon',
        'num_paths', 'num_steps', 'seed', 'expected_return',
        'price_model', 'vol_model',
        # Heston
        'heston_v0', 'heston_kappa', 'heston_theta', 'heston_xi', 'heston_rho',
        # Merton
        'merton_lambda', 'merton_mu_j', 'merton_sigma_j',
        # Bates
        'bates_v0', 'bates_kappa', 'bates_theta', 'bates_xi', 'bates_rho',
        'bates_lambda', 'bates_mu_j', 'bates_sigma_j',
        # SABR
        'sabr_beta', 'sabr_nu', 'sabr_rho',
        # GARCH
        'garch_alpha', 'garch_beta', 'garch_omega',
        'ngarch_theta', 'gjr_gamma', 'egarch_gamma'
    ]

    tracked_params = {k: params.get(k) for k in tracked_keys if k in params}

    # Also track position arrays for P&L
    if 'position_arrays' in params:
        pos = params['position_arrays']
        tracked_params['strikes'] = list(pos.get('strikes', []))
        tracked_params['option_types'] = list(pos.get('option_types', []))
        tracked_params['position_types'] = list(pos.get('position_types', []))
        tracked_params['quantities'] = list(pos.get('quantities', []))

    hash_str = json.dumps(tracked_params, sort_keys=True, default=str)
    return hashlib.md5(hash_str.encode()).hexdigest()


def check_params_changed(params: dict) -> bool:
    """Check if parameters changed and update version if so."""
    current_hash = compute_params_hash(params)
    last_hash = st.session_state.get('last_params_hash')

    if last_hash != current_hash:
        st.session_state.last_params_hash = current_hash
        increment_config_version()
        return True
    return False


# =============================================================================
# ACTIVE TAB STATE
# =============================================================================

def get_active_tab() -> str:
    """Get the currently active main tab."""
    return st.session_state.get('active_main_tab', 'config')


def set_active_tab(tab: str) -> None:
    """Set the currently active main tab."""
    st.session_state.active_main_tab = tab


# =============================================================================
# STRATEGY BUILDER STATE
# =============================================================================

def get_strategy_expanded() -> bool:
    """Get whether strategy builder is expanded."""
    return st.session_state.get('strategy_expanded', True)


def set_strategy_expanded(expanded: bool) -> None:
    """Set strategy builder expanded state."""
    st.session_state.strategy_expanded = expanded


def toggle_strategy_expanded() -> None:
    """Toggle strategy builder expanded state."""
    current = get_strategy_expanded()
    set_strategy_expanded(not current)
