"""
Simulation Runner Service for Monte Carlo Simulation Explorer.

Provides centralized simulation execution logic, separating the simulation
operations from the UI layer for better maintainability and testability.

Updated to use the new backend API structure.
"""

import time as time_module
from typing import Dict, Any, Optional
import numpy as np

from backend.simulation.base import SimulationResult
from backend.simulation.factory import create_simulator
from backend.portfolio.pnl import (
    calculate_portfolio_pnl_vectorized,
    calculate_portfolio_pnl_with_stock,
    compute_risk_metrics,
    RiskMetrics
)


def run_price_simulation(params: Dict[str, Any]) -> SimulationResult:
    """
    Execute price path simulation based on the selected model.

    Parameters
    ----------
    params : dict
        Simulation parameters including:
        - price_model: str - Model type ('gbm', 'heston', 'merton', 'bates', 'garch', 'ngarch', 'gjr_garch')
        - spot_price: float - Initial price S₀
        - expected_return: float - Annual expected return μ
        - volatility: float - Base volatility σ
        - time_horizon: float - Simulation period T in years
        - num_paths: int - Number of paths to simulate
        - num_steps: int - Number of time steps
        - seed: int, optional - Random seed for reproducibility
        - Model-specific parameters

    Returns
    -------
    SimulationResult
        Result object containing paths, time grid, and statistics
    """
    model = params.get('price_model', 'gbm')

    # Common parameters
    s0 = params.get('spot_price', 100.0)
    mu = params.get('expected_return', params.get('risk_free_rate', 0.05))
    sigma = params.get('volatility', 0.20)
    t = params.get('time_horizon', 1.0)
    n_paths = int(params.get('num_paths', 10000))
    n_steps = int(params.get('num_steps', 252))
    seed = params.get('seed')

    # Create simulator based on model type
    if model == 'gbm':
        simulator = create_simulator('gbm', sigma=sigma)

    elif model == 'heston':
        simulator = create_simulator(
            'heston',
            v0=params.get('heston_v0', sigma**2),
            kappa=params.get('heston_kappa', 2.0),
            theta=params.get('heston_theta', sigma**2),
            xi=params.get('heston_xi', 0.3),
            rho=params.get('heston_rho', -0.7)
        )

    elif model == 'merton':
        simulator = create_simulator(
            'merton',
            sigma=sigma,
            lambda_j=params.get('merton_lambda', 0.5),
            mu_j=params.get('merton_mu_j', -0.1),
            sigma_j=params.get('merton_sigma_j', 0.2)
        )

    elif model == 'bates':
        simulator = create_simulator(
            'bates',
            v0=params.get('bates_v0', sigma**2),
            kappa=params.get('bates_kappa', 2.0),
            theta=params.get('bates_theta', sigma**2),
            xi=params.get('bates_xi', 0.3),
            rho=params.get('bates_rho', -0.7),
            lambda_j=params.get('bates_lambda', 0.5),
            mu_j=params.get('bates_mu_j', -0.1),
            sigma_j=params.get('bates_sigma_j', 0.2)
        )

    elif model == 'garch':
        simulator = create_simulator(
            'garch',
            omega=params.get('garch_omega', sigma**2 * 0.05),
            alpha=params.get('garch_alpha', 0.05),
            beta=params.get('garch_beta', 0.90),
            sigma0=sigma
        )

    elif model == 'ngarch':
        simulator = create_simulator(
            'ngarch',
            omega=params.get('garch_omega', sigma**2 * 0.05),
            alpha=params.get('garch_alpha', 0.05),
            beta=params.get('garch_beta', 0.90),
            theta=params.get('ngarch_theta', 0.5),
            sigma0=sigma
        )

    elif model == 'gjr_garch':
        simulator = create_simulator(
            'gjr_garch',
            omega=params.get('garch_omega', sigma**2 * 0.05),
            alpha=params.get('garch_alpha', 0.05),
            beta=params.get('garch_beta', 0.90),
            gamma=params.get('gjr_gamma', 0.05),
            sigma0=sigma
        )
    else:
        # Default to GBM
        simulator = create_simulator('gbm', sigma=sigma)

    # Run simulation
    result = simulator.simulate_paths(
        s0=s0,
        mu=mu,
        t=t,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed
    )

    return result


def calculate_pnl_from_paths(
    price_result: SimulationResult,
    params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Calculate portfolio P&L from existing price simulation result.

    Uses terminal prices from the simulation to compute P&L for all
    option positions and optional stock position.

    Parameters
    ----------
    price_result : SimulationResult
        Result from price simulation containing paths
    params : dict
        Parameters including position_arrays with:
        - strikes: np.ndarray
        - option_types: np.ndarray (1=call, -1=put)
        - position_types: np.ndarray (1=long, -1=short)
        - quantities: np.ndarray
        - premiums: np.ndarray
        - stock_quantity: float
        - stock_entry_price: float

    Returns
    -------
    dict or None
        Dictionary containing:
        - terminal_prices: np.ndarray
        - pnl_values: np.ndarray
        - risk_metrics: RiskMetrics
        - computation_time: float
        - num_paths: int
        Returns None if no positions are defined.
    """
    position_arrays = params.get('position_arrays', {})

    if len(position_arrays.get('strikes', [])) == 0:
        return None

    start_time = time_module.perf_counter()

    # Use terminal prices from the simulation result
    terminal_prices = price_result.terminal_prices

    stock_qty = position_arrays.get('stock_quantity', 0.0)
    stock_entry = position_arrays.get('stock_entry_price', 0.0)

    if stock_qty != 0.0:
        pnl_values = calculate_portfolio_pnl_with_stock(
            terminal_prices,
            position_arrays['strikes'],
            position_arrays['option_types'],
            position_arrays['position_types'],
            position_arrays['quantities'],
            position_arrays['premiums'],
            stock_qty,
            stock_entry,
            multiplier=100.0
        )
    else:
        pnl_values = calculate_portfolio_pnl_vectorized(
            terminal_prices,
            position_arrays['strikes'],
            position_arrays['option_types'],
            position_arrays['position_types'],
            position_arrays['quantities'],
            position_arrays['premiums'],
            multiplier=100.0
        )

    risk_metrics = compute_risk_metrics(pnl_values)
    computation_time = time_module.perf_counter() - start_time

    return {
        'terminal_prices': terminal_prices,
        'pnl_values': pnl_values,
        'risk_metrics': risk_metrics,
        'computation_time': computation_time,
        'num_paths': len(terminal_prices)
    }
