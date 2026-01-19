"""
Simulation Runner Service for Monte Carlo Simulation Explorer.

Provides centralized simulation execution logic, separating the simulation
operations from the UI layer for better maintainability and testability.
"""

import time as time_module
from typing import Dict, Any, Optional

from backend.simulation.simulate_paths import (
    simulate_paths,
    simulate_terminal,
    SimulationResult
)
from backend.simulation.simulate_volatility import (
    simulate_volatility_paths,
    VolatilitySimulationResult
)
from backend.simulation.pnl_engine import (
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
        - price_model: str - Model type ('gbm', 'heston', 'merton', 'bates', 'sabr')
        - spot_price: float - Initial price S₀
        - expected_return: float - Annual expected return μ
        - volatility: float - Base volatility σ
        - time_horizon: float - Simulation period T in years
        - num_paths: int - Number of paths to simulate
        - num_steps: int - Number of time steps
        - seed: int, optional - Random seed for reproducibility
        - Model-specific parameters (heston_*, merton_*, bates_*, sabr_*)

    Returns
    -------
    SimulationResult
        Result object containing paths, time grid, and statistics
    """
    model = params['price_model']

    common_params = {
        's0': params['spot_price'],
        'mu': params.get('expected_return', params['risk_free_rate']),
        'sigma': params['volatility'],
        't': params['time_horizon'],
        'n_paths': params['num_paths'],
        'n_steps': params['num_steps'],
        'seed': params.get('seed')
    }

    if model == 'heston':
        common_params.update({
            'v0': params['heston_v0'],
            'kappa': params['heston_kappa'],
            'theta': params['heston_theta'],
            'xi': params['heston_xi'],
            'rho': params['heston_rho']
        })
    elif model == 'merton':
        common_params.update({
            'lambda_j': params['merton_lambda'],
            'mu_j': params['merton_mu_j'],
            'sigma_j': params['merton_sigma_j']
        })
    elif model == 'bates':
        common_params.update({
            'v0': params['bates_v0'],
            'kappa': params['bates_kappa'],
            'theta': params['bates_theta'],
            'xi': params['bates_xi'],
            'rho': params['bates_rho'],
            'lambda_j': params['bates_lambda'],
            'mu_j': params['bates_mu_j'],
            'sigma_j': params['bates_sigma_j']
        })
    elif model == 'sabr':
        common_params.update({
            'alpha0': params['volatility'],
            'beta': params['sabr_beta'],
            'rho': params['sabr_rho'],
            'nu': params['sabr_nu']
        })

    return simulate_paths(model=model, **common_params)


def run_volatility_simulation(params: Dict[str, Any]) -> VolatilitySimulationResult:
    """
    Execute volatility path simulation based on the selected GARCH model.

    Parameters
    ----------
    params : dict
        Simulation parameters including:
        - vol_model: str - Model type ('garch', 'ngarch', 'gjr_garch', 'egarch')
        - volatility: float - Initial volatility σ₀
        - num_paths: int - Number of paths to simulate
        - num_steps: int - Number of time steps
        - seed: int, optional - Random seed
        - garch_omega, garch_alpha, garch_beta: float - GARCH parameters
        - Model-specific: ngarch_theta, gjr_gamma, egarch_gamma

    Returns
    -------
    VolatilitySimulationResult
        Result object containing volatility paths and statistics
    """
    model = params['vol_model']

    common_params = {
        'sigma0': params['volatility'],
        'n_paths': params['num_paths'],
        'n_steps': params['num_steps'],
        'seed': params.get('seed')
    }

    garch_params = {
        'omega': params.get('garch_omega', 0.00001),
        'alpha': params['garch_alpha'],
        'beta': params['garch_beta']
    }

    if model == 'garch':
        common_params.update(garch_params)
    elif model == 'ngarch':
        common_params.update(garch_params)
        common_params['theta'] = params['ngarch_theta']
    elif model == 'gjr_garch':
        common_params.update(garch_params)
        common_params['gamma'] = params['gjr_gamma']
    elif model == 'egarch':
        common_params.update(garch_params)
        common_params['gamma'] = params['egarch_gamma']

    return simulate_volatility_paths(model=model, **common_params)


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
        - option_types: np.ndarray (0=call, 1=put)
        - position_types: np.ndarray (0=long, 1=short)
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
    terminal_prices = price_result.paths[:, -1]

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


def run_terminal_only_simulation(params: Dict[str, Any]) -> SimulationResult:
    """
    Execute terminal-only price simulation (no intermediate steps stored).

    More memory-efficient for large simulations when only terminal
    values are needed (e.g., for P&L calculation).

    Parameters
    ----------
    params : dict
        Same as run_price_simulation

    Returns
    -------
    SimulationResult
        Result with terminal values (paths array contains only terminal column)
    """
    model = params['price_model']

    common_params = {
        's0': params['spot_price'],
        'mu': params.get('expected_return', params['risk_free_rate']),
        'sigma': params['volatility'],
        't': params['time_horizon'],
        'n_paths': params['num_paths'],
        'n_steps': params['num_steps'],
        'seed': params.get('seed')
    }

    # Model-specific parameters (same as run_price_simulation)
    if model == 'heston':
        common_params.update({
            'v0': params['heston_v0'],
            'kappa': params['heston_kappa'],
            'theta': params['heston_theta'],
            'xi': params['heston_xi'],
            'rho': params['heston_rho']
        })
    elif model == 'merton':
        common_params.update({
            'lambda_j': params['merton_lambda'],
            'mu_j': params['merton_mu_j'],
            'sigma_j': params['merton_sigma_j']
        })
    elif model == 'bates':
        common_params.update({
            'v0': params['bates_v0'],
            'kappa': params['bates_kappa'],
            'theta': params['bates_theta'],
            'xi': params['bates_xi'],
            'rho': params['bates_rho'],
            'lambda_j': params['bates_lambda'],
            'mu_j': params['bates_mu_j'],
            'sigma_j': params['bates_sigma_j']
        })
    elif model == 'sabr':
        common_params.update({
            'alpha0': params['volatility'],
            'beta': params['sabr_beta'],
            'rho': params['sabr_rho'],
            'nu': params['sabr_nu']
        })

    return simulate_terminal(model=model, **common_params)
