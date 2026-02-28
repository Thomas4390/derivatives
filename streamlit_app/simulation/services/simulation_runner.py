"""
Simulation Runner Service for Monte Carlo Simulation Explorer.

Provides P&L calculation from simulation results.
"""

import time as time_module
from typing import Any

from backend.portfolio.pnl import (
    calculate_portfolio_pnl_vectorized,
    calculate_portfolio_pnl_with_stock,
    compute_risk_metrics,
)
from backend.simulation.base import SimulationResult


def calculate_pnl_from_paths(
    price_result: SimulationResult,
    params: dict[str, Any]
) -> dict[str, Any] | None:
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
