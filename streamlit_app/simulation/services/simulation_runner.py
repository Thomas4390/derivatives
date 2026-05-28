"""
Simulation Runner Service for Monte Carlo Simulation Explorer.

Provides P&L calculation from simulation results for vanilla option legs and
optional stock positions.
"""

import time as time_module
from typing import Any

import numpy as np

from backend.portfolio.pnl import (
    calculate_portfolio_pnl_vectorized,
    calculate_portfolio_pnl_with_stock,
    compute_payoff_curve,
    compute_risk_metrics,
)
from backend.simulation.base import SimulationResult


def has_exotic_legs(position_arrays: dict) -> bool:
    """Return False — only vanilla legs are supported in this build."""
    return False


def compute_mtm_pnl_at_step(
    paths: np.ndarray,
    step_idx: int,
    position_arrays: dict,
    rate: float,
    sigma: float,
    time_to_expiry: float,
    multiplier: float = 100.0,
) -> np.ndarray:
    """Compute mark-to-market P&L at an intermediate time step.

    Uses vectorised Black-Scholes repricing for vanilla legs (intrinsic at
    expiry).

    Parameters
    ----------
    paths : np.ndarray
        Shape ``(n_paths, n_steps+1)`` simulated price paths.
    step_idx : int
        Time step index to evaluate at (0 = inception, ``n_steps`` = expiry).
    position_arrays : dict
        Portfolio leg arrays (strikes, option_types, position_types,
        quantities, premiums, stock_quantity, stock_entry_price).
    rate : float
        Risk-free rate (annualised).
    sigma : float
        Volatility (annualised).
    time_to_expiry : float
        Total time to expiry in years (T).
    multiplier : float
        Contract multiplier (default 100).

    Returns
    -------
    np.ndarray
        Shape ``(n_paths,)`` P&L per path at the given step.
    """
    from backend.utils.math import bs_price

    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1
    dt = time_to_expiry / n_steps if n_steps > 0 else 0.0
    tau = time_to_expiry - step_idx * dt
    spot_at_step = paths[:, step_idx]

    strikes = position_arrays["strikes"]
    option_types = position_arrays["option_types"]
    position_types = position_arrays["position_types"]
    quantities = position_arrays["quantities"]
    premiums = position_arrays["premiums"]

    pnl = np.zeros(n_paths)

    for j in range(len(strikes)):
        K = float(strikes[j])
        direction = float(position_types[j])
        qty = float(quantities[j])
        premium = float(premiums[j])
        is_call = option_types[j] == 1.0

        if tau < 1e-10:
            values = np.where(
                is_call,
                np.maximum(spot_at_step - K, 0.0),
                np.maximum(K - spot_at_step, 0.0),
            )
        else:
            values = np.array(
                [
                    bs_price(float(s), K, tau, rate, sigma, is_call)
                    for s in spot_at_step
                ]
            )
        pnl += direction * (values - premium) * qty * multiplier

    stock_qty = position_arrays.get("stock_quantity", 0.0)
    if stock_qty != 0.0:
        stock_entry = position_arrays.get("stock_entry_price", 0.0)
        pnl += stock_qty * (spot_at_step - stock_entry)

    return pnl


def compute_hybrid_payoff_curve(
    spot_range: np.ndarray,
    position_arrays: dict,
    multiplier: float = 100.0,
) -> np.ndarray:
    """Vectorised payoff curve for vanilla legs plus optional stock leg."""
    strikes = position_arrays.get("strikes", np.array([]))

    if len(strikes) > 0:
        payoff = compute_payoff_curve(
            spot_range,
            strikes,
            position_arrays["option_types"],
            position_arrays["position_types"],
            position_arrays["quantities"],
            position_arrays["premiums"],
            0.0,
            0.0,
        )
    else:
        payoff = np.zeros(len(spot_range))

    stock_qty = position_arrays.get("stock_quantity", 0.0)
    if stock_qty != 0.0:
        stock_entry = position_arrays.get("stock_entry_price", 0.0)
        payoff += stock_qty * (spot_range - stock_entry)

    return payoff


def calculate_pnl_from_paths(
    price_result: SimulationResult, params: dict[str, Any]
) -> dict[str, Any] | None:
    """
    Calculate portfolio P&L from an existing price simulation result.

    Uses terminal prices to compute P&L for vanilla option legs and the
    optional stock position.

    Parameters
    ----------
    price_result : SimulationResult
        Result from price simulation containing paths.
    params : dict
        Parameters including ``position_arrays`` with strikes, option_types,
        position_types, quantities, premiums, stock_quantity,
        stock_entry_price.

    Returns
    -------
    dict or None
        Dictionary with terminal_prices, pnl_values, risk_metrics,
        computation_time, num_paths. Returns None if no positions are
        defined.
    """
    position_arrays = params.get("position_arrays", {})

    if len(position_arrays.get("strikes", [])) == 0:
        return None

    start_time = time_module.perf_counter()

    terminal_prices = price_result.terminal_prices

    stock_qty = position_arrays.get("stock_quantity", 0.0)
    stock_entry = position_arrays.get("stock_entry_price", 0.0)

    if stock_qty != 0.0:
        pnl_values = calculate_portfolio_pnl_with_stock(
            terminal_prices,
            position_arrays["strikes"],
            position_arrays["option_types"],
            position_arrays["position_types"],
            position_arrays["quantities"],
            position_arrays["premiums"],
            stock_qty,
            stock_entry,
            multiplier=100.0,
        )
    else:
        pnl_values = calculate_portfolio_pnl_vectorized(
            terminal_prices,
            position_arrays["strikes"],
            position_arrays["option_types"],
            position_arrays["position_types"],
            position_arrays["quantities"],
            position_arrays["premiums"],
            multiplier=100.0,
        )

    risk_metrics = compute_risk_metrics(pnl_values)
    computation_time = time_module.perf_counter() - start_time

    return {
        "terminal_prices": terminal_prices,
        "pnl_values": pnl_values,
        "risk_metrics": risk_metrics,
        "computation_time": computation_time,
        "num_paths": len(terminal_prices),
    }
