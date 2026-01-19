"""
Utility Functions for Monte Carlo Simulation
=============================================

This module provides common utility functions used across
the simulation modules.
"""

import numpy as np
from typing import Tuple, Optional

# Re-export commonly used helpers from constants
from backend.simulation.constants import (
    MIN_VARIANCE_FLOOR,
    floor_variance,
    compute_correlation_decomposition,
)

__all__ = [
    # Re-exports
    'MIN_VARIANCE_FLOOR',
    'floor_variance',
    'compute_correlation_decomposition',
    # New functions
    'generate_correlated_normals',
    'generate_correlated_normal_pair',
    'compute_log_return',
    'compute_annualized_volatility',
    'validate_positive',
    'validate_probability',
    'validate_correlation',
]


# =============================================================================
# CORRELATED RANDOM NUMBER GENERATION
# =============================================================================

def generate_correlated_normals(
    z1: np.ndarray,
    z2: np.ndarray,
    rho: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate correlated normal variates from independent ones.

    Uses Cholesky decomposition to create correlated pairs:
    Z2_correlated = rho * Z1 + sqrt(1 - rho^2) * Z2

    Parameters
    ----------
    z1 : np.ndarray
        First independent standard normal array
    z2 : np.ndarray
        Second independent standard normal array
    rho : float
        Target correlation coefficient in [-1, 1]

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (z1, z2_correlated) where z2_correlated is correlated with z1

    Examples
    --------
    >>> z1 = np.random.randn(1000)
    >>> z2 = np.random.randn(1000)
    >>> z1_out, z2_corr = generate_correlated_normals(z1, z2, rho=0.5)
    >>> np.corrcoef(z1_out, z2_corr)[0, 1]  # Should be ~0.5
    """
    sqrt_one_minus_rho2 = compute_correlation_decomposition(rho)
    z2_correlated = rho * z1 + sqrt_one_minus_rho2 * z2
    return z1, z2_correlated


def generate_correlated_normal_pair(
    rho: float,
    size: Optional[Tuple[int, ...]] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a pair of correlated standard normal random variates.

    Parameters
    ----------
    rho : float
        Target correlation coefficient in [-1, 1]
    size : Optional[Tuple[int, ...]]
        Shape of output arrays (default: scalar)
    seed : Optional[int]
        Random seed for reproducibility

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (z1, z2) correlated normal variates

    Examples
    --------
    >>> z1, z2 = generate_correlated_normal_pair(0.7, size=(10000,), seed=42)
    >>> np.corrcoef(z1, z2)[0, 1]  # Should be ~0.7
    """
    if seed is not None:
        np.random.seed(seed)

    z1 = np.random.standard_normal(size)
    z2_indep = np.random.standard_normal(size)
    _, z2 = generate_correlated_normals(z1, z2_indep, rho)

    return z1, z2


# =============================================================================
# FINANCIAL CALCULATIONS
# =============================================================================

def compute_log_return(prices: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute log returns from a price series.

    Parameters
    ----------
    prices : np.ndarray
        Price array
    axis : int
        Axis along which to compute returns (default: last axis)

    Returns
    -------
    np.ndarray
        Log returns: ln(P_t / P_{t-1})
    """
    return np.diff(np.log(prices), axis=axis)


def compute_annualized_volatility(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized volatility from a return series.

    Parameters
    ----------
    returns : np.ndarray
        Array of periodic returns
    periods_per_year : int
        Number of periods per year (default: 252 for daily data)

    Returns
    -------
    float
        Annualized volatility
    """
    return float(np.std(returns) * np.sqrt(periods_per_year))


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_positive(value: float, name: str) -> None:
    """
    Validate that a value is strictly positive.

    Parameters
    ----------
    value : float
        Value to validate
    name : str
        Parameter name for error message

    Raises
    ------
    ValueError
        If value <= 0
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_probability(value: float, name: str) -> None:
    """
    Validate that a value is a valid probability [0, 1].

    Parameters
    ----------
    value : float
        Value to validate
    name : str
        Parameter name for error message

    Raises
    ------
    ValueError
        If value not in [0, 1]
    """
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def validate_correlation(rho: float, name: str = "rho") -> None:
    """
    Validate that a value is a valid correlation [-1, 1].

    Parameters
    ----------
    rho : float
        Correlation value to validate
    name : str
        Parameter name for error message

    Raises
    ------
    ValueError
        If rho not in [-1, 1]
    """
    if not -1 <= rho <= 1:
        raise ValueError(f"{name} must be in [-1, 1], got {rho}")


# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================

def compute_var(
    values: np.ndarray,
    alpha: float = 0.05
) -> float:
    """
    Compute Value-at-Risk at given confidence level.

    Parameters
    ----------
    values : np.ndarray
        Array of P&L or return values
    alpha : float
        Significance level (default: 0.05 for 95% VaR)

    Returns
    -------
    float
        VaR value (typically negative for losses)
    """
    return float(np.percentile(values, alpha * 100))


def compute_cvar(
    values: np.ndarray,
    alpha: float = 0.05
) -> float:
    """
    Compute Conditional Value-at-Risk (Expected Shortfall).

    Parameters
    ----------
    values : np.ndarray
        Array of P&L or return values
    alpha : float
        Significance level (default: 0.05 for 95% CVaR)

    Returns
    -------
    float
        CVaR value (mean of values below VaR)
    """
    var = compute_var(values, alpha)
    return float(np.mean(values[values <= var]))


def compute_percentiles(
    values: np.ndarray,
    percentiles: Optional[list] = None
) -> dict:
    """
    Compute multiple percentiles of a distribution.

    Parameters
    ----------
    values : np.ndarray
        Array of values
    percentiles : Optional[list]
        List of percentile levels (default: [5, 25, 50, 75, 95])

    Returns
    -------
    dict
        Dictionary mapping percentile level to value
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    return {p: float(np.percentile(values, p)) for p in percentiles}
