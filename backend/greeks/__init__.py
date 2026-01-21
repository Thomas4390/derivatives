"""
Greeks Package
==============

Unified interface for option Greeks calculation.

Modules:
- analytic: Closed-form Greeks for Black-Scholes and other analytic models
- numerical: Finite difference Greeks for any pricing engine
- calculator: Unified calculator dispatching to appropriate method

All 14 Greeks are supported:
- First order: delta, gamma, vega, theta, rho
- Second order: vanna, volga/vomma, charm, veta
- Third order: speed, zomma, color, ultima

Author: Thomas
Created: 2025
"""

from .calculator import GreeksCalculator, calculate_greeks
from .analytic import (
    bs_greeks_first_order,
    bs_greeks_second_order,
    bs_greeks_third_order,
    bs_all_greeks,
)
from .numerical import (
    finite_difference_delta,
    finite_difference_gamma,
    finite_difference_vega,
    finite_difference_theta,
    finite_difference_rho,
    finite_difference_greeks,
)


__all__ = [
    # Unified interface
    "GreeksCalculator",
    "calculate_greeks",
    # Analytic
    "bs_greeks_first_order",
    "bs_greeks_second_order",
    "bs_greeks_third_order",
    "bs_all_greeks",
    # Numerical
    "finite_difference_delta",
    "finite_difference_gamma",
    "finite_difference_vega",
    "finite_difference_theta",
    "finite_difference_rho",
    "finite_difference_greeks",
]
