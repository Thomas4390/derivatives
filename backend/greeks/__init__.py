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
    # Scaling utilities
    unscale_greeks,
    scale_greeks,
    # Scaling constants
    VEGA_SCALE,
    RHO_SCALE,
    THETA_SCALE,
    VANNA_SCALE,
    VOLGA_SCALE,
    CHARM_SCALE,
    VETA_SCALE,
    ZOMMA_SCALE,
    COLOR_SCALE,
    ULTIMA_SCALE,
    # Result types
    AllGreeks,
    FirstOrderGreeks,
    SecondOrderGreeks,
    ThirdOrderGreeks,
)
from .numerical import (
    # Configuration
    GreeksBumpConfig,
    DEFAULT_BUMP_CONFIG,
    # Result types
    NumericalGreeks,
    ModelNumericalGreeks,
    # First-order
    finite_difference_delta,
    finite_difference_gamma,
    finite_difference_vega,
    finite_difference_theta,
    finite_difference_rho,
    # Second-order
    finite_difference_vanna,
    finite_difference_volga,
    finite_difference_charm,
    finite_difference_veta,
    # Third-order
    finite_difference_speed,
    finite_difference_zomma,
    finite_difference_color,
    finite_difference_ultima,
    # Combined
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
    # Scaling utilities
    "unscale_greeks",
    "scale_greeks",
    # Scaling constants
    "VEGA_SCALE",
    "RHO_SCALE",
    "THETA_SCALE",
    "VANNA_SCALE",
    "VOLGA_SCALE",
    "CHARM_SCALE",
    "VETA_SCALE",
    "ZOMMA_SCALE",
    "COLOR_SCALE",
    "ULTIMA_SCALE",
    # Analytic result types
    "AllGreeks",
    "FirstOrderGreeks",
    "SecondOrderGreeks",
    "ThirdOrderGreeks",
    # Numerical - configuration
    "GreeksBumpConfig",
    "DEFAULT_BUMP_CONFIG",
    # Numerical - result types
    "NumericalGreeks",
    "ModelNumericalGreeks",
    # Numerical - first order
    "finite_difference_delta",
    "finite_difference_gamma",
    "finite_difference_vega",
    "finite_difference_theta",
    "finite_difference_rho",
    # Numerical - second order
    "finite_difference_vanna",
    "finite_difference_volga",
    "finite_difference_charm",
    "finite_difference_veta",
    # Numerical - third order
    "finite_difference_speed",
    "finite_difference_zomma",
    "finite_difference_color",
    "finite_difference_ultima",
    # Combined
    "finite_difference_greeks",
]
