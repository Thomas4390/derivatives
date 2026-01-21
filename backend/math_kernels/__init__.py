"""
Math Kernels Package
====================

ALTERNATIVE OPTIMIZATION LAYER - Not integrated into main pipeline.

This module contains low-level Numba-optimized kernels that can be used
as building blocks for custom high-performance implementations.

Current Status: STANDALONE - Not used by simulation/ or engines/ modules.
The simulation/ module has its own integrated implementations.

Modules:
- sde_kernels: Euler, Milstein, Heston QE discretization schemes
- random: Correlated Brownian motion, antithetic variates
- payoff_kernels: Vectorized payoff evaluation (vanilla, digital, Asian, barriers)
- regression: Longstaff-Schwartz regression for American options

All functions are JIT-compiled with Numba for optimal performance.

Use Cases:
- Custom simulation implementations
- Performance-critical research code
- American option pricing via regression

Author: Thomas
Created: 2025
"""

from .sde_kernels import (
    euler_step,
    milstein_step,
    gbm_exact_step,
    gbm_euler_step,
    heston_euler_step,
    heston_truncation_step,
    heston_reflection_step,
    heston_qe_step,
    heston_spot_step,
    merton_jump_step,
)

from .random import (
    generate_normal,
    generate_normal_2d,
    generate_correlated_normals,
    generate_correlated_brownian,
    generate_antithetic_normals,
    generate_antithetic_brownian,
    cholesky_transform,
    compute_cholesky,
    box_muller_transform,
)

from .payoff_kernels import (
    call_payoff,
    put_payoff,
    call_payoff_vec,
    put_payoff_vec,
    digital_call_payoff,
    digital_put_payoff,
    digital_call_payoff_vec,
    digital_put_payoff_vec,
    straddle_payoff,
    strangle_payoff,
    butterfly_payoff,
    asian_arithmetic_payoff,
    barrier_up_out_call_payoff,
    barrier_down_out_put_payoff,
)

from .regression import (
    laguerre_basis,
    polynomial_basis,
    lstsq_regression,
    continuation_value,
)


__all__ = [
    # SDE kernels
    "euler_step",
    "milstein_step",
    "gbm_exact_step",
    "gbm_euler_step",
    "heston_euler_step",
    "heston_truncation_step",
    "heston_reflection_step",
    "heston_qe_step",
    "heston_spot_step",
    "merton_jump_step",
    # Random
    "generate_normal",
    "generate_normal_2d",
    "generate_correlated_normals",
    "generate_correlated_brownian",
    "generate_antithetic_normals",
    "generate_antithetic_brownian",
    "cholesky_transform",
    "compute_cholesky",
    "box_muller_transform",
    # Payoffs
    "call_payoff",
    "put_payoff",
    "call_payoff_vec",
    "put_payoff_vec",
    "digital_call_payoff",
    "digital_put_payoff",
    "digital_call_payoff_vec",
    "digital_put_payoff_vec",
    "straddle_payoff",
    "strangle_payoff",
    "butterfly_payoff",
    "asian_arithmetic_payoff",
    "barrier_up_out_call_payoff",
    "barrier_down_out_put_payoff",
    # Regression
    "laguerre_basis",
    "polynomial_basis",
    "lstsq_regression",
    "continuation_value",
]
