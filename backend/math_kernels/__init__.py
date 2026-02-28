"""
Math Kernels Package
====================

STANDALONE REFERENCE IMPLEMENTATIONS - Not integrated into main pipeline.

This module contains low-level Numba-optimized kernels that serve as:
1. Reference implementations for mathematical algorithms
2. Building blocks for custom high-performance code
3. Testing/validation of the main simulation engine

Architecture Decision
---------------------
This package is INTENTIONALLY SEPARATE from the main `simulation/` and
`engines/` modules. The reasons are:

1. **Separation of concerns**: Pure mathematical kernels vs. application logic
2. **Flexibility**: Can be used independently for research/experimentation
3. **Validation**: Reference implementations for testing the production code
4. **No coupling**: Changes here don't affect the main pricing pipeline

The main `simulation/` module has its own integrated, production-optimized
implementations that are tightly coupled with the Model-Engine architecture.

Modules
-------
- sde_kernels: Euler, Milstein, Heston QE discretization schemes
- random: Correlated Brownian motion, antithetic variates
- payoff_kernels: Vectorized payoff evaluation (vanilla, digital, Asian, barriers)
- regression: Longstaff-Schwartz regression for American options

All functions are JIT-compiled with Numba for optimal performance.

Use Cases
---------
- Custom simulation implementations outside the main framework
- Performance-critical research code
- American option pricing via Longstaff-Schwartz regression
- Algorithm prototyping and validation

Example
-------
    from backend.math_kernels import gbm_exact_step, call_payoff_vec
    import numpy as np

    # Simulate GBM paths manually
    dt = 1/252
    paths = np.zeros((10000, 252))
    paths[:, 0] = 100.0
    for t in range(1, 252):
        dW = np.random.randn(10000) * np.sqrt(dt)
        paths[:, t] = gbm_exact_step(paths[:, t-1], 0.05, 0.2, dt, dW)

    # Evaluate payoffs
    payoffs = call_payoff_vec(paths[:, -1], 100.0)

Author: Thomas
Created: 2025
"""

from backend.math_kernels.payoff_kernels import (
    asian_arithmetic_payoff,
    barrier_down_out_put_payoff,
    barrier_up_out_call_payoff,
    butterfly_payoff,
    call_payoff,
    call_payoff_vec,
    digital_call_payoff,
    digital_call_payoff_vec,
    digital_put_payoff,
    digital_put_payoff_vec,
    put_payoff,
    put_payoff_vec,
    straddle_payoff,
    strangle_payoff,
)
from backend.math_kernels.random import (
    box_muller_transform,
    cholesky_transform,
    compute_cholesky,
    generate_antithetic_brownian,
    generate_antithetic_normals,
    generate_correlated_brownian,
    generate_correlated_normals,
    generate_normal,
    generate_normal_2d,
)
from backend.math_kernels.regression import (
    continuation_value,
    laguerre_basis,
    lstsq_regression,
    polynomial_basis,
)
from backend.math_kernels.sde_kernels import (
    euler_step,
    gbm_euler_step,
    gbm_exact_step,
    heston_euler_step,
    heston_qe_step,
    heston_reflection_step,
    heston_spot_step,
    heston_truncation_step,
    merton_jump_step,
    milstein_step,
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
