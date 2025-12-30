"""
Monte Carlo Simulation Package
==============================

This package provides high-performance Monte Carlo simulation tools for
derivatives pricing and risk management. All implementations use Numba
JIT compilation for optimal performance.

Modules:
--------
- simulate_paths: Price path simulation (GBM, Heston, Merton, SABR)
- simulate_volatility: Volatility path simulation (GARCH, NGARCH, GJR-GARCH, EGARCH)

Quick Start:
------------

    # Price path simulation
    from backend.simulation import simulate_paths, simulate_terminal

    result = simulate_paths('gbm', s0=100, r=0.05, sigma=0.2, t=1.0, n_paths=10000)
    print(f"Terminal price mean: ${result.terminal_values.mean():.2f}")

    # Volatility path simulation
    from backend.simulation import simulate_volatility_paths

    result = simulate_volatility_paths('garch', sigma0=0.20, n_paths=10000)
    print(f"Terminal vol mean: {result.terminal_volatility.mean()*100:.2f}%")

    # Joint price-volatility simulation
    from backend.simulation import simulate_joint_paths

    result = simulate_joint_paths('ngarch', s0=100, r=0.05, sigma0=0.20, t=1.0)
    print(f"Price: ${result.terminal_prices.mean():.2f}")
    print(f"Vol: {result.terminal_volatility.mean()*100:.2f}%")

Available Models:
-----------------

Price Models:
    - 'gbm': Geometric Brownian Motion (Black-Scholes)
    - 'heston': Heston Stochastic Volatility
    - 'merton': Merton Jump Diffusion
    - 'sabr': SABR Model

Volatility Models:
    - 'garch': GARCH(1,1) - Bollerslev (1986)
    - 'ngarch': NGARCH/NAGARCH - Engle & Ng (1993)
    - 'gjr_garch': GJR-GARCH - Glosten, Jagannathan & Runkle (1993)
    - 'egarch': EGARCH - Nelson (1991)
"""

# =============================================================================
# Price Path Simulation Exports
# =============================================================================

from .simulate_paths import (
    # Enums and data classes
    ModelType,
    SimulationResult,

    # Core simulation functions (Numba-optimized)
    simulate_gbm_paths,
    simulate_gbm_paths_vectorized,
    simulate_heston_paths,
    simulate_heston_single_path,
    simulate_merton_jump_paths,
    simulate_sabr_paths,
    simulate_correlated_gbm_paths,
    simulate_gbm_with_control_variate,

    # Terminal-only simulations (memory efficient)
    simulate_gbm_terminal,
    simulate_heston_terminal,
    simulate_merton_terminal,
    simulate_sabr_terminal,

    # High-level interfaces
    simulate_paths,
    simulate_terminal,

    # Option pricing utilities
    price_european_call_mc,
    price_european_put_mc,
    price_asian_arithmetic_call_mc,
    price_lookback_call_mc,
    price_barrier_down_out_call_mc,

    # Benchmarking
    benchmark_simulation,
    run_full_benchmark,
    print_benchmark_results,
)

# =============================================================================
# Volatility Path Simulation Exports
# =============================================================================

from .simulate_volatility import (
    # Enums and data classes
    VolatilityModelType,
    VolatilitySimulationResult,
    JointSimulationResult,

    # Core simulation functions (Numba-optimized)
    simulate_garch_paths,
    simulate_garch_single_path,
    simulate_ngarch_paths,
    simulate_ngarch_single_path,
    simulate_gjr_garch_paths,
    simulate_egarch_paths,

    # Terminal-only simulations
    simulate_garch_terminal,
    simulate_ngarch_terminal,

    # Joint price-volatility simulations
    simulate_gbm_garch_paths,
    simulate_gbm_ngarch_paths,

    # High-level interfaces
    simulate_volatility_paths,
    simulate_terminal_volatility,
    simulate_joint_paths,

    # Parameter utilities
    compute_garch_long_run_variance,
    compute_ngarch_long_run_variance,
    validate_garch_params,
    validate_ngarch_params,
    estimate_garch_params_from_volatility,

    # Benchmarking
    benchmark_volatility_simulation,
    run_volatility_benchmark,
    print_volatility_benchmark_results,
)

# =============================================================================
# Package-level Utilities
# =============================================================================

__all__ = [
    # Price simulation
    'ModelType',
    'SimulationResult',
    'simulate_paths',
    'simulate_terminal',
    'simulate_gbm_paths',
    'simulate_gbm_paths_vectorized',
    'simulate_heston_paths',
    'simulate_merton_jump_paths',
    'simulate_sabr_paths',
    'simulate_correlated_gbm_paths',
    'simulate_gbm_terminal',
    'simulate_heston_terminal',
    'simulate_merton_terminal',
    'simulate_sabr_terminal',
    'price_european_call_mc',
    'price_european_put_mc',
    'price_asian_arithmetic_call_mc',
    'price_lookback_call_mc',
    'price_barrier_down_out_call_mc',

    # Volatility simulation
    'VolatilityModelType',
    'VolatilitySimulationResult',
    'JointSimulationResult',
    'simulate_volatility_paths',
    'simulate_terminal_volatility',
    'simulate_joint_paths',
    'simulate_garch_paths',
    'simulate_ngarch_paths',
    'simulate_gjr_garch_paths',
    'simulate_egarch_paths',
    'simulate_garch_terminal',
    'simulate_ngarch_terminal',
    'simulate_gbm_garch_paths',
    'simulate_gbm_ngarch_paths',

    # Utilities
    'compute_garch_long_run_variance',
    'compute_ngarch_long_run_variance',
    'validate_garch_params',
    'validate_ngarch_params',
    'estimate_garch_params_from_volatility',

    # Benchmarking
    'benchmark_simulation',
    'run_full_benchmark',
    'benchmark_volatility_simulation',
    'run_volatility_benchmark',
]

__version__ = '1.1.0'
