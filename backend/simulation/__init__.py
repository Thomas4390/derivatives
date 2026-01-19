"""
Monte Carlo Simulation Package
==============================

This package provides high-performance Monte Carlo simulation tools for
scenario analysis and risk management. All implementations use Numba
JIT compilation for optimal performance.

P-Measure (Physical Measure) Implementation:
--------------------------------------------
All simulations use the REAL-WORLD (P) measure with drift = μ (expected return).
This is appropriate for:
    - Scenario analysis and risk management
    - VaR and stress testing
    - Backtesting trading strategies
    - Generating realistic price trajectories

For DERIVATIVES PRICING, use the Q-measure (risk-neutral) where drift = r.
Simply pass mu=r (risk-free rate) to use risk-neutral dynamics.

Modules:
--------
- simulate_paths: Price path simulation (GBM, Heston, Merton, SABR)
- simulate_volatility: Volatility path simulation (GARCH, NGARCH, GJR-GARCH, EGARCH)

Quick Start:
------------

    # Price path simulation (P-measure with 8% expected return)
    from backend.simulation import simulate_paths, simulate_terminal

    result = simulate_paths('gbm', s0=100, mu=0.08, sigma=0.2, t=1.0, n_paths=10000)
    print(f"Terminal price mean: ${result.terminal_values.mean():.2f}")

    # For derivatives pricing (Q-measure), use mu=r:
    result = simulate_paths('gbm', s0=100, mu=0.05, sigma=0.2, t=1.0, n_paths=10000)

    # Volatility path simulation
    from backend.simulation import simulate_volatility_paths

    result = simulate_volatility_paths('garch', sigma0=0.20, n_paths=10000)
    print(f"Terminal vol mean: {result.terminal_volatility.mean()*100:.2f}%")

    # Joint price-volatility simulation
    from backend.simulation import simulate_joint_paths

    result = simulate_joint_paths('ngarch', s0=100, mu=0.08, sigma0=0.20, t=1.0)
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
# P&L Engine Exports (Option P&L Simulation)
# =============================================================================

from .pnl_engine import (
    # Data classes
    RiskMetrics,

    # Core P&L calculation (Numba-optimized)
    calculate_portfolio_pnl_vectorized,
    calculate_portfolio_pnl_with_stock,

    # Risk metrics
    compute_risk_metrics,
    compute_risk_metrics_core,
    compute_skewness_kurtosis,
    compute_percentiles,

    # Payoff analysis
    compute_payoff_curve,
    find_breakeven_points,

    # Utilities
    prepare_position_arrays,
    warm_up_jit,
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

    # P&L Engine
    'RiskMetrics',
    'calculate_portfolio_pnl_vectorized',
    'calculate_portfolio_pnl_with_stock',
    'compute_risk_metrics',
    'compute_risk_metrics_core',
    'compute_skewness_kurtosis',
    'compute_percentiles',
    'compute_payoff_curve',
    'find_breakeven_points',
    'prepare_position_arrays',
    'warm_up_jit',
]

# =============================================================================
# Constants, Types and Utilities
# =============================================================================

from .constants import (
    MIN_VARIANCE_FLOOR,
    EXPECTED_ABS_NORMAL,
    VAR_95_ALPHA,
    VAR_99_ALPHA,
    HESTON_DEFAULTS,
    GARCH_DEFAULTS,
    MERTON_DEFAULTS,
    SABR_DEFAULTS,
    NGARCH_DEFAULTS,
    GJR_GARCH_DEFAULTS,
    EGARCH_DEFAULTS,
    floor_variance,
    compute_correlation_decomposition,
)

from .enums import (
    HestonScheme,
    OptionType,
    PositionType,
    PriceModel,
    VolatilityModel,
    SimulationMode,
    RiskMetricType,
)

from .exceptions import (
    SimulationError,
    ParameterValidationError,
    NumericalInstabilityError,
    ModelNotFoundError,
    StationarityViolationError,
    CorrelationMatrixError,
)

# Update __all__ with new exports
__all__ += [
    # Constants
    'MIN_VARIANCE_FLOOR',
    'HESTON_DEFAULTS',
    'GARCH_DEFAULTS',
    'floor_variance',
    'compute_correlation_decomposition',

    # Enums
    'HestonScheme',
    'OptionType',
    'PositionType',
    'PriceModel',
    'VolatilityModel',
    'SimulationMode',
    'RiskMetricType',

    # Exceptions
    'SimulationError',
    'ParameterValidationError',
    'NumericalInstabilityError',
    'ModelNotFoundError',
    'StationarityViolationError',
    'CorrelationMatrixError',
]

__version__ = '2.1.0'  # Added constants, enums, types, exceptions
