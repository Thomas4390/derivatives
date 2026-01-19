"""
Type Definitions for Monte Carlo Simulation
============================================

This module provides TypedDict classes and type aliases for
structured parameter passing and better IDE support.
"""

from typing import TypedDict, Optional, List, Union, Dict, Any
import numpy as np
from numpy.typing import NDArray


# =============================================================================
# NUMPY TYPE ALIASES
# =============================================================================

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


# =============================================================================
# PRICE MODEL PARAMETER TYPES
# =============================================================================

class GBMParams(TypedDict, total=False):
    """Parameters for Geometric Brownian Motion model."""
    s0: float  # Initial stock price
    r: float   # Risk-free rate (or drift mu for P-measure)
    sigma: float  # Volatility
    t: float   # Time to maturity
    n_paths: int  # Number of paths
    n_steps: int  # Number of time steps
    seed: Optional[int]  # Random seed


class HestonParams(TypedDict, total=False):
    """Parameters for Heston stochastic volatility model."""
    s0: float  # Initial stock price
    v0: float  # Initial variance
    r: float   # Risk-free rate
    kappa: float  # Mean reversion speed
    theta: float  # Long-run variance
    xi: float  # Vol of vol
    rho: float  # Correlation between price and variance
    t: float   # Time to maturity
    n_paths: int
    n_steps: int
    scheme: int  # Discretization scheme (0-3)
    seed: Optional[int]


class MertonParams(TypedDict, total=False):
    """Parameters for Merton jump-diffusion model."""
    s0: float
    r: float
    sigma: float  # Diffusion volatility
    lam: float   # Jump intensity (lambda)
    mu_j: float  # Mean of log jump size
    sigma_j: float  # Std of log jump size
    t: float
    n_paths: int
    n_steps: int
    seed: Optional[int]


class BatesParams(TypedDict, total=False):
    """Parameters for Bates model (Heston + jumps)."""
    s0: float
    v0: float
    r: float
    kappa: float
    theta: float
    xi: float
    rho: float
    lam: float  # Jump intensity
    mu_j: float
    sigma_j: float
    t: float
    n_paths: int
    n_steps: int
    scheme: int
    seed: Optional[int]


class SABRParams(TypedDict, total=False):
    """Parameters for SABR model."""
    f0: float   # Initial forward price
    alpha0: float  # Initial vol
    beta: float  # CEV exponent (0 to 1)
    rho: float  # Correlation
    nu: float   # Vol of vol
    t: float
    n_paths: int
    n_steps: int
    seed: Optional[int]


# =============================================================================
# VOLATILITY MODEL PARAMETER TYPES
# =============================================================================

class GARCHParams(TypedDict, total=False):
    """Parameters for GARCH(1,1) model."""
    sigma0: float  # Initial volatility
    omega: float   # Constant term
    alpha: float   # ARCH coefficient
    beta: float    # GARCH coefficient
    n_paths: int
    n_steps: int
    seed: Optional[int]


class NGARCHParams(TypedDict, total=False):
    """Parameters for NGARCH (NAGARCH) model."""
    sigma0: float
    omega: float
    alpha: float
    beta: float
    theta: float  # Leverage parameter
    n_paths: int
    n_steps: int
    seed: Optional[int]


class GJRGARCHParams(TypedDict, total=False):
    """Parameters for GJR-GARCH model."""
    sigma0: float
    omega: float
    alpha: float
    beta: float
    gamma: float  # Asymmetry parameter
    n_paths: int
    n_steps: int
    seed: Optional[int]


class EGARCHParams(TypedDict, total=False):
    """Parameters for EGARCH model."""
    sigma0: float
    omega: float
    alpha: float
    beta: float
    gamma: float  # Leverage parameter
    n_paths: int
    n_steps: int
    seed: Optional[int]


# =============================================================================
# OPTION PARAMETERS
# =============================================================================

class OptionParams(TypedDict, total=False):
    """Parameters for an option position."""
    option_type: str  # 'call' or 'put'
    strike: float
    premium: float
    quantity: int
    position: str  # 'long' or 'short'


class StrategyParams(TypedDict):
    """Parameters for an option strategy."""
    legs: List[OptionParams]
    underlying_quantity: int  # Shares of underlying (0 if options only)
    underlying_cost: float  # Cost basis of underlying


# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================

class SimulationConfig(TypedDict, total=False):
    """General simulation configuration."""
    n_paths: int
    n_steps: int
    seed: Optional[int]
    return_full_paths: bool


class PriceSimulationConfig(SimulationConfig, total=False):
    """Configuration for price path simulation."""
    model: str  # 'gbm', 'heston', 'merton', 'bates', 'sabr'
    measure: str  # 'Q' (risk-neutral) or 'P' (physical)


class VolatilitySimulationConfig(SimulationConfig, total=False):
    """Configuration for volatility path simulation."""
    model: str  # 'garch', 'ngarch', 'gjr_garch', 'egarch'


# =============================================================================
# RESULT TYPES
# =============================================================================

class SimulationStats(TypedDict):
    """Statistics from a simulation run."""
    mean: float
    std: float
    min: float
    max: float
    median: float
    skewness: float
    kurtosis: float


class RiskMetricsDict(TypedDict):
    """Dictionary representation of risk metrics."""
    mean_pnl: float
    std_pnl: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    prob_profit: float
    max_profit: float
    max_loss: float


class BenchmarkStats(TypedDict):
    """Statistics from a benchmark run."""
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    throughput_paths_per_sec: float
    throughput_samples_per_sec: float


# =============================================================================
# UTILITY TYPE ALIASES
# =============================================================================

# Union of all price model parameter types
PriceModelParams = Union[
    GBMParams,
    HestonParams,
    MertonParams,
    BatesParams,
    SABRParams
]

# Union of all volatility model parameter types
VolatilityModelParams = Union[
    GARCHParams,
    NGARCHParams,
    GJRGARCHParams,
    EGARCHParams
]

# Generic parameter dictionary (for backward compatibility)
ParamsDict = Dict[str, Any]
