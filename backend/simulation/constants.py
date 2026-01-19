"""
Backend Simulation Constants
============================

Centralized constants for Monte Carlo simulation modules.
This module eliminates magic numbers and provides a single source of truth
for numerical constants, model defaults, and thresholds.

Usage:
------
    from backend.simulation.constants import (
        MIN_VARIANCE_FLOOR,
        VAR_95_ALPHA,
        HESTON_DEFAULTS,
    )
"""

import numpy as np

# =============================================================================
# NUMERICAL CONSTANTS
# =============================================================================

# Variance floor to prevent numerical instability in stochastic volatility models
MIN_VARIANCE_FLOOR = 1e-10

# Floor for standard deviation in statistical calculations
MIN_STD_FLOOR = 1e-10

# Expected value of |Z| where Z ~ N(0,1): E[|Z|] = sqrt(2/pi)
EXPECTED_ABS_NORMAL = np.sqrt(2.0 / np.pi)  # ≈ 0.7978845608

# Tolerance for eigenvalue checks in correlation matrices
EIGENVALUE_TOLERANCE = 1e-10

# Tolerance for numerical comparisons
NUMERICAL_TOLERANCE = 1e-10

# Small value threshold for psi in QE scheme
QE_PSI_THRESHOLD = 1.5

# Trading days per year (for annualization)
TRADING_DAYS_PER_YEAR = 252

# =============================================================================
# VAR CONFIDENCE LEVELS
# =============================================================================

# VaR at 95% confidence (5% tail)
VAR_95_ALPHA = 0.05

# VaR at 99% confidence (1% tail)
VAR_99_ALPHA = 0.01

# =============================================================================
# OPTION CONTRACT CONSTANTS
# =============================================================================

# Standard contract multiplier (shares per contract)
DEFAULT_CONTRACT_MULTIPLIER = 100.0

# Excess kurtosis adjustment (normal distribution has kurtosis = 3)
EXCESS_KURTOSIS_ADJUSTMENT = 3.0

# =============================================================================
# HESTON MODEL DEFAULTS
# =============================================================================

HESTON_DEFAULTS = {
    'v0': 0.04,      # Initial variance (corresponds to 20% volatility)
    'kappa': 2.0,    # Mean reversion speed
    'theta': 0.04,   # Long-term variance level
    'xi': 0.3,       # Volatility of volatility
    'rho': -0.7,     # Correlation between price and volatility
}

# =============================================================================
# MERTON JUMP DIFFUSION DEFAULTS
# =============================================================================

MERTON_DEFAULTS = {
    'lambda_j': 0.5,   # Jump intensity (expected jumps per year)
    'mu_j': -0.1,      # Mean of log-jump size
    'sigma_j': 0.2,    # Std dev of log-jump size
}

# =============================================================================
# BATES MODEL DEFAULTS (Heston + Jumps)
# =============================================================================

BATES_DEFAULTS = {
    # Heston parameters
    'v0': 0.04,
    'kappa': 2.0,
    'theta': 0.04,
    'xi': 0.3,
    'rho': -0.7,
    # Jump parameters
    'lambda_j': 0.5,
    'mu_j': -0.1,
    'sigma_j': 0.2,
}

# =============================================================================
# SABR MODEL DEFAULTS
# =============================================================================

SABR_DEFAULTS = {
    'beta': 0.5,     # CEV exponent (0 = normal, 1 = lognormal)
    'nu': 0.4,       # Volatility of volatility
    'rho': -0.3,     # Correlation
}

# =============================================================================
# GARCH MODEL DEFAULTS
# =============================================================================

GARCH_DEFAULTS = {
    'omega': 0.000002,  # Base variance (constant term)
    'alpha': 0.05,      # ARCH coefficient (reaction to shocks)
    'beta': 0.90,       # GARCH coefficient (persistence)
}

# =============================================================================
# NGARCH (NAGARCH) MODEL DEFAULTS
# =============================================================================

NGARCH_DEFAULTS = {
    'omega': 0.000002,
    'alpha': 0.05,
    'beta': 0.90,
    'theta': 0.5,       # Leverage/asymmetry parameter
}

# =============================================================================
# GJR-GARCH MODEL DEFAULTS
# =============================================================================

GJR_GARCH_DEFAULTS = {
    'omega': 0.000002,
    'alpha': 0.05,
    'beta': 0.90,
    'gamma': 0.05,      # Asymmetry coefficient for negative shocks
}

# =============================================================================
# EGARCH MODEL DEFAULTS
# =============================================================================

EGARCH_DEFAULTS = {
    'omega': -0.1,      # Constant in log-variance equation
    'alpha': 0.1,       # Magnitude effect
    'beta': 0.98,       # Persistence (high for EGARCH)
    'gamma': -0.1,      # Asymmetry/leverage (negative for leverage effect)
}

# =============================================================================
# SIMULATION DEFAULTS
# =============================================================================

SIMULATION_DEFAULTS = {
    'n_paths': 10000,
    'n_steps': 252,
    'time_horizon': 1.0,  # 1 year
}

# =============================================================================
# HESTON DISCRETIZATION SCHEMES
# =============================================================================

class HestonScheme:
    """Heston model discretization scheme identifiers."""
    EULER = 0
    FULL_TRUNCATION = 1
    REFLECTION = 2
    QE = 3  # Quadratic Exponential

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def floor_variance(variance: float, floor: float = MIN_VARIANCE_FLOOR) -> float:
    """
    Ensure variance stays positive.

    Parameters
    ----------
    variance : float
        Variance value to floor
    floor : float
        Minimum allowed variance (default: MIN_VARIANCE_FLOOR)

    Returns
    -------
    float
        max(variance, floor)
    """
    return max(variance, floor)


def compute_correlation_decomposition(rho: float) -> float:
    """
    Compute sqrt(1 - rho^2) for correlated normal generation.

    This is used in Cholesky-style correlation decomposition where:
    Z2_correlated = rho * Z1 + sqrt(1 - rho^2) * Z2

    Parameters
    ----------
    rho : float
        Correlation coefficient in [-1, 1]

    Returns
    -------
    float
        sqrt(1 - rho^2)
    """
    return np.sqrt(1.0 - rho * rho)
