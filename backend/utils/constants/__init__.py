"""
Centralized Constants Package
==============================

Single source of truth for all backend constants.

Modules:
    time        - Time conventions (DAYS_PER_YEAR, TRADING_DAYS_PER_YEAR)
    numerical   - Tolerances, floors, limits
    greeks      - Greek indices and scaling factors
    monte_carlo - MC defaults and bump sizes
    calibration - Model calibration bounds and parameter names
    market      - Default market environment values (spot, rate, div, maturity)
    constraints - Reparametrization clamps and Numba-path floors

Usage::

    from backend.utils.constants import TRADING_DAYS_PER_YEAR, GREEK_DELTA
    from backend.utils.constants.calibration import HESTON_BOUNDS

Author: Thomas Vaudescal
Created: 2026
"""

# --- time ---
from backend.utils.constants.time import (
    CALENDAR_DAYS_PER_YEAR,
    DAYS_PER_YEAR,
    OBSERVATION_PERIODS,
    TRADING_DAYS_PER_YEAR,
)

# --- numerical ---
from backend.utils.constants.numerical import (
    DEFAULT_TOLERANCE,
    DEFAULT_YIELD_CURVE_TENORS,
    DIVIDEND_YIELD_MAX,
    FFT_DEFAULT_ALPHA,
    FFT_DEFAULT_ETA,
    FFT_DEFAULT_N,
    GARCH_CALIBRATION_VARIANCE_FLOOR,
    GARCH_VARIANCE_FLOOR,
    LOG_2PI,
    MIN_MATURITY,
    PUT_CALL_PARITY_TOLERANCE,
    QE_M_FLOOR,
    QE_PSI_THRESHOLD,
    RATE_MAX,
    RATE_MIN,
    SMOOTHING_EPS,
    VAR_FLOOR,
    VOLATILITY_MAX,
)

# --- greeks ---
from backend.utils.constants.greeks import (
    CHARM_SCALE,
    COLOR_SCALE,
    FIRST_ORDER_GREEKS,
    GREEK_CHARM,
    GREEK_COLOR,
    GREEK_DELTA,
    GREEK_GAMMA,
    GREEK_NAMES,
    GREEK_PRICE,
    GREEK_RHO,
    GREEK_SPEED,
    GREEK_THETA,
    GREEK_ULTIMA,
    GREEK_VANNA,
    GREEK_VEGA,
    GREEK_VETA,
    GREEK_VOLGA,
    GREEK_ZOMMA,
    NUM_GREEKS,
    RHO_SCALE,
    SECOND_ORDER_GREEKS,
    THETA_SCALE,
    THIRD_ORDER_GREEKS,
    ULTIMA_SCALE,
    VALID_GREEKS,
    VANNA_SCALE,
    VEGA_SCALE,
    VETA_SCALE,
    VOLGA_SCALE,
    ZOMMA_SCALE,
)

# --- monte_carlo ---
from backend.utils.constants.monte_carlo import (
    DEFAULT_MC_PATHS,
    DEFAULT_MC_STEPS_PER_YEAR,
    DEFAULT_RATE_BUMP,
    DEFAULT_SPOT_BUMP,
    DEFAULT_TIME_BUMP_DAYS,
    DEFAULT_VOL_BUMP,
)

# --- calibration ---
from backend.utils.constants.calibration import (
    BATES_BOUNDS,
    BATES_PARAM_NAMES,
    GARCH_BOUNDS,
    GJR_BOUNDS,
    HESTON_BOUNDS,
    HESTON_PARAM_NAMES,
    JUMP_BOUNDS,
    JUMP_PARAM_NAMES,
    LM_DEFAULT_DAMPING,
    LM_DEFAULT_MAX_ITER,
    LM_DEFAULT_TOL,
    MERTON_BOUNDS,
    MERTON_PARAM_NAMES,
    NGARCH_BOUNDS,
    VALID_CALIBRATION_OBJECTIVES,
    VALID_GARCH_TYPES,
)

# --- market ---
from backend.utils.constants.market import (
    DEFAULT_DIV,
    DEFAULT_MATURITY,
    DEFAULT_RATE,
    DEFAULT_SPOT,
)

# --- constraints ---
from backend.utils.constants.constraints import (
    NUMBA_EPS_MATURITY,
    NUMBA_EPS_VEGA,
    RHO_CLAMP,
    SOFTPLUS_STABILITY_THRESHOLD,
)

__all__ = [
    # Time
    "CALENDAR_DAYS_PER_YEAR",
    "DAYS_PER_YEAR",
    "OBSERVATION_PERIODS",
    "TRADING_DAYS_PER_YEAR",
    # Numerical
    "DEFAULT_TOLERANCE",
    "DEFAULT_YIELD_CURVE_TENORS",
    "DIVIDEND_YIELD_MAX",
    "FFT_DEFAULT_ALPHA",
    "FFT_DEFAULT_ETA",
    "FFT_DEFAULT_N",
    "GARCH_CALIBRATION_VARIANCE_FLOOR",
    "GARCH_VARIANCE_FLOOR",
    "LOG_2PI",
    "MIN_MATURITY",
    "PUT_CALL_PARITY_TOLERANCE",
    "QE_M_FLOOR",
    "QE_PSI_THRESHOLD",
    "RATE_MAX",
    "RATE_MIN",
    "SMOOTHING_EPS",
    "VAR_FLOOR",
    "VOLATILITY_MAX",
    # Greeks
    "CHARM_SCALE",
    "COLOR_SCALE",
    "FIRST_ORDER_GREEKS",
    "GREEK_CHARM",
    "GREEK_COLOR",
    "GREEK_DELTA",
    "GREEK_GAMMA",
    "GREEK_NAMES",
    "GREEK_PRICE",
    "GREEK_RHO",
    "GREEK_SPEED",
    "GREEK_THETA",
    "GREEK_ULTIMA",
    "GREEK_VANNA",
    "GREEK_VEGA",
    "GREEK_VETA",
    "GREEK_VOLGA",
    "GREEK_ZOMMA",
    "NUM_GREEKS",
    "RHO_SCALE",
    "SECOND_ORDER_GREEKS",
    "THETA_SCALE",
    "THIRD_ORDER_GREEKS",
    "ULTIMA_SCALE",
    "VALID_GREEKS",
    "VANNA_SCALE",
    "VEGA_SCALE",
    "VETA_SCALE",
    "VOLGA_SCALE",
    "ZOMMA_SCALE",
    # Monte Carlo
    "DEFAULT_MC_PATHS",
    "DEFAULT_MC_STEPS_PER_YEAR",
    "DEFAULT_RATE_BUMP",
    "DEFAULT_SPOT_BUMP",
    "DEFAULT_TIME_BUMP_DAYS",
    "DEFAULT_VOL_BUMP",
    # Calibration
    "BATES_BOUNDS",
    "BATES_PARAM_NAMES",
    "GARCH_BOUNDS",
    "GJR_BOUNDS",
    "HESTON_BOUNDS",
    "HESTON_PARAM_NAMES",
    "JUMP_BOUNDS",
    "JUMP_PARAM_NAMES",
    "LM_DEFAULT_DAMPING",
    "LM_DEFAULT_MAX_ITER",
    "LM_DEFAULT_TOL",
    "MERTON_BOUNDS",
    "MERTON_PARAM_NAMES",
    "NGARCH_BOUNDS",
    "VALID_CALIBRATION_OBJECTIVES",
    "VALID_GARCH_TYPES",
    # Market defaults
    "DEFAULT_DIV",
    "DEFAULT_MATURITY",
    "DEFAULT_RATE",
    "DEFAULT_SPOT",
    # Constraints / stability
    "NUMBA_EPS_MATURITY",
    "NUMBA_EPS_VEGA",
    "RHO_CLAMP",
    "SOFTPLUS_STABILITY_THRESHOLD",
]
