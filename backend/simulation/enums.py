"""
Enumeration Types for Simulation Module
=======================================

Centralized enums for type-safe model and parameter selection.
Using enums instead of strings/integers improves code clarity,
enables IDE autocomplete, and catches errors at development time.

Usage:
------
    from backend.simulation.enums import (
        HestonScheme,
        OptionType,
        PositionType,
        PriceModel,
        VolatilityModel,
    )

    # Use in function calls
    scheme = HestonScheme.QE
    option = OptionType.CALL
    position = PositionType.LONG
"""

from enum import Enum, IntEnum


# =============================================================================
# DISCRETIZATION SCHEMES
# =============================================================================

class HestonScheme(IntEnum):
    """
    Heston model variance discretization schemes.

    Different schemes handle the potential negative variance issue:

    - EULER: Simple Euler-Maruyama, can produce negative variance
    - FULL_TRUNCATION: Floor variance at 0, most common choice
    - REFLECTION: Reflect negative variance to positive
    - QE: Quadratic-Exponential, most accurate but complex

    Reference:
        Andersen, L. (2008). "Simple and efficient simulation of the
        Heston stochastic volatility model"
    """
    EULER = 0
    FULL_TRUNCATION = 1
    REFLECTION = 2
    QE = 3  # Quadratic-Exponential


# =============================================================================
# OPTION TYPES
# =============================================================================

class OptionType(Enum):
    """
    Option contract types.

    Used in P&L calculations and strategy building.
    """
    CALL = "call"
    PUT = "put"

    @property
    def numeric_value(self) -> int:
        """Return numeric value for Numba functions (1 = call, -1 = put)."""
        return 1 if self == OptionType.CALL else -1


class PositionType(Enum):
    """
    Trading position types.

    - LONG: Bought the option (pay premium, receive payoff)
    - SHORT: Sold the option (receive premium, pay payoff)
    """
    LONG = "long"
    SHORT = "short"

    @property
    def numeric_value(self) -> int:
        """Return numeric value for Numba functions (1 = long, -1 = short)."""
        return 1 if self == PositionType.LONG else -1


# =============================================================================
# PRICE MODELS
# =============================================================================

class PriceModel(Enum):
    """
    Available price path simulation models.

    Each model captures different market dynamics:

    - GBM: Constant volatility, log-normal prices (Black-Scholes assumption)
    - HESTON: Stochastic volatility with mean reversion
    - MERTON: Jump diffusion for sudden price movements
    - BATES: Combines Heston (stochastic vol) + Merton (jumps)
    - SABR: Popular for interest rate derivatives
    """
    GBM = "gbm"
    HESTON = "heston"
    MERTON = "merton"
    BATES = "bates"
    SABR = "sabr"

    @property
    def display_name(self) -> str:
        """Human-readable model name."""
        names = {
            PriceModel.GBM: "Geometric Brownian Motion (GBM)",
            PriceModel.HESTON: "Heston Stochastic Volatility",
            PriceModel.MERTON: "Merton Jump Diffusion",
            PriceModel.BATES: "Bates Model (Heston + Jumps)",
            PriceModel.SABR: "SABR Model",
        }
        return names[self]


# =============================================================================
# VOLATILITY MODELS
# =============================================================================

class VolatilityModel(Enum):
    """
    Available volatility simulation models (GARCH family).

    All models capture volatility clustering and persistence:

    - GARCH: Symmetric response to shocks
    - NGARCH: Asymmetric response via shifted term (Engle & Ng)
    - GJR_GARCH: Asymmetric via indicator function (GJR)
    - EGARCH: Log-space model, no positivity constraints
    """
    GARCH = "garch"
    NGARCH = "ngarch"
    GJR_GARCH = "gjr_garch"
    EGARCH = "egarch"

    @property
    def display_name(self) -> str:
        """Human-readable model name."""
        names = {
            VolatilityModel.GARCH: "GARCH(1,1)",
            VolatilityModel.NGARCH: "NGARCH (NAGARCH)",
            VolatilityModel.GJR_GARCH: "GJR-GARCH",
            VolatilityModel.EGARCH: "EGARCH",
        }
        return names[self]


# =============================================================================
# SIMULATION MODES
# =============================================================================

class SimulationMode(Enum):
    """
    Application simulation modes.

    - PRICE: Simulate price paths only
    - VOLATILITY: Simulate volatility paths (GARCH models)
    - OPTION_PNL: Full option P&L analysis with risk metrics
    """
    PRICE = "price"
    VOLATILITY = "volatility"
    OPTION_PNL = "option_pnl"

    @property
    def display_name(self) -> str:
        """Human-readable mode name."""
        names = {
            SimulationMode.PRICE: "Price Path Simulation",
            SimulationMode.VOLATILITY: "Volatility Simulation",
            SimulationMode.OPTION_PNL: "Option P&L Analysis",
        }
        return names[self]


# =============================================================================
# RISK METRICS
# =============================================================================

class RiskMetricType(Enum):
    """
    Types of risk metrics computed from P&L distributions.
    """
    MEAN_PNL = "mean_pnl"
    STD_PNL = "std_pnl"
    VAR_95 = "var_95"
    VAR_99 = "var_99"
    CVAR_95 = "cvar_95"
    CVAR_99 = "cvar_99"
    PROB_PROFIT = "prob_profit"
    MAX_PROFIT = "max_profit"
    MAX_LOSS = "max_loss"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"
