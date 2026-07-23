"""
Calibration Module
==================

Model calibration to market data for all supported stochastic models.

All calibrators use JAX for analytical Jacobians / gradients and
produce parameter standard errors alongside the point estimate.

Calibrators:
    - ImpliedVolCalibrator : GBM sigma from option prices (IV inversion)
    - HestonCalibrator     : 5-param stochastic vol — LM + multi-start
    - MertonCalibrator     : 4-param jump-diffusion — LM + Tikhonov on sigma
    - BatesCalibrator      : 8-param SV+jumps — semi-sequential + joint LM
    - GARCHCalibrator      : GARCH/NGARCH/GJR-GARCH MLE — exact JAX gradient

Every option-surface calibrator reports Gauss-Newton standard errors
at the optimum; GARCH reports BHHH (outer product of scores).

Shared infrastructure:
    - transforms.py     : Sigmoid/softplus reparametrization
    - pricing_loop.py   : Factored FFT surface-pricing loop
    - uncertainty.py    : Gauss-Newton LSQ + BHHH MLE covariance

See ``docs/calibration/CHANGELOG.md`` for the history of the 2026-04
migration from the legacy scipy-based calibrators to the JAX stack.

Author: Thomas Vaudescal
Created: 2026
"""

from backend.calibration.base import BaseCalibrator, CalibrationResult
from backend.calibration.feller import (
    DEFAULT_FELLER_WEIGHT,
    FellerMode,
    feller_capped_alpha,
    feller_alpha_to_unit,
    penalty_weight,
)
from backend.calibration.stationarity import (
    DEFAULT_STATIONARITY_WEIGHT,
    StationarityMode,
    ngarch_capped_gamma,
    stationarity_capped_gamma,
    stationarity_gamma_to_unit,
)
from backend.calibration.market_data import (
    HistoricalReturns,
    OptionMarketData,
    OptionQuote,
)
from backend.calibration.implied_vol import ImpliedVolCalibrator
from backend.calibration.heston_calibrator import HestonCalibrator
from backend.calibration.heston_nandi_calibrator import HestonNandiGARCHCalibrator
from backend.calibration.merton_calibrator import MertonCalibrator
from backend.calibration.bates_calibrator import BatesCalibrator
from backend.calibration.garch_calibrator import GARCHCalibrator
from backend.calibration.ngarch_q_calibrator import (
    GARCHRiskNeutralCalibrator,
    NGARCHRiskNeutralCalibrator,
)
from backend.calibration.custom_calibrator import (
    CustomModelCalibrator,
    CustomTerminalSimulator,
)

# Shared infrastructure
from backend.calibration.transforms import (
    IdentityTransform,
    ParameterTransform,
    SigmoidTransform,
    SoftplusTransform,
    bates_transform,
    heston_transform,
    merton_transform,
)
from backend.calibration.pricing_loop import (
    price_residuals,
    price_surface,
    price_surface_safe,
)
from backend.calibration.uncertainty import (
    UncertaintySummary,
    bhhh_covariance,
    least_squares_covariance,
    summary_table,
)

# Optimizer strategies (Strategy Pattern)
from backend.calibration.optimizers import (
    CalibrationProblem,
    DEFAULT_STRATEGIES,
    DifferentialEvolutionStrategy,
    IterationLogger,
    IterationSnapshot,
    LBFGSStrategy,
    LMJaxStrategy,
    NelderMeadStrategy,
    OptimizationResult,
    OptimizerStrategy,
    StrategyMetadata,
    make_strategy,
)

# Objective strategies (Strategy Pattern — mirrors optimizers)
from backend.calibration.objectives import (
    DEFAULT_OBJECTIVES,
    HuberObjective,
    IVMSEObjective,
    LEGACY_OBJECTIVE_ALIASES,
    ObjectiveMetadata,
    ObjectiveStrategy,
    PriceMSEObjective,
    RelativeMSEObjective,
    SpreadWeightedObjective,
    VegaWeightedObjective,
    make_objective,
)

__all__ = [
    # Base / data
    "BaseCalibrator",
    "CalibrationResult",
    "HistoricalReturns",
    "OptionMarketData",
    "OptionQuote",
    # Feller-condition control
    "FellerMode",
    "DEFAULT_FELLER_WEIGHT",
    "feller_capped_alpha",
    "feller_alpha_to_unit",
    "penalty_weight",
    # Stationarity control (Heston-Nandi GARCH)
    "StationarityMode",
    "DEFAULT_STATIONARITY_WEIGHT",
    "stationarity_capped_gamma",
    "stationarity_gamma_to_unit",
    # Calibrators
    "ImpliedVolCalibrator",
    "HestonCalibrator",
    "HestonNandiGARCHCalibrator",
    "MertonCalibrator",
    "BatesCalibrator",
    "GARCHCalibrator",
    "GARCHRiskNeutralCalibrator",
    "NGARCHRiskNeutralCalibrator",
    "CustomModelCalibrator",
    "CustomTerminalSimulator",
    # Shared infrastructure
    "IdentityTransform",
    "ParameterTransform",
    "SigmoidTransform",
    "SoftplusTransform",
    "heston_transform",
    "merton_transform",
    "bates_transform",
    "price_surface",
    "price_surface_safe",
    "price_residuals",
    "UncertaintySummary",
    "least_squares_covariance",
    "bhhh_covariance",
    "summary_table",
    # Optimizer strategies
    "CalibrationProblem",
    "DEFAULT_STRATEGIES",
    "DifferentialEvolutionStrategy",
    "IterationLogger",
    "IterationSnapshot",
    "LBFGSStrategy",
    "LMJaxStrategy",
    "NelderMeadStrategy",
    "OptimizationResult",
    "OptimizerStrategy",
    "StrategyMetadata",
    "make_strategy",
    # Objective strategies
    "DEFAULT_OBJECTIVES",
    "HuberObjective",
    "IVMSEObjective",
    "LEGACY_OBJECTIVE_ALIASES",
    "ObjectiveMetadata",
    "ObjectiveStrategy",
    "PriceMSEObjective",
    "RelativeMSEObjective",
    "SpreadWeightedObjective",
    "VegaWeightedObjective",
    "make_objective",
]
