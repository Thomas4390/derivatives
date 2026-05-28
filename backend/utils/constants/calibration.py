"""
Calibration Constants
======================

Parameter bounds and names for all supported stochastic models.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import Final

# =============================================================================
# Heston Model
# =============================================================================

HESTON_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "v0": (0.001, 1.0),  # initial variance
    "kappa": (0.01, 20.0),  # mean-reversion speed
    "theta": (0.001, 1.0),  # long-run variance
    "alpha": (0.01, 3.0),  # vol-of-vol
    "rho": (-0.99, 0.99),  # spot-variance correlation
}

HESTON_PARAM_NAMES: Final[tuple[str, ...]] = ("v0", "kappa", "theta", "alpha", "rho")

# =============================================================================
# Bates Model (Heston + Jumps)
# =============================================================================

BATES_BOUNDS: Final[tuple[tuple[float, float], ...]] = (
    # Heston params
    (0.001, 1.0),  # v0    - initial variance
    (0.01, 20.0),  # kappa - mean-reversion speed
    (0.001, 1.0),  # theta - long-run variance
    (0.01, 2.0),  # alpha    - vol-of-vol
    (-0.99, 0.99),  # rho   - correlation
    # Jump params
    (0.001, 5.0),  # lam - jump intensity
    (-0.5, 0.1),  # alpha_j    - mean log-jump size
    (0.01, 0.5),  # sigma_j - volatility of log-jump size
)

BATES_PARAM_NAMES: Final[tuple[str, ...]] = (
    "v0",
    "kappa",
    "theta",
    "alpha",
    "rho",
    "lam",
    "alpha_j",
    "sigma_j",
)

JUMP_BOUNDS: Final[tuple[tuple[float, float], ...]] = (
    (0.001, 5.0),  # lam
    (-0.5, 0.1),  # alpha_j
    (0.01, 0.5),  # sigma_j
)

JUMP_PARAM_NAMES: Final[tuple[str, ...]] = ("lam", "alpha_j", "sigma_j")

# =============================================================================
# Merton Jump-Diffusion Model
# =============================================================================

MERTON_BOUNDS: Final[tuple[tuple[float, float], ...]] = (
    (0.01, 1.0),  # sigma   - diffusion volatility
    (0.001, 5.0),  # lam - jump intensity
    (-0.5, 0.1),  # alpha_j    - mean log-jump size
    (0.01, 0.5),  # sigma_j - volatility of log-jump size
)

MERTON_PARAM_NAMES: Final[tuple[str, ...]] = (
    "sigma",
    "lam",
    "alpha_j",
    "sigma_j",
)

# =============================================================================
# GARCH Family
# =============================================================================

GARCH_BOUNDS: Final[tuple[tuple[float, float], ...]] = (
    (1e-10, 0.01),  # omega - constant term
    (1e-6, 0.3),  # alpha - ARCH coefficient (tightened: keep alpha+beta < 1)
    (0.001, 0.98),  # beta  - GARCH persistence (tightened: keep alpha+beta < 1)
)

NGARCH_BOUNDS: Final[tuple[tuple[float, float], ...]] = GARCH_BOUNDS + (
    (0.0, 5.0),  # gamma - leverage parameter
)

GJR_BOUNDS: Final[tuple[tuple[float, float], ...]] = GARCH_BOUNDS + (
    (0.0, 0.5),  # gamma - asymmetry parameter
)

# =============================================================================
# Heston-Nandi GARCH (2000) — risk-neutral, option-surface calibration
# =============================================================================

# Per-period (daily) parameter box for the risk-neutral Heston-Nandi GARCH
# option-pricing model. Unlike the physical-measure GARCH family above, this
# model is calibrated to an option SURFACE (not a return series): it has a
# closed-form discrete characteristic function and prices via Carr-Madan FFT.
# gamma is O(100) because it multiplies sqrt(h_t) ~ vol / sqrt(252).
HESTON_NANDI_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "omega": (1e-9, 1e-4),  # variance-recursion intercept (per period)
    "alpha": (1e-9, 1e-3),  # ARCH coefficient
    "beta": (0.0, 0.999),  # GARCH persistence
    "gamma": (0.0, 1000.0),  # risk-neutral leverage (large in HN-GARCH)
    "h0": (1e-7, 1e-2),  # initial conditional variance (per period)
}

HESTON_NANDI_PARAM_NAMES: Final[tuple[str, ...]] = (
    "omega",
    "alpha",
    "beta",
    "gamma",
    "h0",
)

# Discretization: trading days per year. Heston-Nandi is a discrete-time model;
# the characteristic-function recursion runs N = round(tau * steps_per_year)
# steps with a per-step risk-free rate r / steps_per_year.
HESTON_NANDI_STEPS_PER_YEAR: Final[int] = 252

# =============================================================================
# Duan NGARCH (risk-neutral, nonaffine) — option-surface calibration via MC
# =============================================================================

# Per-period (daily) parameter box for the risk-neutral nonaffine NGARCH option
# model (Duan 1995). Unlike Heston-Nandi (affine, gamma O(100) because it
# multiplies sqrt(h_t)), here the asymmetry enters as (z - gamma)^2 with z a
# standard normal, so gamma is O(1). Persistence is beta + alpha*(1 + gamma^2).
# omega/alpha/beta share GARCH_BOUNDS — under Duan's LRNVR the one-step-ahead
# conditional variance is invariant P -> Q, so the optimiser searches the same
# box for both families. Only gamma (and h0, not present on P) are Q-specific.
NGARCH_Q_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "omega": (1e-10, 0.01),  # variance-recursion intercept (per period) — aligned with GARCH_BOUNDS
    "alpha": (1e-6, 0.3),  # ARCH coefficient — aligned with GARCH_BOUNDS
    "beta": (0.001, 0.98),  # GARCH persistence — aligned with GARCH_BOUNDS
    "gamma": (0.0, 4.0),  # risk-neutral asymmetry (gamma* = gamma_P + lambda)
    "h0": (1e-7, 1e-2),  # initial conditional variance (per period)
}

NGARCH_Q_PARAM_NAMES: Final[tuple[str, ...]] = (
    "omega",
    "alpha",
    "beta",
    "gamma",
    "h0",
)

# Sibling risk-neutral GARCH-Q bounds (same uniform 5-slot order as NGARCH-Q).
# GJR-Q: gamma is the leverage-indicator coefficient (γ ≥ 0). GARCH-Q is
# symmetric — gamma is pinned to 0 so the optimiser never moves it.
GJR_Q_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    **NGARCH_Q_BOUNDS,
    "gamma": (0.0, 2.0),
}
GARCH_Q_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    **NGARCH_Q_BOUNDS,
    "gamma": (0.0, 0.0),
}

# Dispatch table consumed by the generalised GARCHRiskNeutralCalibrator.
RISK_NEUTRAL_GARCH_BOUNDS: Final[dict[str, dict[str, tuple[float, float]]]] = {
    "garch": GARCH_Q_BOUNDS,
    "ngarch": NGARCH_Q_BOUNDS,
    "gjr_garch": GJR_Q_BOUNDS,
}

# =============================================================================
# Valid Calibration Objectives
# =============================================================================

VALID_CALIBRATION_OBJECTIVES: Final[frozenset[str]] = frozenset(
    {"price_rmse", "iv_rmse", "price_weighted"}
)

VALID_GARCH_TYPES: Final[tuple[str, ...]] = ("garch", "ngarch", "gjr_garch")

# =============================================================================
# Levenberg-Marquardt Optimizer Defaults
# =============================================================================

# Default maximum iterations for the LM loop used by every V2 calibrator.
LM_DEFAULT_MAX_ITER: Final[int] = 100

# Default LM damping parameter (lambda).
LM_DEFAULT_DAMPING: Final[float] = 1e-3

# Default LM convergence tolerance.
LM_DEFAULT_TOL: Final[float] = 1e-8
