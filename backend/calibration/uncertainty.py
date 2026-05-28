"""
Parameter Uncertainty Quantification
=====================================

Industry-standard estimators for the covariance matrix of calibrated
parameters, used to report standard errors and confidence intervals
alongside the point estimate in ``CalibrationResult.diagnostics``.

Methods
-------
least_squares_covariance(jac, residuals)
    Gauss-Newton approximation for non-linear least-squares fits
    (Heston, Merton, Bates calibration to an option surface).
    Cov ~ sigma^2 * (J^T J)^{-1}  where  sigma^2 = RSS / (n - p).

bhhh_covariance(scores)
    Berndt-Hall-Hall-Hausman estimator for maximum likelihood
    (GARCH family). Cov ~ (sum_t s_t s_t^T)^{-1}.

Utilities
---------
std_errors_from_cov, confidence_intervals, summary_table:
    Format outputs for diagnostics dicts.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UncertaintySummary:
    """Container for calibration uncertainty diagnostics."""

    std_errors: np.ndarray  # sqrt(diag(cov))
    cov_matrix: np.ndarray  # p x p
    correlation_matrix: np.ndarray  # p x p
    condition_number: float  # cond(J^T J) or cond(sum s_t s_t^T)
    method: str  # "gauss-newton" | "bhhh"

    def ci(self, alpha: float = 0.05) -> np.ndarray:
        """Return symmetric Wald intervals: (p, 2) array of (lo, hi) offsets.

        Callers add these to the point estimate to get intervals.
        """
        from scipy.stats import norm

        z = float(norm.ppf(1.0 - alpha / 2.0))
        half = z * self.std_errors
        return np.stack([-half, +half], axis=1)


def least_squares_covariance(
    jac: np.ndarray,
    residuals: np.ndarray,
    regularize: float = 1e-12,
) -> UncertaintySummary:
    """Gauss-Newton covariance for non-linear least-squares.

    Parameters
    ----------
    jac : np.ndarray
        Jacobian matrix, shape (n_obs, n_params), evaluated at the
        optimum. Each row is the gradient of residual_i wrt theta.
    residuals : np.ndarray
        Residual vector at the optimum, shape (n_obs,).
    regularize : float
        Tikhonov ridge added to J^T J before inversion; guards against
        rank deficiency near the optimum.

    Returns
    -------
    UncertaintySummary
        std_errors, cov matrix, correlation matrix, condition number.

    Notes
    -----
    Formula (Seber & Wild 2003, Ch. 2):
        sigma_hat^2 = RSS / (n_obs - n_params)
        Cov(theta_hat) = sigma_hat^2 * (J^T J)^{-1}

    Degrees-of-freedom corrections require n_obs > n_params.
    """
    jac = np.asarray(jac, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    n, p = jac.shape
    if n <= p:
        raise ValueError(
            f"need more observations ({n}) than parameters ({p}) "
            "for a valid least-squares covariance estimate"
        )

    rss = float(residuals @ residuals)
    sigma2 = rss / (n - p)

    jtj = jac.T @ jac + regularize * np.eye(p)
    try:
        jtj_inv = np.linalg.inv(jtj)
    except np.linalg.LinAlgError as exc:
        raise ValueError("J^T J is singular after regularization") from exc

    cov = sigma2 * jtj_inv
    return _summary(cov, jtj, method="gauss-newton")


def bhhh_covariance(
    scores: np.ndarray,
    regularize: float = 1e-12,
) -> UncertaintySummary:
    """Berndt-Hall-Hall-Hausman (outer-product-of-gradients) MLE covariance.

    Parameters
    ----------
    scores : np.ndarray
        Per-observation score matrix, shape (T, n_params), where
        row t is  s_t = d log L_t / d theta  at the MLE.
    regularize : float
        Small ridge added to the outer product before inversion.

    Returns
    -------
    UncertaintySummary

    Notes
    -----
    BHHH estimate (Berndt et al. 1974):
        I_BHHH = sum_t s_t s_t^T
        Cov(theta_hat) = I_BHHH^{-1}

    BHHH is the fastest asymptotic covariance to compute (no Hessian),
    and is typically used for GARCH family estimation.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D (T, p), got shape {scores.shape}")
    T, p = scores.shape
    if T <= p:
        raise ValueError(f"need T>{p}, got T={T}")

    info = scores.T @ scores + regularize * np.eye(p)
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError as exc:
        raise ValueError("BHHH information matrix is singular") from exc

    return _summary(cov, info, method="bhhh")


def _summary(cov: np.ndarray, info_like: np.ndarray, method: str) -> UncertaintySummary:
    """Wrap a cov matrix into a full UncertaintySummary."""
    diag = np.diag(cov)
    # Numerical guard: negative diagonals can appear under extreme ill-conditioning
    std_errors = np.sqrt(np.maximum(diag, 0.0))

    p = cov.shape[0]
    corr = np.zeros_like(cov)
    for i in range(p):
        for j in range(p):
            denom = std_errors[i] * std_errors[j]
            corr[i, j] = cov[i, j] / denom if denom > 0 else 0.0

    try:
        cond = float(np.linalg.cond(info_like))
    except np.linalg.LinAlgError:
        cond = float("inf")

    return UncertaintySummary(
        std_errors=std_errors,
        cov_matrix=cov,
        correlation_matrix=corr,
        condition_number=cond,
        method=method,
    )


def summary_table(
    param_names: list[str],
    estimates: np.ndarray,
    summary: UncertaintySummary,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Return a flat dict with point estimates, std errors, and CIs.

    Intended to be stored in ``CalibrationResult.diagnostics["uncertainty"]``.
    """
    ci_offsets = summary.ci(alpha=alpha)
    return {
        name: {
            "estimate": float(estimates[i]),
            "std_error": float(summary.std_errors[i]),
            "ci_lo": float(estimates[i] + ci_offsets[i, 0]),
            "ci_hi": float(estimates[i] + ci_offsets[i, 1]),
        }
        for i, name in enumerate(param_names)
    }
