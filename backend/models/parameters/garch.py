"""
GARCH Family Parameter Containers
=================================

Immutable parameters for GARCH(1,1), NGARCH, and GJR-GARCH models.

Models:
    GARCH(1,1):   sigma^2_t = omega + alpha * sigma^2_{t-1} * z^2_{t-1} + beta * sigma^2_{t-1}
    NGARCH:       sigma^2_t = omega + alpha * sigma^2_{t-1} * (z_{t-1} - theta)^2 + beta * sigma^2_{t-1}
    GJR-GARCH:    sigma^2_t = omega + (alpha + gamma * I_{t-1}) * sigma^2_{t-1} * z^2_{t-1} + beta * sigma^2_{t-1}

Author: Derivatives Pricing Project
"""

from dataclasses import dataclass
import numpy as np
from .base import BaseParams


@dataclass(frozen=True)
class GARCHParams(BaseParams):
    """
    GARCH(1,1) model parameters.

    Model:
        sigma^2_t = omega + alpha * sigma^2_{t-1} * z^2_{t-1} + beta * sigma^2_{t-1}

    Parameters
    ----------
    sigma0 : float
        Initial volatility (annualized)
    omega : float
        Constant term in variance equation (omega > 0)
    alpha : float
        ARCH coefficient - reaction to past shocks (alpha >= 0)
    beta : float
        GARCH coefficient - persistence (beta >= 0)

    Notes
    -----
    - Stationarity requires: alpha + beta < 1
    - Long-run variance: omega / (1 - alpha - beta)
    - Typical values: alpha ~ 0.05-0.10, beta ~ 0.85-0.95
    """
    sigma0: float
    omega: float
    alpha: float
    beta: float

    def _validate(self) -> None:
        if self.sigma0 <= 0:
            raise ValueError(f"sigma0 must be positive, got {self.sigma0}")
        if self.omega <= 0:
            raise ValueError(f"omega must be positive, got {self.omega}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")

        persistence = self.alpha + self.beta
        if persistence >= 1:
            raise ValueError(
                f"Process is not stationary: alpha + beta = {persistence:.4f} >= 1. "
                f"Reduce alpha or beta."
            )

    @property
    def persistence(self) -> float:
        """Returns alpha + beta, the persistence of shocks."""
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        """Returns omega / (1 - alpha - beta)."""
        return self.omega / (1 - self.persistence)

    @property
    def long_run_volatility(self) -> float:
        """Returns sqrt of long-run variance."""
        return np.sqrt(self.long_run_variance)

    @property
    def half_life(self) -> float:
        """Half-life of variance shocks in time steps."""
        if self.persistence <= 0 or self.persistence >= 1:
            return np.inf
        return np.log(2) / (-np.log(self.persistence))


@dataclass(frozen=True)
class NGARCHParams(BaseParams):
    """
    NGARCH (Nonlinear Asymmetric GARCH) model parameters.

    Model:
        sigma^2_t = omega + alpha * sigma^2_{t-1} * (z_{t-1} - theta)^2 + beta * sigma^2_{t-1}

    Parameters
    ----------
    sigma0 : float
        Initial volatility
    omega : float
        Constant term
    alpha : float
        ARCH coefficient
    beta : float
        GARCH coefficient
    theta : float
        Leverage parameter (theta > 0 for leverage effect)

    Notes
    -----
    - Stationarity requires: alpha * (1 + theta^2) + beta < 1
    - theta > 0: Bad news increases volatility more than good news
    - theta = 0: Reduces to GARCH(1,1)
    """
    sigma0: float
    omega: float
    alpha: float
    beta: float
    theta: float

    def _validate(self) -> None:
        if self.sigma0 <= 0:
            raise ValueError(f"sigma0 must be positive, got {self.sigma0}")
        if self.omega <= 0:
            raise ValueError(f"omega must be positive, got {self.omega}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")

        persistence = self.alpha * (1 + self.theta ** 2) + self.beta
        if persistence >= 1:
            raise ValueError(
                f"Process is not stationary: alpha*(1+theta^2) + beta = {persistence:.4f} >= 1. "
                f"Reduce alpha, beta, or theta."
            )

    @property
    def persistence(self) -> float:
        """Returns alpha * (1 + theta^2) + beta."""
        return self.alpha * (1 + self.theta ** 2) + self.beta

    @property
    def long_run_variance(self) -> float:
        """Returns omega / (1 - persistence)."""
        return self.omega / (1 - self.persistence)

    @property
    def long_run_volatility(self) -> float:
        """Returns sqrt of long-run variance."""
        return np.sqrt(self.long_run_variance)


@dataclass(frozen=True)
class GJRGARCHParams(BaseParams):
    """
    GJR-GARCH model parameters.

    Model:
        sigma^2_t = omega + (alpha + gamma * I_{t-1}) * sigma^2_{t-1} * z^2_{t-1} + beta * sigma^2_{t-1}

    Where I_{t-1} = 1 if z_{t-1} < 0 (negative return indicator).

    Parameters
    ----------
    sigma0 : float
        Initial volatility
    omega : float
        Constant term
    alpha : float
        ARCH coefficient
    beta : float
        GARCH coefficient
    gamma : float
        Asymmetry coefficient (gamma > 0 for leverage effect)

    Notes
    -----
    - Stationarity requires: alpha + 0.5*gamma + beta < 1
    - gamma > 0: Negative returns add extra volatility
    - gamma = 0: Reduces to GARCH(1,1)
    """
    sigma0: float
    omega: float
    alpha: float
    beta: float
    gamma: float

    def _validate(self) -> None:
        if self.sigma0 <= 0:
            raise ValueError(f"sigma0 must be positive, got {self.sigma0}")
        if self.omega <= 0:
            raise ValueError(f"omega must be positive, got {self.omega}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {self.gamma}")

        persistence = self.alpha + 0.5 * self.gamma + self.beta
        if persistence >= 1:
            raise ValueError(
                f"Process is not stationary: alpha + 0.5*gamma + beta = {persistence:.4f} >= 1. "
                f"Reduce alpha, beta, or gamma."
            )

    @property
    def persistence(self) -> float:
        """Returns alpha + 0.5*gamma + beta."""
        return self.alpha + 0.5 * self.gamma + self.beta

    @property
    def long_run_variance(self) -> float:
        """Returns omega / (1 - persistence)."""
        return self.omega / (1 - self.persistence)

    @property
    def long_run_volatility(self) -> float:
        """Returns sqrt of long-run variance."""
        return np.sqrt(self.long_run_variance)
