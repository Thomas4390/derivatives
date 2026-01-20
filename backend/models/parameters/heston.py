"""
Heston Parameter Container
==========================

Immutable parameters for Heston (1993) stochastic volatility model.

Model:
    dS = mu * S * dt + sqrt(V) * S * dW_S
    dV = kappa * (theta - V) * dt + xi * sqrt(V) * dW_V
    Corr(dW_S, dW_V) = rho

Author: Derivatives Pricing Project
"""

from dataclasses import dataclass
from typing import Dict, Any
from .base import BaseParams


@dataclass(frozen=True)
class HestonParams(BaseParams):
    """
    Heston stochastic volatility model parameters.

    Model:
        dS = mu * S * dt + sqrt(V) * S * dW_S
        dV = kappa * (theta - V) * dt + xi * sqrt(V) * dW_V
        Corr(dW_S, dW_V) = rho

    Parameters
    ----------
    v0 : float
        Initial variance (sigma^2), e.g., 0.04 for 20% initial vol
    kappa : float
        Mean reversion speed of variance (typical: 1-5)
    theta : float
        Long-run variance level (e.g., 0.04 for 20% long-run vol)
    xi : float
        Volatility of volatility (vol-of-vol)
    rho : float
        Correlation between price and variance (-1 to 1)

    Notes
    -----
    - Feller condition: 2*kappa*theta > xi^2 ensures V stays positive
    - Typical equity: rho < 0 (leverage effect)
    """
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def _validate(self) -> None:
        if self.v0 < 0:
            raise ValueError(f"v0 must be non-negative, got {self.v0}")
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.theta < 0:
            raise ValueError(f"theta must be non-negative, got {self.theta}")
        if self.xi <= 0:
            raise ValueError(f"xi must be positive, got {self.xi}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")

    @property
    def feller_satisfied(self) -> bool:
        """
        Check if Feller condition 2*kappa*theta > xi^2 is satisfied.

        When satisfied, variance process stays strictly positive.
        """
        return 2 * self.kappa * self.theta > self.xi ** 2

    @property
    def long_run_volatility(self) -> float:
        """Long-run volatility sqrt(theta)."""
        return self.theta ** 0.5

    @property
    def initial_volatility(self) -> float:
        """Initial volatility sqrt(v0)."""
        return self.v0 ** 0.5
