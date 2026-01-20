"""
Bates Parameter Container
=========================

Immutable parameters for Bates (1996) model combining
Heston stochastic volatility with Merton-style jumps.

Model:
    dS = (mu - lambda*k) * S * dt + sqrt(V) * S * dW_S + (J - 1) * S * dN
    dV = kappa * (theta - V) * dt + xi * sqrt(V) * dW_V
    Corr(dW_S, dW_V) = rho

Author: Derivatives Pricing Project
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from .base import BaseParams
from .heston import HestonParams
from .merton import JumpParams


@dataclass(frozen=True)
class BatesParams(BaseParams):
    """
    Bates model parameters = Heston + Jumps.

    Model:
        dS = (mu - lambda*k) * S * dt + sqrt(V) * S * dW_S + (J - 1) * S * dN
        dV = kappa * (theta - V) * dt + xi * sqrt(V) * dW_V
        Corr(dW_S, dW_V) = rho

    Parameters
    ----------
    v0 : float
        Initial variance
    kappa : float
        Mean reversion speed
    theta : float
        Long-run variance
    xi : float
        Volatility of volatility
    rho : float
        Correlation
    lambda_j : float
        Jump intensity
    mu_j : float
        Mean of log-jump
    sigma_j : float
        Std of log-jump
    """
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    lambda_j: float
    mu_j: float
    sigma_j: float

    def _validate(self) -> None:
        # Heston constraints
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

        # Jump constraints
        if self.lambda_j < 0:
            raise ValueError(f"lambda_j must be non-negative, got {self.lambda_j}")
        if self.sigma_j < 0:
            raise ValueError(f"sigma_j must be non-negative, got {self.sigma_j}")

    @property
    def heston_params(self) -> HestonParams:
        """Extract Heston parameters as HestonParams object."""
        return HestonParams(
            v0=self.v0,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.xi,
            rho=self.rho
        )

    @property
    def jump_params(self) -> JumpParams:
        """Extract jump parameters as JumpParams object."""
        return JumpParams(
            lambda_j=self.lambda_j,
            mu_j=self.mu_j,
            sigma_j=self.sigma_j
        )

    @property
    def feller_satisfied(self) -> bool:
        """Check if Feller condition 2*kappa*theta > xi^2 is satisfied."""
        return 2 * self.kappa * self.theta > self.xi ** 2

    @property
    def expected_jump_size(self) -> float:
        """E[J - 1], expected percentage jump."""
        return np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1

    @property
    def long_run_volatility(self) -> float:
        """Long-run volatility sqrt(theta)."""
        return self.theta ** 0.5

    @classmethod
    def from_components(cls, heston: HestonParams, jumps: JumpParams) -> "BatesParams":
        """Create BatesParams from Heston and Jump parameter objects."""
        return cls(
            v0=heston.v0,
            kappa=heston.kappa,
            theta=heston.theta,
            xi=heston.xi,
            rho=heston.rho,
            lambda_j=jumps.lambda_j,
            mu_j=jumps.mu_j,
            sigma_j=jumps.sigma_j
        )
