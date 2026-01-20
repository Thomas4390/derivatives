"""
Merton Jump-Diffusion Parameter Container
==========================================

Immutable parameters for Merton (1976) jump-diffusion model.

Model:
    dS = (mu - lambda*k) * S * dt + sigma * S * dW + (J - 1) * S * dN

Where:
    - dN is Poisson with intensity lambda
    - J is lognormal: ln(J) ~ N(mu_j, sigma_j^2)
    - k = E[J - 1] = exp(mu_j + 0.5*sigma_j^2) - 1

Author: Derivatives Pricing Project
"""

from dataclasses import dataclass
import numpy as np
from .base import BaseParams


@dataclass(frozen=True)
class JumpParams(BaseParams):
    """
    Jump component parameters for Merton/Bates models.

    Parameters
    ----------
    lambda_j : float
        Jump intensity (expected jumps per year)
    mu_j : float
        Mean of log-jump size
    sigma_j : float
        Standard deviation of log-jump size

    Notes
    -----
    - lambda_j >= 0 (no jumps if 0)
    - mu_j < 0 for downward jumps (typical for equities)
    - sigma_j >= 0
    """
    lambda_j: float
    mu_j: float
    sigma_j: float

    def _validate(self) -> None:
        if self.lambda_j < 0:
            raise ValueError(f"lambda_j must be non-negative, got {self.lambda_j}")
        if self.sigma_j < 0:
            raise ValueError(f"sigma_j must be non-negative, got {self.sigma_j}")

    @property
    def expected_jump_size(self) -> float:
        """E[J - 1], expected percentage jump."""
        return np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1

    @property
    def jump_compensator(self) -> float:
        """lambda * k, the drift adjustment for jump risk."""
        return self.lambda_j * self.expected_jump_size


@dataclass(frozen=True)
class MertonParams(BaseParams):
    """
    Merton jump-diffusion model parameters.

    Combines constant volatility with jump component.

    Parameters
    ----------
    sigma : float
        Diffusion volatility (annualized)
    lambda_j : float
        Jump intensity (expected jumps per year)
    mu_j : float
        Mean of log-jump size
    sigma_j : float
        Standard deviation of log-jump size
    """
    sigma: float
    lambda_j: float
    mu_j: float
    sigma_j: float

    def _validate(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.lambda_j < 0:
            raise ValueError(f"lambda_j must be non-negative, got {self.lambda_j}")
        if self.sigma_j < 0:
            raise ValueError(f"sigma_j must be non-negative, got {self.sigma_j}")

    @property
    def jump_params(self) -> JumpParams:
        """Extract jump parameters as JumpParams object."""
        return JumpParams(
            lambda_j=self.lambda_j,
            mu_j=self.mu_j,
            sigma_j=self.sigma_j
        )

    @property
    def expected_jump_size(self) -> float:
        """E[J - 1], expected percentage jump."""
        return np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1

    @property
    def variance(self) -> float:
        """Diffusion variance sigma^2."""
        return self.sigma ** 2
