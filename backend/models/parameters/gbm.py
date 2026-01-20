"""
GBM Parameter Container
=======================

Immutable parameters for Geometric Brownian Motion model.

Model:
    dS = mu * S * dt + sigma * S * dW

Author: Derivatives Pricing Project
"""

from dataclasses import dataclass
from typing import Dict, Any
from .base import BaseParams


@dataclass(frozen=True)
class GBMParams(BaseParams):
    """
    Geometric Brownian Motion model parameters.

    Model:
        dS = mu * S * dt + sigma * S * dW

    Parameters
    ----------
    sigma : float
        Volatility (annualized), e.g., 0.20 for 20%

    Notes
    -----
    - sigma > 0 is required
    - mu (drift) is provided at simulation time, not stored here
    - For risk-neutral pricing, use r (risk-free rate) as drift
    """
    sigma: float

    def _validate(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

    @property
    def variance(self) -> float:
        """Annualized variance sigma^2."""
        return self.sigma ** 2
