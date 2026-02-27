"""
Merton Jump-Diffusion Model
===========================

Merton (1976) jump-diffusion model.

Model:
    dS = (r - q - lambda_j * k) * S * dt + sigma * S * dW + (J - 1) * S * dN

Where:
    - dN is a Poisson process with intensity lambda_j
    - J is lognormal: ln(J) ~ N(mu_j, sigma_j^2)
    - k = E[J - 1] = exp(mu_j + 0.5*sigma_j^2) - 1

The Merton model extends GBM with jumps, capturing fat tails and
crash risk better than pure diffusion models.

Author: Thomas
Created: 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from backend.models.gbm import GBMModel

from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability
from backend.models.characteristic_functions.merton_cf import (
    merton_characteristic_function,
    merton_cf_vectorized,
)


# =============================================================================
# MERTON MODEL
# =============================================================================

@dataclass(frozen=True)
class MertonModel(Model):
    """
    Merton (1976) Jump-Diffusion Model.

    Extends GBM with Poisson-driven lognormal jumps.
    Captures crash risk and fat tails.

    Model:
        dS = (r - q - lambda_j * k) * S * dt + sigma * S * dW + (J - 1) * S * dN

    Diffusion Parameters
    --------------------
    sigma : float
        Volatility (annualized), e.g., 0.20 for 20%

    Jump Parameters
    ---------------
    lambda_j : float
        Jump intensity (expected number of jumps per year)
    mu_j : float
        Mean of log-jump size (e.g., -0.1 for 10% negative mean jump)
    sigma_j : float
        Volatility of log-jump size

    Examples
    --------
    model = MertonModel(sigma=0.20, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)

    # Expected jump size
    model.expected_jump_size  # E[J - 1]

    # Characteristic function for FFT pricing
    cf = model.characteristic_function(u=1+0.5j, s0=100, t=0.5, r=0.05)

    Notes
    -----
    - When lambda_j = 0, reduces to GBM
    - Jumps add fat tails and negative skewness
    - The Merton formula has a semi-closed form solution as a
      weighted sum of Black-Scholes prices (but FFT is faster)
    """

    sigma: float
    lambda_j: float
    mu_j: float
    sigma_j: float

    def __post_init__(self):
        """Validate parameters."""
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.lambda_j < 0:
            raise ValueError(f"lambda_j must be non-negative, got {self.lambda_j}")
        if not -1.0 <= self.mu_j <= 1.0:
            raise ValueError(f"mu_j should be in [-1, 1], got {self.mu_j}")
        if self.sigma_j < 0:
            raise ValueError(f"sigma_j must be non-negative, got {self.sigma_j}")

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return "Merton Jump-Diffusion"

    @property
    def supported_engines(self) -> List[PricingCapability]:
        """Which pricing methods this model supports."""
        return [
            PricingCapability.FFT,
            PricingCapability.MONTE_CARLO,
        ]

    def get_parameters(self) -> Dict[str, Any]:
        """Return model parameters as dictionary."""
        return {
            "sigma": self.sigma,
            "lambda_j": self.lambda_j,
            "mu_j": self.mu_j,
            "sigma_j": self.sigma_j,
        }

    def characteristic_function(
        self, u: complex, s0: float, t: float, r: float, q: float = 0.0
    ) -> complex:
        """
        Merton characteristic function phi(u) = E^Q[exp(i*u*ln(S_T))].

        Parameters
        ----------
        u : complex
            Fourier transform variable
        s0 : float
            Initial spot price
        t : float
            Time to maturity
        r : float
            Risk-free rate
        q : float
            Dividend yield

        Returns
        -------
        complex
            Value of characteristic function at u
        """
        s0_adj = s0 * np.exp(-q * t)
        return merton_characteristic_function(
            u, s0_adj, t, r,
            self.sigma, self.lambda_j, self.mu_j, self.sigma_j
        )

    def characteristic_function_vectorized(
        self, u_arr: np.ndarray, s0: float, t: float, r: float, q: float = 0.0
    ) -> np.ndarray:
        """Vectorized characteristic function for FFT."""
        s0_adj = s0 * np.exp(-q * t)
        return merton_cf_vectorized(
            u_arr, s0_adj, t, r,
            self.sigma, self.lambda_j, self.mu_j, self.sigma_j
        )

    def drift(self, s: float, v: float, t: float, r: float, q: float) -> float:
        """
        Drift coefficient for SDE discretization.

        For Merton: drift = (r - q - lambda_j * k) * S
        where k = E[J - 1] is the expected jump size.

        Parameters
        ----------
        s : float
            Current spot price
        v : float
            Current variance (unused)
        t : float
            Current time (unused)
        r : float
            Risk-free rate
        q : float
            Dividend yield

        Returns
        -------
        float
            Drift value
        """
        k = self.expected_jump_size
        return (r - q - self.lambda_j * k) * s

    def diffusion(self, s: float, v: float, t: float) -> float:
        """
        Diffusion coefficient for SDE discretization.

        For Merton: diffusion = sigma * S

        Parameters
        ----------
        s : float
            Current spot price
        v : float
            Current variance (unused)
        t : float
            Current time (unused)

        Returns
        -------
        float
            Diffusion value
        """
        return self.sigma * s

    @property
    def expected_jump_size(self) -> float:
        """
        Expected relative jump size k = E[J - 1].

        For lognormal J: k = exp(mu_j + 0.5*sigma_j^2) - 1
        """
        return np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1

    @property
    def expected_jump_return(self) -> float:
        """Expected jump return as percentage."""
        return self.expected_jump_size * 100

    def expected_jumps_per_year(self) -> float:
        """Expected number of jumps per year."""
        return self.lambda_j

    @property
    def variance(self) -> float:
        """Annualized variance sigma^2."""
        return self.sigma ** 2

    def total_variance(self, t: float = 1.0) -> float:
        """
        Total variance over period t, including jump contribution.

        Var[ln(S_T/S_0)] = sigma^2 * t + lambda_j * t * (mu_j^2 + sigma_j^2)
        """
        diffusion_var = self.variance * t
        jump_var = self.lambda_j * t * (self.mu_j ** 2 + self.sigma_j ** 2)
        return diffusion_var + jump_var

    def total_volatility(self, t: float = 1.0) -> float:
        """Total annualized volatility including jumps."""
        return np.sqrt(self.total_variance(t) / t)

    def jump_contribution_to_variance(self) -> float:
        """
        Variance contribution from jumps (annualized).

        For Merton: lambda_j * (mu_j^2 + sigma_j^2)
        """
        return self.lambda_j * (self.mu_j ** 2 + self.sigma_j ** 2)

    def to_gbm(self) -> 'GBMModel':
        """
        Convert to GBM model (drop jumps).

        Returns
        -------
        GBMModel
            Equivalent GBM model without jumps
        """
        from backend.models.gbm import GBMModel
        return GBMModel(sigma=self.sigma)

    def create_simulator(self, **kwargs):
        """
        Create a Merton simulator for Monte Carlo pricing.

        Parameters
        ----------
        **kwargs
            Additional simulator parameters

        Returns
        -------
        MertonSimulator
            Configured simulator instance
        """
        from backend.simulation.models.merton import MertonSimulator
        return MertonSimulator(
            sigma=self.sigma,
            lambda_j=self.lambda_j,
            mu_j=self.mu_j,
            sigma_j=self.sigma_j,
            **kwargs
        )

    def __repr__(self) -> str:
        return (
            f"MertonModel(sigma={self.sigma}, lambda_j={self.lambda_j}, "
            f"mu_j={self.mu_j}, sigma_j={self.sigma_j})"
        )


if __name__ == "__main__":
    print("=" * 50)
    print("Merton Model Smoke Test")
    print("=" * 50)

    # Create model
    model = MertonModel(sigma=0.20, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)
    print(f"\nModel: {model}")
    print(f"Name: {model.name}")
    print(f"Supported engines: {model.supported_engines}")

    # Parameters
    print("\n--- Parameters ---")
    for k, v in model.get_parameters().items():
        print(f"  {k}: {v}")

    # Jump characteristics
    print("\n--- Jump Characteristics ---")
    print(f"Expected jumps/year: {model.expected_jumps_per_year()}")
    print(f"Expected jump size: {model.expected_jump_size:.2%}")
    print(f"Jump variance contribution: {model.jump_contribution_to_variance():.4f}")

    # Variance decomposition
    print("\n--- Variance Decomposition ---")
    print(f"Diffusion variance: {model.variance:.4f}")
    print(f"Jump variance: {model.jump_contribution_to_variance():.4f}")
    print(f"Total variance: {model.total_variance():.4f}")
    print(f"Total volatility: {model.total_volatility():.1%}")

    # Test characteristic function
    print("\n--- Characteristic Function ---")
    s0, t, r, q = 100.0, 0.5, 0.05, 0.02
    u = 1.0 + 0.5j
    cf = model.characteristic_function(u, s0, t, r, q)
    print(f"phi({u}) = {cf}")
    print(f"|phi| = {abs(cf):.6f}")

    # Test vectorized CF
    u_arr = np.array([0.5, 1.0, 1.5]) + 0.5j
    cf_vec = model.characteristic_function_vectorized(u_arr, s0, t, r, q)
    print(f"Vectorized CF: {np.abs(cf_vec)}")

    # Test SDE coefficients
    print("\n--- SDE Coefficients ---")
    drift = model.drift(s=100, v=0, t=0, r=0.05, q=0.02)
    diff = model.diffusion(s=100, v=0, t=0)
    print(f"Drift at S=100: {drift:.2f}")
    print(f"Diffusion at S=100: {diff:.2f}")

    # Compare with GBM
    print("\n--- Comparison with GBM ---")
    gbm = model.to_gbm()
    cf_gbm = gbm.characteristic_function(u, s0, t, r, q)
    print(f"Merton CF: {abs(cf):.6f}")
    print(f"GBM CF: {abs(cf_gbm):.6f}")
    print(f"Difference (jump effect): {abs(abs(cf) - abs(cf_gbm)):.6f}")

    # Test immutability
    print("\n--- Immutability Test ---")
    try:
        model.sigma = 0.30  # type: ignore
        print("ERROR: Mutation should have failed!")
    except Exception as e:
        print(f"Correctly prevented mutation: {type(e).__name__}")

    # Test validation
    print("\n--- Validation Test ---")
    try:
        bad_model = MertonModel(sigma=-0.1, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"Correctly rejected invalid sigma: {e}")

    print("\n" + "=" * 50)
    print("Merton smoke test passed")
    print("=" * 50)
