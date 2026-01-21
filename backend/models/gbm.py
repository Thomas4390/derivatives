"""
GBM Model
=========

Geometric Brownian Motion (Black-Scholes) model.

Model:
    dS = (r - q) * S * dt + sigma * S * dW

The most fundamental model in option pricing. Under risk-neutral measure,
drift equals r - q (risk-free rate minus dividend yield).

Author: Thomas
Created: 2025
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from numba import njit

from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability


# =============================================================================
# NUMBA KERNELS (Hot Path)
# =============================================================================

@njit(cache=True, fastmath=True)
def _gbm_characteristic_function(
    u: complex,
    s0: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
) -> complex:
    """
    GBM characteristic function phi(u) = E^Q[exp(i*u*ln(S_T))].

    For GBM under risk-neutral measure:
        phi(u) = exp(i*u*(ln(s0) + (r - q - 0.5*sigma^2)*t) - 0.5*sigma^2*t*u^2)

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
    sigma : float
        Volatility

    Returns
    -------
    complex
        Characteristic function value
    """
    i = 1j
    log_s0 = np.log(s0)
    drift = r - q - 0.5 * sigma ** 2
    variance = sigma ** 2 * t

    return np.exp(
        i * u * (log_s0 + drift * t) - 0.5 * variance * u ** 2
    )


@njit(cache=True, fastmath=True)
def _gbm_drift(s: float, r: float, q: float) -> float:
    """Risk-neutral drift: (r - q) * S."""
    return (r - q) * s


@njit(cache=True, fastmath=True)
def _gbm_diffusion(s: float, sigma: float) -> float:
    """Diffusion coefficient: sigma * S."""
    return sigma * s


# =============================================================================
# GBM MODEL
# =============================================================================

@dataclass(frozen=True)
class GBMModel(Model):
    """
    Geometric Brownian Motion (Black-Scholes) Model.

    The simplest and most widely used model for option pricing.
    Assumes constant volatility and log-normal stock price distribution.

    Model:
        dS = (r - q) * S * dt + sigma * S * dW

    Parameters
    ----------
    sigma : float
        Volatility (annualized), e.g., 0.20 for 20%

    Examples
    --------
    model = GBMModel(sigma=0.20)

    # Check supported engines
    model.supported_engines  # [ANALYTICAL, FFT, MONTE_CARLO]

    # Characteristic function for FFT pricing
    cf = model.characteristic_function(u=1+0.5j, s0=100, t=0.5, r=0.05)

    # SDE coefficients for Monte Carlo
    drift = model.drift(s=100, v=0, t=0, r=0.05, q=0)
    diff = model.diffusion(s=100, v=0, t=0)
    """

    sigma: float

    def __post_init__(self):
        """Validate parameters."""
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return "Geometric Brownian Motion"

    @property
    def supported_engines(self) -> List[PricingCapability]:
        """Which pricing methods this model supports."""
        return [
            PricingCapability.ANALYTICAL,
            PricingCapability.FFT,
            PricingCapability.MONTE_CARLO,
        ]

    def get_parameters(self) -> Dict[str, Any]:
        """Return model parameters as dictionary."""
        return {"sigma": self.sigma}

    def characteristic_function(
        self, u: complex, s0: float, t: float, r: float, q: float = 0.0
    ) -> complex:
        """
        Characteristic function phi(u) = E^Q[exp(i*u*ln(S_T))].

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
        return _gbm_characteristic_function(u, s0, t, r, q, self.sigma)

    def characteristic_function_vectorized(
        self, u_arr: np.ndarray, s0: float, t: float, r: float, q: float = 0.0
    ) -> np.ndarray:
        """
        Vectorized characteristic function for FFT pricing.

        Parameters
        ----------
        u_arr : np.ndarray
            Array of Fourier transform variables (complex)
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
        np.ndarray
            Array of characteristic function values
        """
        log_s0 = np.log(s0)
        drift = r - q - 0.5 * self.sigma ** 2
        variance = self.sigma ** 2 * t

        return np.exp(
            1j * u_arr * (log_s0 + drift * t) - 0.5 * variance * u_arr ** 2
        )

    def drift(self, s: float, v: float, t: float, r: float, q: float) -> float:
        """
        Drift coefficient for SDE discretization.

        For GBM: drift = (r - q) * S

        Parameters
        ----------
        s : float
            Current spot price
        v : float
            Current variance (unused in GBM)
        t : float
            Current time (unused in GBM)
        r : float
            Risk-free rate
        q : float
            Dividend yield

        Returns
        -------
        float
            Drift value
        """
        return _gbm_drift(s, r, q)

    def diffusion(self, s: float, v: float, t: float) -> float:
        """
        Diffusion coefficient for SDE discretization.

        For GBM: diffusion = sigma * S

        Parameters
        ----------
        s : float
            Current spot price
        v : float
            Current variance (unused in GBM)
        t : float
            Current time (unused in GBM)

        Returns
        -------
        float
            Diffusion value
        """
        return _gbm_diffusion(s, self.sigma)

    @property
    def variance(self) -> float:
        """Annualized variance sigma^2."""
        return self.sigma ** 2

    def __repr__(self) -> str:
        return f"GBMModel(sigma={self.sigma})"


if __name__ == "__main__":
    print("=" * 50)
    print("GBM Model Smoke Test")
    print("=" * 50)

    # Create model
    model = GBMModel(sigma=0.20)
    print(f"\nModel: {model}")
    print(f"Name: {model.name}")
    print(f"Parameters: {model.get_parameters()}")
    print(f"Supported engines: {model.supported_engines}")

    # Test characteristic function
    print("\n--- Characteristic Function ---")
    s0, t, r, q = 100.0, 0.5, 0.05, 0.02
    u = 1.0 + 0.5j
    cf = model.characteristic_function(u, s0, t, r, q)
    print(f"phi({u}) = {cf}")
    print(f"|phi| = {abs(cf):.6f}")

    # Test SDE coefficients
    print("\n--- SDE Coefficients ---")
    drift = model.drift(s=100, v=0, t=0, r=0.05, q=0.02)
    diff = model.diffusion(s=100, v=0, t=0)
    print(f"Drift at S=100: {drift:.2f}")
    print(f"Diffusion at S=100: {diff:.2f}")

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
        bad_model = GBMModel(sigma=-0.1)
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"Correctly rejected invalid params: {e}")

    print("\n" + "=" * 50)
    print("GBM smoke test passed")
    print("=" * 50)
