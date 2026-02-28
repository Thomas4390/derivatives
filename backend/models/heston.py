"""
Heston Model
============

Heston (1993) stochastic volatility model.

Model:
    dS = (r - q) * S * dt + sqrt(V) * S * dW_S
    dV = kappa * (theta - V) * dt + xi * sqrt(V) * dW_V
    Corr(dW_S, dW_V) = rho

The Heston model captures the volatility smile through stochastic variance.
The correlation parameter rho captures the leverage effect (typically negative
for equities - stock drops -> volatility increases).

Author: Thomas
Created: 2025
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability
from backend.models.characteristic_functions.heston_cf import (
    heston_cf_vectorized,
    heston_characteristic_function,
)

# =============================================================================
# HESTON MODEL
# =============================================================================

@dataclass(frozen=True)
class HestonModel(Model):
    """
    Heston (1993) Stochastic Volatility Model.

    The most popular stochastic volatility model in practice.
    Captures the volatility smile and term structure effects.

    Model:
        dS = (r - q) * S * dt + sqrt(V) * S * dW_S
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

    Examples
    --------
    model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

    # Check Feller condition
    model.feller_satisfied  # True if 2*kappa*theta > xi^2

    # Characteristic function for FFT pricing
    cf = model.characteristic_function(u=1+0.5j, s0=100, t=0.5, r=0.05)

    # SDE coefficients for Monte Carlo
    drift = model.drift(s=100, v=0.04, t=0, r=0.05, q=0)

    Notes
    -----
    - Feller condition: 2*kappa*theta > xi^2 ensures V stays strictly positive
    - Typical equity: rho < 0 (leverage effect: stock drops -> vol increases)
    - kappa controls how fast variance reverts to theta
    - xi controls the "volatility of volatility"
    """

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self):
        """Validate parameters."""
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
    def name(self) -> str:
        """Human-readable model name."""
        return "Heston Stochastic Volatility"

    @property
    def supported_engines(self) -> list[PricingCapability]:
        """Which pricing methods this model supports."""
        return [
            PricingCapability.FFT,
            PricingCapability.MONTE_CARLO,
        ]

    def get_parameters(self) -> dict[str, Any]:
        """Return model parameters as dictionary."""
        return {
            "v0": self.v0,
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
        }

    def characteristic_function(
        self, u: complex, s0: float, t: float, r: float, q: float = 0.0
    ) -> complex:
        """
        Heston characteristic function phi(u) = E^Q[exp(i*u*ln(S_T))].

        Uses the Gatheral (2006) formulation for numerical stability.

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
            Dividend yield (adjusts forward price)

        Returns
        -------
        complex
            Value of characteristic function at u
        """
        # Adjust for dividend yield via forward price
        s0_adj = s0 * np.exp(-q * t)
        return heston_characteristic_function(
            u, s0_adj, self.v0, t, r,
            self.kappa, self.theta, self.xi, self.rho
        )

    def characteristic_function_vectorized(
        self, u_arr: np.ndarray, s0: float, t: float, r: float, q: float = 0.0
    ) -> np.ndarray:
        """
        Vectorized characteristic function for FFT.

        Parameters
        ----------
        u_arr : np.ndarray
            Array of frequency arguments
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
        s0_adj = s0 * np.exp(-q * t)
        return heston_cf_vectorized(
            u_arr, s0_adj, self.v0, t, r,
            self.kappa, self.theta, self.xi, self.rho
        )

    def drift(self, s: float, v: float, t: float, r: float, q: float) -> float:
        """
        Drift coefficient for SDE discretization.

        For Heston (spot process): drift = (r - q) * S

        Parameters
        ----------
        s : float
            Current spot price
        v : float
            Current variance
        t : float
            Current time
        r : float
            Risk-free rate
        q : float
            Dividend yield

        Returns
        -------
        float
            Drift value
        """
        return (r - q) * s

    def diffusion(self, s: float, v: float, t: float) -> float:
        """
        Diffusion coefficient for SDE discretization.

        For Heston (spot process): diffusion = sqrt(V) * S

        Parameters
        ----------
        s : float
            Current spot price
        v : float
            Current variance
        t : float
            Current time

        Returns
        -------
        float
            Diffusion value
        """
        return np.sqrt(max(v, 0.0)) * s

    def variance_drift(self, v: float) -> float:
        """
        Drift of the variance process.

        dV = kappa * (theta - V) * dt + ...

        Parameters
        ----------
        v : float
            Current variance

        Returns
        -------
        float
            Variance drift
        """
        return self.kappa * (self.theta - v)

    def variance_diffusion(self, v: float) -> float:
        """
        Diffusion of the variance process.

        dV = ... + xi * sqrt(V) * dW_V

        Parameters
        ----------
        v : float
            Current variance

        Returns
        -------
        float
            Variance diffusion
        """
        return self.xi * np.sqrt(max(v, 0.0))

    @property
    def feller_satisfied(self) -> bool:
        """
        Check if Feller condition 2*kappa*theta > xi^2 is satisfied.

        When satisfied, variance process stays strictly positive.
        If violated, variance can touch zero (need boundary handling).
        """
        return 2 * self.kappa * self.theta > self.xi ** 2

    @property
    def feller_ratio(self) -> float:
        """
        Feller ratio: 2*kappa*theta / xi^2.

        > 1: Feller satisfied (variance stays positive)
        < 1: Feller violated (variance can touch zero)
        """
        return (2 * self.kappa * self.theta) / (self.xi ** 2)

    @property
    def long_run_volatility(self) -> float:
        """Long-run volatility sqrt(theta)."""
        return np.sqrt(self.theta)

    @property
    def initial_volatility(self) -> float:
        """Initial volatility sqrt(v0)."""
        return np.sqrt(self.v0)

    def mean_variance(self, t: float) -> float:
        """
        Expected variance at time t: E[V_t].

        V_t converges to theta as t -> infinity.

        Parameters
        ----------
        t : float
            Time

        Returns
        -------
        float
            Expected variance
        """
        decay = np.exp(-self.kappa * t)
        return self.theta + (self.v0 - self.theta) * decay

    def expected_variance(self, t: float) -> float:
        """
        Expected variance at time t under Q measure.

        Alias for mean_variance for API consistency with MertonModel.

        Parameters
        ----------
        t : float
            Time

        Returns
        -------
        float
            Expected variance
        """
        return self.mean_variance(t)

    def total_variance(self, t: float = 1.0) -> float:
        """
        Approximate total variance over period t.

        For stochastic volatility, this integrates E[V_s] from 0 to t.

        Parameters
        ----------
        t : float
            Time period (default 1 year)

        Returns
        -------
        float
            Integrated expected variance
        """
        # Integral of E[V_s] = theta*t + (v0 - theta)*(1 - exp(-kappa*t))/kappa
        if abs(self.kappa) < 1e-10:
            return self.v0 * t
        return self.theta * t + (self.v0 - self.theta) * (1 - np.exp(-self.kappa * t)) / self.kappa

    def total_volatility(self, t: float = 1.0) -> float:
        """
        Approximate total annualized volatility.

        Returns sqrt(total_variance(t) / t).

        Parameters
        ----------
        t : float
            Time period (default 1 year)

        Returns
        -------
        float
            Annualized volatility
        """
        return np.sqrt(self.total_variance(t) / t)

    def create_simulator(self, **kwargs):
        """
        Create a Heston simulator for Monte Carlo pricing.

        Parameters
        ----------
        **kwargs
            Additional simulator parameters (e.g., scheme)

        Returns
        -------
        HestonSimulator
            Configured simulator instance
        """
        from backend.simulation.models.heston import HestonSimulator
        return HestonSimulator(
            v0=self.v0,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.xi,
            rho=self.rho,
            **kwargs
        )

    def __repr__(self) -> str:
        return (
            f"HestonModel(v0={self.v0}, kappa={self.kappa}, "
            f"theta={self.theta}, xi={self.xi}, rho={self.rho})"
        )


if __name__ == "__main__":
    print("=" * 50)
    print("Heston Model Smoke Test")
    print("=" * 50)

    # Create model
    model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    print(f"\nModel: {model}")
    print(f"Name: {model.name}")
    print(f"Parameters: {model.get_parameters()}")
    print(f"Supported engines: {model.supported_engines}")

    # Feller condition
    print("\n--- Feller Condition ---")
    print(f"Feller satisfied: {model.feller_satisfied}")
    print(f"Feller ratio: {model.feller_ratio:.2f}")
    print(f"Initial vol: {model.initial_volatility:.1%}")
    print(f"Long-run vol: {model.long_run_volatility:.1%}")

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
    v_current = 0.04
    drift = model.drift(s=100, v=v_current, t=0, r=0.05, q=0.02)
    diff = model.diffusion(s=100, v=v_current, t=0)
    var_drift = model.variance_drift(v=v_current)
    var_diff = model.variance_diffusion(v=v_current)
    print(f"Spot drift at S=100, V=0.04: {drift:.2f}")
    print(f"Spot diffusion at S=100, V=0.04: {diff:.2f}")
    print(f"Variance drift at V=0.04: {var_drift:.4f}")
    print(f"Variance diffusion at V=0.04: {var_diff:.4f}")

    # Expected variance
    print("\n--- Variance Evolution ---")
    for t in [0.0, 0.5, 1.0, 5.0]:
        print(f"E[V_{t}] = {model.mean_variance(t):.4f} (vol = {np.sqrt(model.mean_variance(t)):.1%})")

    # Test immutability
    print("\n--- Immutability Test ---")
    try:
        model.v0 = 0.09  # type: ignore
        print("ERROR: Mutation should have failed!")
    except Exception as e:
        print(f"Correctly prevented mutation: {type(e).__name__}")

    # Test validation
    print("\n--- Validation Test ---")
    try:
        bad_model = HestonModel(v0=0.04, kappa=-1.0, theta=0.04, xi=0.3, rho=-0.7)
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"Correctly rejected invalid kappa: {e}")

    try:
        bad_model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-1.5)
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"Correctly rejected invalid rho: {e}")

    print("\n" + "=" * 50)
    print("Heston smoke test passed")
    print("=" * 50)
