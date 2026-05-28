"""
Bates Model
===========

Bates (1996) stochastic volatility with jumps model.

Model:
    dS = (r - q - lam * k) * S * dt + sqrt(V) * S * dW_S + (J - 1) * S * dN
    dV = kappa * (theta - V) * dt + alpha * sqrt(V) * dW_V
    Corr(dW_S, dW_V) = rho

Where:
    - dN is a Poisson process with intensity lam
    - J is lognormal: ln(J) ~ N(alpha_j, sigma_j^2)
    - k = E[J - 1] = exp(alpha_j + 0.5*sigma_j^2) - 1

Notation: long-run variance ``theta`` is
written ``σ²``, vol-of-vol ``alpha`` is ``α``, jump intensity ``lam`` is ``λ``,
and the log-jump law is ``J ~ N(α_J, σ_J²)``. The Python identifier ``theta``
is kept (displayed σ²) to avoid colliding with the option Greek θ.

The Bates model combines Heston stochastic volatility with Merton-style jumps,
allowing for both volatility smile AND fat tails in the distribution.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.utils.validation import (
    feller_ratio as _feller_ratio,
    feller_satisfied as _feller_satisfied,
)

if TYPE_CHECKING:
    from backend.models.heston import HestonModel
    from backend.simulation.models.bates import BatesSimulator

from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability
from backend.models.characteristic_functions.bates_cf import (
    bates_cf_vectorized,
    bates_characteristic_function,
)

# =============================================================================
# BATES MODEL
# =============================================================================


@dataclass(frozen=True)
class BatesModel(Model):
    """
    Bates (1996) Stochastic Volatility with Jumps Model.

    Combines Heston stochastic volatility with Merton-style lognormal jumps.
    The most flexible single-asset model for equity options.

    Model:
        dS = (r - q - lam * k) * S * dt + sqrt(V) * S * dW_S + (J - 1) * S * dN
        dV = kappa * (theta - V) * dt + alpha * sqrt(V) * dW_V
        Corr(dW_S, dW_V) = rho

    Heston Parameters
    -----------------
    v0 : float
        Initial variance (sigma^2), e.g., 0.04 for 20% initial vol
    kappa : float
        Mean reversion speed of variance (typical: 1-5)
    theta : float
        Long-run variance level
    alpha : float
        Volatility of volatility (vol-of-vol)
    rho : float
        Correlation between price and variance (-1 to 1)

    Jump Parameters
    ---------------
    lam : float
        Jump intensity (expected number of jumps per year)
    alpha_j : float
        Mean of log-jump size (e.g., -0.1 for 10% negative mean jump)
    sigma_j : float
        Volatility of log-jump size

    Examples
    --------
    model = BatesModel(
        v0=0.04, kappa=2.0, theta=0.04, alpha=0.3, rho=-0.7,
        lam=0.5, alpha_j=-0.1, sigma_j=0.2
    )

    # Check Feller condition (same as Heston)
    model.feller_satisfied

    # Expected jump size
    model.expected_jump_size  # E[J - 1]

    Notes
    -----
    - When lam = 0, reduces to Heston model
    - Jumps add fat tails to the distribution
    - alpha_j < 0 creates negative skewness (crash risk)
    - sigma_j controls jump size uncertainty
    """

    # Heston parameters
    v0: float
    kappa: float
    theta: float
    alpha: float
    rho: float
    # Jump parameters
    lam: float
    alpha_j: float
    sigma_j: float

    def __post_init__(self) -> None:
        """Validate parameters."""
        # Heston validation
        if self.v0 < 0:
            raise ValueError(f"v0 must be non-negative, got {self.v0}")
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.theta < 0:
            raise ValueError(f"theta must be non-negative, got {self.theta}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        # Jump validation
        if self.lam < 0:
            raise ValueError(f"lam must be non-negative, got {self.lam}")
        if not -1.0 <= self.alpha_j <= 1.0:
            raise ValueError(f"alpha_j should be in [-1, 1], got {self.alpha_j}")
        if self.sigma_j < 0:
            raise ValueError(f"sigma_j must be non-negative, got {self.sigma_j}")

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return "Bates (Heston + Jumps)"

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
            # Heston
            "v0": self.v0,
            "kappa": self.kappa,
            "theta": self.theta,
            "alpha": self.alpha,
            "rho": self.rho,
            # Jumps
            "lam": self.lam,
            "alpha_j": self.alpha_j,
            "sigma_j": self.sigma_j,
        }

    def characteristic_function(
        self, u: complex, s0: float, t: float, r: float, q: float = 0.0
    ) -> complex:
        """
        Bates characteristic function phi(u) = E^Q[exp(i*u*ln(S_T))].

        Combines Heston CF with jump contribution.

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
        return bates_characteristic_function(
            u,
            s0_adj,
            self.v0,
            t,
            r,
            self.kappa,
            self.theta,
            self.alpha,
            self.rho,
            self.lam,
            self.alpha_j,
            self.sigma_j,
        )

    def characteristic_function_vectorized(
        self, u_arr: np.ndarray, s0: float, t: float, r: float, q: float = 0.0
    ) -> np.ndarray:
        """Vectorized characteristic function for FFT."""
        s0_adj = s0 * np.exp(-q * t)
        return bates_cf_vectorized(
            u_arr,
            s0_adj,
            self.v0,
            t,
            r,
            self.kappa,
            self.theta,
            self.alpha,
            self.rho,
            self.lam,
            self.alpha_j,
            self.sigma_j,
        )

    def drift(self, s: float, v: float, t: float, r: float, q: float) -> float:
        """
        Drift coefficient for SDE discretization.

        For Bates: drift = (r - q - lam * k) * S
        where k = E[J - 1] is the expected jump size.

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
        k = self.expected_jump_size
        return (r - q - self.lam * k) * s

    def diffusion(self, s: float, v: float, t: float) -> float:
        """
        Diffusion coefficient for SDE discretization.

        For Bates (spot process): diffusion = sqrt(V) * S

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
        """Drift of the variance process."""
        return self.kappa * (self.theta - v)

    def variance_diffusion(self, v: float) -> float:
        """Diffusion of the variance process."""
        return self.alpha * np.sqrt(max(v, 0.0))

    @property
    def expected_jump_size(self) -> float:
        """
        Expected relative jump size k = E[J - 1].

        For lognormal J: k = exp(alpha_j + 0.5*sigma_j^2) - 1
        """
        return np.exp(self.alpha_j + 0.5 * self.sigma_j**2) - 1

    @property
    def expected_jump_return(self) -> float:
        """
        Expected jump return as percentage.

        Same as expected_jump_size but expressed as percent.
        """
        return self.expected_jump_size * 100

    @property
    def feller_satisfied(self) -> bool:
        """Check if Feller condition 2*kappa*theta > alpha^2 is satisfied."""
        return _feller_satisfied(self.kappa, self.theta, self.alpha)

    @property
    def feller_ratio(self) -> float:
        """Feller ratio: 2*kappa*theta / alpha^2."""
        return _feller_ratio(self.kappa, self.theta, self.alpha)

    @property
    def long_run_volatility(self) -> float:
        """Long-run volatility sqrt(theta)."""
        return np.sqrt(self.theta)

    @property
    def initial_volatility(self) -> float:
        """Initial volatility sqrt(v0)."""
        return np.sqrt(self.v0)

    def expected_jumps_per_year(self) -> float:
        """Expected number of jumps per year."""
        return self.lam

    def jump_contribution_to_variance(self) -> float:
        """
        Variance contribution from jumps.

        For Merton-style jumps: lam * (alpha_j^2 + sigma_j^2)
        """
        return self.lam * (self.alpha_j**2 + self.sigma_j**2)

    def to_heston(self) -> HestonModel:
        """
        Convert to Heston model (drop jumps).

        Returns
        -------
        HestonModel
            Equivalent Heston model without jumps
        """
        from backend.models.heston import HestonModel

        return HestonModel(
            v0=self.v0, kappa=self.kappa, theta=self.theta, alpha=self.alpha, rho=self.rho
        )

    def mean_variance(self, t: float) -> float:
        """
        Expected variance at time t: E[V_t].

        Same as Heston - the variance process is unchanged by jumps.

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

        Alias for mean_variance for API consistency.

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
        Total variance over period t, including jump contribution.

        Combines stochastic volatility and jump contributions.

        Parameters
        ----------
        t : float
            Time period (default 1 year)

        Returns
        -------
        float
            Total variance
        """
        # Stochastic vol contribution (from Heston)
        if abs(self.kappa) < 1e-10:
            sv_var = self.v0 * t
        else:
            sv_var = (
                self.theta * t
                + (self.v0 - self.theta) * (1 - np.exp(-self.kappa * t)) / self.kappa
            )

        # Jump contribution (from Merton-style jumps)
        jump_var = self.lam * t * (self.alpha_j**2 + self.sigma_j**2)

        return sv_var + jump_var

    def total_volatility(self, t: float = 1.0) -> float:
        """
        Total annualized volatility including jumps.

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

    def create_simulator(self, **kwargs: Any) -> BatesSimulator:
        """
        Create a Bates simulator for Monte Carlo pricing.

        Parameters
        ----------
        **kwargs
            Additional simulator parameters

        Returns
        -------
        BatesSimulator
            Configured simulator instance
        """
        from backend.simulation.models.bates import BatesSimulator

        return BatesSimulator(
            v0=self.v0,
            kappa=self.kappa,
            theta=self.theta,
            alpha=self.alpha,
            rho=self.rho,
            lam=self.lam,
            alpha_j=self.alpha_j,
            sigma_j=self.sigma_j,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"BatesModel(v0={self.v0}, kappa={self.kappa}, theta={self.theta}, "
            f"alpha={self.alpha}, rho={self.rho}, lam={self.lam}, "
            f"alpha_j={self.alpha_j}, sigma_j={self.sigma_j})"
        )


if __name__ == "__main__":
    print("=" * 50)
    print("Bates Model Smoke Test")
    print("=" * 50)

    # Create model
    model = BatesModel(
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        alpha=0.3,
        rho=-0.7,
        lam=0.5,
        alpha_j=-0.1,
        sigma_j=0.2,
    )
    print(f"\nModel: {model}")
    print(f"Name: {model.name}")
    print(f"Supported engines: {model.supported_engines}")

    # Parameters
    print("\n--- Parameters ---")
    for k, v in model.get_parameters().items():
        print(f"  {k}: {v}")

    # Feller condition
    print("\n--- Feller Condition ---")
    print(f"Feller satisfied: {model.feller_satisfied}")
    print(f"Feller ratio: {model.feller_ratio:.2f}")

    # Jump characteristics
    print("\n--- Jump Characteristics ---")
    print(f"Expected jumps/year: {model.expected_jumps_per_year()}")
    print(f"Expected jump size: {model.expected_jump_size:.2%}")
    print(f"Jump variance contribution: {model.jump_contribution_to_variance():.4f}")

    # Test characteristic function
    print("\n--- Characteristic Function ---")
    s0, t, r, q = 100.0, 0.5, 0.05, 0.02
    u = 1.0 + 0.5j
    cf = model.characteristic_function(u, s0, t, r, q)
    print(f"phi({u}) = {cf}")
    print(f"|phi| = {abs(cf):.6f}")

    # Test SDE coefficients
    print("\n--- SDE Coefficients ---")
    drift = model.drift(s=100, v=0.04, t=0, r=0.05, q=0.02)
    diff = model.diffusion(s=100, v=0.04, t=0)
    print(f"Spot drift at S=100, V=0.04: {drift:.2f}")
    print(f"Spot diffusion: {diff:.2f}")

    # Compare with Heston
    print("\n--- Comparison with Heston ---")
    heston = model.to_heston()
    cf_heston = heston.characteristic_function(u, s0, t, r, q)
    print(f"Bates CF: {abs(cf):.6f}")
    print(f"Heston CF: {abs(cf_heston):.6f}")
    print(f"Difference (jump effect): {abs(abs(cf) - abs(cf_heston)):.6f}")

    # Test immutability
    print("\n--- Immutability Test ---")
    try:
        model.lam = 1.0  # type: ignore
        print("ERROR: Mutation should have failed!")
    except Exception as e:
        print(f"Correctly prevented mutation: {type(e).__name__}")

    # Test validation
    print("\n--- Validation Test ---")
    try:
        bad_model = BatesModel(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            alpha=0.3,
            rho=-0.7,
            lam=-0.5,
            alpha_j=-0.1,
            sigma_j=0.2,
        )
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"Correctly rejected invalid lam: {e}")

    print("\n" + "=" * 50)
    print("Bates smoke test passed")
    print("=" * 50)
