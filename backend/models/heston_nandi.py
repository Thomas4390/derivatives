"""
Heston-Nandi GARCH Model
========================

Heston & Nandi (2000) discrete-time GARCH option-pricing model — the
risk-neutral GARCH that calibrates to an option / implied-vol surface (unlike
the physical-measure GARCH family, which is MLE-fitted to a return series).

Risk-neutral dynamics (per-period / daily step, ``lambda* = -1/2``):

    R_t = ln(S_t/S_{t-1}) = r_step - 0.5 h_t + sqrt(h_t) z_t,  z_t ~ N(0,1)
    h_{t+1} = omega + beta h_t + alpha (z_t - gamma sqrt(h_t))^2

with ``r_step = r / steps_per_year``. The model admits a closed-form
characteristic function (log-affine recursion), so European options price by
Carr-Madan FFT exactly like Heston / Bates.

Parameters
----------
omega, alpha, beta : float
    Variance-recursion coefficients (per period), all non-negative.
gamma : float
    Risk-neutral leverage. Large (O(100)) because it multiplies
    ``sqrt(h_t) ~ vol / sqrt(steps_per_year)``.
h0 : float
    Initial conditional variance ``h_1`` (per period).
steps_per_year : int
    Trading-day discretization (default 252).

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability
from backend.models.characteristic_functions.heston_nandi_cf import (
    heston_nandi_cf_vectorized,
    heston_nandi_characteristic_function,
)
from backend.utils.constants.calibration import HESTON_NANDI_STEPS_PER_YEAR

if TYPE_CHECKING:
    from backend.simulation.models.heston_nandi_garch import HestonNandiGARCHSimulator


@dataclass(frozen=True)
class HestonNandiGARCHModel(Model):
    """Heston-Nandi (2000) risk-neutral GARCH(1,1) option-pricing model."""

    omega: float
    alpha: float
    beta: float
    gamma: float
    h0: float
    steps_per_year: int = HESTON_NANDI_STEPS_PER_YEAR

    def __post_init__(self) -> None:
        """Validate variance-positivity parameters.

        Stationarity (``beta + alpha*gamma^2 < 1``) is intentionally NOT
        enforced here: finite-horizon prices remain well defined for a
        non-stationary parameterisation, and the calibrator's
        ``StationarityMode.OFF`` must be able to construct such a model.
        """
        if self.omega < 0:
            raise ValueError(f"omega must be non-negative, got {self.omega}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {self.gamma}")
        if self.h0 < 0:
            raise ValueError(f"h0 must be non-negative, got {self.h0}")
        if self.steps_per_year <= 0:
            raise ValueError(
                f"steps_per_year must be positive, got {self.steps_per_year}"
            )

    @property
    def name(self) -> str:
        return "Heston-Nandi GARCH"

    @property
    def supported_engines(self) -> list[PricingCapability]:
        return [PricingCapability.FFT, PricingCapability.MONTE_CARLO]

    def get_parameters(self) -> dict[str, Any]:
        return {
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "h0": self.h0,
        }

    def characteristic_function(
        self, u: complex, s0: float, t: float, r: float, q: float = 0.0
    ) -> complex:
        """Log-price CF ``phi(u) = E^Q[exp(i u ln S_T)]`` (dividend-adjusted)."""
        s0_adj = s0 * np.exp(-q * t)
        return heston_nandi_characteristic_function(
            u,
            s0_adj,
            self.omega,
            self.alpha,
            self.beta,
            self.gamma,
            self.h0,
            t,
            r,
            self.steps_per_year,
        )

    def characteristic_function_vectorized(
        self, u_arr: np.ndarray, s0: float, t: float, r: float, q: float = 0.0
    ) -> np.ndarray:
        """Vectorized log-price CF for FFT pricing."""
        s0_adj = s0 * np.exp(-q * t)
        return heston_nandi_cf_vectorized(
            u_arr,
            s0_adj,
            self.omega,
            self.alpha,
            self.beta,
            self.gamma,
            self.h0,
            t,
            r,
            self.steps_per_year,
        )

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    @property
    def persistence(self) -> float:
        """Variance persistence ``beta + alpha*gamma^2`` (< 1 ⇒ stationary)."""
        return float(self.beta + self.alpha * self.gamma**2)

    @property
    def is_stationary(self) -> bool:
        """True iff the unconditional variance is finite (persistence < 1)."""
        return self.persistence < 1.0

    @property
    def long_run_variance(self) -> float:
        """Per-period unconditional variance ``(omega + alpha) / (1 - persistence)``.

        Returns ``inf`` for a non-stationary parameterisation.
        """
        gap = 1.0 - self.persistence
        if gap <= 0.0:
            return float("inf")
        return float((self.omega + self.alpha) / gap)

    @property
    def long_run_volatility(self) -> float:
        """Annualized long-run volatility ``sqrt(long_run_variance * steps_per_year)``."""
        return float(np.sqrt(self.long_run_variance * self.steps_per_year))

    @property
    def initial_volatility(self) -> float:
        """Annualized initial volatility ``sqrt(h0 * steps_per_year)``."""
        return float(np.sqrt(self.h0 * self.steps_per_year))

    def create_simulator(self, **kwargs: Any) -> HestonNandiGARCHSimulator:
        """Create a risk-neutral Heston-Nandi GARCH path simulator."""
        from backend.simulation.models.heston_nandi_garch import (
            HestonNandiGARCHSimulator,
        )

        return HestonNandiGARCHSimulator(
            omega=self.omega,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            h0=self.h0,
            steps_per_year=self.steps_per_year,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"HestonNandiGARCHModel(omega={self.omega:.3e}, alpha={self.alpha:.3e}, "
            f"beta={self.beta:.4f}, gamma={self.gamma:.2f}, h0={self.h0:.3e})"
        )
