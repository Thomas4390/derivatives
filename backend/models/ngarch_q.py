"""
Risk-neutral GARCH(1,1) option-pricing models (nonaffine, Monte-Carlo)
======================================================================

The risk-neutral counterparts of the *physical* GARCH family, used to calibrate
an option / implied-vol surface directly under Q (Duan 1995 LRNVR; Dorion &
François §7.2). Three variants share one Monte-Carlo simulator
(:class:`backend.simulation.models.garch_q.GARCHRiskNeutralSimulator`) — unlike
the affine Heston-Nandi model they admit **no** closed-form characteristic
function, so European options price by Monte-Carlo only.

Risk-neutral dynamics (per-period / daily step), ``r_step = r / steps_per_year``::

    R_t      = r_step - 0.5 h_t + sqrt(h_t) z_t,           z_t ~ N(0,1)
    garch :  h_{t+1} = omega + alpha h_t z_t^2 + beta h_t                 (symmetric)
    ngarch:  h_{t+1} = omega + alpha h_t (z_t - gamma)^2 + beta h_t       (Duan 1995)
    gjr   :  h_{t+1} = omega + (alpha + gamma 1{z_t<0}) h_t z_t^2 + beta h_t

Here ``gamma`` is the *risk-neutral* asymmetry ``gamma* = gamma_P + lambda`` (the
LRNVR shift by the unit market price of risk); calibrating to options identifies
``gamma*`` directly, so no separate risk-premium parameter is needed. Plain
``garch`` is symmetric (no ``gamma``) and cannot reproduce an equity skew.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability
from backend.utils.constants.time import TRADING_DAYS_PER_YEAR

if TYPE_CHECKING:
    from backend.simulation.models.garch_q import GARCHRiskNeutralSimulator


class _GARCHRiskNeutralMixin:
    """Shared behaviour for the risk-neutral GARCH-Q family.

    Concrete subclasses are frozen dataclasses carrying ``omega, alpha, beta,
    (gamma), h0, steps_per_year`` plus three class attributes: ``_GARCH_TYPE``
    (simulator variant), ``_DISPLAY`` (model name) and ``_PARAM_NAMES`` (the
    calibratable parameters, in slot order).
    """

    _GARCH_TYPE: ClassVar[str]
    _DISPLAY: ClassVar[str]
    _PARAM_NAMES: ClassVar[tuple[str, ...]]

    # Provided as dataclass fields by every concrete subclass.
    omega: float
    alpha: float
    beta: float
    h0: float
    steps_per_year: int

    def _validate(self) -> None:
        if self.omega < 0 or self.alpha < 0 or self.beta < 0 or self.h0 < 0:
            raise ValueError("omega, alpha, beta, h0 must all be non-negative")
        if float(getattr(self, "gamma", 0.0)) < 0:
            raise ValueError(
                f"gamma must be non-negative, got {getattr(self, 'gamma', 0.0)}"
            )
        if self.steps_per_year <= 0:
            raise ValueError(
                f"steps_per_year must be positive, got {self.steps_per_year}"
            )

    @property
    def name(self) -> str:
        return self._DISPLAY

    @property
    def supported_engines(self) -> list[PricingCapability]:
        return [PricingCapability.MONTE_CARLO]

    def get_parameters(self) -> dict[str, Any]:
        return {n: float(getattr(self, n)) for n in self._PARAM_NAMES}

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    @property
    def persistence(self) -> float:
        """Variance persistence (< 1 ⇒ finite unconditional variance)."""
        a, b, g = float(self.alpha), float(self.beta), float(getattr(self, "gamma", 0.0))
        if self._GARCH_TYPE == "ngarch":
            return b + a * (1.0 + g * g)
        if self._GARCH_TYPE == "gjr_garch":
            return b + a + 0.5 * g
        return b + a

    @property
    def is_stationary(self) -> bool:
        return self.persistence < 1.0

    @property
    def long_run_variance(self) -> float:
        """Per-period unconditional variance ``omega / (1 - persistence)`` (``inf`` if non-stationary)."""
        gap = 1.0 - self.persistence
        return float(self.omega / gap) if gap > 0.0 else float("inf")

    @property
    def long_run_volatility(self) -> float:
        """Annualized long-run volatility ``sqrt(long_run_variance * steps_per_year)``."""
        return float(np.sqrt(self.long_run_variance * self.steps_per_year))

    @property
    def initial_volatility(self) -> float:
        """Annualized initial volatility ``sqrt(h0 * steps_per_year)``."""
        return float(np.sqrt(self.h0 * self.steps_per_year))

    def create_simulator(self, **kwargs: Any) -> GARCHRiskNeutralSimulator:
        """Create the matching risk-neutral GARCH path simulator."""
        from backend.simulation.models.garch_q import GARCHRiskNeutralSimulator

        return GARCHRiskNeutralSimulator(
            garch_type=self._GARCH_TYPE,
            omega=self.omega,
            alpha=self.alpha,
            beta=self.beta,
            gamma=float(getattr(self, "gamma", 0.0)),
            h0=self.h0,
            steps_per_year=self.steps_per_year,
            **kwargs,
        )


@dataclass(frozen=True)
class GARCHRiskNeutralModel(_GARCHRiskNeutralMixin, Model):
    """Symmetric risk-neutral GARCH(1,1) (no leverage) — MC-priced surface model."""

    omega: float
    alpha: float
    beta: float
    h0: float
    steps_per_year: int = TRADING_DAYS_PER_YEAR

    _GARCH_TYPE: ClassVar[str] = "garch"
    _DISPLAY: ClassVar[str] = "GARCH (risk-neutral)"
    _PARAM_NAMES: ClassVar[tuple[str, ...]] = ("omega", "alpha", "beta", "h0")

    def __post_init__(self) -> None:
        self._validate()

    def __repr__(self) -> str:
        return (
            f"GARCHRiskNeutralModel(omega={self.omega:.3e}, alpha={self.alpha:.3e}, "
            f"beta={self.beta:.4f}, h0={self.h0:.3e})"
        )


@dataclass(frozen=True)
class NGARCHRiskNeutralModel(_GARCHRiskNeutralMixin, Model):
    """Duan (1995) risk-neutral NGARCH(1,1) option-pricing model (nonaffine)."""

    omega: float
    alpha: float
    beta: float
    gamma: float
    h0: float
    steps_per_year: int = TRADING_DAYS_PER_YEAR

    _GARCH_TYPE: ClassVar[str] = "ngarch"
    _DISPLAY: ClassVar[str] = "Duan NGARCH (risk-neutral)"
    _PARAM_NAMES: ClassVar[tuple[str, ...]] = (
        "omega", "alpha", "beta", "gamma", "h0",
    )

    def __post_init__(self) -> None:
        self._validate()

    def __repr__(self) -> str:
        return (
            f"NGARCHRiskNeutralModel(omega={self.omega:.3e}, alpha={self.alpha:.3e}, "
            f"beta={self.beta:.4f}, gamma={self.gamma:.3f}, h0={self.h0:.3e})"
        )


@dataclass(frozen=True)
class GJRGARCHRiskNeutralModel(_GARCHRiskNeutralMixin, Model):
    """Risk-neutral GJR-GARCH(1,1) — leverage via a negative-shock indicator."""

    omega: float
    alpha: float
    beta: float
    gamma: float
    h0: float
    steps_per_year: int = TRADING_DAYS_PER_YEAR

    _GARCH_TYPE: ClassVar[str] = "gjr_garch"
    _DISPLAY: ClassVar[str] = "GJR-GARCH (risk-neutral)"
    _PARAM_NAMES: ClassVar[tuple[str, ...]] = (
        "omega", "alpha", "beta", "gamma", "h0",
    )

    def __post_init__(self) -> None:
        self._validate()

    def __repr__(self) -> str:
        return (
            f"GJRGARCHRiskNeutralModel(omega={self.omega:.3e}, alpha={self.alpha:.3e}, "
            f"beta={self.beta:.4f}, gamma={self.gamma:.3f}, h0={self.h0:.3e})"
        )
