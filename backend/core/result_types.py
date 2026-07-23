"""
Result Types
=============

Enums and result dataclasses used throughout the pricing system.

Note: This module is named 'result_types' to avoid conflict with the
standard library 'types' module.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TypedDict

import numpy as np

# =============================================================================
# Enums
# =============================================================================


class ExerciseStyle(Enum):
    """Option exercise style."""

    EUROPEAN = auto()
    AMERICAN = auto()
    BERMUDAN = auto()


class PricingCapability(Enum):
    """Type of pricing engine capability."""

    ANALYTICAL = auto()
    FFT = auto()
    MONTE_CARLO = auto()


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass(frozen=True)
class PricingResult:
    """
    Unified result of pricing any Priceable (vanilla or structured product).

    Attributes
    ----------
    price : float
        Absolute price (premium for vanilla, notional-based for structured).
    engine : str
        Name of the engine that produced the result.
    model : str
        Name of the model used.
    error : float, optional
        Estimation error (for MC methods).
    metadata : dict, optional
        Additional engine-specific metadata.

    Structured Product Fields (None for vanilla)
    ---------------------------------------------
    fair_value_pct : float, optional
        Present value as % of notional (e.g., 98.5 means 98.5%).
    notional : float, optional
        Product notional amount.
    bond_floor_pct : float, optional
        PV of the bond component (% of notional).
    option_value_pct : float, optional
        PV of the option component (% of notional).
    expected_coupon_pct : float, optional
        PV of expected coupons (% of notional).
    autocall_probability : float, optional
        Probability of early termination via autocall.
    capital_loss_probability : float, optional
        Probability of receiving less than notional at maturity.
    expected_return : float, optional
        Expected annualized return.
    worst_case_return : float, optional
        5th percentile return.
    best_case_return : float, optional
        95th percentile return.
    """

    price: float
    engine: str = ""
    model: str = ""
    error: float | None = None
    metadata: dict[str, float] | None = None

    # Structured product decomposition (None for vanilla instruments)
    fair_value_pct: float | None = None
    notional: float | None = None
    bond_floor_pct: float | None = None
    option_value_pct: float | None = None
    expected_coupon_pct: float | None = None
    autocall_probability: float | None = None
    capital_loss_probability: float | None = None
    expected_return: float | None = None
    worst_case_return: float | None = None
    best_case_return: float | None = None

    @property
    def is_structured(self) -> bool:
        """True if this result contains structured product decomposition."""
        return self.notional is not None

    def __repr__(self) -> str:
        err_str = f", error={self.error:.6f}" if self.error is not None else ""
        if self.is_structured:
            fv = self.fair_value_pct
            fv_str = f"fair_value={fv:.2f}%" if fv is not None else "fair_value=N/A"
            return f"PricingResult(price={self.price:.2f}, {fv_str}{err_str})"
        return f"PricingResult(price={self.price:.6f}{err_str})"


@dataclass(frozen=True)
class GreeksResult:
    """
    Portfolio or option Greeks.

    All Greeks are per-unit values (multiply by quantity for portfolio totals).

    Price
    -----
    price : float
        Option price (0.0 when not computed, e.g. for portfolio aggregation).

    First Order Greeks (∂V/∂x)
    --------------------------
    delta : float
        ∂V/∂S - sensitivity to spot price
    theta : float
        ∂V/∂t - time decay (typically negative for long options)
    vega : float
        ∂V/∂σ - sensitivity to volatility (often scaled by 1/100)
    rho : float
        ∂V/∂r - sensitivity to interest rate (often scaled by 1/100)

    Second Order Greeks (∂²V/∂x∂y)
    ------------------------------
    gamma : float
        ∂²V/∂S² - sensitivity of delta to spot (convexity)
    vanna : float
        ∂²V/∂S∂σ - sensitivity of delta to volatility
        Also equals ∂vega/∂S (sensitivity of vega to spot)
    volga : float
        ∂²V/∂σ² - sensitivity of vega to volatility (also called vomma)
    charm : float
        ∂²V/∂S∂t - decay of delta over time (delta bleed)
        Also called delta decay
    veta : float
        ∂²V/∂σ∂t - sensitivity of vega to time

    Third Order Greeks (∂³V/∂x∂y∂z)
    -------------------------------
    speed : float
        ∂³V/∂S³ - rate of change of gamma with respect to spot
        Important for delta hedging of large positions
    zomma : float
        ∂³V/∂S²∂σ - rate of change of gamma with respect to volatility
    color : float
        ∂³V/∂S²∂t - decay of gamma over time
    ultima : float
        ∂³V/∂σ³ - third order sensitivity to volatility
    """

    # Price
    price: float = 0.0
    # First order
    delta: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    # Second order
    gamma: float = 0.0
    vanna: float = 0.0
    volga: float = 0.0
    charm: float = 0.0
    veta: float = 0.0
    # Third order
    speed: float = 0.0
    zomma: float = 0.0
    color: float = 0.0
    ultima: float = 0.0

    @property
    def has_higher_order(self) -> bool:
        """True if any second/third-order Greek beyond gamma is populated."""
        return any(
            [
                self.vanna != 0.0,
                self.volga != 0.0,
                self.charm != 0.0,
                self.veta != 0.0,
                self.speed != 0.0,
                self.zomma != 0.0,
                self.color != 0.0,
                self.ultima != 0.0,
            ]
        )

    def __add__(self, other: "GreeksResult") -> "GreeksResult":
        """Sum Greeks (for portfolio aggregation)."""
        return GreeksResult(
            price=self.price + other.price,
            # First order
            delta=self.delta + other.delta,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho,
            # Second order
            gamma=self.gamma + other.gamma,
            vanna=self.vanna + other.vanna,
            volga=self.volga + other.volga,
            charm=self.charm + other.charm,
            veta=self.veta + other.veta,
            # Third order
            speed=self.speed + other.speed,
            zomma=self.zomma + other.zomma,
            color=self.color + other.color,
            ultima=self.ultima + other.ultima,
        )

    def __mul__(self, scalar: float) -> "GreeksResult":
        """Scale Greeks by a scalar."""
        return GreeksResult(
            price=self.price * scalar,
            # First order
            delta=self.delta * scalar,
            theta=self.theta * scalar,
            vega=self.vega * scalar,
            rho=self.rho * scalar,
            # Second order
            gamma=self.gamma * scalar,
            vanna=self.vanna * scalar,
            volga=self.volga * scalar,
            charm=self.charm * scalar,
            veta=self.veta * scalar,
            # Third order
            speed=self.speed * scalar,
            zomma=self.zomma * scalar,
            color=self.color * scalar,
            ultima=self.ultima * scalar,
        )

    def __rmul__(self, scalar: float) -> "GreeksResult":
        return self.__mul__(scalar)

    def __sub__(self, other: "GreeksResult") -> "GreeksResult":
        """Subtract Greeks (for hedging calculations)."""
        return GreeksResult(
            price=self.price - other.price,
            # First order
            delta=self.delta - other.delta,
            theta=self.theta - other.theta,
            vega=self.vega - other.vega,
            rho=self.rho - other.rho,
            # Second order
            gamma=self.gamma - other.gamma,
            vanna=self.vanna - other.vanna,
            volga=self.volga - other.volga,
            charm=self.charm - other.charm,
            veta=self.veta - other.veta,
            # Third order
            speed=self.speed - other.speed,
            zomma=self.zomma - other.zomma,
            color=self.color - other.color,
            ultima=self.ultima - other.ultima,
        )

    def __truediv__(self, scalar: float) -> "GreeksResult":
        """Divide Greeks by a scalar (for normalization)."""
        return self * (1.0 / scalar)

    def __neg__(self) -> "GreeksResult":
        """Negate Greeks (for short positions)."""
        return self * (-1.0)

    @property
    def vomma(self) -> float:
        """Alias for volga (∂²V/∂σ²)."""
        return self.volga

    @property
    def delta_decay(self) -> float:
        """Alias for charm (∂²V/∂S∂t)."""
        return self.charm

    def first_order(self) -> dict[str, float]:
        """Return only first-order Greeks."""
        return {
            "delta": self.delta,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
        }

    def second_order(self) -> dict[str, float]:
        """Return only second-order Greeks."""
        return {
            "gamma": self.gamma,
            "vanna": self.vanna,
            "volga": self.volga,
            "charm": self.charm,
            "veta": self.veta,
        }

    def third_order(self) -> dict[str, float]:
        """Return only third-order Greeks."""
        return {
            "speed": self.speed,
            "zomma": self.zomma,
            "color": self.color,
            "ultima": self.ultima,
        }

    def to_dict(self) -> dict[str, float]:
        """Convert all Greeks to dictionary."""
        return {
            # First order
            "delta": self.delta,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
            # Second order
            "gamma": self.gamma,
            "vanna": self.vanna,
            "volga": self.volga,
            "charm": self.charm,
            "veta": self.veta,
            # Third order
            "speed": self.speed,
            "zomma": self.zomma,
            "color": self.color,
            "ultima": self.ultima,
        }


@dataclass(frozen=True)
class StructuredProductResult(PricingResult):
    """
    Backward-compatible wrapper for structured product results.

    New code should use :class:`PricingResult` directly with the structured
    fields populated. This subclass adds legacy attribute aliases
    (``fair_value``, ``bond_floor`` …) as read-only properties and a
    :meth:`create` factory that maps the legacy constructor signature to the
    new PricingResult fields.

    The class itself is a normal frozen dataclass — ``__init__`` is the one
    generated by ``@dataclass`` so the previous ``object.__setattr__``
    boilerplate is gone.
    """

    @classmethod
    def create(
        cls,
        fair_value: float,
        price: float,
        notional: float,
        engine: str = "",
        model: str = "",
        error: float | None = None,
        bond_floor: float = 0.0,
        option_value: float = 0.0,
        expected_coupon: float = 0.0,
        autocall_probability: float = 0.0,
        capital_loss_probability: float = 0.0,
        expected_return: float = 0.0,
        worst_case_return: float = 0.0,
        best_case_return: float = 0.0,
    ) -> "StructuredProductResult":
        """Build a :class:`StructuredProductResult` from the legacy keyword arguments.

        The factory translates the legacy ``fair_value``/``bond_floor``/...
        names into the canonical ``*_pct`` fields used by ``PricingResult``.
        """
        return cls(
            price=price,
            engine=engine,
            model=model,
            error=error,
            metadata=None,
            fair_value_pct=fair_value,
            notional=notional,
            bond_floor_pct=bond_floor,
            option_value_pct=option_value,
            expected_coupon_pct=expected_coupon,
            autocall_probability=autocall_probability,
            capital_loss_probability=capital_loss_probability,
            expected_return=expected_return,
            worst_case_return=worst_case_return,
            best_case_return=best_case_return,
        )

    # Legacy attribute aliases (read-only properties)
    @property
    def fair_value(self) -> float:
        """PV as % of notional (legacy alias for fair_value_pct)."""
        # Coerce None -> 0.0 to honour the float annotation (and avoid a None
        # crash in __repr__), matching the bond_floor/option_value aliases.
        return self.fair_value_pct or 0.0

    @property
    def bond_floor(self) -> float:
        """Bond floor % of notional (legacy alias for bond_floor_pct)."""
        return self.bond_floor_pct or 0.0

    @property
    def option_value(self) -> float:
        """Option value % of notional (legacy alias for option_value_pct)."""
        return self.option_value_pct or 0.0

    @property
    def expected_coupon(self) -> float:
        """Expected coupon % of notional (legacy alias for expected_coupon_pct)."""
        return self.expected_coupon_pct or 0.0

    def __repr__(self) -> str:
        err_str = f", error={self.error:.6f}" if self.error is not None else ""
        return (
            f"StructuredProductResult(fair_value={self.fair_value:.2f}%, "
            f"price={self.price:.2f}{err_str})"
        )


class StructuredProductPricingComponents(TypedDict):
    """Type-safe return type for StructuredProduct.evaluate_paths()."""

    pv: np.ndarray
    bond_floor_pv: np.ndarray
    option_pv: np.ndarray
    coupon_pv: np.ndarray
    autocall_probability: float
