"""
Result Types
=============

Enums and result dataclasses used throughout the pricing system.

Note: This module is named 'result_types' to avoid conflict with the
standard library 'types' module.

Author: Thomas
Created: 2025
"""

from dataclasses import dataclass
from enum import Enum, auto

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
    Result of option pricing.

    Attributes
    ----------
    price : float
        Option price (premium)
    engine : str
        Name of the engine that produced the result
    model : str
        Name of the model used
    error : Optional[float]
        Estimation error (for MC methods)
    """
    price: float
    engine: str = ""
    model: str = ""
    error: float | None = None

    def __repr__(self) -> str:
        err_str = f", error={self.error:.6f}" if self.error else ""
        return f"PricingResult(price={self.price:.6f}{err_str})"


@dataclass(frozen=True)
class GreeksResult:
    """
    Portfolio or option Greeks.

    All Greeks are per-unit values (multiply by quantity for portfolio totals).

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

    def __add__(self, other: 'GreeksResult') -> 'GreeksResult':
        """Sum Greeks (for portfolio aggregation)."""
        return GreeksResult(
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

    def __mul__(self, scalar: float) -> 'GreeksResult':
        """Scale Greeks by a scalar."""
        return GreeksResult(
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

    def __rmul__(self, scalar: float) -> 'GreeksResult':
        return self.__mul__(scalar)

    def __sub__(self, other: 'GreeksResult') -> 'GreeksResult':
        """Subtract Greeks (for hedging calculations)."""
        return GreeksResult(
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

    def __truediv__(self, scalar: float) -> 'GreeksResult':
        """Divide Greeks by a scalar (for normalization)."""
        return self * (1.0 / scalar)

    def __neg__(self) -> 'GreeksResult':
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

    def first_order(self) -> dict:
        """Return only first-order Greeks."""
        return {
            "delta": self.delta,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
        }

    def second_order(self) -> dict:
        """Return only second-order Greeks."""
        return {
            "gamma": self.gamma,
            "vanna": self.vanna,
            "volga": self.volga,
            "charm": self.charm,
            "veta": self.veta,
        }

    def third_order(self) -> dict:
        """Return only third-order Greeks."""
        return {
            "speed": self.speed,
            "zomma": self.zomma,
            "color": self.color,
            "ultima": self.ultima,
        }

    def to_dict(self) -> dict:
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


if __name__ == "__main__":
    print("=" * 50)
    print("Result Types Smoke Test")
    print("=" * 50)

    result = PricingResult(price=5.123, engine="BS", model="GBM")
    print(f"\nPricing result: {result}")

    greeks = GreeksResult(
        # First order
        delta=0.55, theta=-0.05, vega=0.20, rho=0.15,
        # Second order
        gamma=0.02, vanna=0.03, volga=0.01, charm=-0.001, veta=-0.02,
        # Third order
        speed=0.001, zomma=0.0005, color=-0.0001, ultima=0.0003,
    )
    print("\nGreeks (all orders):")
    print(f"  1st order: {greeks.first_order()}")
    print(f"  2nd order: {greeks.second_order()}")
    print(f"  3rd order: {greeks.third_order()}")

    # Test aliases
    print("\nAliases:")
    print(f"  volga == vomma: {greeks.volga == greeks.vomma}")
    print(f"  charm == delta_decay: {greeks.charm == greeks.delta_decay}")

    # Test aggregation
    greeks2 = GreeksResult(delta=0.30, gamma=0.01, theta=-0.03, vega=0.10, rho=0.10)
    combined = greeks + greeks2
    print(f"\nCombined delta: {combined.delta:.2f}")

    # Test scaling
    scaled = greeks * 10
    print(f"Scaled delta (x10): {scaled.delta:.2f}")

    print("\n" + "=" * 50)
    print("Result types smoke test passed")
    print("=" * 50)
