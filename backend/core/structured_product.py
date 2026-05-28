"""
Structured Product Interfaces
=============================

Abstract base classes for structured products — a parallel hierarchy to
Instrument/Payoff designed for multi-component, path-dependent products
with discrete observation dates and potential early termination.

Architecture:
- ObservationSchedule: Discrete observation dates
- ProductComponent: Atomic building block (bond floor, coupon, barrier, etc.)
- StructuredProduct: ABC assembling components into a tradeable product

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from backend.core.result_types import StructuredProductPricingComponents


# =============================================================================
# Observation Schedule
# =============================================================================


@dataclass(frozen=True)
class ObservationSchedule:
    """
    Discrete observation dates for structured products.

    Parameters
    ----------
    times : tuple[float, ...]
        Observation times in years since valuation date.
    frequency : str
        Observation frequency: 'monthly', 'quarterly', 'semi_annual', 'annual'.
    """

    times: tuple[float, ...]
    frequency: str

    def __post_init__(self) -> None:
        if not self.times:
            raise ValueError("Observation schedule must have at least one time")
        if any(t <= 0 for t in self.times):
            raise ValueError("All observation times must be positive")
        if list(self.times) != sorted(self.times):
            raise ValueError("Observation times must be sorted in ascending order")
        valid_freqs = ("monthly", "quarterly", "semi_annual", "annual", "custom")
        if self.frequency not in valid_freqs:
            raise ValueError(
                f"Frequency must be one of {valid_freqs}, got '{self.frequency}'"
            )

    @classmethod
    def from_frequency(
        cls,
        maturity: float,
        frequency: str,
        first_observation: float | None = None,
    ) -> ObservationSchedule:
        """
        Generate observation schedule from frequency and maturity.

        Parameters
        ----------
        maturity : float
            Product maturity in years.
        frequency : str
            'monthly', 'quarterly', 'semi_annual', 'annual'.
        first_observation : float, optional
            Time of first observation. Defaults to one period from now.

        Returns
        -------
        ObservationSchedule
        """
        freq_map = {
            "monthly": 1 / 12,
            "quarterly": 0.25,
            "semi_annual": 0.5,
            "annual": 1.0,
        }
        if frequency not in freq_map:
            raise ValueError(
                f"Frequency must be one of {list(freq_map.keys())}, got '{frequency}'"
            )

        period = freq_map[frequency]
        start = first_observation if first_observation is not None else period

        times: list[float] = []
        t = start
        while t <= maturity + 1e-10:
            times.append(round(t, 10))
            t += period

        if not times:
            raise ValueError(
                f"No observation dates generated for maturity={maturity}, "
                f"frequency='{frequency}', first_observation={first_observation}"
            )

        return cls(times=tuple(times), frequency=frequency)

    @property
    def n_observations(self) -> int:
        """Number of observation dates."""
        return len(self.times)

    @property
    def last_observation(self) -> float:
        """Time of last observation."""
        return self.times[-1]

    def __repr__(self) -> str:
        return (
            f"ObservationSchedule(n={self.n_observations}, "
            f"freq='{self.frequency}', "
            f"T=[{self.times[0]:.2f}, ..., {self.times[-1]:.2f}])"
        )


# =============================================================================
# Evaluation Context (per-pricing-call mutable state)
# =============================================================================


@dataclass
class EvaluationContext:
    """
    Mutable per-pricing-call state shared between components.

    Created once in _price_generic(), consumed by evaluate_in_context().
    Function-local: created and destroyed in a single price_structured() call.
    """

    # Inputs (set at construction, treated as read-only by components)
    paths: np.ndarray  # (n_paths, n_steps+1)
    time_grid: np.ndarray  # (n_steps+1,)
    obs_indices: np.ndarray  # (n_obs,)
    discount_factors: np.ndarray  # (n_obs,)
    obs_dt: np.ndarray  # (n_obs,)
    df_terminal: float
    s0: np.ndarray  # (n_paths,) — spot de référence

    # Mutable shared state
    alive: np.ndarray = field(init=False)  # bool (n_paths,)
    terminated_at_obs: np.ndarray = field(init=False)  # int (n_paths,)
    ki_breached: np.ndarray = field(init=False)  # bool (n_paths,)
    coupon_accrued: np.ndarray = field(init=False)  # float (n_paths,)

    # Phase 2 state fields
    cmi_unpaid_count: np.ndarray = field(
        init=False
    )  # int (n_paths,) — consecutive missed CMI coupons

    def __post_init__(self) -> None:
        n_paths = self.paths.shape[0]
        self.alive = np.ones(n_paths, dtype=bool)
        self.terminated_at_obs = np.full(n_paths, -1, dtype=np.int64)
        self.ki_breached = np.zeros(n_paths, dtype=bool)
        self.coupon_accrued = np.zeros(n_paths, dtype=np.float64)
        self.cmi_unpaid_count = np.zeros(n_paths, dtype=np.int64)

    @classmethod
    def create(
        cls,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
        obs_dt: np.ndarray,
        df_terminal: float,
        s0_reference: float = -1.0,
    ) -> EvaluationContext:
        """Factory with s0_reference logic (use paths[:,0] if -1)."""
        if s0_reference > 0:
            s0 = np.full(paths.shape[0], s0_reference)
        else:
            s0 = paths[:, 0].copy()
        return cls(
            paths=paths,
            time_grid=time_grid,
            obs_indices=obs_indices,
            discount_factors=discount_factors,
            obs_dt=obs_dt,
            df_terminal=df_terminal,
            s0=s0,
        )

    @property
    def n_paths(self) -> int:
        return self.paths.shape[0]

    @property
    def n_obs(self) -> int:
        return len(self.obs_indices)

    def performance_at(self, obs_idx: int) -> np.ndarray:
        """S(t_obs) / S0 for all paths at observation index obs_idx."""
        return self.paths[:, self.obs_indices[obs_idx]] / self.s0

    def terminal_performance(self) -> np.ndarray:
        """S(T) / S0 for all paths."""
        return self.paths[:, -1] / self.s0


# =============================================================================
# Product Component (Building Block)
# =============================================================================


class ProductComponent(ABC):
    """
    Atomic building block of a structured product.

    Each component evaluates its contribution to the product's present value
    given simulated paths, observation dates, and discount factors.
    """

    @abstractmethod
    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate this component's present value contribution per path.

        Parameters
        ----------
        paths : np.ndarray
            Simulated price paths, shape (n_paths, n_steps+1).
        time_grid : np.ndarray
            Time grid corresponding to path columns, shape (n_steps+1,).
        obs_indices : np.ndarray
            Indices into time_grid for each observation date, shape (n_obs,).
        discount_factors : np.ndarray
            Discount factors D(0, t) for each observation date, shape (n_obs,).

        Returns
        -------
        np.ndarray
            Present value contribution per path, shape (n_paths,).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable component name."""

    def evaluate_in_context(self, ctx: EvaluationContext) -> np.ndarray:
        """
        Evaluate this component using shared EvaluationContext.

        Default: delegates to legacy evaluate(). Override for components
        that need to read/write shared state (alive mask, ki_breached, etc.).
        """
        return self.evaluate(
            ctx.paths, ctx.time_grid, ctx.obs_indices, ctx.discount_factors
        )


# =============================================================================
# Structured Product ABC
# =============================================================================


class StructuredProduct(ABC):
    """
    Abstract base class for all structured products.

    A structured product is an assembly of ProductComponents evaluated
    at discrete observation dates with a notional amount.
    """

    @property
    @abstractmethod
    def notional(self) -> float:
        """Notional amount of the product."""

    @property
    @abstractmethod
    def maturity(self) -> float:
        """Product maturity in years."""

    @property
    @abstractmethod
    def components(self) -> list[ProductComponent]:
        """List of product components."""

    @property
    @abstractmethod
    def observation_schedule(self) -> ObservationSchedule:
        """Discrete observation schedule."""

    @property
    @abstractmethod
    def product_type(self) -> str:
        """Product category: 'capital_protected', 'yield_enhancement', 'participation'."""

    @abstractmethod
    def has_early_termination(self) -> bool:
        """Whether the product can terminate before maturity (e.g. autocall)."""

    def evaluate_paths(
        self,
        paths: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
        obs_dt: np.ndarray,
        df_terminal: float,
        s0_reference: float = -1.0,
    ) -> StructuredProductPricingComponents | None:
        """
        Evaluate this product's pricing components from simulated paths.

        Subclasses override this for product-specific evaluation kernels.
        Default implementation uses generic component-by-component evaluation.

        Parameters
        ----------
        paths : np.ndarray
            Simulated price paths, shape (n_paths, n_steps+1).
        obs_indices : np.ndarray
            Indices into paths for each observation date.
        discount_factors : np.ndarray
            Discount factors at observation dates.
        obs_dt : np.ndarray
            Period lengths between observations.
        df_terminal : float
            Terminal discount factor.
        s0_reference : float
            Reference spot for performance calculation (-1.0 = use paths[:,0]).

        Returns
        -------
        StructuredProductPricingComponents | None
            Keys: 'pv', 'bond_floor_pv', 'option_pv', 'coupon_pv',
                  'autocall_probability'. None signals engine to use generic fallback.
        """
        return None  # Signals engine to use generic fallback

    @property
    def name(self) -> str:
        """Human-readable product name."""
        return self.__class__.__name__


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Structured Product ABCs Smoke Test")
    print("=" * 50)

    # Test ObservationSchedule.from_frequency
    print("\n--- ObservationSchedule ---")
    sched_q = ObservationSchedule.from_frequency(2.0, "quarterly")
    print(f"Quarterly over 2y: {sched_q}")
    print(f"  Times: {sched_q.times}")
    assert sched_q.n_observations == 8

    sched_a = ObservationSchedule.from_frequency(3.0, "annual")
    print(f"Annual over 3y: {sched_a}")
    assert sched_a.n_observations == 3
    assert sched_a.times == (1.0, 2.0, 3.0)

    sched_sa = ObservationSchedule.from_frequency(2.0, "semi_annual")
    print(f"Semi-annual over 2y: {sched_sa}")
    assert sched_sa.n_observations == 4

    sched_m = ObservationSchedule.from_frequency(1.0, "monthly")
    print(f"Monthly over 1y: {sched_m}")
    assert sched_m.n_observations == 12

    # Test with custom first observation
    sched_custom = ObservationSchedule.from_frequency(
        3.0, "annual", first_observation=0.5
    )
    print(f"Annual from 0.5y: {sched_custom}")
    assert sched_custom.times[0] == 0.5

    # Test validation
    try:
        ObservationSchedule(times=(), frequency="annual")
        print("ERROR: Should have raised")
    except ValueError:
        print("Empty schedule rejected: OK")

    try:
        ObservationSchedule(times=(-0.5,), frequency="annual")
        print("ERROR: Should have raised")
    except ValueError:
        print("Negative time rejected: OK")

    print("\n" + "=" * 50)
    print("Structured Product ABCs smoke test passed")
    print("=" * 50)
