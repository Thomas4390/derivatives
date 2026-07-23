"""
Base Classes for Monte Carlo Simulation
=======================================

This module provides abstract base classes and result containers for all
simulation models in the framework.

Architecture:
- BaseSimulator: Abstract interface for all price/volatility simulators
- SimulationResult: Container for simulation results with computed properties
- VolatilityResult: Extended result for models that output volatility paths

All concrete simulators must inherit from BaseSimulator and implement:
- simulate_paths(): Full path simulation
- simulate_terminal(): Memory-efficient terminal-only simulation

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# =============================================================================
# Result Containers
# =============================================================================


@dataclass
class SimulationResult:
    """
    Container for simulation results with metadata and computed properties.

    Attributes
    ----------
    price_paths : np.ndarray
        Simulated price paths, shape (n_paths, n_steps + 1)
    time_grid : np.ndarray
        Time points corresponding to path columns, shape (n_steps + 1,)
    model_name : str
        Name of the model used for simulation
    computation_time : float
        Wall-clock time for simulation in seconds
    n_paths : int
        Number of simulated paths
    n_steps : int
        Number of time steps per path
    volatility_paths : np.ndarray, optional
        Volatility paths for stochastic vol models, shape (n_paths, n_steps + 1)
    parameters : dict
        Model parameters used for simulation
    """

    price_paths: np.ndarray
    time_grid: np.ndarray
    model_name: str
    computation_time: float
    n_paths: int
    n_steps: int
    volatility_paths: np.ndarray | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def terminal_prices(self) -> np.ndarray:
        """Returns terminal prices S(T) for all paths."""
        return self.price_paths[:, -1]

    @property
    def initial_price(self) -> float:
        """Returns the initial price S(0)."""
        return self.price_paths[0, 0]

    @property
    def mean_path(self) -> np.ndarray:
        """Returns the mean price path across all simulations."""
        return np.mean(self.price_paths, axis=0)

    @property
    def std_path(self) -> np.ndarray:
        """Returns the standard deviation of prices at each time step."""
        return np.std(self.price_paths, axis=0)

    @property
    def terminal_mean(self) -> float:
        """Returns mean of terminal prices."""
        return float(np.mean(self.terminal_prices))

    @property
    def terminal_std(self) -> float:
        """Returns standard deviation of terminal prices."""
        return float(np.std(self.terminal_prices))

    @property
    def terminal_volatility(self) -> np.ndarray | None:
        """Returns terminal volatility values if available."""
        if self.volatility_paths is not None:
            return self.volatility_paths[:, -1]
        return None

    @property
    def mean_volatility_path(self) -> np.ndarray | None:
        """Returns mean volatility path if available."""
        if self.volatility_paths is not None:
            return np.mean(self.volatility_paths, axis=0)
        return None

    def percentile_paths(self, percentiles: list[float]) -> np.ndarray:
        """
        Compute percentile paths across simulations.

        Parameters
        ----------
        percentiles : list
            Percentiles to compute (e.g., [5, 50, 95])

        Returns
        -------
        np.ndarray
            Percentile values at each time step, shape (len(percentiles), n_steps + 1)
        """
        return np.percentile(self.price_paths, percentiles, axis=0)

    def log_returns(self) -> np.ndarray:
        """
        Compute log returns from price paths.

        Returns
        -------
        np.ndarray
            Log returns, shape (n_paths, n_steps)
        """
        return np.diff(np.log(self.price_paths), axis=1)

    def realized_volatility(
        self, annualization_factor: float | None = None
    ) -> np.ndarray:
        """
        Compute realized volatility for each path.

        Parameters
        ----------
        annualization_factor : float, optional
            Factor to annualize the per-step return std. If None (default), it
            is derived from the actual time step as ``1/sqrt(dt)`` with
            ``dt = T / n_steps`` (from ``time_grid``), which is correct for any
            step size. The previous hard-coded ``sqrt(252)`` was only valid for
            daily steps and silently mis-scaled every other ``dt``.

        Returns
        -------
        np.ndarray
            Realized volatility for each path, shape (n_paths,)
        """
        log_rets = self.log_returns()
        if annualization_factor is None:
            dt = float(self.time_grid[-1]) / self.n_steps if self.n_steps > 0 else 0.0
            annualization_factor = 1.0 / np.sqrt(dt) if dt > 0 else 1.0
        return np.std(log_rets, axis=1) * annualization_factor

    @property
    def has_volatility(self) -> bool:
        """Returns True if volatility paths are available."""
        return self.volatility_paths is not None

    def __repr__(self) -> str:
        vol_info = f", has_vol={self.has_volatility}" if self.has_volatility else ""
        return (
            f"SimulationResult(model='{self.model_name}', "
            f"n_paths={self.n_paths:,}, n_steps={self.n_steps}, "
            f"time={self.computation_time * 1000:.1f}ms{vol_info})"
        )


# =============================================================================
# Abstract Base Class
# =============================================================================


class BaseSimulator(ABC):
    """
    Abstract base class for all Monte Carlo simulators.

    All concrete simulator classes must inherit from this class and implement
    the required abstract methods. This ensures a consistent interface across
    all model types.

    The design follows the Strategy pattern, allowing different simulation
    algorithms to be used interchangeably through a common interface.

    Attributes
    ----------
    model_name : str
        Human-readable name of the model

    Methods
    -------
    simulate_paths(s0, mu, t, n_paths, n_steps, seed=None) -> SimulationResult
        Simulate full price (and optionally volatility) paths
    simulate_terminal(s0, mu, t, n_paths, n_steps, seed=None) -> np.ndarray
        Simulate only terminal values (memory-efficient)
    get_parameters() -> dict
        Return current model parameters
    """

    def __init__(self) -> None:
        self._model_name: str = "BaseSimulator"

    @property
    def model_name(self) -> str:
        """Returns the model name."""
        return self._model_name

    @abstractmethod
    def simulate_paths(
        self,
        s0: float,
        mu: float,
        t: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
    ) -> SimulationResult:
        """
        Simulate full price paths under the P-measure (physical measure).

        Parameters
        ----------
        s0 : float
            Initial asset price
        mu : float
            Expected return / drift (annualized)
        t : float
            Time horizon in years
        n_paths : int
            Number of Monte Carlo paths
        n_steps : int
            Number of time steps per path
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        SimulationResult
            Container with price paths, time grid, and metadata
        """
        pass

    @abstractmethod
    def simulate_terminal(
        self,
        s0: float,
        mu: float,
        t: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Simulate only terminal values S(T).

        This method is optimized for memory efficiency when only terminal
        values are needed (e.g., European option pricing).

        Parameters
        ----------
        s0 : float
            Initial asset price
        mu : float
            Expected return / drift (annualized)
        t : float
            Time horizon in years
        n_paths : int
            Number of Monte Carlo paths
        n_steps : int
            Number of time steps per path
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        np.ndarray
            Terminal values S(T), shape (n_paths,)
        """
        pass

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """
        Return current model parameters.

        Returns
        -------
        dict
            Dictionary of parameter names and values
        """
        pass

    def validate_inputs(
        self, s0: float, mu: float, t: float, n_paths: int, n_steps: int
    ) -> None:
        """
        Validate common simulation inputs.

        Parameters
        ----------
        s0 : float
            Initial price (must be positive)
        mu : float
            Drift (any real number)
        t : float
            Time horizon (must be positive)
        n_paths : int
            Number of paths (must be positive)
        n_steps : int
            Number of steps (must be positive)

        Raises
        ------
        ValueError
            If any input is invalid
        """
        if s0 <= 0:
            raise ValueError(f"Initial price s0 must be positive, got {s0}")
        if t <= 0:
            raise ValueError(f"Time horizon t must be positive, got {t}")
        if n_paths <= 0:
            raise ValueError(f"Number of paths must be positive, got {n_paths}")
        if n_steps <= 0:
            raise ValueError(f"Number of steps must be positive, got {n_steps}")

    def __repr__(self) -> str:
        params = self.get_parameters()
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"


# =============================================================================
# Mixin for Stochastic Volatility Models
# =============================================================================


class StochasticVolatilityMixin:
    """
    Mixin providing additional methods for stochastic volatility models.

    Models that output volatility paths (Heston, Bates, GARCH variants)
    should inherit from this mixin in addition to BaseSimulator.
    """

    def long_run_variance(self) -> float:
        """
        Compute the theoretical long-run variance.

        Returns
        -------
        float
            Long-run variance (σ²_∞)
        """
        raise NotImplementedError("Subclass must implement long_run_variance()")

    def long_run_volatility(self) -> float:
        """
        Compute the theoretical long-run volatility.

        Returns
        -------
        float
            Long-run volatility (σ_∞)
        """
        return np.sqrt(self.long_run_variance())

    def feller_condition_satisfied(self) -> bool:
        """
        Check if the Feller condition is satisfied (for CIR-type variance).

        The Feller condition (2*kappa*theta > alpha^2) ensures variance stays positive.

        Returns
        -------
        bool
            True if Feller condition is satisfied
        """
        raise NotImplementedError(
            "Subclass must implement feller_condition_satisfied()"
        )


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Simulation Base Classes Smoke Test")
    print("=" * 50)

    # Test SimulationResult
    print("\n--- SimulationResult ---")

    # Create mock simulation data
    n_paths, n_steps = 1000, 100
    np.random.seed(42)

    # Simulate simple GBM-like paths
    s0 = 100.0
    t = 1.0
    dt = t / n_steps
    time_grid = np.linspace(0, t, n_steps + 1)

    # Generate log returns
    log_returns = np.random.normal(0.05 * dt, 0.2 * np.sqrt(dt), (n_paths, n_steps))
    log_prices = np.zeros((n_paths, n_steps + 1))
    log_prices[:, 0] = np.log(s0)
    log_prices[:, 1:] = np.log(s0) + np.cumsum(log_returns, axis=1)
    price_paths = np.exp(log_prices)

    # Create volatility paths (mock)
    vol_paths = np.ones((n_paths, n_steps + 1)) * 0.2

    # Create result
    result = SimulationResult(
        price_paths=price_paths,
        time_grid=time_grid,
        model_name="TestModel",
        computation_time=0.1,
        n_paths=n_paths,
        n_steps=n_steps,
        volatility_paths=vol_paths,
        parameters={"sigma": 0.2},
    )

    print(f"Result: {result}")
    print(f"Terminal prices shape: {result.terminal_prices.shape}")
    print(f"Initial price: {result.initial_price:.2f}")
    print(f"Terminal mean: {result.terminal_mean:.2f}")
    print(f"Terminal std: {result.terminal_std:.2f}")
    print(f"Mean path shape: {result.mean_path.shape}")
    print(f"Has volatility: {result.has_volatility}")

    # Test computed properties
    print("\n--- Computed Properties ---")
    log_rets = result.log_returns()
    print(f"Log returns shape: {log_rets.shape}")
    assert log_rets.shape == (n_paths, n_steps), "Log returns shape mismatch"

    realized_vol = result.realized_volatility()
    print(f"Realized volatility shape: {realized_vol.shape}")
    print(f"Mean realized vol: {np.mean(realized_vol):.4f}")

    percentiles = result.percentile_paths([5, 50, 95])
    print(f"Percentile paths shape: {percentiles.shape}")
    assert percentiles.shape == (3, n_steps + 1), "Percentile shape mismatch"

    # Test volatility properties
    print("\n--- Volatility Properties ---")
    terminal_vol = result.terminal_volatility
    print(f"Terminal volatility shape: {terminal_vol.shape}")

    mean_vol_path = result.mean_volatility_path
    print(f"Mean volatility path shape: {mean_vol_path.shape}")

    # Test without volatility
    print("\n--- Result Without Volatility ---")
    result_no_vol = SimulationResult(
        price_paths=price_paths,
        time_grid=time_grid,
        model_name="NoVolModel",
        computation_time=0.05,
        n_paths=n_paths,
        n_steps=n_steps,
    )
    print(f"Has volatility: {result_no_vol.has_volatility}")
    assert result_no_vol.terminal_volatility is None, "Should be None"
    assert result_no_vol.mean_volatility_path is None, "Should be None"
    print("No volatility handling: ✓")

    # Test BaseSimulator validation
    print("\n--- BaseSimulator Validation ---")

    class DummySimulator(BaseSimulator):
        """Minimal concrete implementation for testing."""

        def __init__(self) -> None:
            super().__init__()
            self._model_name = "DummySimulator"

        def simulate_paths(
            self,
            s0: float,
            mu: float,
            t: float,
            n_paths: int,
            n_steps: int,
            seed: int | None = None,
        ) -> None:  # type: ignore[override]
            return None

        def simulate_terminal(
            self,
            s0: float,
            mu: float,
            t: float,
            n_paths: int,
            n_steps: int,
            seed: int | None = None,
        ) -> None:  # type: ignore[override]
            return None

        def get_parameters(self) -> dict[str, Any]:
            return {"test": 1}

    simulator = DummySimulator()
    print(f"Simulator: {simulator}")

    # Test valid inputs
    try:
        simulator.validate_inputs(100.0, 0.05, 1.0, 1000, 252)
        print("Valid inputs accepted: ✓")
    except ValueError as e:
        print(f"ERROR: Valid inputs rejected: {e}")

    # Test invalid inputs
    invalid_cases = [
        (-100.0, 0.05, 1.0, 1000, 252, "negative s0"),
        (100.0, 0.05, -1.0, 1000, 252, "negative t"),
        (100.0, 0.05, 1.0, -1000, 252, "negative n_paths"),
        (100.0, 0.05, 1.0, 1000, -252, "negative n_steps"),
    ]

    for s0, mu, t, n_paths, n_steps, desc in invalid_cases:
        try:
            simulator.validate_inputs(s0, mu, t, n_paths, n_steps)
            print(f"ERROR: Should have rejected {desc}")
        except ValueError:
            print(f"Correctly rejected {desc}: ✓")

    # Test StochasticVolatilityMixin
    print("\n--- StochasticVolatilityMixin ---")

    class VolSimulator(BaseSimulator, StochasticVolatilityMixin):
        """Test stochastic volatility simulator."""

        def __init__(self) -> None:
            super().__init__()
            self._model_name = "VolSimulator"

        def simulate_paths(
            self,
            s0: float,
            mu: float,
            t: float,
            n_paths: int,
            n_steps: int,
            seed: int | None = None,
        ) -> None:  # type: ignore[override]
            return None

        def simulate_terminal(
            self,
            s0: float,
            mu: float,
            t: float,
            n_paths: int,
            n_steps: int,
            seed: int | None = None,
        ) -> None:  # type: ignore[override]
            return None

        def get_parameters(self) -> dict[str, Any]:
            return {"v0": 0.04}

        def long_run_variance(self) -> float:
            return 0.04

        def feller_condition_satisfied(self) -> bool:
            return True

    vol_sim = VolSimulator()
    print(f"Long-run variance: {vol_sim.long_run_variance():.4f}")
    print(f"Long-run volatility: {vol_sim.long_run_volatility():.4f}")
    print(f"Feller satisfied: {vol_sim.feller_condition_satisfied()}")

    print("\n" + "=" * 50)
    print("Simulation Base Classes smoke test passed")
    print("=" * 50)
