"""
Monte Carlo Pricing Engine
==========================

Generic Monte Carlo pricing engine for European options.
This engine can price options for any model that provides a terminal price simulator.

The key insight is that the Monte Carlo payoff computation and discounting is the
same for all models - only the simulation of terminal prices changes.

Author: Derivatives Pricing Project
"""

import numpy as np
from numba import njit
from typing import Callable, Tuple, Optional, NamedTuple
from dataclasses import dataclass


@dataclass(frozen=True)
class MCConfig:
    """
    Configuration for Monte Carlo pricing.

    Parameters
    ----------
    n_paths : int
        Number of simulation paths (default 100,000)
    n_steps : int
        Number of time steps per path (default 252, i.e., daily for 1 year)
    antithetic : bool
        Use antithetic variates for variance reduction (default True)
    seed : int, optional
        Random seed for reproducibility
    """
    n_paths: int = 100_000
    n_steps: int = 252
    antithetic: bool = True
    seed: Optional[int] = None

    def __post_init__(self):
        if self.n_paths <= 0:
            raise ValueError(f"n_paths must be positive, got {self.n_paths}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")


class MCResult(NamedTuple):
    """Result from Monte Carlo pricing."""
    price: float
    std_error: float
    n_paths: int


# Type alias for terminal price simulator
# Takes (s0, t, r, n_paths, n_steps, seed) and returns terminal prices array
TerminalSimulator = Callable[[float, float, float, int, int, Optional[int]], np.ndarray]


@njit(fastmath=True, cache=True)
def _compute_payoffs(
    terminals: np.ndarray,
    k: float,
    is_call: bool
) -> np.ndarray:
    """Compute option payoffs from terminal prices."""
    n = len(terminals)
    payoffs = np.empty(n, dtype=np.float64)

    for i in range(n):
        if is_call:
            payoffs[i] = max(terminals[i] - k, 0.0)
        else:
            payoffs[i] = max(k - terminals[i], 0.0)

    return payoffs


@njit(fastmath=True, cache=True)
def _compute_price_and_se(
    payoffs: np.ndarray,
    discount: float
) -> Tuple[float, float]:
    """Compute discounted price and standard error."""
    n = len(payoffs)

    # Mean payoff
    mean_payoff = 0.0
    for i in range(n):
        mean_payoff += payoffs[i]
    mean_payoff = mean_payoff / n

    # Variance
    var_payoff = 0.0
    for i in range(n):
        diff = payoffs[i] - mean_payoff
        var_payoff += diff * diff
    var_payoff = var_payoff / (n - 1)

    # Discounted price and standard error
    price = discount * mean_payoff
    std_error = discount * np.sqrt(var_payoff / n)

    return price, std_error


class MonteCarloEngine:
    """
    Generic Monte Carlo pricing engine for European options.

    This engine prices European options using Monte Carlo simulation.
    It accepts a terminal price simulator function that generates
    terminal prices under the risk-neutral measure.

    Parameters
    ----------
    config : MCConfig, optional
        Monte Carlo configuration. Uses defaults if not provided.

    Examples
    --------
    # Create engine
    engine = MonteCarloEngine()

    # Define terminal simulator for your model
    def my_simulator(s0, t, r, n_paths, n_steps, seed=None):
        # ... simulate terminal prices under Q-measure
        return terminal_prices

    # Price an option
    result = engine.price(my_simulator, s0=100, k=100, t=0.25, r=0.05, is_call=True)
    print(f"Price: ${result.price:.4f} +/- ${result.std_error:.4f}")
    """

    def __init__(self, config: MCConfig = None):
        self._config = config or MCConfig()

    @property
    def config(self) -> MCConfig:
        """Returns the Monte Carlo configuration."""
        return self._config

    def price(
        self,
        terminal_simulator: TerminalSimulator,
        s0: float,
        k: float,
        t: float,
        r: float,
        is_call: bool = True,
        n_paths: Optional[int] = None,
        n_steps: Optional[int] = None,
        seed: Optional[int] = None
    ) -> MCResult:
        """
        Price a European option using Monte Carlo simulation.

        Parameters
        ----------
        terminal_simulator : callable
            Function that simulates terminal prices:
            simulator(s0, t, r, n_paths, n_steps, seed) -> np.ndarray
        s0 : float
            Spot price
        k : float
            Strike price
        t : float
            Time to maturity
        r : float
            Risk-free rate
        is_call : bool
            True for call, False for put
        n_paths : int, optional
            Number of paths (uses config default if not specified)
        n_steps : int, optional
            Number of steps (uses config default if not specified)
        seed : int, optional
            Random seed (uses config default if not specified)

        Returns
        -------
        MCResult
            Named tuple with (price, std_error, n_paths)
        """
        n_paths = n_paths or self._config.n_paths
        n_steps = n_steps or self._config.n_steps
        seed = seed if seed is not None else self._config.seed

        # Simulate terminal prices
        terminals = terminal_simulator(s0, t, r, n_paths, n_steps, seed)

        # Compute payoffs
        payoffs = _compute_payoffs(terminals, k, is_call)

        # Discount factor
        discount = np.exp(-r * t)

        # Compute price and standard error
        price, std_error = _compute_price_and_se(payoffs, discount)

        return MCResult(
            price=max(price, 0.0),
            std_error=std_error,
            n_paths=n_paths
        )

    def price_strikes(
        self,
        terminal_simulator: TerminalSimulator,
        s0: float,
        strikes: np.ndarray,
        t: float,
        r: float,
        is_call: bool = True,
        n_paths: Optional[int] = None,
        n_steps: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Price multiple strikes efficiently with a single simulation.

        The terminal prices are simulated once and reused for all strikes.

        Parameters
        ----------
        terminal_simulator : callable
            Function that simulates terminal prices
        s0 : float
            Spot price
        strikes : np.ndarray
            Array of strike prices
        t : float
            Time to maturity
        r : float
            Risk-free rate
        is_call : bool
            True for calls, False for puts
        n_paths : int, optional
            Number of paths
        n_steps : int, optional
            Number of steps
        seed : int, optional
            Random seed

        Returns
        -------
        prices : np.ndarray
            Array of option prices
        std_errors : np.ndarray
            Array of standard errors
        """
        n_paths = n_paths or self._config.n_paths
        n_steps = n_steps or self._config.n_steps
        seed = seed if seed is not None else self._config.seed

        # Single simulation
        terminals = terminal_simulator(s0, t, r, n_paths, n_steps, seed)

        # Discount factor
        discount = np.exp(-r * t)

        # Price each strike
        n_k = len(strikes)
        prices = np.empty(n_k)
        std_errors = np.empty(n_k)

        for i, k in enumerate(strikes):
            payoffs = _compute_payoffs(terminals, k, is_call)
            price, std_error = _compute_price_and_se(payoffs, discount)
            prices[i] = max(price, 0.0)
            std_errors[i] = std_error

        return prices, std_errors

    def price_surface(
        self,
        terminal_simulator: TerminalSimulator,
        s0: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        r: float,
        is_call: bool = True,
        n_paths: Optional[int] = None,
        n_steps_per_year: int = 252,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Price options across a strike-maturity surface.

        Parameters
        ----------
        terminal_simulator : callable
            Function that simulates terminal prices
        s0 : float
            Spot price
        strikes : np.ndarray
            Array of strike prices
        maturities : np.ndarray
            Array of maturities
        r : float
            Risk-free rate
        is_call : bool
            True for calls, False for puts
        n_paths : int, optional
            Number of paths per maturity
        n_steps_per_year : int
            Number of steps per year (steps = n_steps_per_year * t)
        seed : int, optional
            Random seed

        Returns
        -------
        prices : np.ndarray
            2D array of prices [n_strikes x n_maturities]
        std_errors : np.ndarray
            2D array of standard errors [n_strikes x n_maturities]
        """
        n_paths = n_paths or self._config.n_paths

        n_k = len(strikes)
        n_t = len(maturities)
        prices = np.empty((n_k, n_t))
        std_errors = np.empty((n_k, n_t))

        for j, t in enumerate(maturities):
            # Adjust steps for maturity
            n_steps = max(int(n_steps_per_year * t), 10)

            # Use different seed for each maturity for independence
            mat_seed = None if seed is None else seed + j

            # Price all strikes for this maturity
            p, se = self.price_strikes(
                terminal_simulator, s0, strikes, t, r, is_call,
                n_paths=n_paths, n_steps=n_steps, seed=mat_seed
            )
            prices[:, j] = p
            std_errors[:, j] = se

        return prices, std_errors


# =============================================================================
# Convenience function for quick pricing
# =============================================================================

def mc_price(
    terminal_simulator: TerminalSimulator,
    s0: float,
    k: float,
    t: float,
    r: float,
    is_call: bool = True,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: Optional[int] = None
) -> MCResult:
    """
    Quick Monte Carlo pricing with default configuration.

    Parameters
    ----------
    terminal_simulator : callable
        Function that simulates terminal prices
    s0 : float
        Spot price
    k : float
        Strike price
    t : float
        Time to maturity
    r : float
        Risk-free rate
    is_call : bool
        True for call, False for put
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps
    seed : int, optional
        Random seed

    Returns
    -------
    MCResult
        Named tuple with (price, std_error, n_paths)
    """
    config = MCConfig(n_paths=n_paths, n_steps=n_steps, seed=seed)
    engine = MonteCarloEngine(config)
    return engine.price(terminal_simulator, s0, k, t, r, is_call)
