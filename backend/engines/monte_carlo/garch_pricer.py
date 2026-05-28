"""
GARCH Monte Carlo Option Pricer
================================

Monte Carlo option pricing for GARCH family models using the
Locally Risk-Neutral Valuation Relationship (LRNVR) from Duan (1995).

Supported Models:
- GARCH(1,1)
- NGARCH (Nonlinear Asymmetric GARCH)
- GJR-GARCH (Glosten-Jagannathan-Runkle)

Risk-Neutral Dynamics (LRNVR):
    Under the risk-neutral measure Q:
        r_t = r - 0.5·σ²_t + σ_t·z_t,  where z_t ~ N(0,1)
        S_t = S_{t-1}·exp(r_t)

    The variance dynamics remain the same but use risk-neutral innovations.

References:
    - Duan, J.C. (1995). "The GARCH Option Pricing Model."
      Mathematical Finance, 5(1), 13-32.
    - Duan, J.C., Gauthier, G. & Simonato, J.G. (1999). "An Analytical
      Approximation for the GARCH Option Pricing Model."
      Journal of Computational Finance, 2(4), 75-116.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numba import njit, prange

# =============================================================================
# Types and Enums
# =============================================================================


class GARCHType(Enum):
    """GARCH model type enumeration."""

    GARCH = "garch"
    NGARCH = "ngarch"
    GJR_GARCH = "gjr_garch"


class OptionType(Enum):
    """Option type enumeration."""

    CALL = "call"
    PUT = "put"


@dataclass
class GARCHPricingResult:
    """
    Result of GARCH Monte Carlo pricing.

    Attributes
    ----------
    price : float
        Option price
    std_error : float
        Standard error of the MC estimate
    computation_time : float
        Time taken for computation in seconds
    n_paths : int
        Number of MC paths used
    garch_type : str
        Type of GARCH model used
    parameters : Dict[str, Any]
        Dictionary of model and pricing parameters
    """

    price: float
    std_error: float
    computation_time: float
    n_paths: int
    garch_type: str
    parameters: dict[str, Any]


# =============================================================================
# Numba-Optimized Monte Carlo Kernels
# =============================================================================


@njit(parallel=True, cache=True, fastmath=True)
def _simulate_garch_terminal_rn(
    s0: float,
    r: float,
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    t: float,
    n_paths: int,
    n_steps: int,
) -> np.ndarray:
    """
    GARCH(1,1) terminal simulation under risk-neutral measure (LRNVR).

    Variance: σ²_t = ω + α·σ²_{t-1}·z²_{t-1} + β·σ²_{t-1}
    """
    dt = t / n_steps
    var0 = sigma0 * sigma0

    terminals = np.empty(n_paths, dtype=np.float64)

    for i in prange(n_paths):
        s = s0
        var_t = var0

        for j in range(n_steps):
            z = np.random.standard_normal()

            sigma_t = np.sqrt(var_t)

            # Risk-neutral return (LRNVR)
            log_return = (r - 0.5 * var_t) * dt + sigma_t * np.sqrt(dt) * z
            s = s * np.exp(log_return)

            # GARCH variance update
            var_next = omega + alpha * var_t * z * z + beta * var_t
            var_t = max(var_next, 1e-10)

        terminals[i] = s

    return terminals


@njit(parallel=True, cache=True, fastmath=True)
def _simulate_ngarch_terminal_rn(
    s0: float,
    r: float,
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    t: float,
    n_paths: int,
    n_steps: int,
) -> np.ndarray:
    """
    NGARCH terminal simulation under risk-neutral measure.

    Variance: σ²_t = ω + α·σ²_{t-1}·(z_{t-1} - γ)² + β·σ²_{t-1}
    """
    dt = t / n_steps
    var0 = sigma0 * sigma0

    terminals = np.empty(n_paths, dtype=np.float64)

    for i in prange(n_paths):
        s = s0
        var_t = var0

        for j in range(n_steps):
            z = np.random.standard_normal()

            sigma_t = np.sqrt(var_t)
            log_return = (r - 0.5 * var_t) * dt + sigma_t * np.sqrt(dt) * z
            s = s * np.exp(log_return)

            # NGARCH variance update with leverage
            shifted = z - gamma
            var_next = omega + alpha * var_t * shifted * shifted + beta * var_t
            var_t = max(var_next, 1e-10)

        terminals[i] = s

    return terminals


@njit(parallel=True, cache=True, fastmath=True)
def _simulate_gjr_garch_terminal_rn(
    s0: float,
    r: float,
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    t: float,
    n_paths: int,
    n_steps: int,
) -> np.ndarray:
    """
    GJR-GARCH terminal simulation under risk-neutral measure.

    Variance: σ²_t = ω + (α + γ·I_{t-1})·σ²_{t-1}·z²_{t-1} + β·σ²_{t-1}
    where I_{t-1} = 1 if z_{t-1} < 0, else 0
    """
    dt = t / n_steps
    var0 = sigma0 * sigma0

    terminals = np.empty(n_paths, dtype=np.float64)

    for i in prange(n_paths):
        s = s0
        var_t = var0

        for j in range(n_steps):
            z = np.random.standard_normal()

            sigma_t = np.sqrt(var_t)
            log_return = (r - 0.5 * var_t) * dt + sigma_t * np.sqrt(dt) * z
            s = s * np.exp(log_return)

            # GJR-GARCH variance update with asymmetry
            indicator = 1.0 if z < 0.0 else 0.0
            z_sq = z * z
            var_next = omega + (alpha + gamma * indicator) * var_t * z_sq + beta * var_t
            var_t = max(var_next, 1e-10)

        terminals[i] = s

    return terminals


@njit(fastmath=True, cache=True)
def _compute_mc_price_and_se(
    terminals: np.ndarray, k: float, r: float, t: float, is_call: bool
) -> tuple[float, float]:
    """Compute Monte Carlo price and standard error."""
    n = len(terminals)

    # Payoffs
    payoffs = np.empty(n, dtype=np.float64)
    for i in range(n):
        if is_call:
            payoffs[i] = max(terminals[i] - k, 0.0)
        else:
            payoffs[i] = max(k - terminals[i], 0.0)

    # Discounted mean
    discount = np.exp(-r * t)
    mean_payoff = 0.0
    for i in range(n):
        mean_payoff += payoffs[i]
    mean_payoff = mean_payoff / n

    price = discount * mean_payoff

    # Standard error
    var_payoff = 0.0
    for i in range(n):
        diff = payoffs[i] - mean_payoff
        var_payoff += diff * diff
    var_payoff = var_payoff / (n - 1)

    std_error = discount * np.sqrt(var_payoff / n)

    return price, std_error


# =============================================================================
# GARCH Monte Carlo Pricer Class
# =============================================================================


class GARCHMCPricer:
    """
    GARCH family option pricer using Monte Carlo simulation.

    Uses the Locally Risk-Neutral Valuation Relationship (LRNVR)
    from Duan (1995) for risk-neutral pricing.

    Supported models:
    - GARCH(1,1): σ²_t = ω + α·σ²_{t-1}·z² + β·σ²_{t-1}
    - NGARCH: σ²_t = ω + α·σ²_{t-1}·(z - γ)² + β·σ²_{t-1}
    - GJR-GARCH: σ²_t = ω + (α + γ·I)·σ²_{t-1}·z² + β·σ²_{t-1}

    Parameters
    ----------
    garch_type : GARCHType or str
        Type of GARCH model
    sigma0 : float
        Initial volatility
    omega : float
        Constant term in variance equation
    alpha : float
        ARCH coefficient
    beta : float
        GARCH coefficient
    gamma : float, optional
        Leverage parameter for NGARCH (default 0)
    gamma : float, optional
        Asymmetry parameter for GJR-GARCH (default 0)
    n_paths : int
        Number of MC paths (default 100,000)
    n_steps : int
        Number of time steps (default 252)

    Examples
    --------
    # GARCH(1,1)
    pricer = GARCHMCPricer(
        garch_type="garch",
        sigma0=0.20, omega=0.000002, alpha=0.05, beta=0.90
    )
    result = pricer.price(s0=100, k=100, t=0.25, r=0.05)

    # NGARCH with leverage
    pricer = GARCHMCPricer(
        garch_type="ngarch",
        sigma0=0.20, omega=0.000002, alpha=0.05, beta=0.90, gamma=0.5
    )
    """

    def __init__(
        self,
        garch_type: str | GARCHType,
        sigma0: float,
        omega: float,
        alpha: float,
        beta: float,
        gamma: float = 0.0,
        n_paths: int = 100000,
        n_steps: int = 252,
    ) -> None:
        # Parse GARCH type
        if isinstance(garch_type, str):
            garch_type = GARCHType(garch_type.lower())
        self._garch_type = garch_type

        self._model_name = f"{garch_type.value.upper()} (Monte Carlo)"

        # Model parameters
        self._sigma0 = sigma0
        self._omega = omega
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

        # MC parameters
        self._n_paths = n_paths
        self._n_steps = n_steps

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate GARCH parameters."""
        if self._sigma0 <= 0:
            raise ValueError(f"Initial volatility must be positive, got {self._sigma0}")
        if self._omega <= 0:
            raise ValueError(f"Omega must be positive, got {self._omega}")
        if self._alpha < 0:
            raise ValueError(f"Alpha must be non-negative, got {self._alpha}")
        if self._beta < 0:
            raise ValueError(f"Beta must be non-negative, got {self._beta}")

        # Check stationarity
        if self._garch_type == GARCHType.GARCH:
            persistence = self._alpha + self._beta
            if persistence >= 1:
                raise ValueError(f"Non-stationary: α + β = {persistence:.4f} >= 1")

        elif self._garch_type == GARCHType.NGARCH:
            persistence = self._alpha * (1 + self._gamma**2) + self._beta
            if persistence >= 1:
                raise ValueError(
                    f"Non-stationary: α(1+γ²) + β = {persistence:.4f} >= 1"
                )

        elif self._garch_type == GARCHType.GJR_GARCH:
            if self._gamma < 0:
                raise ValueError(f"Gamma must be non-negative, got {self._gamma}")
            persistence = self._alpha + 0.5 * self._gamma + self._beta
            if persistence >= 1:
                raise ValueError(
                    f"Non-stationary: α + 0.5γ + β = {persistence:.4f} >= 1"
                )

    # Properties
    @property
    def garch_type(self) -> GARCHType:
        return self._garch_type

    @property
    def sigma0(self) -> float:
        return self._sigma0

    @property
    def omega(self) -> float:
        return self._omega

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def n_paths(self) -> int:
        return self._n_paths

    @n_paths.setter
    def n_paths(self, value: int) -> None:
        if value <= 0:
            raise ValueError(f"n_paths must be positive, got {value}")
        self._n_paths: int = value

    @property
    def persistence(self) -> float:
        """Returns the persistence of shocks."""
        if self._garch_type == GARCHType.GARCH:
            return self._alpha + self._beta
        if self._garch_type == GARCHType.NGARCH:
            return self._alpha * (1 + self._gamma**2) + self._beta
        # GJR_GARCH
        return self._alpha + 0.5 * self._gamma + self._beta

    def long_run_variance(self) -> float:
        """Returns the theoretical long-run variance."""
        return float(self._omega / (1 - self.persistence))

    def long_run_volatility(self) -> float:
        """Returns the theoretical long-run volatility."""
        return float(np.sqrt(self.long_run_variance()))

    def simulate_terminal(
        self,
        s0: float,
        t: float,
        r: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Simulate terminal prices under risk-neutral measure."""
        if seed is not None:
            np.random.seed(seed)

        if self._garch_type == GARCHType.GARCH:
            return _simulate_garch_terminal_rn(
                s0,
                r,
                self._sigma0,
                self._omega,
                self._alpha,
                self._beta,
                t,
                n_paths,
                n_steps,
            )
        if self._garch_type == GARCHType.NGARCH:
            return _simulate_ngarch_terminal_rn(
                s0,
                r,
                self._sigma0,
                self._omega,
                self._alpha,
                self._beta,
                self._gamma,
                t,
                n_paths,
                n_steps,
            )
        # GJR_GARCH
        return _simulate_gjr_garch_terminal_rn(
            s0,
            r,
            self._sigma0,
            self._omega,
            self._alpha,
            self._beta,
            self._gamma,
            t,
            n_paths,
            n_steps,
        )

    def price(
        self,
        s0: float,
        k: float,
        t: float,
        r: float,
        option_type: str | OptionType = OptionType.CALL,
        n_paths: int | None = None,
        n_steps: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> GARCHPricingResult:
        """
        Price a European option using Monte Carlo simulation.

        Parameters
        ----------
        s0 : float
            Current spot price
        k : float
            Strike price
        t : float
            Time to maturity
        r : float
            Risk-free rate
        option_type : str or OptionType
            'call' or 'put'
        n_paths : int, optional
            Number of paths (uses default if not specified)
        n_steps : int, optional
            Number of steps (uses default if not specified)
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        GARCHPricingResult
            Pricing result with price and standard error
        """
        # Validate inputs
        if s0 <= 0:
            raise ValueError(f"Spot price must be positive, got {s0}")
        if k <= 0:
            raise ValueError(f"Strike must be positive, got {k}")
        if t <= 0:
            raise ValueError(f"Time to maturity must be positive, got {t}")

        # Parse option type
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())
        is_call = option_type == OptionType.CALL

        n_paths = n_paths or self._n_paths
        n_steps = n_steps or self._n_steps

        start_time = time.perf_counter()

        # Simulate terminal prices
        terminals = self.simulate_terminal(s0, t, r, n_paths, n_steps, seed)

        # Compute price and standard error
        price, std_error = _compute_mc_price_and_se(terminals, k, r, t, is_call)

        computation_time = time.perf_counter() - start_time

        return GARCHPricingResult(
            price=max(price, 0.0),
            std_error=std_error,
            computation_time=computation_time,
            n_paths=n_paths,
            garch_type=self._garch_type.value,
            parameters={
                "s0": s0,
                "k": k,
                "t": t,
                "r": r,
                "sigma0": self._sigma0,
                "omega": self._omega,
                "alpha": self._alpha,
                "beta": self._beta,
                "gamma": self._gamma,
                "option_type": option_type.value,
            },
        )

    def price_surface(
        self,
        s0: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        r: float,
        option_type: str | OptionType = OptionType.CALL,
        n_paths: int | None = None,
        n_steps: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Price options across strike-maturity surface.

        Note: For efficiency, simulations are reused across strikes
        for each maturity.
        """
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())
        is_call = option_type == OptionType.CALL

        n_paths = n_paths or self._n_paths
        n_steps = n_steps or self._n_steps

        n_k = len(strikes)
        n_t = len(maturities)
        prices = np.empty((n_k, n_t))

        for j, t in enumerate(maturities):
            # Single simulation for all strikes at this maturity
            steps_for_t = max(int(n_steps * t), 10)
            terminals = self.simulate_terminal(s0, t, r, n_paths, steps_for_t)

            for i, k in enumerate(strikes):
                price, _ = _compute_mc_price_and_se(terminals, k, r, t, is_call)
                prices[i, j] = max(price, 0.0)

        return prices


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def create_garch_pricer(
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    n_paths: int = 100000,
) -> GARCHMCPricer:
    """Create a GARCH(1,1) Monte Carlo pricer."""
    return GARCHMCPricer(
        garch_type=GARCHType.GARCH,
        sigma0=sigma0,
        omega=omega,
        alpha=alpha,
        beta=beta,
        n_paths=n_paths,
    )


def create_ngarch_pricer(
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    n_paths: int = 100000,
) -> GARCHMCPricer:
    """Create an NGARCH Monte Carlo pricer."""
    return GARCHMCPricer(
        garch_type=GARCHType.NGARCH,
        sigma0=sigma0,
        omega=omega,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        n_paths=n_paths,
    )


def create_gjr_garch_pricer(
    sigma0: float,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    n_paths: int = 100000,
) -> GARCHMCPricer:
    """Create a GJR-GARCH Monte Carlo pricer."""
    return GARCHMCPricer(
        garch_type=GARCHType.GJR_GARCH,
        sigma0=sigma0,
        omega=omega,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        n_paths=n_paths,
    )


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GARCH Monte Carlo Pricer Benchmark")
    print("=" * 60)

    # Test parameters
    s0, k, t, r = 100.0, 100.0, 0.25, 0.05
    sigma0, omega, alpha, beta = 0.20, 0.000002, 0.05, 0.90

    # Warmup
    print("\nWarming up JIT...")
    pricer = create_garch_pricer(sigma0, omega, alpha, beta, n_paths=1000)
    _ = pricer.price(s0, k, t, r)

    # 1. GARCH(1,1) pricing
    print("\n1. GARCH(1,1) Pricing")
    print("-" * 40)
    pricer = create_garch_pricer(sigma0, omega, alpha, beta, n_paths=100000)
    result = pricer.price(s0, k, t, r, seed=42)
    print(f"Call Price: ${result.price:.4f} ± ${result.std_error:.4f}")
    print(f"Time: {result.computation_time * 1000:.2f} ms")
    print(f"Paths: {result.n_paths:,}")
    print(f"Long-run vol: {pricer.long_run_volatility() * 100:.2f}%")

    # 2. NGARCH pricing
    print("\n2. NGARCH Pricing (with leverage)")
    print("-" * 40)
    pricer_ngarch = create_ngarch_pricer(
        sigma0, omega, 0.05, 0.90, gamma=0.5, n_paths=100000
    )
    result = pricer_ngarch.price(s0, k, t, r, seed=42)
    print(f"Call Price: ${result.price:.4f} ± ${result.std_error:.4f}")
    print(f"Long-run vol: {pricer_ngarch.long_run_volatility() * 100:.2f}%")

    # 3. GJR-GARCH pricing
    print("\n3. GJR-GARCH Pricing (asymmetric)")
    print("-" * 40)
    pricer_gjr = create_gjr_garch_pricer(
        sigma0, omega, 0.03, 0.90, gamma=0.07, n_paths=100000
    )
    result = pricer_gjr.price(s0, k, t, r, seed=42)
    print(f"Call Price: ${result.price:.4f} ± ${result.std_error:.4f}")
    print(f"Long-run vol: {pricer_gjr.long_run_volatility() * 100:.2f}%")

    # 4. Convergence test
    print("\n4. Convergence Test (GARCH)")
    print("-" * 40)
    path_counts = [10000, 50000, 100000, 500000]
    pricer = create_garch_pricer(sigma0, omega, alpha, beta)

    print(f"{'Paths':>12} {'Price':>10} {'Std Err':>10} {'Time (ms)':>12}")
    print("-" * 46)

    for n in path_counts:
        result = pricer.price(s0, k, t, r, n_paths=n, seed=42)
        print(
            f"{n:>12,} ${result.price:>9.4f} ${result.std_error:>9.4f} {result.computation_time * 1000:>11.2f}"
        )

    print()
