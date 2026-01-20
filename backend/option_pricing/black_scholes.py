"""
Black-Scholes / GBM Option Pricer
=================================

High-performance option pricing for Geometric Brownian Motion.

Available Methods:
- Analytical: Closed-form Black-Scholes formula (default)
- Monte Carlo: Simulation-based pricing

The analytical method is exact and extremely fast. Monte Carlo is useful for
validation and for path-dependent options (not implemented here).

References:
    Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate
    Liabilities." Journal of Political Economy, 81(3), 637-654.

Author: Derivatives Pricing Project
"""

import numpy as np
import math
from numba import njit, prange
import time
from typing import Optional, Union, Tuple
from dataclasses import dataclass

from .base import (
    BasePricer,
    PricingResult,
    PricingMethod,
    OptionType,
)
from .engines.monte_carlo import MonteCarloEngine, MCConfig

# Import simulator for Monte Carlo
from backend.simulation.models.gbm import GBMSimulator


# =============================================================================
# Numba-Optimized Core Functions
# =============================================================================

@njit(fastmath=True, cache=True)
def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@njit(fastmath=True, cache=True)
def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@njit(fastmath=True, cache=True)
def _d1_d2(s: float, k: float, t: float, r: float, sigma: float) -> Tuple[float, float]:
    """Calculate d1 and d2 parameters."""
    if t <= 0 or sigma <= 0:
        return 0.0, 0.0

    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    return d1, d2


@njit(fastmath=True, cache=True)
def _bs_call_price(s: float, k: float, t: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    if t <= 0:
        return max(s - k, 0.0)

    d1, d2 = _d1_d2(s, k, t, r, sigma)
    return s * _norm_cdf(d1) - k * np.exp(-r * t) * _norm_cdf(d2)


@njit(fastmath=True, cache=True)
def _bs_put_price(s: float, k: float, t: float, r: float, sigma: float) -> float:
    """Black-Scholes put price."""
    if t <= 0:
        return max(k - s, 0.0)

    d1, d2 = _d1_d2(s, k, t, r, sigma)
    return k * np.exp(-r * t) * _norm_cdf(-d2) - s * _norm_cdf(-d1)


@njit(fastmath=True, cache=True)
def _bs_greeks(
    s: float, k: float, t: float, r: float, sigma: float, is_call: bool
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate price and all first-order Greeks.

    Returns: (price, delta, gamma, vega, theta, rho)
    """
    if t <= 0:
        if is_call:
            price = max(s - k, 0.0)
            delta = 1.0 if s > k else 0.0
        else:
            price = max(k - s, 0.0)
            delta = -1.0 if s < k else 0.0
        return price, delta, 0.0, 0.0, 0.0, 0.0

    d1, d2 = _d1_d2(s, k, t, r, sigma)
    sqrt_t = np.sqrt(t)
    exp_rt = np.exp(-r * t)

    n_d1 = _norm_cdf(d1)
    n_d2 = _norm_cdf(d2)
    n_prime_d1 = _norm_pdf(d1)

    # Gamma (same for call and put)
    gamma = n_prime_d1 / (s * sigma * sqrt_t)

    # Vega (same for call and put) - per 1% vol change
    vega = s * n_prime_d1 * sqrt_t / 100.0

    if is_call:
        price = s * n_d1 - k * exp_rt * n_d2
        delta = n_d1
        theta = (-s * n_prime_d1 * sigma / (2 * sqrt_t)
                 - r * k * exp_rt * n_d2) / 365.0
        rho = k * t * exp_rt * n_d2 / 100.0
    else:
        n_minus_d1 = _norm_cdf(-d1)
        n_minus_d2 = _norm_cdf(-d2)
        price = k * exp_rt * n_minus_d2 - s * n_minus_d1
        delta = n_d1 - 1.0
        theta = (-s * n_prime_d1 * sigma / (2 * sqrt_t)
                 + r * k * exp_rt * n_minus_d2) / 365.0
        rho = -k * t * exp_rt * n_minus_d2 / 100.0

    return price, delta, gamma, vega, theta, rho


@njit(fastmath=True, cache=True, parallel=True)
def _bs_price_surface(
    s: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    r: float,
    sigma: float,
    is_call: bool
) -> np.ndarray:
    """
    Price options across strike-maturity surface.

    Returns: 2D array [n_strikes x n_maturities]
    """
    n_k = len(strikes)
    n_t = len(maturities)
    prices = np.empty((n_k, n_t), dtype=np.float64)

    for i in prange(n_k):
        for j in range(n_t):
            if is_call:
                prices[i, j] = _bs_call_price(s, strikes[i], maturities[j], r, sigma)
            else:
                prices[i, j] = _bs_put_price(s, strikes[i], maturities[j], r, sigma)

    return prices


@njit(fastmath=True, cache=True)
def _implied_vol_newton(
    market_price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    is_call: bool,
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    """
    Implied volatility via Newton-Raphson method.
    """
    if t <= 0:
        return 0.0

    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2.0 * np.pi / t) * market_price / s

    # Bounds
    sigma = max(0.001, min(sigma, 5.0))

    for _ in range(max_iter):
        if is_call:
            price = _bs_call_price(s, k, t, r, sigma)
        else:
            price = _bs_put_price(s, k, t, r, sigma)

        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        # Vega for Newton step
        d1, _ = _d1_d2(s, k, t, r, sigma)
        vega = s * _norm_pdf(d1) * np.sqrt(t)

        if vega < 1e-10:
            break

        sigma = sigma - diff / vega
        sigma = max(0.001, min(sigma, 5.0))

    return sigma


# =============================================================================
# GBM Parameters Dataclass
# =============================================================================

@dataclass(frozen=True)
class GBMParams:
    """
    GBM model parameters.

    Parameters
    ----------
    sigma : float
        Volatility (annualized)
    """
    sigma: float

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")


# =============================================================================
# Black-Scholes / GBM Pricer Class
# =============================================================================

class BlackScholesPricer(BasePricer):
    """
    Black-Scholes option pricer for GBM models.

    Supports both analytical (closed-form) and Monte Carlo pricing.
    Analytical is the default and recommended method - it's exact and fast.

    Parameters
    ----------
    sigma : float
        Volatility (annualized)
    mc_config : MCConfig, optional
        Monte Carlo configuration (for MC method)

    Examples
    --------
    # Create pricer
    pricer = BlackScholesPricer(sigma=0.20)

    # Analytical pricing (default, recommended)
    result = pricer.price(s0=100, k=100, t=0.25, r=0.05)
    print(f"Call price: ${result.price:.4f}")
    print(f"Delta: {result.delta:.4f}")

    # Monte Carlo pricing (for validation)
    result_mc = pricer.price(s0=100, k=100, t=0.25, r=0.05,
                             method=PricingMethod.MONTE_CARLO)
    print(f"MC price: ${result_mc.price:.4f} +/- ${result_mc.std_error:.4f}")
    """

    # Class-level engine (shared across instances)
    _mc_engine: Optional[MonteCarloEngine] = None

    def __init__(
        self,
        sigma: float,
        mc_config: Optional[MCConfig] = None,
    ):
        super().__init__()
        self._model_name = "Black-Scholes"
        self._method = PricingMethod.ANALYTICAL  # Default method

        # Store parameters
        self._params = GBMParams(sigma=sigma)

        # Initialize engine config
        self._mc_config = mc_config or MCConfig()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def params(self) -> GBMParams:
        """Returns the GBM parameters."""
        return self._params

    @property
    def sigma(self) -> float:
        """Returns volatility."""
        return self._params.sigma

    @property
    def supported_methods(self) -> list:
        """Returns list of supported pricing methods."""
        return [PricingMethod.ANALYTICAL, PricingMethod.MONTE_CARLO]

    # -------------------------------------------------------------------------
    # Engine Access
    # -------------------------------------------------------------------------

    def _get_mc_engine(self) -> MonteCarloEngine:
        """Get or create Monte Carlo engine."""
        if BlackScholesPricer._mc_engine is None or BlackScholesPricer._mc_engine.config != self._mc_config:
            BlackScholesPricer._mc_engine = MonteCarloEngine(self._mc_config)
        return BlackScholesPricer._mc_engine

    # -------------------------------------------------------------------------
    # Monte Carlo Simulator
    # -------------------------------------------------------------------------

    def _create_terminal_simulator(self):
        """Create a terminal price simulator for the MC engine."""
        simulator = GBMSimulator(sigma=self._params.sigma)

        def terminal_sim(s0, t, r, n_paths, n_steps, seed=None):
            # Use risk-neutral measure (mu = r)
            return simulator.simulate_terminal(s0, r, t, n_paths, n_steps, seed)

        return terminal_sim

    # -------------------------------------------------------------------------
    # Pricing Methods
    # -------------------------------------------------------------------------

    def price(
        self,
        s0: float,
        k: float,
        t: float,
        r: float,
        option_type: Union[str, OptionType] = OptionType.CALL,
        method: Optional[PricingMethod] = None,
        compute_greeks: bool = True,
        n_paths: Optional[int] = None,
        n_steps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> PricingResult:
        """
        Price a European option.

        Parameters
        ----------
        s0 : float
            Current spot price
        k : float
            Strike price
        t : float
            Time to maturity in years
        r : float
            Risk-free interest rate
        option_type : str or OptionType
            'call' or 'put'
        method : PricingMethod, optional
            ANALYTICAL (default) or MONTE_CARLO
        compute_greeks : bool
            Whether to compute Greeks (only for analytical method)
        n_paths : int, optional
            Number of simulation paths (for MC method)
        n_steps : int, optional
            Number of time steps (for MC method)
        seed : int, optional
            Random seed for reproducibility (for MC method)

        Returns
        -------
        PricingResult
            Pricing result with price and optionally Greeks
        """
        self._validate_inputs(s0, k, t, r)
        opt_type = self._parse_option_type(option_type)

        # Default to analytical method
        method = method or PricingMethod.ANALYTICAL

        # Validate method
        if method not in self.supported_methods:
            raise ValueError(
                f"Method {method} not supported. Supported: {self.supported_methods}"
            )

        if method == PricingMethod.ANALYTICAL:
            return self._price_analytical(s0, k, t, r, opt_type, compute_greeks)
        else:  # MONTE_CARLO
            return self._price_mc(s0, k, t, r, opt_type, n_paths, n_steps, seed)

    def _price_analytical(
        self,
        s0: float,
        k: float,
        t: float,
        r: float,
        opt_type: OptionType,
        compute_greeks: bool = True,
    ) -> PricingResult:
        """Price using analytical Black-Scholes formula."""
        is_call = opt_type == OptionType.CALL
        start_time = time.perf_counter()

        if compute_greeks:
            price, delta, gamma, vega, theta, rho = _bs_greeks(
                s0, k, t, r, self._params.sigma, is_call
            )
            computation_time = time.perf_counter() - start_time

            return PricingResult(
                price=price,
                delta=delta,
                gamma=gamma,
                vega=vega,
                theta=theta,
                rho=rho,
                method=PricingMethod.ANALYTICAL,
                computation_time=computation_time,
                parameters={
                    "s0": s0, "k": k, "t": t, "r": r,
                    "sigma": self._params.sigma, "option_type": opt_type.value
                }
            )
        else:
            if is_call:
                price = _bs_call_price(s0, k, t, r, self._params.sigma)
            else:
                price = _bs_put_price(s0, k, t, r, self._params.sigma)

            computation_time = time.perf_counter() - start_time

            return PricingResult(
                price=price,
                method=PricingMethod.ANALYTICAL,
                computation_time=computation_time,
                parameters={
                    "s0": s0, "k": k, "t": t, "r": r,
                    "sigma": self._params.sigma, "option_type": opt_type.value
                }
            )

    def _price_mc(
        self,
        s0: float,
        k: float,
        t: float,
        r: float,
        opt_type: OptionType,
        n_paths: Optional[int] = None,
        n_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> PricingResult:
        """Price using Monte Carlo simulation."""
        start_time = time.perf_counter()

        engine = self._get_mc_engine()
        terminal_sim = self._create_terminal_simulator()

        is_call = opt_type == OptionType.CALL

        result = engine.price(
            terminal_sim, s0, k, t, r, is_call,
            n_paths=n_paths, n_steps=n_steps, seed=seed
        )

        computation_time = time.perf_counter() - start_time

        return PricingResult(
            price=result.price,
            method=PricingMethod.MONTE_CARLO,
            computation_time=computation_time,
            parameters={
                "s0": s0, "k": k, "t": t, "r": r,
                "sigma": self._params.sigma,
                "option_type": opt_type.value,
            },
            std_error=result.std_error,
            n_paths=result.n_paths,
        )

    # -------------------------------------------------------------------------
    # Multi-Strike / Surface Pricing
    # -------------------------------------------------------------------------

    def price_strikes(
        self,
        s0: float,
        strikes: np.ndarray,
        t: float,
        r: float,
        option_type: Union[str, OptionType] = OptionType.CALL,
        method: Optional[PricingMethod] = None,
        n_paths: Optional[int] = None,
        n_steps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Price multiple strikes efficiently.

        Parameters
        ----------
        s0 : float
            Spot price
        strikes : array-like
            Array of strike prices
        t : float
            Time to maturity
        r : float
            Risk-free rate
        option_type : str or OptionType
            'call' or 'put'
        method : PricingMethod, optional
            ANALYTICAL (default) or MONTE_CARLO

        Returns
        -------
        np.ndarray
            Array of option prices
        """
        strikes = np.asarray(strikes)
        opt_type = self._parse_option_type(option_type)
        is_call = opt_type == OptionType.CALL
        method = method or PricingMethod.ANALYTICAL

        if method == PricingMethod.ANALYTICAL:
            # Vectorized analytical pricing
            prices = np.empty(len(strikes))
            for i, k in enumerate(strikes):
                if is_call:
                    prices[i] = _bs_call_price(s0, k, t, r, self._params.sigma)
                else:
                    prices[i] = _bs_put_price(s0, k, t, r, self._params.sigma)
            return prices
        else:
            # Monte Carlo with single simulation
            engine = self._get_mc_engine()
            terminal_sim = self._create_terminal_simulator()

            prices, _ = engine.price_strikes(
                terminal_sim, s0, strikes, t, r, is_call,
                n_paths=n_paths, n_steps=n_steps, seed=seed
            )
            return prices

    def price_surface(
        self,
        s0: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        r: float,
        option_type: Union[str, OptionType] = OptionType.CALL,
        method: Optional[PricingMethod] = None,
        n_paths: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Price options across strike-maturity surface.

        Parameters
        ----------
        s0 : float
            Current spot price
        strikes : np.ndarray
            Array of strike prices
        maturities : np.ndarray
            Array of maturities
        r : float
            Risk-free interest rate
        option_type : str or OptionType
            'call' or 'put'
        method : PricingMethod, optional
            ANALYTICAL (default) or MONTE_CARLO

        Returns
        -------
        np.ndarray
            2D array of prices [n_strikes x n_maturities]
        """
        strikes = np.asarray(strikes)
        maturities = np.asarray(maturities)
        opt_type = self._parse_option_type(option_type)
        is_call = opt_type == OptionType.CALL
        method = method or PricingMethod.ANALYTICAL

        if method == PricingMethod.ANALYTICAL:
            return _bs_price_surface(s0, strikes, maturities, r, self._params.sigma, is_call)
        else:
            # Monte Carlo surface pricing
            engine = self._get_mc_engine()
            terminal_sim = self._create_terminal_simulator()

            prices, _ = engine.price_surface(
                terminal_sim, s0, strikes, maturities, r, is_call,
                n_paths=n_paths, seed=seed
            )
            return prices

    # -------------------------------------------------------------------------
    # Implied Volatility
    # -------------------------------------------------------------------------

    def implied_volatility(
        self,
        market_price: float,
        s0: float,
        k: float,
        t: float,
        r: float,
        option_type: Union[str, OptionType] = OptionType.CALL
    ) -> float:
        """
        Calculate implied volatility from market price.

        Parameters
        ----------
        market_price : float
            Market price of the option
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

        Returns
        -------
        float
            Implied volatility
        """
        opt_type = self._parse_option_type(option_type)
        is_call = opt_type == OptionType.CALL

        return _implied_vol_newton(market_price, s0, k, t, r, is_call)


# =============================================================================
# Alias for backward compatibility
# =============================================================================

GBMPricer = BlackScholesPricer


# =============================================================================
# Convenience Functions
# =============================================================================

def bs_call_price(s: float, k: float, t: float, r: float, sigma: float) -> float:
    """Black-Scholes call option price."""
    return _bs_call_price(s, k, t, r, sigma)


def bs_put_price(s: float, k: float, t: float, r: float, sigma: float) -> float:
    """Black-Scholes put option price."""
    return _bs_put_price(s, k, t, r, sigma)


def bs_greeks(
    s: float, k: float, t: float, r: float, sigma: float, is_call: bool = True
) -> dict:
    """
    Calculate Black-Scholes price and Greeks.

    Returns dict with: price, delta, gamma, vega, theta, rho
    """
    price, delta, gamma, vega, theta, rho = _bs_greeks(s, k, t, r, sigma, is_call)
    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def implied_volatility(
    market_price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    is_call: bool = True
) -> float:
    """Calculate implied volatility from market price."""
    return _implied_vol_newton(market_price, s, k, t, r, is_call)


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Black-Scholes Pricer Benchmark")
    print("=" * 60)

    # Test parameters
    s0, k, t, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.20

    # Warmup
    print("\nWarming up JIT...")
    pricer = BlackScholesPricer(sigma=sigma)
    _ = pricer.price(s0, k, t, r)

    # 1. Analytical pricing
    print("\n1. Analytical Pricing")
    print("-" * 40)
    result = pricer.price(s0, k, t, r, option_type="call")
    print(f"Call Price: ${result.price:.4f}")
    print(f"Delta: {result.delta:.4f}")
    print(f"Gamma: {result.gamma:.4f}")
    print(f"Vega: {result.vega:.4f}")
    print(f"Theta: {result.theta:.4f}")
    print(f"Rho: {result.rho:.4f}")
    print(f"Time: {result.computation_time*1e6:.2f} μs")

    # 2. Monte Carlo pricing
    print("\n2. Monte Carlo Pricing")
    print("-" * 40)
    result_mc = pricer.price(
        s0, k, t, r, option_type="call",
        method=PricingMethod.MONTE_CARLO,
        n_paths=100_000, seed=42
    )
    print(f"MC Call Price: ${result_mc.price:.4f} +/- ${result_mc.std_error:.4f}")
    print(f"Analytical: ${result.price:.4f}")
    print(f"Difference: ${abs(result_mc.price - result.price):.4f}")
    print(f"Time: {result_mc.computation_time*1000:.2f} ms")

    # 3. Surface pricing benchmark
    print("\n3. Surface Pricing Benchmark")
    print("-" * 40)
    strikes = np.linspace(80, 120, 100)
    maturities = np.linspace(0.01, 2.0, 50)

    start = time.perf_counter()
    surface = pricer.price_surface(s0, strikes, maturities, r)
    elapsed = time.perf_counter() - start

    print(f"Surface size: {len(strikes)} x {len(maturities)} = {len(strikes)*len(maturities):,}")
    print(f"Time: {elapsed*1000:.2f} ms")
    print(f"Speed: {len(strikes)*len(maturities)/elapsed:,.0f} prices/sec")

    # 4. Implied volatility
    print("\n4. Implied Volatility")
    print("-" * 40)
    market_price = result.price
    iv = pricer.implied_volatility(market_price, s0, k, t, r)
    print(f"Original sigma: {sigma:.4f}")
    print(f"Recovered IV: {iv:.4f}")
    print(f"Error: {abs(iv - sigma)*10000:.4f} bps")

    # 5. Supported methods
    print(f"\nSupported methods: {pricer.supported_methods}")

    print()
