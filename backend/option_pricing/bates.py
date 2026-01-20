"""
Bates Option Pricer
===================

Unified option pricer for the Bates stochastic volatility with jumps model
with multiple pricing methods.

Available Methods:
- FFT (Carr-Madan): Fast and accurate for European options
- Monte Carlo: Flexible, can be extended to path-dependent options

The Bates model combines Heston with Merton-style jumps:
    dS = μ·S·dt + √V·S·dW_S + (J - 1)·S·dN
    dV = κ·(θ - V)·dt + ξ·√V·dW_V
    Corr(dW_S, dW_V) = ρ

Where dN is a Poisson process with intensity λ and ln(J) ~ N(μ_J, σ²_J).

Under the risk-neutral measure (Q), the drift is adjusted for the jump risk premium.

References:
    - Bates, D.S. (1996). "Jumps and Stochastic Volatility: Exchange Rate
      Processes Implicit in Deutsche Mark Options." Review of Financial
      Studies, 9(1), 69-107.
    - Carr, P. and Madan, D.B. (1999). "Option valuation using the fast
      Fourier transform." Journal of Computational Finance, 2(4), 61-73.

Author: Derivatives Pricing Project
"""

import numpy as np
import time
import warnings
from typing import Optional, Union
from dataclasses import dataclass

from .base import (
    BasePricer,
    PricingResult,
    PricingMethod,
    OptionType,
)
from .engines.carr_madan import CarrMadanFFTEngine, FFTConfig
from .engines.monte_carlo import MonteCarloEngine, MCConfig

# Import characteristic function from models
from backend.models.characteristic_functions.bates_cf import (
    bates_characteristic_function,
    bates_cf_vectorized,
)

# Import simulator for Monte Carlo
from backend.simulation.models.bates import BatesSimulator


@dataclass(frozen=True)
class BatesParams:
    """
    Bates model parameters.

    Parameters
    ----------
    v0 : float
        Initial variance (σ²_0)
    kappa : float
        Mean reversion speed
    theta : float
        Long-run variance
    xi : float
        Volatility of volatility
    rho : float
        Correlation between price and variance
    lambda_j : float
        Jump intensity (expected jumps per year)
    mu_j : float
        Mean of log-jump size
    sigma_j : float
        Std of log-jump size
    """
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    lambda_j: float
    mu_j: float
    sigma_j: float

    def __post_init__(self):
        if self.v0 < 0:
            raise ValueError(f"Initial variance v0 must be non-negative, got {self.v0}")
        if self.kappa <= 0:
            raise ValueError(f"Mean reversion kappa must be positive, got {self.kappa}")
        if self.theta < 0:
            raise ValueError(f"Long-run variance theta must be non-negative, got {self.theta}")
        if self.xi <= 0:
            raise ValueError(f"Vol of vol xi must be positive, got {self.xi}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"Correlation rho must be in [-1, 1], got {self.rho}")
        if self.lambda_j < 0:
            raise ValueError(f"Jump intensity must be non-negative, got {self.lambda_j}")
        if self.sigma_j < 0:
            raise ValueError(f"Jump vol must be non-negative, got {self.sigma_j}")

    @property
    def feller_satisfied(self) -> bool:
        """Check if Feller condition (2κθ > ξ²) is satisfied."""
        return 2 * self.kappa * self.theta > self.xi ** 2

    @property
    def expected_jump_size(self) -> float:
        """E[J - 1], the expected percentage jump."""
        return np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1


class BatesPricer(BasePricer):
    """
    Unified Bates option pricer with multiple pricing methods.

    This pricer supports both FFT (Carr-Madan) and Monte Carlo methods,
    allowing you to choose the most appropriate method for your use case:
    - FFT: Fast, accurate for vanilla European options
    - Monte Carlo: Flexible, supports exotic payoffs

    Parameters
    ----------
    v0 : float
        Initial variance
    kappa : float
        Mean reversion speed
    theta : float
        Long-run variance
    xi : float
        Volatility of volatility
    rho : float
        Correlation between price and variance
    lambda_j : float
        Jump intensity (expected jumps per year)
    mu_j : float
        Mean of log-jump size
    sigma_j : float
        Std of log-jump size
    default_method : PricingMethod
        Default pricing method (default: FFT)

    Examples
    --------
    # Create pricer
    pricer = BatesPricer(
        v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
    )

    # Price with default method (FFT)
    result = pricer.price(s0=100, k=100, t=0.25, r=0.05)

    # Price with Monte Carlo
    result_mc = pricer.price(s0=100, k=100, t=0.25, r=0.05, method=PricingMethod.MONTE_CARLO)
    """

    # Class-level engines (shared across instances for efficiency)
    _fft_engine: Optional[CarrMadanFFTEngine] = None
    _mc_engine: Optional[MonteCarloEngine] = None

    def __init__(
        self,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        lambda_j: float,
        mu_j: float,
        sigma_j: float,
        default_method: PricingMethod = PricingMethod.FFT,
        fft_config: Optional[FFTConfig] = None,
        mc_config: Optional[MCConfig] = None,
    ):
        super().__init__()
        self._model_name = "Bates"
        self._method = default_method

        # Store parameters
        self._params = BatesParams(
            v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
            lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j
        )

        # Initialize engines lazily
        self._fft_config = fft_config or FFTConfig()
        self._mc_config = mc_config or MCConfig()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def params(self) -> BatesParams:
        """Returns the Bates parameters."""
        return self._params

    @property
    def v0(self) -> float:
        return self._params.v0

    @property
    def kappa(self) -> float:
        return self._params.kappa

    @property
    def theta(self) -> float:
        return self._params.theta

    @property
    def xi(self) -> float:
        return self._params.xi

    @property
    def rho(self) -> float:
        return self._params.rho

    @property
    def lambda_j(self) -> float:
        return self._params.lambda_j

    @property
    def mu_j(self) -> float:
        return self._params.mu_j

    @property
    def sigma_j(self) -> float:
        return self._params.sigma_j

    @property
    def expected_jump_size(self) -> float:
        """E[J - 1], the expected percentage jump."""
        return self._params.expected_jump_size

    @property
    def feller_condition_satisfied(self) -> bool:
        """Check if Feller condition (2κθ > ξ²) is satisfied."""
        return self._params.feller_satisfied

    @property
    def supported_methods(self) -> list:
        """Returns list of supported pricing methods."""
        return [PricingMethod.FFT, PricingMethod.MONTE_CARLO]

    # -------------------------------------------------------------------------
    # Engine Access (Lazy Initialization)
    # -------------------------------------------------------------------------

    def _get_fft_engine(self) -> CarrMadanFFTEngine:
        """Get or create FFT engine."""
        if BatesPricer._fft_engine is None or BatesPricer._fft_engine.config != self._fft_config:
            BatesPricer._fft_engine = CarrMadanFFTEngine(self._fft_config)
        return BatesPricer._fft_engine

    def _get_mc_engine(self) -> MonteCarloEngine:
        """Get or create Monte Carlo engine."""
        if BatesPricer._mc_engine is None or BatesPricer._mc_engine.config != self._mc_config:
            BatesPricer._mc_engine = MonteCarloEngine(self._mc_config)
        return BatesPricer._mc_engine

    # -------------------------------------------------------------------------
    # Characteristic Function
    # -------------------------------------------------------------------------

    def characteristic_function(
        self,
        u: np.ndarray,
        s0: float,
        t: float,
        r: float,
    ) -> np.ndarray:
        """
        Compute Bates characteristic function.

        Parameters
        ----------
        u : np.ndarray
            Complex frequency values
        s0 : float
            Spot price
        t : float
            Time to maturity
        r : float
            Risk-free rate

        Returns
        -------
        np.ndarray
            Characteristic function values phi(u)
        """
        return bates_cf_vectorized(
            u.astype(np.complex128),
            s0,
            self._params.v0,
            t,
            r,
            self._params.kappa,
            self._params.theta,
            self._params.xi,
            self._params.rho,
            self._params.lambda_j,
            self._params.mu_j,
            self._params.sigma_j,
        )

    def _create_cf(self, s0: float, t: float, r: float):
        """Create a characteristic function closure for the FFT engine."""
        def cf(u: np.ndarray) -> np.ndarray:
            return self.characteristic_function(u, s0, t, r)
        return cf

    # -------------------------------------------------------------------------
    # Monte Carlo Simulator
    # -------------------------------------------------------------------------

    def _create_terminal_simulator(self):
        """Create a terminal price simulator for the MC engine."""
        simulator = BatesSimulator(
            v0=self._params.v0,
            kappa=self._params.kappa,
            theta=self._params.theta,
            xi=self._params.xi,
            rho=self._params.rho,
            lambda_j=self._params.lambda_j,
            mu_j=self._params.mu_j,
            sigma_j=self._params.sigma_j,
        )

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
            Time to maturity
        r : float
            Risk-free rate
        option_type : str or OptionType
            'call' or 'put'
        method : PricingMethod, optional
            Pricing method. Uses default if not specified.
            Options: PricingMethod.FFT, PricingMethod.MONTE_CARLO
        **kwargs
            Method-specific arguments:
            - For Monte Carlo: n_paths, n_steps, seed

        Returns
        -------
        PricingResult
            Pricing result with price and metadata
        """
        self._validate_inputs(s0, k, t, r)
        opt_type = self._parse_option_type(option_type)
        method = method or self._method

        if method == PricingMethod.FFT:
            return self._price_fft(s0, k, t, r, opt_type)
        elif method == PricingMethod.MONTE_CARLO:
            return self._price_mc(s0, k, t, r, opt_type, **kwargs)
        else:
            raise ValueError(
                f"Method {method} not supported for Bates. "
                f"Supported: {self.supported_methods}"
            )

    def _price_fft(
        self,
        s0: float,
        k: float,
        t: float,
        r: float,
        opt_type: OptionType,
    ) -> PricingResult:
        """Price using FFT (Carr-Madan) method."""
        start_time = time.perf_counter()

        engine = self._get_fft_engine()
        cf = self._create_cf(s0, t, r)

        is_call = opt_type == OptionType.CALL

        if is_call:
            price = engine.price_call(cf, s0, k, t, r)
        else:
            price = engine.price_put(cf, s0, k, t, r)

        # Warn if price is negative
        if price < 0:
            warnings.warn(
                f"Negative {opt_type.value} price ({price:.6f}) computed via FFT. "
                f"Price floored to 0.",
                RuntimeWarning
            )
            price = 0.0

        computation_time = time.perf_counter() - start_time

        return PricingResult(
            price=price,
            method=PricingMethod.FFT,
            computation_time=computation_time,
            parameters={
                "s0": s0, "k": k, "t": t, "r": r,
                "v0": self._params.v0, "kappa": self._params.kappa,
                "theta": self._params.theta, "xi": self._params.xi,
                "rho": self._params.rho,
                "lambda_j": self._params.lambda_j,
                "mu_j": self._params.mu_j, "sigma_j": self._params.sigma_j,
                "option_type": opt_type.value,
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
        **kwargs
    ) -> PricingResult:
        """Price using Monte Carlo method."""
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
                "v0": self._params.v0, "kappa": self._params.kappa,
                "theta": self._params.theta, "xi": self._params.xi,
                "rho": self._params.rho,
                "lambda_j": self._params.lambda_j,
                "mu_j": self._params.mu_j, "sigma_j": self._params.sigma_j,
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
        **kwargs
    ) -> np.ndarray:
        """
        Price multiple strikes efficiently.

        For FFT, uses a single FFT for all strikes.
        For MC, uses a single simulation for all strikes.

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
            Pricing method

        Returns
        -------
        np.ndarray
            Array of option prices
        """
        strikes = np.asarray(strikes)
        opt_type = self._parse_option_type(option_type)
        method = method or self._method
        is_call = opt_type == OptionType.CALL

        if method == PricingMethod.FFT:
            engine = self._get_fft_engine()
            cf = self._create_cf(s0, t, r)
            return engine.price_strikes(cf, s0, strikes, t, r, is_call)

        elif method == PricingMethod.MONTE_CARLO:
            engine = self._get_mc_engine()
            terminal_sim = self._create_terminal_simulator()
            prices, _ = engine.price_strikes(
                terminal_sim, s0, strikes, t, r, is_call, **kwargs
            )
            return prices

        else:
            raise ValueError(f"Method {method} not supported")

    def price_surface(
        self,
        s0: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        r: float,
        option_type: Union[str, OptionType] = OptionType.CALL,
        method: Optional[PricingMethod] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Price options across a strike-maturity surface.

        Parameters
        ----------
        s0 : float
            Spot price
        strikes : np.ndarray
            Array of strike prices
        maturities : np.ndarray
            Array of maturities
        r : float
            Risk-free rate
        option_type : str or OptionType
            'call' or 'put'
        method : PricingMethod, optional
            Pricing method

        Returns
        -------
        np.ndarray
            2D array of prices [n_strikes x n_maturities]
        """
        strikes = np.asarray(strikes)
        maturities = np.asarray(maturities)
        opt_type = self._parse_option_type(option_type)
        method = method or self._method
        is_call = opt_type == OptionType.CALL

        if method == PricingMethod.FFT:
            engine = self._get_fft_engine()

            # Factory that creates CF for each maturity
            def cf_factory(t):
                return self._create_cf(s0, t, r)

            return engine.price_surface(cf_factory, s0, strikes, maturities, r, is_call)

        elif method == PricingMethod.MONTE_CARLO:
            engine = self._get_mc_engine()
            terminal_sim = self._create_terminal_simulator()
            prices, _ = engine.price_surface(
                terminal_sim, s0, strikes, maturities, r, is_call, **kwargs
            )
            return prices

        else:
            raise ValueError(f"Method {method} not supported")


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

# =============================================================================
# Convenience Functions
# =============================================================================

def bates_call_price(
    s0: float,
    k: float,
    t: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    method: PricingMethod = PricingMethod.FFT,
) -> float:
    """Bates call option price."""
    pricer = BatesPricer(
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j
    )
    result = pricer.price(s0, k, t, r, option_type=OptionType.CALL, method=method)
    return result.price


def bates_put_price(
    s0: float,
    k: float,
    t: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    method: PricingMethod = PricingMethod.FFT,
) -> float:
    """Bates put option price."""
    pricer = BatesPricer(
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j
    )
    result = pricer.price(s0, k, t, r, option_type=OptionType.PUT, method=method)
    return result.price


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Bates Unified Pricer Benchmark")
    print("=" * 60)

    # Test parameters
    s0, k, t, r = 100.0, 100.0, 0.25, 0.05
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7
    lambda_j, mu_j, sigma_j = 0.5, -0.1, 0.2

    # Create pricer
    pricer = BatesPricer(
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j
    )

    # Warmup
    print("\nWarming up...")
    _ = pricer.price(s0, k, t, r)
    _ = pricer.price(s0, k, t, r, method=PricingMethod.MONTE_CARLO, n_paths=1000)

    # 1. FFT pricing
    print("\n1. FFT Pricing (Carr-Madan)")
    print("-" * 40)
    result_fft = pricer.price(s0, k, t, r, option_type="call")
    print(f"Call Price: ${result_fft.price:.4f}")
    print(f"Time: {result_fft.computation_time*1000:.2f} ms")

    result_put = pricer.price(s0, k, t, r, option_type="put")
    print(f"Put Price: ${result_put.price:.4f}")

    # 2. Monte Carlo pricing
    print("\n2. Monte Carlo Pricing")
    print("-" * 40)
    result_mc = pricer.price(
        s0, k, t, r, option_type="call",
        method=PricingMethod.MONTE_CARLO,
        n_paths=100_000, seed=42
    )
    print(f"Call Price: ${result_mc.price:.4f} +/- ${result_mc.std_error:.4f}")
    print(f"Time: {result_mc.computation_time*1000:.2f} ms")
    print(f"FFT vs MC diff: ${abs(result_fft.price - result_mc.price):.4f}")

    # 3. Compare with Heston (no jumps)
    print("\n3. Bates vs Heston (λ=0)")
    print("-" * 40)
    pricer_no_jumps = BatesPricer(
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        lambda_j=0.0, mu_j=0.0, sigma_j=0.0
    )
    result_no_jumps = pricer_no_jumps.price(s0, k, t, r)
    print(f"Bates (λ=0): ${result_no_jumps.price:.4f}")
    print(f"Bates (λ={lambda_j}): ${result_fft.price:.4f}")
    print(f"Jump premium: ${result_fft.price - result_no_jumps.price:.4f}")

    # 4. Multi-strike pricing
    print("\n4. Multi-Strike Pricing (FFT)")
    print("-" * 40)
    strikes = np.linspace(80, 120, 41)

    start = time.perf_counter()
    prices = pricer.price_strikes(s0, strikes, t, r)
    elapsed = time.perf_counter() - start

    print(f"Strikes: {len(strikes)}")
    print(f"Time: {elapsed*1000:.2f} ms")
    print(f"Speed: {len(strikes)/elapsed:,.0f} prices/sec")

    # 5. Model info
    print(f"\nExpected jump size: {pricer.expected_jump_size*100:.1f}%")
    print(f"Feller condition satisfied: {pricer.feller_condition_satisfied}")
    print(f"Supported methods: {pricer.supported_methods}")

    print()
