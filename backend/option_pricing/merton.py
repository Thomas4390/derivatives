"""
Merton Jump Diffusion Option Pricer
====================================

Option pricer for the Merton (1976) jump-diffusion model.

Available Methods:
- FFT (Carr-Madan): Fast semi-analytical pricing via characteristic function
- Monte Carlo: Simulation-based pricing

The Merton model adds jumps to GBM:
    dS/S = (μ - λ·k)·dt + σ·dW + (J - 1)·dN

Where:
    - dN is a Poisson process with intensity λ
    - J is lognormally distributed: ln(J) ~ N(μ_J, σ²_J)
    - k = E[J - 1] = exp(μ_J + 0.5·σ²_J) - 1 (compensator)

Note: Merton also has an exact semi-closed form solution as an infinite series
of Black-Scholes prices (not implemented). FFT is typically faster and more
general for models with known characteristic functions.

References:
    - Merton, R.C. (1976). "Option Pricing When Underlying Stock Returns
      Are Discontinuous." Journal of Financial Economics, 3(1-2), 125-144.
    - Carr, P. and Madan, D.B. (1999). "Option valuation using the fast
      Fourier transform." Journal of Computational Finance, 2(4), 61-73.

Author: Derivatives Pricing Project
"""

import numpy as np
import time
from typing import Optional, Union
from dataclasses import dataclass

from .base import (
    BasePricer,
    PricingResult,
    PricingMethod,
    OptionType,
)
from .engines.monte_carlo import MonteCarloEngine, MCConfig
from .engines.carr_madan import CarrMadanFFTEngine, FFTConfig

# Import simulator for Monte Carlo
from backend.simulation.models.merton import MertonSimulator

# Import characteristic function for FFT
from backend.models.characteristic_functions.merton_cf import create_merton_cf


@dataclass(frozen=True)
class MertonParams:
    """
    Merton model parameters.

    Parameters
    ----------
    sigma : float
        Diffusion volatility
    lambda_j : float
        Jump intensity (expected jumps per year)
    mu_j : float
        Mean of log-jump size
    sigma_j : float
        Std of log-jump size
    """
    sigma: float
    lambda_j: float
    mu_j: float
    sigma_j: float

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")
        if self.lambda_j < 0:
            raise ValueError(f"Jump intensity must be non-negative, got {self.lambda_j}")
        if self.sigma_j < 0:
            raise ValueError(f"Jump vol must be non-negative, got {self.sigma_j}")

    @property
    def expected_jump_size(self) -> float:
        """E[J - 1], the expected percentage jump."""
        return np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1

    @property
    def total_variance_per_year(self) -> float:
        """Total variance including jump contribution."""
        # Diffusion variance + jump variance contribution
        jump_var = self.lambda_j * (self.sigma_j ** 2 + self.mu_j ** 2)
        return self.sigma ** 2 + jump_var


class MertonPricer(BasePricer):
    """
    Merton jump-diffusion option pricer.

    Supports both FFT (Carr-Madan) and Monte Carlo pricing methods.
    FFT is the default and recommended method for European options.

    Parameters
    ----------
    sigma : float
        Diffusion volatility
    lambda_j : float
        Jump intensity (expected jumps per year)
    mu_j : float
        Mean of log-jump size
    sigma_j : float
        Std of log-jump size
    fft_config : FFTConfig, optional
        FFT configuration
    mc_config : MCConfig, optional
        Monte Carlo configuration

    Examples
    --------
    # Create pricer
    pricer = MertonPricer(sigma=0.15, lambda_j=1.0, mu_j=-0.05, sigma_j=0.10)

    # Price with FFT (default)
    result = pricer.price(s0=100, k=100, t=0.25, r=0.05)
    print(f"FFT Price: ${result.price:.4f}")

    # Price with Monte Carlo
    result_mc = pricer.price(s0=100, k=100, t=0.25, r=0.05,
                             method=PricingMethod.MONTE_CARLO)
    print(f"MC Price: ${result_mc.price:.4f} +/- ${result_mc.std_error:.4f}")
    """

    # Class-level engines (shared across instances)
    _fft_engine: Optional[CarrMadanFFTEngine] = None
    _mc_engine: Optional[MonteCarloEngine] = None

    def __init__(
        self,
        sigma: float,
        lambda_j: float,
        mu_j: float,
        sigma_j: float,
        fft_config: Optional[FFTConfig] = None,
        mc_config: Optional[MCConfig] = None,
    ):
        super().__init__()
        self._model_name = "Merton"
        self._method = PricingMethod.FFT  # Default method

        # Store parameters
        self._params = MertonParams(
            sigma=sigma, lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j
        )

        # Initialize engine configs
        self._fft_config = fft_config or FFTConfig()
        self._mc_config = mc_config or MCConfig()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def params(self) -> MertonParams:
        """Returns the Merton parameters."""
        return self._params

    @property
    def sigma(self) -> float:
        return self._params.sigma

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
    def supported_methods(self) -> list:
        """Returns list of supported pricing methods."""
        return [PricingMethod.FFT, PricingMethod.MONTE_CARLO]

    # -------------------------------------------------------------------------
    # Engine Access
    # -------------------------------------------------------------------------

    def _get_fft_engine(self) -> CarrMadanFFTEngine:
        """Get or create FFT engine."""
        if MertonPricer._fft_engine is None or MertonPricer._fft_engine.config != self._fft_config:
            MertonPricer._fft_engine = CarrMadanFFTEngine(self._fft_config)
        return MertonPricer._fft_engine

    def _get_mc_engine(self) -> MonteCarloEngine:
        """Get or create Monte Carlo engine."""
        if MertonPricer._mc_engine is None or MertonPricer._mc_engine.config != self._mc_config:
            MertonPricer._mc_engine = MonteCarloEngine(self._mc_config)
        return MertonPricer._mc_engine

    # -------------------------------------------------------------------------
    # Characteristic Function Factory
    # -------------------------------------------------------------------------

    def _create_cf(self, s0: float, t: float, r: float):
        """Create characteristic function for FFT pricing."""
        return create_merton_cf(
            s0=s0, t=t, r=r,
            sigma=self._params.sigma,
            lambda_j=self._params.lambda_j,
            mu_j=self._params.mu_j,
            sigma_j=self._params.sigma_j,
        )

    def _create_cf_factory(self, s0: float, r: float):
        """Create characteristic function factory for surface pricing."""
        params = self._params

        def cf_factory(t: float):
            return create_merton_cf(
                s0=s0, t=t, r=r,
                sigma=params.sigma,
                lambda_j=params.lambda_j,
                mu_j=params.mu_j,
                sigma_j=params.sigma_j,
            )

        return cf_factory

    # -------------------------------------------------------------------------
    # Monte Carlo Simulator
    # -------------------------------------------------------------------------

    def _create_terminal_simulator(self):
        """Create a terminal price simulator for the MC engine."""
        simulator = MertonSimulator(
            sigma=self._params.sigma,
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
            Time to maturity
        r : float
            Risk-free rate
        option_type : str or OptionType
            'call' or 'put'
        method : PricingMethod, optional
            FFT (default) or MONTE_CARLO
        n_paths : int, optional
            Number of simulation paths (for MC method)
        n_steps : int, optional
            Number of time steps (for MC method)
        seed : int, optional
            Random seed for reproducibility (for MC method)

        Returns
        -------
        PricingResult
            Pricing result with price, std_error (for MC), and metadata
        """
        self._validate_inputs(s0, k, t, r)
        opt_type = self._parse_option_type(option_type)

        # Default to FFT method
        method = method or PricingMethod.FFT

        # Validate method
        if method not in self.supported_methods:
            raise ValueError(
                f"Method {method} not supported for Merton. "
                f"Supported: {self.supported_methods}"
            )

        if method == PricingMethod.FFT:
            return self._price_fft(s0, k, t, r, opt_type)
        else:  # MONTE_CARLO
            return self._price_mc(s0, k, t, r, opt_type, n_paths, n_steps, seed)

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

        computation_time = time.perf_counter() - start_time

        return PricingResult(
            price=price,
            method=PricingMethod.FFT,
            computation_time=computation_time,
            parameters={
                "s0": s0, "k": k, "t": t, "r": r,
                "sigma": self._params.sigma,
                "lambda_j": self._params.lambda_j,
                "mu_j": self._params.mu_j,
                "sigma_j": self._params.sigma_j,
                "option_type": opt_type.value,
            },
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
                "sigma": self._params.sigma,
                "lambda_j": self._params.lambda_j,
                "mu_j": self._params.mu_j,
                "sigma_j": self._params.sigma_j,
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

        For FFT: Single FFT computation prices all strikes.
        For MC: Single simulation prices all strikes.

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
            FFT (default) or MONTE_CARLO

        Returns
        -------
        np.ndarray
            Array of option prices
        """
        strikes = np.asarray(strikes)
        opt_type = self._parse_option_type(option_type)
        is_call = opt_type == OptionType.CALL
        method = method or PricingMethod.FFT

        if method == PricingMethod.FFT:
            engine = self._get_fft_engine()
            cf = self._create_cf(s0, t, r)
            return engine.price_strikes(cf, s0, strikes, t, r, is_call)
        else:
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
            FFT (default) or MONTE_CARLO

        Returns
        -------
        np.ndarray
            2D array of prices [n_strikes x n_maturities]
        """
        strikes = np.asarray(strikes)
        maturities = np.asarray(maturities)
        opt_type = self._parse_option_type(option_type)
        is_call = opt_type == OptionType.CALL
        method = method or PricingMethod.FFT

        if method == PricingMethod.FFT:
            engine = self._get_fft_engine()
            cf_factory = self._create_cf_factory(s0, r)
            return engine.price_surface(cf_factory, s0, strikes, maturities, r, is_call)
        else:
            engine = self._get_mc_engine()
            terminal_sim = self._create_terminal_simulator()

            prices, _ = engine.price_surface(
                terminal_sim, s0, strikes, maturities, r, is_call,
                n_paths=n_paths, seed=seed
            )
            return prices


# =============================================================================
# Convenience Functions
# =============================================================================

def merton_call_price(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    method: PricingMethod = PricingMethod.FFT,
    n_paths: int = 100_000,
    seed: Optional[int] = None,
) -> float:
    """Merton call option price."""
    pricer = MertonPricer(
        sigma=sigma, lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j
    )
    result = pricer.price(
        s0, k, t, r,
        option_type=OptionType.CALL,
        method=method,
        n_paths=n_paths,
        seed=seed
    )
    return result.price


def merton_put_price(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    method: PricingMethod = PricingMethod.FFT,
    n_paths: int = 100_000,
    seed: Optional[int] = None,
) -> float:
    """Merton put option price."""
    pricer = MertonPricer(
        sigma=sigma, lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j
    )
    result = pricer.price(
        s0, k, t, r,
        option_type=OptionType.PUT,
        method=method,
        n_paths=n_paths,
        seed=seed
    )
    return result.price


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Merton Jump Diffusion Pricer Benchmark")
    print("=" * 60)

    # Test parameters
    s0, k, t, r = 100.0, 100.0, 0.25, 0.05
    sigma = 0.15
    lambda_j, mu_j, sigma_j = 1.0, -0.05, 0.10

    # Create pricer
    pricer = MertonPricer(
        sigma=sigma, lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j
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

    result_put_fft = pricer.price(s0, k, t, r, option_type="put")
    print(f"Put Price: ${result_put_fft.price:.4f}")

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

    # 3. Comparison
    print("\n3. FFT vs Monte Carlo Comparison")
    print("-" * 40)
    print(f"FFT:         ${result_fft.price:.4f}")
    print(f"Monte Carlo: ${result_mc.price:.4f}")
    print(f"Difference:  ${abs(result_fft.price - result_mc.price):.4f}")
    print(f"Std Errors:  {abs(result_fft.price - result_mc.price) / result_mc.std_error:.1f}σ")

    # 4. Compare with Black-Scholes (no jumps)
    print("\n4. Merton vs Black-Scholes (λ=0)")
    print("-" * 40)
    from .black_scholes import BlackScholesPricer
    bs_pricer = BlackScholesPricer(sigma=sigma)
    bs_result = bs_pricer.price(s0, k, t, r)
    print(f"Black-Scholes: ${bs_result.price:.4f}")
    print(f"Merton (λ={lambda_j}): ${result_fft.price:.4f}")
    print(f"Jump premium: ${result_fft.price - bs_result.price:.4f}")

    # 5. Surface pricing benchmark
    print("\n5. Surface Pricing (FFT)")
    print("-" * 40)
    strikes = np.linspace(80, 120, 50)
    maturities = np.array([0.1, 0.25, 0.5, 1.0])

    start = time.perf_counter()
    surface = pricer.price_surface(s0, strikes, maturities, r)
    elapsed = time.perf_counter() - start

    print(f"Surface: {len(strikes)} strikes x {len(maturities)} maturities")
    print(f"Time: {elapsed*1000:.2f} ms")
    print(f"Speed: {len(strikes)*len(maturities)/elapsed:,.0f} prices/sec")

    # Model info
    print(f"\nExpected jump size: {pricer.expected_jump_size*100:.2f}%")
    print(f"Supported methods: {pricer.supported_methods}")

    print()
