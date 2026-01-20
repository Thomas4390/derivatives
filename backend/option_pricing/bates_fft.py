"""
Bates FFT Option Pricer
=======================

Fast Fourier Transform pricing for the Bates stochastic volatility
with jumps model using the Carr-Madan (1999) method.

The Bates model combines Heston with Merton-style jumps:
    dS = μ·S·dt + √V·S·dW_S + (J - 1)·S·dN
    dV = κ·(θ - V)·dt + ξ·√V·dW_V
    Corr(dW_S, dW_V) = ρ

Where dN is a Poisson process with intensity λ and
ln(J) ~ N(μ_J, σ²_J).

References:
    - Bates, D.S. (1996). "Jumps and Stochastic Volatility: Exchange Rate
      Processes Implicit in Deutsche Mark Options." Review of Financial
      Studies, 9(1), 69-107.
    - Carr, P. and Madan, D.B. (1999). "Option valuation using the fast
      Fourier transform." Journal of Computational Finance, 2(4), 61-73.

Author: Derivatives Pricing Project
"""

import numpy as np
from numba import njit
import time
import sys
import warnings
from pathlib import Path
from typing import Union

# Handle imports for both package and direct execution
try:
    from .base import (
        FFTPricer,
        PricingResult,
        PricingMethod,
        OptionType,
    )
except ImportError:
    _project_root = Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from backend.option_pricing.base import (
        FFTPricer,
        PricingResult,
        PricingMethod,
        OptionType,
    )

# =============================================================================
# Characteristic Function - Import from shared module
# =============================================================================

# Import shared characteristic function (single source of truth)
try:
    from backend.models.characteristic_functions.bates_cf import (
        bates_characteristic_function as _bates_characteristic_function,
        bates_cf_vectorized as _bates_cf_array,
    )
except ImportError:
    _project_root = Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from backend.models.characteristic_functions.bates_cf import (
        bates_characteristic_function as _bates_characteristic_function,
        bates_cf_vectorized as _bates_cf_array,
    )


# =============================================================================
# Carr-Madan FFT Implementation for Bates
# =============================================================================

@njit(fastmath=True, cache=True)
def _bates_carr_madan_integrand(
    v: np.ndarray,
    alpha: float,
    s0: float,
    v0: float,
    t: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float
) -> np.ndarray:
    """Compute the Carr-Madan integrand for FFT with Bates model."""
    n = len(v)
    result = np.empty(n, dtype=np.complex128)

    i = 1j

    for idx in range(n):
        u = v[idx] - (alpha + 1) * i

        # Characteristic function
        cf = _bates_characteristic_function(
            u, s0, v0, t, r, kappa, theta, xi, rho,
            lambda_j, mu_j, sigma_j
        )

        # Carr-Madan integrand
        denominator = alpha ** 2 + alpha - v[idx] ** 2 + i * (2 * alpha + 1) * v[idx]

        result[idx] = np.exp(-r * t) * cf / denominator

    return result


def _bates_fft_price(
    s0: float,
    k: float,
    v0: float,
    t: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    alpha: float = 1.5,
    n_fft: int = 4096,
    eta: float = 0.25
) -> float:
    """Price a call option using Bates FFT method."""
    # Log-strike spacing
    lambda_spacing = 2 * np.pi / (n_fft * eta)

    # Integration grid
    v_grid = np.arange(n_fft) * eta

    # Simpson's rule weights
    simpson = np.ones(n_fft)
    simpson[1::2] = 4.0
    simpson[2::2] = 2.0
    simpson[0] = 1.0
    simpson[-1] = 1.0
    simpson = simpson * eta / 3.0

    # Compute integrand
    integrand = _bates_carr_madan_integrand(
        v_grid, alpha, s0, v0, t, r, kappa, theta, xi, rho,
        lambda_j, mu_j, sigma_j
    )

    # Apply Simpson weights and phase shift
    log_s0 = np.log(s0)
    x = integrand * simpson * np.exp(-1j * v_grid * (-log_s0))

    # FFT
    fft_result = np.fft.fft(x)

    # Log-strike grid
    log_strikes = -log_s0 + lambda_spacing * np.arange(n_fft)

    # Damped call prices
    damped_prices = np.exp(-alpha * log_strikes) / np.pi * np.real(fft_result)

    # Interpolate to get price at desired strike
    log_k = np.log(k)
    idx = int((log_k + log_s0) / lambda_spacing)
    idx = min(max(idx, 0), n_fft - 2)

    w = (log_k - log_strikes[idx]) / lambda_spacing
    price = (1 - w) * damped_prices[idx] + w * damped_prices[idx + 1]

    return max(price, 0.0)


def _bates_fft_surface(
    s0: float,
    strikes: np.ndarray,
    v0: float,
    t: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    alpha: float = 1.5,
    n_fft: int = 4096,
    eta: float = 0.25
) -> np.ndarray:
    """Price multiple strikes efficiently with single FFT."""
    lambda_spacing = 2 * np.pi / (n_fft * eta)
    v_grid = np.arange(n_fft) * eta

    simpson = np.ones(n_fft)
    simpson[1::2] = 4.0
    simpson[2::2] = 2.0
    simpson[0] = 1.0
    simpson[-1] = 1.0
    simpson = simpson * eta / 3.0

    integrand = _bates_carr_madan_integrand(
        v_grid, alpha, s0, v0, t, r, kappa, theta, xi, rho,
        lambda_j, mu_j, sigma_j
    )

    log_s0 = np.log(s0)
    x = integrand * simpson * np.exp(-1j * v_grid * (-log_s0))

    fft_result = np.fft.fft(x)
    log_strikes_grid = -log_s0 + lambda_spacing * np.arange(n_fft)
    damped_prices = np.exp(-alpha * log_strikes_grid) / np.pi * np.real(fft_result)

    prices = np.empty(len(strikes))
    for i, k in enumerate(strikes):
        log_k = np.log(k)
        idx = int((log_k + log_s0) / lambda_spacing)
        idx = min(max(idx, 0), n_fft - 2)

        w = (log_k - log_strikes_grid[idx]) / lambda_spacing
        w = min(max(w, 0.0), 1.0)
        prices[i] = max((1 - w) * damped_prices[idx] + w * damped_prices[idx + 1], 0.0)

    return prices


# =============================================================================
# Bates FFT Pricer Class
# =============================================================================

class BatesFFTPricer(FFTPricer):
    """
    Bates stochastic volatility with jumps option pricer using FFT.

    Combines Heston stochastic volatility with Merton-style price jumps.

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
    alpha : float
        Damping factor for FFT (default 1.5)
    n_fft : int
        Number of FFT points (default 4096)

    Examples
    --------
    pricer = BatesFFTPricer(
        v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
    )
    result = pricer.price(s0=100, k=100, t=0.25, r=0.05)
    """

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
        alpha: float = 1.5,
        n_fft: int = 4096
    ):
        super().__init__()
        self._model_name = "Bates (FFT)"

        # Heston parameters
        self._v0 = v0
        self._kappa = kappa
        self._theta = theta
        self._xi = xi
        self._rho = rho

        # Jump parameters
        self._lambda_j = lambda_j
        self._mu_j = mu_j
        self._sigma_j = sigma_j

        # FFT parameters
        self._alpha = alpha
        self._n_fft = n_fft

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate Bates parameters."""
        if self._v0 < 0:
            raise ValueError(f"Initial variance v0 must be non-negative, got {self._v0}")
        if self._kappa <= 0:
            raise ValueError(f"Mean reversion kappa must be positive, got {self._kappa}")
        if self._theta < 0:
            raise ValueError(f"Long-run variance theta must be non-negative, got {self._theta}")
        if self._xi <= 0:
            raise ValueError(f"Vol of vol xi must be positive, got {self._xi}")
        if not -1 <= self._rho <= 1:
            raise ValueError(f"Correlation rho must be in [-1, 1], got {self._rho}")
        if self._lambda_j < 0:
            raise ValueError(f"Jump intensity must be non-negative, got {self._lambda_j}")
        if self._sigma_j < 0:
            raise ValueError(f"Jump vol must be non-negative, got {self._sigma_j}")

    # Properties
    @property
    def v0(self) -> float:
        return self._v0

    @property
    def kappa(self) -> float:
        return self._kappa

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def xi(self) -> float:
        return self._xi

    @property
    def rho(self) -> float:
        return self._rho

    @property
    def lambda_j(self) -> float:
        return self._lambda_j

    @property
    def mu_j(self) -> float:
        return self._mu_j

    @property
    def sigma_j(self) -> float:
        return self._sigma_j

    @property
    def expected_jump_size(self) -> float:
        """E[J - 1], the expected percentage jump."""
        return np.exp(self._mu_j + 0.5 * self._sigma_j ** 2) - 1

    def feller_condition_satisfied(self) -> bool:
        """Check if Feller condition (2κθ > ξ²) is satisfied."""
        return 2 * self._kappa * self._theta > self._xi ** 2

    def characteristic_function(
        self,
        u: np.ndarray,
        s0: float,
        t: float,
        r: float,
        **kwargs
    ) -> np.ndarray:
        """Compute Bates characteristic function."""
        return _bates_cf_array(
            u.astype(np.complex128), s0, self._v0, t, r,
            self._kappa, self._theta, self._xi, self._rho,
            self._lambda_j, self._mu_j, self._sigma_j
        )

    def price(
        self,
        s0: float,
        k: float,
        t: float,
        r: float,
        option_type: Union[str, OptionType] = OptionType.CALL,
        **kwargs
    ) -> PricingResult:
        """
        Price a European option using Bates FFT.

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

        Returns
        -------
        PricingResult
            Pricing result
        """
        self._validate_inputs(s0, k, t, r)
        opt_type = self._parse_option_type(option_type)

        start_time = time.perf_counter()

        call_price = _bates_fft_price(
            s0, k, self._v0, t, r,
            self._kappa, self._theta, self._xi, self._rho,
            self._lambda_j, self._mu_j, self._sigma_j,
            self._alpha, self._n_fft
        )

        if opt_type == OptionType.PUT:
            price = call_price - s0 + k * np.exp(-r * t)
        else:
            price = call_price

        # Warn if price is negative (indicates numerical issues)
        if price < 0:
            warnings.warn(
                f"Negative {opt_type.value} price ({price:.6f}) computed via FFT. "
                f"This may indicate numerical instability or put-call parity violation. "
                f"Price floored to 0.",
                RuntimeWarning
            )

        computation_time = time.perf_counter() - start_time

        return PricingResult(
            price=max(price, 0.0),
            method=PricingMethod.FFT,
            computation_time=computation_time,
            parameters={
                "s0": s0, "k": k, "t": t, "r": r,
                "v0": self._v0, "kappa": self._kappa,
                "theta": self._theta, "xi": self._xi, "rho": self._rho,
                "lambda_j": self._lambda_j, "mu_j": self._mu_j,
                "sigma_j": self._sigma_j,
                "option_type": opt_type.value
            }
        )

    def price_surface(
        self,
        s0: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        r: float,
        option_type: Union[str, OptionType] = OptionType.CALL,
        **kwargs
    ) -> np.ndarray:
        """Price options across strike-maturity surface."""
        opt_type = self._parse_option_type(option_type)
        is_put = opt_type == OptionType.PUT

        n_k = len(strikes)
        n_t = len(maturities)
        prices = np.empty((n_k, n_t))

        for j, t in enumerate(maturities):
            call_prices = _bates_fft_surface(
                s0, strikes, self._v0, t, r,
                self._kappa, self._theta, self._xi, self._rho,
                self._lambda_j, self._mu_j, self._sigma_j,
                self._alpha, self._n_fft
            )

            if is_put:
                prices[:, j] = call_prices - s0 + strikes * np.exp(-r * t)
            else:
                prices[:, j] = call_prices

        return np.maximum(prices, 0.0)

    def price_strikes(
        self,
        s0: float,
        strikes: np.ndarray,
        t: float,
        r: float,
        option_type: Union[str, OptionType] = OptionType.CALL
    ) -> np.ndarray:
        """Price multiple strikes at single maturity."""
        opt_type = self._parse_option_type(option_type)

        call_prices = _bates_fft_surface(
            s0, strikes, self._v0, t, r,
            self._kappa, self._theta, self._xi, self._rho,
            self._lambda_j, self._mu_j, self._sigma_j,
            self._alpha, self._n_fft
        )

        if opt_type == OptionType.PUT:
            return np.maximum(call_prices - s0 + strikes * np.exp(-r * t), 0.0)
        return np.maximum(call_prices, 0.0)


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
    sigma_j: float
) -> float:
    """Bates call option price using FFT."""
    return _bates_fft_price(
        s0, k, v0, t, r, kappa, theta, xi, rho,
        lambda_j, mu_j, sigma_j
    )


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
    sigma_j: float
) -> float:
    """Bates put option price using FFT."""
    call = _bates_fft_price(
        s0, k, v0, t, r, kappa, theta, xi, rho,
        lambda_j, mu_j, sigma_j
    )
    return max(call - s0 + k * np.exp(-r * t), 0.0)


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Bates FFT Pricer Benchmark")
    print("=" * 60)

    # Test parameters
    s0, k, t, r = 100.0, 100.0, 0.25, 0.05
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7
    lambda_j, mu_j, sigma_j = 0.5, -0.1, 0.2

    # Warmup
    print("\nWarming up...")
    pricer = BatesFFTPricer(
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j
    )
    _ = pricer.price(s0, k, t, r)

    # Single option pricing
    print("\n1. Single Option Pricing")
    print("-" * 40)
    result = pricer.price(s0, k, t, r, option_type="call")
    print(f"Call Price: ${result.price:.4f}")
    print(f"Time: {result.computation_time*1000:.2f} ms")

    result_put = pricer.price(s0, k, t, r, option_type="put")
    print(f"Put Price: ${result_put.price:.4f}")

    print(f"\nExpected jump size: {pricer.expected_jump_size*100:.1f}%")
    print(f"Feller condition: {pricer.feller_condition_satisfied()}")

    # Compare Bates vs Heston (no jumps)
    print("\n2. Bates vs Heston (λ=0)")
    print("-" * 40)
    pricer_no_jumps = BatesFFTPricer(
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        lambda_j=0.0, mu_j=0.0, sigma_j=0.0
    )
    result_no_jumps = pricer_no_jumps.price(s0, k, t, r)
    print(f"Bates (λ=0): ${result_no_jumps.price:.4f}")
    print(f"Bates (λ={lambda_j}): ${result.price:.4f}")
    print(f"Jump premium: ${result.price - result_no_jumps.price:.4f}")

    # Multi-strike pricing
    print("\n3. Multi-Strike Pricing")
    print("-" * 40)
    strikes = np.linspace(80, 120, 41)

    start = time.perf_counter()
    prices = pricer.price_strikes(s0, strikes, t, r)
    elapsed = time.perf_counter() - start

    print(f"Strikes: {len(strikes)}")
    print(f"Time: {elapsed*1000:.2f} ms")
    print(f"Speed: {len(strikes)/elapsed:,.0f} prices/sec")

    # Surface pricing
    print("\n4. Surface Pricing Benchmark")
    print("-" * 40)
    strikes = np.linspace(80, 120, 50)
    maturities = np.array([0.1, 0.25, 0.5, 1.0, 2.0])

    start = time.perf_counter()
    surface = pricer.price_surface(s0, strikes, maturities, r)
    elapsed = time.perf_counter() - start

    print(f"Surface: {len(strikes)} x {len(maturities)} = {len(strikes)*len(maturities)}")
    print(f"Time: {elapsed*1000:.2f} ms")
    print(f"Speed: {len(strikes)*len(maturities)/elapsed:,.0f} prices/sec")

    print()
