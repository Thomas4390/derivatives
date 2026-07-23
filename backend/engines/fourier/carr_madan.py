"""
Carr-Madan FFT Engine
=====================

Generic Fast Fourier Transform pricing engine using the Carr-Madan (1999) method.
This engine can price options for any model that provides a characteristic function.

The key insight is that the FFT algorithm is the same for all models - only the
characteristic function changes. This engine abstracts the FFT machinery.

References:
    Carr, P. and Madan, D.B. (1999). "Option valuation using the fast Fourier
    transform." Journal of Computational Finance, 2(4), 61-73.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numba import njit

from backend.utils.constants import (
    FFT_DEFAULT_ALPHA,
    FFT_DEFAULT_ETA,
    FFT_DEFAULT_N,
)


@dataclass(frozen=True)
class FFTConfig:
    """
    Configuration for FFT pricing.

    Parameters
    ----------
    alpha : float
        Damping factor (default 1.5). Controls the decay of the integrand.
        Typical values: 1.0 to 2.0. Higher values improve convergence for
        deep ITM/OTM options but may reduce accuracy for ATM.
    n_fft : int
        Number of FFT points (default 4096). Must be a power of 2.
        Higher values improve accuracy but increase computation time.
    eta : float
        Integration step size (default 0.25). Controls the grid spacing.
        Smaller values give finer grids but may cause numerical issues.
    """

    alpha: float = FFT_DEFAULT_ALPHA
    n_fft: int = FFT_DEFAULT_N
    eta: float = FFT_DEFAULT_ETA

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {self.alpha}")
        if self.n_fft <= 0 or (self.n_fft & (self.n_fft - 1)) != 0:
            raise ValueError(f"n_fft must be a positive power of 2, got {self.n_fft}")
        if self.eta <= 0:
            raise ValueError(f"eta must be positive, got {self.eta}")


# Type alias for characteristic function signature
# Takes u (complex array), returns phi(u) (complex array)
CharacteristicFunction = Callable[[np.ndarray], np.ndarray]


@njit(fastmath=True, cache=True)
def _simpson_weights(n: int, eta: float) -> np.ndarray:
    """Compute Simpson's rule integration weights."""
    weights = np.ones(n, dtype=np.float64)
    weights[1::2] = 4.0
    weights[2::2] = 2.0
    weights[0] = 1.0
    weights[-1] = 1.0
    return weights * eta / 3.0


class CarrMadanFFTEngine:
    """
    Generic FFT pricing engine using Carr-Madan method.

    This engine can price European options for any model that provides
    a characteristic function. The FFT algorithm transforms the
    characteristic function into option prices across a grid of strikes.

    Parameters
    ----------
    config : FFTConfig, optional
        FFT configuration. Uses defaults if not provided.

    Examples
    --------
    # Create engine
    engine = CarrMadanFFTEngine()

    # Define characteristic function for your model
    def my_cf(u: np.ndarray) -> np.ndarray:
        # ... compute phi(u) for your model
        return phi_u

    # Price a single strike
    call_price = engine.price_call(my_cf, s0=100, k=100, t=0.25, r=0.05)

    # Price multiple strikes efficiently (single FFT)
    prices = engine.price_strikes(my_cf, s0=100, strikes=np.array([90,100,110]), t=0.25, r=0.05)
    """

    def __init__(self, config: FFTConfig | None = None) -> None:
        self._config: FFTConfig = config or FFTConfig()

        # Pre-compute fixed arrays for efficiency
        self._v_grid: np.ndarray = np.arange(self._config.n_fft) * self._config.eta
        self._simpson: np.ndarray = _simpson_weights(
            self._config.n_fft, self._config.eta
        )
        self._lambda_spacing: float = (
            2 * np.pi / (self._config.n_fft * self._config.eta)
        )

    @property
    def config(self) -> FFTConfig:
        """Returns the FFT configuration."""
        return self._config

    def _compute_integrand(
        self,
        characteristic_fn: CharacteristicFunction,
        s0: float,
        t: float,
        r: float,
        alpha: float | None = None,
    ) -> np.ndarray:
        """
        Compute the Carr-Madan integrand.

        The damped price transform is:
            psi(v) = exp(-rT) * phi(v - (alpha+1)i) / (alpha^2 + alpha - v^2 + i(2alpha+1)v)

        With ``alpha > 0`` this transform yields CALL prices; by Lee's duality
        the very same expression with ``alpha < -1`` yields PUT prices directly
        (see :meth:`price_strikes`). The damping is therefore parameterised
        rather than always read from the config.

        Parameters
        ----------
        characteristic_fn : callable
            Function that computes phi(u) for complex u
        s0 : float
            Spot price (used for log-moneyness centering)
        t : float
            Time to maturity
        r : float
            Risk-free rate
        alpha : float, optional
            Damping factor. Defaults to the configured call damping
            ``self._config.alpha``.

        Returns
        -------
        np.ndarray
            Complex integrand values
        """
        if alpha is None:
            alpha = self._config.alpha
        v = self._v_grid

        # Shifted argument for characteristic function
        u = v - (alpha + 1) * 1j

        # Evaluate characteristic function
        cf_values = characteristic_fn(u)

        # Carr-Madan denominator
        denominator = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v

        # Integrand. The characteristic function can overflow to ±inf/nan on the
        # high-frequency tail of the v-grid (e.g. the Heston/Bates exponent
        # exp(C + D v0) blows up before its asymptotic decay kicks in
        # numerically). The true damped transform decays to 0 there, so we
        # neutralise the non-finite tail rather than let a single nan/inf poison
        # the whole FFT (which would NaN out every price on the slice). For a
        # finite cf_values this is a no-op, leaving valid prices bit-identical.
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            integrand = np.exp(-r * t) * cf_values / denominator
        integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)

        return integrand

    def _fft_transform(
        self, integrand: np.ndarray, log_s0: float, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply FFT and return damped prices and log-strike grid.

        The damping ``alpha`` must match the one used to build ``integrand``
        (positive for calls, ``< -1`` for the direct put transform). The
        log-strike grid is independent of ``alpha``.

        Returns
        -------
        damped_prices : np.ndarray
            Damped option prices on log-strike grid
        log_strikes : np.ndarray
            Log-strike grid
        """
        if alpha is None:
            alpha = self._config.alpha

        # Apply Simpson weights and phase shift
        x = integrand * self._simpson * np.exp(-1j * self._v_grid * (-log_s0))

        # FFT
        fft_result = np.fft.fft(x)

        # Log-strike grid (centered at log(S0))
        log_strikes = -log_s0 + self._lambda_spacing * np.arange(self._config.n_fft)

        # Damped prices
        damped_prices = np.exp(-alpha * log_strikes) / np.pi * np.real(fft_result)

        return damped_prices, log_strikes

    def _interpolate_price(
        self,
        damped_prices: np.ndarray,
        log_strikes_grid: np.ndarray,
        log_k: float,
        log_s0: float,
    ) -> float:
        """Interpolate to get price at specific log-strike."""
        # Find index in grid
        idx = int((log_k + log_s0) / self._lambda_spacing)
        idx = min(max(idx, 0), self._config.n_fft - 2)

        # Linear interpolation weight
        w = (log_k - log_strikes_grid[idx]) / self._lambda_spacing
        w = min(max(w, 0.0), 1.0)

        # Interpolated price
        price = (1 - w) * damped_prices[idx] + w * damped_prices[idx + 1]

        return max(price, 0.0)

    def price_call(
        self,
        characteristic_fn: CharacteristicFunction,
        s0: float,
        k: float,
        t: float,
        r: float,
        q: float = 0.0,
    ) -> float:
        """
        Price a single European call option.

        Parameters
        ----------
        characteristic_fn : callable
            Characteristic function phi(u) for complex u
        s0 : float
            Spot price
        k : float
            Strike price
        t : float
            Time to maturity
        r : float
            Risk-free rate
        q : float
            Dividend yield (default 0)

        Returns
        -------
        float
            Call option price
        """
        log_s0 = np.log(s0)
        log_k = np.log(k)

        # Compute integrand (characteristic function already includes q)
        integrand = self._compute_integrand(characteristic_fn, s0, t, r)

        # FFT transform
        damped_prices, log_strikes = self._fft_transform(integrand, log_s0)

        # Interpolate to target strike
        return self._interpolate_price(damped_prices, log_strikes, log_k, log_s0)

    def price_put(
        self,
        characteristic_fn: CharacteristicFunction,
        s0: float,
        k: float,
        t: float,
        r: float,
        q: float = 0.0,
    ) -> float:
        """
        Price a single European put option via the direct Carr-Madan transform.

        Uses the negative-damping (Lee duality) transform rather than
        ``call FFT + put-call parity`` so deep-OTM puts stay accurate (see
        :meth:`price_strikes`).

        Parameters
        ----------
        characteristic_fn : callable
            Characteristic function phi(u) for complex u
        s0 : float
            Spot price
        k : float
            Strike price
        t : float
            Time to maturity
        r : float
            Risk-free rate
        q : float
            Dividend yield (default 0)

        Returns
        -------
        float
            Put option price
        """
        prices = self.price_strikes(
            characteristic_fn,
            s0,
            np.array([k], dtype=np.float64),
            t,
            r,
            is_call=False,
            q=q,
        )
        return float(prices[0])

    def price_strikes(
        self,
        characteristic_fn: CharacteristicFunction,
        s0: float,
        strikes: np.ndarray,
        t: float,
        r: float,
        is_call: bool = True,
        q: float = 0.0,
    ) -> np.ndarray:
        """
        Price multiple strikes efficiently with a single FFT.

        This is much more efficient than calling price_call/price_put
        repeatedly, as the FFT is computed only once.

        Parameters
        ----------
        characteristic_fn : callable
            Characteristic function phi(u) for complex u
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
        q : float
            Dividend yield (default 0)

        Returns
        -------
        np.ndarray
            Array of option prices
        """
        log_s0 = np.log(s0)
        strikes = np.asarray(strikes, dtype=np.float64)

        # Single FFT computation (call damping)
        alpha = self._config.alpha
        integrand = self._compute_integrand(characteristic_fn, s0, t, r, alpha)
        damped_prices, log_strikes_grid = self._fft_transform(integrand, log_s0, alpha)

        # Vectorized linear interpolation over all strikes at once (same formula
        # as _interpolate_price, lifted out of the per-strike Python loop). The
        # grid and hence (idx, w) are independent of the damping, so the same
        # weights interpolate both the call and the direct-put transforms.
        log_k = np.log(strikes)
        idx = ((log_k + log_s0) / self._lambda_spacing).astype(np.int64)
        idx = np.clip(idx, 0, self._config.n_fft - 2)
        w = np.clip((log_k - log_strikes_grid[idx]) / self._lambda_spacing, 0.0, 1.0)
        call_prices = (1.0 - w) * damped_prices[idx] + w * damped_prices[idx + 1]
        call_prices = np.maximum(call_prices, 0.0)

        if is_call:
            return call_prices

        # Direct put pricing (Lee duality): the same damped transform with
        # alpha_put = -alpha_call - 1 (< -1) returns puts directly. This avoids
        # the catastrophic cancellation of ``call FFT + parity`` in the deep put
        # wing, where the ~1e-4 absolute FFT error on the O(S) call swamps the
        # O(1e-6) put and yields negative/garbage prices.
        alpha_put = -alpha - 1.0
        put_integrand = self._compute_integrand(characteristic_fn, s0, t, r, alpha_put)
        put_damped, _ = self._fft_transform(put_integrand, log_s0, alpha_put)
        put_prices = (1.0 - w) * put_damped[idx] + w * put_damped[idx + 1]

        # Safety net: some exotic characteristic functions may not be analytic
        # in the negative-damping band. Fall back cell-by-cell to put-call parity
        # wherever the direct put value is non-finite.
        parity_prices = call_prices - s0 * np.exp(-q * t) + strikes * np.exp(-r * t)
        put_prices = np.where(np.isfinite(put_prices), put_prices, parity_prices)
        put_prices = np.nan_to_num(put_prices, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(put_prices, 0.0)

    def price_surface(
        self,
        characteristic_fn_factory: Callable[[float], CharacteristicFunction],
        s0: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        r: float,
        is_call: bool = True,
        q: float = 0.0,
    ) -> np.ndarray:
        """
        Price options across a strike-maturity surface.

        Parameters
        ----------
        characteristic_fn_factory : callable
            Function that takes maturity t and returns the characteristic
            function for that maturity: factory(t) -> cf(u)
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
        q : float
            Dividend yield (default 0)

        Returns
        -------
        np.ndarray
            2D array of prices [n_strikes x n_maturities]
        """
        n_k = len(strikes)
        n_t = len(maturities)
        prices = np.empty((n_k, n_t))

        for j, t in enumerate(maturities):
            # Get characteristic function for this maturity
            cf = characteristic_fn_factory(t)

            # Price all strikes with single FFT
            prices[:, j] = self.price_strikes(cf, s0, strikes, t, r, is_call, q)

        return prices


# =============================================================================
# Convenience function for quick pricing
# =============================================================================


def fft_price(
    characteristic_fn: CharacteristicFunction,
    s0: float,
    k: float,
    t: float,
    r: float,
    is_call: bool = True,
    q: float = 0.0,
    alpha: float = FFT_DEFAULT_ALPHA,
    n_fft: int = FFT_DEFAULT_N,
) -> float:
    """
    Quick FFT pricing with default configuration.

    Parameters
    ----------
    characteristic_fn : callable
        Characteristic function phi(u)
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
    q : float
        Dividend yield (default 0)
    alpha : float
        Damping factor
    n_fft : int
        Number of FFT points

    Returns
    -------
    float
        Option price
    """
    config = FFTConfig(alpha=alpha, n_fft=n_fft)
    engine = CarrMadanFFTEngine(config)

    if is_call:
        return engine.price_call(characteristic_fn, s0, k, t, r, q)
    return engine.price_put(characteristic_fn, s0, k, t, r, q)


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    from backend.models.characteristic_functions.heston_cf import heston_cf_vectorized

    print("=" * 50)
    print("Carr-Madan FFT Engine Smoke Test")
    print("=" * 50)

    # Test parameters
    s0, k, t, r = 100.0, 100.0, 0.5, 0.05
    v0, kappa, theta, alpha, rho = 0.04, 2.0, 0.04, 0.3, -0.7

    # Create characteristic function for Heston model
    def heston_cf(u: np.ndarray) -> np.ndarray:
        return heston_cf_vectorized(u, s0, v0, t, r, kappa, theta, alpha, rho)

    # Test FFT config
    print("\n--- FFT Configuration ---")
    config = FFTConfig(alpha=1.5, n_fft=4096)
    print(f"Alpha: {config.alpha}")
    print(f"N_FFT: {config.n_fft}")
    print(f"Eta:   {config.eta}")

    # Test config validation
    try:
        FFTConfig(alpha=-1.0)
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"Config validation works: {e}")

    # Test call pricing
    print("\n--- Call Pricing ---")
    engine = CarrMadanFFTEngine(config)
    call_price = engine.price_call(heston_cf, s0, k, t, r)
    print(f"ATM Call (K={k}): ${call_price:.4f}")

    # Test put pricing
    print("\n--- Put Pricing ---")
    put_price = engine.price_put(heston_cf, s0, k, t, r)
    print(f"ATM Put (K={k}): ${put_price:.4f}")

    # Test put-call parity: C - P = S - K*exp(-rT)
    parity_check = call_price - put_price
    parity_expected = s0 - k * np.exp(-r * t)
    print("\nPut-Call Parity:")
    print(f"  C - P = ${parity_check:.4f}")
    print(f"  S - K*e^(-rT) = ${parity_expected:.4f}")
    print(f"  Difference: ${abs(parity_check - parity_expected):.6f}")
    assert abs(parity_check - parity_expected) < 0.01, "Put-call parity violated"
    print("  Put-call parity: ✓")

    # Test multiple strikes
    print("\n--- Multiple Strikes ---")
    strikes = np.array([90, 95, 100, 105, 110])
    call_prices = engine.price_strikes(heston_cf, s0, strikes, t, r, is_call=True)
    put_prices = engine.price_strikes(heston_cf, s0, strikes, t, r, is_call=False)
    print(f"{'Strike':>8} {'Call':>10} {'Put':>10}")
    print("-" * 30)
    for i, strike in enumerate(strikes):
        print(f"{strike:>8} ${call_prices[i]:>9.4f} ${put_prices[i]:>9.4f}")

    # Test convenience function
    print("\n--- Convenience Function ---")
    quick_price = fft_price(heston_cf, s0, k, t, r, is_call=True)
    print(f"fft_price result: ${quick_price:.4f}")
    assert abs(quick_price - call_price) < 1e-10, "Convenience function mismatch"
    print("Convenience function matches engine: ✓")

    print("\n" + "=" * 50)
    print("Carr-Madan FFT Engine smoke test passed")
    print("=" * 50)
