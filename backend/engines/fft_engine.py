"""
FFT Pricing Engine
==================

Carr-Madan FFT pricing engine wrapper for the new architecture.

This engine wraps the generic CarrMadanFFTEngine and implements the
PricingEngine interface, bridging Instrument/Model/Market to the
underlying FFT machinery.

Author: Thomas
Created: 2025
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from backend.core.interfaces import PricingEngine, Instrument, Model
from backend.core.market import MarketEnvironment
from backend.core.result_types import (
    PricingResult, PricingCapability, ExerciseStyle
)
from backend.instruments.options import VanillaOption
from backend.engines.fourier.carr_madan import (
    CarrMadanFFTEngine, FFTConfig
)


# =============================================================================
# FFT ENGINE
# =============================================================================

@dataclass
class FFTEngine(PricingEngine):
    """
    FFT pricing engine using Carr-Madan method.

    Prices European options for any model that provides a characteristic
    function. The engine extracts the characteristic function from the
    model and delegates to the underlying CarrMadanFFTEngine.

    Supported Instruments
    ---------------------
    - European vanilla calls and puts (VanillaOption with ExerciseStyle.EUROPEAN)

    Supported Models
    ----------------
    - Any model with characteristic_function() method
    - GBMModel, HestonModel, BatesModel, MertonModel

    Parameters
    ----------
    config : FFTConfig, optional
        FFT configuration (alpha, n_fft, eta)

    Examples
    --------
    engine = FFTEngine()

    # Works with any model that has characteristic function
    option = VanillaOption(strike=100, maturity=0.5, is_call=True)
    model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    market = MarketEnvironment(spot=100, rate=0.05)

    result = engine.price(option, model, market)
    """

    config: FFTConfig = None

    def __post_init__(self):
        """Initialize the underlying FFT engine."""
        if self.config is None:
            self.config = FFTConfig()
        self._fft_engine = CarrMadanFFTEngine(self.config)

    @property
    def capability(self) -> PricingCapability:
        """This is an FFT engine."""
        return PricingCapability.FFT

    @property
    def supported_exercises(self) -> List[ExerciseStyle]:
        """Only European exercise supported by FFT."""
        return [ExerciseStyle.EUROPEAN]

    def can_price(self, instrument: Instrument, model: Model) -> bool:
        """
        Check if this engine can price the given combination.

        FFTEngine requires:
        - European exercise style
        - Model supports FFT (has characteristic function)
        """
        if instrument.exercise_style != ExerciseStyle.EUROPEAN:
            return False

        if PricingCapability.FFT not in model.supported_engines:
            return False

        return True

    def price(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
    ) -> PricingResult:
        """
        Price a European option using FFT.

        Parameters
        ----------
        instrument : VanillaOption
            European vanilla option
        model : Model
            Model with characteristic function
        market : MarketEnvironment
            Current market conditions

        Returns
        -------
        PricingResult
            FFT price
        """
        if not self.can_price(instrument, model):
            raise ValueError(
                f"FFTEngine cannot price {type(instrument).__name__} "
                f"with {type(model).__name__}"
            )

        # Extract option parameters
        option = instrument  # type: VanillaOption
        s0 = market.spot
        k = option.strike
        t = option.maturity
        r = market.rate
        q = market.dividend_yield

        # Build characteristic function closure
        def cf(u: np.ndarray) -> np.ndarray:
            """Characteristic function for FFT."""
            return model.characteristic_function_vectorized(u, s0, t, r, q)

        # Price using FFT
        if option.is_call:
            price = self._fft_engine.price_call(cf, s0, k, t, r)
        else:
            price = self._fft_engine.price_put(cf, s0, k, t, r, q)

        return PricingResult(
            price=price,
            engine="FFTEngine",
            model=model.name
        )

    def price_strikes(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
        strikes: np.ndarray,
    ) -> np.ndarray:
        """
        Price multiple strikes efficiently with single FFT.

        Parameters
        ----------
        instrument : VanillaOption
            Template option (strike will be ignored)
        model : Model
            Model with characteristic function
        market : MarketEnvironment
            Current market conditions
        strikes : np.ndarray
            Array of strikes to price

        Returns
        -------
        np.ndarray
            Array of prices
        """
        if not self.can_price(instrument, model):
            raise ValueError(
                f"FFTEngine cannot price {type(instrument).__name__} "
                f"with {type(model).__name__}"
            )

        option = instrument  # type: VanillaOption
        s0 = market.spot
        t = option.maturity
        r = market.rate
        q = market.dividend_yield

        # Build characteristic function closure
        def cf(u: np.ndarray) -> np.ndarray:
            return model.characteristic_function_vectorized(u, s0, t, r, q)

        # Price all strikes with single FFT
        return self._fft_engine.price_strikes(cf, s0, strikes, t, r, option.is_call, q)

    def price_surface(
        self,
        model: Model,
        market: MarketEnvironment,
        strikes: np.ndarray,
        maturities: np.ndarray,
        is_call: bool = True,
    ) -> np.ndarray:
        """
        Price options across strike-maturity surface.

        Parameters
        ----------
        model : Model
            Model with characteristic function
        market : MarketEnvironment
            Current market conditions
        strikes : np.ndarray
            Array of strikes
        maturities : np.ndarray
            Array of maturities
        is_call : bool
            True for calls, False for puts

        Returns
        -------
        np.ndarray
            2D array of prices [n_strikes x n_maturities]
        """
        s0 = market.spot
        r = market.rate
        q = market.dividend_yield

        # Factory that creates CF for each maturity
        def cf_factory(t: float):
            def cf(u: np.ndarray) -> np.ndarray:
                return model.characteristic_function_vectorized(u, s0, t, r, q)
            return cf

        return self._fft_engine.price_surface(
            cf_factory, s0, strikes, maturities, r, is_call, q
        )


if __name__ == "__main__":
    from backend.instruments.options import VanillaOption
    from backend.models.gbm import GBMModel
    from backend.models.heston import HestonModel
    from backend.models.bates import BatesModel
    from backend.models.merton import MertonModel
    from backend.core.market import MarketEnvironment
    from backend.engines import BSAnalyticEngine

    print("=" * 50)
    print("FFTEngine Smoke Test")
    print("=" * 50)

    # Create components
    engine = FFTEngine()
    option = VanillaOption(strike=100, maturity=0.5, is_call=True)
    market = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.02)

    print(f"\n--- Setup ---")
    print(f"Option: K={option.strike}, T={option.maturity}, Call={option.is_call}")
    print(f"Market: S0={market.spot}, r={market.rate}, q={market.dividend_yield}")

    # Test with GBM (compare to analytical)
    print(f"\n--- GBM Model (vs Analytical) ---")
    gbm = GBMModel(sigma=0.20)
    fft_result = engine.price(option, gbm, market)
    print(f"FFT Price: ${fft_result.price:.4f}")

    bs_engine = BSAnalyticEngine()
    bs_result = bs_engine.price(option, gbm, market)
    print(f"BS Price: ${bs_result.price:.4f}")
    print(f"Difference: ${abs(fft_result.price - bs_result.price):.6f}")

    # Test with Heston
    print(f"\n--- Heston Model ---")
    heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    heston_result = engine.price(option, heston, market)
    print(f"Heston Price: ${heston_result.price:.4f}")
    print(f"Model: {heston_result.model}")

    # Test with Bates
    print(f"\n--- Bates Model ---")
    bates = BatesModel(
        v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
    )
    bates_result = engine.price(option, bates, market)
    print(f"Bates Price: ${bates_result.price:.4f}")

    # Test with Merton
    print(f"\n--- Merton Model ---")
    merton = MertonModel(sigma=0.20, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)
    merton_result = engine.price(option, merton, market)
    print(f"Merton Price: ${merton_result.price:.4f}")

    # Test put option
    print(f"\n--- Put Options ---")
    put_option = VanillaOption(strike=100, maturity=0.5, is_call=False)
    for model_name, model in [("GBM", gbm), ("Heston", heston), ("Bates", bates)]:
        put_result = engine.price(put_option, model, market)
        print(f"{model_name} Put: ${put_result.price:.4f}")

    # Test multiple strikes
    print(f"\n--- Multiple Strikes (Heston) ---")
    strikes = np.array([90, 95, 100, 105, 110])
    prices = engine.price_strikes(option, heston, market, strikes)
    for k, p in zip(strikes, prices):
        print(f"  K={k}: ${p:.4f}")

    # Test price surface
    print(f"\n--- Price Surface ---")
    maturities = np.array([0.25, 0.5, 1.0])
    surface = engine.price_surface(heston, market, strikes, maturities)
    print(f"Surface shape: {surface.shape}")
    print(f"ATM prices by maturity: {surface[2, :]}")

    # Test can_price
    print(f"\n--- Compatibility Check ---")
    print(f"Can price European call with Heston: {engine.can_price(option, heston)}")
    print(f"Can price European call with GBM: {engine.can_price(option, gbm)}")

    print("\n" + "=" * 50)
    print("FFTEngine smoke test passed")
    print("=" * 50)
