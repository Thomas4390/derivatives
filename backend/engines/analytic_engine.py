"""
Analytical Pricing Engine
=========================

Black-Scholes analytical pricing for GBM model.

This engine provides closed-form solutions for European vanilla options
under the GBM (Black-Scholes) model. It's the fastest and most accurate
method when applicable.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from backend.utils.math import bs_price as _bs_price_canonical
from backend.utils.math import d1_d2 as _d1_d2_canonical

from backend.core.interfaces import Instrument, Model, PricingEngine
from backend.core.market import MarketEnvironment
from backend.core.result_types import (
    ExerciseStyle,
    GreeksResult,
    PricingCapability,
    PricingResult,
)
from backend.instruments.options import VanillaOption
from backend.models.gbm import GBMModel

# =============================================================================
# BLACK-SCHOLES ANALYTICAL ENGINE
# =============================================================================


@dataclass(frozen=True)
class BSAnalyticEngine(PricingEngine):
    """
    Black-Scholes analytical pricing engine.

    Provides closed-form pricing and Greeks for European vanilla options
    under the GBM model. This is the most efficient method when applicable.

    Supported Instruments
    ---------------------
    - European vanilla calls and puts (VanillaOption with ExerciseStyle.EUROPEAN)

    Supported Models
    ----------------
    - GBMModel only (constant volatility, no jumps/stochastic vol)

    Examples
    --------
    engine = BSAnalyticEngine()

    option = VanillaOption(strike=100, maturity=0.5, is_call=True)
    model = GBMModel(sigma=0.20)
    market = MarketEnvironment(spot=100, rate=0.05)

    result = engine.price(option, model, market)
    greeks = engine.greeks(option, model, market)
    """

    @property
    def capability(self) -> PricingCapability:
        """This is an analytical engine."""
        return PricingCapability.ANALYTICAL

    @property
    def supported_exercises(self) -> list[ExerciseStyle]:
        """Only European exercise supported."""
        return [ExerciseStyle.EUROPEAN]

    def can_price(self, instrument: Instrument, model: Model) -> bool:
        """
        Check if this engine can price the given combination.

        BSAnalyticEngine requires:
        - European exercise style
        - Instrument with fixed strike
        - Model supporting analytical pricing (GBMModel)
        """
        # Check exercise style
        if instrument.exercise_style != ExerciseStyle.EUROPEAN:
            return False

        # Only vanilla options — reject exotics (barrier, Asian, etc.)
        if not isinstance(instrument, VanillaOption):
            return False

        # Check model supports analytical pricing
        if PricingCapability.ANALYTICAL not in model.supported_engines:
            return False

        return True

    def price(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
    ) -> PricingResult:
        """
        Price a European vanilla option analytically.

        Parameters
        ----------
        instrument : VanillaOption
            European vanilla option
        model : GBMModel
            GBM model with constant volatility
        market : MarketEnvironment
            Current market conditions

        Returns
        -------
        PricingResult
            Analytical price
        """
        if not self.can_price(instrument, model):
            raise ValueError(
                f"BSAnalyticEngine cannot price {type(instrument).__name__} "
                f"with {type(model).__name__}"
            )

        option = instrument  # type: VanillaOption
        gbm = model  # type: GBMModel

        # Extract parameters
        s0 = market.spot
        k = option.strike
        t = option.maturity
        r = market.rate
        q = market.dividend_yield
        sigma = gbm.sigma

        # Black-Scholes formula
        price = self._bs_price(
            s0=s0, k=k, t=t, r=r, q=q, sigma=sigma, is_call=option.is_call
        )

        return PricingResult(price=price, engine="BSAnalyticEngine", model=model.name)

    def greeks(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
    ) -> GreeksResult:
        """
        Compute analytical Greeks.

        Parameters
        ----------
        instrument : VanillaOption
            European vanilla option
        model : GBMModel
            GBM model with constant volatility
        market : MarketEnvironment
            Current market conditions

        Returns
        -------
        GreeksResult
            All analytical Greeks (first, second, and third order)
        """
        if not self.can_price(instrument, model):
            raise ValueError(
                f"BSAnalyticEngine cannot compute Greeks for "
                f"{type(instrument).__name__} with {type(model).__name__}"
            )

        option = instrument  # type: VanillaOption
        gbm = model  # type: GBMModel

        # Extract parameters
        s0 = market.spot
        k = option.strike
        t = option.maturity
        r = market.rate
        q = market.dividend_yield
        sigma = gbm.sigma
        is_call = option.is_call

        from backend.greeks.analytic import bs_all_greeks

        g = bs_all_greeks(s=s0, k=k, t=t, r=r, q=q, sigma=sigma, is_call=is_call)
        return GreeksResult(
            price=g[0],
            delta=g[1],
            gamma=g[2],
            vega=g[3],
            theta=g[4],
            rho=g[5],
            vanna=g[6],
            volga=g[7],
            charm=g[8],
            veta=g[9],
            speed=g[10],
            zomma=g[11],
            color=g[12],
            ultima=g[13],
        )

    def implied_volatility(
        self,
        price: float,
        instrument: Instrument,
        market: MarketEnvironment,
        initial_guess: float = 0.20,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> float:
        """
        Compute implied volatility from market price.

        Uses Newton-Raphson iteration with vega.

        Parameters
        ----------
        price : float
            Market price of the option
        instrument : VanillaOption
            European vanilla option
        market : MarketEnvironment
            Current market conditions
        initial_guess : float
            Initial volatility guess
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum iterations

        Returns
        -------
        float
            Implied volatility

        Raises
        ------
        ValueError
            If convergence fails
        """
        option = instrument  # type: VanillaOption

        s0 = market.spot
        k = option.strike
        t = option.maturity
        r = market.rate
        q = market.dividend_yield
        is_call = option.is_call

        sigma = initial_guess
        diff = float("nan")

        for i in range(max_iter):
            bs_price = self._bs_price(
                s0=s0, k=k, t=t, r=r, q=q, sigma=sigma, is_call=is_call
            )
            vega = self._bs_vega(s0=s0, k=k, t=t, r=r, q=q, sigma=sigma)

            if vega < 1e-12:
                break

            diff = bs_price - price
            if abs(diff) < tol:
                return sigma

            sigma = sigma - diff / vega
            sigma = max(sigma, 0.001)  # Keep positive
            sigma = min(sigma, 10.0)  # Cap at 1000% vol

        raise ValueError(
            f"IV did not converge after {max_iter} iterations. "
            f"Last sigma={sigma:.4f}, diff={diff:.6f}"
        )

    # =========================================================================
    # PRIVATE BLACK-SCHOLES FORMULAS
    # =========================================================================

    def _d1_d2(
        self, s0: float, k: float, t: float, r: float, q: float, sigma: float
    ) -> tuple[float, float]:
        """Compute d1 and d2 parameters.  Delegates to canonical ``d1_d2``."""
        return _d1_d2_canonical(s0, k, t, r, sigma, q)

    def _bs_price(
        self,
        s0: float,
        k: float,
        t: float,
        r: float,
        q: float,
        sigma: float,
        is_call: bool,
    ) -> float:
        """Black-Scholes price formula.  Delegates to canonical ``bs_price``."""
        return _bs_price_canonical(s0, k, t, r, sigma, is_call, q)

    def _bs_vega(
        self, s0: float, k: float, t: float, r: float, q: float, sigma: float
    ) -> float:
        """Black-Scholes vega."""
        if t <= 0:
            return 0.0

        d1, _ = _d1_d2_canonical(s0, k, t, r, sigma, q)
        forward_discount = np.exp(-q * t)
        return s0 * forward_discount * norm.pdf(d1) * np.sqrt(t)


if __name__ == "__main__":
    from backend.core.market import MarketEnvironment
    from backend.instruments.options import VanillaOption
    from backend.models.gbm import GBMModel

    print("=" * 50)
    print("BSAnalyticEngine Smoke Test")
    print("=" * 50)

    # Create components
    engine = BSAnalyticEngine()
    option = VanillaOption(strike=100, maturity=0.5, is_call=True)
    model = GBMModel(sigma=0.20)
    market = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.02)

    print("\n--- Setup ---")
    print(f"Option: K={option.strike}, T={option.maturity}, Call={option.is_call}")
    print(f"Model: {model}")
    print(f"Market: S0={market.spot}, r={market.rate}, q={market.dividend_yield}")

    # Test pricing
    print("\n--- Pricing ---")
    result = engine.price(option, model, market)
    print(f"Price: ${result.price:.4f}")
    print(f"Engine: {result.engine}")
    print(f"Model: {result.model}")

    # Test Greeks
    print("\n--- Greeks ---")
    greeks = engine.greeks(option, model, market)
    print(f"Delta: {greeks.delta:.4f}")
    print(f"Gamma: {greeks.gamma:.6f}")
    print(f"Theta: {greeks.theta:.4f}")
    print(f"Vega: {greeks.vega:.4f}")
    print(f"Rho: {greeks.rho:.4f}")
    print(f"Vanna: {greeks.vanna:.6f}")
    print(f"Volga: {greeks.volga:.6f}")

    # Test put option
    print("\n--- Put Option ---")
    put_option = VanillaOption(strike=100, maturity=0.5, is_call=False)
    put_result = engine.price(put_option, model, market)
    put_greeks = engine.greeks(put_option, model, market)
    print(f"Put Price: ${put_result.price:.4f}")
    print(f"Put Delta: {put_greeks.delta:.4f}")

    # Verify put-call parity
    print("\n--- Put-Call Parity Check ---")
    call_price = result.price
    put_price = put_result.price
    parity_rhs = market.spot * np.exp(
        -market.dividend_yield * option.maturity
    ) - option.strike * np.exp(-market.rate * option.maturity)
    parity_lhs = call_price - put_price
    print(f"C - P = {parity_lhs:.4f}")
    print(f"S*e^(-qT) - K*e^(-rT) = {parity_rhs:.4f}")
    print(f"Parity holds: {abs(parity_lhs - parity_rhs) < 0.001}")

    # Test implied volatility
    print("\n--- Implied Volatility ---")
    target_price = result.price
    iv = engine.implied_volatility(target_price, option, market)
    print(f"Original sigma: {model.sigma:.4f}")
    print(f"Implied vol: {iv:.4f}")
    print(f"IV recovery: {abs(iv - model.sigma) < 1e-6}")

    # Test can_price
    print("\n--- Compatibility Check ---")
    print(f"Can price European call with GBM: {engine.can_price(option, model)}")

    from backend.models.heston import HestonModel

    heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, alpha=0.3, rho=-0.7)
    print(f"Can price European call with Heston: {engine.can_price(option, heston)}")

    print("\n" + "=" * 50)
    print("BSAnalyticEngine smoke test passed")
    print("=" * 50)
