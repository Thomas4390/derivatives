"""
Implied Volatility Calibrator (GBM)
====================================

Calibrates GBM sigma from observed option prices via IV inversion.

- Single quote: direct Newton-Raphson inversion
- Multiple quotes: vega-weighted average of individual IVs

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import logging
import time

import numpy as np

from backend.calibration.base import BaseCalibrator, CalibrationResult
from backend.calibration.market_data import OptionMarketData
from backend.models.gbm import GBMModel
from backend.utils.math import implied_volatility

logger = logging.getLogger(__name__)


class ImpliedVolCalibrator(BaseCalibrator):
    """Calibrate GBM sigma from option market data via IV inversion.

    For a single quote, returns the exact implied volatility.
    For multiple quotes, returns the vega-weighted average IV.

    Parameters
    ----------
    weighting : str
        Weighting scheme for multiple quotes:
        - "equal": simple average
        - "vega": vega-weighted average (default, more robust)
    """

    def __init__(self, weighting: str = "vega") -> None:
        self.weighting = weighting

    def calibrate(self, market_data: OptionMarketData) -> CalibrationResult:
        t0 = time.time()

        iv_mat_pairs = []
        for q in market_data.quotes:
            try:
                iv = implied_volatility(
                    price=q.market_price,
                    spot=market_data.spot,
                    strike=q.strike,
                    time_to_expiry=q.maturity,
                    rate=market_data.rate,
                    is_call=q.is_call,
                    dividend_yield=market_data.dividend_yield,
                )
                if iv > 0 and np.isfinite(iv):
                    iv_mat_pairs.append((iv, q.maturity))
            except (ValueError, RuntimeError) as e:
                logger.debug(
                    "IV inversion failed for K=%.1f T=%.3f: %s", q.strike, q.maturity, e
                )

        if not iv_mat_pairs:
            return CalibrationResult(
                model=GBMModel(sigma=0.20),
                objective_value=float("inf"),
                n_iterations=0,
                success=False,
                method="iv_inversion",
                elapsed_seconds=time.time() - t0,
                diagnostics={"error": "all IV inversions failed"},
            )

        iv_arr = np.array([p[0] for p in iv_mat_pairs])

        if self.weighting == "vega" and len(iv_arr) > 1:
            # Weight by sqrt(T) as vega proxy (higher T → more weight)
            mats = np.array([p[1] for p in iv_mat_pairs])
            if len(mats) == len(iv_arr):
                weights = np.sqrt(mats)
                weights /= np.sum(weights)
                sigma = float(np.dot(weights, iv_arr))
            else:
                sigma = float(np.mean(iv_arr))
        else:
            sigma = float(np.mean(iv_arr))

        sigma = max(sigma, 0.001)
        model = GBMModel(sigma=sigma)

        # Compute RMSE
        from backend.engines.analytic_engine import BSAnalyticEngine
        from backend.core.market import MarketEnvironment
        from backend.instruments.options import VanillaOption

        engine = BSAnalyticEngine()
        market = MarketEnvironment(
            spot=market_data.spot,
            rate=market_data.rate,
            dividend_yield=market_data.dividend_yield,
        )
        model_prices = []
        market_prices = []
        for q in market_data.quotes:
            try:
                option = VanillaOption(
                    strike=q.strike, maturity=q.maturity, is_call=q.is_call
                )
                mp = engine.price(option, model, market).price
                model_prices.append(mp)
                market_prices.append(q.market_price)
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                logger.warning(
                    "Price computation failed for T=%.4f: %s", q.maturity, exc
                )

        rmse = 0.0
        if model_prices:
            rmse = float(
                np.sqrt(
                    np.mean((np.array(model_prices) - np.array(market_prices)) ** 2)
                )
            )

        elapsed = time.time() - t0
        return CalibrationResult(
            model=model,
            objective_value=rmse,
            n_iterations=len(iv_mat_pairs),
            success=True,
            method="iv_inversion",
            rmse_price=rmse,
            rmse_iv=float(np.sqrt(np.mean((iv_arr - sigma) ** 2)) * 10_000)
            if len(iv_arr) > 1
            else 0.0,
            elapsed_seconds=elapsed,
            diagnostics={
                "n_successful_inversions": len(iv_mat_pairs),
                "n_total_quotes": market_data.n_quotes,
                "iv_mean": float(np.mean(iv_arr)),
                "iv_std": float(np.std(iv_arr)) if len(iv_arr) > 1 else 0.0,
                "iv_min": float(np.min(iv_arr)),
                "iv_max": float(np.max(iv_arr)),
            },
        )

    def objective(self, params: np.ndarray, market_data: OptionMarketData) -> float:
        sigma = params[0]
        if sigma <= 0:
            return 1e10

        from backend.engines.analytic_engine import BSAnalyticEngine
        from backend.core.market import MarketEnvironment
        from backend.instruments.options import VanillaOption

        model = GBMModel(sigma=sigma)
        engine = BSAnalyticEngine()
        market = MarketEnvironment(
            spot=market_data.spot,
            rate=market_data.rate,
            dividend_yield=market_data.dividend_yield,
        )

        total_error = 0.0
        for q in market_data.quotes:
            option = VanillaOption(
                strike=q.strike, maturity=q.maturity, is_call=q.is_call
            )
            model_price = engine.price(option, model, market).price
            total_error += (model_price - q.market_price) ** 2

        return total_error

    def default_bounds(self) -> list[tuple[float, float]]:
        return [(0.01, 2.0)]


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("ImpliedVolCalibrator Smoke Test")
    print("=" * 50)

    from backend.calibration.market_data import OptionQuote, OptionMarketData
    from backend.engines.analytic_engine import BSAnalyticEngine
    from backend.core.market import MarketEnvironment
    from backend.instruments.options import VanillaOption

    # Generate synthetic BS prices with known sigma
    true_sigma = 0.25
    true_model = GBMModel(sigma=true_sigma)
    engine = BSAnalyticEngine()
    market = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.02)

    strikes = [90, 95, 100, 105, 110]
    maturities = [0.25, 0.50]
    quotes = []
    for T in maturities:
        for K in strikes:
            option = VanillaOption(strike=K, maturity=T, is_call=True)
            price = engine.price(option, true_model, market).price
            quotes.append(
                OptionQuote(strike=K, maturity=T, is_call=True, market_price=price)
            )

    data = OptionMarketData(
        spot=100, rate=0.05, dividend_yield=0.02, quotes=tuple(quotes)
    )

    # Calibrate
    calibrator = ImpliedVolCalibrator(weighting="vega")
    result = calibrator.calibrate(data)

    print(f"\nTrue sigma: {true_sigma:.4f}")
    print(f"Calibrated sigma: {result.model.sigma:.4f}")
    print(f"RMSE price: {result.rmse_price:.6f}")
    print(f"Success: {result.success}")
    print(f"Diagnostics: {result.diagnostics}")

    assert abs(result.model.sigma - true_sigma) < 0.001, (
        f"Should recover true sigma, got {result.model.sigma}"
    )

    print("\n✓ Smoke test passed")
