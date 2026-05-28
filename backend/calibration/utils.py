"""
Calibration Utilities
=====================

Shared helpers for all calibrators.

Author: Thomas Vaudescal
Created: 2026
"""

import numpy as np

from backend.utils.math import implied_volatility


def compute_rmse_price(model_prices: np.ndarray, market_prices: np.ndarray) -> float:
    """Root mean squared error in price space."""
    return float(np.sqrt(np.mean((model_prices - market_prices) ** 2)))


def compute_rmse_iv(model_ivs: np.ndarray, market_ivs: np.ndarray) -> float:
    """Root mean squared error in IV space (basis points)."""
    return float(np.sqrt(np.mean((model_ivs - market_ivs) ** 2)) * 10_000)


def vega_weights(
    spot: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    rate: float,
    ivs: np.ndarray,
    dividend_yield: float = 0.0,
) -> np.ndarray:
    """Compute vega-based weights for objective function normalization.

    Weighting by 1/vega prevents deep OTM/ITM options from dominating.
    Fully vectorized with NumPy broadcasting.
    """
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    ivs = np.asarray(ivs, dtype=float)

    sigmas = np.where(ivs > 0, ivs, 0.2)
    sqrt_t = np.where(maturities > 0, np.sqrt(np.maximum(maturities, 1e-10)), 0.01)
    d1 = (
        np.log(spot / strikes) + (rate - dividend_yield + 0.5 * sigmas**2) * maturities
    ) / (sigmas * sqrt_t)
    pdf_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    vegas = spot * np.exp(-dividend_yield * maturities) * pdf_d1 * sqrt_t
    weights = 1.0 / np.maximum(vegas, 1e-6)
    # Normalize so weights sum to number of observations
    weights *= len(weights) / np.sum(weights)
    return weights


def model_prices_to_ivs(
    model_prices: np.ndarray,
    spot: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    rate: float,
    is_calls: np.ndarray,
    dividend_yield: float = 0.0,
) -> np.ndarray:
    """Convert model prices to implied volatilities."""
    ivs = np.full(len(model_prices), np.nan)
    for i in range(len(model_prices)):
        try:
            ivs[i] = implied_volatility(
                price=model_prices[i],
                spot=spot,
                strike=strikes[i],
                time_to_expiry=maturities[i],
                rate=rate,
                is_call=bool(is_calls[i]),
                dividend_yield=dividend_yield,
            )
        except (ValueError, RuntimeError):
            ivs[i] = np.nan
    return ivs


def get_atm_iv(market_data) -> float:
    """Extract ATM implied volatility from market data.

    Returns the IV of the quote closest to ATM (strike ≈ spot).
    Falls back to 0.20 if no IV available.
    """
    spot = market_data.spot
    best_quote = min(market_data.quotes, key=lambda q: abs(q.strike - spot))
    if best_quote.implied_vol is not None:
        return best_quote.implied_vol
    # Compute IV from price
    try:
        return implied_volatility(
            price=best_quote.market_price,
            spot=spot,
            strike=best_quote.strike,
            time_to_expiry=best_quote.maturity,
            rate=market_data.rate,
            is_call=best_quote.is_call,
            dividend_yield=market_data.dividend_yield,
        )
    except (ValueError, RuntimeError):
        return 0.20
