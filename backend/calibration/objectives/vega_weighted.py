"""
Vega-Weighted Objective
========================

Industry-standard rescaling of price residuals by Black-Scholes vega.
Under first-order Taylor expansion ``ΔP ≈ V·Δσ``, dividing the residual
by vega recovers an IV-error scale without paying the cost of an IV
inversion at every evaluation (Cont & Tankov 2004, §13.1).

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.calibration.objectives.base import ObjectiveMetadata, ObjectiveStrategy
from backend.calibration.utils import vega_weights


class VegaWeightedObjective(ObjectiveStrategy):
    """Residuals ``r_i = sqrt(w_i) (P^{mod}_i - P^{mkt}_i)`` with ``w_i ∝ 1/V_i^2``."""

    metadata = ObjectiveMetadata(
        name="vega_weighted",
        display_name="Vega-weighted",
        description=(
            "Rescaling by 1/vega — fast linear approximation of IV-MSE "
            "(Cont & Tankov 2004). Industry default for surface calibration."
        ),
        formula=r"r_i = (P^{mod}_i - P^{mkt}_i) / \mathcal{V}_i^{BS}",
        jax_compatible=True,
        requires_iv=False,
        requires_spreads=False,
        references=(
            "Cont & Tankov (2004) §13.1",
            "Schoutens, Simons & Tistaert (2003)",
        ),
    )

    def __init__(self, fallback_iv: float = 0.20, min_vega: float = 1e-6) -> None:
        self.fallback_iv = float(fallback_iv)
        self.min_vega = float(min_vega)

    def precompute_weights(self, market_data: Any) -> np.ndarray:
        market_ivs = np.array(
            [
                q.implied_vol if q.implied_vol is not None else self.fallback_iv
                for q in market_data.quotes
            ],
            dtype=float,
        )
        return vega_weights(
            spot=market_data.spot,
            strikes=market_data.strikes,
            maturities=market_data.maturities,
            rate=market_data.rate,
            ivs=market_ivs,
            dividend_yield=market_data.dividend_yield,
        )

    def compute_residuals(
        self,
        model_prices: np.ndarray,
        market_data: Any,
    ) -> np.ndarray:
        weights = self.precompute_weights(market_data)
        diff = np.asarray(model_prices, dtype=float) - market_data.market_prices
        return np.sqrt(weights) * diff


__all__ = ["VegaWeightedObjective"]
