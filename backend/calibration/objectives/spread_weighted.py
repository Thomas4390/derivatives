"""
Spread-Weighted Objective
==========================

Bid-ask aware loss — rescales each residual by the inverse of the
observed bid-ask spread. Down-weights illiquid quotes that contribute
noise rather than signal. Recommended for SPX surfaces with patchy
liquidity and for cryptocurrency markets where spreads are 10-50× wider
than equity (Guillaume & Schoutens 2014).

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.calibration.objectives.base import ObjectiveMetadata, ObjectiveStrategy


class SpreadWeightedObjective(ObjectiveStrategy):
    """Residuals ``r_i = (P^{mod}_i - P^{mkt}_i) / spread^{BA}_i``."""

    metadata = ObjectiveMetadata(
        name="spread_weighted",
        display_name="Spread-weighted",
        description=(
            "Rescaling by 1/(bid-ask spread). Down-weights illiquid options "
            "whose mid is noisy. Falls back to a configurable median spread "
            "when bid/ask are missing."
        ),
        formula=r"r_i = (P^{mod}_i - P^{mkt}_i) / \mathrm{spread}^{BA}_i",
        jax_compatible=True,
        requires_iv=False,
        requires_spreads=True,
        references=(
            "Guillaume & Schoutens (2014)",
            "Hagan & West (2008)",
        ),
    )

    def __init__(
        self,
        fallback_spread_pct: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        # Default fallback : 5% of the quote price when bid/ask are missing.
        self.fallback_spread_pct = float(fallback_spread_pct)
        self.eps = float(eps)

    def precompute_weights(self, market_data: Any) -> np.ndarray:
        market_prices = market_data.market_prices
        spreads = np.empty(len(market_data.quotes), dtype=float)
        for i, q in enumerate(market_data.quotes):
            s = q.spread  # property: ask - bid, None when missing
            if s is None or not np.isfinite(s) or s <= 0:
                spreads[i] = self.fallback_spread_pct * max(market_prices[i], self.eps)
            else:
                spreads[i] = float(s)
        # Weights w_i = 1/spread_i^2 so that sqrt(w_i)*(diff) = diff/spread_i
        weights = 1.0 / (spreads + self.eps) ** 2
        # Normalize to sum to n quotes (consistent with vega_weights convention)
        weights *= len(weights) / float(np.sum(weights))
        return weights

    def compute_residuals(
        self,
        model_prices: np.ndarray,
        market_data: Any,
    ) -> np.ndarray:
        weights = self.precompute_weights(market_data)
        diff = np.asarray(model_prices, dtype=float) - market_data.market_prices
        return np.sqrt(weights) * diff


__all__ = ["SpreadWeightedObjective"]
