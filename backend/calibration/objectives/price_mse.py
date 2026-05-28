"""
Price MSE Objective
===================

Standard sum-of-squared price residuals — the unweighted baseline used
in the original Heston (1993) calibration. Robust, JAX-friendly, and
trivial to differentiate, but dominated by ATM options where the
nominal price is largest.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.calibration.objectives.base import ObjectiveMetadata, ObjectiveStrategy


class PriceMSEObjective(ObjectiveStrategy):
    """Residuals ``r_i = P^{mod}_i - P^{mkt}_i``."""

    metadata = ObjectiveMetadata(
        name="price_mse",
        display_name="Price MSE",
        description=(
            "Standard sum-of-squared price errors. Robust and JAX-friendly "
            "but ATM-dominated because nominal prices grow with maturity."
        ),
        formula=r"r_i = P^{mod}_i - P^{mkt}_i",
        jax_compatible=True,
        requires_iv=False,
        requires_spreads=False,
        references=("Heston (1993)", "Bates (1996)"),
    )

    def precompute_weights(self, market_data: Any) -> np.ndarray | None:
        return None

    def compute_residuals(
        self,
        model_prices: np.ndarray,
        market_data: Any,
    ) -> np.ndarray:
        return np.asarray(model_prices, dtype=float) - market_data.market_prices


__all__ = ["PriceMSEObjective"]
