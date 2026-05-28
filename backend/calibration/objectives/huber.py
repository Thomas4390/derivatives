"""
Huber Objective — Robust to Outliers
=====================================

Huber's M-estimator (Huber 1964) interpolates between quadratic loss
(small residuals) and linear loss (large residuals), with crossover
at a user-chosen threshold ``δ``. Recommended for noisy markets:
patchy SPX, crypto options with 10-50× wider spreads, or end-of-day
snapshots where stale quotes appear as outliers.

For use with Levenberg-Marquardt, the residual is reshaped as
``ψ_i = sqrt(w_i) · r_i`` with ``w_i = min(1, δ/|r_i|)`` so that the
sum-of-squares objective ``½ Σ ψ_i²`` equals the standard Huber loss
formula. This is the IRLS (Iteratively Reweighted Least Squares)
re-parametrisation used implicitly inside the LM trust-region update.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from backend.calibration.objectives.base import (
    JaxResidualFn,
    ObjectiveMetadata,
    ObjectiveStrategy,
)


class HuberObjective(ObjectiveStrategy):
    """Robust Huber loss with crossover threshold ``δ``.

    Parameters
    ----------
    delta : float
        Crossover threshold in price units. Residuals below ``δ`` are
        squared, residuals above are linearly penalised. Typical choice:
        5 % of the median quote price.
    """

    def __init__(self, delta: float = 0.05) -> None:
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        self.delta = float(delta)
        self.metadata = ObjectiveMetadata(
            name="huber",
            display_name="Huber",
            description=(
                f"Robust loss: quadratic for |r|<δ ({self.delta:g}), "
                "linear beyond. Reduces the influence of outliers and "
                "stale quotes."
            ),
            formula=(
                r"\rho_\delta(r) = \tfrac{1}{2}r^2 \text{ si }|r|<\delta;\;"
                r"\delta(|r|-\tfrac{\delta}{2}) \text{ sinon.}"
            ),
            jax_compatible=True,
            requires_iv=False,
            requires_spreads=False,
            references=(
                "Huber (1964)",
                "Schoutens, Simons & Tistaert (2003)",
                "arxiv 2207.02989 (2022)",
            ),
        )

    def precompute_weights(self, market_data: Any) -> np.ndarray | None:
        return None  # dynamic weights — computed inside the residual

    # ------------------------------------------------------------------ #
    # Pure-NumPy path
    # ------------------------------------------------------------------ #

    def _irls_residuals_np(self, diff: np.ndarray) -> np.ndarray:
        abs_diff = np.abs(diff)
        weights = np.minimum(1.0, self.delta / np.maximum(abs_diff, 1e-12))
        return np.sqrt(weights) * diff

    def compute_residuals(
        self,
        model_prices: np.ndarray,
        market_data: Any,
    ) -> np.ndarray:
        diff = np.asarray(model_prices, dtype=float) - market_data.market_prices
        return self._irls_residuals_np(diff)

    def compute_loss(self, model_prices: np.ndarray, market_data: Any) -> float:
        """Returns the true Huber loss aggregate, not the RMSE of the IRLS residuals."""
        diff = np.asarray(model_prices, dtype=float) - market_data.market_prices
        abs_d = np.abs(diff)
        quadratic = 0.5 * (diff * diff)
        linear = self.delta * (abs_d - 0.5 * self.delta)
        huber = np.where(abs_d <= self.delta, quadratic, linear)
        return float(np.mean(huber))

    # ------------------------------------------------------------------ #
    # JAX path (LM-JAX uses these residuals; dynamic weights via jnp.where)
    # ------------------------------------------------------------------ #

    def make_jax_residual_fn(self, market_data: Any) -> JaxResidualFn:
        market_p_jnp = jnp.asarray(market_data.market_prices, dtype=jnp.float64)
        delta = self.delta

        def _huber_irls(model_p_jnp: jnp.ndarray) -> jnp.ndarray:
            diff = model_p_jnp - market_p_jnp
            abs_diff = jnp.abs(diff)
            weights = jnp.minimum(1.0, delta / jnp.maximum(abs_diff, 1e-12))
            return jnp.sqrt(weights) * diff

        return _huber_irls


__all__ = ["HuberObjective"]
