"""
Relative / Log Price Objective
===============================

Scale-invariant loss using either the relative error
``(model − market) / market`` or the log price ratio
``log(model) − log(market)``. Useful when the price level varies
strongly across maturities (typical for SPX surfaces with weeks-to-
years range) so the residual contributions are commensurable.

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


class RelativeMSEObjective(ObjectiveStrategy):
    """Relative or log-price residuals.

    Parameters
    ----------
    use_log : bool
        If ``True``, residual is ``log(P_mod / P_mkt)`` (log-price MSE).
        Otherwise ``(P_mod - P_mkt) / P_mkt`` (relative price MSE).
    eps : float
        Floor added under prices to avoid division-by-zero on deep-OTM
        quotes that may price to a tiny positive number.
    """

    def __init__(self, use_log: bool = False, eps: float = 1e-6) -> None:
        self.use_log = bool(use_log)
        self.eps = float(eps)
        mode = "log" if self.use_log else "relative"
        self.metadata = ObjectiveMetadata(
            name="relative",
            display_name=f"{'Log-price' if self.use_log else 'Relative'} MSE",
            description=(
                "Log-price ratio residuals — scale-invariant."
                if self.use_log
                else "Relative price errors — scale-invariant across maturities."
            ),
            formula=(
                r"r_i = \log(P^{mod}_i) - \log(P^{mkt}_i)"
                if self.use_log
                else r"r_i = (P^{mod}_i - P^{mkt}_i) / P^{mkt}_i"
            ),
            jax_compatible=True,
            requires_iv=False,
            requires_spreads=False,
            references=(
                "Bakshi, Cao & Chen (1997)",
                "Carr & Madan (1999)",
            ),
        )
        self._mode = mode

    def precompute_weights(self, market_data: Any) -> np.ndarray | None:
        # Not used — relative scaling is computed directly from market prices
        # at residual evaluation. We still encode the eps + use_log in the
        # closure built by make_jax_residual_fn.
        return None

    def compute_residuals(
        self,
        model_prices: np.ndarray,
        market_data: Any,
    ) -> np.ndarray:
        model_p = np.asarray(model_prices, dtype=float)
        market_p = np.asarray(market_data.market_prices, dtype=float)
        if self.use_log:
            return np.log(np.maximum(model_p, self.eps)) - np.log(
                np.maximum(market_p, self.eps)
            )
        return (model_p - market_p) / np.maximum(market_p, self.eps)

    def make_jax_residual_fn(self, market_data: Any) -> JaxResidualFn:
        market_p_jnp = jnp.asarray(market_data.market_prices, dtype=jnp.float64)
        eps = self.eps
        if self.use_log:
            log_mkt = jnp.log(jnp.maximum(market_p_jnp, eps))

            def _log(model_p_jnp: jnp.ndarray) -> jnp.ndarray:
                return jnp.log(jnp.maximum(model_p_jnp, eps)) - log_mkt

            return _log

        inv_mkt = 1.0 / jnp.maximum(market_p_jnp, eps)

        def _rel(model_p_jnp: jnp.ndarray) -> jnp.ndarray:
            return (model_p_jnp - market_p_jnp) * inv_mkt

        return _rel


__all__ = ["RelativeMSEObjective"]
