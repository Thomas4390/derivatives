"""
Objective Strategy — Base Protocol & Metadata
==============================================

Strategy Pattern for the *loss function* of a calibration. Each strategy
encapsulates a definition of "how close is the model to the market"
that drives both the LM-style residual vector and the scalar reporting
metric.

The pattern mirrors :mod:`backend.calibration.optimizers` — a small
package of independently composable strategies with a factory.

Two execution paths
-------------------

* **NumPy path** — :meth:`compute_residuals` takes
  ``(model_prices, market_data)`` and returns a 1-D ``np.ndarray`` of
  signed residuals. Always available, never traced.
* **JAX path** — :meth:`make_jax_residual_fn` returns a closure
  ``(model_prices_jnp) -> jnp.ndarray`` ready to be wrapped in
  :func:`jax.jit` and :func:`jax.jacfwd`. Available only when
  ``metadata.jax_compatible`` is ``True``; otherwise the calibrator is
  expected to fall back to a JAX-friendly approximation (typically
  ``vega_weighted``).

References
----------
* Cont & Tankov (2004), *Financial Modelling with Jump Processes*, §13.
* Schoutens, Simons & Tistaert (2003), *A Perfect Calibration ! Now What?*
* Guillaume & Schoutens (2014), *Heston Model: The Variance Swap
  Calibration*.
* Pacati, Pompa & Renò (2018), *Smiling Twice: The Heston++ Model*.
* Huber (1964), *Robust Estimation of a Location Parameter*.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np

JaxResidualFn = Callable[[jnp.ndarray], jnp.ndarray]


@dataclass(frozen=True)
class ObjectiveMetadata:
    """Static description of an objective function for the UI and docs."""

    name: str
    display_name: str
    description: str
    formula: str  # LaTeX-friendly snippet for the pedagogical panel
    jax_compatible: bool
    requires_iv: bool
    requires_spreads: bool
    references: tuple[str, ...] = field(default_factory=tuple)


class ObjectiveStrategy(ABC):
    """Abstract objective: residual transform + scalar loss.

    Subclasses define how the per-quote price error is reshaped into the
    residual vector. The scalar loss reported by :meth:`compute_loss`
    defaults to the RMSE of the residuals (square-root of mean squared
    residual) but may be overridden by subclasses with non-square losses
    (e.g. Huber, where the scalar loss is the Huber sum, not the RMSE of
    the weighted residuals).
    """

    metadata: ObjectiveMetadata

    # ------------------------------------------------------------------ #
    # Public properties (sugar)
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def display_name(self) -> str:
        return self.metadata.display_name

    @property
    def jax_compatible(self) -> bool:
        return self.metadata.jax_compatible

    @property
    def requires_iv(self) -> bool:
        return self.metadata.requires_iv

    @property
    def requires_spreads(self) -> bool:
        return self.metadata.requires_spreads

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #

    def precompute_weights(self, market_data: Any) -> np.ndarray | None:
        """One-shot precomputation of static weights.

        Linear-weight objectives (vega, spread, relative-via-mkt-prices)
        override this to return the per-quote weight array. Stateless
        objectives (price_mse) return ``None`` to short-circuit the
        weighting branch. Dynamic-weight objectives (Huber) also return
        ``None`` and override :meth:`make_jax_residual_fn` directly.
        """
        return None

    @abstractmethod
    def compute_residuals(
        self,
        model_prices: np.ndarray,
        market_data: Any,
    ) -> np.ndarray:
        """Pure-NumPy residuals (always available, no JAX tracing)."""
        ...

    def compute_loss(self, model_prices: np.ndarray, market_data: Any) -> float:
        """Scalar reporting metric: RMSE of the residual vector by default."""
        r = self.compute_residuals(model_prices, market_data)
        return float(np.sqrt(np.mean(r * r)))

    # ------------------------------------------------------------------ #
    # JAX path (default = sqrt(weights) * (model - market))
    # ------------------------------------------------------------------ #

    def make_jax_residual_fn(self, market_data: Any) -> JaxResidualFn:
        """Build a JAX-traceable residual closure.

        Default implementation handles the "linear weights" family
        (price_mse, vega_weighted, spread_weighted): precomputes the
        weight vector once, packs the market prices, and returns
        ``sqrt(weights) * (model_prices - market_prices)``.

        Subclasses with custom JAX logic (Huber, relative/log) override
        this method.
        """
        if not self.jax_compatible:
            raise NotImplementedError(
                f"Objective '{self.name}' is not JAX-compatible. "
                "Calibrators using JAX-only solvers (LM-JAX) should fall "
                "back to a JAX-friendly approximation such as vega_weighted."
            )

        weights_np = self.precompute_weights(market_data)
        market_p_jnp = jnp.asarray(market_data.market_prices, dtype=jnp.float64)

        if weights_np is None:

            def _no_weights(model_p_jnp: jnp.ndarray) -> jnp.ndarray:
                return model_p_jnp - market_p_jnp

            return _no_weights

        sqrt_w_jnp = jnp.sqrt(jnp.asarray(weights_np, dtype=jnp.float64))

        def _weighted(model_p_jnp: jnp.ndarray) -> jnp.ndarray:
            return sqrt_w_jnp * (model_p_jnp - market_p_jnp)

        return _weighted


__all__ = [
    "JaxResidualFn",
    "ObjectiveMetadata",
    "ObjectiveStrategy",
]
