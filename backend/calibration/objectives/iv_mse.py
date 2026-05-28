"""
IV-MSE Objective — Implied-Volatility Residuals
================================================

Sum-of-squared *implied volatility* errors. Considered the most
pedagogically uniform objective across moneyness because IV magnitudes
do not scale with the option's nominal price (Pacati, Pompa & Renò
2018). Each evaluation inverts the BS price back to a volatility per
quote — historically the cost that pushed surface calibrators to the
``vega_weighted`` first-order approximation.

This objective is now **fully JAX-traceable** via
:func:`backend.utils.math_jax.implied_volatility_jax`: the forward
inversion delegates to the existing Numba rtsafe code through
:func:`jax.pure_callback`, and a custom JVP propagates the
implicit-function gradient ``∂σ/∂P = 1/Vega(σ*)`` evaluated at the
converged root. ``LM-JAX`` can therefore minimise ``iv_mse`` directly,
without coercing it down to ``vega_weighted``.

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
from backend.calibration.utils import model_prices_to_ivs
from backend.utils.math_jax import implied_volatility_jax


class IVMSEObjective(ObjectiveStrategy):
    """Residuals ``r_i = σ^{mod}_i - σ^{mkt}_i`` (basis-points reporting).

    Parameters
    ----------
    invalid_penalty : float
        Residual value substituted when the IV inversion fails on a
        given quote (price out of arbitrage range). Defaults to ``1.0``
        — large enough to penalise the solver but finite to avoid
        derailing the LM update.
    """

    metadata = ObjectiveMetadata(
        name="iv_mse",
        display_name="IV MSE",
        description=(
            "Sum-of-squared implied-volatility errors. Moneyness-uniform. "
            "JAX-traceable via implicit function theorem: forward IV "
            "inversion runs the existing Numba rtsafe through "
            "jax.pure_callback, backward propagates dσ/dP = 1/Vega(σ*)."
        ),
        formula=r"r_i = \sigma^{mod}_i - \sigma^{mkt}_i",
        jax_compatible=True,
        requires_iv=True,
        requires_spreads=False,
        references=(
            "Pacati, Pompa & Renò (2018)",
            "Andersen, Brotherton-Ratcliffe (1998)",
            "Cont & Tankov (2004) §13.1 — vega-weighting as the first-order "
            "approximation this objective makes exact.",
        ),
    )

    def __init__(self, invalid_penalty: float = 1.0) -> None:
        self.invalid_penalty = float(invalid_penalty)

    def precompute_weights(self, market_data: Any) -> np.ndarray:
        # Stores market IVs (per quote, NaN if missing).
        return np.array(
            [
                q.implied_vol if q.implied_vol is not None else np.nan
                for q in market_data.quotes
            ],
            dtype=float,
        )

    def compute_residuals(
        self,
        model_prices: np.ndarray,
        market_data: Any,
    ) -> np.ndarray:
        is_calls = np.array([q.is_call for q in market_data.quotes])
        model_ivs = model_prices_to_ivs(
            model_prices=np.asarray(model_prices, dtype=float),
            spot=market_data.spot,
            strikes=market_data.strikes,
            maturities=market_data.maturities,
            rate=market_data.rate,
            is_calls=is_calls,
            dividend_yield=market_data.dividend_yield,
        )
        market_ivs = self.precompute_weights(market_data)
        valid = ~np.isnan(model_ivs) & ~np.isnan(market_ivs)
        residuals = np.full_like(model_ivs, self.invalid_penalty, dtype=float)
        residuals[valid] = model_ivs[valid] - market_ivs[valid]
        return residuals

    # ------------------------------------------------------------------ #
    # JAX path — implicit-function-theorem custom JVP
    # ------------------------------------------------------------------ #

    def make_jax_residual_fn(self, market_data: Any) -> JaxResidualFn:
        """JAX-traceable residual ``r_i = σ^{mod}_i − σ^{mkt}_i``.

        The model prices are inverted in-graph through
        :func:`backend.utils.math_jax.implied_volatility_jax`, whose
        custom JVP gives ``LM-JAX`` an *exact* per-quote Jacobian
        ``∂σ_i/∂P_i = 1/Vega(σ*_i)``. Quotes for which either the
        market IV is missing or the model price violates no-arbitrage
        bounds are masked to the static ``invalid_penalty`` value, with
        the standard ``jnp.where(stop_gradient_mask, …)`` trick to keep
        NaN gradients from poisoning the rest of the residual.
        """
        spot = float(market_data.spot)
        rate = float(market_data.rate)
        dividend_yield = float(market_data.dividend_yield)
        strikes_jnp = jnp.asarray(market_data.strikes, dtype=jnp.float64)
        maturities_jnp = jnp.asarray(market_data.maturities, dtype=jnp.float64)
        is_calls_jnp = jnp.asarray(
            [bool(q.is_call) for q in market_data.quotes], dtype=jnp.bool_,
        )
        market_ivs_np = self.precompute_weights(market_data)
        market_ivs_jnp = jnp.asarray(market_ivs_np, dtype=jnp.float64)
        invalid_penalty = float(self.invalid_penalty)

        def _residual(model_prices_jnp: jnp.ndarray) -> jnp.ndarray:
            model_ivs = implied_volatility_jax(
                model_prices_jnp,
                spot,
                strikes_jnp,
                maturities_jnp,
                rate,
                is_calls_jnp,
                dividend_yield,
            )
            invalid = jnp.isnan(model_ivs) | jnp.isnan(market_ivs_jnp)
            # Replace NaNs with the market IV before differencing so the
            # ``jnp.where`` selector picks a finite value on both
            # branches — otherwise ``0 × NaN`` would leak NaN gradients
            # back through the chain rule.
            safe_model_ivs = jnp.where(invalid, market_ivs_jnp, model_ivs)
            diff = safe_model_ivs - market_ivs_jnp
            return jnp.where(invalid, invalid_penalty, diff)

        return _residual


__all__ = ["IVMSEObjective"]
