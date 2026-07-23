"""
Shared reparametrization & residual helpers for the CF calibrators
==================================================================

Functional sigmoid-box helpers and the JAX residual-factory scaffold that
were hand-copied across the Heston / Bates / Merton / Heston-Nandi
calibrators, consolidated into a single home. These are the *functional*
counterparts of the class-based :mod:`backend.calibration.transforms`
framework (a separate, currently unused paradigm); the calibrators use
these directly.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


def logit(x: float, lo: float, hi: float) -> float:
    """Inverse-sigmoid of ``x`` within the open box ``(lo, hi)``.

    Maps a constrained scalar to the unconstrained real line for the
    sigmoid reparametrisation: ``frac = (x - lo) / (hi - lo)`` clipped to
    ``[1e-8, 1 - 1e-8]`` so the log stays finite at the box boundaries.
    """
    frac = (x - lo) / (hi - lo)
    frac = float(np.clip(frac, 1e-8, 1.0 - 1e-8))
    return float(np.log(frac / (1.0 - frac)))


@dataclass
class _CompiledResiduals:
    """Bundle of JIT-compiled residual and Jacobian callables."""

    residual: Callable
    jacobian: Callable
    n_quote_residuals: int  # excludes the final penalty element


def build_cf_residual_fn(
    market_data: Any,
    objective: Any,
    *,
    theta_to_params: Callable[[jnp.ndarray], tuple],
    price_calls: Callable[[tuple, jnp.ndarray, float], jnp.ndarray],
    penalty_fn: Callable[[tuple], jnp.ndarray],
) -> _CompiledResiduals:
    """JIT-compile the (residual, Jacobian) pair shared by the CF calibrators.

    The per-surface boilerplate — packing strikes / call flags per
    maturity, pricing each maturity, recovering puts by parity, clamping
    to non-negative, applying the objective's residual transform, and
    appending a single scalar model penalty — is identical across the
    Heston / Bates / Merton / Heston-Nandi calibrators. The three
    model-specific pieces are injected as closures:

    ``theta_to_params(theta)``
        Unconstrained R^n -> constrained parameter tuple.
    ``price_calls(params, strikes_T, tau)``
        JAX call prices for one maturity (closes over the model CF,
        spot, rate, dividend yield and FFT grids).
    ``penalty_fn(params)``
        Scalar penalty residual appended last (Feller / Tikhonov /
        stationarity). The optimiser minimises ``½‖r‖²`` over
        ``[objective.transform(model_prices), penalty]``.
    """
    spot = float(market_data.spot)
    rate = float(market_data.rate)
    q = float(market_data.dividend_yield)
    unique_mats = [float(T) for T in market_data.unique_maturities]

    strikes_per_T: list[jnp.ndarray] = []
    is_calls_per_T: list[jnp.ndarray] = []
    running = 0
    for T in unique_mats:
        quotes = market_data.quotes_for_maturity(T)
        strikes_per_T.append(jnp.asarray([qt.strike for qt in quotes], dtype=jnp.float64))
        is_calls_per_T.append(jnp.asarray([qt.is_call for qt in quotes], dtype=jnp.bool_))
        running += len(quotes)
    n_quotes = running

    objective_residual_fn = objective.make_jax_residual_fn(market_data)

    def _residual_untraced(theta: jnp.ndarray) -> jnp.ndarray:
        params = theta_to_params(theta)

        model_prices_blocks: list[jnp.ndarray] = []
        for i, tau in enumerate(unique_mats):
            strikes_T = strikes_per_T[i]
            is_calls_T = is_calls_per_T[i]

            call_prices = price_calls(params, strikes_T, tau)

            # Put via parity: P = C - S*exp(-qT) + K*exp(-rT)
            disc_spot = spot * jnp.exp(-q * tau)
            disc_k = strikes_T * jnp.exp(-rate * tau)
            put_prices = call_prices - disc_spot + disc_k

            prices_T = jnp.where(is_calls_T, call_prices, put_prices)
            prices_T = jnp.maximum(prices_T, 0.0)
            model_prices_blocks.append(prices_T)

        model_prices = jnp.concatenate(model_prices_blocks)
        quote_residuals = objective_residual_fn(model_prices)

        penalty_res = penalty_fn(params)
        return jnp.concatenate([quote_residuals, jnp.array([penalty_res])])

    return _CompiledResiduals(
        residual=jax.jit(_residual_untraced),
        jacobian=jax.jit(jax.jacfwd(_residual_untraced)),
        n_quote_residuals=n_quotes,
    )
