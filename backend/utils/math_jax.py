"""
JAX-traceable Black-Scholes Implied-Volatility Inversion
=========================================================

Wraps the safeguarded Newton-bisection IV inverter
(``backend.utils.math.implied_volatility``) into a JAX-compatible
primitive whose backward pass is the **closed-form** implicit-function
gradient

.. math::

    \\frac{\\partial \\sigma}{\\partial P}
        \\;=\\; \\frac{1}{\\partial P / \\partial \\sigma}
        \\;=\\; \\frac{1}{\\mathcal V_{BS}(\\sigma^\\star)} \\,,

derived by implicitly differentiating ``BS(σ; S, K, T, r, q) = P`` at the
converged root ``σ* = implied_volatility(P, …)``. The same identity that
motivates ``vega_weighted`` as a *first-order approximation* of
``iv_mse`` becomes **exact** when applied at ``σ = σ*`` instead of
``σ = σ_market``.

The forward pass uses :func:`jax.pure_callback` to delegate the
iteration to the existing Numba rtsafe code — we do **not** re-implement
the inversion in JAX. Only the gradient is JAX-side, and that is just a
vectorised division by a closed-form vega.

This is what lets ``LM-JAX`` calibrate against the exact ``iv_mse``
objective instead of the ``vega_weighted`` fallback.

References
----------
* Implicit function theorem applied to the monotone BS price/IV bijection.
* Cont & Tankov (2004), *Financial Modelling with Jump Processes*, §13.1 —
  the original vega-weighting derivation this primitive makes exact.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from functools import partial

import numpy as np

# Importing the AAD JAX config side-effects ``jax_enable_x64=True``. We
# repeat the call below in case ``_jax_config`` was stripped (light
# branch) so this module remains self-sufficient.
try:  # pragma: no cover - exercised by import order in dev/main only
    import backend.engines.aad._jax_config  # noqa: F401
except ImportError:  # pragma: no cover - light branch removes the helper
    pass

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.scipy.stats as jstats  # noqa: E402

# Idempotent — already true if ``_jax_config`` ran above.
jax.config.update("jax_enable_x64", True)

from backend.utils.math import implied_volatility as _iv_numba  # noqa: E402

# Floor on the vega used in the JVP denominator. Deep OTM / very short
# expiries can drive vega to zero; dividing by raw zero would NaN the
# Jacobian and derail the optimizer. Clipping below to a tiny floor
# turns the JVP into a large-but-finite number (the optimizer's
# trust-region machinery handles large steps gracefully; NaNs it does
# not).
_EPS_VEGA: float = 1e-12


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _bs_vega_jax(
    spot: float,
    strikes: jnp.ndarray,
    maturities: jnp.ndarray,
    rate: float,
    sigmas: jnp.ndarray,
    dividend_yield: float,
) -> jnp.ndarray:
    """Vectorised Black-Scholes vega ``V = S e^{-qT} φ(d₁) √T`` in pure JAX.

    Self-contained re-implementation (not imported from
    ``backend.engines.aad.calibration.calibrator``) so this module stays
    callable on the ``light`` branch where that file is intentionally
    absent.
    """
    sigma = jnp.maximum(sigmas, 1e-8)
    T = jnp.maximum(maturities, 1e-8)
    sqrt_t = jnp.sqrt(T)
    d1 = (
        jnp.log(spot / strikes) + (rate - dividend_yield + 0.5 * sigma**2) * T
    ) / (sigma * sqrt_t)
    return spot * jnp.exp(-dividend_yield * T) * jstats.norm.pdf(d1) * sqrt_t


def _iv_callback_vec(
    prices: np.ndarray,
    spot: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    rate: np.ndarray,
    is_calls: np.ndarray,
    dividend_yield: np.ndarray,
) -> np.ndarray:
    """Pure-NumPy callback wrapping the Numba scalar inverter.

    Invoked by ``jax.pure_callback`` once per residual evaluation. Loops
    over ``n_quotes`` and dispatches each quote to the safeguarded
    Newton-bisection in :func:`backend.utils.math.implied_volatility`,
    which returns ``NaN`` for out-of-arb-bounds prices (the IVMSE
    residual masks those downstream).
    """
    spot_f = float(spot)
    rate_f = float(rate)
    q_f = float(dividend_yield)
    n = prices.shape[0]
    sigmas = np.empty(n, dtype=np.float64)
    for i in range(n):
        sigmas[i] = _iv_numba(
            float(prices[i]),
            spot_f,
            float(strikes[i]),
            float(maturities[i]),
            rate_f,
            bool(is_calls[i]),
            q_f,
        )
    return sigmas


def _iv_primal(
    prices: jnp.ndarray,
    spot: float,
    strikes: jnp.ndarray,
    maturities: jnp.ndarray,
    rate: float,
    is_calls: jnp.ndarray,
    dividend_yield: float,
) -> jnp.ndarray:
    """Concrete (non-traceable) IV inversion via :func:`jax.pure_callback`.

    Used by both the primal entry point and the JVP rule so we only need
    one path through the Numba inverter per evaluation regardless of
    whether the call site is the residual, the Jacobian, or a one-off
    diagnostic.
    """
    return jax.pure_callback(
        _iv_callback_vec,
        jax.ShapeDtypeStruct(prices.shape, jnp.float64),
        prices,
        spot,
        strikes,
        maturities,
        rate,
        is_calls,
        dividend_yield,
    )


# --------------------------------------------------------------------------- #
# Public primitive — implied_volatility_jax
# --------------------------------------------------------------------------- #


@partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3, 4, 5, 6))
def implied_volatility_jax(
    prices: jnp.ndarray,
    spot: float,
    strikes: jnp.ndarray,
    maturities: jnp.ndarray,
    rate: float,
    is_calls: jnp.ndarray,
    dividend_yield: float,
) -> jnp.ndarray:
    """Vectorised IV inversion, JAX-traceable through ``jacfwd`` / ``grad``.

    Parameters
    ----------
    prices :
        Model prices (one per quote). The **only** differentiable
        argument — the others are market data held fixed at trace time.
    spot, rate, dividend_yield :
        Market scalars.
    strikes, maturities, is_calls :
        Per-quote arrays of length ``n_quotes``.

    Returns
    -------
    jnp.ndarray
        Implied volatilities, shape ``(n_quotes,)``. ``NaN`` for prices
        outside the no-arbitrage bounds (the caller is expected to mask
        those samples via ``jnp.where`` — see
        :meth:`backend.calibration.objectives.iv_mse.IVMSEObjective.make_jax_residual_fn`).

    Notes
    -----
    Forward: delegates to the existing Numba ``implied_volatility`` via
    ``jax.pure_callback`` — no JAX-side iteration loop, no fixed
    iteration count, no Numba-vs-JAX numerical drift.

    JVP (custom): the implicit function theorem applied to
    ``BS(σ; S, K, T, r, q) = P`` gives ``∂σ/∂P = 1 / Vega(σ*)``. The
    rule below evaluates ``Vega`` in pure JAX at the converged ``σ*``
    and propagates ``σ̇ = Ṗ / Vega(σ*)``.
    """
    return _iv_primal(
        prices, spot, strikes, maturities, rate, is_calls, dividend_yield,
    )


@implied_volatility_jax.defjvp
def _implied_volatility_jax_jvp(
    spot,
    strikes,
    maturities,
    rate,
    is_calls,
    dividend_yield,
    primals,
    tangents,
):
    (prices,) = primals
    (prices_dot,) = tangents
    sigmas = _iv_primal(
        prices, spot, strikes, maturities, rate, is_calls, dividend_yield,
    )
    vegas = _bs_vega_jax(spot, strikes, maturities, rate, sigmas, dividend_yield)
    # Safeguard against vega ≈ 0 at the wings — see ``_EPS_VEGA`` doc.
    vegas_safe = jnp.maximum(vegas, _EPS_VEGA)
    sigmas_dot = prices_dot / vegas_safe
    return sigmas, sigmas_dot


__all__ = ["implied_volatility_jax"]
