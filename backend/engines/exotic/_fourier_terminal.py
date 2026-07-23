"""
Fourier (COS) pricing of terminal exotics under any characteristic-function model.

The COS method (Fang-Oosterlee 2008) recovers ``E^Q[g(S_T)]`` from the model's
characteristic function of ``X = ln S_T``:

    E[g(X)] ~= sum'_{k=0}^{N-1} A_k V_k ,   price = e^{-rT} E[g(X)]

with density cosine coefficients ``A_k`` (from the CF) and payoff cosine
coefficients ``V_k`` (from ``g``); ``sum'`` halves the ``k = 0`` term. This is the
model-consistent fast path for the terminal exotics (log contract/option,
powered, capped power, supershare, gap, asset-or-nothing) under FFT-capable
models (Heston / Bates / Merton). GARCH-style models (no CF) fall back to MC.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from backend.core.interfaces import Instrument, Model, PricingEngine
from backend.core.market import MarketEnvironment
from backend.core.result_types import (
    ExerciseStyle,
    GreeksResult,
    PricingCapability,
    PricingResult,
)
from backend.engines.exotic._greeks_bumps import bump_model_vol
from backend.engines.exotic._terminal_payoffs import build_terminal_payoff
from backend.utils.constants.exotic import (
    FD_RATE_BUMP,
    FD_SPOT_REL_BUMP,
    FD_TIME_BUMP,
    FD_VOL_BUMP,
)
from backend.utils.logging import get_logger

logger = get_logger(__name__)

CFVectorized = Callable[[np.ndarray], np.ndarray]
TerminalPayoff = Callable[[np.ndarray], np.ndarray]


def cos_terminal_price(
    cf_vectorized: CFVectorized,
    payoff_fn: TerminalPayoff,
    s0: float,
    t: float,
    r: float,
    q: float,
    *,
    n_terms: int = 192,
    trunc_L: float = 12.0,
    n_quad: int = 4096,
) -> float:
    """Price ``e^{-rT} E^Q[g(S_T)]`` for a terminal payoff via the COS method.

    Parameters
    ----------
    cf_vectorized : callable
        ``u -> E^Q[exp(i u ln S_T)]`` for a real array ``u`` (e.g.
        ``model.characteristic_function_vectorized(u, s0, t, r, q)``).
    payoff_fn : callable
        Vectorised terminal payoff ``g(s_t) -> payoff``.
    s0, t, r, q : float
        Spot, maturity, rate, dividend yield.
    n_terms : int
        Number of COS terms.
    trunc_L : float
        Truncation width in standard deviations of ``ln S_T``.
    n_quad : int
        Quadrature points for the payoff cosine coefficients ``V_k``.

    Returns
    -------
    float
        Present value of the terminal claim.
    """
    # Cumulants of X = ln S_T from the log-CF via central finite differences.
    h = 1e-4
    psi = np.log(cf_vectorized(np.array([-h, 0.0, h], dtype=np.complex128)))
    c1 = float(np.imag((psi[2] - psi[0]) / (2.0 * h)))
    c2 = float(-np.real((psi[2] - 2.0 * psi[1] + psi[0]) / (h * h)))

    half = trunc_L * max(np.sqrt(abs(c2)), 1e-3)
    a, b = c1 - half, c1 + half
    width = b - a

    k = np.arange(n_terms)
    u = k * np.pi / width
    cf_u = cf_vectorized(u.astype(np.complex128))
    a_k = (2.0 / width) * np.real(cf_u * np.exp(-1j * u * a))

    # Payoff cosine coefficients V_k via vectorised trapezoid over x in [a, b].
    xs = np.linspace(a, b, n_quad)
    gx = np.asarray(payoff_fn(np.exp(xs)), dtype=np.float64)
    cos_km = np.cos(np.outer(k, xs - a) * (np.pi / width))
    v_k = np.trapezoid(gx[None, :] * cos_km, xs, axis=1)

    value = float(a_k @ v_k - 0.5 * a_k[0] * v_k[0])
    return float(np.exp(-r * t) * value)


@dataclass(frozen=True)
class ExoticFourierEngine(PricingEngine):
    """COS pricing of terminal exotics under any FFT-capable (CF) model."""

    n_terms: int = 192
    trunc_L: float = 12.0
    n_quad: int = 4096

    @property
    def capability(self) -> PricingCapability:
        return PricingCapability.FFT

    @property
    def supported_exercises(self) -> list[ExerciseStyle]:
        return [ExerciseStyle.EUROPEAN]

    def can_price(self, instrument: Instrument, model: Model) -> bool:
        """True for a terminal exotic on a model exposing a characteristic function."""
        return (
            instrument.exercise_style == ExerciseStyle.EUROPEAN
            and PricingCapability.FFT in model.supported_engines
            and build_terminal_payoff(instrument) is not None
        )

    def price(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
    ) -> PricingResult:
        """COS price of a terminal exotic under ``model``."""
        payoff_fn = build_terminal_payoff(instrument)
        if payoff_fn is None:
            raise ValueError(
                f"ExoticFourierEngine has no terminal payoff for "
                f"{type(instrument).__name__}"
            )

        s0: float = market.spot
        r: float = market.rate
        q: float = market.dividend_yield
        t: float = instrument.maturity

        def cf(u: np.ndarray) -> np.ndarray:
            return np.asarray(model.characteristic_function_vectorized(u, s0, t, r, q))

        price = cos_terminal_price(
            cf,
            payoff_fn,
            s0,
            t,
            r,
            q,
            n_terms=self.n_terms,
            trunc_L=self.trunc_L,
            n_quad=self.n_quad,
        )
        return PricingResult(
            price=price,
            engine="ExoticFourierEngine",
            model=model.name,
            error=None,
        )

    def greeks(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
    ) -> GreeksResult:
        """Deterministic FD bump-and-reprice Greeks via the COS pricer.

        The COS price is noise-free, so plain central differences are precise (no
        CRN needed). Spot/rate/maturity bumps re-evaluate the model CF at the
        bumped argument; vega bumps the model vol level (``0`` if the model has no
        scalar vol). Bump sizes and per-100bp vega/rho scaling match ``fd_greeks``.
        """
        payoff_fn = build_terminal_payoff(instrument)
        if payoff_fn is None:
            raise ValueError(
                f"ExoticFourierEngine has no terminal payoff for "
                f"{type(instrument).__name__}"
            )

        q: float = market.dividend_yield

        def price_at(m: Model, spot: float, rate: float, mat: float) -> float:
            def cf(u: np.ndarray) -> np.ndarray:
                return np.asarray(
                    m.characteristic_function_vectorized(u, spot, mat, rate, q)
                )

            return cos_terminal_price(
                cf,
                payoff_fn,
                spot,
                mat,
                rate,
                q,
                n_terms=self.n_terms,
                trunc_L=self.trunc_L,
                n_quad=self.n_quad,
            )

        s0: float = market.spot
        r: float = market.rate
        t: float = instrument.maturity
        price = price_at(model, s0, r, t)

        d_s = s0 * FD_SPOT_REL_BUMP
        v_up = price_at(model, s0 + d_s, r, t)
        v_dn = price_at(model, s0 - d_s, r, t)
        delta = (v_up - v_dn) / (2.0 * d_s)
        gamma = (v_up - 2.0 * price + v_dn) / (d_s * d_s)

        d_r = FD_RATE_BUMP
        rho = (
            (price_at(model, s0, r + d_r, t) - price_at(model, s0, r - d_r, t))
            / (2.0 * d_r)
            / 100.0
        )

        d_v = FD_VOL_BUMP
        model_up = bump_model_vol(model, d_v)
        model_dn = bump_model_vol(model, -d_v)
        if model_up is not None and model_dn is not None:
            vega = (
                (price_at(model_up, s0, r, t) - price_at(model_dn, s0, r, t))
                / (2.0 * d_v)
                / 100.0
            )
        else:
            vega = 0.0

        d_t = FD_TIME_BUMP
        theta = price_at(model, s0, r, t - d_t) - price if d_t < t else 0.0

        return GreeksResult(
            price=price,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
        )

    def price_curve(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
        spot_range: np.ndarray,
    ) -> np.ndarray | None:
        """COS price of a terminal exotic across a spot grid (deterministic).

        The COS pricer re-evaluates the model characteristic function at each spot,
        so every node is exact and noise-free -- no simulation or homogeneity
        rescale needed. Finite differences of this curve are the model-consistent
        spot Greeks (they match ``greeks()``). Returns prices aligned with
        ``spot_range``, or ``None`` if the instrument has no terminal payoff under
        this engine.
        """
        payoff_fn = build_terminal_payoff(instrument)
        if payoff_fn is None:
            return None

        q: float = market.dividend_yield
        r: float = market.rate
        t: float = instrument.maturity
        spots = np.ascontiguousarray(np.asarray(spot_range, dtype=np.float64))
        out = np.empty(spots.shape[0], dtype=np.float64)
        for i in range(spots.shape[0]):
            s = float(spots[i])

            def cf(u: np.ndarray, _s: float = s) -> np.ndarray:
                return np.asarray(
                    model.characteristic_function_vectorized(u, _s, t, r, q)
                )

            out[i] = cos_terminal_price(
                cf,
                payoff_fn,
                s,
                t,
                r,
                q,
                n_terms=self.n_terms,
                trunc_L=self.trunc_L,
                n_quad=self.n_quad,
            )
        return out
