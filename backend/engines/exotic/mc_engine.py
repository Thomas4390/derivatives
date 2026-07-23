"""
Model-dependent Monte-Carlo engine for exotic options.

The closed-form Haug catalog (``ExoticAnalyticEngine``) prices exotics under GBM
only. This engine prices the *same* exotics consistently with any model that
exposes a path simulator (Heston, Bates, Merton, GARCH variants, and GBM itself)
by simulating risk-neutral price paths and averaging a batch payoff.

Design
------
- ``model.create_simulator()`` is the polymorphic bridge to the right per-model
  ``BaseSimulator`` (the same one ``MonteCarloEngine`` uses for vanillas).
- Paths are simulated under the risk-neutral drift ``mu = r - q``; the payoff is
  discounted by ``exp(-r * T)``.
- The instrument -> batch-payoff mapping lives in ``_mc_payoff_registry`` so new
  exotics are added without touching this engine (Open-Closed).
- ``error`` carries the Monte-Carlo standard error of the price.

Greeks are intentionally not implemented here yet (bump-and-reprice with common
random numbers is a separate follow-up); use ``price`` for non-GBM exotics.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass

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
from backend.engines.exotic._mc_payoff_registry import lookup_mc_payoff
from backend.utils.constants.exotic import (
    FD_RATE_BUMP,
    FD_SPOT_REL_BUMP,
    FD_TIME_BUMP,
    FD_VOL_BUMP,
)
from backend.utils.constants.monte_carlo import (
    DEFAULT_MC_PATHS,
    DEFAULT_MC_SEED,
    DEFAULT_MC_STEPS_PER_YEAR,
)
from backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExoticMonteCarloEngine(PricingEngine):
    """Price exotic options under any MC-capable model by path simulation.

    Parameters
    ----------
    n_paths : int
        Number of simulated paths.
    steps_per_year : int
        Monitoring resolution; ``n_steps = max(1, round(T * steps_per_year))``.
    seed : int or None
        RNG seed. A fixed seed keeps prices reproducible across calls.
    """

    n_paths: int = DEFAULT_MC_PATHS
    steps_per_year: int = DEFAULT_MC_STEPS_PER_YEAR
    seed: int | None = DEFAULT_MC_SEED

    @property
    def capability(self) -> PricingCapability:
        return PricingCapability.MONTE_CARLO

    @property
    def supported_exercises(self) -> list[ExerciseStyle]:
        return [ExerciseStyle.EUROPEAN]

    def can_price(self, instrument: Instrument, model: Model) -> bool:
        """True if the instrument has an MC payoff and the model supports MC."""
        return (
            instrument.exercise_style == ExerciseStyle.EUROPEAN
            and PricingCapability.MONTE_CARLO in model.supported_engines
            and lookup_mc_payoff(instrument) is not None
        )

    def price(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
    ) -> PricingResult:
        """Monte-Carlo price of an exotic under ``model``.

        Raises
        ------
        ValueError
            If the instrument is not registered or the model has no simulator.
        """
        payoff_fn = lookup_mc_payoff(instrument)
        if payoff_fn is None:
            raise ValueError(
                f"ExoticMonteCarloEngine has no payoff registered for "
                f"{type(instrument).__name__}"
            )

        spot: float = market.spot
        rate: float = market.rate
        q: float = market.dividend_yield
        maturity: float = instrument.maturity
        n_steps = max(1, int(round(maturity * self.steps_per_year)))

        try:
            simulator = model.create_simulator()
        except NotImplementedError as exc:  # model has no MC bridge
            raise ValueError(
                f"{model.name} does not support Monte-Carlo simulation"
            ) from exc

        result = simulator.simulate_paths(
            s0=spot,
            mu=rate - q,  # risk-neutral drift (dividend-adjusted)
            t=maturity,
            n_paths=self.n_paths,
            n_steps=n_steps,
            seed=self.seed,
        )

        payoffs = np.ascontiguousarray(
            payoff_fn(result.price_paths, instrument), dtype=np.float64
        )
        discount = float(np.exp(-rate * maturity))
        price = discount * float(payoffs.mean())
        # Sample standard error of the discounted MC estimator.
        n = payoffs.shape[0]
        std_error = discount * float(payoffs.std(ddof=1)) / np.sqrt(n) if n > 1 else 0.0

        return PricingResult(
            price=price,
            engine="ExoticMonteCarloEngine",
            model=model.name,
            error=std_error,
            metadata={"n_paths": float(n), "n_steps": float(n_steps)},
        )

    def greeks(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
    ) -> GreeksResult:
        """Common-random-numbers bump-and-reprice Greeks under ``model``.

        Delta, gamma and rho are computed *exactly* from a single simulated path
        matrix by rescaling: under risk-neutral ``dS/S`` dynamics ``S_t = S0 * R_t``
        so a spot bump is ``paths * (S0+-dS)/S0`` and a rate bump is
        ``paths * exp(+-dr * t)`` plus the discount change -- no re-simulation and
        no Monte-Carlo noise on those Greeks. Vega and theta re-simulate with the
        SAME seed and step count (CRN) so only the bumped quantity moves the paths.
        Vega is ``0`` for models with no scalar vol level (e.g. GARCH). Bump sizes
        and per-100bp vega/rho scaling match the closed-form ``fd_greeks``.

        Note
        ----
        Theta re-simulates to ``T - 1 day`` holding the contract fields fixed; for
        partial-window exotics the window-index mapping uses the original maturity,
        a sub-one-day approximation.
        """
        payoff_fn = lookup_mc_payoff(instrument)
        if payoff_fn is None:
            raise ValueError(
                f"ExoticMonteCarloEngine has no payoff registered for "
                f"{type(instrument).__name__}"
            )

        s0: float = market.spot
        rate: float = market.rate
        q: float = market.dividend_yield
        maturity: float = instrument.maturity
        n_steps = max(1, int(round(maturity * self.steps_per_year)))

        try:
            simulator = model.create_simulator()
        except NotImplementedError as exc:
            raise ValueError(
                f"{model.name} does not support Monte-Carlo simulation"
            ) from exc

        base = simulator.simulate_paths(
            s0=s0,
            mu=rate - q,
            t=maturity,
            n_paths=self.n_paths,
            n_steps=n_steps,
            seed=self.seed,
        )
        paths = base.price_paths
        time_grid = np.asarray(base.time_grid, dtype=np.float64)
        disc = float(np.exp(-rate * maturity))

        def mc(p: np.ndarray, discount: float) -> float:
            payoffs = np.asarray(payoff_fn(p, instrument), dtype=np.float64)
            return discount * float(payoffs.mean())

        price = mc(paths, disc)

        # Delta / gamma -- exact spot rescale (CRN, no re-simulation).
        d_s = s0 * FD_SPOT_REL_BUMP
        v_up = mc(paths * ((s0 + d_s) / s0), disc)
        v_dn = mc(paths * ((s0 - d_s) / s0), disc)
        delta = (v_up - v_dn) / (2.0 * d_s)
        gamma = (v_up - 2.0 * price + v_dn) / (d_s * d_s)

        # Rho -- exact rate rescale (deterministic drift shift) + bumped discount.
        d_r = FD_RATE_BUMP
        v_ru = mc(
            paths * np.exp(d_r * time_grid)[None, :],
            float(np.exp(-(rate + d_r) * maturity)),
        )
        v_rd = mc(
            paths * np.exp(-d_r * time_grid)[None, :],
            float(np.exp(-(rate - d_r) * maturity)),
        )
        rho = (v_ru - v_rd) / (2.0 * d_r) / 100.0

        # Vega -- re-simulate with the same seed and a bumped vol level (CRN).
        d_v = FD_VOL_BUMP
        model_up = bump_model_vol(model, d_v)
        model_dn = bump_model_vol(model, -d_v)
        if model_up is not None and model_dn is not None:
            p_vu = model_up.create_simulator().simulate_paths(
                s0=s0,
                mu=rate - q,
                t=maturity,
                n_paths=self.n_paths,
                n_steps=n_steps,
                seed=self.seed,
            )
            p_vd = model_dn.create_simulator().simulate_paths(
                s0=s0,
                mu=rate - q,
                t=maturity,
                n_paths=self.n_paths,
                n_steps=n_steps,
                seed=self.seed,
            )
            vega = (
                (mc(p_vu.price_paths, disc) - mc(p_vd.price_paths, disc))
                / (2.0 * d_v)
                / 100.0
            )
        else:
            vega = 0.0

        # Theta -- re-simulate to T - 1 day, same seed/steps (CRN); one-sided.
        d_t = FD_TIME_BUMP
        if d_t < maturity:
            base_dn = simulator.simulate_paths(
                s0=s0,
                mu=rate - q,
                t=maturity - d_t,
                n_paths=self.n_paths,
                n_steps=n_steps,
                seed=self.seed,
            )
            v_td = mc(base_dn.price_paths, float(np.exp(-rate * (maturity - d_t))))
            theta = v_td - price
        else:
            theta = 0.0

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
        *,
        n_steps: int | None = None,
    ) -> np.ndarray | None:
        """Risk-neutral price across a spot grid from ONE simulated path matrix.

        Simulates a single path matrix at ``market.spot`` and rescales it by
        ``S_i / s0`` across ``spot_range``: under risk-neutral ``dS/S`` dynamics
        ``S_t = S0 * R_t`` and the payoff is degree-1 homogeneous in
        (spot, strike, barrier), so ``e^{-rT} E[payoff(paths * S_i/s0)]`` is the
        *exact* price at spot ``S_i`` for the original contract. This is the same
        exact spot rescale ``greeks()`` uses for delta/gamma, swept over the grid,
        so the whole curve costs one simulation and its finite differences are the
        common-random-number (noise-free) model-consistent delta/gamma.

        ``n_steps`` overrides the step count (default ``round(maturity *
        steps_per_year)``). Pass a fixed value across a maturity finite difference
        (theta) so the bumped curves keep the same RNG draw count and stay
        common-random-number aligned -- otherwise the step count jumps by a step or
        two and MC noise swamps the one-day decay.

        Returns prices aligned with ``spot_range``, or ``None`` if the instrument
        has no registered payoff or the model exposes no simulator -- the caller
        then falls back to the closed-form Greeks path.
        """
        payoff_fn = lookup_mc_payoff(instrument)
        if payoff_fn is None:
            return None

        s0: float = market.spot
        rate: float = market.rate
        q: float = market.dividend_yield
        maturity: float = instrument.maturity
        if n_steps is None:
            n_steps = max(1, int(round(maturity * self.steps_per_year)))
        else:
            n_steps = max(1, int(n_steps))

        try:
            simulator = model.create_simulator()
        except NotImplementedError:
            return None

        base = simulator.simulate_paths(
            s0=s0,
            mu=rate - q,
            t=maturity,
            n_paths=self.n_paths,
            n_steps=n_steps,
            seed=self.seed,
        )
        paths = base.price_paths
        disc = float(np.exp(-rate * maturity))
        spots = np.ascontiguousarray(np.asarray(spot_range, dtype=np.float64))
        out = np.empty(spots.shape[0], dtype=np.float64)
        for i in range(spots.shape[0]):
            payoffs = np.asarray(
                payoff_fn(paths * (spots[i] / s0), instrument), dtype=np.float64
            )
            out[i] = disc * float(payoffs.mean())
        return out
