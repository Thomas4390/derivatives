"""
Exotic Analytical Engine
========================

Closed-form pricing engine for exotic options under GBM:
- Barrier Options (Reiner-Rubinstein 1991) -- all 8 types
- Asian Geometric Options (Kemna-Vorst 1990)
- Digital/Binary Options (cash-or-nothing)
- Lookback Options (Goldman-Sosin-Gatto 1979, Conze-Viswanathan 1991)
- Chooser Options (Rubinstein 1991)
- Asset-or-Nothing Options
- Power Options
- Gap Options

All pricing kernels are Numba-compiled for performance.
Greeks are computed via central finite differences aligned with
``GreeksBumpConfig`` defaults from ``backend.greeks.numerical``.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from backend.core.interfaces import Instrument, Model, PricingEngine
from backend.core.market import MarketEnvironment
from backend.core.result_types import (
    ExerciseStyle,
    GreeksResult,
    PricingCapability,
    PricingResult,
)
from backend.engines.exotic._registry import (
    EXOTIC_PRICER_REGISTRY,
    fd_greeks,
    lookup,
    registry_price,
)
from backend.engines.exotic.asian import asian_geometric_price
from backend.engines.exotic.asset_or_nothing import asset_or_nothing_price
from backend.engines.exotic.barrier import barrier_option_price
from backend.engines.exotic.chooser import chooser_price
from backend.engines.exotic.digital import digital_price
from backend.engines.exotic.gap import gap_option_price
from backend.engines.exotic.lookback import (
    lookback_fixed_price,
    lookback_floating_price,
)
from backend.engines.exotic.power import power_option_price
from backend.instruments.options import (
    AsianOption,
    AssetOrNothingOption,
    BarrierOption,
    ChooserOption,
    DigitalOption,
    GapOption,
    LookbackOption,
    PowerOption,
)
from backend.models.gbm import GBMModel
from backend.engines.exotic._greeks_kernels import (  # noqa: F401
    _exotic_price,
    exotic_calculate_greeks,
    exotic_greeks_batch,
    exotic_greeks_surface,
    exotic_price_param_sweep,
    exotic_price_surface,
)
from backend.engines.exotic._option_types import (
    ASIAN_GEO,
    ASSET_OR_NOTHING,
    BARRIER,
    CHOOSER,
    DIGITAL,
    GAP,
    LOOKBACK_FIXED,
    LOOKBACK_FLOATING,
    POWER,
)


class _NumericalExoticEngine(Protocol):
    """Structural type for the non-GBM exotic routes (MC / Fourier).

    Both ``ExoticMonteCarloEngine`` and ``ExoticFourierEngine`` satisfy it; it
    exposes the ``greeks`` method that ``PricingEngine`` does not declare.
    """

    def can_price(self, instrument: Instrument, model: Model) -> bool: ...

    def price(
        self, instrument: Instrument, model: Model, market: MarketEnvironment
    ) -> PricingResult: ...

    def greeks(
        self, instrument: Instrument, model: Model, market: MarketEnvironment
    ) -> GreeksResult: ...


# =============================================================================
# ENGINE CLASS
# =============================================================================


@dataclass(frozen=True)
class ExoticAnalyticEngine(PricingEngine):
    """
    Analytical pricing engine for exotic options under GBM.

    Supports:
    - Barrier Options (all 8 types) via Reiner-Rubinstein 1991
    - Asian Geometric Options via Kemna-Vorst 1990
    - Digital/Binary Options (cash-or-nothing)
    - Lookback Options (floating and fixed strike)
    - Chooser Options via Rubinstein 1991
    - Asset-or-Nothing Options
    - Power Options
    - Gap Options

    All kernels are Numba-compiled for performance.

    Examples
    --------
    engine = ExoticAnalyticEngine()
    gbm = GBMModel(sigma=0.25)
    market = MarketEnvironment(spot=100, rate=0.05)

    barrier = BarrierOption(100, 110, 0.25, is_call=True, is_up=True)
    result = engine.price(barrier, gbm, market)
    greeks = engine.greeks(barrier, gbm, market)
    """

    @property
    def capability(self) -> PricingCapability:
        return PricingCapability.ANALYTICAL

    @property
    def supported_exercises(self) -> list[ExerciseStyle]:
        return [ExerciseStyle.EUROPEAN]

    def _numerical_engine(
        self, instrument: Instrument, model: Model
    ) -> _NumericalExoticEngine:
        """Pick the model-dependent route for a non-GBM exotic.

        Terminal exotics on a characteristic-function (FFT-capable) model take the
        fast Fourier/COS route; everything else (path-dependent exotics, or models
        without a CF such as GARCH) takes Monte-Carlo. Lazily imported so the
        package ``__init__`` never triggers an import cycle through these engines.
        """
        from backend.engines.exotic._fourier_terminal import ExoticFourierEngine
        from backend.engines.exotic.mc_engine import ExoticMonteCarloEngine

        fourier = ExoticFourierEngine()
        if fourier.can_price(instrument, model):
            return fourier
        return ExoticMonteCarloEngine()

    def can_price(self, instrument: Instrument, model: Model) -> bool:
        """
        Check if this engine can price the given combination.

        Under GBM the closed-form catalog is used. Under any other MC-capable
        model (Heston, Bates, Merton, GARCH...) the request is routed to the
        model-dependent Monte-Carlo engine. For AsianOption, only geometric
        average is supported by the GBM closed form (the MC route also handles
        the arithmetic average).
        """
        if instrument.exercise_style != ExerciseStyle.EUROPEAN:
            return False

        # Non-GBM models: the closed forms are GBM-only, so route to the
        # model-dependent numerical engine (Fourier for terminal exotics, else MC).
        if not isinstance(model, GBMModel):
            return self._numerical_engine(instrument, model).can_price(
                instrument, model
            )

        # Registry-dispatched (Haug catalog) exotics: Open-Closed fast path.
        if type(instrument) in EXOTIC_PRICER_REGISTRY:
            return True

        if isinstance(instrument, BarrierOption):
            return True
        if isinstance(instrument, DigitalOption):
            return True
        if isinstance(instrument, LookbackOption):
            return True
        if isinstance(instrument, AsianOption):
            return instrument.average_type == "geometric"
        if isinstance(instrument, ChooserOption):
            return True
        if isinstance(instrument, AssetOrNothingOption):
            return True
        if isinstance(instrument, PowerOption):
            return True
        if isinstance(instrument, GapOption):
            return True

        return False

    def price(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
    ) -> PricingResult:
        """
        Price an exotic option analytically.

        Parameters
        ----------
        instrument : Instrument
            Exotic option (Barrier, Asian geometric, Digital, or Lookback)
        model : GBMModel
            GBM model with constant volatility
        market : MarketEnvironment
            Current market conditions

        Returns
        -------
        PricingResult
            Analytical price
        """
        if not self.can_price(instrument, model):
            raise ValueError(
                f"ExoticAnalyticEngine cannot price {type(instrument).__name__} "
                f"with {type(model).__name__}"
            )

        # Non-GBM models price through the model-dependent numerical route.
        if not isinstance(model, GBMModel):
            return self._numerical_engine(instrument, model).price(
                instrument, model, market
            )

        S: float = market.spot
        r: float = market.rate
        q: float = market.dividend_yield
        sigma: float = model.sigma
        T: float = instrument.maturity

        # Registry-dispatched (Haug catalog) exotics: Open-Closed fast path.
        spec = lookup(instrument)
        if spec is not None:
            return PricingResult(
                price=registry_price(spec, instrument, model, market),
                engine="ExoticAnalyticEngine",
                model=model.name,
            )

        if isinstance(instrument, BarrierOption):
            p = barrier_option_price(
                S=S,
                K=instrument.strike,
                H=instrument.barrier,
                T=T,
                r=r,
                q=q,
                sigma=sigma,
                is_call=instrument.is_call,
                is_knock_in=instrument.is_knock_in,
                is_up=instrument.is_up,
                rebate=instrument.rebate,
            )
        elif isinstance(instrument, AsianOption):
            p = asian_geometric_price(
                S=S,
                K=instrument.strike,
                T=T,
                r=r,
                q=q,
                sigma=sigma,
                is_call=instrument.is_call,
            )
        elif isinstance(instrument, DigitalOption):
            p = digital_price(
                S=S,
                K=instrument.strike,
                T=T,
                r=r,
                q=q,
                sigma=sigma,
                is_call=instrument.is_call,
                payout=instrument.payout,
            )
        elif isinstance(instrument, LookbackOption):
            if instrument.lookback_type == "fixed":
                p = lookback_fixed_price(
                    S=S,
                    K=instrument.strike,
                    M_min=S,
                    M_max=S,
                    T=T,
                    r=r,
                    q=q,
                    sigma=sigma,
                    is_call=instrument.is_call,
                )
            else:
                p = lookback_floating_price(
                    S=S,
                    M_min=S,
                    M_max=S,
                    T=T,
                    r=r,
                    q=q,
                    sigma=sigma,
                    is_call=instrument.is_call,
                )
        elif isinstance(instrument, ChooserOption):
            p = chooser_price(
                S=S,
                K=instrument.strike,
                T=T,
                t_c=instrument.choice_time,
                r=r,
                q=q,
                sigma=sigma,
            )
        elif isinstance(instrument, AssetOrNothingOption):
            p = asset_or_nothing_price(
                S=S,
                K=instrument.strike,
                T=T,
                r=r,
                q=q,
                sigma=sigma,
                is_call=instrument.is_call,
            )
        elif isinstance(instrument, PowerOption):
            p = power_option_price(
                S=S,
                K=instrument.strike,
                T=T,
                r=r,
                q=q,
                sigma=sigma,
                is_call=instrument.is_call,
                n=instrument.power,
            )
        elif isinstance(instrument, GapOption):
            p = gap_option_price(
                S=S,
                K1=instrument.strike,
                K2=instrument.trigger,
                T=T,
                r=r,
                q=q,
                sigma=sigma,
                is_call=instrument.is_call,
            )
        else:
            raise ValueError(f"Unsupported instrument: {type(instrument).__name__}")

        return PricingResult(
            price=p,
            engine="ExoticAnalyticEngine",
            model=model.name,
        )

    def greeks_surface(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
        spot_range: np.ndarray,
    ) -> dict[str, np.ndarray] | None:
        """First-order Greeks over a spot range in one parallel kernel sweep.

        Returns ``{price, delta, gamma, vega, theta, rho}`` arrays computed by
        the ``@njit(parallel=True)`` ``exotic_greeks_surface`` kernel — the same
        ``exotic_greeks_batch`` finite differences that ``greeks()`` runs per
        point, so it is identical to the per-spot path but pays one Numba
        dispatch and runs the spots across cores.

        Consumed as a fast path by ``GreeksCalculator.calculate_surface`` (the
        duck-typed analogue of the AAD ``greeks_profile``). Returns ``None`` for
        registry-dispatched (Haug catalog) exotics — those route through
        ``greeks()``'s ``fd_greeks`` per spot, which this kernel does not
        reproduce — and for anything this engine cannot price; the caller then
        falls back to the per-spot loop, keeping results consistent.
        """
        if not isinstance(model, GBMModel) or not self.can_price(instrument, model):
            return None
        # Registry (Haug) exotics use fd_greeks in greeks(); don't shadow them.
        if lookup(instrument) is not None:
            return None

        r: float = market.rate
        q: float = market.dividend_yield
        sigma: float = model.sigma
        T: float = instrument.maturity
        spots = np.ascontiguousarray(np.asarray(spot_range, dtype=np.float64))

        H = 0.0
        # M_min == M_max == 0 ⇒ the kernel tracks the running extreme per spot
        # (= S), matching greeks()'s per-spot M_min == M_max == S for lookbacks.
        M_min = 0.0
        M_max = 0.0
        is_knock_in = False
        is_up = True
        rebate_val = 0.0
        payout_val = 1.0
        extra1_val = 0.0

        if isinstance(instrument, BarrierOption):
            opt_type = BARRIER
            K = instrument.strike
            H = instrument.barrier
            is_call_flag = instrument.is_call
            is_knock_in = instrument.is_knock_in
            is_up = instrument.is_up
            rebate_val = instrument.rebate
        elif isinstance(instrument, AsianOption):
            opt_type = ASIAN_GEO
            K = instrument.strike
            is_call_flag = instrument.is_call
        elif isinstance(instrument, DigitalOption):
            opt_type = DIGITAL
            K = instrument.strike
            is_call_flag = instrument.is_call
            payout_val = instrument.payout
        elif isinstance(instrument, LookbackOption):
            if instrument.lookback_type == "fixed":
                opt_type = LOOKBACK_FIXED
                K = instrument.strike
            else:
                opt_type = LOOKBACK_FLOATING
                K = 0.0
            is_call_flag = instrument.is_call
        elif isinstance(instrument, ChooserOption):
            opt_type = CHOOSER
            K = instrument.strike
            is_call_flag = True  # not used by the chooser kernel
            extra1_val = instrument.choice_time
        elif isinstance(instrument, AssetOrNothingOption):
            opt_type = ASSET_OR_NOTHING
            K = instrument.strike
            is_call_flag = instrument.is_call
        elif isinstance(instrument, PowerOption):
            opt_type = POWER
            K = instrument.strike
            is_call_flag = instrument.is_call
            extra1_val = instrument.power
        elif isinstance(instrument, GapOption):
            opt_type = GAP
            K = instrument.strike
            is_call_flag = instrument.is_call
            extra1_val = instrument.trigger
        else:
            return None

        matrix = exotic_greeks_surface(
            opt_type,
            spots,
            K,
            T,
            r,
            q,
            sigma,
            is_call_flag,
            H,
            M_min,
            M_max,
            is_knock_in,
            is_up,
            rebate_val,
            payout_val,
            extra1_val,
        )
        return {
            "price": matrix[:, 0],
            "delta": matrix[:, 1],
            "gamma": matrix[:, 2],
            "vega": matrix[:, 3],
            "theta": matrix[:, 4],
            "rho": matrix[:, 5],
        }

    def greeks(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
    ) -> GreeksResult:
        """
        Compute Greeks for exotic options via finite differences.

        Parameters
        ----------
        instrument : Instrument
            Exotic option
        model : GBMModel
            GBM model
        market : MarketEnvironment
            Current market conditions

        Returns
        -------
        GreeksResult
            First-order Greeks (delta, gamma, vega, theta, rho)
        """
        if not self.can_price(instrument, model):
            raise ValueError(
                f"ExoticAnalyticEngine cannot compute Greeks for "
                f"{type(instrument).__name__} with {type(model).__name__}"
            )

        # Non-GBM models compute Greeks through the model-dependent numerical
        # route (CRN bump-and-reprice for MC, deterministic FD for Fourier).
        if not isinstance(model, GBMModel):
            return self._numerical_engine(instrument, model).greeks(
                instrument, model, market
            )

        S: float = market.spot
        r: float = market.rate
        q: float = market.dividend_yield
        sigma: float = model.sigma
        T: float = instrument.maturity

        # Registry-dispatched (Haug catalog) exotics: Open-Closed fast path.
        spec = lookup(instrument)
        if spec is not None:
            _p, delta, gamma, vega, theta, rho = fd_greeks(
                spec, instrument, model, market
            )
            return GreeksResult(
                delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho
            )

        # Determine option type and extract parameters
        opt_type: int
        K: float
        H: float
        M_min: float
        M_max: float
        is_call_flag: bool
        is_knock_in: bool
        is_up: bool
        rebate_val: float
        payout_val: float
        extra1_val: float = 0.0

        if isinstance(instrument, BarrierOption):
            opt_type = BARRIER
            K = instrument.strike
            H = instrument.barrier
            M_min = 0.0
            M_max = 0.0
            is_call_flag = instrument.is_call
            is_knock_in = instrument.is_knock_in
            is_up = instrument.is_up
            rebate_val = instrument.rebate
            payout_val = 1.0
        elif isinstance(instrument, AsianOption):
            opt_type = ASIAN_GEO
            K = instrument.strike
            H = 0.0
            M_min = 0.0
            M_max = 0.0
            is_call_flag = instrument.is_call
            is_knock_in = False
            is_up = True
            rebate_val = 0.0
            payout_val = 1.0
        elif isinstance(instrument, DigitalOption):
            opt_type = DIGITAL
            K = instrument.strike
            H = 0.0
            M_min = 0.0
            M_max = 0.0
            is_call_flag = instrument.is_call
            is_knock_in = False
            is_up = True
            rebate_val = 0.0
            payout_val = instrument.payout
        elif isinstance(instrument, LookbackOption):
            if instrument.lookback_type == "fixed":
                opt_type = LOOKBACK_FIXED
                K = instrument.strike
            else:
                opt_type = LOOKBACK_FLOATING
                K = 0.0
            H = 0.0
            M_min = S
            M_max = S
            is_call_flag = instrument.is_call
            is_knock_in = False
            is_up = True
            rebate_val = 0.0
            payout_val = 1.0
        elif isinstance(instrument, ChooserOption):
            opt_type = CHOOSER
            K = instrument.strike
            H = 0.0
            M_min = 0.0
            M_max = 0.0
            is_call_flag = True  # not used by chooser kernel
            is_knock_in = False
            is_up = True
            rebate_val = 0.0
            payout_val = 1.0
            extra1_val = instrument.choice_time
        elif isinstance(instrument, AssetOrNothingOption):
            opt_type = ASSET_OR_NOTHING
            K = instrument.strike
            H = 0.0
            M_min = 0.0
            M_max = 0.0
            is_call_flag = instrument.is_call
            is_knock_in = False
            is_up = True
            rebate_val = 0.0
            payout_val = 1.0
        elif isinstance(instrument, PowerOption):
            opt_type = POWER
            K = instrument.strike
            H = 0.0
            M_min = 0.0
            M_max = 0.0
            is_call_flag = instrument.is_call
            is_knock_in = False
            is_up = True
            rebate_val = 0.0
            payout_val = 1.0
            extra1_val = instrument.power
        elif isinstance(instrument, GapOption):
            opt_type = GAP
            K = instrument.strike
            H = 0.0
            M_min = 0.0
            M_max = 0.0
            is_call_flag = instrument.is_call
            is_knock_in = False
            is_up = True
            rebate_val = 0.0
            payout_val = 1.0
            extra1_val = instrument.trigger
        else:
            raise ValueError(f"Unsupported instrument: {type(instrument).__name__}")

        price, delta, gamma, vega, theta, rho = exotic_calculate_greeks(
            option_type=opt_type,
            S=S,
            K=K,
            T=T,
            r=r,
            q=q,
            sigma=sigma,
            is_call=is_call_flag,
            H=H,
            M_min=M_min,
            M_max=M_max,
            is_knock_in=is_knock_in,
            is_up=is_up,
            rebate=rebate_val,
            payout=payout_val,
            extra1=extra1_val,
        )

        return GreeksResult(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
        )
