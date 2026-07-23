"""
Model/engine-architecture-aware finite-difference Greeks: bump-and-reprice
through ``engine.price(instrument, model, market)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.greeks.numerical._guards import (
    _check_finite,
    _require_positive_spot,
    _safe_div,
)
from backend.greeks.numerical.config import (
    DEFAULT_BUMP_CONFIG,
    GreeksBumpConfig,
    NumericalGreeks,
)
from backend.utils.math import DAYS_PER_YEAR

if TYPE_CHECKING:
    from backend.core.interfaces import Instrument, Model, PricingEngine
    from backend.core.market import MarketEnvironment


class ModelNumericalGreeks:
    """
    Numerical Greeks calculator that works with model/engine architecture.

    Provides finite difference Greeks for any pricing engine.

    Parameters
    ----------
    config : GreeksBumpConfig, optional
        Bump configuration. If not provided, uses DEFAULT_BUMP_CONFIG.
    """

    config: GreeksBumpConfig

    def __init__(self, config: GreeksBumpConfig | None = None) -> None:
        """
        Initialize numerical Greeks calculator.

        Parameters
        ----------
        config : GreeksBumpConfig, optional
            Bump configuration. If not provided, uses DEFAULT_BUMP_CONFIG.
        """
        self.config = config or DEFAULT_BUMP_CONFIG

    @property
    def spot_bump(self) -> float:
        """Relative spot bump."""
        return self.config.spot_bump

    @property
    def vol_bump(self) -> float:
        """Absolute volatility bump."""
        return self.config.vol_bump

    @property
    def time_bump_days(self) -> float:
        """Time decay bump in days."""
        return self.config.time_bump_days

    @property
    def rate_bump(self) -> float:
        """Absolute rate bump."""
        return self.config.rate_bump

    def calculate(
        self,
        engine: PricingEngine,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
    ) -> NumericalGreeks:
        """
        Calculate numerical Greeks for an instrument.

        Parameters
        ----------
        engine : PricingEngine
            Pricing engine
        instrument : Instrument
            Option instrument
        model : Model
            Pricing model
        market : MarketEnvironment
            Market conditions

        Returns
        -------
        NumericalGreeks
            Numerical Greeks
        """
        _require_positive_spot(market.spot)

        # Base price
        v_mid: float = _check_finite(
            engine.price(instrument, model, market).price, label="v_mid"
        )

        # Spot bumps
        h_s: float = market.spot * self.spot_bump
        market_s_up = market.bump_spot(h_s)
        market_s_down = market.bump_spot(-h_s)
        v_s_up: float = _check_finite(
            engine.price(instrument, model, market_s_up).price, label="v_spot_up"
        )
        v_s_down: float = _check_finite(
            engine.price(instrument, model, market_s_down).price,
            label="v_spot_down",
        )

        # Rate bumps
        h_r: float = self.rate_bump
        market_r_up = market.bump_rate(h_r)
        market_r_down = market.bump_rate(-h_r)
        v_r_up: float = _check_finite(
            engine.price(instrument, model, market_r_up).price, label="v_rate_up"
        )
        v_r_down: float = _check_finite(
            engine.price(instrument, model, market_r_down).price,
            label="v_rate_down",
        )

        # Time bump (requires instrument modification)
        h_t: float = self.time_bump_days / DAYS_PER_YEAR
        from backend.greeks._instrument_utils import create_decayed_instrument

        new_T: float = max(instrument.maturity - h_t, 0.001)
        decayed = create_decayed_instrument(instrument, new_T)
        if decayed is not None:
            v_t_bump: float = _check_finite(
                engine.price(decayed, model, market).price, label="v_time_bump"
            )
        else:
            # Unsupported instrument type - skip theta
            v_t_bump = v_mid  # This will result in theta = 0

        # Vol bump (requires model modification)
        from backend.models.vol_bump import create_vol_bumped_pair

        model_v_up, model_v_down = create_vol_bumped_pair(model, self.vol_bump)
        vega: float
        if model_v_up is not None and model_v_down is not None:
            v_v_up: float = _check_finite(
                engine.price(instrument, model_v_up, market).price, label="v_vol_up"
            )
            v_v_down: float = _check_finite(
                engine.price(instrument, model_v_down, market).price,
                label="v_vol_down",
            )
            vega = _safe_div(v_v_up - v_v_down, 2 * self.vol_bump, label="vega") / 100.0
        else:
            vega = 0.0

        # Calculate Greeks (safe-div guards against pathological zero bumps)
        delta: float = _safe_div(v_s_up - v_s_down, 2 * h_s, label="delta")
        gamma: float = _safe_div(v_s_up - 2 * v_mid + v_s_down, h_s**2, label="gamma")
        theta: float = _safe_div(v_t_bump - v_mid, self.time_bump_days, label="theta")
        rho: float = _safe_div(v_r_up - v_r_down, 2 * h_r, label="rho") / 100.0

        return NumericalGreeks(
            delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho
        )
