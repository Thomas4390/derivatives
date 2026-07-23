"""
Calibration Base Classes
========================

Abstract calibrator interface and result dataclass.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from backend.core.interfaces import Model

if TYPE_CHECKING:
    from backend.calibration.objectives import ObjectiveStrategy


@dataclass(frozen=True)
class CalibrationResult:
    """Immutable result of a model calibration."""

    model: Model
    objective_value: float
    n_iterations: int
    success: bool
    method: str
    rmse_price: float | None = None
    rmse_iv: float | None = None
    elapsed_seconds: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    iteration_history: tuple = ()  # tuple[IterationSnapshot, ...] when log_iterations=True
    optimizer_name: str | None = None  # name of the OptimizerStrategy used

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAILED"
        parts = [f"CalibrationResult({status}, model={self.model.name}"]
        if self.rmse_price is not None:
            parts.append(f"rmse_price={self.rmse_price:.6f}")
        if self.rmse_iv is not None:
            parts.append(f"rmse_iv={self.rmse_iv:.2f}bp")
        parts.append(f"iter={self.n_iterations}")
        parts.append(f"time={self.elapsed_seconds:.2f}s")
        return ", ".join(parts) + ")"


@runtime_checkable
class Calibrator(Protocol):
    """Any calibrator -- scipy-based or JAX differentiable."""

    def calibrate(self, market_data: Any) -> Any: ...


class BaseCalibrator(ABC):
    """Abstract base class for all model calibrators."""

    @abstractmethod
    def calibrate(self, market_data: Any) -> CalibrationResult:
        """Calibrate model to market data."""
        ...

    @abstractmethod
    def objective(self, params: np.ndarray, market_data: Any) -> float:
        """Objective function to minimize."""
        ...

    @abstractmethod
    def default_bounds(self) -> list[tuple[float, float]]:
        """Default parameter bounds for optimization."""
        ...

    # --------------------------------------------------------------------- #
    # Optional: residuals() for Levenberg-Marquardt (scipy.optimize.least_squares)
    # --------------------------------------------------------------------- #

    def residuals(self, params: np.ndarray, market_data: Any) -> np.ndarray:
        """Return the per-observation residual vector (model - market).

        Optional hook for gradient-based least-squares optimizers
        (e.g. ``scipy.optimize.least_squares`` with the Trust-Region-
        Reflective method). The scalar objective is the half squared
        norm of this vector, so implementations can delegate:

            def objective(self, params, market_data):
                r = self.residuals(params, market_data)
                return 0.5 * float(r @ r)

        Subclasses that want Levenberg-Marquardt access should override.
        The default raises ``NotImplementedError`` so that optimizers
        relying on residuals fail loudly rather than silently falling
        back to a scalar-only path.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a residual vector; "
            "override residuals() to enable Levenberg-Marquardt."
        )

    # --------------------------------------------------------------------- #
    # Objective resolution (JAX-compatibility fallback)
    # --------------------------------------------------------------------- #

    def _resolve_objective(self, obj: "ObjectiveStrategy") -> "ObjectiveStrategy":
        """Coerce a JAX-incompatible objective into a tractable fallback.

        ``LM-JAX`` is the production solver and traces residuals through
        ``jax.jacfwd``; objectives requiring scipy IV-inversion
        (``iv_mse``) cannot be JIT-compiled and fall back to
        ``vega_weighted`` — the first-order Taylor expansion of the IV
        residuals around the market vega (Cont & Tankov 2004). The notice
        is logged at INFO via the concrete calibrator's own module logger
        so it surfaces in the Streamlit panel without being noisy.
        """
        if obj.jax_compatible:
            return obj
        from backend.calibration.objectives import VegaWeightedObjective

        logging.getLogger(type(self).__module__).info(
            "%s: objective '%s' is not JAX-compatible. Falling back to "
            "'vega_weighted' (first-order IV approximation, Cont & Tankov 2004).",
            type(self).__name__,
            obj.name,
        )
        return VegaWeightedObjective()

    # --------------------------------------------------------------------- #
    # Shared objective computation
    # --------------------------------------------------------------------- #

    @staticmethod
    def _compute_objective_value_static(
        objective_type: str,
        model_prices: np.ndarray,
        market_prices: np.ndarray,
        market_data: Any,
    ) -> float:
        """Compute the raw objective value (before any penalty/regularization).

        Parameters
        ----------
        objective_type : str
            One of ``"price_rmse"``, ``"iv_rmse"``, ``"price_weighted"``.
        model_prices : np.ndarray
            Model-computed option prices.
        market_prices : np.ndarray
            Observed market option prices.
        market_data : OptionMarketData
            Market data containing quotes, strikes, maturities, etc.

        Returns
        -------
        float
            Scalar objective value (lower is better).
        """
        # Lazy imports to avoid circular dependencies
        from backend.calibration.utils import (
            compute_rmse_iv,
            compute_rmse_price,
            model_prices_to_ivs,
            vega_weights,
        )

        if objective_type == "price_rmse":
            return compute_rmse_price(model_prices, market_prices)

        if objective_type == "price_weighted":
            # Vega-weighted RMSE: down-weight deep OTM/ITM quotes
            strikes = market_data.strikes
            maturities = market_data.maturities

            # Need market IVs for vega computation
            market_ivs = np.array(
                [
                    q.implied_vol if q.implied_vol is not None else 0.2
                    for q in market_data.quotes
                ]
            )
            weights = vega_weights(
                spot=market_data.spot,
                strikes=strikes,
                maturities=maturities,
                rate=market_data.rate,
                ivs=market_ivs,
                dividend_yield=market_data.dividend_yield,
            )
            weighted_sq_errors = weights * (model_prices - market_prices) ** 2
            return float(np.sqrt(np.mean(weighted_sq_errors)))

        # objective_type == "iv_rmse"
        strikes = market_data.strikes
        maturities = market_data.maturities
        is_calls = np.array([q.is_call for q in market_data.quotes])

        model_ivs = model_prices_to_ivs(
            model_prices=model_prices,
            spot=market_data.spot,
            strikes=strikes,
            maturities=maturities,
            rate=market_data.rate,
            is_calls=is_calls,
            dividend_yield=market_data.dividend_yield,
        )
        market_ivs = np.array(
            [
                q.implied_vol if q.implied_vol is not None else np.nan
                for q in market_data.quotes
            ]
        )

        # If any IV inversion failed, fall back to a large value
        valid = ~np.isnan(model_ivs) & ~np.isnan(market_ivs)
        if valid.sum() == 0:
            return 1e10

        return compute_rmse_iv(model_ivs[valid], market_ivs[valid])

    def _compute_objective_value(
        self,
        model_prices: np.ndarray,
        market_prices: np.ndarray,
        market_data: Any,
    ) -> float:
        """Compute the raw objective value using this calibrator's objective_type."""
        return self._compute_objective_value_static(
            self.objective_type, model_prices, market_prices, market_data
        )
