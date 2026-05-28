"""
Objective Strategies for Calibration
======================================

Strategy Pattern for the *loss function*. Each strategy encapsulates a
definition of "how close is the model to the market" that drives both
the LM residual vector and the scalar reporting metric.

Mirrors the API of :mod:`backend.calibration.optimizers`.

Available strategies
--------------------

============== =============== ============== ============= =================
Name           Type            Needs          JAX-friendly  Best for
============== =============== ============== ============= =================
price_mse      Baseline LSQ    nothing        ✓             Generic
iv_mse         Direct LSQ      IV inversion   ✗             Pedagogy
vega_weighted  Linear weights  market IVs     ✓             Industry default
spread_weighted Linear weights bid/ask        ✓             Liquidity-aware
relative       Scale-invariant nothing        ✓             Wide maturity range
huber          Robust M-est    δ threshold    ✓ (IRLS)      Outliers / crypto
============== =============== ============== ============= =================

Quick example
-------------

>>> from backend.calibration.objectives import make_objective
>>> obj = make_objective("vega_weighted")
>>> r = obj.compute_residuals(model_prices, market_data)
>>> rmse = obj.compute_loss(model_prices, market_data)

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import Any

from backend.calibration.objectives.base import (
    JaxResidualFn,
    ObjectiveMetadata,
    ObjectiveStrategy,
)
from backend.calibration.objectives.huber import HuberObjective
from backend.calibration.objectives.iv_mse import IVMSEObjective
from backend.calibration.objectives.price_mse import PriceMSEObjective
from backend.calibration.objectives.relative import RelativeMSEObjective
from backend.calibration.objectives.spread_weighted import SpreadWeightedObjective
from backend.calibration.objectives.vega_weighted import VegaWeightedObjective

DEFAULT_OBJECTIVES: dict[str, type[ObjectiveStrategy]] = {
    "price_mse": PriceMSEObjective,
    "iv_mse": IVMSEObjective,
    "vega_weighted": VegaWeightedObjective,
    "spread_weighted": SpreadWeightedObjective,
    "relative": RelativeMSEObjective,
    "huber": HuberObjective,
}

# Legacy aliases — accepted by ``make_objective`` for backward compatibility
# with the previous ``objective_type`` string interface on calibrators.
LEGACY_OBJECTIVE_ALIASES: dict[str, str] = {
    "price_rmse": "price_mse",
    "price_weighted": "vega_weighted",
    "iv_rmse": "iv_mse",
}


def make_objective(name: str, **kwargs: Any) -> ObjectiveStrategy:
    """Factory : build an objective strategy by canonical or legacy name.

    Parameters
    ----------
    name : str
        Canonical name (``"price_mse"``, …) or legacy alias
        (``"price_rmse"``).
    **kwargs
        Strategy-specific keyword arguments. ``HuberObjective`` accepts
        ``delta``, ``RelativeMSEObjective`` accepts ``use_log`` and
        ``eps``, etc.

    Raises
    ------
    KeyError
        If ``name`` is neither canonical nor a legacy alias.
    """
    canonical = LEGACY_OBJECTIVE_ALIASES.get(name, name)
    try:
        cls = DEFAULT_OBJECTIVES[canonical]
    except KeyError as exc:
        raise KeyError(
            f"Unknown objective '{name}'. "
            f"Available: {list(DEFAULT_OBJECTIVES)} "
            f"(legacy aliases: {list(LEGACY_OBJECTIVE_ALIASES)})"
        ) from exc
    return cls(**kwargs)


__all__ = [
    # Contracts
    "JaxResidualFn",
    "ObjectiveMetadata",
    "ObjectiveStrategy",
    # Concrete strategies
    "PriceMSEObjective",
    "IVMSEObjective",
    "VegaWeightedObjective",
    "SpreadWeightedObjective",
    "RelativeMSEObjective",
    "HuberObjective",
    # Factory + registry
    "DEFAULT_OBJECTIVES",
    "LEGACY_OBJECTIVE_ALIASES",
    "make_objective",
]
