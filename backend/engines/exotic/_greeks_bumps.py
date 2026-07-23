"""
Volatility-bump helper for model-dependent exotic Greeks (vega).

Vega bumps the model's volatility *level*. Spot/rate/maturity bumps are
model-agnostic (they live on ``MarketEnvironment`` / the simulation horizon), but
"the vol" is model-specific, so this helper returns a fresh model with the vol
nudged by ``dv`` (vol points), or ``None`` when the model has no scalar vol level
(e.g. GARCH variants — their conditional variance is a path, so a single vega is
ill-defined and the caller reports vega = 0).

- GBM / Merton: bump the diffusion ``sigma`` directly.
- Heston / Bates: bump the vol level ``sqrt(v0)`` and square back to ``v0`` so a
  ``dv`` move in vol maps to the initial variance.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np

from backend.core.interfaces import Model
from backend.utils.constants.exotic import FD_SIGMA_FLOOR


def bump_model_vol(model: Model, dv: float) -> Model | None:
    """Return ``model`` with its volatility level moved by ``dv``, or ``None``.

    Parameters
    ----------
    model : Model
        A frozen-dataclass model.
    dv : float
        Volatility bump in vol points (may be negative).

    Returns
    -------
    Model or None
        Bumped model, or ``None`` if the model exposes no scalar vol level.
    """
    # ``Model`` is an abstract base, not a recognised dataclass type, so the
    # concrete frozen-dataclass instance is bumped via ``replace`` through ``Any``.
    dc = cast(Any, model)
    # GBM / Merton: scalar diffusion sigma.
    if hasattr(model, "sigma"):
        new_sigma = max(float(dc.sigma) + dv, FD_SIGMA_FLOOR)
        try:
            return cast(Model, replace(dc, sigma=new_sigma))
        except (TypeError, ValueError):
            return None
    # Heston / Bates: bump the vol level sqrt(v0), square back to v0.
    if hasattr(model, "v0"):
        vol = max(float(np.sqrt(dc.v0)) + dv, FD_SIGMA_FLOOR)
        try:
            return cast(Model, replace(dc, v0=vol * vol))
        except (TypeError, ValueError):
            return None
    return None
