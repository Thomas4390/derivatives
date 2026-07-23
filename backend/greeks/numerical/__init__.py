"""
Numerical Greeks
================

Finite-difference Greeks for any pricing engine (models without closed-form
Greeks, analytic-Greek validation, portfolio Greeks).

The implementation lives in cohesive sub-modules (guards, config, first/second/
third-order families, the aggregator, and the model-aware class); this package
re-exports every public symbol so ``from backend.greeks.numerical import X``
resolves identically.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from backend.greeks.numerical._guards import (  # noqa: F401
    _DIV_EPS,
    _check_finite,
    _require_positive_spot,
    _safe_div,
)
from backend.greeks.numerical.aggregate import finite_difference_greeks  # noqa: F401
from backend.greeks.numerical.config import (  # noqa: F401
    DEFAULT_BUMP_CONFIG,
    GreeksBumpConfig,
    NumericalGreeks,
    PricingFunc,
)
from backend.greeks.numerical.first_order import (  # noqa: F401
    finite_difference_delta,
    finite_difference_gamma,
    finite_difference_rho,
    finite_difference_theta,
    finite_difference_vega,
)
from backend.greeks.numerical.model_greeks import ModelNumericalGreeks  # noqa: F401
from backend.greeks.numerical.second_order import (  # noqa: F401
    finite_difference_charm,
    finite_difference_vanna,
    finite_difference_veta,
    finite_difference_volga,
)
from backend.greeks.numerical.third_order import (  # noqa: F401
    finite_difference_color,
    finite_difference_speed,
    finite_difference_ultima,
    finite_difference_zomma,
)

__all__ = [
    "GreeksBumpConfig",
    "DEFAULT_BUMP_CONFIG",
    "NumericalGreeks",
    "PricingFunc",
    "ModelNumericalGreeks",
    "finite_difference_delta",
    "finite_difference_gamma",
    "finite_difference_vega",
    "finite_difference_theta",
    "finite_difference_rho",
    "finite_difference_vanna",
    "finite_difference_volga",
    "finite_difference_charm",
    "finite_difference_veta",
    "finite_difference_speed",
    "finite_difference_zomma",
    "finite_difference_color",
    "finite_difference_ultima",
    "finite_difference_greeks",
]
