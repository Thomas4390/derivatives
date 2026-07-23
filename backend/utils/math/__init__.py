"""
Mathematical Utilities
======================

Shared mathematical primitives used across the backend (Numba-optimized).

This package is the SINGLE SOURCE OF TRUTH for:
- Normal distribution functions (CDF, PDF, inverse-CDF, bivariate)
- Black-Scholes d1/d2 parameters and pricing
- First/Second/Third-order Greeks
- Implied-volatility inversion
- Discount/forward and moneyness conversions

IMPORTANT: Do NOT duplicate these formulas elsewhere. Import from here; the
implementation lives in the cohesive sub-modules and is re-exported below so
``from backend.utils.math import X`` resolves identically for every X.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from backend.utils.constants.numerical import (  # noqa: F401
    IV_SIGMA_MIN,
    IV_VEGA_FLOOR,
    VOLATILITY_MAX,
)
from backend.utils.constants.time import DAYS_PER_YEAR  # noqa: F401
from backend.utils.math._constants import SQRT_2, SQRT_2PI  # noqa: F401
from backend.utils.math.bivariate import cbnd  # noqa: F401
from backend.utils.math.black_scholes import bs_price, d1_d2  # noqa: F401
from backend.utils.math.conversions import (  # noqa: F401
    delta_to_strike,
    discount_factor,
    forward_log_moneyness,
    forward_price,
    log_moneyness,
)
from backend.utils.math.distributions import (  # noqa: F401
    norm_cdf,
    norm_cdf_vec,
    norm_inv_cdf,
    norm_pdf,
    norm_pdf_vec,
)
from backend.utils.math.greeks_first_order import (  # noqa: F401
    bs_delta,
    bs_gamma,
    bs_greeks,
    bs_rho,
    bs_theta,
    bs_vega,
)
from backend.utils.math.greeks_higher_order import (  # noqa: F401
    bs_second_order_greeks,
    bs_third_order_greeks,
)
from backend.utils.math.implied_vol import implied_volatility  # noqa: F401

__all__ = [
    "SQRT_2",
    "SQRT_2PI",
    "DAYS_PER_YEAR",
    "IV_SIGMA_MIN",
    "IV_VEGA_FLOOR",
    "VOLATILITY_MAX",
    "norm_cdf",
    "norm_pdf",
    "norm_inv_cdf",
    "norm_cdf_vec",
    "norm_pdf_vec",
    "cbnd",
    "d1_d2",
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "bs_theta",
    "bs_rho",
    "bs_greeks",
    "bs_second_order_greeks",
    "bs_third_order_greeks",
    "implied_volatility",
    "discount_factor",
    "forward_price",
    "log_moneyness",
    "forward_log_moneyness",
    "delta_to_strike",
]
