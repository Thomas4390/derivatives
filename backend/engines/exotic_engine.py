"""
Backward-compatibility shim -- use backend.engines.exotic instead.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

# Re-export the engine class
from backend.engines.exotic import ExoticAnalyticEngine  # noqa: F401

# Re-export all pricing kernels (tests and streamlit import these directly)
from backend.engines.exotic.asian import asian_geometric_price  # noqa: F401
from backend.engines.exotic.asset_or_nothing import asset_or_nothing_price  # noqa: F401
from backend.engines.exotic.barrier import (  # noqa: F401
    _bs_vanilla_price,
    barrier_option_price,
)
from backend.engines.exotic.chooser import chooser_price  # noqa: F401
from backend.engines.exotic.digital import digital_price  # noqa: F401
from backend.engines.exotic.engine import (  # noqa: F401
    ASIAN_GEO,
    ASSET_OR_NOTHING,
    BARRIER,
    CHOOSER,
    DIGITAL,
    GAP,
    LOOKBACK_FIXED,
    LOOKBACK_FLOATING,
    POWER,
    _exotic_price,
    exotic_calculate_greeks,
    exotic_greeks_batch,
    exotic_greeks_surface,
    exotic_price_param_sweep,
    exotic_price_surface,
)
from backend.engines.exotic.gap import gap_option_price  # noqa: F401
from backend.engines.exotic.lookback import (  # noqa: F401
    lookback_fixed_price,
    lookback_floating_price,
)
from backend.engines.exotic.power import power_option_price  # noqa: F401
