"""
Exotic Option Instruments
=========================

Immutable exotic option classes and their factory functions, split by family
(asian, barrier, chooser, asset_or_nothing, power, gap, lookback). This package
re-exports every public symbol so ``from backend.instruments.exotic_options
import X`` resolves identically.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from backend.instruments.exotic_options.asian import (  # noqa: F401
    AsianCall,
    AsianGeometricCall,
    AsianGeometricPut,
    AsianOption,
    AsianPut,
)
from backend.instruments.exotic_options.asset_or_nothing import (  # noqa: F401
    AssetOrNothingCall,
    AssetOrNothingOption,
    AssetOrNothingPut,
)
from backend.instruments.exotic_options.barrier import (  # noqa: F401
    BarrierDownInCall,
    BarrierDownInPut,
    BarrierDownOutCall,
    BarrierDownOutPut,
    BarrierOption,
    BarrierUpInCall,
    BarrierUpInPut,
    BarrierUpOutCall,
    BarrierUpOutPut,
)
from backend.instruments.exotic_options.chooser import (  # noqa: F401
    Chooser,
    ChooserOption,
)
from backend.instruments.exotic_options.gap import (  # noqa: F401
    GapCall,
    GapOption,
    GapPut,
)
from backend.instruments.exotic_options.lookback import (  # noqa: F401
    LookbackCall,
    LookbackFixedCall,
    LookbackFixedPut,
    LookbackOption,
    LookbackPut,
)
from backend.instruments.exotic_options.power import (  # noqa: F401
    PowerCall,
    PowerOption,
    PowerPut,
)

__all__ = [
    "AsianOption",
    "BarrierOption",
    "ChooserOption",
    "AssetOrNothingOption",
    "PowerOption",
    "GapOption",
    "LookbackOption",
    "AsianCall",
    "AsianPut",
    "AsianGeometricCall",
    "AsianGeometricPut",
    "BarrierUpOutCall",
    "BarrierUpInCall",
    "BarrierDownOutCall",
    "BarrierDownInCall",
    "BarrierUpOutPut",
    "BarrierUpInPut",
    "BarrierDownOutPut",
    "BarrierDownInPut",
    "LookbackCall",
    "LookbackPut",
    "LookbackFixedCall",
    "LookbackFixedPut",
    "Chooser",
    "AssetOrNothingCall",
    "AssetOrNothingPut",
    "PowerCall",
    "PowerPut",
    "GapCall",
    "GapPut",
]
