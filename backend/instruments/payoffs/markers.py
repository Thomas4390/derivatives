"""
Non-callable marker payoff for analytical-only instruments.
"""

from __future__ import annotations

import numpy as np

from backend.core.interfaces import Payoff


class AnalyticalOnlyPayoff(Payoff):
    """
    Marker payoff for instruments that only support analytical pricing.

    Used by exotic options whose payoff has no Monte Carlo implementation
    (e.g., geometric Asian, fixed-strike lookback, knock-in barriers, chooser,
    asset-or-nothing, power, gap). Calling it raises ``NotImplementedError``
    with a precise diagnostic message instead of letting downstream code
    fail with ``TypeError: 'NoneType' object is not callable``.
    """

    _instrument_type: str
    _reason: str

    def __init__(
        self,
        instrument_type: str,
        reason: str = "no Monte Carlo payoff implemented; use the analytical engine",
    ) -> None:
        self._instrument_type = instrument_type
        self._reason = reason

    @property
    def instrument_type(self) -> str:
        return self._instrument_type

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def is_path_dependent(self) -> bool:
        return False

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{self._instrument_type} has no callable payoff: {self._reason}."
        )

    def __repr__(self) -> str:
        return f"AnalyticalOnlyPayoff({self._instrument_type!r})"
