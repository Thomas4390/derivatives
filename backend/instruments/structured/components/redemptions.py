"""
Redemption components — capital protection / bond floors.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.core.structured_product import ProductComponent


# =============================================================================
# Bond Floor
# =============================================================================


@dataclass(frozen=True)
class BondFloor(ProductComponent):
    """
    Capital protection component.

    Pays `protection_level * notional` at maturity, discounted to PV.

    Parameters
    ----------
    protection_level : float
        Fraction of notional protected (e.g., 1.0 = 100%, 0.9 = 90%).
    notional : float
        Product notional.
    maturity : float
        Product maturity in years.
    """

    protection_level: float
    notional: float
    maturity: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.protection_level <= 1.0:
            raise ValueError(
                f"protection_level must be in [0, 1], got {self.protection_level}"
            )

    @property
    def name(self) -> str:
        return f"BondFloor({self.protection_level:.0%})"

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
    ) -> np.ndarray:
        n_paths = paths.shape[0]
        # Terminal discount factor is the last one
        df_terminal = discount_factors[-1]
        pv = self.protection_level * self.notional * df_terminal
        return np.full(n_paths, pv)


__all__ = ["BondFloor"]
