"""IV-surface x-axis display resolver.

The synthetic surface grid is generated on σ√T-standardized moneyness so it
fills without NaN holes for any vol level (see ``synthetic_data_service``).
That axis is, however, non-standard to read. This module re-expresses the
*already priced* quotes in a more familiar unit at plot time — log-moneyness
``ln(K/F)`` (the academic default), ``K/F``, the dollar strike, or the native
σ√T axis — **without regenerating the surface**.

Because the synthetic strikes differ per maturity (and forwards always do),
the standard axes are intrinsically 2D ``(n_T, n_K)``; the σ√T axis is the only
shared-1D one. ``_collapse`` returns a 1D vector only when every maturity's row
is numerically identical (e.g. real data with a single shared strike ladder),
keeping the cheap 1D path where it is genuinely available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config.constants import DEFAULT_X_AXIS
from services import state_manager


@dataclass(frozen=True)
class AxisSpec:
    """Resolved plot-axis specification for the surface/smile charts.

    ``x`` is 1D ``(n_K,)`` when the axis is shared across maturities, else 2D
    ``(n_T, n_K)``. ``hover_label``/``hover_fmt`` are ``None`` for the σ√T
    passthrough, in which case the charts derive them from ``atm_x`` (preserving
    the legacy ``m`` / ``K/S₀`` behaviour).
    """

    x: np.ndarray
    x_label: str
    atm_x: float
    hover_label: str | None
    hover_fmt: str | None
    tickformat: str

    def kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the surface/smile render functions."""
        return {
            "moneyness": self.x,
            "x_label": self.x_label,
            "atm_x": self.atm_x,
            "hover_label": self.hover_label,
            "hover_fmt": self.hover_fmt,
            "tickformat": self.tickformat,
        }

    def heatmap_kwargs(self, meta: dict) -> dict[str, Any]:
        """1D-safe kwargs for the residual heatmap.

        A heatmap needs a single shared x per column. When the resolved axis is
        2D (synthetic standard units), fall back to the σ√T-moneyness stored in
        ``meta`` — the residual grid is naturally indexed by those bins.
        """
        if np.ndim(self.x) == 1:
            return {
                "moneyness": self.x,
                "x_label": self.x_label,
                "atm_x": self.atm_x,
                "hover_label": self.hover_label,
                "hover_fmt": self.hover_fmt,
            }
        return {
            "moneyness": np.asarray(meta["moneyness"], dtype=float),
            "x_label": meta.get("x_label", "Moneyness  ln(K/F) / (σ√T)"),
            "atm_x": float(meta.get("atm_x", 0.0)),
            "hover_label": None,
            "hover_fmt": None,
        }


def _collapse(x: np.ndarray) -> np.ndarray:
    """Return a 1D axis if every maturity row is identical, else the 2D grid."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 2 and x.shape[0] > 0 and np.allclose(x, x[0:1], atol=1e-12, rtol=0.0):
        return x[0]
    return x


def resolve_display_axis(meta: dict) -> AxisSpec:
    """Build the display ``AxisSpec`` from a market-data ``meta`` + session choice.

    Falls back to the σ√T-moneyness stored in ``meta`` whenever the chosen
    transform cannot be computed (missing strikes / non-surface meta).
    """
    choice = state_manager.get("calib_x_axis") or DEFAULT_X_AXIS

    sigma_axis = AxisSpec(
        x=np.asarray(meta.get("moneyness", []), dtype=float),
        x_label=meta.get("x_label", "Moneyness  ln(K/F) / (σ√T)"),
        atm_x=float(meta.get("atm_x", 0.0)),
        hover_label=None,
        hover_fmt=None,
        tickformat=".2f",
    )
    if choice == "moneyness_sigma" or "strikes" not in meta:
        return sigma_axis

    strikes = np.asarray(meta["strikes"], dtype=float)  # (n_T, n_K) dollar strikes
    if strikes.ndim != 2:
        return sigma_axis
    maturities = np.asarray(meta["maturities"], dtype=float)
    spot = float(meta.get("spot", 100.0))
    rate = float(meta.get("rate", 0.0))
    q = float(meta.get("dividend_yield", 0.0))
    forwards = spot * np.exp((rate - q) * maturities)  # (n_T,)

    if choice == "strike":
        return AxisSpec(
            x=_collapse(strikes),
            x_label="Strike  K ($)",
            atm_x=spot,
            hover_label="K",
            hover_fmt=".0f",
            tickformat=".0f",
        )
    if choice == "k_over_f":
        return AxisSpec(
            x=_collapse(strikes / forwards[:, None]),
            x_label="Moneyness  K / F",
            atm_x=1.0,
            hover_label="K/F",
            hover_fmt=".3f",
            tickformat=".2f",
        )
    # Default / "log_moneyness": ln(K/F), ATM at 0 for every maturity.
    return AxisSpec(
        x=_collapse(np.log(strikes / forwards[:, None])),
        x_label="Log-moneyness  ln(K/F)",
        atm_x=0.0,
        hover_label="ln(K/F)",
        hover_fmt=".3f",
        tickformat=".2f",
    )
