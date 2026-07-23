"""
Market-data bundle constructors
=================================

Adapts the raw outputs of the synthetic/real data services into a
single ``MarketDataBundle`` (market_data + UI-friendly metadata) ready
for caching in ``st.session_state`` and consumption by the tabs.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from services.real_data_service import RealMarketSnapshot
from services.synthetic_data_service import ReturnsData, SurfaceData


class InvalidMarketData(ValueError):
    """Raised when a bundle's numerical content contains NaN or Inf.

    Subclasses ``ValueError`` so existing broad catches still cover it
    as a fallback, but the dedicated type lets the UI surface a
    targeted, source-pointing message instead of blaming the solver.
    """


def _assert_finite(meta: dict) -> None:
    """Reject bundles whose numerical content is unusable.

    Sparse ``NaN`` cells in ``iv_grid`` are **legitimate** — Feller
    violations, deep-wing inverter failures, and other pricing
    pathologies show up as isolated NaN cells that the loss function
    filters out per-cell. We therefore only flag:

    - any ``Inf`` cell (always signals an overflow or a corrupted feed),
    - arrays that are 100% non-finite (degenerate — nothing to calibrate),
    - non-finite float scalars (e.g. ``spot = inf`` from a bad config).

    Strings, ``None``, booleans, integers, and the precomputed cache
    hash are skipped.
    """
    bad: list[str] = []
    for key, val in meta.items():
        if val is None or isinstance(val, (str, bool)):
            continue
        if isinstance(val, np.ndarray):
            if val.dtype.kind in "fc" and val.size > 0:
                n_inf = int(np.sum(np.isinf(val)))
                n_finite = int(np.sum(np.isfinite(val)))
                if n_inf > 0:
                    bad.append(f"{key} ({n_inf} Inf cells of {val.size})")
                elif n_finite == 0:
                    bad.append(f"{key} (all {val.size} cells are NaN)")
        elif isinstance(val, (int, np.integer)):
            # Integers cannot be NaN/Inf — skip.
            continue
        elif isinstance(val, (float, np.floating)):
            if not math.isfinite(float(val)):
                bad.append(f"{key} = {val!r}")
    if bad:
        raise InvalidMarketData(
            "Market data is unusable: " + "; ".join(bad)
        )


@dataclass(frozen=True)
class MarketDataBundle:
    market_data: Any
    meta: dict

    def __post_init__(self) -> None:
        _assert_finite(self.meta)


def _content_hash(payload: dict[str, Any]) -> str:
    """Stable MD5 over the salient bundle content.

    Lets the landscape and other expensive caches key on a short
    deterministic string instead of re-pickling the whole IV grid /
    returns array on every cache lookup. Computed once per bundle.
    """
    h = hashlib.md5()
    for key in sorted(payload):
        val = payload[key]
        h.update(str(key).encode())
        if hasattr(val, "tobytes"):
            h.update(np.ascontiguousarray(val).tobytes())
        else:
            h.update(repr(val).encode())
    return h.hexdigest()


def from_surface(sd: SurfaceData, data_config: dict) -> MarketDataBundle:
    meta = {
        "iv_grid": sd.iv_grid,
        "strikes": sd.strikes,  # (n_T, n_K) per-maturity dollar strikes
        "maturities": sd.maturities,
        "moneyness": sd.moneyness,  # shared 1D plot axis (σ√T-standardized)
        "x_label": sd.x_label,
        "atm_x": sd.atm_x,
        "true_model": sd.true_model,
        "spot": data_config["spot"],
        "rate": data_config["rate"],
        "dividend_yield": data_config["dividend_yield"],
    }
    meta["market_data_hash"] = _content_hash({
        "iv_grid": sd.iv_grid,
        "strikes": sd.strikes,
        "maturities": sd.maturities,
        "spot": data_config["spot"],
        "rate": data_config["rate"],
        "dividend_yield": data_config["dividend_yield"],
    })
    return MarketDataBundle(market_data=sd.market_data, meta=meta)


def from_returns(rd: ReturnsData, data_config: dict) -> MarketDataBundle:
    meta = {
        "prices": rd.prices,
        "log_returns": rd.log_returns,
        "sample_volatility_ann": rd.sample_volatility_ann,
        "annualization_factor": data_config["annualization_factor"],
        # Extra context for the model-implied risk-neutral IV surface shown in
        # the returns-family diagnostics (Part A). ``rate`` is the assumed
        # risk-free rate for the risk-neutral pricing (the returns config has no
        # rate, only a physical drift), defaulting to the surface default.
        "spot": float(data_config.get("spot", 100.0)),
        "rate": float(data_config.get("rate", 0.05)),
        "drift": float(data_config.get("drift", 0.05)),
        "true_sigma0": float(data_config.get("true_sigma0", 0.20)),
    }
    meta["market_data_hash"] = _content_hash({
        "log_returns": rd.log_returns,
        "annualization_factor": data_config["annualization_factor"],
    })
    return MarketDataBundle(market_data=rd.market_data, meta=meta)


def from_real_snapshot(snap: RealMarketSnapshot) -> MarketDataBundle:
    # Real quotes share one dollar-strike axis across maturities → tile to the
    # uniform 2D-strikes representation and use K/S₀ as the plot axis (σ√T binning
    # of sparse, irregular real quotes would be lossy — deliberate asymmetry vs the
    # synthetic σ-moneyness surface).
    strikes_1d = np.asarray(snap.strikes, dtype=float)
    strikes_2d = np.tile(strikes_1d, (len(snap.maturities), 1))
    moneyness = strikes_1d / float(snap.spot) if snap.spot else strikes_1d
    meta = {
        "iv_grid": snap.iv_grid,
        "strikes": strikes_2d,
        "maturities": snap.maturities,
        "moneyness": moneyness,
        "x_label": "Moneyness  K / S₀",
        "atm_x": 1.0,
        "true_model": None,
        "real_label": snap.snapshot_label,
        "spot": snap.spot,
        "rate": snap.rate,
        "dividend_yield": snap.dividend_yield,
        "n_quotes_total": snap.n_quotes_total,
        "n_quotes_dropped": snap.n_quotes_dropped,
    }
    meta["market_data_hash"] = _content_hash({
        "iv_grid": snap.iv_grid,
        "strikes": snap.strikes,
        "maturities": snap.maturities,
        "spot": snap.spot,
        "rate": snap.rate,
        "dividend_yield": snap.dividend_yield,
        "real_label": snap.snapshot_label,
    })
    return MarketDataBundle(market_data=snap.market_data, meta=meta)
