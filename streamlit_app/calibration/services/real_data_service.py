"""
SPX real-market option-chain loader
=====================================

Reads pre-extracted SPX option snapshots that ship with the repo
(``streamlit_app/calibration/data/spx_*.parquet``), applies liquidity
filters and converts the chosen quotes into an :class:`OptionMarketData`
ready to feed any of the surface-based calibrators.

File discovery, parsing and timezone handling live in
:mod:`services.snapshot_repository`. This module only does the
business-level work (filter and assemble).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.calibration.market_data import OptionMarketData, OptionQuote

from config.constants import IV_FILTER_MAX, IV_FILTER_MIN
from services.snapshot_repository import (
    SnapshotEntry,  # noqa: F401  — re-exported for sidebar / tests
    find_entry,
    is_dataset_available,  # noqa: F401  — re-exported for sidebar / tests
    list_embedded_snapshots,  # noqa: F401  — re-exported for sidebar / tests
    load_raw_dataframe,
)

DEFAULT_RATE = 0.045
DEFAULT_DIVIDEND_YIELD = 0.013


@dataclass(frozen=True)
class RealMarketSnapshot:
    market_data: OptionMarketData
    spot: float
    rate: float
    dividend_yield: float
    snapshot_label: str
    iv_grid: np.ndarray  # (n_T, n_K) IVs (call mid)
    strikes: np.ndarray
    maturities: np.ndarray
    n_quotes_total: int
    n_quotes_dropped: int


def load_snapshot(
    *,
    snapshot_key: str,
    dte_min: int = 7,
    dte_max: int = 60,
    moneyness_min: float = 0.85,
    moneyness_max: float = 1.15,
    rate: float = DEFAULT_RATE,
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD,
) -> RealMarketSnapshot:
    """Load and filter one of the embedded SPX snapshots.

    Parameters
    ----------
    snapshot_key : str
        ISO datetime that matches one of the embedded files
        (see :func:`snapshot_repository.list_embedded_snapshots`).
    dte_min, dte_max : int
        Days-to-expiry filter — keep options with maturity in
        ``[dte_min, dte_max]``.
    moneyness_min, moneyness_max : float
        Filter on K / S₀ to keep liquid near-the-money quotes only.
    rate, dividend_yield : float
        Risk-free rate and continuous dividend yield assumptions.
    """
    entry = find_entry(snapshot_key)
    if entry is None:
        raise FileNotFoundError(
            f"Snapshot '{snapshot_key}' not found among embedded files."
        )

    df = load_raw_dataframe(entry.file)
    n_raw = len(df)
    # The constant underlying quote for the snapshot is ``spotPrice`` (one value
    # per file); ``stockPrice`` varies row to row (it carries a per-row quote
    # print, including the discarded dte=1 row), so ``stockPrice.iloc[0]`` gave
    # a spot up to ~0.3 % off — shifting the moneyness filter and the OTM
    # call/put side selection.
    spot = float(df["spotPrice"].iloc[0])

    df = df[(df["dte"] >= dte_min) & (df["dte"] <= dte_max)]
    df["moneyness"] = df["strike"] / spot
    df = df[(df["moneyness"] >= moneyness_min) & (df["moneyness"] <= moneyness_max)]

    # Each expiry carries an AM and a PM (SPX / SPXW) series with the SAME
    # (expiry, strike) but contradictory IVs; the grid keeps only the last, so
    # the residual vector double-weighted them against inconsistent targets.
    # Keep the tighter-spread series per (expiry, strike) so quotes and iv_grid
    # describe one unique surface.
    df["_spread"] = (
        df["callAskPrice"].fillna(0) - df["callBidPrice"].fillna(0)
    ).abs() + (df["putAskPrice"].fillna(0) - df["putBidPrice"].fillna(0)).abs()
    df = (
        df.sort_values("_spread")
        .drop_duplicates(subset=["expirDate", "strike"], keep="first")
        .sort_index()
    )
    n_total = len(df)

    df["call_mid"] = 0.5 * (df["callBidPrice"].fillna(0) + df["callAskPrice"].fillna(0))
    df["put_mid"] = 0.5 * (df["putBidPrice"].fillna(0) + df["putAskPrice"].fillna(0))

    quotes: list[OptionQuote] = []
    for _, r in df.iterrows():
        # Vendor ``dte`` is calendar-days + 1 (a same-day expiry reports dte=1),
        # so the time to expiry is (dte-1)/365; the dte FILTER above stays in
        # vendor units. The +1-day bias overstated every maturity (up to
        # ~14-17 % at the short end), tilting the fitted short-dated term
        # structure and biasing mean-reversion / vol-of-vol.
        T = (float(r["dte"]) - 1.0) / 365.0
        K = float(r["strike"])
        is_call = K >= spot  # OTM convention — picks the more liquid side
        if is_call:
            mid = float(r["call_mid"])
            iv = float(r["callMidIv"]) if pd.notna(r["callMidIv"]) else None
        else:
            mid = float(r["put_mid"])
            iv = float(r["putMidIv"]) if pd.notna(r["putMidIv"]) else None
        if (
            mid <= 0
            or T <= 0
            or iv is None
            or not np.isfinite(iv)
            or iv <= IV_FILTER_MIN
            or iv > IV_FILTER_MAX
        ):
            continue
        quotes.append(
            OptionQuote(
                strike=K,
                maturity=T,
                is_call=bool(is_call),
                market_price=mid,
                implied_vol=iv,
            )
        )
    # Report against the RAW row count so the caption is honest: n_total used
    # to be the post-range-filter count and n_dropped excluded every dte /
    # moneyness / dedup drop (e.g. '2221 raw quotes · 0 dropped' for a file
    # that started at 2950 rows).
    n_kept = len(quotes)
    n_dropped = n_raw - n_kept
    if not quotes:
        raise ValueError(
            "After filtering, no usable quotes remain. Loosen the dte/moneyness range."
        )

    md = OptionMarketData(
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        quotes=tuple(quotes),
    )

    strikes = np.array(sorted({q.strike for q in quotes}))
    maturities = np.array(sorted({q.maturity for q in quotes}))
    iv_grid = np.full((len(maturities), len(strikes)), np.nan)
    sidx = {s: j for j, s in enumerate(strikes)}
    midx = {t: i for i, t in enumerate(maturities)}
    for q in quotes:
        if q.implied_vol is not None:
            iv_grid[midx[q.maturity], sidx[q.strike]] = q.implied_vol

    return RealMarketSnapshot(
        market_data=md,
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        snapshot_label=entry.label,
        iv_grid=iv_grid,
        strikes=strikes,
        maturities=maturities,
        n_quotes_total=n_raw,
        n_quotes_dropped=n_dropped,
    )
