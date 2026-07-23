"""
SPX snapshot repository
========================

Owns everything related to *finding and reading* the embedded SPX option
parquet files: filename parsing, timezone conversion, file I/O. The
business logic (filtering quotes, building :class:`OptionMarketData`)
lives in :mod:`services.real_data_service`.

Keeping these concerns separate makes both layers easier to test and
swap (e.g. plugging a remote data source).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq

# Filename token validators (spx_YYYY-MM-DD_HHMMSS.parquet).
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_HMS_RE = re.compile(r"\d{6}")

# SPX trades on the CBOE — quote times are most natural in New York time.
# zoneinfo handles the EDT ↔ EST DST switch automatically.
NYC_TZ = ZoneInfo("America/New_York")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class SnapshotEntry:
    file: Path
    label: str  # e.g., "2025-06-02 · 13:30 EDT"
    trade_date: str
    snap_iso: str  # e.g., "2025-06-02T17:30:00Z"

    @property
    def key(self) -> str:
        return self.snap_iso


def _parse_filename(p: Path) -> tuple[str, str] | None:
    """File name format: spx_YYYY-MM-DD_HHMMSS.parquet (UTC).

    Validates the date and time tokens (a non-numeric / short HHMMSS used to
    slip through and build a malformed ISO string that later crashed
    ``datetime.fromisoformat`` inside :func:`list_embedded_snapshots`, outside
    any try — breaking the whole real-data path over one stray file)."""
    stem = p.stem  # spx_2025-06-02_173000
    parts = stem.split("_")
    if len(parts) != 3:
        return None
    _, date, hms = parts
    if not _DATE_RE.fullmatch(date) or not _HMS_RE.fullmatch(hms):
        return None
    hh, mm, ss = hms[:2], hms[2:4], hms[4:6]
    return date, f"{date}T{hh}:{mm}:{ss}Z"


def to_nyc(snap_iso: str, *, fmt: str = "full") -> str:
    """Convert a UTC ISO timestamp to a New York-local label.

    Parameters
    ----------
    snap_iso :
        ISO 8601 string ending in ``Z`` (UTC), e.g. ``2025-06-02T17:30:00Z``.
    fmt :
        ``"full"`` → ``"2025-06-02  ·  13:30 EDT"`` (default).
        ``"time"`` → ``"13:30 EDT"`` (just the clock part).
    """
    dt_utc = datetime.fromisoformat(snap_iso.replace("Z", "+00:00"))
    dt_nyc = dt_utc.astimezone(NYC_TZ)
    if fmt == "time":
        return f"{dt_nyc.strftime('%H:%M')} {dt_nyc.strftime('%Z')}"
    if fmt == "full":
        return (
            f"{dt_nyc.strftime('%Y-%m-%d')}"
            f"  ·  "
            f"{dt_nyc.strftime('%H:%M')} {dt_nyc.strftime('%Z')}"
        )
    raise ValueError(f"Unknown fmt: {fmt!r} (expected 'full' or 'time')")


def list_embedded_snapshots() -> list[SnapshotEntry]:
    """Return entries for every parquet snapshot bundled under ``data/``."""
    if not DATA_DIR.exists():
        return []
    entries: list[SnapshotEntry] = []
    for p in sorted(DATA_DIR.glob("spx_*.parquet")):
        parsed = _parse_filename(p)
        if parsed is None:
            continue
        date, snap_iso = parsed
        entries.append(
            SnapshotEntry(
                file=p,
                label=to_nyc(snap_iso, fmt="full"),
                trade_date=date,
                snap_iso=snap_iso,
            )
        )
    return entries


def is_dataset_available() -> bool:
    """True if at least one embedded snapshot is present."""
    return bool(list_embedded_snapshots())


def find_entry(snapshot_key: str) -> SnapshotEntry | None:
    """Return the entry whose ``snap_iso`` matches ``snapshot_key``, or None."""
    return next(
        (e for e in list_embedded_snapshots() if e.snap_iso == snapshot_key),
        None,
    )


def load_raw_dataframe(path: Path) -> pd.DataFrame:
    """Load a parquet snapshot into a pandas DataFrame."""
    return pq.read_table(str(path)).to_pandas()
