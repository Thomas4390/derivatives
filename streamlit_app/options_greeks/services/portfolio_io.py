"""
Portfolio snapshot encode / decode for the Options Greeks Explorer.

Streamlit keeps no state between sessions, so a professor or student cannot return
to a setup (market context + option/stock positions) they built earlier. This module
serialises the current setup to a compact, copy-paste **"setup code"** and restores it
later. It is pure logic (no Streamlit) so it is unit-testable headless.

Design / best practices
-----------------------
* The code is ``OGX1-`` + base64-url(gzip(json)) — a copy-paste string, **not** a
  downloaded file (the apps deliberately offer no downloads; a setup is an *input*,
  not a result).
* **Strict boundary validation**: every imported field is type/range-checked against
  the position schema (reusing :mod:`backend.utils.validation`); only known keys are
  kept; the input is parsed with ``json`` only — never ``eval``/``exec``. DoS guards
  cap the decoded size and the number of legs.
* :func:`decode_snapshot` accepts **either** the compact code **or** raw JSON, so a
  hand-written setup and the educational "view as JSON" panel both round-trip.

Author: Thomas Vaudescal
"""

from __future__ import annotations

import base64
import gzip
import json
import math
from dataclasses import dataclass
from typing import Any

from backend.utils.logging import get_logger
from backend.utils.validation import (
    ValidationError,
    validate_dividend_yield,
    validate_rate,
    validate_spot,
    validate_strike,
    validate_volatility,
)

logger = get_logger(__name__)

SNAPSHOT_SCHEMA_VERSION = 1
_CODE_TAG = "OGX1-"  # Options-Greeks eXplorer, schema v1

# Anti-DoS guards on untrusted pasted input.
_MAX_INPUT_CHARS = 1_000_000
_MAX_DECODED_BYTES = 256 * 1024
_MAX_LEGS = 100

_OPTION_TYPES = frozenset({"call", "put"})
_POSITION_TYPES = frozenset({"long", "short"})
_INSTRUMENT_CLASSES = frozenset(
    {
        "vanilla",
        "barrier",
        "digital",
        "lookback_floating",
        "chooser",
        "asset_or_nothing",
        "power",
        "gap",
        "powered",
        "capped_power",
        "log_contract",
        "log_option",
        "supershare",
        "double_barrier",
        "discrete_barrier",
        "partial_barrier",
        "binary_barrier",
        "arithmetic_asian",
    }
)
# Optional exotic fields → coercion callable. Only present keys are kept.
_EXOTIC_FIELDS: dict[str, Any] = {
    "barrier": float,
    "is_up": bool,
    "is_knock_in": bool,
    "rebate": float,
    "payout": float,
    "extra1": float,
    "choice_time_pct": float,
    "power_n": float,
    "gap_trigger": float,
    "cap": float,
    "lower_strike": float,
    "upper_strike": float,
    "dbl_lower": float,
    "dbl_upper": float,
    "adv_barrier": float,
    "adv_is_up": bool,
    "adv_in": bool,
    "monitoring_points": int,
    "t1_pct": float,
    "partial_type": str,
    "cash": float,
    "binary_type": int,
    "avg_elapsed_pct": float,
    "avg_realized": float,
}


class SnapshotError(ValueError):
    """Raised when a pasted setup code / JSON cannot be decoded or is invalid."""


@dataclass(frozen=True)
class Snapshot:
    """A validated, restorable Options-Greeks setup."""

    market: dict[str, float]  # {"spot", "rate", "dividend_yield"}
    positions: list[dict[str, Any]]  # sanitised option position dicts
    stock: dict[str, Any] | None  # sanitised stock position dict, or None


# --------------------------------------------------------------------------- #
# Encode
# --------------------------------------------------------------------------- #


def _build_payload(
    market: dict[str, Any],
    positions: list[dict[str, Any]],
    stock: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "app": "options_greeks",
        "market": {
            "spot": float(market.get("spot", market.get("spot_price", 0.0))),
            "rate": float(market.get("rate", market.get("risk_free_rate", 0.0))),
            "dividend_yield": float(market.get("dividend_yield", 0.0)),
        },
        "positions": [dict(p) for p in positions],
        "stock": dict(stock) if stock else None,
    }


def encode_snapshot(
    market: dict[str, Any],
    positions: list[dict[str, Any]],
    stock: dict[str, Any] | None,
) -> str:
    """Serialise the current setup to a compact, copy-paste 'setup code'."""
    raw = json.dumps(
        _build_payload(market, positions, stock), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    packed = gzip.compress(raw, compresslevel=9)
    return _CODE_TAG + base64.urlsafe_b64encode(packed).decode("ascii")


def to_pretty_json(
    market: dict[str, Any],
    positions: list[dict[str, Any]],
    stock: dict[str, Any] | None,
) -> str:
    """Human-readable JSON of the setup (educational 'view as JSON' panel)."""
    return json.dumps(
        _build_payload(market, positions, stock), indent=2, sort_keys=True
    )


# --------------------------------------------------------------------------- #
# Decode
# --------------------------------------------------------------------------- #


def decode_snapshot(text: str) -> Snapshot:
    """Decode a pasted 'setup code' **or** raw JSON into a validated :class:`Snapshot`.

    Raises :class:`SnapshotError` (with a user-facing message) on any malformation.
    """
    text = (text or "").strip()
    if not text:
        raise SnapshotError("Empty input — paste a setup code or JSON.")
    if len(text) > _MAX_INPUT_CHARS:
        raise SnapshotError("Input is too large.")

    try:
        if text.startswith("{"):
            data = json.loads(text)
        else:
            blob = text[len(_CODE_TAG) :] if text.startswith(_CODE_TAG) else text
            try:
                packed = base64.urlsafe_b64decode(blob.encode("ascii"))
                raw = gzip.decompress(packed)
            except Exception as exc:
                raise SnapshotError(
                    "This does not look like a valid setup code."
                ) from exc
            if len(raw) > _MAX_DECODED_BYTES:
                raise SnapshotError("Setup code is too large.")
            data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"Malformed JSON: {exc.msg}") from exc

    return validate_snapshot(data)


def validate_snapshot(data: Any) -> Snapshot:
    """Validate a decoded snapshot dict at the trust boundary."""
    if not isinstance(data, dict):
        raise SnapshotError("Snapshot must be a JSON object.")
    version = data.get("schema_version")
    if version != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotError(
            f"Unsupported setup version {version!r} (expected {SNAPSHOT_SCHEMA_VERSION})."
        )

    market = _validate_market(data.get("market", {}))

    raw_positions = data.get("positions", [])
    if not isinstance(raw_positions, list):
        raise SnapshotError("'positions' must be a list.")
    if len(raw_positions) > _MAX_LEGS:
        raise SnapshotError(f"Too many positions ({len(raw_positions)} > {_MAX_LEGS}).")
    positions = [_validate_position(p, i) for i, p in enumerate(raw_positions)]

    stock = _validate_stock(data.get("stock"))
    return Snapshot(market=market, positions=positions, stock=stock)


def _validate_market(market: Any) -> dict[str, float]:
    if not isinstance(market, dict):
        raise SnapshotError("'market' must be an object.")
    try:
        spot = validate_spot(float(market["spot"]))
        rate = validate_rate(float(market["rate"]))
        q = validate_dividend_yield(float(market.get("dividend_yield", 0.0)))
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise SnapshotError(f"Invalid market parameters: {exc}") from exc
    return {"spot": spot, "rate": rate, "dividend_yield": q}


def _validate_position(pos: Any, idx: int) -> dict[str, Any]:
    if not isinstance(pos, dict):
        raise SnapshotError(f"Position {idx + 1} must be an object.")
    option_type = pos.get("option_type")
    if option_type not in _OPTION_TYPES:
        raise SnapshotError(f"Position {idx + 1}: option_type must be 'call' or 'put'.")
    position_type = pos.get("position_type")
    if position_type not in _POSITION_TYPES:
        raise SnapshotError(
            f"Position {idx + 1}: position_type must be 'long' or 'short'."
        )
    instrument_class = pos.get("instrument_class", "vanilla")
    if instrument_class not in _INSTRUMENT_CLASSES:
        raise SnapshotError(
            f"Position {idx + 1}: unknown instrument_class {instrument_class!r}."
        )
    try:
        strike = validate_strike(float(pos["strike"]))
        quantity = int(pos["quantity"])
        premium = float(pos.get("premium_paid", 0.0))
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise SnapshotError(
            f"Position {idx + 1}: invalid strike/quantity/premium ({exc})."
        ) from exc
    if quantity <= 0:
        raise SnapshotError(f"Position {idx + 1}: quantity must be a positive integer.")
    if not math.isfinite(premium) or premium < 0:
        raise SnapshotError(
            f"Position {idx + 1}: premium must be a finite, non-negative number."
        )

    out: dict[str, Any] = {
        "option_type": option_type,
        "position_type": position_type,
        "strike": strike,
        "quantity": quantity,
        "premium_paid": premium,
    }
    if pos.get("dte_days") is not None:
        out["dte_days"] = int(pos["dte_days"])
    if pos.get("volatility") is not None:
        try:
            out["volatility"] = validate_volatility(float(pos["volatility"]))
        except (TypeError, ValueError, ValidationError) as exc:
            raise SnapshotError(
                f"Position {idx + 1}: invalid volatility ({exc})."
            ) from exc

    if instrument_class != "vanilla":
        out["instrument_class"] = instrument_class
        for field, cast in _EXOTIC_FIELDS.items():
            if field in pos and pos[field] is not None:
                try:
                    out[field] = cast(pos[field])
                except (TypeError, ValueError) as exc:
                    raise SnapshotError(
                        f"Position {idx + 1}: invalid {field} ({exc})."
                    ) from exc
    return out


def _validate_stock(stock: Any) -> dict[str, Any] | None:
    if stock is None:
        return None
    if not isinstance(stock, dict):
        raise SnapshotError("'stock' must be an object or null.")
    position_type = stock.get("position_type")
    if position_type not in _POSITION_TYPES:
        raise SnapshotError("Stock position_type must be 'long' or 'short'.")
    try:
        quantity = int(stock["quantity"])
        entry_price = float(stock["entry_price"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotError(f"Invalid stock position ({exc}).") from exc
    if quantity <= 0:
        raise SnapshotError("Stock quantity must be a positive integer.")
    if not math.isfinite(entry_price) or entry_price <= 0:
        raise SnapshotError("Stock entry price must be a finite, positive number.")
    return {
        "position_type": position_type,
        "quantity": quantity,
        "entry_price": entry_price,
    }
