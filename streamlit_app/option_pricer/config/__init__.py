"""Configuration module for Options Greeks Explorer."""

from .constants import (
    CONTRACT_MULTIPLIER,
    DEFAULT_DTE,
    DEFAULT_IV,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_SPOT_PRICE,
    DTE_RANGE,
    FIRST_ORDER,
    GREEK_COLORS,
    GREEK_NAMES,
    GREEK_TITLES,
    IV_RANGE,
    SECOND_ORDER,
    SPOT_RANGE_FACTOR,
    SPOT_RANGE_POINTS,
    THIRD_ORDER,
)
from .styles import CUSTOM_CSS, inject_styles

__all__ = [
    "CONTRACT_MULTIPLIER",
    "GREEK_NAMES",
    "FIRST_ORDER",
    "SECOND_ORDER",
    "THIRD_ORDER",
    "GREEK_TITLES",
    "GREEK_COLORS",
    "DEFAULT_SPOT_PRICE",
    "DEFAULT_RISK_FREE_RATE",
    "DEFAULT_IV",
    "DEFAULT_DTE",
    "SPOT_RANGE_FACTOR",
    "SPOT_RANGE_POINTS",
    "DTE_RANGE",
    "IV_RANGE",
    "inject_styles",
    "CUSTOM_CSS",
]
