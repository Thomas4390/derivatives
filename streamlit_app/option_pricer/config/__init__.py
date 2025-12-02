"""Configuration module for Options Greeks Explorer."""

from .constants import (
    CONTRACT_MULTIPLIER,
    GREEK_NAMES,
    FIRST_ORDER,
    SECOND_ORDER,
    THIRD_ORDER,
    GREEK_TITLES,
    GREEK_COLORS,
    DEFAULT_SPOT_PRICE,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_IV,
    DEFAULT_DTE,
    SPOT_RANGE_FACTOR,
    SPOT_RANGE_POINTS,
    DTE_RANGE,
    IV_RANGE,
)

from .styles import inject_styles, CUSTOM_CSS

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
