"""
Base Model Classes
==================

Common enums and utilities for financial models.

Provides:
- Measure: Enum for probability measure (P/Q)

Note: The canonical PricingCapability enum is in backend.core.result_types.
      All models should inherit from backend.core.interfaces.Model.

Author: Thomas
Created: 2025
"""

from enum import Enum


class Measure(Enum):
    """
    Probability measure for simulation/pricing.

    P (Physical): Real-world measure, uses expected return mu as drift
    Q (Risk-Neutral): Risk-neutral measure, uses risk-free rate r as drift
    """
    P = "physical"
    Q = "risk_neutral"
