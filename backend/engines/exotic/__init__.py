"""
Exotic options analytical pricing engine.

Author: Thomas Vaudescal
Created: 2026
"""

from backend.engines.exotic.engine import ExoticAnalyticEngine

# Populate EXOTIC_PRICER_REGISTRY with the advanced (Haug-catalog) pricers as a
# side effect, so ExoticAnalyticEngine can dispatch them via the registry.
from backend.engines.exotic import _advanced_registration  # noqa: E402, F401

__all__ = ["ExoticAnalyticEngine"]
