"""
Tests for Engine Registry
=========================

Verify all models are properly registered with the EngineRegistry.

Author: Thomas
Created: 2025
"""

import pytest

from backend.core import EngineRegistry
from backend.core.result_types import PricingCapability
from backend.models import (
    GBMModel,
    HestonModel,
    BatesModel,
    MertonModel,
)
from backend.models.garch import GARCHModel, NGARCHModel, GJRGARCHModel

# Ensure engines are registered
from backend.engines import ensure_registered
ensure_registered()


def get_registered_engines_for_model(model_name: str):
    """
    Helper to get all registered engines for a given model name.

    Returns a list of (capability, engine_class) tuples.
    """
    engines = EngineRegistry.list_engines()
    return [(cap, name) for name, cap in engines if name == model_name]


class TestEngineRegistry:
    """Tests for EngineRegistry functionality."""

    def test_gbm_has_all_engines(self):
        """GBM should support analytical, FFT, and Monte Carlo."""
        model = GBMModel(sigma=0.20)
        engines = get_registered_engines_for_model(model.name)

        assert len(engines) >= 3
        capabilities = {cap for cap, _ in engines}
        assert "ANALYTICAL" in capabilities
        assert "FFT" in capabilities
        assert "MONTE_CARLO" in capabilities

    def test_heston_has_fft_and_mc(self):
        """Heston should support FFT and Monte Carlo."""
        model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        engines = get_registered_engines_for_model(model.name)

        assert len(engines) >= 2
        capabilities = {cap for cap, _ in engines}
        assert "FFT" in capabilities
        assert "MONTE_CARLO" in capabilities

    def test_bates_has_fft_and_mc(self):
        """Bates should support FFT and Monte Carlo."""
        model = BatesModel(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
        )
        engines = get_registered_engines_for_model(model.name)

        assert len(engines) >= 2
        capabilities = {cap for cap, _ in engines}
        assert "FFT" in capabilities
        assert "MONTE_CARLO" in capabilities

    def test_merton_has_fft_and_mc(self):
        """Merton should support FFT and Monte Carlo."""
        model = MertonModel(sigma=0.20, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)
        engines = get_registered_engines_for_model(model.name)

        assert len(engines) >= 2
        capabilities = {cap for cap, _ in engines}
        assert "FFT" in capabilities
        assert "MONTE_CARLO" in capabilities

    def test_garch_has_mc(self):
        """GARCH(1,1) should support Monte Carlo."""
        model = GARCHModel(sigma0=0.20, omega=0.002, alpha=0.05, beta=0.90)
        engines = get_registered_engines_for_model(model.name)

        assert len(engines) >= 1
        capabilities = {cap for cap, _ in engines}
        assert "MONTE_CARLO" in capabilities

    def test_ngarch_has_mc(self):
        """NGARCH should support Monte Carlo."""
        # NGARCH uses 'theta' for leverage, not 'gamma'
        model = NGARCHModel(sigma0=0.20, omega=0.002, alpha=0.05, beta=0.90, theta=0.3)
        engines = get_registered_engines_for_model(model.name)

        assert len(engines) >= 1
        capabilities = {cap for cap, _ in engines}
        assert "MONTE_CARLO" in capabilities

    def test_gjr_garch_has_mc(self):
        """GJR-GARCH should support Monte Carlo."""
        # GJR-GARCH uses 'gamma' for leverage
        # alpha + 0.5*gamma + beta < 1 for stationarity
        model = GJRGARCHModel(sigma0=0.20, omega=0.002, alpha=0.04, beta=0.85, gamma=0.10)
        engines = get_registered_engines_for_model(model.name)

        assert len(engines) >= 1
        capabilities = {cap for cap, _ in engines}
        assert "MONTE_CARLO" in capabilities

    def test_all_models_registered(self):
        """Verify all models have at least one engine registered."""
        all_models = [
            GBMModel(sigma=0.20),
            HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7),
            BatesModel(
                v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
            ),
            MertonModel(sigma=0.20, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2),
            GARCHModel(sigma0=0.20, omega=0.002, alpha=0.05, beta=0.90),
            NGARCHModel(sigma0=0.20, omega=0.002, alpha=0.05, beta=0.90, theta=0.3),
            GJRGARCHModel(sigma0=0.20, omega=0.002, alpha=0.04, beta=0.85, gamma=0.10),
        ]

        for model in all_models:
            engines = get_registered_engines_for_model(model.name)
            assert len(engines) > 0, f"No engines registered for {model.name}"


class TestPricingCapabilityIdentity:
    """Tests to ensure there's only one PricingCapability definition."""

    def test_single_pricing_capability_enum(self):
        """Ensure PricingCapability from result_types is used everywhere."""
        from backend.core.result_types import PricingCapability as PC1

        # Check that GBM model uses the same enum
        model = GBMModel(sigma=0.20)
        assert PC1.ANALYTICAL in model.supported_engines
        assert PC1.FFT in model.supported_engines
        assert PC1.MONTE_CARLO in model.supported_engines

    def test_pricing_capability_consistency(self):
        """Registry capabilities should be identical to model capabilities."""
        from backend.core.result_types import PricingCapability

        model = GBMModel(sigma=0.20)
        engines = get_registered_engines_for_model(model.name)
        registry_capabilities = {cap for cap, _ in engines}

        # All model capabilities should be in registry (as strings)
        for cap in model.supported_engines:
            assert cap.name in registry_capabilities, f"{cap.name} not in registry for {model.name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
