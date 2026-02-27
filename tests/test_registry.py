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

    def test_garch_has_no_engines(self):
        """GARCH(1,1) should have no registered engines (uses own create_pricer)."""
        model = GARCHModel(sigma0=0.20, omega=0.002, alpha=0.05, beta=0.90)
        engines = get_registered_engines_for_model(model.name)
        assert len(engines) == 0

    def test_ngarch_has_no_engines(self):
        """NGARCH should have no registered engines (uses own create_pricer)."""
        model = NGARCHModel(sigma0=0.20, omega=0.002, alpha=0.05, beta=0.90, theta=0.3)
        engines = get_registered_engines_for_model(model.name)
        assert len(engines) == 0

    def test_gjr_garch_has_no_engines(self):
        """GJR-GARCH should have no registered engines (uses own create_pricer)."""
        model = GJRGARCHModel(sigma0=0.20, omega=0.002, alpha=0.04, beta=0.85, gamma=0.10)
        engines = get_registered_engines_for_model(model.name)
        assert len(engines) == 0

    def test_all_non_garch_models_registered(self):
        """Verify all non-GARCH models have at least one engine registered."""
        all_models = [
            GBMModel(sigma=0.20),
            HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7),
            BatesModel(
                v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
            ),
            MertonModel(sigma=0.20, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2),
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

        model = GBMModel(sigma=0.20)
        engines = get_registered_engines_for_model(model.name)
        registry_capabilities = {cap for cap, _ in engines}

        # All model capabilities should be in registry (as strings)
        for cap in model.supported_engines:
            assert cap.name in registry_capabilities, f"{cap.name} not in registry for {model.name}"


class TestEngineRegistryLifecycle:
    """Tests for EngineRegistry lifecycle operations."""

    def test_unregister_engine(self):
        """Verify unregister returns True for existing registration."""
        # Register a test entry, then unregister it
        from backend.engines.analytic_engine import BSAnalyticEngine

        EngineRegistry.register(
            "TestModel_Lifecycle",
            PricingCapability.ANALYTICAL,
            BSAnalyticEngine
        )
        result = EngineRegistry.unregister("TestModel_Lifecycle", PricingCapability.ANALYTICAL)
        assert result is True

    def test_unregister_nonexistent_returns_false(self):
        """Verify unregister returns False for non-existent registration."""
        result = EngineRegistry.unregister("NonexistentModel_XYZ", PricingCapability.ANALYTICAL)
        assert result is False

    def test_clear_and_re_register(self):
        """Verify clear + re-register lifecycle."""
        from backend.engines.analytic_engine import BSAnalyticEngine

        # Register a test entry
        EngineRegistry.register(
            "TestModel_Clear",
            PricingCapability.ANALYTICAL,
            BSAnalyticEngine
        )

        # Clear all registrations
        EngineRegistry.clear()

        # No engines should be registered
        engines = EngineRegistry.list_engines()
        assert len(engines) == 0

        # Re-register everything
        from backend.engines._registration import register_all_engines
        register_all_engines()

        # GBM should be back
        model = GBMModel(sigma=0.20)
        engines = get_registered_engines_for_model(model.name)
        assert len(engines) >= 3

    def test_get_engine_preferred_capability(self):
        """Verify preferred capability selection."""
        from backend.instruments.options import VanillaOption
        from backend.engines import FFTEngine

        model = GBMModel(sigma=0.20)
        option = VanillaOption(strike=100, maturity=0.5, is_call=True)

        # Request FFT specifically for GBM
        engine = EngineRegistry.get_engine(option, model, preferred=PricingCapability.FFT)
        assert isinstance(engine, FFTEngine)

    def test_get_engine_no_match_raises(self):
        """Verify ValueError for GARCH model with no registered engines."""
        from backend.instruments.options import VanillaOption

        model = GARCHModel(sigma0=0.20, omega=0.002, alpha=0.05, beta=0.90)
        option = VanillaOption(strike=100, maturity=0.5, is_call=True)

        with pytest.raises(ValueError):
            EngineRegistry.get_engine(option, model)


class TestModuleLevelPrice:
    """Tests for the module-level price() function."""

    def test_price_function_gbm(self):
        """price() auto-selects BSAnalyticEngine for GBM."""
        from backend.core.registry import price
        from backend.instruments.options import VanillaOption
        from backend.core.market import MarketEnvironment

        model = GBMModel(sigma=0.20)
        option = VanillaOption(strike=100, maturity=0.5, is_call=True)
        market = MarketEnvironment(spot=100, rate=0.05)

        result = price(option, model, market)
        assert result.price > 0
        assert result.engine == "BSAnalyticEngine"

    def test_price_function_heston(self):
        """price() auto-selects FFTEngine for Heston."""
        from backend.core.registry import price
        from backend.instruments.options import VanillaOption
        from backend.core.market import MarketEnvironment

        model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        option = VanillaOption(strike=100, maturity=0.5, is_call=True)
        market = MarketEnvironment(spot=100, rate=0.05)

        result = price(option, model, market)
        assert result.price > 0
        assert result.engine == "FFTEngine"

    def test_price_function_with_preferred_method(self):
        """price() respects preferred method override."""
        from backend.core.registry import price
        from backend.instruments.options import VanillaOption
        from backend.core.market import MarketEnvironment

        model = GBMModel(sigma=0.20)
        option = VanillaOption(strike=100, maturity=0.5, is_call=True)
        market = MarketEnvironment(spot=100, rate=0.05)

        result = price(option, model, market, method=PricingCapability.FFT)
        assert result.price > 0
        assert result.engine == "FFTEngine"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
