"""
Interface Compliance Tests
==========================

Tests to verify all models and payoffs implement their interfaces correctly.

Author: Thomas
Created: 2025
"""

import pytest
import numpy as np
import warnings

from backend.core.interfaces import Model
from backend.core.result_types import ExerciseStyle, PricingCapability
from backend.models import GBMModel, HestonModel, MertonModel, BatesModel
from backend.instruments import (
    VanillaCallPayoff, VanillaPutPayoff,
    DigitalCallPayoff, DigitalPutPayoff,
    EuropeanCall, VanillaOption,
)
from backend.instruments.strategies import StrategyLeg
from backend.models.registry import registry


# =============================================================================
# Model Interface Tests
# =============================================================================

class TestModelInterface:
    """Verify all models implement required interface."""

    @pytest.mark.parametrize("model_cls,params", [
        (GBMModel, {"sigma": 0.2}),
        (HestonModel, {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7}),
        (MertonModel, {"sigma": 0.2, "lambda_j": 0.5, "mu_j": -0.1, "sigma_j": 0.2}),
        (BatesModel, {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7,
                      "lambda_j": 0.5, "mu_j": -0.1, "sigma_j": 0.2}),
    ])
    def test_all_models_have_name(self, model_cls, params):
        """All models must have a name property."""
        model = model_cls(**params)
        assert hasattr(model, 'name')
        assert isinstance(model.name, str)
        assert len(model.name) > 0

    @pytest.mark.parametrize("model_cls,params", [
        (GBMModel, {"sigma": 0.2}),
        (HestonModel, {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7}),
        (MertonModel, {"sigma": 0.2, "lambda_j": 0.5, "mu_j": -0.1, "sigma_j": 0.2}),
        (BatesModel, {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7,
                      "lambda_j": 0.5, "mu_j": -0.1, "sigma_j": 0.2}),
    ])
    def test_all_models_have_supported_engines(self, model_cls, params):
        """All models must declare supported engines."""
        model = model_cls(**params)
        assert hasattr(model, 'supported_engines')
        assert isinstance(model.supported_engines, list)
        assert len(model.supported_engines) > 0
        for cap in model.supported_engines:
            assert isinstance(cap, PricingCapability)

    @pytest.mark.parametrize("model_cls,params", [
        (GBMModel, {"sigma": 0.2}),
        (HestonModel, {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7}),
        (MertonModel, {"sigma": 0.2, "lambda_j": 0.5, "mu_j": -0.1, "sigma_j": 0.2}),
        (BatesModel, {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7,
                      "lambda_j": 0.5, "mu_j": -0.1, "sigma_j": 0.2}),
    ])
    def test_all_models_have_get_parameters(self, model_cls, params):
        """All models must have get_parameters method."""
        model = model_cls(**params)
        assert hasattr(model, 'get_parameters')
        result = model.get_parameters()
        assert isinstance(result, dict)

    @pytest.mark.parametrize("model_cls,params", [
        (GBMModel, {"sigma": 0.2}),
        (HestonModel, {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7}),
        (MertonModel, {"sigma": 0.2, "lambda_j": 0.5, "mu_j": -0.1, "sigma_j": 0.2}),
        (BatesModel, {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7,
                      "lambda_j": 0.5, "mu_j": -0.1, "sigma_j": 0.2}),
    ])
    def test_fft_models_have_vectorized_cf(self, model_cls, params):
        """All FFT-capable models must have characteristic_function_vectorized."""
        model = model_cls(**params)
        if PricingCapability.FFT in model.supported_engines:
            assert hasattr(model, 'characteristic_function_vectorized')
            # Test it works
            u = np.array([1.0, 2.0])
            cf = model.characteristic_function_vectorized(u, 100, 0.5, 0.05, 0.0)
            assert cf.shape == u.shape
            assert np.all(np.isfinite(cf))


# =============================================================================
# Payoff Edge Case Tests
# =============================================================================

class TestPayoffEdgeCases:
    """Test payoff edge cases."""

    def test_payoff_rejects_nan(self):
        """Payoffs should reject NaN inputs."""
        call = VanillaCallPayoff(strike=100)
        with pytest.raises(ValueError, match="finite"):
            call(np.array([np.nan]))

    def test_payoff_rejects_inf(self):
        """Payoffs should reject Inf inputs."""
        call = VanillaCallPayoff(strike=100)
        with pytest.raises(ValueError, match="finite"):
            call(np.array([np.inf]))

    def test_payoff_rejects_negative_inf(self):
        """Payoffs should reject -Inf inputs."""
        put = VanillaPutPayoff(strike=100)
        with pytest.raises(ValueError, match="finite"):
            put(np.array([-np.inf]))

    def test_payoff_rejects_negative_spots(self):
        """Payoffs should reject negative spot prices."""
        call = VanillaCallPayoff(strike=100)
        with pytest.raises(ValueError, match="non-negative"):
            call(np.array([-10.0]))

    def test_digital_call_at_strike_boundary(self):
        """Digital call should pay at spot == strike (standard convention)."""
        digital = DigitalCallPayoff(strike=100.0, payout=1.0)
        # At strike, call should pay
        result = digital(np.array([100.0]))
        assert result[0] == 1.0

    def test_digital_put_at_strike_boundary(self):
        """Digital put should NOT pay at spot == strike."""
        digital = DigitalPutPayoff(strike=100.0, payout=1.0)
        # At strike, put should not pay (only pays if S < K)
        result = digital(np.array([100.0]))
        assert result[0] == 0.0

    def test_digital_call_above_strike(self):
        """Digital call should pay above strike."""
        digital = DigitalCallPayoff(strike=100.0, payout=5.0)
        result = digital(np.array([100.01, 110.0, 200.0]))
        np.testing.assert_array_equal(result, [5.0, 5.0, 5.0])

    def test_digital_put_below_strike(self):
        """Digital put should pay below strike."""
        digital = DigitalPutPayoff(strike=100.0, payout=5.0)
        result = digital(np.array([99.99, 90.0, 50.0]))
        np.testing.assert_array_equal(result, [5.0, 5.0, 5.0])


# =============================================================================
# Simulation Edge Case Tests
# =============================================================================

class TestSimulationEdgeCases:
    """Test simulation edge cases."""

    def test_antithetic_odd_paths_warning(self):
        """Antithetic with odd paths should warn and adjust."""
        from backend.simulation.models.gbm import GBMSimulator

        sim = GBMSimulator(sigma=0.2, antithetic=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = sim.simulate_terminal(s0=100, mu=0.05, t=1.0, n_paths=101, n_steps=10)
            # Should have issued a warning
            assert len(w) == 1
            assert "even n_paths" in str(w[0].message).lower()
            # Should have used 100 paths instead of 101
            assert len(result) == 100

    def test_antithetic_single_path_handled(self):
        """Antithetic with n_paths=1 should use at least 2 paths."""
        from backend.simulation.models.gbm import GBMSimulator

        sim = GBMSimulator(sigma=0.2, antithetic=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = sim.simulate_terminal(s0=100, mu=0.05, t=1.0, n_paths=1, n_steps=10)
            # Should have issued a warning
            assert len(w) == 1
            # Should have used 2 paths (minimum for antithetic)
            assert len(result) == 2


# =============================================================================
# Registry Parameter Naming Tests
# =============================================================================

class TestRegistryNaming:
    """Test registry parameter naming."""

    def test_gbm_model_registration(self):
        """GBM model should be registrable."""
        registry.register('test_gbm_compliance', GBMModel, aliases=['test_bs'])
        assert registry.is_registered('test_gbm_compliance')
        assert registry.is_registered('test_bs')
        # Cleanup
        registry._models.pop('test_gbm_compliance', None)
        registry._aliases.pop('test_bs', None)

    def test_create_model_with_antithetic_param(self):
        """Registry should work with 'antithetic' not 'use_antithetic'."""
        registry.register('test_gbm_sim', GBMModel)
        try:
            # Should work with 'antithetic' parameter
            sim = registry.create_simulator('test_gbm_sim', sigma=0.2, antithetic=True)
            assert sim is not None
            params = sim.get_parameters()
            assert params.get('antithetic', False) is True
        finally:
            registry._models.pop('test_gbm_sim', None)


# =============================================================================
# StrategyLeg Validation Tests
# =============================================================================

class TestStrategyLegValidation:
    """Test StrategyLeg validation."""

    def test_quantity_zero_rejected(self):
        """StrategyLeg should reject quantity=0."""
        with pytest.raises(ValueError, match="quantity cannot be zero"):
            StrategyLeg(strike=100.0, is_call=True, quantity=0)

    def test_negative_strike_rejected(self):
        """StrategyLeg should reject negative strike."""
        with pytest.raises(ValueError, match="strike must be positive"):
            StrategyLeg(strike=-100.0, is_call=True, quantity=1)

    def test_valid_leg_creation(self):
        """Valid StrategyLeg should be created."""
        leg = StrategyLeg(strike=100.0, is_call=True, quantity=1)
        assert leg.strike == 100.0
        assert leg.is_call is True
        assert leg.quantity == 1
        assert leg.is_long is True
        assert leg.is_short is False


# =============================================================================
# mu_j Parameter Validation Tests
# =============================================================================

class TestMuJValidation:
    """Test mu_j parameter validation in jump models."""

    def test_merton_mu_j_in_range(self):
        """Merton model should accept mu_j in [-1, 1]."""
        model = MertonModel(sigma=0.2, lambda_j=0.5, mu_j=0.0, sigma_j=0.2)
        assert model.mu_j == 0.0

        model = MertonModel(sigma=0.2, lambda_j=0.5, mu_j=-1.0, sigma_j=0.2)
        assert model.mu_j == -1.0

        model = MertonModel(sigma=0.2, lambda_j=0.5, mu_j=1.0, sigma_j=0.2)
        assert model.mu_j == 1.0

    def test_merton_mu_j_out_of_range_rejected(self):
        """Merton model should reject mu_j outside [-1, 1]."""
        with pytest.raises(ValueError, match="mu_j"):
            MertonModel(sigma=0.2, lambda_j=0.5, mu_j=1.5, sigma_j=0.2)

        with pytest.raises(ValueError, match="mu_j"):
            MertonModel(sigma=0.2, lambda_j=0.5, mu_j=-1.5, sigma_j=0.2)

    def test_bates_mu_j_in_range(self):
        """Bates model should accept mu_j in [-1, 1]."""
        model = BatesModel(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            lambda_j=0.5, mu_j=0.0, sigma_j=0.2
        )
        assert model.mu_j == 0.0

    def test_bates_mu_j_out_of_range_rejected(self):
        """Bates model should reject mu_j outside [-1, 1]."""
        with pytest.raises(ValueError, match="mu_j"):
            BatesModel(
                v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                lambda_j=0.5, mu_j=2.0, sigma_j=0.2
            )


# =============================================================================
# Variance Method Tests
# =============================================================================

class TestVarianceMethods:
    """Test variance methods for API consistency."""

    def test_heston_has_expected_variance(self):
        """HestonModel should have expected_variance method."""
        model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        assert hasattr(model, 'expected_variance')
        ev = model.expected_variance(1.0)
        assert np.isfinite(ev)
        assert ev > 0

    def test_heston_has_total_variance(self):
        """HestonModel should have total_variance method."""
        model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        assert hasattr(model, 'total_variance')
        tv = model.total_variance(1.0)
        assert np.isfinite(tv)
        assert tv > 0

    def test_bates_has_variance_methods(self):
        """BatesModel should have variance methods."""
        model = BatesModel(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
        )
        assert hasattr(model, 'expected_variance')
        assert hasattr(model, 'total_variance')
        assert hasattr(model, 'total_volatility')

        # Total variance should include jump contribution
        tv = model.total_variance(1.0)
        heston = model.to_heston()
        tv_heston = heston.total_variance(1.0)
        # Bates should have higher total variance due to jumps
        assert tv > tv_heston


# =============================================================================
# ExerciseStyle Unification Tests
# =============================================================================

class TestExerciseStyleUnification:
    """Test ExerciseStyle/ExerciseType unification."""

    def test_exercise_style_available_from_result_types(self):
        """ExerciseStyle should be available from result_types."""
        from backend.core.result_types import ExerciseStyle
        assert ExerciseStyle.EUROPEAN is not None
        assert ExerciseStyle.AMERICAN is not None
        assert ExerciseStyle.BERMUDAN is not None

    def test_exercise_type_is_alias(self):
        """ExerciseType should be an alias for ExerciseStyle."""
        from backend.instruments.exercise import ExerciseType
        from backend.core.result_types import ExerciseStyle
        assert ExerciseType is ExerciseStyle

    def test_vanilla_option_uses_exercise_style(self):
        """VanillaOption should use ExerciseStyle enum."""
        option = EuropeanCall(strike=100, maturity=0.5)
        assert isinstance(option.exercise_style, ExerciseStyle)
        assert option.exercise_style == ExerciseStyle.EUROPEAN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
