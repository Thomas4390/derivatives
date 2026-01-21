"""
Model Tests
============

Tests for stochastic models and their characteristic functions.

Author: Thomas
Created: 2025
"""

import pytest
import numpy as np
from typing import Callable

from backend.models.gbm import GBMModel
from backend.models.heston import HestonModel
from backend.models.bates import BatesModel
from backend.models.merton import MertonModel


# =============================================================================
# CHARACTERISTIC FUNCTION PROPERTIES
# =============================================================================

class TestCharacteristicFunctionProperties:
    """Test mathematical properties of characteristic functions."""

    @pytest.fixture
    def models(self, gbm_model, heston_model, bates_model, merton_model, market_atm):
        """Return all models with market for testing."""
        return [
            (gbm_model, market_atm, "GBM"),
            (heston_model, market_atm, "Heston"),
            (bates_model, market_atm, "Bates"),
            (merton_model, market_atm, "Merton"),
        ]

    def test_cf_at_zero_equals_one(self, market_atm, gbm_model, heston_model, bates_model, merton_model):
        """CF(0) = 1 for all models (definition of characteristic function)."""
        models = [gbm_model, heston_model, bates_model, merton_model]
        t = 0.5

        for model in models:
            cf_value = model.characteristic_function(
                u=0.0,
                s0=market_atm.spot,
                t=t,
                r=market_atm.rate,
                q=market_atm.dividend_yield
            )
            np.testing.assert_allclose(
                cf_value, 1.0 + 0j,
                rtol=1e-10,
                err_msg=f"{model.name} CF(0) != 1"
            )

    def test_cf_conjugate_symmetry(self, market_atm, gbm_model, heston_model, bates_model, merton_model):
        """CF(-u) = conj(CF(u)) for real-valued distributions."""
        models = [gbm_model, heston_model, bates_model, merton_model]
        t = 0.5
        u_values = [0.5, 1.0, 2.0, 5.0]

        for model in models:
            for u in u_values:
                cf_pos = model.characteristic_function(
                    u=u, s0=market_atm.spot, t=t, r=market_atm.rate, q=market_atm.dividend_yield
                )
                cf_neg = model.characteristic_function(
                    u=-u, s0=market_atm.spot, t=t, r=market_atm.rate, q=market_atm.dividend_yield
                )
                np.testing.assert_allclose(
                    cf_neg, np.conj(cf_pos),
                    rtol=1e-10,
                    err_msg=f"{model.name} conjugate symmetry failed for u={u}"
                )

    def test_cf_bounded_by_one(self, market_atm, gbm_model, heston_model, bates_model, merton_model):
        """|CF(u)| <= 1 for all u (probability measure)."""
        models = [gbm_model, heston_model, bates_model, merton_model]
        t = 0.5
        u_values = np.linspace(-10, 10, 50)

        for model in models:
            for u in u_values:
                cf_value = model.characteristic_function(
                    u=u, s0=market_atm.spot, t=t, r=market_atm.rate, q=market_atm.dividend_yield
                )
                assert np.abs(cf_value) <= 1.0 + 1e-10, \
                    f"{model.name} |CF({u})| = {np.abs(cf_value)} > 1"

    def test_cf_vectorized_consistency(self, market_atm, gbm_model, heston_model):
        """Vectorized CF should match scalar CF."""
        models = [gbm_model, heston_model]
        t = 0.5
        u_array = np.array([0.0, 0.5, 1.0, 2.0, 5.0])

        for model in models:
            # Vectorized (parameter is u_arr, not u)
            cf_vec = model.characteristic_function_vectorized(
                u_arr=u_array, s0=market_atm.spot, t=t, r=market_atm.rate, q=market_atm.dividend_yield
            )

            # Scalar loop
            cf_scalar = np.array([
                model.characteristic_function(u=u, s0=market_atm.spot, t=t, r=market_atm.rate, q=market_atm.dividend_yield)
                for u in u_array
            ])

            np.testing.assert_allclose(
                cf_vec, cf_scalar,
                rtol=1e-10,
                err_msg=f"{model.name} vectorized vs scalar mismatch"
            )


# =============================================================================
# GBM MODEL TESTS
# =============================================================================

class TestGBMModel:
    """Tests specific to the GBM model."""

    def test_gbm_parameters(self):
        """Test GBM parameter validation."""
        # Valid
        model = GBMModel(sigma=0.2)
        assert model.sigma == 0.2

        # Invalid (negative volatility)
        with pytest.raises(ValueError):
            GBMModel(sigma=-0.1)

    def test_gbm_variance(self, gbm_model):
        """GBM variance equals sigma^2."""
        variance = gbm_model.variance  # Property, not method
        expected = gbm_model.sigma ** 2
        np.testing.assert_allclose(variance, expected, rtol=1e-10)

    def test_gbm_drift_coefficients(self, gbm_model, market_atm):
        """Test drift calculation."""
        S = 100.0
        v = 0.0  # Not used in GBM but required by signature
        t = 0.0

        drift = gbm_model.drift(S, v, t, market_atm.rate, market_atm.dividend_yield)
        expected_drift = (market_atm.rate - market_atm.dividend_yield) * S

        np.testing.assert_allclose(drift, expected_drift, rtol=1e-10)

    def test_gbm_diffusion_coefficients(self, gbm_model, market_atm):
        """Test diffusion calculation."""
        S = 100.0
        v = 0.0  # Not used in GBM but required by signature
        t = 0.0

        diffusion = gbm_model.diffusion(S, v, t)
        expected_diffusion = gbm_model.sigma * S

        np.testing.assert_allclose(diffusion, expected_diffusion, rtol=1e-10)

    def test_gbm_cf_closed_form(self, gbm_model, market_atm):
        """
        Test GBM characteristic function against known closed form.

        For log(S_T) under GBM:
        CF(u) = exp(iu*(ln(s0) + mu*T) - 0.5*u^2*sigma^2*T)
        where mu = r - q - 0.5*sigma^2
        """
        t = 0.5
        r = market_atm.rate
        q = market_atm.dividend_yield
        sigma = gbm_model.sigma
        s0 = market_atm.spot
        u = 2.0

        cf_value = gbm_model.characteristic_function(
            u=u, s0=s0, t=t, r=r, q=q
        )

        # Closed form for log(S_T), includes log(s0)
        log_s0 = np.log(s0)
        drift = r - q - 0.5 * sigma ** 2
        variance = sigma ** 2 * t
        expected = np.exp(1j * u * (log_s0 + drift * t) - 0.5 * variance * u ** 2)

        np.testing.assert_allclose(cf_value, expected, rtol=1e-10)


# =============================================================================
# HESTON MODEL TESTS
# =============================================================================

class TestHestonModel:
    """Tests specific to the Heston model."""

    def test_heston_parameters(self):
        """Test Heston parameter validation."""
        # Valid
        model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        assert model.v0 == 0.04
        assert model.kappa == 2.0

        # Invalid rho (outside [-1, 1])
        with pytest.raises(ValueError):
            HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-1.5)

        # Invalid variance (negative)
        with pytest.raises(ValueError):
            HestonModel(v0=-0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

    def test_feller_condition(self):
        """Test Feller condition checking."""
        # Feller satisfied: 2*kappa*theta > xi^2
        # 2 * 2.0 * 0.04 = 0.16 > 0.09 = 0.3^2
        model_satisfied = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        assert model_satisfied.feller_satisfied is True  # Property

        # Feller not satisfied: 2*kappa*theta < xi^2
        # 2 * 0.5 * 0.04 = 0.04 < 0.81 = 0.9^2
        model_violated = HestonModel(v0=0.04, kappa=0.5, theta=0.04, xi=0.9, rho=-0.7)
        assert model_violated.feller_satisfied is False  # Property

    def test_feller_ratio(self, heston_model):
        """Test Feller ratio calculation."""
        expected_ratio = 2 * heston_model.kappa * heston_model.theta / (heston_model.xi ** 2)
        np.testing.assert_allclose(heston_model.feller_ratio, expected_ratio, rtol=1e-10)  # Property

    def test_long_run_volatility(self, heston_model):
        """Long-run volatility = sqrt(theta)."""
        expected = np.sqrt(heston_model.theta)
        np.testing.assert_allclose(heston_model.long_run_volatility, expected, rtol=1e-10)  # Property

    def test_initial_volatility(self, heston_model):
        """Initial volatility = sqrt(v0)."""
        expected = np.sqrt(heston_model.v0)
        np.testing.assert_allclose(heston_model.initial_volatility, expected, rtol=1e-10)  # Property

    def test_heston_reduces_to_bs_zero_vol_of_vol(self, market_atm):
        """With xi=0, Heston should give similar prices to BS."""
        from backend.engines import BSAnalyticEngine, FFTEngine
        from backend.instruments.options import VanillaOption

        # Heston with zero vol of vol (effectively constant volatility)
        heston_zero_xi = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=1e-6, rho=0.0)
        gbm = GBMModel(sigma=0.2)  # sqrt(0.04) = 0.2

        option = VanillaOption(strike=100.0, maturity=0.5, is_call=True)

        bs_engine = BSAnalyticEngine()
        fft_engine = FFTEngine()

        price_bs = bs_engine.price(option, gbm, market_atm).price
        price_fft = fft_engine.price(option, heston_zero_xi, market_atm).price

        # Should be close (not exact due to different model dynamics)
        np.testing.assert_allclose(price_fft, price_bs, rtol=0.05)


# =============================================================================
# BATES MODEL TESTS
# =============================================================================

class TestBatesModel:
    """Tests specific to the Bates model."""

    def test_bates_parameters(self):
        """Test Bates parameter validation."""
        # Valid
        model = BatesModel(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
        )
        assert model.lambda_j == 0.5

        # Invalid jump intensity (negative)
        with pytest.raises(ValueError):
            BatesModel(
                v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                lambda_j=-0.5, mu_j=-0.1, sigma_j=0.2
            )

    def test_bates_to_heston(self, bates_model):
        """Test conversion to Heston model."""
        heston = bates_model.to_heston()

        assert isinstance(heston, HestonModel)
        assert heston.v0 == bates_model.v0
        assert heston.kappa == bates_model.kappa
        assert heston.theta == bates_model.theta
        assert heston.xi == bates_model.xi
        assert heston.rho == bates_model.rho

    def test_bates_reduces_to_heston(self, market_atm):
        """With lambda_j=0, Bates should equal Heston."""
        from backend.engines import FFTEngine
        from backend.instruments.options import VanillaOption

        heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        bates_no_jumps = BatesModel(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            lambda_j=0.0, mu_j=0.0, sigma_j=0.1
        )

        option = VanillaOption(strike=100.0, maturity=0.5, is_call=True)
        fft_engine = FFTEngine()

        price_heston = fft_engine.price(option, heston, market_atm).price
        price_bates = fft_engine.price(option, bates_no_jumps, market_atm).price

        np.testing.assert_allclose(price_bates, price_heston, rtol=1e-6)

    def test_expected_jump_size(self, bates_model):
        """Test expected jump size calculation."""
        expected = np.exp(bates_model.mu_j + 0.5 * bates_model.sigma_j ** 2) - 1
        np.testing.assert_allclose(bates_model.expected_jump_size, expected, rtol=1e-10)  # Property

    def test_expected_jumps_per_year(self, bates_model):
        """Expected jumps per year = lambda_j."""
        assert bates_model.expected_jumps_per_year() == bates_model.lambda_j


# =============================================================================
# MERTON MODEL TESTS
# =============================================================================

class TestMertonModel:
    """Tests specific to the Merton jump-diffusion model."""

    def test_merton_parameters(self):
        """Test Merton parameter validation."""
        # Valid
        model = MertonModel(sigma=0.2, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)
        assert model.sigma == 0.2
        assert model.lambda_j == 0.5

        # Invalid volatility
        with pytest.raises(ValueError):
            MertonModel(sigma=-0.2, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)

    def test_merton_reduces_to_gbm(self, market_atm):
        """With lambda_j=0, Merton should equal GBM."""
        from backend.engines import FFTEngine, BSAnalyticEngine
        from backend.instruments.options import VanillaOption

        gbm = GBMModel(sigma=0.2)
        merton_no_jumps = MertonModel(sigma=0.2, lambda_j=0.0, mu_j=0.0, sigma_j=0.1)

        option = VanillaOption(strike=100.0, maturity=0.5, is_call=True)

        bs_engine = BSAnalyticEngine()
        fft_engine = FFTEngine()

        price_bs = bs_engine.price(option, gbm, market_atm).price
        price_merton = fft_engine.price(option, merton_no_jumps, market_atm).price

        np.testing.assert_allclose(price_merton, price_bs, rtol=1e-4)

    def test_merton_cf_matches_closed_form(self, merton_model, market_atm):
        """
        Test Merton CF against closed form.

        For Merton model (CF of ln(S_T)):
        CF(u) = exp(iu*(ln(s0) + mu*T) - 0.5*u^2*sigma^2*T) * exp(lambda*T*(E[e^{iu*J}] - 1))
        where E[e^{iu*J}] = exp(iu*mu_j - 0.5*u^2*sigma_j^2)
        and mu = r - q - 0.5*sigma^2 - lambda_j*k (with q adjustment handled separately)
        """
        t = 0.5
        r = market_atm.rate
        q = market_atm.dividend_yield
        sigma = merton_model.sigma
        lambda_j = merton_model.lambda_j
        mu_j = merton_model.mu_j
        sigma_j = merton_model.sigma_j
        s0 = market_atm.spot
        u = 2.0

        cf_value = merton_model.characteristic_function(
            u=u, s0=s0, t=t, r=r, q=q
        )

        # Implementation adjusts s0 for dividend: s0_adj = s0 * exp(-q * t)
        s0_adj = s0 * np.exp(-q * t)

        # Jump compensation: k = E[J - 1] = exp(mu_j + 0.5*sigma_j^2) - 1
        k = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1

        # Drift (without q as it's in s0_adj)
        drift = r - 0.5 * sigma ** 2 - lambda_j * k

        # Diffusion part (includes ln(s0_adj))
        diffusion_exponent = (
            1j * u * np.log(s0_adj) +
            1j * u * drift * t -
            0.5 * sigma ** 2 * u ** 2 * t
        )

        # Jump part
        jump_cf = np.exp(1j * u * mu_j - 0.5 * u ** 2 * sigma_j ** 2)
        jump_exponent = lambda_j * t * (jump_cf - 1)

        expected = np.exp(diffusion_exponent + jump_exponent)

        np.testing.assert_allclose(cf_value, expected, rtol=1e-8)


# =============================================================================
# MODEL COMPARISON TESTS
# =============================================================================

class TestModelComparison:
    """Test that models produce consistent results."""

    def test_model_smile_patterns(self, market_atm):
        """Test that stochastic vol models produce volatility smile."""
        from backend.engines import FFTEngine, BSAnalyticEngine
        from backend.instruments.options import VanillaOption

        gbm = GBMModel(sigma=0.2)
        heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

        fft_engine = FFTEngine()
        bs_engine = BSAnalyticEngine()

        strikes = [85, 90, 95, 100, 105, 110, 115]
        t = 0.5

        ivs_gbm = []
        ivs_heston = []

        for K in strikes:
            option = VanillaOption(strike=K, maturity=t, is_call=True)

            # GBM - should have flat IV
            price_gbm = fft_engine.price(option, gbm, market_atm).price
            iv_gbm = bs_engine.implied_volatility(price_gbm, option, market_atm)
            ivs_gbm.append(iv_gbm)

            # Heston - should have IV smile
            price_heston = fft_engine.price(option, heston, market_atm).price
            iv_heston = bs_engine.implied_volatility(price_heston, option, market_atm)
            ivs_heston.append(iv_heston)

        # GBM should have nearly flat IV (close to input sigma)
        gbm_iv_std = np.std(ivs_gbm)
        assert gbm_iv_std < 0.001, f"GBM IV should be flat, std={gbm_iv_std}"

        # Heston should have IV smile (higher IV for OTM options)
        heston_iv_std = np.std(ivs_heston)
        assert heston_iv_std > 0.001, f"Heston should produce smile, std={heston_iv_std}"

    def test_jump_models_higher_otm_prices(self, market_atm):
        """Jump models should price OTM options higher than diffusion models."""
        from backend.engines import FFTEngine
        from backend.instruments.options import VanillaOption

        gbm = GBMModel(sigma=0.2)
        merton = MertonModel(sigma=0.2, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)

        fft_engine = FFTEngine()

        # Deep OTM put
        otm_put = VanillaOption(strike=80.0, maturity=0.5, is_call=False)

        price_gbm = fft_engine.price(otm_put, gbm, market_atm).price
        price_merton = fft_engine.price(otm_put, merton, market_atm).price

        # Merton should price higher due to crash risk (negative jumps)
        assert price_merton > price_gbm, \
            f"Merton OTM put ({price_merton:.4f}) should > GBM ({price_gbm:.4f})"


# =============================================================================
# NUMERICAL STABILITY TESTS
# =============================================================================

class TestNumericalStability:
    """Test numerical stability of model implementations."""

    def test_cf_large_u(self, gbm_model, heston_model, market_atm):
        """Test CF stability for large u values."""
        t = 0.5
        large_u_values = [10, 50, 100]

        for model in [gbm_model, heston_model]:
            for u in large_u_values:
                cf_value = model.characteristic_function(
                    u=u, s0=market_atm.spot, t=t, r=market_atm.rate, q=market_atm.dividend_yield
                )

                # Should not be NaN or Inf
                assert np.isfinite(cf_value), f"{model.name} CF({u}) = {cf_value} is not finite"

                # Should still satisfy |CF| <= 1
                assert np.abs(cf_value) <= 1.0 + 1e-6, \
                    f"{model.name} |CF({u})| = {np.abs(cf_value)} > 1"

    def test_cf_small_maturity(self, gbm_model, heston_model, market_atm):
        """Test CF stability for very small maturities."""
        u = 1.0
        small_t_values = [1e-4, 1e-6, 1e-8]

        for model in [gbm_model, heston_model]:
            for t in small_t_values:
                cf_value = model.characteristic_function(
                    u=u, s0=market_atm.spot, t=t, r=market_atm.rate, q=market_atm.dividend_yield
                )

                # Should be finite and bounded
                # Note: CF(u) = E[exp(i*u*ln(S_T))] approaches exp(i*u*ln(s0)) as t→0
                # so |CF| should remain <= 1 and be finite
                assert np.isfinite(cf_value), f"{model.name} CF at t={t} is not finite"
                assert np.abs(cf_value) <= 1.0 + 1e-6, f"{model.name} |CF| at t={t} exceeds 1"

    def test_cf_extreme_parameters(self, market_atm):
        """Test CF with extreme but valid parameters."""
        # High volatility
        high_vol_gbm = GBMModel(sigma=1.0)
        cf = high_vol_gbm.characteristic_function(
            u=1.0, s0=market_atm.spot, t=1.0, r=0.05, q=0.0
        )
        assert np.isfinite(cf), "High vol GBM CF should be finite"

        # High vol of vol Heston
        high_xi_heston = HestonModel(v0=0.04, kappa=5.0, theta=0.04, xi=1.5, rho=-0.7)
        cf = high_xi_heston.characteristic_function(
            u=1.0, s0=market_atm.spot, t=1.0, r=0.05, q=0.0
        )
        assert np.isfinite(cf), "High xi Heston CF should be finite"

        # High jump intensity
        high_jump_merton = MertonModel(sigma=0.2, lambda_j=5.0, mu_j=-0.05, sigma_j=0.1)
        cf = high_jump_merton.characteristic_function(
            u=1.0, s0=market_atm.spot, t=1.0, r=0.05, q=0.0
        )
        assert np.isfinite(cf), "High jump intensity Merton CF should be finite"
