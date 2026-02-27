"""
Model Tests
============

Tests for stochastic models and their characteristic functions.

Author: Thomas
Created: 2025
"""

import pytest
import numpy as np

from backend.models.gbm import GBMModel
from backend.models.heston import HestonModel
from backend.models.bates import BatesModel
from backend.models.merton import MertonModel

# Import reporter from conftest
from tests.conftest import report


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

        report.header("CF(0) = 1 Property Test")
        report.info("Validates fundamental CF property: CF(0) must equal 1 by definition")
        report.info("This holds for any probability distribution's characteristic function")
        print("  CF(0) for each model:")
        for model in models:
            cf_value = model.characteristic_function(
                u=0.0,
                s0=market_atm.spot,
                t=t,
                r=market_atm.rate,
                q=market_atm.dividend_yield
            )
            print(f"    {model.name}: {cf_value} (|CF|={np.abs(cf_value):.10f})")

            np.testing.assert_allclose(
                cf_value, 1.0 + 0j,
                rtol=1e-10,
                err_msg=f"{model.name} CF(0) != 1"
            )
        report.success("All models satisfy CF(0) = 1")

    def test_cf_conjugate_symmetry(self, market_atm, gbm_model, heston_model, bates_model, merton_model):
        """CF(-u) = conj(CF(u)) for real-valued distributions."""
        models = [gbm_model, heston_model, bates_model, merton_model]
        t = 0.5
        u_values = [0.5, 1.0, 2.0, 5.0]

        report.header("Conjugate Symmetry Test")
        report.info("Verifies CF(-u) = conj(CF(u)), a property of real-valued random variables")
        report.info("This symmetry is essential for efficient FFT pricing implementations")
        for model in models:
            print(f"  {model.name} - Conjugate symmetry:")
            for u in u_values:
                cf_pos = model.characteristic_function(
                    u=u, s0=market_atm.spot, t=t, r=market_atm.rate, q=market_atm.dividend_yield
                )
                cf_neg = model.characteristic_function(
                    u=-u, s0=market_atm.spot, t=t, r=market_atm.rate, q=market_atm.dividend_yield
                )
                diff = np.abs(cf_neg - np.conj(cf_pos))
                print(f"    u={u}: CF(-u)={cf_neg:.6f}, conj(CF(u))={np.conj(cf_pos):.6f}, diff={diff:.2e}")

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

        report.header("CF Bounded by 1 Test")
        report.info("Verifies |CF(u)| <= 1 for all u, a fundamental probability measure property")
        report.info("CF is the expectation of unit-modulus complex exponentials, so must be bounded")
        for model in models:
            max_abs = 0.0
            for u in u_values:
                cf_value = model.characteristic_function(
                    u=u, s0=market_atm.spot, t=t, r=market_atm.rate, q=market_atm.dividend_yield
                )
                max_abs = max(max_abs, np.abs(cf_value))
                assert np.abs(cf_value) <= 1.0 + 1e-10, \
                    f"{model.name} |CF({u})| = {np.abs(cf_value)} > 1"

            print(f"  {model.name}: max|CF(u)| = {max_abs:.10f} (must be <= 1)")

    def test_cf_vectorized_consistency(self, market_atm, gbm_model, heston_model):
        """Vectorized CF should match scalar CF."""
        models = [gbm_model, heston_model]
        t = 0.5
        u_array = np.array([0.0, 0.5, 1.0, 2.0, 5.0])

        report.header("Vectorized vs Scalar CF Test")
        report.info("Ensures vectorized CF implementation produces identical results to scalar version")
        report.info("Vectorization is critical for FFT performance but must maintain accuracy")
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

            max_diff = np.max(np.abs(cf_vec - cf_scalar))
            print(f"  {model.name}: max diff vectorized vs scalar = {max_diff:.2e}")

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
        report.header("GBM Parameter Validation")
        report.info("Tests that GBM model correctly validates input parameters")
        report.info("Volatility must be non-negative; negative values should raise ValueError")

        # Valid
        model = GBMModel(sigma=0.2)
        print(f"  GBM created with sigma={model.sigma}")
        assert model.sigma == 0.2

        # Invalid (negative volatility)
        print("  Test negative volatility -> should raise ValueError")
        with pytest.raises(ValueError):
            GBMModel(sigma=-0.1)
        report.success("ValueError raised correctly")

    def test_gbm_variance(self, gbm_model):
        """GBM variance equals sigma^2."""
        report.header("GBM Variance Test")
        report.info("Validates that GBM variance property returns sigma squared")
        report.info("For GBM: instantaneous variance = sigma^2 (constant volatility)")

        variance = gbm_model.variance  # Property, not method
        expected = gbm_model.sigma ** 2

        report.value("Variance", variance, expected=expected)

        np.testing.assert_allclose(variance, expected, rtol=1e-10)

    def test_gbm_drift_coefficients(self, gbm_model, market_atm):
        """Test drift calculation."""
        report.header("GBM Drift Coefficients")
        report.info("Tests the drift term for Monte Carlo simulation: drift = (r - q) * S")
        report.info("Under risk-neutral measure, stock grows at risk-free rate minus dividends")

        S = 100.0
        v = 0.0  # Not used in GBM but required by signature
        t = 0.0

        drift = gbm_model.drift(S, v, t, market_atm.rate, market_atm.dividend_yield)
        expected_drift = (market_atm.rate - market_atm.dividend_yield) * S

        report.value("Drift", drift, expected=expected_drift)

        np.testing.assert_allclose(drift, expected_drift, rtol=1e-10)

    def test_gbm_diffusion_coefficients(self, gbm_model, market_atm):
        """Test diffusion calculation."""
        report.header("GBM Diffusion Coefficients")
        report.info("Tests the diffusion term for Monte Carlo: diffusion = sigma * S")
        report.info("This is the volatility component that drives randomness in price paths")

        S = 100.0
        v = 0.0  # Not used in GBM but required by signature
        t = 0.0

        diffusion = gbm_model.diffusion(S, v, t)
        expected_diffusion = gbm_model.sigma * S

        report.value("Diffusion", diffusion, expected=expected_diffusion)

        np.testing.assert_allclose(diffusion, expected_diffusion, rtol=1e-10)

    def test_gbm_cf_closed_form(self, gbm_model, market_atm):
        """
        Test GBM characteristic function against known closed form.

        For log(S_T) under GBM:
        CF(u) = exp(iu*(ln(s0) + mu*T) - 0.5*u^2*sigma^2*T)
        where mu = r - q - 0.5*sigma^2
        """
        report.header("GBM CF Closed Form Validation")
        report.info("Verifies GBM CF implementation against mathematically derived closed form")
        report.info("The CF of log-normal distribution has an explicit formula we can validate against")

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

        report.params(s0=s0, t=t, r=r, q=q, sigma=sigma, u=u)
        print(f"  Computed CF: {cf_value}")
        print(f"  Expected CF: {expected}")
        print(f"  Difference:  {np.abs(cf_value - expected):.2e}")

        np.testing.assert_allclose(cf_value, expected, rtol=1e-10)


# =============================================================================
# HESTON MODEL TESTS
# =============================================================================

class TestHestonModel:
    """Tests specific to the Heston model."""

    def test_heston_parameters(self):
        """Test Heston parameter validation."""
        report.header("Heston Parameter Validation")
        report.info("Tests that Heston model validates all parameters correctly")
        report.info("rho must be in [-1,1], variances (v0, theta) must be non-negative")

        # Valid
        model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        print(f"  Heston created: v0={model.v0}, kappa={model.kappa}, theta={model.theta}, xi={model.xi}, rho={model.rho}")
        assert model.v0 == 0.04
        assert model.kappa == 2.0

        # Invalid rho (outside [-1, 1])
        print("  Test invalid rho (-1.5) -> should raise ValueError")
        with pytest.raises(ValueError):
            HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-1.5)
        report.success("ValueError raised for invalid rho")

        # Invalid variance (negative)
        print("  Test negative variance -> should raise ValueError")
        with pytest.raises(ValueError):
            HestonModel(v0=-0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        report.success("ValueError raised for negative v0")

    def test_feller_condition(self):
        """Test Feller condition checking."""
        report.header("Feller Condition Test")
        report.info("Tests Feller condition: 2*kappa*theta > xi^2 ensures variance stays positive")
        report.info("When violated, variance process can hit zero (important for simulation)")

        # Feller satisfied: 2*kappa*theta > xi^2
        # 2 * 2.0 * 0.04 = 0.16 > 0.09 = 0.3^2
        model_satisfied = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        feller_val_sat = 2 * 2.0 * 0.04
        xi_sq_sat = 0.3**2
        print(f"  Model satisfied: 2*kappa*theta={feller_val_sat:.4f} > xi^2={xi_sq_sat:.4f} -> {model_satisfied.feller_satisfied}")
        assert model_satisfied.feller_satisfied is True  # Property

        # Feller not satisfied: 2*kappa*theta < xi^2
        # 2 * 0.5 * 0.04 = 0.04 < 0.81 = 0.9^2
        model_violated = HestonModel(v0=0.04, kappa=0.5, theta=0.04, xi=0.9, rho=-0.7)
        feller_val_vio = 2 * 0.5 * 0.04
        xi_sq_vio = 0.9**2
        print(f"  Model violated: 2*kappa*theta={feller_val_vio:.4f} < xi^2={xi_sq_vio:.4f} -> {model_violated.feller_satisfied}")
        assert model_violated.feller_satisfied is False  # Property

    def test_feller_ratio(self, heston_model):
        """Test Feller ratio calculation."""
        report.header("Feller Ratio Test")
        report.info("Tests calculation of Feller ratio = 2*kappa*theta / xi^2")
        report.info("Ratio > 1 means Feller condition satisfied; process won't touch zero")

        expected_ratio = 2 * heston_model.kappa * heston_model.theta / (heston_model.xi ** 2)
        actual_ratio = heston_model.feller_ratio

        report.value("Feller Ratio", actual_ratio, expected=expected_ratio)
        report.info(f"Feller satisfied if ratio > 1: {actual_ratio > 1}")

        np.testing.assert_allclose(heston_model.feller_ratio, expected_ratio, rtol=1e-10)  # Property

    def test_long_run_volatility(self, heston_model):
        """Long-run volatility = sqrt(theta)."""
        report.header("Long-Run Volatility Test")
        report.info("Validates long-term volatility = sqrt(theta) for Heston model")
        report.info("theta is the long-term variance; variance mean-reverts to this level")

        expected = np.sqrt(heston_model.theta)
        actual = heston_model.long_run_volatility

        report.value("Long-term vol", actual, expected=expected)

        np.testing.assert_allclose(actual, expected, rtol=1e-10)  # Property

    def test_initial_volatility(self, heston_model):
        """Initial volatility = sqrt(v0)."""
        report.header("Initial Volatility Test")
        report.info("Validates initial volatility = sqrt(v0) for Heston model")
        report.info("v0 is the initial variance at time t=0")

        expected = np.sqrt(heston_model.v0)
        actual = heston_model.initial_volatility

        report.value("Initial vol", actual, expected=expected)

        np.testing.assert_allclose(actual, expected, rtol=1e-10)  # Property

    def test_heston_reduces_to_bs_zero_vol_of_vol(self, market_atm):
        """With xi=0, Heston should give similar prices to BS."""
        report.header("Heston -> BS Reduction Test (xi=0)")
        report.info("When vol-of-vol (xi) approaches zero, Heston becomes constant-vol like GBM")
        report.info("Prices should then match Black-Scholes (within numerical tolerance)")

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

        report.comparison("Heston (xi~0)", price_fft, "BS (GBM)", price_bs, unit="$")

        # Should be close (not exact due to different model dynamics)
        np.testing.assert_allclose(price_fft, price_bs, rtol=0.05)


# =============================================================================
# BATES MODEL TESTS
# =============================================================================

class TestBatesModel:
    """Tests specific to the Bates model."""

    def test_bates_parameters(self):
        """Test Bates parameter validation."""
        report.header("Bates Parameter Validation")
        report.info("Tests that Bates model validates jump and stochastic vol parameters")
        report.info("Jump intensity (lambda_j) must be non-negative")

        # Valid
        model = BatesModel(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
        )
        print(f"  Bates created: lambda_j={model.lambda_j}, mu_j={model.mu_j}, sigma_j={model.sigma_j}")
        assert model.lambda_j == 0.5

        # Invalid jump intensity (negative)
        print("  Test negative jump intensity -> should raise ValueError")
        with pytest.raises(ValueError):
            BatesModel(
                v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                lambda_j=-0.5, mu_j=-0.1, sigma_j=0.2
            )
        report.success("ValueError raised for negative lambda_j")

    def test_bates_to_heston(self, bates_model):
        """Test conversion to Heston model."""
        report.header("Bates to Heston Conversion")
        report.info("Tests extraction of Heston parameters from Bates model")
        report.info("Bates = Heston + jumps, so conversion strips out jump components")

        heston = bates_model.to_heston()

        print("  Bates -> Heston conversion:")
        print(f"    v0:    {heston.v0} == {bates_model.v0}")
        print(f"    kappa: {heston.kappa} == {bates_model.kappa}")
        print(f"    theta: {heston.theta} == {bates_model.theta}")
        print(f"    xi:    {heston.xi} == {bates_model.xi}")
        print(f"    rho:   {heston.rho} == {bates_model.rho}")

        assert isinstance(heston, HestonModel)
        assert heston.v0 == bates_model.v0
        assert heston.kappa == bates_model.kappa
        assert heston.theta == bates_model.theta
        assert heston.xi == bates_model.xi
        assert heston.rho == bates_model.rho

    def test_bates_reduces_to_heston(self, market_atm):
        """With lambda_j=0, Bates should equal Heston."""
        report.header("Bates -> Heston Reduction Test (lambda_j=0)")
        report.info("When jump intensity is zero, Bates should produce identical prices to Heston")
        report.info("This validates the jump component implementation")

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

        report.comparison("Bates (lambda_j=0)", price_bates, "Heston", price_heston, unit="$")

        np.testing.assert_allclose(price_bates, price_heston, rtol=1e-6)

    def test_expected_jump_size(self, bates_model):
        """Test expected jump size calculation."""
        report.header("Expected Jump Size Test")
        report.info("Tests expected jump size: E[J-1] = exp(mu_j + 0.5*sigma_j^2) - 1")
        report.info("Negative values indicate expected downward jumps (crash risk)")

        expected = np.exp(bates_model.mu_j + 0.5 * bates_model.sigma_j ** 2) - 1
        actual = bates_model.expected_jump_size

        report.value("Expected jump size", actual, expected=expected)

        np.testing.assert_allclose(actual, expected, rtol=1e-10)  # Property

    def test_expected_jumps_per_year(self, bates_model):
        """Expected jumps per year = lambda_j."""
        report.header("Expected Jumps Per Year Test")
        report.info("Verifies that expected number of jumps equals lambda_j")
        report.info("lambda_j=0.5 means on average 1 jump every 2 years")

        expected = bates_model.lambda_j
        actual = bates_model.expected_jumps_per_year()

        report.value("Expected jumps/year", actual, expected=expected)

        assert actual == expected


# =============================================================================
# MERTON MODEL TESTS
# =============================================================================

class TestMertonModel:
    """Tests specific to the Merton jump-diffusion model."""

    def test_merton_parameters(self):
        """Test Merton parameter validation."""
        report.header("Merton Parameter Validation")
        report.info("Tests that Merton model validates diffusion and jump parameters")
        report.info("Volatility (sigma) must be non-negative")

        # Valid
        model = MertonModel(sigma=0.2, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)
        print(f"  Merton created: sigma={model.sigma}, lambda_j={model.lambda_j}, mu_j={model.mu_j}, sigma_j={model.sigma_j}")
        assert model.sigma == 0.2
        assert model.lambda_j == 0.5

        # Invalid volatility
        print("  Test negative volatility -> should raise ValueError")
        with pytest.raises(ValueError):
            MertonModel(sigma=-0.2, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)
        report.success("ValueError raised for negative sigma")

    def test_merton_reduces_to_gbm(self, market_atm):
        """With lambda_j=0, Merton should equal GBM."""
        report.header("Merton -> GBM Reduction Test (lambda_j=0)")
        report.info("When jump intensity is zero, Merton should match Black-Scholes/GBM")
        report.info("This validates that the jump-diffusion properly nests the pure diffusion case")

        from backend.engines import FFTEngine, BSAnalyticEngine
        from backend.instruments.options import VanillaOption

        gbm = GBMModel(sigma=0.2)
        merton_no_jumps = MertonModel(sigma=0.2, lambda_j=0.0, mu_j=0.0, sigma_j=0.1)

        option = VanillaOption(strike=100.0, maturity=0.5, is_call=True)

        bs_engine = BSAnalyticEngine()
        fft_engine = FFTEngine()

        price_bs = bs_engine.price(option, gbm, market_atm).price
        price_merton = fft_engine.price(option, merton_no_jumps, market_atm).price

        report.comparison("Merton (lambda_j=0)", price_merton, "BS (GBM)", price_bs, unit="$")

        np.testing.assert_allclose(price_merton, price_bs, rtol=1e-4)

    def test_merton_cf_matches_closed_form(self, merton_model, market_atm):
        """
        Test Merton CF against closed form.

        For Merton model (CF of ln(S_T)):
        CF(u) = exp(iu*(ln(s0) + mu*T) - 0.5*u^2*sigma^2*T) * exp(lambda*T*(E[e^{iu*J}] - 1))
        where E[e^{iu*J}] = exp(iu*mu_j - 0.5*u^2*sigma_j^2)
        and mu = r - q - 0.5*sigma^2 - lambda_j*k (with q adjustment handled separately)
        """
        report.header("Merton CF Closed Form Validation")
        report.info("Verifies Merton CF implementation against mathematically derived formula")
        report.info("The Merton CF combines GBM diffusion with compound Poisson jump process")

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

        print("  Merton parameters:")
        report.params(s0=s0, t=t, r=r, q=q, sigma=sigma, lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j, u=u)
        print(f"  Computed CF: {cf_value}")
        print(f"  Expected CF: {expected}")
        print(f"  Difference:  {np.abs(cf_value - expected):.2e}")

        np.testing.assert_allclose(cf_value, expected, rtol=1e-8)


# =============================================================================
# MODEL COMPARISON TESTS
# =============================================================================

class TestModelComparison:
    """Test that models produce consistent results."""

    def test_model_smile_patterns(self, market_atm):
        """Test that stochastic vol models produce volatility smile."""
        report.header("Implied Volatility Smile Test")
        report.info("Tests that Heston produces implied volatility smile while GBM stays flat")
        report.info("The smile pattern is a key feature that distinguishes stochastic vol models")

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

        rows = [(k, iv_g, iv_h) for k, iv_g, iv_h in zip(strikes, ivs_gbm, ivs_heston)]
        report.table(
            ["Strike", "IV GBM", "IV Heston"],
            rows,
            title="Implied Volatility Surface",
            precision=4
        )

        # GBM should have nearly flat IV (close to input sigma)
        gbm_iv_std = np.std(ivs_gbm)
        print(f"  GBM IV std:    {gbm_iv_std:.6f} (should be < 0.001)")
        assert gbm_iv_std < 0.001, f"GBM IV should be flat, std={gbm_iv_std}"

        # Heston should have IV smile (higher IV for OTM options)
        heston_iv_std = np.std(ivs_heston)
        print(f"  Heston IV std: {heston_iv_std:.6f} (should be > 0.001)")
        assert heston_iv_std > 0.001, f"Heston should produce smile, std={heston_iv_std}"

    def test_jump_models_higher_otm_prices(self, market_atm):
        """Jump models should price OTM options higher than diffusion models."""
        report.header("Jump Model OTM Pricing Test")
        report.info("Jump models should price deep OTM puts higher than pure diffusion models")
        report.info("Negative jumps create crash risk that increases OTM put values")

        from backend.engines import FFTEngine
        from backend.instruments.options import VanillaOption

        gbm = GBMModel(sigma=0.2)
        merton = MertonModel(sigma=0.2, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)

        fft_engine = FFTEngine()

        # Deep OTM put
        otm_put = VanillaOption(strike=80.0, maturity=0.5, is_call=False)

        price_gbm = fft_engine.price(otm_put, gbm, market_atm).price
        price_merton = fft_engine.price(otm_put, merton, market_atm).price

        report.comparison("Merton (OTM put)", price_merton, "GBM (OTM put)", price_gbm, unit="$")
        report.info("Merton should price higher (crash risk)")

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
        report.header("CF Stability for Large u")
        report.info("Tests that CF remains finite and bounded for large frequency values")
        report.info("FFT pricing requires stable CF behavior across wide u range")

        t = 0.5
        large_u_values = [10, 50, 100]

        for model in [gbm_model, heston_model]:
            print(f"  {model.name} - Stability for large u:")
            for u in large_u_values:
                cf_value = model.characteristic_function(
                    u=u, s0=market_atm.spot, t=t, r=market_atm.rate, q=market_atm.dividend_yield
                )
                abs_cf = np.abs(cf_value)
                print(f"    u={u:3d}: |CF|={abs_cf:.10f}, finite={np.isfinite(cf_value)}")

                # Should not be NaN or Inf
                assert np.isfinite(cf_value), f"{model.name} CF({u}) = {cf_value} is not finite"

                # Should still satisfy |CF| <= 1
                assert abs_cf <= 1.0 + 1e-6, \
                    f"{model.name} |CF({u})| = {abs_cf} > 1"

    def test_cf_small_maturity(self, gbm_model, heston_model, market_atm):
        """Test CF stability for very small maturities."""
        report.header("CF Stability for Small Maturity")
        report.info("Tests CF behavior as maturity approaches zero")
        report.info("CF should converge to exp(i*u*ln(s0)) and remain finite")

        u = 1.0
        small_t_values = [1e-4, 1e-6, 1e-8]

        for model in [gbm_model, heston_model]:
            print(f"  {model.name} - Stability for small maturity:")
            for t in small_t_values:
                cf_value = model.characteristic_function(
                    u=u, s0=market_atm.spot, t=t, r=market_atm.rate, q=market_atm.dividend_yield
                )
                abs_cf = np.abs(cf_value)
                print(f"    t={t:.0e}: |CF|={abs_cf:.10f}, finite={np.isfinite(cf_value)}")

                # Should be finite and bounded
                # Note: CF(u) = E[exp(i*u*ln(S_T))] approaches exp(i*u*ln(s0)) as t->0
                # so |CF| should remain <= 1 and be finite
                assert np.isfinite(cf_value), f"{model.name} CF at t={t} is not finite"
                assert abs_cf <= 1.0 + 1e-6, f"{model.name} |CF| at t={t} exceeds 1"

    def test_cf_extreme_parameters(self, market_atm):
        """Test CF with extreme but valid parameters."""
        report.header("CF with Extreme Parameters")
        report.info("Tests CF stability with extreme but valid parameter values")
        report.info("High volatility, high vol-of-vol, high jump intensity should all remain finite")
        print("  Test with extreme parameters:")

        # High volatility
        high_vol_gbm = GBMModel(sigma=1.0)
        cf = high_vol_gbm.characteristic_function(
            u=1.0, s0=market_atm.spot, t=1.0, r=0.05, q=0.0
        )
        print(f"    GBM sigma=100%: CF={cf}, finite={np.isfinite(cf)}")
        assert np.isfinite(cf), "High vol GBM CF should be finite"

        # High vol of vol Heston
        high_xi_heston = HestonModel(v0=0.04, kappa=5.0, theta=0.04, xi=1.5, rho=-0.7)
        cf = high_xi_heston.characteristic_function(
            u=1.0, s0=market_atm.spot, t=1.0, r=0.05, q=0.0
        )
        print(f"    Heston xi=1.5: CF={cf}, finite={np.isfinite(cf)}")
        assert np.isfinite(cf), "High xi Heston CF should be finite"

        # High jump intensity
        high_jump_merton = MertonModel(sigma=0.2, lambda_j=5.0, mu_j=-0.05, sigma_j=0.1)
        cf = high_jump_merton.characteristic_function(
            u=1.0, s0=market_atm.spot, t=1.0, r=0.05, q=0.0
        )
        print(f"    Merton lambda_j=5: CF={cf}, finite={np.isfinite(cf)}")
        assert np.isfinite(cf), "High jump intensity Merton CF should be finite"

        report.success("All CFs with extreme parameters are finite")
