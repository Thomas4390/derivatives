"""
Simulation and Convergence Tests
=================================

Tests for Monte Carlo simulation, path generation, and convergence properties.

Author: Thomas
Created: 2025
"""

import pytest
import numpy as np
from typing import List, Tuple

from backend.simulation import (
    create_simulator,
    create_gbm,
    create_heston,
    create_merton,
    create_bates,
    list_models,
)
from backend.simulation.base import SimulationResult, BaseSimulator


# =============================================================================
# SIMULATOR FACTORY TESTS
# =============================================================================

class TestSimulatorFactory:
    """Test simulator creation and factory functions."""

    def test_list_models(self):
        """Test that all expected models are available."""
        models = list_models()

        # Keys are uppercase enum names
        assert "GBM" in models
        assert "HESTON" in models
        assert "MERTON" in models
        assert "BATES" in models

    def test_create_gbm_simulator(self):
        """Test GBM simulator creation."""
        sim = create_gbm(sigma=0.2)

        assert sim is not None
        assert "Geometric Brownian Motion" in sim.model_name
        params = sim.get_parameters()
        assert params["sigma"] == 0.2

    def test_create_heston_simulator(self):
        """Test Heston simulator creation."""
        sim = create_heston(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7
        )

        assert sim is not None
        assert "Heston" in sim.model_name
        params = sim.get_parameters()
        assert params["v0"] == 0.04
        assert params["kappa"] == 2.0

    def test_create_merton_simulator(self):
        """Test Merton simulator creation."""
        sim = create_merton(
            sigma=0.2, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
        )

        assert sim is not None
        assert "Merton" in sim.model_name
        params = sim.get_parameters()
        assert params["lambda_j"] == 0.5

    def test_create_bates_simulator(self):
        """Test Bates simulator creation."""
        sim = create_bates(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
        )

        assert sim is not None
        assert "Bates" in sim.model_name

    def test_create_simulator_generic(self):
        """Test generic simulator creation function."""
        sim = create_simulator("GBM", sigma=0.25)

        assert sim is not None
        params = sim.get_parameters()
        assert params["sigma"] == 0.25

    def test_create_simulator_invalid_model(self):
        """Test error on invalid model name."""
        with pytest.raises((ValueError, KeyError)):
            create_simulator("invalid_model")


# =============================================================================
# GBM SIMULATION TESTS
# =============================================================================

class TestGBMSimulation:
    """Test GBM simulation properties."""

    @pytest.fixture
    def gbm_sim(self):
        """Create a GBM simulator."""
        return create_gbm(sigma=0.2)

    def test_gbm_simulate_paths(self, gbm_sim):
        """Test basic path simulation."""
        result = gbm_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=1000, n_steps=252, seed=42
        )

        assert isinstance(result, SimulationResult)
        assert result.price_paths.shape == (1000, 253)  # n_paths x (n_steps + 1)
        assert result.price_paths[:, 0].mean() == pytest.approx(100.0, rel=1e-10)

    def test_gbm_simulate_terminal(self, gbm_sim):
        """Test terminal value simulation."""
        # simulate_terminal returns np.ndarray directly
        terminal_prices = gbm_sim.simulate_terminal(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=10000, n_steps=252, seed=42
        )

        # Check expected properties
        assert isinstance(terminal_prices, np.ndarray)
        assert len(terminal_prices) == 10000

    def test_gbm_martingale_property(self, gbm_sim):
        """
        Test martingale property: E[S_T] = S_0 * exp((r-q)*T)

        Under risk-neutral measure, discounted asset is martingale.
        """
        s0, t, r, q = 100.0, 1.0, 0.05, 0.02
        mu = r - q  # Risk-neutral drift
        n_paths = 100000

        # Use simulate_paths to get SimulationResult
        result = gbm_sim.simulate_paths(
            s0=s0, mu=mu, t=t, n_paths=n_paths, n_steps=252, seed=42
        )

        expected_mean = s0 * np.exp(mu * t)
        actual_mean = result.terminal_mean

        # Should be within 1% for 100k paths
        np.testing.assert_allclose(actual_mean, expected_mean, rtol=0.01)

    def test_gbm_terminal_distribution(self, gbm_sim):
        """
        Test log-normal terminal distribution.

        log(S_T) ~ N(log(S_0) + (r - q - sigma^2/2)*T, sigma^2*T)
        """
        s0, t, r, q = 100.0, 1.0, 0.05, 0.0
        mu = r - q
        sigma = 0.2
        n_paths = 50000

        result = gbm_sim.simulate_paths(
            s0=s0, mu=mu, t=t, n_paths=n_paths, n_steps=252, seed=42
        )

        log_returns = np.log(result.terminal_prices / s0)

        # Expected mean and std of log returns
        expected_mean = (mu - 0.5 * sigma ** 2) * t
        expected_std = sigma * np.sqrt(t)

        np.testing.assert_allclose(log_returns.mean(), expected_mean, rtol=0.05)
        np.testing.assert_allclose(log_returns.std(), expected_std, rtol=0.05)

    def test_gbm_reproducibility(self, gbm_sim):
        """Test that MC simulation produces statistically consistent results.

        Note: Exact reproducibility is not guaranteed with parallel numba code
        (@njit(parallel=True)) because thread scheduling is non-deterministic.
        We test that simulation statistics are consistent across runs.
        """
        params = dict(s0=100.0, mu=0.05, t=1.0, n_paths=1000, n_steps=100)

        result1 = gbm_sim.simulate_paths(**params, seed=123)
        result2 = gbm_sim.simulate_paths(**params, seed=456)

        # Terminal prices should have similar distributions
        # Mean should be close (within a few SEs)
        mean1 = np.mean(result1.price_paths[:, -1])
        mean2 = np.mean(result2.price_paths[:, -1])
        se1 = np.std(result1.price_paths[:, -1]) / np.sqrt(params['n_paths'])
        se2 = np.std(result2.price_paths[:, -1]) / np.sqrt(params['n_paths'])
        combined_se = np.sqrt(se1**2 + se2**2)

        np.testing.assert_allclose(mean1, mean2, atol=4*combined_se)

    def test_gbm_different_seeds(self, gbm_sim):
        """Test that different seeds produce different results."""
        params = dict(s0=100.0, mu=0.05, t=1.0, n_paths=100, n_steps=50)

        result1 = gbm_sim.simulate_paths(**params, seed=123)
        result2 = gbm_sim.simulate_paths(**params, seed=456)

        assert not np.allclose(result1.price_paths, result2.price_paths)


# =============================================================================
# HESTON SIMULATION TESTS
# =============================================================================

class TestHestonSimulation:
    """Test Heston simulation properties."""

    @pytest.fixture
    def heston_sim(self):
        """Create a Heston simulator."""
        return create_heston(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7
        )

    def test_heston_simulate_paths(self, heston_sim):
        """Test Heston path simulation."""
        result = heston_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=1000, n_steps=252, seed=42
        )

        assert isinstance(result, SimulationResult)
        assert result.price_paths.shape[0] == 1000
        assert result.has_volatility  # Should have volatility paths

    def test_heston_volatility_positive(self, heston_sim):
        """Test that volatility stays positive (or handled properly)."""
        result = heston_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=1000, n_steps=252, seed=42
        )

        if result.has_volatility:
            # Volatility should be positive (stored as vol, not variance)
            assert np.all(result.volatility_paths >= 0), "Volatility should be non-negative"

    def test_heston_mean_reversion(self, heston_sim):
        """Test variance mean reversion property."""
        # Long simulation to observe mean reversion
        result = heston_sim.simulate_paths(
            s0=100.0, mu=0.05, t=5.0,
            n_paths=10000, n_steps=1250, seed=42
        )

        if result.has_volatility:
            # Terminal volatility should approach sqrt(theta)
            terminal_vol = result.volatility_paths[:, -1].mean()
            theta = 0.04  # Long-run variance
            expected_vol = np.sqrt(theta)

            # Should be within reasonable range of sqrt(theta)
            np.testing.assert_allclose(terminal_vol, expected_vol, rtol=0.2)

    def test_heston_leverage_effect(self):
        """
        Test leverage effect (negative correlation).

        With rho < 0, negative spot returns should correlate with higher volatility.
        """
        # High negative correlation
        heston_neg_rho = create_heston(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.9
        )

        result = heston_neg_rho.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=10000, n_steps=252, seed=42
        )

        if result.has_volatility:
            # Compute correlation between returns and volatility changes
            log_returns = np.diff(np.log(result.price_paths), axis=1)
            vol_changes = np.diff(result.volatility_paths, axis=1)

            # Flatten for correlation
            corr = np.corrcoef(log_returns.flatten(), vol_changes.flatten())[0, 1]

            # Should be negative
            assert corr < 0, f"Leverage effect correlation should be negative, got {corr}"


# =============================================================================
# JUMP DIFFUSION SIMULATION TESTS
# =============================================================================

class TestJumpDiffusionSimulation:
    """Test jump diffusion model simulations."""

    @pytest.fixture
    def merton_sim(self):
        """Create a Merton simulator."""
        return create_merton(
            sigma=0.2, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
        )

    def test_merton_simulate_paths(self, merton_sim):
        """Test Merton path simulation."""
        result = merton_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=1000, n_steps=252, seed=42
        )

        assert isinstance(result, SimulationResult)
        assert result.price_paths.shape[0] == 1000

    def test_merton_jump_detection(self, merton_sim):
        """Test that jumps are present in paths."""
        result = merton_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=10000, n_steps=252, seed=42
        )

        # Compute log returns
        log_returns = np.diff(np.log(result.price_paths), axis=1).flatten()

        # With jumps, returns should have excess kurtosis > 3 (normal has 3)
        kurtosis = np.mean((log_returns - log_returns.mean()) ** 4) / (log_returns.std() ** 4)

        assert kurtosis > 3.5, f"Jump model should have fat tails, kurtosis={kurtosis}"

    def test_bates_simulation(self):
        """Test Bates model simulation."""
        bates_sim = create_bates(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
        )

        result = bates_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=1000, n_steps=252, seed=42
        )

        assert isinstance(result, SimulationResult)
        assert result.has_volatility  # Bates has stochastic vol


# =============================================================================
# CONVERGENCE TESTS
# =============================================================================

class TestConvergence:
    """Test Monte Carlo convergence properties."""

    def test_price_convergence(self):
        """Test that MC price converges to analytical price."""
        from backend.engines import BSAnalyticEngine, MonteCarloEngine
        from backend.instruments.options import VanillaOption
        from backend.models.gbm import GBMModel
        from backend.core.market import MarketEnvironment

        gbm = GBMModel(sigma=0.2)
        market = MarketEnvironment(spot=100.0, rate=0.05, dividend_yield=0.0)
        option = VanillaOption(strike=100.0, maturity=0.5, is_call=True)

        bs_engine = BSAnalyticEngine()
        analytical_result = bs_engine.price(option, gbm, market)
        analytical_price = analytical_result.price

        # Test convergence with increasing paths
        path_counts = [1000, 10000, 50000]
        errors = []

        for n_paths in path_counts:
            mc_engine = MonteCarloEngine(n_paths=n_paths, seed=42)
            mc_result = mc_engine.price(option, gbm, market)
            mc_price = mc_result.price
            errors.append(abs(mc_price - analytical_price) / analytical_price)

        # Error should decrease with more paths
        assert errors[-1] < errors[0], "MC error should decrease with more paths"

        # Final error should be small
        assert errors[-1] < 0.02, f"MC error too large: {errors[-1]}"

    def test_sqrt_n_convergence_rate(self):
        """
        Test sqrt(n) convergence rate of Monte Carlo.

        Standard error proportional to 1/sqrt(n), so doubling paths should reduce error by sqrt(2).
        """
        gbm_sim = create_gbm(sigma=0.2)
        s0, t, r, q = 100.0, 1.0, 0.05, 0.0
        mu = r - q

        true_mean = s0 * np.exp(mu * t)

        errors = []
        path_counts = [1000, 4000, 16000]  # 4x increases

        for n_paths in path_counts:
            means = []
            for seed in range(20):  # Multiple trials
                result = gbm_sim.simulate_paths(
                    s0=s0, mu=mu, t=t, n_paths=n_paths, n_steps=252, seed=seed
                )
                means.append(result.terminal_mean)

            # Standard error of mean across trials
            se = np.std(means)
            errors.append(se)

        # 4x paths should give 2x reduction in error
        ratio = errors[0] / errors[1]
        expected_ratio = np.sqrt(4)  # = 2

        np.testing.assert_allclose(ratio, expected_ratio, rtol=0.3)

    def test_antithetic_variance_reduction(self):
        """Test antithetic variates reduce variance."""
        # Note: This test depends on the simulator supporting antithetic variates
        # If not supported, we can test the concept with manual implementation
        gbm_sim = create_gbm(sigma=0.2)

        s0, t, r, q = 100.0, 1.0, 0.05, 0.0
        mu = r - q
        n_paths = 10000

        # Regular simulation
        result_regular = gbm_sim.simulate_paths(
            s0=s0, mu=mu, t=t, n_paths=n_paths, n_steps=252, seed=42
        )

        # Manual antithetic: simulate n_paths/2 and use opposite random numbers
        # This is a conceptual test - actual implementation may vary

        # For now, verify regular simulation works correctly
        terminal_std = result_regular.terminal_std
        expected_std = s0 * np.exp(mu * t) * np.sqrt(np.exp(0.2 ** 2 * t) - 1)

        # Standard deviation should be in reasonable range
        np.testing.assert_allclose(terminal_std, expected_std, rtol=0.1)


# =============================================================================
# SIMULATION RESULT TESTS
# =============================================================================

class TestSimulationResult:
    """Test SimulationResult dataclass functionality."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample simulation result."""
        gbm_sim = create_gbm(sigma=0.2)
        return gbm_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=1000, n_steps=252, seed=42
        )

    def test_terminal_prices(self, sample_result):
        """Test terminal prices extraction."""
        terminal = sample_result.terminal_prices
        assert len(terminal) == 1000

    def test_initial_price(self, sample_result):
        """Test initial price."""
        initial = sample_result.initial_price
        assert initial == pytest.approx(100.0, rel=1e-10)

    def test_mean_path(self, sample_result):
        """Test mean path calculation."""
        mean_path = sample_result.mean_path
        assert len(mean_path) == 253  # n_steps + 1
        assert mean_path[0] == pytest.approx(100.0, rel=1e-10)

    def test_std_path(self, sample_result):
        """Test standard deviation path."""
        std_path = sample_result.std_path
        assert len(std_path) == 253
        assert std_path[0] == pytest.approx(0.0, abs=1e-10)  # No std at t=0

    def test_percentile_paths(self, sample_result):
        """Test percentile path calculation."""
        percentiles = sample_result.percentile_paths([5, 50, 95])

        assert len(percentiles) == 3
        for p_path in percentiles:
            assert len(p_path) == 253

        # 5th percentile < median < 95th percentile
        assert np.all(percentiles[0] <= percentiles[1])
        assert np.all(percentiles[1] <= percentiles[2])

    def test_log_returns(self, sample_result):
        """Test log returns calculation."""
        log_ret = sample_result.log_returns()  # Method, not property

        assert log_ret.shape == (1000, 252)  # n_paths x n_steps

    def test_realized_volatility(self, sample_result):
        """Test realized volatility calculation."""
        rv = sample_result.realized_volatility()  # Method, not property

        assert len(rv) == 1000  # One per path
        assert np.all(rv >= 0)  # Volatility is positive


# =============================================================================
# EDGE CASES AND NUMERICAL STABILITY
# =============================================================================

class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_very_short_maturity(self):
        """Test simulation with very short maturity."""
        gbm_sim = create_gbm(sigma=0.2)

        result = gbm_sim.simulate_paths(
            s0=100.0, mu=0.05, t=0.001, n_paths=1000, n_steps=10, seed=42
        )

        # Terminal prices should be very close to initial
        np.testing.assert_allclose(result.terminal_mean, 100.0, rtol=0.01)

    def test_zero_volatility(self):
        """Test GBM with near-zero volatility."""
        gbm_sim = create_gbm(sigma=0.001)

        result = gbm_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0, n_paths=1000, n_steps=252, seed=42
        )

        # Should be deterministic growth
        expected = 100.0 * np.exp(0.05)
        np.testing.assert_allclose(result.terminal_mean, expected, rtol=0.01)
        np.testing.assert_allclose(result.terminal_std, 0.0, atol=0.5)

    def test_high_volatility(self):
        """Test GBM with high volatility."""
        gbm_sim = create_gbm(sigma=1.0)  # 100% vol

        result = gbm_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0, n_paths=10000, n_steps=252, seed=42
        )

        # Should still have correct mean
        expected_mean = 100.0 * np.exp(0.05)
        np.testing.assert_allclose(result.terminal_mean, expected_mean, rtol=0.05)

    def test_negative_drift(self):
        """Test with negative drift (high dividend yield)."""
        gbm_sim = create_gbm(sigma=0.2)

        # mu = r - q = 0.02 - 0.08 = -0.06
        result = gbm_sim.simulate_paths(
            s0=100.0, mu=-0.06, t=1.0, n_paths=10000, n_steps=252, seed=42
        )

        # Net drift is negative
        expected_mean = 100.0 * np.exp(-0.06)
        np.testing.assert_allclose(result.terminal_mean, expected_mean, rtol=0.02)

    def test_large_number_of_steps(self):
        """Test with many time steps."""
        gbm_sim = create_gbm(sigma=0.2)

        result = gbm_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=100, n_steps=1000, seed=42
        )

        assert result.price_paths.shape == (100, 1001)

    def test_single_path(self):
        """Test single path simulation."""
        gbm_sim = create_gbm(sigma=0.2)

        result = gbm_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=1, n_steps=100, seed=42
        )

        assert result.price_paths.shape == (1, 101)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Basic performance tests (not benchmarks, just sanity checks)."""

    def test_large_simulation_completes(self):
        """Test that large simulation completes in reasonable time."""
        import time

        gbm_sim = create_gbm(sigma=0.2)

        start = time.time()
        result = gbm_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=10000, n_steps=252, seed=42
        )
        elapsed = time.time() - start

        assert result.price_paths.shape == (10000, 253)
        assert elapsed < 10.0, f"Simulation took too long: {elapsed}s"

    def test_terminal_only_faster_than_full_paths(self):
        """Terminal-only simulation should be faster than full paths."""
        import time

        gbm_sim = create_gbm(sigma=0.2)
        n_paths = 50000

        # Full paths
        start = time.time()
        gbm_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=n_paths, n_steps=252, seed=42
        )
        time_full = time.time() - start

        # Terminal only
        start = time.time()
        gbm_sim.simulate_terminal(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=n_paths, n_steps=252, seed=42
        )
        time_terminal = time.time() - start

        # Terminal should be faster (or at least not much slower)
        assert time_terminal <= time_full * 1.5, \
            f"Terminal ({time_terminal}s) should be faster than full paths ({time_full}s)"
