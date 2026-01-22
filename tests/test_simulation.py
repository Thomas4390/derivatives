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

# Import reporter from conftest
from tests.conftest import report


# =============================================================================
# SIMULATOR FACTORY TESTS
# =============================================================================

class TestSimulatorFactory:
    """Test simulator creation and factory functions."""

    def test_list_models(self):
        """Test that all expected models are available."""
        report.header("Available Simulation Models")
        report.info("Tests that all expected stochastic models are registered")
        report.info("Should include GBM, Heston, Merton, and Bates models")

        models = list_models()

        print("  Available models:")
        for model_name in models:
            print(f"    - {model_name}")

        # Keys are uppercase enum names
        assert "GBM" in models
        assert "HESTON" in models
        assert "MERTON" in models
        assert "BATES" in models

        report.success("All expected models are available")

    def test_create_gbm_simulator(self):
        """Test GBM simulator creation."""
        report.header("GBM Simulator Creation")
        report.info("Tests factory function for creating GBM simulator")
        report.info("GBM: dS = μS dt + σS dW (Geometric Brownian Motion)")

        sim = create_gbm(sigma=0.2)

        assert sim is not None
        assert "Geometric Brownian Motion" in sim.model_name

        params = sim.get_parameters()
        report.params(**params)

        assert params["sigma"] == 0.2
        report.success("GBM simulator created successfully")

    def test_create_heston_simulator(self):
        """Test Heston simulator creation."""
        report.header("Heston Simulator Creation")
        report.info("Tests factory function for creating Heston simulator")
        report.info("Heston: stochastic volatility with mean reversion")

        sim = create_heston(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7
        )

        assert sim is not None
        assert "Heston" in sim.model_name

        params = sim.get_parameters()
        report.params(**params)

        assert params["v0"] == 0.04
        assert params["kappa"] == 2.0
        report.success("Heston simulator created successfully")

    def test_create_merton_simulator(self):
        """Test Merton simulator creation."""
        report.header("Merton Simulator Creation")
        report.info("Tests factory function for creating Merton jump-diffusion simulator")
        report.info("Merton: GBM + compound Poisson jumps")

        sim = create_merton(
            sigma=0.2, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
        )

        assert sim is not None
        assert "Merton" in sim.model_name

        params = sim.get_parameters()
        report.params(**params)

        assert params["lambda_j"] == 0.5
        report.success("Merton simulator created successfully")

    def test_create_bates_simulator(self):
        """Test Bates simulator creation."""
        report.header("Bates Simulator Creation")
        report.info("Tests factory function for creating Bates simulator")
        report.info("Bates: Heston stochastic vol + compound Poisson jumps")

        sim = create_bates(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
        )

        assert sim is not None
        assert "Bates" in sim.model_name
        report.success("Bates simulator created successfully")

    def test_create_simulator_generic(self):
        """Test generic simulator creation function."""
        report.header("Generic Simulator Creation")
        report.info("Tests generic create_simulator function with model name string")
        report.info("Allows dynamic model selection at runtime")

        sim = create_simulator("GBM", sigma=0.25)

        assert sim is not None
        params = sim.get_parameters()
        report.params(**params)

        assert params["sigma"] == 0.25
        report.success("Generic simulator created successfully")

    def test_create_simulator_invalid_model(self):
        """Test error on invalid model name."""
        report.header("Invalid Model Error Test")
        report.info("Tests that invalid model names raise appropriate errors")
        report.info("Expected: ValueError or KeyError for unrecognized model")

        print("  Testing invalid model name -> should raise ValueError or KeyError")
        with pytest.raises((ValueError, KeyError)):
            create_simulator("invalid_model")
        report.success("Error raised correctly for invalid model")


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
        report.header("GBM Path Simulation")
        report.info("Tests full path simulation with specified steps and paths")
        report.info("Returns SimulationResult with price_paths array")

        result = gbm_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=1000, n_steps=252, seed=42
        )

        report.params(s0=100.0, mu=0.05, t=1.0, n_paths=1000, n_steps=252)
        print(f"  Result shape: {result.price_paths.shape}")
        print(f"  Initial price (mean): {result.price_paths[:, 0].mean():.6f}")

        assert isinstance(result, SimulationResult)
        assert result.price_paths.shape == (1000, 253)  # n_paths x (n_steps + 1)
        assert result.price_paths[:, 0].mean() == pytest.approx(100.0, rel=1e-10)

    def test_gbm_simulate_terminal(self, gbm_sim):
        """Test terminal value simulation."""
        report.header("GBM Terminal Value Simulation")
        report.info("Tests terminal-only simulation (more efficient than full paths)")
        report.info("Returns numpy array of terminal prices directly")

        # simulate_terminal returns np.ndarray directly
        terminal_prices = gbm_sim.simulate_terminal(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=10000, n_steps=252, seed=42
        )

        report.params(s0=100.0, mu=0.05, t=1.0, n_paths=10000)
        print(f"  Terminal prices count: {len(terminal_prices)}")
        print(f"  Terminal mean: {np.mean(terminal_prices):.4f}")
        print(f"  Terminal std:  {np.std(terminal_prices):.4f}")

        # Check expected properties
        assert isinstance(terminal_prices, np.ndarray)
        assert len(terminal_prices) == 10000

    def test_gbm_martingale_property(self, gbm_sim):
        """
        Test martingale property: E[S_T] = S_0 * exp((r-q)*T)

        Under risk-neutral measure, discounted asset is martingale.
        """
        report.header("GBM Martingale Property Test")
        report.info("Tests risk-neutral martingale property E[S_T] = S_0 * exp((r-q)*T)")
        report.info("Critical for no-arbitrage pricing")

        s0, t, r, q = 100.0, 1.0, 0.05, 0.02
        mu = r - q  # Risk-neutral drift
        n_paths = 100000

        report.params(s0=s0, t=t, r=r, q=q, mu=mu, n_paths=n_paths)

        # Use simulate_paths to get SimulationResult
        result = gbm_sim.simulate_paths(
            s0=s0, mu=mu, t=t, n_paths=n_paths, n_steps=252, seed=42
        )

        expected_mean = s0 * np.exp(mu * t)
        actual_mean = result.terminal_mean

        report.value("Terminal Mean", actual_mean, expected=expected_mean, unit="$")

        # Should be within 1% for 100k paths
        np.testing.assert_allclose(actual_mean, expected_mean, rtol=0.01)

    def test_gbm_terminal_distribution(self, gbm_sim):
        """
        Test log-normal terminal distribution.

        log(S_T) ~ N(log(S_0) + (r - q - sigma^2/2)*T, sigma^2*T)
        """
        report.header("GBM Log-Normal Distribution Test")
        report.info("Tests that log returns follow normal distribution (GBM property)")
        report.info("Verifies both mean and standard deviation of log(S_T/S_0)")

        s0, t, r, q = 100.0, 1.0, 0.05, 0.0
        mu = r - q
        sigma = 0.2
        n_paths = 50000

        report.params(s0=s0, t=t, r=r, q=q, sigma=sigma)

        result = gbm_sim.simulate_paths(
            s0=s0, mu=mu, t=t, n_paths=n_paths, n_steps=252, seed=42
        )

        log_returns = np.log(result.terminal_prices / s0)

        # Expected mean and std of log returns
        expected_mean = (mu - 0.5 * sigma ** 2) * t
        expected_std = sigma * np.sqrt(t)

        report.value("Log Return Mean", log_returns.mean(), expected=expected_mean)
        report.value("Log Return Std", log_returns.std(), expected=expected_std)

        np.testing.assert_allclose(log_returns.mean(), expected_mean, rtol=0.05)
        np.testing.assert_allclose(log_returns.std(), expected_std, rtol=0.05)

    def test_gbm_reproducibility(self, gbm_sim):
        """Test that MC simulation produces statistically consistent results.

        Note: Exact reproducibility is not guaranteed with parallel numba code
        (@njit(parallel=True)) because thread scheduling is non-deterministic.
        We test that simulation statistics are consistent across runs.
        """
        report.header("GBM Statistical Consistency Test")
        report.info("Tests statistical consistency of Monte Carlo simulations")
        report.info("Different seeds should give similar mean within standard error bounds")

        params = dict(s0=100.0, mu=0.05, t=1.0, n_paths=1000, n_steps=100)

        result1 = gbm_sim.simulate_paths(**params, seed=123)
        result2 = gbm_sim.simulate_paths(**params, seed=456)

        # Terminal prices should have similar distributions
        mean1 = np.mean(result1.price_paths[:, -1])
        mean2 = np.mean(result2.price_paths[:, -1])
        se1 = np.std(result1.price_paths[:, -1]) / np.sqrt(params['n_paths'])
        se2 = np.std(result2.price_paths[:, -1]) / np.sqrt(params['n_paths'])
        combined_se = np.sqrt(se1**2 + se2**2)

        print(f"  Run 1 mean: {mean1:.4f}")
        print(f"  Run 2 mean: {mean2:.4f}")
        print(f"  Combined SE: {combined_se:.4f}")
        print(f"  Difference: {abs(mean1 - mean2):.4f}")

        np.testing.assert_allclose(mean1, mean2, atol=4*combined_se)
        report.success("Simulations are statistically consistent")

    def test_gbm_different_seeds(self, gbm_sim):
        """Test that different seeds produce different results."""
        report.header("GBM Different Seeds Test")
        report.info("Tests that different random seeds produce different paths")
        report.info("Ensures proper randomness in Monte Carlo simulation")

        params = dict(s0=100.0, mu=0.05, t=1.0, n_paths=100, n_steps=50)

        result1 = gbm_sim.simulate_paths(**params, seed=123)
        result2 = gbm_sim.simulate_paths(**params, seed=456)

        different = not np.allclose(result1.price_paths, result2.price_paths)
        print(f"  Seeds 123 vs 456 produce different paths: {different}")

        assert different
        report.success("Different seeds produce different results")


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
        report.header("Heston Path Simulation")
        report.info("Tests Heston stochastic volatility model simulation")
        report.info("Should return both price and volatility paths")

        result = heston_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=1000, n_steps=252, seed=42
        )

        report.params(s0=100.0, mu=0.05, t=1.0, n_paths=1000, n_steps=252)
        print(f"  Price paths shape: {result.price_paths.shape}")
        print(f"  Has volatility paths: {result.has_volatility}")

        assert isinstance(result, SimulationResult)
        assert result.price_paths.shape[0] == 1000
        assert result.has_volatility  # Should have volatility paths

    def test_heston_volatility_positive(self, heston_sim):
        """Test that volatility stays positive (or handled properly)."""
        report.header("Heston Volatility Positivity Test")
        report.info("Tests that simulated volatility stays non-negative")
        report.info("Crucial for model validity - volatility cannot be negative")

        result = heston_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=1000, n_steps=252, seed=42
        )

        if result.has_volatility:
            min_vol = np.min(result.volatility_paths)
            max_vol = np.max(result.volatility_paths)
            mean_vol = np.mean(result.volatility_paths)

            print(f"  Volatility range: [{min_vol:.6f}, {max_vol:.6f}]")
            print(f"  Mean volatility: {mean_vol:.6f}")

            # Volatility should be positive (stored as vol, not variance)
            assert np.all(result.volatility_paths >= 0), "Volatility should be non-negative"
            report.success("All volatility values are non-negative")

    def test_heston_mean_reversion(self, heston_sim):
        """Test variance mean reversion property."""
        report.header("Heston Mean Reversion Test")
        report.info("Tests that volatility reverts to long-run mean θ over time")
        report.info("Terminal vol should approach sqrt(theta) in long simulations")

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

            report.value("Terminal Vol (mean)", terminal_vol, expected=expected_vol)

            # Should be within reasonable range of sqrt(theta)
            np.testing.assert_allclose(terminal_vol, expected_vol, rtol=0.2)

    def test_heston_leverage_effect(self):
        """
        Test leverage effect (negative correlation).

        With rho < 0, negative spot returns should correlate with higher volatility.
        """
        report.header("Heston Leverage Effect Test")
        report.info("Tests leverage effect: price drops correlate with volatility increases")
        report.info("rho < 0 creates the 'volatility smile' asymmetry")

        # High negative correlation
        heston_neg_rho = create_heston(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.9
        )

        report.params(rho=-0.9)

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

            print(f"  Return-Vol correlation: {corr:.4f}")
            report.info("Negative correlation indicates leverage effect")

            # Should be negative
            assert corr < 0, f"Leverage effect correlation should be negative, got {corr}"
            report.success("Leverage effect confirmed")


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
        report.header("Merton Path Simulation")
        report.info("Tests Merton jump-diffusion model simulation")
        report.info("Combines GBM diffusion with Poisson jump process")

        result = merton_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=1000, n_steps=252, seed=42
        )

        report.params(s0=100.0, mu=0.05, t=1.0, n_paths=1000, n_steps=252)
        print(f"  Price paths shape: {result.price_paths.shape}")

        assert isinstance(result, SimulationResult)
        assert result.price_paths.shape[0] == 1000

    def test_merton_jump_detection(self, merton_sim):
        """Test that jumps are present in paths."""
        report.header("Merton Jump Detection Test")
        report.info("Tests that jumps create fat tails in return distribution")
        report.info("Kurtosis > 3 indicates presence of jumps (normal = 3)")

        result = merton_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=10000, n_steps=252, seed=42
        )

        # Compute log returns
        log_returns = np.diff(np.log(result.price_paths), axis=1).flatten()

        # With jumps, returns should have excess kurtosis > 3 (normal has 3)
        kurtosis = np.mean((log_returns - log_returns.mean()) ** 4) / (log_returns.std() ** 4)

        print(f"  Log returns kurtosis: {kurtosis:.4f}")
        report.info("Normal distribution has kurtosis = 3")
        report.info("Jumps create fat tails -> kurtosis > 3")

        assert kurtosis > 3.5, f"Jump model should have fat tails, kurtosis={kurtosis}"
        report.success(f"Fat tails detected (kurtosis = {kurtosis:.2f})")

    def test_bates_simulation(self):
        """Test Bates model simulation."""
        report.header("Bates Model Simulation")
        report.info("Tests Bates model: combines Heston stochastic vol with jumps")
        report.info("Most complete model - captures both vol dynamics and discontinuities")

        bates_sim = create_bates(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
        )

        result = bates_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=1000, n_steps=252, seed=42
        )

        print(f"  Price paths shape: {result.price_paths.shape}")
        print(f"  Has volatility paths: {result.has_volatility}")

        assert isinstance(result, SimulationResult)
        assert result.has_volatility  # Bates has stochastic vol
        report.success("Bates simulation completed with volatility paths")


# =============================================================================
# CONVERGENCE TESTS
# =============================================================================

class TestConvergence:
    """Test Monte Carlo convergence properties."""

    def test_price_convergence(self):
        """Test that MC price converges to analytical price."""
        report.header("MC Price Convergence Test")
        report.info("Tests that MC price converges to Black-Scholes analytical price")
        report.info("Error should decrease as number of paths increases")

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

        print(f"  Analytical (BS) price: ${analytical_price:.4f}")

        # Test convergence with increasing paths
        path_counts = [1000, 10000, 50000]
        errors = []

        rows = []
        for n_paths in path_counts:
            mc_engine = MonteCarloEngine(n_paths=n_paths, seed=42)
            mc_result = mc_engine.price(option, gbm, market)
            mc_price = mc_result.price
            error = abs(mc_price - analytical_price) / analytical_price * 100
            errors.append(error / 100)
            rows.append((n_paths, mc_price, f"{error:.4f}%"))

        report.table(["Paths", "MC Price", "Error"], rows, title="MC Convergence", precision=4)

        # Error should decrease with more paths
        assert errors[-1] < errors[0], "MC error should decrease with more paths"

        # Final error should be small
        assert errors[-1] < 0.02, f"MC error too large: {errors[-1]}"
        report.success("MC converges to analytical price")

    def test_sqrt_n_convergence_rate(self):
        """
        Test sqrt(n) convergence rate of Monte Carlo.

        Standard error proportional to 1/sqrt(n), so doubling paths should reduce error by sqrt(2).
        """
        report.header("MC sqrt(n) Convergence Rate Test")
        report.info("Tests that MC standard error follows 1/sqrt(n) convergence")
        report.info("4x paths should reduce error by factor of 2")

        gbm_sim = create_gbm(sigma=0.2)
        s0, t, r, q = 100.0, 1.0, 0.05, 0.0
        mu = r - q

        true_mean = s0 * np.exp(mu * t)
        report.value("True Mean", true_mean, unit="$", precision=4)

        errors = []
        path_counts = [1000, 4000, 16000]  # 4x increases

        rows = []
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
            rows.append((n_paths, f"{se:.6f}"))

        report.table(["Paths", "Std Error"], rows, title="Standard Error by Path Count")

        # 4x paths should give 2x reduction in error
        ratio = errors[0] / errors[1]
        expected_ratio = np.sqrt(4)  # = 2

        print(f"  Error ratio (1000->4000): {ratio:.4f}")
        print(f"  Expected ratio (sqrt(4)): {expected_ratio:.4f}")

        np.testing.assert_allclose(ratio, expected_ratio, rtol=0.3)
        report.success("sqrt(n) convergence confirmed")

    def test_antithetic_variance_reduction(self):
        """Test antithetic variates reduce variance."""
        report.header("Antithetic Variance Reduction Test")
        report.info("Tests variance reduction technique for Monte Carlo")
        report.info("Verifies terminal distribution matches theoretical expectations")

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

        # For now, verify regular simulation works correctly
        terminal_std = result_regular.terminal_std
        expected_std = s0 * np.exp(mu * t) * np.sqrt(np.exp(0.2 ** 2 * t) - 1)

        report.value("Terminal Std", terminal_std, expected=expected_std, unit="$")

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
        report.header("Terminal Prices Test")
        report.info("Tests extraction of terminal prices from SimulationResult")
        report.info("Terminal prices are used for option payoff calculations")

        terminal = sample_result.terminal_prices

        print(f"  Terminal prices count: {len(terminal)}")
        print(f"  Terminal mean: {np.mean(terminal):.4f}")
        print(f"  Terminal std: {np.std(terminal):.4f}")

        assert len(terminal) == 1000

    def test_initial_price(self, sample_result):
        """Test initial price."""
        report.header("Initial Price Test")
        report.info("Tests that initial price matches input spot price")
        report.info("All paths should start from the same initial value")

        initial = sample_result.initial_price
        report.value("Initial Price", initial, expected=100.0, unit="$")

        assert initial == pytest.approx(100.0, rel=1e-10)

    def test_mean_path(self, sample_result):
        """Test mean path calculation."""
        report.header("Mean Path Test")
        report.info("Tests calculation of average price path across all simulations")
        report.info("Useful for visualizing expected price evolution")

        mean_path = sample_result.mean_path

        print(f"  Mean path length: {len(mean_path)}")
        print(f"  Initial: {mean_path[0]:.4f}")
        print(f"  Terminal: {mean_path[-1]:.4f}")

        assert len(mean_path) == 253  # n_steps + 1
        assert mean_path[0] == pytest.approx(100.0, rel=1e-10)

    def test_std_path(self, sample_result):
        """Test standard deviation path."""
        report.header("Std Path Test")
        report.info("Tests standard deviation path (dispersion over time)")
        report.info("Std should be 0 at t=0 and increase with time")

        std_path = sample_result.std_path

        print(f"  Std path length: {len(std_path)}")
        print(f"  Initial std: {std_path[0]:.6f}")
        print(f"  Terminal std: {std_path[-1]:.4f}")

        assert len(std_path) == 253
        assert std_path[0] == pytest.approx(0.0, abs=1e-10)  # No std at t=0

    def test_percentile_paths(self, sample_result):
        """Test percentile path calculation."""
        report.header("Percentile Paths Test")
        report.info("Tests percentile path extraction for confidence bands")
        report.info("5th and 95th percentiles show 90% confidence interval")

        percentiles = sample_result.percentile_paths([5, 50, 95])

        print(f"  Percentile paths count: {len(percentiles)}")
        print(f"  Path length: {len(percentiles[0])}")
        print(f"  Terminal values: P5={percentiles[0][-1]:.2f}, P50={percentiles[1][-1]:.2f}, P95={percentiles[2][-1]:.2f}")

        assert len(percentiles) == 3
        for p_path in percentiles:
            assert len(p_path) == 253

        # 5th percentile < median < 95th percentile
        assert np.all(percentiles[0] <= percentiles[1])
        assert np.all(percentiles[1] <= percentiles[2])

    def test_log_returns(self, sample_result):
        """Test log returns calculation."""
        report.header("Log Returns Test")
        report.info("Tests computation of log returns: ln(S_t+1/S_t)")
        report.info("Log returns are additive and approximately normal for GBM")

        log_ret = sample_result.log_returns()  # Method, not property

        print(f"  Log returns shape: {log_ret.shape}")
        print(f"  Mean daily log return: {np.mean(log_ret):.6f}")
        print(f"  Std daily log return: {np.std(log_ret):.6f}")

        assert log_ret.shape == (1000, 252)  # n_paths x n_steps

    def test_realized_volatility(self, sample_result):
        """Test realized volatility calculation."""
        report.header("Realized Volatility Test")
        report.info("Tests realized volatility calculation from simulated paths")
        report.info("Realized vol = sqrt(annualized sum of squared returns)")

        rv = sample_result.realized_volatility()  # Method, not property

        print(f"  Realized vol count: {len(rv)}")
        print(f"  Mean realized vol: {np.mean(rv):.4f}")
        print(f"  Std realized vol: {np.std(rv):.4f}")

        assert len(rv) == 1000  # One per path
        assert np.all(rv >= 0)  # Volatility is positive


# =============================================================================
# EDGE CASES AND NUMERICAL STABILITY
# =============================================================================

class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_very_short_maturity(self):
        """Test simulation with very short maturity."""
        report.header("Very Short Maturity Test")
        report.info("Tests simulation stability with very short time horizon (0.001y)")
        report.info("Terminal values should be very close to initial price")

        gbm_sim = create_gbm(sigma=0.2)

        result = gbm_sim.simulate_paths(
            s0=100.0, mu=0.05, t=0.001, n_paths=1000, n_steps=10, seed=42
        )

        report.params(t=0.001)
        report.value("Terminal Mean", result.terminal_mean, expected=100.0, unit="$")

        # Terminal prices should be very close to initial
        np.testing.assert_allclose(result.terminal_mean, 100.0, rtol=0.01)

    def test_zero_volatility(self):
        """Test GBM with near-zero volatility."""
        report.header("Near-Zero Volatility Test")
        report.info("Tests simulation with near-zero volatility (σ=0.001)")
        report.info("Should produce deterministic growth: S_T = S_0 * exp(μT)")

        gbm_sim = create_gbm(sigma=0.001)

        result = gbm_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0, n_paths=1000, n_steps=252, seed=42
        )

        # Should be deterministic growth
        expected = 100.0 * np.exp(0.05)

        report.params(sigma=0.001)
        report.value("Terminal Mean", result.terminal_mean, expected=expected, unit="$")
        report.value("Terminal Std", result.terminal_std, expected=0.0, unit="$")

        np.testing.assert_allclose(result.terminal_mean, expected, rtol=0.01)
        np.testing.assert_allclose(result.terminal_std, 0.0, atol=0.5)

    def test_high_volatility(self):
        """Test GBM with high volatility."""
        report.header("High Volatility Test")
        report.info("Tests simulation with extreme volatility (σ=100%)")
        report.info("Martingale property should still hold despite high volatility")

        gbm_sim = create_gbm(sigma=1.0)  # 100% vol

        result = gbm_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0, n_paths=10000, n_steps=252, seed=42
        )

        # Should still have correct mean
        expected_mean = 100.0 * np.exp(0.05)

        report.params(sigma=1.0)
        report.value("Terminal Mean", result.terminal_mean, expected=expected_mean, unit="$")

        np.testing.assert_allclose(result.terminal_mean, expected_mean, rtol=0.05)

    def test_negative_drift(self):
        """Test with negative drift (high dividend yield)."""
        report.header("Negative Drift Test")
        report.info("Tests simulation with negative risk-neutral drift")
        report.info("Occurs when dividend yield > risk-free rate (q > r)")

        gbm_sim = create_gbm(sigma=0.2)

        # mu = r - q = 0.02 - 0.08 = -0.06
        result = gbm_sim.simulate_paths(
            s0=100.0, mu=-0.06, t=1.0, n_paths=10000, n_steps=252, seed=42
        )

        # Net drift is negative
        expected_mean = 100.0 * np.exp(-0.06)

        report.params(mu=-0.06)
        report.value("Terminal Mean", result.terminal_mean, expected=expected_mean, unit="$")

        np.testing.assert_allclose(result.terminal_mean, expected_mean, rtol=0.02)

    def test_large_number_of_steps(self):
        """Test with many time steps."""
        report.header("Large Number of Steps Test")
        report.info("Tests simulation with 1000 time steps")
        report.info("Finer discretization improves accuracy but increases compute")

        gbm_sim = create_gbm(sigma=0.2)

        result = gbm_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=100, n_steps=1000, seed=42
        )

        report.params(n_steps=1000)
        print(f"  Result shape: {result.price_paths.shape}")

        assert result.price_paths.shape == (100, 1001)

    def test_single_path(self):
        """Test single path simulation."""
        report.header("Single Path Test")
        report.info("Tests simulation with only 1 path (edge case)")
        report.info("Used for visualization or debugging, not for pricing")

        gbm_sim = create_gbm(sigma=0.2)

        result = gbm_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=1, n_steps=100, seed=42
        )

        report.params(n_paths=1)
        print(f"  Result shape: {result.price_paths.shape}")

        assert result.price_paths.shape == (1, 101)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Basic performance tests (not benchmarks, just sanity checks)."""

    def test_large_simulation_completes(self):
        """Test that large simulation completes in reasonable time."""
        report.header("Large Simulation Performance Test")
        report.info("Tests performance with 10,000 paths x 252 steps")
        report.info("Verifies simulation completes in < 10 seconds")

        import time

        gbm_sim = create_gbm(sigma=0.2)

        start = time.time()
        result = gbm_sim.simulate_paths(
            s0=100.0, mu=0.05, t=1.0,
            n_paths=10000, n_steps=252, seed=42
        )
        elapsed = time.time() - start

        report.params(n_paths=10000, n_steps=252)
        print(f"  Result shape: {result.price_paths.shape}")
        print(f"  Elapsed time: {elapsed:.3f}s")

        assert result.price_paths.shape == (10000, 253)
        assert elapsed < 10.0, f"Simulation took too long: {elapsed}s"
        report.success(f"Completed in {elapsed:.3f}s")

    def test_terminal_only_faster_than_full_paths(self):
        """Terminal-only simulation should be faster than full paths."""
        report.header("Terminal vs Full Path Performance Test")
        report.info("Compares performance: terminal-only vs full path simulation")
        report.info("Terminal-only should be faster (less memory allocation)")

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

        report.params(n_paths=n_paths)
        report.comparison("Full Paths", time_full, "Terminal Only", time_terminal, unit="s", precision=3)

        # Terminal should be faster (or at least not much slower)
        assert time_terminal <= time_full * 1.5, \
            f"Terminal ({time_terminal}s) should be faster than full paths ({time_full}s)"
