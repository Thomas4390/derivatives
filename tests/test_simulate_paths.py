"""
Tests for Monte Carlo Price Path Simulation Module
==================================================

These tests were extracted from backend/simulation/simulate_paths.py
to separate test code from production code.

Run with: pytest tests/test_simulate_paths.py -v
"""

import numpy as np
import pytest
import time
from scipy.stats import norm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.simulation.simulate_paths import (
    simulate_paths,
    simulate_gbm_paths,
    simulate_gbm_paths_vectorized,
    simulate_heston_paths,
    simulate_merton_jump_paths,
    simulate_sabr_paths,
    simulate_correlated_gbm_paths,
    validate_correlation_matrix,
    price_european_call_mc,
    price_european_put_mc,
    price_asian_arithmetic_call_mc,
    price_lookback_call_mc,
    price_barrier_down_out_call_mc,
    benchmark_simulation,
    run_full_benchmark,
)


# =============================================================================
# Test Configuration
# =============================================================================

S0 = 100.0      # Initial stock price
R = 0.05        # Risk-free rate (5%)
SIGMA = 0.2     # Volatility (20%)
T = 1.0         # Time to maturity (1 year)


def black_scholes_call(s, k, t, r, sigma):
    """Analytical Black-Scholes call price for validation."""
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestBasicFunctionality:
    """Test basic functionality of each model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)

    @pytest.mark.parametrize("model", ['gbm', 'heston', 'merton', 'sabr'])
    def test_model_runs(self, model):
        """Each model should run without errors."""
        result = simulate_paths(
            model=model,
            s0=S0,
            r=R,
            sigma=SIGMA,
            t=T,
            n_paths=1000,
            n_steps=252,
            seed=42
        )
        assert result.paths.shape == (1000, 253)
        assert result.terminal_values.shape == (1000,)
        assert result.computation_time > 0

    @pytest.mark.parametrize("model", ['gbm', 'heston', 'merton', 'sabr'])
    def test_terminal_values_positive(self, model):
        """All terminal prices should be positive."""
        result = simulate_paths(
            model=model,
            s0=S0,
            r=R,
            sigma=SIGMA,
            t=T,
            n_paths=1000,
            n_steps=252,
            seed=42
        )
        assert np.all(result.terminal_values > 0)


# =============================================================================
# Black-Scholes Validation Tests
# =============================================================================

class TestBlackScholesValidation:
    """Validate Monte Carlo vs analytical Black-Scholes."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)

    def test_gbm_convergence_to_bs(self):
        """GBM Monte Carlo should converge to Black-Scholes price."""
        K = 100.0
        bs_price = black_scholes_call(S0, K, T, R, SIGMA)

        # Large number of paths for convergence
        result = simulate_paths(
            'gbm', S0, R, SIGMA, T,
            n_paths=500000, n_steps=252, seed=42
        )
        mc_price = price_european_call_mc(result.terminal_values, K, R, T)

        # Should be within 1% of analytical
        assert abs(mc_price - bs_price) / bs_price < 0.01

    @pytest.mark.parametrize("n_paths,max_error_pct", [
        (10000, 5.0),
        (50000, 2.0),
        (100000, 1.5),
        (500000, 0.8),
    ])
    def test_convergence_rate(self, n_paths, max_error_pct):
        """Error should decrease with increasing path count."""
        K = 100.0
        bs_price = black_scholes_call(S0, K, T, R, SIGMA)

        result = simulate_paths(
            'gbm', S0, R, SIGMA, T,
            n_paths=n_paths, n_steps=252, seed=42
        )
        mc_price = price_european_call_mc(result.terminal_values, K, R, T)

        error_pct = abs(mc_price - bs_price) / bs_price * 100
        assert error_pct < max_error_pct


# =============================================================================
# Exotic Option Tests
# =============================================================================

class TestExoticOptions:
    """Test exotic option pricing functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.result = simulate_paths(
            'gbm', S0, R, SIGMA, T,
            n_paths=100000, n_steps=252, seed=42
        )

    def test_european_call(self):
        """European call should have positive value."""
        price = price_european_call_mc(self.result.terminal_values, 100, R, T)
        assert price > 0

    def test_european_put(self):
        """European put should have positive value."""
        price = price_european_put_mc(self.result.terminal_values, 100, R, T)
        assert price > 0

    def test_put_call_parity(self):
        """Put-call parity: C - P = S0 - K*exp(-rT)."""
        K = 100.0
        call_price = price_european_call_mc(self.result.terminal_values, K, R, T)
        put_price = price_european_put_mc(self.result.terminal_values, K, R, T)

        parity_rhs = S0 - K * np.exp(-R * T)
        parity_lhs = call_price - put_price

        # Should hold within numerical tolerance
        assert abs(parity_lhs - parity_rhs) < 0.5

    def test_asian_call(self):
        """Asian call should be cheaper than European call."""
        K = 100.0
        euro_price = price_european_call_mc(self.result.terminal_values, K, R, T)
        asian_price = price_asian_arithmetic_call_mc(self.result.paths, K, R, T)

        # Asian options have lower value due to averaging
        assert asian_price < euro_price

    def test_lookback_call(self):
        """Lookback call should be more expensive than European."""
        K = 100.0
        euro_price = price_european_call_mc(self.result.terminal_values, K, R, T)
        lookback_price = price_lookback_call_mc(self.result.paths, R, T)

        # Lookback has higher value (always optimal exercise)
        assert lookback_price > euro_price

    def test_barrier_option(self):
        """Down-and-out call should be cheaper than vanilla call."""
        K = 100.0
        B = 90.0
        euro_price = price_european_call_mc(self.result.terminal_values, K, R, T)
        barrier_price = price_barrier_down_out_call_mc(self.result.paths, K, B, R, T)

        # Barrier option has lower value (can knock out)
        assert barrier_price < euro_price


# =============================================================================
# Multi-Asset Tests
# =============================================================================

class TestMultiAsset:
    """Test correlated multi-asset simulation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)

    def test_correlation_matrix_validation(self):
        """Valid correlation matrix should pass validation."""
        corr_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ])
        # Should not raise
        validate_correlation_matrix(corr_matrix)

    def test_invalid_correlation_matrix(self):
        """Invalid correlation matrix should raise ValueError."""
        # Non-positive definite
        bad_matrix = np.array([
            [1.0, 0.99],
            [0.5, 1.0]  # Asymmetric
        ])
        with pytest.raises(ValueError):
            validate_correlation_matrix(bad_matrix)

    def test_realized_correlations(self):
        """Simulated correlations should match target."""
        s0_multi = np.array([100.0, 50.0, 200.0])
        sigmas_multi = np.array([0.2, 0.3, 0.15])
        corr_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ])

        multi_paths = simulate_correlated_gbm_paths(
            s0_multi, R, sigmas_multi, corr_matrix, T, 100000, 252
        )

        # Compute realized correlations
        terminal_returns = np.log(multi_paths[:, -1, :] / multi_paths[:, 0, :])
        realized_corr = np.corrcoef(terminal_returns.T)

        # Should be within 0.05 of target
        assert abs(realized_corr[0, 1] - 0.5) < 0.05
        assert abs(realized_corr[0, 2] - 0.3) < 0.05
        assert abs(realized_corr[1, 2] - 0.2) < 0.05


# =============================================================================
# Performance Benchmarks
# =============================================================================

class TestPerformance:
    """Performance benchmark tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)

    @pytest.mark.parametrize("model", ['gbm', 'heston', 'merton', 'sabr'])
    def test_simulation_speed(self, model):
        """Each model should complete 10K paths in reasonable time."""
        start = time.perf_counter()
        simulate_paths(
            model=model,
            s0=S0,
            r=R,
            sigma=SIGMA,
            t=T,
            n_paths=10000,
            n_steps=252,
            seed=42
        )
        elapsed = time.perf_counter() - start
        # Should complete in under 5 seconds (generous for CI)
        assert elapsed < 5.0

    def test_benchmark_function(self):
        """Benchmark function should return valid statistics."""
        stats = benchmark_simulation(
            simulate_gbm_paths,
            (S0, R, SIGMA, T, 1000, 252, True),
            n_runs=3
        )
        assert 'mean_time' in stats
        assert 'std_time' in stats
        assert 'throughput_paths_per_sec' in stats
        assert stats['mean_time'] > 0


# =============================================================================
# Heston Model Tests
# =============================================================================

class TestHestonModel:
    """Specific tests for Heston model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.v0 = 0.04
        self.kappa = 2.0
        self.theta = 0.04
        self.xi = 0.3
        self.rho = -0.7

    @pytest.mark.parametrize("scheme", [0, 1, 2, 3])
    def test_all_schemes(self, scheme):
        """All Heston discretization schemes should work."""
        s_paths, v_paths = simulate_heston_paths(
            S0, self.v0, R, self.kappa, self.theta,
            self.xi, self.rho, T, 1000, 252, scheme=scheme
        )
        assert s_paths.shape == (1000, 253)
        assert v_paths.shape == (1000, 253)
        assert np.all(s_paths > 0)

    def test_variance_mean_reversion(self):
        """Variance should mean-revert towards theta."""
        s_paths, v_paths = simulate_heston_paths(
            S0, self.v0 * 2, R, self.kappa, self.theta,
            self.xi, self.rho, T, 10000, 252, scheme=1
        )
        # Terminal variance should be closer to theta than initial
        initial_diff = abs(self.v0 * 2 - self.theta)
        terminal_diff = abs(v_paths[:, -1].mean() - self.theta)
        assert terminal_diff < initial_diff


# =============================================================================
# Main Entry Point (for running outside pytest)
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 80)
    print("MONTE CARLO PATH SIMULATION - PERFORMANCE TEST")
    print("=" * 80)

    np.random.seed(42)

    # Test 1: Basic functionality
    print("\n[TEST 1] Basic Functionality Test")
    print("-" * 40)

    n_paths_test = 10000
    n_steps_test = 252

    for model in ['gbm', 'heston', 'merton', 'sabr']:
        result = simulate_paths(
            model=model,
            s0=S0,
            r=R,
            sigma=SIGMA,
            t=T,
            n_paths=n_paths_test,
            n_steps=n_steps_test,
            seed=42
        )
        print(f"  {result.model}:")
        print(f"    Computation time: {result.computation_time*1000:.2f} ms")
        print(f"    Terminal mean: ${result.terminal_values.mean():.2f}")
        print(f"    Terminal std: ${result.terminal_values.std():.2f}")

    # Test 2: Black-Scholes validation
    print("\n[TEST 2] Option Pricing Validation (GBM vs Black-Scholes)")
    print("-" * 40)

    K = 100.0
    bs_price = black_scholes_call(S0, K, T, R, SIGMA)
    print(f"  Black-Scholes analytical price: ${bs_price:.4f}")

    path_counts = [10000, 50000, 100000, 500000]
    print(f"\n  Monte Carlo convergence:")
    for n in path_counts:
        result = simulate_paths('gbm', S0, R, SIGMA, T, n_paths=n, n_steps=252, seed=42)
        mc_price = price_european_call_mc(result.terminal_values, K, R, T)
        error = abs(mc_price - bs_price)
        print(f"    {n:>10,} paths: ${mc_price:.4f} (error: ${error:.4f})")

    # Test 3: Performance benchmark
    print("\n[TEST 3] Performance Benchmark (100K paths)")
    print("-" * 40)

    results = run_full_benchmark(n_paths=100000, n_steps=252)
    for model, stats in results.items():
        print(f"  {model}: {stats['mean_time']*1000:.2f} ms "
              f"({stats['throughput_samples_per_sec']/1e6:.2f} M samples/sec)")

    print("\nAll tests completed!")
