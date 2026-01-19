"""
Tests for Volatility Path Simulation Module
===========================================

These tests were extracted from backend/simulation/simulate_volatility.py
to separate test code from production code.

Run with: pytest tests/test_simulate_volatility.py -v
"""

import numpy as np
import pytest
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.simulation.simulate_volatility import (
    simulate_volatility_paths,
    simulate_garch_paths,
    simulate_ngarch_paths,
    simulate_gjr_garch_paths,
    simulate_egarch_paths,
    simulate_garch_terminal,
    simulate_ngarch_terminal,
    simulate_joint_paths,
    validate_garch_params,
    compute_garch_long_run_variance,
    compute_ngarch_long_run_variance,
    estimate_garch_params_from_volatility,
    run_volatility_benchmark,
)


# =============================================================================
# Test Configuration
# =============================================================================

SIGMA0 = 0.20  # 20% initial volatility


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestBasicFunctionality:
    """Test basic functionality of each volatility model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)

    @pytest.mark.parametrize("model", ['garch', 'ngarch', 'gjr_garch', 'egarch'])
    def test_model_runs(self, model):
        """Each model should run without errors."""
        result = simulate_volatility_paths(
            model=model,
            sigma0=SIGMA0,
            n_paths=1000,
            n_steps=252,
            seed=42
        )
        assert result.variance_paths.shape == (1000, 253)
        assert result.return_paths.shape == (1000, 252)
        assert result.computation_time > 0

    @pytest.mark.parametrize("model", ['garch', 'ngarch', 'gjr_garch', 'egarch'])
    def test_variance_positive(self, model):
        """All variance values should be positive."""
        result = simulate_volatility_paths(
            model=model,
            sigma0=SIGMA0,
            n_paths=1000,
            n_steps=252,
            seed=42
        )
        assert np.all(result.variance_paths > 0)

    @pytest.mark.parametrize("model", ['garch', 'ngarch', 'gjr_garch', 'egarch'])
    def test_volatility_reasonable_range(self, model):
        """Volatility should stay in reasonable range (not explode)."""
        result = simulate_volatility_paths(
            model=model,
            sigma0=SIGMA0,
            n_paths=1000,
            n_steps=252,
            seed=42
        )
        # Volatility should not exceed 500%
        assert np.all(result.terminal_volatility < 5.0)


# =============================================================================
# Parameter Validation Tests
# =============================================================================

class TestParameterValidation:
    """Test GARCH parameter validation."""

    def test_valid_garch_params(self):
        """Valid GARCH parameters should pass validation."""
        is_valid, msg = validate_garch_params(0.0001, 0.05, 0.90)
        assert is_valid
        assert "Valid" in msg

    def test_nonstationary_garch_params(self):
        """Non-stationary parameters should be detected."""
        is_valid, msg = validate_garch_params(0.0001, 0.10, 0.92)
        assert not is_valid
        assert "stationarity" in msg.lower() or "constraint" in msg.lower()

    def test_negative_omega(self):
        """Negative omega should be invalid."""
        is_valid, msg = validate_garch_params(-0.0001, 0.05, 0.90)
        assert not is_valid


# =============================================================================
# Long-Run Variance Tests
# =============================================================================

class TestLongRunVariance:
    """Test long-run variance computations."""

    def test_garch_long_run_variance(self):
        """GARCH long-run variance should be computable."""
        omega, alpha, beta = 0.0001, 0.05, 0.90
        lr_var = compute_garch_long_run_variance(omega, alpha, beta)

        # Should equal omega / (1 - alpha - beta)
        expected = omega / (1 - alpha - beta)
        assert abs(lr_var - expected) < 1e-10

    def test_ngarch_long_run_variance(self):
        """NGARCH long-run variance should be computable."""
        omega, alpha, beta, theta = 0.0001, 0.05, 0.90, 0.5
        lr_var = compute_ngarch_long_run_variance(omega, alpha, beta, theta)

        # Should be positive
        assert lr_var > 0

    def test_garch_converges_to_long_run(self):
        """GARCH simulations should converge to long-run variance."""
        omega, alpha, beta = 0.0001, 0.05, 0.90
        lr_var = compute_garch_long_run_variance(omega, alpha, beta)
        lr_vol = np.sqrt(lr_var)

        # Simulate with initial vol different from long-run
        result = simulate_volatility_paths(
            'garch',
            sigma0=lr_vol * 1.5,  # Start above long-run
            n_paths=10000,
            n_steps=500,
            seed=42,
            omega=omega,
            alpha=alpha,
            beta=beta
        )

        # Terminal vol should be closer to long-run than initial
        initial_diff = abs(lr_vol * 1.5 - lr_vol)
        terminal_diff = abs(result.terminal_volatility.mean() - lr_vol)
        assert terminal_diff < initial_diff


# =============================================================================
# Parameter Estimation Tests
# =============================================================================

class TestParameterEstimation:
    """Test GARCH parameter estimation utilities."""

    def test_estimate_from_volatility(self):
        """Should estimate valid parameters from target volatility."""
        params = estimate_garch_params_from_volatility(
            target_long_run_vol=0.20,
            half_life_days=20,
            alpha_ratio=0.1
        )

        assert 'omega' in params
        assert 'alpha' in params
        assert 'beta' in params
        assert params['alpha'] > 0
        assert params['beta'] > 0
        assert params['alpha'] + params['beta'] < 1


# =============================================================================
# Joint Simulation Tests
# =============================================================================

class TestJointSimulation:
    """Test joint price-volatility simulation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)

    @pytest.mark.parametrize("vol_model", ['garch', 'ngarch'])
    def test_joint_simulation_runs(self, vol_model):
        """Joint simulation should produce both price and volatility paths."""
        result = simulate_joint_paths(
            volatility_model=vol_model,
            s0=100.0,
            mu=0.08,
            sigma0=0.20,
            t=1.0,
            n_paths=1000,
            n_steps=252,
            seed=42
        )

        assert result.price_paths.shape == (1000, 253)
        assert result.variance_paths.shape == (1000, 253)
        assert np.all(result.terminal_prices > 0)
        assert np.all(result.terminal_volatility > 0)

    def test_joint_simulation_drift(self):
        """Joint simulation should show expected drift direction."""
        result = simulate_joint_paths(
            volatility_model='ngarch',
            s0=100.0,
            mu=0.10,  # Positive drift
            sigma0=0.20,
            t=1.0,
            n_paths=10000,
            n_steps=252,
            seed=42
        )

        # Mean terminal price should be above initial
        # (with high probability given positive drift)
        mean_terminal = result.terminal_prices.mean()
        assert mean_terminal > 100.0


# =============================================================================
# Model-Specific Tests
# =============================================================================

class TestGARCH:
    """Specific tests for GARCH(1,1)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)

    def test_volatility_clustering(self):
        """GARCH should exhibit volatility clustering."""
        variance_paths, return_paths = simulate_garch_paths(
            SIGMA0, 0.0001, 0.10, 0.85, 1000, 500
        )

        # Check autocorrelation of squared returns
        returns_sq = return_paths[0] ** 2
        autocorr = np.corrcoef(returns_sq[:-1], returns_sq[1:])[0, 1]

        # Should show positive autocorrelation (clustering)
        assert autocorr > 0.1


class TestNGARCH:
    """Specific tests for NGARCH leverage effect."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)

    def test_leverage_effect(self):
        """NGARCH should show leverage effect (negative returns -> higher vol)."""
        result = simulate_volatility_paths(
            'ngarch', SIGMA0, 5000, 252, seed=42, theta=0.7
        )

        # Compute correlation between returns and volatility changes
        returns = result.return_paths.flatten()
        vol_changes = np.diff(result.volatility_paths, axis=1).flatten()

        # Subsample for efficiency
        n = min(len(returns), len(vol_changes))
        returns = returns[:n]
        vol_changes = vol_changes[:n]

        # Should show negative correlation (leverage effect)
        corr = np.corrcoef(returns, vol_changes)[0, 1]
        # Note: With NGARCH theta > 0, negative returns increase vol
        # so correlation should be negative
        assert corr < 0


class TestEGARCH:
    """Specific tests for EGARCH."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)

    def test_log_variance_stability(self):
        """EGARCH should maintain stable variance (no NaN/Inf)."""
        variance_paths, return_paths = simulate_egarch_paths(
            SIGMA0, -0.1, 0.1, 0.98, -0.1, 1000, 252
        )

        # Should have no NaN or Inf values
        assert not np.any(np.isnan(variance_paths))
        assert not np.any(np.isinf(variance_paths))


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance benchmark tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)

    @pytest.mark.parametrize("model", ['garch', 'ngarch', 'gjr_garch', 'egarch'])
    def test_simulation_speed(self, model):
        """Each model should complete 10K paths in reasonable time."""
        start = time.perf_counter()
        simulate_volatility_paths(
            model=model,
            sigma0=SIGMA0,
            n_paths=10000,
            n_steps=252,
            seed=42
        )
        elapsed = time.perf_counter() - start
        # Should complete in under 5 seconds (generous for CI)
        assert elapsed < 5.0

    def test_benchmark_function(self):
        """Benchmark function should return valid statistics."""
        results = run_volatility_benchmark(n_paths=10000, n_steps=252)
        assert 'GARCH' in results
        assert 'mean_time' in results['GARCH']
        assert results['GARCH']['mean_time'] > 0


# =============================================================================
# Terminal-Only Simulation Tests
# =============================================================================

class TestTerminalOnly:
    """Test terminal-only simulations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)

    def test_garch_terminal(self):
        """GARCH terminal-only should match full simulation."""
        # Full simulation
        full_var, _ = simulate_garch_paths(
            SIGMA0, 0.0001, 0.05, 0.90, 10000, 252
        )
        full_terminal = full_var[:, -1]

        # Terminal-only
        np.random.seed(42)  # Reset seed
        terminal_only = simulate_garch_terminal(
            SIGMA0, 0.0001, 0.05, 0.90, 10000, 252
        )

        # Mean should be similar (not exact due to parallel execution)
        assert abs(full_terminal.mean() - terminal_only.mean()) < 0.001

    def test_ngarch_terminal(self):
        """NGARCH terminal-only should produce valid results."""
        terminal_var = simulate_ngarch_terminal(
            SIGMA0, 0.0001, 0.05, 0.90, 0.5, 10000, 252
        )
        assert terminal_var.shape == (10000,)
        assert np.all(terminal_var > 0)


# =============================================================================
# Main Entry Point (for running outside pytest)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VOLATILITY PATH SIMULATION - PERFORMANCE TEST")
    print("=" * 70)

    np.random.seed(42)

    # Test 1: Basic functionality
    print("\n[TEST 1] Basic Functionality Test")
    print("-" * 40)

    n_paths_test = 10000
    n_steps_test = 252

    for model in ['garch', 'ngarch', 'gjr_garch', 'egarch']:
        result = simulate_volatility_paths(
            model=model,
            sigma0=SIGMA0,
            n_paths=n_paths_test,
            n_steps=n_steps_test,
            seed=42
        )
        print(f"  {result.model}:")
        print(f"    Computation time: {result.computation_time*1000:.2f} ms")
        print(f"    Terminal vol mean: {result.terminal_volatility.mean()*100:.2f}%")
        print(f"    Terminal vol std: {result.terminal_volatility.std()*100:.2f}%")

    # Test 2: Parameter validation
    print("\n[TEST 2] Parameter Validation")
    print("-" * 40)

    test_cases = [
        (0.0001, 0.05, 0.90, "Valid GARCH"),
        (0.0001, 0.10, 0.92, "Non-stationary (alpha+beta>1)"),
        (-0.0001, 0.05, 0.90, "Invalid omega"),
    ]

    for omega, alpha, beta, description in test_cases:
        is_valid, msg = validate_garch_params(omega, alpha, beta)
        status = "PASS" if is_valid else "FAIL"
        print(f"  {description}: {status} - {msg}")

    # Test 3: Long-run variance
    print("\n[TEST 3] Long-Run Variance")
    print("-" * 40)

    omega, alpha, beta = 0.0001, 0.05, 0.90
    lr_var = compute_garch_long_run_variance(omega, alpha, beta)
    print(f"  GARCH long-run variance: {lr_var:.6f}")
    print(f"  GARCH long-run volatility: {np.sqrt(lr_var)*100:.2f}%")

    # Test 4: Joint simulation
    print("\n[TEST 4] Joint Price-Volatility Simulation")
    print("-" * 40)

    joint_result = simulate_joint_paths(
        volatility_model='ngarch',
        s0=100.0,
        mu=0.08,
        sigma0=0.20,
        t=1.0,
        n_paths=10000,
        n_steps=252,
        seed=42,
        theta=0.5
    )

    print(f"  Volatility model: {joint_result.volatility_model}")
    print(f"  Computation time: {joint_result.computation_time*1000:.2f} ms")
    print(f"  Terminal price mean: ${joint_result.terminal_prices.mean():.2f}")
    print(f"  Terminal vol mean: {joint_result.terminal_volatility.mean()*100:.2f}%")

    # Test 5: Performance benchmark
    print("\n[TEST 5] Performance Benchmark (100K paths)")
    print("-" * 40)

    results = run_volatility_benchmark(n_paths=100000, n_steps=252)
    for model, stats in results.items():
        print(f"  {model}: {stats['mean_time']*1000:.2f} ms "
              f"({stats['throughput_samples_per_sec']/1e6:.2f} M samples/sec)")

    print("\nAll tests completed!")
