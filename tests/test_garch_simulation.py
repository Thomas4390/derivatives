"""
Tests for GARCH Simulation
============================

Tests for GARCH(1,1), NGARCH, and GJR-GARCH model simulation and MC pricing.

Author: Thomas
Created: 2025
"""

import pytest
import numpy as np

from backend.models.garch import GARCHModel, NGARCHModel, GJRGARCHModel


class TestGARCHSimulation:
    """Tests for GARCH simulator and MC pricer."""

    def test_garch_simulator_creation(self, garch_model):
        """create_simulator() returns a functional simulator."""
        simulator = garch_model.create_simulator()
        assert simulator is not None
        result = simulator.simulate_paths(
            s0=100.0, mu=0.08, t=1.0, n_paths=100, n_steps=50, seed=42
        )
        assert result.price_paths.shape == (100, 51)

    def test_garch_positive_prices(self, garch_model):
        """All simulated prices are positive."""
        simulator = garch_model.create_simulator()
        result = simulator.simulate_paths(
            s0=100.0, mu=0.08, t=1.0, n_paths=1000, n_steps=252, seed=42
        )
        assert np.all(result.price_paths > 0), "All prices must be positive"

    def test_garch_volatility_paths_exist(self, garch_model):
        """has_volatility is True, vol paths are non-negative."""
        simulator = garch_model.create_simulator()
        result = simulator.simulate_paths(
            s0=100.0, mu=0.08, t=1.0, n_paths=500, n_steps=252, seed=42
        )
        assert result.has_volatility is True
        assert np.all(result.volatility_paths >= 0), "Volatility must be non-negative"

    def test_garch_vol_mean_reversion(self, garch_model):
        """Terminal vol converges toward sqrt(omega / (1 - alpha - beta))."""
        params = garch_model.get_parameters()
        long_run_var = params['omega'] / (1 - params['alpha'] - params['beta'])
        long_run_vol = np.sqrt(long_run_var)

        simulator = garch_model.create_simulator()
        result = simulator.simulate_paths(
            s0=100.0, mu=0.08, t=5.0, n_paths=5000, n_steps=1260, seed=42
        )

        terminal_vol = result.volatility_paths[:, -1]
        mean_terminal = np.mean(terminal_vol)

        # Should converge toward long-run vol (generous tolerance for MC)
        np.testing.assert_allclose(mean_terminal, long_run_vol, rtol=0.30)

    def test_garch_mc_pricing_positive(self, garch_model):
        """MC pricer returns a positive call price."""
        pricer = garch_model.create_pricer(n_paths=50000)
        result = pricer.price(s0=100.0, k=100.0, t=0.25, r=0.05, option_type="call")
        assert result.price > 0, "Call price must be positive"
        assert result.std_error > 0, "Standard error must be positive"

    def test_ngarch_simulation_runs(self, ngarch_model):
        """NGARCH simulator produces valid results."""
        simulator = ngarch_model.create_simulator()
        result = simulator.simulate_paths(
            s0=100.0, mu=0.08, t=1.0, n_paths=500, n_steps=252, seed=42
        )
        assert result.price_paths.shape == (500, 253)
        assert np.all(result.price_paths > 0)
        assert result.has_volatility is True

    def test_gjr_garch_simulation_runs(self, gjr_garch_model):
        """GJR-GARCH simulator produces valid results."""
        simulator = gjr_garch_model.create_simulator()
        result = simulator.simulate_paths(
            s0=100.0, mu=0.08, t=1.0, n_paths=500, n_steps=252, seed=42
        )
        assert result.price_paths.shape == (500, 253)
        assert np.all(result.price_paths > 0)
        assert result.has_volatility is True

    def test_gjr_garch_leverage_effect(self, gjr_garch_model):
        """Negative returns correlate with increased vol (gamma > 0)."""
        simulator = gjr_garch_model.create_simulator()
        result = simulator.simulate_paths(
            s0=100.0, mu=0.0, t=1.0, n_paths=5000, n_steps=252, seed=42
        )

        prices = result.price_paths
        vols = result.volatility_paths

        # Compute log returns and vol changes at each step
        log_returns = np.log(prices[:, 1:] / prices[:, :-1])
        vol_changes = vols[:, 1:] - vols[:, :-1]

        # Flatten across all paths and steps
        all_returns = log_returns.flatten()
        all_vol_changes = vol_changes.flatten()

        # Correlation between returns and subsequent vol changes should be negative
        corr = np.corrcoef(all_returns, all_vol_changes)[0, 1]
        assert corr < 0, f"Expected negative correlation (leverage), got {corr:.4f}"

    def test_garch_put_call_parity(self, garch_model):
        """Approximate put-call parity under MC: C - P ≈ S - K*exp(-rT)."""
        pricer = garch_model.create_pricer(n_paths=100000)

        s0, k, t, r = 100.0, 100.0, 0.25, 0.05

        call_result = pricer.price(s0=s0, k=k, t=t, r=r, option_type="call", seed=42)
        put_result = pricer.price(s0=s0, k=k, t=t, r=r, option_type="put", seed=42)

        # C - P should approximate S - K*exp(-rT)
        c_minus_p = call_result.price - put_result.price
        expected = s0 - k * np.exp(-r * t)

        np.testing.assert_allclose(c_minus_p, expected, atol=0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
