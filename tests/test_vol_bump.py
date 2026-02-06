"""
Tests for Vol Bump Utility
===========================

Tests for the shared vol bumping utility in backend/models/vol_bump.py.

Author: Thomas
Created: 2025
"""

import pytest
import numpy as np

from backend.models.vol_bump import create_vol_bumped_model, create_vol_bumped_pair
from backend.models.gbm import GBMModel
from backend.models.heston import HestonModel
from backend.models.bates import BatesModel
from backend.models.merton import MertonModel
from backend.models.garch import GARCHModel, NGARCHModel, GJRGARCHModel


class TestVolBump:
    """Tests for create_vol_bumped_model and create_vol_bumped_pair."""

    def test_gbm_bump_symmetric(self, gbm_model):
        """Bump is symmetric around sigma."""
        h = 0.01
        up = create_vol_bumped_model(gbm_model, h)
        down = create_vol_bumped_model(gbm_model, -h)

        assert up is not None
        assert down is not None
        sigma = gbm_model.get_parameters()['sigma']
        assert up.get_parameters()['sigma'] == pytest.approx(sigma + h)
        assert down.get_parameters()['sigma'] == pytest.approx(sigma - h)

    def test_heston_bump_preserves_params(self, heston_model):
        """kappa, theta, xi, rho unchanged after bump."""
        h = 0.01
        bumped = create_vol_bumped_model(heston_model, h)

        assert bumped is not None
        orig = heston_model.get_parameters()
        new = bumped.get_parameters()

        assert new['kappa'] == orig['kappa']
        assert new['theta'] == orig['theta']
        assert new['xi'] == orig['xi']
        assert new['rho'] == orig['rho']
        # v0 should have changed
        assert new['v0'] != orig['v0']

    def test_heston_bump_vol_space(self, heston_model):
        """Heston bump operates in vol space: v0_new = (sqrt(v0) + h)^2."""
        h = 0.01
        bumped = create_vol_bumped_model(heston_model, h)

        v0 = heston_model.get_parameters()['v0']
        expected_v0 = (np.sqrt(v0) + h) ** 2
        assert bumped.get_parameters()['v0'] == pytest.approx(expected_v0)

    def test_bates_bump_preserves_jump_params(self, bates_model):
        """lambda_j, mu_j, sigma_j unchanged after bump."""
        h = 0.01
        bumped = create_vol_bumped_model(bates_model, h)

        assert bumped is not None
        orig = bates_model.get_parameters()
        new = bumped.get_parameters()

        assert new['lambda_j'] == orig['lambda_j']
        assert new['mu_j'] == orig['mu_j']
        assert new['sigma_j'] == orig['sigma_j']
        assert new['kappa'] == orig['kappa']
        assert new['theta'] == orig['theta']
        assert new['xi'] == orig['xi']
        assert new['rho'] == orig['rho']

    def test_merton_bump_preserves_jump_params(self, merton_model):
        """lambda_j, mu_j, sigma_j unchanged after bump."""
        h = 0.01
        bumped = create_vol_bumped_model(merton_model, h)

        assert bumped is not None
        orig = merton_model.get_parameters()
        new = bumped.get_parameters()

        assert new['lambda_j'] == orig['lambda_j']
        assert new['mu_j'] == orig['mu_j']
        assert new['sigma_j'] == orig['sigma_j']
        assert new['sigma'] == pytest.approx(orig['sigma'] + h)

    def test_garch_bump_preserves_params(self, garch_model):
        """omega, alpha, beta unchanged after bump."""
        h = 0.01
        bumped = create_vol_bumped_model(garch_model, h)

        assert bumped is not None
        orig = garch_model.get_parameters()
        new = bumped.get_parameters()

        assert new['omega'] == orig['omega']
        assert new['alpha'] == orig['alpha']
        assert new['beta'] == orig['beta']
        assert new['sigma0'] == pytest.approx(orig['sigma0'] + h)

    def test_unknown_model_returns_none(self):
        """Unknown model type returns None."""
        class FakeModel:
            name = "fake"
            def get_parameters(self):
                return {"x": 1.0}

        result = create_vol_bumped_model(FakeModel(), 0.01)
        assert result is None

    def test_negative_bump_floored(self):
        """Volatility cannot go negative — floored at 1e-8."""
        model = GBMModel(sigma=0.005)
        bumped = create_vol_bumped_model(model, -0.01)
        assert bumped is not None
        assert bumped.get_parameters()['sigma'] > 0

    def test_heston_negative_bump_floored(self):
        """Heston v0 cannot go negative — floored at 1e-8."""
        model = HestonModel(v0=0.0001, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        bumped = create_vol_bumped_model(model, -0.05)
        assert bumped is not None
        assert bumped.get_parameters()['v0'] > 0

    def test_create_vol_bumped_pair(self, gbm_model):
        """Pair returns consistent up and down models."""
        h = 0.01
        up, down = create_vol_bumped_pair(gbm_model, h)

        assert up is not None
        assert down is not None
        sigma = gbm_model.get_parameters()['sigma']
        assert up.get_parameters()['sigma'] == pytest.approx(sigma + h)
        assert down.get_parameters()['sigma'] == pytest.approx(sigma - h)

    def test_pair_unknown_model_returns_none_pair(self):
        """Pair with unknown model returns (None, None)."""
        class FakeModel:
            name = "fake"
            def get_parameters(self):
                return {"x": 1.0}

        up, down = create_vol_bumped_pair(FakeModel(), 0.01)
        assert up is None
        assert down is None


    def test_vol_bump_negative_floors_at_zero(self):
        """Bug 5: Large negative bump must not produce phantom vol via squaring."""
        # Heston with sqrt(v0) = 0.2, bump by -0.3 → should floor at 0
        model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        bumped = create_vol_bumped_model(model, -0.3)

        assert bumped is not None
        new_v0 = bumped.get_parameters()['v0']
        # Without the floor fix, new_vol = 0.2 - 0.3 = -0.1 → v0 = 0.01 (phantom)
        # With the fix, new_vol = max(-0.1, 0) = 0 → v0 = 1e-8 (floor)
        assert new_v0 == pytest.approx(1e-8), f"Expected v0=1e-8 (floored), got {new_v0}"

        # Same test for Bates
        bates = BatesModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                           lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)
        bumped_bates = create_vol_bumped_model(bates, -0.3)
        assert bumped_bates is not None
        assert bumped_bates.get_parameters()['v0'] == pytest.approx(1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
