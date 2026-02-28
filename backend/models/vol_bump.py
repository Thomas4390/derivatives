"""
Volatility Bumping Utility
===========================

Shared utility for creating volatility-bumped model copies.

Provides a unified interface for bumping volatility across all model types:
- GBM: bump sigma directly
- Merton: bump sigma (preserves jump params)
- Heston/Bates: bump in vol space then convert to variance → v0_new = (sqrt(v0) + h)^2
- GARCH family: bump sigma0 (initial volatility)

Author: Thomas
Created: 2025
"""


import numpy as np

from backend.core.interfaces import Model


def create_vol_bumped_model(model: Model, vol_bump: float) -> Model | None:
    """
    Create a model with bumped volatility, preserving all other parameters.

    For variance-based models (Heston, Bates), the bump is applied in
    volatility space and converted back to variance:
        v0_new = (sqrt(v0) + vol_bump)^2

    Parameters
    ----------
    model : Model
        Original model instance
    vol_bump : float
        Absolute volatility bump (e.g., 0.01 for 1%)

    Returns
    -------
    Model or None
        New model with bumped volatility, or None if model type is unknown
    """
    from backend.models.bates import BatesModel
    from backend.models.garch import GARCHModel, GJRGARCHModel, NGARCHModel
    from backend.models.gbm import GBMModel
    from backend.models.heston import HestonModel
    from backend.models.merton import MertonModel

    params = model.get_parameters()

    if isinstance(model, GBMModel):
        new_sigma = max(params['sigma'] + vol_bump, 1e-8)
        return GBMModel(sigma=new_sigma)

    if isinstance(model, MertonModel):
        new_sigma = max(params['sigma'] + vol_bump, 1e-8)
        return MertonModel(
            sigma=new_sigma,
            lambda_j=params['lambda_j'],
            mu_j=params['mu_j'],
            sigma_j=params['sigma_j']
        )

    if isinstance(model, BatesModel):
        # Bump in vol space: new_vol = sqrt(v0) + h, then v0_new = new_vol^2
        new_vol = max(np.sqrt(params['v0']) + vol_bump, 0.0)
        new_v0 = max(new_vol ** 2, 1e-8)
        return BatesModel(
            v0=new_v0,
            kappa=params['kappa'],
            theta=params['theta'],
            xi=params['xi'],
            rho=params['rho'],
            lambda_j=params['lambda_j'],
            mu_j=params['mu_j'],
            sigma_j=params['sigma_j']
        )

    if isinstance(model, HestonModel):
        new_vol = max(np.sqrt(params['v0']) + vol_bump, 0.0)
        new_v0 = max(new_vol ** 2, 1e-8)
        return HestonModel(
            v0=new_v0,
            kappa=params['kappa'],
            theta=params['theta'],
            xi=params['xi'],
            rho=params['rho']
        )

    if isinstance(model, (GARCHModel, NGARCHModel, GJRGARCHModel)):
        new_sigma0 = max(params['sigma0'] + vol_bump, 1e-8)
        p = dict(params)
        p['sigma0'] = new_sigma0
        return type(model)(**p)

    return None


def create_vol_bumped_pair(
    model: Model, vol_bump: float
) -> tuple[Model | None, Model | None]:
    """
    Create a pair of models bumped up and down by vol_bump.

    Parameters
    ----------
    model : Model
        Original model instance
    vol_bump : float
        Absolute volatility bump size (positive)

    Returns
    -------
    tuple of (Model or None, Model or None)
        (model_up, model_down) with volatility bumped by +/- vol_bump
    """
    model_up = create_vol_bumped_model(model, vol_bump)
    model_down = create_vol_bumped_model(model, -vol_bump)
    return model_up, model_down
