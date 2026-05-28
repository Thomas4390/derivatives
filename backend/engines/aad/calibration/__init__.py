"""
JAX kernels used by the V2 model calibrators (Heston/Merton/Bates/GARCH).

This package contains only the differentiable building blocks needed for
non-linear least-squares calibration with analytical Jacobians:

- :mod:`fft`        : differentiable Carr-Madan FFT call pricer
- :mod:`heston_cf`  : Heston and Bates characteristic functions in pure JAX
- :mod:`merton_cf`  : Merton jump-diffusion characteristic function in pure JAX
- :mod:`garch_nll`  : GARCH/NGARCH/GJR-GARCH negative log-likelihoods

The differentiable end-to-end Heston calibrator and the COS pricer that ship
on ``main`` are intentionally absent on the ``light`` branch.

Author: Thomas Vaudescal
Created: 2026
"""

from backend.engines.aad.calibration.fft import (
    JaxFFTGrids,
    carr_madan_call_prices,
    interpolate_strikes,
    price_call_strikes_jax,
)
from backend.engines.aad.calibration.garch_nll import (
    garch_per_obs_logL,
    gjr_per_obs_logL,
    neg_log_likelihood_garch,
    neg_log_likelihood_gjr,
    neg_log_likelihood_ngarch,
    ngarch_per_obs_logL,
)
from backend.engines.aad.calibration.heston_cf import (
    bates_cf_jax,
    heston_cf_jax,
    heston_cf_jax_jit,
)
from backend.engines.aad.calibration.heston_nandi_cf import heston_nandi_cf_jax
from backend.engines.aad.calibration.merton_cf import merton_cf_jax

__all__ = [
    # FFT pricer
    "JaxFFTGrids",
    "carr_madan_call_prices",
    "interpolate_strikes",
    "price_call_strikes_jax",
    # Characteristic functions
    "heston_cf_jax",
    "heston_cf_jax_jit",
    "bates_cf_jax",
    "heston_nandi_cf_jax",
    "merton_cf_jax",
    # GARCH NLLs
    "garch_per_obs_logL",
    "ngarch_per_obs_logL",
    "gjr_per_obs_logL",
    "neg_log_likelihood_garch",
    "neg_log_likelihood_ngarch",
    "neg_log_likelihood_gjr",
]
