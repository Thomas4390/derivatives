"""
JAX-based numerical kernels.

On the ``light`` branch this package only ships the calibration helpers
(:mod:`backend.engines.aad.calibration`). The full AAD pricing/Greeks engine
lives on ``main``.

Importing this package enables JAX's 64-bit precision globally — the
calibration FFT pricer and characteristic-function residuals require x64
to converge to a tight RMSE.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

try:
    import jax  # noqa: F401

    jax.config.update("jax_enable_x64", True)
except ImportError as e:
    raise ImportError(
        "JAX is required for the calibration kernels under "
        "backend.engines.aad.calibration. Install with `pip install jax jaxopt`."
    ) from e
