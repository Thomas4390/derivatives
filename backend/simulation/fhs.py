"""
Filtered Historical Simulation (FHS)
=====================================

Resamples standardized innovations from a filtered NGARCH history to generate
forward-looking Monte Carlo paths under the P-measure.

Unlike standard Q-measure simulation (LRNVR with N(0,1) draws), FHS:
1. Draws innovations from the empirical distribution of filtered residuals
2. Preserves fat tails, skewness, and serial dependence structure
3. Cross-sectionally demeans each time step to remove sampling bias

References:
    Barone-Adesi, G., Giannopoulos, K., and Vosper, L. (1999).
    "VaR without correlations for portfolios of derivative securities."
    Journal of Futures Markets, 19(5), 583-602.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _resample_and_demean(
    filtered_innovations: np.ndarray,
    n_days: int,
    n_paths: int,
    uniform_draws: np.ndarray,
) -> np.ndarray:
    """
    Resample innovations with replacement and cross-sectionally demean.

    Parameters
    ----------
    filtered_innovations : np.ndarray
        Historical filtered innovations, shape (n_history,).
    n_days : int
        Number of forward simulation days.
    n_paths : int
        Number of Monte Carlo paths.
    uniform_draws : np.ndarray
        Pre-generated U(0,1) draws, shape (n_days * n_paths,).

    Returns
    -------
    np.ndarray
        Resampled innovations, shape (n_days, n_paths), cross-sectionally demeaned.
    """
    n_history = filtered_innovations.shape[0]
    shocks = np.empty((n_days, n_paths), dtype=np.float64)

    for idx in range(n_days * n_paths):
        hist_idx = int(uniform_draws[idx] * n_history)
        if hist_idx >= n_history:
            hist_idx = n_history - 1
        day = idx // n_paths
        path = idx % n_paths
        shocks[day, path] = filtered_innovations[hist_idx]

    # Cross-sectional demeaning: each time step independently
    for day in range(n_days):
        mean_val = 0.0
        for path in range(n_paths):
            mean_val += shocks[day, path]
        mean_val /= n_paths
        for path in range(n_paths):
            shocks[day, path] -= mean_val

    return shocks


class FHSInnovationSampler:
    """
    Filtered Historical Simulation innovation resampler.

    Takes filtered standardized innovations from an NGARCH estimation
    and resamples them with replacement for forward simulation.

    Parameters
    ----------
    filtered_innovations : np.ndarray
        Standardized residuals from NGARCH filtering, shape (n_history,).
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    sampler = FHSInnovationSampler(filtered_eps, seed=42)
    eps_market = sampler.sample(n_days=252, n_paths=10000)
    # eps_market.shape == (252, 10000), each row demeaned
    """

    _innovations: np.ndarray
    _rng: np.random.Generator

    def __init__(
        self, filtered_innovations: np.ndarray, seed: int | None = None
    ) -> None:
        innovations = np.asarray(filtered_innovations, dtype=np.float64).ravel()
        if len(innovations) == 0:
            raise ValueError("filtered_innovations must be non-empty")
        self._innovations = innovations
        self._rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))

    @property
    def n_history(self) -> int:
        return len(self._innovations)

    @property
    def innovations(self) -> np.ndarray:
        return self._innovations

    def sample(self, n_days: int, n_paths: int) -> np.ndarray:
        """
        Resample innovations for forward simulation.

        Parameters
        ----------
        n_days : int
            Number of forward days to simulate.
        n_paths : int
            Number of Monte Carlo paths.

        Returns
        -------
        np.ndarray
            Resampled innovations, shape (n_days, n_paths).
            Each row (time step) is cross-sectionally demeaned.
        """
        uniform_draws = self._rng.random(n_days * n_paths)
        return _resample_and_demean(self._innovations, n_days, n_paths, uniform_draws)
