"""Warmup helper for the structured-product Numba kernels consumed by sn_interface.

Calling ``precompile_structured_kernels()`` forces the JIT compilation (or cache
load) of each kernel used by the new backend route, so that subsequent timings
exclude the one-shot compile/load overhead.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import numpy as np

from backend.engines.structured_kernels import (
    BARRIER_MATURITY,
    evaluate_cmi_ac_paths,
    evaluate_cpn_paths,
    evaluate_daily_barrier_paths,
    evaluate_fixings_call_paths,
    evaluate_generalized_ac_paths,
    evaluate_lowpoint_forward_paths,
    evaluate_marathon_paths,
    evaluate_terminal_product_paths,
    FIXINGS_AVERAGE,
)


def precompile_structured_kernels() -> None:
    """Invoke each sn_interface kernel once on minimal dummy inputs."""
    paths = np.array(
        [[100.0, 101.0, 102.0], [100.0, 99.0, 98.0]], dtype=np.float64
    )
    obs_int = np.array([1, 2], dtype=np.int64)
    obs_float = np.array([1.0, 1.0], dtype=np.float64)
    discount = np.array([0.99, 0.98], dtype=np.float64)

    evaluate_cpn_paths(
        paths, 0.98, 100.0, 1.0, 1.0, 1.5, True,
    )

    evaluate_generalized_ac_paths(
        paths, obs_int, obs_float, obs_float, obs_int, obs_float, obs_float, obs_float,
        1.0, 0.7, BARRIER_MATURITY, 1.0, 100.0, discount, discount, 0.98,
    )

    evaluate_terminal_product_paths(
        paths, obs_int, obs_float, obs_float, discount,
        100.0, 1.0, 0.7, BARRIER_MATURITY, 1.0, 0.0, 1.0, 1.0,
        False, True, 1.5, 0.98,
    )

    evaluate_marathon_paths(
        paths, 100.0, 1.0, 0.7, BARRIER_MATURITY, 0.9, 0.98,
    )

    evaluate_cmi_ac_paths(
        paths, obs_int, obs_float, obs_float, obs_float, obs_float,
        obs_int, obs_float, obs_float, obs_float,
        1.0, 0.7, BARRIER_MATURITY, 1.0, 100.0, discount, discount, 0.98,
    )

    evaluate_lowpoint_forward_paths(
        paths, obs_int, 100.0, 1.0, 0.98,
    )

    evaluate_daily_barrier_paths(
        paths, 100.0, 0.7, True, 1.3, 0.98,
    )

    evaluate_fixings_call_paths(
        paths, obs_int, 100.0, 1.0, FIXINGS_AVERAGE, False, True, 1.5, 0.98,
    )
