"""Lazy-loader for exotic pricing/payoff functions.

Shared by pricing_service, simulation_runner and consistent_pricing to avoid
duplicate importlib boilerplate. The options_greeks exotic adapter is loaded
once on first use and its functions are cached.
"""

from pathlib import Path

_adapter = None


def _load_adapter():
    """Load the options_greeks exotic pricing adapter module once."""
    global _adapter
    if _adapter is None:
        import importlib.util

        adapter_path = (
            Path(__file__).parent.parent.parent
            / "options_greeks"
            / "services"
            / "exotic_pricing_adapter.py"
        )
        spec = importlib.util.spec_from_file_location(
            "exotic_pricing_adapter", adapter_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _adapter = mod
    return _adapter


def get_exotic_payoff_fn():
    """Lazy-load calculate_exotic_payoff_at_expiry on first use."""
    return _load_adapter().calculate_exotic_payoff_at_expiry


def get_exotic_payoff_vec_fn():
    """Lazy-load the vectorized calculate_exotic_payoff_at_expiry_vec."""
    return _load_adapter().calculate_exotic_payoff_at_expiry_vec


def get_exotic_price_fn():
    """Lazy-load calculate_exotic_price (GBM closed-form) on first use.

    For Haug-catalog families (powered/capped-power/log/supershare/double-,
    discrete-, partial-time- & binary-barrier) this routes through the
    Open/Closed registry behind ``ExoticAnalyticEngine``; basic-8 types keep the
    integer Numba kernels.
    """
    return _load_adapter().calculate_exotic_price


def get_exotic_instrument_fn():
    """Lazy-load ``_create_exotic_instrument`` (exotic_type -> backend Instrument).

    Used by the model-consistent pricer to build the backend instrument for the
    full-path ``ExoticMonteCarloEngine`` route (path-dependent exotics under a
    non-GBM model), reusing the single table-driven factory.
    """
    return _load_adapter()._create_exotic_instrument


def get_exotic_all_greeks_fn():
    """Lazy-load calculate_exotic_all_greeks (registry-aware 14-Greeks).

    Used by the Greeks service for Haug-catalog families, which have no integer
    Numba kernel: the adapter builds the ``exotic_advanced`` instrument and
    differentiates it through the backend ``GreeksCalculator``.
    """
    return _load_adapter().calculate_exotic_all_greeks


def get_binary_parse_map() -> dict[int, tuple]:
    """Reiner-Rubinstein binary-barrier type map ``{1..28: (is_down, is_in,
    is_asset, gate)}``, reused from the adapter so the simulation path-monitoring
    shares the single source of truth (no duplicated 28-type table).
    """
    return _load_adapter()._BINARY_PARSE
