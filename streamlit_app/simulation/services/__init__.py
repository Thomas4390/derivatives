"""Services module for Monte Carlo Simulation Explorer."""

from .state_manager import (
    init_session_state,
    get_simulation_params,
    set_simulation_params,
    get_price_model,
    set_price_model,
    get_volatility_model,
    set_volatility_model
)

# New unified simulation service
from .simulation_service import (
    run_simulation,
    run_terminal_simulation,
    get_model_characteristics,
    check_model_conditions,
    compute_long_run_volatility,
    get_initial_volatility,
    MODEL_NAMES,
)

# New pricing service
from .pricing_service import (
    compare_pricing,
    price_from_terminals,
    price_with_analytical,
    price_with_fft,
    price_multiple_strikes,
    get_available_pricing_methods,
    compute_option_pnl,
    PricingComparison,
)

# Try to import old simulation_runner for backward compatibility
try:
    from .simulation_runner import (
        run_price_simulation,
        run_volatility_simulation,
        calculate_pnl_from_paths,
        run_terminal_only_simulation
    )
except ImportError:
    # Old backend not available, provide stubs
    def run_price_simulation(*args, **kwargs):
        raise NotImplementedError("Use run_simulation from simulation_service instead")

    def run_volatility_simulation(*args, **kwargs):
        raise NotImplementedError("Use run_simulation from simulation_service instead")

    def calculate_pnl_from_paths(*args, **kwargs):
        raise NotImplementedError("Use compute_option_pnl from pricing_service instead")

    def run_terminal_only_simulation(*args, **kwargs):
        raise NotImplementedError("Use run_terminal_simulation from simulation_service instead")
