"""Services module for Monte Carlo Simulation Explorer."""

from .simulation_service import (
    run_simulation,
    run_terminal_simulation,
    get_model_characteristics,
    check_model_conditions,
    compute_long_run_volatility,
    get_initial_volatility,
    get_model_display_name,
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
    PricingComparison,
)

from .simulation_runner import calculate_pnl_from_paths
