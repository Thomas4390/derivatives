"""Components module for Monte Carlo Simulation Explorer."""

# New unified components
from .model_selector import render_model_selector
from .parameter_panel import render_parameter_panel
from .pricing_comparison import render_pricing_comparison
from .results_summary import render_results_summary

__all__ = [
    # New components
    'render_model_selector',
    'render_parameter_panel',
    'render_pricing_comparison',
    'render_results_summary',
]

# Import additional components
try:
    from .sidebar import render_sidebar
    from .analysis_header import (
        render_analysis_header,
        render_strategy_summary_compact,
        render_no_results_message
    )
    from .strategy_builder import (
        render_strategy_builder,
        render_strategy_builder_compact,
        SimulationOptionPosition,
        SimulationStockPosition,
        export_positions_for_pnl_engine,
        get_net_premium
    )
    __all__.extend([
        'render_sidebar',
        'render_analysis_header',
        'render_strategy_summary_compact',
        'render_no_results_message',
        'render_strategy_builder',
        'render_strategy_builder_compact',
        'SimulationOptionPosition',
        'SimulationStockPosition',
        'export_positions_for_pnl_engine',
        'get_net_premium'
    ])
except ImportError:
    # Components not available, skip
    pass
