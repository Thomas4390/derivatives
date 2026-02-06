"""Charts module for Monte Carlo Simulation Explorer."""

from .volatility_paths import render_volatility_paths_tab
from .distributions import render_distributions_tab
from .statistics import render_statistics_tab
from .unified_paths import (
    render_unified_paths,
    render_price_paths_only,
    render_volatility_paths_only,
    render_path_controls
)
from .pnl_distribution import render_pnl_distribution_tab, render_risk_metrics_tab
from .scenario_analysis import render_scenario_analysis_tab
from .greeks_decomposition import render_greeks_decomposition_tab
from .volatility_impact import render_volatility_impact_tab
from .convergence_analysis import render_convergence_analysis_tab
from .strategy_comparison import render_strategy_comparison_tab
from .scenario_dashboard import render_scenario_dashboard_tab

__all__ = [
    'render_volatility_paths_tab',
    'render_distributions_tab',
    'render_statistics_tab',
    'render_unified_paths',
    'render_price_paths_only',
    'render_volatility_paths_only',
    'render_path_controls',
    'render_pnl_distribution_tab',
    'render_risk_metrics_tab',
    'render_scenario_analysis_tab',
    'render_greeks_decomposition_tab',
    'render_volatility_impact_tab',
    'render_convergence_analysis_tab',
    'render_strategy_comparison_tab',
    'render_scenario_dashboard_tab',
]
