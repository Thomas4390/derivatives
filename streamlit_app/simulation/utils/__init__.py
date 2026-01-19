"""
Utility modules for Monte Carlo Simulation Streamlit App.
"""

from .chart_helpers import (
    apply_default_layout,
    add_percentile_band,
    add_horizontal_line,
    add_vertical_line,
    create_base_figure,
    add_confidence_band,
    format_currency,
    format_percentage,
    add_sample_paths,
    add_mean_path,
    format_large_number,
    get_profit_loss_color,
)

from .distribution_stats import (
    DistributionStats,
    PnLStats,
    compute_histogram_data,
    compute_kde,
    compute_empirical_cdf,
    compute_confidence_interval,
    compare_distributions,
)

__all__ = [
    # Chart helpers
    'apply_default_layout',
    'add_percentile_band',
    'add_horizontal_line',
    'add_vertical_line',
    'create_base_figure',
    'add_confidence_band',
    'format_currency',
    'format_percentage',
    'add_sample_paths',
    'add_mean_path',
    'format_large_number',
    'get_profit_loss_color',
    # Distribution stats
    'DistributionStats',
    'PnLStats',
    'compute_histogram_data',
    'compute_kde',
    'compute_empirical_cdf',
    'compute_confidence_interval',
    'compare_distributions',
]
