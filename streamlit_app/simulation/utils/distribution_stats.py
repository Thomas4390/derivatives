"""
Distribution Statistics Utilities for Monte Carlo Simulation Explorer.

Provides dataclasses and functions for computing and representing
distribution statistics from simulation results.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class DistributionStats:
    """
    Statistical summary of a distribution.

    Attributes
    ----------
    mean : float
        Mean value
    std : float
        Standard deviation
    min_val : float
        Minimum value
    max_val : float
        Maximum value
    skewness : float
        Skewness (measure of asymmetry)
    kurtosis : float
        Excess kurtosis (measure of tail heaviness)
    percentiles : Dict[int, float]
        Dictionary mapping percentile levels to values
    """
    mean: float
    std: float
    min_val: float
    max_val: float
    skewness: float
    kurtosis: float
    percentiles: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_array(
        cls,
        values: np.ndarray,
        percentile_levels: list[int] | None = None
    ) -> 'DistributionStats':
        """
        Create DistributionStats from a numpy array.

        Parameters
        ----------
        values : np.ndarray
            Array of values to analyze
        percentile_levels : Optional[List[int]]
            List of percentile levels to compute (default: [5, 25, 50, 75, 95])

        Returns
        -------
        DistributionStats
            Statistical summary of the distribution
        """
        if percentile_levels is None:
            percentile_levels = [5, 25, 50, 75, 95]

        return cls(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min_val=float(np.min(values)),
            max_val=float(np.max(values)),
            skewness=float(stats.skew(values)),
            kurtosis=float(stats.kurtosis(values)),
            percentiles={p: float(np.percentile(values, p)) for p in percentile_levels}
        )

    @property
    def median(self) -> float:
        """Get the median (50th percentile)."""
        return self.percentiles.get(50, np.nan)

    @property
    def iqr(self) -> float:
        """Get the interquartile range (75th - 25th percentile)."""
        p25 = self.percentiles.get(25, np.nan)
        p75 = self.percentiles.get(75, np.nan)
        return p75 - p25

    @property
    def range(self) -> float:
        """Get the range (max - min)."""
        return self.max_val - self.min_val

    @property
    def coefficient_of_variation(self) -> float:
        """Get the coefficient of variation (std/mean)."""
        if self.mean == 0:
            return np.inf
        return self.std / abs(self.mean)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'mean': self.mean,
            'std': self.std,
            'min': self.min_val,
            'max': self.max_val,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'median': self.median,
            'iqr': self.iqr,
            **{f'p{k}': v for k, v in self.percentiles.items()}
        }


@dataclass
class PnLStats(DistributionStats):
    """
    Extended statistics for P&L distributions.

    Includes risk metrics like VaR and probability of profit.
    """
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    prob_profit: float = 0.0
    prob_loss: float = 0.0
    expected_profit: float = 0.0
    expected_loss: float = 0.0

    @classmethod
    def from_pnl_array(
        cls,
        pnl: np.ndarray,
        percentile_levels: list[int] | None = None
    ) -> 'PnLStats':
        """
        Create PnLStats from a P&L array.

        Parameters
        ----------
        pnl : np.ndarray
            Array of P&L values
        percentile_levels : Optional[List[int]]
            List of percentile levels to compute

        Returns
        -------
        PnLStats
            Complete P&L statistics including risk metrics
        """
        if percentile_levels is None:
            percentile_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]

        # Base statistics
        base = DistributionStats.from_array(pnl, percentile_levels)

        # VaR calculations
        var_95 = float(np.percentile(pnl, 5))
        var_99 = float(np.percentile(pnl, 1))

        # CVaR calculations
        cvar_95 = float(np.mean(pnl[pnl <= var_95])) if np.any(pnl <= var_95) else var_95
        cvar_99 = float(np.mean(pnl[pnl <= var_99])) if np.any(pnl <= var_99) else var_99

        # Profit/Loss probabilities
        prob_profit = float(np.mean(pnl > 0))
        prob_loss = float(np.mean(pnl < 0))

        # Expected values conditional on profit/loss
        expected_profit = float(np.mean(pnl[pnl > 0])) if np.any(pnl > 0) else 0.0
        expected_loss = float(np.mean(pnl[pnl < 0])) if np.any(pnl < 0) else 0.0

        return cls(
            mean=base.mean,
            std=base.std,
            min_val=base.min_val,
            max_val=base.max_val,
            skewness=base.skewness,
            kurtosis=base.kurtosis,
            percentiles=base.percentiles,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            prob_profit=prob_profit,
            prob_loss=prob_loss,
            expected_profit=expected_profit,
            expected_loss=expected_loss
        )

    @property
    def profit_loss_ratio(self) -> float:
        """
        Calculate profit/loss ratio.

        Returns the ratio of expected profit to expected loss (absolute value).
        """
        if self.expected_loss == 0:
            return np.inf
        return abs(self.expected_profit / self.expected_loss)

    @property
    def expected_value(self) -> float:
        """
        Calculate expected value using profit/loss probabilities.

        E[PnL] = P(profit) * E[profit|profit] + P(loss) * E[loss|loss]
        """
        return (self.prob_profit * self.expected_profit +
                self.prob_loss * self.expected_loss)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            'prob_profit': self.prob_profit,
            'prob_loss': self.prob_loss,
            'expected_profit': self.expected_profit,
            'expected_loss': self.expected_loss,
            'profit_loss_ratio': self.profit_loss_ratio,
        })
        return base_dict


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_histogram_data(
    values: np.ndarray,
    bins: int = 50,
    density: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute histogram data for visualization.

    Parameters
    ----------
    values : np.ndarray
        Values to histogram
    bins : int
        Number of bins
    density : bool
        If True, normalize to density

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (counts, bin_edges, bin_centers)
    """
    counts, bin_edges = np.histogram(values, bins=bins, density=density)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return counts, bin_edges, bin_centers


def compute_kde(
    values: np.ndarray,
    n_points: int = 200,
    bandwidth: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute kernel density estimate.

    Parameters
    ----------
    values : np.ndarray
        Values for KDE
    n_points : int
        Number of points for KDE curve
    bandwidth : Optional[float]
        KDE bandwidth (None for automatic)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (x_values, kde_values)
    """
    try:
        if bandwidth is not None:
            kde = stats.gaussian_kde(values, bw_method=bandwidth)
        else:
            kde = stats.gaussian_kde(values)

        x_range = np.linspace(values.min(), values.max(), n_points)
        kde_values = kde(x_range)
        return x_range, kde_values
    except Exception:
        # Return empty arrays if KDE fails
        return np.array([]), np.array([])


def compute_empirical_cdf(
    values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical cumulative distribution function.

    Parameters
    ----------
    values : np.ndarray
        Values for ECDF

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (sorted_values, cdf_values)
    """
    sorted_values = np.sort(values)
    cdf_values = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    return sorted_values, cdf_values


def compute_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95
) -> tuple[float, float]:
    """
    Compute confidence interval for the mean.

    Parameters
    ----------
    values : np.ndarray
        Sample values
    confidence : float
        Confidence level (default: 0.95)

    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound)
    """
    n = len(values)
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(n)

    # t-value for given confidence level
    alpha = 1 - confidence
    t_val = stats.t.ppf(1 - alpha / 2, n - 1)

    margin = t_val * se
    return float(mean - margin), float(mean + margin)


def compare_distributions(
    values1: np.ndarray,
    values2: np.ndarray
) -> dict[str, float]:
    """
    Compare two distributions using various statistical tests.

    Parameters
    ----------
    values1 : np.ndarray
        First distribution
    values2 : np.ndarray
        Second distribution

    Returns
    -------
    Dict[str, float]
        Dictionary with test statistics and p-values
    """
    results = {}

    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(values1, values2)
    results['ks_statistic'] = float(ks_stat)
    results['ks_pvalue'] = float(ks_pval)

    # Mann-Whitney U test (non-parametric)
    try:
        mw_stat, mw_pval = stats.mannwhitneyu(values1, values2, alternative='two-sided')
        results['mannwhitney_statistic'] = float(mw_stat)
        results['mannwhitney_pvalue'] = float(mw_pval)
    except Exception:
        results['mannwhitney_statistic'] = np.nan
        results['mannwhitney_pvalue'] = np.nan

    # Welch's t-test
    t_stat, t_pval = stats.ttest_ind(values1, values2, equal_var=False)
    results['ttest_statistic'] = float(t_stat)
    results['ttest_pvalue'] = float(t_pval)

    return results
