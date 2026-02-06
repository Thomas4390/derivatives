"""
Test Configuration and Fixtures
================================

Shared fixtures and configuration for the test suite.

Author: Thomas
Created: 2025
"""

import pytest
import numpy as np
from typing import Tuple

# =============================================================================
# Backend Imports
# =============================================================================

from backend.instruments.options import VanillaOption
from backend.models.gbm import GBMModel
from backend.models.heston import HestonModel
from backend.models.bates import BatesModel
from backend.models.merton import MertonModel
from backend.models.garch import GARCHModel, NGARCHModel, GJRGARCHModel
from backend.engines import BSAnalyticEngine, FFTEngine, MonteCarloEngine
from backend.core.market import MarketEnvironment
from backend.portfolio import (
    OptionsPortfolio,
    PortfolioPosition,
    StockPosition,
    long_call,
    short_call,
    long_put,
    short_put,
    long_stock,
)


# =============================================================================
# Test Tolerances
# =============================================================================

# Absolute tolerance for price comparisons
PRICE_ATOL = 0.01  # $0.01

# Relative tolerance for price comparisons
PRICE_RTOL = 0.001  # 0.1%

# Tolerance for Monte Carlo (higher due to sampling error)
MC_RTOL = 0.02  # 2%

# Tolerance for Greeks
GREEK_ATOL = 1e-4
GREEK_RTOL = 0.01  # 1%

# Tolerance for theoretical tests (very tight)
THEORY_ATOL = 1e-6


# =============================================================================
# Market Environment Fixtures
# =============================================================================

@pytest.fixture
def market_atm() -> MarketEnvironment:
    """Standard ATM market environment."""
    return MarketEnvironment(spot=100.0, rate=0.05, dividend_yield=0.0)


@pytest.fixture
def market_with_dividend() -> MarketEnvironment:
    """Market with dividend yield."""
    return MarketEnvironment(spot=100.0, rate=0.05, dividend_yield=0.02)


@pytest.fixture
def market_zero_rate() -> MarketEnvironment:
    """Zero interest rate market."""
    return MarketEnvironment(spot=100.0, rate=0.0, dividend_yield=0.0)


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def gbm_model() -> GBMModel:
    """Standard GBM model with 20% volatility."""
    return GBMModel(sigma=0.20)


@pytest.fixture
def gbm_high_vol() -> GBMModel:
    """High volatility GBM model."""
    return GBMModel(sigma=0.40)


@pytest.fixture
def heston_model() -> HestonModel:
    """Standard Heston model."""
    return HestonModel(
        v0=0.04,      # Initial variance (20% vol)
        kappa=2.0,    # Mean reversion speed
        theta=0.04,   # Long-term variance
        xi=0.3,       # Vol of vol
        rho=-0.7      # Correlation
    )


@pytest.fixture
def bates_model() -> BatesModel:
    """Standard Bates model (Heston + jumps)."""
    return BatesModel(
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        lambda_j=0.5,   # Jump intensity
        mu_j=-0.1,      # Mean jump size
        sigma_j=0.2     # Jump volatility
    )


@pytest.fixture
def merton_model() -> MertonModel:
    """Standard Merton jump-diffusion model."""
    return MertonModel(
        sigma=0.20,
        lambda_j=0.5,
        mu_j=-0.1,
        sigma_j=0.2
    )


@pytest.fixture
def garch_model() -> GARCHModel:
    """Standard GARCH(1,1) model."""
    return GARCHModel(sigma0=0.20, omega=0.000002, alpha=0.05, beta=0.90)


@pytest.fixture
def ngarch_model() -> NGARCHModel:
    """NGARCH model with leverage effect."""
    return NGARCHModel(sigma0=0.20, omega=0.000002, alpha=0.04, beta=0.90, theta=0.5)


@pytest.fixture
def gjr_garch_model() -> GJRGARCHModel:
    """GJR-GARCH model with asymmetric response."""
    return GJRGARCHModel(sigma0=0.20, omega=0.000002, alpha=0.03, beta=0.90, gamma=0.04)


# =============================================================================
# Engine Fixtures
# =============================================================================

@pytest.fixture
def bs_engine() -> BSAnalyticEngine:
    """Black-Scholes analytical engine."""
    return BSAnalyticEngine()


@pytest.fixture
def fft_engine() -> FFTEngine:
    """FFT pricing engine."""
    return FFTEngine()


@pytest.fixture
def mc_engine() -> MonteCarloEngine:
    """Monte Carlo engine with moderate paths."""
    return MonteCarloEngine(n_paths=50000, seed=42)


@pytest.fixture
def mc_engine_high_precision() -> MonteCarloEngine:
    """High precision Monte Carlo engine."""
    return MonteCarloEngine(n_paths=200000, seed=42)


# =============================================================================
# Option Fixtures
# =============================================================================

@pytest.fixture
def call_atm() -> VanillaOption:
    """ATM call option, 3 months maturity."""
    return VanillaOption(strike=100.0, maturity=0.25, is_call=True)


@pytest.fixture
def put_atm() -> VanillaOption:
    """ATM put option, 3 months maturity."""
    return VanillaOption(strike=100.0, maturity=0.25, is_call=False)


@pytest.fixture
def call_itm() -> VanillaOption:
    """ITM call option (S=100, K=90)."""
    return VanillaOption(strike=90.0, maturity=0.25, is_call=True)


@pytest.fixture
def call_otm() -> VanillaOption:
    """OTM call option (S=100, K=110)."""
    return VanillaOption(strike=110.0, maturity=0.25, is_call=True)


@pytest.fixture
def put_itm() -> VanillaOption:
    """ITM put option (S=100, K=110)."""
    return VanillaOption(strike=110.0, maturity=0.25, is_call=False)


@pytest.fixture
def put_otm() -> VanillaOption:
    """OTM put option (S=100, K=90)."""
    return VanillaOption(strike=90.0, maturity=0.25, is_call=False)


@pytest.fixture
def call_long_maturity() -> VanillaOption:
    """Long maturity call (1 year)."""
    return VanillaOption(strike=100.0, maturity=1.0, is_call=True)


# =============================================================================
# Portfolio Fixtures
# =============================================================================

@pytest.fixture
def bull_call_spread(gbm_model) -> OptionsPortfolio:
    """Bull call spread: long K=95, short K=105."""
    portfolio = OptionsPortfolio(model=gbm_model)
    portfolio.add(long_call(strike=95, maturity=0.25, premium=8.0))
    portfolio.add(short_call(strike=105, maturity=0.25, premium=3.0))
    return portfolio


@pytest.fixture
def long_straddle(gbm_model) -> OptionsPortfolio:
    """Long straddle: long call + long put at same strike."""
    portfolio = OptionsPortfolio(model=gbm_model)
    portfolio.add(long_call(strike=100, maturity=0.25, premium=5.0))
    portfolio.add(long_put(strike=100, maturity=0.25, premium=4.5))
    return portfolio


@pytest.fixture
def covered_call(gbm_model) -> OptionsPortfolio:
    """Covered call: long stock + short call."""
    portfolio = OptionsPortfolio(model=gbm_model)
    portfolio.add(long_stock(quantity=100, entry_price=100.0))
    portfolio.add(short_call(strike=105, maturity=0.25, premium=3.0))
    return portfolio


@pytest.fixture
def iron_condor(gbm_model) -> OptionsPortfolio:
    """Iron condor strategy."""
    portfolio = OptionsPortfolio(model=gbm_model)
    # Bull put spread
    portfolio.add(short_put(strike=95, maturity=0.25, premium=2.0))
    portfolio.add(long_put(strike=90, maturity=0.25, premium=1.0))
    # Bear call spread
    portfolio.add(short_call(strike=105, maturity=0.25, premium=2.0))
    portfolio.add(long_call(strike=110, maturity=0.25, premium=1.0))
    return portfolio


# =============================================================================
# Helper Functions
# =============================================================================

def assert_price_close(actual: float, expected: float, rtol: float = PRICE_RTOL, atol: float = PRICE_ATOL):
    """Assert that two prices are close within tolerances."""
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def assert_greek_close(actual: float, expected: float, rtol: float = GREEK_RTOL, atol: float = GREEK_ATOL):
    """Assert that two Greeks are close within tolerances."""
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


# =============================================================================
# Test Output Helpers
# =============================================================================

class TestReporter:
    """
    Helper class for consistent test output formatting.

    Usage in tests:
        from tests.conftest import report

        def test_something():
            result = compute_something()
            report.value("Price", result, expected=5.0, unit="$")
    """

    def __init__(self, precision: int = 6):
        self.precision = precision

    def header(self, title: str) -> None:
        """Print a section header."""
        print(f"\n{'─' * 50}")
        print(f"  {title}")
        print(f"{'─' * 50}")

    def value(
        self,
        label: str,
        actual: float,
        expected: float = None,
        unit: str = "",
        precision: int = None
    ) -> None:
        """
        Print a single value comparison.

        Args:
            label: Description of the value
            actual: Computed value
            expected: Expected value (optional)
            unit: Unit string (e.g., "$", "%")
            precision: Override default precision
        """
        prec = precision if precision is not None else self.precision

        if expected is not None:
            diff = actual - expected
            rel_diff = abs(diff / expected) * 100 if expected != 0 else 0
            print(f"  {label}:")
            print(f"    Computed: {actual:>{prec + 8}.{prec}f} {unit}")
            print(f"    Expected: {expected:>{prec + 8}.{prec}f} {unit}")
            print(f"    Diff:     {diff:>+{prec + 8}.{prec}f} {unit} ({rel_diff:.4f}%)")
        else:
            print(f"  {label}: {actual:.{prec}f} {unit}")

    def comparison(
        self,
        label1: str,
        value1: float,
        label2: str,
        value2: float,
        unit: str = "",
        precision: int = None
    ) -> None:
        """
        Print a comparison between two computed values.

        Args:
            label1: Description of first value
            value1: First computed value
            label2: Description of second value
            value2: Second computed value
            unit: Unit string
            precision: Override default precision
        """
        prec = precision if precision is not None else self.precision
        diff = value1 - value2
        rel_diff = abs(diff / value2) * 100 if value2 != 0 else 0

        print(f"  {label1}: {value1:.{prec}f} {unit}")
        print(f"  {label2}: {value2:.{prec}f} {unit}")
        print(f"  Difference: {diff:+.{prec}f} {unit} ({rel_diff:.4f}%)")

    def greeks(self, greeks_obj, label: str = "Greeks") -> None:
        """
        Print all Greeks from a Greeks result object.

        Args:
            greeks_obj: Object with delta, gamma, vega, theta, rho attributes
            label: Optional label for the section
        """
        print(f"  {label}:")
        print(f"    Delta: {greeks_obj.delta:>12.6f}")
        print(f"    Gamma: {greeks_obj.gamma:>12.6f}")
        print(f"    Vega:  {greeks_obj.vega:>12.6f}")
        print(f"    Theta: {greeks_obj.theta:>12.6f}")
        if hasattr(greeks_obj, 'rho') and greeks_obj.rho is not None:
            print(f"    Rho:   {greeks_obj.rho:>12.6f}")

    def table(
        self,
        headers: list,
        rows: list,
        title: str = None,
        precision: int = None
    ) -> None:
        """
        Print a formatted table of results.

        Args:
            headers: List of column headers
            rows: List of row tuples/lists
            title: Optional table title
            precision: Override default precision
        """
        prec = precision if precision is not None else self.precision

        if title:
            print(f"\n  {title}:")

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                if isinstance(val, float):
                    val_str = f"{val:.{prec}f}"
                else:
                    val_str = str(val)
                col_widths[i] = max(col_widths[i], len(val_str))

        # Print header
        header_str = "  │ " + " │ ".join(h.center(col_widths[i]) for i, h in enumerate(headers)) + " │"
        separator = "  ├─" + "─┼─".join("─" * w for w in col_widths) + "─┤"
        top_border = "  ┌─" + "─┬─".join("─" * w for w in col_widths) + "─┐"
        bottom_border = "  └─" + "─┴─".join("─" * w for w in col_widths) + "─┘"

        print(top_border)
        print(header_str)
        print(separator)

        # Print rows
        for row in rows:
            row_strs = []
            for i, val in enumerate(row):
                if isinstance(val, float):
                    val_str = f"{val:.{prec}f}"
                else:
                    val_str = str(val)
                row_strs.append(val_str.rjust(col_widths[i]))
            print("  │ " + " │ ".join(row_strs) + " │")

        print(bottom_border)

    def array_stats(
        self,
        arr: np.ndarray,
        label: str = "Array",
        precision: int = None
    ) -> None:
        """
        Print statistics for a numpy array.

        Args:
            arr: Numpy array
            label: Description label
            precision: Override default precision
        """
        prec = precision if precision is not None else self.precision

        print(f"  {label} statistics:")
        print(f"    Mean:   {np.mean(arr):.{prec}f}")
        print(f"    Std:    {np.std(arr):.{prec}f}")
        print(f"    Min:    {np.min(arr):.{prec}f}")
        print(f"    Max:    {np.max(arr):.{prec}f}")
        print(f"    Median: {np.median(arr):.{prec}f}")

    def success(self, message: str) -> None:
        """Print a success message."""
        print(f"  ✓ {message}")

    def info(self, message: str) -> None:
        """Print an info message."""
        print(f"  ℹ {message}")

    def params(self, **kwargs) -> None:
        """Print test parameters."""
        print("  Parameters:")
        for key, value in kwargs.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.6f}")
            else:
                print(f"    {key}: {value}")


# Global reporter instance for easy access
report = TestReporter()


# =============================================================================
# Known Values (for validation)
# =============================================================================

# Black-Scholes benchmark values (S=100, K=100, T=0.25, r=5%, σ=20%, q=0%)
# Calculated using standard BS formula with scipy.stats.norm
BS_BENCHMARK = {
    "call_price": 4.6150,
    "put_price": 3.3728,
    "delta_call": 0.5695,
    "delta_put": -0.4305,
    "gamma": 0.0393,
    "vega": 19.6440,  # Per 100% vol move
    "theta_call": -10.4742,  # Per year
    "rho_call": 13.0828,  # Per 100% rate move
}
