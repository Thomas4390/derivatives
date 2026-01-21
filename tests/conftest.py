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
