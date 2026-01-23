"""
Pricing Service - Compare Monte Carlo vs Analytical/FFT pricing.

Provides:
- Unified pricing comparison across methods
- Model-specific pricing engine selection
- Standard error estimation for MC
- Greeks computation where available
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

# Backend imports
from backend.simulation.base import SimulationResult
from backend.engines.analytic_engine import BSAnalyticEngine
from backend.engines.fft_engine import FFTEngine, FFTConfig
from backend.engines.mc_engine import MonteCarloEngine
from backend.models.gbm import GBMModel
from backend.models.heston import HestonModel
from backend.models.merton import MertonModel
from backend.models.bates import BatesModel
from backend.instruments.options import VanillaOption
from backend.core.result_types import ExerciseStyle
from backend.core.market import MarketEnvironment


@dataclass
class PricingComparison:
    """Results from pricing comparison."""
    # Monte Carlo
    mc_price: float
    mc_std_error: float
    mc_confidence_interval: Tuple[float, float]
    mc_n_paths: int

    # Analytical (BS) - only for GBM
    analytical_price: Optional[float] = None
    analytical_delta: Optional[float] = None
    analytical_gamma: Optional[float] = None
    analytical_vega: Optional[float] = None
    analytical_theta: Optional[float] = None

    # FFT - for models with characteristic function
    fft_price: Optional[float] = None

    # Comparison metrics
    mc_vs_analytical_error: Optional[float] = None
    mc_vs_fft_error: Optional[float] = None

    # Model info
    model: str = ""
    available_methods: List[str] = None

    def __post_init__(self):
        if self.available_methods is None:
            self.available_methods = ["monte_carlo"]

        # Compute errors
        if self.analytical_price is not None:
            self.mc_vs_analytical_error = self.mc_price - self.analytical_price
        if self.fft_price is not None:
            self.mc_vs_fft_error = self.mc_price - self.fft_price


def _create_vanilla_option(
    strike: float,
    time_to_maturity: float,
    is_call: bool = True
) -> VanillaOption:
    """Create a vanilla option instrument."""
    return VanillaOption(
        strike=strike,
        maturity=time_to_maturity,
        is_call=is_call,
        exercise=ExerciseStyle.EUROPEAN
    )


def _create_market(spot: float, rate: float) -> MarketEnvironment:
    """Create market environment."""
    return MarketEnvironment(spot=spot, rate=rate)


def _create_model(model_key: str, params: Dict[str, Any]):
    """Create pricing model from parameters."""
    model_lower = model_key.lower()

    if model_lower == "gbm":
        return GBMModel(sigma=params.get("sigma", 0.20))

    elif model_lower == "heston":
        return HestonModel(
            v0=params.get("v0", 0.04),
            kappa=params.get("kappa", 2.0),
            theta=params.get("theta", 0.04),
            xi=params.get("xi", 0.3),
            rho=params.get("rho", -0.7)
        )

    elif model_lower == "merton":
        return MertonModel(
            sigma=params.get("sigma", 0.20),
            lambda_j=params.get("lambda_j", 0.5),
            mu_j=params.get("mu_j", -0.1),
            sigma_j=params.get("sigma_j", 0.2)
        )

    elif model_lower == "bates":
        return BatesModel(
            v0=params.get("v0", 0.04),
            kappa=params.get("kappa", 2.0),
            theta=params.get("theta", 0.04),
            xi=params.get("xi", 0.3),
            rho=params.get("rho", -0.7),
            lambda_j=params.get("lambda_j", 0.5),
            mu_j=params.get("mu_j", -0.1),
            sigma_j=params.get("sigma_j", 0.2)
        )

    else:
        return None


def price_from_terminals(
    terminal_prices: np.ndarray,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    is_call: bool = True,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Price option from terminal prices using Monte Carlo.

    Args:
        terminal_prices: Array of terminal asset prices
        strike: Option strike price
        time_to_maturity: Time to maturity in years
        risk_free_rate: Risk-free rate
        is_call: True for call, False for put
        confidence_level: Confidence level for interval (default 95%)

    Returns:
        Dictionary with price, std_error, confidence_interval
    """
    # Compute payoffs
    if is_call:
        payoffs = np.maximum(terminal_prices - strike, 0)
    else:
        payoffs = np.maximum(strike - terminal_prices, 0)

    # Discount
    discount_factor = np.exp(-risk_free_rate * time_to_maturity)
    discounted_payoffs = payoffs * discount_factor

    # Statistics
    n_paths = len(terminal_prices)
    price = np.mean(discounted_payoffs)
    std = np.std(discounted_payoffs, ddof=1)
    std_error = std / np.sqrt(n_paths)

    # Confidence interval
    from scipy import stats
    z = stats.norm.ppf((1 + confidence_level) / 2)
    ci_lower = price - z * std_error
    ci_upper = price + z * std_error

    return {
        "price": price,
        "std_error": std_error,
        "confidence_interval": (ci_lower, ci_upper),
        "n_paths": n_paths
    }


def price_with_analytical(
    model_key: str,
    params: Dict[str, Any],
    strike: float,
    time_to_maturity: float,
    spot: float,
    risk_free_rate: float,
    is_call: bool = True
) -> Optional[Dict[str, float]]:
    """
    Price option using analytical formula (Black-Scholes).
    Only available for GBM model.

    Returns:
        Dictionary with price and Greeks, or None if not available
    """
    if model_key.lower() != "gbm":
        return None

    try:
        engine = BSAnalyticEngine()
        option = _create_vanilla_option(strike, time_to_maturity, is_call)
        model = GBMModel(sigma=params.get("sigma", 0.20))
        market = _create_market(spot, risk_free_rate)

        # Check if engine can price this
        if not engine.can_price(option, model):
            return None

        # Get price
        price_result = engine.price(option, model, market)

        # Get Greeks
        greeks_result = engine.greeks(option, model, market)

        return {
            "price": price_result.price,
            "delta": greeks_result.delta,
            "gamma": greeks_result.gamma,
            "vega": greeks_result.vega,
            "theta": greeks_result.theta,
            "rho": greeks_result.rho,
        }
    except Exception as e:
        print(f"Analytical pricing failed: {e}")
        return None


def price_with_fft(
    model_key: str,
    params: Dict[str, Any],
    strike: float,
    time_to_maturity: float,
    spot: float,
    risk_free_rate: float,
    is_call: bool = True
) -> Optional[float]:
    """
    Price option using FFT (Carr-Madan).
    Available for GBM, Heston, Merton, Bates.

    Returns:
        Option price or None if not available
    """
    model_lower = model_key.lower()

    # FFT not available for GARCH family
    if model_lower in ["garch", "ngarch", "gjr_garch"]:
        return None

    try:
        model = _create_model(model_key, params)
        if model is None:
            return None

        engine = FFTEngine(config=FFTConfig(alpha=1.5, n_fft=4096, eta=0.25))
        option = _create_vanilla_option(strike, time_to_maturity, is_call)
        market = _create_market(spot, risk_free_rate)

        # Check if engine can price this
        if not engine.can_price(option, model):
            return None

        result = engine.price(option, model, market)
        return result.price

    except Exception as e:
        print(f"FFT pricing failed: {e}")
        return None


def compare_pricing(
    model_key: str,
    params: Dict[str, Any],
    terminal_prices: np.ndarray,
    strike: float,
    time_to_maturity: float,
    spot: float,
    risk_free_rate: float,
    is_call: bool = True
) -> PricingComparison:
    """
    Compare pricing across all available methods for a model.

    Args:
        model_key: Model identifier
        params: Model parameters
        terminal_prices: Terminal prices from MC simulation
        strike: Option strike
        time_to_maturity: Time to maturity
        spot: Spot price
        risk_free_rate: Risk-free rate
        is_call: True for call

    Returns:
        PricingComparison with all available prices and errors
    """
    available_methods = ["monte_carlo"]

    # Monte Carlo pricing from terminals
    mc_result = price_from_terminals(
        terminal_prices=terminal_prices,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        is_call=is_call
    )

    comparison = PricingComparison(
        mc_price=mc_result["price"],
        mc_std_error=mc_result["std_error"],
        mc_confidence_interval=mc_result["confidence_interval"],
        mc_n_paths=mc_result["n_paths"],
        model=model_key
    )

    # Analytical pricing (GBM only)
    analytical_result = price_with_analytical(
        model_key=model_key,
        params=params,
        strike=strike,
        time_to_maturity=time_to_maturity,
        spot=spot,
        risk_free_rate=risk_free_rate,
        is_call=is_call
    )

    if analytical_result is not None:
        available_methods.append("analytical")
        comparison.analytical_price = analytical_result["price"]
        comparison.analytical_delta = analytical_result.get("delta")
        comparison.analytical_gamma = analytical_result.get("gamma")
        comparison.analytical_vega = analytical_result.get("vega")
        comparison.analytical_theta = analytical_result.get("theta")
        comparison.mc_vs_analytical_error = comparison.mc_price - comparison.analytical_price

    # FFT pricing
    fft_price = price_with_fft(
        model_key=model_key,
        params=params,
        strike=strike,
        time_to_maturity=time_to_maturity,
        spot=spot,
        risk_free_rate=risk_free_rate,
        is_call=is_call
    )

    if fft_price is not None:
        available_methods.append("fft")
        comparison.fft_price = fft_price
        comparison.mc_vs_fft_error = comparison.mc_price - comparison.fft_price

    comparison.available_methods = available_methods

    return comparison


def price_multiple_strikes(
    model_key: str,
    params: Dict[str, Any],
    simulation_result: SimulationResult,
    strikes: np.ndarray,
    time_to_maturity: float,
    spot: float,
    risk_free_rate: float,
    is_call: bool = True
) -> Dict[str, np.ndarray]:
    """
    Price options at multiple strikes.

    Returns:
        Dictionary with:
        - strikes: Strike array
        - mc_prices: MC prices
        - mc_errors: MC standard errors
        - analytical_prices: BS prices (if available)
        - fft_prices: FFT prices (if available)
    """
    terminal_prices = simulation_result.terminal_prices
    n_strikes = len(strikes)

    mc_prices = np.zeros(n_strikes)
    mc_errors = np.zeros(n_strikes)

    for i, strike in enumerate(strikes):
        mc_result = price_from_terminals(
            terminal_prices=terminal_prices,
            strike=strike,
            time_to_maturity=time_to_maturity,
            risk_free_rate=risk_free_rate,
            is_call=is_call
        )
        mc_prices[i] = mc_result["price"]
        mc_errors[i] = mc_result["std_error"]

    result = {
        "strikes": strikes,
        "mc_prices": mc_prices,
        "mc_errors": mc_errors,
    }

    # Analytical prices (GBM only)
    if model_key.lower() == "gbm":
        analytical_prices = np.zeros(n_strikes)
        for i, strike in enumerate(strikes):
            ana_result = price_with_analytical(
                model_key=model_key,
                params=params,
                strike=strike,
                time_to_maturity=time_to_maturity,
                spot=spot,
                risk_free_rate=risk_free_rate,
                is_call=is_call
            )
            if ana_result:
                analytical_prices[i] = ana_result["price"]
        result["analytical_prices"] = analytical_prices

    # FFT prices
    if model_key.lower() not in ["garch", "ngarch", "gjr_garch"]:
        try:
            model = _create_model(model_key, params)
            if model is not None:
                engine = FFTEngine()
                option = _create_vanilla_option(strikes[0], time_to_maturity, is_call)
                market = _create_market(spot, risk_free_rate)
                fft_prices = engine.price_strikes(option, model, market, strikes)
                result["fft_prices"] = fft_prices
        except Exception:
            pass

    return result


def get_available_pricing_methods(model_key: str) -> List[str]:
    """Get available pricing methods for a model."""
    model_lower = model_key.lower()

    if model_lower == "gbm":
        return ["analytical", "fft", "monte_carlo"]
    elif model_lower in ["heston", "merton", "bates"]:
        return ["fft", "monte_carlo"]
    elif model_lower in ["garch", "ngarch", "gjr_garch"]:
        return ["monte_carlo"]
    else:
        return ["monte_carlo"]


def compute_option_pnl(
    price_paths: np.ndarray,
    strike: float,
    premium: float,
    is_call: bool = True,
    is_long: bool = True,
    quantity: int = 1
) -> np.ndarray:
    """
    Compute P&L distribution for an option position.

    Args:
        price_paths: Price paths (n_paths, n_steps+1)
        strike: Option strike
        premium: Option premium paid/received
        is_call: True for call
        is_long: True for long position
        quantity: Number of contracts

    Returns:
        Array of terminal P&L values
    """
    terminal_prices = price_paths[:, -1]

    if is_call:
        payoff = np.maximum(terminal_prices - strike, 0)
    else:
        payoff = np.maximum(strike - terminal_prices, 0)

    if is_long:
        pnl = (payoff - premium) * quantity
    else:
        pnl = (premium - payoff) * quantity

    return pnl
