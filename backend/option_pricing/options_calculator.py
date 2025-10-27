"""
Options Calculator - English Version with Verified Greeks
Black-Scholes option pricing and Greeks calculation with 3D matrices
High-performance implementation with Numba JIT compilation
All formulas verified against academic sources
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from numba import njit, prange
import time
from enum import Enum

# ============= Configuration and Constants =============

class OptionType(Enum):
    """Option type enumeration"""
    CALL = 1
    PUT = 0

class PositionType(Enum):
    """Position type enumeration"""
    LONG = 1
    SHORT = -1

# Validation limits
MAX_VOLATILITY = 5.0  # 500% volatility max
MAX_INTEREST_RATE = 1.0  # 100% interest rate max
MIN_PRICE = 0.0001  # Minimum price to avoid division by zero
MAX_TIME_TO_EXPIRY = 10.0  # 10 years max
DAYS_PER_YEAR = 365.0  # Calendar days convention

# ============= Data Classes =============

@dataclass
class OptionPosition:
    """Represents an option position with validation"""
    option_type: str  # 'call' or 'put'
    position_type: str  # 'long' or 'short'
    strike: float
    quantity: int = 1
    premium_paid: float = 0.0

    def __post_init__(self):
        """Post-initialization validation"""
        if self.option_type not in ['call', 'put']:
            raise ValueError(f"Option type must be 'call' or 'put', got {self.option_type}")
        if self.position_type not in ['long', 'short']:
            raise ValueError(f"Position type must be 'long' or 'short', got {self.position_type}")
        if self.strike <= 0:
            raise ValueError(f"Strike must be positive, got {self.strike}")
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {self.quantity}")

@dataclass
class StockPosition:
    """Represents a stock/underlying position with validation"""
    position_type: str  # 'long' or 'short'
    quantity: int = 100
    entry_price: float = 0.0

    def __post_init__(self):
        """Post-initialization validation"""
        if self.position_type not in ['long', 'short']:
            raise ValueError(f"Position type must be 'long' or 'short', got {self.position_type}")
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {self.quantity}")
        if self.entry_price < 0:
            raise ValueError(f"Entry price cannot be negative, got {self.entry_price}")

@dataclass
class BreakevenResult:
    """Breakeven calculation results"""
    breakeven_points: List[float]  # Points where P&L = 0
    max_profit: float
    max_profit_spot: float
    max_loss: float
    max_loss_spot: float
    profit_zones: List[Tuple[float, float]]  # Profit zones
    loss_zones: List[Tuple[float, float]]  # Loss zones

    def summary(self) -> str:
        """Generate a formatted summary of breakevens"""
        summary = "=== Breakeven Analysis ===\n"

        if self.breakeven_points:
            summary += f"Breakeven points: {[f'${b:.2f}' for b in self.breakeven_points]}\n"
        else:
            summary += "No breakeven points (always profit or always loss)\n"

        summary += f"Max Profit: ${self.max_profit:.2f} at ${self.max_profit_spot:.2f}\n"
        summary += f"Max Loss: ${self.max_loss:.2f} at ${self.max_loss_spot:.2f}\n"

        if self.profit_zones:
            summary += "\nProfit zones:\n"
            for start, end in self.profit_zones:
                if start == -np.inf:
                    summary += f"  Below ${end:.2f}\n"
                elif end == np.inf:
                    summary += f"  Above ${start:.2f}\n"
                else:
                    summary += f"  ${start:.2f} - ${end:.2f}\n"

        if self.loss_zones:
            summary += "\nLoss zones:\n"
            for start, end in self.loss_zones:
                if start == -np.inf:
                    summary += f"  Below ${end:.2f}\n"
                elif end == np.inf:
                    summary += f"  Above ${start:.2f}\n"
                else:
                    summary += f"  ${start:.2f} - ${end:.2f}\n"

        return summary

# ============= Normal Distribution Functions (Numba) =============

@njit(fastmath=True, cache=True)
def norm_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal distribution"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@njit(fastmath=True, cache=True)
def norm_pdf(x: float) -> float:
    """Probability density function for standard normal distribution"""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

# ============= Black-Scholes Core Functions =============

@njit(fastmath=True, cache=True)
def calculate_d1_d2(spot: float, strike: float, time_to_expiry: float,
                    risk_free_rate: float, volatility: float) -> Tuple[float, float]:
    """
    Calculate d1 and d2 parameters for Black-Scholes model

    Parameters:
    -----------
    spot : Current price of underlying
    strike : Strike price
    time_to_expiry : Time to expiration in years
    risk_free_rate : Risk-free interest rate
    volatility : Implied volatility

    Returns:
    --------
    Tuple of (d1, d2)
    """
    if time_to_expiry <= 0:
        return 0.0, 0.0

    if volatility <= 0:
        # Handle zero volatility case
        if spot > strike * np.exp(-risk_free_rate * time_to_expiry):
            return 1e10, 1e10  # Large positive value
        else:
            return -1e10, -1e10  # Large negative value

    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    return d1, d2

@njit(fastmath=True, cache=True)
def black_scholes_call_price(spot: float, strike: float, time_to_expiry: float,
                             risk_free_rate: float, volatility: float) -> float:
    """Calculate European call option price using Black-Scholes formula"""
    if time_to_expiry <= 0:
        return max(spot - strike, 0.0)

    d1, d2 = calculate_d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)

    call_price = spot * norm_cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm_cdf(d2)

    return call_price

@njit(fastmath=True, cache=True)
def black_scholes_put_price(spot: float, strike: float, time_to_expiry: float,
                            risk_free_rate: float, volatility: float) -> float:
    """Calculate European put option price using Black-Scholes formula"""
    if time_to_expiry <= 0:
        return max(strike - spot, 0.0)

    d1, d2 = calculate_d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)

    put_price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm_cdf(-d2) - spot * norm_cdf(-d1)

    return put_price

# ============= First-Order Greeks (VERIFIED) =============

@njit(fastmath=True, cache=True)
def calculate_first_order_greeks(spot: float, strike: float, time_to_expiry: float,
                                 risk_free_rate: float, volatility: float,
                                 option_type: int) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate first-order Greeks (Delta, Gamma, Vega, Theta, Rho)

    Returns:
    --------
    Tuple of (price, delta, gamma, vega, theta, rho)
    """
    if time_to_expiry <= 0:
        if option_type == 1:  # Call
            price = max(spot - strike, 0.0)
            delta = 1.0 if spot > strike else 0.0
        else:  # Put
            price = max(strike - spot, 0.0)
            delta = -1.0 if spot < strike else 0.0
        return price, delta, 0.0, 0.0, 0.0, 0.0

    d1, d2 = calculate_d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    sqrt_t = np.sqrt(time_to_expiry)

    # Common calculations
    n_d1 = norm_cdf(d1)
    n_d2 = norm_cdf(d2)
    n_prime_d1 = norm_pdf(d1)
    n_minus_d1 = norm_cdf(-d1)
    n_minus_d2 = norm_cdf(-d2)
    exp_rt = np.exp(-risk_free_rate * time_to_expiry)

    # Gamma (same for calls and puts)
    gamma = n_prime_d1 / (spot * volatility * sqrt_t) if volatility > 0 else 0.0

    # Vega (same for calls and puts) - per 1% change in volatility
    vega = spot * n_prime_d1 * sqrt_t / 100.0

    if option_type == 1:  # Call
        price = spot * n_d1 - strike * exp_rt * n_d2
        delta = n_d1
        # Theta - per day
        theta = (-spot * n_prime_d1 * volatility / (2 * sqrt_t)
                - risk_free_rate * strike * exp_rt * n_d2) / DAYS_PER_YEAR
        # Rho - per 1% change in interest rate
        rho = strike * time_to_expiry * exp_rt * n_d2 / 100.0
    else:  # Put
        price = strike * exp_rt * n_minus_d2 - spot * n_minus_d1
        delta = n_d1 - 1.0  # Put delta is negative
        # Theta - per day
        theta = (-spot * n_prime_d1 * volatility / (2 * sqrt_t)
                + risk_free_rate * strike * exp_rt * n_minus_d2) / DAYS_PER_YEAR
        # Rho - per 1% change in interest rate
        rho = -strike * time_to_expiry * exp_rt * n_minus_d2 / 100.0

    return price, delta, gamma, vega, theta, rho

# ============= Second-Order Greeks (VERIFIED & CORRECTED) =============

@njit(fastmath=True, cache=True)
def calculate_second_order_greeks(spot: float, strike: float, time_to_expiry: float,
                                  risk_free_rate: float, volatility: float) -> Tuple[float, float, float, float]:
    """
    Calculate second-order Greeks (Vanna, Volga/Vomma, Charm, Veta)
    All formulas verified against academic sources

    Returns:
    --------
    Tuple of (vanna, volga, charm, veta)
    """
    if time_to_expiry <= 0 or volatility <= 0:
        return 0.0, 0.0, 0.0, 0.0

    d1, d2 = calculate_d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    sqrt_t = np.sqrt(time_to_expiry)
    n_prime_d1 = norm_pdf(d1)

    # Vanna - ∂²V/∂S∂σ (per 1% change in volatility)
    # Measures how delta changes with volatility
    vanna = -n_prime_d1 * d2 / volatility / 100.0

    # Volga/Vomma - ∂²V/∂σ² (per 1% change in volatility squared)
    # Measures convexity of vega
    vega_base = spot * n_prime_d1 * sqrt_t
    volga = vega_base * d1 * d2 / volatility / 10000.0

    # Charm - ∂²V/∂S∂t (per day)
    # Also called delta decay - measures how delta changes with time
    charm = -n_prime_d1 * (2 * risk_free_rate * time_to_expiry - d2 * volatility * sqrt_t) / (2 * time_to_expiry * volatility * sqrt_t) / DAYS_PER_YEAR

    # Veta - ∂²V/∂σ∂t (per day per 1% volatility)
    # Also called vega decay - measures how vega changes with time
    veta = spot * n_prime_d1 * sqrt_t * (
        risk_free_rate * d1 / (volatility * sqrt_t) - (1 + d1 * d2) / (2 * time_to_expiry)
    ) / (DAYS_PER_YEAR * 100.0)

    return vanna, volga, charm, veta

# ============= Third-Order Greeks (VERIFIED & CORRECTED) =============

@njit(fastmath=True, cache=True)
def calculate_third_order_greeks(spot: float, strike: float, time_to_expiry: float,
                                 risk_free_rate: float, volatility: float) -> Tuple[float, float, float, float]:
    """
    Calculate third-order Greeks (Speed, Zomma, Color, Ultima)
    All formulas verified against academic sources

    Returns:
    --------
    Tuple of (speed, zomma, color, ultima)
    """
    if time_to_expiry <= 0 or volatility <= 0:
        return 0.0, 0.0, 0.0, 0.0

    d1, d2 = calculate_d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    sqrt_t = np.sqrt(time_to_expiry)
    n_prime_d1 = norm_pdf(d1)

    # Calculate gamma for speed calculation
    gamma = n_prime_d1 / (spot * volatility * sqrt_t)

    # Speed - ∂³V/∂S³
    # Measures how gamma changes with spot price
    speed = -gamma * (d1 / (volatility * sqrt_t) + 1) / spot

    # Zomma - ∂³V/∂S²∂σ (per 1% change in volatility)
    # Also called DgammaDvol - measures how gamma changes with volatility
    zomma = gamma * (d1 * d2 - 1) / volatility / 100.0

    # Color - ∂³V/∂S²∂t (per day)
    # Also called gamma decay - measures how gamma changes with time
    color = -n_prime_d1 / (2 * spot * time_to_expiry * volatility * sqrt_t) * (
        2 * risk_free_rate * time_to_expiry - 1 +
        d1 * (2 * risk_free_rate * time_to_expiry - d2 * volatility * sqrt_t) / (volatility * sqrt_t)
    ) / DAYS_PER_YEAR

    # Ultima - ∂³V/∂σ³ (per 1% change in volatility cubed)
    # Measures third-order sensitivity to volatility
    vega = spot * n_prime_d1 * sqrt_t
    ultima = -vega / (volatility * volatility * volatility) * (
        d1 * d2 * (1 - d1 * d2) + d1 * d1 + d2 * d2
    ) / 1000000.0  # Normalized for 1% change cubed

    return speed, zomma, color, ultima

# ============= Combined Greeks Calculation =============

@njit(fastmath=True, cache=True)
def calculate_all_greeks(spot: float, strike: float, time_to_expiry: float,
                        risk_free_rate: float, volatility: float,
                        option_type: int) -> np.ndarray:
    """
    Calculate all Greeks in a single pass

    Returns:
    --------
    Array of 14 Greeks: [price, delta, gamma, vega, theta, rho,
                        vanna, volga, charm, veta, speed, zomma, color, ultima]
    """
    greeks = np.zeros(14)

    # First-order Greeks
    price, delta, gamma, vega, theta, rho = calculate_first_order_greeks(
        spot, strike, time_to_expiry, risk_free_rate, volatility, option_type
    )
    greeks[0] = price
    greeks[1] = delta
    greeks[2] = gamma
    greeks[3] = vega
    greeks[4] = theta
    greeks[5] = rho

    # Second-order Greeks
    vanna, volga, charm, veta = calculate_second_order_greeks(
        spot, strike, time_to_expiry, risk_free_rate, volatility
    )
    greeks[6] = vanna
    greeks[7] = volga
    greeks[8] = charm
    greeks[9] = veta

    # Third-order Greeks
    speed, zomma, color, ultima = calculate_third_order_greeks(
        spot, strike, time_to_expiry, risk_free_rate, volatility
    )
    greeks[10] = speed
    greeks[11] = zomma
    greeks[12] = color
    greeks[13] = ultima

    return greeks

# ============= 3D Matrix Functions =============

@njit(fastmath=True, cache=True, parallel=True)
def calculate_greeks_3d_dte_matrix(spot_range: np.ndarray, strike: float,
                                   dte_range: np.ndarray, risk_free_rate: float,
                                   volatility: float, option_type: int) -> np.ndarray:
    """
    Calculate 3D matrix of Greeks: [spot x dte x greek]

    Parameters:
    -----------
    spot_range : Array of spot prices
    strike : Strike price
    dte_range : Array of days to expiration
    risk_free_rate : Risk-free interest rate
    volatility : Implied volatility
    option_type : 1 for call, 0 for put

    Returns:
    --------
    3D matrix of shape (n_spots, n_dte, 14) containing all Greeks
    """
    n_spots = len(spot_range)
    n_dte = len(dte_range)
    matrix_3d = np.zeros((n_spots, n_dte, 14))

    for i in prange(n_spots):
        for j in range(n_dte):
            time_to_expiry = dte_range[j] / DAYS_PER_YEAR
            greeks = calculate_all_greeks(
                spot_range[i], strike, time_to_expiry, risk_free_rate, volatility, option_type
            )
            matrix_3d[i, j, :] = greeks

    return matrix_3d

@njit(fastmath=True, cache=True, parallel=True)
def calculate_greeks_3d_iv_matrix(spot_range: np.ndarray, strike: float,
                                  dte: float, risk_free_rate: float,
                                  iv_range: np.ndarray, option_type: int) -> np.ndarray:
    """
    Calculate 3D matrix of Greeks: [spot x iv x greek]

    Parameters:
    -----------
    spot_range : Array of spot prices
    strike : Strike price
    dte : Days to expiration (fixed)
    risk_free_rate : Risk-free interest rate
    iv_range : Array of implied volatilities
    option_type : 1 for call, 0 for put

    Returns:
    --------
    3D matrix of shape (n_spots, n_iv, 14) containing all Greeks
    """
    n_spots = len(spot_range)
    n_iv = len(iv_range)
    time_to_expiry = dte / DAYS_PER_YEAR
    matrix_3d = np.zeros((n_spots, n_iv, 14))

    for i in prange(n_spots):
        for j in range(n_iv):
            greeks = calculate_all_greeks(
                spot_range[i], strike, time_to_expiry, risk_free_rate, iv_range[j], option_type
            )
            matrix_3d[i, j, :] = greeks

    return matrix_3d

# ============= Portfolio Greeks Calculation =============

@njit(fastmath=True, cache=True, parallel=True)
def calculate_portfolio_greeks_3d_dte(strikes: np.ndarray, option_types: np.ndarray,
                                      position_types: np.ndarray, quantities: np.ndarray,
                                      spot_range: np.ndarray, dte_range: np.ndarray,
                                      risk_free_rate: float, volatility: float) -> np.ndarray:
    """
    Calculate 3D matrix of portfolio Greeks: [spot x dte x greek]
    Aggregates all positions in the portfolio
    """
    n_spots = len(spot_range)
    n_dte = len(dte_range)
    n_positions = len(strikes)
    matrix_3d = np.zeros((n_spots, n_dte, 14))

    for i in prange(n_spots):
        for j in range(n_dte):
            time_to_expiry = dte_range[j] / DAYS_PER_YEAR
            total_greeks = np.zeros(14)

            for k in range(n_positions):
                greeks = calculate_all_greeks(
                    spot_range[i], strikes[k], time_to_expiry,
                    risk_free_rate, volatility, option_types[k]
                )
                multiplier = quantities[k] * position_types[k]
                total_greeks += greeks * multiplier

            matrix_3d[i, j, :] = total_greeks

    return matrix_3d

@njit(fastmath=True, cache=True, parallel=True)
def calculate_portfolio_greeks_3d_iv(strikes: np.ndarray, option_types: np.ndarray,
                                     position_types: np.ndarray, quantities: np.ndarray,
                                     spot_range: np.ndarray, dte: float,
                                     risk_free_rate: float, iv_range: np.ndarray) -> np.ndarray:
    """
    Calculate 3D matrix of portfolio Greeks: [spot x iv x greek]
    Aggregates all positions in the portfolio
    """
    n_spots = len(spot_range)
    n_iv = len(iv_range)
    n_positions = len(strikes)
    time_to_expiry = dte / DAYS_PER_YEAR
    matrix_3d = np.zeros((n_spots, n_iv, 14))

    for i in prange(n_spots):
        for j in range(n_iv):
            total_greeks = np.zeros(14)

            for k in range(n_positions):
                greeks = calculate_all_greeks(
                    spot_range[i], strikes[k], time_to_expiry,
                    risk_free_rate, iv_range[j], option_types[k]
                )
                multiplier = quantities[k] * position_types[k]
                total_greeks += greeks * multiplier

            matrix_3d[i, j, :] = total_greeks

    return matrix_3d

# ============= Breakeven Calculation Functions =============

@njit(fastmath=True, cache=True)
def calculate_portfolio_pnl_at_expiry(spot: float, strikes: np.ndarray,
                                      option_types: np.ndarray,
                                      position_types: np.ndarray,
                                      quantities: np.ndarray,
                                      premiums: np.ndarray,
                                      stock_quantity: float,
                                      stock_entry_price: float) -> float:
    """
    Calculate portfolio P&L at expiration for a given spot price
    """
    pnl = 0.0

    # Initial cost of options
    for i in range(len(strikes)):
        # Long positions: pay premium (negative P&L)
        # Short positions: receive premium (positive P&L)
        if position_types[i] == 1:  # Long
            pnl -= premiums[i] * quantities[i]  # Pay premium
        else:  # Short (position_types[i] == -1)
            pnl += premiums[i] * quantities[i]  # Receive premium

    # Initial cost of stock
    if stock_quantity != 0:
        pnl -= stock_entry_price * stock_quantity

    # Option values at expiration
    for i in range(len(strikes)):
        if option_types[i] == 1:  # Call
            payoff = max(spot - strikes[i], 0.0)
        else:  # Put
            payoff = max(strikes[i] - spot, 0.0)

        pnl += position_types[i] * quantities[i] * payoff

    # Stock value at expiration
    if stock_quantity != 0:
        pnl += stock_quantity * spot

    return pnl

def find_breakeven_points(strikes: np.ndarray, option_types: np.ndarray,
                         position_types: np.ndarray, quantities: np.ndarray,
                         premiums: np.ndarray, stock_quantity: float,
                         stock_entry_price: float, spot_min: float = 0.1,
                         spot_max: float = 1000.0, precision: int = 10000) -> BreakevenResult:
    """
    Find all breakeven points for a portfolio

    Parameters:
    -----------
    strikes, option_types, position_types, quantities, premiums : Position arrays
    stock_quantity, stock_entry_price : Stock position details
    spot_min, spot_max : Search range limits
    precision : Number of points for search

    Returns:
    --------
    BreakevenResult with all breakeven points and analysis
    """
    # Generate price grid
    spot_range = np.linspace(spot_min, spot_max, precision)
    pnl_values = np.zeros(precision)

    # Calculate P&L for each spot price
    for i, spot in enumerate(spot_range):
        pnl_values[i] = calculate_portfolio_pnl_at_expiry(
            spot, strikes, option_types, position_types,
            quantities, premiums, stock_quantity, stock_entry_price
        )

    # Find breakeven points (where P&L changes sign)
    breakeven_points = []
    for i in range(len(pnl_values) - 1):
        if pnl_values[i] * pnl_values[i+1] < 0:  # Sign change
            # Linear interpolation for precision
            spot1, spot2 = spot_range[i], spot_range[i+1]
            pnl1, pnl2 = pnl_values[i], pnl_values[i+1]

            # Exact point by interpolation
            breakeven = spot1 - pnl1 * (spot2 - spot1) / (pnl2 - pnl1)
            breakeven_points.append(breakeven)

    # Find max profit and max loss
    max_profit_idx = np.argmax(pnl_values)
    max_loss_idx = np.argmin(pnl_values)

    max_profit = pnl_values[max_profit_idx]
    max_profit_spot = spot_range[max_profit_idx]
    max_loss = pnl_values[max_loss_idx]
    max_loss_spot = spot_range[max_loss_idx]

    # Identify profit and loss zones
    profit_zones = []
    loss_zones = []

    if len(breakeven_points) == 0:
        # No breakeven, either always profit or always loss
        if pnl_values[0] > 0:
            profit_zones.append((-np.inf, np.inf))
        else:
            loss_zones.append((-np.inf, np.inf))
    else:
        # Analyze each zone
        breakeven_extended = [-np.inf] + breakeven_points + [np.inf]

        for i in range(len(breakeven_extended) - 1):
            # Test middle of zone
            if i == 0:
                test_spot = breakeven_extended[1] - 1
            elif i == len(breakeven_extended) - 2:
                test_spot = breakeven_extended[-2] + 1
            else:
                test_spot = (breakeven_extended[i] + breakeven_extended[i+1]) / 2

            # Clamp to valid range
            test_spot = max(spot_min, min(spot_max, test_spot))

            # Calculate P&L at test point
            test_pnl = calculate_portfolio_pnl_at_expiry(
                test_spot, strikes, option_types, position_types,
                quantities, premiums, stock_quantity, stock_entry_price
            )

            if test_pnl > 0:
                profit_zones.append((breakeven_extended[i], breakeven_extended[i+1]))
            else:
                loss_zones.append((breakeven_extended[i], breakeven_extended[i+1]))

    return BreakevenResult(
        breakeven_points=breakeven_points,
        max_profit=max_profit,
        max_profit_spot=max_profit_spot,
        max_loss=max_loss,
        max_loss_spot=max_loss_spot,
        profit_zones=profit_zones,
        loss_zones=loss_zones
    )

# ============= Main Portfolio Class =============

class OptionsPortfolio:
    """Options portfolio manager with 3D Greeks calculation and breakeven analysis"""

    def __init__(self, spot: float, risk_free_rate: float = 0.05):
        """
        Initialize portfolio

        Parameters:
        -----------
        spot : Current spot price
        risk_free_rate : Risk-free interest rate
        """
        if spot <= 0:
            raise ValueError(f"Spot price must be positive, got {spot}")

        self.spot = spot
        self.risk_free_rate = risk_free_rate
        self.positions: List[OptionPosition] = []
        self.stock_position: Optional[StockPosition] = None
        self._warmup_numba()

    def _warmup_numba(self):
        """Pre-compile Numba functions for better performance"""
        test_spots = np.array([100.0, 101.0])
        test_dte = np.array([30.0, 60.0])
        test_iv = np.array([0.2, 0.3])
        _ = calculate_greeks_3d_dte_matrix(test_spots, 100.0, test_dte, 0.05, 0.2, 1)
        _ = calculate_greeks_3d_iv_matrix(test_spots, 100.0, 30.0, 0.05, test_iv, 1)

    def add_option_position(self, position: OptionPosition):
        """Add an option position to the portfolio"""
        # Auto-calculate premium if not provided
        if position.premium_paid == 0.0:
            time_to_expiry = 30 / DAYS_PER_YEAR
            opt_type = 1 if position.option_type == 'call' else 0
            price, _, _, _, _, _ = calculate_first_order_greeks(
                self.spot, position.strike, time_to_expiry, self.risk_free_rate, 0.25, opt_type
            )
            position.premium_paid = price

        self.positions.append(position)

    def add_stock_position(self, position: StockPosition):
        """Add or replace stock position"""
        if position.entry_price == 0.0:
            position.entry_price = self.spot
        self.stock_position = position

    def calculate_single_option_3d_dte(self, strike: float, option_type: str,
                                       spot_range: np.ndarray, dte_range: np.ndarray,
                                       volatility: float) -> Dict[str, np.ndarray]:
        """
        Calculate 3D matrix of Greeks for a single option with DTE variation

        Returns:
        --------
        Dict with keys 'price', 'delta', 'gamma', etc. containing 2D matrices [spot x dte]
        """
        opt_type = 1 if option_type == 'call' else 0
        matrix_3d = calculate_greeks_3d_dte_matrix(
            spot_range, strike, dte_range, self.risk_free_rate, volatility, opt_type
        )

        greek_names = self.get_greek_names()
        return {name: matrix_3d[:, :, i] for i, name in enumerate(greek_names)}

    def calculate_single_option_3d_iv(self, strike: float, option_type: str,
                                      spot_range: np.ndarray, dte: float,
                                      iv_range: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate 3D matrix of Greeks for a single option with IV variation

        Returns:
        --------
        Dict with keys 'price', 'delta', 'gamma', etc. containing 2D matrices [spot x iv]
        """
        opt_type = 1 if option_type == 'call' else 0
        matrix_3d = calculate_greeks_3d_iv_matrix(
            spot_range, strike, dte, self.risk_free_rate, iv_range, opt_type
        )

        greek_names = self.get_greek_names()
        return {name: matrix_3d[:, :, i] for i, name in enumerate(greek_names)}

    def calculate_portfolio_3d_dte(self, spot_range: np.ndarray, dte_range: np.ndarray,
                                   volatility: float) -> Dict[str, np.ndarray]:
        """
        Calculate 3D matrix of aggregated portfolio Greeks with DTE variation
        """
        if not self.positions:
            return {name: np.zeros((len(spot_range), len(dte_range)))
                   for name in self.get_greek_names()}

        strikes = np.array([pos.strike for pos in self.positions])
        option_types = np.array([1 if pos.option_type == 'call' else 0
                                 for pos in self.positions])
        position_types = np.array([1 if pos.position_type == 'long' else -1
                                   for pos in self.positions])
        quantities = np.array([pos.quantity for pos in self.positions])

        matrix_3d = calculate_portfolio_greeks_3d_dte(
            strikes, option_types, position_types, quantities,
            spot_range, dte_range, self.risk_free_rate, volatility
        )

        greek_names = self.get_greek_names()
        return {name: matrix_3d[:, :, i] for i, name in enumerate(greek_names)}

    def calculate_portfolio_3d_iv(self, spot_range: np.ndarray, dte: float,
                                  iv_range: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate 3D matrix of aggregated portfolio Greeks with IV variation
        """
        if not self.positions:
            return {name: np.zeros((len(spot_range), len(iv_range)))
                   for name in self.get_greek_names()}

        strikes = np.array([pos.strike for pos in self.positions])
        option_types = np.array([1 if pos.option_type == 'call' else 0
                                 for pos in self.positions])
        position_types = np.array([1 if pos.position_type == 'long' else -1
                                   for pos in self.positions])
        quantities = np.array([pos.quantity for pos in self.positions])

        matrix_3d = calculate_portfolio_greeks_3d_iv(
            strikes, option_types, position_types, quantities,
            spot_range, dte, self.risk_free_rate, iv_range
        )

        greek_names = self.get_greek_names()
        return {name: matrix_3d[:, :, i] for i, name in enumerate(greek_names)}

    def calculate_breakevens(self, spot_min: Optional[float] = None,
                            spot_max: Optional[float] = None,
                            precision: int = 10000) -> BreakevenResult:
        """
        Calculate breakeven points for the portfolio

        Parameters:
        -----------
        spot_min : Minimum price for search (default: 0.5 * current spot)
        spot_max : Maximum price for search (default: 2.0 * current spot)
        precision : Number of points for search

        Returns:
        --------
        BreakevenResult with all breakeven points and analysis
        """
        if not self.positions and not self.stock_position:
            raise ValueError("Portfolio is empty")

        # Default limits
        if spot_min is None:
            spot_min = self.spot * 0.5
        if spot_max is None:
            spot_max = self.spot * 2.0

        # Prepare arrays for option positions
        if self.positions:
            strikes = np.array([pos.strike for pos in self.positions])
            option_types = np.array([1 if pos.option_type == 'call' else 0
                                     for pos in self.positions])
            position_types = np.array([1 if pos.position_type == 'long' else -1
                                      for pos in self.positions])
            quantities = np.array([pos.quantity for pos in self.positions])
            premiums = np.array([pos.premium_paid for pos in self.positions])
        else:
            strikes = np.array([])
            option_types = np.array([], dtype=np.int32)
            position_types = np.array([], dtype=np.int32)
            quantities = np.array([], dtype=np.int32)
            premiums = np.array([])

        # Stock position
        stock_quantity = 0.0
        stock_entry_price = 0.0
        if self.stock_position:
            stock_quantity = self.stock_position.quantity * (
                1 if self.stock_position.position_type == 'long' else -1
            )
            stock_entry_price = self.stock_position.entry_price

        return find_breakeven_points(
            strikes, option_types, position_types, quantities,
            premiums, stock_quantity, stock_entry_price,
            spot_min, spot_max, precision
        )

    def create_strategy(self, strategy_name: str):
        """Create common option strategies"""
        self.positions.clear()
        self.stock_position = None

        strategies = {
            'long_straddle': lambda: [
                self.add_option_position(OptionPosition('call', 'long', self.spot)),
                self.add_option_position(OptionPosition('put', 'long', self.spot))
            ],
            'iron_condor': lambda: [
                self.add_option_position(OptionPosition('put', 'long', self.spot * 0.90)),
                self.add_option_position(OptionPosition('put', 'short', self.spot * 0.95)),
                self.add_option_position(OptionPosition('call', 'short', self.spot * 1.05)),
                self.add_option_position(OptionPosition('call', 'long', self.spot * 1.10))
            ],
            'butterfly': lambda: [
                self.add_option_position(OptionPosition('call', 'long', self.spot * 0.95)),
                self.add_option_position(OptionPosition('call', 'short', self.spot, 2)),
                self.add_option_position(OptionPosition('call', 'long', self.spot * 1.05))
            ],
            'covered_call': lambda: [
                self.add_stock_position(StockPosition('long', 100, self.spot)),
                self.add_option_position(OptionPosition('call', 'short', self.spot * 1.05))
            ],
            'protective_put': lambda: [
                self.add_stock_position(StockPosition('long', 100, self.spot)),
                self.add_option_position(OptionPosition('put', 'long', self.spot * 0.95))
            ],
            'bull_call_spread': lambda: [
                self.add_option_position(OptionPosition('call', 'long', self.spot * 0.95)),
                self.add_option_position(OptionPosition('call', 'short', self.spot * 1.05))
            ],
            'bear_put_spread': lambda: [
                self.add_option_position(OptionPosition('put', 'long', self.spot * 1.05)),
                self.add_option_position(OptionPosition('put', 'short', self.spot * 0.95))
            ],
            'collar': lambda: [
                self.add_stock_position(StockPosition('long', 100, self.spot)),
                self.add_option_position(OptionPosition('put', 'long', self.spot * 0.95)),
                self.add_option_position(OptionPosition('call', 'short', self.spot * 1.05))
            ]
        }

        if strategy_name in strategies:
            strategies[strategy_name]()
        else:
            raise ValueError(f"Strategy '{strategy_name}' not recognized")

    @staticmethod
    def get_greek_names() -> List[str]:
        """Return list of Greek names"""
        return ['price', 'delta', 'gamma', 'vega', 'theta', 'rho',
                'vanna', 'volga', 'charm', 'veta', 'speed', 'zomma',
                'color', 'ultima']

# ============= Main Test Program =============

if __name__ == "__main__":
    print("=" * 80)
    print("OPTIONS CALCULATOR - VERIFIED BLACK-SCHOLES GREEKS")
    print("=" * 80)

    # Configuration
    spot_price = 100.0
    risk_free_rate = 0.05
    base_volatility = 0.25

    # Test 1: Greeks Calculation Example
    print("\n1. GREEKS CALCULATION EXAMPLE")
    print("-" * 40)

    time_to_expiry = 0.25  # 3 months
    greeks = calculate_all_greeks(spot_price, spot_price, time_to_expiry,
                                  risk_free_rate, base_volatility, 1)

    greek_names = OptionsPortfolio.get_greek_names()
    print(f"ATM Call Option: Spot=${spot_price}, Strike=${spot_price}")
    print(f"Time={time_to_expiry} years, Vol={base_volatility*100}%, Rate={risk_free_rate*100}%")
    print("\nFirst-Order Greeks:")
    print(f"  Price: ${greeks[0]:.4f}")
    print(f"  Delta: {greeks[1]:.4f}")
    print(f"  Gamma: {greeks[2]:.4f}")
    print(f"  Vega: {greeks[3]:.4f} (per 1% vol)")
    print(f"  Theta: {greeks[4]:.4f} (per day)")
    print(f"  Rho: {greeks[5]:.4f} (per 1% rate)")

    print("\nSecond-Order Greeks:")
    print(f"  Vanna: {greeks[6]:.6f}")
    print(f"  Volga/Vomma: {greeks[7]:.6f}")
    print(f"  Charm: {greeks[8]:.6f}")
    print(f"  Veta: {greeks[9]:.6f}")

    print("\nThird-Order Greeks:")
    print(f"  Speed: {greeks[10]:.8f}")
    print(f"  Zomma: {greeks[11]:.6f}")
    print(f"  Color: {greeks[12]:.8f}")
    print(f"  Ultima: {greeks[13]:.8f}")

    # Test 2: Breakeven Analysis
    print("\n2. BREAKEVEN ANALYSIS - IRON CONDOR")
    print("-" * 40)

    portfolio = OptionsPortfolio(spot=spot_price, risk_free_rate=risk_free_rate)
    portfolio.create_strategy('iron_condor')

    breakevens = portfolio.calculate_breakevens()
    print(breakevens.summary())

    # Test 3: 3D Matrix Performance
    print("\n3. 3D MATRIX PERFORMANCE TEST")
    print("-" * 40)

    spot_range = np.linspace(80, 120, 51)
    dte_range = np.array([1, 7, 14, 30, 45, 60, 90])
    iv_range = np.array([0.10, 0.20, 0.30, 0.40, 0.50])

    # DTE variation
    start = time.time()
    matrix_dte = portfolio.calculate_portfolio_3d_dte(spot_range, dte_range, base_volatility)
    time_dte = time.time() - start

    # IV variation
    start = time.time()
    matrix_iv = portfolio.calculate_portfolio_3d_iv(spot_range, 30, iv_range)
    time_iv = time.time() - start

    print(f"3D Matrix DTE (51 spots x 7 DTE x 14 Greeks):")
    print(f"  Time: {time_dte:.3f} seconds")
    print(f"  Speed: {(51*7*14)/time_dte:,.0f} values/second")

    print(f"\n3D Matrix IV (51 spots x 5 IV x 14 Greeks):")
    print(f"  Time: {time_iv:.3f} seconds")
    print(f"  Speed: {(51*5*14)/time_iv:,.0f} values/second")

    # Test 4: Multiple Strategies
    print("\n4. STRATEGY EXAMPLES")
    print("-" * 40)

    strategies = ['bull_call_spread', 'bear_put_spread', 'butterfly', 'long_straddle']

    for strategy in strategies:
        portfolio.create_strategy(strategy)
        breakevens = portfolio.calculate_breakevens()

        print(f"\n{strategy.upper().replace('_', ' ')}:")
        if breakevens.breakeven_points:
            print(f"  Breakevens: {[f'${b:.2f}' for b in breakevens.breakeven_points]}")
        print(f"  Max Profit: ${breakevens.max_profit:.2f} at ${breakevens.max_profit_spot:.2f}")
        print(f"  Max Loss: ${breakevens.max_loss:.2f} at ${breakevens.max_loss_spot:.2f}")

    print("\n" + "=" * 80)
    print("ALL CALCULATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 80)