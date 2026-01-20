"""
Greeks Calculation Strategies
=============================

Strategy pattern for calculating portfolio Greeks.
Supports analytical formulas (when available) and finite differences fallback.

Includes:
- First-order Greeks: delta, gamma, theta, vega, rho
- Second-order Greeks: vanna, volga (vomma), charm, veta
- Third-order Greeks: speed, zomma, color, ultima

Author: Derivatives Pricing Project
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Protocol, runtime_checkable

from .positions import OptionPosition, StockPosition

# Import Numba-optimized math functions from utils
from backend.utils.math import (
    bs_second_order_greeks,
    bs_third_order_greeks,
    DAYS_PER_YEAR,
)


# =============================================================================
# Greeks Data Container
# =============================================================================

@dataclass
class GreeksResult:
    """
    Container for Greeks calculation results.

    All values are for the total portfolio position.

    First-order Greeks:
        delta: dV/dS
        gamma: d²V/dS²
        theta: dV/dt (per day)
        vega: dV/dσ (per 1% vol change)
        rho: dV/dr (per 1% rate change)

    Second-order Greeks:
        vanna: d²V/dSdσ (per 1% vol change)
        volga: d²V/dσ² (vomma, per 1% vol change squared)
        charm: d²V/dSdt (delta decay, per day)
        veta: d²V/dσdt (vega decay, per day per 1% vol)

    Third-order Greeks:
        speed: d³V/dS³
        zomma: d³V/dS²dσ (per 1% vol change)
        color: d³V/dS²dt (gamma decay, per day)
        ultima: d³V/dσ³ (per 1% vol change cubed)
    """
    # First-order
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    # Second-order
    vanna: float = 0.0
    volga: float = 0.0  # Also known as vomma
    charm: float = 0.0
    veta: float = 0.0

    # Third-order
    speed: float = 0.0
    zomma: float = 0.0
    color: float = 0.0
    ultima: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            # First-order
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
            # Second-order
            "vanna": self.vanna,
            "volga": self.volga,
            "charm": self.charm,
            "veta": self.veta,
            # Third-order
            "speed": self.speed,
            "zomma": self.zomma,
            "color": self.color,
            "ultima": self.ultima,
        }

    def first_order(self) -> Dict[str, float]:
        """Return only first-order Greeks."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
        }

    def second_order(self) -> Dict[str, float]:
        """Return only second-order Greeks."""
        return {
            "vanna": self.vanna,
            "volga": self.volga,
            "charm": self.charm,
            "veta": self.veta,
        }

    def third_order(self) -> Dict[str, float]:
        """Return only third-order Greeks."""
        return {
            "speed": self.speed,
            "zomma": self.zomma,
            "color": self.color,
            "ultima": self.ultima,
        }

    def __add__(self, other: 'GreeksResult') -> 'GreeksResult':
        """Add two Greeks results."""
        return GreeksResult(
            # First-order
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho,
            # Second-order
            vanna=self.vanna + other.vanna,
            volga=self.volga + other.volga,
            charm=self.charm + other.charm,
            veta=self.veta + other.veta,
            # Third-order
            speed=self.speed + other.speed,
            zomma=self.zomma + other.zomma,
            color=self.color + other.color,
            ultima=self.ultima + other.ultima,
        )

    def __mul__(self, scalar: float) -> 'GreeksResult':
        """Multiply Greeks by a scalar."""
        return GreeksResult(
            # First-order
            delta=self.delta * scalar,
            gamma=self.gamma * scalar,
            theta=self.theta * scalar,
            vega=self.vega * scalar,
            rho=self.rho * scalar,
            # Second-order
            vanna=self.vanna * scalar,
            volga=self.volga * scalar,
            charm=self.charm * scalar,
            veta=self.veta * scalar,
            # Third-order
            speed=self.speed * scalar,
            zomma=self.zomma * scalar,
            color=self.color * scalar,
            ultima=self.ultima * scalar,
        )

    def __rmul__(self, scalar: float) -> 'GreeksResult':
        """Right multiply by scalar."""
        return self.__mul__(scalar)


# =============================================================================
# Wrapper functions for backward compatibility
# =============================================================================

def calculate_bs_second_order_greeks(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
) -> tuple:
    """
    Calculate second-order Greeks for Black-Scholes.

    This is a wrapper around the Numba-optimized function in utils.math.

    Returns
    -------
    tuple
        (vanna, volga, charm, veta)
    """
    return bs_second_order_greeks(spot, strike, t, r, sigma)


def calculate_bs_third_order_greeks(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
) -> tuple:
    """
    Calculate third-order Greeks for Black-Scholes.

    This is a wrapper around the Numba-optimized function in utils.math.

    Returns
    -------
    tuple
        (speed, zomma, color, ultima)
    """
    return bs_third_order_greeks(spot, strike, t, r, sigma)


# =============================================================================
# Pricer Protocol (for type checking)
# =============================================================================

@runtime_checkable
class GreeksPricer(Protocol):
    """Protocol for pricers that support analytical Greeks."""

    def price(self, s0: float, k: float, t: float, r: float,
              option_type: str = "call", **kwargs) -> object:
        """Price an option, returning object with .price attribute."""
        ...


@runtime_checkable
class AnalyticalGreeksPricer(Protocol):
    """Protocol for pricers with analytical Greeks support via PricingResult."""

    def price(self, s0: float, k: float, t: float, r: float,
              option_type: str = "call", compute_greeks: bool = True,
              **kwargs) -> object:
        """
        Price an option, returning PricingResult with Greeks.

        The result should have attributes: price, delta, gamma, vega, theta, rho
        """
        ...


# =============================================================================
# Greeks Strategy Interface
# =============================================================================

class GreeksStrategy(ABC):
    """Abstract base class for Greeks calculation strategies."""

    @abstractmethod
    def calculate_single(
        self,
        position: OptionPosition,
        spot: float,
        rate: float,
        time_to_expiry: float,
    ) -> GreeksResult:
        """Calculate Greeks for a single option position."""
        pass

    def calculate_portfolio(
        self,
        options: List[OptionPosition],
        stock: Optional[StockPosition],
        spot: float,
        rate: float,
        time_to_expiry: float,
    ) -> GreeksResult:
        """Calculate aggregate Greeks for the portfolio."""
        total = GreeksResult()

        # Sum option Greeks
        for pos in options:
            greeks = self.calculate_single(pos, spot, rate, time_to_expiry)
            total = total + greeks

        # Add stock delta
        if stock is not None:
            total.delta += stock.delta

        return total


# =============================================================================
# Analytical Greeks Strategy
# =============================================================================

class AnalyticalGreeksStrategy(GreeksStrategy):
    """
    Calculate Greeks using pricer's analytical formulas.

    For Black-Scholes pricers, also computes second and third-order Greeks.
    """

    def __init__(self, pricer: AnalyticalGreeksPricer):
        """
        Initialize with a pricer that supports analytical Greeks.

        Parameters
        ----------
        pricer : AnalyticalGreeksPricer
            Pricer with analytical Greeks support
        """
        self._pricer = pricer

    def _is_black_scholes(self) -> bool:
        """Check if pricer is Black-Scholes (has sigma attribute)."""
        return hasattr(self._pricer, 'sigma') or hasattr(self._pricer, '_sigma')

    def _get_sigma(self) -> Optional[float]:
        """Get volatility from pricer if available."""
        if hasattr(self._pricer, 'sigma'):
            return self._pricer.sigma
        if hasattr(self._pricer, '_sigma'):
            return self._pricer._sigma
        return None

    def calculate_single(
        self,
        position: OptionPosition,
        spot: float,
        rate: float,
        time_to_expiry: float,
    ) -> GreeksResult:
        """Calculate Greeks for a single option using analytical formulas."""
        if time_to_expiry <= 0:
            # At expiry, only delta matters for ITM options
            intrinsic = position.intrinsic_value(spot)
            if intrinsic > 0:
                delta = position.sign * position.quantity * (1.0 if position.is_call else -1.0)
            else:
                delta = 0.0
            return GreeksResult(delta=delta)

        # Get option type string for pricer
        opt_type = position.option_type.value  # 'call' or 'put'

        # Call pricer's price method with compute_greeks=True
        result = self._pricer.price(
            s0=spot,
            k=position.strike,
            t=time_to_expiry,
            r=rate,
            option_type=opt_type,
            compute_greeks=True,
        )

        # Scale by position sign and quantity
        multiplier = position.sign * position.quantity

        # Extract first-order Greeks from PricingResult
        greeks = GreeksResult(
            delta=multiplier * (result.delta or 0.0),
            gamma=multiplier * (result.gamma or 0.0),
            theta=multiplier * (result.theta or 0.0),
            vega=multiplier * (result.vega or 0.0),
            rho=multiplier * (result.rho or 0.0),
        )

        # For Black-Scholes, compute higher-order Greeks analytically
        if self._is_black_scholes():
            sigma = self._get_sigma()
            if sigma is not None:
                # Second-order Greeks
                vanna, volga, charm, veta = calculate_bs_second_order_greeks(
                    spot, position.strike, time_to_expiry, rate, sigma
                )
                greeks.vanna = multiplier * vanna
                greeks.volga = multiplier * volga
                greeks.charm = multiplier * charm
                greeks.veta = multiplier * veta

                # Third-order Greeks
                speed, zomma, color, ultima = calculate_bs_third_order_greeks(
                    spot, position.strike, time_to_expiry, rate, sigma
                )
                greeks.speed = multiplier * speed
                greeks.zomma = multiplier * zomma
                greeks.color = multiplier * color
                greeks.ultima = multiplier * ultima

        return greeks


# =============================================================================
# Finite Differences Strategy
# =============================================================================

class FiniteDiffGreeksStrategy(GreeksStrategy):
    """
    Calculate Greeks using finite differences (bump-and-revalue).

    This strategy works with any pricer that has a price() method.
    Less accurate than analytical but universally applicable.
    """

    def __init__(
        self,
        pricer: GreeksPricer,
        spot_bump: float = 0.01,  # 1% of spot
        vol_bump: float = 0.01,  # 1% absolute
        time_bump: float = 1/365,  # 1 day
        rate_bump: float = 0.0001,  # 1 bp
    ):
        """
        Initialize with a pricer and bump sizes.

        Parameters
        ----------
        pricer : GreeksPricer
            Any pricer with a price() method
        spot_bump : float
            Relative bump for spot (as fraction)
        vol_bump : float
            Absolute bump for volatility
        time_bump : float
            Time bump in years
        rate_bump : float
            Rate bump (absolute)
        """
        self._pricer = pricer
        self._spot_bump = spot_bump
        self._vol_bump = vol_bump
        self._time_bump = time_bump
        self._rate_bump = rate_bump

    def _get_price(self, spot: float, strike: float, t: float, r: float,
                   option_type: str) -> float:
        """Get price from pricer, handling different return types."""
        result = self._pricer.price(s0=spot, k=strike, t=t, r=r, option_type=option_type)
        # Handle both PricingResult objects and raw floats
        if hasattr(result, 'price'):
            return result.price
        return float(result)

    def calculate_single(
        self,
        position: OptionPosition,
        spot: float,
        rate: float,
        time_to_expiry: float,
    ) -> GreeksResult:
        """Calculate Greeks using finite differences."""
        if time_to_expiry <= 0:
            intrinsic = position.intrinsic_value(spot)
            if intrinsic > 0:
                delta = position.sign * position.quantity * (1.0 if position.is_call else -1.0)
            else:
                delta = 0.0
            return GreeksResult(delta=delta)

        opt_type = position.option_type.value
        k = position.strike
        t = time_to_expiry
        r = rate

        # Base price
        p0 = self._get_price(spot, k, t, r, opt_type)

        # Delta: dP/dS (central difference)
        ds = spot * self._spot_bump
        p_up = self._get_price(spot + ds, k, t, r, opt_type)
        p_down = self._get_price(spot - ds, k, t, r, opt_type)
        delta = (p_up - p_down) / (2 * ds)

        # Gamma: d²P/dS² (central second difference)
        gamma = (p_up - 2 * p0 + p_down) / (ds ** 2)

        # Theta: -dP/dt (forward difference, negative by convention)
        dt = self._time_bump
        if t > dt:
            p_later = self._get_price(spot, k, t - dt, r, opt_type)
            theta = (p_later - p0) / dt  # Theta per day (negative for long)
        else:
            theta = 0.0

        # Vega: dP/dσ - requires pricer with volatility
        # This is tricky for stochastic vol models, skip if not applicable
        vega = 0.0

        # Rho: dP/dr
        dr = self._rate_bump
        p_r_up = self._get_price(spot, k, t, r + dr, opt_type)
        p_r_down = self._get_price(spot, k, t, r - dr, opt_type)
        rho = (p_r_up - p_r_down) / (2 * dr)

        # Scale by position
        multiplier = position.sign * position.quantity

        return GreeksResult(
            delta=multiplier * delta,
            gamma=multiplier * gamma,
            theta=multiplier * theta,
            vega=multiplier * vega,
            rho=multiplier * rho,
            # Higher-order Greeks not computed with finite differences
        )


# =============================================================================
# Greeks Calculator (Context)
# =============================================================================

class GreeksCalculator:
    """
    Context class for calculating portfolio Greeks.

    Uses the Strategy pattern to allow different calculation methods.
    """

    def __init__(self, strategy: GreeksStrategy):
        """
        Initialize with a Greeks calculation strategy.

        Parameters
        ----------
        strategy : GreeksStrategy
            Strategy to use (Analytical or FiniteDiff)
        """
        self._strategy = strategy

    @property
    def strategy(self) -> GreeksStrategy:
        """Current calculation strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: GreeksStrategy):
        """Change calculation strategy."""
        self._strategy = strategy

    def calculate(
        self,
        options: List[OptionPosition],
        stock: Optional[StockPosition],
        spot: float,
        rate: float,
        time_to_expiry: float,
    ) -> GreeksResult:
        """
        Calculate portfolio Greeks.

        Parameters
        ----------
        options : List[OptionPosition]
            List of option positions
        stock : StockPosition, optional
            Stock position (if any)
        spot : float
            Current spot price
        rate : float
            Risk-free rate
        time_to_expiry : float
            Time to expiry in years

        Returns
        -------
        GreeksResult
            Portfolio Greeks (first, second, and third order)
        """
        return self._strategy.calculate_portfolio(
            options=options,
            stock=stock,
            spot=spot,
            rate=rate,
            time_to_expiry=time_to_expiry,
        )

    def calculate_single(
        self,
        position: OptionPosition,
        spot: float,
        rate: float,
        time_to_expiry: float,
    ) -> GreeksResult:
        """Calculate Greeks for a single position."""
        return self._strategy.calculate_single(
            position=position,
            spot=spot,
            rate=rate,
            time_to_expiry=time_to_expiry,
        )


# =============================================================================
# Standalone Functions for Frontend Compatibility
# =============================================================================

import numpy as np
from backend.option_pricing import BlackScholesPricer


def calculate_all_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: int  # 1 for call, 0 for put
) -> np.ndarray:
    """
    Calculate all Greeks for a single option.

    Returns a numpy array with 14 values:
    [price, delta, gamma, theta, vega, rho, vanna, volga, charm, veta, speed, zomma, color, ultima]

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Strike price
    time_to_expiry : float
        Time to expiry in years
    risk_free_rate : float
        Risk-free rate (decimal)
    volatility : float
        Implied volatility (decimal)
    option_type : int
        1 for call, 0 for put

    Returns
    -------
    np.ndarray
        Array of 14 Greek values
    """
    # Handle edge cases
    if time_to_expiry <= 0 or volatility <= 0:
        opt_type_str = 'call' if option_type == 1 else 'put'
        if opt_type_str == 'call':
            intrinsic = max(0, spot - strike)
            delta = 1.0 if spot > strike else 0.0
        else:
            intrinsic = max(0, strike - spot)
            delta = -1.0 if spot < strike else 0.0
        return np.array([intrinsic, delta, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Use Black-Scholes pricer for first-order Greeks
    pricer = BlackScholesPricer(sigma=volatility)
    opt_type_str = 'call' if option_type == 1 else 'put'

    result = pricer.price(
        s0=spot,
        k=strike,
        t=time_to_expiry,
        r=risk_free_rate,
        option_type=opt_type_str,
        compute_greeks=True,
    )

    # Calculate higher-order Greeks using Numba-optimized functions
    vanna, volga, charm, veta = bs_second_order_greeks(
        spot, strike, time_to_expiry, risk_free_rate, volatility
    )
    speed, zomma, color, ultima = bs_third_order_greeks(
        spot, strike, time_to_expiry, risk_free_rate, volatility
    )

    return np.array([
        result.price,
        result.delta,
        result.gamma,
        result.theta,
        result.vega,
        result.rho,
        vanna,
        volga,
        charm,
        veta,
        speed,
        zomma,
        color,
        ultima,
    ])


def calculate_portfolio_greeks_3d_dte(
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    spot_range: np.ndarray,
    dte_range: np.ndarray,
    risk_free_rate: float,
    volatility: float
) -> np.ndarray:
    """
    Calculate 3D matrix of portfolio Greeks: [spot x dte x greek]

    Parameters
    ----------
    strikes : np.ndarray
        Strike prices
    option_types : np.ndarray
        1 for call, 0 for put
    position_types : np.ndarray
        1 for long, -1 for short
    quantities : np.ndarray
        Position quantities
    spot_range : np.ndarray
        Array of spot prices
    dte_range : np.ndarray
        Array of days to expiration
    risk_free_rate : float
        Risk-free interest rate
    volatility : float
        Implied volatility

    Returns
    -------
    np.ndarray
        3D array of shape (n_spots, n_dte, 14) containing all Greeks
    """
    n_spots = len(spot_range)
    n_dte = len(dte_range)
    n_positions = len(strikes)
    matrix_3d = np.zeros((n_spots, n_dte, 14))

    for i in range(n_spots):
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


def calculate_portfolio_greeks_3d_iv(
    strikes: np.ndarray,
    option_types: np.ndarray,
    position_types: np.ndarray,
    quantities: np.ndarray,
    spot_range: np.ndarray,
    dte: float,
    risk_free_rate: float,
    iv_range: np.ndarray
) -> np.ndarray:
    """
    Calculate 3D matrix of portfolio Greeks: [spot x iv x greek]

    Parameters
    ----------
    strikes : np.ndarray
        Strike prices
    option_types : np.ndarray
        1 for call, 0 for put
    position_types : np.ndarray
        1 for long, -1 for short
    quantities : np.ndarray
        Position quantities
    spot_range : np.ndarray
        Array of spot prices
    dte : float
        Days to expiration
    risk_free_rate : float
        Risk-free interest rate
    iv_range : np.ndarray
        Array of implied volatilities

    Returns
    -------
    np.ndarray
        3D array of shape (n_spots, n_iv, 14) containing all Greeks
    """
    n_spots = len(spot_range)
    n_iv = len(iv_range)
    n_positions = len(strikes)
    time_to_expiry = dte / DAYS_PER_YEAR
    matrix_3d = np.zeros((n_spots, n_iv, 14))

    for i in range(n_spots):
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


def calculate_greeks_3d_strike(
    spot_range: np.ndarray,
    strike_range: np.ndarray,
    dte: float,
    risk_free_rate: float,
    volatility: float,
    option_type: int,
    position_type: int,
    quantity: int
) -> np.ndarray:
    """
    Calculate 3D matrix of Greeks for single-leg option varying strike.

    Parameters
    ----------
    spot_range : np.ndarray
        Array of spot prices
    strike_range : np.ndarray
        Array of strike prices
    dte : float
        Days to expiration
    risk_free_rate : float
        Risk-free interest rate
    volatility : float
        Implied volatility
    option_type : int
        1 for call, 0 for put
    position_type : int
        1 for long, -1 for short
    quantity : int
        Number of contracts (already multiplied by 100)

    Returns
    -------
    np.ndarray
        3D array of shape (n_spots, n_strikes, 14) containing all Greeks
    """
    n_spots = len(spot_range)
    n_strikes = len(strike_range)
    time_to_expiry = dte / DAYS_PER_YEAR
    matrix_3d = np.zeros((n_spots, n_strikes, 14))
    multiplier = quantity * position_type

    for i in range(n_spots):
        for j in range(n_strikes):
            greeks = calculate_all_greeks(
                spot_range[i], strike_range[j], time_to_expiry,
                risk_free_rate, volatility, option_type
            )
            matrix_3d[i, j, :] = greeks * multiplier

    return matrix_3d
