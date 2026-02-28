"""
Input Validation
================

Validation utilities for option pricing inputs.

Provides:
- Parameter validation with clear error messages
- Range checks for market parameters
- Feller condition checks for Heston model
- Arbitrage constraint validation

Author: Thomas
Created: 2025
"""


import numpy as np

# =============================================================================
# Validation Exceptions
# =============================================================================

class ValidationError(ValueError):
    """Exception raised for validation failures."""
    pass


class ParameterOutOfRangeError(ValidationError):
    """Exception raised when a parameter is out of valid range."""
    pass


class ArbitrageViolationError(ValidationError):
    """Exception raised when arbitrage constraints are violated."""
    pass


class FellerConditionError(ValidationError):
    """Exception raised when Feller condition is violated."""
    pass


# =============================================================================
# Basic Parameter Validation
# =============================================================================

def validate_positive(
    value: float,
    name: str,
    strict: bool = True
) -> float:
    """
    Validate that a value is positive.

    Parameters
    ----------
    value : float
        Value to validate
    name : str
        Parameter name (for error message)
    strict : bool
        If True, value must be > 0. If False, value must be >= 0.

    Returns
    -------
    float
        Validated value

    Raises
    ------
    ParameterOutOfRangeError
        If validation fails
    """
    if strict and value <= 0:
        raise ParameterOutOfRangeError(
            f"{name} must be strictly positive, got {value}"
        )
    if not strict and value < 0:
        raise ParameterOutOfRangeError(
            f"{name} must be non-negative, got {value}"
        )
    return value


def validate_in_range(
    value: float,
    name: str,
    min_val: float | None = None,
    max_val: float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True
) -> float:
    """
    Validate that a value is within a specified range.

    Parameters
    ----------
    value : float
        Value to validate
    name : str
        Parameter name
    min_val : float, optional
        Minimum allowed value
    max_val : float, optional
        Maximum allowed value
    min_inclusive : bool
        Include minimum in valid range
    max_inclusive : bool
        Include maximum in valid range

    Returns
    -------
    float
        Validated value

    Raises
    ------
    ParameterOutOfRangeError
        If value is outside range
    """
    if min_val is not None:
        if min_inclusive and value < min_val:
            raise ParameterOutOfRangeError(
                f"{name} must be >= {min_val}, got {value}"
            )
        if not min_inclusive and value <= min_val:
            raise ParameterOutOfRangeError(
                f"{name} must be > {min_val}, got {value}"
            )

    if max_val is not None:
        if max_inclusive and value > max_val:
            raise ParameterOutOfRangeError(
                f"{name} must be <= {max_val}, got {value}"
            )
        if not max_inclusive and value >= max_val:
            raise ParameterOutOfRangeError(
                f"{name} must be < {max_val}, got {value}"
            )

    return value


def validate_not_nan(value: float, name: str) -> float:
    """Validate that a value is not NaN."""
    if np.isnan(value):
        raise ValidationError(f"{name} cannot be NaN")
    return value


def validate_finite(value: float, name: str) -> float:
    """Validate that a value is finite (not NaN or Inf)."""
    if not np.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value}")
    return value


# =============================================================================
# Market Parameter Validation
# =============================================================================

def validate_spot(spot: float) -> float:
    """Validate spot price."""
    validate_finite(spot, "Spot price")
    return validate_positive(spot, "Spot price", strict=True)


def validate_strike(strike: float) -> float:
    """Validate strike price."""
    validate_finite(strike, "Strike price")
    return validate_positive(strike, "Strike price", strict=True)


def validate_maturity(maturity: float) -> float:
    """Validate time to maturity."""
    validate_finite(maturity, "Maturity")
    return validate_positive(maturity, "Maturity", strict=True)


def validate_rate(rate: float, name: str = "Interest rate") -> float:
    """
    Validate interest rate.

    Allows negative rates (as seen in some markets), but warns.
    """
    validate_finite(rate, name)
    validate_in_range(rate, name, min_val=-0.10, max_val=0.50)
    return rate


def validate_volatility(sigma: float) -> float:
    """
    Validate volatility.

    Parameters
    ----------
    sigma : float
        Volatility (annualized, as decimal, e.g., 0.20 for 20%)

    Returns
    -------
    float
        Validated volatility
    """
    validate_finite(sigma, "Volatility")
    validate_positive(sigma, "Volatility", strict=True)
    validate_in_range(sigma, "Volatility", max_val=5.0)  # 500% vol cap
    return sigma


def validate_dividend_yield(q: float) -> float:
    """Validate dividend yield."""
    validate_finite(q, "Dividend yield")
    validate_in_range(q, "Dividend yield", min_val=0.0, max_val=0.50)
    return q


def validate_correlation(rho: float) -> float:
    """Validate correlation coefficient."""
    validate_finite(rho, "Correlation")
    return validate_in_range(
        rho, "Correlation",
        min_val=-1.0, max_val=1.0,
        min_inclusive=True, max_inclusive=True
    )


# =============================================================================
# Option Validation
# =============================================================================

def validate_vanilla_option(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    dividend_yield: float = 0.0
) -> None:
    """
    Validate all inputs for vanilla option pricing.

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Strike price
    maturity : float
        Time to maturity (years)
    rate : float
        Risk-free interest rate
    sigma : float
        Volatility
    dividend_yield : float
        Continuous dividend yield

    Raises
    ------
    ValidationError
        If any input is invalid
    """
    validate_spot(spot)
    validate_strike(strike)
    validate_maturity(maturity)
    validate_rate(rate)
    validate_volatility(sigma)
    validate_dividend_yield(dividend_yield)


# =============================================================================
# Model-Specific Validation
# =============================================================================

def validate_heston_parameters(
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    check_feller: bool = True
) -> None:
    """
    Validate Heston model parameters.

    Parameters
    ----------
    v0 : float
        Initial variance
    kappa : float
        Mean reversion speed
    theta : float
        Long-run variance
    xi : float
        Volatility of volatility
    rho : float
        Correlation between spot and variance
    check_feller : bool, default True
        Check Feller condition (2*kappa*theta > xi^2).
        **Strongly recommended to keep enabled.** The Feller condition
        ensures variance stays positive in continuous-time limit.
        Disabling may cause numerical instabilities in Monte Carlo.

    Raises
    ------
    ValidationError
        If parameters are invalid
    FellerConditionError
        If Feller condition is violated and check_feller=True

    Warnings
    --------
    UserWarning
        If check_feller=False, warns about potential numerical issues
    """
    import warnings

    validate_positive(v0, "Initial variance (v0)", strict=True)
    validate_positive(kappa, "Mean reversion (kappa)", strict=True)
    validate_positive(theta, "Long-run variance (theta)", strict=True)
    validate_positive(xi, "Vol of vol (xi)", strict=True)
    validate_correlation(rho)

    if not check_feller:
        # Warn when Feller check is disabled
        feller_lhs = 2 * kappa * theta
        feller_rhs = xi ** 2
        if feller_lhs <= feller_rhs:
            warnings.warn(
                f"Feller condition is violated (2κθ={feller_lhs:.4f} <= ξ²={feller_rhs:.4f}) "
                "and check_feller=False. This may cause negative variance in simulations. "
                "Consider using truncation/reflection schemes or adjusting parameters.",
                UserWarning,
                stacklevel=2
            )

    if check_feller:
        feller_lhs = 2 * kappa * theta
        feller_rhs = xi ** 2

        if feller_lhs <= feller_rhs:
            raise FellerConditionError(
                f"Feller condition violated: 2*kappa*theta = {feller_lhs:.6f} "
                f"must be > xi^2 = {feller_rhs:.6f}. "
                "This may cause variance to become negative in simulations."
            )


def validate_merton_jump_parameters(
    lambda_j: float,
    mu_j: float,
    sigma_j: float
) -> None:
    """
    Validate Merton jump-diffusion parameters.

    Parameters
    ----------
    lambda_j : float
        Jump intensity (jumps per year)
    mu_j : float
        Mean jump size (log)
    sigma_j : float
        Jump size volatility

    Raises
    ------
    ValidationError
        If parameters are invalid
    """
    validate_positive(lambda_j, "Jump intensity", strict=False)
    validate_finite(mu_j, "Mean jump size")
    validate_positive(sigma_j, "Jump volatility", strict=False)


# =============================================================================
# Arbitrage Validation
# =============================================================================

def check_put_call_parity(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float = 0.0,
    tolerance: float = 0.01
) -> bool:
    """
    Check put-call parity relation.

    C - P = S*exp(-qT) - K*exp(-rT)

    Parameters
    ----------
    call_price : float
        Call option price
    put_price : float
        Put option price
    spot : float
        Current spot
    strike : float
        Strike price
    maturity : float
        Time to maturity
    rate : float
        Risk-free rate
    dividend_yield : float
        Dividend yield
    tolerance : float
        Acceptable relative error

    Returns
    -------
    bool
        True if parity holds within tolerance
    """
    lhs = call_price - put_price
    rhs = spot * np.exp(-dividend_yield * maturity) - strike * np.exp(-rate * maturity)

    error = abs(lhs - rhs) / max(abs(rhs), 1e-10)
    return error <= tolerance


def validate_no_arbitrage(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    is_call: bool,
    dividend_yield: float = 0.0
) -> float:
    """
    Validate option price satisfies no-arbitrage bounds.

    For calls: max(0, S*exp(-qT) - K*exp(-rT)) <= C <= S*exp(-qT)
    For puts:  max(0, K*exp(-rT) - S*exp(-qT)) <= P <= K*exp(-rT)

    Parameters
    ----------
    price : float
        Option price to validate
    spot : float
        Current spot
    strike : float
        Strike price
    maturity : float
        Time to maturity
    rate : float
        Risk-free rate
    is_call : bool
        True for call, False for put
    dividend_yield : float
        Dividend yield

    Returns
    -------
    float
        Validated price

    Raises
    ------
    ArbitrageViolationError
        If price violates no-arbitrage bounds
    """
    df = np.exp(-rate * maturity)
    dq = np.exp(-dividend_yield * maturity)

    pv_spot = spot * dq
    pv_strike = strike * df

    if is_call:
        lower = max(0.0, pv_spot - pv_strike)
        upper = pv_spot
        option_type = "Call"
    else:
        lower = max(0.0, pv_strike - pv_spot)
        upper = pv_strike
        option_type = "Put"

    if price < lower - 1e-8:
        raise ArbitrageViolationError(
            f"{option_type} price {price:.4f} is below lower bound {lower:.4f}"
        )

    if price > upper + 1e-8:
        raise ArbitrageViolationError(
            f"{option_type} price {price:.4f} is above upper bound {upper:.4f}"
        )

    return price


# =============================================================================
# Array Validation
# =============================================================================

def validate_array_positive(arr: np.ndarray, name: str) -> np.ndarray:
    """Validate all elements of array are positive."""
    if np.any(arr <= 0):
        raise ParameterOutOfRangeError(
            f"All elements of {name} must be positive"
        )
    return arr


def validate_array_finite(arr: np.ndarray, name: str) -> np.ndarray:
    """Validate all elements of array are finite."""
    if not np.all(np.isfinite(arr)):
        raise ValidationError(
            f"All elements of {name} must be finite"
        )
    return arr


def validate_monotonic_increasing(
    arr: np.ndarray,
    name: str,
    strict: bool = False
) -> np.ndarray:
    """Validate array is monotonically increasing."""
    diff = np.diff(arr)
    if strict and np.any(diff <= 0):
        raise ValidationError(f"{name} must be strictly increasing")
    if not strict and np.any(diff < 0):
        raise ValidationError(f"{name} must be non-decreasing")
    return arr


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Validation Module Smoke Test")
    print("=" * 50)

    # Test basic validation
    print("\n--- Basic Validation ---")
    try:
        validate_positive(-1.0, "Test value")
        print("  ERROR: Should have raised")
    except ParameterOutOfRangeError as e:
        print(f"  validate_positive(-1): Caught {type(e).__name__}")

    # Test market parameters
    print("\n--- Market Parameter Validation ---")
    try:
        validate_vanilla_option(
            spot=100, strike=100, maturity=1.0,
            rate=0.05, sigma=0.20
        )
        print("  Valid vanilla option: OK")
    except ValidationError as e:
        print(f"  ERROR: {e}")

    try:
        validate_vanilla_option(
            spot=-100, strike=100, maturity=1.0,
            rate=0.05, sigma=0.20
        )
        print("  ERROR: Should have raised for negative spot")
    except ValidationError as e:
        print(f"  Negative spot: Caught {type(e).__name__}")

    # Test Heston validation
    print("\n--- Heston Validation ---")
    try:
        validate_heston_parameters(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7
        )
        print("  Valid Heston params (Feller OK): OK")
    except ValidationError as e:
        print(f"  ERROR: {e}")

    try:
        validate_heston_parameters(
            v0=0.04, kappa=1.0, theta=0.04, xi=1.0, rho=-0.7
        )
        print("  ERROR: Should have raised FellerConditionError")
    except FellerConditionError:
        print("  Feller violation: Caught FellerConditionError")

    # Test arbitrage bounds
    print("\n--- Arbitrage Validation ---")
    try:
        validate_no_arbitrage(
            price=10, spot=100, strike=90,
            maturity=1.0, rate=0.05, is_call=True
        )
        print("  Valid call price: OK")
    except ArbitrageViolationError as e:
        print(f"  ERROR: {e}")

    try:
        validate_no_arbitrage(
            price=200, spot=100, strike=90,
            maturity=1.0, rate=0.05, is_call=True
        )
        print("  ERROR: Should have raised for price > spot")
    except ArbitrageViolationError:
        print("  Price too high: Caught ArbitrageViolationError")

    print("\n" + "=" * 50)
    print("Validation module smoke test passed")
    print("=" * 50)
