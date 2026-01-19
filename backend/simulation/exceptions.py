"""
Custom Exceptions for Monte Carlo Simulation
=============================================

This module defines custom exception classes for handling
simulation-specific errors with meaningful messages.
"""

from typing import Any, Optional


class SimulationError(Exception):
    """
    Base exception for all simulation-related errors.

    All custom exceptions in this module inherit from this class,
    making it easy to catch all simulation errors with a single handler.
    """
    pass


class ParameterValidationError(SimulationError):
    """
    Exception raised when parameter validation fails.

    Attributes
    ----------
    param : str
        Name of the invalid parameter
    value : Any
        The invalid value that was provided
    constraint : str
        Description of the constraint that was violated
    """

    def __init__(self, param: str, value: Any, constraint: str):
        self.param = param
        self.value = value
        self.constraint = constraint
        message = f"Parameter '{param}' = {value} violates constraint: {constraint}"
        super().__init__(message)


class NumericalInstabilityError(SimulationError):
    """
    Exception raised when numerical instability is detected.

    This can occur when:
    - Variance becomes negative
    - Values become NaN or Inf
    - Numerical overflow occurs

    Attributes
    ----------
    operation : str
        The operation that caused the instability
    details : Optional[str]
        Additional details about the error
    """

    def __init__(self, operation: str, details: Optional[str] = None):
        self.operation = operation
        self.details = details
        message = f"Numerical instability in {operation}"
        if details:
            message += f": {details}"
        super().__init__(message)


class ModelNotFoundError(SimulationError):
    """
    Exception raised when an unknown model is requested.

    Attributes
    ----------
    model_name : str
        The name of the model that was not found
    available_models : list
        List of available model names
    """

    def __init__(self, model_name: str, available_models: list):
        self.model_name = model_name
        self.available_models = available_models
        message = (
            f"Unknown model '{model_name}'. "
            f"Available models: {', '.join(available_models)}"
        )
        super().__init__(message)


class StationarityViolationError(SimulationError):
    """
    Exception raised when GARCH stationarity conditions are violated.

    For GARCH(1,1), we need alpha + beta < 1 for stationarity.

    Attributes
    ----------
    model : str
        The GARCH variant model name
    alpha : float
        Alpha parameter value
    beta : float
        Beta parameter value
    persistence : float
        Computed persistence (alpha + beta)
    """

    def __init__(self, model: str, alpha: float, beta: float,
                 persistence: Optional[float] = None):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.persistence = persistence or (alpha + beta)
        message = (
            f"{model} stationarity violation: alpha + beta = {self.persistence:.4f} >= 1. "
            f"Need alpha + beta < 1 for covariance stationarity."
        )
        super().__init__(message)


class CorrelationMatrixError(SimulationError):
    """
    Exception raised when a correlation matrix is invalid.

    Attributes
    ----------
    reason : str
        Reason why the matrix is invalid
    """

    def __init__(self, reason: str):
        self.reason = reason
        message = f"Invalid correlation matrix: {reason}"
        super().__init__(message)


class InsufficientDataError(SimulationError):
    """
    Exception raised when there is insufficient data for an operation.

    Attributes
    ----------
    required : int
        Minimum number of data points required
    actual : int
        Actual number of data points provided
    """

    def __init__(self, required: int, actual: int, operation: str = "operation"):
        self.required = required
        self.actual = actual
        self.operation = operation
        message = (
            f"Insufficient data for {operation}: "
            f"requires {required}, got {actual}"
        )
        super().__init__(message)


class ConvergenceError(SimulationError):
    """
    Exception raised when an iterative algorithm fails to converge.

    Attributes
    ----------
    algorithm : str
        Name of the algorithm
    iterations : int
        Number of iterations attempted
    tolerance : float
        Convergence tolerance used
    """

    def __init__(self, algorithm: str, iterations: int, tolerance: float):
        self.algorithm = algorithm
        self.iterations = iterations
        self.tolerance = tolerance
        message = (
            f"{algorithm} failed to converge after {iterations} iterations "
            f"(tolerance: {tolerance})"
        )
        super().__init__(message)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_positive(value: float, name: str) -> None:
    """
    Validate that a value is strictly positive.

    Parameters
    ----------
    value : float
        Value to validate
    name : str
        Parameter name for error message

    Raises
    ------
    ParameterValidationError
        If value <= 0
    """
    if value <= 0:
        raise ParameterValidationError(name, value, "must be > 0")


def validate_non_negative(value: float, name: str) -> None:
    """
    Validate that a value is non-negative.

    Parameters
    ----------
    value : float
        Value to validate
    name : str
        Parameter name for error message

    Raises
    ------
    ParameterValidationError
        If value < 0
    """
    if value < 0:
        raise ParameterValidationError(name, value, "must be >= 0")


def validate_probability(value: float, name: str) -> None:
    """
    Validate that a value is a valid probability [0, 1].

    Parameters
    ----------
    value : float
        Value to validate
    name : str
        Parameter name for error message

    Raises
    ------
    ParameterValidationError
        If value not in [0, 1]
    """
    if not 0 <= value <= 1:
        raise ParameterValidationError(name, value, "must be in [0, 1]")


def validate_correlation(rho: float, name: str = "rho") -> None:
    """
    Validate that a value is a valid correlation [-1, 1].

    Parameters
    ----------
    rho : float
        Correlation value to validate
    name : str
        Parameter name for error message

    Raises
    ------
    ParameterValidationError
        If rho not in [-1, 1]
    """
    if not -1 <= rho <= 1:
        raise ParameterValidationError(name, rho, "must be in [-1, 1]")


def validate_in_range(
    value: float,
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    inclusive: bool = True
) -> None:
    """
    Validate that a value is within a specified range.

    Parameters
    ----------
    value : float
        Value to validate
    name : str
        Parameter name for error message
    min_val : Optional[float]
        Minimum allowed value (None for no lower bound)
    max_val : Optional[float]
        Maximum allowed value (None for no upper bound)
    inclusive : bool
        If True, bounds are inclusive (default: True)

    Raises
    ------
    ParameterValidationError
        If value is outside the specified range
    """
    if min_val is not None:
        if inclusive and value < min_val:
            raise ParameterValidationError(name, value, f"must be >= {min_val}")
        elif not inclusive and value <= min_val:
            raise ParameterValidationError(name, value, f"must be > {min_val}")

    if max_val is not None:
        if inclusive and value > max_val:
            raise ParameterValidationError(name, value, f"must be <= {max_val}")
        elif not inclusive and value >= max_val:
            raise ParameterValidationError(name, value, f"must be < {max_val}")


def validate_garch_stationarity(
    alpha: float,
    beta: float,
    model: str = "GARCH"
) -> None:
    """
    Validate GARCH stationarity condition.

    Parameters
    ----------
    alpha : float
        ARCH coefficient
    beta : float
        GARCH coefficient
    model : str
        Model name for error message

    Raises
    ------
    StationarityViolationError
        If alpha + beta >= 1
    """
    persistence = alpha + beta
    if persistence >= 1:
        raise StationarityViolationError(model, alpha, beta, persistence)
