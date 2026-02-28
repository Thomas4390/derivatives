"""
Model Helpers - Utility functions for model handling.

Provides:
- Model categorization helpers
- Parameter validation
- Display formatting
- LaTeX equation rendering
"""

from typing import Any

import numpy as np

from streamlit_app.simulation.config.model_registry import (
    MARKET_PARAMETERS,
    MODEL_REGISTRY,
    SIMULATION_PARAMETERS,
    ModelCategory,
    ParameterSpec,
    PricingMethod,
    get_model,
)

# ============================================================================
# CATEGORIZATION HELPERS
# ============================================================================

def get_model_category(model_key: str) -> str:
    """Get human-readable category for a model."""
    model = get_model(model_key)
    category_names = {
        ModelCategory.CONTINUOUS: "Continuous Diffusion",
        ModelCategory.DISCRETE: "Discrete Time (GARCH)",
        ModelCategory.JUMP: "Jump-Diffusion",
        ModelCategory.STOCHASTIC_VOL: "Stochastic Volatility",
    }
    return category_names.get(model.category, "Unknown")


def get_volatility_type(model_key: str) -> str:
    """Get volatility type description."""
    try:
        model = get_model(model_key)
    except ValueError:
        return "Custom"

    if not model.has_stochastic_vol:
        return "Constant"
    if model.category == ModelCategory.DISCRETE:
        return "Time-Varying (GARCH)"
    return "Stochastic"


def get_model_features(model_key: str) -> dict[str, bool]:
    """Get feature flags for a model."""
    model = get_model(model_key)
    return {
        "stochastic_vol": model.has_stochastic_vol,
        "jumps": model.has_jumps,
        "analytical_pricing": PricingMethod.ANALYTICAL in model.pricing_methods,
        "fft_pricing": PricingMethod.FFT in model.pricing_methods,
        "has_feller_condition": model.feller_condition is not None,
        "has_stationarity_condition": model.stationarity_condition is not None,
    }


def get_feature_badges(model_key: str) -> list[tuple[str, str]]:
    """
    Get badge labels for model features.

    Returns:
        List of (label, color) tuples for Streamlit badges
    """
    try:
        model = get_model(model_key)
    except ValueError:
        return [("Custom", "violet"), ("MC", "green")]
    badges = []

    # Volatility type
    if not model.has_stochastic_vol:
        badges.append(("σ Constant", "blue"))
    elif model.category == ModelCategory.DISCRETE:
        badges.append(("σ Time-Varying", "orange"))
    else:
        badges.append(("σ Stochastic", "green"))

    # Jumps
    if model.has_jumps:
        badges.append(("Jumps", "red"))

    # Pricing methods
    if PricingMethod.ANALYTICAL in model.pricing_methods:
        badges.append(("BS", "violet"))
    if PricingMethod.FFT in model.pricing_methods:
        badges.append(("FFT", "violet"))

    return badges


def group_models_by_category() -> dict[str, list[str]]:
    """Group models by their category."""
    groups = {}
    for key, model in MODEL_REGISTRY.items():
        cat_name = get_model_category(key)
        if cat_name not in groups:
            groups[cat_name] = []
        groups[cat_name].append(key)
    return groups


# ============================================================================
# PARAMETER HELPERS
# ============================================================================

def get_model_parameters(model_key: str) -> list[ParameterSpec]:
    """Get parameter specifications for a model."""
    model = get_model(model_key)
    return model.parameters


def get_all_parameters(model_key: str) -> list[ParameterSpec]:
    """Get all parameters (market + simulation + model)."""
    model = get_model(model_key)
    return MARKET_PARAMETERS + model.parameters + SIMULATION_PARAMETERS


def validate_parameters(model_key: str, params: dict[str, Any]) -> list[str]:
    """
    Validate parameters for a model.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    model = get_model(model_key)

    for param_spec in model.parameters:
        if param_spec.name in params:
            value = params[param_spec.name]
            if value < param_spec.min_value:
                errors.append(
                    f"{param_spec.display_name} ({value}) below minimum ({param_spec.min_value})"
                )
            elif value > param_spec.max_value:
                errors.append(
                    f"{param_spec.display_name} ({value}) above maximum ({param_spec.max_value})"
                )

    return errors


def format_parameter_value(param_spec: ParameterSpec, value: float) -> str:
    """Format parameter value for display."""
    return param_spec.format % value


# ============================================================================
# EQUATION HELPERS
# ============================================================================

def get_model_equations(model_key: str) -> dict[str, str]:
    """Get LaTeX equations for a model."""
    model = get_model(model_key)
    equations = {"main": model.equation_main}

    if model.equation_vol:
        equations["volatility"] = model.equation_vol
    if model.equation_jump:
        equations["jump"] = model.equation_jump
    eq_analytical = getattr(model, "equation_analytical", None)
    eq_cf = getattr(model, "equation_cf", None)
    eq_mc = getattr(model, "equation_mc", None)
    if eq_analytical:
        equations["analytical"] = eq_analytical
    if eq_cf:
        equations["cf"] = eq_cf
    if eq_mc:
        equations["mc"] = eq_mc

    return equations


def get_equation_with_values(
    model_key: str,
    params: dict[str, Any]
) -> str:
    """
    Get equation with parameter values substituted.

    For educational display showing actual numbers.
    """
    model_lower = model_key.lower()

    if model_lower == "gbm":
        sigma = params.get("sigma", 0.20)
        return rf"dS = \mu S \, dt + {sigma:.2f} S \, dW"

    if model_lower == "heston":
        kappa = params.get("kappa", 2.0)
        theta = params.get("theta", 0.04)
        xi = params.get("xi", 0.3)
        rho = params.get("rho", -0.7)
        return (
            rf"dV = {kappa:.1f}({theta:.3f} - V) \, dt + {xi:.2f} \sqrt{{V}} \, dW_V"
            rf"\quad (\rho = {rho:.2f})"
        )

    if model_lower == "garch":
        omega = params.get("omega", 0.002)
        alpha = params.get("alpha", 0.06)
        beta = params.get("beta", 0.90)
        return rf"\sigma_t^2 = {omega:.2e} + {alpha:.3f} \epsilon_{{t-1}}^2 + {beta:.2f} \sigma_{{t-1}}^2"

    # Add more as needed
    return get_model(model_key).equation_main


# ============================================================================
# CONDITION HELPERS
# ============================================================================

def check_feller_condition(params: dict[str, Any]) -> tuple[bool, float, float]:
    """
    Check Feller condition for Heston/Bates.

    Returns:
        (is_satisfied, lhs_value, rhs_value)
    """
    kappa = params.get("kappa", 2.0)
    theta = params.get("theta", 0.04)
    xi = params.get("xi", 0.3)

    lhs = 2 * kappa * theta
    rhs = xi ** 2

    return (lhs > rhs, lhs, rhs)


def check_garch_stationarity(
    model_key: str,
    params: dict[str, Any]
) -> tuple[bool, float]:
    """
    Check stationarity condition for GARCH models.

    Returns:
        (is_stationary, persistence)
    """
    alpha = params.get("alpha", 0.06)
    beta = params.get("beta", 0.90)

    model_lower = model_key.lower()

    if model_lower == "garch":
        persistence = alpha + beta
    elif model_lower == "ngarch":
        theta = params.get("theta_ngarch", params.get("theta", 0.5))
        persistence = alpha * (1 + theta ** 2) + beta
    elif model_lower == "gjr_garch":
        gamma = params.get("gamma", 0.03)
        persistence = alpha + beta + gamma / 2
    else:
        return (True, 0.0)

    return (persistence < 1, persistence)


def get_condition_display(model_key: str, params: dict[str, Any]) -> dict[str, Any] | None:
    """
    Get condition check display info.

    Returns:
        Dictionary with condition info for display, or None if no conditions
    """
    model = get_model(model_key)

    if model.feller_condition:
        satisfied, lhs, rhs = check_feller_condition(params)
        return {
            "name": "Feller Condition",
            "equation": model.feller_condition,
            "satisfied": satisfied,
            "message": f"2κθ = {lhs:.4f} {'>' if satisfied else '≤'} {rhs:.4f} = ξ²",
            "help": "Ensures variance stays positive" if satisfied
                   else "Variance may become negative. Consider using Full Truncation scheme.",
        }

    if model.stationarity_condition:
        satisfied, persistence = check_garch_stationarity(model_key, params)
        return {
            "name": "Stationarity Condition",
            "equation": model.stationarity_condition,
            "satisfied": satisfied,
            "message": f"Persistence = {persistence:.4f} {'<' if satisfied else '≥'} 1",
            "help": "Volatility mean-reverts to long-run level" if satisfied
                   else "Non-stationary: volatility may explode over time",
        }

    return None


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def get_model_icon(model_key: str) -> str:
    """Get emoji icon for model."""
    icons = {
        "gbm": "📈",
        "heston": "🌊",
        "merton": "⚡",
        "bates": "🔀",
        "garch": "📊",
        "ngarch": "📉",
        "gjr_garch": "⚖️",
    }
    return icons.get(model_key.lower(), "🧪")


def get_short_description(model_key: str) -> str:
    """Get short one-line description."""
    descriptions = {
        "gbm": "Log-normal model with constant volatility",
        "heston": "Mean-reverting stochastic variance",
        "merton": "GBM with random jumps",
        "bates": "Stochastic vol + jumps",
        "garch": "Volatility clustering from past returns",
        "ngarch": "GARCH with leverage effect",
        "gjr_garch": "Asymmetric response to negative shocks",
    }
    if model_key.lower() in descriptions:
        return descriptions[model_key.lower()]
    # Fallback for custom model
    try:
        import streamlit as st
        custom = st.session_state.get("custom_model")
        if custom and "spec" in custom:
            return custom["spec"].description
    except Exception:
        pass
    return "Custom user-defined model"


def format_volatility_display(vol: float) -> str:
    """Format volatility for display (as percentage)."""
    return f"{vol * 100:.1f}%"


def format_price_display(price: float) -> str:
    """Format price for display."""
    return f"${price:,.2f}"


# ============================================================================
# STATISTICS HELPERS
# ============================================================================

def compute_summary_statistics(
    terminal_prices: np.ndarray,
    initial_price: float
) -> dict[str, float]:
    """Compute summary statistics for terminal distribution."""
    returns = np.log(terminal_prices / initial_price)

    return {
        "mean_price": np.mean(terminal_prices),
        "median_price": np.median(terminal_prices),
        "std_price": np.std(terminal_prices),
        "min_price": np.min(terminal_prices),
        "max_price": np.max(terminal_prices),
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "skewness": compute_skewness(returns),
        "kurtosis": compute_kurtosis(returns),
        "var_95": np.percentile(terminal_prices, 5),
        "var_99": np.percentile(terminal_prices, 1),
    }


def compute_skewness(x: np.ndarray) -> float:
    """Compute skewness."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    return (n / ((n - 1) * (n - 2))) * np.sum(((x - mean) / std) ** 3)


def compute_kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis."""
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    m4 = np.mean((x - mean) ** 4)
    return (m4 / (std ** 4)) - 3
