"""
Model Registry - Unified metadata for all simulation models.

Provides comprehensive information about each model including:
- Display name and category
- Model characteristics (stochastic vol, jumps)
- Available pricing methods
- Parameter definitions
- Mathematical equations (LaTeX)
"""

from dataclasses import dataclass
from enum import Enum


class ModelCategory(str, Enum):
    """Model categorization."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    JUMP = "jump"
    STOCHASTIC_VOL = "stochastic_volatility"


class PricingMethod(str, Enum):
    """Available pricing methods."""
    ANALYTICAL = "analytical"
    FFT = "fft"
    MONTE_CARLO = "monte_carlo"


@dataclass
class ParameterSpec:
    """Specification for a model parameter."""
    name: str
    display_name: str
    default: float
    min_value: float
    max_value: float
    step: float
    description: str
    format: str = "%.4f"


@dataclass
class ModelSpec:
    """Complete specification for a simulation model."""
    key: str
    name: str
    short_name: str
    category: ModelCategory
    has_stochastic_vol: bool
    has_jumps: bool
    pricing_methods: list[PricingMethod]
    parameters: list[ParameterSpec]
    equation_main: str
    equation_vol: str | None = None
    equation_jump: str | None = None
    equation_analytical: str | None = None   # Closed-form pricing (BS)
    equation_cf: str | None = None           # Characteristic function (FFT)
    equation_mc: str | None = None           # Monte Carlo discretization
    description: str = ""
    stationarity_condition: str | None = None
    feller_condition: str | None = None


# ============================================================================
# PARAMETER DEFINITIONS
# ============================================================================

# Common market parameters (shared by all models)
MARKET_PARAMETERS = [
    ParameterSpec(
        name="spot",
        display_name="Spot Price",
        default=100.0,
        min_value=1.0,
        max_value=10000.0,
        step=1.0,
        description="Current asset price",
        format="%.2f"
    ),
    ParameterSpec(
        name="drift",
        display_name="Expected Return (μ)",
        default=0.08,
        min_value=-0.20,
        max_value=0.50,
        step=0.01,
        description="Annualized expected return",
        format="%.2f"
    ),
    ParameterSpec(
        name="risk_free_rate",
        display_name="Risk-Free Rate (r)",
        default=0.05,
        min_value=0.0,
        max_value=0.20,
        step=0.005,
        description="Annualized risk-free interest rate",
        format="%.3f"
    ),
    ParameterSpec(
        name="time_horizon",
        display_name="Time Horizon (T)",
        default=1.0,
        min_value=0.1,
        max_value=10.0,
        step=0.1,
        description="Simulation horizon in years",
        format="%.1f"
    ),
]

# Simulation settings
SIMULATION_PARAMETERS = [
    ParameterSpec(
        name="n_paths",
        display_name="Number of Paths",
        default=10000,
        min_value=100,
        max_value=100000,
        step=100,
        description="Number of Monte Carlo paths",
        format="%d"
    ),
    ParameterSpec(
        name="n_steps",
        display_name="Time Steps",
        default=252,
        min_value=10,
        max_value=1000,
        step=1,
        description="Number of time steps (252 = daily)",
        format="%d"
    ),
    ParameterSpec(
        name="seed",
        display_name="Random Seed",
        default=42,
        min_value=0,
        max_value=99999,
        step=1,
        description="Random seed for reproducibility (0 = random)",
        format="%d"
    ),
]

# GBM parameters
GBM_PARAMETERS = [
    ParameterSpec(
        name="sigma",
        display_name="Volatility (σ)",
        default=0.20,
        min_value=0.01,
        max_value=1.0,
        step=0.01,
        description="Annualized volatility",
        format="%.2f"
    ),
]

# Heston parameters
HESTON_PARAMETERS = [
    ParameterSpec(
        name="v0",
        display_name="Initial Variance (V₀)",
        default=0.04,
        min_value=0.001,
        max_value=1.0,
        step=0.01,
        description="Initial variance level (0.04 = 20% vol)",
        format="%.4f"
    ),
    ParameterSpec(
        name="kappa",
        display_name="Mean Reversion (κ)",
        default=2.0,
        min_value=0.1,
        max_value=10.0,
        step=0.1,
        description="Speed of variance mean reversion",
        format="%.2f"
    ),
    ParameterSpec(
        name="theta",
        display_name="Long-Run Variance (θ)",
        default=0.04,
        min_value=0.001,
        max_value=1.0,
        step=0.01,
        description="Long-run variance level",
        format="%.4f"
    ),
    ParameterSpec(
        name="xi",
        display_name="Vol of Vol (ξ)",
        default=0.3,
        min_value=0.01,
        max_value=1.0,
        step=0.01,
        description="Volatility of variance process",
        format="%.2f"
    ),
    ParameterSpec(
        name="rho",
        display_name="Correlation (ρ)",
        default=-0.7,
        min_value=-0.99,
        max_value=0.99,
        step=0.01,
        description="Correlation between price and variance",
        format="%.2f"
    ),
]

# Merton jump parameters
MERTON_PARAMETERS = [
    ParameterSpec(
        name="sigma",
        display_name="Diffusion Vol (σ)",
        default=0.20,
        min_value=0.01,
        max_value=1.0,
        step=0.01,
        description="Diffusion volatility",
        format="%.2f"
    ),
    ParameterSpec(
        name="lambda_j",
        display_name="Jump Intensity (λ)",
        default=0.5,
        min_value=0.0,
        max_value=5.0,
        step=0.1,
        description="Expected number of jumps per year",
        format="%.2f"
    ),
    ParameterSpec(
        name="mu_j",
        display_name="Mean Jump Size (μⱼ)",
        default=-0.1,
        min_value=-0.5,
        max_value=0.5,
        step=0.01,
        description="Mean of log-jump size",
        format="%.2f"
    ),
    ParameterSpec(
        name="sigma_j",
        display_name="Jump Vol (σⱼ)",
        default=0.2,
        min_value=0.01,
        max_value=0.5,
        step=0.01,
        description="Volatility of log-jump size",
        format="%.2f"
    ),
]

# Bates parameters (Heston + Merton jumps)
BATES_PARAMETERS = HESTON_PARAMETERS + [
    ParameterSpec(
        name="lambda_j",
        display_name="Jump Intensity (λ)",
        default=0.5,
        min_value=0.0,
        max_value=5.0,
        step=0.1,
        description="Expected number of jumps per year",
        format="%.2f"
    ),
    ParameterSpec(
        name="mu_j",
        display_name="Mean Jump Size (μⱼ)",
        default=-0.1,
        min_value=-0.5,
        max_value=0.5,
        step=0.01,
        description="Mean of log-jump size",
        format="%.2f"
    ),
    ParameterSpec(
        name="sigma_j",
        display_name="Jump Vol (σⱼ)",
        default=0.2,
        min_value=0.01,
        max_value=0.5,
        step=0.01,
        description="Volatility of log-jump size",
        format="%.2f"
    ),
]

# GARCH parameters
GARCH_PARAMETERS = [
    ParameterSpec(
        name="sigma0",
        display_name="Initial Vol (σ₀)",
        default=0.20,
        min_value=0.01,
        max_value=1.0,
        step=0.01,
        description="Initial volatility level",
        format="%.2f"
    ),
    ParameterSpec(
        name="omega",
        display_name="Constant (ω)",
        default=0.002,
        min_value=0.00001,
        max_value=0.01,
        step=0.0001,
        description="Variance constant — long-run vol ≈ √(ω/(1-α-β))",
        format="%.4f"
    ),
    ParameterSpec(
        name="alpha",
        display_name="ARCH Coef (α)",
        default=0.06,
        min_value=0.001,
        max_value=0.5,
        step=0.01,
        description="Response to past shocks",
        format="%.3f"
    ),
    ParameterSpec(
        name="beta",
        display_name="GARCH Coef (β)",
        default=0.90,
        min_value=0.0,
        max_value=0.99,
        step=0.01,
        description="Persistence of volatility",
        format="%.2f"
    ),
]

# NGARCH parameters
NGARCH_PARAMETERS = GARCH_PARAMETERS + [
    ParameterSpec(
        name="theta",
        display_name="Leverage (θ)",
        default=0.5,
        min_value=0.0,
        max_value=2.0,
        step=0.1,
        description="Leverage effect parameter",
        format="%.2f"
    ),
]

# GJR-GARCH parameters
GJR_GARCH_PARAMETERS = GARCH_PARAMETERS + [
    ParameterSpec(
        name="gamma",
        display_name="Asymmetry (γ)",
        default=0.03,
        min_value=0.0,
        max_value=0.3,
        step=0.01,
        description="Asymmetry coefficient for negative shocks",
        format="%.3f"
    ),
]


# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "gbm": ModelSpec(
        key="gbm",
        name="Geometric Brownian Motion",
        short_name="GBM",
        category=ModelCategory.CONTINUOUS,
        has_stochastic_vol=False,
        has_jumps=False,
        pricing_methods=[PricingMethod.ANALYTICAL, PricingMethod.FFT, PricingMethod.MONTE_CARLO],
        parameters=GBM_PARAMETERS,
        equation_main=r"dS = \mu S \, dt + \sigma S \, dW",
        equation_analytical=r"C = S_0 N(d_1) - Ke^{-rT}N(d_2), \quad d_{1,2} = \frac{\ln(S_0/K) + (r \pm \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}",
        equation_cf=r"\varphi(u) = \exp\!\left[iu\!\left(\ln S_0 + (r - q - \tfrac{1}{2}\sigma^2)T\right) - \tfrac{1}{2}\sigma^2 T u^2\right]",
        equation_mc=r"S_{t+\Delta t} = S_t \exp\!\left[\left(r - q - \tfrac{1}{2}\sigma^2\right)\!\Delta t + \sigma\sqrt{\Delta t}\, Z\right], \quad Z \sim \mathcal{N}(0,1)",
        description="Classic log-normal model. Constant volatility, continuous paths.",
    ),

    "heston": ModelSpec(
        key="heston",
        name="Heston Stochastic Volatility",
        short_name="Heston",
        category=ModelCategory.STOCHASTIC_VOL,
        has_stochastic_vol=True,
        has_jumps=False,
        pricing_methods=[PricingMethod.FFT, PricingMethod.MONTE_CARLO],
        parameters=HESTON_PARAMETERS,
        equation_main=r"dS = \mu S \, dt + \sqrt{V} S \, dW_S",
        equation_vol=r"dV = \kappa(\theta - V) \, dt + \xi \sqrt{V} \, dW_V",
        equation_cf=r"\varphi(u) = \exp\!\bigl(C(u,T) + D(u,T)\,v_0 + iu\ln S_0\bigr), \quad d = \sqrt{(\kappa - i\rho\xi u)^2 + \xi^2(iu + u^2)}",
        equation_mc=r"\begin{aligned} S_{t+\Delta t} &= S_t \exp\!\left[(r-q-\tfrac{V_t}{2})\Delta t + \sqrt{V_t\Delta t}\, Z_1\right] \\ V_{t+\Delta t} &= V_t + \kappa(\theta - V_t)\Delta t + \xi\sqrt{V_t\Delta t}\, Z_2, \quad \mathrm{corr}(Z_1,Z_2) = \rho \end{aligned}",
        description="Mean-reverting stochastic variance. Captures volatility smile.",
        feller_condition=r"2\kappa\theta > \xi^2",
    ),

    "merton": ModelSpec(
        key="merton",
        name="Merton Jump-Diffusion",
        short_name="Merton",
        category=ModelCategory.JUMP,
        has_stochastic_vol=False,
        has_jumps=True,
        pricing_methods=[PricingMethod.FFT, PricingMethod.MONTE_CARLO],
        parameters=MERTON_PARAMETERS,
        equation_main=r"dS/S = (\mu - \lambda k) \, dt + \sigma \, dW + (J-1) \, dN",
        equation_jump=r"J \sim \log\mathcal{N}(\mu_J, \sigma_J^2)",
        equation_cf=r"\varphi(u) = \varphi_{\text{GBM}}(u) \cdot \exp\!\left[\lambda T\!\left(e^{iu\mu_J - \frac{1}{2}\sigma_J^2 u^2} - 1\right)\right]",
        equation_mc=r"S_{t+\Delta t} = S_t \exp\!\left[(r-q-\lambda k - \tfrac{\sigma^2}{2})\Delta t + \sigma\sqrt{\Delta t}\, Z\right] \cdot \prod_{j=1}^{N_t} e^{J_j}, \;\; N_t \sim \mathrm{Poisson}(\lambda\Delta t)",
        description="GBM with random jumps. Models crash risk and fat tails.",
    ),

    "bates": ModelSpec(
        key="bates",
        name="Bates (Heston + Jumps)",
        short_name="Bates",
        category=ModelCategory.STOCHASTIC_VOL,
        has_stochastic_vol=True,
        has_jumps=True,
        pricing_methods=[PricingMethod.FFT, PricingMethod.MONTE_CARLO],
        parameters=BATES_PARAMETERS,
        equation_main=r"dS = (\mu - \lambda k) S \, dt + \sqrt{V} S \, dW_S + (J-1) S \, dN",
        equation_vol=r"dV = \kappa(\theta - V) \, dt + \xi \sqrt{V} \, dW_V",
        equation_jump=r"J \sim \log\mathcal{N}(\mu_J, \sigma_J^2)",
        equation_cf=r"\varphi(u) = \varphi_{\text{Heston}}(u) \cdot \exp\!\left[\lambda T\!\left(e^{iu\mu_J - \frac{1}{2}\sigma_J^2 u^2} - 1\right)\right]",
        equation_mc=r"\begin{aligned} S_{t+\Delta t} &= S_t \exp\!\left[(r-q-\lambda k - \tfrac{V_t}{2})\Delta t + \sqrt{V_t\Delta t}\, Z_1\right] \cdot \prod_{j=1}^{N_t} e^{J_j} \\ V_{t+\Delta t} &= V_t + \kappa(\theta - V_t)\Delta t + \xi\sqrt{V_t\Delta t}\, Z_2 \end{aligned}",
        description="Combines stochastic vol and jumps. Most flexible continuous-time model.",
        feller_condition=r"2\kappa\theta > \xi^2",
    ),

    "garch": ModelSpec(
        key="garch",
        name="GARCH(1,1)",
        short_name="GARCH",
        category=ModelCategory.DISCRETE,
        has_stochastic_vol=True,
        has_jumps=False,
        pricing_methods=[PricingMethod.MONTE_CARLO],
        parameters=GARCH_PARAMETERS,
        equation_main=r"r_t = \mu - \frac{1}{2}\sigma_t^2 + \sigma_t z_t",
        equation_vol=r"\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2",
        equation_mc=r"S_t = S_{t-1}\exp\!\left(\mu - \tfrac{1}{2}\sigma_t^2 + \sigma_t z_t\right), \quad z_t \sim \mathcal{N}(0,1)",
        description="Volatility clustering from past returns. Industry standard.",
        stationarity_condition=r"\alpha + \beta < 1",
    ),

    "ngarch": ModelSpec(
        key="ngarch",
        name="NGARCH (Nonlinear)",
        short_name="NGARCH",
        category=ModelCategory.DISCRETE,
        has_stochastic_vol=True,
        has_jumps=False,
        pricing_methods=[PricingMethod.MONTE_CARLO],
        parameters=NGARCH_PARAMETERS,
        equation_main=r"r_t = \mu - \frac{1}{2}\sigma_t^2 + \sigma_t z_t",
        equation_vol=r"\sigma_t^2 = \omega + \alpha(\epsilon_{t-1} - \theta\sigma_{t-1})^2 + \beta \sigma_{t-1}^2",
        equation_mc=r"S_t = S_{t-1}\exp\!\left(\mu - \tfrac{1}{2}\sigma_t^2 + \sigma_t z_t\right), \quad z_t \sim \mathcal{N}(0,1)",
        description="GARCH with leverage effect. Bad news increases vol more than good news.",
        stationarity_condition=r"\alpha(1 + \theta^2) + \beta < 1",
    ),

    "gjr_garch": ModelSpec(
        key="gjr_garch",
        name="GJR-GARCH (Threshold)",
        short_name="GJR-GARCH",
        category=ModelCategory.DISCRETE,
        has_stochastic_vol=True,
        has_jumps=False,
        pricing_methods=[PricingMethod.MONTE_CARLO],
        parameters=GJR_GARCH_PARAMETERS,
        equation_main=r"r_t = \mu - \frac{1}{2}\sigma_t^2 + \sigma_t z_t",
        equation_vol=r"\sigma_t^2 = \omega + (\alpha + \gamma I_{t-1}) \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2",
        equation_mc=r"S_t = S_{t-1}\exp\!\left(\mu - \tfrac{1}{2}\sigma_t^2 + \sigma_t z_t\right), \quad z_t \sim \mathcal{N}(0,1)",
        description="GARCH with asymmetric response to negative shocks.",
        stationarity_condition=r"\alpha + \beta + \gamma/2 < 1",
    ),
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model(key: str) -> ModelSpec:
    """Get model specification by key."""
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]
    # Check for custom model in session state
    if key == "custom":
        try:
            import streamlit as st
            custom = st.session_state.get("custom_model")
            if custom and "spec" in custom:
                return custom["spec"]
        except Exception:
            pass
    raise ValueError(f"Unknown model: {key}. Available: {list(MODEL_REGISTRY.keys())}")


def get_all_models() -> dict[str, ModelSpec]:
    """Get all model specifications."""
    return MODEL_REGISTRY


def get_models_by_category(category: ModelCategory) -> dict[str, ModelSpec]:
    """Get models filtered by category."""
    return {k: v for k, v in MODEL_REGISTRY.items() if v.category == category}


def get_models_with_stochastic_vol() -> dict[str, ModelSpec]:
    """Get models with stochastic volatility."""
    return {k: v for k, v in MODEL_REGISTRY.items() if v.has_stochastic_vol}


def get_models_with_jumps() -> dict[str, ModelSpec]:
    """Get models with jump components."""
    return {k: v for k, v in MODEL_REGISTRY.items() if v.has_jumps}


def get_models_with_pricing_method(method: PricingMethod) -> dict[str, ModelSpec]:
    """Get models supporting a specific pricing method."""
    return {k: v for k, v in MODEL_REGISTRY.items() if method in v.pricing_methods}


def get_parameter_defaults(model_key: str) -> dict[str, float]:
    """Get default parameter values for a model."""
    model = get_model(model_key)
    return {p.name: p.default for p in model.parameters}


def get_all_parameter_defaults(model_key: str) -> dict[str, float]:
    """Get all defaults including market and simulation parameters."""
    defaults = {}
    for p in MARKET_PARAMETERS + SIMULATION_PARAMETERS:
        defaults[p.name] = p.default
    defaults.update(get_parameter_defaults(model_key))
    return defaults


# Model display order for UI
MODEL_DISPLAY_ORDER = ["gbm", "heston", "merton", "bates", "garch", "ngarch", "gjr_garch"]


# Category display names
CATEGORY_DISPLAY_NAMES = {
    ModelCategory.CONTINUOUS: "Continuous Diffusion",
    ModelCategory.DISCRETE: "Discrete Time (GARCH)",
    ModelCategory.JUMP: "Jump-Diffusion",
    ModelCategory.STOCHASTIC_VOL: "Stochastic Volatility",
}
