"""
Per-model parameter specifications
====================================

Single source of truth for each model's calibratable parameters: name,
display label, range (lo/hi), default true-value, formatting.  Used by
the sidebar's "True Parameters" panel to render model-specific sliders.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    lo: float
    hi: float
    default: float
    step: float
    fmt: str
    description: str = ""
    # SI-style unit hint appended to the slider label as "[units]" when
    # populated. Left blank for dimensionless quantities and for params
    # whose meaning is already obvious from the symbol (e.g. ρ, log-jump
    # statistics) — adding "[—]" everywhere is visual noise.
    units: str = ""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    params: tuple[ParamSpec, ...]
    extras: dict = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        return len(self.params)

    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    def true_param_dict(self) -> dict[str, float]:
        return {p.name: p.default for p in self.params}


# Canonical parameter descriptions, shared by every model that uses them
# so Bates / NGARCH / GJR carry the same tooltip text as Heston / Merton
# / GARCH without copy-paste drift.
_HESTON_DESCRIPTIONS = {
    "v0": "Initial spot variance — square of initial vol.",
    "kappa": "Speed at which variance reverts to its long-run mean.",
    "theta": "Long-run variance level — square of the long-run vol.",
    "alpha": (
        "Diffusion coefficient of the variance process — controls smile "
        "curvature."
    ),
    "rho": (
        "Correlation between spot and variance Brownian motions — negative "
        "for equities (leverage effect)."
    ),
}
_MERTON_DESCRIPTIONS = {
    "lam": "Expected number of jumps per year.",
    "alpha_j": "Mean of the log-jump size — usually negative for equity tail risk.",
    "sigma_j": "Standard deviation of the log-jump distribution.",
}
_GARCH_DESCRIPTIONS = {
    "omega": "Constant term ω of the variance recursion.",
    "alpha": "Sensitivity of variance to last squared return.",
    "beta": (
        "Persistence of past variance — α + β must be < 1 for stationarity."
    ),
}

HESTON_SPEC = ModelSpec(
    key="heston",
    display_name="Heston",
    params=(
        ParamSpec("v0", "v₀ (initial variance)", 0.001, 0.5, 0.04, 0.001, "%.4f",
                  _HESTON_DESCRIPTIONS["v0"], units="σ²"),
        ParamSpec("kappa", "κ (mean-reversion speed)", 0.1, 10.0, 2.0, 0.1, "%.2f",
                  _HESTON_DESCRIPTIONS["kappa"], units="yr⁻¹"),
        ParamSpec("theta", "σ² (long-run variance)", 0.001, 0.5, 0.04, 0.001, "%.4f",
                  _HESTON_DESCRIPTIONS["theta"], units="σ²"),
        ParamSpec("alpha", "α (vol of vol)", 0.01, 1.5, 0.3, 0.01, "%.3f",
                  _HESTON_DESCRIPTIONS["alpha"]),
        ParamSpec("rho", "ρ (spot-vol correlation)", -0.999, 0.999, -0.7, 0.01, "%.3f",
                  _HESTON_DESCRIPTIONS["rho"]),
    ),
)

MERTON_SPEC = ModelSpec(
    key="merton",
    display_name="Merton",
    params=(
        ParamSpec("sigma", "σ (diffusion vol)", 0.01, 1.0, 0.18, 0.01, "%.3f",
                  "Brownian diffusion volatility (continuous component)."),
        ParamSpec("lam", "λ (jump intensity, annualised)", 0.0, 5.0, 0.5, 0.05, "%.2f",
                  _MERTON_DESCRIPTIONS["lam"], units="yr⁻¹"),
        ParamSpec("alpha_j", "α_J (mean log-jump)", -0.5, 0.1, -0.10, 0.01, "%.3f",
                  _MERTON_DESCRIPTIONS["alpha_j"]),
        ParamSpec("sigma_j", "σ_J (jump std)", 0.01, 0.5, 0.20, 0.01, "%.3f",
                  _MERTON_DESCRIPTIONS["sigma_j"]),
    ),
)

BATES_SPEC = ModelSpec(
    key="bates",
    display_name="Bates",
    params=(
        ParamSpec("v0", "v₀", 0.001, 0.5, 0.04, 0.001, "%.4f",
                  _HESTON_DESCRIPTIONS["v0"], units="σ²"),
        ParamSpec("kappa", "κ", 0.1, 10.0, 1.5, 0.1, "%.2f",
                  _HESTON_DESCRIPTIONS["kappa"], units="yr⁻¹"),
        ParamSpec("theta", "σ²", 0.001, 0.5, 0.04, 0.001, "%.4f",
                  _HESTON_DESCRIPTIONS["theta"], units="σ²"),
        ParamSpec("alpha", "α", 0.01, 1.5, 0.3, 0.01, "%.3f",
                  _HESTON_DESCRIPTIONS["alpha"]),
        ParamSpec("rho", "ρ", -0.999, 0.999, -0.65, 0.01, "%.3f",
                  _HESTON_DESCRIPTIONS["rho"]),
        ParamSpec("lam", "λ", 0.0, 5.0, 0.5, 0.05, "%.2f",
                  _MERTON_DESCRIPTIONS["lam"], units="yr⁻¹"),
        ParamSpec("alpha_j", "α_J", -0.5, 0.1, -0.10, 0.01, "%.3f",
                  _MERTON_DESCRIPTIONS["alpha_j"]),
        ParamSpec("sigma_j", "σ_J", 0.01, 0.5, 0.15, 0.01, "%.3f",
                  _MERTON_DESCRIPTIONS["sigma_j"]),
    ),
)

_HESTON_NANDI_DESCRIPTIONS = {
    "omega": "Constant term ω of the risk-neutral variance recursion (per period).",
    "alpha": "ARCH coefficient — sensitivity of variance to the squared shock.",
    "beta": "GARCH persistence of past variance.",
    "gamma": (
        "Risk-neutral leverage — asymmetric shock shift. Large (O(100)) because "
        "it scales √hₜ; stationarity requires β + αγ² < 1."
    ),
    "h0": "Initial conditional variance h₁ (per period).",
}

HESTON_NANDI_SPEC = ModelSpec(
    key="heston_nandi",
    display_name="AGARCH (Heston-Nandi)",
    params=(
        ParamSpec("omega", "ω (variance intercept)", 1e-9, 1e-4, 2e-6, 1e-7, "%.2e",
                  _HESTON_NANDI_DESCRIPTIONS["omega"]),
        ParamSpec("alpha", "α (ARCH effect)", 1e-9, 1e-3, 3e-6, 1e-7, "%.2e",
                  _HESTON_NANDI_DESCRIPTIONS["alpha"]),
        ParamSpec("beta", "β (GARCH persistence)", 0.0, 0.999, 0.82, 0.01, "%.3f",
                  _HESTON_NANDI_DESCRIPTIONS["beta"]),
        ParamSpec("gamma", "γ (risk-neutral leverage)", 0.0, 1000.0, 180.0, 5.0, "%.1f",
                  _HESTON_NANDI_DESCRIPTIONS["gamma"]),
        ParamSpec("h0", "h₀ (initial variance)", 1e-7, 1e-2, 1.2e-4, 1e-6, "%.2e",
                  _HESTON_NANDI_DESCRIPTIONS["h0"]),
    ),
)

_NGARCH_Q_DESCRIPTIONS = {
    "omega": "Constant term ω of the risk-neutral variance recursion (per period).",
    "alpha": "ARCH coefficient — sensitivity of variance to the squared shock.",
    "beta": "GARCH persistence of past variance.",
    "gamma": (
        "Risk-neutral asymmetry γ* = γ + λ. Unlike Heston-Nandi (O(100)) it is "
        "O(1) here, entering as (z − γ)²; stationarity requires β + α(1+γ²) < 1."
    ),
    "h0": "Initial conditional variance h₁ (per period).",
}

NGARCH_Q_SPEC = ModelSpec(
    key="ngarch_q",
    display_name="Duan NGARCH (risk-neutral)",
    params=(
        # omega/alpha/beta sliders mirror the P-family ranges: under Duan's
        # LRNVR the one-step-ahead conditional variance is invariant P -> Q,
        # so the same numerical box applies to both measures. Only γ is
        # Q-specific (γ* = γ + λ).
        ParamSpec("omega", "ω (variance intercept)", 1e-6, 1e-3, 2e-6, 1e-7, "%.2e",
                  _NGARCH_Q_DESCRIPTIONS["omega"]),
        ParamSpec("alpha", "α (ARCH effect)", 0.0, 0.5, 0.04, 0.005, "%.3f",
                  _NGARCH_Q_DESCRIPTIONS["alpha"]),
        ParamSpec("beta", "β (GARCH persistence)", 0.0, 0.999, 0.80, 0.01, "%.3f",
                  _NGARCH_Q_DESCRIPTIONS["beta"]),
        ParamSpec("gamma", "γ (risk-neutral asymmetry)", 0.0, 4.0, 0.8, 0.05, "%.3f",
                  _NGARCH_Q_DESCRIPTIONS["gamma"]),
        ParamSpec("h0", "h₀ (initial variance)", 1e-7, 1e-2, 4e-5, 1e-6, "%.2e",
                  _NGARCH_Q_DESCRIPTIONS["h0"]),
    ),
)

# Sibling risk-neutral GARCH-Q surface models (MC-priced, like ngarch_q). All
# per-period parameters. garch_q is symmetric (no γ); gjr_q carries leverage.
GARCH_Q_SPEC = ModelSpec(
    key="garch_q",
    display_name="GARCH (risk-neutral)",
    params=(
        # omega/alpha/beta sliders aligned with the P-family — LRNVR-invariant.
        ParamSpec("omega", "ω (variance intercept)", 1e-6, 1e-3, 3e-6, 1e-7, "%.2e",
                  "Constant term ω of the risk-neutral variance recursion (per period)."),
        ParamSpec("alpha", "α (ARCH effect)", 0.0, 0.5, 0.06, 0.005, "%.3f",
                  "ARCH coefficient — sensitivity of variance to the squared shock."),
        ParamSpec("beta", "β (GARCH persistence)", 0.0, 0.999, 0.90, 0.01, "%.3f",
                  "GARCH persistence of past variance. Stationarity needs β + α < 1."),
        ParamSpec("h0", "h₀ (initial variance)", 1e-7, 1e-2, 4e-5, 1e-6, "%.2e",
                  "Initial conditional variance h₁ (per period)."),
    ),
)

GJR_Q_SPEC = ModelSpec(
    key="gjr_q",
    display_name="GJR-GARCH (risk-neutral)",
    params=(
        # omega/alpha/beta sliders aligned with the P-family — LRNVR-invariant.
        # Only γ stays Q-specific (γ* = γ + λ, positive on this side).
        ParamSpec("omega", "ω (variance intercept)", 1e-6, 1e-3, 2.5e-6, 1e-7, "%.2e",
                  "Constant term ω of the risk-neutral variance recursion (per period)."),
        ParamSpec("alpha", "α (ARCH effect)", 0.0, 0.5, 0.04, 0.005, "%.3f",
                  "Symmetric ARCH coefficient (applies to every shock)."),
        ParamSpec("beta", "β (GARCH persistence)", 0.0, 0.999, 0.78, 0.01, "%.3f",
                  "GARCH persistence of past variance. With α and γ, stationarity "
                  "needs β + α + γ/2 < 1 (default keeps it ≈ 0.97)."),
        ParamSpec("gamma", "γ (risk-neutral leverage)", 0.0, 2.0, 0.3, 0.02, "%.3f",
                  "Risk-neutral leverage-indicator coefficient (γ* = γ + λ): extra "
                  "ARCH on negative shocks → skew. Stationarity needs β + α + γ/2 < 1."),
        ParamSpec("h0", "h₀ (initial variance)", 1e-7, 1e-2, 4e-5, 1e-6, "%.2e",
                  "Initial conditional variance h₁ (per period)."),
    ),
)

GARCH_SPEC = ModelSpec(
    key="garch",
    display_name="GARCH(1,1)",
    params=(
        ParamSpec("omega", "ω (intercept)", 1e-6, 1e-3, 2e-6, 1e-7, "%.2e",
                  _GARCH_DESCRIPTIONS["omega"]),
        ParamSpec("alpha", "α (ARCH effect)", 0.0, 0.5, 0.08, 0.01, "%.3f",
                  _GARCH_DESCRIPTIONS["alpha"]),
        ParamSpec("beta", "β (GARCH persistence)", 0.0, 0.999, 0.90, 0.01, "%.3f",
                  _GARCH_DESCRIPTIONS["beta"]),
    ),
)

NGARCH_SPEC = ModelSpec(
    key="ngarch",
    display_name="NGARCH",
    params=(
        ParamSpec("omega", "ω", 1e-6, 1e-3, 2e-6, 1e-7, "%.2e",
                  _GARCH_DESCRIPTIONS["omega"]),
        ParamSpec("alpha", "α", 0.0, 0.5, 0.08, 0.01, "%.3f",
                  _GARCH_DESCRIPTIONS["alpha"]),
        ParamSpec("beta", "β", 0.0, 0.999, 0.88, 0.01, "%.3f",
                  _GARCH_DESCRIPTIONS["beta"]),
        ParamSpec("gamma", "γ (leverage, physical)", -2.0, 2.0, 0.5, 0.05, "%.3f",
                  "Physical-measure (P) leverage — negative shocks raise variance "
                  "more when γ > 0. Under risk-neutral pricing it maps to "
                  "γ* = γ + λ (the app's forward map assumes λ = 0)."),
    ),
)

GJR_SPEC = ModelSpec(
    key="gjr_garch",
    display_name="GJR-GARCH",
    params=(
        ParamSpec("omega", "ω", 1e-6, 1e-3, 2e-6, 1e-7, "%.2e",
                  _GARCH_DESCRIPTIONS["omega"]),
        ParamSpec("alpha", "α", 0.0, 0.5, 0.05, 0.01, "%.3f",
                  _GARCH_DESCRIPTIONS["alpha"]),
        ParamSpec("beta", "β", 0.0, 0.999, 0.88, 0.01, "%.3f",
                  _GARCH_DESCRIPTIONS["beta"]),
        ParamSpec("gamma", "γ (leverage, physical)", 0.0, 0.5, 0.04, 0.01, "%.3f",
                  "Physical-measure (P) leverage-indicator coefficient "
                  "(non-negative). Risk-neutral pricing uses γ* = γ + λ "
                  "(forward map assumes λ = 0)."),
    ),
)

IV_SPEC = ModelSpec(
    key="iv_gbm",
    display_name="GBM (IV)",
    params=(
        ParamSpec("sigma", "σ (volatility)", 0.05, 1.0, 0.20, 0.01, "%.3f",
                  "Constant Black-Scholes implied volatility."),
    ),
)

REGISTRY: dict[str, ModelSpec] = {
    "heston": HESTON_SPEC,
    "merton": MERTON_SPEC,
    "bates": BATES_SPEC,
    "heston_nandi": HESTON_NANDI_SPEC,
    "ngarch_q": NGARCH_Q_SPEC,
    "garch_q": GARCH_Q_SPEC,
    "gjr_q": GJR_Q_SPEC,
    "garch": GARCH_SPEC,
    "ngarch": NGARCH_SPEC,
    "gjr_garch": GJR_SPEC,
    "iv_gbm": IV_SPEC,
}


def get_spec(model_key: str) -> ModelSpec:
    if model_key not in REGISTRY:
        raise KeyError(f"Unknown model '{model_key}'. Available: {list(REGISTRY)}")
    return REGISTRY[model_key]


def supported_solvers(model_key: str) -> tuple[str, ...]:
    """Solvers the user can pick for this model."""
    if model_key in ("garch", "ngarch", "gjr_garch"):
        # GARCH MLE is scalar — LM is rejected
        return ("L-BFGS-B", "DE", "NM")
    if model_key in ("ngarch_q", "garch_q", "gjr_q"):
        # Nonaffine MC surface fit — scalar objective, no analytical Jacobian,
        # so LM-JAX is rejected (it needs a residual vector + Jacobian).
        return ("DE", "NM", "L-BFGS-B")
    if model_key == "iv_gbm":
        # IV is a closed-form inversion, optimisers don't apply
        return ()
    return ("LM-JAX", "DE", "NM", "L-BFGS-B")


def supported_objectives(model_key: str) -> tuple[str, ...]:
    """Objective functions the user can pick for this model.

    Surface models (Heston/Bates/Merton) support the full catalogue:
    price MSE, vega-weighted, spread-weighted, relative/log, Huber, and
    IV MSE (the latter falling back to vega_weighted under LM-JAX).
    GARCH-family calibration is MLE-on-returns and does not consume an
    objective function in the same sense — we return an empty tuple so
    the UI masks the selector.
    """
    if model_key in ("garch", "ngarch", "gjr_garch"):
        return ()
    if model_key == "iv_gbm":
        # IV-GBM is a closed-form IV inversion; no objective to choose.
        return ()
    return (
        "price_mse",
        "iv_mse",
        "vega_weighted",
        "spread_weighted",
        "relative",
        "huber",
    )
