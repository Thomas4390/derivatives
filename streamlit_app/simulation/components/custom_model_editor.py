"""
Custom Model Editor — Define, validate, and register user-defined stochastic models.

Provides a step-by-step workflow:
1. Code editor with syntax highlighting
2. Validation suite
3. Registration
"""

import streamlit as st
from code_editor import code_editor
from services.custom_model_service import (
    compile_and_validate,
    register_custom_model,
    unregister_custom_model,
)

# ── Template code ────────────────────────────────────────────────────────

_TEMPLATES = {
    "GBM (Simple)": '''\
import numpy as np
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability

class MyGBM(Model):
    """Geometric Brownian Motion: dS = (r-q)*S*dt + sigma*S*dW.
    The simplest continuous-time model for asset prices.
    Produces log-normal terminal distribution."""

    PARAMETER_SPECS = [
        {"name": "sigma", "display_name": "Volatility (sigma)",
         "default": 0.20, "min_value": 0.01, "max_value": 1.0,
         "step": 0.01, "description": "Annualized volatility (0.20 = 20%)"},
    ]

    EQUATION_LATEX = {
        "main": r"dS = (r - q) \\, S \\, dt + \\sigma \\, S \\, dW",
        "mc": r"S_{t+\\Delta t} = S_t \\exp\\!\\left[\\left(r - q - \\tfrac{1}{2}\\sigma^2\\right)\\!\\Delta t + \\sigma\\sqrt{\\Delta t}\\, Z\\right]",
    }

    def __init__(self, sigma=0.20):
        self._sigma = sigma

    @property
    def name(self):
        return "Custom GBM"

    @property
    def supported_engines(self):
        return [PricingCapability.MONTE_CARLO]

    def get_parameters(self):
        return {"sigma": self._sigma}

    def drift(self, s, v, t, r, q):
        return (r - q) * s

    def diffusion(self, s, v, t):
        return self._sigma * s
''',

    "CEV (Constant Elasticity of Variance)": '''\
import numpy as np
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability

class CEVModel(Model):
    """Constant Elasticity of Variance (CEV) model.
    dS = (r-q)*S*dt + sigma * S^gamma * dW

    gamma controls how volatility scales with price level:
      gamma=1: GBM (constant relative vol)
      gamma=0.5: square-root diffusion
      gamma=0: normal model (constant absolute vol)

    Effective vol at S0: sigma * S0^(gamma-1).
    Default: sigma=0.65, gamma=0.75 gives ~20% vol at S0=100."""

    EQUATION_LATEX = {
        "main": r"dS = (r - q) \\, S \\, dt + \\sigma \\, S^{\\gamma} \\, dW",
        "vol": r"\\text{Effective vol}(S) = \\sigma \\, S^{\\gamma - 1} \\quad (\\gamma=1 \\Rightarrow \\text{GBM}, \\; \\gamma=0 \\Rightarrow \\text{Normal})",
        "mc": r"S_{t+\\Delta t} = S_t + (r - q) S_t \\Delta t + \\sigma S_t^{\\gamma} \\sqrt{\\Delta t}\\, Z",
    }

    PARAMETER_SPECS = [
        {"name": "sigma", "display_name": "Volatility (sigma)",
         "default": 0.65, "min_value": 0.01, "max_value": 5.0,
         "step": 0.01, "description": "CEV vol param (effective vol = sigma * S^(gamma-1))"},
        {"name": "gamma", "display_name": "Elasticity (gamma)",
         "default": 0.75, "min_value": 0.0, "max_value": 1.5,
         "step": 0.05, "description": "Elasticity of variance (1=GBM, 0.5=sqrt, 0=normal)"},
    ]

    def __init__(self, sigma=0.65, gamma=0.75):
        self._sigma = sigma
        self._gamma = gamma

    @property
    def name(self):
        return "CEV Model"

    @property
    def supported_engines(self):
        return [PricingCapability.MONTE_CARLO]

    def get_parameters(self):
        return {"sigma": self._sigma, "gamma": self._gamma}

    def drift(self, s, v, t, r, q):
        return (r - q) * s

    def diffusion(self, s, v, t):
        return self._sigma * np.power(np.maximum(s, 1e-10), self._gamma)
''',

    "Mean-Reverting (Ornstein-Uhlenbeck)": '''\
import numpy as np
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability

class OUModel(Model):
    """Ornstein-Uhlenbeck mean-reverting model for prices.
    dS = kappa*(theta - S)*dt + sigma*S*dW

    Price reverts to theta with speed kappa.
    Useful for commodities, pairs trading, interest rates.
    Default: reverts to 100 with ~20% vol and half-life ~4 months."""

    EQUATION_LATEX = {
        "main": r"dS = \\kappa \\, (\\theta - S) \\, dt + \\sigma \\, S \\, dW",
        "vol": r"t_{1/2} = \\frac{\\ln 2}{\\kappa} \\quad \\text{(half-life of mean reversion)}",
        "mc": r"S_{t+\\Delta t} = S_t + \\kappa(\\theta - S_t)\\Delta t + \\sigma S_t \\sqrt{\\Delta t}\\, Z",
    }

    PARAMETER_SPECS = [
        {"name": "kappa", "display_name": "Mean Reversion (kappa)",
         "default": 2.0, "min_value": 0.1, "max_value": 10.0,
         "step": 0.1, "description": "Speed of mean reversion (half-life = ln(2)/kappa)"},
        {"name": "theta", "display_name": "Long-Run Level (theta)",
         "default": 100.0, "min_value": 10.0, "max_value": 500.0,
         "step": 1.0, "description": "Long-run equilibrium price level"},
        {"name": "sigma", "display_name": "Volatility (sigma)",
         "default": 0.20, "min_value": 0.01, "max_value": 1.0,
         "step": 0.01, "description": "Annualized volatility (0.20 = 20%)"},
    ]

    def __init__(self, kappa=2.0, theta=100.0, sigma=0.20):
        self._kappa = kappa
        self._theta = theta
        self._sigma = sigma

    @property
    def name(self):
        return "OU Mean-Reverting"

    @property
    def supported_engines(self):
        return [PricingCapability.MONTE_CARLO]

    def get_parameters(self):
        return {"kappa": self._kappa, "theta": self._theta, "sigma": self._sigma}

    def drift(self, s, v, t, r, q):
        return self._kappa * (self._theta - s)

    def diffusion(self, s, v, t):
        return self._sigma * s
''',

    "Merton Jump-Diffusion": '''\
import numpy as np
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability

class MertonJD(Model):
    """Merton (1976) Jump-Diffusion Model.
    dS/S = (r - q - lambda*k)*dt + sigma*dW + (exp(J) - 1)*dN

    Combines GBM diffusion with compound Poisson jumps:
      - dN ~ Poisson(lambda*dt): jump arrival
      - J ~ N(mu_j, sigma_j^2): log-jump size
      - k = E[exp(J)-1]: jump compensator

    Defines jump() for MC and characteristic_function() for FFT.
    Default: ~18% diffusive vol + ~0.5 jumps/year = ~22% total vol."""

    EQUATION_LATEX = {
        "main": r"\\frac{dS}{S} = (r - q - \\lambda k) \\, dt + \\sigma \\, dW + (e^J - 1) \\, dN",
        "jump": r"dN \\sim \\mathrm{Poisson}(\\lambda \\, dt), \\quad J \\sim \\mathcal{N}(\\mu_J, \\sigma_J^2), \\quad k = e^{\\mu_J + \\frac{1}{2}\\sigma_J^2} - 1",
        "cf": r"\\varphi(u) = \\varphi_{\\text{GBM}}(u) \\cdot \\exp\\!\\left[\\lambda T\\!\\left(e^{iu\\mu_J - \\frac{1}{2}\\sigma_J^2 u^2} - 1\\right)\\right]",
        "mc": r"S_{t+\\Delta t} = S_t \\exp\\!\\left[(r-q-\\lambda k - \\tfrac{\\sigma^2}{2})\\Delta t + \\sigma\\sqrt{\\Delta t}\\, Z\\right] \\cdot \\prod_{j=1}^{N_t} e^{J_j}",
    }

    PARAMETER_SPECS = [
        {"name": "sigma", "display_name": "Diffusion Vol (sigma)",
         "default": 0.18, "min_value": 0.01, "max_value": 1.0,
         "step": 0.01, "description": "Diffusive volatility (continuous part)"},
        {"name": "lambda_j", "display_name": "Jump Intensity (lambda)",
         "default": 0.5, "min_value": 0.0, "max_value": 5.0,
         "step": 0.1, "description": "Expected number of jumps per year"},
        {"name": "mu_j", "display_name": "Mean Log-Jump (mu_j)",
         "default": -0.10, "min_value": -0.5, "max_value": 0.5,
         "step": 0.01, "description": "Mean of log-jump size (negative = downward jumps)"},
        {"name": "sigma_j", "display_name": "Jump Vol (sigma_j)",
         "default": 0.15, "min_value": 0.01, "max_value": 0.5,
         "step": 0.01, "description": "Std dev of log-jump size"},
    ]

    def __init__(self, sigma=0.18, lambda_j=0.5, mu_j=-0.10, sigma_j=0.15):
        self._sigma = sigma
        self._lambda_j = lambda_j
        self._mu_j = mu_j
        self._sigma_j = sigma_j
        # Jump compensator: k = E[exp(J) - 1]
        self._k = np.exp(mu_j + 0.5 * sigma_j**2) - 1

    @property
    def name(self):
        return "Merton Jump-Diffusion"

    @property
    def supported_engines(self):
        return [PricingCapability.MONTE_CARLO, PricingCapability.FFT]

    def get_parameters(self):
        return {"sigma": self._sigma, "lambda_j": self._lambda_j,
                "mu_j": self._mu_j, "sigma_j": self._sigma_j}

    # ── Monte Carlo: drift + diffusion + jump ──
    def drift(self, s, v, t, r, q):
        # Compensated drift: subtract jump premium
        return (r - q - self._lambda_j * self._k) * s

    def diffusion(self, s, v, t):
        return self._sigma * s

    def jump(self, s, dt):
        """Compound Poisson jump at each timestep.
        Returns additive jump increment: S * sum(exp(J_i) - 1)."""
        n = len(s) if hasattr(s, \'__len__\') else 1
        n_jumps = np.random.poisson(self._lambda_j * dt, n)
        total = np.zeros(n)
        max_j = n_jumps.max() if n > 0 else 0
        for k in range(1, max_j + 1):
            mask = n_jumps >= k
            j = np.random.normal(self._mu_j, self._sigma_j, mask.sum())
            total[mask] += np.exp(j) - 1
        return s * total

    # ── Characteristic function for FFT ──
    def characteristic_function(self, u, s0, t, r, q=0.0):
        """Merton CF: GBM CF * jump CF."""
        sigma2 = self._sigma**2
        x = np.log(s0)

        # GBM part (compensated for jumps)
        drift_adj = (r - q - 0.5 * sigma2 - self._lambda_j * self._k) * t
        gbm_cf = np.exp(1j * u * (x + drift_adj) - 0.5 * sigma2 * t * u**2)

        # Jump part: exp(lambda*t * (E[exp(iuJ)] - 1))
        jump_cf_inner = np.exp(1j * u * self._mu_j
                               - 0.5 * self._sigma_j**2 * u**2) - 1
        jump_cf = np.exp(self._lambda_j * t * jump_cf_inner)

        return gbm_cf * jump_cf
''',

    "Square-Root / CIR Process": '''\
import numpy as np
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability

class CIRModel(Model):
    """Cox-Ingersoll-Ross (CIR) square-root process.
    dS = kappa*(theta - S)*dt + sigma*sqrt(S)*dW

    Mean-reverting with volatility proportional to sqrt(S):
      - Low prices -> low vol (prevents negative prices)
      - Feller condition: 2*kappa*theta > sigma^2 ensures S > 0

    Effective vol at S: sigma/sqrt(S).
    Default: sigma=2.0 gives ~20% vol at S0=100."""

    EQUATION_LATEX = {
        "main": r"dS = \\kappa \\, (\\theta - S) \\, dt + \\sigma \\sqrt{S} \\, dW",
        "vol": r"\\text{Feller condition: } 2\\kappa\\theta > \\sigma^2 \\; \\Rightarrow \\; S_t > 0 \\;\\; \\forall t",
        "mc": r"S_{t+\\Delta t} = S_t + \\kappa(\\theta - S_t)\\Delta t + \\sigma\\sqrt{\\max(S_t, 0) \\cdot \\Delta t}\\, Z",
    }

    PARAMETER_SPECS = [
        {"name": "kappa", "display_name": "Mean Reversion (kappa)",
         "default": 2.0, "min_value": 0.1, "max_value": 10.0,
         "step": 0.1, "description": "Speed of mean reversion"},
        {"name": "theta", "display_name": "Long-Run Level (theta)",
         "default": 100.0, "min_value": 10.0, "max_value": 500.0,
         "step": 1.0, "description": "Long-run equilibrium level"},
        {"name": "sigma", "display_name": "Vol of sqrt(S) (sigma)",
         "default": 2.0, "min_value": 0.1, "max_value": 10.0,
         "step": 0.1, "description": "Diffusion param (effective vol = sigma/sqrt(S))"},
    ]

    def __init__(self, kappa=2.0, theta=100.0, sigma=2.0):
        self._kappa = kappa
        self._theta = theta
        self._sigma = sigma

    @property
    def name(self):
        return "CIR Square-Root"

    @property
    def supported_engines(self):
        return [PricingCapability.MONTE_CARLO]

    def get_parameters(self):
        return {"kappa": self._kappa, "theta": self._theta, "sigma": self._sigma}

    def drift(self, s, v, t, r, q):
        return self._kappa * (self._theta - s)

    def diffusion(self, s, v, t):
        return self._sigma * np.sqrt(np.maximum(s, 1e-10))
''',

    "Exponential OU (Schwartz)": '''\
import numpy as np
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability

class ExpOUModel(Model):
    """Schwartz (1997) Exponential Ornstein-Uhlenbeck model.
    ln(S) follows OU: d(lnS) = kappa*(ln(theta) - lnS)*dt + sigma*dW

    Equivalently: dS = S*[kappa*(ln(theta) - lnS) + 0.5*sigma^2]*dt + sigma*S*dW

    Price mean-reverts in log-space to theta. Unlike additive OU,
    this guarantees positive prices. Widely used for commodities.
    Default: reverts to $100 with 20% vol."""

    EQUATION_LATEX = {
        "main": r"d(\\ln S) = \\kappa \\, (\\ln \\theta - \\ln S) \\, dt + \\sigma \\, dW",
        "vol": r"dS = S\\!\\left[\\kappa(\\ln\\theta - \\ln S) + \\tfrac{1}{2}\\sigma^2\\right] dt + \\sigma \\, S \\, dW",
        "mc": r"\\ln S_{t+\\Delta t} = \\ln S_t + \\kappa(\\ln\\theta - \\ln S_t)\\Delta t + \\sigma\\sqrt{\\Delta t}\\, Z",
    }

    PARAMETER_SPECS = [
        {"name": "kappa", "display_name": "Mean Reversion (kappa)",
         "default": 1.0, "min_value": 0.1, "max_value": 10.0,
         "step": 0.1, "description": "Speed of log-price mean reversion"},
        {"name": "theta", "display_name": "Long-Run Price (theta)",
         "default": 100.0, "min_value": 10.0, "max_value": 500.0,
         "step": 1.0, "description": "Long-run equilibrium price level"},
        {"name": "sigma", "display_name": "Volatility (sigma)",
         "default": 0.20, "min_value": 0.01, "max_value": 1.0,
         "step": 0.01, "description": "Log-price volatility (0.20 = 20%)"},
    ]

    def __init__(self, kappa=1.0, theta=100.0, sigma=0.20):
        self._kappa = kappa
        self._theta = theta
        self._sigma = sigma

    @property
    def name(self):
        return "Exp-OU (Schwartz)"

    @property
    def supported_engines(self):
        return [PricingCapability.MONTE_CARLO]

    def get_parameters(self):
        return {"kappa": self._kappa, "theta": self._theta, "sigma": self._sigma}

    def drift(self, s, v, t, r, q):
        log_s = np.log(np.maximum(s, 1e-10))
        log_theta = np.log(self._theta)
        return s * (self._kappa * (log_theta - log_s) + 0.5 * self._sigma**2)

    def diffusion(self, s, v, t):
        return self._sigma * s
''',

    "Heston-Like (Stochastic Volatility)": '''\
import numpy as np
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability

class HestonLike(Model):
    """Heston-style stochastic volatility model.
    dS = (r - q)*S*dt + sqrt(V)*S*dW_1
    dV = kappa*(theta - V)*dt + xi*sqrt(V)*dW_2
    corr(dW_1, dW_2) = rho

    Variance V follows a CIR process, correlated with price.
    Feller condition 2*kappa*theta > xi^2 ensures V > 0.

    The simulator detects variance_drift/variance_diffusion/get_correlation
    and automatically generates correlated Brownian motions."""

    EQUATION_LATEX = {
        "main": r"dS = (r - q) \\, S \\, dt + \\sqrt{V} \\, S \\, dW_1",
        "vol": r"dV = \\kappa(\\theta - V) \\, dt + \\xi \\sqrt{V} \\, dW_2, \\quad \\mathrm{corr}(dW_1, dW_2) = \\rho",
        "cf": r"\\varphi(u) = \\exp\\!\\bigl(C(u,T) + D(u,T)\\,v_0 + iu\\ln S_0\\bigr)",
        "mc": r"\\begin{aligned} S_{t+\\Delta t} &= S_t \\exp\\!\\left[(r-q-\\tfrac{V_t}{2})\\Delta t + \\sqrt{V_t\\Delta t}\\, Z_1\\right] \\\\ V_{t+\\Delta t} &= V_t + \\kappa(\\theta - V_t)\\Delta t + \\xi\\sqrt{V_t\\Delta t}\\, Z_2 \\end{aligned}",
    }

    PARAMETER_SPECS = [
        {"name": "v0", "display_name": "Initial Variance (V0)",
         "default": 0.04, "min_value": 0.001, "max_value": 0.50,
         "step": 0.005, "description": "Initial variance (0.04 = 20% vol)"},
        {"name": "kappa", "display_name": "Mean Reversion (kappa)",
         "default": 2.0, "min_value": 0.1, "max_value": 10.0,
         "step": 0.1, "description": "Speed of variance mean reversion"},
        {"name": "theta", "display_name": "Long-Run Variance (theta)",
         "default": 0.04, "min_value": 0.001, "max_value": 0.50,
         "step": 0.005, "description": "Long-run variance level (0.04 = 20% vol)"},
        {"name": "xi", "display_name": "Vol of Vol (xi)",
         "default": 0.3, "min_value": 0.01, "max_value": 1.5,
         "step": 0.01, "description": "Volatility of variance (vol-of-vol)"},
        {"name": "rho", "display_name": "Correlation (rho)",
         "default": -0.7, "min_value": -0.99, "max_value": 0.99,
         "step": 0.01, "description": "Price-variance correlation (negative = leverage)"},
    ]

    def __init__(self, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7):
        self._v0 = v0
        self._kappa = kappa
        self._theta = theta
        self._xi = xi
        self._rho = rho

    @property
    def name(self):
        return "Heston-Like Stoch Vol"

    @property
    def supported_engines(self):
        return [PricingCapability.MONTE_CARLO, PricingCapability.FFT]

    def get_parameters(self):
        return {"v0": self._v0, "kappa": self._kappa,
                "theta": self._theta, "xi": self._xi, "rho": self._rho}

    # ── Price SDE coefficients ──
    def drift(self, s, v, t, r, q):
        return (r - q) * s

    def diffusion(self, s, v, t):
        return np.sqrt(np.maximum(v, 1e-10)) * s

    # ── Variance SDE coefficients (detected by simulator) ──
    def variance_drift(self, v, s, t):
        return self._kappa * (self._theta - v)

    def variance_diffusion(self, v, s, t):
        return self._xi * np.sqrt(np.maximum(v, 1e-10))

    def get_correlation(self):
        return self._rho

    # ── Characteristic function for FFT pricing ──
    def characteristic_function(self, u, s0, t, r, q=0.0):
        """Heston semi-analytical CF."""
        kappa, theta, xi, rho, v0 = (
            self._kappa, self._theta, self._xi, self._rho, self._v0
        )
        x = np.log(s0)
        d = np.sqrt((rho * xi * 1j * u - kappa) ** 2
                     + xi ** 2 * (1j * u + u ** 2))
        g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)
        C = (r - q) * 1j * u * t + (kappa * theta / xi ** 2) * (
            (kappa - rho * xi * 1j * u - d) * t
            - 2.0 * np.log((1.0 - g * np.exp(-d * t)) / (1.0 - g))
        )
        D = ((kappa - rho * xi * 1j * u - d) / xi ** 2) * (
            (1.0 - np.exp(-d * t)) / (1.0 - g * np.exp(-d * t))
        )
        return np.exp(C + D * v0 + 1j * u * x)
''',

    "GBM with FFT (Characteristic Function)": '''\
import numpy as np
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability

class GBMWithFFT(Model):
    """GBM with both Monte Carlo and FFT pricing.
    dS = (r-q)*S*dt + sigma*S*dW

    This template shows how to implement characteristic_function()
    to enable FFT-based option pricing alongside Monte Carlo.

    phi(u) = E[exp(i*u*ln(S_T))] for the GBM log-price process.
    FFT is much faster than MC for European options."""

    EQUATION_LATEX = {
        "main": r"dS = (r - q) \\, S \\, dt + \\sigma \\, S \\, dW",
        "cf": r"\\varphi(u) = \\exp\\!\\left[iu\\left(\\ln S_0 + (r - q - \\tfrac{1}{2}\\sigma^2)T\\right) - \\tfrac{1}{2}\\sigma^2 T u^2\\right]",
        "mc": r"S_{t+\\Delta t} = S_t \\exp\\!\\left[\\left(r - q - \\tfrac{1}{2}\\sigma^2\\right)\\!\\Delta t + \\sigma\\sqrt{\\Delta t}\\, Z\\right]",
    }

    PARAMETER_SPECS = [
        {"name": "sigma", "display_name": "Volatility (sigma)",
         "default": 0.20, "min_value": 0.01, "max_value": 1.0,
         "step": 0.01, "description": "Annualized volatility (0.20 = 20%)"},
    ]

    def __init__(self, sigma=0.20):
        self._sigma = sigma

    @property
    def name(self):
        return "GBM + FFT"

    @property
    def supported_engines(self):
        return [PricingCapability.MONTE_CARLO, PricingCapability.FFT]

    def get_parameters(self):
        return {"sigma": self._sigma}

    # ── Monte Carlo SDE coefficients ──
    def drift(self, s, v, t, r, q):
        return (r - q) * s

    def diffusion(self, s, v, t):
        return self._sigma * s

    # ── Characteristic function for FFT ──
    def characteristic_function(self, u, s0, t, r, q=0.0):
        """phi(u) = exp(i*u*x + i*u*(r-q-0.5*sigma^2)*t - 0.5*sigma^2*t*u^2)
        where x = ln(s0)."""
        sigma2 = self._sigma ** 2
        x = np.log(s0)
        drift_adj = (r - q - 0.5 * sigma2) * t
        return np.exp(1j * u * (x + drift_adj) - 0.5 * sigma2 * t * u ** 2)
''',
}


# ── Editor configuration ────────────────────────────────────────────────

_EDITOR_BUTTONS = [
    {
        "name": "Copy",
        "feather": "Copy",
        "hasText": True,
        "alwaysOn": True,
        "commands": ["copyAll"],
        "style": {"top": "0.46rem", "right": "0.4rem"},
    },
]


# ── Main component ──────────────────────────────────────────────────────

def render_custom_model_editor():
    """Render the Custom Model tab with editor, validation, and registration."""

    st.markdown("### Define Your Own Stochastic Model")
    st.caption(
        "Write a Python class inheriting from `Model` with `drift()` and `diffusion()` methods. "
        "After validation, the model appears in the sidebar dropdown and works with all tabs."
    )

    # ── Status panel: currently registered model ─────────────────────
    custom = st.session_state.get("custom_model")
    if custom:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.success(f"Active custom model: **{custom['spec'].name}**")
        with c2:
            if st.button("Remove", key="remove_custom", type="secondary"):
                unregister_custom_model()
                if st.session_state.get("selected_model") == "custom":
                    st.session_state.selected_model = "gbm"
                st.rerun()

    st.markdown("---")

    # ── Step 1: Template selector + Code editor ──────────────────────
    st.markdown("**Step 1:** Write your model code")

    # Track which template is selected; reload editor when it changes
    if "custom_editor_version" not in st.session_state:
        st.session_state.custom_editor_version = 0
    if "custom_prev_template" not in st.session_state:
        st.session_state.custom_prev_template = None

    template_name = st.selectbox(
        "Start from template",
        options=list(_TEMPLATES.keys()),
        key="custom_template_select",
        help="Select a template — code loads automatically",
    )

    # Auto-load on template change (or first render)
    if template_name != st.session_state.custom_prev_template:
        st.session_state.custom_editor_code = _TEMPLATES[template_name]
        st.session_state.custom_editor_version += 1
        st.session_state.custom_prev_template = template_name
        st.session_state.pop("custom_validation", None)
        st.rerun()

    editor_key = f"custom_code_editor_{st.session_state.custom_editor_version}"
    response = code_editor(
        st.session_state.custom_editor_code,
        lang="python",
        height=[20, 40],
        buttons=_EDITOR_BUTTONS,
        key=editor_key,
        options={"wrap": True},
    )

    # Capture editor output
    if response and response.get("type") == "submit" and response.get("text"):
        st.session_state.custom_editor_code = response["text"]

    st.markdown("---")

    # ── Step 2: Validate ─────────────────────────────────────────────
    st.markdown("**Step 2:** Validate your model")

    if st.button("Validate Model", key="validate_custom_btn", type="primary"):
        # Get current code from editor response or session state
        code = response.get("text") if (response and response.get("text")) else st.session_state.custom_editor_code
        st.session_state.custom_editor_code = code

        with st.spinner("Running validation suite..."):
            result = compile_and_validate(code)
            st.session_state.custom_validation = result

    # Display validation results
    validation = st.session_state.get("custom_validation")
    if validation:
        for test in validation.tests:
            if test.passed:
                st.markdown(f"&ensp; :green[PASS] **{test.name}** — {test.message}")
            else:
                st.markdown(f"&ensp; :red[FAIL] **{test.name}** — {test.message}")

        if validation.all_passed:
            st.success("All validation tests passed!")
        elif len(validation.tests) > 0:
            st.error("Some tests failed. Fix the issues above and re-validate.")

    st.markdown("---")

    # ── Step 3: Register ─────────────────────────────────────────────
    st.markdown("**Step 3:** Register your model")

    can_register = validation is not None and validation.all_passed
    if st.button(
        "Register Model",
        key="register_custom_btn",
        type="primary",
        disabled=not can_register,
    ):
        code = st.session_state.custom_editor_code
        register_custom_model(validation.model_class, code)
        st.session_state.selected_model = "custom"
        st.session_state.custom_just_registered = validation.model_class.__name__
        st.rerun()

    # Show registration confirmation (survives rerun)
    just_registered = st.session_state.pop("custom_just_registered", None)
    if just_registered:
        st.success(f"Model **{just_registered}** registered and selected in the sidebar.")

    if not can_register:
        st.caption("Validate your model first. All tests must pass before registration.")
