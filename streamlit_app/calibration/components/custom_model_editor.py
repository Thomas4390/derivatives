"""
Custom-model editor — define, validate, and register a user model.

A three-step workflow rendered inside the "Custom Model" calibration tab:

1. Write a Python class inheriting from ``Model`` (start from a template).
2. Validate it against the interface / SDE / characteristic-function suite.
3. Register it so it can be calibrated to the current option surface.

Every template is a **risk-neutral** asset model (martingale drift ``(r - q) S``)
so the generated price surface is economically meaningful for calibration. Each
advertises a Monte-Carlo route (``drift`` / ``diffusion``) and, where a closed
form exists, an FFT route (``characteristic_function``).
"""

from __future__ import annotations

import streamlit as st
from code_editor import code_editor

from services.custom_model_service import (
    compile_and_validate,
    is_registered,
    register_custom_model,
    unregister_custom_model,
)

# ── Template code ────────────────────────────────────────────────────────

_TEMPLATES: dict[str, str] = {
    "GBM (Black–Scholes)": '''\
import numpy as np
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability

class MyGBM(Model):
    """Geometric Brownian Motion: dS = (r-q)*S*dt + sigma*S*dW.
    The Black-Scholes constant-volatility benchmark — a flat IV surface.
    Calibrating it to a smile recovers an average level (intentional baseline)."""

    PARAMETER_SPECS = [
        {
            "name": "sigma",
            "display_name": "Volatility (sigma)",
            "default": 0.20,
            "min_value": 0.05,
            "max_value": 1.0,
            "step": 0.01,
            "description": "Annualized volatility (0.20 = 20%)",
        },
    ]

    EQUATION_LATEX = {
        "main": r"dS = (r - q) \\, S \\, dt + \\sigma \\, S \\, dW",
        "mc": r"S_{t+\\Delta t} = S_t \\exp\\!\\left[\\left(r - q - \\tfrac{1}{2}\\sigma^2\\right)\\!\\Delta t + \\sigma\\sqrt{\\Delta t}\\, Z\\right]",
    }

    def __init__(self, sigma: float = 0.20) -> None:
        self._sigma = sigma

    @property
    def name(self) -> str:
        return "Custom GBM"

    @property
    def supported_engines(self) -> list[PricingCapability]:
        return [PricingCapability.MONTE_CARLO]

    def get_parameters(self) -> dict[str, float]:
        return {"sigma": self._sigma}

    def drift(self, s: float, v: float, t: float, r: float, q: float) -> float:
        # Risk-neutral drift of the spot under Q: (r - q) S.
        return (r - q) * s

    def diffusion(self, s: float, v: float, t: float) -> float:
        # Constant-volatility diffusion: sigma S (the coefficient of dW).
        return self._sigma * s
''',
    "GBM + FFT (characteristic function)": '''\
import numpy as np
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability

class GBMWithFFT(Model):
    """GBM priced by FFT via its characteristic function (fast, exact).
    dS = (r-q)*S*dt + sigma*S*dW

    phi(u) = E[exp(i*u*ln(S_T))] for the GBM log-price process. Advertising
    FFT makes calibration use the closed-form surface pricer instead of MC,
    so a flat-vol fit converges in a few evaluations."""

    EQUATION_LATEX = {
        "main": r"dS = (r - q) \\, S \\, dt + \\sigma \\, S \\, dW",
        "cf": r"\\varphi(u) = \\exp\\!\\left[iu\\left(\\ln S_0 + (r - q - \\tfrac{1}{2}\\sigma^2)T\\right) - \\tfrac{1}{2}\\sigma^2 T u^2\\right]",
    }

    PARAMETER_SPECS = [
        {
            "name": "sigma",
            "display_name": "Volatility (sigma)",
            "default": 0.20,
            "min_value": 0.05,
            "max_value": 1.0,
            "step": 0.01,
            "description": "Annualized volatility (0.20 = 20%)",
        },
    ]

    def __init__(self, sigma: float = 0.20) -> None:
        self._sigma = sigma

    @property
    def name(self) -> str:
        return "GBM + FFT"

    @property
    def supported_engines(self) -> list[PricingCapability]:
        return [PricingCapability.MONTE_CARLO, PricingCapability.FFT]

    def get_parameters(self) -> dict[str, float]:
        return {"sigma": self._sigma}

    def drift(self, s: float, v: float, t: float, r: float, q: float) -> float:
        # Risk-neutral drift: (r - q) S (used only by the MC fallback route).
        return (r - q) * s

    def diffusion(self, s: float, v: float, t: float) -> float:
        # Constant-volatility diffusion: sigma S.
        return self._sigma * s

    def characteristic_function(
        self, u: complex, s0: float, t: float, r: float, q: float = 0.0
    ) -> complex:
        # phi(u) = E[exp(i u ln S_T)] in closed form: under Q the log-price is
        # Normal(ln S0 + (r - q - sigma^2/2) T, sigma^2 T).
        sigma2 = self._sigma ** 2
        x = np.log(s0)
        drift_adj = (r - q - 0.5 * sigma2) * t
        return np.exp(1j * u * (x + drift_adj) - 0.5 * sigma2 * t * u ** 2)
''',
    "CEV (local volatility)": '''\
import numpy as np
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability

class CEVModel(Model):
    """Constant Elasticity of Variance: dS = (r-q)*S*dt + sigma*S^gamma*dW.

    gamma controls how local vol scales with the price level:
      gamma=1: GBM (flat smile); gamma<1: a downward skew (equity-like).
    Effective vol at S0: sigma * S0^(gamma-1).
    Default sigma=0.65, gamma=0.75 gives ~20% vol at S0=100."""

    EQUATION_LATEX = {
        "main": r"dS = (r - q) \\, S \\, dt + \\sigma \\, S^{\\gamma} \\, dW",
        "vol": r"\\text{Effective vol}(S) = \\sigma \\, S^{\\gamma - 1} \\quad (\\gamma=1 \\Rightarrow \\text{GBM})",
        "mc": r"S_{t+\\Delta t} = S_t + (r - q) S_t \\Delta t + \\sigma S_t^{\\gamma} \\sqrt{\\Delta t}\\, Z",
    }

    PARAMETER_SPECS = [
        {
            "name": "sigma",
            "display_name": "Volatility (sigma)",
            "default": 0.65,
            "min_value": 0.05,
            "max_value": 5.0,
            "step": 0.01,
            "description": "CEV vol param (effective vol = sigma * S^(gamma-1))",
        },
        {
            "name": "gamma",
            "display_name": "Elasticity (gamma)",
            "default": 0.75,
            "min_value": 0.0,
            "max_value": 1.5,
            "step": 0.05,
            "description": "Elasticity of variance (1=GBM, 0.5=sqrt, 0=normal)",
        },
    ]

    def __init__(self, sigma: float = 0.65, gamma: float = 0.75) -> None:
        self._sigma = sigma
        self._gamma = gamma

    @property
    def name(self) -> str:
        return "CEV Model"

    @property
    def supported_engines(self) -> list[PricingCapability]:
        return [PricingCapability.MONTE_CARLO]

    def get_parameters(self) -> dict[str, float]:
        return {"sigma": self._sigma, "gamma": self._gamma}

    def drift(self, s: float, v: float, t: float, r: float, q: float) -> float:
        # Risk-neutral drift: (r - q) S. The skew comes from diffusion, not here.
        return (r - q) * s

    def diffusion(self, s: float, v: float, t: float) -> float:
        # Local volatility sigma*S^gamma. Floor S at a tiny positive value so the
        # fractional power stays real if an Euler step overshoots below zero.
        return self._sigma * np.power(np.maximum(s, 1e-10), self._gamma)
''',
    "Merton jump-diffusion (MC + FFT)": '''\
import numpy as np
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability

class MertonJD(Model):
    """Merton (1976) jump-diffusion: GBM diffusion + compound-Poisson jumps.
    dS/S = (r - q - lambda*k)*dt + sigma*dW + (exp(J) - 1)*dN

      dN ~ Poisson(lambda*dt); J ~ N(alpha_j, sigma_j^2); k = E[exp(J)-1].

    Defines characteristic_function() (FFT route, used for calibration) and
    jump() (Monte-Carlo route). Default ~18% diffusion + ~0.5 jumps/yr."""

    EQUATION_LATEX = {
        "main": r"\\frac{dS}{S} = (r - q - \\lambda k) \\, dt + \\sigma \\, dW + (e^J - 1) \\, dN",
        "jump": r"dN \\sim \\mathrm{Poisson}(\\lambda \\, dt), \\quad J \\sim \\mathcal{N}(\\alpha_j, \\sigma_j^2), \\quad k = e^{\\alpha_j + \\frac{1}{2}\\sigma_j^2} - 1",
        "cf": r"\\varphi(u) = \\varphi_{\\text{GBM}}(u) \\cdot \\exp\\!\\left[\\lambda T\\!\\left(e^{iu\\alpha_j - \\frac{1}{2}\\sigma_j^2 u^2} - 1\\right)\\right]",
    }

    PARAMETER_SPECS = [
        {
            "name": "sigma",
            "display_name": "Diffusion Vol (sigma)",
            "default": 0.18,
            "min_value": 0.05,
            "max_value": 1.0,
            "step": 0.01,
            "description": "Diffusive volatility (continuous part)",
        },
        {
            "name": "lam",
            "display_name": "Jump Intensity (lambda)",
            "default": 0.5,
            "min_value": 0.0,
            "max_value": 5.0,
            "step": 0.1,
            "description": "Expected number of jumps per year",
        },
        {
            "name": "alpha_j",
            "display_name": "Mean Log-Jump (alpha_j)",
            "default": -0.10,
            "min_value": -0.5,
            "max_value": 0.5,
            "step": 0.01,
            "description": "Mean of log-jump size (negative = downward jumps)",
        },
        {
            "name": "sigma_j",
            "display_name": "Jump Vol (sigma_j)",
            "default": 0.15,
            "min_value": 0.01,
            "max_value": 0.5,
            "step": 0.01,
            "description": "Std dev of log-jump size",
        },
    ]

    def __init__(self, sigma=0.18, lam=0.5, alpha_j=-0.10, sigma_j=0.15) -> None:
        self._sigma = sigma
        self._lambda_j = lam
        self._mu_j = alpha_j
        self._sigma_j = sigma_j
        # Martingale compensator k = E[e^J] - 1, so the discounted spot drifts at
        # the risk-free rate despite the jumps.
        self._k = np.exp(alpha_j + 0.5 * sigma_j ** 2) - 1

    @property
    def name(self) -> str:
        return "Merton Jump-Diffusion"

    @property
    def supported_engines(self) -> list[PricingCapability]:
        return [PricingCapability.MONTE_CARLO, PricingCapability.FFT]

    def get_parameters(self) -> dict[str, float]:
        return {
            "sigma": self._sigma,
            "lam": self._lambda_j,
            "alpha_j": self._mu_j,
            "sigma_j": self._sigma_j,
        }

    def drift(self, s: float, v: float, t: float, r: float, q: float) -> float:
        # Drift compensated by -lambda*k so the discounted spot is a martingale.
        return (r - q - self._lambda_j * self._k) * s

    def diffusion(self, s: float, v: float, t: float) -> float:
        # Continuous (Brownian) part only; the jumps are added by jump().
        return self._sigma * s

    def jump(self, s: np.ndarray, dt: float) -> np.ndarray:
        # Poisson number of jumps per path; summing N i.i.d. log-jumps
        # ~ N(alpha_j, sigma_j^2) gives one N(N*alpha_j, N*sigma_j^2) draw.
        n_jumps = np.random.poisson(self._lambda_j * dt, len(s))
        log_jump = (
            n_jumps * self._mu_j
            + np.sqrt(n_jumps) * self._sigma_j * np.random.standard_normal(len(s))
        )
        return s * (np.exp(log_jump) - 1.0)

    def characteristic_function(
        self, u: complex, s0: float, t: float, r: float, q: float = 0.0
    ) -> complex:
        # By Levy-Khintchine the CF factorizes into the GBM diffusion CF times a
        # compound-Poisson jump CF (independent parts multiply).
        sigma2 = self._sigma ** 2
        x = np.log(s0)
        drift_adj = (r - q - 0.5 * sigma2 - self._lambda_j * self._k) * t
        gbm_cf = np.exp(1j * u * (x + drift_adj) - 0.5 * sigma2 * t * u ** 2)
        jump_inner = np.exp(1j * u * self._mu_j - 0.5 * self._sigma_j ** 2 * u ** 2) - 1
        jump_cf = np.exp(self._lambda_j * t * jump_inner)
        return gbm_cf * jump_cf
''',
    "Heston-like stochastic vol (MC + FFT)": '''\
import numpy as np
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability

class HestonLike(Model):
    """Heston-style stochastic volatility.
    dS = (r-q)*S*dt + sqrt(V)*S*dW_1
    dV = kappa*(theta - V)*dt + alpha*sqrt(V)*dW_2,  corr(dW_1, dW_2) = rho

    Variance follows a CIR process correlated with the price (rho<0 → skew).
    Defines the variance SDE hooks (MC route) and the Heston characteristic
    function (FFT route, used for calibration)."""

    EQUATION_LATEX = {
        "main": r"dS = (r - q) \\, S \\, dt + \\sqrt{V} \\, S \\, dW_1",
        "vol": r"dV = \\kappa(\\theta - V) \\, dt + \\alpha \\sqrt{V} \\, dW_2, \\quad \\mathrm{corr}(dW_1, dW_2) = \\rho",
        "cf": r"\\varphi(u) = \\exp\\!\\bigl(C(u,T) + D(u,T)\\,v_0 + iu\\ln S_0\\bigr)",
    }

    PARAMETER_SPECS = [
        {"name": "v0", "display_name": "Initial Variance (v0)", "default": 0.04,
         "min_value": 0.001, "max_value": 0.50, "step": 0.005,
         "description": "Initial variance (0.04 = 20% vol)"},
        {"name": "kappa", "display_name": "Mean Reversion (kappa)", "default": 2.0,
         "min_value": 0.1, "max_value": 10.0, "step": 0.1,
         "description": "Speed of variance mean reversion"},
        {"name": "theta", "display_name": "Long-Run Variance (theta)", "default": 0.04,
         "min_value": 0.001, "max_value": 0.50, "step": 0.005,
         "description": "Long-run variance level (0.04 = 20% vol)"},
        {"name": "alpha", "display_name": "Vol of Vol (alpha)", "default": 0.3,
         "min_value": 0.01, "max_value": 1.5, "step": 0.01,
         "description": "Volatility of variance (vol-of-vol)"},
        {"name": "rho", "display_name": "Correlation (rho)", "default": -0.7,
         "min_value": -0.99, "max_value": 0.99, "step": 0.01,
         "description": "Price-variance correlation (negative = leverage)"},
    ]

    def __init__(self, v0=0.04, kappa=2.0, theta=0.04, alpha=0.3, rho=-0.7) -> None:
        self._v0 = v0
        self._kappa = kappa
        self._theta = theta
        self._xi = alpha
        self._rho = rho

    @property
    def name(self) -> str:
        return "Heston-Like Stoch Vol"

    @property
    def supported_engines(self) -> list[PricingCapability]:
        return [PricingCapability.MONTE_CARLO, PricingCapability.FFT]

    def get_parameters(self) -> dict[str, float]:
        return {"v0": self._v0, "kappa": self._kappa, "theta": self._theta,
                "alpha": self._xi, "rho": self._rho}

    def drift(self, s: float, v: float, t: float, r: float, q: float) -> float:
        return (r - q) * s

    def diffusion(self, s: float, v: float, t: float) -> float:
        # sqrt(V) * S; floor V at >= 0 since an Euler step can push it negative.
        return np.sqrt(np.maximum(v, 1e-10)) * s

    def variance_drift(self, v: np.ndarray, s: np.ndarray, t: float) -> np.ndarray:
        # CIR mean reversion: the variance is pulled to theta at speed kappa.
        return self._kappa * (self._theta - v)

    def variance_diffusion(self, v: np.ndarray, s: np.ndarray, t: float) -> np.ndarray:
        # Vol-of-vol term alpha * sqrt(V) (same non-negativity floor).
        return self._xi * np.sqrt(np.maximum(v, 1e-10))

    def get_correlation(self) -> float:
        return self._rho

    def characteristic_function(
        self, u: complex, s0: float, t: float, r: float, q: float = 0.0
    ) -> complex:
        # Heston (1993) CF: exp(C(u,T) + D(u,T) v0 + i u ln S0), where C and D
        # solve the Riccati ODEs; d and g below are the standard helper terms.
        kappa, theta, alpha, rho, v0 = (
            self._kappa, self._theta, self._xi, self._rho, self._v0
        )
        x = np.log(s0)
        d = np.sqrt((rho * alpha * 1j * u - kappa) ** 2
                    + alpha ** 2 * (1j * u + u ** 2))
        g = (kappa - rho * alpha * 1j * u - d) / (kappa - rho * alpha * 1j * u + d)
        C = (r - q) * 1j * u * t + (kappa * theta / alpha ** 2) * (
            (kappa - rho * alpha * 1j * u - d) * t
            - 2.0 * np.log((1.0 - g * np.exp(-d * t)) / (1.0 - g))
        )
        D = ((kappa - rho * alpha * 1j * u - d) / alpha ** 2) * (
            (1.0 - np.exp(-d * t)) / (1.0 - g * np.exp(-d * t))
        )
        return np.exp(C + D * v0 + 1j * u * x)
''',
    "Kou double-exponential jumps (MC + FFT)": '''\
import numpy as np
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability

class KouJD(Model):
    """Kou (2002) double-exponential jump-diffusion.
    dS/S = (r - q - lambda*k)*dt + sigma*dW + (e^Y - 1)*dN

    Log-jumps Y are ASYMMETRIC double-exponential: up-jumps ~ Exp(eta_up)
    with prob p, down-jumps ~ -Exp(eta_down) with prob 1-p. Heavier,
    one-sided tails than Merton's Gaussian jumps -> steeper short-dated
    skew. Requires eta_up > 1 so E[e^Y] is finite.
    k = E[e^Y] - 1 = p*eta_up/(eta_up-1) + (1-p)*eta_down/(eta_down+1) - 1."""

    EQUATION_LATEX = {
        "main": r"\\frac{dS}{S} = (r - q - \\lambda k) \\, dt + \\sigma \\, dW + (e^Y - 1) \\, dN",
        "jump": r"f_Y(y) = p\\,\\eta_u e^{-\\eta_u y} \\mathbf{1}_{y \\ge 0} + (1-p)\\,\\eta_d e^{\\eta_d y} \\mathbf{1}_{y < 0}",
        "cf": r"\\varphi(u) = \\varphi_{\\text{GBM}}(u) \\cdot \\exp\\!\\left[\\lambda T\\!\\left(\\frac{p\\,\\eta_u}{\\eta_u - iu} + \\frac{(1-p)\\,\\eta_d}{\\eta_d + iu} - 1\\right)\\right]",
    }

    PARAMETER_SPECS = [
        {
            "name": "sigma",
            "display_name": "Diffusion Vol (sigma)",
            "default": 0.16,
            "min_value": 0.05,
            "max_value": 1.0,
            "step": 0.01,
            "description": "Diffusive volatility (continuous part)",
        },
        {
            "name": "lam",
            "display_name": "Jump Intensity (lambda)",
            "default": 1.0,
            "min_value": 0.0,
            "max_value": 5.0,
            "step": 0.1,
            "description": "Expected number of jumps per year",
        },
        {
            "name": "p_up",
            "display_name": "Up-Jump Probability (p)",
            "default": 0.3,
            "min_value": 0.0,
            "max_value": 1.0,
            "step": 0.05,
            "description": "Probability a jump is upward",
        },
        {
            "name": "eta_up",
            "display_name": "Up-Jump Decay (eta_up)",
            "default": 25.0,
            "min_value": 1.5,
            "max_value": 100.0,
            "step": 0.5,
            "description": "Up-jump decay rate (mean up-jump = 1/eta_up; must be > 1)",
        },
        {
            "name": "eta_down",
            "display_name": "Down-Jump Decay (eta_down)",
            "default": 15.0,
            "min_value": 1.5,
            "max_value": 100.0,
            "step": 0.5,
            "description": "Down-jump decay rate (mean down-jump = -1/eta_down)",
        },
    ]

    def __init__(self, sigma=0.16, lam=1.0, p_up=0.3, eta_up=25.0, eta_down=15.0):
        self._sigma = sigma
        self._lam = lam
        self._p = p_up
        self._eta_u = eta_up
        self._eta_d = eta_down
        # Martingale compensator k = E[e^Y] - 1 (finite because eta_up > 1).
        self._k = (
            p_up * eta_up / (eta_up - 1.0)
            + (1.0 - p_up) * eta_down / (eta_down + 1.0)
            - 1.0
        )

    @property
    def name(self) -> str:
        return "Kou Jump-Diffusion"

    @property
    def supported_engines(self) -> list[PricingCapability]:
        return [PricingCapability.MONTE_CARLO, PricingCapability.FFT]

    def get_parameters(self) -> dict[str, float]:
        return {"sigma": self._sigma, "lam": self._lam, "p_up": self._p,
                "eta_up": self._eta_u, "eta_down": self._eta_d}

    def drift(self, s: float, v: float, t: float, r: float, q: float) -> float:
        # Drift compensated by -lambda*k so the discounted spot is a martingale.
        return (r - q - self._lam * self._k) * s

    def diffusion(self, s: float, v: float, t: float) -> float:
        # Continuous (Brownian) part only; the jumps are added by jump().
        return self._sigma * s

    def jump(self, s: np.ndarray, dt: float) -> np.ndarray:
        # Compound Poisson: draw N jumps per path, sample each jump's
        # asymmetric double-exponential size, scatter-add back to its path.
        n_jumps = np.random.poisson(self._lam * dt, len(s))
        idx = np.repeat(np.arange(len(s)), n_jumps)  # path of each individual jump
        up = np.random.random(len(idx)) < self._p
        y = np.where(
            up,
            np.random.exponential(1.0 / self._eta_u, len(idx)),
            -np.random.exponential(1.0 / self._eta_d, len(idx)),
        )
        total = np.zeros(len(s))
        np.add.at(total, idx, np.exp(y) - 1.0)
        return s * total

    def characteristic_function(
        self, u: complex, s0: float, t: float, r: float, q: float = 0.0
    ) -> complex:
        # Same factorization as Merton (GBM diffusion CF times a jump CF); here
        # the jump term is the double-exponential characteristic exponent.
        sigma2 = self._sigma ** 2
        x = np.log(s0)
        drift_adj = (r - q - 0.5 * sigma2 - self._lam * self._k) * t
        gbm_cf = np.exp(1j * u * (x + drift_adj) - 0.5 * sigma2 * t * u ** 2)
        jump_inner = (
            self._p * self._eta_u / (self._eta_u - 1j * u)
            + (1.0 - self._p) * self._eta_d / (self._eta_d + 1j * u)
            - 1.0
        )
        return gbm_cf * np.exp(self._lam * t * jump_inner)
''',
    "Variance-Gamma pure jumps (MC + FFT)": '''\
import numpy as np
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability

class VarianceGamma(Model):
    """Madan-Carr-Chang (1998) Variance-Gamma: a PURE-JUMP Levy process.
    ln S_T = ln S_0 + (r - q + omega)T + X_T,  X = theta*G + sigma*W(G)

    Brownian motion with drift theta evaluated on a random gamma clock G
    (variance rate nu) — no diffusion at all, yet a full smile: nu fattens
    both tails (kurtosis), theta < 0 tilts the smile into a skew.
    omega = ln(1 - theta*nu - sigma^2*nu/2)/nu is the martingale correction
    (parameter bounds keep its argument positive)."""

    EQUATION_LATEX = {
        "main": r"\\ln S_T = \\ln S_0 + (r - q + \\omega)T + \\theta G_T + \\sigma W_{G_T}",
        "clock": r"G_T \\sim \\Gamma(T/\\nu, \\nu), \\quad \\omega = \\tfrac{1}{\\nu}\\ln\\!\\left(1 - \\theta\\nu - \\tfrac{1}{2}\\sigma^2\\nu\\right)",
        "cf": r"\\varphi(u) = e^{iu(\\ln S_0 + (r - q + \\omega)T)} \\left(1 - iu\\theta\\nu + \\tfrac{1}{2}\\sigma^2\\nu u^2\\right)^{-T/\\nu}",
    }

    PARAMETER_SPECS = [
        {
            "name": "sigma",
            "display_name": "VG Volatility (sigma)",
            "default": 0.18,
            "min_value": 0.05,
            "max_value": 0.5,
            "step": 0.01,
            "description": "Volatility of the time-changed Brownian motion",
        },
        {
            "name": "theta",
            "display_name": "Drift on Gamma Clock (theta)",
            "default": -0.14,
            "min_value": -0.5,
            "max_value": 0.3,
            "step": 0.01,
            "description": "Drift of the subordinated BM (negative = skew)",
        },
        {
            "name": "nu",
            "display_name": "Clock Variance Rate (nu)",
            "default": 0.20,
            "min_value": 0.01,
            "max_value": 1.0,
            "step": 0.01,
            "description": "Variance rate of the gamma clock (kurtosis knob)",
        },
    ]

    def __init__(self, sigma=0.18, theta=-0.14, nu=0.20) -> None:
        self._sigma = sigma
        self._theta = theta
        self._nu = nu
        # Martingale correction; the spec box keeps the log argument > 0
        # (worst case: 1 - (-0.5)(1.0) - 0.5*0.25*1.0 > 0).
        self._omega = np.log(1.0 - theta * nu - 0.5 * sigma ** 2 * nu) / nu

    @property
    def name(self) -> str:
        return "Variance-Gamma"

    @property
    def supported_engines(self) -> list[PricingCapability]:
        return [PricingCapability.MONTE_CARLO, PricingCapability.FFT]

    def get_parameters(self) -> dict[str, float]:
        return {"sigma": self._sigma, "theta": self._theta, "nu": self._nu}

    def drift(self, s: float, v: float, t: float, r: float, q: float) -> float:
        # Risk-neutral drift (r - q + omega); omega offsets the jumps' mean so
        # the discounted spot stays a martingale.
        return (r - q + self._omega) * s

    def diffusion(self, s: float, v: float, t: float) -> float:
        return 0.0 * s  # pure jump — every move comes from jump()

    def jump(self, s: np.ndarray, dt: float) -> np.ndarray:
        # One VG increment per Euler step: BM with drift on a gamma clock.
        n = len(s)
        g = np.random.gamma(dt / self._nu, self._nu, n)
        x = self._theta * g + self._sigma * np.sqrt(g) * np.random.standard_normal(n)
        return s * (np.exp(x) - 1.0)

    def characteristic_function(
        self, u: complex, s0: float, t: float, r: float, q: float = 0.0
    ) -> complex:
        # VG CF: GBM-style phase times the gamma-subordinated factor
        # (1 - i u theta nu + sigma^2 nu u^2 / 2)^(-T/nu).
        x = np.log(s0)
        drift_adj = (r - q + self._omega) * t
        vg = (
            1.0 - 1j * u * self._theta * self._nu
            + 0.5 * self._sigma ** 2 * self._nu * u ** 2
        ) ** (-t / self._nu)
        return np.exp(1j * u * (x + drift_adj)) * vg
''',
    "SABR-like lognormal vol (MC)": '''\
import numpy as np
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability

class SABRLike(Model):
    """Lognormal stochastic volatility (SABR with beta = 1, no closed form).
    dS = (r-q)*S*dt + sqrt(V)*S*dW_1
    dV = nu^2*V*dt + 2*nu*V*dW_2,   corr(dW_1, dW_2) = rho

    The vol itself is a driftless GBM (d sigma = nu*sigma*dW_2), so V = sigma^2
    follows the SDE above by Ito. Unlike Heston there is NO mean reversion:
    vol paths wander, fattening long-dated wings — compare its term structure
    of smiles against the Heston-like template. Monte-Carlo route only."""

    EQUATION_LATEX = {
        "main": r"dS = (r - q) \\, S \\, dt + \\sqrt{V} \\, S \\, dW_1",
        "vol": r"d\\sigma = \\nu \\, \\sigma \\, dW_2 \\;\\Leftrightarrow\\; dV = \\nu^2 V \\, dt + 2\\nu V \\, dW_2, \\quad \\mathrm{corr}(dW_1, dW_2) = \\rho",
    }

    PARAMETER_SPECS = [
        {
            "name": "v0",
            "display_name": "Initial Variance (v0)",
            "default": 0.04,
            "min_value": 0.001,
            "max_value": 0.50,
            "step": 0.005,
            "description": "Initial variance (0.04 = 20% vol)",
        },
        {
            "name": "nu",
            "display_name": "Vol of Vol (nu)",
            "default": 0.6,
            "min_value": 0.05,
            "max_value": 2.0,
            "step": 0.05,
            "description": "Lognormal volatility of the vol process",
        },
        {
            "name": "rho",
            "display_name": "Correlation (rho)",
            "default": -0.5,
            "min_value": -0.99,
            "max_value": 0.99,
            "step": 0.01,
            "description": "Price-vol correlation (negative = leverage/skew)",
        },
    ]

    def __init__(self, v0=0.04, nu=0.6, rho=-0.5) -> None:
        self._v0 = v0
        self._nu = nu
        self._rho = rho

    @property
    def name(self) -> str:
        return "SABR-Like Lognormal Vol"

    @property
    def supported_engines(self) -> list[PricingCapability]:
        return [PricingCapability.MONTE_CARLO]

    def get_parameters(self) -> dict[str, float]:
        return {"v0": self._v0, "nu": self._nu, "rho": self._rho}

    def drift(self, s: float, v: float, t: float, r: float, q: float) -> float:
        return (r - q) * s

    def diffusion(self, s: float, v: float, t: float) -> float:
        # sqrt(V) * S; floor V at >= 0 for the Euler scheme.
        return np.sqrt(np.maximum(v, 1e-10)) * s

    def variance_drift(self, v: np.ndarray, s: np.ndarray, t: float) -> np.ndarray:
        # V = sigma^2 of a driftless lognormal vol (d sigma = nu sigma dW); by
        # Ito that is dV = nu^2 V dt + 2 nu V dW — no mean reversion (unlike CIR).
        return self._nu ** 2 * v

    def variance_diffusion(self, v: np.ndarray, s: np.ndarray, t: float) -> np.ndarray:
        return 2.0 * self._nu * v

    def get_correlation(self) -> float:
        return self._rho
''',
}


_EDITOR_BUTTONS = [
    {
        "name": "Copy",
        "feather": "Copy",
        "hasText": True,
        "alwaysOn": True,
        "commands": ["copyAll"],
        "style": {"top": "0.46rem", "right": "0.4rem"},
    },
    {
        # Always-on submit so the on-screen text reaches Python without the
        # built-in Ctrl/Cmd+Enter — otherwise Validate/Register ran against the
        # previously submitted (template) code and silently activated the wrong
        # model.
        "name": "Apply",
        "feather": "Check",
        "hasText": True,
        "alwaysOn": True,
        "commands": ["submit"],
        "style": {"bottom": "0.46rem", "right": "0.4rem"},
    },
]


def _current_editor_code(response, fallback: str) -> str:
    """The freshest editor text: the widget's delivered text if any, else the
    last stored code. With ``response_mode='blur'`` the editor delivers its
    text whenever it loses focus (e.g. clicking Validate), so this is the
    on-screen code — not a stale snapshot."""
    if response and response.get("text"):
        return response["text"]
    return fallback


def _is_validation_stale(validated_source: str | None, current_code: str) -> bool:
    """True when a passed validation no longer matches the on-screen code.

    Register pairs the validated ``model_class`` with the current editor text;
    if the user edits after validating, that couples the OLD compiled class
    with NEW source. Treat the validation as stale so Register is disabled
    until the user re-validates."""
    if validated_source is None:
        return False
    return validated_source != current_code


def render_custom_model_editor() -> None:
    """Render the define → validate → register workflow."""
    st.markdown("### Define Your Own Model")
    st.caption(
        "Write a Python class inheriting from `Model`. Provide either the SDE "
        "coefficients `drift()` / `diffusion()` (Monte-Carlo route) or a "
        "`characteristic_function()` (FFT route). Validate, register, then "
        "calibrate it to the current surface below."
    )

    # ── Status panel ─────────────────────────────────────────────────
    if is_registered():
        meta = st.session_state.get("calib_custom_model", {})
        c1, c2 = st.columns([3, 1])
        with c1:
            st.success(
                f"Active custom model: **{meta.get('name', '?')}** "
                f"· engines: {', '.join(meta.get('engines', []))}"
            )
        with c2:
            if st.button("Remove", key="calib_remove_custom", type="secondary"):
                unregister_custom_model()
                st.session_state.pop("calib_custom_result", None)
                st.rerun()

    st.markdown("---")

    # ── Step 1: template + editor ────────────────────────────────────
    st.markdown("**Step 1:** Write your model code")

    st.session_state.setdefault("calib_custom_editor_version", 0)
    st.session_state.setdefault("calib_custom_prev_template", None)
    st.session_state.setdefault(
        "calib_custom_editor_code", next(iter(_TEMPLATES.values()))
    )

    template_name = st.selectbox(
        "Start from template",
        options=list(_TEMPLATES.keys()),
        key="calib_custom_template_select",
        help="Select a template — code loads automatically.",
    )
    if template_name != st.session_state.calib_custom_prev_template:
        st.session_state.calib_custom_editor_code = _TEMPLATES[template_name]
        st.session_state.calib_custom_editor_version += 1
        st.session_state.calib_custom_prev_template = template_name
        st.session_state.pop("calib_custom_validation", None)
        st.session_state.pop("calib_custom_validated_source", None)
        st.rerun()

    editor_key = (
        f"calib_custom_code_editor_{st.session_state.calib_custom_editor_version}"
    )
    response = code_editor(
        st.session_state.calib_custom_editor_code,
        lang="python",
        height=[20, 40],
        buttons=_EDITOR_BUTTONS,
        key=editor_key,
        # 'blur' delivers the on-screen text whenever the editor loses focus
        # (e.g. clicking Validate), so edits are picked up without Ctrl+Enter.
        response_mode="blur",
        options={"wrap": True},
    )
    if response and response.get("text"):
        st.session_state.calib_custom_editor_code = response["text"]
    st.caption(
        "Edits sync automatically when you click away or the **Apply** button; "
        "then Validate."
    )

    st.markdown("---")

    # ── Step 2: validate ─────────────────────────────────────────────
    st.markdown("**Step 2:** Validate your model")
    if st.button("Validate Model", key="calib_validate_custom_btn", type="primary"):
        code = _current_editor_code(response, st.session_state.calib_custom_editor_code)
        st.session_state.calib_custom_editor_code = code
        st.session_state.calib_custom_validated_source = code
        with st.spinner("Running validation suite…"):
            st.session_state.calib_custom_validation = compile_and_validate(code)

    validation = st.session_state.get("calib_custom_validation")
    if validation:
        for test in validation.tests:
            badge = ":green[PASS]" if test.passed else ":red[FAIL]"
            st.markdown(f"&ensp; {badge} **{test.name}** — {test.message}")
        if validation.all_passed:
            st.success("All validation tests passed!")
        elif validation.tests:
            st.error("Some tests failed. Fix the issues above and re-validate.")

    st.markdown("---")

    # ── Step 3: register ─────────────────────────────────────────────
    st.markdown("**Step 3:** Register your model")
    stale = _is_validation_stale(
        st.session_state.get("calib_custom_validated_source"),
        st.session_state.calib_custom_editor_code,
    )
    can_register = validation is not None and validation.all_passed and not stale
    if st.button(
        "Register Model",
        key="calib_register_custom_btn",
        type="primary",
        disabled=not can_register,
    ):
        # Register the exact source that was validated, so the compiled class
        # and the stored source can never disagree.
        register_custom_model(
            validation.model_class,
            st.session_state.calib_custom_validated_source,
        )
        st.session_state.calib_custom_just_registered = validation.model_class.__name__
        st.session_state.pop("calib_custom_result", None)
        st.rerun()

    just_registered = st.session_state.pop("calib_custom_just_registered", None)
    if just_registered:
        st.success(f"Model **{just_registered}** registered — calibrate it below.")
    if stale and validation is not None and validation.all_passed:
        st.caption("Code changed since validation — re-validate before registering.")
    elif not can_register:
        st.caption(
            "Validate your model first. All tests must pass before registration."
        )
