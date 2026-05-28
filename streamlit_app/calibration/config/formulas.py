"""LaTeX formulas + parameter interpretations used by the Theory tab
and the glossary popover.

Kept in a single module so the cheat-sheet and the glossary draw from
the same DRY source of truth.

Notation: the Heston variance
reverts to its long-run level ``σ²`` with vol-of-vol ``α``; Merton jumps
are ``J ~ N(α_J, σ_J²)`` with intensity ``λ``; the affine GARCH of
Heston-Nandi is labelled *AGARCH*. The symbol ``θ`` is reserved here for
the **generic parameter vector** being calibrated (the optimisation
variable ``L(θ)``), never for a model parameter.
"""

from __future__ import annotations

# ── Stochastic dynamics (SDE / variance recursion) ──────────────────────

MODEL_SDE_LATEX: dict[str, str] = {
    "heston": r"""
\begin{aligned}
dS_t &= (r - y)\,S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S \\
dv_t &= \kappa\,(\sigma^2 - v_t)\,dt + \alpha\,\sqrt{v_t}\,dW_t^v \\
\langle dW^S, dW^v\rangle &= \rho\,dt
\end{aligned}
""",
    "merton": r"""
\begin{aligned}
\frac{dS_t}{S_{t^-}} &= (r - y - \lambda\,m)\,dt + \sigma\,dW_t + (e^{J} - 1)\,dN_t \\
J &\sim \mathcal{N}(\alpha_J,\,\sigma_J^2),\quad m = e^{\alpha_J + \tfrac{1}{2}\sigma_J^2} - 1,\quad N_t \sim \text{Poi}(\lambda t)
\end{aligned}
""",
    "bates": r"""
\begin{aligned}
\frac{dS_t}{S_{t^-}} &= (r - y - \lambda m)\,dt + \sqrt{v_t}\,dW_t^S + (e^{J}-1)\,dN_t \\
dv_t &= \kappa(\sigma^2 - v_t)\,dt + \alpha\sqrt{v_t}\,dW_t^v,\quad \langle dW^S, dW^v\rangle = \rho\,dt \\
J &\sim \mathcal{N}(\alpha_J,\,\sigma_J^2)
\end{aligned}
""",
    "iv_gbm": r"""
\frac{dS_t}{S_t} = (r - y)\,dt + \sigma\,dW_t
""",
    "heston_nandi": r"""
\begin{aligned}
R_t &= r_{\text{step}} - \tfrac{1}{2} h_t + \sqrt{h_t}\,z_t,\quad z_t \sim \mathcal{N}(0,1) \\
h_{t+1} &= \omega + \beta\,h_t + \alpha\,\bigl(z_t - \gamma\sqrt{h_t}\bigr)^2 \\
\text{stationarity:}\;\;& \beta + \alpha\gamma^2 < 1\quad\text{(closed-form characteristic function under }\mathbb{Q}\text{)}
\end{aligned}
""",
    "ngarch_q": r"""
\begin{aligned}
R_t &= r_{\text{step}} - \tfrac{1}{2} h_t + \sqrt{h_t}\,z_t,\quad z_t \sim \mathcal{N}(0,1) \\
h_{t+1} &= \omega + \alpha\,h_t\,(z_t - \gamma)^2 + \beta\,h_t \\
\text{stationarity:}\;\;& \beta + \alpha\,(1 + \gamma^2) < 1\quad\text{(nonaffine — MC pricing under }\mathbb{Q}\text{)}
\end{aligned}
""",
    "garch_q": r"""
\begin{aligned}
R_t &= r_{\text{step}} - \tfrac{1}{2} h_t + \sqrt{h_t}\,z_t,\quad z_t \sim \mathcal{N}(0,1) \\
h_{t+1} &= \omega + \alpha\,h_t\,z_t^2 + \beta\,h_t\quad\text{(symmetric — no leverage)} \\
\text{stationarity:}\;\;& \beta + \alpha < 1
\end{aligned}
""",
    "gjr_q": r"""
\begin{aligned}
R_t &= r_{\text{step}} - \tfrac{1}{2} h_t + \sqrt{h_t}\,z_t,\quad z_t \sim \mathcal{N}(0,1) \\
h_{t+1} &= \omega + \bigl(\alpha + \gamma\,\mathbf{1}_{\{z_t < 0\}}\bigr)\,h_t\,z_t^2 + \beta\,h_t \\
\text{stationarity:}\;\;& \beta + \alpha + \tfrac{\gamma}{2} < 1
\end{aligned}
""",
    "garch": r"""
\begin{aligned}
r_t &= \sigma_t\,z_t,\quad z_t \sim \mathcal{N}(0,1) \\
\sigma_t^2 &= \omega + \alpha\,r_{t-1}^2 + \beta\,\sigma_{t-1}^2 \\
\text{stationarity:}\;\;& \alpha + \beta < 1
\end{aligned}
""",
    "ngarch": r"""
\sigma_t^2 = \omega + \alpha\,(r_{t-1} - \gamma\,\sigma_{t-1})^2 + \beta\,\sigma_{t-1}^2
""",
    "gjr_garch": r"""
\sigma_t^2 = \omega + \alpha\,r_{t-1}^2 + \gamma\,\mathbf{1}_{\{r_{t-1} < 0\}}\,r_{t-1}^2 + \beta\,\sigma_{t-1}^2
""",
}

# ── Calibration objective ───────────────────────────────────────────────
# θ here is the generic vector of calibrated parameters (optimisation
# variable), not a model parameter — see the module docstring.

SURFACE_LOSS_LATEX = r"""
\mathcal{L}(\theta) = \frac{1}{N}\sum_{i,j} w_{ij}\,\bigl(\mathrm{IV}^{\text{model}}_{ij}(\theta)
- \mathrm{IV}^{\text{mkt}}_{ij}\bigr)^2 + \lambda_{\text{pen}}\,\mathrm{pen}(\theta)
"""

GARCH_LOSS_LATEX = r"""
-2\,\log L(\theta) = \sum_{t=1}^{N}\bigl[\,\log(2\pi) + \log\sigma_t^2(\theta) + \frac{r_t^2}{\sigma_t^2(\theta)}\bigr]
"""

# ── Penalty terms (model-specific) ──────────────────────────────────────
# Heston Feller boundary: 2κσ² > α².

FELLER_PENALTY_LATEX = r"\mathrm{pen}(\theta) = \bigl(\max(0,\,\alpha^2 - 2\kappa\sigma^2)\bigr)^2"

TIKHONOV_PENALTY_LATEX = (
    r"\mathrm{pen}(\theta) = (\sigma - \mathrm{IV}_{\text{ATM}})^2"
)

# Generic stationarity penalty for the four risk-neutral GARCH-Q variants.
# ``ρ(θ)`` is the model-specific persistence (β + αγ² for AGARCH, β + α(1+γ²)
# for Duan NGARCH-Q, β + α for symmetric GARCH-Q, β + α + γ/2 for GJR-Q) —
# kept in ``STATIONARITY_PERSISTENCE_LATEX`` below.
STATIONARITY_PENALTY_LATEX = (
    r"\mathrm{pen}(\theta) = \bigl(\max\bigl(0,\,\rho(\theta) - (1 - \epsilon)\bigr)\bigr)^2"
)

STATIONARITY_PERSISTENCE_LATEX: dict[str, str] = {
    "heston_nandi": r"\rho(\theta) = \beta + \alpha\gamma^2",
    "ngarch_q": r"\rho(\theta) = \beta + \alpha\,(1 + \gamma^2)",
    "garch_q": r"\rho(\theta) = \beta + \alpha",
    "gjr_q": r"\rho(\theta) = \beta + \alpha + \tfrac{\gamma}{2}",
}

# Duan (1995) Local Risk-Neutral Valuation Relationship — the forward map
# from physical-measure to risk-neutral GARCH parameters used when pricing
# options from a P-measure GARCH fit. ``λ`` is the unit market price of
# risk; the diagnostics in this app assume λ = 0 by default.
DUAN_LRNVR_LATEX = r"\gamma^{*} = \gamma + \lambda"


# ── Calibration residual & aggregate per objective ──────────────────────
# Mirrors backend.calibration.objectives.<obj>.metadata.formula but kept
# as a module-level constant so the UI does not have to instantiate an
# objective just to render its formula. Update both sides if either
# changes.

OBJECTIVE_RESIDUAL_LATEX: dict[str, str] = {
    "price_mse": r"r_i = P^{\text{mod}}_i - P^{\text{mkt}}_i",
    "iv_mse": r"r_i = \sigma^{\text{mod}}_i - \sigma^{\text{mkt}}_i",
    "vega_weighted": (
        r"r_i = \dfrac{P^{\text{mod}}_i - P^{\text{mkt}}_i}{\mathcal{V}^{\text{BS}}_i}"
    ),
    "spread_weighted": (
        r"r_i = \dfrac{P^{\text{mod}}_i - P^{\text{mkt}}_i}{\mathrm{spread}^{\text{BA}}_i}"
    ),
    "relative": (
        r"r_i = \dfrac{P^{\text{mod}}_i - P^{\text{mkt}}_i}{P^{\text{mkt}}_i}"
        r"\quad\text{or}\quad r_i = \log P^{\text{mod}}_i - \log P^{\text{mkt}}_i"
    ),
    "huber": (
        r"\rho_\delta(r) = \tfrac{1}{2}r^2\ \text{if }|r| < \delta;\quad "
        r"\delta\bigl(|r| - \tfrac{\delta}{2}\bigr)\ \text{otherwise}"
    ),
}

OBJECTIVE_AGGREGATE_LATEX: dict[str, str] = {
    "price_mse": r"L(\theta) = \sqrt{\tfrac{1}{N}\sum_i r_i^{2}}",
    "iv_mse": r"L(\theta) = \sqrt{\tfrac{1}{N}\sum_i r_i^{2}}",
    "vega_weighted": r"L(\theta) = \sqrt{\tfrac{1}{N}\sum_i r_i^{2}}",
    "spread_weighted": r"L(\theta) = \sqrt{\tfrac{1}{N}\sum_i r_i^{2}}",
    "relative": r"L(\theta) = \sqrt{\tfrac{1}{N}\sum_i r_i^{2}}",
    "huber": r"L(\theta) = \tfrac{1}{N}\sum_i \rho_\delta(r_i)",
}


# ── Parameter cheat-sheet rows ──────────────────────────────────────────
# Each row = (symbol, name, typical_range, smile_or_returns_effect).
# Used to populate a Markdown table per model in the Theory tab.

PARAM_CHEATSHEET: dict[str, list[tuple[str, str, str, str]]] = {
    "heston": [
        ("v₀", "initial variance", "[0.01, 0.10]",
         "front-month ATM level (square-root = ATM σ short-dated)"),
        ("κ", "mean-reversion speed", "[0.5, 5]",
         "term-structure curvature; large κ → ATM σ flattens toward √σ²"),
        ("σ²", "long-run variance", "[0.02, 0.15]",
         "long-dated ATM level (square-root = ATM σ long-dated)"),
        ("α", "vol-of-vol", "[0.1, 1.0]",
         "smile convexity (wings); large α → deeper smile"),
        ("ρ", "spot-vol correlation", "[-0.9, -0.2]",
         "skew; negative → left wing lifted, equities typical"),
    ],
    "merton": [
        ("σ", "diffusion vol", "[0.10, 0.40]",
         "ATM level; identified jointly with jumps (Tikhonov anchors it)"),
        ("λ", "jump intensity", "[0, 2]",
         "rate at which jumps occur per year (Poisson)"),
        ("α_J", "mean log-jump", "[-0.3, 0.05]",
         "skew direction; negative → left-skewed smile"),
        ("σ_J", "jump std-dev", "[0.05, 0.35]",
         "smile kurtosis (fat tails); small σ_J → sharp wings"),
    ],
    "bates": [
        ("v₀", "initial variance", "[0.01, 0.10]", "as Heston"),
        ("κ", "mean-reversion speed", "[0.5, 5]", "as Heston"),
        ("σ²", "long-run variance", "[0.02, 0.15]", "as Heston"),
        ("α", "vol-of-vol", "[0.1, 1.0]", "as Heston"),
        ("ρ", "spot-vol correlation", "[-0.9, -0.2]", "as Heston"),
        ("λ", "jump intensity", "[0, 2]", "as Merton"),
        ("α_J", "mean log-jump", "[-0.3, 0.05]", "as Merton"),
        ("σ_J", "jump std-dev", "[0.05, 0.35]", "as Merton"),
    ],
    "iv_gbm": [
        ("σ", "Black-Scholes vol", "[0.05, 0.50]",
         "flat smile — fitted as a weighted average of quote-level IV inversions"),
    ],
    "heston_nandi": [
        ("ω", "variance intercept (per period)", "[1e-9, 1e-4]",
         "anchors the unconditional variance; small absolute scale (per-period units)"),
        ("α", "ARCH coefficient", "[1e-9, 1e-3]",
         "weight of the squared shock — multiplies (z − γ√h)² so it stays tiny"),
        ("β", "GARCH persistence", "[0.0, 0.999]",
         "memory of past variance; closed-form CF holds while β + αγ² < 1"),
        ("γ", "risk-neutral leverage", "[0, 1000]",
         "asymmetry shift on the shock; large O(100) because it scales √h"),
        ("h₀", "initial variance (per period)", "[1e-7, 1e-2]",
         "seed for the recursion — square is the short-dated front-month variance"),
    ],
    "ngarch_q": [
        ("ω", "variance intercept (per period)", "[1e-9, 1e-4]",
         "long-run per-period variance ≈ ω / (1 − β − α(1+γ²))"),
        ("α", "ARCH coefficient", "[1e-6, 0.2]",
         "weight of the centred squared shock (z − γ)²"),
        ("β", "GARCH persistence", "[0.0, 0.98]",
         "memory of past variance; stationarity requires β + α(1+γ²) < 1"),
        ("γ", "risk-neutral asymmetry γ\\* = γ + λ", "[0, 4]",
         "Duan leverage shift, O(1) here — generates the equity skew"),
        ("h₀", "initial variance (per period)", "[1e-7, 1e-2]",
         "seed for the recursion — short-dated front-month variance"),
    ],
    "garch_q": [
        ("ω", "variance intercept (per period)", "[1e-9, 1e-4]",
         "long-run per-period variance ≈ ω / (1 − β − α)"),
        ("α", "ARCH coefficient", "[1e-6, 0.2]",
         "weight of the squared shock z²"),
        ("β", "GARCH persistence", "[0.0, 0.98]",
         "memory of past variance; stationarity requires β + α < 1"),
        ("h₀", "initial variance (per period)", "[1e-7, 1e-2]",
         "no leverage → smile stays symmetric (no equity skew)"),
    ],
    "gjr_q": [
        ("ω", "variance intercept (per period)", "[1e-9, 1e-4]",
         "long-run per-period variance ≈ ω / (1 − β − α − γ/2)"),
        ("α", "ARCH coefficient (symmetric)", "[1e-6, 0.2]",
         "weight of the squared shock z² on every observation"),
        ("β", "GARCH persistence", "[0.0, 0.98]",
         "memory of past variance; stationarity requires β + α + γ/2 < 1"),
        ("γ", "risk-neutral leverage indicator", "[0, 2]",
         "extra ARCH on negative shocks only — generates the downward skew"),
        ("h₀", "initial variance (per period)", "[1e-7, 1e-2]",
         "seed for the recursion — short-dated front-month variance"),
    ],
    "garch": [
        ("ω", "intercept", "[1e-6, 1e-3]",
         "long-run variance ≈ ω / (1 − α − β)"),
        ("α", "ARCH effect", "[0.02, 0.20]",
         "short-term shock reaction (yesterday's squared return)"),
        ("β", "GARCH persistence", "[0.75, 0.97]",
         "memory of past variance; α + β → 1 ⇒ near-IGARCH"),
    ],
    "ngarch": [
        ("ω", "intercept", "[1e-6, 1e-3]", "as GARCH"),
        ("α", "ARCH effect", "[0.02, 0.20]", "as GARCH"),
        ("β", "GARCH persistence", "[0.75, 0.97]", "as GARCH"),
        ("γ", "leverage / asymmetry", "[0.0, 1.5]",
         "negative-return amplification; equity γ > 0 typical"),
    ],
    "gjr_garch": [
        ("ω", "intercept", "[1e-6, 1e-3]", "as GARCH"),
        ("α", "ARCH effect", "[0.02, 0.20]", "as GARCH"),
        ("β", "GARCH persistence", "[0.75, 0.97]", "as GARCH"),
        ("γ", "leverage indicator", "[0.0, 0.20]",
         "extra ARCH effect on **negative** shocks only"),
    ],
}

# ── Glossary (alphabetical) ─────────────────────────────────────────────
# Symbol → short definition. Used by the floating glossary popover.

GLOSSARY: dict[str, str] = {
    "α": "Heston/Bates vol-of-vol (diffusion of the variance process); "
         "in the GARCH family, the ARCH effect — weight of yesterday's squared return.",
    "α_J": "Mean of the log-jump distribution (Merton/Bates), J ~ N(α_J, σ_J²).",
    "β": "GARCH persistence: weight of yesterday's variance on today's variance.",
    "γ": "Asymmetry / leverage parameter — NGARCH leverage, GJR-GARCH leverage "
         "indicator, and AGARCH (Heston-Nandi) risk-neutral leverage.",
    "θ": "Generic vector of calibrated parameters — the optimisation variable in L(θ). "
         "Not a model parameter (Heston long-run variance is σ²).",
    "κ": "Heston mean-reversion speed for the variance process.",
    "λ": "Jump intensity — expected number of jumps per year (Merton/Bates).",
    "ρ": "Spot-vol correlation between asset and variance Brownian motions.",
    "σ": "Diffusion volatility (Merton, iv_gbm).",
    "σ²": "Heston/Bates long-run variance — the level the variance reverts to "
          "(square-root = long-run vol).",
    "σ_J": "Standard deviation of the log-jump distribution (Merton/Bates).",
    "σ_t": "Conditional volatility at time t (GARCH-style filtered).",
    "ω": "GARCH intercept term — sets the unconditional variance.",
    "v₀": "Heston initial variance v — square of the short-dated ATM IV.",
    "S₀": "Underlying spot price at calibration time.",
    "r": "Continuously-compounded risk-free rate.",
    "y": "Continuous dividend yield.",
    "τ": "Time to maturity (years).",
    "K": "Strike price.",
    "IV": "Black-Scholes implied volatility — inverted from price quotes.",
    "RMSE": "Root mean squared error of the model residuals.",
    "AIC": "Akaike Information Criterion: 2·k − 2·log L. Lower is better.",
    "BIC": "Bayesian Information Criterion: k·log N − 2·log L. Penalises more harshly than AIC.",
    "LR": "Likelihood Ratio test statistic: 2·(log L_full − log L_nested).",
    "Feller condition": "2κσ² ≥ α² — prevents the Heston variance process from hitting zero.",
    "Tikhonov regularisation": "Adds (σ − IV_ATM)² to the Merton loss to pin σ near the ATM vol and break the σ/jumps identifiability.",
    "Pareto frontier": "Set of solutions that aren't dominated on both speed AND accuracy.",
}
