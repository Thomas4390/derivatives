"""
Calibration App — Constants & Descriptions
=============================================

Static metadata used by the sidebar dropdowns, info boxes, and tooltips.
"""

from __future__ import annotations

# ── Model metadata ─────────────────────────────────────────────────────

MODEL_CHOICES: tuple[str, ...] = (
    "heston",
    "merton",
    "bates",
    "heston_nandi",
    "ngarch_q",
    "garch_q",
    "gjr_q",
    "garch",
    "ngarch",
    "gjr_garch",
    "iv_gbm",
)

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "heston": "Heston (5 params)",
    "merton": "Merton Jump-Diffusion (4 params)",
    "bates": "Bates SV+Jumps (8 params)",
    "heston_nandi": "AGARCH (Heston-Nandi) (5 params)",
    "ngarch_q": "Duan NGARCH (risk-neutral, 5 params)",
    "garch_q": "GARCH-Q (risk-neutral, 4 params)",
    "gjr_q": "GJR-GARCH-Q (risk-neutral, 5 params)",
    "garch": "GARCH(1,1) (3 params)",
    "ngarch": "NGARCH (4 params)",
    "gjr_garch": "GJR-GARCH (4 params)",
    "iv_gbm": "GBM Implied-Vol (1 param)",
}

# Emoji prefix per model — keeps the sidebar dropdown visually
# consistent with the tab strip ("🌐 Setup", "▶️ Live", "🔁 Multi-Start",
# …) where every entry is already glyph-prefixed.
MODEL_ICONS: dict[str, str] = {
    "heston": "🌀",      # SV swirl
    "merton": "💥",      # jump
    "bates": "🌪️",      # SV + jumps
    "heston_nandi": "🔁",  # risk-neutral GARCH recursion → surface
    "ngarch_q": "🎲",    # nonaffine risk-neutral GARCH → MC surface
    "garch_q": "🎰",     # symmetric risk-neutral GARCH → MC surface
    "gjr_q": "🃏",       # GJR risk-neutral GARCH → MC surface
    "garch": "📊",      # variance recursion
    "ngarch": "📈",     # asymmetric variance
    "gjr_garch": "📉",  # leverage indicator
    "iv_gbm": "🟦",     # closed-form baseline
}

MODEL_DESCRIPTIONS: dict[str, str] = {
    "heston": (
        r"Heston (1993) stochastic-volatility model. Five parameters: "
        r"$v_0$ (initial variance), $\kappa$ (mean-reversion speed), "
        r"$\sigma^2$ (long-run variance), $\alpha$ (vol-of-vol), $\rho$ "
        r"(spot-vol correlation). The Feller condition "
        r"$2\kappa\sigma^2 \geq \alpha^2$ is enforced via a soft penalty "
        r"during calibration."
    ),
    "merton": (
        r"Merton (1976) jump-diffusion model. Combines a Brownian "
        r"diffusion (volatility $\sigma$) with a compound-Poisson jump "
        r"process (intensity $\lambda$, log-normal sizes with mean "
        r"$\alpha_J$ and std $\sigma_J$). Tikhonov regularisation pulls "
        r"$\sigma$ toward the ATM IV to lift the diffusion / jump "
        r"identifiability issue."
    ),
    "bates": (
        r"Bates (1996) = Heston stochastic vol + Merton compound-Poisson "
        r"jumps. Eight parameters jointly calibrated via a 3-phase "
        r"semi-sequential procedure (Heston warm-up, then jumps-only, "
        r"then joint LM) to dodge the high-dimensional local minima."
    ),
    "heston_nandi": (
        r"AGARCH: the Heston-Nandi (2000) affine, risk-neutral "
        r"GARCH(1,1) that prices options in closed form. Variance "
        r"recursion "
        r"$h_{t+1} = \omega + \beta h_t + \alpha (z_t - \gamma\sqrt{h_t})^2$. "
        r"Calibrated to the option SURFACE (not a return series) by "
        r"Levenberg-Marquardt on a closed-form characteristic function. "
        r"Stationarity $\beta + \alpha\gamma^2 < 1$ is enforced via "
        r"OFF / soft / hard modes."
    ),
    "ngarch_q": (
        r"Duan (1995) NGARCH: the *nonaffine* risk-neutral GARCH(1,1) "
        r"(Engle-Ng 1993 dynamics, Dorion-François §7.2.1). Variance "
        r"recursion "
        r"$h_{t+1} = \omega + \alpha h_t (z_t - \gamma)^2 + \beta h_t$ "
        r"under $\mathbb{Q}$, where $\gamma$ is the risk-neutral "
        r"asymmetry $\gamma^* = \gamma + \lambda$. Unlike the affine "
        r"Heston-Nandi it has no closed-form characteristic function, "
        r"so it is calibrated to the option SURFACE by Monte-Carlo "
        r"(Duan LRNVR). Slower than the affine version but a better "
        r"empirical fit (Christoffersen-Jacobs 2004). Stationarity "
        r"$\beta + \alpha(1 + \gamma^2) < 1$ via OFF / soft / hard modes."
    ),
    "garch_q": (
        r"Symmetric risk-neutral GARCH(1,1) under $\mathbb{Q}$ "
        r"(Duan LRNVR), priced to the option SURFACE by Monte-Carlo. "
        r"Variance recursion "
        r"$h_{t+1} = \omega + \alpha h_t z_t^2 + \beta h_t$. "
        r"**No leverage**, so it produces a symmetric smile and cannot "
        r"reproduce an equity skew (instructive baseline). "
        r"Stationarity $\beta + \alpha < 1$ via OFF / soft / hard modes."
    ),
    "gjr_q": (
        r"Risk-neutral GJR-GARCH(1,1) under $\mathbb{Q}$ "
        r"(Glosten-Jagannathan-Runkle), priced to the option SURFACE by "
        r"Monte-Carlo. Leverage enters through a negative-shock "
        r"indicator: "
        r"$h_{t+1} = \omega + (\alpha + \gamma\,\mathbf{1}_{\{z_t < 0\}})\,h_t\,z_t^2 + \beta h_t$, "
        r"so $\gamma > 0$ generates a downward skew. Stationarity "
        r"$\beta + \alpha + \gamma/2 < 1$ via OFF / soft / hard modes."
    ),
    "garch": (
        r"GARCH(1,1) (Bollerslev 1986). Recursive volatility filter on "
        r"daily log-returns: "
        r"$\sigma_t^2 = \omega + \alpha\,r_{t-1}^2 + \beta\,\sigma_{t-1}^2$. "
        r"Calibrated by maximum likelihood with an exact JAX gradient."
    ),
    "ngarch": (
        r"NGARCH (Engle & Ng 1993): asymmetric variant adding a leverage "
        r"term $\gamma$ that lets negative shocks raise variance more "
        r"than positive ones of equal magnitude."
    ),
    "gjr_garch": (
        r"GJR-GARCH (Glosten, Jagannathan & Runkle 1993): uses an "
        r"indicator-based asymmetry term "
        r"$\gamma\,\mathbf{1}_{\{r_t < 0\}}\,r_t^2$. Captures the "
        r"leverage effect with one extra parameter."
    ),
    "iv_gbm": (
        r"GBM (Black-Scholes) constant-volatility baseline. Calibrated "
        r"by Black-Scholes IV inversion on each quote, then averaged "
        r"(equal-weight or vega-weighted)."
    ),
}

# Tight one-liner per model — fed to the per-button hover chip strip in
# the generator / candidate pickers (the long MODEL_DESCRIPTIONS above
# stays for the Theory / Setup deep-dive surfaces). Keys must cover
# MODEL_CHOICES.
MODEL_HOVER: dict[str, str] = {
    "heston": (
        "Stochastic-volatility model (5 params). Mean-reverting variance "
        "with spot-vol correlation; Feller enforced via a soft penalty."
    ),
    "merton": (
        "Jump-diffusion (4 params): Brownian diffusion plus log-normal "
        "Poisson jumps. Tikhonov-regularised."
    ),
    "bates": (
        "Heston volatility + Merton jumps (8 params). Fitted with a "
        "3-phase semi-sequential procedure."
    ),
    "heston_nandi": (
        "Risk-neutral GARCH(1,1) with a closed-form CF (5 params). "
        "Calibrates to the option surface like Heston; stationarity β + αγ² "
        "< 1 controllable (off / soft / hard)."
    ),
    "ngarch_q": (
        "Nonaffine risk-neutral NGARCH (Duan 1995, 5 params). Fits the option "
        "surface by Monte-Carlo — no closed form, slower than affine HN, but "
        "empirically a better fit. Stationarity β + α(1+γ²) < 1 controllable."
    ),
    "garch_q": (
        "Symmetric risk-neutral GARCH-Q (4 params). MC surface fit; no leverage "
        "→ symmetric smile (no skew). Stationarity β + α < 1 controllable."
    ),
    "gjr_q": (
        "Risk-neutral GJR-GARCH-Q (5 params). MC surface fit; leverage via a "
        "negative-shock indicator → skew. Stationarity β + α + γ/2 < 1 controllable."
    ),
    "garch": (
        "GARCH(1,1) variance recursion on daily log-returns (3 params). "
        "Maximum-likelihood with an exact JAX gradient."
    ),
    "ngarch": (
        "Asymmetric GARCH (4 params): a leverage term lets negative "
        "shocks raise variance more than positive ones."
    ),
    "gjr_garch": (
        "GJR-GARCH (4 params): indicator-based leverage term for the "
        "asymmetric variance response."
    ),
    "iv_gbm": (
        "Black-Scholes constant-vol baseline (1 param). Per-quote IV "
        "inversion, then averaged."
    ),
}

# General group-level explanations — these become the widget-level
# ``help=`` ("?") text, replacing the old per-option blob.
MODEL_GROUP_HELP_CANDIDATES: str = (
    "Candidate models are fitted independently against the same market "
    "data — total runs = candidates x solvers. Pick two or more to "
    "compare fit quality and model misspecification. Hover a button for "
    "that model's specifics."
)
MODEL_GROUP_HELP_GENERATOR: str = (
    "The generator's true parameters produce the synthetic ground-truth "
    "data the candidate models are calibrated against; a different "
    "generator demonstrates model misspecification. Hover a button for "
    "that model's specifics."
)

GARCH_FAMILY: tuple[str, ...] = ("garch", "ngarch", "gjr_garch")
SURFACE_FAMILY: tuple[str, ...] = (
    "heston",
    "merton",
    "bates",
    "heston_nandi",
    "ngarch_q",
    "garch_q",
    "gjr_q",
    "iv_gbm",
)

# Public mapping: data family → ordered tuple of eligible model keys.
# Used by sidebar widgets to filter generator / candidate dropdowns when
# the user toggles the family switch.
FAMILY_MODELS: dict[str, tuple[str, ...]] = {
    "surface": SURFACE_FAMILY,
    "returns": GARCH_FAMILY,
}

FAMILY_DEFAULT_MODEL: dict[str, str] = {
    "surface": "heston",
    "returns": "garch",
}


def model_data_mode(model_key: str) -> str:
    """Return ``'surface'`` or ``'returns'`` depending on the input the
    calibrator expects."""
    if model_key in GARCH_FAMILY:
        return "returns"
    return "surface"


def model_family(model_key: str) -> str:
    """Inverse of ``FAMILY_MODELS`` — ``"surface"`` or ``"returns"``."""
    return "returns" if model_key in GARCH_FAMILY else "surface"


# ── Solver metadata ────────────────────────────────────────────────────

SOLVER_CHOICES: tuple[str, ...] = ("LM-JAX", "DE", "NM", "L-BFGS-B")

SOLVER_DESCRIPTIONS: dict[str, str] = {
    "LM-JAX": (
        "Levenberg-Marquardt (Trust-Region-Reflective) with analytical "
        "JAX Jacobian — production V2 default. Local, gradient-based, "
        "fast convergence. Requires a least-squares problem so cannot be "
        "used for GARCH MLE."
    ),
    "DE": (
        "Differential Evolution (scipy). Population-based stochastic "
        "global solver — robust against local minima but ~10× slower "
        "than LM. Re-introduced from the legacy V1 stack."
    ),
    "NM": (
        "Nelder-Mead simplex (scipy). Local derivative-free method — "
        "geometric simplex moves, intuitive but slow on >5-dim problems. "
        "Re-introduced from the legacy V1 stack."
    ),
    "L-BFGS-B": (
        "Limited-memory BFGS with box bounds (scipy). Quasi-Newton local "
        "solver that uses the analytical gradient when available. "
        "Default for GARCH MLE."
    ),
}

SOLVER_BADGES: dict[str, str] = {
    "LM-JAX": "Local • Least-squares • Gradient",
    "DE": "Global • Scalar • Derivative-free",
    "NM": "Local • Scalar • Derivative-free",
    "L-BFGS-B": "Local • Scalar • Gradient",
}

# Compact glyph per solver — used as a prefix in the active-solver
# picker (segmented control) so the user can spot the chosen solver
# at a glance without reading the full name.
SOLVER_ICONS: dict[str, str] = {
    "LM-JAX": "⚡",     # trust-region, fast convergence
    "DE": "🧬",         # population-based, evolution-inspired
    "NM": "🔺",         # simplex / triangle reflections
    "L-BFGS-B": "📐",   # quasi-Newton with bounds
}

# Tight one-liner per solver — fed to the per-button hover chip strip in
# the solver picker (the longer SOLVER_DESCRIPTIONS above stays for the
# Theory tab). Sourced from each backend StrategyMetadata.description.
# Keys must cover SOLVER_CHOICES.
SOLVER_HOVER: dict[str, str] = {
    "LM-JAX": (
        "Levenberg-Marquardt with an analytical JAX Jacobian. Fast local "
        "least-squares solver — the production default for option-surface "
        "fits. Needs a residual vector, so it cannot run GARCH MLE."
    ),
    "DE": (
        "Differential Evolution: population-based stochastic GLOBAL "
        "solver. Robust against local minima but ~10x slower than LM on "
        "smooth problems."
    ),
    "NM": (
        "Nelder-Mead simplex: derivative-free LOCAL solver using "
        "geometric reflections. Intuitive but slow above ~5 parameters."
    ),
    "L-BFGS-B": (
        "Limited-memory BFGS with box bounds: quasi-Newton LOCAL solver "
        "on a scalar objective. Uses the gradient when available; the "
        "default for GARCH MLE."
    ),
}

# General group-level explanation — becomes the solver widget's ``help=``
# ("?") text, replacing the old per-option blob.
SOLVER_GROUP_HELP: str = (
    "A solver is the optimisation algorithm that searches for the "
    "parameters minimising the objective. Pick one or more to compare; "
    "each runs on every candidate model. Rule of thumb: LM-JAX for "
    "speed, DE for a global sanity check, NM / L-BFGS-B as scalar "
    "counter-examples. Hover a button for that solver's specifics."
)


# ── Objective function metadata ───────────────────────────────────────

OBJECTIVE_CHOICES: tuple[str, ...] = (
    "price_mse",
    "iv_mse",
    "vega_weighted",
    "spread_weighted",
    "relative",
    "huber",
)

OBJECTIVE_DISPLAY_NAMES: dict[str, str] = {
    "price_mse": "Price MSE",
    "iv_mse": "IV MSE",
    "vega_weighted": "Vega-weighted",
    "spread_weighted": "Spread-weighted",
    "relative": "Relative / Log",
    "huber": "Huber",
}

# Single-glyph prefix per objective — keeps the sidebar pills tight.
OBJECTIVE_ICONS: dict[str, str] = {
    "price_mse": "💲",
    "iv_mse": "📏",
    "vega_weighted": "🎚️",
    "spread_weighted": "↔️",
    "relative": "⚖️",
    "huber": "🛡️",
}

OBJECTIVE_DESCRIPTIONS: dict[str, str] = {
    "price_mse": (
        r"Standard $\|P^{\text{mod}} - P^{\text{mkt}}\|^2$. Robust and "
        r"JAX-friendly but ATM-dominated, since nominal prices grow with "
        r"maturity. Historical baseline (Heston 1993)."
    ),
    "iv_mse": (
        r"$\|\sigma^{\text{mod}} - \sigma^{\text{mkt}}\|^2$. "
        r"Moneyness-uniform; one IV inversion per quote per evaluation. "
        r"LM-JAX uses the implicit function theorem to get an *exact* "
        r"Jacobian without re-implementing the inversion in JAX "
        r"(custom JVP = $1/\mathcal{V}^{\text{BS}}(\sigma^\star)$)."
    ),
    "vega_weighted": (
        r"$(P^{\text{mod}} - P^{\text{mkt}}) / \mathcal{V}^{\text{BS}}$. "
        r"Fast linear approximation of IV-MSE. Industry default (Cont & "
        r"Tankov 2004)."
    ),
    "spread_weighted": (
        r"$(P^{\text{mod}} - P^{\text{mkt}}) / \mathrm{spread}^{\text{BA}}$. "
        r"Down-weights options with a wide bid-ask. Relevant for SPX and "
        r"crypto."
    ),
    "relative": (
        r"$(P^{\text{mod}} - P^{\text{mkt}}) / P^{\text{mkt}}$. "
        r"Scale-invariant, so short and long maturities count equally."
    ),
    "huber": (
        r"Quadratic for $|r| < \delta$, linear beyond. Reduces the "
        r"influence of outliers and stale ticks (Huber 1964, "
        r"Schoutens 2003)."
    ),
}

OBJECTIVE_BADGES: dict[str, str] = {
    "price_mse": "Local • Smooth • Baseline",
    "iv_mse": "Moneyness-uniform • Costly",
    "vega_weighted": "Smooth • Fast • Industry-default",
    "spread_weighted": "Liquidity-aware • Needs spreads",
    "relative": "Scale-invariant • Maturity-uniform",
    "huber": "Outlier-robust • Threshold δ",
}

# Tight one-liner per objective — fed to the per-button hover chip strip
# in the objective picker (the longer OBJECTIVE_DESCRIPTIONS / pedagogy
# expander stay for the deep-dive surface). Keys must cover
# OBJECTIVE_CHOICES.
OBJECTIVE_HOVER: dict[str, str] = {
    "price_mse": (
        "Plain price least-squares. Robust and JAX-friendly but "
        "dominated by large at-the-money prices."
    ),
    "iv_mse": (
        "Implied-vol least-squares — uniform across moneyness but "
        "inverts IV every evaluation; falls back to vega-weighted under "
        "LM-JAX."
    ),
    "vega_weighted": (
        "Price error divided by Black-Scholes vega — a fast IV-MSE proxy "
        "and the industry default."
    ),
    "spread_weighted": (
        "Price error divided by the bid-ask spread — down-weights "
        "illiquid quotes (SPX / crypto)."
    ),
    "relative": (
        "Relative price error — scale-invariant, so short and long "
        "maturities count equally."
    ),
    "huber": (
        "Quadratic for small errors, linear beyond a threshold — robust "
        "to outliers and stale ticks."
    ),
}

# General group-level explanation — becomes the objective widget's
# ``help=`` ("?") text, replacing the old per-option blob.
OBJECTIVE_GROUP_HELP: str = (
    "An objective is the loss that defines what 'a good fit' means. Pick "
    "one or more to compare side-by-side. Price MSE is the safe default; "
    "weighted / relative / Huber variants trade robustness for "
    "liquidity- or scale-awareness. Hover a button for that objective's "
    "specifics."
)


# ── Default synthetic-data settings ────────────────────────────────────

DEFAULT_SURFACE_CONFIG: dict = {
    "spot": 100.0,
    "rate": 0.05,
    "dividend_yield": 0.0,
    "n_strikes": 11,
    "n_maturities": 5,
    "strike_min": 80.0,
    "strike_max": 120.0,
    "maturity_min": 1.0 / 12.0,
    "maturity_max": 2.0,
    "noise_std": 0.0,
    "seed": 42,
}

DEFAULT_RETURNS_CONFIG: dict = {
    "n_periods": 1000,
    "annualization_factor": 252,
    "frequency": "daily",
    "seed": 42,
    "true_sigma0": 0.20,
    "spot": 100.0,
    "drift": 0.05,
}


# ── Numerical guardrails ──────────────────────────────────────────────
# Implied-vol sanity bounds applied when reading SPX quotes — anything
# outside [_IV_FILTER_MIN, _IV_FILTER_MAX] is treated as a stale tick.
IV_FILTER_MIN: float = 0.01
IV_FILTER_MAX: float = 4.99

# Lower bound on |true_param| used by the recovery-error denominator to
# avoid blow-up when a true parameter is zero.
RECOVERY_DENOM_CLAMP: float = 1e-12


# ── Live runner tuning ────────────────────────────────────────────────
# Polling cadence between snapshot drains in the Streamlit live UI —
# ~5 polls/sec keeps the chart smooth without thrashing the rerun loop.
LIVE_POLL_INTERVAL_SEC: float = 0.18
# Idle sleep when the worker is alive but produced no new snapshot —
# keeps CPU usage low.
LIVE_IDLE_SLEEP_SEC: float = 0.05


# ── st.cache_data sizes ───────────────────────────────────────────────
# Bumped from 8 (audit recommendation) — typical exploration round-trips
# through several configs and the smaller value caused unnecessary cache
# evictions. Memory cost stays trivial.
SNAPSHOT_CACHE_MAX_ENTRIES: int = 16
SURFACE_CACHE_MAX_ENTRIES: int = 16
RETURNS_CACHE_MAX_ENTRIES: int = 16
