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
# consistent with the tab strip ("🌐 Setup", "▶️ Live", "🗺️ Loss
# Landscape", …) where every entry is already glyph-prefixed.
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

# Nonaffine risk-neutral GARCH surface models (Duan-Q): no closed-form
# characteristic function, so they are calibrated to the option SURFACE by
# Monte-Carlo. Their scalar MC objective has no analytical Jacobian, so they
# share special calibration wiring — a single seeded start (multi-start hurts),
# a higher per-eval budget floor, a final polish, and a non-DE default solver.
# Reused across solver_panel, calibration_service, and synthetic_data_service.
RN_GARCH_SURFACE_MODELS: tuple[str, ...] = ("ngarch_q", "garch_q", "gjr_q")

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
    # Pseudo-objective shown for the returns-GARCH family, whose calibration
    # target is the MLE negative log-likelihood (not an ObjectiveStrategy).
    # Deliberately NOT in OBJECTIVE_CHOICES — it is never user-selectable.
    "nll": "Returns NLL",
}

# Single-glyph prefix per objective — keeps the sidebar pills tight.
OBJECTIVE_ICONS: dict[str, str] = {
    "price_mse": "💲",
    "iv_mse": "📏",
    "vega_weighted": "🎚️",
    "spread_weighted": "↔️",
    "relative": "⚖️",
    "huber": "🛡️",
    "nll": "📉",
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
    # Half-width of the strike grid in standardized-moneyness units (σ√T). Each
    # maturity's strikes span F·exp(±moneyness_width·σ_T·√T), so the surface fills
    # for any vol level. Replaces the old fixed dollar strike_min/strike_max.
    # Defaults to ±5σ√T; the sidebar slider can dial it back toward the forward.
    # ``generate_surface`` clamps the effective wings inward whenever the outermost
    # strikes would price below the inversion floor, so the grid renders NaN-free
    # even when the requested width reaches into the dead wings (see the wing-clamp
    # thresholds in the "Numerical guardrails" section).
    "moneyness_width": 5.0,
    "maturity_min": 1.0 / 12.0,
    "maturity_max": 2.0,
    "noise_std": 0.0,
    "seed": 42,
}

# ── IV-surface display x-axis ──────────────────────────────────────────
# Display-only choice of how the surface/smile x-axis is labelled. The
# synthetic grid is always generated on σ√T-standardized moneyness (with the
# wings clamped in so it fills without NaN holes for any vol level); these
# options re-express the
# *already priced* quotes in more familiar units at plot time — no surface
# regeneration. ``moneyness_sigma`` (the σ√T-standardized ``ln(K/F) / (σ√T)``
# the grid is natively built on) is the default; ``log_moneyness`` is the
# academic alternative. Keyed display-label → internal key (consumed by
# services/axis_display.py).
X_AXIS_OPTIONS: dict[str, str] = {
    "Log-moneyness  ln(K/F)": "log_moneyness",
    "σ√T-moneyness": "moneyness_sigma",
    "Strike  K ($)": "strike",
    "Moneyness  K / F": "k_over_f",
}
DEFAULT_X_AXIS: str = "moneyness_sigma"


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

# Smallest *model* option price (price units, spot ≈ 100) we trust to carry
# information for a Black-Scholes IV inversion. Below it the price is either a
# floored zero (no Monte-Carlo path crossed a deep-OTM short-maturity strike) or
# a tiny negative value from the FFT call→put parity on a deep-ITM call — in both
# cases the cell holds no model signal, so we mark it NaN instead of inverting the
# floor into a fabricated, model-independent IV. Mirrors the long-standing 1e-6
# price floor: any cell that rounds to the floor is exactly the artefact.
MIN_PRICE_FOR_IV: float = 1e-6

# Wing-clamp thresholds for the adaptive synthetic surface. A ±moneyness_width·σ√T
# grid can push its outermost strikes into wings where the model price falls below
# the IV-inversion floor, leaving NaN holes. Rather than fabricate an IV from a
# floored price, ``generate_surface`` shrinks the grid's wings to the domain where
# the *production* re-pricing still yields an informative price. A cell priced below
# the applicable threshold is treated as "dead" and clamped out. The threshold is
# route-dependent, and carries a margin above MIN_PRICE_FOR_IV because the overlay
# re-prices at the FIT (not the truth):
#   - FFT / closed-form surfaces: a ×10 margin on MIN_PRICE_FOR_IV is enough — the
#     re-price is deterministic, only the fitted parameters differ from the truth.
#   - Monte-Carlo surfaces (GARCH-Q trio, nonaffine custom): a much wider margin,
#     because a true price of ~2.9e-4 can round to 0 under a finite path count. The
#     ground-truth surface prices at MC_PATHS_TRUTH (80k), but the post-calibration
#     overlay re-prices at only MC_PATHS_SURFACE (30k), so the clamp keeps the wings
#     where even the coarser overlay stays above the inversion floor.
SURFACE_WING_MIN_PRICE_FFT: float = 1e-5
SURFACE_WING_MIN_PRICE_MC: float = 1e-3
# Never pull a wing tighter than ±1σ√T (the near-ATM core always prices well above
# the floor); a persistent dead cell inside this core keeps the NaN convention.
SURFACE_WING_MIN_WIDTH: float = 1.0
# The MC re-price of a clamped grid can resurface a boundary dead cell, so the clamp
# iterates a few times before falling back to the NaN convention + a logged warning.
SURFACE_WING_CLAMP_MAX_ITERS: int = 3

# Lower bound on |true_param| used by the recovery-error denominator to
# avoid blow-up when a true parameter is zero.
RECOVERY_DENOM_CLAMP: float = 1e-12

# Risk-free rate assumed when pricing a returns-family (GARCH) model's
# model-implied IV surface under Q (Duan LRNVR). The returns data carries a
# physical drift μ, not a rate, so the diagnostics surface uses this fixed
# assumption rather than silently defaulting an inline literal.
RETURNS_IMPLIED_SURFACE_RATE: float = 0.05

# Strike span of the returns-family model-implied IV surface (Diagnostics tab).
# The grid is shared by every displayed maturity, so its half-width follows the
# SHORTEST maturity: ±RETURNS_IMPLIED_STRIKE_SPAN_SIGMAS·σ_LR·√T_min, where σ_LR
# is the smallest long-run annualised vol on display. The old fixed ±20 % grid
# was ~7σ√T at 1 month for a low-vol GARCH — the MC prices there round to zero
# and the IV inversion fails, punching NaN holes in the chart. Clamped to
# [MIN, MAX] so the chart never collapses nor exceeds the old span.
RETURNS_IMPLIED_STRIKE_SPAN_SIGMAS: float = 4.0
RETURNS_IMPLIED_HALFWIDTH_MIN: float = 0.05
RETURNS_IMPLIED_HALFWIDTH_MAX: float = 0.20


# ── Calibration numeric defaults ──────────────────────────────────────
# Domain defaults that were previously hard-coded (some duplicated across the UI
# panels and the calibration service). Centralised here so they have one home.

# Vega-weighted objective fallback IV (used when a quote carries no implied vol).
FALLBACK_IV: float = 0.20
# Huber objective threshold δ default (price units).
HUBER_DELTA_DEFAULT: float = 0.05
# Default volatility for the closed-form GBM (iv_gbm) synthetic surface.
DEFAULT_GBM_SIGMA: float = 0.2

# Monte-Carlo paths / seed for the risk-neutral GARCH pricing paths. A fixed seed
# gives common random numbers, so repeated re-pricing is reproducible.
MC_PATHS_INTERACTIVE: int = 20_000  # per-eval cap during interactive calibration
MC_PATHS_SURFACE: int = 30_000  # post-calibration surface re-pricing / animation
MC_PATHS_TRUTH: int = 80_000  # clean synthetic ground-truth surface (nonaffine GARCH-Q)
MC_SEED: int = 12_345

# Loss-landscape sweep budgets. The grid re-prices the model at EVERY cell, so
# MC-priced models (GARCH-Q trio, MC-only custom models) get a reduced path
# budget — the fixed MC_SEED gives common random numbers, so the surface stays
# smooth and deterministic; only the loss LEVEL shifts slightly vs the 20k-path
# calibration objective, never the basin shape — and a coarser default grid
# than the closed-form / NLL backends.
LANDSCAPE_MC_PATHS: int = 5_000
LANDSCAPE_RESOLUTION_DEFAULT: int = 20
LANDSCAPE_RESOLUTION_DEFAULT_MC: int = 15

# Per-eval budget *floor* for the nonaffine GARCH-Q calibrators (ngarch_q / garch_q
# / gjr_q). Their scalar MC objective has no Jacobian, so only derivative-free
# solvers run (NM / DE / L-BFGS-B); the shared default of 200 stalls far from the
# optimum even when the truth is attainable. Measured sweet spot is a single seeded
# start at ~300 evals + a ~100-eval polish ≈ 1 min wall-clock (400 raises accuracy
# but overshoots the responsiveness budget; 500 worse still). See
# ``services/calibration_service._calibrator_for``.
GARCH_Q_MAX_NFEV_FLOOR: int = 300
# Budget for the calibrator's final local polish (a fresh local solve from the
# multi-start incumbent). Buys ~30–40 % RMSE for ~16 s on the trio. See the
# ``polish_nfev`` / ``polish_optimizer`` params of ``GARCHRiskNeutralCalibrator``.
GARCH_Q_POLISH_NFEV: int = 100

# Soft Feller / stationarity penalty-weight slider (shared by both panels).
PENALTY_WEIGHT_DEFAULT: float = 1000.0
PENALTY_WEIGHT_MAX: float = 5000.0
PENALTY_WEIGHT_STEP: float = 100.0


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
