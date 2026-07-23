"""Loss-landscape computation for the Loss Landscape tab.

Given a model and two of its parameters, sweep a 2-D meshgrid fixing every
other parameter at the user-provided base point and record the calibration
loss at each ``(p_x, p_y)`` cell. The cell evaluation dispatches per family
(:func:`loss_backend`):

- ``"fft"`` — affine surface models + FFT-capable custom models: closed-form
  pricing, loss under the **chosen calibration objective** (the same
  ``ObjectiveStrategy`` minimised by the solver).
- ``"mc"`` — the risk-neutral GARCH-Q trio + MC-only custom models: same
  objective, but Monte-Carlo pricing at a reduced ``LANDSCAPE_MC_PATHS``
  budget. The fixed ``MC_SEED`` gives common random numbers, so the sweep is
  smooth and deterministic (only the loss *level* shifts slightly vs the
  full-path calibration objective, never the basin shape).
- ``"nll"`` — the physical returns-GARCH family: the MLE negative
  log-likelihood itself (the very objective the calibrator minimises),
  evaluated by the jitted JAX closures in
  ``backend.engines.aad.calibration.garch_nll``. All axes are per-period
  (daily) units — the same scale as the calibrator's search vector and the
  overlaid solver trajectories. A ``jax.jit(jax.vmap(...))`` batch over the
  whole grid would be a further speed-up, but at ~2-5 ms per jitted call the
  plain loop already matches the FFT route.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from backend.calibration.objectives import ObjectiveStrategy
from backend.calibration.pricing_loop import price_surface
from config.constants import (
    GARCH_FAMILY,
    LANDSCAPE_MC_PATHS,
    MC_SEED,
    RN_GARCH_SURFACE_MODELS,
)
from config.model_registry import get_spec
from services.post_calibration import _engine, rebuild_model, surface_model_prices


@dataclass(frozen=True)
class LandscapeResult:
    param_x: str
    param_y: str
    x_values: np.ndarray
    y_values: np.ndarray
    loss_grid: np.ndarray  # shape (n_y, n_x), NaN for failures
    base_params: dict[str, float]  # frozen at compute time


def loss_backend(model_key: str) -> str:
    """How a model's landscape cell-loss is evaluated: ``fft``/``mc``/``nll``.

    The registered custom model is ``"fft"`` when its class exposes a
    characteristic function (the registration metadata lists the FFT engine),
    ``"mc"`` otherwise. Import is lazy so tests can patch the custom-model
    service without a Streamlit runtime.
    """
    if model_key in GARCH_FAMILY:
        return "nll"
    if model_key in RN_GARCH_SURFACE_MODELS:
        return "mc"
    if model_key == "custom":
        from services.custom_model_service import get_custom_meta

        meta = get_custom_meta()
        engines = (meta or {}).get("engines") or ()
        return "fft" if "FFT" in engines else "mc"
    return "fft"


def is_supported(model_key: str) -> tuple[bool, str | None]:
    """Return ``(ok, reason_if_not)``. The tab uses this to short-circuit.

    Every calibratable model family now has a landscape backend (FFT, MC, or
    returns-NLL); the only exclusions left are structural — fewer than two
    parameters means there is no 2-D slice to draw.
    """
    if model_key == "iv_gbm":
        return False, (
            "iv_gbm has a single parameter (σ) — a 2-D slice needs two "
            "parameters to vary, so there is no landscape to draw."
        )
    if model_key == "custom":
        from services.custom_model_service import is_registered

        if not is_registered():
            return False, "register a custom model in the 🧪 Custom Model tab first."
    spec = get_spec(model_key)
    if spec.n_params < 2:
        return False, "Need ≥ 2 parameters to render a 2-D slice."
    return True, None


# Parameter order each NLL closure expects — mirrors GARCHCalibrator._param_names.
_NLL_ORDER: dict[str, tuple[str, ...]] = {
    "garch": ("omega", "alpha", "beta"),
    "ngarch": ("omega", "alpha", "beta", "gamma"),
    "gjr_garch": ("omega", "alpha", "beta", "gamma"),
}


def _nll_fn(model_key: str):
    """The jitted NLL closure for one returns-GARCH variant (lazy import)."""
    from backend.engines.aad.calibration.garch_nll import (
        nll_garch_jit,
        nll_gjr_jit,
        nll_ngarch_jit,
    )

    return {
        "garch": nll_garch_jit,
        "ngarch": nll_ngarch_jit,
        "gjr_garch": nll_gjr_jit,
    }[model_key]


def _make_cell_loss(
    model_key: str,
    market_data,
    objective: ObjectiveStrategy | None,
    n_paths_mc: int | None,
):
    """Build the per-cell loss callable for one grid sweep.

    Hoists everything reusable (jitted NLL closure, returns array, MC budget)
    out of the cell loop; the returned callable maps a full natural-scale
    parameter dict to a scalar loss.
    """
    backend = loss_backend(model_key)

    if backend == "nll":
        import jax.numpy as jnp

        nll = _nll_fn(model_key)
        order = _NLL_ORDER[model_key]
        returns = jnp.asarray(np.asarray(market_data.log_returns, dtype=np.float64))

        def _cell_nll(params: dict[str, float]) -> float:
            vec = jnp.asarray([float(params[name]) for name in order])
            return float(nll(vec, returns))

        return _cell_nll

    if objective is None:
        raise ValueError(
            f"model '{model_key}' needs an ObjectiveStrategy for its landscape"
        )

    if backend == "mc":
        paths = int(n_paths_mc or LANDSCAPE_MC_PATHS)

        # Deliberately NO stationarity penalty here, unlike the GARCH-Q
        # calibrator's objective: the SOFT penalty is zero everywhere inside
        # the stationary region and the non-stationary region is masked for
        # display (mask_nonstationary), so adding it would only change hidden
        # cells while dragging the user's constraint settings into the grid
        # cache key. The remaining loss-LEVEL shift vs the solver (reduced
        # path budget) is surfaced in the slice-diagnostics card instead.
        def _cell_mc(params: dict[str, float]) -> float:
            model = rebuild_model(model_key, params)
            if model is None:
                return float("nan")
            prices = surface_model_prices(
                model, market_data, n_paths=paths, mc_seed=MC_SEED
            )
            return float(objective.compute_loss(prices, market_data))

        return _cell_mc

    def _cell_fft(params: dict[str, float]) -> float:
        model = rebuild_model(model_key, params)
        if model is None:
            return float("nan")
        prices = price_surface(model, market_data, _engine())
        return float(objective.compute_loss(prices, market_data))

    return _cell_fft


def evaluate_slice_points(
    model_key: str,
    market_data,
    meta: dict,
    base_params: dict[str, float],
    param_x: str,
    param_y: str,
    points: tuple[tuple[float, float], ...],
    objective: ObjectiveStrategy | None,
    objective_key: str | None = None,
    n_paths_mc: int | None = None,
) -> np.ndarray:
    """Exact slice loss at arbitrary ``(x, y)`` points — the same cell loss
    the grid sweeps, with every other parameter frozen at ``base_params``.

    Used for the 3-D overlay markers (trajectory endpoints, x₀, ground
    truth): a nearest-cell grid lookup snaps a converged point onto the
    valley wall when the basin is narrower than a cell — for the GARCH
    families that snapped height can even land beyond the stationarity
    boundary, drawing the *final* fit above the *seed*. Evaluating the loss
    at the marker's own coordinates removes the quantization entirely.
    ``meta`` / ``objective_key`` mirror :func:`compute_loss_grid`'s cache
    contract (consumed by the cached wrapper, ignored here).
    """
    cell_loss = _make_cell_loss(model_key, market_data, objective, n_paths_mc)
    base = dict(base_params)
    out = np.full(len(points), np.nan, dtype=np.float64)
    for k, (xv, yv) in enumerate(points):
        params = dict(base)
        params[param_x] = float(xv)
        params[param_y] = float(yv)
        try:
            out[k] = cell_loss(params)
        except (ValueError, RuntimeError, FloatingPointError):
            out[k] = float("nan")
    return out


def compute_loss_grid(
    model_key: str,
    market_data,
    meta: dict,
    base_params: dict[str, float],
    param_x: str,
    param_y: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    objective: ObjectiveStrategy | None,
    resolution: int = 30,
    objective_key: str | None = None,
    n_paths_mc: int | None = None,
) -> LandscapeResult:
    """Sweep a ``resolution × resolution`` grid of ``(p_x, p_y)`` values.

    ``base_params`` provides the held-fixed coordinates; ``x_range`` and
    ``y_range`` set the slice extents. ``objective`` is the
    ``ObjectiveStrategy`` whose loss is evaluated at every cell — pass the
    same strategy the user selected so the landscape matches the solver's
    target. It is ignored (pass ``None``) for the returns-GARCH family, whose
    cell loss is the MLE negative log-likelihood itself. ``n_paths_mc``
    overrides the ``LANDSCAPE_MC_PATHS`` budget for the MC backend (tests).
    NaN cells correspond to model invocations that failed (e.g.
    Feller-violating Heston configurations). ``meta`` is retained for call
    compatibility with the cached wrapper; ``objective_key`` is a cache-key
    hint consumed by that wrapper and ignored by the direct computation.
    """
    if param_x == param_y:
        raise ValueError("param_x and param_y must differ to form a 2-D slice")
    x_values = np.linspace(float(x_range[0]), float(x_range[1]), int(resolution))
    y_values = np.linspace(float(y_range[0]), float(y_range[1]), int(resolution))

    base = dict(base_params)
    cell_loss = _make_cell_loss(model_key, market_data, objective, n_paths_mc)

    loss = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
    for j, yv in enumerate(y_values):
        for i, xv in enumerate(x_values):
            params = dict(base)
            params[param_x] = float(xv)
            params[param_y] = float(yv)
            try:
                loss[j, i] = cell_loss(params)
            except (ValueError, RuntimeError, FloatingPointError):
                loss[j, i] = float("nan")

    return LandscapeResult(
        param_x=param_x,
        param_y=param_y,
        x_values=x_values,
        y_values=y_values,
        loss_grid=loss,
        base_params=dict(base_params),
    )


def trajectory_points(
    history,
    param_x: str,
    param_y: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the ``(p_x, p_y)`` optimiser path from an iteration history.

    Prefers the per-iteration *best* (``source == "callback"``) snapshots when
    present (≥2), falling back to the full history otherwise. This matters for
    population solvers (DE): their per-evaluation snapshots are a cloud that
    spans the whole bound box, which — since this polyline also feeds the
    window-fit — would blow the landscape window up to the full bounds and
    flatten the surface. The callback snapshots track the population best and
    give the actual descent path. (Opposite of the convergence chart's filter,
    which deliberately shows every evaluation.) LM has no callback → falls back
    to its tightly-converging evaluation path.

    When we fall through to the callback path, the *first* callback fires
    **after** iteration 1 — already off the seed x₀ — so a polyline drawn
    from ``cb[0]`` would leave a visual gap to the ``initial guess (x₀)``
    marker rendered separately, and the enlarged first callback dot would
    read as a duplicate starting point. To close that gap we prepend the
    very first ``source == "evaluation"`` snapshot (the seed the wrapped
    objective sees on call #1) so the trajectory starts exactly at x₀.
    """
    cb = [s for s in history if getattr(s, "source", None) == "callback"]
    if len(cb) >= 2:
        first_eval = next(
            (s for s in history if getattr(s, "source", None) == "evaluation"),
            None,
        )
        snaps = [first_eval, *cb] if first_eval is not None else list(cb)
    else:
        snaps = list(history)
    xs: list[float] = []
    ys: list[float] = []
    for snap in snaps:
        nat = snap.params_natural
        if param_x in nat and param_y in nat:
            xs.append(float(nat[param_x]))
            ys.append(float(nat[param_y]))
    return np.asarray(xs), np.asarray(ys)


def initial_point_from_history(
    history,
    param_x: str,
    param_y: str,
) -> tuple[float, float] | None:
    """Return the *first evaluation* snapshot's natural (p_x, p_y).

    The previous implementation used ``ParamSpec.default`` for the
    'initial' marker, which is the registry default — not the seed the
    calibrator actually started from (ATM-IV-derived for Heston, etc.).
    Pulling the first ``source='evaluation'`` snapshot gives the true
    starting point, which is what the user expects when they read the
    contour plot.
    """
    for snap in history:
        if getattr(snap, "source", None) == "evaluation":
            nat = snap.params_natural
            if param_x in nat and param_y in nat:
                return float(nat[param_x]), float(nat[param_y])
            return None
    # No 'evaluation' snapshot — fall back to the very first snapshot.
    if history:
        nat = history[0].params_natural
        if param_x in nat and param_y in nat:
            return float(nat[param_x]), float(nat[param_y])
    return None


def slice_minimum(
    result: "LandscapeResult",
) -> tuple[float, float, float] | None:
    """Return ``(x_min, y_min, loss_min)`` — the lowest cell on the slice.

    Returns ``None`` when every cell is NaN (e.g. every model evaluation
    raised). The user-facing caption uses this to anchor the plot
    visually against a numerical value.
    """
    grid = result.loss_grid
    if not np.isfinite(grid).any():
        return None
    flat_idx = int(np.nanargmin(grid))
    j, i = np.unravel_index(flat_idx, grid.shape)
    return (
        float(result.x_values[i]),
        float(result.y_values[j]),
        float(grid[j, i]),
    )


def basin_curvature(
    result: "LandscapeResult",
) -> dict[str, float] | None:
    """Estimate the 2×2 Hessian by finite differences around the slice
    minimum.

    Uses the 9-point stencil ``(i±1, j±1)`` centred on the minimum cell.
    Returns ``{"lambda_min", "lambda_max", "kappa", "angle_deg"}``
    where ``kappa = lambda_max / lambda_min`` is the condition number
    and ``angle_deg`` is the principal axis orientation (degrees from
    the x-axis). Returns ``None`` when the minimum sits on a boundary
    or NaN cells prevent the stencil.
    """
    grid = result.loss_grid
    if not np.isfinite(grid).any():
        return None
    flat_idx = int(np.nanargmin(grid))
    j, i = np.unravel_index(flat_idx, grid.shape)
    nx, ny = grid.shape[1], grid.shape[0]
    if not (1 <= i < nx - 1 and 1 <= j < ny - 1):
        return None
    # Step sizes in natural units
    dx = float(result.x_values[1] - result.x_values[0])
    dy = float(result.y_values[1] - result.y_values[0])
    stencil = grid[j - 1 : j + 2, i - 1 : i + 2]
    if not np.isfinite(stencil).all():
        return None
    # Central finite differences for the Hessian of f(x, y)
    fxx = (stencil[1, 2] - 2 * stencil[1, 1] + stencil[1, 0]) / (dx * dx)
    fyy = (stencil[2, 1] - 2 * stencil[1, 1] + stencil[0, 1]) / (dy * dy)
    fxy = (stencil[2, 2] - stencil[2, 0] - stencil[0, 2] + stencil[0, 0]) / (
        4.0 * dx * dy
    )
    H = np.array([[fxx, fxy], [fxy, fyy]], dtype=np.float64)
    try:
        eigs = np.linalg.eigvalsh(H)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(eigs).all() or (eigs <= 0).any():
        # Not a positive-definite Hessian → not a clean basin minimum.
        return None
    lam_min = float(eigs[0])
    lam_max = float(eigs[-1])
    # Principal axis from the eigenvector of the largest eigenvalue
    try:
        _, vecs = np.linalg.eigh(H)
        v = vecs[:, -1]
        angle_deg = float(np.degrees(np.arctan2(v[1], v[0])))
    except np.linalg.LinAlgError:
        angle_deg = float("nan")
    return {
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "kappa": lam_max / max(lam_min, 1e-30),
        "angle_deg": angle_deg,
    }


def feller_boundary_segments(
    spec_lookup: dict[str, tuple[float, float]],
    param_x: str,
    param_y: str,
    base_params: dict[str, float],
    n: int = 80,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> dict[str, np.ndarray] | None:
    """Return ``(x_curve, y_curve)`` for the 2κθ = α² boundary in
    (κ, θ) / (κ, α) / (θ, α) slices. Returns ``None`` when the slice does
    not contain the relevant pair.

    When ``x_range`` / ``y_range`` are given (the visible slice window),
    the curve is **clipped** to that rectangle so it stays on the plotted
    surface instead of running off to the full parameter bounds. Density is
    bumped when clipping so the visible arc stays smooth, and ``None`` is
    returned if no point of the curve falls inside the window.
    """
    pair = {param_x, param_y}
    if not pair.issubset({"kappa", "theta", "alpha"}):
        return None

    clip = x_range is not None and y_range is not None
    n_gen = max(n, 400) if clip else n

    if pair == {"kappa", "theta"}:
        alpha = float(base_params.get("alpha", 0.0))
        kappa_lo, kappa_hi = spec_lookup["kappa"]
        kappa = np.linspace(max(kappa_lo, 1e-3), kappa_hi, n_gen)
        theta = alpha * alpha / (2.0 * kappa)
        x_arr, y_arr = (kappa, theta) if param_x == "kappa" else (theta, kappa)
    elif pair == {"kappa", "alpha"}:
        theta = float(base_params.get("theta", 0.0))
        kappa_lo, kappa_hi = spec_lookup["kappa"]
        kappa = np.linspace(max(kappa_lo, 1e-3), kappa_hi, n_gen)
        alpha = np.sqrt(np.clip(2.0 * kappa * theta, 0.0, None))
        x_arr, y_arr = (kappa, alpha) if param_x == "kappa" else (alpha, kappa)
    else:  # pair == {"theta", "alpha"}
        kappa = float(base_params.get("kappa", 0.0))
        xi_lo, xi_hi = spec_lookup["alpha"]
        alpha = np.linspace(max(xi_lo, 1e-3), xi_hi, n_gen)
        theta = alpha * alpha / (2.0 * max(kappa, 1e-3))
        x_arr, y_arr = (theta, alpha) if param_x == "theta" else (alpha, theta)

    if clip:
        mask = (
            (x_arr >= x_range[0])
            & (x_arr <= x_range[1])
            & (y_arr >= y_range[0])
            & (y_arr <= y_range[1])
        )
        if not np.any(mask):
            return None
        x_arr, y_arr = x_arr[mask], y_arr[mask]

    return {"x": x_arr, "y": y_arr}


def feller_feasible_window(
    param_x: str,
    param_y: str,
    base_params: dict[str, float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    n: int = 200,
) -> tuple[float, float, float, float] | None:
    """Bounding box of the Feller-feasible region (``2κθ ≥ α²``) inside
    ``[x_range] × [y_range]`` for a κ/θ/α slice.

    Used to crop the landscape window onto the feasible side so the infeasible
    band — which prices to NaN and renders blank — doesn't waste the plot
    (notably in Feller-HARD mode, where the optimum sits on the boundary). The
    test is purely algebraic (no pricing), so this is instant.

    Returns ``(x_lo, x_hi, y_lo, y_hi)``, or ``None`` when the pair isn't
    Feller-constrained or no cell in the window is feasible.
    """
    if not {param_x, param_y}.issubset({"kappa", "theta", "alpha"}):
        return None
    xs = np.linspace(x_range[0], x_range[1], n)
    ys = np.linspace(y_range[0], y_range[1], n)
    grid_x, grid_y = np.meshgrid(xs, ys)
    axes = {param_x: grid_x, param_y: grid_y}
    kappa = axes.get("kappa", float(base_params.get("kappa", 0.0)))
    theta = axes.get("theta", float(base_params.get("theta", 0.0)))
    alpha = axes.get("alpha", float(base_params.get("alpha", 0.0)))
    feasible = 2.0 * kappa * theta >= alpha**2
    if not np.any(feasible):
        return None
    fx, fy = grid_x[feasible], grid_y[feasible]
    return float(fx.min()), float(fx.max()), float(fy.min()), float(fy.max())


# Per-family persistence as a function of (alpha, beta, gamma) — the
# stationarity boundary is the persistence = 1 level set. P- and Q-measure
# variants share the same algebra (verified against garch_calibrator's
# stationarity penalty, the _GARCHRiskNeutralMixin caps, garch_nll's soft
# barrier, and HestonNandiGARCHModel's β + αγ² constraint).
_PERSISTENCE: dict[str, str] = {
    "garch": "garch",
    "garch_q": "garch",
    "ngarch": "ngarch",
    "ngarch_q": "ngarch",
    "gjr_garch": "gjr",
    "gjr_q": "gjr",
    "heston_nandi": "heston_nandi",
}

# LaTeX-ish legend label per persistence form (UI strings, English).
STATIONARITY_LABELS: dict[str, str] = {
    "garch": "stationarity boundary  α + β = 1",
    "ngarch": "stationarity boundary  β + α(1+γ²) = 1",
    "gjr": "stationarity boundary  α + β + γ/2 = 1",
    "heston_nandi": "stationarity boundary  β + αγ² = 1",
}


def stationarity_label(model_key: str) -> str | None:
    """Legend label of the model's stationarity boundary, or ``None``."""
    form = _PERSISTENCE.get(model_key)
    return STATIONARITY_LABELS[form] if form else None


def _persistence_terms(form: str, alpha, beta, gamma):
    """Persistence value for arrays/scalars of (α, β, γ) under one form."""
    if form == "garch":
        return alpha + beta
    if form == "ngarch":
        return beta + alpha * (1.0 + gamma**2)
    if form == "gjr":
        return alpha + beta + 0.5 * gamma
    return beta + alpha * gamma**2  # heston_nandi


def mask_nonstationary(
    result: LandscapeResult,
    model_key: str,
    *,
    threshold: float = 1.0,
) -> LandscapeResult:
    """Copy of ``result`` with cells at persistence ≥ ``threshold`` set to NaN.

    The NLL's stationarity soft barrier makes non-stationary cells reach
    10⁶-10⁷ while the basin sits around −6500 — one wall cell crushes the
    whole colour/z scale into a flat sheet (the NLL view cannot log-scale:
    the likelihood is negative). Masking the infeasible region — the GARCH
    analogue of the Feller-infeasible crop for Heston/Bates — restores the
    basin. α/β/γ come from the slice axes when swept, else from the frozen
    ``base_params``, so hidden-parameter slices (e.g. ω, β) mask correctly.

    :func:`compute_loss_grid` deliberately stays unmasked (its unit-level
    contract and the app cache key don't know about constraints); callers
    mask for display. Models without a persistence form pass through.
    """
    form = _PERSISTENCE.get(model_key)
    if form is None:
        return result
    grid_x, grid_y = np.meshgrid(result.x_values, result.y_values)
    axes = {result.param_x: grid_x, result.param_y: grid_y}

    def _coord(name: str):
        return axes.get(name, float(result.base_params.get(name, 0.0)))

    persistence = np.broadcast_to(
        np.asarray(
            _persistence_terms(form, _coord("alpha"), _coord("beta"), _coord("gamma")),
            dtype=float,
        ),
        result.loss_grid.shape,
    )
    return replace(
        result,
        loss_grid=np.where(persistence >= threshold, np.nan, result.loss_grid),
    )


def stationarity_boundary_segments(
    model_key: str,
    spec_lookup: dict[str, tuple[float, float]],
    param_x: str,
    param_y: str,
    base_params: dict[str, float],
    n: int = 400,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> dict[str, np.ndarray] | None:
    """``(x, y)`` polyline of the persistence = 1 boundary in an (α, β, γ) slice.

    The GARCH analogue of :func:`feller_boundary_segments` — same window
    clipping contract. The third persistence parameter (if any) is frozen at
    ``base_params``; ω / h₀ never enter persistence, so slices involving them
    get no curve (the boundary would be a constant line in a direction the
    pedagogy panel already explains). Returns ``None`` when the model has no
    persistence constraint, the pair isn't persistence-bound, or no point of
    the curve falls inside the window.
    """
    form = _PERSISTENCE.get(model_key)
    if form is None:
        return None
    persistence_vars = (
        {"alpha", "beta"} if form == "garch" else {"alpha", "beta", "gamma"}
    )
    pair = {param_x, param_y}
    if not pair.issubset(persistence_vars):
        return None

    clip = x_range is not None and y_range is not None
    n_gen = max(n, 400) if clip else n

    if param_x in spec_lookup:
        x_lo, x_hi = spec_lookup[param_x]
    else:
        return None
    x_arr = np.linspace(float(x_lo), float(x_hi), n_gen)

    def _solve_other(x_vals: np.ndarray) -> np.ndarray:
        """Solve persistence(α, β, γ) = 1 for ``param_y`` given ``param_x``."""
        fixed = {
            name: float(base_params.get(name, 0.0))
            for name in ("alpha", "beta", "gamma")
        }
        coords = dict(fixed)
        coords[param_x] = x_vals
        if param_y == "beta":
            # persistence is linear in β with unit coefficient in every form.
            return 1.0 - (
                _persistence_terms(form, coords["alpha"], 0.0, coords["gamma"])
            )
        if param_y == "alpha":
            denom = {
                "garch": 1.0,
                "ngarch": 1.0 + coords["gamma"] ** 2,
                "gjr": 1.0,
                "heston_nandi": max(coords["gamma"] ** 2, 1e-12),
            }[form]
            extra = 0.5 * coords["gamma"] if form == "gjr" else 0.0
            return (1.0 - coords["beta"] - extra) / denom
        # param_y == "gamma"
        alpha = np.maximum(np.asarray(coords["alpha"], dtype=float), 1e-12)
        if form == "ngarch":
            return np.sqrt(np.clip((1.0 - coords["beta"]) / alpha - 1.0, 0.0, None))
        if form == "gjr":
            return 2.0 * (1.0 - coords["alpha"] - coords["beta"])
        # heston_nandi
        return np.sqrt(np.clip((1.0 - coords["beta"]) / alpha, 0.0, None))

    with np.errstate(invalid="ignore", divide="ignore"):
        y_arr = np.broadcast_to(
            np.asarray(_solve_other(x_arr), dtype=float), x_arr.shape
        ).copy()

    finite = np.isfinite(y_arr)
    if not np.any(finite):
        return None
    x_arr, y_arr = x_arr[finite], y_arr[finite]

    if clip:
        mask = (
            (x_arr >= x_range[0])
            & (x_arr <= x_range[1])
            & (y_arr >= y_range[0])
            & (y_arr <= y_range[1])
        )
        if not np.any(mask):
            return None
        x_arr, y_arr = x_arr[mask], y_arr[mask]

    return {"x": x_arr, "y": y_arr}
