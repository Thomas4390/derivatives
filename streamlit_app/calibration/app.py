"""
Model Calibration Explorer — "Quant Lab Terminal"
==================================================

Interactive pedagogical interface for stochastic-model calibration:

* Generate a synthetic ground-truth surface OR load real SPX market data
* Calibrate one or several solvers (LM-JAX, DE, NM, L-BFGS-B) and watch
  the loss converge in real time
* Compare solvers on accuracy / speed / parameter recovery
* Inspect post-fit diagnostics (residual heatmap, parameter
  correlations, QQ-plot)

Author: Thomas Vaudescal
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# --- JAX cache & XLA noise suppression -----------------------------------
_JAX_CACHE_DIR = os.environ.get(
    "JAX_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "jax_cache_derivatives"),
)
# Streamlit re-executes this script on every rerun. Deleting the cache here
# (the old behaviour) wiped it mid-session, and since the makedirs in
# backend._jax_config only runs once at import, JAX then wrote compiled
# entries into a now-missing directory → "FileNotFoundError … writing
# persistent compilation cache entry". Just *ensure the directory exists*
# on every rerun and let the (content-addressed) cache persist — stale
# entries are harmless and a warm cache skips recompilation.
os.makedirs(_JAX_CACHE_DIR, exist_ok=True)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "0")

# --- Path bootstrap -----------------------------------------------------
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

import streamlit as st

from components import glossary
from components.live_progress import run_multi_model_with_live_progress
from components.sidebar import render_sidebar
from config.constants import (
    RETURNS_CACHE_MAX_ENTRIES,
    SNAPSHOT_CACHE_MAX_ENTRIES,
    SURFACE_CACHE_MAX_ENTRIES,
)
from services import (
    data_loader,
    landscape_service,
    real_data_service,
    state_manager,
    synthetic_data_service,
)
from streamlit_app.simulation.config.styles import (  # type: ignore
    inject_styles,
    render_compact_header,
)
from tabs import (
    render_comparison,
    render_diagnostics,
    render_landscape,
    render_live,
    render_setup,
    render_theory,
)

# Wrap the framework-free service functions with Streamlit's data cache so
# successive reruns reuse the same surface/returns/snapshot when arguments are
# unchanged. The services themselves stay free of `import streamlit`.
_load_snapshot_cached = st.cache_data(
    show_spinner=False, max_entries=SNAPSHOT_CACHE_MAX_ENTRIES,
)(real_data_service.load_snapshot)
_generate_surface_cached = st.cache_data(
    show_spinner=False, max_entries=SURFACE_CACHE_MAX_ENTRIES,
)(synthetic_data_service.generate_surface)
_generate_returns_cached = st.cache_data(
    show_spinner=False, max_entries=RETURNS_CACHE_MAX_ENTRIES,
)(synthetic_data_service.generate_returns)

# Loss-landscape cache. The mesh sweep is expensive (400-3600 model
# repricings) and the user is likely to flip Log scale / 3-D view / zoom
# toggles repeatedly with the same underlying compute — caching the
# numeric grid keeps that flow snappy.
#
# Cache key is built from short hashable proxies (the bundle's
# precomputed ``market_data_hash`` + a tuple of sorted base_params
# items) rather than re-pickling the full IV grid / ``meta`` dict on
# every lookup. ``_market_data`` / ``_meta`` carry a leading underscore
# so ``st.cache_data`` skips them when computing the key — they are
# only forwarded to the wrapped service call.
@st.cache_data(
    show_spinner="Computing loss landscape…",
    max_entries=16,
)
def _cached_compute_loss_grid_inner(
    *,
    model_key: str,
    market_data_hash: str,
    base_params_items: tuple[tuple[str, float], ...],
    param_x: str,
    param_y: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    objective_key: str,
    resolution: int,
    _market_data,
    _meta,
    _objective,
):
    return landscape_service.compute_loss_grid(
        model_key=model_key,
        market_data=_market_data,
        meta=_meta,
        base_params=dict(base_params_items),
        param_x=param_x,
        param_y=param_y,
        x_range=x_range,
        y_range=y_range,
        objective=_objective,
        resolution=resolution,
    )


def _compute_loss_grid_cached(
    *,
    model_key: str,
    market_data,
    meta: dict,
    base_params: dict[str, float],
    param_x: str,
    param_y: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    objective,
    objective_key: str,
    resolution: int,
):
    """Adapter preserving the legacy call signature on top of the
    underscored-args cached implementation.

    ``objective_key`` (objective name + settings digest) goes into the
    cache key so switching objective busts the grid; the ``_objective``
    strategy object itself is excluded from the key (underscored) and only
    forwarded to the service call."""
    market_data_hash = str(meta.get("market_data_hash", ""))
    base_items = tuple(sorted((k, float(v)) for k, v in base_params.items()))
    return _cached_compute_loss_grid_inner(
        model_key=model_key,
        market_data_hash=market_data_hash,
        base_params_items=base_items,
        param_x=param_x,
        param_y=param_y,
        x_range=(float(x_range[0]), float(x_range[1])),
        y_range=(float(y_range[0]), float(y_range[1])),
        objective_key=objective_key,
        resolution=int(resolution),
        _market_data=market_data,
        _meta=meta,
        _objective=objective,
    )


# ======================================================================
# Page setup
# ======================================================================

st.set_page_config(
    page_title="Model Calibration Explorer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()
state_manager.init_state()

results = state_manager.get("calib_results") or {}
header_badge = (
    f"{len(results)} solver{'s' if len(results) > 1 else ''} · ready to inspect"
    if results else "Educational Tool"
)
render_compact_header(
    title="Model Calibration Explorer",
    subtitle="Watch stochastic-model calibration converge in real time — "
             "compare LM-JAX, Differential Evolution, Nelder-Mead and L-BFGS-B "
             "on synthetic ground truth or real SPX option chains.",
    badge=header_badge,
)

# One-shot banner consumed when the previous script run ended because
# the user pressed the ⏹ Stop button. The flag is flipped by the Stop
# button's on_click callback in components.live_progress; clear it as
# soon as we have shown the message so it doesn't reappear on every
# subsequent rerun.
if state_manager.get("calib_was_cancelled"):
    st.warning(
        "Calibration stopped — kept each interrupted solver's best-so-far "
        "parameters (marked **⏸ stopped** below).",
        icon="⏹",
    )
    state_manager.set("calib_was_cancelled", False)

# Glossary popover lives in a thin row right under the header so the
# user can pop it open from any tab without scrolling.
_glossary_col, _ = st.columns([1, 6])
with _glossary_col:
    glossary.render()

# ======================================================================
# Sidebar
# ======================================================================

ctx = render_sidebar()
data_family = ctx["data_family"]
generator_model = ctx["generator_model"]
candidate_models = ctx["candidate_models"]
data_source = ctx["data_source"]
true_params = ctx["true_params"]
data_config = ctx["data_config"]
solvers = ctx["solvers"]
solver_settings = ctx["solver_settings"]
objectives = ctx.get("objectives") or ()
objective_settings = ctx.get("objective_settings") or {}
constraint_settings = ctx.get("constraint_settings") or {}
run_clicked = ctx["run_clicked"]
mode = data_family  # data family already encodes the calibration input shape


# ======================================================================
# Data hash — regenerate if any sidebar choice changed
# ======================================================================

data_hash_payload = {
    "family": data_family,
    "generator": generator_model,
    "candidates": tuple(candidate_models),
    "data_source": data_source,
    "true_params": true_params if data_source == "synthetic" else None,
    "data_config": {k: v for k, v in data_config.items() if not k.startswith("_")},
}
data_hash = state_manager.hash_config(data_hash_payload)
if data_hash != state_manager.get("calib_data_hash"):
    state_manager.update(calib_data_hash=data_hash)
    state_manager.update(calib_market_data=None)
    state_manager.reset_results()


def _ensure_data():
    """Materialise the current synthetic or real market data (memoised in session_state)."""
    if state_manager.get("calib_market_data") is not None:
        return (
            state_manager.get("calib_market_data"),
            state_manager.get("calib_market_data_meta"),
        )

    with st.spinner("Loading market data…"):
        if data_source == "real":
            real_kwargs = {k: v for k, v in data_config.items() if not k.startswith("_")}
            try:
                snap = _load_snapshot_cached(**real_kwargs)
                bundle = data_loader.from_real_snapshot(snap)
            except data_loader.InvalidMarketData as exc:
                st.error(
                    f"SPX snapshot contains non-finite quotes: {exc}. "
                    "The snapshot is corrupted or the moneyness / DTE filter "
                    "is too aggressive — try widening the range in the sidebar."
                )
                return None, None
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load SPX snapshot: {exc}")
                return None, None
        elif mode == "surface":
            try:
                sd = _generate_surface_cached(
                    model_key=generator_model,
                    true_params=true_params,
                    **{k: data_config[k] for k in (
                        "spot", "rate", "dividend_yield",
                        "n_strikes", "n_maturities",
                        "strike_min", "strike_max",
                        "maturity_min", "maturity_max",
                        "noise_std", "seed",
                    )},
                )
                bundle = data_loader.from_surface(sd, data_config)
            except data_loader.InvalidMarketData as exc:
                st.error(
                    f"Synthetic surface contains non-finite IV cells: {exc}. "
                    "The true parameters likely sit in a pathological region "
                    "(Feller violation, extreme jump intensity) — adjust them "
                    "in the sidebar before re-running."
                )
                return None, None
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                st.error(
                    f"Could not generate synthetic surface for "
                    f"`{generator_model}`: {type(exc).__name__}: {exc}. "
                    "Try adjusting the true parameters or surface bounds in "
                    "the sidebar."
                )
                return None, None
        else:
            try:
                rd = _generate_returns_cached(
                    garch_type=generator_model,
                    true_params=true_params,
                    **{k: data_config[k] for k in (
                        "n_periods", "annualization_factor",
                        "spot", "drift", "seed",
                    )},
                )
                bundle = data_loader.from_returns(rd, data_config)
            except data_loader.InvalidMarketData as exc:
                st.error(
                    f"Synthetic returns series contains non-finite values: "
                    f"{exc}. The GARCH parameters likely violate stationarity "
                    "(α + β ≥ 1) — adjust them in the sidebar before re-running."
                )
                return None, None
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                st.error(
                    f"Could not generate synthetic returns for "
                    f"`{generator_model}`: {type(exc).__name__}: {exc}. "
                    "Try adjusting the GARCH parameters in the sidebar."
                )
                return None, None

        state_manager.update(
            calib_market_data=bundle.market_data,
            calib_market_data_meta=bundle.meta,
        )
        return bundle.market_data, bundle.meta


# ======================================================================
# Calibration trigger
# ======================================================================

# Stable placeholder reserved BEFORE the tabs so the DOM ordering does not
# change between "calibration running" and "idle" reruns. Without this, the
# st.status widget rendered by run_solvers_with_live_progress only exists
# during the run, which shifts the tab container's position and forces
# st.tabs() back to its first index on the next rerun.
live_run_slot = st.empty()

if run_clicked:
    blocking_error = ctx.get("blocking_error")
    if blocking_error:
        # Defensive check — the run button itself is disabled when
        # blocking_error is set, so reaching this branch implies a stale
        # click captured before the error appeared.
        st.error(blocking_error)
    elif not solvers:
        st.error("No solver selected — pick at least one in the sidebar.")
    elif not candidate_models:
        st.error("No candidate model selected — pick at least one in the sidebar.")
    else:
        market_data, _ = _ensure_data()
        if market_data is None:
            # _ensure_data already surfaced the underlying error via
            # st.error — no need to add a generic banner that drowns it.
            pass
        else:
            # Only the generator's model has a known ground truth; other
            # candidates can still record a recovery error against their
            # own defaults, but it's not pedagogically meaningful.
            truth_per_model = {generator_model: dict(true_params)} if (
                generator_model is not None and true_params
            ) else {}
            with live_run_slot.container():
                results = run_multi_model_with_live_progress(
                    candidate_models=tuple(candidate_models),
                    market_data=market_data,
                    true_params_per_model=truth_per_model,
                    solver_names=tuple(solvers),
                    objective_names=tuple(objectives) or ("price_mse",),
                    objective_settings=dict(objective_settings),
                    constraint_settings=dict(constraint_settings),
                    n_restarts=int(solver_settings.get("n_restarts", 5)),
                    max_nfev=int(solver_settings.get("max_nfev", 200)),
                    de_seed=int(solver_settings.get("de_seed", 42)),
                )
            first_model = next(iter(results), None)
            first_solver = (
                next(iter(results[first_model])) if first_model is not None else None
            )
            first_objective = None
            if first_model is not None and first_solver is not None:
                per_objective = results[first_model][first_solver]
                if isinstance(per_objective, dict):
                    first_objective = next(iter(per_objective), None)
            state_manager.update(
                calib_results=results,
                calib_active_model=first_model,
                calib_active_solver=first_solver,
                calib_active_objective=first_objective,
            )
            state_manager.mark_run_complete({
                "data_hash": data_hash,
                "candidates": tuple(candidate_models),
                "solvers": tuple(solvers),
                "objectives": tuple(objectives) or ("price_mse",),
                "solver_settings": solver_settings,
                "objective_settings": objective_settings,
                "constraint_settings": constraint_settings,
                "de_seed": int(solver_settings.get("de_seed", 42)),
            })


# ======================================================================
# Tabs
# ======================================================================

TAB_LABELS = (
    "🌐 Setup & Data",
    "▶️ Live Calibration",
    "🗺️ Loss Landscape",
    "🔬 Diagnostics",
    "⚖️ Compare & Restarts",
    "📚 Theory",
)
tab_setup, tab_live, tab_land, tab_diag, tab_cmp, tab_theory = st.tabs(list(TAB_LABELS))

tab_ctx = {
    "data_family": data_family,
    "generator_model": generator_model,
    "candidate_models": candidate_models,
    # Back-compat: many existing tabs read ``model_key`` and ``mode``.
    # We keep them populated with the *generator* (synthetic) or the
    # currently-active candidate (real), so single-model tabs continue
    # to work while we migrate them to the multi-model schema.
    "model_key": generator_model or (candidate_models[0] if candidate_models else None),
    "mode": mode,
    "data_source": data_source,
    "true_params": true_params,
    "data_config": data_config,
    "ensure_data": _ensure_data,
    "compute_loss_grid": _compute_loss_grid_cached,
}

with tab_setup:
    render_setup(tab_ctx)
with tab_live:
    render_live(tab_ctx)
with tab_land:
    render_landscape(tab_ctx)
with tab_diag:
    render_diagnostics(tab_ctx)
with tab_cmp:
    render_comparison(tab_ctx)
with tab_theory:
    render_theory(tab_ctx)
