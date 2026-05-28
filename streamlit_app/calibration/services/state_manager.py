"""
Session-state Container & Hash-based change detection
======================================================

Centralises every key read/written to ``st.session_state`` so the rest
of the app interacts with a typed surface.  Includes a configuration-
hashing helper used to decide whether the cached calibration result is
still valid for the current sidebar choices.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import streamlit as st

# Keys we own under st.session_state — kept minimal to avoid collisions
# with other Streamlit pages.

DEFAULT_STATE: dict[str, Any] = {
    # ── User selections ──
    # Data family decides which models are eligible: surface models
    # (Heston/Merton/Bates/iv_gbm) consume an IV grid, returns models
    # (GARCH/NGARCH/GJR) consume a log-return series. Switching family
    # resets generator + candidates to family defaults.
    "calib_data_family": "surface",
    # Synthetic mode only: the model whose `true_params` produce the
    # ground-truth market data. None in real-data mode.
    "calib_generator_model": "heston",
    # Tuple of model keys to calibrate against the (synthetic or real)
    # market data. Always ≥ 1.
    "calib_candidate_models": ("heston",),
    "calib_true_params": {},
    "calib_data_config": {},
    # Default data source on first load. Synthetic shows the analytic
    # ground-truth surface from the user's chosen parameters, which is
    # the most pedagogically useful starting view for a fresh session.
    "calib_data_source": "synthetic",
    "calib_solver_selection": ("LM-JAX",),
    "calib_solver_settings": {
        "max_nfev": 200,
        "n_restarts": 3,
        "tol": 1e-10,
        "init_perturbation": 0.5,
        # Seed forwarded to DifferentialEvolution (the only stochastic
        # solver here). Exposed in the sidebar so students can observe
        # how DE's path varies with the seed — a core pedagogical point
        # the previous hard-coded `seed=42` hid.
        "de_seed": 42,
    },
    # Objective function multi-selection (mirrors solver selection).
    "calib_objective_selection": ("price_mse",),
    "calib_objective_settings": {
        "huber_delta": 0.05,
        "relative_use_log": False,
        "fallback_iv": 0.20,
    },
    # Optimisation constraints. ``feller_mode`` ∈ {off, soft, hard} decides how
    # the Feller condition 2κθ > ξ² is enforced for Heston/Bates; ``feller_weight``
    # is the soft-penalty weight (only used in soft mode). Default = legacy soft@1000.
    "calib_constraint_settings": {
        "feller_mode": "soft",
        "feller_weight": 1000.0,
    },
    # ── Derived data ──
    "calib_market_data": None,           # OptionMarketData | HistoricalReturns
    "calib_market_data_meta": {},        # auxiliary info: ivs, true model
    # Nested results: dict[model_key, dict[solver_name, CalibrationRunSummary]]
    # (post multi-model refactor). Empty when no run has executed yet.
    "calib_results": {},
    "calib_active_model": None,          # which model the inspection tabs focus on
    "calib_active_solver": None,         # within active_model
    "calib_active_objective": None,      # within active_model × active_solver
    "calib_data_hash": None,             # hash of the current data config
    "calib_results_hash": None,          # hash of the calibration config
    # Wall-clock timestamp (UTC) of the last completed multi-model run.
    # Surfaced as a small caption in the Compare tab so students can tell
    # at a glance whether sidebar tweaks have invalidated the on-screen
    # numbers — pedagogical only, never persisted to disk.
    "calib_last_run_ts": None,
    # ── UI tabs ──
    "calib_active_tab": 0,
    # Hard validation: when set, the run button is disabled and the message
    # is shown to the user. Used by true_params for stationarity violations
    # that would make the calibration meaningless.
    "calib_blocking_error": None,
    # One-shot flag flipped by the ⏹ Stop button's on_click callback so
    # the next script run can render a "calibration cancelled" banner
    # before clearing the flag. The actual cancellation signal travels
    # to the worker threads via LiveRunHandle.cancel_event, not here.
    "calib_was_cancelled": False,
}


def init_state() -> None:
    """Initialise any missing keys under st.session_state."""
    for key, default in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get(key: str) -> Any:
    return st.session_state.get(key)


def set(key: str, value: Any) -> None:  # noqa: A001 — shadowing built-in OK in service
    st.session_state[key] = value


def update(**kwargs: Any) -> None:
    for k, v in kwargs.items():
        st.session_state[k] = v


def hash_config(payload: dict | tuple | list) -> str:
    """Stable hash for change-detection in the sidebar."""

    def _default(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    blob = json.dumps(payload, sort_keys=True, default=_default).encode()
    return hashlib.md5(blob).hexdigest()


def reset_results() -> None:
    """Clear cached calibration results — call when data config changes."""
    st.session_state["calib_results"] = {}
    st.session_state["calib_active_model"] = None
    st.session_state["calib_active_solver"] = None
    st.session_state["calib_active_objective"] = None
    st.session_state["calib_results_hash"] = None
    st.session_state["calib_last_run_ts"] = None
    st.session_state["calib_market_data_meta"] = {}


def mark_run_complete(run_config: dict[str, Any]) -> None:
    """Stamp the session with completion time + config hash.

    Called once the multi-model orchestration has finished writing
    ``calib_results``. The hash captures solver / objective config so
    the caption in the Compare tab can distinguish a fresh run from a
    stale display where the user has since moved a slider.
    """
    st.session_state["calib_last_run_ts"] = datetime.now(timezone.utc)
    st.session_state["calib_results_hash"] = hash_config(run_config)
