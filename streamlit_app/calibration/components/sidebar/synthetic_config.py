"""Synthetic-data configuration panel — surface grid OR returns sample."""

from __future__ import annotations

from typing import Any

import streamlit as st

from config.constants import (
    DEFAULT_RETURNS_CONFIG,
    DEFAULT_SURFACE_CONFIG,
    model_data_mode,
)
from services import state_manager


def _render_surface_config(cfg: dict) -> dict:
    c1, c2 = st.columns(2)
    with c1:
        cfg["n_strikes"] = st.slider(
            "# Strikes", 5, 30, int(cfg["n_strikes"]),
            help="Number of strike points across the synthetic surface. "
                 "More strikes → smoother smile but more quotes to fit.",
        )
        cfg["spot"] = st.number_input(
            "Spot S₀", value=float(cfg["spot"]), step=1.0,
            help="Synthetic underlying price. Strike grid is read in dollars.",
        )
        cfg["rate"] = st.number_input(
            "Risk-free rate r", value=float(cfg["rate"]), step=0.005, format="%.4f",
            help="Annual continuously-compounded risk-free rate.",
        )
    with c2:
        cfg["n_maturities"] = st.slider(
            "# Maturities", 3, 12, int(cfg["n_maturities"]),
            help="Number of maturity slices on the surface.",
        )
        cfg["dividend_yield"] = st.number_input(
            "Dividend yield q",
            value=float(cfg["dividend_yield"]), step=0.005, format="%.4f",
            help="Annual continuous dividend yield used in forward pricing.",
        )
        cfg["seed"] = st.number_input(
            "Seed", min_value=0, max_value=99999, value=int(cfg["seed"]), step=1,
            help="RNG seed for the synthetic noise. Same seed + same params "
                 "= reproducible quotes.",
        )
    cfg["strike_min"], cfg["strike_max"] = st.slider(
        "Strike range ($)",
        50.0, 200.0,
        value=(float(cfg["strike_min"]), float(cfg["strike_max"])),
        step=1.0,
        help="Strike bounds for the synthetic surface. Asymmetric ranges "
             "(say 80-130 with S₀=100) bias the calibration toward the "
             "wing the range emphasises.",
    )
    cfg["maturity_min"], cfg["maturity_max"] = st.slider(
        "Maturity range (years)", 0.05, 5.0,
        value=(float(cfg["maturity_min"]), float(cfg["maturity_max"])),
        step=0.05,
        help="Maturity bounds. Very short tenors stress the variance "
             "process (Heston small-time behaviour); long tenors stress θ.",
    )
    cfg["noise_std"] = st.slider(
        "Bid-ask noise", 0.0, 0.05, float(cfg["noise_std"]),
        # 4-decimal step + 4-decimal format keeps the slider value and the
        # displayed number in lockstep. The previous 3/3 combo rounded the
        # label to "0.000" while the underlying value still moved.
        step=0.0005, format="%.4f",
        help="Proportional Gaussian noise on prices (relative standard deviation).",
    )
    cfg["_mode"] = "surface"
    return cfg


def _render_returns_config(cfg: dict) -> dict:
    cfg["n_periods"] = st.slider(
        "# Periods (days)", 200, 5000, int(cfg["n_periods"]), step=50,
        help="Length of the synthetic log-return series. GARCH MLE needs "
             "≥ ~500 observations for stable parameter estimates.",
    )
    cfg["annualization_factor"] = st.selectbox(
        "Annualisation", options=[252, 12, 4, 1],
        index=[252, 12, 4, 1].index(int(cfg["annualization_factor"])),
        help="Periods per year — 252 = daily, 52 = weekly, 12 = monthly. "
             "Affects the σ-annualised display only, not the fit itself.",
    )
    c1, c2 = st.columns(2)
    with c1:
        cfg["spot"] = st.number_input(
            "Initial S₀", value=float(cfg["spot"]), step=1.0,
            help="Starting price level for the simulated path.",
        )
    with c2:
        cfg["drift"] = st.number_input(
            "Drift μ", value=float(cfg["drift"]), step=0.005, format="%.4f",
            help="Annualised expected log-return. GARCH families filter out "
                 "the mean before fitting variance so this rarely matters.",
        )
    cfg["seed"] = st.number_input(
        "Seed", min_value=0, max_value=99999, value=int(cfg["seed"]), step=1,
        help="RNG seed for the simulated returns. Same seed = reproducible.",
    )
    cfg["_mode"] = "returns"
    return cfg


def render(model_key: str) -> dict[str, Any]:
    mode = model_data_mode(model_key)
    st.subheader("⚙️ Synthetic Data Config")

    stored = state_manager.get("calib_data_config") or {}
    default = DEFAULT_SURFACE_CONFIG if mode == "surface" else DEFAULT_RETURNS_CONFIG
    cfg = dict(stored or default)
    if cfg.get("_mode") != mode:
        cfg = dict(default)

    if mode == "surface":
        cfg = _render_surface_config(cfg)
    else:
        cfg = _render_returns_config(cfg)

    state_manager.set("calib_data_config", cfg)
    return cfg
