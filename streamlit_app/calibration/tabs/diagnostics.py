"""Tab 4 — Diagnostics: residual heatmap, QQ-plot, parameter uncertainty.

Overlays / small-multiples every selected ``(model, solver, objective)``
run (via ``series_view_filter``) instead of inspecting one at a time:

* residual heatmaps and the squared-residual ACF can't share a plane, so
  they render as **small multiples** (one mini-chart per run);
* QQ-plots and the conditional-volatility series are **superimposed** with
  a legend;
* the uncertainty / recovery table gets one block of rows per run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from backend.utils.logging import get_logger
from charts.diagnostics import (
    render_qq_overlay,
    render_residual_heatmap,
    uncertainty_table,
)
from charts.iv_surface import (
    render_iv_surface_3d,
    render_iv_surface_overlay_3d,
    render_smile_slices,
    render_smile_slices_overlay,
)
from charts.returns_diagnostics import (
    render_conditional_volatility_overlay,
    render_squared_residuals_acf,
)
from config.constants import (
    GARCH_FAMILY,
    RETURNS_CACHE_MAX_ENTRIES,
    RETURNS_IMPLIED_HALFWIDTH_MAX,
    RETURNS_IMPLIED_HALFWIDTH_MIN,
    RETURNS_IMPLIED_STRIKE_SPAN_SIGMAS,
    RETURNS_IMPLIED_SURFACE_RATE,
)
from services import state_manager
from services.axis_display import resolve_display_axis
from streamlit_app.simulation.config.styles import section_header_html  # type: ignore
from tabs._helpers import (
    Series,
    facet_grid,
    reference_surface_label,
    series_view_filter,
)
from utils.numba_kernels import reshape_residuals_to_grid
from utils.plotly_theme import series_style

logger = get_logger(__name__)


def render(ctx: dict) -> None:
    mode = ctx["mode"]
    data_source = ctx["data_source"]
    true_params = ctx["true_params"]
    generator_model = ctx.get("generator_model")
    ensure_data = ctx["ensure_data"]

    results = state_manager.get("calib_results") or {}
    if not results:
        st.info("Run a calibration to populate this view.")
        return

    selected = series_view_filter(results, key="diag")
    if not selected:
        st.info("Select at least one run above to display its diagnostics.")
        return

    market_data, meta = ensure_data()
    if market_data is None:
        st.info("Reload the data tab once the market data is available.")
        return

    multi_model = len({s.model for s in selected}) > 1

    if mode == "surface":
        _render_surface_diagnostics(selected, market_data, meta, multi_model)
    else:
        _render_returns_diagnostics(
            selected,
            meta,
            multi_model,
            data_source=data_source,
            generator_model=generator_model,
            true_params=true_params,
        )

    _render_uncertainty_tables(selected, true_params, data_source, generator_model)


# ──────────────────────────────────────────────────────────────────────
# Surface-family diagnostics
# ──────────────────────────────────────────────────────────────────────


def _surface_residuals_1d(res, market_data) -> np.ndarray | None:
    """Flat ``model − market`` price residuals for one fit, or ``None`` when
    re-pricing the surface fails (narrowed to the pricing-pipeline errors so
    a genuine bug upstream still surfaces)."""
    try:
        from services.post_calibration import surface_model_prices

        model_prices = surface_model_prices(res.model, market_data)
        return (model_prices - market_data.market_prices).astype(np.float64)
    except (ValueError, RuntimeError, FloatingPointError, AttributeError):
        logger.debug("surface residual computation failed", exc_info=True)
        return None


def _render_surface_diagnostics(
    selected: list[Series],
    market_data,
    meta: dict,
    multi_model: bool,
) -> None:
    # Re-price each selected fit once; reuse the residuals for both charts.
    resids_by_key: dict[str, np.ndarray] = {}
    for s in selected:
        resids = _surface_residuals_1d(s.summary.result, market_data)
        if resids is not None:
            resids_by_key[s.key] = resids

    if not resids_by_key:
        st.warning(
            "Residual diagnostics unavailable for the selected runs. "
            "Try a different solver or re-run after adjusting the data config."
        )
        return

    quote_strikes = market_data.strikes.astype(np.float64)
    quote_maturities = market_data.maturities.astype(np.float64)
    grid_strikes = np.asarray(meta["strikes"], dtype=np.float64)
    grid_maturities = np.asarray(meta["maturities"], dtype=np.float64)

    st.markdown(
        section_header_html("🔥", "Residual heatmaps · model − market"),
        unsafe_allow_html=True,
    )

    # 1D-safe display axis: honours the x-axis choice for real data (shared
    # strikes) and falls back to σ√T-moneyness when the choice is a 2D
    # per-maturity axis (a heatmap needs a single shared x per column).
    axis = resolve_display_axis(meta)
    heatmap_axis = axis.heatmap_kwargs(meta)

    def _heatmap_for(s: Series):
        resids = resids_by_key.get(s.key)
        if resids is None:
            return None
        grid = reshape_residuals_to_grid(
            resids, quote_strikes, quote_maturities, grid_strikes, grid_maturities
        )
        return render_residual_heatmap(
            grid,
            meta["strikes"],
            meta["maturities"],
            **heatmap_axis,
            spot=meta.get("spot"),
        )

    facet_grid(
        [s for s in selected if s.key in resids_by_key],
        _heatmap_for,
        key_prefix="diag_resid",
    )

    st.markdown(
        section_header_html("📐", "Residual normality · QQ vs N(0, 1)"),
        unsafe_allow_html=True,
    )
    labeled = [
        (
            s.label,
            resids_by_key[s.key],
            series_style(s.model, s.solver, multi_model=multi_model, index=i),
        )
        for i, s in enumerate(selected)
        if s.key in resids_by_key
    ]
    st.plotly_chart(
        render_qq_overlay(labeled, standardise=True), width="stretch", key="diag_qq"
    )


# ──────────────────────────────────────────────────────────────────────
# Returns-family (GARCH) diagnostics
# ──────────────────────────────────────────────────────────────────────


def _garch_filter(res, model_key: str, meta: dict):
    """Run the variance filter at one calibrated optimum.

    Returns ``(variance_series, z_residuals)`` or ``(None, None)`` on
    failure. ``ω`` is stored annualised, so divide it back by the
    annualization factor before filtering on the per-period returns (the
    GARCH scale convention — see the info box rendered above).
    """
    try:
        from backend.calibration.garch_calibrator import _filter_variance
    except ImportError:
        return None, None

    model = res.model
    log_returns = np.asarray(meta["log_returns"], dtype=np.float64)
    annualization_factor = float(meta["annualization_factor"])
    omega_per_period = float(model.omega) / annualization_factor
    params = [omega_per_period, model.alpha, model.beta]
    if model_key == "ngarch":
        params.append(getattr(model, "gamma", 0.0))
    elif model_key == "gjr_garch":
        params.append(getattr(model, "gamma", 0.0))
    try:
        var_series = _filter_variance(
            model_key, np.asarray(params, dtype=np.float64), log_returns
        )
    except (ValueError, RuntimeError, FloatingPointError):
        return None, None
    # The conditional variance for r_t is var_series[t] (the recursion
    # standardises with returns[t]/sqrt(var_series[t]) and stores the prior at
    # index 0). Using var_series[1:] would divide r_t by h_{t+1} — an
    # off-by-one that biases the QQ slope sub-unit and damps the ACF-of-z².
    sigma_t = np.sqrt(np.maximum(var_series[:-1], 1e-18))
    return var_series, log_returns / sigma_t


def _render_returns_diagnostics(
    selected: list[Series],
    meta: dict,
    multi_model: bool,
    *,
    data_source: str = "synthetic",
    generator_model: str | None = None,
    true_params: dict | None = None,
) -> None:
    """GARCH-family diagnostics: σ_t series, standardised-residuals QQ, and
    squared-residuals ACF — the textbook trio for assessing whether a GARCH
    model has absorbed the heteroskedasticity in the input series — plus the
    model-implied risk-neutral IV surface so the returns family shares the
    same surface charts as the surface family."""
    if meta.get("log_returns") is None:
        st.info("Reload the data tab once the input returns are available.")
        return

    st.info(
        "**GARCH scale convention** — `ω` is stored on the **annualised** "
        "scale (× `annualization_factor`, 252 by default). The per-period "
        "diagnostics below divide `ω` by that factor before running the "
        "variance filter; `σ_t` is then re-annualised by √252 for "
        "display. α, β, γ, θ are dimensionless and unchanged.",
        icon="ℹ️",
    )

    log_returns = np.asarray(meta["log_returns"], dtype=np.float64)
    annualization_factor = int(meta["annualization_factor"])

    var_by_key: dict[str, np.ndarray] = {}
    z_by_key: dict[str, np.ndarray] = {}
    for s in selected:
        var_series, z = _garch_filter(s.summary.result, s.model, meta)
        if var_series is not None and z is not None:
            var_by_key[s.key] = var_series
            z_by_key[s.key] = z

    if not z_by_key:
        st.warning(
            "Could not run the variance filter at the calibrated optimum "
            "for any selected run."
        )
        return

    st.markdown(
        section_header_html(
            "📈", "Conditional volatility (P) · model-implied σ_t over the sample"
        ),
        unsafe_allow_html=True,
    )
    labeled_sigmas = [
        (
            s.label,
            var_by_key[s.key],
            series_style(s.model, s.solver, multi_model=multi_model, index=i),
        )
        for i, s in enumerate(selected)
        if s.key in var_by_key
    ]
    st.plotly_chart(
        render_conditional_volatility_overlay(
            log_returns, labeled_sigmas, annualization_factor
        ),
        width="stretch",
        key="diag_condvol",
    )

    st.markdown(
        section_header_html("📐", "Standardised residuals · QQ vs N(0, 1)"),
        unsafe_allow_html=True,
    )
    labeled_z = [
        (
            s.label,
            z_by_key[s.key],
            series_style(s.model, s.solver, multi_model=multi_model, index=i),
        )
        for i, s in enumerate(selected)
        if s.key in z_by_key
    ]
    st.plotly_chart(
        render_qq_overlay(labeled_z, standardise=False),
        width="stretch",
        key="diag_zqq",
    )

    st.markdown(
        section_header_html("🔁", "ACF of squared residuals · remaining ARCH effect?"),
        unsafe_allow_html=True,
    )
    facet_grid(
        [s for s in selected if s.key in z_by_key],
        lambda s: render_squared_residuals_acf(z_by_key[s.key]),
        key_prefix="diag_acf",
    )

    _render_returns_implied_surface(
        selected, meta, multi_model, data_source, generator_model, true_params
    )


# ──────────────────────────────────────────────────────────────────────
# Model-implied risk-neutral IV surface (returns family) — Part A coherence
# ──────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False, max_entries=RETURNS_CACHE_MAX_ENTRIES)
def _cached_implied_grid(
    garch_type: str,
    omega_per: float,
    alpha: float,
    beta: float,
    gamma: float,
    h0_per: float,
    spot: float,
    rate: float,
    strikes: tuple[float, ...],
    maturities: tuple[float, ...],
) -> np.ndarray:
    """MC-priced model-implied IV grid for a per-period risk-neutral GARCH."""
    from backend.simulation.models.garch_q import GARCHRiskNeutralSimulator
    from services.post_calibration import iv_grid_from_simulator

    sim = GARCHRiskNeutralSimulator(
        garch_type, omega=omega_per, alpha=alpha, beta=beta, gamma=gamma, h0=h0_per
    )
    return iv_grid_from_simulator(
        sim,
        spot=spot,
        rate=rate,
        strikes=np.asarray(strikes),
        maturities=np.asarray(maturities),
        n_paths=25_000,
    )


def _garch_persistence(
    garch_type: str, alpha: float, beta: float, gamma: float
) -> float:
    if garch_type == "ngarch":
        return beta + alpha * (1.0 + gamma**2)
    if garch_type == "gjr_garch":
        return beta + alpha + 0.5 * gamma
    return beta + alpha


def _long_run_vol_ann(
    garch_type: str,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float = 0.0,
    *,
    per_period: bool = False,
    annualization_factor: float = 252.0,
) -> float | None:
    """Annualised long-run vol √(ω/(1−persistence)), or ``None`` if non-stationary.

    Fitted physical GARCH models store ω **annualised**; the synthetic
    generator's true ω is **per-period** — pass ``per_period=True`` to scale
    the long-run variance by ``annualization_factor``.
    """
    persistence = _garch_persistence(garch_type, alpha, beta, gamma)
    if persistence >= 1.0:
        return None
    long_run_var = omega / (1.0 - persistence)
    if per_period:
        long_run_var *= annualization_factor
    return float(np.sqrt(long_run_var))


def _implied_strike_halfwidth(vols_ann: list[float], t_min: float) -> float:
    """Half-width (fraction of spot) of the shared implied-surface strike grid.

    ±``RETURNS_IMPLIED_STRIKE_SPAN_SIGMAS``·σ·√T at the *shortest* displayed
    maturity, driven by the smallest long-run vol on display so every model's
    wing cells keep an invertible MC price (a fixed ±20 % span is ~7σ√T at one
    month for a low-vol GARCH → zero prices → NaN holes). Clamped to
    [``RETURNS_IMPLIED_HALFWIDTH_MIN``, ``RETURNS_IMPLIED_HALFWIDTH_MAX``].
    """
    if not vols_ann:
        return RETURNS_IMPLIED_HALFWIDTH_MAX
    raw = RETURNS_IMPLIED_STRIKE_SPAN_SIGMAS * min(vols_ann) * float(np.sqrt(t_min))
    return float(
        np.clip(raw, RETURNS_IMPLIED_HALFWIDTH_MIN, RETURNS_IMPLIED_HALFWIDTH_MAX)
    )


def _fitted_implied_grid(
    model_key, model, spot, rate, strikes, maturities, annualization_factor
):
    """Implied IV grid of one fitted physical GARCH model (annualised → per-period).

    ``ω`` is stored annualised on the model and ``σ₀`` is an annualised vol, so
    both are scaled back by ``annualization_factor`` (the periods-per-year the
    synthetic series was generated at — 252 daily, 12 monthly, …) before the
    per-period risk-neutral GARCH is priced. The previous hard-coded 252 silently
    mis-scaled every non-daily series.
    """
    af = float(annualization_factor)
    try:
        return _cached_implied_grid(
            model_key,
            float(model.omega) / af,  # annualised → per-period
            float(model.alpha),
            float(model.beta),
            float(getattr(model, "gamma", 0.0)),
            float(model.sigma0) ** 2 / af,  # annualised var → per-period var
            spot,
            rate,
            strikes,
            maturities,
        )
    except (ValueError, RuntimeError, FloatingPointError, AttributeError, KeyError):
        return None


def _true_implied_grid(garch_type, true_params, spot, rate, strikes, maturities):
    """Implied IV grid of the synthetic generator (per-period params as entered)."""
    try:
        omega_per = float(true_params["omega"])  # entered per-period
        alpha = float(true_params["alpha"])
        beta = float(true_params["beta"])
        gamma = float(true_params.get("gamma", 0.0))
        persistence = _garch_persistence(garch_type, alpha, beta, gamma)
        if persistence >= 1.0:
            return None  # non-stationary generator — no long-run vol to anchor on
        h0_per = omega_per / (1.0 - persistence)  # generator starts at long-run var
        return _cached_implied_grid(
            garch_type,
            omega_per,
            alpha,
            beta,
            gamma,
            h0_per,
            spot,
            rate,
            strikes,
            maturities,
        )
    except (ValueError, RuntimeError, FloatingPointError, KeyError):
        return None


def _render_returns_implied_surface(
    selected: list[Series],
    meta: dict,
    multi_model: bool,
    data_source: str,
    generator_model: str | None,
    true_params: dict | None,
) -> None:
    """Risk-neutral model-implied IV surface for the fitted GARCH models.

    Gives the returns family the *same* IV-surface 3D + smile charts as the
    surface family. In synthetic mode the true generator's implied surface is
    overlaid as the reference (the analogue of 'market' for the surface family).
    """
    spot = float(meta.get("spot", 100.0))
    rate = float(meta.get("rate", RETURNS_IMPLIED_SURFACE_RATE))
    annualization_factor = float(meta.get("annualization_factor", 252))
    maturities = tuple(np.array([1.0, 3.0, 6.0, 12.0, 24.0]) / 12.0)

    # σ√T-aware strike span: the smallest long-run vol on display sets the
    # half-width at the shortest maturity so no model's wing prices to zero.
    vols_ann: list[float] = []
    for s in selected:
        m = s.summary.result.model
        vol = _long_run_vol_ann(
            s.model,
            float(m.omega),  # fitted physical GARCH ω is annualised
            float(m.alpha),
            float(m.beta),
            float(getattr(m, "gamma", 0.0)),
        )
        if vol is not None:
            vols_ann.append(vol)
    if data_source == "synthetic" and generator_model in GARCH_FAMILY and true_params:
        vol = _long_run_vol_ann(
            generator_model,
            float(true_params["omega"]),  # generator ω is per-period
            float(true_params["alpha"]),
            float(true_params["beta"]),
            float(true_params.get("gamma", 0.0)),
            per_period=True,
            annualization_factor=annualization_factor,
        )
        if vol is not None:
            vols_ann.append(vol)
    half = _implied_strike_halfwidth(vols_ann, min(maturities))
    strikes = tuple(np.linspace((1.0 - half) * spot, (1.0 + half) * spot, 9))

    st.markdown(
        section_header_html(
            "📈", "Model-implied IV surface · risk-neutral (Duan LRNVR)"
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "Forward map only — the GARCH was estimated by **maximum likelihood on "
        "returns** (under P), then its European options are priced under Q by "
        f"Monte-Carlo (Duan 1995 LRNVR; assumed risk-free r = {rate:.1%}, market "
        "price of risk λ = 0). This is the *same* IV-surface view the "
        "surface-family models show — not a fit to option quotes."
    )

    with st.spinner("Pricing model-implied surfaces by Monte-Carlo…"):
        fit_grids: dict[str, np.ndarray] = {}
        for s in selected:
            grid = _fitted_implied_grid(
                s.model,
                s.summary.result.model,
                spot,
                rate,
                strikes,
                maturities,
                annualization_factor,
            )
            if grid is not None and np.isfinite(grid).any():
                fit_grids[s.label] = grid
        ref_grid = None
        if (
            data_source == "synthetic"
            and generator_model in GARCH_FAMILY
            and true_params
        ):
            ref_grid = _true_implied_grid(
                generator_model, true_params, spot, rate, strikes, maturities
            )

    if not fit_grids:
        st.info("Model-implied surface unavailable for the selected runs.")
        return

    strikes_arr = np.asarray(strikes)
    maturities_arr = np.asarray(maturities)
    # This returns-family view uses a fixed dollar-strike grid (no σ√T frame).
    # Tile the 1D strikes to the uniform 2D-strikes hover convention and resolve
    # the display axis from the same session choice as the surface family, so the
    # x-axis is consistent across the app (defaults to ln(K/F); σ√T falls back to
    # the native K/S₀ here since there is no σ√T axis for a fixed grid).
    ret_strikes_2d = np.tile(strikes_arr, (len(maturities_arr), 1))
    ret_moneyness = strikes_arr / spot if spot else strikes_arr
    ret_meta = {
        "strikes": ret_strikes_2d,
        "maturities": maturities_arr,
        "spot": float(spot) if spot else 100.0,
        "rate": float(rate),
        "dividend_yield": 0.0,
        "moneyness": ret_moneyness,
        "x_label": "Moneyness  K / S₀",
        "atm_x": 1.0,
    }
    ret_axis = resolve_display_axis(ret_meta).kwargs()

    # The reference (target) surface is the synthetic generator's model-implied
    # surface — name it after that model rather than the generic "Market".
    ref_label = reference_surface_label(data_source, generator_model)
    if ref_grid is not None and np.isfinite(ref_grid).any():
        st.plotly_chart(
            render_iv_surface_overlay_3d(
                ref_grid,
                fit_grids,
                ret_strikes_2d,
                maturities_arr,
                **ret_axis,
                title="",
                spot=spot,
                solver_name="true generator",
                reference_label=ref_label,
            ),
            width="stretch",
            key="diag_returns_surface3d",
        )
        st.plotly_chart(
            render_smile_slices_overlay(
                ref_grid,
                fit_grids,
                ret_strikes_2d,
                maturities_arr,
                **ret_axis,
                spot=spot,
                solver_name="true generator",
                reference_label=ref_label,
            ),
            width="stretch",
            key="diag_returns_smiles",
        )
    else:
        label, grid = next(iter(fit_grids.items()))
        st.caption(f"Showing **{label}** (no synthetic ground truth to overlay).")
        st.plotly_chart(
            render_iv_surface_3d(
                grid,
                ret_strikes_2d,
                maturities_arr,
                **ret_axis,
                title="",
                spot=spot,
                reference_label=label,
            ),
            width="stretch",
            key="diag_returns_surface3d",
        )
        st.plotly_chart(
            render_smile_slices(
                grid, ret_strikes_2d, maturities_arr, **ret_axis, spot=spot
            ),
            width="stretch",
            key="diag_returns_smiles",
        )


# ──────────────────────────────────────────────────────────────────────
# Uncertainty / recovery table (both modes)
# ──────────────────────────────────────────────────────────────────────


def _render_uncertainty_tables(
    selected: list[Series],
    true_params: dict,
    data_source: str,
    generator_model: str | None,
) -> None:
    rows: list[dict] = []
    for s in selected:
        diag = s.summary.result.diagnostics or {}
        unc = diag.get("uncertainty")
        if not unc:
            continue
        # The recovery (True / Rel. error) columns only make sense when the
        # run's model IS the synthetic generator — otherwise there is no
        # ground truth to compare against.
        show_truth = data_source == "synthetic" and s.model == generator_model
        for row in uncertainty_table(
            unc, true_params=true_params if show_truth else None
        ):
            rows.append({"Series": s.label, **row})

    if rows:
        st.markdown(
            section_header_html("📏", "Standard errors & 95% CI · Gauss-Newton / BHHH"),
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.info("No uncertainty quantification available for the selected runs.")
