"""Tab 1 — Setup & Data: model description + market data preview.

In surface mode the IV-surface and smile charts overlay every selected
``(model, solver, objective)`` fit (via ``series_view_filter``) against
the market, instead of showing one solver at a time. The surface-fit
animation is inherently single-run, so it animates one chosen series.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from charts.iv_surface import (
    render_iv_surface_3d,
    render_iv_surface_animation,
    render_iv_surface_overlay_3d,
    render_returns_summary,
    render_smile_slices,
    render_smile_slices_overlay,
)
from components.equations_panel import render as render_equations_panel
from config.constants import MODEL_DESCRIPTIONS, MODEL_DISPLAY_NAMES
from config.model_registry import get_spec
from services import state_manager
from services.post_calibration import (
    iv_grid_animation_frames,
    model_iv_grid,
)
from streamlit_app.simulation.config.styles import (  # type: ignore
    render_metric_row,
    section_header_html,
)
from tabs._helpers import Series, series_view_filter


def render(ctx: dict) -> None:
    mode = ctx["mode"]
    ensure_data = ctx["ensure_data"]

    # First-run guidance: while no calibration has populated the other
    # tabs, surface a 3-step quickstart card. It disappears as soon as
    # ``calib_results`` is non-empty so returning users don't have it
    # pinned at the top of the tab.
    if not (st.session_state.get("calib_results") or {}):
        _render_onboarding_card(ctx)

    market_data, meta = ensure_data()
    if market_data is None:
        # ensure_data has already raised st.error with the actionable
        # cause. Add an explicit recovery hint so the user doesn't see a
        # half-rendered tab and assume the app is stuck.
        st.info(
            "Adjust the sidebar configuration (model, parameters, or "
            "synthetic surface bounds) and the data will load on the "
            "next interaction."
        )
        st.stop()

    # Equations reference for the current sidebar selection — generator
    # SDE, candidate dynamics, optimiser update rules, and objective
    # residuals. Default-open until a calibration result exists; the
    # student can collapse it once the page is familiar.
    render_equations_panel(ctx)

    if mode == "surface":
        _render_surface_setup(ctx, market_data, meta)
    else:
        _render_returns_setup(ctx, market_data, meta)


# ──────────────────────────────────────────────────────────────────────
# Surface family
# ──────────────────────────────────────────────────────────────────────


def _render_surface_setup(ctx: dict, market_data, meta: dict) -> None:
    data_source = ctx["data_source"]
    true_params = ctx["true_params"]

    results = state_manager.get("calib_results") or {}
    selected = series_view_filter(results, key="setup") if results else []

    # Which models to describe: the distinct models present in the
    # selection (so comparing Heston vs Bates shows both write-ups), or
    # the sidebar's reference model before any run.
    models_in_view = _distinct_models(selected) or [ctx["model_key"]]
    badge = (
        "Synthetic ground truth" if data_source == "synthetic" else "Real SPX market"
    )
    header = " · ".join(
        [", ".join(get_spec(m).display_name for m in models_in_view), badge]
    )
    # Single model: drop the description straight into help. Multiple
    # candidates: stitch them together with a separator so the tooltip
    # walks through each one.
    if len(models_in_view) == 1:
        help_text = MODEL_DESCRIPTIONS[models_in_view[0]]
    else:
        help_text = "\n\n---\n\n".join(
            f"**{MODEL_DISPLAY_NAMES.get(m, m)}** — {MODEL_DESCRIPTIONS[m]}"
            for m in models_in_view
        )
    st.subheader(f"📐 {header}", help=help_text)

    render_metric_row(
        [
            ("Quotes", f"{market_data.n_quotes:,}", "after filtering"),
            ("Spot S₀", f"${meta['spot']:.2f}", "underlying"),
            ("Risk-free rate r", f"{meta['rate']:.2%}", "annual, cont. comp."),
            (
                "Dividend yield q",
                f"{meta['dividend_yield']:.2%}",
                "annual, cont. comp.",
            ),
        ]
    )
    if data_source == "real":
        st.caption(
            f"Snapshot: **{meta['real_label']}** · "
            f"{meta['n_quotes_total']} raw quotes · "
            f"{meta['n_quotes_dropped']} dropped after filtering"
        )

    # Price each selected fit's IV grid once; reuse for the 3D surface and
    # the smile slices. Keyed by composite label so the overlay legends
    # name each run unambiguously.
    fit_grids = _compute_fit_grids(selected, market_data, meta)

    st.markdown(section_header_html("📈", "IV surface (3D)"), unsafe_allow_html=True)
    if fit_grids:
        if len(fit_grids) > 4:
            st.caption(
                "Several model surfaces overlaid — the 2D smile slices below "
                "stay readable with more fits than the 3D view."
            )
        st.plotly_chart(
            render_iv_surface_overlay_3d(
                meta["iv_grid"],
                fit_grids,
                meta["strikes"],
                meta["maturities"],
                title="",
                spot=meta["spot"],
            ),
            width="stretch",
            key="setup_surface_overlay",
        )
    else:
        _render_surface_fallback(ctx, market_data, meta)

    st.markdown(
        section_header_html(
            "📊", f"IV smiles · {len(meta['maturities'])} maturities (in days)"
        ),
        unsafe_allow_html=True,
    )
    if fit_grids:
        # Smiles-specific per-fit filter: the 2D slices get busy fast (market
        # × every fit × every maturity), so let the user hide/show individual
        # fits here without touching the 3D surface above. Defaults to all
        # selected fits; the in-chart maturity dropdown stacks on top.
        smile_grids = fit_grids
        if len(fit_grids) > 1:
            chosen = st.pills(
                "Show fits on smiles",
                options=list(fit_grids),
                default=list(fit_grids),
                selection_mode="multi",
                key="setup_smiles_fits",
            )
            chosen_set = set(chosen) if chosen else set()
            smile_grids = {k: v for k, v in fit_grids.items() if k in chosen_set}
        if smile_grids:
            st.plotly_chart(
                render_smile_slices_overlay(
                    meta["iv_grid"],
                    smile_grids,
                    meta["strikes"],
                    meta["maturities"],
                    spot=meta["spot"],
                ),
                width="stretch",
                key="setup_smiles_overlay",
            )
        else:
            st.caption("No fit selected — re-enable at least one above to overlay.")
            st.plotly_chart(
                render_smile_slices(
                    meta["iv_grid"],
                    meta["strikes"],
                    meta["maturities"],
                    spot=meta["spot"],
                ),
                width="stretch",
                key="setup_smiles",
            )
    else:
        st.plotly_chart(
            render_smile_slices(
                meta["iv_grid"], meta["strikes"], meta["maturities"], spot=meta["spot"]
            ),
            width="stretch",
            key="setup_smiles",
        )

    if data_source == "synthetic":
        with st.expander("True parameters used"):
            st.json(true_params, expanded=True)

    # Surface-fit animation is inherently single-run (it morphs one
    # solver's surface across its iterations), so it animates one chosen
    # series rather than the whole overlay.
    st.markdown(
        section_header_html("🎞️", "Surface fit animation"), unsafe_allow_html=True
    )
    _render_fit_animation(selected, market_data, meta)


def _distinct_models(selected: list[Series]) -> list[str]:
    """Models present in the selection, de-duplicated, order preserved."""
    seen: dict[str, None] = {}
    for s in selected:
        seen.setdefault(s.model, None)
    return list(seen)


def _compute_fit_grids(
    selected: list[Series],
    market_data,
    meta: dict,
) -> dict[str, np.ndarray]:
    """``{series_label: model_iv_grid}`` for every selected fit that prices.

    Skips runs whose model can't produce a Fourier-priceable surface or
    that raise during pricing, so one bad fit can't blank the chart.
    """
    grids: dict[str, np.ndarray] = {}
    for s in selected:
        try:
            grids[s.label] = model_iv_grid(s.summary.result.model, market_data, meta)
        except (ValueError, RuntimeError, FloatingPointError, KeyError):
            continue
    return grids


def _render_surface_fallback(ctx: dict, market_data, meta: dict) -> None:
    """Pre-run / no-fit view: show the market surface alone.

    No model surface is overlaid before a calibration runs. The only model
    parameters available pre-run are the synthetic ground truth, whose
    surface coincides with the market and would read as an (uncalibrated)
    perfect fit — confusing. The fitted model overlay appears only once a
    real calibration result exists (see ``_render_surface_setup``).
    """
    st.caption("Run a calibration to overlay the model surface here.")
    st.plotly_chart(
        render_iv_surface_3d(
            meta["iv_grid"],
            meta["strikes"],
            meta["maturities"],
            title="",
            spot=meta["spot"],
        ),
        width="stretch",
        key="setup_surface_market",
    )


# ──────────────────────────────────────────────────────────────────────
# Returns family (GARCH) — data preview only, no per-fit overlay
# ──────────────────────────────────────────────────────────────────────


def _render_returns_setup(ctx: dict, market_data, meta: dict) -> None:
    true_params = ctx["true_params"]
    model_key = ctx["model_key"]

    badge = (
        "Synthetic ground truth" if ctx["data_source"] == "synthetic" else "Real market"
    )
    st.subheader(
        f"📐 {get_spec(model_key).display_name} · {badge}",
        help=MODEL_DESCRIPTIONS[model_key],
    )
    render_metric_row(
        [
            ("Periods", f"{len(meta['log_returns']):,}", "log-return observations"),
            ("σ annualised", f"{meta['sample_volatility_ann']:.2%}", "sample"),
            ("Ann. factor", str(meta["annualization_factor"]), "obs / year"),
            (
                "Mean return / yr",
                f"{float(np.mean(meta['log_returns']) * meta['annualization_factor']):.2%}",
                "annualised",
            ),
        ]
    )
    st.markdown(section_header_html("📈", "Synthetic returns"), unsafe_allow_html=True)
    st.plotly_chart(
        render_returns_summary(
            meta["prices"], meta["log_returns"], meta["annualization_factor"]
        ),
        width="stretch",
        key="setup_returns_summary",
    )
    with st.expander("True parameters used"):
        st.json(true_params, expanded=True)


# ──────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────


def _render_onboarding_card(ctx: dict) -> None:
    """Three-step quickstart shown on the Setup tab until the first run."""
    family = ctx.get("data_family") or "surface"
    candidates = ctx.get("candidate_models") or ()
    family_label = (
        "Surface · Heston / Merton / Bates / IV-GBM"
        if family == "surface"
        else "Returns · GARCH / NGARCH / GJR-GARCH"
    )
    candidates_done = len(candidates) > 0
    candidates_text = (
        f"**{len(candidates)} model"
        f"{'s' if len(candidates) > 1 else ''}** selected — "
        "add more in the sidebar to compare them in one run."
        if candidates_done
        else "**no model selected** — pick at least one in the sidebar."
    )
    step1_marker = "✅"  # family is always set (defaults to surface)
    step2_marker = "✅" if candidates_done else "2."
    st.info(
        f"**Three-step quickstart**\n\n"
        f"{step1_marker} **Model family**: {family_label} — change in the "
        "sidebar's *Data family* picker.\n"
        f"{step2_marker} **Candidate models**: {candidates_text}\n"
        "3. **Click ▶ Run calibration** in the sidebar.\n\n"
        "_The Live, Loss Landscape, Diagnostics and Compare tabs populate "
        "as soon as the first solver finishes._",
        icon="🎓",
    )


def _pick_animation_history(result) -> tuple:
    """Pick the iteration_history that morphs the most visibly.

    Scores each multi-start restart by its first→last objective ratio
    (largest = biggest visible improvement) and returns the winner, falling
    back to ``result.iteration_history`` when no multi-start data exists.
    """
    primary = tuple(result.iteration_history or ())
    diag = result.diagnostics or {}
    msh = diag.get("multi_start_history") or ()
    if not msh:
        return primary

    best_history = primary
    best_ratio = 0.0
    for history in msh:
        if not history or len(history) < 2:
            continue
        first_obj = float(history[0].objective)
        last_obj = float(history[-1].objective)
        if last_obj <= 0 or first_obj <= 0:
            continue
        ratio = first_obj / max(last_obj, 1e-30)
        if ratio > best_ratio:
            best_ratio = ratio
            best_history = tuple(history)
    return best_history


def _render_fit_animation(
    selected: list[Series],
    market_data,
    meta: dict,
) -> None:
    """Animate one chosen run's model surface across its iteration callbacks."""
    animatable = [s for s in selected if s.summary.result.iteration_history]
    if not animatable:
        st.info(
            "No iteration history to animate yet — run a calibration with a "
            "solver that logs callbacks, then pick it here."
        )
        return

    if len(animatable) > 1:
        labels = [s.label for s in animatable]
        chosen_label = st.selectbox(
            "Animate which fit?",
            options=labels,
            index=0,
            key="setup_anim_series",
        )
        series = next(s for s in animatable if s.label == chosen_label)
    else:
        series = animatable[0]

    history = _pick_animation_history(series.summary.result)
    if not history:
        st.info(
            "Calibrator did not log any iteration callbacks "
            "(closed-form solver?) — animation skipped."
        )
        return

    st.caption(
        f"Animating **{series.label}**. Each frame re-prices the surface from "
        "the parameters logged at an iteration callback. Drag to rotate."
    )
    max_frames = st.slider(
        "Animation frames", 4, 30, 12, step=2, key="setup_anim_frames"
    )
    with st.spinner("Pricing intermediate surfaces…"):
        frames = iv_grid_animation_frames(
            series.model, history, market_data, meta, max_frames=max_frames
        )
    if not frames:
        st.info("No usable frames — every snapshot raised during pricing.")
        return
    st.plotly_chart(
        render_iv_surface_animation(
            frames,
            meta["iv_grid"],
            meta["strikes"],
            meta["maturities"],
            solver_name=series.solver,
            spot=meta["spot"],
        ),
        width="stretch",
        key="setup_fit_animation",
    )
