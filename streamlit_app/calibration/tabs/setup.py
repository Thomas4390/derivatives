"""Tab 1 — Setup & Data: model description + market data preview.

In surface mode the IV-surface and smile charts overlay every selected
``(model, solver, objective)`` fit (via ``series_view_filter``) against
the market, instead of showing one solver at a time. The surface-fit
animation renders every selected model in ONE figure: all model surfaces
morph on a shared 3D scene and each model owns its own parameter column.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from charts.iv_surface import (
    render_iv_surface_3d,
    render_iv_surface_overlay_3d,
    render_returns_summary,
    render_smile_slices,
    render_smile_slices_overlay,
    render_surface_fit_anatomy_animated,
)
from components.equations_panel import render as render_equations_panel
from config.constants import MODEL_DESCRIPTIONS, MODEL_DISPLAY_NAMES
from config.model_registry import get_spec
from services import state_manager
from services.axis_display import resolve_display_axis
from services.post_calibration import (
    iv_grid_animation_frames,
    model_iv_grid,
)
from streamlit_app.simulation.config.styles import (  # type: ignore
    render_metric_row,
    section_header_html,
)
from tabs._helpers import Series, reference_surface_label, series_view_filter


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
    # Name the reference (target) surface after its generating model for
    # synthetic data; real SPX quotes stay "Market".
    ref_label = reference_surface_label(data_source, ctx.get("generator_model"))

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

    # Display-axis choice (ln(K/F) by default) — re-labels the already priced
    # quotes without regenerating the surface.
    axis = resolve_display_axis(meta)

    st.markdown(section_header_html("📈", "IV surface (3D)"), unsafe_allow_html=True)
    # Presentation-only visual settings: retune the red market markers and the
    # surface opacity without regenerating the priced surface (mirrors the
    # display-axis picker — these keys never enter the data hash).
    with st.expander("🎨 Surface visual settings", expanded=False):
        _vc = st.columns(5, vertical_alignment="center")
        with _vc[0]:
            st.slider("Market point size", 2, 16, 5, key="calib_vis_marker_size")
        with _vc[1]:
            st.slider(
                "Market point opacity",
                0.1,
                1.0,
                0.95,
                0.05,
                key="calib_vis_marker_opacity",
            )
        with _vc[2]:
            st.slider(
                "Surface opacity",
                0.1,
                1.0,
                0.45,
                0.05,
                key="calib_vis_market_surf_opacity",
            )
        with _vc[3]:
            st.color_picker(
                "Market point colour", "#dc2626", key="calib_vis_marker_color"
            )
        with _vc[4]:
            st.toggle("Fill market points", value=True, key="calib_vis_marker_fill")
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
                **axis.kwargs(),
                title="",
                spot=meta["spot"],
                reference_label=ref_label,
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
                    **axis.kwargs(),
                    spot=meta["spot"],
                    reference_label=ref_label,
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
                    **axis.kwargs(),
                    spot=meta["spot"],
                ),
                width="stretch",
                key="setup_smiles",
            )
    else:
        st.plotly_chart(
            render_smile_slices(
                meta["iv_grid"],
                meta["strikes"],
                meta["maturities"],
                **axis.kwargs(),
                spot=meta["spot"],
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
        section_header_html("🎞️", "Surface fit progression + parameter roles"),
        unsafe_allow_html=True,
    )
    _render_fit_animation(
        selected,
        market_data,
        meta,
        true_params=true_params,
        gen_model=ctx["model_key"],
        data_source=data_source,
    )


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
    ref_label = reference_surface_label(ctx["data_source"], ctx.get("generator_model"))
    st.plotly_chart(
        render_iv_surface_3d(
            meta["iv_grid"],
            meta["strikes"],
            meta["maturities"],
            **resolve_display_axis(meta).kwargs(),
            title="",
            spot=meta["spot"],
            reference_label=ref_label,
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
    *,
    true_params: dict[str, float] | None = None,
    gen_model: str | None = None,
    data_source: str = "real",
) -> None:
    """Overlay the selected models' surfaces + parameter paths in one figure.

    Runs are grouped by model. Variations of one model (different solvers /
    objectives) are overlaid with a clickable legend. Several models can be
    animated together: their morphing surfaces all share the single 3D scene,
    while each model gets its own column of parameter panels on the right —
    distinct models' parameter sets are not comparable, so they are never merged
    onto one axis. One ▶ Replay / slider drives everything in lock-step.
    """
    animatable = [s for s in selected if s.summary.result.iteration_history]
    if not animatable:
        st.info(
            "No iteration history to animate yet — run a calibration with a "
            "solver that logs callbacks, then pick it here."
        )
        return

    by_model: dict[str, list[Series]] = {}
    for s in animatable:
        by_model.setdefault(s.model, []).append(s)

    # Model picker: multi-select pills (was a single-choice menu). Several models
    # animate together in one figure — surfaces share the 3D scene, each model
    # keeps its own column of parameter panels.
    models = list(by_model)
    chosen_models = st.pills(
        "Animate models",
        options=models,
        selection_mode="multi",
        format_func=MODEL_DISPLAY_NAMES.get,
        key="setup_anim_models",
        default=[models[0]],
    )
    chosen_models = list(chosen_models) if chosen_models else []
    if not chosen_models:
        st.caption("No model selected — pick at least one above to animate.")
        return

    if len(chosen_models) > 1:
        st.caption(
            "Every model's surfaces are overlaid on the same 3D scene, but each "
            "model gets its own column of parameter panels on the right: their "
            "parameter spaces are not comparable, so same-named parameters are "
            "never merged onto one axis. Drag to rotate."
        )
    else:
        st.caption(
            f"Animating **{MODEL_DISPLAY_NAMES.get(chosen_models[0], chosen_models[0])}**"
            " — every selected variation (solver / objective) is overlaid: each "
            "morphing 3D surface and its parameter paths share a colour, and the "
            "legend toggles a variation on or off. Drag to rotate."
        )

    # Presentation-only visual settings for the animation (mirrors the "Surface
    # visual settings" expander): all cosmetic except "Animation frames", which
    # re-prices the intermediate surfaces. Applied to the whole figure.
    with st.expander("🎨 Animation visual settings", expanded=False):
        _ac = st.columns(4, vertical_alignment="center")
        with _ac[0]:
            max_frames = st.slider(
                "Animation frames",
                4,
                30,
                12,
                step=2,
                key="setup_anim_frames",
                help=(
                    "Re-prices the intermediate surfaces — unlike the other "
                    "settings here, this one is not purely cosmetic."
                ),
            )
        with _ac[1]:
            st.slider(
                "Ghost trajectory opacity",
                0.1,
                1.0,
                0.55,
                0.05,
                key="calib_vis_ghost_opacity",
            )
        with _ac[2]:
            st.slider(
                "Model surface opacity",
                0.2,
                1.0,
                0.8,
                0.05,
                key="calib_vis_model_surf_opacity",
            )
        with _ac[3]:
            st.slider(
                "Frame duration (ms)",
                100,
                1500,
                400,
                50,
                key="calib_vis_frame_duration_ms",
            )

    # Per-model "Variations to overlay" pickers — rendered side by side (widgets
    # only, no chart duplication). The key is namespaced per model so each keeps
    # its own selection.
    picker_cols = (
        st.columns(len(chosen_models)) if len(chosen_models) > 1 else [st.container()]
    )
    selected_series: dict[str, list[Series]] = {}
    for ctx, model in zip(picker_cols, chosen_models):
        model_series = by_model[model]
        with ctx:
            if len(model_series) > 1:
                label_by_key = {s.key: s.label for s in model_series}
                picked = st.pills(
                    f"Variations to overlay — {MODEL_DISPLAY_NAMES.get(model, model)}"
                    if len(chosen_models) > 1
                    else "Variations to overlay",
                    options=list(label_by_key),
                    default=list(label_by_key),
                    selection_mode="multi",
                    format_func=lambda k, _lbk=label_by_key: _lbk.get(k, str(k)),
                    key=f"setup_anim_runs_{model}",
                )
                picked_set = set(picked) if picked else set()
                if not picked_set:
                    st.caption(
                        "No variation selected — re-enable at least one to animate."
                    )
                    model_series = []
                else:
                    model_series = [s for s in model_series if s.key in picked_set]
        selected_series[model] = model_series

    # Price each selected model's intermediate surfaces and assemble the groups.
    with st.spinner("Pricing intermediate surfaces…"):
        model_groups = []
        for model in chosen_models:
            runs = []
            for s in selected_series.get(model, []):
                history = _pick_animation_history(s.summary.result)
                if not history:
                    continue
                frames = iv_grid_animation_frames(
                    s.model, history, market_data, meta, max_frames=max_frames
                )
                if frames:
                    runs.append((s.label, s.solver, frames))
            if not runs:
                continue
            # True-value reference lines only when the animated model IS the
            # synthetic generator (its params share the generator's truth).
            tp = (
                true_params
                if (
                    data_source == "synthetic"
                    and gen_model is not None
                    and model == gen_model
                )
                else None
            )
            model_groups.append(
                (model, MODEL_DISPLAY_NAMES.get(model, model), runs, tp)
            )
    if not model_groups:
        st.info(
            "No usable frames — the selected solvers logged no callbacks, or every "
            "snapshot raised during pricing."
        )
        return
    st.plotly_chart(
        render_surface_fit_anatomy_animated(
            model_groups,
            meta["iv_grid"],
            meta["strikes"],
            meta["maturities"],
            **resolve_display_axis(meta).kwargs(),
            spot=meta["spot"],
            reference_label=reference_surface_label(data_source, gen_model),
        ),
        width="stretch",
        key="setup_fit_animation",
    )
