"""Search-space panel — per-parameter calibration bounds (the search universe).

Lets the user choose the ``[min, max]`` box the optimiser explores for each
calibratable parameter, via three presets:

- **Default** — each model searches its full admissible parameter box (the
  legacy behaviour; no override is sent to the backend).
- **Tight around truth** — brackets each parameter's true value (synthetic data
  only). Isolates *local* identifiability from the *global* search problem.
- **Custom** — a per-model expander where each parameter's range can be
  tightened to a sub-interval of its admissible box (a range slider for linear
  params; two number inputs for tiny log-scale params such as ω / h₀).

The resolved override ``{model: {param: (lo, hi)}}`` is stored in
``calib_search_bounds`` and threaded through
``services.calibration_service.calibrate_with(search_bounds=...)`` into each
calibrator. Because the search universe is a *sub-box* of each model's
admissible region, no new (invalid) regime is ever introduced.
"""

from __future__ import annotations

import streamlit as st

from backend.calibration.search_space import (
    default_search_bounds,
    tighten_around_truth,
)
from config.model_registry import get_spec
from services import state_manager

# UI label <-> internal mode. Order = increasing control (default → custom).
_MODE_LABELS: dict[str, str] = {
    "default": "Default",
    "tight": "Tight around truth",
    "custom": "Custom",
}
_LABEL_TO_MODE: dict[str, str] = {v: k for k, v in _MODE_LABELS.items()}

_TIGHT_FRAC_DEFAULT_PCT = 50


def _admissible(model_key: str) -> dict[str, tuple[float, float]]:
    """Admissible search box ``{param: (lo, hi)}`` for one candidate model.

    The backend table only knows the built-in models; the session-registered
    custom model reads its box from its own ``ParamSpec`` (the ``min_value`` /
    ``max_value`` the user declared in the Custom Model tab), which is also what
    ``build_custom_calibrator`` falls back to.
    """
    if model_key == "custom":
        return {p.name: (float(p.lo), float(p.hi)) for p in get_spec("custom").params}
    return default_search_bounds(model_key)


_HELP = (
    "The **search universe** is the ``[min, max]`` box the optimiser explores "
    "for each parameter — a sub-interval of the model's admissible region.\n\n"
    "- **Default** — full admissible box (the standard search).\n"
    "- **Tight around truth** — brackets each true value (synthetic only); "
    "shows how a well-located universe isolates *local* identifiability from the "
    "*global* search.\n"
    "- **Custom** — tighten each parameter's range by hand.\n\n"
    "Tightening the box can speed up and stabilise the fit, but a box that "
    "*excludes* the truth makes recovery impossible — a useful thing to witness."
)


def _resolve_tight_boxes(
    eligible: list[str],
    generator_model: str | None,
    true_params: dict[str, float],
    frac: float,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Tight-around-truth boxes, applied ONLY to the generator model.

    ``true_params`` are the single generator's true values, keyed by bare
    parameter name. Broadcasting them to every candidate collided by name — a
    Heston-Nandi generator's ``alpha`` ≈ 3e-6 collapsed GARCH-Q's ARCH ``alpha``
    box to ~(1.5e-6, 4.5e-6), 4-5 orders off. Only the model that actually
    produced the truth gets a tight box; the others keep their default
    admissible box (empty override).
    """
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for m in eligible:
        if m == generator_model:
            out[m] = tighten_around_truth(
                m, true_params, frac, admissible=_admissible(m)
            )
    return out


def render(
    candidates: tuple[str, ...],
    true_params: dict[str, float],
    data_source: str,
    generator_model: str | None = None,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Render the search-space picker. Returns ``{model: {param: (lo, hi)}}``.

    Only models with an iterative search universe are shown (``iv_gbm`` is a
    closed-form IV inversion and is skipped). An empty return means *no
    override* — every calibrator falls back to its canonical default box.
    ``generator_model`` names the synthetic DGP so 'Tight around truth' only
    brackets that model (the only one whose true params are known).
    """
    eligible = [m for m in candidates if _admissible(m)]
    if not eligible:
        state_manager.set("calib_search_bounds", {})
        return {}

    st.subheader("🔍 Search space")

    is_synth = data_source == "synthetic"
    modes = ["default", "tight", "custom"] if is_synth else ["default", "custom"]
    stored_mode = str(state_manager.get("calib_search_space_mode") or "default")
    if stored_mode not in modes:
        stored_mode = "default"

    picked_label = st.segmented_control(
        "Parameter search universe",
        options=[_MODE_LABELS[m] for m in modes],
        default=_MODE_LABELS[stored_mode],
        selection_mode="single",
        help=_HELP,
        key="search_space_mode_pick",
    )
    if picked_label is None:  # single-select can deselect — keep current mode
        picked_label = _MODE_LABELS[stored_mode]
    mode = _LABEL_TO_MODE[picked_label]
    state_manager.set("calib_search_space_mode", mode)

    if not is_synth:
        st.caption(
            "ℹ️  Real data — no ground truth, so 'Tight around truth' is unavailable."
        )

    if mode == "default":
        st.caption("Each model searches its full admissible parameter box (default).")
        out: dict[str, dict[str, tuple[float, float]]] = {}
    elif mode == "tight":
        frac = (
            st.slider(
                "Window around truth (±%)",
                min_value=5,
                max_value=100,
                value=_TIGHT_FRAC_DEFAULT_PCT,
                step=5,
                help="Half-width of the box around each true value, as a "
                "percentage of |true value|.",
                key="search_tight_frac",
            )
            / 100.0
        )
        st.caption(
            "Brackets each parameter's true value — the optimiser starts already "
            "near the answer, isolating local identifiability."
        )
        out = _resolve_tight_boxes(eligible, generator_model, true_params, frac)
    else:  # custom
        st.caption(
            "Tighten each parameter's [min, max] to a sub-interval of its "
            "admissible box."
        )
        stored = state_manager.get("calib_search_bounds") or {}
        out = {m: _render_model_editor(m, dict(stored.get(m, {}))) for m in eligible}

    if is_synth and mode != "default":
        _warn_truth_outside(out, true_params)

    state_manager.set("calib_search_bounds", out)
    return out


def _render_model_editor(
    model_key: str,
    stored_model: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Per-parameter min/max editor for one model, inside an expander."""
    spec = get_spec(model_key)
    admissible = _admissible(model_key)
    box: dict[str, tuple[float, float]] = {}

    with st.expander(f"{spec.display_name} bounds", expanded=False):
        for p in spec.params:
            if p.name not in admissible:  # e.g. a pinned param absent from spec
                continue
            a_lo, a_hi = (float(admissible[p.name][0]), float(admissible[p.name][1]))
            cur = stored_model.get(p.name, (a_lo, a_hi))
            cur_lo = max(a_lo, min(float(cur[0]), a_hi))
            cur_hi = max(a_lo, min(float(cur[1]), a_hi))
            if cur_lo >= cur_hi:  # degenerate stored range — reset to full box
                cur_lo, cur_hi = a_lo, a_hi

            if p.log_scale:
                c_lo, c_hi = st.columns(2)
                lo = c_lo.number_input(
                    f"{p.label} · min",
                    min_value=a_lo,
                    max_value=a_hi,
                    value=cur_lo,
                    format=p.fmt,
                    key=f"search_{model_key}_{p.name}_min",
                )
                hi = c_hi.number_input(
                    f"{p.label} · max",
                    min_value=a_lo,
                    max_value=a_hi,
                    value=cur_hi,
                    format=p.fmt,
                    key=f"search_{model_key}_{p.name}_max",
                )
            else:
                lo, hi = st.slider(
                    p.label,
                    min_value=a_lo,
                    max_value=a_hi,
                    value=(cur_lo, cur_hi),
                    step=float(p.step),
                    format=p.fmt,
                    help=p.description,
                    key=f"search_{model_key}_{p.name}",
                )

            lo, hi = float(lo), float(hi)
            if lo >= hi:
                st.warning(f"{p.label}: min ≥ max → using the full admissible range.")
                lo, hi = a_lo, a_hi
            box[p.name] = (lo, hi)

    return box


def _warn_truth_outside(
    resolved: dict[str, dict[str, tuple[float, float]]],
    true_params: dict[str, float],
) -> None:
    """Warn (non-blocking) when a true value falls outside the chosen box.

    A search universe that excludes the ground truth makes recovery impossible —
    surfaced as a pedagogical warning, not a blocking error.
    """
    for model_key, box in resolved.items():
        for name, (lo, hi) in box.items():
            if name not in true_params:
                continue
            tv = float(true_params[name])
            if tv < lo or tv > hi:
                st.warning(
                    f"{get_spec(model_key).display_name} · {name}: true value "
                    f"{tv:g} lies outside the search box [{lo:g}, {hi:g}] — "
                    "recovery is impossible.",
                    icon="⚠️",
                )
