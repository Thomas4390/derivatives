"""Optimisation-constraints panel — Feller + GARCH-stationarity controls.

Mirrors :mod:`components.sidebar.objective_panel`. Exposes how variance-positivity
constraints are enforced during calibration:

- the **Feller** condition ``2κσ² > α²`` for the CIR stochastic-variance models
  (Heston, Bates), and
- the **stationarity** condition ``β + αγ² < 1`` for Heston-Nandi GARCH.

Each control is shown only when the candidate set contains a model that carries
the corresponding condition. Selections are stored in
``calib_constraint_settings`` and consumed by
``services.calibration_service._build_feller`` / ``._build_stationarity``.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from services import state_manager

# Models whose CIR variance process carries a Feller condition.
_FELLER_MODELS: frozenset[str] = frozenset({"heston", "bates"})
# Models carrying a GARCH variance-stationarity condition (persistence < 1).
# The persistence form is variant-specific: Heston-Nandi β + αγ²; Duan
# NGARCH-Q β + α(1 + γ²); GARCH-Q β + α; GJR-Q β + α + γ/2.
_STATIONARITY_MODELS: frozenset[str] = frozenset(
    {"heston_nandi", "ngarch_q", "garch_q", "gjr_q"}
)

# UI label <-> backend mode. Order defines the segmented-control order
# (increasing strictness: off → soft → hard).
_MODE_LABELS: dict[str, str] = {
    "off": "Off",
    "soft": "Soft penalty",
    "hard": "Hard enforce",
}
_LABEL_TO_MODE: dict[str, str] = {v: k for k, v in _MODE_LABELS.items()}

# Leading icon per label — mirrors the status captions below each picker
# (ℹ️ free, ⚖️ penalised, ✅ guaranteed) and the icon-in-label convention used
# by the other segmented controls (see ``model_picker._format``).
_MODE_ICONS: dict[str, str] = {
    "Off": "ℹ️",
    "Soft penalty": "⚖️",
    "Hard enforce": "✅",
}


def _format_mode(label: str) -> str:
    """Prefix a constraint-mode label with its status icon."""
    return f"{_MODE_ICONS.get(label, '·')} {label}"

_FELLER_HELP = (
    "How the Feller condition **2κσ² > α²** (strictly positive variance) is "
    "treated during calibration:\n\n"
    "- **Off** — no constraint; the optimiser is free to land in the "
    "Feller-violating region (the empirically realistic regime for equity "
    "index surfaces).\n"
    "- **Soft penalty** — violations are penalised in the loss with the weight "
    "below; Feller is *discouraged* but still reachable (legacy default).\n"
    "- **Hard enforce** — α is reparametrised so α ≤ √(2κσ²) always holds; "
    "Feller is *guaranteed* at the cost of fit quality."
)

_STATIONARITY_HELP = (
    "How the GARCH variance-stationarity condition (**persistence < 1**) is "
    "treated during calibration. The persistence is variant-specific — "
    "β + αγ² (Heston-Nandi), β + α(1+γ²) (NGARCH-Q), β + α (GARCH-Q), "
    "β + α + γ/2 (GJR-Q):\n\n"
    "- **Off** — no constraint; the fit may be non-stationary (finite-maturity "
    "prices stay valid, only the unconditional variance diverges).\n"
    "- **Soft penalty** — violations are penalised in the loss with the weight "
    "below (default).\n"
    "- **Hard enforce** — the leverage γ (or α for symmetric GARCH-Q) is capped "
    "so the condition always holds; stationarity is *guaranteed*."
)


def render(candidates: tuple[str, ...]) -> dict[str, Any]:
    """Render the constraint pickers. Returns the ``constraint_settings`` dict.

    Renders the Feller control when any candidate is a CIR model and the
    stationarity control when any candidate is Heston-Nandi GARCH. With neither,
    nothing is drawn and the stored settings are returned unchanged.
    """
    settings = dict(state_manager.get("calib_constraint_settings") or {})
    has_feller = any(c in _FELLER_MODELS for c in candidates)
    has_stationarity = any(c in _STATIONARITY_MODELS for c in candidates)
    if not (has_feller or has_stationarity):
        return settings

    st.subheader("⛓️ Optimisation constraints")
    if has_feller:
        _render_feller(settings)
    if has_stationarity:
        _render_stationarity(settings)

    state_manager.update(calib_constraint_settings=settings)
    return settings


# --------------------------------------------------------------------------- #
# Feller (Heston / Bates)
# --------------------------------------------------------------------------- #


def _render_feller(settings: dict[str, Any]) -> None:
    current_mode = str(settings.get("feller_mode", "soft")).lower()
    if current_mode not in _MODE_LABELS:
        current_mode = "soft"
    options = list(_MODE_LABELS.values())
    picked_label = st.segmented_control(
        "Feller condition  2κσ² > α²",
        options=options,
        default=_MODE_LABELS[current_mode],
        selection_mode="single",
        format_func=_format_mode,
        help=_FELLER_HELP,
        key="constraint_feller_mode",
    )
    # Single-select segmented controls can return None on deselection — keep
    # the current mode rather than dropping the constraint.
    if picked_label is None:
        picked_label = _MODE_LABELS[current_mode]
    mode = _LABEL_TO_MODE[picked_label]
    settings["feller_mode"] = mode

    if mode == "soft":
        settings["feller_weight"] = float(
            st.slider(
                "Feller penalty weight",
                min_value=0.0,
                max_value=5000.0,
                value=float(settings.get("feller_weight", 1000.0)),
                step=100.0,
                help=(
                    "Strength of the soft Feller penalty added to the loss. "
                    "0 ≈ Off; higher values push the fit harder toward the "
                    "Feller-satisfying region (default 1000)."
                ),
                key="constraint_feller_weight",
            )
        )

    if mode == "off":
        st.caption(
            "ℹ️  Feller free — calibrated κ, σ², α may violate 2κσ² > α² "
            "(variance can touch zero)."
        )
    elif mode == "hard":
        st.caption(
            "✅  Feller guaranteed — α is capped at √(2κσ²) by reparametrisation."
        )
    else:
        st.caption("⚖️  Feller penalised — violations are discouraged, not forbidden.")

    _render_feller_pedagogy()


# --------------------------------------------------------------------------- #
# Stationarity (Heston-Nandi GARCH)
# --------------------------------------------------------------------------- #


def _render_stationarity(settings: dict[str, Any]) -> None:
    current_mode = str(settings.get("stationarity_mode", "soft")).lower()
    if current_mode not in _MODE_LABELS:
        current_mode = "soft"
    options = list(_MODE_LABELS.values())
    picked_label = st.segmented_control(
        "GARCH stationarity  (persistence < 1)",
        options=options,
        default=_MODE_LABELS[current_mode],
        selection_mode="single",
        format_func=_format_mode,
        help=_STATIONARITY_HELP,
        key="constraint_stationarity_mode",
    )
    # Single-select segmented controls can return None on deselection — keep
    # the current mode rather than dropping the constraint.
    if picked_label is None:
        picked_label = _MODE_LABELS[current_mode]
    mode = _LABEL_TO_MODE[picked_label]
    settings["stationarity_mode"] = mode

    if mode == "soft":
        settings["stationarity_weight"] = float(
            st.slider(
                "Stationarity penalty weight",
                min_value=0.0,
                max_value=5000.0,
                value=float(settings.get("stationarity_weight", 1000.0)),
                step=100.0,
                help=(
                    "Strength of the soft stationarity penalty added to the "
                    "loss. 0 ≈ Off; higher values push the fit toward "
                    "β + αγ² < 1 (default 1000)."
                ),
                key="constraint_stationarity_weight",
            )
        )

    if mode == "off":
        st.caption(
            "ℹ️  Stationarity free — calibrated β + αγ² may exceed 1 "
            "(unconditional variance diverges)."
        )
    elif mode == "hard":
        st.caption(
            "✅  Stationarity guaranteed — γ is capped so β + αγ² < 1 by "
            "reparametrisation."
        )
    else:
        st.caption(
            "⚖️  Stationarity penalised — violations are discouraged, not forbidden."
        )


def _render_feller_pedagogy() -> None:
    """Collapsible explainer on the Feller condition and the fit trade-off."""
    with st.expander("📖 About the Feller condition", expanded=False):
        st.markdown(
            r"""
The variance of Heston / Bates follows a CIR process

$$
\mathrm{d}v_t = \kappa(\sigma^2 - v_t)\,\mathrm{d}t + \alpha\sqrt{v_t}\,\mathrm{d}W_t .
$$

The **Feller condition** $2\kappa\sigma^2 > \alpha^2$ guarantees that $v_t$ stays
*strictly positive* and never touches zero. When it is violated the variance
can hit zero (and reflect), which is harder to simulate but perfectly valid as
a pricing model.

**Why expose three modes?** Empirically, equity-index option surfaces almost
always calibrate a Heston model that *violates* Feller — forcing the condition
typically **degrades the fit**. The trade-off is real and worth seeing:

| Mode | Guarantee | Effect on fit |
|---|---|---|
| **Off** | none | best fit, may touch zero variance |
| **Soft** | discouraged | small fit cost, lands near the boundary |
| **Hard** | $\alpha \le \sqrt{2\kappa\sigma^2}$ always | larger fit cost, always positive |

The *Loss Landscape* tab draws the boundary $2\kappa\sigma^2 = \alpha^2$ so you can
watch where each mode places the calibrated point relative to it.

**Sources**: Feller (1951); Andersen (2008) §2; Gatheral (2006) ch. 2.
"""
        )
