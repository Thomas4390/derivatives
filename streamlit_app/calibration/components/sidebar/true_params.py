"""True-parameters editor — sliders/inputs for the synthetic ground truth."""

from __future__ import annotations

import streamlit as st

from config.constants import GARCH_FAMILY
from config.model_registry import get_spec
from services import state_manager


def render(model_key: str, *, hide_for_real: bool) -> dict[str, float]:
    spec = get_spec(model_key)
    if hide_for_real:
        st.subheader("🎯 True Params (synthetic only)")
        st.caption("Switch back to synthetic to control the ground truth.")
    else:
        st.subheader("🎯 True Parameters")
        st.caption("These generate the synthetic data we'll calibrate against.")

    stored = state_manager.get("calib_true_params") or {}
    if stored.get("_model") != model_key:
        stored = {p.name: p.default for p in spec.params}
        stored["_model"] = model_key

    out: dict[str, float] = {}
    if hide_for_real:
        for p in spec.params:
            out[p.name] = float(stored.get(p.name, p.default))
        out["_model"] = model_key
        state_manager.set("calib_true_params", out)
        return {k: v for k, v in out.items() if k != "_model"}

    for p in spec.params:
        v = float(stored.get(p.name, p.default))
        use_number_input = p.log_scale
        widget = st.number_input if use_number_input else st.slider
        value = widget(
            p.label,
            min_value=float(p.lo),
            max_value=float(p.hi),
            value=v,
            step=float(p.step),
            format=p.fmt,
            help=p.description,
            key=f"true_{model_key}_{p.name}",
        )
        out[p.name] = float(value)
        # number_input doesn't expose its bounds in the widget chrome the
        # way a slider does. For tiny scales (e.g. GARCH ω ∈ [1e-6, 1e-3])
        # users have no way to know the valid range without a caption.
        if use_number_input:
            st.caption(
                f"min = {p.lo:.2e} · max = {p.hi:.2e}",
            )

    blocking_error: str | None = None
    if model_key in ("heston", "bates"):
        feller = 2.0 * out["kappa"] * out["theta"] - out["alpha"] ** 2
        if feller < 0:
            st.warning(
                f"Feller violated: 2κθ − α² = {feller:.4f} < 0 — soft penalty active."
            )
    if model_key == "heston_nandi":
        persistence = out["beta"] + out["alpha"] * out["gamma"] ** 2
        if persistence >= 1.0:
            st.warning(
                f"Stationarity violated: β + αγ² = {persistence:.3f} ≥ 1 — "
                "finite-maturity prices still valid (soft penalty active)."
            )
    if model_key in GARCH_FAMILY:
        # Match each variant's backend persistence (backend/models/garch.py):
        # plain GARCH diverges at α+β ≥ 1, NGARCH at α(1+γ²)+β ≥ 1, GJR at
        # α+γ/2+β ≥ 1. A flat α+β check let non-stationary NGARCH/GJR configs
        # pass this guard only to be rejected by the model constructor mid-run.
        gamma = out.get("gamma", 0.0)
        if model_key == "ngarch":
            persistence = out["alpha"] * (1.0 + gamma**2) + out["beta"]
            expr = "α(1+γ²) + β"
        elif model_key == "gjr_garch":
            persistence = out["alpha"] + 0.5 * gamma + out["beta"]
            expr = "α + γ/2 + β"
        else:  # garch
            persistence = out["alpha"] + out["beta"]
            expr = "α + β"
        if persistence >= 1.0:
            blocking_error = (
                f"Stationarity violated: {expr} = {persistence:.3f} ≥ 1. "
                "The variance recursion would diverge — the run button is disabled."
            )
            st.error(blocking_error)

    state_manager.set("calib_blocking_error", blocking_error)
    out["_model"] = model_key
    state_manager.set("calib_true_params", out)
    return {k: v for k, v in out.items() if k != "_model"}
