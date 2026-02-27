"""
Path Explorer Parameters - Compact inline controls for single-path exploration.

Renders parameter controls in the main content area (not sidebar) using a unique
"explorer_" key prefix to avoid conflicts with sidebar widgets.
Parameter labels use compact centered HTML with serif italic font and proper subscripts.
"""

import streamlit as st
from typing import Dict, Any

from config.model_registry import get_model, get_parameter_defaults
from utils.model_helpers import check_feller_condition

# ── Compact centered label ─────────────────────────────────────────────────

_LABEL_CSS = (
    "text-align:center;"
    "font-family:'Times New Roman',Georgia,serif;"
    "font-style:italic;"
    "font-size:1.05rem;"
    "padding:0 0 4px;"
    "line-height:1.4;"
    "color:rgba(255,255,255,0.85);"
)


def _label(html: str) -> None:
    """Render a compact, centered, math-styled label above a widget."""
    st.markdown(
        f"<div style='{_LABEL_CSS}'>{html}</div>",
        unsafe_allow_html=True,
    )


# ── Main entry point ──────────────────────────────────────────────────────

def render_explorer_params(model_key: str) -> Dict[str, Any]:
    """
    Render compact inline parameter controls for the Path Explorer tab.

    Row 1: Market & Simulation parameters (S0, mu, r, T, seed)
    Row 2+: Model-specific parameters (varies by model)

    Args:
        model_key: Selected model key

    Returns:
        Dictionary with all parameters needed for run_simulation()
    """
    kp = "explorer_"
    defaults = get_parameter_defaults(model_key)
    params: Dict[str, Any] = {}

    # ── Row 1: Market & Simulation ──────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        _label("S<sub>0</sub>")
        params["spot"] = st.number_input(
            "S0", min_value=1.0, max_value=10000.0,
            value=st.session_state.get(f"{kp}spot", 100.0),
            step=1.0, key=f"{kp}spot", label_visibility="collapsed",
        )

    with c2:
        _label("&mu;")
        params["drift"] = st.number_input(
            "drift", min_value=-0.20, max_value=0.50,
            value=st.session_state.get(f"{kp}drift", 0.08),
            step=0.01, format="%.2f", key=f"{kp}drift",
            label_visibility="collapsed",
        )

    with c3:
        _label("r")
        params["risk_free_rate"] = st.number_input(
            "rate", min_value=0.0, max_value=0.20,
            value=st.session_state.get(f"{kp}rate", 0.05),
            step=0.005, format="%.3f", key=f"{kp}rate",
            label_visibility="collapsed",
        )

    with c4:
        _label("T")
        params["time_horizon"] = st.number_input(
            "T", min_value=0.1, max_value=10.0,
            value=st.session_state.get(f"{kp}T", 1.0),
            step=0.1, format="%.1f", key=f"{kp}T",
            label_visibility="collapsed",
        )

    with c5:
        _label("Seed")
        params["seed"] = st.number_input(
            "seed", min_value=1, max_value=99999,
            value=st.session_state.get(f"{kp}seed", 42),
            step=1, key=f"{kp}seed", label_visibility="collapsed",
            help="Fixed seed — same random draws across parameter changes",
        )

    # Fixed single-path settings; n_steps from sidebar
    params["n_paths"] = 1
    params["n_steps"] = int(st.session_state.get("sim_steps", 252))

    # ── Row 2+: Model-specific parameters ───────────────────────────────
    model_lower = model_key.lower()

    if model_lower == "gbm":
        _render_gbm(params, defaults, kp)

    elif model_lower == "heston":
        _render_heston(params, defaults, kp)

    elif model_lower == "merton":
        _render_merton(params, defaults, kp)

    elif model_lower == "bates":
        _render_heston(params, defaults, kp)
        _render_jump(params, defaults, kp)

    elif model_lower == "garch":
        _render_garch(params, defaults, kp)

    elif model_lower == "ngarch":
        _render_garch(params, defaults, kp)
        c = st.columns(1)[0]
        with c:
            _label("&theta;<sub>NG</sub>")
            params["theta_ngarch"] = st.slider(
                "theta_ng", min_value=0.0, max_value=2.0,
                value=st.session_state.get(f"{kp}theta_ng", defaults.get("theta", 0.5)),
                step=0.1, format="%.1f", key=f"{kp}theta_ng",
                label_visibility="collapsed",
            )

    elif model_lower == "gjr_garch":
        _render_garch(params, defaults, kp)
        c = st.columns(1)[0]
        with c:
            _label("&gamma;")
            params["gamma"] = st.slider(
                "gamma", min_value=0.0, max_value=0.5,
                value=st.session_state.get(f"{kp}gamma", defaults.get("gamma", 0.03)),
                step=0.01, format="%.3f", key=f"{kp}gamma",
                label_visibility="collapsed",
            )

    else:
        _render_custom(params, defaults, kp, model_key)

    params["model"] = model_key
    return params


# ── Private renderers ──────────────────────────────────────────────────────


def _render_gbm(params: Dict, defaults: Dict, kp: str) -> None:
    c = st.columns(1)[0]
    with c:
        _label("&sigma;")
        params["sigma"] = st.slider(
            "sigma", min_value=0.01, max_value=1.0,
            value=st.session_state.get(f"{kp}sigma", defaults.get("sigma", 0.20)),
            step=0.01, format="%.2f", key=f"{kp}sigma",
            label_visibility="collapsed",
        )


def _render_heston(params: Dict, defaults: Dict, kp: str) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        _label("V<sub>0</sub>")
        params["v0"] = st.slider(
            "v0", min_value=0.001, max_value=0.50,
            value=st.session_state.get(f"{kp}v0", defaults.get("v0", 0.04)),
            step=0.005, format="%.3f", key=f"{kp}v0",
            label_visibility="collapsed",
        )
    with c2:
        _label("&kappa;")
        params["kappa"] = st.slider(
            "kappa", min_value=0.1, max_value=10.0,
            value=st.session_state.get(f"{kp}kappa", defaults.get("kappa", 2.0)),
            step=0.1, format="%.1f", key=f"{kp}kappa",
            label_visibility="collapsed",
        )
    with c3:
        _label("&theta;")
        params["theta"] = st.slider(
            "theta", min_value=0.001, max_value=0.50,
            value=st.session_state.get(f"{kp}theta", defaults.get("theta", 0.04)),
            step=0.005, format="%.3f", key=f"{kp}theta",
            label_visibility="collapsed",
        )
    with c4:
        _label("&xi;")
        params["xi"] = st.slider(
            "xi", min_value=0.01, max_value=1.0,
            value=st.session_state.get(f"{kp}xi", defaults.get("xi", 0.3)),
            step=0.01, format="%.2f", key=f"{kp}xi",
            label_visibility="collapsed",
        )
    with c5:
        _label("&rho;")
        params["rho"] = st.slider(
            "rho", min_value=-0.99, max_value=0.99,
            value=st.session_state.get(f"{kp}rho", defaults.get("rho", -0.7)),
            step=0.01, format="%.2f", key=f"{kp}rho",
            label_visibility="collapsed",
        )

    # Feller condition caption
    satisfied, lhs, rhs = check_feller_condition(params)
    if satisfied:
        st.caption(f"Feller OK: 2\u03ba\u03b8 = {lhs:.4f} > {rhs:.4f} = \u03be\u00b2")
    else:
        st.caption(f"\u26a0\ufe0f Feller violated: 2\u03ba\u03b8 = {lhs:.4f} \u2264 {rhs:.4f} = \u03be\u00b2")


def _render_merton(params: Dict, defaults: Dict, kp: str) -> None:
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        _label("&sigma;")
        params["sigma"] = st.slider(
            "sigma", min_value=0.01, max_value=1.0,
            value=st.session_state.get(f"{kp}sigma", defaults.get("sigma", 0.20)),
            step=0.01, format="%.2f", key=f"{kp}sigma",
            label_visibility="collapsed",
        )
    with c2:
        _label("&lambda;<sub>J</sub>")
        params["lambda_j"] = st.slider(
            "lambda_j", min_value=0.0, max_value=5.0,
            value=st.session_state.get(f"{kp}lambda", defaults.get("lambda_j", 0.5)),
            step=0.1, format="%.1f", key=f"{kp}lambda",
            label_visibility="collapsed",
        )
    with c3:
        _label("&mu;<sub>J</sub>")
        params["mu_j"] = st.slider(
            "mu_j", min_value=-0.5, max_value=0.5,
            value=st.session_state.get(f"{kp}muj", defaults.get("mu_j", -0.1)),
            step=0.01, format="%.2f", key=f"{kp}muj",
            label_visibility="collapsed",
        )
    with c4:
        _label("&sigma;<sub>J</sub>")
        params["sigma_j"] = st.slider(
            "sigma_j", min_value=0.01, max_value=0.5,
            value=st.session_state.get(f"{kp}sigmaj", defaults.get("sigma_j", 0.2)),
            step=0.01, format="%.2f", key=f"{kp}sigmaj",
            label_visibility="collapsed",
        )


def _render_jump(params: Dict, defaults: Dict, kp: str) -> None:
    """Jump parameters row for Bates (Heston sliders already rendered)."""
    c1, c2, c3 = st.columns(3)

    with c1:
        _label("&lambda;<sub>J</sub>")
        params["lambda_j"] = st.slider(
            "lambda_j", min_value=0.0, max_value=5.0,
            value=st.session_state.get(f"{kp}lambda", defaults.get("lambda_j", 0.5)),
            step=0.1, format="%.1f", key=f"{kp}lambda",
            label_visibility="collapsed",
        )
    with c2:
        _label("&mu;<sub>J</sub>")
        params["mu_j"] = st.slider(
            "mu_j", min_value=-0.5, max_value=0.5,
            value=st.session_state.get(f"{kp}muj", defaults.get("mu_j", -0.1)),
            step=0.01, format="%.2f", key=f"{kp}muj",
            label_visibility="collapsed",
        )
    with c3:
        _label("&sigma;<sub>J</sub>")
        params["sigma_j"] = st.slider(
            "sigma_j", min_value=0.01, max_value=0.5,
            value=st.session_state.get(f"{kp}sigmaj", defaults.get("sigma_j", 0.2)),
            step=0.01, format="%.2f", key=f"{kp}sigmaj",
            label_visibility="collapsed",
        )


def _render_garch(params: Dict, defaults: Dict, kp: str) -> None:
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        _label("&sigma;<sub>0</sub>")
        params["sigma0"] = st.slider(
            "sigma0", min_value=0.01, max_value=2.0,
            value=st.session_state.get(f"{kp}sigma0", defaults.get("sigma0", 0.20)),
            step=0.01, format="%.2f", key=f"{kp}sigma0",
            label_visibility="collapsed",
        )
    with c2:
        _label("&omega;")
        params["omega"] = st.number_input(
            "omega", min_value=1e-7, max_value=0.1,
            value=st.session_state.get(f"{kp}omega", defaults.get("omega", 0.002)),
            step=0.0001, format="%.4f", key=f"{kp}omega",
            label_visibility="collapsed",
        )
    with c3:
        _label("&alpha;")
        params["alpha"] = st.slider(
            "alpha", min_value=0.0, max_value=0.50,
            value=st.session_state.get(f"{kp}alpha", defaults.get("alpha", 0.06)),
            step=0.01, format="%.3f", key=f"{kp}alpha",
            label_visibility="collapsed",
        )
    with c4:
        _label("&beta;")
        params["beta"] = st.slider(
            "beta", min_value=0.0, max_value=0.99,
            value=st.session_state.get(f"{kp}beta", defaults.get("beta", 0.90)),
            step=0.01, format="%.2f", key=f"{kp}beta",
            label_visibility="collapsed",
        )

    # Persistence caption
    persistence = params["alpha"] + params["beta"]
    if persistence >= 1.0:
        st.caption(f"\u26a0\ufe0f \u03b1+\u03b2 = {persistence:.3f} \u2265 1 \u2014 non-stationary")
    else:
        lr_vol = (params["omega"] / (1 - persistence)) ** 0.5 * 100
        st.caption(f"Persistence \u03b1+\u03b2 = {persistence:.3f} \u00b7 Long-run vol \u2248 {lr_vol:.1f}%")


def _render_custom(params: Dict, defaults: Dict, kp: str, model_key: str) -> None:
    """Render custom model parameters dynamically from ModelSpec."""
    try:
        model_spec = get_model(model_key)
    except ValueError:
        return

    n_params = len(model_spec.parameters)
    if n_params == 0:
        return

    cols = st.columns(min(n_params, 5))
    for i, p in enumerate(model_spec.parameters):
        with cols[i % len(cols)]:
            _label(p.display_name)
            params[p.name] = st.slider(
                p.name, min_value=float(p.min_value), max_value=float(p.max_value),
                value=st.session_state.get(f"{kp}{p.name}", float(defaults.get(p.name, p.default))),
                step=float(p.step), format=p.format, key=f"{kp}{p.name}",
                label_visibility="collapsed",
            )
