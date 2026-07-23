"""Objective function picker + advanced settings + pedagogical explainer.

Mirrors :mod:`components.sidebar.solver_panel`. The user can pick one or
more objective functions to compare side-by-side; results are nested as
``calib_results[model][solver][objective]``. GARCH and iv_gbm models do
not expose objective selection because their calibration is either MLE
on returns or a closed-form IV inversion that doesn't consume a loss in
the same sense.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from config.constants import (
    GARCH_FAMILY,
    HUBER_DELTA_DEFAULT,
    OBJECTIVE_DISPLAY_NAMES,
    OBJECTIVE_GROUP_HELP,
    OBJECTIVE_HOVER,
    OBJECTIVE_ICONS,
)
from config.model_registry import supported_objectives
from services import state_manager

from ._option_help import compose_help


def _format_objective(name: str) -> str:
    return f"{OBJECTIVE_ICONS.get(name, '·')} {OBJECTIVE_DISPLAY_NAMES.get(name, name)}"


def _intersection(candidates: tuple[str, ...]) -> tuple[str, ...]:
    """Objectives supported by **every** candidate model in the run."""
    if not candidates:
        return ()
    sets = [set(supported_objectives(m)) for m in candidates]
    inter = sets[0]
    for s in sets[1:]:
        inter &= s
    # Preserve the canonical order: follow supported_objectives() of the
    # first candidate.
    first_order = supported_objectives(candidates[0])
    return tuple(o for o in first_order if o in inter)


def render(candidates: tuple[str, ...]) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Render the objective multi-select + pedagogical expander.

    Returns ``(objective_tuple, objective_settings)``. ``objective_tuple``
    is empty when the candidate models don't expose an objective
    selection (GARCH/iv_gbm) — the UI displays an info banner instead.
    """
    inter = _intersection(candidates)
    st.subheader("🎯 Objective functions")
    if not inter:
        if not candidates:
            st.info("Pick at least one candidate model to enable objective selection.")
        elif all(c in GARCH_FAMILY for c in candidates):
            st.info(
                "GARCH-family calibration uses maximum-likelihood on log-returns — "
                "the loss is the log-likelihood, not a price-error norm."
            )
        else:
            st.info(
                "Selected model uses a closed-form inversion — no iterative loss to choose."
            )
        return (), {}

    current = state_manager.get("calib_objective_selection") or ("price_mse",)
    valid_current = [o for o in current if o in inter] or [inter[0]]

    picked = st.pills(
        "Active objectives",
        options=list(inter),
        default=valid_current,
        selection_mode="multi",
        format_func=_format_objective,
        key="objective_pick_multi",
        help=compose_help(
            OBJECTIVE_GROUP_HELP, inter, OBJECTIVE_HOVER, _format_objective
        ),
    )
    selection = list(picked) if picked else []
    if not selection:
        selection = [inter[0]]
        st.warning(f"No objective selected → falling back to {_format_objective(inter[0])}.")

    settings = dict(state_manager.get("calib_objective_settings") or {})

    # Advanced settings appear only when the relevant objective is picked.
    needs_huber = "huber" in selection
    needs_relative = "relative" in selection
    if needs_huber or needs_relative:
        with st.expander("⚙️ Objective settings", expanded=False):
            if needs_huber:
                settings["huber_delta"] = float(
                    st.slider(
                        "Huber threshold δ (price units)",
                        min_value=0.001,
                        max_value=1.0,
                        value=float(settings.get("huber_delta", HUBER_DELTA_DEFAULT)),
                        step=0.005,
                        format="%.3f",
                        help=(
                            "Below δ: quadratic penalty (sensitive to small errors). "
                            "Above δ: linear penalty (robust to outliers). "
                            "Rule of thumb: 5% of the median quote price."
                        ),
                        key="objective_huber_delta",
                    )
                )
            if needs_relative:
                settings["relative_use_log"] = st.checkbox(
                    "Log-price mode",
                    value=bool(settings.get("relative_use_log", False)),
                    help=(
                        "Checked: residual = log(P_mod) − log(P_mkt). "
                        "Otherwise: (P_mod − P_mkt) / P_mkt. Log mode is more "
                        "stable for large errors but asymmetric on over/under-pricing."
                    ),
                    key="objective_relative_use_log",
                )

    if "iv_mse" in selection:
        st.caption(
            "ℹ️  **IV MSE** is JAX-traceable via the implicit function theorem: "
            "the forward IV inversion runs through `jax.pure_callback`, and the "
            "JVP propagates $\\partial\\sigma/\\partial P = 1/\\mathcal{V}^{BS}(\\sigma^\\star)$ "
            "— the same identity that motivates `vega_weighted` as a first-order "
            "approximation, but evaluated *exactly* at the converged $\\sigma^\\star$."
        )

    # ── Pedagogy expander ───────────────────────────────────────────
    _render_objective_pedagogy()

    state_manager.update(
        calib_objective_selection=tuple(selection),
        calib_objective_settings=settings,
    )
    return tuple(selection), settings


def _render_objective_pedagogy() -> None:
    """Collapsible explainer comparing the 6 objectives with formulas + table."""
    with st.expander(
        "📖 Compare the objective functions (literature)",
        expanded=False,
    ):
        st.markdown(
            r"""
**Why several?** Each objective function $L(\theta)$ encodes a different *definition*
of "good fit". The choice depends on the data (liquidity, moneyness range) and on
the end use (vanilla pricing / exotics / hedging / market-making).

#### Catalogue (6 objectives, signed residual $r_i$ — LM minimises $\tfrac{1}{2}\sum_i r_i^2$)

| Objective | Residual $r_i$ | Strength | Risk |
|---|---|---|---|
| **Price MSE** | $P^{\mathrm{mod}}_i - P^{\mathrm{mkt}}_i$ | Standard, JAX-friendly, numerically robust | ATM (large prices) dominate |
| **IV MSE** | $\sigma^{\mathrm{mod}}_i - \sigma^{\mathrm{mkt}}_i$ | Uniform across moneyness; **exact** under LM-JAX via implicit-function-theorem JVP ($1/\mathcal{V}^{BS}$) | One IV inversion per quote per eval (slower than vega-weighted) |
| **Vega-weighted** | $\dfrac{P^{\mathrm{mod}}_i - P^{\mathrm{mkt}}_i}{\mathcal{V}^{BS}_i}$ | Approximates IV-MSE at near-zero cost | Assumes spread $\propto$ vega |
| **Spread-weighted** | $\dfrac{P^{\mathrm{mod}}_i - P^{\mathrm{mkt}}_i}{\mathrm{spread}^{BA}_i}$ | Down-weights illiquid quotes | Requires clean bid-asks |
| **Relative / Log** | $\dfrac{P^{\mathrm{mod}}_i - P^{\mathrm{mkt}}_i}{P^{\mathrm{mkt}}_i}$ or $\log\tfrac{P^{\mathrm{mod}}_i}{P^{\mathrm{mkt}}_i}$ | Scale-invariant across maturities | Unstable for deep OTM |
| **Huber** | $\sqrt{w_i}\,r_i,\;w_i=\min(1,\delta/|r_i|)$ | Robust to outliers (Schoutens 2003) | Requires δ calibration |

#### Rule of thumb (industry recommendation)
- **Clean synthetic** → `price_mse` or `vega_weighted` (gold-standard Heston).
- **Real SPX surface with spreads** → `spread_weighted` or `vega_weighted`.
- **Crypto / low liquidity** → `huber` ($\delta\approx$ 5–10% of the median price).
- **Wide maturity range** → `relative` or `log` to avoid long-dated dominance.
- **Pure pedagogy** → `iv_mse` with a scalar solver (DE/NM) to see the *direct* effect on implied volatility.

#### The calibration "triangle"
$$
\big(\textbf{model},\;\textbf{solver},\;\textbf{objective}\big)
$$
Changing **one** of these three dimensions generally moves the global optimum and modifies parameter sensitivity. The *Loss Landscape* and *Diagnostics* tabs let you visualise this effect by varying the objective without recalibrating the model.

**Sources**: Cont & Tankov (2004) §13.1; Schoutens, Simons & Tistaert (2003);
Guillaume & Schoutens (2014); Pacati, Pompa & Renò (2018); Huber (1964); arxiv 2207.02989.
"""
        )
