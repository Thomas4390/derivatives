"""Reusable per-model equation card.

Renders one model's SDE / variance recursion, its calibration loss
shape, the penalty terms relevant under the user-selected constraint
mode, and a Markdown parameter cheat-sheet — sharing identical content
between the Theory tab (full pedagogy) and the Setup-tab Equations
panel (live, sidebar-coupled view).

The Theory tab keeps its existing visual signature by setting
``wrap_in_expander=True`` (default), which renders the loss section
inside an ``st.expander`` titled "📐 Equations · <model name>". The
Setup-tab panel sits already inside an outer expander + tabs container,
so it passes ``wrap_in_expander=False`` to render the same content flat.
"""

from __future__ import annotations

import streamlit as st

from config.constants import (
    GARCH_FAMILY,
    MODEL_DISPLAY_NAMES,
    SURFACE_FAMILY,
)
from config.formulas import (
    DUAN_LRNVR_LATEX,
    FELLER_PENALTY_LATEX,
    GARCH_LOSS_LATEX,
    MODEL_SDE_LATEX,
    PARAM_CHEATSHEET,
    STATIONARITY_PENALTY_LATEX,
    STATIONARITY_PERSISTENCE_LATEX,
    SURFACE_LOSS_LATEX,
    TIKHONOV_PENALTY_LATEX,
)
from streamlit_app.simulation.config.styles import section_header_html  # type: ignore

# Q-measure surface GARCH variants — calibrated by Levenberg-Marquardt
# on a closed-form characteristic function (heston_nandi) or by
# Monte-Carlo under Duan's LRNVR (ngarch_q / garch_q / gjr_q).
_Q_GARCH_MODELS: tuple[str, ...] = ("heston_nandi", "ngarch_q", "garch_q", "gjr_q")

# Q-measure variants that carry a leverage parameter γ. The Duan map
# γ* = γ + λ is meaningful only when γ exists, so the symmetric
# ``garch_q`` is intentionally excluded.
_Q_GARCH_LEVERAGE_MODELS: tuple[str, ...] = ("heston_nandi", "ngarch_q", "gjr_q")


def render_model_card(
    model_key: str,
    *,
    wrap_in_expander: bool = True,
    show_cheatsheet: bool = True,
    feller_mode: str | None = None,
    stationarity_mode: str | None = None,
) -> None:
    """Render one model's equation card.

    Parameters
    ----------
    model_key:
        Canonical model key — must be present in ``MODEL_SDE_LATEX``.
    wrap_in_expander:
        ``True`` (default) wraps the loss section in an ``st.expander``
        with label ``"📐 Equations · <display name>"`` — the Theory-tab
        voice. ``False`` renders flat for callers that already provide
        their own container (the Setup-tab panel).
    show_cheatsheet:
        Render the Markdown parameter cheat-sheet under the loss
        section. ``True`` by default to keep the Theory tab unchanged.
    feller_mode, stationarity_mode:
        Constraint modes (``"off" | "soft" | "hard"``) read from the
        sidebar. ``None`` keeps the historical Theory-tab behaviour of
        always showing the relevant penalty (educational default).
    """
    if wrap_in_expander:
        label = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        with st.expander(f"📐 Equations · {label}", expanded=False):
            _render_loss_section(model_key, feller_mode, stationarity_mode)
    else:
        _render_loss_section(model_key, feller_mode, stationarity_mode)

    if show_cheatsheet:
        _render_cheatsheet_section(model_key)


# ─────────────────────────────────────────────────────────────────────
# Loss section (SDE + objective + penalties + Duan note)
# ─────────────────────────────────────────────────────────────────────


def _render_loss_section(
    model_key: str,
    feller_mode: str | None,
    stationarity_mode: str | None,
) -> None:
    sde_latex = MODEL_SDE_LATEX.get(model_key)
    if sde_latex:
        st.markdown("**Underlying dynamics**")
        st.latex(sde_latex)

    if model_key in SURFACE_FAMILY:
        st.markdown("**Surface calibration objective**")
        st.latex(SURFACE_LOSS_LATEX)
        st.markdown(
            r"- $\mathrm{IV}^{\text{model}}_{ij}(\theta)$ is obtained by "
            r"inverting Black-Scholes on the model price "
            r"$C^{\text{model}}_{ij}(\theta)$ (Fourier / FFT pricer)."
        )
        st.markdown(
            r"- $w_{ij} = 1$ for equal-weight calibration, "
            r"or $w_{ij} = \mathrm{vega}_{ij}$ for vega-weighted. "
            r"Vega weighting puts ATM and wing errors on the same scale."
        )
        if model_key in ("heston", "bates") and _should_show(feller_mode):
            st.markdown("**Soft penalty: Feller condition**")
            st.latex(FELLER_PENALTY_LATEX)
            st.caption(
                r"Kicks in only when $2\kappa\sigma^2 < \alpha^2$. Keeps the "
                r"variance process strictly positive so paths never touch zero."
            )
        if model_key == "merton":
            st.markdown("**Tikhonov regularisation**")
            st.latex(TIKHONOV_PENALTY_LATEX)
            st.caption(
                r"Pins the diffusion volatility $\sigma$ near the ATM IV. "
                r"Without it, the calibrator can trade jumps for diffusion "
                r"and the parameters stop being identifiable."
            )
        if model_key in _Q_GARCH_MODELS and _should_show(stationarity_mode):
            st.markdown("**Soft penalty: stationarity**")
            persistence = STATIONARITY_PERSISTENCE_LATEX.get(model_key)
            if persistence:
                st.latex(persistence)
            st.latex(STATIONARITY_PENALTY_LATEX)
            st.caption(
                r"Kicks in when the persistence $\rho(\theta)$ gets close to "
                r"or crosses 1. HARD mode reparametrises $\rho < 1$ through "
                r"an interior transform. SOFT adds the quadratic penalty "
                r"above."
            )
        if model_key in _Q_GARCH_LEVERAGE_MODELS:
            st.markdown("**Risk-neutral parameterisation: Duan LRNVR forward map**")
            st.latex(DUAN_LRNVR_LATEX)
            st.caption(
                r"Here the leverage $\gamma$ already lives under $\mathbb{Q}$ "
                r"because the surface fit is itself a $\mathbb{Q}$-measure "
                r"calibration. The relation $\gamma^* = \gamma + \lambda$ is "
                r"how it links back to a physical-measure ($\mathbb{P}$) "
                r"leverage. The app's diagnostics assume $\lambda = 0$ in "
                r"that forward map. "
                r"**$\omega$, $\alpha$, $\beta$ are LRNVR-invariant**: the "
                r"one-step-ahead conditional variance does not change "
                r"$\mathbb{P} \to \mathbb{Q}$, so the sliders for the P and "
                r"Q variants of the same recursion share the same range "
                r"(Heston-Nandi excepted — its $\gamma$ is $\mathcal{O}(100)$ "
                r"and absorbs scale differently)."
            )
    elif model_key in GARCH_FAMILY:
        st.markdown("**GARCH MLE objective (negative log-likelihood)**")
        st.latex(GARCH_LOSS_LATEX)
        st.caption(
            r"The recursion in $\sigma^2_t$ means the calibrator unrolls the "
            r"whole filter at every function evaluation. JAX returns exact "
            r"gradients via automatic differentiation."
        )
        st.markdown("**From P to Q: Duan LRNVR forward map**")
        st.latex(DUAN_LRNVR_LATEX)
        st.caption(
            r"This fit lives under the **physical measure $\mathbb{P}$** "
            r"(real returns). Pricing options means mapping it to the "
            r"**risk-neutral measure $\mathbb{Q}$** through the Duan (1995) "
            r"Local Risk-Neutral Valuation Relationship. The drift becomes "
            r"the risk-free rate. The leverage shifts to "
            r"$\gamma^* = \gamma + \lambda$, where $\lambda$ is the unit "
            r"market price of risk. **$\omega$, $\alpha$, $\beta$ are "
            r"LRNVR-invariant** — the one-step-ahead conditional variance is "
            r"the same under $\mathbb{P}$ and $\mathbb{Q}$, so the sliders "
            r"and calibration bounds for $\omega/\alpha/\beta$ match between "
            r"`garch` $\leftrightarrow$ `garch_q`, `ngarch` $\leftrightarrow$ "
            r"`ngarch_q`, `gjr_garch` $\leftrightarrow$ `gjr_q`. The "
            r"Diagnostics IV surface assumes $\lambda = 0$, so "
            r"$\gamma^* = \gamma$. That is a forward map, not a fit to "
            r"option quotes. To calibrate a risk-neutral GARCH directly "
            r"against a surface, use the Surface family: the Heston-Nandi "
            r"affine model, or the MC-priced NGARCH-Q / GARCH-Q / GJR-Q."
        )


def _should_show(mode: str | None) -> bool:
    """Whether to render a constraint-mode-dependent penalty block.

    ``None`` keeps the Theory-tab default of always showing the
    pedagogical block. ``"off"`` hides it; ``"soft"`` / ``"hard"`` show
    it. Anything else (including empty strings, defensive against UI
    drift) falls back to showing it.
    """
    if mode is None:
        return True
    normalised = str(mode).strip().lower()
    return normalised != "off"


# ─────────────────────────────────────────────────────────────────────
# Cheat-sheet section (Markdown table of parameter symbols)
# ─────────────────────────────────────────────────────────────────────


def _render_cheatsheet_section(model_key: str) -> None:
    rows = PARAM_CHEATSHEET.get(model_key)
    if not rows:
        return
    st.markdown(
        section_header_html(
            "📋",
            f"Parameter cheat sheet · {MODEL_DISPLAY_NAMES.get(model_key, model_key)}",
        ),
        unsafe_allow_html=True,
    )
    family_label = "smile effect" if model_key in SURFACE_FAMILY else "returns effect"
    md_lines = [
        "| Symbol | Name | Typical range | " + family_label.capitalize() + " |",
        "|---|---|---|---|",
    ]
    for sym, name, rng, effect in rows:
        md_lines.append(f"| `{sym}` | {name} | {rng} | {effect} |")
    st.markdown("\n".join(md_lines))


__all__ = ["render_model_card"]
