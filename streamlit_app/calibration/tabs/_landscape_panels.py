"""Presentation panels of the Loss Landscape tab.

Extracted from ``tabs/landscape.py`` (file-size budget): the pedagogy
expander, the converged-at-seed caption, and the slice-diagnostics block.
Pure presentation — every value is read from already-computed objects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from charts.landscape import LandscapeLayer
from services.landscape_service import basin_curvature, slice_minimum
from tabs._helpers import fmt_param_value
from streamlit_app.simulation.config.styles import (  # type: ignore
    render_stats_row,
    section_header_html,
)


def render_pedagogy_panel(model_key_hint: str | None) -> None:
    """Pedagogical explainer rendered above the chart.

    Collapsed by default so the panel does not push the chart below the
    fold for repeat users, but always available for first-timers. The
    content focuses on *reading* the chart — what every colour, marker,
    and *empty* region means — and on the subtle but important fact that
    a 2-D slice is not the full objective.
    """
    with st.expander(
        "📖 How to read this chart (read once before exploring)",
        expanded=False,
    ):
        st.markdown(
            """
**What is a loss landscape?**

Calibration tunes the model parameters $\\theta$ to reproduce the market.
For every candidate $\\theta$, we measure how badly the model misses its
target — that number is the **loss** $L(\\theta)$. The set of all
$(\\theta, L(\\theta))$ pairs is a high-dimensional landscape; calibration
is the descent from a starting guess down into a valley.

For a model with $p$ parameters, the landscape lives in $\\mathbb{R}^p$ and
you cannot see it. **What this tab shows is a 2-D *slice*** of that
landscape: two parameters vary on the $(x, y)$ axes, every other parameter
is **frozen** at the values listed at the bottom of the chart.

**What "loss" means per family**

- **Surface models (FFT)** — Heston, Bates, Merton, Heston-Nandi and an
  FFT-capable custom model: the surface is re-priced in closed form at
  every cell and the loss is the calibration objective you selected
  (Price MSE, IV MSE, …).
- **Monte-Carlo surface models** — the risk-neutral GARCH-Q trio and
  MC-only custom models: same objective, but each cell prices by MC with a
  **fixed seed** (common random numbers). The surface is smooth and
  deterministic; only its *level* shifts slightly vs the full-path
  objective the solver saw, never the basin shape.
- **Returns models (GARCH / NGARCH / GJR)** — the loss is the **negative
  log-likelihood** of the return series, the very objective the MLE
  calibrator minimises. Axes are in per-period (daily) units — ω is the
  per-period intercept, exactly the scale the solver searched. The NLL can
  be negative, so the log-scale toggle is disabled.

**How to read the colour map**

- **Dark blue / purple** → low loss → model fits the target well.
- **Yellow / red** → high loss → model misses the target.
- The **valley** (lowest dark region) is the slice's local optimum.

**The overlays**

| Marker | Meaning |
|---|---|
| 🟢 *Green star (2-D) / green diamond (3-D)* | Synthetic-mode ground truth — the parameters that **generated** the data. The solver should land here. (Plotly's 3-D `Scatter3d` has no star symbol, so the 3-D view falls back to a diamond.) |
| ⬤ *Dark circle (white ring)* | The solver's **initial guess** $x_0$ — typically a heuristic (ATM IV for Heston, etc.). |
| *Solver-coloured polyline* | The **trajectory** — every step the optimiser made, from $x_0$ to its final answer. Each solver keeps its own colour (LM-JAX teal, DE amber, NM purple, L-BFGS-B blue). |
| 🩷 *Pink open diamonds* | **Multi-start endpoints** (LM-JAX only) — final point of each restart. They tell you whether all restarts agreed. |
| ⚪ *Dotted curve* | The **Feller boundary** $2\\kappa\\sigma^2 = \\alpha^2$ (Heston/Bates only). Crossing it makes variance negative. |
| ⚪ *Dashed curve* | The **stationarity boundary** (GARCH families) — persistence = 1 (e.g. $\\alpha + \\beta = 1$). Beyond it the variance process explodes; the soft barrier in the likelihood starts at persistence ≈ 0.995, so the loss wall is visible just inside the line. |

**Why doesn't the colour cover the whole rectangle?**

The blank / white regions are **not bugs** — they are **infeasible
regions** of parameter space:

1. **Feller violation (Heston/Bates).** If $2\\kappa\\sigma^2 < \\alpha^2$, the
   variance process can hit zero in finite time and the model is rejected.
   The pricer returns `NaN`, the contour stays blank.
2. **Pricing failures.** Some configurations make the characteristic
   function or the FFT inversion diverge (e.g. $\\alpha \\to 0$, extreme
   correlations $|\\rho| \\to 1$). The cell is left as `NaN` and stays
   blank.
3. **Numerical exceptions** (overflow, division by zero in the closed-form
   formulas) are caught and reported as `NaN` so a single bad cell does
   not crash the whole grid.

This is also why **the contour map is sometimes very narrow**: the user
toggle "Full spec range" is OFF by default and we zoom **±3 ×|estimate|**
around the solver's best point — a tight window where the curvature is
visible. Turn the toggle ON to see the full bounded region, including
how much of it is actually infeasible.

**Caveat: 2-D slice ≠ full optimisation problem**

The solver does not walk along this slice. It walks in $\\mathbb{R}^p$,
varying *all* parameters at once. The trajectory you see is **projected**
onto the $(x, y)$ plane. A point that looks like a yellow plateau here
may actually be a deep valley in the full $p$-dimensional space because
the hidden parameters were different at that step. Use this chart to
build intuition, not to second-guess the optimiser.
"""
        )


# Below this fraction of the slice window, a solver trajectory is
# indistinguishable from the x₀ marker and the user sees no path at
# all. Two-axis tolerance (max of relative spans) so we only fire when
# *both* parameters barely moved.
_DEGENERATE_TRAJECTORY_THRESHOLD: float = 0.01


def render_convergence_at_seed_caption(
    layers: list[LandscapeLayer],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    """Tell the student why a solver path can shrink into the seed marker.

    LM-JAX seeded from the ATM-IV inverse converges almost instantly when
    the synthetic surface uses the default true parameters — the path
    spans a fraction of a percent of the window and disappears under x₀.
    Flag the case explicitly so they don't read it as a bug.
    """
    window_x = max(abs(x_range[1] - x_range[0]), 1e-12)
    window_y = max(abs(y_range[1] - y_range[0]), 1e-12)
    degenerate: list[tuple[str, int]] = []
    for layer in layers:
        for solver_name, (xs, ys) in (layer.solver_trajectories or {}).items():
            if len(xs) < 2:
                continue
            rel_x = float(np.ptp(xs)) / window_x
            rel_y = float(np.ptp(ys)) / window_y
            if max(rel_x, rel_y) < _DEGENERATE_TRAJECTORY_THRESHOLD:
                degenerate.append((solver_name, len(xs)))
    if not degenerate:
        return
    names = ", ".join(f"`{name}` ({nfev} evals)" for name, nfev in degenerate)
    st.caption(
        f"ℹ️ {names} converged right at the seed. On synthetic surfaces "
        f"the ATM-IV-derived x₀ already sits inside the basin, so the "
        f"trajectory spans less than 1% of the window and hides under "
        f"the x₀ marker. Change the true parameters in the sidebar to "
        f"displace the optimum and watch the path stretch out."
    )


# Basin condition-number verdict → stats-card colour variant.
_KAPPA_VARIANT: dict[str, str] = {
    "well-conditioned": "teal",
    "moderately elongated": "amber",
    "highly elongated": "red",
}


def _anchor_loss_card_value(
    anchor_slice_loss: float | None,
    anchor_objective_value: float | None,
) -> tuple[str, str]:
    """``(card value, sub-label suffix)`` for the "{anchor} optimum" card.

    The primary value is the slice loss re-evaluated at the anchor's estimate
    with the landscape's own cell loss — grid-comparable by construction for
    every backend. The solver's own ``objective_value`` goes to the sub-label:
    it is NOT grid-comparable (surface calibrators report rss/2, the grid
    shows the objective's RMSE-style ``compute_loss``; the MC backend also
    prices with a reduced path budget), but showing it beside the slice loss
    makes that level difference explicit instead of silently mixing units —
    the old card squared ``rmse_iv`` regardless of the calibrated objective.
    """
    value = "—"
    if anchor_slice_loss is not None and np.isfinite(anchor_slice_loss):
        value = f"{anchor_slice_loss:.3e}"
    suffix = ""
    if anchor_objective_value is not None and np.isfinite(anchor_objective_value):
        suffix = f" · solver objective {anchor_objective_value:.2e}"
    return value, suffix


def render_slice_diagnostics(
    *,
    result: Any,
    anchor_summary: Any,
    anchor_name: str | None,
    anchor_slice_loss: float | None = None,
    px: str,
    py: str,
    out_of_range: int,
    solver_trajectories: list[str],
    base_params: dict[str, float],
    param_names: list[str],
    data_source: str,
    inspected: str,
    generator: str | None,
) -> None:
    """Render the slice-diagnostics panel below the landscape chart.

    A coloured stat-card row (slice minimum, solver optimum, the gap between
    them, and the basin condition number κ(H)), a one-line legend, and a
    collapsible explainer holding the 2-D-slice caveat and the anchoring rule.
    Pure presentation — every value is read from already-computed objects.
    """
    minimum = slice_minimum(result)

    anchor_coords: tuple[float, float] | None = None
    anchor_objective_value: float | None = None
    if anchor_summary is not None and anchor_summary.estimated_params:
        est = anchor_summary.estimated_params
        if px in est and py in est:
            anchor_coords = (float(est[px]), float(est[py]))
        if anchor_summary.result is not None:
            anchor_objective_value = float(anchor_summary.result.objective_value)

    curvature = basin_curvature(result)

    # ── Stat cards — built dynamically; render_stats_row fits the columns ──
    stats: list[tuple[str, str, str]] = []
    variants: list[str] = []
    if minimum is not None:
        xm, ym, lm = minimum
        stats.append(
            ("Slice minimum", f"{lm:.3e}", f"{px} = {xm:.3g} · {py} = {ym:.3g}")
        )
        variants.append("blue")
    if anchor_name is not None and anchor_coords is not None:
        loss_val, obj_suffix = _anchor_loss_card_value(
            anchor_slice_loss, anchor_objective_value
        )
        stats.append(
            (
                f"{anchor_name} optimum",
                loss_val,
                f"{px} = {anchor_coords[0]:.3g} · {py} = {anchor_coords[1]:.3g}"
                f"{obj_suffix}",
            )
        )
        variants.append("green")
    if minimum is not None and anchor_coords is not None:
        dx = abs(anchor_coords[0] - minimum[0])
        dy = abs(anchor_coords[1] - minimum[1])
        stats.append(("Min ↔ optimum gap", f"Δ{px} = {dx:.3g}", f"Δ{py} = {dy:.3g}"))
        variants.append("amber")
    if curvature is not None:
        kappa = curvature["kappa"]
        lam_min = curvature["lambda_min"]
        lam_max = curvature["lambda_max"]
        verdict = (
            "well-conditioned"
            if kappa < 5
            else "moderately elongated"
            if kappa < 30
            else "highly elongated"
        )
        stats.append(
            (
                "Condition κ(H)",
                f"{kappa:,.0f}",
                f"{verdict} · λ ∈ [{lam_min:.1e}, {lam_max:.1e}]",
            )
        )
        variants.append(_KAPPA_VARIANT[verdict])

    if stats:
        st.markdown(
            section_header_html("📐", "Slice diagnostics"), unsafe_allow_html=True
        )
        render_stats_row(stats, variants)

    # ── Out-of-window trajectory warning ──────────────────────────────
    if out_of_range > 0:
        st.caption(
            f"⚠ {out_of_range} trajectory point(s) fall outside the slice "
            "window — toggle **Full spec range** above to widen the view."
        )

    # ── Legend (the map key — stays visible) ──────────────────────────
    overlaid = (
        ", ".join(solver_trajectories) if solver_trajectories else "none selected"
    )
    st.caption(
        "Legend: 🟢 ground truth (green star in 2-D, diamond in 3-D) · "
        "⬤ initial guess x₀ (dark circle, white ring) · "
        "solver-coloured trajectory · 🩷 pink multi-start endpoints (LM-JAX) · "
        "⚪ Feller (dotted) / stationarity (dashed) boundary.  "
        f"Trajectories overlaid: {overlaid}."
    )

    # ── Collapsible explainer: slice caveat + anchoring rule ──────────
    with st.expander("ℹ️ How to read this slice", expanded=False):
        if minimum is not None and anchor_coords is not None:
            st.markdown(
                "Large gaps between the **slice minimum** and the **solver "
                "optimum** mean the basin curves through the *hidden* parameters."
            )
        if curvature is not None and curvature["kappa"] > 30:
            st.markdown(
                "Gradient-free solvers (NM, DE) struggle when the condition "
                "number κ(H) ≫ 1 — the basin is a long, narrow valley."
            )
        hidden_str = ", ".join(
            f"`{n}` = {fmt_param_value(base_params[n])}"
            for n in param_names
            if n not in (px, py)
        )
        st.markdown(
            "**⚠ This is a 2-D slice of a high-dimensional objective.** The "
            f"contour shows the loss with hidden parameters frozen at "
            f"{hidden_str or '(none)'}. Solver trajectories pass through "
            "*different* hidden-param values along the way, so the true loss at "
            "a trajectory point is generally lower than what the contour reads "
            "at the same (x, y) location."
        )
        if anchor_name is not None:
            stopped = (
                " *(best-so-far — solver was stopped)*"
                if getattr(anchor_summary, "partial", False)
                else ""
            )
            st.markdown(
                f"Slice anchored at the **{anchor_name}** estimated point.{stopped} "
                "Toggle **Center on solver optimum** is implicit here — the "
                f"hidden parameters and zoom window are derived from {anchor_name}'s fit."
            )
        elif data_source == "synthetic" and inspected == generator:
            st.markdown(
                "Slice anchored at the **synthetic ground truth** parameters "
                "(no calibration has run yet). Run a calibration to anchor the "
                "slice at the optimiser's optimum instead."
            )
        else:
            st.markdown(
                "Slice anchored at the model's **registry defaults** (no run "
                "yet). Run a calibration to anchor the slice at the optimum."
            )
