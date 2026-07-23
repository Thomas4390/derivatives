"""
Calibration search space — canonical parameter bounds
=====================================================

Single source of truth for the **search universe** (per-parameter ``(lo, hi)``
box) that every calibrator explores. It is consumed in two places:

1. the calibrators themselves — the sigmoid-bijection models (Heston, Bates,
   Merton, Heston-Nandi) read their canonical box from here, and the
   direct-box models (GARCH family, risk-neutral GARCH-Q) fall back to it when
   no per-parameter override is supplied; and
2. the Streamlit *Search space* sidebar panel — which seeds its widgets with
   :func:`default_search_bounds` and lets the user tighten each parameter's
   ``[min, max]`` to a sub-interval of the admissible box.

This module is intentionally **pure** (no JAX / NumPy / Streamlit) so it can be
imported from the calibrators *and* the UI without pulling heavy dependencies
or risking a circular import — it depends only on
:mod:`backend.utils.constants.calibration`.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from typing import Final

from backend.utils.constants.calibration import (
    GARCH_BOUNDS,
    GJR_BOUNDS,
    HESTON_NANDI_BOUNDS,
    NGARCH_BOUNDS,
    RISK_NEUTRAL_GARCH_BOUNDS,
)

# --------------------------------------------------------------------------- #
# Canonical boxes for the sigmoid-bijection models
# --------------------------------------------------------------------------- #
# These ARE the live optimiser bounds: each calibrator's sigmoid map sends the
# unconstrained coordinate to ``lo + (hi - lo)·σ(θ)``. They are kept here (not
# in ``constants/calibration.py``) so the calibrators and the UI share one
# definition. NB: they deliberately differ from the legacy ``HESTON_BOUNDS`` /
# ``BATES_BOUNDS`` finals in ``constants/calibration.py`` (v0 floor 1e-5 vs
# 1e-3; Heston α ceiling 2.0 vs 3.0) — those finals are re-export-only and were
# never consumed by the calibrators, which carried their own module constants.
HESTON_SEARCH_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "v0": (1e-5, 1.0),
    "kappa": (0.01, 20.0),
    "theta": (1e-5, 1.0),
    "alpha": (1e-3, 2.0),
    "rho": (-0.999, 0.999),
}

MERTON_SEARCH_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "sigma": (0.01, 1.0),
    "lam": (1e-3, 5.0),
    "alpha_j": (-0.5, 0.1),
    "sigma_j": (0.01, 0.5),
}

BATES_SEARCH_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "v0": (1e-5, 1.0),
    "kappa": (0.01, 20.0),
    "theta": (1e-5, 1.0),
    "alpha": (1e-3, 2.0),
    "rho": (-0.999, 0.999),
    "lam": (1e-3, 5.0),
    "alpha_j": (-0.5, 0.1),
    "sigma_j": (0.01, 0.5),
}

# Ordered parameter names for the direct-box GARCH families (the bounds tuples
# in ``constants`` are positional). Mirrors ``GARCHCalibrator._param_names``.
_GARCH_NAMES: Final[tuple[str, ...]] = ("omega", "alpha", "beta")
_NGARCH_NAMES: Final[tuple[str, ...]] = ("omega", "alpha", "beta", "gamma")
_GJR_NAMES: Final[tuple[str, ...]] = ("omega", "alpha", "beta", "gamma")

# App model-key -> risk-neutral GARCH simulator variant (the RN-GARCH dict key).
_RN_VARIANT: Final[dict[str, str]] = {
    "ngarch_q": "ngarch",
    "garch_q": "garch",
    "gjr_q": "gjr_garch",
}


def default_search_bounds(model_key: str) -> dict[str, tuple[float, float]]:
    """Canonical per-parameter search box ``{param: (lo, hi)}`` for a model.

    The dict is keyed by the model's calibratable parameter names in their
    natural order. Returns an empty dict for models with no iterative search
    (``iv_gbm`` is a closed-form IV inversion).

    Raises
    ------
    KeyError
        If ``model_key`` is not a known calibration model.
    """
    if model_key == "heston":
        return dict(HESTON_SEARCH_BOUNDS)
    if model_key == "bates":
        return dict(BATES_SEARCH_BOUNDS)
    if model_key == "merton":
        return dict(MERTON_SEARCH_BOUNDS)
    if model_key == "heston_nandi":
        return dict(HESTON_NANDI_BOUNDS)
    if model_key == "garch":
        return dict(zip(_GARCH_NAMES, GARCH_BOUNDS))
    if model_key == "ngarch":
        return dict(zip(_NGARCH_NAMES, NGARCH_BOUNDS))
    if model_key == "gjr_garch":
        return dict(zip(_GJR_NAMES, GJR_BOUNDS))
    if model_key in _RN_VARIANT:
        return dict(RISK_NEUTRAL_GARCH_BOUNDS[_RN_VARIANT[model_key]])
    if model_key == "iv_gbm":
        return {}
    raise KeyError(f"Unknown calibration model '{model_key}'")


def clamp_box(
    box: dict[str, tuple[float, float]],
    admissible: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Clamp each ``(lo, hi)`` in ``box`` to the matching admissible interval.

    Keeps only parameters present in ``admissible``, orders ``(lo, hi)`` and
    guarantees ``lo <= hi`` (a degenerate / inverted user range collapses to the
    admissible interval for that parameter). Used to keep a user-supplied search
    box inside each model's admissible region.
    """
    out: dict[str, tuple[float, float]] = {}
    for name, (a_lo, a_hi) in admissible.items():
        if name not in box:
            out[name] = (a_lo, a_hi)
            continue
        lo, hi = float(box[name][0]), float(box[name][1])
        if lo > hi:
            lo, hi = hi, lo
        lo = max(a_lo, min(lo, a_hi))
        hi = max(a_lo, min(hi, a_hi))
        if lo >= hi:  # degenerate after clamping — fall back to the full box
            lo, hi = a_lo, a_hi
        out[name] = (lo, hi)
    return out


def tighten_around_truth(
    model_key: str,
    true_params: dict[str, float],
    frac: float = 0.5,
    admissible: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    """Sub-box of ±``frac`` around each true value, clamped to the admissible box.

    Pedagogical preset: starting the optimiser from a universe that brackets the
    ground truth isolates *local* identifiability from the *global* search
    problem. Parameters absent from ``true_params`` keep their full admissible
    interval. ``admissible`` overrides the canonical box for models unknown to
    :func:`default_search_bounds` (e.g. a session-registered custom model).
    """
    if admissible is None:
        admissible = default_search_bounds(model_key)
    proposed: dict[str, tuple[float, float]] = {}
    for name, value in true_params.items():
        if name not in admissible:
            continue
        span = abs(float(value)) * float(frac)
        if span == 0.0:  # value is exactly 0 (e.g. a pinned γ) — use half-box width
            a_lo, a_hi = admissible[name]
            span = 0.5 * frac * (a_hi - a_lo)
        proposed[name] = (float(value) - span, float(value) + span)
    return clamp_box(proposed, admissible)
