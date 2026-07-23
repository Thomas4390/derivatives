"""
Model dispatch adapter — single source of model → parameter/volatility/name/method
resolution for the simulation app, backed by ``config.model_registry``.

Before this module, the same ``if model == "gbm": ... elif model == "heston": ...``
chain was hand-written in ~10 places across services. They now all delegate here,
and the registry (``ModelSpec`` / ``ParameterSpec``) is the only place that knows a
model's parameter names, defaults, display name and pricing methods.

The mapping is **behaviour-preserving**: the registry's display names and pricing
methods match the previously hard-coded values exactly; the volatility handle is
derived from which parameter the model exposes (``v0`` → √v0, ``sigma0``, ``sigma``).

Author: Thomas Vaudescal
"""

from __future__ import annotations

from typing import Any

import numpy as np

from config.model_registry import ModelSpec, get_model


def _spec(model_key: str) -> ModelSpec | None:
    """Registry spec for ``model_key`` (incl. the session-state custom model), or None."""
    try:
        return get_model(model_key)
    except ValueError:
        return None


def extract_params(model_key: str, params: dict[str, Any]) -> dict[str, Any]:
    """Model-specific parameters pulled from ``params`` with registry defaults.

    For each registry ``ParameterSpec`` the value is the first present key among
    ``(*spec.aliases, spec.name)`` (so UI aliases such as NGARCH ``gamma_ngarch``
    map back to canonical ``gamma``), else the registry default. Raises
    ``ValueError`` for an unknown model — same contract as the old hand-coded
    ``_extract_model_params``.
    """
    spec = _spec(model_key)
    if spec is None:
        raise ValueError(f"Unknown model: {model_key}")
    extracted: dict[str, Any] = {}
    for p in spec.parameters:
        value: Any = p.default
        for key in (*p.aliases, p.name):
            if key in params:
                value = params[key]
                break
        extracted[p.name] = value
    return extracted


def initial_volatility(model_key: str, params: dict[str, Any]) -> float:
    """Representative instantaneous volatility for the model.

    ``v0`` models report √v0 (variance handle), ``sigma0``/``sigma`` models report
    the value directly. Custom models fall back to the first vol-like parameter,
    taking √ when the value looks like a variance (>= 1.0).
    """
    spec = _spec(model_key)
    if spec is None:
        return 0.20
    names = {p.name for p in spec.parameters}
    if "v0" in names:
        return float(np.sqrt(params.get("v0", 0.04)))
    if "sigma0" in names:
        return float(params.get("sigma0", 0.20))
    if "sigma" in names:
        return float(params.get("sigma", 0.20))
    for p in spec.parameters:
        lname = p.name.lower()
        if "sigma" in lname or "vol" in lname or "v0" in lname:
            val = float(params.get(p.name, p.default))
            return val if val < 1.0 else float(np.sqrt(val))
    return 0.20


def display_name(model_key: str) -> str:
    """Human-readable model name from the registry (custom-aware)."""
    spec = _spec(model_key)
    return spec.name if spec is not None else model_key


def available_pricing_methods(model_key: str) -> list[str]:
    """Pricing method keys the model supports, from the registry."""
    spec = _spec(model_key)
    if spec is None:
        return ["monte_carlo"]
    return [m.value for m in spec.pricing_methods]
