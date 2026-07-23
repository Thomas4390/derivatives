"""
Custom-model service — compile, validate, register, and calibrate user models.
==============================================================================

Lets the user define their **own** stochastic model in the calibration app and
fit it to the current option surface, mirroring the simulation app's custom-model
workflow but routed through the calibration backend.

Pipeline
--------
1. :func:`compile_and_validate` — execute the user code in a restricted sandbox
   (a curated builtins set + a blocklist of dangerous imports) and run a
   validation suite (interface, SDE, characteristic function, quick simulation).
2. :func:`register_custom_model` / :func:`unregister_custom_model` — store /
   clear the validated class in ``st.session_state['calib_custom_model']``.
   The global model :data:`REGISTRY` is intentionally **not** mutated: the
   custom model lives only in its own tab.
3. :func:`build_custom_calibrator` — wire the registered class to
   :class:`~backend.calibration.CustomModelCalibrator` (FFT or Monte-Carlo route
   picked from ``supported_engines``). It is consumed by
   ``services.calibration_service._calibrator_for`` so the custom model is
   calibrated through the same Run pipeline as the built-in models.

Security note
-------------
``exec`` of user code is **best-effort** sandboxed (no timeout / memory cap); it
remains arbitrary *local* code execution. This matches the deliberate trade-off
already made in the simulation app for a local, single-user pedagogical tool —
there is no network surface and no untrusted multi-tenant exposure.
"""

from __future__ import annotations

import builtins as _builtins
import math
from dataclasses import dataclass, field
from typing import Any

import hashlib

import numpy as np

from backend.calibration.custom_calibrator import (
    DEFAULT_STEPS_PER_YEAR,
    CustomModelCalibrator,
)
from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability

from config.constants import MC_PATHS_INTERACTIVE, MC_SEED

# Session-state key holding the active custom model (calibration app).
SESSION_KEY = "calib_custom_model"

# Module-level copy of the registered metadata. Calibrations run in worker
# threads (services.live_runner) that carry no Streamlit ScriptRunContext, so
# ``st.session_state`` is unreachable there — :func:`get_custom_meta` falls
# back to this copy. Same last-writer-wins scope as the REGISTRY/MODEL_*
# injection in :func:`_install_custom`, which maintains it.
_ACTIVE_META: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# Validation result containers
# --------------------------------------------------------------------------- #


@dataclass
class TestResult:
    """Outcome of a single validation check."""

    name: str
    passed: bool
    message: str


@dataclass
class ValidationResult:
    """Aggregate outcome of the validation suite."""

    tests: list[TestResult] = field(default_factory=list)
    model_class: type[Model] | None = None
    all_passed: bool = False

    def __post_init__(self) -> None:
        self.all_passed = bool(self.tests) and all(t.passed for t in self.tests)


# --------------------------------------------------------------------------- #
# Sandbox configuration
# --------------------------------------------------------------------------- #

_SAFE_BUILTINS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "frozenset",
    "getattr",
    "hasattr",
    "int",
    "isinstance",
    "issubclass",
    "len",
    "list",
    "map",
    "max",
    "min",
    "print",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
    "__build_class__",
    "__name__",
}

_BLOCKED_MODULES = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "http",
    "importlib",
    "ctypes",
    "signal",
    "multiprocessing",
    "threading",
}

_REQUIRED_SPEC_KEYS = {
    "name",
    "display_name",
    "default",
    "min_value",
    "max_value",
    "step",
    "description",
}


def _restricted_namespace() -> dict[str, Any]:
    """Build the restricted global namespace for ``exec``."""
    # Whitelist ONLY the curated names. The previous
    # ``not k.startswith("_") or k in _SAFE_BUILTINS`` admitted every public
    # builtin (open/eval/exec/compile included) because the first clause is
    # already True for any non-dunder name — the whitelist was dead code and
    # the import blocklist was trivially bypassable.
    restricted_builtins = {
        k: getattr(_builtins, k) for k in _SAFE_BUILTINS if hasattr(_builtins, k)
    }
    restricted_builtins["__build_class__"] = _builtins.__build_class__
    restricted_builtins["__name__"] = "__custom_model__"

    original_import = _builtins.__import__

    def _safe_import(name, *args, **kwargs):  # noqa: ANN001, ANN202
        if name.split(".")[0] in _BLOCKED_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in custom models")
        return original_import(name, *args, **kwargs)

    restricted_builtins["__import__"] = _safe_import

    return {
        "__builtins__": restricted_builtins,
        "np": np,
        "numpy": np,
        "math": math,
        "Model": Model,
        "PricingCapability": PricingCapability,
    }


# --------------------------------------------------------------------------- #
# Validation suite
# --------------------------------------------------------------------------- #


def _spec_bounds_error(spec: dict[str, Any]) -> str | None:
    """Return a message when a PARAMETER_SPECS numeric bound is malformed.

    Catches the specs that pass the shape check but crash later: a ``None``
    bound raises ``TypeError`` at ``ModelSpec``/``ParamSpec`` construction
    (leaving a half-registered ghost), and ``min > max`` silently builds a
    reversed box so every scipy Run fails ``'upper bound < lower bound'``.
    """
    try:
        lo = float(spec["min_value"])
        hi = float(spec["max_value"])
        step = float(spec["step"])
        default = float(spec["default"])
    except (TypeError, ValueError):
        return "min_value / max_value / step / default must be finite numbers"
    if not (
        np.isfinite(lo)
        and np.isfinite(hi)
        and np.isfinite(step)
        and np.isfinite(default)
    ):
        return "min_value / max_value / step / default must be finite numbers"
    if lo >= hi:
        return f"min_value ({lo}) must be < max_value ({hi})"
    if step <= 0.0:
        return f"step ({step}) must be > 0"
    if not (lo <= default <= hi):
        return f"default ({default}) must lie in [{lo}, {hi}]"
    return None


def compile_and_validate(source_code: str) -> ValidationResult:
    """Compile user code and run the validation suite (see module docstring)."""
    tests: list[TestResult] = []

    # 1. Compile in the restricted namespace.
    namespace = _restricted_namespace()
    try:
        exec(compile(source_code, "<custom_model>", "exec"), namespace)
        tests.append(TestResult("Compilation", True, "Code compiled successfully"))
    except SyntaxError as exc:
        tests.append(
            TestResult(
                "Compilation", False, f"Syntax error line {exc.lineno}: {exc.msg}"
            )
        )
        return ValidationResult(tests=tests)
    except Exception as exc:  # noqa: BLE001 — surface any sandbox/import error
        tests.append(TestResult("Compilation", False, f"Compilation error: {exc}"))
        return ValidationResult(tests=tests)

    # 2. Discover exactly one public Model subclass.
    model_classes = [
        obj
        for name, obj in namespace.items()
        if isinstance(obj, type)
        and issubclass(obj, Model)
        and obj is not Model
        and not name.startswith("_")
    ]
    if not model_classes:
        tests.append(
            TestResult(
                "Class Discovery",
                False,
                "No Model subclass found. Define a class inheriting from Model.",
            )
        )
        return ValidationResult(tests=tests)
    if len(model_classes) > 1:
        names = ", ".join(c.__name__ for c in model_classes)
        tests.append(
            TestResult(
                "Class Discovery",
                False,
                f"Multiple Model subclasses found ({names}). Define exactly one.",
            )
        )
        return ValidationResult(tests=tests)
    model_class = model_classes[0]
    tests.append(
        TestResult("Class Discovery", True, f"Found class: {model_class.__name__}")
    )

    # 3. PARAMETER_SPECS shape.
    specs = getattr(model_class, "PARAMETER_SPECS", None)
    if not isinstance(specs, list) or not specs:
        tests.append(
            TestResult(
                "PARAMETER_SPECS",
                False,
                "Class must define a non-empty PARAMETER_SPECS list of dicts.",
            )
        )
        return ValidationResult(tests=tests)
    for i, spec in enumerate(specs):
        if not isinstance(spec, dict):
            tests.append(
                TestResult("PARAMETER_SPECS", False, f"Entry {i} is not a dict.")
            )
            return ValidationResult(tests=tests)
        missing = _REQUIRED_SPEC_KEYS - set(spec.keys())
        if missing:
            tests.append(
                TestResult(
                    "PARAMETER_SPECS",
                    False,
                    f"Entry '{spec.get('name', i)}' missing keys: {sorted(missing)}",
                )
            )
            return ValidationResult(tests=tests)
        bounds_error = _spec_bounds_error(spec)
        if bounds_error is not None:
            tests.append(
                TestResult(
                    "PARAMETER_SPECS",
                    False,
                    f"Entry '{spec.get('name', i)}': {bounds_error}",
                )
            )
            return ValidationResult(tests=tests)
    tests.append(
        TestResult("PARAMETER_SPECS", True, f"{len(specs)} parameter(s) defined")
    )

    # 4. Instantiate with defaults.
    defaults = {s["name"]: s["default"] for s in specs}
    try:
        instance = model_class(**defaults)
        tests.append(
            TestResult(
                "Instantiation", True, "Created instance with default parameters"
            )
        )
    except Exception as exc:  # noqa: BLE001
        tests.append(
            TestResult("Instantiation", False, f"Failed to instantiate: {exc}")
        )
        return ValidationResult(tests=tests)

    # 5. Interface compliance.
    try:
        name = instance.name
        assert isinstance(name, str) and name, "name must be a non-empty string"
        engines = instance.supported_engines
        assert (
            isinstance(engines, list) and engines
        ), "supported_engines must be a non-empty list"
        for eng in engines:
            assert isinstance(
                eng, PricingCapability
            ), "supported_engines must contain PricingCapability values"
        assert isinstance(
            instance.get_parameters(), dict
        ), "get_parameters() must return a dict"
        tests.append(
            TestResult(
                "Interface Compliance",
                True,
                f"name='{name}', engines={[e.name for e in engines]}",
            )
        )
    except (AssertionError, Exception) as exc:  # noqa: BLE001
        tests.append(TestResult("Interface Compliance", False, str(exc)))
        return ValidationResult(tests=tests)

    # 6. SDE test (Monte-Carlo route).
    if PricingCapability.MONTE_CARLO in instance.supported_engines:
        try:
            drift_val = instance.drift(100.0, 0.04, 0.0, 0.05, 0.0)
            diff_val = instance.diffusion(100.0, 0.04, 0.0)
            assert np.isfinite(drift_val), f"drift returned non-finite: {drift_val}"
            assert np.isfinite(diff_val), f"diffusion returned non-finite: {diff_val}"
            tests.append(
                TestResult(
                    "SDE Test", True, f"drift={drift_val:.4f}, diffusion={diff_val:.4f}"
                )
            )
        except Exception as exc:  # noqa: BLE001
            tests.append(TestResult("SDE Test", False, f"SDE test failed: {exc}"))
            return ValidationResult(tests=tests)

    # 7. Characteristic-function test (FFT route).
    if PricingCapability.FFT in instance.supported_engines:
        try:
            cf_0 = instance.characteristic_function(0.0, 100.0, 1.0, 0.05, 0.0)
            assert abs(cf_0 - 1.0) < 0.01, f"CF(0) should be ~1.0, got {cf_0}"
            for u in (0.5, 1.0, 2.0, 5.0):
                cf_u = instance.characteristic_function(u, 100.0, 1.0, 0.05, 0.0)
                assert abs(cf_u) <= 1.01, f"|CF({u})| = {abs(cf_u):.4f} > 1"
            tests.append(
                TestResult("CF Test", True, "Characteristic function passed all checks")
            )
        except Exception as exc:  # noqa: BLE001
            tests.append(TestResult("CF Test", False, f"CF test failed: {exc}"))
            return ValidationResult(tests=tests)

    # 8. Quick Monte-Carlo surface-pricing test.
    if PricingCapability.MONTE_CARLO in instance.supported_engines:
        try:
            from backend.calibration.custom_calibrator import CustomTerminalSimulator

            sim = CustomTerminalSimulator(instance, steps_per_year=20)
            terminals = sim.terminals(
                100.0, 0.05, 1.0, n_paths=512, rng=np.random.default_rng(42)
            )
            assert len(terminals) == 512, "unexpected terminal count"
            assert np.all(np.isfinite(terminals)), "non-finite terminal prices"
            assert np.all(terminals > 0), "non-positive terminal prices"
            tests.append(
                TestResult(
                    "Pricing Test", True, f"MC OK, E[S_T]={np.mean(terminals):.2f}"
                )
            )
        except Exception as exc:  # noqa: BLE001
            tests.append(
                TestResult("Pricing Test", False, f"Pricing test failed: {exc}")
            )

    return ValidationResult(tests=tests, model_class=model_class)


# --------------------------------------------------------------------------- #
# Registration (session-state + module-level registries)
# --------------------------------------------------------------------------- #
#
# A registered custom model becomes a first-class **surface** model under the
# key ``"custom"``: it is injected into the module-level registries the rest of
# the app already reads (``REGISTRY`` for parameter specs, the ``MODEL_*`` label
# dicts, ``MODEL_SDE_LATEX`` for the equations card). The sidebar picker then
# offers it, ``true_params`` renders its sliders, and the normal Run pipeline
# calibrates it — no per-call-site edits. Visibility is still gated on
# :func:`is_registered` (a per-session flag) so a registration in one session
# can't surface a ghost option in another.


def _install_custom(meta: dict[str, Any]) -> None:
    """Inject the custom model into the module-level model registries."""
    global _ACTIVE_META

    from config import constants as C
    from config import formulas
    from config.model_registry import REGISTRY, ModelSpec, ParamSpec

    _ACTIVE_META = meta

    params = tuple(
        ParamSpec(
            name=s["name"],
            label=s.get("display_name", s["name"]),
            lo=float(s["min_value"]),
            hi=float(s["max_value"]),
            default=float(s["default"]),
            step=float(s["step"]),
            fmt=s.get("format", "%.4f"),
            description=s.get("description", ""),
        )
        for s in meta["specs"]
    )
    REGISTRY["custom"] = ModelSpec(
        key="custom", display_name=meta["name"], params=params
    )
    C.MODEL_DISPLAY_NAMES["custom"] = meta["name"]
    C.MODEL_ICONS["custom"] = "🧪"
    C.MODEL_HOVER["custom"] = (
        "Your own user-defined model, registered in the Custom Model tab. "
        "Calibrated through the same pipeline as the built-in models."
    )
    C.MODEL_DESCRIPTIONS["custom"] = (
        "User-defined custom model. Priced to the option surface by FFT when it "
        "exposes a characteristic function, otherwise by Monte-Carlo, and fitted "
        "with the scalar solvers (DE / Nelder-Mead / L-BFGS-B)."
    )
    main_eq = (meta.get("eq_latex") or {}).get("main")
    if main_eq:
        formulas.MODEL_SDE_LATEX["custom"] = main_eq


def _remove_custom() -> None:
    """Drop the custom model from the module-level registries."""
    global _ACTIVE_META

    from config import constants as C
    from config import formulas
    from config.model_registry import REGISTRY

    _ACTIVE_META = None
    REGISTRY.pop("custom", None)
    for d in (
        C.MODEL_DISPLAY_NAMES,
        C.MODEL_ICONS,
        C.MODEL_HOVER,
        C.MODEL_DESCRIPTIONS,
    ):
        d.pop("custom", None)
    formulas.MODEL_SDE_LATEX.pop("custom", None)


def register_custom_model(model_class: type[Model], source_code: str) -> None:
    """Validate-already-passed model → session state + module registries."""
    import streamlit as st

    specs = list(model_class.PARAMETER_SPECS)
    instance = model_class(**{s["name"]: s["default"] for s in specs})
    meta = {
        "class": model_class,
        "source": source_code,
        "specs": specs,
        "engines": [e.name for e in instance.supported_engines],
        "name": instance.name,
        "eq_latex": dict(getattr(model_class, "EQUATION_LATEX", {})),
    }
    # Install into the module registries FIRST; only flag the session as
    # registered if that succeeds. Setting SESSION_KEY before a failing install
    # left a ghost: is_registered() True with no REGISTRY['custom'], so the
    # picker offered it and get_spec('custom') KeyErrored on every render.
    _install_custom(meta)
    st.session_state[SESSION_KEY] = meta


def unregister_custom_model() -> None:
    """Remove the custom model from session state and the module registries."""
    import streamlit as st

    st.session_state.pop(SESSION_KEY, None)
    _remove_custom()
    # Fall back to the family default if the custom model was selected.
    for slot in ("calib_generator_model",):
        if st.session_state.get(slot) == "custom":
            st.session_state[slot] = "heston"
    cands = st.session_state.get("calib_candidate_models")
    if cands and "custom" in cands:
        kept = tuple(m for m in cands if m != "custom") or ("heston",)
        st.session_state["calib_candidate_models"] = kept


def is_registered() -> bool:
    import streamlit as st

    return SESSION_KEY in st.session_state


def get_custom_meta() -> dict[str, Any] | None:
    """Registered custom-model metadata — session state first.

    Falls back to the module-level copy when ``st.session_state`` is
    unreachable or empty: calibration workers are plain daemon threads with no
    ScriptRunContext, so without the fallback every custom run died with
    "No custom model registered." and silently vanished from the result tabs.
    """
    try:
        import streamlit as st

        meta = st.session_state.get(SESSION_KEY)
    except Exception:  # noqa: BLE001 — no ScriptRunContext (worker thread)
        meta = None
    return meta if meta is not None else _ACTIVE_META


def get_custom_model_class() -> type[Model] | None:
    meta = get_custom_meta()
    return meta["class"] if meta else None


def custom_source_hash() -> str:
    """Short digest of the registered source — a cache-buster for the surface."""
    meta = get_custom_meta()
    if not meta:
        return ""
    return hashlib.sha1(str(meta.get("source", "")).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Calibrator factory (consumed by services.calibration_service._calibrator_for)
# --------------------------------------------------------------------------- #


def _seed_away_from_default(default: float, lo: float, hi: float) -> float:
    """Deterministic optimiser seed offset 25 % toward the farther bound.

    The spec defaults double as the True-parameters sliders' defaults, so
    seeding the optimiser AT the defaults starts the fit exactly on the
    synthetic ground truth: the surface arrives already fitted and the loss /
    parameter trajectories are flat lines (same artifact as the Heston
    ATM-IV-seed-equals-default-truth case). Nudging each coordinate a quarter
    of the way toward its farther bound keeps the seed inside the (possibly
    tightened) box while guaranteeing a visible convergence path.
    """
    d = min(max(default, lo), hi)
    if hi <= lo:
        return d
    return d + 0.25 * (hi - d) if (hi - d) >= (d - lo) else d - 0.25 * (d - lo)


def build_custom_calibrator(
    *,
    optimizer,
    objective,
    max_nfev: int,
    iteration_callback=None,
    log_iterations: bool = False,
    param_bounds: dict[str, tuple[float, float]] | None = None,
) -> CustomModelCalibrator:
    """Build a :class:`CustomModelCalibrator` for the registered custom model.

    ``param_bounds`` is the per-run search universe ``{param: (lo, hi)}`` from the
    sidebar (a partial dict overrides only the named parameters); absent entries
    fall back to the model's PARAMETER_SPECS ``min_value`` / ``max_value``. The
    start point is the spec defaults nudged off-default
    (:func:`_seed_away_from_default`) so a fit against a surface generated at
    the default true parameters still shows real convergence.
    """
    meta = get_custom_meta()
    if meta is None:
        raise RuntimeError("No custom model registered.")

    specs = meta["specs"]
    names = tuple(s["name"] for s in specs)
    bounds: list[tuple[float, float]] = []
    for s in specs:
        lo, hi = float(s["min_value"]), float(s["max_value"])
        if param_bounds and s["name"] in param_bounds:
            blo, bhi = param_bounds[s["name"]]
            lo, hi = float(blo), float(bhi)
        bounds.append((lo, hi))
    x0 = [
        _seed_away_from_default(float(s["default"]), b[0], b[1])
        for s, b in zip(specs, bounds)
    ]

    return CustomModelCalibrator(
        meta["class"],
        param_names=names,
        bounds=bounds,
        objective=objective,
        optimizer=optimizer,
        n_paths=MC_PATHS_INTERACTIVE,
        mc_seed=MC_SEED,
        steps_per_year=DEFAULT_STEPS_PER_YEAR,
        x0=x0,
        max_nfev=int(max_nfev),
        log_iterations=log_iterations,
        iteration_callback=iteration_callback,
    )
