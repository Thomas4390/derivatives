"""
Custom Model Service — Compile, validate, and register user-defined models.

Provides:
- Safe compilation of user Python code
- Validation suite (interface, SDE, simulation tests)
- Registration/unregistration in session state
"""

from dataclasses import dataclass, field
from typing import Optional, Type, List
import numpy as np
import math

from backend.core.interfaces import Model
from backend.core.result_types import PricingCapability


@dataclass
class TestResult:
    """Result of a single validation test."""
    name: str
    passed: bool
    message: str


@dataclass
class ValidationResult:
    """Result of the full validation suite."""
    tests: List[TestResult] = field(default_factory=list)
    model_class: Optional[Type[Model]] = None
    all_passed: bool = False

    def __post_init__(self):
        self.all_passed = all(t.passed for t in self.tests)


# Allowed builtins and modules for user code execution
_SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "frozenset",
    "getattr", "hasattr", "int", "isinstance", "issubclass", "len", "list",
    "map", "max", "min", "print", "property", "range", "repr", "reversed",
    "round", "set", "sorted", "str", "sum", "tuple", "type", "zip",
    "__build_class__", "__name__",
}


def compile_and_validate(source_code: str) -> ValidationResult:
    """
    Compile user code and run a validation suite.

    Steps:
    1. Compile — exec() in restricted namespace
    2. Discover — Find the Model subclass
    3. Check PARAMETER_SPECS
    4. Instantiate with defaults
    5. Interface compliance
    6. SDE test (if MC)
    7. CF test (if FFT)
    8. Quick simulation test
    """
    tests: List[TestResult] = []
    model_class = None

    # 1. Compilation — use restricted namespace with pre-injected safe modules
    import builtins as _builtins
    restricted_builtins = {k: getattr(_builtins, k) for k in dir(_builtins) if not k.startswith('_') or k in _SAFE_BUILTINS}
    # Allow __build_class__ and __name__ for class definitions
    restricted_builtins["__build_class__"] = _builtins.__build_class__
    restricted_builtins["__name__"] = "__custom_model__"

    # Custom __import__ that blocks dangerous modules
    _BLOCKED_MODULES = {"os", "sys", "subprocess", "shutil", "socket", "http",
                        "importlib", "ctypes", "signal", "multiprocessing", "threading"}
    _original_import = _builtins.__import__

    def _safe_import(name, *args, **kwargs):
        top_level = name.split(".")[0]
        if top_level in _BLOCKED_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in custom models")
        return _original_import(name, *args, **kwargs)

    restricted_builtins["__import__"] = _safe_import

    namespace = {
        "__builtins__": restricted_builtins,
        "np": np,
        "numpy": np,
        "math": math,
        "Model": Model,
        "PricingCapability": PricingCapability,
    }

    try:
        compiled = compile(source_code, "<custom_model>", "exec")
        exec(compiled, namespace)
        tests.append(TestResult("Compilation", True, "Code compiled successfully"))
    except SyntaxError as e:
        tests.append(TestResult("Compilation", False, f"Syntax error on line {e.lineno}: {e.msg}"))
        return ValidationResult(tests=tests)
    except Exception as e:
        tests.append(TestResult("Compilation", False, f"Compilation error: {e}"))
        return ValidationResult(tests=tests)

    # 2. Discover Model subclass
    model_classes = []
    for name, obj in namespace.items():
        if (isinstance(obj, type)
            and issubclass(obj, Model)
            and obj is not Model
            and not name.startswith("_")):
            model_classes.append(obj)

    if len(model_classes) == 0:
        tests.append(TestResult("Class Discovery", False, "No Model subclass found. Define a class inheriting from Model."))
        return ValidationResult(tests=tests)
    elif len(model_classes) > 1:
        names = ", ".join(c.__name__ for c in model_classes)
        tests.append(TestResult("Class Discovery", False, f"Multiple Model subclasses found ({names}). Define exactly one."))
        return ValidationResult(tests=tests)

    model_class = model_classes[0]
    tests.append(TestResult("Class Discovery", True, f"Found class: {model_class.__name__}"))

    # 3. Check PARAMETER_SPECS
    specs = getattr(model_class, "PARAMETER_SPECS", None)
    if specs is None:
        tests.append(TestResult("PARAMETER_SPECS", False, "Class must have a PARAMETER_SPECS class attribute (list of dicts)."))
        return ValidationResult(tests=tests)

    if not isinstance(specs, list) or len(specs) == 0:
        tests.append(TestResult("PARAMETER_SPECS", False, "PARAMETER_SPECS must be a non-empty list of dicts."))
        return ValidationResult(tests=tests)

    required_keys = {"name", "display_name", "default", "min_value", "max_value", "step", "description"}
    for i, spec in enumerate(specs):
        if not isinstance(spec, dict):
            tests.append(TestResult("PARAMETER_SPECS", False, f"Entry {i} is not a dict."))
            return ValidationResult(tests=tests)
        missing = required_keys - set(spec.keys())
        if missing:
            tests.append(TestResult("PARAMETER_SPECS", False, f"Entry '{spec.get('name', i)}' missing keys: {missing}"))
            return ValidationResult(tests=tests)

    tests.append(TestResult("PARAMETER_SPECS", True, f"{len(specs)} parameter(s) defined"))

    # 4. Instantiate with defaults
    defaults = {s["name"]: s["default"] for s in specs}
    try:
        instance = model_class(**defaults)
        tests.append(TestResult("Instantiation", True, "Created instance with default parameters"))
    except Exception as e:
        tests.append(TestResult("Instantiation", False, f"Failed to instantiate: {e}"))
        return ValidationResult(tests=tests)

    # 5. Interface compliance
    try:
        name = instance.name
        assert isinstance(name, str) and len(name) > 0, "name must be a non-empty string"

        engines = instance.supported_engines
        assert isinstance(engines, list) and len(engines) > 0, "supported_engines must be a non-empty list"
        for eng in engines:
            assert isinstance(eng, PricingCapability), f"supported_engines must contain PricingCapability values, got {type(eng)}"

        params = instance.get_parameters()
        assert isinstance(params, dict), "get_parameters() must return a dict"

        tests.append(TestResult("Interface Compliance", True, f"name='{name}', engines={[e.name for e in engines]}"))
    except (AssertionError, Exception) as e:
        tests.append(TestResult("Interface Compliance", False, str(e)))
        return ValidationResult(tests=tests)

    # 6. SDE test (if MONTE_CARLO supported)
    if PricingCapability.MONTE_CARLO in instance.supported_engines:
        try:
            drift_val = instance.drift(100.0, 0.04, 0.0, 0.05, 0.0)
            diff_val = instance.diffusion(100.0, 0.04, 0.0)

            assert np.isfinite(drift_val), f"drift returned non-finite: {drift_val}"
            assert np.isfinite(diff_val), f"diffusion returned non-finite: {diff_val}"

            tests.append(TestResult("SDE Test", True, f"drift={drift_val:.4f}, diffusion={diff_val:.4f}"))
        except Exception as e:
            tests.append(TestResult("SDE Test", False, f"SDE test failed: {e}"))
            return ValidationResult(tests=tests)

    # 6b. Jump test (if model has jump() method)
    has_jump = hasattr(instance, 'jump') and callable(getattr(instance, 'jump', None))
    if has_jump:
        try:
            s_arr = np.full(5, 100.0)
            jump_val = instance.jump(s_arr, 1.0 / 252)
            assert len(jump_val) == 5, f"jump() must return array of same length as s, got {len(jump_val)}"
            assert np.all(np.isfinite(jump_val)), "jump() returned non-finite values"
            tests.append(TestResult("Jump Test", True, "jump() method works correctly"))
        except Exception as e:
            tests.append(TestResult("Jump Test", False, f"Jump test failed: {e}"))
            return ValidationResult(tests=tests)

    # 6c. Stochastic volatility test (if model has variance_drift/variance_diffusion)
    has_stoch_vol = (
        hasattr(instance, 'variance_drift') and callable(getattr(instance, 'variance_drift', None))
        and hasattr(instance, 'variance_diffusion') and callable(getattr(instance, 'variance_diffusion', None))
    )
    if has_stoch_vol:
        try:
            v_arr = np.full(5, 0.04)
            s_arr = np.full(5, 100.0)
            vd = instance.variance_drift(v_arr, s_arr, 0.0)
            vdiff = instance.variance_diffusion(v_arr, s_arr, 0.0)
            assert len(vd) == 5, f"variance_drift must return array of length 5, got {len(vd)}"
            assert len(vdiff) == 5, f"variance_diffusion must return array of length 5, got {len(vdiff)}"
            assert np.all(np.isfinite(vd)), "variance_drift returned non-finite values"
            assert np.all(np.isfinite(vdiff)), "variance_diffusion returned non-finite values"

            rho = instance.get_correlation() if hasattr(instance, 'get_correlation') else 0.0
            assert -1.0 <= rho <= 1.0, f"get_correlation() must be in [-1, 1], got {rho}"

            tests.append(TestResult("Stoch Vol Test", True,
                                    f"variance SDE OK, rho={rho:.2f}"))
        except Exception as e:
            tests.append(TestResult("Stoch Vol Test", False, f"Stoch vol test failed: {e}"))
            return ValidationResult(tests=tests)

    # 7. CF test (if FFT supported)
    if PricingCapability.FFT in instance.supported_engines:
        try:
            cf_0 = instance.characteristic_function(0.0, 100.0, 1.0, 0.05, 0.0)
            assert abs(cf_0 - 1.0) < 0.01, f"CF(0) should be ~1.0, got {cf_0}"

            for u in [0.5, 1.0, 2.0, 5.0]:
                cf_u = instance.characteristic_function(u, 100.0, 1.0, 0.05, 0.0)
                assert abs(cf_u) <= 1.01, f"|CF({u})| = {abs(cf_u):.4f} > 1"

            tests.append(TestResult("CF Test", True, "Characteristic function passed all checks"))
        except Exception as e:
            tests.append(TestResult("CF Test", False, f"CF test failed: {e}"))
            return ValidationResult(tests=tests)

    # 8. Quick simulation test
    if PricingCapability.MONTE_CARLO in instance.supported_engines:
        try:
            from backend.simulation.models.generic_euler import GenericEulerSimulator

            sim = GenericEulerSimulator(instance)
            terminals = sim.simulate_terminal(s0=100.0, mu=0.05, t=1.0, n_paths=10, n_steps=10, seed=42)

            assert len(terminals) == 10, f"Expected 10 terminal values, got {len(terminals)}"
            assert np.all(np.isfinite(terminals)), "Some terminal values are non-finite"
            assert np.all(terminals > 0), "Some terminal values are non-positive"

            tests.append(TestResult("Simulation Test", True, f"10 paths OK, mean={np.mean(terminals):.2f}"))
        except Exception as e:
            tests.append(TestResult("Simulation Test", False, f"Simulation test failed: {e}"))

    result = ValidationResult(tests=tests, model_class=model_class)
    return result


def register_custom_model(model_class: Type[Model], source_code: str) -> None:
    """Register a validated custom model in session state."""
    import streamlit as st
    from streamlit_app.simulation.config.model_registry import (
        ModelSpec, ModelCategory, PricingMethod, ParameterSpec,
    )

    specs = model_class.PARAMETER_SPECS
    instance = model_class(**{s["name"]: s["default"] for s in specs})

    # Map PricingCapability to PricingMethod
    method_map = {
        PricingCapability.ANALYTICAL: PricingMethod.ANALYTICAL,
        PricingCapability.FFT: PricingMethod.FFT,
        PricingCapability.MONTE_CARLO: PricingMethod.MONTE_CARLO,
    }
    pricing_methods = [method_map[e] for e in instance.supported_engines if e in method_map]

    # Build ParameterSpec list
    param_specs = [
        ParameterSpec(
            name=s["name"],
            display_name=s["display_name"],
            default=s["default"],
            min_value=s["min_value"],
            max_value=s["max_value"],
            step=s["step"],
            description=s["description"],
            format=s.get("format", "%.4f"),
        )
        for s in specs
    ]

    # Extract LaTeX equations from optional EQUATION_LATEX class attribute
    eq_latex = getattr(model_class, "EQUATION_LATEX", {})
    eq_main = eq_latex.get("main", r"\text{User-defined SDE}")
    eq_vol = eq_latex.get("vol")
    eq_jump = eq_latex.get("jump")
    eq_cf = eq_latex.get("cf")
    eq_mc = eq_latex.get("mc")

    # Detect features from model methods
    _has_stoch_vol = (
        hasattr(instance, 'variance_drift') and callable(getattr(instance, 'variance_drift', None))
    )
    _has_jumps = (
        hasattr(instance, 'jump') and callable(getattr(instance, 'jump', None))
    )

    model_spec = ModelSpec(
        key="custom",
        name=instance.name,
        short_name=instance.name,
        category=ModelCategory.CONTINUOUS,
        has_stochastic_vol=_has_stoch_vol,
        has_jumps=_has_jumps,
        pricing_methods=pricing_methods,
        parameters=param_specs,
        equation_main=eq_main,
        equation_vol=eq_vol,
        equation_jump=eq_jump,
        equation_cf=eq_cf,
        equation_mc=eq_mc,
        description="Custom user-defined model",
    )

    st.session_state["custom_model"] = {
        "class": model_class,
        "source": source_code,
        "spec": model_spec,
    }


def unregister_custom_model() -> None:
    """Remove custom model from session state."""
    import streamlit as st
    if "custom_model" in st.session_state:
        del st.session_state["custom_model"]


def is_custom_model(key: str) -> bool:
    """Check if model key refers to a custom model."""
    return key == "custom"


def get_custom_model_class() -> Optional[Type[Model]]:
    """Get the compiled custom model class, or None."""
    import streamlit as st
    custom = st.session_state.get("custom_model")
    if custom:
        return custom["class"]
    return None
