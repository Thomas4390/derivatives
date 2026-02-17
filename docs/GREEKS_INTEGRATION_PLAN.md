# Greeks Integration Plan: All Models + Custom Models

**Status**: Implementation plan
**Author**: Thomas
**Date**: 2025

## Context

The backend supports full Greeks computation (14 Greeks, 1st/2nd/3rd order) for **all models** via the `GreeksCalculator` in `backend/greeks/calculator.py`. It automatically dispatches to analytic formulas for GBM or numerical finite differences for all other models. The `vol_bump.py` utility already handles vol-bumping for GBM, Heston, Merton, Bates, and GARCH.

However, the Streamlit simulation app currently only displays **4 Greeks** (delta, gamma, vega, theta) and **only for GBM** (Black-Scholes analytic). This is the sole significant integration gap between backend and frontend.

---

## Gap Summary

| What | Backend | Frontend | Gap |
|------|---------|----------|-----|
| Greeks for GBM | 14 Greeks (analytic) | 4 Greeks (delta, gamma, vega, theta) | Missing rho + 2nd/3rd order |
| Greeks for Heston/Merton/Bates | 14 Greeks (numerical via FFT engine) | None | Full gap |
| Greeks for GARCH family | 14 Greeks (numerical via MC engine) | None | Full gap |
| Greeks for custom models | Partial (delta, gamma, theta, rho work; vega needs vol-bump) | None | Full gap + vol-bump missing |
| Vol-bump for custom models | Not supported in `vol_bump.py` | N/A | Backend gap |

---

## Step 1: Generic Vol-Bump for Custom Models

### File: `backend/models/vol_bump.py`

### Current behavior

`create_vol_bumped_model()` handles GBM, Merton, Heston, Bates, and GARCH via `isinstance` checks. For any unknown model type (including custom models), it returns `None`, which causes all vol-related Greeks (vega, vanna, volga, veta, zomma, ultima) to be 0.

### Implementation

Add a generic fallback at the end of the function that:

1. Calls `model.get_parameters()` to inspect available parameters
2. Looks for a `sigma` parameter (flat vol models) or `v0` parameter (stochastic vol models)
3. Bumps the appropriate parameter and reconstructs the model via `type(model)(**new_params)`

```python
# At the end of create_vol_bumped_model(), before `return None`:

# Generic fallback for custom/unknown models
params = model.get_parameters()

if 'sigma' in params:
    new_params = dict(params)
    new_params['sigma'] = max(params['sigma'] + vol_bump, 1e-8)
    try:
        return type(model)(**new_params)
    except Exception:
        pass

if 'v0' in params:
    # Bump in vol space like Heston: v0_new = (sqrt(v0) + h)^2
    new_vol = max(np.sqrt(params['v0']) + vol_bump, 0.0)
    new_params = dict(params)
    new_params['v0'] = max(new_vol ** 2, 1e-8)
    try:
        return type(model)(**new_params)
    except Exception:
        pass

return None
```

### Why this works

- Custom models define `get_parameters()` (required by the Model interface)
- Custom models are instantiated via `cls(**params)` (validated during registration)
- The `type(model)(**new_params)` pattern recreates the same class with bumped params
- If the model has both `sigma` and `v0` (e.g., a Bates-like custom model), `sigma` is tried first (diffusive vol), which matches the convention for Merton

### Tests

```python
# Test with a custom GBM-like model
class CustomGBM(Model):
    def __init__(self, sigma=0.20):
        self._sigma = sigma
    def get_parameters(self):
        return {"sigma": self._sigma}
    # ... other methods

model = CustomGBM(sigma=0.20)
bumped = create_vol_bumped_model(model, 0.01)
assert bumped is not None
assert bumped.get_parameters()["sigma"] == pytest.approx(0.21)

# Test with a custom Heston-like model
class CustomSV(Model):
    def __init__(self, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7):
        ...
    def get_parameters(self):
        return {"v0": self._v0, "kappa": self._kappa, ...}

model = CustomSV(v0=0.04)
bumped = create_vol_bumped_model(model, 0.01)
# v0_new = (sqrt(0.04) + 0.01)^2 = (0.2 + 0.01)^2 = 0.0441
assert bumped.get_parameters()["v0"] == pytest.approx(0.0441)
```

---

## Step 2: Add Greeks Computation to Pricing Service

### File: `streamlit_app/simulation/services/pricing_service.py`

### Current behavior

The `PricingComparison` dataclass has only 4 individual Greek fields:
```python
analytical_delta: Optional[float] = None
analytical_gamma: Optional[float] = None
analytical_vega: Optional[float] = None
analytical_theta: Optional[float] = None
```

These are only populated for GBM (via `price_with_analytical()`). No Greeks are computed for any other model.

### Implementation

#### 2a. Add a unified `greeks` field to `PricingComparison`

```python
from typing import Dict

@dataclass
class PricingComparison:
    # ... existing MC/analytical/FFT fields ...

    # NEW: unified Greeks for all models
    greeks: Optional[Dict[str, float]] = None
    greeks_method: Optional[str] = None  # "analytic" or "numerical"

    # Keep existing analytical_delta/gamma/vega/theta for backwards compat
    # (they'll be populated from greeks dict when available)
```

The `greeks` dict will contain all 14 Greeks:
```python
{
    "price": 5.1234,
    "delta": 0.5500, "gamma": 0.0200, "vega": 0.2000,
    "theta": -0.0500, "rho": 0.1200,
    "vanna": 0.0100, "volga": 0.0050, "charm": -0.0020,
    "veta": -0.0010,
    "speed": 0.0001, "zomma": 0.0003, "color": -0.0001,
    "ultima": 0.0000,
}
```

#### 2b. Add `compute_greeks()` function

New function that wraps the backend `GreeksCalculator`:

```python
from backend.greeks.calculator import GreeksCalculator, AllGreeksResult

def compute_greeks(
    model_key: str,
    params: Dict[str, Any],
    strike: float,
    time_to_maturity: float,
    spot: float,
    risk_free_rate: float,
    is_call: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Compute Greeks for any model using the best available engine.

    Returns dict with all 14 Greeks + 'method' key, or None on failure.

    Engine selection logic:
    - GBM: BSAnalyticEngine (analytic Greeks)
    - Heston/Merton/Bates/custom with FFT: FFTEngine (numerical Greeks)
    - GARCH/custom without FFT: MonteCarloEngine (numerical Greeks)
    """
    try:
        model = _create_model(model_key, params)
        if model is None:
            return None

        option = _create_vanilla_option(strike, time_to_maturity, is_call)
        market = _create_market(spot, risk_free_rate)

        # Choose engine based on model capabilities
        from backend.core.result_types import PricingCapability

        if model_key.lower() == "gbm":
            engine = BSAnalyticEngine()
            method = "analytic"
        elif PricingCapability.FFT in model.supported_engines:
            engine = FFTEngine(config=FFTConfig(alpha=1.5, n_fft=4096, eta=0.25))
            method = "numerical (FFT)"
        else:
            engine = MonteCarloEngine(config=MCConfig(n_paths=50_000, n_steps=126))
            method = "numerical (MC)"

        calc = GreeksCalculator()
        result = calc.calculate(
            engine, option, model, market,
            include_higher_order=True
        )

        # Convert AllGreeksResult to dict
        greeks_dict = {
            "price": result.price,
            "delta": result.delta,
            "gamma": result.gamma,
            "vega": result.vega,
            "theta": result.theta,
            "rho": result.rho,
            "vanna": result.vanna,
            "volga": result.volga,
            "charm": result.charm,
            "veta": result.veta,
            "speed": result.speed,
            "zomma": result.zomma,
            "color": result.color,
            "ultima": result.ultima,
            "method": method,
        }
        return greeks_dict

    except Exception as e:
        import logging
        logging.warning(f"Greeks computation failed for {model_key}: {e}")
        return None
```

**Note on MonteCarloEngine**: The MC engine requires additional imports and config. The existing `MCConfig` from `backend/engines/mc_engine.py` should be used. MC Greeks are noisy — use 50k paths for a decent balance between speed and accuracy.

#### 2c. Integrate into `compare_pricing()`

Add Greeks computation at the end of `compare_pricing()`:

```python
def compare_pricing(...) -> PricingComparison:
    # ... existing MC + analytical + FFT pricing logic ...

    # NEW: compute Greeks for all models
    greeks = compute_greeks(
        model_key=model_key,
        params=params,
        strike=strike,
        time_to_maturity=time_to_maturity,
        spot=spot,
        risk_free_rate=risk_free_rate,
        is_call=is_call,
    )
    if greeks is not None:
        comparison.greeks = greeks
        comparison.greeks_method = greeks.pop("method")

        # Backfill analytical_ fields for backwards compat
        if comparison.analytical_delta is None:
            comparison.analytical_delta = greeks.get("delta")
            comparison.analytical_gamma = greeks.get("gamma")
            comparison.analytical_vega = greeks.get("vega")
            comparison.analytical_theta = greeks.get("theta")

    comparison.available_methods = available_methods
    return comparison
```

---

## Step 3: Display Greeks for All Models in UI

### File: `streamlit_app/simulation/components/pricing_comparison.py`

### Current behavior

`_render_greeks()` (lines 150-170):
- Only renders if `comparison.analytical_delta is not None` (i.e., only for GBM)
- Shows 4 Greeks in a single row
- Labeled "Greeks (Black-Scholes)"

### Implementation

Replace `_render_greeks()` with a new version that handles all models and shows all 3 orders:

```python
def _render_greeks(comparison: PricingComparison):
    """Render Greeks for any model, organized by order."""
    greeks = comparison.greeks
    if greeks is None:
        return

    method_label = comparison.greeks_method or "unknown"

    with st.expander(f"Greeks ({method_label})", expanded=False):

        # First order
        st.markdown("**First Order**")
        cols = st.columns(5)
        with cols[0]:
            st.metric("Delta", f"{greeks['delta']:.4f}")
        with cols[1]:
            st.metric("Gamma", f"{greeks['gamma']:.6f}")
        with cols[2]:
            st.metric("Vega", f"{greeks['vega']:.4f}")
        with cols[3]:
            st.metric("Theta", f"{greeks['theta']:.4f}")
        with cols[4]:
            st.metric("Rho", f"{greeks['rho']:.4f}")

        # Second order
        st.markdown("**Second Order**")
        cols = st.columns(4)
        with cols[0]:
            st.metric("Vanna", f"{greeks['vanna']:.6f}")
        with cols[1]:
            st.metric("Volga", f"{greeks['volga']:.6f}")
        with cols[2]:
            st.metric("Charm", f"{greeks['charm']:.6f}")
        with cols[3]:
            st.metric("Veta", f"{greeks['veta']:.6f}")

        # Third order
        st.markdown("**Third Order**")
        cols = st.columns(4)
        with cols[0]:
            st.metric("Speed", f"{greeks['speed']:.8f}")
        with cols[1]:
            st.metric("Zomma", f"{greeks['zomma']:.8f}")
        with cols[2]:
            st.metric("Color", f"{greeks['color']:.8f}")
        with cols[3]:
            st.metric("Ultima", f"{greeks['ultima']:.10f}")
```

### Update the condition in `render_pricing_comparison()`

Change line 45 from:
```python
if show_greeks and comparison.analytical_delta is not None:
```
to:
```python
if show_greeks and comparison.greeks is not None:
```

This ensures Greeks render for all models, not just GBM.

---

## File Change Summary

| File | Change | Lines |
|------|--------|-------|
| `backend/models/vol_bump.py` | Add generic fallback for custom models | ~15 lines added before `return None` |
| `streamlit_app/simulation/services/pricing_service.py` | Add `greeks` + `greeks_method` fields to `PricingComparison` | 2 fields |
| `streamlit_app/simulation/services/pricing_service.py` | Add `compute_greeks()` function | ~60 lines |
| `streamlit_app/simulation/services/pricing_service.py` | Call `compute_greeks()` in `compare_pricing()` | ~15 lines |
| `streamlit_app/simulation/components/pricing_comparison.py` | Rewrite `_render_greeks()` to show all 3 orders | ~40 lines |
| `streamlit_app/simulation/components/pricing_comparison.py` | Update condition in `render_pricing_comparison()` | 1 line |

**Total**: ~130 lines of new/modified code across 3 files.

---

## Engine Selection Logic for Greeks

| Model | Engine for Greeks | Method | Speed |
|-------|-------------------|--------|-------|
| GBM | `BSAnalyticEngine` | Analytic formulas | Instant |
| Heston | `FFTEngine` | Numerical FD on FFT prices | Fast (~1s) |
| Merton | `FFTEngine` | Numerical FD on FFT prices | Fast (~1s) |
| Bates | `FFTEngine` | Numerical FD on FFT prices | Fast (~1s) |
| GARCH / NGARCH / GJR-GARCH | `MonteCarloEngine` | Numerical FD on MC prices | Slow (~5-10s) |
| Custom + FFT | `FFTEngine` | Numerical FD on FFT prices | Fast (~1s) |
| Custom MC-only | `MonteCarloEngine` | Numerical FD on MC prices | Slow (~5-10s) |

### Performance note

For MC-based Greeks, each finite difference requires 2 repricing calls (central differences). With higher-order Greeks, the total can reach ~20-30 repricing calls. At 50k paths each, this takes 5-10 seconds. This is acceptable for interactive use, but a spinner/progress indicator should be shown.

For FFT-based Greeks, each repricing is ~10ms, so even 30 calls take <0.5s. This is effectively instant.

---

## What Already Works (No Changes Needed)

| Feature | Where | Status |
|---------|-------|--------|
| Stochastic vol in custom models | `GenericEulerSimulator` auto-detects `variance_drift()` / `variance_diffusion()` / `get_correlation()` | Working |
| Jumps in custom models | `GenericEulerSimulator` auto-detects `jump(s, dt)` method | Working |
| FFT pricing for custom models | `price_with_fft()` checks `PricingCapability.FFT` dynamically | Working |
| Correlated Brownian motions | Simulator generates correlated paths when `get_correlation()` exists | Working |
| Dynamic parameter UI | `PARAMETER_SPECS` drives slider generation | Working |
| LaTeX equation display | `EQUATION_LATEX` renders in model info panel | Working |
| 8 validation tests | `compile_and_validate()` in `custom_model_service.py` | Working |
| 7 code templates | Covers GBM, CEV, OU, Merton, CIR, Schwartz, Heston-like, GBM+FFT | Working |

---

## Verification Plan

### Manual Testing

1. **GBM**: Run simulation, go to Pricing Comparison tab. Verify 14 Greeks shown (should match existing 4 + new rho + 2nd/3rd order). Method should say "analytic".

2. **Heston**: Run simulation with Heston model. Verify Greeks appear with method "numerical (FFT)". Delta should be close to but not identical to GBM delta. Vega should reflect sensitivity to sqrt(v0).

3. **Merton**: Same check. Vega should reflect diffusive vol sensitivity.

4. **Bates**: Same check. Greeks should combine stochastic vol and jump sensitivities.

5. **GARCH**: Run simulation. Verify Greeks appear with method "numerical (MC)". Values will be noisy — verify they are reasonable (delta ~0.5 for ATM).

6. **Custom GBM template**: Register custom GBM, run simulation. Verify vega works (requires vol-bump generic fallback). Compare to built-in GBM Greeks.

7. **Custom Heston-like template**: Register Heston-like custom model. Verify full Greeks including vega (vol-bump via v0).

### Automated Tests

```python
# test_vol_bump_generic.py
def test_custom_model_sigma_bump():
    """Custom model with sigma param gets vol-bumped correctly."""
    ...

def test_custom_model_v0_bump():
    """Custom model with v0 param gets vol-bumped in vol space."""
    ...

def test_custom_model_no_vol_param():
    """Custom model without sigma or v0 returns None."""
    ...

# test_greeks_all_models.py
def test_greeks_gbm():
    """GBM Greeks match analytic formulas."""
    ...

def test_greeks_heston():
    """Heston Greeks are computed via FFT engine."""
    ...

def test_greeks_merton():
    """Merton Greeks are computed via FFT engine."""
    ...
```

---

## Dependencies

This plan requires no new dependencies. All needed modules already exist:
- `backend.greeks.calculator.GreeksCalculator`
- `backend.greeks.calculator.AllGreeksResult`
- `backend.models.vol_bump.create_vol_bumped_model`
- `backend.engines.fft_engine.FFTEngine`
- `backend.engines.mc_engine.MonteCarloEngine`
