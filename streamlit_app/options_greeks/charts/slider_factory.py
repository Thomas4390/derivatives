"""
Slider factory for Options Greeks Explorer.

Centralizes shared slider creation logic for DTE/IV parameter sliders
used by both P&L and Greeks charts.
"""

from config.chart_theme import SLIDER_DEFAULTS


def find_default_param_value(slider_type: str, param_values: list) -> tuple[int, int]:
    """Find the best default value and its index for a parameter slider.

    For DTE: prefer 31 if available, else midpoint of the range.
    For IV: prefer 25 if available, else closest value to 25.

    Args:
        slider_type: "DTE" or "IV"
        param_values: List of parameter values for the slider

    Returns:
        Tuple of (default_value, active_idx)
    """
    if slider_type == "DTE":
        if 31 in param_values:
            default_value = 31
        else:
            default_value = param_values[len(param_values) // 2]
    else:
        if 25 in param_values:
            default_value = 25
        else:
            default_value = min(param_values, key=lambda v: abs(v - 25))

    if default_value in param_values:
        active_idx = param_values.index(default_value)
    else:
        active_idx = min(10, len(param_values) - 1)

    return default_value, active_idx


def create_param_slider(
    slider_type: str,
    param_values: list,
    default_value: int,
    traces_per_step: int,
    num_subplots: int = 1,
) -> dict:
    """Create a unified DTE/IV parameter slider for both P&L and Greeks charts.

    For P&L (num_subplots=1):
        Each step has len(param_values) traces + 1 always-visible expiry trace.
        Visibility array: [False]*len(param_values) + [True], then visible[idx] = True.

    For Greeks (num_subplots>1):
        Total traces = num_subplots * len(param_values) * traces_per_step.
        For each subplot, the traces for the active step are made visible.

    Args:
        slider_type: "DTE" or "IV"
        param_values: List of parameter values
        default_value: Default selected value (used to find active index)
        traces_per_step: Number of traces each slider step controls per subplot
            - PnL: 1 (just the P&L curve; expiry is always-visible separately)
            - Greeks: 1 + num_legs (aggregate + individual leg traces)
        num_subplots: 1 for P&L charts, N for Greeks subplot grids

    Returns:
        Plotly slider configuration dict
    """
    # Find active index
    if default_value is not None and default_value in param_values:
        active_idx = param_values.index(default_value)
    else:
        active_idx = min(10, len(param_values) - 1)

    prefix = "Days to Expiration: " if slider_type == "DTE" else "Implied Volatility: "

    if num_subplots == 1:
        steps = _build_pnl_steps(slider_type, param_values)
    else:
        steps = _build_greeks_steps(
            slider_type, param_values, traces_per_step, num_subplots
        )

    slider = SLIDER_DEFAULTS.copy()
    slider.update(
        {
            "active": active_idx,
            "currentvalue": {**SLIDER_DEFAULTS["currentvalue"], "prefix": prefix},
            "steps": steps,
        }
    )

    return slider


def _build_pnl_steps(slider_type: str, param_values: list) -> list[dict]:
    """Build slider steps for P&L charts (1 subplot, expiry always visible).

    Visibility layout: [param_trace_0, param_trace_1, ..., param_trace_N, expiry_trace]
    Each step shows one param trace + the always-visible expiry trace.
    """
    steps = []
    for idx, value in enumerate(param_values):
        visible = [False] * len(param_values) + [True]
        visible[idx] = True

        label = str(value) if slider_type == "DTE" else f"{value}%"
        step = dict(method="update", args=[{"visible": visible}], label=label)
        steps.append(step)

    return steps


def _build_greeks_steps(
    slider_type: str,
    param_values: list,
    traces_per_step: int,
    num_subplots: int,
) -> list[dict]:
    """Build slider steps for Greeks charts (multiple subplots, no always-visible traces).

    Trace layout per subplot: [step0_agg, step0_leg0, ..., step1_agg, step1_leg0, ...]
    Total traces = num_subplots * len(param_values) * traces_per_step.
    """
    total_traces = num_subplots * len(param_values) * traces_per_step
    steps = []

    for idx, value in enumerate(param_values):
        visible = [False] * total_traces

        for subplot_idx in range(num_subplots):
            base_idx = subplot_idx * len(param_values) * traces_per_step
            step_base = base_idx + idx * traces_per_step
            for trace_offset in range(traces_per_step):
                visible[step_base + trace_offset] = True

        label = str(value) if slider_type == "DTE" else f"{value}%"
        step = dict(method="update", args=[{"visible": visible}], label=label)
        steps.append(step)

    return steps
