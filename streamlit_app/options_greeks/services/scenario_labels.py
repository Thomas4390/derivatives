"""Pedagogical, leg-contextual labels for discrete-event exotic scenarios.

Pure helper (no Streamlit / Plotly): turns a leg ``position`` dict into
plain-language scenario labels and the barrier level(s) to draw. The scenario
KEYS are unchanged from ``config.exotic_config.PAYOFF_SCENARIOS`` so the P&L
recompute path is untouched; only the human-facing text is contextualised by
the leg's direction / knock type. Resolution of direction/knock/levels mirrors
``exotic_pricing_adapter.conditional_exotic_payoff_vec`` exactly (basic ``barrier``
uses ``barrier``/``is_up``/``is_knock_in``; ``discrete_barrier`` uses the ``adv_*``
keys; ``double_barrier`` uses ``dbl_lower``/``dbl_upper``/``adv_in``;
``binary_barrier`` decodes its Reiner-Rubinstein type number).
"""

from __future__ import annotations


def _barrier_dir_knock(position: dict) -> tuple[bool, bool]:
    """Return (is_up, is_knock_in) resolved by instrument class."""
    inst = position.get("instrument_class")
    if inst == "discrete_barrier":
        return bool(position.get("adv_is_up", True)), bool(
            position.get("adv_in", False)
        )
    if inst == "double_barrier":
        # corridor has no single direction; knock resolved by adv_in
        return True, bool(position.get("adv_in", False))
    if inst == "binary_barrier":
        # Reiner-Rubinstein type numbers alternate down/up (odd = down);
        # the knock-in types are 1-8 and 13-20 (mirrors _BINARY_LABELS in
        # exotic_pricing_adapter, kept numeric here to stay dependency-free).
        binary_type = int(position.get("binary_type", 13))
        return binary_type % 2 == 0, binary_type <= 8 or 13 <= binary_type <= 20
    # default (is_up=True, is_knock_in=False) applies to basic barrier and unknown classes
    return bool(position.get("is_up", True)), bool(position.get("is_knock_in", False))


def scenario_options(position: dict) -> list[tuple[str, str]]:
    """[(scenario_key, plain_label)] for a discrete-event leg (keys unchanged)."""
    inst = position.get("instrument_class")
    if inst == "chooser":
        return [("call", "Chose the call"), ("put", "Chose the put")]

    is_up, is_ki = _barrier_dir_knock(position)

    if inst == "double_barrier":
        if is_ki:
            return [
                ("not_touched", "Never activated — stayed in the corridor"),
                ("touched", "Activated — touched a barrier"),
            ]
        return [
            ("not_touched", "Stayed in the corridor [L, U]"),
            ("touched", "Knocked out — touched a barrier"),
        ]

    word = "rose to" if is_up else "fell to"
    if is_ki:
        return [
            ("not_touched", f"Never activated — never {word} the barrier"),
            ("touched", f"Activated — {word} the barrier"),
        ]
    return [
        ("not_touched", f"Survived — never {word} the barrier"),
        ("touched", f"Knocked out — {word} the barrier"),
    ]


def barrier_levels(position: dict) -> list[float]:
    """Barrier level(s) to draw on the chart (corridor = two; chooser = none)."""
    inst = position.get("instrument_class")
    if inst == "chooser":
        return []
    if inst == "double_barrier":
        return [
            float(position.get("dbl_lower", 0.8 * float(position["strike"]))),
            float(position.get("dbl_upper", 1.2 * float(position["strike"]))),
        ]
    if inst in ("discrete_barrier", "binary_barrier", "partial_barrier"):
        lvl = (
            position.get("adv_barrier")
            if position.get("adv_barrier") is not None
            else position.get("barrier")
        )
    else:  # basic barrier and any unknown class
        lvl = position.get("barrier")
    return [float(lvl)] if lvl else []
