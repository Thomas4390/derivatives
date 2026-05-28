"""
Structured Product Builder for Simulation Explorer sidebar.

Renders configuration UI for CPN, Reverse Convertible, and Autocallable products.
Integrated into the strategy builder dropdown (not a separate mode).

Author: Thomas Vaudescal
"""

from __future__ import annotations

import streamlit as st


# Frequency options for observation schedule
FREQUENCY_OPTIONS = {
    "monthly": "Monthly",
    "quarterly": "Quarterly",
    "semi_annual": "Semi-Annual",
    "annual": "Annual",
}

PRODUCT_TYPES = {
    "cpn": "Capital Protected Note (CPN)",
    "reverse_convertible": "Reverse Convertible",
    "autocallable": "Autocallable",
    "phoenix": "Phoenix Autocallable",
    "shark_note": "Shark Note",
    "twin_win": "Twin Win",
    "snowball": "Snowball Autocallable",
}

PRODUCT_HELP = {
    "cpn": "100% capital protection + capped upside participation",
    "reverse_convertible": "High fixed coupon with capital at risk via knock-in put",
    "autocallable": "Conditional coupons with early redemption trigger and capital protection barrier",
    "phoenix": "Monthly conditional coupons with memory + autocall + knock-in put",
    "shark_note": "Capital protected + capped upside with knock-out barrier and rebate",
    "twin_win": "Profit from both up and down moves, capital at risk if barrier breached",
    "snowball": "Autocall + growing snowball coupon + knock-in put",
}

PRODUCT_DESCRIPTIONS = {
    "cpn": "A <b>Capital Protected Note</b> guarantees return of the notional at maturity (bond floor) while offering participation in the upside. The upside may be capped. Decomposition: zero-coupon bond + call option spread.",
    "reverse_convertible": "A <b>Reverse Convertible</b> pays an above-market fixed coupon in exchange for the investor bearing downside risk via a knock-in put. If the underlying breaches the barrier, the investor receives depreciated shares at maturity.",
    "autocallable": "An <b>Autocallable</b> pays conditional coupons if the underlying stays above a coupon barrier. If above the autocall trigger at any observation date, it terminates early returning notional + coupon. Capital is at risk via a knock-in put.",
    "phoenix": "A <b>Phoenix Autocallable</b> pays frequent conditional coupons (monthly) with memory feature. If the underlying is above the autocall trigger, it terminates early. Capital is protected by a knock-in put barrier.",
    "shark_note": "A <b>Shark Note</b> offers capital protection at maturity plus leveraged participation in the upside. If the underlying touches the upper knock-out barrier, participation is replaced by a fixed rebate.",
    "twin_win": "A <b>Twin Win</b> certificate profits from both upside and downside moves (absolute performance). If the knock-in barrier is breached, the investor bears the full downside loss.",
    "snowball": "A <b>Snowball Autocallable</b> features a growing coupon that increases with time (snowball effect). Combined with autocall trigger and knock-in put for capital protection.",
}

# Product-specific section styling
_PRODUCT_SECTION_STYLES: dict[str, dict] = {
    "cpn": {
        "color": "#0d9488",
        "icon": "\U0001f6e1\ufe0f",
        "label": "Protection & Participation",
    },
    "reverse_convertible": {
        "color": "#f59e0b",
        "icon": "\U0001f4b0",
        "label": "Coupon & Barrier",
    },
    "autocallable": {
        "color": "#6366f1",
        "icon": "\u26a1",
        "label": "Triggers & Barriers",
    },
    "phoenix": {
        "color": "#8b5cf6",
        "icon": "\U0001f525",
        "label": "Phoenix Parameters",
    },
    "shark_note": {
        "color": "#06b6d4",
        "icon": "\U0001f988",
        "label": "Shark Fin Parameters",
    },
    "twin_win": {
        "color": "#ec4899",
        "icon": "\U0001f503",
        "label": "Twin Win Parameters",
    },
    "snowball": {
        "color": "#84cc16",
        "icon": "\u2744\ufe0f",
        "label": "Snowball Parameters",
    },
}


def _render_header_card(product_type: str) -> None:
    """Render teal gradient header card with product name, badge, and hover tooltip."""
    desc = PRODUCT_DESCRIPTIONS.get(product_type, "")
    tooltip_html = f'<div class="sp-header-tip">{desc}</div>' if desc else ""

    st.markdown(
        f"""
    <style>
    .sp-header {{position:relative;cursor:default;}}
    .sp-header-tip {{visibility:hidden;opacity:0;position:absolute;z-index:1000;top:100%;left:0;right:0;margin-top:0.25rem;padding:0.55rem 0.7rem;background:#1e293b;color:#e2e8f0;border-radius:8px;font-size:0.73rem;line-height:1.5;box-shadow:0 4px 16px rgba(0,0,0,0.4);border:1px solid #334155;transition:opacity 0.15s,visibility 0.15s;pointer-events:none;font-weight:400;}}
    .sp-header:hover .sp-header-tip {{visibility:visible;opacity:1;}}
    </style>
    <div class="sp-header" style="background: linear-gradient(135deg, #134e4a 0%, #115e59 100%);
         padding: 0.875rem 1rem; border-radius: 10px; margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-weight: 600; color: #ffffff; font-size: 0.95rem;">
                    {PRODUCT_TYPES[product_type]}
                </div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.75rem; margin-top: 0.2rem;">
                    {PRODUCT_HELP[product_type]}
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.15); padding: 0.35rem 0.65rem; border-radius: 6px;">
                <span style="color: #5eead4; font-size: 0.7rem; font-weight: 600; text-transform: uppercase;">
                    Structured
                </span>
            </div>
        </div>
        {tooltip_html}
    </div>
    """,
        unsafe_allow_html=True,
    )


def _render_section_header(icon: str, label: str, color: str) -> None:
    """Render a styled section header with colored underline."""
    st.markdown(
        f"""
    <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
         color: {color}; font-weight: 700; margin: 0.75rem 0 0.25rem 0;
         padding-bottom: 0.25rem; border-bottom: 2px solid {color};">
        {icon} {label}
    </div>
    """,
        unsafe_allow_html=True,
    )


def _render_sp_summary_card(product_type: str, params: dict) -> None:
    """Render a visual summary card recapping key parameters."""
    name = PRODUCT_TYPES[product_type]

    rows = [
        ("Notional", f"${params['notional']:,.0f}"),
        ("Maturity", f"{params['maturity']:.2g}Y"),
        (
            "Obs. Freq.",
            FREQUENCY_OPTIONS.get(
                params["observation_frequency"], params["observation_frequency"]
            ),
        ),
    ]

    if product_type == "cpn":
        rows.append(("Protection", f"{params.get('protection_level', 1.0):.2%}"))
        rows.append(("Particip.", f"{params.get('participation_rate', 0.8):.2%}"))
        cap = params.get("cap")
        rows.append(("Cap", f"{cap:.2%}" if cap is not None else "None"))
    elif product_type == "reverse_convertible":
        rows.append(("Coupon", f"{params.get('coupon_rate', 0.10):.2%}"))
        rows.append(("Barrier", f"{params.get('barrier', 0.60):.2%}"))
        rows.append(
            ("Monitoring", params.get("barrier_monitoring", "continuous").title())
        )
    elif product_type == "autocallable":
        rows.append(("Coupon", f"{params.get('coupon_rate', 0.07):.2%}"))
        rows.append(("Autocall", f"{params.get('autocall_trigger', 1.0):.2%}"))
        rows.append(("Cpn Barrier", f"{params.get('coupon_barrier', 0.70):.2%}"))
        rows.append(("KI Barrier", f"{params.get('ki_barrier', 0.60):.2%}"))
        rows.append(("Memory", "Yes" if params.get("memory_coupon", True) else "No"))
    elif product_type == "phoenix":
        rows.append(("Coupon", f"{params.get('coupon_rate', 0.08):.2%}"))
        rows.append(("Autocall", f"{params.get('autocall_trigger', 1.0):.2%}"))
        rows.append(("Cpn Barrier", f"{params.get('coupon_barrier', 0.65):.2%}"))
        rows.append(("KI Barrier", f"{params.get('ki_barrier', 0.55):.2%}"))
        rows.append(("Memory", "Yes" if params.get("memory_coupon", True) else "No"))
    elif product_type == "shark_note":
        rows.append(("Protection", f"{params.get('protection_level', 1.0):.2%}"))
        rows.append(("Particip.", f"{params.get('participation_rate', 1.5):.2%}"))
        rows.append(("KO Barrier", f"{params.get('ko_barrier', 1.40):.2%}"))
        rows.append(("Rebate", f"{params.get('rebate', 0.05):.2%}"))
    elif product_type == "twin_win":
        rows.append(("Particip.", f"{params.get('participation_rate', 1.0):.2%}"))
        rows.append(("KI Barrier", f"{params.get('ki_barrier', 0.60):.2%}"))
        rows.append(
            ("Monitoring", params.get("barrier_monitoring", "continuous").title())
        )
    elif product_type == "snowball":
        rows.append(("Coupon", f"{params.get('coupon_rate', 0.10):.2%}"))
        rows.append(("Autocall", f"{params.get('autocall_trigger', 1.0):.2%}"))
        rows.append(("KI Barrier", f"{params.get('ki_barrier', 0.60):.2%}"))

    rows_html = "".join(
        f'<div style="display: flex; justify-content: space-between; padding: 0.2rem 0;">'
        f'<span style="color: rgba(255,255,255,0.7); font-size: 0.75rem;">{label}</span>'
        f"<span style=\"color: #ffffff; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; font-weight: 600;\">{value}</span>"
        f"</div>"
        for label, value in rows
    )

    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, #134e4a 0%, #115e59 100%);
         border-radius: 10px; padding: 0.875rem 1rem; margin-top: 0.75rem;">
        <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
             color: #5eead4; font-weight: 600; margin-bottom: 0.5rem;">
            Summary
        </div>
        <div style="font-weight: 600; color: #ffffff; font-size: 0.85rem; margin-bottom: 0.5rem;">
            {name}
        </div>
        <div style="border-top: 1px solid rgba(255,255,255,0.15); padding-top: 0.4rem;">
            {rows_html}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_structured_product_builder(
    spot_price: float,
    risk_free_rate: float,
    time_to_expiry: float,
    volatility: float,
    product_type_key: str | None = None,
) -> dict | None:
    """
    Render sidebar UI for structured product configuration.

    Parameters
    ----------
    product_type_key : str, optional
        Product type key (e.g. "cpn", "reverse_convertible", "autocallable").
        If provided, skips the product type selector.

    Returns a dict with keys: product_type, product_params, notional, maturity.
    """
    product_type = product_type_key
    if product_type:
        _render_header_card(product_type)

    # ── Common parameters ──
    _render_section_header("\U0001f4cb", "Common Parameters", "#0d9488")

    c1, c2 = st.columns(2)
    with c1:
        notional = st.number_input(
            "Notional",
            min_value=100.0,
            max_value=1_000_000.0,
            value=1000.0,
            step=100.0,
            key="sp_notional",
        )
    with c2:
        maturity = st.slider(
            "Maturity (years)",
            min_value=0.5,
            max_value=5.0,
            value=min(time_to_expiry, 3.0),
            step=0.25,
            key="sp_maturity",
        )

    freq_key = st.selectbox(
        "Observation Frequency",
        options=list(FREQUENCY_OPTIONS.keys()),
        format_func=lambda x: FREQUENCY_OPTIONS[x],
        index=1,  # quarterly default
        key="sp_obs_freq",
    )

    # ── Product-specific parameters ──
    product_params = {
        "notional": notional,
        "maturity": maturity,
        "observation_frequency": freq_key,
    }

    if product_type and product_type in _PRODUCT_SECTION_STYLES:
        style = _PRODUCT_SECTION_STYLES[product_type]
        _render_section_header(style["icon"], style["label"], style["color"])

    if product_type == "cpn":
        _render_cpn_params(product_params)
    elif product_type == "reverse_convertible":
        _render_rc_params(product_params)
    elif product_type == "autocallable":
        _render_autocallable_params(product_params)
    elif product_type == "phoenix":
        _render_phoenix_params(product_params)
    elif product_type == "shark_note":
        _render_shark_note_params(product_params)
    elif product_type == "twin_win":
        _render_twin_win_params(product_params)
    elif product_type == "snowball":
        _render_snowball_params(product_params)

    # ── Summary card ──
    if product_type:
        _render_sp_summary_card(product_type, product_params)

    return {
        "product_type": product_type,
        "product_params": product_params,
        "notional": notional,
        "maturity": maturity,
    }


def _render_cpn_params(params: dict) -> None:
    """Render CPN-specific parameters."""
    c1, c2 = st.columns(2)
    with c1:
        params["protection_level"] = st.slider(
            "Protection Level",
            min_value=0.80,
            max_value=1.00,
            value=1.00,
            step=0.01,
            format="%.2f",
            key="sp_cpn_protection",
            help="Capital protection at maturity (1.0 = 100%)",
        )
    with c2:
        params["participation_rate"] = st.slider(
            "Participation Rate",
            min_value=0.50,
            max_value=1.50,
            value=0.80,
            step=0.01,
            format="%.2f",
            key="sp_cpn_participation",
            help="Participation in upside performance",
        )

    has_cap = st.checkbox("Cap upside", value=True, key="sp_cpn_has_cap")
    if has_cap:
        params["cap"] = st.slider(
            "Cap Level",
            min_value=1.10,
            max_value=2.00,
            value=1.50,
            step=0.01,
            format="%.2f",
            key="sp_cpn_cap",
            help="Maximum performance level (e.g., 1.50 = 150%)",
        )
    else:
        params["cap"] = None


def _render_rc_params(params: dict) -> None:
    """Render Reverse Convertible-specific parameters."""
    c1, c2 = st.columns(2)
    with c1:
        params["coupon_rate"] = st.slider(
            "Coupon Rate (annual)",
            min_value=0.01,
            max_value=0.20,
            value=0.10,
            step=0.005,
            format="%.3f",
            key="sp_rc_coupon",
            help="Fixed annual coupon rate",
        )
    with c2:
        params["barrier"] = st.slider(
            "Knock-In Barrier",
            min_value=0.40,
            max_value=0.90,
            value=0.60,
            step=0.01,
            format="%.2f",
            key="sp_rc_barrier",
            help="Barrier as fraction of initial spot (e.g., 0.60 = 60%)",
        )

    params["barrier_monitoring"] = st.selectbox(
        "Barrier Monitoring",
        options=["continuous", "discrete"],
        index=0,
        key="sp_rc_monitoring",
    )


def _render_autocallable_params(params: dict) -> None:
    """Render Autocallable-specific parameters."""
    c1, c2 = st.columns(2)
    with c1:
        params["coupon_rate"] = st.slider(
            "Coupon Rate (annual)",
            min_value=0.01,
            max_value=0.20,
            value=0.07,
            step=0.005,
            format="%.3f",
            key="sp_auto_coupon",
            help="Conditional annual coupon rate",
        )
    with c2:
        params["autocall_trigger"] = st.slider(
            "Autocall Trigger",
            min_value=0.80,
            max_value=1.20,
            value=1.00,
            step=0.01,
            format="%.2f",
            key="sp_auto_trigger",
            help="Performance level that triggers early redemption",
        )

    c3, c4 = st.columns(2)
    with c3:
        params["coupon_barrier"] = st.slider(
            "Coupon Barrier",
            min_value=0.50,
            max_value=1.00,
            value=0.70,
            step=0.01,
            format="%.2f",
            key="sp_auto_coupon_barrier",
            help="Minimum performance for coupon payment",
        )
    with c4:
        params["ki_barrier"] = st.slider(
            "Knock-In Barrier",
            min_value=0.40,
            max_value=0.90,
            value=0.60,
            step=0.01,
            format="%.2f",
            key="sp_auto_ki_barrier",
            help="Capital protection barrier (below = capital at risk)",
        )

    c5, c6 = st.columns(2)
    with c5:
        params["memory_coupon"] = st.checkbox(
            "Memory Coupon",
            value=True,
            key="sp_auto_memory",
            help="Unpaid coupons accumulate and are paid when barrier is above coupon level",
        )
    with c6:
        params["barrier_monitoring"] = st.selectbox(
            "Barrier Monitoring",
            options=["continuous", "discrete"],
            index=0,
            key="sp_auto_monitoring",
        )


def _render_phoenix_params(params: dict) -> None:
    """Render Phoenix Autocallable-specific parameters."""
    c1, c2 = st.columns(2)
    with c1:
        params["coupon_rate"] = st.slider(
            "Coupon Rate (annual)",
            min_value=0.01,
            max_value=0.20,
            value=0.08,
            step=0.005,
            format="%.3f",
            key="sp_phoenix_coupon",
            help="Conditional annual coupon rate",
        )
    with c2:
        params["autocall_trigger"] = st.slider(
            "Autocall Trigger",
            min_value=0.80,
            max_value=1.20,
            value=1.00,
            step=0.01,
            format="%.2f",
            key="sp_phoenix_trigger",
            help="Performance level that triggers early redemption",
        )

    c3, c4 = st.columns(2)
    with c3:
        params["coupon_barrier"] = st.slider(
            "Coupon Barrier",
            min_value=0.50,
            max_value=1.00,
            value=0.65,
            step=0.01,
            format="%.2f",
            key="sp_phoenix_coupon_barrier",
            help="Minimum performance for coupon payment",
        )
    with c4:
        params["ki_barrier"] = st.slider(
            "Knock-In Barrier",
            min_value=0.30,
            max_value=0.80,
            value=0.55,
            step=0.01,
            format="%.2f",
            key="sp_phoenix_ki_barrier",
            help="Capital protection barrier",
        )

    c5, c6 = st.columns(2)
    with c5:
        params["memory_coupon"] = st.checkbox(
            "Memory Coupon",
            value=True,
            key="sp_phoenix_memory",
            help="Unpaid coupons accumulate and are paid when barrier is met",
        )
    with c6:
        params["barrier_monitoring"] = st.selectbox(
            "Barrier Monitoring",
            options=["continuous", "discrete"],
            index=0,
            key="sp_phoenix_monitoring",
        )


def _render_shark_note_params(params: dict) -> None:
    """Render Shark Note-specific parameters."""
    c1, c2 = st.columns(2)
    with c1:
        params["protection_level"] = st.slider(
            "Protection Level",
            min_value=0.80,
            max_value=1.00,
            value=1.00,
            step=0.01,
            format="%.2f",
            key="sp_shark_protection",
            help="Capital protection at maturity (1.0 = 100%)",
        )
    with c2:
        params["participation_rate"] = st.slider(
            "Participation Rate",
            min_value=0.50,
            max_value=3.00,
            value=1.50,
            step=0.10,
            format="%.2f",
            key="sp_shark_participation",
            help="Leveraged participation in upside",
        )

    c3, c4 = st.columns(2)
    with c3:
        params["ko_barrier"] = st.slider(
            "Knock-Out Barrier",
            min_value=1.10,
            max_value=2.00,
            value=1.40,
            step=0.05,
            format="%.2f",
            key="sp_shark_ko_barrier",
            help="Upper barrier (e.g., 1.40 = 140% of initial)",
        )
    with c4:
        params["rebate"] = st.slider(
            "Rebate",
            min_value=0.00,
            max_value=0.15,
            value=0.05,
            step=0.01,
            format="%.2f",
            key="sp_shark_rebate",
            help="Fixed rebate if knocked out (fraction of notional)",
        )


def _render_twin_win_params(params: dict) -> None:
    """Render Twin Win-specific parameters."""
    c1, c2 = st.columns(2)
    with c1:
        params["participation_rate"] = st.slider(
            "Participation Rate",
            min_value=0.50,
            max_value=2.00,
            value=1.00,
            step=0.10,
            format="%.2f",
            key="sp_twin_participation",
            help="Participation in absolute performance",
        )
    with c2:
        params["ki_barrier"] = st.slider(
            "Knock-In Barrier",
            min_value=0.40,
            max_value=0.90,
            value=0.60,
            step=0.01,
            format="%.2f",
            key="sp_twin_ki_barrier",
            help="If breached, twin-win becomes capital loss",
        )

    params["barrier_monitoring"] = st.selectbox(
        "Barrier Monitoring",
        options=["continuous", "discrete"],
        index=0,
        key="sp_twin_monitoring",
    )


def _render_snowball_params(params: dict) -> None:
    """Render Snowball Autocallable-specific parameters."""
    c1, c2 = st.columns(2)
    with c1:
        params["coupon_rate"] = st.slider(
            "Snowball Coupon Rate",
            min_value=0.01,
            max_value=0.25,
            value=0.10,
            step=0.005,
            format="%.3f",
            key="sp_snowball_coupon",
            help="Annualized coupon rate (grows with time)",
        )
    with c2:
        params["autocall_trigger"] = st.slider(
            "Autocall Trigger",
            min_value=0.80,
            max_value=1.20,
            value=1.00,
            step=0.01,
            format="%.2f",
            key="sp_snowball_trigger",
            help="Performance level for early redemption",
        )

    c3, c4 = st.columns(2)
    with c3:
        params["ki_barrier"] = st.slider(
            "Knock-In Barrier",
            min_value=0.40,
            max_value=0.90,
            value=0.60,
            step=0.01,
            format="%.2f",
            key="sp_snowball_ki_barrier",
            help="Capital protection barrier",
        )
    with c4:
        params["barrier_monitoring"] = st.selectbox(
            "Barrier Monitoring",
            options=["continuous", "discrete"],
            index=0,
            key="sp_snowball_monitoring",
        )
