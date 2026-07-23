"""
Greeks Charts — Greeks vs Spot with DTE slider animation + 3D surface.

Patterns adapted from options_greeks/charts/greeks_chart.py and surface_3d.py.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from config.chart_theme import (
    AXIS_LABEL as _AXIS_LABEL,
    AXIS_STYLE as _AXIS_STYLE,
    CHART_HEIGHT_LG,
    CHART_HEIGHT_SM,
    PAPER_BG as _PAPER_BG,
    PLOT_BG as _PLOT_BG,
    TICK_COLOR as _TICK_COLOR,
)
from services.greeks_service import DISPLAY_GREEKS, GREEK_TITLES

# Greek-specific colors (bright for dark background)
_COLORS = {
    "price": "#fbbf24",  # amber
    "delta": "#60a5fa",  # blue
    "gamma": "#34d399",  # green
    "vega": "#a78bfa",  # purple
    "theta": "#f87171",  # red
    "rho": "#38bdf8",  # sky
}

_3D_COLORSCALES = [
    "Viridis",
    "Plasma",
    "Inferno",
    "Magma",
    "Cividis",
    "Turbo",
    "RdBu",
]


# ═════════════════════════════════════════════════════════════════════════
# 2D Greeks with DTE slider (for option strategies)
# ═════════════════════════════════════════════════════════════════════════


def render_greeks_with_dte_slider(
    surface_data: dict,
    spot: float,
    selected_greek: str,
    model_surface: dict | None = None,
    primary_label: str = "Black-Scholes",
    model_label: str = "Model-consistent",
) -> None:
    """
    Render a Greek vs Spot chart with a Plotly DTE slider.

    When ``model_surface`` is supplied (and shares the DTE grid), the primary
    (practitioner-BS) and model-consistent series are overlaid — same colour per
    Greek, primary solid and model dotted, with a legend — and the DTE slider
    toggles the visible pair. Otherwise a single series is shown.
    """
    spot_range = surface_data["spot_range"]
    dte_values = surface_data["dte_values"]
    n_dte = len(dte_values)

    color = _COLORS.get(selected_greek, "#ffffff")
    greek_title = GREEK_TITLES.get(selected_greek, selected_greek.capitalize())

    overlay = (
        model_surface is not None
        and len(model_surface.get("dte_values", [])) == n_dte
    )

    def _curve(sd: dict, dte) -> np.ndarray:
        if selected_greek == "price":
            return sd.get("price_by_dte", {}).get(
                dte, np.zeros_like(sd["spot_range"])
            )
        return sd["greeks_by_dte"][dte].get(
            selected_greek, np.zeros_like(sd["spot_range"])
        )

    fig = go.Figure()

    # Primary (practitioner Black-Scholes) traces — one per DTE value.
    for idx, dte in enumerate(dte_values):
        fig.add_trace(
            go.Scatter(
                x=spot_range,
                y=_curve(surface_data, dte),
                mode="lines",
                name=primary_label if overlay else f"DTE {dte}",
                legendgroup="primary",
                line=dict(color=color, width=2.5),
                visible=(idx == n_dte - 1),  # show max DTE by default
                showlegend=overlay,
                hovertemplate=(
                    f"<b>{primary_label if overlay else greek_title}</b><br>"
                    "Spot: $%{x:,.2f}<br>"
                    "Value: %{y:.4f}<br>"
                    f"DTE: {dte}<extra></extra>"
                ),
            )
        )

    # Model-consistent overlay traces — dotted, same colour, aligned by DTE index.
    if overlay:
        model_spot = model_surface["spot_range"]
        for idx, dte in enumerate(model_surface["dte_values"]):
            fig.add_trace(
                go.Scatter(
                    x=model_spot,
                    y=_curve(model_surface, dte),
                    mode="lines",
                    name=model_label,
                    legendgroup="model",
                    line=dict(color=color, width=2.0, dash="dot"),
                    visible=(idx == n_dte - 1),
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{model_label}</b><br>"
                        "Spot: $%{x:,.2f}<br>"
                        "Value: %{y:.4f}<br>"
                        f"DTE: {dte}<extra></extra>"
                    ),
                )
            )

    # Spot reference line
    fig.add_vline(
        x=spot,
        line_dash="dash",
        line_color="#fbbf24",
        line_width=1,
        opacity=0.6,
    )
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="rgba(255,255,255,0.25)",
        line_width=1,
    )

    # Build slider steps (toggle the primary [+ model] trace for each DTE).
    total_traces = 2 * n_dte if overlay else n_dte
    steps = []
    for idx, dte in enumerate(dte_values):
        visible = [False] * total_traces
        visible[idx] = True
        if overlay:
            visible[n_dte + idx] = True
        steps.append(
            dict(
                method="update",
                args=[{"visible": visible}],
                label=str(dte),
            )
        )

    slider = dict(
        active=n_dte - 1,
        currentvalue=dict(
            prefix="Days to Expiration: ",
            font=dict(size=13, color=_AXIS_LABEL),
        ),
        pad=dict(t=30),
        steps=steps,
        bgcolor="#1e293b",
        activebgcolor="#0d9488",
        bordercolor="rgba(255,255,255,0.15)",
        font=dict(color=_TICK_COLOR, size=10),
    )

    fig.update_layout(
        height=CHART_HEIGHT_SM,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(color=_AXIS_LABEL),
        margin=dict(l=60, r=30, t=40, b=20),
        xaxis=dict(title="Underlying Price ($)", **_AXIS_STYLE),
        yaxis=dict(title=greek_title, **_AXIS_STYLE),
        sliders=[slider],
        showlegend=overlay,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=_AXIS_LABEL),
        ),
        xaxis_showspikes=True,
        xaxis_spikemode="across",
        xaxis_spikesnap="cursor",
        xaxis_spikethickness=1,
        xaxis_spikecolor="#a5b4fc",
        xaxis_spikedash="dot",
    )

    st.plotly_chart(fig, width="stretch", theme=None)


# ═════════════════════════════════════════════════════════════════════════
# 3D Greeks Surface
# ═════════════════════════════════════════════════════════════════════════


def render_greeks_3d_surface(
    surface_data: dict,
    spot: float,
    selected_greek: str,
    colorscale: str = "Viridis",
) -> None:
    """
    Render a 3D surface: Greek value = f(spot, DTE).
    """
    spot_range = surface_data["spot_range"]
    dte_values = surface_data["dte_values"]
    greeks_by_dte = surface_data["greeks_by_dte"]
    price_by_dte = surface_data.get("price_by_dte", {})

    greek_title = GREEK_TITLES.get(selected_greek, selected_greek.capitalize())

    # Build Z matrix: rows = DTE, cols = spot
    Z = np.zeros((len(dte_values), len(spot_range)))
    for i, dte in enumerate(dte_values):
        if selected_greek == "price":
            Z[i, :] = price_by_dte.get(dte, np.zeros_like(spot_range))
        else:
            Z[i, :] = greeks_by_dte[dte][selected_greek]

    X, Y = np.meshgrid(spot_range, dte_values)

    # ── Dark 3D scene axis style ──
    _3d_axis = dict(
        showbackground=True,
        backgroundcolor="#1e293b",
        gridcolor="rgba(255,255,255,0.12)",
        zerolinecolor="rgba(255,255,255,0.15)",
        title_font=dict(size=12, color="#e2e8f0"),
        tickfont=dict(size=10, color="rgba(255,255,255,0.70)"),
    )

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text=greek_title,
                        font=dict(size=12, color="#e2e8f0"),
                    ),
                    thickness=18,
                    len=0.7,
                    x=1.02,
                    tickfont=dict(size=10, color="rgba(255,255,255,0.70)"),
                    bgcolor="rgba(30,41,59,0.85)",
                    bordercolor="rgba(255,255,255,0.15)",
                    borderwidth=1,
                ),
                contours=dict(z=dict(show=False)),
                lighting=dict(
                    ambient=0.55,
                    diffuse=0.7,
                    specular=0.35,
                    roughness=0.5,
                    fresnel=0.2,
                ),
                lightposition=dict(x=1000, y=1000, z=500),
                hovertemplate=(
                    "<b>Spot:</b> $%{x:,.2f}<br>"
                    "<b>DTE:</b> %{y:.0f}<br>"
                    f"<b>{greek_title}:</b> %{{z:.4f}}<br>"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        height=CHART_HEIGHT_LG + 30,
        paper_bgcolor=_PAPER_BG,
        scene=dict(
            xaxis=dict(title="Underlying Price ($)", **_3d_axis),
            yaxis=dict(title="DTE (days)", **_3d_axis),
            zaxis=dict(title=greek_title, **_3d_axis),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        margin=dict(l=0, r=30, t=30, b=0),
    )

    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": True})


# ═════════════════════════════════════════════════════════════════════════
# SP mode: simple 2D chart with Greek selector (no DTE slider)
# ═════════════════════════════════════════════════════════════════════════


def render_sp_greeks_chart(
    greeks_data: dict,
    spot: float,
    selected_greek: str,
) -> None:
    """Render a single Greek vs Spot chart for structured products (MC data)."""
    spot_range = greeks_data["spot_range"]
    values = greeks_data["greeks"][selected_greek]
    color = _COLORS.get(selected_greek, "#ffffff")
    greek_title = GREEK_TITLES.get(selected_greek, selected_greek.capitalize())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=spot_range,
            y=values,
            mode="lines",
            line=dict(color=color, width=2.5),
            name=greek_title,
            hovertemplate=(
                f"<b>{greek_title}</b><br>"
                "Spot: $%{x:,.2f}<br>"
                "Value: %{y:.4f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(
        x=spot,
        line_dash="dash",
        line_color="#fbbf24",
        line_width=1,
        opacity=0.6,
    )
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="rgba(255,255,255,0.25)",
        line_width=1,
    )

    fig.update_layout(
        height=CHART_HEIGHT_SM,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(color=_AXIS_LABEL),
        margin=dict(l=60, r=30, t=40, b=40),
        xaxis=dict(title="Underlying Price ($)", **_AXIS_STYLE),
        yaxis=dict(title=greek_title, **_AXIS_STYLE),
        showlegend=False,
    )

    st.plotly_chart(fig, width="stretch", theme=None)


# ═════════════════════════════════════════════════════════════════════════
# Greek selector widget
# ═════════════════════════════════════════════════════════════════════════


def render_greek_selector(key: str = "greeks_tab") -> str:
    """Render a selectbox to choose which Greek to display. Returns selected name."""
    options = list(DISPLAY_GREEKS)
    return st.selectbox(  # pyright: ignore[reportCallIssue]
        "Greek",
        options=options,
        format_func=lambda x: GREEK_TITLES.get(x, x.capitalize()),
        index=0,
        key=f"greek_selector_{key}",
    )


def render_colorscale_selector(key: str = "greeks_3d") -> str:
    """Render a selectbox for 3D surface colorscale."""
    return st.selectbox(
        "Color Scheme",
        _3D_COLORSCALES,
        index=0,
        key=f"colorscale_{key}",
    )
