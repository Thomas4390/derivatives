"""
Chart theme configuration for Options Greeks Explorer.

Consistent visual styling for all Plotly charts.
"""

# =============================================================================
# COLOR PALETTE FOR CHARTS
# =============================================================================

CHART_COLORS = {
    # Primary chart colors
    "primary": "#1a365d",
    "secondary": "#2c5282",
    "accent": "#0d9488",
    # P&L colors
    "profit": "#059669",
    "loss": "#dc2626",
    "neutral": "#64748b",
    # Line colors for different series
    "current": "#1a365d",
    "expiry": "#dc2626",
    "breakeven": "#d97706",
    "reference": "#94a3b8",
    # Grid and background
    "grid": "#e2e8f0",
    "background": "#ffffff",
    "paper": "#ffffff",
}

# Greek-specific colors (professional palette)
GREEK_COLORS = {
    "price": "#1a365d",
    "delta": "#1a365d",
    "gamma": "#0d9488",
    "vega": "#7c3aed",
    "theta": "#dc2626",
    "rho": "#0284c7",
    "vanna": "#db2777",
    "volga": "#0891b2",
    "charm": "#ca8a04",
    "veta": "#ea580c",
    "speed": "#4f46e5",
    "zomma": "#059669",
    "color": "#64748b",
    "ultima": "#9333ea",
}

# =============================================================================
# LAYOUT CONFIGURATION
# =============================================================================

LAYOUT_DEFAULTS = {
    "font": {
        "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        "size": 12,
        "color": "#334155",
    },
    "paper_bgcolor": CHART_COLORS["paper"],
    "plot_bgcolor": CHART_COLORS["background"],
    "margin": {"l": 60, "r": 40, "t": 60, "b": 60},
    "hovermode": "x",
    "hoverlabel": {
        "bgcolor": "rgba(255, 255, 255, 0.95)",
        "font_size": 12,
        "font_family": "Inter, sans-serif",
        "bordercolor": "#e2e8f0",
    },
    "spikedistance": -1,
}

AXIS_DEFAULTS = {
    "gridcolor": CHART_COLORS["grid"],
    "gridwidth": 1,
    "linecolor": "#cbd5e1",
    "linewidth": 1,
    "tickfont": {"family": "JetBrains Mono, monospace", "size": 11, "color": "#64748b"},
    "title": {"font": {"family": "Inter, sans-serif", "size": 12, "color": "#475569"}},
    "zeroline": True,
    "zerolinecolor": "#94a3b8",
    "zerolinewidth": 1,
}

# =============================================================================
# SLIDER CONFIGURATION
# =============================================================================

SLIDER_DEFAULTS = {
    "active": 10,
    "y": -0.12,
    "len": 0.9,
    "x": 0.05,
    "pad": {"b": 10, "t": 50},
    "currentvalue": {
        "prefix": "",
        "font": {"family": "Inter, sans-serif", "size": 13, "color": "#1e293b"},
        "xanchor": "center",
        "offset": 20,
        "visible": True,
    },
    "transition": {"duration": 100},
    "bordercolor": "#94a3b8",
    "bgcolor": "#e2e8f0",
    "activebgcolor": "#1a365d",
    "ticklen": 4,
    "tickcolor": "#64748b",
    "font": {"family": "JetBrains Mono, monospace", "size": 10, "color": "#475569"},
}

# =============================================================================
# 3D SURFACE CONFIGURATION
# =============================================================================

SURFACE_COLORSCALES = {
    "default": "Viridis",
    "alternatives": [
        "Plasma",
        "Inferno",
        "Magma",
        "Cividis",
        "Turbo",
        "RdBu",
        "Spectral",
    ],
}

SCENE_DEFAULTS = {
    "xaxis": {
        "title": {
            "text": "Underlying Price",
            "font": {"family": "Inter, sans-serif", "size": 12, "color": "#475569"},
        },
        "gridcolor": "#e2e8f0",
        "showbackground": True,
        "backgroundcolor": "#f8fafc",
        "tickfont": {
            "family": "JetBrains Mono, monospace",
            "size": 10,
            "color": "#64748b",
        },
    },
    "yaxis": {
        "title": {
            "text": "",
            "font": {"family": "Inter, sans-serif", "size": 12, "color": "#475569"},
        },
        "gridcolor": "#e2e8f0",
        "showbackground": True,
        "backgroundcolor": "#f8fafc",
        "tickfont": {
            "family": "JetBrains Mono, monospace",
            "size": 10,
            "color": "#64748b",
        },
    },
    "zaxis": {
        "title": {
            "text": "",
            "font": {"family": "Inter, sans-serif", "size": 12, "color": "#475569"},
        },
        "gridcolor": "#e2e8f0",
        "showbackground": True,
        "backgroundcolor": "#f8fafc",
        "tickfont": {
            "family": "JetBrains Mono, monospace",
            "size": 10,
            "color": "#64748b",
        },
    },
    "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.2}},
    "aspectratio": {"x": 1, "y": 1, "z": 0.7},
}

# =============================================================================
# LINE STYLES
# =============================================================================

LINE_STYLES = {
    "primary": {"width": 2.5, "color": CHART_COLORS["primary"]},
    "expiry": {"width": 2.5, "color": CHART_COLORS["expiry"], "dash": "dash"},
    "breakeven": {"width": 1.5, "color": CHART_COLORS["breakeven"], "dash": "dash"},
    "reference": {"width": 1, "color": CHART_COLORS["reference"], "dash": "dot"},
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_layout_config(title: str = None, height: int = 600) -> dict:
    """Get a complete layout configuration dict."""
    config = LAYOUT_DEFAULTS.copy()
    config["height"] = height

    if title:
        config["title"] = {
            "text": title,
            "font": {
                "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                "size": 16,
                "color": "#1e293b",
            },
            "x": 0.5,
            "xanchor": "center",
        }

    config["xaxis"] = AXIS_DEFAULTS.copy()
    config["yaxis"] = AXIS_DEFAULTS.copy()

    return config


def get_greek_color(greek_name: str) -> str:
    """Get the color for a specific Greek."""
    return GREEK_COLORS.get(greek_name, CHART_COLORS["primary"])


def format_currency_hover(value: float) -> str:
    """Format a value as currency for hover text."""
    if value >= 0:
        return f"${value:,.2f}"
    return f"-${abs(value):,.2f}"
