"""
Post-calibration diagnostics charts (themed)
==============================================

* Residual heatmap (strike × maturity)
* Parameter correlation matrix (from Gauss-Newton covariance)
* QQ plot of standardised residuals
* Standard-errors / 95 % CI table helper
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from utils.numba_kernels import correlation_matrix_from_cov, standardised_residuals
from utils.plotly_theme import COLORS, FONT_FAMILY, apply_lab_theme, empty_state_figure


def _diverging_scale() -> list[list]:
    return [
        [0.0, "#ef4444"],
        [0.25, "#f59e0b"],
        [0.5, "#0f1722"],
        [0.75, "#5eead4"],
        [1.0, "#00ff9c"],
    ]


def render_residual_heatmap(
    residuals: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    *,
    spot: float | None = None,
) -> go.Figure:
    if (
        residuals is None
        or np.size(residuals) == 0
        or np.all(np.isnan(np.asarray(residuals, dtype=np.float64)))
    ):
        return empty_state_figure("No residuals available for this run.")
    # Symmetric clamp around 0 keeps the diverging colorscale honest —
    # without this, an asymmetric residual distribution would visually
    # claim "more red than blue" even though the magnitudes match.
    z_abs = float(np.nanmax(np.abs(np.asarray(residuals, dtype=np.float64))))
    z_clip = z_abs if z_abs > 0.0 else 1e-6
    # Industry-standard surface convention: moneyness K/S₀ on the strike
    # axis. Falls back to raw strikes when spot is not provided.
    if spot is not None and spot > 0.0:
        x_vals = strikes / float(spot)
        x_title = "Moneyness K / S₀"
        x_tickformat = ".2f"
        x_hover_label = "K/S₀"
        x_hover_fmt = ".3f"
    else:
        x_vals = strikes
        x_title = "Strike K"
        x_tickformat = ".0f"
        x_hover_label = "K"
        x_hover_fmt = ".2f"
    fig = go.Figure(
        data=go.Heatmap(
            z=residuals,
            x=x_vals,
            y=maturities * 365.0,
            colorscale=_diverging_scale(),
            zmid=0.0, zmin=-z_clip, zmax=z_clip,
            colorbar=dict(
                title=dict(
                    text="Model − Market<br>(price units)",
                    font=dict(family=FONT_FAMILY, color=COLORS["axis"]),
                ),
                tickfont=dict(family=FONT_FAMILY, color=COLORS["axis"]),
                thickness=14,
                xpad=8,
                outlinewidth=0,
            ),
            hovertemplate=(
                f"{x_hover_label}=%{{x:{x_hover_fmt}}}<br>T=%{{y:.0f}}d<br>"
                "residual=%{z:.4f} (price units)<extra></extra>"
            ),
        )
    )
    apply_lab_theme(fig, height=440, title="Residual heatmap  ·  model − market price")
    fig.update_xaxes(title=x_title, tickformat=x_tickformat)
    fig.update_yaxes(title="Maturity T (days)", tickformat=".0f")
    return fig


def render_correlation_matrix(
    cov_matrix: np.ndarray,
    param_names: list[str],
) -> go.Figure:
    if cov_matrix is None or len(param_names) == 0:
        return empty_state_figure("No covariance available for this run.")
    corr = correlation_matrix_from_cov(np.asarray(cov_matrix, dtype=np.float64))
    text = [[f"{corr[i, j]:.2f}" for j in range(len(param_names))]
            for i in range(len(param_names))]
    fig = go.Figure(
        data=go.Heatmap(
            z=corr,
            x=param_names,
            y=param_names,
            colorscale=_diverging_scale(),
            zmid=0.0, zmin=-1.0, zmax=1.0,
            text=text,
            texttemplate="%{text}",
            textfont=dict(family=FONT_FAMILY, color=COLORS["text"], size=12),
            colorbar=dict(
                title=dict(text="ρ", font=dict(family=FONT_FAMILY, color=COLORS["axis"])),
                tickfont=dict(family=FONT_FAMILY, color=COLORS["axis"]),
                thickness=12,
                outlinewidth=0,
            ),
        )
    )
    apply_lab_theme(fig, height=440,
                     title="Parameter correlation  ·  large |ρ| → identifiability issue")
    return fig


def render_qq_plot(residuals: np.ndarray) -> go.Figure:
    z = standardised_residuals(np.asarray(residuals, dtype=np.float64))
    z_sorted = np.sort(z)
    n = z_sorted.size
    if n < 2:
        return empty_state_figure("Not enough residuals for a QQ-plot.")
    quantiles = np.array([(i + 0.5) / n for i in range(n)])
    theoretical = stats.norm.ppf(quantiles)

    lo = float(min(theoretical.min(), z_sorted.min()))
    hi = float(max(theoretical.max(), z_sorted.max()))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=theoretical, y=z_sorted,
            mode="markers",
            marker=dict(color=COLORS["info"], size=8,
                        line=dict(color=COLORS["plot"], width=1)),
            name="empirical",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines",
            line=dict(color=COLORS["danger"], dash="dash", width=2),
            name="y = x",
        )
    )
    apply_lab_theme(fig, height=420, title="QQ-plot  ·  standardised residuals vs N(0, 1)")
    fig.update_xaxes(title="Theoretical Quantiles  N(0, 1)")
    fig.update_yaxes(title="Empirical Quantiles  (standardised residuals)")
    return fig


def render_qq_overlay(labeled_series, *, standardise: bool = True) -> go.Figure:
    """Overlay QQ-plots of several residual sets against N(0, 1).

    ``labeled_series`` is a list of ``(label, values, style)`` where
    ``style`` is the ``utils.plotly_theme.series_style`` dict. Each set
    becomes one empirical scatter (coloured/shaped by its style); a single
    ``y = x`` reference line spans the combined data range so the user
    compares how each run's residuals depart from normality on one chart.

    Surface residuals are centred/scaled via
    :func:`utils.numba_kernels.standardised_residuals` (``standardise=True``);
    GARCH ``z_t`` arrive already standardised, so pass ``standardise=False``.
    """
    fig = go.Figure()
    lo = float("inf")
    hi = float("-inf")
    any_points = False
    for label, values, style in labeled_series:
        v = np.asarray(values, dtype=np.float64)
        z = standardised_residuals(v) if standardise else v[np.isfinite(v)]
        z = z[np.isfinite(z)]
        if z.size < 2:
            continue
        z_sorted = np.sort(z)
        n = z_sorted.size
        theoretical = stats.norm.ppf((np.arange(n) + 0.5) / n)
        any_points = True
        lo = min(lo, float(theoretical.min()), float(z_sorted.min()))
        hi = max(hi, float(theoretical.max()), float(z_sorted.max()))
        fig.add_trace(
            go.Scatter(
                x=theoretical,
                y=z_sorted,
                mode="markers",
                marker=dict(
                    color=style["color"],
                    size=6,
                    symbol=style["symbol"],
                    line=dict(color=COLORS["plot"], width=0.5),
                ),
                name=label,
                legendgroup=label,
                hovertemplate=(
                    f"<b>{label}</b><br>theory %{{x:.2f}}<br>"
                    "empirical %{y:.2f}<extra></extra>"
                ),
            )
        )
    if not any_points:
        return empty_state_figure("Not enough residuals for a QQ-plot.")
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color=COLORS["danger"], dash="dash", width=2),
            name="y = x",
        )
    )
    apply_lab_theme(fig, height=420, title="QQ-plot  ·  standardised residuals vs N(0, 1)")
    fig.update_xaxes(title="Theoretical Quantiles  N(0, 1)")
    fig.update_yaxes(title="Empirical Quantiles  (standardised residuals)")
    return fig


def uncertainty_table(uncertainty_dict: dict, true_params: dict | None = None) -> list[dict]:
    rows = []
    for name, stats_dict in uncertainty_dict.items():
        row = {
            "Parameter": name,
            "Estimate": stats_dict["estimate"],
            "Std error": stats_dict["std_error"],
            "95 % CI low": stats_dict["ci_lo"],
            "95 % CI high": stats_dict["ci_hi"],
        }
        if true_params is not None and name in true_params:
            row["True"] = true_params[name]
            denom = max(abs(true_params[name]), 1e-12)
            row["Rel. error"] = abs(stats_dict["estimate"] - true_params[name]) / denom
        rows.append(row)
    return rows
