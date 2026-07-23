"""
Diagnostics charts for returns-based (GARCH-family) calibrations
================================================================

Surface models get a residual heatmap and a QQ-plot. GARCH-family fits
need a different lens — the model captures heteroskedasticity, so the
right diagnostics are:

* the conditional-volatility series ``σ_t`` over time,
* the distribution of standardised residuals ``z_t = r_t / σ_t`` against
  N(0, 1) (QQ-plot),
* the auto-correlation of squared standardised residuals (no remaining
  ARCH effect ↔ flat ACF inside the ±2/√T band).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from utils.plotly_theme import COLORS, apply_lab_theme, empty_state_figure


def _autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample auto-correlation up to ``max_lag``, lag 0 excluded.

    Uses the unbiased estimator ``Σ(x_t − x̄)(x_{t-k} − x̄) / Σ(x_t − x̄)²``
    so the lag-0 result equals 1 by construction (we skip it for the
    chart). ``scipy.signal.correlate`` would be O(n²) for n=10k samples;
    a simple loop with vectorised inner products is plenty fast at
    typical n ≤ 10k and avoids the FFT-rounding artefacts that show up
    around lag 0.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    denom = float(np.dot(x, x))
    if denom <= 0.0 or n < 2:
        return np.zeros(max_lag, dtype=np.float64)
    out = np.empty(max_lag, dtype=np.float64)
    for k in range(1, max_lag + 1):
        out[k - 1] = float(np.dot(x[:-k], x[k:])) / denom
    return out


def render_conditional_volatility(
    log_returns: np.ndarray,
    variance_series: np.ndarray,
    annualization_factor: int,
) -> go.Figure:
    """Two-pane chart: returns at top, annualised σ_t at bottom."""
    log_returns = np.asarray(log_returns, dtype=np.float64)
    # variance_series has length T+1: index t is the conditional variance for
    # r_t (index T is the one-step forecast). Keep var[:-1] so σ_t aligns with
    # the return r_t it generated — var[1:] would plot h_{t+1} against r_t.
    sigma_series = np.sqrt(np.maximum(variance_series[:-1], 0.0))
    sigma_ann = sigma_series * np.sqrt(annualization_factor) * 100.0

    t = np.arange(log_returns.size)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=log_returns * 100.0,
            mode="lines",
            line=dict(color=COLORS["text_muted"], width=0.9),
            name="log-returns (%)",
            hovertemplate="t=%{x}<br>r=%{y:.3f}%<extra></extra>",
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=sigma_ann,
            mode="lines",
            line=dict(color=COLORS["primary"], width=2.0),
            name="σ_t (annualised, %)",
            hovertemplate="t=%{x}<br>σ_t=%{y:.2f}%<extra></extra>",
            yaxis="y2",
        )
    )
    apply_lab_theme(
        fig,
        height=380,
        title="Conditional volatility · model-implied σ_t over the sample",
    )
    fig.update_layout(
        yaxis=dict(
            title="Log-Return rₜ (%)",
            side="left",
            showgrid=False,
            zeroline=True,
            zerolinecolor=COLORS["grid"],
            ticksuffix="%",
        ),
        yaxis2=dict(
            title=dict(
                text="Conditional Volatility σₜ  (annualised, %)",
                font=dict(color=COLORS["primary"]),
            ),
            side="right",
            overlaying="y",
            showgrid=False,
            tickfont=dict(color=COLORS["primary"]),
            ticksuffix="%",
        ),
    )
    fig.update_xaxes(title="Time Step t  (observation index)")
    return fig


def render_conditional_volatility_overlay(
    log_returns: np.ndarray,
    labeled_sigmas,
    annualization_factor: int,
) -> go.Figure:
    """Returns (drawn once) + one overlaid annualised σ_t line per run.

    ``labeled_sigmas`` is a list of ``(label, variance_series, style)``
    where each ``variance_series`` has length ``T+1`` (the recursion's t=0
    prior plus t=1..T) and ``style`` is the ``series_style`` dict. The
    log-returns are identical across runs (same input series), so they are
    shown once on the left axis; every run's σ_t is superimposed on the
    right axis, coloured/dashed by its style.
    """
    log_returns = np.asarray(log_returns, dtype=np.float64)
    t = np.arange(log_returns.size)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=log_returns * 100.0,
            mode="lines",
            line=dict(color=COLORS["text_muted"], width=0.9),
            name="log-returns (%)",
            hovertemplate="t=%{x}<br>r=%{y:.3f}%<extra></extra>",
            yaxis="y",
        )
    )
    for label, variance_series, style in labeled_sigmas:
        sigma_series = np.sqrt(
            np.maximum(np.asarray(variance_series, dtype=np.float64)[:-1], 0.0)
        )
        sigma_ann = sigma_series * np.sqrt(annualization_factor) * 100.0
        fig.add_trace(
            go.Scatter(
                x=t[: sigma_ann.size],
                y=sigma_ann,
                mode="lines",
                line=dict(color=style["color"], dash=style["dash"], width=2.0),
                name=label,
                legendgroup=label,
                hovertemplate=f"<b>{label}</b><br>t=%{{x}}<br>σ_t=%{{y:.2f}}%<extra></extra>",
                yaxis="y2",
            )
        )
    apply_lab_theme(
        fig,
        height=380,
        title="Conditional volatility · model-implied σ_t over the sample",
    )
    fig.update_layout(
        yaxis=dict(
            title="Log-Return rₜ (%)",
            side="left",
            showgrid=False,
            zeroline=True,
            zerolinecolor=COLORS["grid"],
            ticksuffix="%",
        ),
        yaxis2=dict(
            title=dict(
                text="Conditional Volatility σₜ  (annualised, %)",
                font=dict(color=COLORS["primary"]),
            ),
            side="right",
            overlaying="y",
            showgrid=False,
            tickfont=dict(color=COLORS["primary"]),
            ticksuffix="%",
        ),
    )
    fig.update_xaxes(title="Time Step t  (observation index)")
    return fig


def render_standardised_residuals_qq(z_residuals: np.ndarray) -> go.Figure:
    """QQ-plot of z_t against N(0, 1) — same convention as the surface tab."""
    z = np.asarray(z_residuals, dtype=np.float64)
    z = z[np.isfinite(z)]
    n = z.size
    if n < 2:
        return empty_state_figure("Not enough standardised residuals for a QQ-plot.")
    z_sorted = np.sort(z)
    quantiles = np.array([(i + 0.5) / n for i in range(n)])
    theoretical = stats.norm.ppf(quantiles)

    lo = float(min(theoretical.min(), z_sorted.min()))
    hi = float(max(theoretical.max(), z_sorted.max()))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=z_sorted,
            mode="markers",
            marker=dict(
                color=COLORS["info"], size=6, line=dict(color=COLORS["plot"], width=0.5)
            ),
            name="empirical",
            hovertemplate="theory=%{x:.2f}<br>empirical=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color=COLORS["danger"], dash="dash", width=2),
            name="y = x",
        )
    )
    apply_lab_theme(
        fig,
        height=380,
        title="QQ-plot · standardised residuals z_t = r_t / σ_t vs N(0, 1)",
    )
    fig.update_xaxes(title="Theoretical Quantiles  N(0, 1)")
    fig.update_yaxes(title="Empirical Quantiles  zₜ = rₜ / σₜ")
    return fig


def render_squared_residuals_acf(
    z_residuals: np.ndarray,
    *,
    max_lag: int = 20,
) -> go.Figure:
    """Stem chart of ρ(z²_t, z²_{t-k}) with ±2/√T confidence bands.

    A model that absorbs heteroskedasticity correctly should leave its
    standardised residuals roughly i.i.d., which means the squared-z
    autocorrelations should sit inside the ±2/√T band (Bartlett's
    formula, 95 % under the null of no serial correlation).
    """
    z = np.asarray(z_residuals, dtype=np.float64)
    z = z[np.isfinite(z)]
    n = z.size
    if n < max_lag + 4:
        return empty_state_figure(
            f"Need at least {max_lag + 4} residuals for a {max_lag}-lag ACF — "
            f"got {n}."
        )
    acf = _autocorrelation(z * z, max_lag=max_lag)
    band = 2.0 / np.sqrt(n)
    lags = np.arange(1, max_lag + 1)

    inside = np.abs(acf) <= band
    bar_colors = np.where(inside, COLORS["primary"], COLORS["danger"])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=lags,
            y=acf,
            marker_color=bar_colors,
            name="ρ(z², lag)",
            hovertemplate="lag %{x}<br>ρ=%{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_hline(y=band, line=dict(color=COLORS["text_muted"], dash="dot", width=1))
    fig.add_hline(y=-band, line=dict(color=COLORS["text_muted"], dash="dot", width=1))
    fig.add_hline(y=0.0, line=dict(color=COLORS["axis"], width=1))
    apply_lab_theme(
        fig,
        height=320,
        title="ACF of squared standardised residuals · "
        "bars outside ±2/√T (dotted) → remaining ARCH effect",
    )
    fig.update_xaxes(title="Lag k  (observations)", tickmode="linear", dtick=1)
    fig.update_yaxes(
        title="Autocorrelation  ρ(z²ₜ, z²ₜ₋ₖ)",
        range=[
            float(min(acf.min(), -band) - 0.05),
            float(max(acf.max(), band) + 0.05),
        ],
    )
    return fig


__all__ = [
    "render_conditional_volatility",
    "render_conditional_volatility_overlay",
    "render_standardised_residuals_qq",
    "render_squared_residuals_acf",
]
