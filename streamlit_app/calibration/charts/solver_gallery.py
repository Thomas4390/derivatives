"""Conceptual mini-figures illustrating the four solver strategies.

Each helper produces a *static* Plotly figure on the 2D Rosenbrock
function so a student can see — at a glance — how the geometry of the
search differs between LM trust regions, Differential Evolution
populations, Nelder-Mead simplex reflections, and L-BFGS line search.

The figures are not live calibrations; the trajectories are hand-tuned
demonstrations chosen to highlight each solver's characteristic shape.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from utils.plotly_theme import COLORS, FONT_FAMILY, apply_lab_theme


def _rosenbrock_grid(n: int = 80, lim: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-lim, lim, n)
    y = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
    return X, Y, Z


def _base_figure(title: str, sub: str) -> go.Figure:
    X, Y, Z = _rosenbrock_grid()
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=X[0], y=Y[:, 0], z=np.log10(Z + 1e-3),
        colorscale="Greys", showscale=False,
        contours=dict(coloring="lines", showlines=True),
        line=dict(width=0.6),
        opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=[1.0], y=[1.0],
        mode="markers",
        marker=dict(size=14, color="#10b981", symbol="star",
                     line=dict(color="white", width=1.5)),
        name="optimum (1, 1)",
        hovertemplate="optimum<extra></extra>",
    ))
    sub_color = COLORS["text_dim"]
    sub_html = (
        f"<span style='font-size:0.78rem;color:{sub_color};"
        f"font-family:{FONT_FAMILY}'>{sub}</span>"
    )
    apply_lab_theme(
        fig, height=320,
        title=f"<b>{title}</b><br>{sub_html}",
        legend_horizontal=False,
        margin=(40, 20, 60, 30),
    )
    fig.update_xaxes(range=[-2, 2], title="θ₁")
    fig.update_yaxes(range=[-2, 2], title="θ₂", scaleanchor="x", scaleratio=1)
    fig.update_layout(showlegend=False)
    return fig


def render_lm_concept() -> go.Figure:
    """Trust-region Gauss-Newton: contracting circles along a banana."""
    fig = _base_figure(
        "LM-JAX · Levenberg-Marquardt (trust-region Gauss-Newton)",
        "Locally quadratic model + adaptive trust radius. Fastest convergence "
        "when residuals are smooth and a Jacobian is available.",
    )
    traj = np.array([
        [-1.6, 1.8], [-1.1, 1.0], [-0.6, 0.4], [-0.1, 0.0],
        [0.3, 0.05], [0.65, 0.4], [0.9, 0.75], [1.0, 1.0],
    ])
    # Trust regions: shrink as we approach the optimum.
    for i, (cx, cy) in enumerate(traj[:-1]):
        r = 0.35 * (1 - i / len(traj)) + 0.06
        fig.add_shape(
            type="circle",
            x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
            line=dict(color="rgba(13,148,136,0.55)", width=1.2),
        )
    fig.add_trace(go.Scatter(
        x=traj[:, 0], y=traj[:, 1],
        mode="lines+markers",
        line=dict(color="#0d9488", width=2.4),
        marker=dict(size=8, color="#0d9488", line=dict(color="white", width=1.2)),
        name="trajectory",
    ))
    return fig


def render_de_concept() -> go.Figure:
    """Differential Evolution: population shrinking onto the optimum."""
    fig = _base_figure(
        "DE · Differential Evolution",
        "Population-based stochastic search. Robust against local minima at "
        "the cost of ~10× more function evaluations than LM.",
    )
    rng = np.random.default_rng(7)
    for gen, scale in enumerate([1.6, 1.1, 0.7, 0.35, 0.12]):
        center = (1.0, 1.0)
        pts = rng.normal(loc=center, scale=scale, size=(18, 2))
        opacity = 0.25 + 0.7 * (gen / 4.0)
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1],
            mode="markers",
            marker=dict(
                size=7 + gen * 1.6, color="#d97706", opacity=opacity,
                line=dict(color="white", width=0.8),
            ),
            name=f"generation {gen}",
        ))
    return fig


def render_nm_concept() -> go.Figure:
    """Nelder-Mead: a simplex that reflects and contracts."""
    fig = _base_figure(
        "NM · Nelder-Mead simplex",
        "Derivative-free reflections, expansions, contractions of a triangle. "
        "Intuitive geometry but slow on problems with > 5 parameters.",
    )
    simplices = [
        np.array([[-1.6, 1.5], [-0.5, 1.8], [-1.0, 0.5]]),
        np.array([[-0.5, 1.8], [-1.0, 0.5], [0.2, 1.1]]),
        np.array([[-1.0, 0.5], [0.2, 1.1], [0.4, 0.2]]),
        np.array([[0.2, 1.1], [0.4, 0.2], [0.85, 0.7]]),
        np.array([[0.4, 0.2], [0.85, 0.7], [1.0, 1.0]]),
    ]
    for i, simplex in enumerate(simplices):
        closed = np.vstack([simplex, simplex[0:1]])
        fig.add_trace(go.Scatter(
            x=closed[:, 0], y=closed[:, 1],
            mode="lines",
            line=dict(color="#7c3aed", width=1.5 + 0.4 * i),
            opacity=0.25 + 0.15 * i,
            name=f"step {i}",
        ))
    return fig


def render_lbfgs_concept() -> go.Figure:
    """L-BFGS-B: gradient descent with line search."""
    fig = _base_figure(
        "L-BFGS-B · quasi-Newton with box bounds",
        "Approximates the Hessian from gradient history. Default for GARCH "
        "MLE; on surface problems it lacks the residual structure LM exploits.",
    )
    traj = np.array([
        [-1.8, 1.3], [-1.4, 0.7], [-0.9, 0.35], [-0.4, 0.1],
        [0.15, 0.05], [0.55, 0.25], [0.85, 0.6], [1.0, 1.0],
    ])
    # Arrows along the steps to suggest gradient direction.
    fig.add_trace(go.Scatter(
        x=traj[:, 0], y=traj[:, 1],
        mode="lines+markers",
        line=dict(color="#0284c7", width=2.4, dash="dash"),
        marker=dict(size=8, color="#0284c7", line=dict(color="white", width=1.2)),
        name="trajectory",
    ))
    for (x0, y0), (x1, y1) in zip(traj[:-1], traj[1:]):
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=2, arrowsize=1.1, arrowwidth=1.3, arrowcolor="#0284c7",
            showarrow=True, opacity=0.55, text="",
        )
    return fig


SOLVER_FIGURES = {
    "LM-JAX": render_lm_concept,
    "DE": render_de_concept,
    "NM": render_nm_concept,
    "L-BFGS-B": render_lbfgs_concept,
}
