"""Per-tab renderers for the calibration explorer.

Each module exposes ``render(ctx)`` (signature taking the shared context
dict prepared in ``app.py``) and is responsible for one Streamlit tab.
"""

from tabs.comparison import render as render_comparison
from tabs.diagnostics import render as render_diagnostics
from tabs.landscape import render as render_landscape
from tabs.live import render as render_live
from tabs.setup import render as render_setup
from tabs.theory import render as render_theory

__all__ = [
    "render_comparison",
    "render_diagnostics",
    "render_landscape",
    "render_live",
    "render_setup",
    "render_theory",
]
