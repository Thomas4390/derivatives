"""Compose the native Streamlit ``help=`` tooltip for a sidebar option group.

``st.pills`` / ``st.segmented_control`` expose a single widget-level
``help=`` "?" icon (rendered to the right of the buttons). This builds its
markdown content: a general group explanation followed by one tight line
per available option, so every option's detail is reachable from the
native icon — no duplicated button row.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping


def compose_help(
    group_help: str,
    names: tuple[str, ...],
    descriptions: Mapping[str, str],
    label: Callable[[str], str],
) -> str:
    """Return ``group_help`` plus one ``**label** — description`` line per name.

    Parameters
    ----------
    group_help :
        The general, group-level explanation shown first.
    names :
        Available option keys — pass the same tuple the widget uses for
        ``options=`` so the tooltip lists exactly the selectable options.
    descriptions :
        ``key -> short description``. Missing keys degrade to an empty
        blurb rather than raising.
    label :
        The panel's display-name builder (e.g. ``_format_solver``) so the
        tooltip headings match the button text.
    """
    lines = "\n\n".join(
        f"**{label(n)}** — {descriptions.get(n, '')}" for n in names
    )
    return f"{group_help}\n\n{lines}" if lines else group_help
