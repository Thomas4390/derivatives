"""Floating glossary popover.

Sources its entries from ``config.formulas.GLOSSARY`` so the symbols
shown to the user stay in sync with the cheat-sheet and the loss
formulas rendered in the Theory tab.
"""

from __future__ import annotations

import streamlit as st

from config.formulas import GLOSSARY


def render() -> None:
    """Render the glossary popover inline (caller decides placement)."""
    with st.popover("📖 Glossary", help="Alphabetical symbol reference"):
        st.markdown("**Symbol reference** — same definitions used everywhere in the app.")
        for symbol, definition in sorted(GLOSSARY.items(), key=lambda kv: kv[0].lower()):
            st.markdown(f"- **`{symbol}`** — {definition}")
