"""Save / Load setup section of the sidebar.

Lets a user save the current market + positions as a compact copy-paste **"setup
code"** and restore it in a later session. No file is downloaded (a setup is an
input, not a result) — the code is shown in a copyable block and pasted back to load.
"""

import streamlit as st
from services.portfolio_io import (
    SnapshotError,
    decode_snapshot,
    encode_snapshot,
    to_pretty_json,
)
from services.state_manager import request_snapshot_restore


def render_portfolio_io(
    positions: list[dict], stock_position: dict | None, market: dict[str, float]
) -> None:
    """Render the 'Save / Load Setup' expander in the sidebar."""
    with st.expander("💾 Save / Load Setup", expanded=False):
        st.caption(
            "Save your market + positions as a code, and paste it back next session "
            "to pick up where you left off. Nothing is downloaded — just copy the code."
        )

        # ── Save: a copyable, always-current code (st.code has a copy button) ──
        st.markdown("**Save** — copy this setup code:")
        st.code(encode_snapshot(market, positions, stock_position), language=None)
        with st.expander("View as readable JSON", expanded=False):
            st.code(
                to_pretty_json(market, positions, stock_position), language="json"
            )

        st.markdown("---")

        # ── Load: paste a code (or JSON) and restore ──
        pasted = st.text_area(
            "**Load** — paste a setup code (or JSON):",
            height=70,
            key="og_import_code",
            placeholder="OGX1-…",
        )
        if st.button("Load setup", key="og_load_btn", use_container_width=True):
            text = (pasted or "").strip()
            if not text:
                st.warning("Paste a setup code first.")
                return
            try:
                snapshot = decode_snapshot(text)
            except SnapshotError as exc:
                st.error(f"Could not load: {exc}")
                return
            request_snapshot_restore(snapshot)
            st.success("Setup loaded — restoring…")
            st.rerun()
