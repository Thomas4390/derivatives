"""
Centralized logging helpers for the backend package.

This module is the single entry point for module-level loggers and root
configuration. Library code should call :func:`get_logger` only — never
:func:`logging.basicConfig` — so applications (Streamlit, scripts, tests)
remain free to choose their own handlers and levels.

Usage
-----
Library module:

    from backend.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.debug("calibration started", extra={"market": market.id})

Application entry point (Streamlit app, smoke test, CLI):

    from backend.utils.logging import configure_root

    configure_root()  # honours BACKEND_LOG_LEVEL env var

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import logging
import os
from typing import Final

DEFAULT_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)
DEFAULT_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"
ENV_LEVEL_VAR: Final[str] = "BACKEND_LOG_LEVEL"


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Parameters
    ----------
    name
        Should always be ``__name__`` so loggers form a tree mirroring the
        package layout (``backend.calibration.heston_calibrator`` etc.).

    Returns
    -------
    logging.Logger
        A logger that inherits handlers/level from the root logger. No
        handler is attached by this helper — the application is responsible
        for configuring sinks via :func:`configure_root` or its own setup.
    """
    return logging.getLogger(name)


def configure_root(
    level: str | int | None = None,
    *,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
    force: bool = False,
) -> None:
    """Attach a single ``StreamHandler`` to the root logger.

    Idempotent by default: if the root logger already has handlers (e.g.
    Streamlit, pytest, or another framework configured logging first), the
    call is a no-op unless ``force=True``.

    Parameters
    ----------
    level
        Log level name (``"DEBUG"``) or numeric. If ``None``, reads the
        ``BACKEND_LOG_LEVEL`` env var, defaulting to ``"WARNING"``.
    fmt, datefmt
        Standard :mod:`logging` format strings.
    force
        Replace existing handlers when ``True``. Use only at top-level
        application entry points.
    """
    root = logging.getLogger()
    if root.handlers and not force:
        return

    if level is None:
        level = os.environ.get(ENV_LEVEL_VAR, "WARNING")

    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(handler)
    root.setLevel(level)


__all__ = ["get_logger", "configure_root", "DEFAULT_FORMAT", "ENV_LEVEL_VAR"]
