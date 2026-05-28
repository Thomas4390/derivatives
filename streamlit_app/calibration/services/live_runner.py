"""
Real-time calibration runner — pure orchestration layer
========================================================

Backend calibrators are synchronous, so to give the user real-time
feedback we run each calibration in a daemon thread and stream
``IterationSnapshot`` updates through a thread-safe queue. This module
deliberately knows **nothing** about Streamlit or Plotly: it exposes a
small handle that any caller (CLI, Streamlit UI, tests) can consume.

Streamlit-side rendering lives in ``components.live_progress``.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from config.constants import LIVE_IDLE_SLEEP_SEC, LIVE_POLL_INTERVAL_SEC
from services.calibration_service import CalibrationRunSummary, calibrate_with

logger = logging.getLogger(__name__)


# Sentinel objects pushed on the queue when the worker thread terminates.
_DONE = object()
_ERROR = object()


class CalibrationCancelled(BaseException):
    """Cooperative cancellation signal from the UI.

    Subclasses :class:`BaseException` (like ``KeyboardInterrupt``) so a
    broad ``except Exception`` inside the backend calibrator can't
    swallow it before it propagates back to :func:`_run`. The
    iteration callback raises this when the matching handle's
    ``cancel_event`` is set, which is the UI's signal that the user
    pressed the Stop button.
    """


@dataclass
class LiveRunHandle:
    """Handle to an in-progress calibration run.

    ``history`` accumulates :class:`IterationSnapshot` objects as they
    drain off the queue. ``dropped_snapshots`` counts callbacks that
    couldn't be enqueued (UI consumer too slow). ``cancel_event`` is
    flipped by the UI when the user requests a stop — the next
    iteration callback notices it and raises
    :class:`CalibrationCancelled`.
    """

    solver_name: str
    queue: "queue.Queue"
    thread: threading.Thread
    result_holder: list = field(default_factory=lambda: [None])
    error_holder: list = field(default_factory=lambda: [None])
    history: list = field(default_factory=list)
    dropped_snapshots: list[int] = field(default_factory=lambda: [0])
    cancel_event: threading.Event = field(default_factory=threading.Event)
    n_restarts: int = 1

    @property
    def result(self) -> CalibrationRunSummary | None:
        return self.result_holder[0]

    @property
    def error(self) -> BaseException | None:
        return self.error_holder[0]

    @property
    def n_dropped(self) -> int:
        return self.dropped_snapshots[0]

    @property
    def cancelled(self) -> bool:
        """True when the worker exited because the UI requested a stop."""
        return isinstance(self.error_holder[0], CalibrationCancelled)


def start_run(
    *,
    model_key: str,
    market_data: Any,
    solver_name: str,
    true_params: dict[str, float],
    objective_name: str = "price_mse",
    objective_settings: dict[str, Any] | None = None,
    constraint_settings: dict[str, Any] | None = None,
    n_restarts: int = 5,
    max_nfev: int = 200,
    de_seed: int = 42,
) -> LiveRunHandle:
    """Spawn the calibration in a daemon thread + queue for snapshots."""
    q: queue.Queue = queue.Queue()
    handle = LiveRunHandle(
        solver_name=solver_name,
        queue=q,
        thread=None,  # type: ignore[arg-type]  # set below before returning
        n_restarts=int(n_restarts),
    )

    def _on_snapshot(snap):
        # Cooperative cancellation check: as soon as the UI flips the
        # event, raise out of the callback. The exception subclasses
        # BaseException so the broad `except Exception` in the
        # calibrator does not swallow it.
        if handle.cancel_event.is_set():
            raise CalibrationCancelled(
                f"calibration cancelled by user (solver={solver_name})"
            )
        try:
            q.put(snap, block=False)
        except queue.Full:
            handle.dropped_snapshots[0] += 1

    def _run():
        try:
            summary = calibrate_with(
                model_key=model_key,
                market_data=market_data,
                solver_name=solver_name,
                true_params=true_params,
                objective_name=objective_name,
                objective_settings=objective_settings,
                constraint_settings=constraint_settings,
                n_restarts=n_restarts,
                max_nfev=max_nfev,
                de_seed=de_seed,
                log_iterations=True,
                iteration_callback=_on_snapshot,
            )
            handle.result_holder[0] = summary
            q.put(_DONE)
        except CalibrationCancelled as exc:
            # User-initiated cancellation — not an error, no traceback log.
            handle.error_holder[0] = exc
            q.put(_ERROR)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Live calibration thread crashed for %s", solver_name)
            handle.error_holder[0] = exc
            q.put(_ERROR)

    th = threading.Thread(target=_run, daemon=True, name=f"calib-{solver_name}")
    handle.thread = th
    th.start()
    return handle


def drain_handle(handle: LiveRunHandle, *, timeout: float = 0.05) -> bool:
    """Pull whatever is in the queue right now into ``handle.history``.

    Returns ``True`` iff a terminal sentinel (``_DONE`` or ``_ERROR``) was
    seen during the drain. The caller is expected to call this repeatedly
    until it returns True.
    """
    finished = False
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            item = handle.queue.get_nowait()
        except queue.Empty:
            break
        if item is _DONE or item is _ERROR:
            finished = True
            continue
        handle.history.append(item)
    return finished


def iter_snapshots(
    handle: LiveRunHandle,
    *,
    poll_interval: float = LIVE_POLL_INTERVAL_SEC,
    idle_sleep: float = LIVE_IDLE_SLEEP_SEC,
) -> Iterator[int]:
    """Yield the new history length each time at least one snapshot arrived.

    Consumers use this generator to drive UI re-renders without owning the
    polling logic. The generator returns when the worker thread signals
    termination (success or error).
    """
    finished = False
    last_count = -1
    while not finished:
        finished = drain_handle(handle, timeout=poll_interval)
        n = len(handle.history)
        if n != last_count:
            last_count = n
            yield n
        else:
            time.sleep(idle_sleep)
    drain_handle(handle, timeout=0.2)
    if last_count != len(handle.history):
        yield len(handle.history)
