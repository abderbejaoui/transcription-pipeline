"""Tiny pipeline tracer.

Anything anywhere in the pipeline can call `emit(stage, payload)` and the
event lands on the active per-request queue (a thread-local). The HTTP
streaming endpoint pumps the queue out as NDJSON. If no tracer is active
on the current thread, `emit` is a silent no-op — services don't need to
know whether they're in a traced request.
"""

from __future__ import annotations

import contextvars
import json
import queue
import time
from typing import Any, Dict, Optional


_active: contextvars.ContextVar[Optional["Tracer"]] = contextvars.ContextVar(
    "active_tracer", default=None
)


class Tracer:
    """A queue-backed event sink. One per request."""

    def __init__(self) -> None:
        self._q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._t0 = time.time()
        self._closed = False

    def emit(self, stage: str, payload: Dict[str, Any]) -> None:
        if self._closed:
            return
        event = {
            "t": round(time.time() - self._t0, 3),
            "stage": stage,
            "payload": payload,
        }
        self._q.put(event)

    def close(self, final: Optional[Dict[str, Any]] = None) -> None:
        if self._closed:
            return
        self._closed = True
        if final is not None:
            self._q.put({"t": round(time.time() - self._t0, 3), "stage": "final", "payload": final})
        self._q.put(None)  # sentinel

    def stream_lines(self):
        """Generator: yields one NDJSON line per event until close() is called."""
        while True:
            event = self._q.get()
            if event is None:
                return
            try:
                yield json.dumps(event, ensure_ascii=False, default=str) + "\n"
            except (TypeError, ValueError):
                yield json.dumps(
                    {"t": event["t"], "stage": event["stage"], "payload": "<unserialisable>"},
                    ensure_ascii=False,
                ) + "\n"


def emit(stage: str, payload: Dict[str, Any]) -> None:
    t = _active.get()
    if t is not None:
        t.emit(stage, payload)


def set_active(tracer: Optional[Tracer]) -> contextvars.Token:
    return _active.set(tracer)


def reset_active(token: contextvars.Token) -> None:
    _active.reset(token)
