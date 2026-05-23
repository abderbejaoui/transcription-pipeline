"""Simple JSONL-backed doctor review queue."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE_PATH = PROJECT_ROOT / "data" / "doctor_review_queue.jsonl"

_lock = threading.Lock()


def enqueue(item: Dict[str, Any], *, path: Path = DEFAULT_QUEUE_PATH) -> Dict[str, Any]:
    rec = dict(item)
    rec["id"] = uuid.uuid4().hex
    rec["created_at"] = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec
